import logging
import operator
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.fx import GraphModule, Node

from .utils import get_arg_value
from ..banking import require_allocation
from ..mapping import duplicate_shared_nodes
from ..mapping_utils import (
    is_conv2d,
    is_depthwise_conv,
    is_elementwise_op,
    is_fully_connected,
    is_linear,
    is_matmul,
    is_nop,
    is_pooling,
    is_reshape_op,
)
from ...pt2e_utils import deduplicate_nodes, fetch_attr, propagate_shape
from ...layout_ops import (
    NCHW_TO_NHWC,
    NHWC_OP_VARIANTS,
    NHWC_TO_NCHW,
    WEIGHT_NCHW_TO_HWIO,
)
from ...quantize_pt2e import create_getattr_from_value

logger = logging.getLogger(__name__)

__all__ = [
    "eliminate_reshape_with_no_effect",
    "normalize_conv2d_layout",
    "normalize_gemm_weight_layout",
]

AXES_ARG_INDEX_MAP = {
    torch.ops.quantized_ops.dequantize.default: 3,
    torch.ops.quantized_ops.quantize.default: 3,
    torch.ops.quantized_ops.quantize_mx.default: 2,
}


def extract_conv2d_graph(
    model: GraphModule, start: Node, visited: Set[Node]
) -> List[Node]:
    """
    Depth-first worklist traversal over both consumers (users) and
    producers (input nodes), restricted to reshape/elementwise/transpose
    /indexing/quantization ops that are considered fusable.
    """

    # ``slice`` / ``cat`` are what L2 tiling emits when it splits a conv along
    # the reduction dim and concatenates the partial outputs.  Both preserve
    # rank and carry an int ``dim`` at args[1], so their layout can be remapped
    # (see ``_rewrite_node_args_for_layout``).  ``stack`` (adds a dim),
    # ``select`` (drops a dim) and ``index`` (args[1] is not a dim) must stay
    # out of the island.
    ALLOW_LIST_OPS = {
        torch.ops.aten.pad.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.cat.default,
        torch.ops.quantized_ops.quantize_mx.default,
        torch.ops.quantized_ops.conv2d_mx.default,
    }

    def should_traverse(node: Node) -> bool:
        # Only include reshape if the input is a 4D tensor
        if is_reshape_op(node):
            return len(node.shape) == 4

        if node.target == operator.getitem:
            src = node.args[0]
            return src.target == torch.ops.quantized_ops.quantize_mx.default

        return (
            node.target in NHWC_OP_VARIANTS
            or node.target in ALLOW_LIST_OPS
            or is_elementwise_op(node)
        )

    stack = [start]
    nodes_in_graph = set()

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        nodes_in_graph.add(node)

        adjacent_nodes = list(node.users) + list(node.all_input_nodes)

        for n in adjacent_nodes:
            if n not in visited and should_traverse(n):
                stack.append(n)

    node_to_idx = {n: i for i, n in enumerate(model.graph.nodes)}
    return sorted(nodes_in_graph, key=lambda n: node_to_idx[n])


def remap_pad_after_permute(
    pad: Tuple[int, ...], dims: Tuple[int, ...], ndim: int
) -> Tuple[int, ...]:
    """
    Remap padding after permuting a tensor.

    Args:
        pad: Original pad tuple as in torch.nn.functional.pad (starts from last dim).
        dims: Permutation dimensions.
        ndim: Number of dimensions in the original tensor.

    Returns:
        Tuple[int, ...]: New pad tuple corresponding to permuted tensor.
    """
    # number of padded dimensions
    k = len(pad) // 2
    assert k <= ndim, "Pad dimensions exceed tensor dimensions"

    # original padded dims (from last to first)
    original_padded_dims = list(range(ndim - k, ndim))

    dim_to_new_index = {d: dims.index(d) for d in range(ndim)}

    new_pad_pairs = {i: (0, 0) for i in range(ndim)}

    # Assign padding for dimensions that were originally padded
    for i, orig_dim in enumerate(reversed(original_padded_dims)):
        left = pad[2 * i]
        right = pad[2 * i + 1]
        new_pad_pairs[dim_to_new_index[orig_dim]] = (left, right)

    # Collect pads in reverse order (last-first)
    new_pad = []
    for i in sorted(new_pad_pairs.keys(), reverse=True):
        new_pad.extend(new_pad_pairs[i])

    return tuple(new_pad)


def _process_conv2d_input_nodes(
    node: Node, model: GraphModule, island_set: Set[Node]
):
    graph = model.graph

    # Case A: Input is a weight (Parameter) or weight scale.
    if node.op == "get_attr":
        user = next((n for n in node.users if is_conv2d(n)), None)
        if user is None or is_depthwise_conv(user):
            return

        w = get_arg_value(user, 1, "weight")
        ws = user.kwargs.get("weight_scale")
        if node not in [w, ws]:
            return

        logger.debug(f"Permuting {user} parameter: {node}")
        param = fetch_attr(model, node.target)
        param.data = param.data.permute(2, 3, 1, 0)

        node.meta["dims"] = WEIGHT_NCHW_TO_HWIO
        return

    # Case B: Input is an activation flowing into the island from outside
    if len(node.shape) == 4:
        logger.debug(f"Insert permute after {node} with dims {NCHW_TO_NHWC}")
        with graph.inserting_after(node):
            permute_node = graph.call_function(
                torch.ops.aten.permute.default,
                (node, NCHW_TO_NHWC),
            )

        permute_node.meta["dims"] = NCHW_TO_NHWC
        permute_node.meta["dtype"] = node.meta.get("dtype")

        for user in list(node.users.keys()):
            if user in island_set:
                user.replace_input_with(node, permute_node)


def _rewrite_node_args_for_layout(node: Node) -> None:
    input_dims = node.all_input_nodes[0].meta.get("dims")
    node.meta["dims"] = input_dims

    args = tuple(node.args)

    if node.target == torch.ops.aten.pad.default:
        pad = remap_pad_after_permute(args[1], input_dims, node.value.ndim)
        node.update_arg(1, pad)

    if node.target in (torch.ops.aten.slice.Tensor, torch.ops.aten.cat.default):
        dim = get_arg_value(node, 1, "dim", 0)
        if dim < 0:
            dim = dim + len(input_dims)
        node.update_arg(1, input_dims.index(dim))

    if is_reshape_op(node):
        if node.target == torch.ops.aten.transpose.int:
            dims = (args[1], args[2])
        else:
            dims = args[1]
        dims = [d + max(input_dims) + 1 if d < 0 else d for d in dims]
        dims = tuple(input_dims.index(d) for d in dims)
        node.update_arg(1, dims)

    idx = AXES_ARG_INDEX_MAP.get(node.target)
    if idx is not None and idx < len(args) and args[idx] is not None:
        axes = [a + len(input_dims) if a < 0 else a for a in args[idx]]
        axes = tuple(input_dims.index(a) for a in axes)
        node.update_arg(idx, axes)

    if node.target in NHWC_OP_VARIANTS:
        node.target = NHWC_OP_VARIANTS[node.target]
        node.meta["transposed"] = True
    elif node.target is torch.ops.quantized_ops.conv2d_mx.default:
        node.kwargs = {**node.kwargs, "layout": "nhwc"}
        node.meta["transposed"] = True

    def update_shape(d, key, order):
        if key in d:
            d[key] = tuple(d[key][i] for i in order)

    shapes = node.meta.get("tiled_shapes")
    if shapes is not None and (is_pooling(node) or is_conv2d(node)):
        for key in ("input", "input_scale", "output"):
            update_shape(shapes, key, NCHW_TO_NHWC)
        if not is_depthwise_conv(node):
            update_shape(shapes, "weight", WEIGHT_NCHW_TO_HWIO)
            update_shape(shapes, "weight_scale", WEIGHT_NCHW_TO_HWIO)
        update_shape(node.meta, "l2_tiling", NCHW_TO_NHWC)
        if stride := node.meta.get("tile_strides"):
            update_shape(stride, "input", NCHW_TO_NHWC)
            update_shape(stride, "input_scale", NCHW_TO_NHWC)


def normalize_conv2d_layout(model: GraphModule):
    graph = model.graph
    visited_nodes: Set[Node] = set()

    for node in list(graph.nodes):
        if node in visited_nodes or (
            node.target not in NHWC_OP_VARIANTS
            and node.target != torch.ops.quantized_ops.conv2d_mx.default
        ):
            continue

        # Extract the cluster of nodes that can share the NHWC layout
        island_nodes = extract_conv2d_graph(model, node, visited_nodes)
        island_set = set(island_nodes)

        for node_to_treat in island_nodes:
            # Inspect inputs to see if they come from outside the island (NCHW)
            for input_node in list(node_to_treat.all_input_nodes):
                if input_node in island_set or "dims" in input_node.meta:
                    continue

                _process_conv2d_input_nodes(input_node, model, island_set)

            for user in list(node_to_treat.users.keys()):
                if user in island_set or "dims" in user.meta:
                    continue

                logger.debug(
                    f"Insert permute before {user} with dims (0, 3, 1, 2)"
                )
                with graph.inserting_before(user):
                    permute_node = graph.call_function(
                        torch.ops.aten.permute.default,
                        (node_to_treat, NHWC_TO_NCHW),
                    )
                permute_node.meta["dtype"] = node_to_treat.meta.get("dtype")
                user.replace_input_with(node_to_treat, permute_node)

            _rewrite_node_args_for_layout(node_to_treat)

    graph.lint()
    model.recompile()
    return model


def eliminate_reshape_with_no_effect(model: GraphModule):
    deleted_nodes = set()
    for node in list(model.graph.nodes):
        if not is_reshape_op(node) or node in deleted_nodes:
            continue

        curr_node = node
        input_node = node.all_input_nodes[0]

        group = []
        while len(curr_node.users) == 1 and (
            is_reshape_op(curr_node) or is_nop(curr_node)
        ):
            group.append(curr_node)
            curr_node = next(iter(curr_node.users))

        val = input_node.value
        orig_x = torch.arange(val.numel(), dtype=torch.int32)

        x = orig_x.reshape(val.shape)

        last_valid_idx = -1

        for i, gn in enumerate(group):
            args = torch.fx.graph.map_arg(gn.args, lambda n: x)
            x = gn.target(*args)

            if torch.equal(x.reshape(-1), orig_x):
                last_valid_idx = i

        del group[last_valid_idx + 1 :]

        if len(group) <= 1:
            continue

        logger.debug(f"Eliminating reshape group: {[n.name for n in group]}")

        output_shape = group[-1].value.shape

        with model.graph.inserting_before(node):
            reshape_node = model.graph.call_function(
                torch.ops.aten.reshape.default,
                (input_node, output_shape),
            )

        propagate_shape(reshape_node)

        group[-1].replace_all_uses_with(reshape_node)

        for n in reversed(group):
            model.graph.erase_node(n)
            deleted_nodes.add(n)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


ALLOWED_UPSTREAM_OPS: Set[any] = {
    torch.ops.aten.select.int,
    torch.ops.quantized_ops.dequantize.default,
    torch.ops.quantized_ops.quantize.default,
    torch.ops.quantized_ops.quantize_mx.default,
}


def is_transpose_2d(node: torch.fx.Node) -> bool:
    """Checks if node is a transpose on the last two dimensions."""
    if node.target != torch.ops.aten.transpose.int:
        return False
    # Check args to ensure it is specifically swapping -2 and -1
    # args format: (input, dim0, dim1)
    rank = len(node.shape)
    dims = set(d if d >= 0 else rank + d for d in node.args[1:])
    return dims == {rank - 2, rank - 1}


def find_upstream_transpose_or_param(
    node: torch.fx.Node,
    *,
    max_depth: int = 16,
) -> Optional[List[torch.fx.Node]]:
    """
    Starting from a transpose node, walks upstream through a list
    of allowed operations until reaching another transpose node
    or a constant param node.
    """
    if not is_transpose_2d(node):
        return None

    def dfs(curr: Node, depth: int) -> Optional[List[Node]]:
        if is_transpose_2d(curr):
            return [curr]

        if curr.op == "get_attr" and require_allocation(curr):
            return [curr]

        if depth > max_depth:
            return None

        path = []
        if curr.target in ALLOWED_UPSTREAM_OPS or is_nop(curr):
            for inp in curr.all_input_nodes:
                path.extend(dfs(inp, depth + 1) or [])

        return [curr] + path if path else None

    if (found_path := dfs(node.args[0], 0)) is None:
        return None

    return list(set([node] + found_path))


def _insert_transposed_input(arg: Node, model: GraphModule):
    with model.graph.inserting_after(arg):
        if arg.op == "get_attr":
            value = fetch_attr(model, arg.target)
            transposed = create_getattr_from_value(
                model, model.graph, arg.name + "_T", value.mT
            )
        else:
            transposed = model.graph.call_function(
                torch.ops.aten.transpose.int, (arg, -2, -1)
            )
    transposed.meta["dtype"] = arg.meta.get("dtype")
    return transposed


def _fix_axes_after_transpose(node: Node) -> List[int]:
    if (index := AXES_ARG_INDEX_MAP.get(node.target)) is None:
        return

    axes = get_arg_value(node, index, "axes")
    rank = len(node.shape)

    # Build forward and inverse permutation for transpose(-2, -1)
    perm = list(range(rank))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    inv_perm = [perm.index(i) for i in range(rank)]

    # Normalize negative axes and apply inverse permutation
    norm_axes = [(a + rank) % rank for a in axes]
    node.update_arg(index, tuple(inv_perm[a] for a in norm_axes))


def eliminate_canceling_transposes(
    model: GraphModule,
    chain: List[Node],
    transposed_nodes: Dict[Node, Node],
) -> bool:
    """
    Optimizes a chain like [select_3, select_2, quantize_default_1, transpose_3]
    when there's a matching matmul-side transpose (user of chain[0]).

    Steps:
      1. Check if the two transposes cancel (considering selects).
      2. If yes, detach intermediate nodes and remove the redundant transpose.

    Returns:
        bool: True if optimization was applied, else False.
    """
    graph = model.graph

    up_t = chain[0]
    down_t = chain[-1]
    if up_t.target != torch.ops.aten.transpose.int:
        return False

    # Ensure selects are on first dimension only
    selects = [n for n in chain if n.target == torch.ops.aten.select.int]
    rank = len(up_t.shape)
    if rank < len(selects) + 2 or any(n.args[1] != 0 for n in selects):
        return False

    # We don't need to duplicate the upstream transpose node
    chain = duplicate_shared_nodes(graph, chain[1:])

    # Rewrite graph to remove cancelling transposes
    for n in chain:
        for arg in n.all_input_nodes:
            if arg == up_t:
                n.replace_input_with(up_t, up_t.args[0])
                continue
            if arg in chain or arg.value.ndim < 2:
                continue
            if arg not in transposed_nodes:
                transposed_nodes[arg] = _insert_transposed_input(arg, model)
            n.replace_input_with(arg, transposed_nodes[arg])
        _fix_axes_after_transpose(n)

    down_t.replace_all_uses_with(down_t.args[0])
    graph.erase_node(down_t)

    if not up_t.users:
        graph.erase_node(up_t)

    logger.info(f"Eliminated redundant transposes: {up_t} and {down_t}")

    return True


def fold_transpose_into_constant(
    model: GraphModule,
    chain: List[Node],
    transposed_nodes: Dict[Node, Node],
) -> bool:
    graph = model.graph

    attr_node = chain[0]
    down_t = chain[-1]
    if attr_node.op != "get_attr":
        return False

    # Ensure selects are on first dimension only
    selects = [n for n in chain if n.target == torch.ops.aten.select.int]
    rank = len(attr_node.shape)
    if rank < len(selects) + 2 or any(n.args[1] != 0 for n in selects):
        return False

    # We don't need to duplicate the transpose node
    chain = duplicate_shared_nodes(model.graph, chain[1:])

    for n in chain:
        for arg in n.all_input_nodes:
            if arg in chain or arg.value.ndim < 2:
                continue
            if arg not in transposed_nodes:
                transposed_nodes[arg] = _insert_transposed_input(arg, model)
            n.replace_input_with(arg, transposed_nodes[arg])

    down_t.replace_all_uses_with(down_t.args[0])
    graph.erase_node(down_t)

    if not attr_node.users:
        graph.erase_node(attr_node)

    logger.info(f"Folded {down_t} into constant: {attr_node}")

    return True


def _insert_transpose_op(
    model: GraphModule, node: Node, user: Node, transposed_nodes: dict
) -> Optional[List[Node]]:
    """Inserts a transpose operation before the user node.

    A ``dequantize`` on the way in decodes element by element, so it is the same
    tensor either side of it -- but the transpose goes *above* it, because a
    decode the GEMM reads through is one it folds into its own fetch, where one
    it reads from is an op with a buffer of its own.  Its scale and zero point
    turn with it, and the axes it blocks along are restated.
    """
    # The legacy path splits the heads before the decode, so the GEMM's operand
    # arrives through a chain of ``select``s on the leading dims.
    chain = [node]
    while chain[-1].target == torch.ops.aten.select.int:
        chain.append(chain[-1].args[0])

    dq = chain[-1]
    selects = [n for n in chain if n.target == torch.ops.aten.select.int]
    if dq.target != torch.ops.quantized_ops.dequantize.default or any(
        n.args[1] != 0 for n in selects
    ):
        dq = None
    else:
        # Every head shares the decode, so give this GEMM one of its own to
        # turn; once all of them are turned the copies are identical again and
        # ``deduplicate_nodes`` folds them back.
        chain = duplicate_shared_nodes(model.graph, chain)
        dq = chain[0]

    # Transpose what the decode reads, or what the GEMM reads if there is none.
    src, consumer = (dq.args[0], dq) if dq is not None else (node, user)

    with model.graph.inserting_before(consumer):
        new_node = model.graph.call_function(
            torch.ops.aten.transpose.int, (src, -2, -1)
        )

    new_node.meta["dtype"] = src.meta.get("dtype")
    propagate_shape(new_node, model)
    consumer.replace_input_with(src, new_node)

    if dq is not None:
        for arg in list(dq.all_input_nodes):
            if arg is new_node or arg.value.ndim < 2:
                continue
            if arg not in transposed_nodes:
                transposed_nodes[arg] = _insert_transposed_input(arg, model)
                propagate_shape(transposed_nodes[arg], model)
            dq.replace_input_with(arg, transposed_nodes[arg])
        _fix_axes_after_transpose(dq)
        # The decode -- and every select over it -- now hands on a transposed
        # tensor, which is the one the GEMM wanted.
        for n in chain:
            propagate_shape(n, model)

    path = find_upstream_transpose_or_param(new_node)
    if path is None or len(path) < 2:
        return

    node_order = {n: i for i, n in enumerate(model.graph.nodes)}
    path = sorted(path, key=lambda n: node_order[n])

    eliminate_canceling_transposes(model, path, transposed_nodes)
    fold_transpose_into_constant(model, path, transposed_nodes)


# The two GEMM weight layouts: ``"kc"`` = [out, contraction] (aten
# linear's native weight layout), ``"ck"`` = [contraction, out] (aten
# matmul's native right-operand layout).
_GEMM_WEIGHT_LAYOUTS = ("kc", "ck")


def _node_layout(node: Node, mm_layout: str, mv_layout: str) -> str:
    """The target layout of ``node``'s class: matrix-vector
    (fully-connected / batch-1) nodes follow ``mv_layout``, matrix-matrix
    nodes ``mm_layout``."""
    return mv_layout if is_fully_connected(node) else mm_layout


def _update_shapes(node: Node) -> None:
    """Updates the tiled_shapes metadata for a transposed node."""
    if (tiled_shapes := node.meta.get("tiled_shapes")) is None:
        return

    for key in ["weight", "other", "weight_scale"]:
        if key in tiled_shapes:
            d0, d1 = tiled_shapes[key]
            tiled_shapes[key] = (d1, d0)


def _process_linear_node(model: GraphModule, node: Node) -> None:
    """Flip a linear's KC-native weight storage to CK and retarget the
    node to the layout twin, whose weight arrives swapped.  The caller
    decides which nodes to flip."""
    logger.info(f"Transposing weight for linear node {node.name}")

    weight_node = node.args[1]
    weight = fetch_attr(model, weight_node.target)
    weight.data = weight.data.T

    if (scale_node := node.kwargs.get("weight_scale")) is not None:
        scale = fetch_attr(model, scale_node.target)
        scale.data = scale.data.T

    # Mark mx / spmm_csr users as consuming a CK-stored weight.
    for user in list(weight_node.users):
        if user.target in [
            torch.ops.quantized_ops.linear_mx.default,
            torch.ops.quantized_ops.spmm_csr.default,
        ]:
            user.kwargs = {**user.kwargs, "weight_layout": "ck"}

    if node.target == torch.ops.aten.linear.default:
        node.target = torch.ops.quantized_ops.linear.default

    _update_shapes(node)
    node.meta["transposed"] = True


def _process_matmul_node(
    model: GraphModule, node: Node, transposed_nodes: dict
) -> None:
    """Flip a matmul's CK-native right operand storage to KC and
    retarget the node to the layout twin, whose operand arrives swapped.
    The caller decides which nodes to flip."""
    logger.info(f"Transposing weight for matmul node {node.name}")

    weight_node = node.args[1]
    _insert_transpose_op(model, weight_node, node, transposed_nodes)

    if (scale_node := node.kwargs.get("weight_scale")) is not None:
        _insert_transpose_op(model, scale_node, node, transposed_nodes)

    if node.target == torch.ops.aten.matmul.default:
        node.target = torch.ops.quantized_ops.matmul.default
    elif node.target == torch.ops.quantized_ops.matmul_mx.default:
        node.kwargs = {**node.kwargs, "weight_layout": "kc"}

    _update_shapes(node)
    node.meta["transposed"] = True


def normalize_gemm_weight_layout(
    model: GraphModule, mm_layout: str = "kc", mv_layout: str = "kc"
) -> GraphModule:
    """Normalize every GEMM right-operand (weight) to one layout per
    operation class: matrix-matrix nodes to ``mm_layout``, matrix-vector
    (fully-connected / batch-1) nodes to ``mv_layout``; each is ``"kc"``
    ([out, contraction]) or ``"ck"`` ([contraction, out]).

    ``aten.linear`` is KC-native and ``aten.matmul`` CK-native, so
    exactly the nodes whose class targets the other layout have their
    weight storage flipped and are retargeted to the ``layout_ops``
    twins (same name/schema, second operand arrives swapped).  A node is
    retargeted iff its storage was flipped, so eager execution stays
    correct in every configuration with no global patching.
    """
    assert mm_layout in _GEMM_WEIGHT_LAYOUTS, mm_layout
    assert mv_layout in _GEMM_WEIGHT_LAYOUTS, mv_layout

    transposed_nodes = {}

    for node in list(model.graph.nodes):
        # A node is flipped exactly when its class's target layout differs
        # from the op's native one (linear: KC, matmul: CK).
        if is_linear(node):
            if _node_layout(node, mm_layout, mv_layout) == "ck":
                _process_linear_node(model, node)
        elif is_matmul(node):
            if _node_layout(node, mm_layout, mv_layout) == "kc":
                _process_matmul_node(model, node, transposed_nodes)

    deduplicate_nodes(model)

    model.graph.lint()
    model.recompile()
    return model
