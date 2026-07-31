"""Operator fusion: group a compute op with the relayout / dequantize ops
around it into one fused ``call_module``, so the accelerator reads and
writes the tensor once instead of once per op.
"""

import logging
from collections import defaultdict
from itertools import permutations
from typing import Any, Callable, Dict, List, Union

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import (
    dtype_byte_size,
    get_arg_value,
    is_elementwise_op,
    is_gemm_op,
    is_mha_qkv_permute,
    is_nop,
    is_prunable_op,
    is_reshape_op,
    repeat_of,
    swaps_last_two_dims,
)
from voyager_compiler.codegen.subgraph import (
    create_and_insert_subgraph,
    update_submod_user_meta,
)
from voyager_compiler.shape_prop import propagate_shape

logger = logging.getLogger(__name__)


def _nodes_sequential(
    nodes: List[Node], order: Dict[Node, int], hoisted: bool = True
) -> bool:
    """Whether ``nodes`` can run as one group: each a user of the one before it,
    and a ``dequantize`` only ever sitting on a GEMM.

    ``hoisted`` says the caller runs the group up where its *first* node stands,
    so an operand read from outside the chain has to be computed above that
    point as well.  A group that instead stays where its *last* node stands --
    what ``create_and_insert_subgraph`` does -- has every operand it needs by
    construction, since each one precedes the node consuming it and that node
    precedes the group.
    """
    prev_node = None
    for n in nodes:
        # Check if the current node is a user of the previous node
        if prev_node is not None and n not in prev_node.users:
            return False
        # We only handle dequantize after GEMM here
        if (
            n.target == torch.ops.quantized_ops.dequantize.default
            and not is_gemm_op(n.args[0])
        ):
            return False
        if hoisted and prev_node is not None:
            for arg in n.all_input_nodes:
                if id(arg) == id(prev_node):
                    continue

                while is_nop(arg):
                    arg = arg.all_input_nodes[0]

                if arg.op == "call_function" and order[arg] > order[prev_node]:
                    return False
        prev_node = n
    return True


def find_sequential_nodes_(
    pattern: List[List[Callable]],
    order: Dict[Node, int],
    nodes_by_source: Dict[Callable, List[Node]],
):
    def get_matched_nodes(matcher):
        return [
            n
            for tgt in matcher.targets
            for n in nodes_by_source[tgt]
            if matcher.matches(n)
        ]

    def collect_nop_chain(node):
        nops = []
        cur = next(iter(node.users), None)
        while cur and is_nop(cur) and len(cur.users) == 1:
            nops.append(cur)
            cur = next(iter(cur.users), None)
        return nops

    fused_chain = []
    fused_nodes = set()
    singleton_nodes = set(get_matched_nodes(pattern[0]))

    for stage_sources in pattern[1:]:
        stage_nodes = [
            n for n in get_matched_nodes(stage_sources) if n not in fused_nodes
        ]
        if not stage_nodes:
            continue

        fusion_candidates = fused_chain + [
            [n] for n in sorted(singleton_nodes, key=order.__getitem__)
        ]
        new_chains = []
        for nodes in fusion_candidates:
            if len(nodes) == 1 and nodes[0] in fused_nodes:
                continue

            last_node = nodes[-1]

            # Skip if the last node has multiple users
            if len(last_node.users) > 1:
                if len(nodes) > 1:
                    new_chains.append(nodes)
                continue

            nops = collect_nop_chain(last_node)
            matched = False

            for node in list(stage_nodes):
                if node in fused_nodes or order[node] < order[last_node]:
                    continue
                candidate = nodes + nops + [node]
                if _nodes_sequential(candidate, order, hoisted=False):
                    new_chains.append(candidate)
                    fused_nodes.update(candidate)
                    matched = True
                    break

            if not matched and len(nodes) > 1:
                new_chains.append(nodes)

        fused_chain = new_chains
        singleton_nodes = (singleton_nodes | set(stage_nodes)) - fused_nodes

    return fused_chain


def _materialize_bytes(node: Node) -> int:
    """DRAM an output costs if it has to live in memory: one write, plus one
    read per consumer, weighted by dtype -- an NF4 tensor is an eighth of the
    same shape in bf16, so a group boundary landing on a quantized value is
    cheap to leave materialized.
    """
    val = getattr(node, "value", None)
    tensors = list(val) if isinstance(val, (list, tuple)) else [val]
    dtypes = node.meta.get("dtype")
    if not isinstance(dtypes, (list, tuple)):
        dtypes = [dtypes] * len(tensors)
    size = 0
    for t, d in zip(tensors, dtypes):
        if isinstance(t, torch.Tensor):
            size += t.numel() * dtype_byte_size(d if d is not None else t.dtype)
    return size * (1 + len(node.users))


def _saved_bytes(groups: List[List[Node]]) -> int:
    """DRAM these groups keep off the bus.  A node is free only if it is
    interior (some later node in its group consumes it) *and* every one of its
    users is inside that group -- otherwise its output still materializes for
    the outside reader.  Each group's last node is the boundary and stays.
    """
    total = 0
    for group in groups:
        members = set(group)
        for node in group[:-1]:
            if all(u in members for u in node.users):
                total += _materialize_bytes(node)
    return total


def _commit(order: List[List[Node]]) -> List[List[Node]]:
    """Keep each chain in ``order``, truncated at the first node an earlier one
    already took (a prefix of a chain is still a chain); drop what shrinks to a
    single node."""
    kept: List[List[Node]] = []
    taken: set = set()
    for group in order:
        prefix = []
        for node in group:
            if node in taken:
                break
            prefix.append(node)
        if len(prefix) > 1:
            kept.append(prefix)
            taken.update(prefix)
    return kept


def select_groups_min_dram(
    all_candidates: List[List[Node]],
) -> List[List[Node]]:
    """Resolve chains that share a node by keeping the node where it removes the
    most DRAM.

    A node claimed by several chains can stay in only one; the others truncate
    to the prefix that led up to it.  Which chain keeps it decides which values
    stay on-chip and which not -- fusing a GEMM's output into the next op
    keeps that large output off the bus, and a boundary on a quantized value is
    cheap -- so pick the assignment that keeps the most bytes on-chip
    (``_saved_bytes``, dtype-weighted).

    Chains that transitively share a node form a cluster.  A small cluster tries
    every commit order and takes the one whose survivors save the most; a large
    one falls back to a single greedy order (most-saving, GEMM-anchored first).
    """
    node_to_groups: Dict[Node, List[int]] = defaultdict(list)
    for i, group in enumerate(all_candidates):
        for node in group:
            node_to_groups[node].append(i)

    parent = list(range(len(all_candidates)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for shared in node_to_groups.values():
        for j in shared[1:]:
            parent[find(j)] = find(shared[0])

    clusters: Dict[int, List[List[Node]]] = defaultdict(list)
    for i, group in enumerate(all_candidates):
        clusters[find(i)].append(group)

    final: List[List[Node]] = []
    for members in clusters.values():
        if len(members) == 1:
            final.extend(_commit(members))
        elif len(members) <= 6:
            final.extend(
                max(map(_commit, permutations(members)), key=_saved_bytes)
            )
        else:
            greedy = sorted(
                members,
                key=lambda g: (
                    _saved_bytes([g]),
                    any(is_gemm_op(n) for n in g),
                ),
                reverse=True,
            )
            final.extend(_commit(greedy))
    return final


def find_sequential_nodes(model: GraphModule, patterns: List[List[List[Any]]]):
    graph = model.graph
    nodes_order = {node: i for i, node in enumerate(graph.nodes)}

    all_sources = {
        op for pattern in patterns for stage in pattern for op in stage.targets
    }

    nodes_by_source = defaultdict(list)
    for node in list(graph.nodes):
        if node.target in all_sources:
            nodes_by_source[node.target].append(node)

    all_candidates = []
    for pattern in patterns:
        candidates = find_sequential_nodes_(
            pattern, nodes_order, nodes_by_source
        )
        all_candidates.extend(candidates)

    return select_groups_min_dram(all_candidates)


def search_group(node, node_lists):
    for l in node_lists:
        if node in l:
            return l
    return None


def duplicate_shared_nodes(
    graph: torch.fx.Graph, nodes: List[Node]
) -> List[Node]:
    """
    Ensures that nodes in the given list are independent by duplicating any
    node that has multiple users.

    This function processes the given list of nodes in topological order,
    identifying any node with multiple users. If such a node exists, it is
    duplicated so that all nodes in the list can be grouped together without
    affecting other nodes in the DAG.

    Args:
        graph (torch.fx.Graph): The FX graph being processed.
        nodes (List[Node]): A list of nodes to check for shared usage.

    Returns:
        List[Node]: A new list where shared nodes have been duplicated.
    """
    nodes_order = {node: idx for idx, node in enumerate(graph.nodes)}
    nodes = sorted(nodes, key=lambda n: nodes_order[n])

    for i in reversed(range(len(nodes) - 1)):
        node, next_node = nodes[i], nodes[i + 1]

        if len(node.users) == 1:
            continue

        with graph.inserting_before(next_node):
            new_node = graph.node_copy(node, lambda n: n)
        propagate_shape(new_node)

        next_node.replace_input_with(node, new_node)
        nodes[i] = new_node

    return nodes


def move_transpose_after_select(graph: torch.fx.Graph, nodes: List[Node]):
    transpose_node = nodes[0]

    select_nodes = [n for n in nodes if n.target == torch.ops.aten.select.int]
    chain = [transpose_node] + select_nodes
    for n, next_n in zip(chain[:-1], chain[1:]):
        if next_n not in n.users or next_n.args[1] != 0:
            return nodes

    if len(select_nodes) == 0:
        return nodes

    user_node = next(iter(select_nodes[-1].users))
    ndim = transpose_node.value.ndim
    dims = [
        (x + ndim if x < 0 else x) - len(select_nodes)
        for x in transpose_node.args[1:]
    ]

    with graph.inserting_before(user_node):
        new_node = graph.call_function(
            torch.ops.aten.transpose.int,
            (select_nodes[-1], *dims),
        )

    user_node.replace_input_with(select_nodes[-1], new_node)
    select_nodes[0].replace_input_with(transpose_node, transpose_node.args[0])
    graph.erase_node(transpose_node)

    for n in select_nodes + [new_node]:
        propagate_shape(n)

    nodes = [n for n in nodes if n not in select_nodes and n != transpose_node]
    nodes.insert(0, new_node)
    return nodes


def _fuse_reshape_with_input_impl(
    graph: torch.fx.Graph,
    candidates: List[List[Node]],
    nodes_map: Dict[Node, Node],
    current_node: Node,
    fused_nodes: List[Node],
    simulate: bool = False,
) -> Union[bool, List[Node]]:
    reshape_node = fused_nodes[0]
    fused_nodes.append(current_node)

    can_fuse = False
    is_transpose = swaps_last_two_dims(reshape_node)
    # TODO: support fusing reshape with elementwise operations
    if is_gemm_op(current_node):
        can_fuse = is_transpose and fused_nodes[-2] in (
            current_node.args[1],
            current_node.kwargs.get("weight_scale"),
            current_node.kwargs.get("other_scale"),
        )

    if can_fuse:
        if simulate:
            return True
        fused_nodes = duplicate_shared_nodes(graph, fused_nodes)
        fused_nodes = move_transpose_after_select(graph, fused_nodes)
        nodes_map[fused_nodes[0]] = fused_nodes[-2]
        if (group := search_group(current_node, candidates)) is not None:
            group.extend(n for n in fused_nodes if n not in group)
        else:
            candidates.append(fused_nodes)
        return [fused_nodes[0]]
    else:
        logger.warning(f"Cannot fuse {reshape_node} with {current_node}")

    if not is_nop(current_node) and not (
        is_transpose
        and current_node.target == torch.ops.aten.select.int
        and current_node.args[1] == 0
    ):
        logger.info(f"Cannot fuse {reshape_node} with {current_node}")
        return False if simulate else []

    all_results = []
    for user in list(current_node.users):
        result = _fuse_reshape_with_input_impl(
            graph,
            candidates,
            nodes_map,
            user,
            list(fused_nodes),
            simulate,
        )
        if simulate:
            if not result:
                return False
        else:
            all_results.extend(result)

    return True if simulate else all_results


def fuse_reshape_with_input(
    graph: torch.fx.Graph,
    candidates: List[List[Node]],
    nodes_map: Dict[Node, Node],
    reshape_node: Node,
):
    # First pass: simulate fusion to ensure all users can be fused
    for user in list(reshape_node.users):
        result = _fuse_reshape_with_input_impl(
            graph,
            candidates,
            nodes_map,
            user,
            [reshape_node],
            simulate=True,
        )
        if not result:
            logger.info(
                f"Skipping fusion for {reshape_node} due to unfusable path"
            )
            return

    # Second pass: perform actual fusion
    for user in list(reshape_node.users):
        result = _fuse_reshape_with_input_impl(
            graph,
            candidates,
            nodes_map,
            user,
            [reshape_node],
            simulate=False,
        )


def _tile_holds_whole_heads(gemm_node: Node, reshape_node: Node) -> bool:
    """Whether ``gemm_node``'s N tile is a whole number of heads.

    A projection's ``N`` is really ``(heads, head_dim)``, and the fused relayout
    stores the tile straight to the permuted block -- which it can only do if no
    head straddles a tile boundary.  Leaving the reshape unfused when it does
    costs a separate permute, but keeps the op lowerable.  An untiled gemm (or
    one interstellar tiles later) has no N tile to check yet.
    """
    if len(reshape_node.shape) <= len(gemm_node.shape):
        return True  # context matmul: keeps its rank, no head packing

    tiling = gemm_node.meta.get("l2_tiling")
    if tiling is None:
        return True

    head_dim = reshape_node.shape[-1]
    n_tile = gemm_node.shape[-1] // tiling[1]
    return n_tile % head_dim == 0


def fuse_reshape_with_output(
    graph: torch.fx.Graph,
    candidates: List[List[Node]],
    nodes_map: Dict[Node, Node],
    reshape_node: Node,
) -> bool:
    if not is_mha_qkv_permute(reshape_node):
        return False

    curr_node = reshape_node.all_input_nodes[0]
    fused_nodes = [reshape_node]

    while not (is_gemm_op(curr_node) or is_elementwise_op(curr_node)):
        if (
            len(curr_node.users) > 1
            or len(curr_node.all_input_nodes) != 1
            or not is_nop(curr_node)
        ):
            logger.debug(f"Cannot fuse {reshape_node} with {curr_node}")
            return False

        fused_nodes.insert(0, curr_node)
        curr_node = curr_node.all_input_nodes[0]

    if len(curr_node.users) > 1:
        return False

    if not _tile_holds_whole_heads(curr_node, reshape_node):
        logger.debug(
            f"Cannot fuse {reshape_node} with {curr_node}: "
            f"N tile would split a head"
        )
        return False

    # A microscaling quantize reading the permute comes along as the group's
    # last op: the tile is quantized on its way out of the GEMM, rather than
    # written, read back and written again.  Nothing sits between the two.
    users = list(reshape_node.users)
    if (
        len(users) == 1
        and users[0].target is torch.ops.quantized_ops.quantize_mx.default
        and users[0].args[0] is reshape_node
        and search_group(users[0], candidates) is None
    ):
        fused_nodes.append(users[0])

    group = search_group(curr_node, candidates)

    if group is not None:
        group.extend(n for n in fused_nodes if n not in group)
    else:
        candidates.append([curr_node, *fused_nodes])

    nodes_map[reshape_node] = curr_node

    return True


def fuse_repeat_with_input(graph, candidates, node) -> bool:
    """Fuse a repeated tensor into the GEMM that reads it.

    Grouped-query attention hands the attention matmul 8 KV heads blown up to
    32.  In the group, the ops that repeat them are not emitted: the builder
    reads head ``h // n_rep`` of the un-repeated tensor (``_InputSpec.repeat``).

    Traced from the repeat down, not from the GEMM up: a repeated tensor need
    not reach it as an operand.  A KIVI cache is decoded on the way in, so its
    scale and zero point repeat into the ``dequantize`` instead -- which, like a
    transpose, folds into the fetch and is crossed.

    Bufferized only: the legacy path copies an ``expand`` into memory, so it
    never sees one of these groups (hence no ``nodes_map`` entry).
    """
    if (found := repeat_of(node)) is None:
        return False
    _, chain, _ = found

    crossed = []
    curr = node
    while True:
        if len(curr.users) != 1:
            return False
        user = next(iter(curr.users))
        if is_gemm_op(user):
            break
        if swaps_last_two_dims(user):
            crossed.append(user)
        elif user.target != torch.ops.quantized_ops.dequantize.default:
            return False
        curr = user

    chain = [*chain, *crossed]
    if (group := search_group(user, candidates)) is not None:
        group.extend(n for n in chain if n not in group)
    else:
        candidates.append([*chain, user])
    return True


def fuse_dequantize_with_input(graph, candidates, nodes_map, node_to_fuse):
    for user in list(node_to_fuse.users):
        _fuse_dequantize_recursive(
            graph, candidates, nodes_map, user, [node_to_fuse]
        )


def _fuse_dequantize_recursive(
    graph, candidates, nodes_map, current_node, fused_nodes
):
    fused_nodes.append(current_node)

    if (
        is_gemm_op(current_node)
        or is_elementwise_op(current_node)
        or current_node.target
        in [
            torch.ops.aten.layer_norm.default,
            torch.ops.aten.softmax.int,
        ]
    ):
        fused_nodes = duplicate_shared_nodes(graph, fused_nodes)
        fused_nodes = move_dq_after_select(graph, fused_nodes)
        nodes_map[fused_nodes[0]] = fused_nodes[-2]

        if (group := search_group(current_node, candidates)) is not None:
            group.extend(n for n in fused_nodes if n not in group)
        else:
            candidates.append(fused_nodes)
        return

    # A transpose folds into the fetch (the DMA swaps the tile), so the decode
    # still reaches the GEMM that runs it.
    if (
        current_node.target != torch.ops.aten.select.int
        and not is_nop(current_node)
        and not swaps_last_two_dims(current_node)
    ):
        logger.info(f"Cannot fuse {fused_nodes[0]} with {current_node}")
        return

    for user in list(current_node.users):
        _fuse_dequantize_recursive(
            graph, candidates, nodes_map, user, list(fused_nodes)
        )


def move_dq_after_select(graph: torch.fx.Graph, nodes: List[Node]):
    node_to_move = nodes[0]

    for n in nodes[1:]:
        if is_prunable_op(n):
            n.replace_all_uses_with(n.args[0])
            graph.erase_node(n)
            nodes.remove(n)

    # Pick select nodes in the chain
    select_nodes = [n for n in nodes if n.target == torch.ops.aten.select.int]
    chain = [node_to_move] + select_nodes
    for n, next_n in zip(chain[:-1], chain[1:]):
        if next_n not in n.users or next_n.args[1] != 0:
            return nodes

    if len(select_nodes) == 0:
        return nodes

    user_node = next(iter(select_nodes[-1].users))

    def insert_select_ops(arg):
        if arg == node_to_move.args[0]:
            return select_nodes[-1]

        if "qmap" in arg.name or "code" in arg.name:
            return arg

        # Some dims are broadcasted, thus don't need to apply all selects
        for sel_node in select_nodes:
            arg_ndim = arg.value.ndim
            ndim = sel_node.args[0].value.ndim
            if sel_node.args[1] < ndim - arg_ndim:
                continue
            with graph.inserting_before(user_node):
                arg = graph.call_function(
                    torch.ops.aten.select.int,
                    (arg,) + sel_node.args[1:],
                )
            propagate_shape(arg)
            arg.meta["dtype"] = arg.args[0].meta.get("dtype", None)
        return arg

    with graph.inserting_before(user_node):
        new_node = graph.node_copy(node_to_move, insert_select_ops)

    # Update dequantize axes arguments
    axes = get_arg_value(new_node, 3, "axes")
    if axes is not None:
        axes = tuple(a - len(select_nodes) if a >= 0 else a for a in axes)
        new_node.update_arg(3, axes)

    user_node.replace_input_with(select_nodes[-1], new_node)
    select_nodes[0].replace_input_with(node_to_move, node_to_move.args[0])

    if len(node_to_move.users) == 0:
        graph.erase_node(node_to_move)

    for n in select_nodes + [new_node]:
        propagate_shape(n)
        n.meta["dtype"] = n.args[0].meta.get("dtype", None)

    # Respect the order of nodes appearing in the graph
    nodes = [n for n in nodes if n not in select_nodes and n != node_to_move]
    nodes.insert(0, new_node)
    return nodes


def fuse_operator(
    model: GraphModule,
    operations: List[List[Callable]] = None,
    fuse_reshape: bool = True,
):
    """
    Fuse reshape, slicing, and dequantize operations with their immediate users.

    The logic for fusing reshape operations is that we first trace up the graph
    until we reach a GEMM or elementwise operation. If we encounter a reshape operation
    or a node that has either multiple inputs or users along the way, we stop and
    try to fuse the node with its immediate user. For fusing a node with its user,
    we trace down the graph until we reach a GEMM or elementwise operation. If we
    encounter a node with multiple users, we duplicate all the elements on the path
    and perform fusion on each branch.
    """
    graph = model.graph

    nodes_map = {}
    fused_nodes_list = []

    if operations is not None:
        fused_nodes_list = find_sequential_nodes(model, operations)

    for node in list(graph.nodes):
        # Try to fuse MHA QKV permute with preceeding GEMM
        if fuse_reshape_with_output(graph, fused_nodes_list, nodes_map, node):
            continue

        # Attempt to fuse it with its immediate user
        if fuse_reshape and is_reshape_op(node):
            fuse_reshape_with_input(graph, fused_nodes_list, nodes_map, node)

    for node in list(graph.nodes):
        if node.target != torch.ops.quantized_ops.dequantize.default:
            continue

        # If the node is already fused, skip it
        if search_group(node, fused_nodes_list) is not None:
            continue

        fuse_dequantize_with_input(graph, fused_nodes_list, nodes_map, node)

    for node in list(graph.nodes):
        fuse_repeat_with_input(graph, fused_nodes_list, node)

    # Sort nodes based on their order of appearance in the graph
    nodes_order = {node: i for i, node in enumerate(graph.nodes)}
    for nodes in fused_nodes_list:
        nodes.sort(key=lambda n: nodes_order[n])
    fused_nodes_list.sort(key=lambda g: nodes_order[g[-1]])

    nodes_map = {v.name: k.name for k, v in nodes_map.items()}

    for fused_nodes in fused_nodes_list:
        node = create_and_insert_subgraph(
            fused_nodes, model, normalize_iteration_space=True
        )
        if node is None:
            continue
        update_submod_user_meta(model, node)
        propagate_shape(node, model)
        gm = node.meta.get("submodule")

        for n in list(gm.graph.nodes):
            if (name := nodes_map.get(n.name, None)) is None:
                continue

            fused_node = next(iter(n for n in gm.graph.nodes if n.name == name))
            fused_node.meta["fused"] = True

            if is_reshape_op(fused_node):
                if next(iter(fused_node.users)).op == "output":
                    node.meta["reshape"] = fused_node
                else:
                    n.meta["reshape"] = fused_node

            if fused_node.target == torch.ops.quantized_ops.dequantize.default:
                n.meta["dequantize"] = fused_node

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model
