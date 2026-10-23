import logging
import re
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Interpreter, Node
from torch.fx.node import map_arg
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher
from torch.library import Library, impl
from torchao.quantization.pt2e.utils import _get_aten_graph_module_for_pattern

from voyager_compiler.codegen.aten_classifier import is_elementwise_op
from voyager_compiler.codegen.node_info import (
    _pair,
    get_arg_value,
    is_gemm_op,
    is_nop,
    is_prunable_op,
)
from voyager_compiler.codegen.subgraph import replace_node_with_graph_module
from voyager_compiler.codegen.transform.operator_fusion import _nodes_sequential
from voyager_compiler.export_utils import (
    create_getattr_from_value,
    get_aten_graph_module,
)
from voyager_compiler.quantization.quantizer.xnnpack_quantizer_utils import (
    _convert_scalars_to_attrs,
)
from voyager_compiler.shape_prop import (
    fetch_attr,
    propagate_shape,
    set_node_value,
)

logger = logging.getLogger(__name__)

__all__ = [
    "deduplicate_nodes",
    "replace_interpolate",
    "replace_rmsnorm_with_layer_norm",
    "replace_conv2d_with_im2col",
    "extract_input_preprocessor",
    "inline_autocast_modules",
    "fold_constant_generators",
    "remove_prunable_ops",
    "remove_softmax_dtype_cast",
    "remove_zero_attention_mask",
]


class WrapperModule(torch.nn.Module):
    """Wrap a callable in a ``Module`` so it can be exported."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def deduplicate_nodes(model: GraphModule):
    """Collapse identical nodes — same op, target, args and kwargs — into one.

    Args:
        model: Graph module to rewrite in place.

    Returns:
        A mapping from each erased node to the node that replaced it.
    """
    seen = {}
    mapping = {}
    graph = model.graph

    for node in list(graph.nodes):
        if node.op in ("placeholder", "output"):
            continue

        key = (
            node.op,
            node.target,
            tuple(node.args),
            frozenset(node.kwargs.items()),
        )

        if key in seen:
            orig = seen[key]
            node.replace_all_uses_with(orig)
            graph.erase_node(node)
            mapping[node] = orig
        else:
            seen[key] = node

    for old, new in mapping.items():
        logger.debug(f"Deduplicated {old} to {new}")

    graph.lint()
    model.recompile()
    return mapping


def remove_prunable_ops(model: GraphModule) -> None:
    """Delete identity ops — a full ``slice``, a unit ``expand``, a same-dtype
    ``to``, a zero-prob ``dropout`` (see ``is_prunable_op``) — by rewiring each
    to its input.  They survive fusion (they have users, so dead-code
    elimination skips them); dropping them shrinks the graph every later pass
    and the bufferizer must walk.
    """
    graph = model.graph
    removed = 0
    for n in list(graph.nodes):
        if is_prunable_op(n):
            n.replace_all_uses_with(n.all_input_nodes[0])
            graph.erase_node(n)
            removed += 1
    if removed:
        graph.lint()
        model.recompile()
    logger.debug("[transform] removed %d prunable ops", removed)


def replace_interpolate():
    template = (
        "interpolate(Tensor input, SymInt[] size, float[]? scale_factor = None,"
        "str mode = 'nearest', bool? align_corners = None, "
        "bool? recompute_scale_factor = None, bool antialias = False) -> Tensor"
    )

    global m
    m = Library("custom", "DEF")
    m.define(template)

    orig_interpolate = torch.nn.functional.interpolate

    @impl(m, "interpolate", "CompositeExplicitAutograd")
    def interpolate(*args, **kwargs):
        return orig_interpolate(*args, **kwargs)

    torch.nn.functional.interpolate = torch.ops.custom.interpolate


def replace_rmsnorm_with_layer_norm(
    model: GraphModule,
    layer_norm: torch.nn.Module,
    example_input,
    convert_scalars_to_attrs=False,
):
    """Replace LLaMA RMSNorm with ATen layer_norm"""
    original_graph = model.graph

    pattern = get_aten_graph_module(layer_norm, example_input)
    if convert_scalars_to_attrs:
        _convert_scalars_to_attrs(pattern)
    pattern_graph = pattern.graph

    matcher = SubgraphMatcher(
        pattern_graph,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=False,
    )
    _matches: List[InternalMatch] = matcher.match(original_graph)
    logger.info(f"Found {len(_matches)} matches")

    weight_node = next(
        iter(n for n in pattern_graph.nodes if n.target == "weight")
    )

    for match in _matches:
        input_node = match.placeholder_nodes[0]
        output_node = match.returning_nodes[0]
        input_shape = input_node.meta["val"].shape
        new_weight_node = match.nodes_map[weight_node]
        layer_norm_inputs = [input_node, [input_shape[-1]], new_weight_node]

        with original_graph.inserting_before(output_node):
            new_node = original_graph.call_function(
                torch.ops.aten.layer_norm.default, tuple(layer_norm_inputs), {}
            )

        output_node.replace_all_uses_with(new_node)
        original_graph.erase_node(output_node)

        new_node.meta = output_node.meta

    original_graph.lint()
    original_graph.eliminate_dead_code()
    model.recompile()


def _get_im2col_gemm_pattern(
    output_shape: Tuple[int],
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
):

    def _im2col_gemm_pattern(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor]:
        inp_unf = F.unfold(input, weight.shape[-2:], dilation, padding, stride)
        wt = weight.view(weight.size(0), -1)
        out_unf = F.linear(inp_unf.transpose(-1, -2), wt, bias)
        out = out_unf.transpose(-1, -2).view(*output_shape)
        return out

    return WrapperModule(_im2col_gemm_pattern)


def replace_conv2d_with_im2col(model: GraphModule):
    """
    Replace Conv2d operations that has input channel dimension equal to 3 with
    In2col operations in the given FX graph module.  Usually this is the first
    Conv2D layer in torchvision models.

    Args:
        model (GraphModule): The FX graph module to transform.

    Returns:
        GraphModule: The transformed FX graph module.
    """
    graph = model.graph

    def get_shape(n):
        if n.meta and "val" in n.meta:
            return n.meta["val"].shape
        return getattr(n, "shape", None)

    for node in list(graph.nodes):
        if node.target != torch.ops.aten.conv2d.default:
            continue

        input_node = node.args[0]
        weight_node = node.args[1]
        bias_node = get_arg_value(node, 2, "bias")
        stride = get_arg_value(node, 3, "stride", 1)
        padding = get_arg_value(node, 4, "padding", 0)
        dilation = get_arg_value(node, 5, "dilation", 1)
        group = get_arg_value(node, 6, "groups", 1)

        input_shape = get_shape(input_node)
        weight_shape = get_shape(weight_node)
        output_shape = get_shape(node)

        if (
            input_shape is None
            or input_shape[1] != 3
            or output_shape is None
            or group != 1
        ):
            continue

        logger.info(f"Replacing Conv2d node {node} with Im2col + GEMM")

        _example_inputs = (
            torch.randn(input_shape),
            torch.randn(weight_shape),
            torch.randn((weight_shape[0],)) if bias_node is not None else None,
        )

        match_pattern = _get_im2col_gemm_pattern(
            output_shape, _pair(stride), _pair(padding), _pair(dilation)
        )
        match_pattern = _get_aten_graph_module_for_pattern(
            match_pattern,
            _example_inputs,
        )

        val_maps = {}
        output = replace_node_with_graph_module(
            model, node, match_pattern, val_maps
        )[0]
        graph.erase_node(node)

        # Fold the view operation into the parameter
        view_node = next(iter(weight_node.users))
        assert view_node.target == torch.ops.aten.view.default

        val = fetch_attr(model, weight_node.target).detach()
        val = val.reshape(val.size(0), -1)

        with graph.inserting_before(view_node):
            new_weight = create_getattr_from_value(
                model, graph, f"{weight_node.target}_im2col", val
            )

        propagate_shape(new_weight, model)
        view_node.replace_all_uses_with(new_weight)

        # Move elementwise operations after view to after the linear node
        linear_node = next((n for n in val_maps.values() if is_gemm_op(n)))

        order = {n: i for i, n in enumerate(graph.nodes)}
        fusable_ops = []
        next_node = next(iter(output.users))
        while is_elementwise_op(next_node):
            chain = fusable_ops + [next_node]
            if _nodes_sequential(chain, order):
                fusable_ops.append(next_node)
            else:
                break
            # Stop fusing if last node is a quantize op
            if (
                len(next_node.users) != 1
                or next_node.target == torch.ops.quantized_ops.quantize.default
            ):
                break
            next_node = next(iter(next_node.users))

        if not fusable_ops:
            continue

        linear_node.replace_all_uses_with(fusable_ops[-1])
        fusable_ops[0].replace_input_with(output, linear_node)
        next_node.replace_input_with(fusable_ops[-1], output)

        for n in reversed(fusable_ops):
            linear_node.append(n)

    graph.eliminate_dead_code()
    graph.lint()
    model.recompile()
    return model


def extract_input_preprocessor(model: GraphModule, input_name=None):
    """
    Extract the input preprocessing operations from the given FX GraphModule
    and create a separate GraphModule.

    Args:
        model (GraphModule): The FX graph module to transform.
        input_name: Target of the placeholder to peel preprocessing from.
            Defaults to the first placeholder (the activation input); pass
            e.g. ``"attention_mask"`` to extract that input's quantize.

    Returns:
        GraphModule: The transformed FX graph module with the input
            preprocessor extracted.
    """
    placeholder = next(
        iter(
            n
            for n in model.graph.nodes
            if n.op == "placeholder"
            and (input_name is None or n.target == input_name)
        )
    )
    preprocess_nodes = [placeholder]

    user = next(iter(placeholder.users))

    while is_nop(user) or user.target in [
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.im2col.default,
        torch.ops.aten.pad.default,
        torch.ops.quantized_ops.quantize.default,
    ]:
        preprocess_nodes.extend(
            n for n in user.all_input_nodes if n not in preprocess_nodes
        )
        preprocess_nodes.append(user)
        user = next(iter(user.users))

    m = torch.nn.Module()

    new_graph = torch.fx.Graph()
    value_remap = {}
    for node in preprocess_nodes:
        if node.op == "placeholder":
            value_remap[node] = new_graph.placeholder(node.name)
        else:
            value_remap[node] = new_graph.node_copy(
                node, lambda n: value_remap[n]
            )

            if node.op == "get_attr":
                param = fetch_attr(model, node.target)
                m.register_buffer(node.target, param)
    new_graph.output(value_remap[preprocess_nodes[-1]])
    new_graph.lint()
    new_graph.print_tabular()

    with model.graph.inserting_before(placeholder):
        new_placeholder = model.graph.placeholder(
            f"{placeholder.name}_preprocess"
        )
    preprocess_nodes[-1].replace_all_uses_with(new_placeholder)

    new_placeholder.meta["dtype"] = preprocess_nodes[-1].meta.get("dtype")
    set_node_value(new_placeholder, preprocess_nodes[-1].value)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    # Placeholder node needs to be manually erased
    model.graph.erase_node(placeholder)
    model.recompile()
    return model, GraphModule(m, new_graph)


def inline_autocast_modules(model: torch.fx.GraphModule):
    """Inline wrap HOPs (``autocast`` / ``set_grad_enabled``) by replacing the
    wrap node with a direct ``call_module`` to its wrapped submodule.

    torch.export emits ``WrapWithAutocast`` for ``torch.autocast`` regions and
    ``wrap_with_set_grad_enabled`` for ``@torch.no_grad`` regions (e.g. Llama's
    ``LlamaRotaryEmbedding.forward``).  Both wrap a submodule that the rest of
    the lowering can only handle once flattened into the parent graph.  The
    wrapped function is at ``args[fn_idx]`` and its operands follow it.
    """
    graph = model.graph
    named_modules = dict(model.named_modules())
    set_grad = torch.ops.higher_order.wrap_with_set_grad_enabled

    for node in list(graph.nodes):
        if isinstance(
            node.target, torch._higher_order_ops.wrap.WrapWithAutocast
        ):
            fn_idx = 4
        elif node.target is set_grad:
            fn_idx = 1
        else:
            continue

        wrapped_func = node.args[fn_idx]
        mod = named_modules.get(wrapped_func.target, None)
        if mod is None:
            continue

        with graph.inserting_before(node):
            new_node = graph.call_module(
                wrapped_func.target, tuple(node.args[fn_idx + 1 :])
            )
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)

        replace_node_with_graph_module(model, new_node, mod)

    graph.eliminate_dead_code()
    model.graph.lint()
    model.recompile()


def fold_constant_generators(model: GraphModule):
    """Constant-fold ``call_function`` nodes whose inputs are all constants:
    input-free generators (``arange`` / ``zeros`` / …) and, transitively, any op
    fed only by already-folded constants — so a whole constant subgraph (e.g.
    RoPE's ``arange -> … -> cos/sin`` position setup) collapses to one
    ``get_attr`` buffer and is not lowered or scheduled as a compute op.

    Walking in program order, a node is constant iff every FX-Node input is a
    ``get_attr`` (an initial buffer or one this pass just created); it is then
    evaluated with the real buffer values and replaced by a ``get_attr`` to the
    result.  Orphaned constant ancestors are dropped by dead-code elimination.
    """
    graph = model.graph
    constants = {n for n in graph.nodes if n.op == "get_attr"}

    def resolve(n: Node):
        return fetch_attr(model, n.target)

    for node in list(graph.nodes):
        if node.op != "call_function" or any(
            inp not in constants for inp in node.all_input_nodes
        ):
            continue
        # Folding a dequantize offsets the benefit of quantizing the params;
        # folding an expand materializes the replication it stands for.
        if node.target in [
            torch.ops.quantized_ops.dequantize.default,
            torch.ops.aten.expand.default,
        ]:
            continue
        # Bufferization pass can absorb expand into indexing to avoid copying.
        if any(u.target is torch.ops.aten.expand.default for u in node.users):
            continue
        if not isinstance(getattr(node, "value", None), torch.Tensor):
            continue
        const = node.target(
            *map_arg(node.args, resolve), **map_arg(node.kwargs, resolve)
        )
        src = next((n.target for n in node.all_input_nodes), "const")
        prefix = re.sub(r"_folded(_\d+)?$", "", str(src)) + "_folded"
        with graph.inserting_before(node):
            attr = create_getattr_from_value(model, graph, prefix, const)
        attr.meta["dtype"] = node.meta.get("dtype")
        set_node_value(attr, const)
        node.replace_all_uses_with(attr)
        graph.erase_node(node)
        constants.add(attr)

    graph.eliminate_dead_code()
    graph.lint()
    model.recompile()


def remove_zero_attention_mask(model: GraphModule, example_inputs):
    """Drop additive attention masks that are provably all-zero.

    Eager attention in transformers materializes ``add(scores, mask)`` where
    ``mask = where(valid, 0.0, min_value)``.  For bidirectional models with no
    padding (e.g. ViT) every position is valid, so the mask is identically
    zero and the add is a no-op.  The mask is evaluated on ``example_inputs``
    and an add is bypassed only when its mask operand is in fact all zero;
    dead-code elimination then removes the mask-building chain.  Masks that
    are not all zero (e.g. real causal masks) are left untouched.
    """
    masks = {}

    class _Capture(Interpreter):
        def run_node(self, n):
            out = super().run_node(n)
            if n.op == "call_function" and "where" in str(n.target):
                masks[n] = out
            return out

    _Capture(model).run(*example_inputs)

    removed = 0
    for mask_node, value in masks.items():
        # Guard: only drop the add when the mask is genuinely all zero.
        if not isinstance(value, torch.Tensor) or not bool((value == 0).all()):
            continue
        for add_node in list(mask_node.users):
            if add_node.target != torch.ops.aten.add.Tensor:
                continue
            others = [a for a in add_node.args if a is not mask_node]
            if len(others) != 1:
                continue
            add_node.replace_all_uses_with(others[0])
            model.graph.erase_node(add_node)
            removed += 1

    logger.info(f"Removed {removed} zero attention-mask add(s)")
    model.graph.eliminate_dead_code()
    model.graph.lint()
    model.recompile()
    return model


def remove_softmax_dtype_cast(model: torch.fx.GraphModule):
    graph = model.graph
    for node in list(model.graph.nodes):
        if node.target == torch.ops.aten.softmax.int:
            node.args = node.args[:2]

    graph.lint()
    model.recompile()
    return model
