import itertools
import logging
import operator
from typing import Tuple, Union, Callable, List
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node
from torch.fx.node import map_arg
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher
from torchao.quantization.pt2e.utils import _get_aten_graph_module_for_pattern

from .utils import _pair, get_arg_value
from ..mapping import (
    replace_node_with_graph_module,
    _nodes_sequential,
    _create_and_insert_subgraph,
)
from ..mapping_utils import is_elementwise_op, is_gemm_op, is_nop
from ...pt2e_utils import WrapperModule, get_aten_graph_module, fetch_attr, propagate_shape
from ...quantize_pt2e import create_getattr_from_value, export_model
from ...quantizer.xnnpack_quantizer_utils import _convert_scalars_to_attrs

logger = logging.getLogger(__name__)

__all__ = [
    "convert_cat_and_stack_as_stack_on_dim0",
    "convert_cat_with_mismatched_shapes_to_stack",
    "convert_expand_to_memory_copy",
    "replace_interpolate",
    "replace_rmsnorm_with_layer_norm",
    "replace_conv2d_with_im2col",
    "extract_input_preprocessor",
    "rewrite_fx_graph",
    "inline_autocast_modules",
    "remove_softmax_dtype_cast",
    "split_multi_head_attention",
]


def _decompose_bmm(model: GraphModule, node: Node):
    input1 = node.args[0].value
    input2 = node.args[1].value

    input1_dims = sum(1 for d in input1.shape if d > 1)
    input2_dims = sum(1 for d in input2.shape if d > 1)
    if input1_dims < 3 and input2_dims < 3:
        return None

    class BMM(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Loop through each element in the batch dimensions
            batch_shape = x.shape[:-2]
            result = []
            for idx in itertools.product(*[range(dim) for dim in batch_shape]):
                result.append(torch.matmul(x[idx], y[idx]))
            result = torch.stack(result)
            result = result.view(*batch_shape, *result.shape[-2:])
            return result

    gm = export_model(BMM(), (input1, input2))

    value_remap = {}
    output_nodes = replace_node_with_graph_module(model, node, gm, value_remap)
    model.graph.erase_node(node)

    for n in list(value_remap.values()):
        if n.target == torch.ops.aten.select.int:
            n.meta["dtype"] = n.args[0].meta.get("dtype")

        if n.target in [
            node.target,
            torch.ops.aten.stack.default,
            torch.ops.aten.view.default
        ]:
            n.meta["dtype"] = node.meta.get("dtype")

    return output_nodes[0]


def _decompose_bmm_mx(model: GraphModule, node: Node):
    input1 = node.args[0].value
    input2 = node.args[1].value

    input1_dims = sum(1 for d in input1.shape if d > 1)
    input2_dims = sum(1 for d in input2.shape if d > 1)
    if input1_dims < 3 and input2_dims < 3:
        return None

    block_size = node.kwargs['block_size']

    class BMM(torch.nn.Module):
        def forward(
                self, input: torch.Tensor, other: torch.Tensor, input_scale=None,
                weight_scale=None, input_code=None, weight_code=None, A_data=None,
                A_indices=None, A_indptr=None, weight_transposed=False):
            # Loop through each element in the batch dimensions
            batch_shape = input.shape[:-2]
            result = []
            for idx in itertools.product(*[range(dim) for dim in batch_shape]):
                kwargs = {
                    "input_scale": input_scale[idx],
                    "weight_scale": weight_scale[idx],
                    "block_size": block_size,
                    "input_code": input_code,
                    "weight_code": weight_code,
                }
                if A_data is not None:
                    kwargs.update({
                        "A_data": A_data[idx],
                        "A_indices": A_indices[idx],
                        "A_indptr": A_indptr[idx],
                        "weight_transposed": weight_transposed,
                    })
                result.append(torch.ops.quantized_ops.matmul_mx(
                    input[idx], other[idx], **kwargs,
                ))
            result = torch.stack(result)
            result = result.view(*batch_shape, *result.shape[-2:])
            return result

    input_code = node.kwargs.get('input_code', None)
    weight_code = node.kwargs.get('weight_code', None)
    A_data = node.kwargs.get('A_data', None)
    A_indices = node.kwargs.get('A_indices', None)
    A_indptr = node.kwargs.get('A_indptr', None)

    kwargs = {
        'input_scale': node.kwargs['input_scale'].value,
        'weight_scale': node.kwargs['weight_scale'].value,
        'input_code': input_code.value if input_code is not None else None,
        'weight_code': weight_code.value if weight_code is not None else None,
        'A_data': A_data.value if A_data is not None else None,
        'A_indices': A_indices.value if A_indices is not None else None,
        'A_indptr': A_indptr.value if A_indptr is not None else None,
        'weight_transposed': node.kwargs.get('weight_transposed', True),
    }

    gm = export_model(BMM(), (input1, input2), kwargs)

    # Remove unused placeholder nodes
    for n in gm.graph.nodes:
        if n.op == 'placeholder' and len(n.users) == 0:
            gm.graph.erase_node(n)
    gm.graph.lint()

    value_remap = {}
    output_nodes = replace_node_with_graph_module(model, node, gm, value_remap)
    model.graph.erase_node(node)

    for n in list(value_remap.values()):
        if n.target == torch.ops.aten.select.int:
            n.meta["dtype"] = n.args[0].meta.get("dtype")

        if n.target in [
            node.target,
            torch.ops.aten.stack.default,
            torch.ops.aten.view.default
        ]:
            n.meta["dtype"] = node.meta.get("dtype")

    return output_nodes[0]


def _decompose_bmm_mx_with_outlier_inputs(model: GraphModule, node: Node):
    from .tiling import _get_node_attribute

    quantized_ops = torch.ops.quantized_ops

    input1 = node.args[0].value
    input2 = node.args[1].value

    input1_dims = sum(1 for d in input1.shape if d > 1)
    input2_dims = sum(1 for d in input2.shape if d > 1)
    if input1_dims < 3 and input2_dims < 3:
        return None

    input_node = node.args[0]
    quantize_node = input_node.all_input_nodes[0]

    example_inputs = (
        quantize_node.args[0],
        get_arg_value(quantize_node, 1, 'qmap'),
        get_arg_value(quantize_node, 6, 'scale_qmap'),
        get_arg_value(quantize_node, 7, 'output_code'),
        node.args[1],
        node.kwargs.get('weight_scale'),
        node.kwargs.get('input_code'),
        node.kwargs.get('weight_code'),
    )

    quantize_mx_outlier_kwargs = _get_node_attribute(quantize_node)
    gemm_kwargs = _get_node_attribute(node)

    def forward(
        input,
        input_qmap=None,
        scale_qmap=None,
        output_code=None,
        other=None,
        weight_scale=None,
        input_code=None,
        weight_code=None,
    ):
        batch_shape = input.shape[:-2]
        result = []
        for idx in itertools.product(*[range(dim) for dim in batch_shape]):
            (
                data, indices, indptr, input_scale, inliers
            ) = quantized_ops.quantize_mx_outlier(
                input[idx],
                qmap=input_qmap,
                scale_qmap=scale_qmap,
                output_code=output_code,
                **quantize_mx_outlier_kwargs,
            )
            output = quantized_ops.matmul_mx(
                inliers,
                other[idx],
                input_scale=input_scale,
                weight_scale=weight_scale[idx],
                input_code=input_code,
                weight_code=weight_code,
                A_data=data,
                A_indices=indices,
                A_indptr=indptr,
                **gemm_kwargs,
            )
            result.append(output)
        result = torch.stack(result)
        result = result.view(*batch_shape, *result.shape[-2:])
        return result

    node_group = [quantize_node, node] + list(quantize_node.users)
    named_modules = dict(model.named_modules())
    submod_node = _create_and_insert_subgraph(node_group, model, named_modules)

    example_inputs = map_arg(
        example_inputs, lambda n: n.value if isinstance(n, Node) else n
    )

    gm = export_model(WrapperModule(forward), example_inputs)

    # Remove unused placeholder nodes
    for n in gm.graph.nodes:
        if n.op == 'placeholder' and len(n.users) == 0:
            gm.graph.erase_node(n)
    gm.graph.lint()

    value_remap = {}
    output_nodes = replace_node_with_graph_module(
        model, submod_node, gm, value_remap
    )
    model.graph.erase_node(submod_node)

    for n in list(value_remap.values()):
        if n.target == torch.ops.aten.select.int:
            n.meta["dtype"] = n.args[0].meta.get("dtype")

        if n.target == operator.getitem:
            if (dtypes := n.args[0].meta.get("dtype")) is not None:
                idx = n.args[1]
                n.meta["dtype"] = dtypes[idx]

        if n.target in (node.target, quantize_node.target):
            source_node = node if n.target == node.target else quantize_node
            n.meta["dtype"] = source_node.meta.get("dtype")

    return output_nodes[0]


def _decompose_bmm_helper(model: GraphModule, node: Node):
    if node.target == torch.ops.aten.matmul.default:
        return _decompose_bmm(model, node)
    elif node.kwargs.get('A_data', None) is None:
        return _decompose_bmm_mx(model, node)
    else:
        return _decompose_bmm_mx_with_outlier_inputs(model, node)


def split_multi_head_attention(model: GraphModule):
    graph = model.graph

    grouped_nodes = defaultdict(list)
    for node in list(graph.nodes):
        if node.target not in [
            torch.ops.aten.matmul.default,
            torch.ops.quantized_ops.matmul_mx.default,
        ]:
            continue

        if (nn_module_stack := node.meta.get('nn_module_stack', None)) is not None:
            bt = list(nn_module_stack.values())[-1]
            grouped_nodes[bt[0]].append(node)

    for nodes in grouped_nodes.values():
        if len(nodes) != 2:
            for node in nodes:
                _decompose_bmm_helper(model, node)
            continue

        qk_matmul, pv_matmul = nodes[0], nodes[1]

        # Find the nodes between the qk and av matmuls
        def dfs(current_node, visited, max_depth=20):
            if len(visited) > max_depth:
                return None
            if current_node == pv_matmul:
                return [visited]
            paths = []
            for user in current_node.users:
                if user not in visited:
                    if (result := dfs(user, visited + [user], max_depth)) is None:
                        return None
                    paths.extend(result)
            return paths

        paths = dfs(qk_matmul, [qk_matmul])

        # Decompose BMM into multiple matmuls
        qk_output = _decompose_bmm_helper(model, qk_matmul)
        pv_output = _decompose_bmm_helper(model, pv_matmul)

        if paths is None:
            logger.warning(
                f"Failed to find paths between {qk_matmul} and {pv_matmul}. "
                "Skipping fusion."
            )
            continue

        nodes_between = set()
        for path in paths:
            nodes_between.update(path[1:-1])

        # Sort the nodes between the qk and pv matmuls
        order = {node: idx for idx, node in enumerate(graph.nodes)}
        nodes_between = sorted(nodes_between, key=lambda n: order[n])

        # Duplicate the nodes between the qk and pv matmuls to perform fusion
        qk_matmuls = qk_output.args[0].args[0]
        pv_matmuls = pv_output.args[0].args[0]

        def select_tensor(n, target_dim):
            dim = len(n.shape)
            for _ in range(dim - target_dim):
                n = graph.call_function(torch.ops.aten.select.int, (n, 0, 0))
                propagate_shape(n)
            return n

        for qk_matmul, pv_matmul in zip(qk_matmuls, pv_matmuls):
            value_remap = {qk_output: qk_matmul}
            for node in nodes_between:
                with graph.inserting_before(pv_matmul):
                    for n in node.all_input_nodes:
                        if n not in value_remap and len(n.shape) > 2:
                            value_remap[n] = select_tensor(n, 2)

                    value_remap[node] = graph.node_copy(
                        node, lambda n: value_remap.get(n, n)
                    )
                propagate_shape(value_remap[node])

            has_spmm_arg = "A_data" in pv_matmul.kwargs
            arg_idx = 1 if has_spmm_arg else 0
            pv_matmul.replace_input_with(
                pv_matmul.args[arg_idx], value_remap[nodes_between[-1]]
            )

            arg_key = 'weight_scale' if has_spmm_arg else 'input_scale'
            if (scale_node := pv_matmul.kwargs.get(arg_key)) is not None:
                pv_matmul.replace_input_with(
                    scale_node, value_remap[nodes_between[-2]]
                )

    def is_impure_node(n):
        return n.op in ['placeholder', 'output']

    graph.lint()
    graph.eliminate_dead_code(is_impure_node=is_impure_node)
    model.recompile()

    return model


def convert_cat_and_stack_as_stack_on_dim0(model: GraphModule):
    """
    Transforms occurrences of `torch.cat` and `torch.stack` operations on
    non-zero dimensions into a `torch.stack` on the 0th dimension, followed by
    a `permute` operation to restore the original order.

    Args:
        model (GraphModule): The PyTorch FX GraphModule to be modified.

    Returns:
        GraphModule: The transformed GraphModule with `torch.cat` and `torch.stack`
        operations adjusted.
    """
    graph = model.graph
    for node in list(graph.nodes):
        if node.target not in [
            torch.ops.aten.cat.default, torch.ops.aten.stack.default
        ]:
            continue
        cat_node = node

        if not all(hasattr(n, "shape") for n in cat_node.args[0]):
            logger.warning(f"Node {cat_node} does not have shape attributes for all inputs.")
            continue

        shapes = [n.shape for n in cat_node.args[0]]
        input_shape = list(shapes[0])

        if not all(list(s) == input_shape for s in shapes):
            logger.warning(
                "Concatenated tensors have different shapes in node %s. Shapes: %s",
                cat_node, shapes
            )
            continue

        concat_dim = cat_node.args[1] if len(cat_node.args) > 1 else 0
        if concat_dim < 0:
            concat_dim += len(input_shape)

        if len(cat_node.args) == 1 or concat_dim == 0:
            continue

        # Always stack along the first dimension
        if cat_node.target == torch.ops.aten.stack.default:
            cat_node.args = (cat_node.args[0], 0)
            stack_node = cat_node
        else:
            with graph.inserting_after(cat_node):
                stack_node = graph.call_function(
                    torch.ops.aten.stack.default, (cat_node.args[0], 0)
                )
        propagate_shape(stack_node)

        # Permute the concatenated tensor to match the original order
        dims = list(range(len(input_shape) + 1))[1:]
        dims = dims[:concat_dim] + [0] + dims[concat_dim:]

        logger.info(f"Converting {cat_node} to stack on dim 0 with permute {dims}")

        with graph.inserting_after(stack_node):
            permute_node = graph.call_function(
                torch.ops.aten.permute.default, (stack_node, dims),
            )
        propagate_shape(permute_node)
        output_node = permute_node

        # Flatten the permuted tensor if it is a cat operation
        if cat_node.target == torch.ops.aten.cat.default:
            with graph.inserting_after(permute_node):
                output_node = graph.call_function(
                    torch.ops.aten.flatten.using_ints,
                    (permute_node, concat_dim, concat_dim + 1),
                )
            propagate_shape(output_node)

        # Replace all use of the cat node with the new node
        for node in list(cat_node.users):
            if id(node) == id(output_node):
                continue
            node.replace_input_with(cat_node, output_node)

        if cat_node.target == torch.ops.aten.cat.default:
            graph.erase_node(cat_node)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


def convert_cat_with_mismatched_shapes_to_stack(model: GraphModule):
    """
    Convert `torch.cat` operations where input tensors have different shapes by
    replacing them with a `torch.stack` operation along the concatenated dimensions.

    Args:
        model (GraphModule): The PyTorch FX GraphModule to be modified.

    Returns:
        GraphModule: The transformed GraphModule with `torch.cat` operations
        adjusted to `torch.stack`.
    """
    for node in model.graph.nodes:
        if node.target != torch.ops.aten.cat.default:
            continue

        input_shape = list(node.args[0][0].shape)
        if all(list(n.shape) == input_shape for n in node.args[0][1:]):
            continue

        logger.info(f"Node {node} has different input shapes")
        dim = node.args[1]

        args = map_arg(node.args, lambda n: n.value)
        shape = list(args[0][0].shape[:dim])

        class Concat(torch.nn.Module):
            def forward(self, *inputs):
                result = []
                for idx in itertools.product(*[range(dim) for dim in shape]):
                    tensor = torch.cat([x[idx] for x in inputs], dim=0)
                    result.append(tensor)
                output = torch.stack(result, dim=0)
                return output.reshape(*shape, *output.shape[1:])

        gm = export_model(Concat(), (*args[0],))
        replace_node_with_graph_module(model, node, gm)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def convert_expand_to_memory_copy(model: torch.fx.GraphModule):
    """
    Convert `torch.expand` operations into explicit memory copying by replicating
    input elements. This replaces implicit broadcasting with actual memory
    duplication, ensuring that expanded  dimensions are materialized as stacked tensors.

    Args:
        model (torch.fx.GraphModule): The PyTorch FX GraphModule to be modified.

    Returns:
        torch.fx.GraphModule: The transformed GraphModule where `torch.expand`
        operations are replaced with explicit memory copies.
    """
    for node in list(model.graph.nodes):
        if node.target != torch.ops.aten.expand.default:
            continue

        # Skip if the expand operation is a no-op
        if all(x == 1 or x == -1 for x in node.args[1]):
            continue

        input_node = node.args[0]
        sizes = node.args[1]
        original_shape = input_node.meta["val"].shape
        assert len(sizes) >= len(original_shape), (
            "Sizes must have at least as many dimensions as the original tensor."
        )

        # Add singleton dimensions to match the size length
        while len(original_shape) < len(sizes):
            input = input.unsqueeze(0)
            original_shape = input.shape

        class Expand(torch.nn.Module):
            def forward(self, input):
                # Stack along the first dimension to create the expanded shape
                for dim, size in enumerate(sizes):
                    if input.shape[dim] == 1 and size > 1:
                        input = torch.stack([input.squeeze(dim)] * size, dim=dim)
                    elif input.shape[dim] != size:
                        raise ValueError(
                            f"Cannot expand dimension {dim} from {input.shape[dim]} to {size}."
                        )
                return input

        gm = export_model(Expand(), (input_node.meta["val"],))
        replace_node_with_graph_module(model, node, gm)
        model.graph.erase_node(node)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def replace_interpolate():
    from torch.library import Library, impl

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
    """Replace LLaMA RMSNorm with ATen layer_norm
    """
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

    weight_node = next(iter(n for n in pattern_graph.nodes if n.target == "weight"))

    for match in _matches:
        input_node = match.placeholder_nodes[0]
        output_node = match.returning_nodes[0]
        input_shape = input_node.meta["val"].shape
        new_weight_node = match.nodes_map[weight_node]
        layer_norm_inputs = [input_node, [input_shape[-1]], new_weight_node]

        with original_graph.inserting_before(output_node):
            new_node = original_graph.call_function(
                torch.ops.aten.layer_norm.default,
                tuple(layer_norm_inputs),
                {}
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
    In2col operations in the given FX graph module. Usually this is the first
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
        output = replace_node_with_graph_module(model, node, match_pattern, val_maps)[0]
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

        linear_node.replace_all_uses_with(fusable_ops[-1])
        fusable_ops[0].replace_input_with(output, linear_node)
        next_node.replace_input_with(fusable_ops[-1], output)

        for n in reversed(fusable_ops):
            linear_node.append(n)

    graph.eliminate_dead_code()
    graph.lint()
    model.recompile()
    return model


def extract_input_preprocessor(model: GraphModule):
    """
    Extract the input preprocessing operations from the given FX GraphModule
    and create a separate GraphModule.

    Args:
        model (GraphModule): The FX graph module to transform.

    Returns:
        GraphModule: The transformed FX graph module with the input preprocessor extracted.
    """
    placeholder = next(iter(n for n in model.graph.nodes if n.op == "placeholder"))
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
        if node.op == 'placeholder':
            value_remap[node] = new_graph.placeholder(node.name)
        else:
            value_remap[node] = new_graph.node_copy(node, lambda n: value_remap[n])

            if node.op == "get_attr":
                param = fetch_attr(model, node.target)
                m.register_buffer(node.target, param)
    new_graph.output(value_remap[preprocess_nodes[-1]])
    new_graph.lint()
    new_graph.print_tabular()

    with model.graph.inserting_before(placeholder):
        new_placeholder = model.graph.placeholder(f"{placeholder.name}_preprocess")
    preprocess_nodes[-1].replace_all_uses_with(new_placeholder)

    new_placeholder.meta["dtype"] =  preprocess_nodes[-1].meta.get("dtype")

    model.graph.lint()
    model.graph.eliminate_dead_code()
    # Placeholder node needs to be manually erased
    model.graph.erase_node(placeholder)
    model.recompile()
    return model, GraphModule(m, new_graph)


def rewrite_fx_graph(model: torch.fx.GraphModule, fn: Callable):
    """
    Transforms a given PyTorch FX GraphModule by identifying and replacing
    nodes that match a user-defined match_and_rewrite with alternative implementations.

    Args:
        model (torch.fx.GraphModule): The input FX GraphModule to be transformed.
        fn (Callable): A function that takes three arguments:
            - sources: The underlying function, module, or primitive operation
              responsible for a given FX node (from node.meta["source_fn_stack"]).
            - example_args (Tuple): A tuple of example arguments for the node,
              extracted from node metadata.
            - example_kwargs (Dict): A dictionary of example keyword arguments for the node.

            The `match_and_rewrite` function should return:
                - A `torch.nn.Module` or callable implementing an equivalent
                  or decomposed version of the operation if a match is found.
                - `None` otherwise.

    Returns:
        torch.fx.GraphModule: The transformed GraphModule with selected nodes
        replaced by decomposed modules returned by `match_and_rewrite`.

    Notes:
        - Each matched node is replaced using `export_model` with the returned
          module from `match_and_rewrite`.
        - The original node is erased from the graph after replacement.
        - The transformed graph is cleaned up via linting, dead code elimination,
          and recompilation.

    Example:
        >>> def match_and_rewrite(source_fn, args, kwargs):
        ...     if source_fn not in [torch.nn.Conv2d, torch.nn.functional.conv2d]:
        ...         return None
        ...     # Replace with a no-op or alternative module
        ...     class Identity(nn.Module):
        ...         def forward(self, x): return x
        ...     return Identity
        >>> transformed = rewrite_fx_graph(fx_model, match_and_rewrite)
    """
    for node in list(model.graph.nodes):
        if node.op != "call_function":
            continue

        if (source_fn_st := node.meta.get("source_fn_stack")) is None:
            continue

        source_fn = source_fn_st[-1][1]

        def get_value(n: Node):
            if "val" in n.meta:
                return n.meta["val"]
            return getattr(n, "value", None)

        example_args = map_arg(node.args, get_value)
        example_kwargs = map_arg(node.kwargs, get_value)

        if (cls := fn(source_fn, example_args, example_kwargs)) is None:
            continue

        new_args = map_arg(tuple(node.all_input_nodes), get_value)
        gm = export_model(cls(), new_args, example_kwargs)

        # PyTorch PT2E expect nodes to have no kwargs in the exported graph.
        # Clone has a memory_format kwarg, zeros_like has a pin_memory kwarg, and
        # gelu has a has an approximate kwarg that persist in exported graph.
        # This is just a work around for these.
        for n in list(gm.graph.nodes):
            if n.target == torch.ops.aten.zeros.default:
                n.kwargs = {}

        replace_node_with_graph_module(model, node, gm)

        model.graph.erase_node(node)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def inline_autocast_modules(model: torch.fx.GraphModule):
    """
    Handle autocast HOP by replacing the autocast node with its wrapped module
    and directly calling the arguments.
    """
    graph = model.graph
    named_modules = dict(model.named_modules())

    for node in list(graph.nodes):
        if isinstance(node.target, torch._higher_order_ops.wrap.WrapWithAutocast):
            wrapped_func = node.args[4]
            mod = named_modules.get(wrapped_func.target, None)

            if mod is None:
                continue

            with graph.inserting_before(node):
                new_node = graph.call_module(
                    wrapped_func.target, tuple(node.args[5:])
                )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)

            replace_node_with_graph_module(model, new_node, mod)

    graph.eliminate_dead_code()
    model.graph.lint()
    model.compile()


def remove_softmax_dtype_cast(model: torch.fx.GraphModule):
    graph = model.graph
    for node in list(model.graph.nodes):
        if node.target == torch.ops.aten.softmax.int:
            node.args = node.args[:2]

    graph.lint()
    model.recompile()
    return model


def split_dense_spmm_node(model: GraphModule):
    for node in list(model.graph.nodes):
        if node.target != torch.ops.quantized_ops.linear_mx.default:
            continue

        if "A_data" not in node.kwargs:
            continue

        kwargs = dict(node.kwargs)
        A_data = kwargs.pop("A_data")
        A_indices = kwargs.pop("A_indices")
        A_indptr = kwargs.pop("A_indptr")
        weight_transposed = kwargs.pop("weight_transposed", False)

        node.kwargs = kwargs

        with model.graph.inserting_before(node):
            spmm_node = model.graph.call_function(
                torch.ops.quantized_ops.spmm_csr.default,
                (
                    A_data,
                    A_indices,
                    A_indptr,
                    node.args[1],
                    kwargs.get("weight_scale"),
                    kwargs.get("weight_code"),
                    kwargs.get("block_size"),
                    weight_transposed,
                ),
            )

        with model.graph.inserting_after(node):
            add_node = model.graph.call_function(
                torch.ops.aten.add.Tensor,
                (node, spmm_node),
            )

        for user in list(node.users):
            if id(user) != id(add_node):
                user.replace_input_with(node, add_node)

        propagate_shape(spmm_node)
        propagate_shape(node)
        propagate_shape(add_node)

        tiled_shapes = node.meta.get("tiled_shapes")
        tile_strides = node.meta.get("tile_strides")
        tiling = node.meta.get("l2_tiling")

        if tiled_shapes is not None:
            spmm_node.meta["tiled_shapes"] = {
                "data": tiled_shapes["A_data"],
                "indices": tiled_shapes["A_indices"],
                "indptr": tiled_shapes["A_indptr"],
                "B": tiled_shapes["weight"],
                "B_scale": tiled_shapes["weight_scale"],
                "output": tiled_shapes["output"],
            }
            spmm_node.meta["tile_strides"] = {
                "indptr": tile_strides["A_indptr"],
            }
            spmm_node.meta["l2_tiling"] = tiling

    model.graph.lint()
    model.recompile()
    return model
