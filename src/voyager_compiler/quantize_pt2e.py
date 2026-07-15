import copy
import logging
import math
import operator
import re
from collections import OrderedDict
from dataclasses import asdict, replace
from typing import Dict, Tuple, Any, Optional, Callable, List

import torch
from torch import Tensor
from torch.ao.quantization.fx.utils import assert_and_get_unique_device
from torch.fx import GraphModule, Graph, Node
from torchao.quantization.pt2e import FakeQuantizeBase, ObserverOrFakeQuantize
from torchao.quantization.pt2e.quantizer import (
    EdgeOrNode,
    QuantizationSpecBase,
)

import voyager_compiler as qt
from voyager_compiler.fake_quantize import (
    _DerivedObserverOrFakeQuantize,
    FusedAmaxObsFakeQuantize,
    get_quantization_map,
)
from voyager_compiler.quantizer.quantizer import (
    QuantizationSpec,
    DerivedQuantizationSpec,
)
from voyager_compiler.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from voyager_compiler.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
)

from .codegen.aten_classifier import is_compute_op
from .codegen.passes.utils import get_arg_value
from .codegen.mapping_utils import (
    is_gemm_op,
    is_mha_qkv_permute,
    is_nop,
    is_reshape_op,
    is_matmul,
    reshape_preserves_full_blocks,
)
from .decomposed import quantized_ops_lib

logger = logging.getLogger(__name__)


def _create_obs_or_fq_from_qspec(quantization_spec, obs_or_fq_map, is_qat):
    """Create observer or fake quantize objects based on quantization spec

    Args:
       quantization_spec: used to store parameters to create the observer or fake quantizer
       obs_or_fq_map: this is a map from edge/output to the corresponding observer/fake_quant
       instance, it may be reused for different edge/output depending on configuration
    """
    if quantization_spec is None:
        return None
    if isinstance(quantization_spec, DerivedQuantizationSpec):
        kwargs = {
            "dtype": quantization_spec.dtype,
            "derive_qparams_fn": quantization_spec.derive_qparams_fn,
        }
        edge_or_nodes = quantization_spec.derived_from
        obs_or_fqs = [obs_or_fq_map[k] for k in edge_or_nodes]
        kwargs["obs_or_fqs"] = obs_or_fqs
        return _DerivedObserverOrFakeQuantize.with_args(**kwargs)()

    assert isinstance(quantization_spec, QuantizationSpec)
    observer_or_fake_quant_ctr = quantization_spec.observer_or_fake_quant_ctr
    kwargs_dict = asdict(quantization_spec)
    kwargs = copy.deepcopy(kwargs_dict)
    kwargs.pop("observer_or_fake_quant_ctr")
    return observer_or_fake_quant_ctr.with_args(**kwargs)()


def _get_obs_or_fq_map(
    edge_or_node_to_group_id: Dict[EdgeOrNode, int],
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase],
    is_qat: bool,
) -> Dict[EdgeOrNode, ObserverOrFakeQuantize]:
    """Generates the EdgeOrNode to observer/fake_quant instances
    Makes sure that for EdgeOrNode that has the same group_id should have the same observer or fake quant
    instances
    """
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize] = {}
    group_id_to_obs_or_fq: Dict[int, ObserverOrFakeQuantize] = {}
    for edge_or_node, qspec in edge_or_node_to_qspec.items():
        group_id = edge_or_node_to_group_id[edge_or_node]
        if group_id not in group_id_to_obs_or_fq:
            # TODO: maybe edge_or_node_to_qspec should be edge_or_node_to_root_qspec, this will simplify
            # the implementation for _create_obs_or_fq_from_qspec
            group_id_to_obs_or_fq[group_id] = _create_obs_or_fq_from_qspec(
                qspec, obs_or_fq_map, is_qat
            )
        obs_or_fq_map[edge_or_node] = group_id_to_obs_or_fq[group_id]
    return obs_or_fq_map


def _set_ch_axis(qspec: Optional[QuantizationSpec], ch_axis: int):
    if qspec is None:
        return None
    return replace(qspec, ch_axis=ch_axis)


def get_microscaling_quantizer(
    activation: Optional[QuantizationSpec], weight: Optional[QuantizationSpec]
):
    # Microscaling performs quantization along the reduction dimension
    act_qspec = _set_ch_axis(activation, 1)
    weight_qspec = _set_ch_axis(weight, 1)
    qconfig_conv2d = QuantizationConfig(act_qspec, None, weight_qspec, None)

    act_qspec = _set_ch_axis(activation, -1)
    weight_qspec = _set_ch_axis(weight, -1)
    qconfig_linear = QuantizationConfig(act_qspec, None, weight_qspec, None)

    act0_qspec = _set_ch_axis(activation, -1)
    act1_qspec = _set_ch_axis(activation, -2)
    qconfig_matmul = QuantizationConfig(act0_qspec, None, act1_qspec, None)

    return (
        XNNPACKQuantizer()
        .set_object_type(torch.ops.aten.conv2d.default, qconfig_conv2d)
        .set_object_type(torch.ops.aten.linear.default, qconfig_linear)
        .set_object_type(torch.ops.aten.matmul.default, qconfig_matmul)
    )


def get_per_channel_act_quantizer(
    input_activation: Optional[QuantizationSpec],
    output_activation: Optional[QuantizationSpec],
    weight: Optional[QuantizationSpec],
    bias: Optional[QuantizationSpec],
):
    # Convolution layer only support per-tensor activation quantization
    act_qspec = replace(input_activation, qscheme=qt.per_tensor_symmetric)
    qconfig_conv2d = QuantizationConfig(
        act_qspec, output_activation, weight, bias
    )

    # Perform quantization along the outer dimension
    act_qspec = replace(input_activation, ch_axis=-2)
    qconfig_linear = QuantizationConfig(
        act_qspec, output_activation, weight, bias
    )

    act0_qspec = replace(input_activation, ch_axis=-2)
    act1_qspec = replace(input_activation, ch_axis=-1)
    qconfig_matmul = QuantizationConfig(
        act0_qspec, output_activation, act1_qspec, None
    )

    return (
        XNNPACKQuantizer()
        .set_object_type(torch.ops.aten.conv2d.default, qconfig_conv2d)
        .set_object_type(torch.ops.aten.linear.default, qconfig_linear)
        .set_object_type(torch.ops.aten.matmul.default, qconfig_matmul)
    )


def derive_bias_qparams_fn(
    obs_or_fqs: List[ObserverOrFakeQuantize],
) -> Tuple[Tensor, Tensor]:
    assert (
        len(obs_or_fqs) == 2
    ), "Expecting two obs/fqs, one for activation and one for weight, got: {}".format(
        len(obs_or_fqs)
    )
    act_obs_or_fq = obs_or_fqs[0]
    weight_obs_or_fq = obs_or_fqs[1]
    act_scale = act_obs_or_fq.calculate_qparams()
    weight_scale = weight_obs_or_fq.calculate_qparams()
    return act_scale * weight_scale.flatten()


def get_default_quantizer(
    input_activation: Optional[QuantizationSpec] = None,
    output_activation: Optional[QuantizationSpec] = None,
    weight: Optional[QuantizationSpec] = None,
    bias: Optional[QuantizationSpec] = None,
    record_histogram: bool = False,
    force_scale_power_of_two: bool = False,
    **kwargs: Any,
) -> XNNPACKQuantizer:
    """
    Create a quantizer for the given activation and weight quantization specifications.

    Parameters:
    - activation: The quantization spec for activations.
    - weight: The quantization spec for weights.
    - record_histogram: Whether to record histogram of input.
    - force_scale_power_of_two: Whether to force the scaling factor to be a power of two.

    Returns:
    - A configured XNNPACKQuantizer.
    """

    observer_or_fake_quant_ctr = FusedAmaxObsFakeQuantize.with_args(
        record_histogram=record_histogram,
        force_scale_power_of_two=force_scale_power_of_two,
    )

    qschemes = []
    if input_activation is not None:
        input_activation = QuantizationSpec.from_str(input_activation)
        input_activation.observer_or_fake_quant_ctr = observer_or_fake_quant_ctr
        qschemes.append(input_activation.qscheme)

    if output_activation is not None:
        output_activation = QuantizationSpec.from_str(output_activation)
        output_activation.observer_or_fake_quant_ctr = (
            observer_or_fake_quant_ctr
        )

    if weight is not None:
        weight = QuantizationSpec.from_str(weight)
        weight.observer_or_fake_quant_ctr = observer_or_fake_quant_ctr
        qschemes.append(weight.qscheme)

    qschemes = [qs for qs in qschemes if qs is not None]
    if len(qschemes) > 0 and qt.microscaling not in qschemes:
        assert (
            bias is not None
        ), "Bias quantization is required when quantizing activations and weights."

    # We will specify derived_from later in the quantizer.
    # We use bias data type to imply the accumulation data type for the output.
    if bias is not None:
        bias = DerivedQuantizationSpec(
            derived_from=None,
            derive_qparams_fn=derive_bias_qparams_fn,
            dtype=bias,
        )

    if qt.microscaling in qschemes:
        assert (
            len(set(qschemes)) == 1
        ), f"Quantization scheme {qschemes[0]} does not work with {qschemes[1]}"
        return get_microscaling_quantizer(input_activation, weight)

    if weight is not None and weight.qscheme == qt.per_channel_symmetric:
        assert weight.ch_axis == 0, (
            f"Per-channel weight quantization only supports quantizing output "
            "channel dimension (dim=0)."
        )

    if (
        input_activation is not None
        and input_activation.qscheme == qt.per_channel_symmetric
    ):
        return get_per_channel_act_quantizer(
            input_activation, output_activation, weight, bias
        )

    qconfig = QuantizationConfig(
        input_activation, output_activation, weight, bias
    )
    qconfig_matmul = QuantizationConfig(
        input_activation, output_activation, input_activation, None
    )
    return (
        XNNPACKQuantizer()
        .set_object_type(torch.ops.aten.conv2d.default, qconfig)
        .set_object_type(torch.ops.aten.linear.default, qconfig)
        .set_object_type(torch.ops.aten.matmul.default, qconfig_matmul)
        .set_object_type(torch.ops.aten.add.Tensor, qconfig)
        .set_object_type(torch.ops.aten.add_.Tensor, qconfig)
    )


def export_model(
    model: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
    strict: bool = False,
):
    from transformers.utils.import_utils import is_torch_greater_or_equal

    export_args = (model, args, kwargs)
    export_kwargs = {"dynamic_shapes": dynamic_shapes, "strict": strict}

    if is_torch_greater_or_equal("2.10"):
        from torch._export.utils import _disable_aten_to_metadata_assertions

        # Each ``.to(dtype)`` makes export emit an ``_assert_tensor_metadata``
        # node pinning the dtype seen at trace time.
        with _disable_aten_to_metadata_assertions():
            gm = torch.export.export(*export_args, **export_kwargs)
        return gm.module(check_guards=False)
    elif is_torch_greater_or_equal("2.8"):
        from torch._export.utils import _disable_aten_to_metadata_assertions

        with _disable_aten_to_metadata_assertions():
            gm = torch.export.export_for_training(*export_args, **export_kwargs)
        return gm.module()
    elif is_torch_greater_or_equal("2.5"):
        return torch.export.export_for_training(
            *export_args, **export_kwargs
        ).module()
    elif is_torch_greater_or_equal("2.0"):
        return torch._export.capture_pre_autograd_graph(
            model, args, kwargs, dynamic_shapes=dynamic_shapes
        )
    else:
        raise RuntimeError(f"Require torch>=2.0, but found {torch.__version__}")


def prepare_pt2e(model, quantizer, args=None, kwargs=None, dynamic_shapes=None):
    from torchao.quantization.pt2e import prepare
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e

    # replace the default implementation of _create_obs_or_fq_from_qspec
    prepare._get_obs_or_fq_map = _get_obs_or_fq_map

    if not isinstance(model, GraphModule):
        model = export_model(model, args, kwargs, dynamic_shapes=dynamic_shapes)

    return prepare_pt2e(model, quantizer)


def _get_module(
    node: Node, named_modules: Dict[str, torch.nn.Module]
) -> Optional[torch.nn.Module]:
    """
    If `node` refers to a call_module node, return the module, else None.
    """
    if node.op == "call_module" and str(node.target) in named_modules:
        return named_modules[str(node.target)]
    else:
        return None


# Returns a function that can get a new attribute name for module with given
# prefix, for example,
# >> get_new_observer_name = get_new_attr_name_with_prefix('_observer')
# >> new_name = get_new_observer_name(module)
# new_name will be an unused attribute name on module, e.g. `_observer_1`
def get_new_attr_name_with_prefix(prefix: str) -> Callable:
    prefix = prefix.replace(".", "_")

    def get_new_attr_name(module: torch.nn.Module):
        def get_attr_name(i: int):
            return prefix if i == 0 else prefix + f"_{i}"

        i = 0
        attr_name = get_attr_name(i)
        while hasattr(module, attr_name):
            i += 1
            attr_name = get_attr_name(i)
        return attr_name

    return get_new_attr_name


def create_getattr_from_value(
    module: torch.nn.Module, graph: Graph, prefix: str, value: Any
) -> Node:
    """
    Given a value of any type, creates a getattr node corresponding to the value and
    registers the value as a buffer to the module.
    """
    get_new_attr_name = get_new_attr_name_with_prefix(prefix)
    attr_name = get_new_attr_name(module)
    new_value = (
        value.clone().detach()
        if isinstance(value, torch.Tensor)
        else torch.tensor(value)
    )
    module.register_buffer(attr_name, new_value)
    # Create get_attr with value
    attr_node = graph.create_node("get_attr", attr_name)
    return attr_node


def _replace_observer_with_quantize_dequantize_node_decomposed(
    model: torch.fx.GraphModule,
    node: Node,
    modules: Dict[str, torch.nn.Module],
    output_dtype: str = None,
):
    graph = model.graph
    assert modules is not None
    assert isinstance(node.target, str)
    activation_post_process = modules[node.target]
    device = assert_and_get_unique_device(activation_post_process)

    dtype = next(iter(model.parameters())).dtype
    scale = activation_post_process.calculate_qparams().to(dtype)

    orig_fq_users = list(node.users.keys())
    input_node = node.args[0]
    if input_node.op == "get_attr":
        # Quantize weight and remove the fq module
        param = model.get_parameter(input_node.target)
        param.data = torch.ops.quantized_ops.quantize(
            param.data, scale, qmap=activation_post_process.qmap
        )
        node.replace_all_uses_with(input_node)

        # Annotate weight dtype
        input_node.meta["dtype"] = activation_post_process.dtype

        # Reshape the scale to match the shape of the output tensor for
        # per-channel weight quantization.
        if scale.ndim == 4:
            scale = scale.view(-1, 1, 1)
        elif scale.ndim == 2:
            scale = scale.view(-1)
    else:
        # Replace fake quant module with a quantize node
        with graph.inserting_before(node):
            qparam_node = create_getattr_from_value(
                model, graph, next(iter(node.users)).name + "_scale", scale
            )
            # TODO quantization map can be shared among multiple quantize nodes?
            get_attr_node = create_getattr_from_value(
                model, graph, "qmap", activation_post_process.qmap
            )
            quantized_node = graph.call_function(
                torch.ops.quantized_ops.quantize.default,
                (node.args[0], qparam_node, None, None, None, get_attr_node),
            )

        # Annotate input dtype
        quantized_node.meta["dtype"] = activation_post_process.dtype

        node.replace_all_uses_with(quantized_node)
    graph.erase_node(node)

    # We don't need to insert dequantize node for bias
    if (
        isinstance(activation_post_process, _DerivedObserverOrFakeQuantize)
        or activation_post_process.qscheme is None
    ):
        return

    for user_node in orig_fq_users:
        if is_gemm_op(user_node):
            user_node.meta["dtype"] = output_dtype

            # Insert dequantize node before the node that appear the earlist in the graph
            node_index_map = {n: i for i, n in enumerate(graph.nodes)}
            all_user_nodes = sorted(
                user_node.users.keys(),
                key=lambda n: node_index_map.get(n, float("inf")),
            )
            maybe_dq_node = all_user_nodes[0]

            if (
                maybe_dq_node.op != "call_function"
                or maybe_dq_node.target
                != torch.ops.quantized_ops.dequantize.default
            ):
                # Insert a dequantize node after the gemm operation
                quant_map = get_quantization_map(output_dtype, device)
                with graph.inserting_before(maybe_dq_node):
                    qparam_node = create_getattr_from_value(
                        model, graph, user_node.name + "_scale", scale
                    )
                    get_attr_node = create_getattr_from_value(
                        model, graph, "qmap", quant_map
                    )
                    dequantized_node = graph.call_function(
                        torch.ops.quantized_ops.dequantize.default,
                        (
                            user_node,
                            qparam_node,
                            None,
                            None,
                            None,
                            get_attr_node,
                        ),
                    )

                # We need to save orig users before updating users because
                # the list of users will change as we update users
                orig_users = list(user_node.users.keys())
                for user in orig_users:
                    if id(user) == id(dequantized_node):
                        continue
                    user.replace_input_with(user_node, dequantized_node)
            else:
                # Update the scale if a dequantize node already exists
                qparam_node = maybe_dq_node.args[1]
                buffer = model.get_buffer(qparam_node.target)
                model.register_buffer(qparam_node.target, scale * buffer)
        else:
            # Insert a dequantize node after the quantize node
            with graph.inserting_before(quantized_node.next):
                qparam_node = create_getattr_from_value(
                    model, graph, user_node.name + "_scale", scale
                )
                dequantized_node = graph.call_function(
                    torch.ops.quantized_ops.dequantize.default,
                    (quantized_node, qparam_node),
                )

            user_node.replace_input_with(quantized_node, dequantized_node)


MX_OP_MAPPING = {
    torch.ops.aten.conv2d.default: torch.ops.quantized_ops.conv2d_mx.default,
    torch.ops.aten.linear.default: torch.ops.quantized_ops.linear_mx.default,
    torch.ops.aten.matmul.default: torch.ops.quantized_ops.matmul_mx.default,
}


def _replace_observer_with_quantize_mx_node_decomposed(
    model: torch.fx.GraphModule, node: Node, modules: Dict[str, torch.nn.Module]
):
    graph = model.graph
    assert modules is not None
    assert isinstance(node.target, str)
    activation_post_process = modules[node.target]
    device = assert_and_get_unique_device(activation_post_process)

    input_node = node.args[0]
    input_dtype = activation_post_process.dtype
    node_to_quantize = input_node

    if isinstance(activation_post_process.ch_axis, int):
        activation_post_process.ch_axis = (activation_post_process.ch_axis,)

    if activation_post_process.outlier_threshold is not None:
        max_outlier_pct = (
            math.ceil(activation_post_process.max_outlier_pct * 100) / 100.0
        )
        activation_post_process.max_outlier_pct = max(max_outlier_pct, 0.05)
        logger.info(
            f"{node.target} has maximum outlier percentage {max_outlier_pct:.2%}"
        )

    quant_map = get_quantization_map(activation_post_process.dtype, device)
    dequant_code, quant_code = None, None

    if isinstance(quant_map, tuple):
        matches = re.findall(r"\d+", activation_post_process.dtype)
        activation_post_process.dtype = f"int{matches[0]}"

        indices, values = quant_map
        activation_post_process.qmap = indices

        with graph.inserting_before(node):
            dequant_code = create_getattr_from_value(
                model, graph, "code", values
            )
            if input_node.op != "get_attr":
                midpoints = (values[:-1] + values[1:]) / 2
                quant_code = create_getattr_from_value(
                    model, graph, "code", midpoints
                )

        # NF4_[B] means approximate NormalFloat4 with B-bit integer
        if len(matches) > 1:
            dequant_code.meta["dtype"] = f"int{matches[1]}"

    if input_node.op == "get_attr":
        # quantize model parameter and remove the fq module
        try:
            param = model.get_parameter(input_node.target)
        except AttributeError:
            param = model.get_buffer(input_node.target)

        scale = torch.ops.quantized_ops.calculate_mx_qparam(
            param.data,
            activation_post_process.ch_axis,
            activation_post_process.block_size,
            activation_post_process.quant_max,
            activation_post_process.force_scale_power_of_two,
            activation_post_process.scale_qmap,
        )

        weight = torch.ops.quantized_ops.quantize(
            param.data,
            scale,
            axes=activation_post_process.ch_axis,
            block_size=activation_post_process.block_size,
            qmap=activation_post_process.qmap,
        )

        with graph.inserting_before(node):
            quantized_node = create_getattr_from_value(
                model, graph, input_node.name + "_" + input_dtype, weight
            )
            scale_node = create_getattr_from_value(
                model, graph, input_node.name + "_scale", scale
            )
    else:
        with graph.inserting_before(node):
            get_attr_node = create_getattr_from_value(
                model, graph, "qmap", activation_post_process.qmap
            )

            scale_qmap = None
            if activation_post_process.scale_qmap is not None:
                scale_qmap = create_getattr_from_value(
                    model, graph, "qmap", activation_post_process.scale_qmap
                )

            target = torch.ops.quantized_ops.quantize_mx.default
            args = [
                node_to_quantize,
                get_attr_node,
                activation_post_process.ch_axis,
                activation_post_process.block_size,
                activation_post_process.quant_max,
                activation_post_process.force_scale_power_of_two,
                scale_qmap,
                quant_code,
            ]

            if activation_post_process.outlier_threshold is not None:
                target = torch.ops.quantized_ops.quantize_mx_outlier.default
                args.extend(
                    [
                        float(activation_post_process.outlier_threshold),
                        activation_post_process.max_outlier_pct,
                    ]
                )
                num_outputs = 5
            else:
                num_outputs = 2

            quantize_mx_node = graph.call_function(target, tuple(args))

            output_nodes = [
                graph.call_function(operator.getitem, (quantize_mx_node, i))
                for i in range(num_outputs)
            ]

        scale_dtype = (
            "fp8_e8m0"
            if activation_post_process.force_scale_power_of_two
            else activation_post_process.scale_dtype
        )

        if num_outputs == 5:
            csr_data_node = output_nodes[0]
            csr_indices_node = output_nodes[1]
            csr_indptr_node = output_nodes[2]
            scale_node = output_nodes[3]
            quantized_node = output_nodes[4]
            dtype_tuple = (
                None,
                None,
                None,
                scale_dtype,
                activation_post_process.dtype,
            )
        else:
            scale_node, quantized_node = output_nodes
            dtype_tuple = (scale_dtype, activation_post_process.dtype)

        quantize_mx_node.meta["dtype"] = dtype_tuple

    quantized_node.meta["dtype"] = activation_post_process.dtype

    if activation_post_process.force_scale_power_of_two:
        scale_node.meta["dtype"] = "fp8_e8m0"
    elif activation_post_process.scale_dtype is not None:
        scale_node.meta["dtype"] = activation_post_process.scale_dtype

    orig_fq_users = list(node.users.keys())

    node.replace_all_uses_with(quantized_node)
    graph.erase_node(node)

    if len(input_node.users) == 0:
        graph.erase_node(input_node)

    for user in orig_fq_users:
        # Keep the original nodes for other users
        kwarg1, kwarg2 = dequant_code, scale_node

        # Skip device alignment node
        if user.target == torch.Tensor.to:
            user_device = user.args[1]
            with graph.inserting_before(user):
                if kwarg1 is not None:
                    kwarg1 = graph.call_function(
                        torch.Tensor.to, (dequant_code, user_device)
                    )
                kwarg2 = graph.call_function(
                    torch.Tensor.to, (scale_node, user_device)
                )
            user = next(iter(user.users))

        kwargs = OrderedDict(user.kwargs)
        kwargs.setdefault("block_size", activation_post_process.block_size)
        if input_node.op == "get_attr" or id(quantized_node) == id(
            user.args[1]
        ):
            kwargs.setdefault("weight_code", kwarg1)
            kwargs.setdefault("weight_scale", kwarg2)
        else:
            kwargs.setdefault("input_code", kwarg1)
            kwargs.setdefault("input_scale", kwarg2)

        # Sort kwargs so that they can be accessed sequentially during MHA splitting
        order = [
            "input_scale",
            "weight_scale",
            "block_size",
            "input_code",
            "weight_code",
            "A_data",
            "A_indices",
            "A_indptr",
        ]
        kwargs = OrderedDict(
            (key, kwargs[key]) for key in order if key in kwargs
        )

        # Replace the node with its MX counterpart
        if user.target in MX_OP_MAPPING:
            with graph.inserting_before(user):
                mx_op_node = graph.call_function(
                    MX_OP_MAPPING[user.target], user.args, kwargs
                )

            user.replace_all_uses_with(mx_op_node)
            graph.erase_node(user)

            mx_op_node.meta = user.meta
        elif user.target in MX_OP_MAPPING.values():
            mx_op_node = user
            mx_op_node.kwargs = kwargs
        elif user.target == torch.ops.quantized_ops.spmm_csr.default:
            assert (
                input_node.op == "get_attr"
            ), f"Expect input node to be a get_attr, but found {input_node.op}"
            user.args = user.args[:-1] + (quantized_node,)
            user.kwargs = {
                "B_scale": kwargs.get("weight_scale"),
                "B_code": kwargs.get("weight_code"),
                "block_size": activation_post_process.block_size,
            }
        else:
            raise RuntimeError(
                f"Unsupported user node {user.target} for quantization, "
                f"expected one of {list(MX_OP_MAPPING.keys())}"
            )

        if (
            activation_post_process.outlier_threshold is not None
            and input_node.op != "get_attr"
        ):
            # For now only support linear layers
            assert mx_op_node.target in [
                torch.ops.aten.linear.default,
                torch.ops.aten.matmul.default,
                torch.ops.quantized_ops.linear_mx.default,
                torch.ops.quantized_ops.matmul_mx.default,
            ], f"Only GEMM is supported for outlier suppresion, got {user.target}"

            weight_node = mx_op_node.args[1]

            if mx_op_node.target in [
                torch.ops.quantized_ops.linear_mx.default,
                torch.ops.quantized_ops.matmul_mx.default,
            ]:
                mx_op_node.kwargs = {
                    **mx_op_node.kwargs,
                    "A_data": csr_data_node,
                    "A_indices": csr_indices_node,
                    "A_indptr": csr_indptr_node,
                }
            else:
                with graph.inserting_before(mx_op_node):
                    spmm_node = graph.call_function(
                        torch.ops.quantized_ops.spmm_csr.default,
                        (
                            csr_data_node,
                            csr_indices_node,
                            csr_indptr_node,
                            weight_node,
                        ),
                        {
                            "B_scale": kwargs.get("weight_scale"),
                            "B_code": kwargs.get("weight_code"),
                            "block_size": activation_post_process.block_size,
                        },
                    )

                with graph.inserting_after(mx_op_node):
                    add_node = graph.call_function(
                        torch.ops.aten.add.Tensor, (spmm_node, mx_op_node)
                    )

                mx_op_node.replace_all_uses_with(add_node)
                add_node.replace_input_with(add_node, mx_op_node)


def _replace_observer_with_groupwise_affine_q_dq_node_decomposed(
    model: torch.fx.GraphModule, node: Node, modules: Dict[str, torch.nn.Module]
):
    graph = model.graph
    assert modules is not None
    assert isinstance(node.target, str)
    activation_post_process = modules[node.target]
    device = assert_and_get_unique_device(activation_post_process)

    if isinstance(activation_post_process.ch_axis, int):
        activation_post_process.ch_axis = (activation_post_process.ch_axis,)

    input_node = node.args[0]

    if input_node.op == "get_attr":
        try:
            param = model.get_parameter(input_node.target)
        except AttributeError:
            param = model.get_buffer(input_node.target)

        activation_post_process(param.data)
        scale, zero_point = activation_post_process.calculate_qparams()
        scale = scale.to(param.data.dtype)
        zero_point = zero_point.to(param.data.dtype)

        weight = torch.ops.quantized_ops.quantize(
            param.data,
            scale,
            zero_point,
            activation_post_process.ch_axis,
            activation_post_process.block_size,
            activation_post_process.qmap,
        )

        with graph.inserting_before(node):
            quantized_node = create_getattr_from_value(
                model,
                graph,
                input_node.name + "_" + activation_post_process.dtype,
                weight,
            )
            scale_node = create_getattr_from_value(
                model, graph, input_node.name + "_scale", scale
            )
            zero_point_node = create_getattr_from_value(
                model, graph, input_node.name + "_zero_point", zero_point
            )
    else:
        raise NotImplementedError

    quantized_node.meta["dtype"] = activation_post_process.dtype

    if activation_post_process.scale_dtype is not None:
        scale_node.meta["dtype"] = activation_post_process.scale_dtype
        zero_point_node.meta["dtype"] = activation_post_process.scale_dtype

    # Insert a dequantize node after the quantize node
    with graph.inserting_before(node):
        dequantized_node = graph.call_function(
            torch.ops.quantized_ops.dequantize.default,
            (
                quantized_node,
                scale_node,
                zero_point_node,
                activation_post_process.ch_axis,
                activation_post_process.block_size,
            ),
        )

    node.replace_all_uses_with(dequantized_node)
    graph.erase_node(node)

    if len(input_node.users) == 0:
        graph.erase_node(input_node)


def _eliminate_dequantize_with_no_effect(model: GraphModule):
    for node in model.graph.nodes:
        if node.target != torch.ops.quantized_ops.dequantize.default:
            continue

        scale_node = node.args[1]
        scale = model.get_buffer(scale_node.target)
        if scale_node.op != "get_attr" or torch.any(scale != 1):
            continue

        # During integer quantization, the dequantize node also perform a
        # quantization to the output dtype
        output_qmap = get_arg_value(node, 6, "output_qmap")
        if output_qmap is not None:
            continue

        node.replace_all_uses_with(node.args[0])
        model.graph.erase_node(node)
        logger.info(f"Eliminate dequantize node {node} with no effect")

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()

    return model


# Ops a quantize can be lifted over: each only moves data, and takes a single
# tensor to do it.  ``expand`` is the one that earns its keep here -- lifting a
# quantize over GQA's ``repeat_kv`` is what stops it quantizing 4x the heads.
_HOISTABLE_OPS = (
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.select.int,
    torch.ops.aten.expand.default,
    torch.ops.aten.repeat.default,
)

# Ops that regroup dims without moving an element: they preserve row-major
# order, so a quantize can be lifted over one even when it cuts across the axis
# the quantize blocks along -- as long as the blocks come out the same
# (``_blocks_survive_regroup``).  These are also the only ops whose size
# argument ``_replay_relayout`` knows how to rebuild for the scale.
_REGROUP_OPS = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
)

# MHA head splitting rejoins the per-head results with one of these, so a
# quantize lifted over it has to be duplicated onto every branch.
_FORK_OPS = (
    torch.ops.aten.stack.default,
    torch.ops.aten.cat.default,
)

_QUANTIZE_MX = torch.ops.quantized_ops.quantize_mx.default
_QUANTIZE_OPS = (
    torch.ops.quantized_ops.quantize.default,
    torch.ops.quantized_ops.dequantize.default,
    _QUANTIZE_MX,
)

# ``quantize_mx`` returns ``(scale, value)``.  The value inherits the relayout
# ops the quantize was lifted over; the scale gets a replayed copy of them.
_MX_VALUE = 1


def _is_relayout(node) -> bool:
    return (
        isinstance(node, Node)
        and (
            is_nop(node) or is_reshape_op(node) or node.target in _HOISTABLE_OPS
        )
        and len(node.all_input_nodes) == 1
    )


def _axes_above(
    node: Node, axes: Tuple[int, ...], block_size: Optional[int]
) -> Optional[Tuple[int, ...]]:
    """``axes`` -- the axes a microscaling quantize blocks along, read against
    ``node``'s output -- restated against its input.  ``None`` if the blocks do
    not survive ``node``, which is then as far as the quantize can be lifted.

    Axes count from the end, so an op that only rearranges dims to the *left*
    of a block axis leaves it alone: that covers every op on the ``repeat_kv``
    path (``unsqueeze``, ``expand``, the head-flattening ``reshape``).  A
    transpose or a permute genuinely moves the axis, so it is remapped.  A
    reshape that regroups the block axis itself still passes if the blocks come
    out the same set of elements.  A per-tensor quantize passes ``()`` and is
    unaffected.
    """
    out_shape = tuple(node.value.shape)
    in_shape = tuple(node.args[0].value.shape)
    rank = len(out_shape)

    if node.target is torch.ops.aten.transpose.int:
        a, b = (int(d) % rank - rank for d in node.args[1:3])
        swap = {a: b, b: a}
        return tuple(swap.get(x, x) for x in axes)

    if node.target is torch.ops.aten.permute.default:
        perm = [int(p) % rank for p in node.args[1]]  # out dim i <- in perm[i]
        return tuple(perm[x + rank] - rank for x in axes)

    if any(in_shape[x:] != out_shape[x:] for x in axes):
        # A reshape regrouping the block axis is still crossable if the blocks
        # come out the same sets of elements.
        if node.target not in _REGROUP_OPS or len(axes) != 1:
            return None
        a = axes[0]
        if block_size is None or len(in_shape) < -a or len(out_shape) < -a:
            return None
        if not reshape_preserves_full_blocks(
            in_shape,
            a + len(in_shape),
            out_shape,
            a + len(out_shape),
            block_size,
        ):
            return None
    return axes


def _relayout_path(
    start,
    axes: Tuple[int, ...],
    block_size: Optional[int],
    keep_head_permute=False,
):
    """Walk up from ``start`` over relayout ops, restating ``axes`` at each.

    Returns ``(src, path, axes_at, src_axes)``: ``path`` is the ops crossed,
    nearest ``start`` first; ``src`` the node that actually computed the data;
    ``axes_at[k]`` the block axes as seen at ``path[k]``'s output; ``src_axes``
    those at ``src``.  The walk stops at the first node that computes, that
    someone else also reads, or that the blocks would not survive.

    ``keep_head_permute`` also stops it at an MHA head permute -- a *fusable*
    reshape, one the GEMM below can store straight through
    (``fuse_reshape_with_output``).  That is exactly where a multi-output
    quantize wants to sit: fused as the last op of that group, quantizing the
    tile on its way out.  Lifted past it, it would leave a ``getitem`` between
    the two and neither could fuse.  A single-output quantize has no ``getitem``
    and steps over freely.
    """
    path, axes_at = [], []
    src = start
    while (
        _is_relayout(src)
        and len(src.users) == 1
        and not (keep_head_permute and is_mha_qkv_permute(src))
    ):
        above = _axes_above(src, axes, block_size)
        if above is None:
            break
        path.append(src)
        axes_at.append(axes)
        axes = above
        src = src.args[0]
    return src, path, axes_at, axes


def _copy_quantize_above(model: GraphModule, node: Node, src: Node, axes):
    """A copy of quantize ``node`` reading ``src``, inserted right after it."""
    from .pt2e_utils import propagate_shape

    graph = model.graph
    remap = {node.args[0]: src}
    with graph.inserting_before(src.next):
        for n in node.all_input_nodes:
            if n not in remap:
                remap[n] = graph.node_copy(n)
        new = graph.node_copy(node, lambda n: remap[n])

    if node.target is _QUANTIZE_MX:
        args = list(new.args)
        args[2] = list(axes)
        new.args = tuple(args)

    for n in list(remap.values()) + [new]:
        propagate_shape(n, model)
    new.meta = {
        k: copy.deepcopy(v) if k != "val" else v.clone()
        for k, v in node.meta.items()
    }
    return new


def _replay_relayout(
    graph: Graph, node: Node, src: Node, axes, block_size: int
) -> Node:
    """Copy relayout ``node`` onto ``src``.  ``src`` is the scale of a hoisted
    ``quantize_mx``, so along ``axes`` it holds one element per *block* where
    the original input held one per element -- a shape argument keeps that dim's
    own extent and divides it.  Every other dim is untouched: ``_axes_above``
    already proved the blocks survive.
    """
    new = graph.node_copy(node, lambda n: src if n is node.args[0] else n)
    out_shape = tuple(node.value.shape)

    if node.target in _REGROUP_OPS:
        shape = list(out_shape)
        for a in axes:
            shape[a] = out_shape[a] // block_size
        new.args = (src, shape)
    elif node.target is torch.ops.aten.expand.default:
        # ``-1`` keeps a dim, so naming only the dims this expand actually grows
        # makes the sizes independent of how long the block axis is.
        in_shape = tuple(node.args[0].value.shape)
        new.args = (
            src,
            [
                out_shape[d] if out_shape[d] != in_shape[d] else -1
                for d in range(len(out_shape))
            ],
        )
    return new


def _annotate(path, source: Node) -> None:
    """The relayout ops now sit *below* the quantize, so they carry the dtype of
    the tensor that flows through them -- or none at all, once a dequantize has
    put it back in the clear.  ``source`` is that tensor: the quantize itself,
    or, for a multi-output one, the single output the path was rewired onto (not
    the op, whose ``dtype`` is the pair it returns).
    """
    is_dequantize = source.target is torch.ops.quantized_ops.dequantize.default
    for n in path:
        if is_dequantize:
            n.meta.pop("dtype", None)
        else:
            n.meta["dtype"] = source.meta.get("dtype", None)


def _hoist_forked(model: GraphModule, node: Node) -> bool:
    """Lift a single-output quantize over the relayout ops feeding it.

    A ``stack`` / ``cat`` on the way up (MHA head splitting rejoining its heads)
    forks the walk: every branch is quantized on its own, and the concat then
    joins pieces that are already quantized.
    """
    graph = model.graph
    on_path, moved = [], False
    todo = [(node.args[0], node)]  # (tensor to lift over, the node reading it)

    while todo:
        start, reader = todo.pop()
        src, path, _, _ = _relayout_path(start, (), None)
        on_path.extend(path)
        if path:
            reader = path[-1]

        if src.target in _FORK_OPS and len(src.users) == 1:
            on_path.append(src)
            todo.extend((a, src) for a in src.all_input_nodes)
            continue

        if not path and reader is node:
            continue  # already sitting on its producer

        new = _copy_quantize_above(model, node, src, ())
        reader.replace_input_with(src, new)
        moved = True

    if not moved:
        return False
    _annotate(on_path, node)
    node.replace_all_uses_with(node.args[0])
    graph.erase_node(node)
    return True


def _hoist_microscaling(model: GraphModule, node: Node) -> bool:
    """Lift a ``quantize_mx`` over the relayout ops feeding it, so it quantizes
    the tensor they re-address rather than the one they hand on.

    Those ops move no element, so quantizing above them is the same arithmetic
    on less data -- and what they were going to do (broadcast a KV head, lay a
    tile out for the MXU) the consumer folds into its addressing rather than
    materializing.  Two things halt the walk: an op the quantization blocks do
    not survive (``_axes_above``), and an MHA head permute, where the quantize
    wants to stop -- the GEMM below stores straight through that permute, and
    the quantize fuses onto the end of it (``fuse_reshape_with_output``).

    The op has two outputs, so the value keeps the relayout ops it was lifted
    over and the scale gets a replayed copy of them.  It is never forked: a
    concat can move the axis it blocks along, and head splitting -- the only
    thing that forks -- never runs where microscaling is used.
    """
    from .pt2e_utils import propagate_shape

    graph = model.graph
    outs = {}
    for user in node.users:
        if user.target is not operator.getitem:
            return False
        outs[user.args[1]] = user

    block_size = node.args[3]
    src, path, axes_at, src_axes = _relayout_path(
        node.args[0], node.args[2], block_size, keep_head_permute=True
    )
    if not path:
        return False

    # TODO: we should skip moving quantize_mx if there is no fusable anchor.
    # In the future maybe move this into operator fusion.
    if not is_compute_op(src) and not is_mha_qkv_permute(src):
        return False

    new = _copy_quantize_above(model, node, src, src_axes)

    def unpack(i: int) -> Node:
        """Output ``i`` of the hoisted quantize.  ``quantize_mx``'s ``dtype`` is
        the *pair* it returns, so each output takes its own element of it -- the
        one the ``getitem`` it replaces carried."""
        out = graph.call_function(operator.getitem, (new, i))
        out.meta["dtype"] = outs[i].meta.get("dtype", None)
        propagate_shape(out, model)
        return out

    # The value keeps the relayout ops it was lifted over: rewire them onto it.
    with graph.inserting_before(path[-1]):
        value = unpack(_MX_VALUE)
    path[-1].replace_input_with(src, value)
    _annotate(path, outs[_MX_VALUE])
    outs[_MX_VALUE].replace_all_uses_with(path[0])

    # The scale is one element per block, so it needs its own copy of them.
    for i, old in outs.items():
        if i == _MX_VALUE:
            continue
        with graph.inserting_before(node):
            cur = unpack(i)
            for k in reversed(range(len(path))):
                cur = _replay_relayout(
                    graph, path[k], cur, axes_at[k], block_size
                )
                cur.meta["dtype"] = old.meta.get("dtype", None)
                propagate_shape(cur, model)
        old.replace_all_uses_with(cur)

    return True


def fuse_quantize_dequantize_with_previous_op(
    model: GraphModule, bufferize: bool = False
):
    """Move each quantize / dequantize up the graph to sit directly after the
    op that computed its input, so the two can fuse into one kernel.

    Everything it is lifted over only relayouts data -- a reshape, a transpose,
    the ``stack`` MHA splitting leaves behind, the ``expand`` of GQA's
    ``repeat_kv`` -- so quantizing above them is the same arithmetic on less
    data.  A microscaling ``quantize_mx`` also blocks along an axis and returns
    a scale beside its value, so it takes the ``_hoist_microscaling`` route;
    the rest share the walk but fork over a concat.

    Only the bufferized backend can lower a ``quantize_mx`` that has moved --
    it fuses onto the store of the GEMM it lands on -- so ``bufferize`` says
    whether to lift one at all.
    """
    graph = model.graph

    for node in list(graph.nodes):
        if node.target not in _QUANTIZE_OPS:
            continue
        if node.target is _QUANTIZE_MX:
            if bufferize:
                _hoist_microscaling(model, node)
            continue
        # A blocked plain quantize would need the same axis bookkeeping as
        # quantize_mx, which it does not have; only per-tensor is lifted.
        block_size = get_arg_value(node, 4, "block_size")
        if block_size is not None and block_size > 1:
            continue
        _hoist_forked(model, node)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()

    return model


def swap_matmul_inputs(model: GraphModule):
    graph = model.graph
    modules = dict(model.named_modules(remove_duplicate=False))

    def get_fake_quant_mod(node: Node):
        if node.op == "call_module":
            mod = _get_module(node, modules)
            if isinstance(mod, FakeQuantizeBase):
                return mod

        return None

    target = torch.ops.aten.transpose.int

    def transpose_input(node):
        input_node = node.args[0]
        if input_node.target != target:
            with graph.inserting_before(node):
                transposed_node = graph.call_function(
                    target, (input_node, -1, -2)
                )
            node.replace_input_with(input_node, transposed_node)
        else:
            node.replace_input_with(input_node, input_node.args[0])

    for node in list(graph.nodes):
        if not is_matmul(node):
            continue

        input_node = node.args[0]
        other_node = node.args[1]

        input_fq = get_fake_quant_mod(input_node)
        other_fq = get_fake_quant_mod(other_node)

        if other_fq is None or other_fq.outlier_threshold is None:
            continue

        assert (
            input_fq is None or input_fq.outlier_threshold is None
        ), "Only one input of matmul can have outlier filter"

        node.args = (other_node, input_node)

        transpose_input(input_node)
        transpose_input(other_node)

        other_fq.ch_axis = -1
        input_fq.ch_axis = -2

        with graph.inserting_after(node):
            transposed = graph.call_function(target, (node, -1, -2))

        for user in list(node.users):
            if id(user) != id(transposed):
                user.replace_input_with(node, transposed)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()


def convert_pt2e(
    model: GraphModule,
    output_dtype: str = None,
    eliminate_no_effect: bool = True,
):
    modules = dict(model.named_modules(remove_duplicate=False))

    swap_matmul_inputs(model)

    for node in list(model.graph.nodes):
        if node.op == "call_module":
            mod = _get_module(node, modules)
            assert mod is not None
            if isinstance(mod, FakeQuantizeBase):
                if mod.qscheme == qt.microscaling:
                    _replace_observer_with_quantize_mx_node_decomposed(
                        model, node, modules
                    )
                elif mod.qscheme == qt.group_wise_affine:
                    _replace_observer_with_groupwise_affine_q_dq_node_decomposed(
                        model, node, modules
                    )
                else:
                    _replace_observer_with_quantize_dequantize_node_decomposed(
                        model, node, modules, output_dtype
                    )

    if eliminate_no_effect:
        _eliminate_dequantize_with_no_effect(model)

    model.graph.lint()
    model.graph.eliminate_dead_code(
        is_impure_node=lambda n: n.op in {"placeholder", "output"}
    )
    model.recompile()
    model.delete_all_unused_submodules()

    return model
