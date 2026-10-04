import copy
import logging
import operator
import os
import types
from collections import OrderedDict
from dataclasses import asdict, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.ao.quantization.fx.utils import assert_and_get_unique_device
from torch.fx import Graph, GraphModule, Node, map_arg
from torchao.quantization.pt2e import FakeQuantizeBase, ObserverOrFakeQuantize
from torchao.quantization.pt2e.qat_utils import _fuse_conv_bn_qat
from torchao.quantization.pt2e.quantizer import (
    EdgeOrNode,
    QuantizationSpecBase,
)
from torchao.utils import _assert_and_get_unique_device

from voyager_compiler.codegen.node_info import (
    get_arg_value,
    is_gemm_op,
    is_matmul,
)
from voyager_compiler.export_utils import (
    create_getattr_from_value,
    export_model,
)
from voyager_compiler.quantization.fake_quantize import (
    DirectCastFakeQuantize,
    GroupWiseAffineFakeQuantize,
    MXFakeQuantize,
    RandomHadamardTransform,
    _DerivedObserverOrFakeQuantize,
    _FreezableFlags,
    get_quantization_map,
)
from voyager_compiler.quantization.quantizer.quantizer import (
    DerivedQuantizationSpec,
    QScheme,
    QuantizationSpec,
)
from voyager_compiler.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
)
from voyager_compiler.quantization.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
    _set_ch_axis,
)
from voyager_compiler.shape_prop import fetch_attr

logger = logging.getLogger(__name__)


def _create_obs_or_fq_from_qspec(quantization_spec, obs_or_fq_map, is_qat):
    """Create observer or fake quantize objects based on quantization spec

    Args:
       quantization_spec: used to store parameters to create the observer or
           fake quantizer
       obs_or_fq_map: this is a map from edge/output to the corresponding
           observer/fake_quant instance, it may be reused for different
           edge/output depending on configuration
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


# Spec fields torchao's implicit sharing does not compare, but that the
# readers of one tensor must agree on to share a fake-quant.
_SHARED_FIELDS = (
    "amax_history_len",
    "block_size",
    "power_2_scale",
    "scale_dtype",
    "outlier_threshold",
    "outlier_pct",
    "rht_axis",
)


def _get_obs_or_fq_map(
    edge_or_node_to_group_id: Dict[EdgeOrNode, int],
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase],
    is_qat: bool,
) -> Dict[EdgeOrNode, ObserverOrFakeQuantize]:
    """Generates the EdgeOrNode to observer/fake_quant instances
    Makes sure that for EdgeOrNode that has the same group_id should have the
    same observer or fake quant instances
    """
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize] = {}
    group_id_to_obs_or_fq: Dict[Tuple, ObserverOrFakeQuantize] = {}
    for edge_or_node, qspec in edge_or_node_to_qspec.items():
        # torchao's union-find groups readers by its own spec attributes
        # only; split each group by the fields it does not compare.
        group_id = (edge_or_node_to_group_id[edge_or_node],) + tuple(
            getattr(qspec, field, None) for field in _SHARED_FIELDS
        )
        if group_id not in group_id_to_obs_or_fq:
            # TODO: maybe edge_or_node_to_qspec should be
            # edge_or_node_to_root_qspec, this will simplify the implementation
            # for _create_obs_or_fq_from_qspec
            group_id_to_obs_or_fq[group_id] = _create_obs_or_fq_from_qspec(
                qspec, obs_or_fq_map, is_qat
            )
        obs_or_fq_map[edge_or_node] = group_id_to_obs_or_fq[group_id]
    return obs_or_fq_map


_SDPA = torch.ops.aten.scaled_dot_product_attention.default
# Dropouts, each taking ``(input, p, train)``.
_DROPOUTS = (
    torch.ops.aten.dropout.default,
    torch.ops.aten.dropout_.default,
    torch.ops.aten.feature_dropout.default,
    torch.ops.aten.feature_dropout_.default,
)


def get_microscaling_quantizer(
    activation: Optional[QuantizationSpec],
    weight: Optional[QuantizationSpec],
    error: Optional[QuantizationSpec] = None,
    random_hadamard_transform: Tuple[str, ...] = (),
):
    # Microscaling performs quantization along the reduction dimension
    act_qspec = _set_ch_axis(activation, 1)
    weight_qspec = _set_ch_axis(weight, 1)
    qconfig_conv2d = QuantizationConfig(act_qspec, None, weight_qspec, None)

    act_qspec = _set_ch_axis(activation, -1)
    weight_qspec = _set_ch_axis(weight, -1)
    qconfig_linear = QuantizationConfig(
        act_qspec,
        None,
        weight_qspec,
        None,
        error=error,
        random_hadamard_transform=random_hadamard_transform,
    )

    act0_qspec = _set_ch_axis(activation, -1)
    act1_qspec = _set_ch_axis(activation, -2)
    # With an error spec, training GEMMs pick specs by operand type.
    qconfig_matmul = QuantizationConfig(
        act0_qspec,
        None,
        act1_qspec if error is None else weight,
        None,
        error=error,
        random_hadamard_transform=random_hadamard_transform,
    )

    return (
        XNNPACKQuantizer()
        .set_object_type(torch.ops.aten.conv2d.default, qconfig_conv2d)
        .set_object_type(torch.ops.aten.linear.default, qconfig_linear)
        .set_object_type(torch.ops.aten.matmul.default, qconfig_matmul)
        .set_object_type(_SDPA, qconfig_matmul)
        .set_object_type(MX_OP_MAPPING[_SDPA], qconfig_matmul)
    )


def get_per_channel_act_quantizer(
    input_activation: Optional[QuantizationSpec],
    output_activation: Optional[QuantizationSpec],
    weight: Optional[QuantizationSpec],
    bias: Optional[QuantizationSpec],
):
    # Convolution layer only support per-tensor activation quantization
    act_qspec = replace(input_activation, qscheme=QScheme.PER_TENSOR_SYMMETRIC)
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
    assert len(obs_or_fqs) == 2, (
        "Expecting two obs/fqs, one for activation and one for weight, "
        "got: {}".format(len(obs_or_fqs))
    )
    act_obs_or_fq = obs_or_fqs[0]
    weight_obs_or_fq = obs_or_fqs[1]
    act_scale = act_obs_or_fq.calculate_qparams()
    weight_scale = weight_obs_or_fq.calculate_qparams()
    return act_scale * weight_scale.flatten()


def get_default_quantizer(
    input_activation: Optional[str] = None,
    output_activation: Optional[str] = None,
    weight: Optional[str] = None,
    bias: Optional[str] = None,
    error: Optional[str] = None,
    random_hadamard_transform: Tuple[str, ...] = (),
    **kwargs: Any,
) -> XNNPACKQuantizer:
    """Build a quantizer for conv2d, linear and matmul from spec strings.

    Args:
        input_activation: Spec for GEMM inputs, or None.
        output_activation: Spec for GEMM outputs, or None.
        weight: Spec for weights, or None.
        bias: Dtype the bias is quantized to, with the product of its GEMM's
            input and weight scales; required with non-microscaling input
            and weight specs.
        error: Spec for gradients, or None.  Setting it builds the quantizer
            of ``prepare_training``, whose GEMM configs give each operand
            the spec of its type.
        random_hadamard_transform: GEMM kinds -- ``"fprop"``, ``"dgrad"``,
            ``"wgrad"`` -- whose operands are rotated before quantizing;
            microscaling specs only.
        **kwargs: Ignored.

    Returns:
        The configured quantizer.

    Raises:
        ValueError: ``random_hadamard_transform`` is set with specs that are
            not microscaling.
    """
    qschemes = []
    if input_activation is not None:
        input_activation = QuantizationSpec.from_str(input_activation)
        qschemes.append(input_activation.qscheme)

    if output_activation is not None:
        output_activation = QuantizationSpec.from_str(output_activation)

    if weight is not None:
        weight = QuantizationSpec.from_str(weight)
        qschemes.append(weight.qscheme)

    if error is not None:
        error = QuantizationSpec.from_str(error)

    qschemes = [qs for qs in qschemes if qs is not None]
    if len(qschemes) > 0 and QScheme.MICROSCALING not in qschemes:
        assert bias is not None, (
            "Bias quantization is required when quantizing activations and "
            "weights."
        )

    # We will specify derived_from later in the quantizer.
    # We use bias data type to imply the accumulation data type for the output.
    if bias is not None:
        bias = DerivedQuantizationSpec(
            derived_from=None,
            derive_qparams_fn=derive_bias_qparams_fn,
            dtype=bias,
        )

    if QScheme.MICROSCALING in qschemes:
        return get_microscaling_quantizer(
            input_activation, weight, error, random_hadamard_transform
        )

    if random_hadamard_transform:
        raise ValueError(
            "The random Hadamard transform supports microscaling specs only"
        )

    if weight is not None and weight.qscheme == QScheme.PER_CHANNEL_SYMMETRIC:
        assert weight.ch_axis == 0, (
            "Per-channel weight quantization only supports quantizing output "
            "channel dimension (dim=0)."
        )

    if (
        input_activation is not None
        and input_activation.qscheme == QScheme.PER_CHANNEL_SYMMETRIC
    ):
        return get_per_channel_act_quantizer(
            input_activation, output_activation, weight, bias
        )

    qconfig = QuantizationConfig(
        input_activation, output_activation, weight, bias, error=error
    )
    # With an error spec, training GEMMs pick specs by operand type.
    qconfig_matmul = QuantizationConfig(
        input_activation,
        output_activation,
        input_activation if error is None else weight,
        None,
        error=error,
    )
    return (
        XNNPACKQuantizer()
        .set_object_type(torch.ops.aten.conv2d.default, qconfig)
        .set_object_type(torch.ops.aten.linear.default, qconfig)
        .set_object_type(torch.ops.aten.matmul.default, qconfig_matmul)
    )


def prepare_pt2e(model, quantizer, args=None, kwargs=None, dynamic_shapes=None):
    from torchao.quantization.pt2e import prepare
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e

    # replace the default implementation of _create_obs_or_fq_from_qspec
    prepare._get_obs_or_fq_map = _get_obs_or_fq_map

    if not isinstance(model, GraphModule):
        model = export_model(model, args, kwargs, dynamic_shapes=dynamic_shapes)

    model = prepare_pt2e(model, quantizer)
    # Both device lookups are memoized on the module they are given; the
    # intermediate graph prepare passes in would keep every parameter alive.
    assert_and_get_unique_device.cache_clear()
    _assert_and_get_unique_device.cache_clear()
    return model


def prepare_qat_pt2e(model: GraphModule, quantizer) -> GraphModule:
    """Prepare a graph for QAT with each conv-BN fold simulated.

    Every batch norm must follow a conv2d that feeds nothing else; torchao
    rewrites each pair into the simulated fold: the weight is scaled by the
    BN factor ``s = gamma / sqrt(running_var + eps)`` before its fake-quant,
    the conv output is divided by ``s``, and the batch norm keeps
    normalizing with batch statistics.  A conv without a bias first gets a
    zero buffer as one, so the quantizer annotates the bias the fold will
    produce.

    Args:
        model: Graph exported in training mode.
        quantizer: Annotates the rewritten graph.

    Returns:
        The prepared graph, whose ``train()`` and ``eval()`` do nothing:
        ``set_training`` sets its mode.

    Raises:
        ValueError: A batch norm does not follow a conv2d alone, or its
            momentum or eps is not 0.1 or 1e-5, the values torchao's
            rewrite writes into the graph.
    """
    graph = model.graph
    for bn in list(graph.nodes):
        if bn.target != torch.ops.aten.batch_norm.default:
            continue
        conv = bn.args[0]
        if conv.target != torch.ops.aten.conv2d.default or len(conv.users) > 1:
            raise ValueError(f"{bn} does not follow a conv2d alone")
        momentum, eps = bn.args[6], bn.args[7]
        if (momentum, eps) != (0.1, 1e-5):
            raise ValueError(
                f"{bn} has momentum {momentum} and eps {eps}; the QAT fold "
                "supports only 0.1 and 1e-5"
            )
        if get_arg_value(conv, 2, "bias") is not None:
            continue
        weight = conv.args[1]
        value = fetch_attr(model, weight.target)
        with graph.inserting_before(conv):
            bias = create_getattr_from_value(
                model,
                graph,
                weight.target + "_bias",
                value.new_zeros(value.shape[0]),
            )
        conv.args = conv.args[:2] + (bias,) + conv.args[3:]
    model.recompile()

    # torchao traces its patterns on random inputs; the caller's random
    # stream stays as it was.
    with torch.random.fork_rng():
        _fuse_conv_bn_qat(model)
    # The rewrite builds each conv's zero bias in the dtype it was traced in.
    for conv in graph.nodes:
        if conv.target != torch.ops.aten.conv2d.default:
            continue
        bias = get_arg_value(conv, 2, "bias")
        if bias is None or bias.target != torch.ops.aten.zeros_like.default:
            continue
        bias.update_kwarg("dtype", None)
    model = prepare_pt2e(model, quantizer)

    def _eval(self, mode: bool = True):
        return self

    model.eval = types.MethodType(_eval, model)
    model.train = types.MethodType(_eval, model)
    return model


def set_batch_norm_training(model: GraphModule, training: bool) -> None:
    """Set whether ``model``'s batch norms normalize with batch statistics.

    ``train()`` and ``eval()`` do not reach the ops of an exported graph, so
    this sets each batch norm's ``training`` argument.  Out of training a
    batch norm normalizes with its running statistics and leaves them
    unchanged.

    Args:
        model: Exported graph, rewritten in place.
        training: Whether the batch norms use batch statistics.
    """
    for node in model.graph.nodes:
        if node.target == torch.ops.aten.batch_norm.default:
            node.update_arg(5, training)
    model.recompile()


def set_training(model: GraphModule, training: bool) -> None:
    """Switch ``model``'s dropouts and batch norms between training and eval.

    Out of training a dropout passes its input through.  Attention's dropout
    probability is fixed when the graph is exported; it is kept in the
    node's meta and restored for training.  Batch norms switch as
    ``set_batch_norm_training`` switches them.

    Args:
        model: Exported graph, rewritten in place.
        training: Whether the ops behave as in training.
    """
    for node in model.graph.nodes:
        if node.target in _DROPOUTS:
            node.update_arg(2, training)
        elif node.target == _SDPA and len(node.args) > 4:
            dropout_p = node.meta.setdefault("dropout_p", node.args[4])
            node.update_arg(4, dropout_p if training else 0.0)
    set_batch_norm_training(model, training)


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
        param = fetch_attr(model, input_node.target)
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

    # A bias, and a cast with no scale, get no dequantize node.
    if isinstance(
        activation_post_process,
        (_DerivedObserverOrFakeQuantize, DirectCastFakeQuantize),
    ):
        return

    for user_node in orig_fq_users:
        if is_gemm_op(user_node):
            user_node.meta["dtype"] = output_dtype

            # Insert dequantize node before the node that appear the earlist in
            # the graph
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
    _SDPA: torch.ops.quantized_ops.sdpa_mx.default,
}

# An attention operand's scale kwarg, by its position among the operands.
_ATTENTION_SCALES = ("query_scale", "key_scale", "value_scale")


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
        # The stream takes the observed maximum plus one percentage point,
        # rounded to the nearest whole percent: headroom for the blocks
        # above the mean.
        observed = activation_post_process.max_outlier_pct
        stream_pct = round((observed + 0.01) * 100) / 100
        # ``CSR_STREAM_PCT`` declares the stream at that rate instead (never
        # below what was observed).
        if os.environ.get("CSR_STREAM_PCT"):
            stream_pct = max(observed, float(os.environ["CSR_STREAM_PCT"]))
        logger.info(
            f"{node.target}: {observed:.4%} outliers observed, stream "
            f"declared at {stream_pct:.4%}"
        )

    dequant_code, quant_code = None, None
    if activation_post_process.is_codebook_quantization:
        table = activation_post_process.qmap
        # A codebook is already the levels, one row per attention head; a
        # lookup table holds them once per bfloat16 bit pattern.
        per_head = table.dim() > 1
        values = table if per_head else torch.unique(table)

        activation_post_process.dtype = (
            f"int{activation_post_process.index_bits}"
        )
        # A lookup table answers with the index directly, since it is
        # indexed by the value itself.  A codebook stays the levels, and
        # the midpoints it is compared against travel beside it as the
        # quantize's ``output_code``, which is what turns a value into an
        # index there.
        activation_post_process.qmap = (
            values
            if per_head
            else torch.searchsorted(values, table).to(torch.int64)
        )
        # Entries without a dtype of their own are stored in the model's, and
        # a graph with no parameters keeps the table's.  The midpoints are
        # taken from the levels as fitted.
        entries = values
        parameter = next(iter(model.parameters()), None)
        if activation_post_process.code_dtype is None and parameter is not None:
            entries = values.to(parameter.dtype)

        with graph.inserting_before(node):
            dequant_code = create_getattr_from_value(
                model, graph, "code", entries
            )
            if input_node.op != "get_attr":
                midpoints = (values[..., :-1] + values[..., 1:]) / 2
                quant_code = create_getattr_from_value(
                    model, graph, "code", midpoints
                )

        if activation_post_process.code_dtype is not None:
            dequant_code.meta["dtype"] = activation_post_process.code_dtype

    get_attr_node = scale_qmap = None
    if input_node.op == "get_attr":
        # quantize model parameter and remove the fq module
        param = fetch_attr(model, input_node.target)

        scale = torch.ops.quantized_ops.calculate_mx_qparam(
            param.data,
            activation_post_process.ch_axis,
            activation_post_process.block_size,
            activation_post_process.quant_max,
            activation_post_process.power_2_scale,
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
                activation_post_process.power_2_scale,
                scale_qmap,
                quant_code,
            ]

            if activation_post_process.outlier_threshold is not None:
                target = torch.ops.quantized_ops.quantize_mx_outlier.default
                args.extend(
                    [
                        float(activation_post_process.outlier_threshold),
                        stream_pct,
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
            if activation_post_process.power_2_scale
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
            quantize_mx_node.meta["outlier_rate"] = observed
        else:
            scale_node, quantized_node = output_nodes
            dtype_tuple = (scale_dtype, activation_post_process.dtype)

        quantize_mx_node.meta["dtype"] = dtype_tuple

    quantized_node.meta["dtype"] = activation_post_process.dtype

    if activation_post_process.power_2_scale:
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
        operand = quantized_node

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
            operand = user
            user = next(iter(user.users))

        kwargs = OrderedDict(user.kwargs)
        kwargs.setdefault("block_size", activation_post_process.block_size)
        if user.target in (_SDPA, MX_OP_MAPPING[_SDPA]):
            # The query, key and value each carry their own scale; the
            # query and the probabilities share a codebook, the key and the
            # value the other.  The probabilities are quantized inside the
            # attention kernel with the query's parameters.
            position = user.args.index(operand)
            kwargs.setdefault(_ATTENTION_SCALES[position], kwarg2)
            if position == 0:
                kwargs.setdefault("input_code", kwarg1)
                kwargs.setdefault("probs_qmap", get_attr_node)
                kwargs.setdefault(
                    "probs_quant_max", activation_post_process.quant_max
                )
                kwargs.setdefault("probs_scale_qmap", scale_qmap)
                kwargs.setdefault("probs_code", quant_code)
                kwargs.setdefault(
                    "force_scale_power_of_two",
                    activation_post_process.power_2_scale,
                )
            else:
                kwargs.setdefault("weight_code", kwarg1)
        elif input_node.op == "get_attr" or id(quantized_node) == id(
            user.args[1]
        ):
            kwargs.setdefault("weight_code", kwarg1)
            kwargs.setdefault("weight_scale", kwarg2)
        else:
            kwargs.setdefault("input_code", kwarg1)
            kwargs.setdefault("input_scale", kwarg2)

        order = [
            "input_scale",
            "weight_scale",
            "query_scale",
            "key_scale",
            "value_scale",
            "block_size",
            "input_code",
            "weight_code",
            "probs_qmap",
            "probs_quant_max",
            "probs_scale_qmap",
            "probs_code",
            "force_scale_power_of_two",
            "A_data",
            "A_indices",
            "A_indptr",
        ]
        kwargs = OrderedDict(
            [(key, kwargs[key]) for key in order if key in kwargs]
            + [(key, val) for key, val in kwargs.items() if key not in order]
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
            ], (
                "Only GEMM is supported for outlier suppresion, got "
                f"{user.target}"
            )

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
                mx_op_node.meta["outlier_rate"] = observed
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


def _is_written_buffer(model: GraphModule, node: Node) -> bool:
    """Whether the buffer ``get_attr`` ``node`` names is written in the
    graph -- the destination of an in-place op (a KV cache and its fold).
    Such a buffer changes at run time, so it is quantized dynamically, in
    the graph, not baked once from the observer's statistics."""
    written = (
        torch.ops.aten.index_copy_.default,
        torch.ops.aten.copy_.default,
    )
    for n in model.graph.nodes:
        if n.op != "call_function" or n.target not in written:
            continue
        for operand in n.all_input_nodes:
            if operand.op == "get_attr" and operand.target == node.target:
                return True
    return False


def _replace_observer_with_groupwise_affine_q_dq_node_decomposed(
    model: torch.fx.GraphModule, node: Node, modules: Dict[str, torch.nn.Module]
):
    graph = model.graph
    assert modules is not None
    assert isinstance(node.target, str)
    activation_post_process = modules[node.target]
    assert_and_get_unique_device(activation_post_process)

    if isinstance(activation_post_process.ch_axis, int):
        activation_post_process.ch_axis = (activation_post_process.ch_axis,)

    input_node = node.args[0]

    if input_node.op == "get_attr" and not _is_written_buffer(
        model, input_node
    ):
        param = fetch_attr(model, input_node.target)
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
        # An activation's blocks change every step, so the qparams are
        # computed in the graph and come out of the quantize beside the value.
        with graph.inserting_before(node):
            qmap_node = create_getattr_from_value(
                model, graph, "qmap", activation_post_process.qmap
            )
            scale_qmap_node = None
            if activation_post_process.scale_qmap is not None:
                scale_qmap_node = create_getattr_from_value(
                    model, graph, "qmap", activation_post_process.scale_qmap
                )
            quantize_node = graph.call_function(
                torch.ops.quantized_ops.quantize_affine.default,
                (
                    input_node,
                    qmap_node,
                    activation_post_process.ch_axis,
                    activation_post_process.block_size,
                    float(activation_post_process.quant_min),
                    float(activation_post_process.quant_max),
                    scale_qmap_node,
                ),
            )
            scale_node, zero_point_node, quantized_node = [
                graph.call_function(operator.getitem, (quantize_node, i))
                for i in range(3)
            ]
        quantize_node.meta["dtype"] = (
            activation_post_process.scale_dtype,
            activation_post_process.scale_dtype,
            activation_post_process.dtype,
        )

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
        if scale_node.op != "get_attr":
            continue
        if torch.any(model.get_buffer(scale_node.target) != 1):
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
def sink_obs_or_fq(model: GraphModule) -> GraphModule:
    """Move each observer / fake-quant on a parameter down to its first use.

    An observer inserted directly on a ``get_attr`` runs at the top of the
    graph, keeping the dequantized parameter live from there until it is read.
    Re-inserting it before its earliest user shortens that live range.

    Args:
        model: Prepared graph module, rewritten in place.

    Returns:
        The same module, recompiled.
    """
    graph = model.graph

    def is_obs_or_fq(node):
        return (
            node.op == "call_module"
            and "activation_post_process" in node.target
        )

    for node in reversed(graph.nodes):
        if not is_obs_or_fq(node):
            continue

        input_node = node.args[0]

        # Handle double quantization case where input is also an obs or fq
        if is_obs_or_fq(input_node):
            input_node = input_node.args[0]

        if input_node.op != "get_attr":
            continue

        order = {n: i for i, n in enumerate(graph.nodes)}
        users = list(node.users.keys())
        first_user = min(users, key=lambda n: order[n])

        with graph.inserting_before(first_user):
            new_fq = graph.node_copy(node, lambda x: x)

        logger.info(f"Replacing {node} with {new_fq} before {first_user}")

        node.replace_all_uses_with(new_fq)
        graph.erase_node(node)

    graph.lint()
    model.recompile()
    return model


def _is_mutating(node: Node) -> bool:
    """Whether ``node`` writes to one of its operands."""
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target._schema.is_mutable
    )


def _finish_freezing(model: GraphModule) -> None:
    """Mark ``model`` frozen, fix its fake-quants' enable flags, and drop
    what the freezing left unused."""
    for module in model.modules():
        if isinstance(module, _FreezableFlags):
            module.freeze_flags()
    model.meta["frozen_weights"] = True
    model.graph.lint()
    model.recompile()
    model.delete_all_unused_submodules()


@torch.no_grad()
def freeze_weights(*models: GraphModule) -> None:
    """Bake each weight's fake-quantization into the weight.

    Every fake-quant reading a parameter runs once and is removed; its users
    read the fake-quantized value instead.  A buffer, such as a KV cache,
    changes between calls and is left alone.  The graphs may share parameters
    -- graphs exported from one model do -- and a shared one is quantized
    once for all of them: when every fake-quant reading it gives the same
    value and nothing else reads it, the value is written into the
    parameter itself, so every holder of the tensor (the source model, the
    other graphs) sees it too; otherwise each fake-quant's value gets a
    buffer of its own and the other readers keep the raw weight.  The
    graphs compute the same values without re-quantizing their weights on
    every forward.  Call it once every weight scale is final, i.e. after
    calibration.  A frozen graph is for evaluation only: its remaining
    fake-quants keep the enable flags they have now, and ``convert_pt2e``
    rejects it, since quantizing the weights needs the raw ones.

    Args:
        *models: Prepared graph modules, rewritten in place.
    """
    # Each tensor's fake-quants and other readers across the graphs, keyed
    # by its memory: graphs exported from one model hold the same tensors.
    readers = {}
    for model in models:
        modules = dict(model.named_modules())
        params = dict(model.named_parameters(remove_duplicate=False))
        for node in model.graph.nodes:
            if node.op != "get_attr" or node.target not in params:
                continue
            param = params[node.target]
            key = (param.data_ptr(), tuple(param.shape))
            _, fake_quants, others = readers.setdefault(key, (param, [], []))
            for user in node.users:
                module = _get_module(user, modules)
                if isinstance(module, FakeQuantizeBase):
                    fake_quants.append((model, user, module))
                else:
                    others.append(user)

    for param, fake_quants, others in readers.values():
        if not fake_quants:
            continue
        values = [module(param) for _, _, module in fake_quants]
        if not others and all(torch.equal(values[0], v) for v in values):
            param.copy_(values[0])
            for _, node, _ in fake_quants:
                node.replace_all_uses_with(node.args[0])
                node.graph.erase_node(node)
            continue
        for (model, node, _), value in zip(fake_quants, values):
            with model.graph.inserting_before(node):
                frozen = create_getattr_from_value(
                    model, model.graph, node.args[0].target + "_frozen", value
                )
            node.replace_all_uses_with(frozen)
            model.graph.erase_node(node)

    for model in models:
        _finish_freezing(model)


@torch.no_grad()
def freeze_cache_reads(
    model: GraphModule, caches: List[str]
) -> Callable[[], None]:
    """Precompute what ``model`` computes from caches its caller writes.

    A cache here is a buffer that changes only between calls, and seldom:
    the main half of a split KV cache (``split_kv_cache``) takes a chunk
    only when the residual fills.  The graph's own writes to ``caches`` are
    removed, so the caller must make them.  Everything the graph then
    computes from the caches and from buffers it never writes -- their
    fake-quants included -- is the same on every call until the caches
    change: it moves into a module of its own, and the graph reads the
    results from buffers.  Freeze the weights first, so their fake-quants
    are not mistaken for such reads.

    Args:
        model: Prepared graph module, rewritten in place.
        caches: Names of the buffers the caller writes.

    Returns:
        A function recomputing the results from the caches' current
        contents; call it after writing them.

    Raises:
        ValueError: The graph reads the value of one of its cache writes.
    """
    graph = model.graph
    caches = set(caches)
    for node in list(graph.nodes):
        if (
            _is_mutating(node)
            and isinstance(node.args[0], Node)
            and node.args[0].op == "get_attr"
            and node.args[0].target in caches
        ):
            if node.users:
                raise ValueError(f"{node} writes a cache and is read")
            graph.erase_node(node)
    written = {
        node.args[0].target
        for node in graph.nodes
        if _is_mutating(node)
        and isinstance(node.args[0], Node)
        and node.args[0].op == "get_attr"
    }
    modules = dict(model.named_modules())

    # Nodes computed only from buffers the graph never writes, mapped to
    # whether they read a cache.
    static = {}
    for node in graph.nodes:
        inputs = node.all_input_nodes
        if node.op == "get_attr":
            if node.target not in written:
                static[node] = node.target in caches
        elif all(n in static for n in inputs) and (
            (node.op == "call_function" and not _is_mutating(node))
            or isinstance(_get_module(node, modules), FakeQuantizeBase)
        ):
            static[node] = any(static[n] for n in inputs)

    results = [
        node
        for node, reads_cache in static.items()
        if reads_cache
        and node.op != "get_attr"
        and any(user not in static for user in node.users)
    ]
    needed = set()
    pending = list(results)
    while pending:
        node = pending.pop()
        if node not in needed:
            needed.add(node)
            pending.extend(node.all_input_nodes)
    compute = Graph()
    copies = {}
    for node in graph.nodes:
        if node in needed:
            copies[node] = compute.node_copy(node, lambda n: copies[n])
    compute.output(tuple(copies[n] for n in results))
    computed = GraphModule(model, compute)

    buffers = []
    for node, value in zip(results, computed()):
        with graph.inserting_before(node):
            frozen = create_getattr_from_value(
                model, graph, node.name + "_frozen", value
            )
        buffers.append(fetch_attr(model, frozen.target))
        node.replace_all_uses_with(
            frozen, delete_user_cb=lambda user: user not in static
        )
    graph.eliminate_dead_code(
        is_impure_node=lambda n: n.op in {"placeholder", "output"}
        or _is_mutating(n)
    )
    _finish_freezing(model)

    @torch.no_grad()
    def refresh():
        for buffer, value in zip(buffers, computed()):
            buffer.copy_(value)

    return refresh


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

        if (
            not isinstance(other_fq, MXFakeQuantize)
            or other_fq.outlier_threshold is None
        ):
            continue

        assert (
            not isinstance(input_fq, MXFakeQuantize)
            or input_fq.outlier_threshold is None
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


@torch.no_grad()
def fold_conv_bn_qat(model: GraphModule) -> None:
    """Fold each batch norm of a QAT graph into the conv before it.

    ``prepare_qat_pt2e`` leaves every conv-BN pair as the simulated fold,
    ``conv(x, fq(W * s), fq(zeros)) / s + b`` into the batch norm, with
    ``s = gamma / sqrt(running_var + eps)``.  With the running statistics
    taken as final this is ``conv(x, fq(W'), fq(b'))``, where
    ``W' = W * s`` and ``b' = (b - running_mean) * s + beta``: the fold
    writes ``W'`` and ``b'`` into the conv's parameters, points the
    fake-quants at them and removes the rest of the chain.  ``W'`` is
    computed by the graph's own nodes, so the weight fake-quant sees the
    values it was trained on.  Any other batch norm is left alone.

    Args:
        model: Graph prepared by ``prepare_qat_pt2e``, rewritten in place.
    """
    graph = model.graph
    modules = dict(model.named_modules())

    def fake_quant_input(node):
        """The node ``node`` fake-quantizes, or ``node`` itself."""
        if isinstance(_get_module(node, modules), FakeQuantizeBase):
            return node.args[0]
        return node

    def value(node):
        """``node``'s value, computed from the parameters it reads."""
        if node.op == "get_attr":
            return fetch_attr(model, node.target)
        args = map_arg(node.args, value)
        return node.target(*args, **map_arg(node.kwargs, value))

    for bn in list(graph.nodes):
        if bn.target != torch.ops.aten.batch_norm.default:
            continue
        add_bias = bn.args[0]
        if (
            add_bias.target != torch.ops.aten.add.Tensor
            or add_bias.args[0].target != torch.ops.aten.div.Tensor
        ):
            continue
        divide = add_bias.args[0]
        conv = fake_quant_input(divide.args[0])
        if conv.target != torch.ops.aten.conv2d.default:
            continue
        scaled_weight = fake_quant_input(conv.args[1])
        zeros = fake_quant_input(conv.args[2])
        weight, bias = scaled_weight.args[0], zeros.args[0]

        scale = value(divide.args[1]).flatten()
        mean, beta = value(bn.args[3]), value(bn.args[2])
        value(weight).copy_(value(scaled_weight))
        bias_value = value(bias)
        bias_value.copy_((bias_value - mean) * scale + beta)

        scaled_weight.replace_all_uses_with(weight)
        zeros.replace_all_uses_with(bias)
        # Any fake-quant on the conv output now quantizes the folded output.
        bn.replace_all_uses_with(divide.args[0])
        graph.erase_node(bn)

    graph.eliminate_dead_code()
    model.recompile()


def convert_pt2e(
    model: GraphModule,
    output_dtype: str = None,
    eliminate_no_effect: bool = True,
):
    if model.meta.get("frozen_weights"):
        raise ValueError(
            "convert_pt2e needs the raw weights that freeze_weights "
            "replaced; convert a graph that was not frozen"
        )

    fold_conv_bn_qat(model)

    modules = dict(model.named_modules(remove_duplicate=False))

    swap_matmul_inputs(model)

    for node in list(model.graph.nodes):
        if node.op != "call_module":
            continue
        mod = _get_module(node, modules)
        assert mod is not None
        if not isinstance(mod, FakeQuantizeBase):
            continue
        if (
            isinstance(mod, RandomHadamardTransform)
            and mod.rht_axis is not None
        ):
            raise NotImplementedError(
                "The random Hadamard transform has no lowering yet"
            )
        if isinstance(mod, MXFakeQuantize):
            _replace_observer_with_quantize_mx_node_decomposed(
                model, node, modules
            )
        elif isinstance(mod, GroupWiseAffineFakeQuantize):
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
    # A KV-cache fold writes a buffer in place and yields nothing; it is the
    # one side effect kept.
    model.graph.eliminate_dead_code(
        is_impure_node=lambda n: n.op in {"placeholder", "output"}
        or (
            n.target is torch.ops.aten.index_copy_.default
            and n.args[0].op == "get_attr"
        )
    )
    model.recompile()
    model.delete_all_unused_submodules()

    return model
