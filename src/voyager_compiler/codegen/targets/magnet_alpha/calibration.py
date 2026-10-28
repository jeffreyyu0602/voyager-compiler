# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""What Alpha's observers see: the epilogue's output, not the accumulator.

A matrix call does not store its accumulator. It drains through the PPU, which
normalizes into a signed Q8.8 intermediate and truncates from there, so the
value that reaches the next layer's observer has already been through a
16-bit fixed-point domain. Calibrating on the float accumulator measures a
range the datapath never holds, and every scale derived from it is a scale for
a value the machine does not produce.

This inserts one reference at each PE boundary, before calibration runs, so
the observer downstream of it sees what the endpoint will actually store. The
reference is removed again before conversion; the record's epilogue is still
``ppu_fixed_relu``.

**One branch is not instrumentation.** When a boundary has transparent views
*and* an activation, the activation is rebuilt against the reference and the
original erased -- so after unwrapping, ``conv -> views -> relu`` has become
``conv -> relu -> views`` and stays that way. That is the fork's shape and it
is sound (a ReLU is elementwise and the views move no bytes), but it is a
permanent canonicalization riding in a pass named for a temporary one, and it
is called out here because the rest of this module really is removable.

After that first pass, the same references are configured from finalized
observer qparams and replayed over the same calibration samples. That second
pass is PTQ, not QAT: it reproduces the PE and NMP integer datapaths so the
observers see the values the endpoint actually stores. Gradient-only state,
learned binary points, and training refresh hooks remain outside this module.
"""

import copy
import logging
import math
import operator
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.codegen.targets.magnet_alpha.calibration_epilogues import (
    ALPHA_Q8_8_REFERENCE_FORMAT,
    AlphaCalibrationEpilogue,
    AlphaNMPCalibrationEpilogue,
)
from voyager_compiler.codegen.targets.magnet_alpha.fixed_point import (
    alpha_w8_accumulator_real_bound,
)
from voyager_compiler.codegen.targets.magnet_alpha.hardware import OUTPUT_LANES
from voyager_compiler.quantization.fake_quantize import (
    FusedAmaxObsFakeQuantize,
    _DerivedObserverOrFakeQuantize,
)
from voyager_compiler.quantization.contracts import (
    PreparedDomainInputFormat,
    PreparedDomainRelation,
    PreparedDomainRuleSet,
    PreparedDomainTransferRule,
    PreparedQuantizationContract,
)
from voyager_compiler.shape_prop import fetch_attr

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ALPHA_CALIBRATION_REFERENCE",
    "ALPHA_Q8_8_REFERENCE_FORMAT",
    "AlphaCalibrationEpilogue",
    "AlphaNMPCalibrationEpilogue",
    "cleanup_quantization_preparation",
    "finalize_quantization_calibration",
    "insert_calibration_epilogues",
    "insert_nmp_calibration_epilogues",
    "prepare_quantization",
    "prepare_quantization_recalibration",
    "unwrap_calibration_epilogues",
]


#: Marks a node as an instrument rather than a lowering, so a pass that walks
#: the graph can tell the two apart without importing this module. The domain
#: rules need exactly that: a ReLU after a reference is still a ReLU the array
#: absorbs, and treating it as free-standing would put a region frontier back
#: on the accumulator -- the defect the reference is inserted around.
ALPHA_CALIBRATION_REFERENCE = "magnet_alpha_calibration_reference"

#: What a matrix call the PPU drains can be. The prepare-time graph still has
#: ATen convolutions -- Alpha's own operators do not exist until lowering.
_MATRIX_SOURCES = (torch.ops.aten.conv2d.default,)

#: The activation a matrix call absorbs.
#:
#: Out-of-place only, and narrower than ``regions.ACTIVATION_EPILOGUES`` on
#: purpose: the views branch below *rebuilds* the activation against the
#: reference, and it can only rebuild it as ``relu.default``. Accepting
#: ``relu_`` here would match an in-place activation and silently emit an
#: out-of-place one in its place. The fork matches the same single target,
#: for the same reason.
_ACTIVATION_EPILOGUES = (torch.ops.aten.relu.default,)


def _observer(node: Node, modules) -> bool:
    """Whether ``node`` runs an output observer."""
    from voyager_compiler.quantization.fake_quantize import (
        FusedAmaxObsFakeQuantize,
    )

    if node.op != "call_module" or not isinstance(node.target, str):
        return False
    return isinstance(modules.get(node.target), FusedAmaxObsFakeQuantize)


def _transparent_view(node: Node, source: Node) -> bool:
    """Whether ``node`` reads ``source``'s bytes without moving them.

    A view between a convolution and its observer does not change what the
    epilogue produced, so the reference still belongs on the convolution.
    Anything this cannot prove transparent ends the walk rather than being
    assumed harmless -- a reference placed across a real copy would round a
    value the PPU never saw.
    """
    if node.op != "call_function":
        return False
    operand = get_arg_value(node, 0, "input")
    if operand is not source:
        return False
    # This is deliberately narrower than the generic storage predicate.  PE
    # placement may commute only through a slice that leaves the value alone
    # or removes a leading run of NCHW channels.  A spatial window can be
    # addressable and still be the wrong side of the resident PE output.
    if _slice_kind(node) is not None:
        return True

    # The fork's PE-boundary walk also admits a contiguous reshape.  This is
    # load-bearing for YOLO's first stem: ``conv -> reshape -> relu`` must be
    # canonicalized to ``conv -> relu -> reshape`` so the PPU can absorb the
    # activation.  Keep the proof local and fail closed when shape propagation
    # did not establish the two contiguous tensor views.
    if node.target is not torch.ops.aten.reshape.default:
        return False
    source_value = getattr(source, "value", None)
    if source_value is None:
        source_value = source.meta.get("val")
    result_value = getattr(node, "value", None)
    if result_value is None:
        result_value = node.meta.get("val")
    return (
        isinstance(source_value, torch.Tensor)
        and isinstance(result_value, torch.Tensor)
        and source_value.numel() == result_value.numel()
        and source_value.dtype == result_value.dtype
        and source_value.is_contiguous()
        and result_value.is_contiguous()
    )


def _prepared_pe_boundary(conv: Node, modules):
    """One matrix boundary's views, optional ReLU and output observer.

    ``None`` when the boundary is not explicit -- no observer, a fan-out, or
    a view this cannot prove transparent. Declining is the honest outcome:
    the boundary keeps calibrating on its accumulator, which is what happens
    today everywhere.
    """
    current = conv
    views: List[Node] = []
    visited = set()
    while current not in visited and len(current.users) == 1:
        visited.add(current)
        user = next(iter(current.users))
        if _observer(user, modules):
            return tuple(views), None, user
        if (
            user.op == "call_function"
            and user.target in _ACTIVATION_EPILOGUES
        ):
            observers = tuple(
                candidate
                for candidate in user.users
                if _observer(candidate, modules)
            )
            if len(observers) == 1:
                return tuple(views), user, observers[0]
            return None
        if not _transparent_view(user, current):
            return None
        views.append(user)
        current = user
    return None


def _unique_module_name(model: GraphModule, stem: str) -> str:
    name, index = stem, 0
    while hasattr(model, name):
        index += 1
        name = f"{stem}_{index}"
    return name


def _insert_reference(
    model: GraphModule,
    conv: Node,
    views,
    relu: Optional[Node],
    observer: Node,
) -> None:
    """Put one reference between a matrix call and what observes it.

    The ReLU commutes with a transparent view -- it is elementwise and the
    view moves no bytes -- so when both are present the ReLU is *relocated*
    to sit against the reference and the original erased. That is the fork's
    shape and it is not cosmetic: it is what keeps the epilogue, its
    activation and the value the observer sees adjacent, which is the whole
    property the boundary matcher later depends on.

    That relocation **survives** ``unwrap_calibration_epilogues``, unlike
    everything else this pass does. Only the reference module is temporary.
    """
    graph = model.graph
    name = _unique_module_name(model, f"{conv.name}_ppu_reference")
    reference = AlphaCalibrationEpilogue()
    reference.train(model.training)
    model.add_module(name, reference)

    if relu is None and not views:
        with graph.inserting_before(observer):
            node = graph.call_module(name, (conv,))
        observer.replace_input_with(conv, node)
    elif relu is None:
        with graph.inserting_after(conv):
            node = graph.call_module(name, (conv,))
        views[0].replace_input_with(conv, node)
    elif not views:
        with graph.inserting_before(relu):
            node = graph.call_module(name, (conv,))
        relu.replace_input_with(conv, node)
    else:
        with graph.inserting_after(conv):
            node = graph.call_module(name, (conv,))
        with graph.inserting_after(node):
            relocated = graph.call_function(
                torch.ops.aten.relu.default, (node,)
            )
        relocated.meta = dict(conv.meta)
        views[0].replace_input_with(conv, relocated)
        relu.replace_all_uses_with(views[-1])
        graph.erase_node(relu)

    node.meta = dict(conv.meta)
    node.meta[ALPHA_CALIBRATION_REFERENCE] = True


def insert_calibration_epilogues(model: GraphModule) -> int:
    """Put one Q8.8 reference at every explicit matrix boundary.

    Returns the number inserted. Idempotent by construction: a boundary whose
    walk already reaches a reference stops at the reference's own
    ``call_module`` rather than at an observer, so a second run declines it.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    inserted = 0
    declined = 0
    for conv in list(model.graph.nodes):
        if conv.op != "call_function" or conv.target not in _MATRIX_SOURCES:
            continue
        if get_arg_value(conv, 6, "groups", 1) != 1:
            # Depthwise drains through the near-memory unit, whose reference
            # is an identity while calibrating. Nothing to insert.
            continue
        boundary = _prepared_pe_boundary(conv, modules)
        if boundary is None:
            declined += 1
            continue
        views, relu, observer = boundary
        _insert_reference(model, conv, views, relu, observer)
        inserted += 1

    if inserted:
        model.graph.lint()
        model.recompile()
    LOGGER.info(
        "inserted %d Alpha Q8.8 calibration references; %d matrix boundaries "
        "declined and keep calibrating on their accumulator",
        inserted,
        declined,
    )
    return inserted


def _prepared_w8_observer(value: object, modules):
    """Return a prepared scalar/per-channel W8 observer, if present."""
    from voyager_compiler.quantization.fake_quantize import (
        FusedAmaxObsFakeQuantize,
    )

    if not isinstance(value, Node) or value.op != "call_module":
        return None
    observer = modules.get(str(value.target))
    if not isinstance(observer, FusedAmaxObsFakeQuantize):
        return None
    return observer if observer.dtype == "int8" else None


def _node_shape(node: Node) -> Tuple[int, ...]:
    """Read a prepared node's tensor shape without executing the graph."""
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    shape = getattr(node, "shape", None)
    return () if shape is None else tuple(shape)


def _insert_nmp_reference(
    model: GraphModule,
    conv: Node,
    views,
    relu: Optional[Node],
    observer: Node,
) -> None:
    """Attach one exact-capable reference to a depthwise boundary."""
    input_value = get_arg_value(conv, 0, "input")
    weight_value = get_arg_value(conv, 1, "weight")
    if not isinstance(input_value, Node) or not isinstance(weight_value, Node):
        raise ValueError(f"{conv.name}: depthwise operands are missing")
    input_source = input_value.args[0] if input_value.args else None
    weight_source = weight_value.args[0] if weight_value.args else None
    if not isinstance(input_source, Node) or not isinstance(weight_source, Node):
        raise ValueError(
            f"{conv.name}: depthwise W8 observers have no tensor sources"
        )

    name = _unique_module_name(model, f"{conv.name}_nmp_reference")
    reference = AlphaNMPCalibrationEpilogue(
        lane_width=OUTPUT_LANES,
        stride=get_arg_value(conv, 3, "stride", (1, 1)),
        padding=get_arg_value(conv, 4, "padding", (0, 0)),
        dilation=get_arg_value(conv, 5, "dilation", (1, 1)),
        groups=int(get_arg_value(conv, 6, "groups", 1)),
    )
    reference.train(model.training)
    model.add_module(name, reference)
    with model.graph.inserting_after(conv):
        fixed_node = model.graph.call_module(
            name,
            (conv, input_source, weight_source),
        )
    first_user = views[0] if views else relu
    if first_user is None:
        first_user = observer
    first_user.replace_input_with(conv, fixed_node)
    fixed_node.meta = dict(conv.meta)
    fixed_node.meta[ALPHA_CALIBRATION_REFERENCE] = True


def insert_nmp_calibration_epilogues(model: GraphModule) -> int:
    """Put one exact-capable reference at every prepared depthwise boundary."""
    modules = dict(model.named_modules(remove_duplicate=False))
    inserted = 0
    declined = 0
    for conv in list(model.graph.nodes):
        if conv.op != "call_function" or conv.target not in _MATRIX_SOURCES:
            continue
        groups = int(get_arg_value(conv, 6, "groups", 1))
        if groups <= 1:
            continue
        input_value = get_arg_value(conv, 0, "input")
        weight_value = get_arg_value(conv, 1, "weight")
        input_observer = _prepared_w8_observer(input_value, modules)
        weight_observer = _prepared_w8_observer(weight_value, modules)
        weight_source = (
            weight_value.args[0]
            if isinstance(weight_value, Node) and weight_value.args
            else None
        )
        weight_shape = (
            _node_shape(weight_source)
            if isinstance(weight_source, Node)
            else ()
        )
        if (
            input_observer is None
            or weight_observer is None
            or len(weight_shape) != 4
            or weight_shape[0] != groups
            or weight_shape[1] != 1
        ):
            declined += 1
            continue
        boundary = _prepared_pe_boundary(conv, modules)
        if boundary is None:
            declined += 1
            continue
        views, relu, observer = boundary
        _insert_nmp_reference(model, conv, views, relu, observer)
        inserted += 1

    if inserted:
        model.graph.lint()
        model.recompile()
    LOGGER.info(
        "inserted %d Alpha NMP calibration references; %d depthwise "
        "boundaries declined",
        inserted,
        declined,
    )
    return inserted


def _is_calibration_side_user(current: Node, user: Node, modules) -> bool:
    """Whether ``user`` reads ``current`` only as an exact operand."""
    if user.op != "call_module":
        return False
    module = modules.get(str(user.target))
    if not isinstance(
        module, (AlphaCalibrationEpilogue, AlphaNMPCalibrationEpilogue)
    ):
        return False
    return any(argument is current for argument in user.args[1:])


def _prepared_exact_output(fixed_node: Node, modules):
    """Find one reference's output observer and absorbed activation."""
    current = fixed_node
    relu = False
    visited = set()
    while current not in visited:
        visited.add(current)
        users = [
            user
            for user in current.users
            if not _is_calibration_side_user(current, user, modules)
        ]
        if len(users) != 1:
            return None
        user = users[0]
        if (
            user.op == "call_function"
            and user.target in _ACTIVATION_EPILOGUES
        ):
            if relu:
                return None
            relu = True
            current = user
            continue
        observer = (
            modules.get(str(user.target))
            if user.op == "call_module"
            else None
        )
        if isinstance(observer, FusedAmaxObsFakeQuantize):
            if observer.dtype == "int8" and not observer.is_per_channel:
                return user, observer, relu
            coarse = getattr(observer, "coarse", {})
            if (
                observer.dtype == "int4"
                and observer.block_size == OUTPUT_LANES
                and observer.scale_dtype == "uint8"
                and coarse.get("dtype") == "int8"
                and not observer.coarse_is_per_channel
            ):
                return user, observer, relu
            return None
        if not _transparent_view(user, current):
            return None
        current = user
    return None


def _prepared_vsq_observer(value: object, modules):
    if not isinstance(value, Node) or value.op != "call_module":
        return None
    observer = modules.get(str(value.target))
    if not isinstance(observer, FusedAmaxObsFakeQuantize):
        return None
    coarse = getattr(observer, "coarse", {})
    if (
        observer.dtype != "int4"
        or observer.block_size != OUTPUT_LANES
        or observer.scale_dtype != "uint8"
        or coarse.get("dtype") != "int8"
    ):
        return None
    return observer


@dataclass(frozen=True)
class _PreparedExactPEOperands:
    mode: str
    real_bias: torch.Tensor
    output_scale: torch.Tensor
    output_precision: int
    uses_pvs: bool
    relu: bool
    output_observer_node: Node
    output_observer: FusedAmaxObsFakeQuantize
    accumulator_scale: Optional[torch.Tensor] = None
    accumulator_real_amax: Optional[torch.Tensor] = None
    input_observer: Optional[FusedAmaxObsFakeQuantize] = None
    weight_observer: Optional[FusedAmaxObsFakeQuantize] = None
    input_value: Optional[Node] = None
    weight_value: Optional[Node] = None
    stride: object = (1, 1)
    padding: object = (0, 0)
    dilation: object = (1, 1)
    groups: int = 1


@dataclass(frozen=True)
class _PreparedExactNMPOperands:
    input_observer: FusedAmaxObsFakeQuantize
    weight_observer: FusedAmaxObsFakeQuantize
    real_bias: torch.Tensor
    output_scale: torch.Tensor
    output_precision: int
    uses_pvs: bool
    relu: bool
    output_observer_node: Node
    output_observer: FusedAmaxObsFakeQuantize


def _prepared_real_bias(
    model: GraphModule,
    conv: Node,
    modules,
    channel_scale: torch.Tensor,
):
    """Resolve a prepared convolution bias in semantic real units."""
    bias_node = get_arg_value(conv, 2, "bias")
    if bias_node is None:
        return torch.zeros_like(channel_scale)
    if not isinstance(bias_node, Node):
        return None
    if bias_node.op == "get_attr":
        bias = fetch_attr(model, bias_node.target)
        return bias.detach() if isinstance(bias, torch.Tensor) else None
    if bias_node.op != "call_module":
        return None
    bias_observer = modules.get(str(bias_node.target))
    source = bias_node.args[0] if bias_node.args else None
    if (
        not isinstance(bias_observer, _DerivedObserverOrFakeQuantize)
        or not isinstance(source, Node)
        or source.op != "get_attr"
    ):
        return None
    bias = fetch_attr(model, source.target)
    if not isinstance(bias, torch.Tensor):
        return None
    derived_scale = bias_observer.calculate_qparams().detach().reshape(-1)
    expected_scale = channel_scale.detach().reshape(-1).to(
        device=derived_scale.device,
        dtype=derived_scale.dtype,
    )
    if derived_scale.shape != expected_scale.shape or not torch.allclose(
        derived_scale, expected_scale, rtol=1e-6, atol=0.0
    ):
        raise ValueError(
            f"{conv.name}: derived bias qparams do not match the active "
            "input and weight observers"
        )
    with torch.no_grad():
        return bias_observer(bias.detach()).detach()


def _prepared_w8_weight_payload(
    model: GraphModule,
    weight_node: Node,
    observer: FusedAmaxObsFakeQuantize,
):
    source = weight_node.args[0] if weight_node.args else None
    if not isinstance(source, Node) or source.op != "get_attr":
        return None
    weight = fetch_attr(model, source.target)
    if not isinstance(weight, torch.Tensor):
        return None
    return torch.ops.quantized_ops.quantize.default(
        weight.detach(),
        observer.calculate_qparams().detach().to(
            device=weight.device, dtype=weight.dtype
        ),
        None,
        None,
        None,
        observer.qmap.to(weight.device),
    )


def _prepared_exact_pe_operands(
    model: GraphModule, fixed_node: Node, modules
):
    if len(fixed_node.args) not in (1, 3):
        return None
    conv = fixed_node.args[0]
    if (
        not isinstance(conv, Node)
        or conv.target not in _MATRIX_SOURCES
        or int(get_arg_value(conv, 6, "groups", 1)) != 1
    ):
        return None
    output = _prepared_exact_output(fixed_node, modules)
    if output is None:
        return None
    output_node, output_observer, relu = output
    uses_pvs = output_observer.dtype == "int4"
    output_precision = 4 if uses_pvs else 8
    input_value = get_arg_value(conv, 0, "input")
    weight_value = get_arg_value(conv, 1, "weight")

    input_w8 = _prepared_w8_observer(input_value, modules)
    weight_w8 = _prepared_w8_observer(weight_value, modules)
    if input_w8 is not None and weight_w8 is not None:
        input_scale = input_w8.calculate_qparams().detach()
        weight_scale = weight_w8.calculate_qparams().detach()
        accumulator_scale = input_scale * weight_scale.reshape(-1)
        real_bias = _prepared_real_bias(
            model, conv, modules, accumulator_scale
        )
        if real_bias is None or not isinstance(weight_value, Node):
            return None
        weight_payload = _prepared_w8_weight_payload(
            model, weight_value, weight_w8
        )
        if weight_payload is None:
            return None
        return _PreparedExactPEOperands(
            mode="w8_o4" if uses_pvs else "w8_o8",
            accumulator_scale=accumulator_scale,
            accumulator_real_amax=alpha_w8_accumulator_real_bound(
                weight_payload, accumulator_scale
            ),
            real_bias=real_bias,
            output_scale=output_observer.calculate_qparams().detach(),
            output_precision=output_precision,
            uses_pvs=uses_pvs,
            relu=relu,
            output_observer_node=output_node,
            output_observer=output_observer,
        )

    input_vsq = _prepared_vsq_observer(input_value, modules)
    weight_vsq = _prepared_vsq_observer(weight_value, modules)
    if input_vsq is None or weight_vsq is None:
        return None
    input_source = input_value.args[0] if input_value.args else None
    weight_source = weight_value.args[0] if weight_value.args else None
    if not isinstance(input_source, Node) or not isinstance(weight_source, Node):
        return None
    if input_vsq.block_size != weight_vsq.block_size:
        return None
    weight_scale = weight_vsq.calculate_qparams().detach().reshape(-1)
    real_bias = _prepared_real_bias(model, conv, modules, weight_scale)
    if real_bias is None:
        return None
    return _PreparedExactPEOperands(
        mode="vsq_o4" if uses_pvs else "vsq_o8",
        real_bias=real_bias,
        output_scale=output_observer.calculate_qparams().detach(),
        output_precision=output_precision,
        uses_pvs=uses_pvs,
        relu=relu,
        output_observer_node=output_node,
        output_observer=output_observer,
        input_observer=input_vsq,
        weight_observer=weight_vsq,
        input_value=input_source,
        weight_value=weight_source,
        stride=get_arg_value(conv, 3, "stride", (1, 1)),
        padding=get_arg_value(conv, 4, "padding", (0, 0)),
        dilation=get_arg_value(conv, 5, "dilation", (1, 1)),
        groups=int(get_arg_value(conv, 6, "groups", 1)),
    )


def _prepared_exact_nmp_operands(
    model: GraphModule, fixed_node: Node, modules
):
    if len(fixed_node.args) != 3:
        return None
    conv = fixed_node.args[0]
    if not isinstance(conv, Node) or conv.target not in _MATRIX_SOURCES:
        return None
    groups = int(get_arg_value(conv, 6, "groups", 1))
    weight_value = get_arg_value(conv, 1, "weight")
    weight_source = (
        weight_value.args[0]
        if isinstance(weight_value, Node) and weight_value.args
        else None
    )
    weight_shape = (
        _node_shape(weight_source) if isinstance(weight_source, Node) else ()
    )
    if (
        groups <= 1
        or len(weight_shape) != 4
        or weight_shape[0] != groups
        or weight_shape[1] != 1
    ):
        return None
    input_observer = _prepared_w8_observer(
        get_arg_value(conv, 0, "input"), modules
    )
    weight_observer = _prepared_w8_observer(weight_value, modules)
    output = _prepared_exact_output(fixed_node, modules)
    if input_observer is None or weight_observer is None or output is None:
        return None
    output_node, output_observer, relu = output
    input_scale = input_observer.calculate_qparams().detach().reshape(-1)
    weight_scale = weight_observer.calculate_qparams().detach().reshape(-1)
    if input_scale.numel() != 1 or weight_scale.numel() != 1:
        return None
    real_bias = _prepared_real_bias(
        model, conv, modules, input_scale * weight_scale
    )
    if real_bias is None:
        return None
    uses_pvs = output_observer.dtype == "int4"
    return _PreparedExactNMPOperands(
        input_observer=input_observer,
        weight_observer=weight_observer,
        real_bias=real_bias,
        output_scale=output_observer.calculate_qparams().detach(),
        output_precision=4 if uses_pvs else 8,
        uses_pvs=uses_pvs,
        relu=relu,
        output_observer_node=output_node,
        output_observer=output_observer,
    )


_ALPHA_EXACT_OUTPUT_OBSERVER = "magnet_alpha_exact_output_observer"


def _rebind_downstream_derived_bias_observers(
    model: GraphModule,
    original: FusedAmaxObsFakeQuantize,
    isolated: FusedAmaxObsFakeQuantize,
) -> None:
    """Retarget derived bias qparams to an isolated activation observer."""
    modules = dict(model.named_modules(remove_duplicate=False))
    for bias_node in model.graph.nodes:
        if bias_node.op != "call_module":
            continue
        bias_observer = modules.get(str(bias_node.target))
        if not isinstance(bias_observer, _DerivedObserverOrFakeQuantize):
            continue
        consumers = [
            user
            for user in bias_node.users
            if user.target in _MATRIX_SOURCES
            and get_arg_value(user, 2, "bias") is bias_node
        ]
        if not consumers:
            continue
        if len(consumers) != 1:
            raise ValueError(
                f"{bias_node.name}: derived bias observer must feed one Conv2d"
            )
        conv = consumers[0]
        operand_modules = []
        for position, keyword in ((0, "input"), (1, "weight")):
            operand = get_arg_value(conv, position, keyword)
            if isinstance(operand, Node) and operand.op == "call_module":
                operand_modules.append(modules.get(str(operand.target)))
        if isolated not in operand_modules:
            continue
        dependencies = list(bias_observer.obs_or_fqs)
        updated = [
            isolated if dependency is original else dependency
            for dependency in dependencies
        ]
        if any(
            dependency is not replacement
            for dependency, replacement in zip(dependencies, updated)
        ):
            bias_observer.obs_or_fqs = updated


def _isolate_exact_output_observer(
    model: GraphModule,
    observer_node: Node,
    observer: FusedAmaxObsFakeQuantize,
):
    """Give an exact boundary independently resettable output history."""
    if observer_node.meta.get(_ALPHA_EXACT_OUTPUT_OBSERVER):
        isolated = model.get_submodule(str(observer_node.target))
        if not isinstance(isolated, FusedAmaxObsFakeQuantize):
            raise TypeError(
                f"{observer_node.name}: exact output is not an Alpha observer"
            )
    else:
        isolated = copy.deepcopy(observer)
        name = _unique_module_name(
            model, f"{observer_node.name}_magnet_alpha_exact"
        )
        model.add_module(name, isolated)
        observer_node.target = name
        observer_node.meta[_ALPHA_EXACT_OUTPUT_OBSERVER] = True
        _rebind_downstream_derived_bias_observers(
            model, observer, isolated
        )

    # Preserve the first-pass scale while collecting new exact-output range.
    # Fake quantization here would apply the already encoded output twice.
    if isolated.amax_history.numel():
        isolated.amax_history.zero_()
    isolated.enable_observer()
    isolated.disable_fake_quant()
    return isolated


def _configure_pe_calibration_epilogues(
    model: GraphModule,
    *,
    configured_only: bool,
    reset_output_observers: bool,
    recompile: bool = True,
) -> int:
    modules = dict(model.named_modules(remove_duplicate=False))
    configured = 0
    for fixed_node in model.graph.nodes:
        if fixed_node.op != "call_module":
            continue
        fixed_module = modules.get(str(fixed_node.target))
        if not isinstance(fixed_module, AlphaCalibrationEpilogue):
            continue
        if configured_only and not fixed_module.exact_enabled:
            continue
        operands = _prepared_exact_pe_operands(model, fixed_node, modules)
        if operands is None:
            continue
        fine_endpoint, coarse_endpoint = fixed_module._observer_endpoints(
            operands.output_observer
        )
        if operands.mode.startswith("w8_"):
            assert operands.accumulator_scale is not None
            assert operands.accumulator_real_amax is not None
            fixed_module.configure_exact(
                operands.accumulator_scale,
                operands.real_bias,
                operands.output_scale,
                accumulator_real_amax=operands.accumulator_real_amax,
                output_precision=operands.output_precision,
                uses_pvs=operands.uses_pvs,
                relu=operands.relu,
                fine_quant_max=fine_endpoint,
                coarse_scale_quant_max=coarse_endpoint,
            )
        else:
            assert operands.input_observer is not None
            assert operands.weight_observer is not None
            assert operands.input_value is not None
            assert operands.weight_value is not None
            fixed_module.configure_exact_vsq(
                operands.input_observer,
                operands.weight_observer,
                operands.real_bias,
                operands.output_scale,
                output_precision=operands.output_precision,
                uses_pvs=operands.uses_pvs,
                stride=operands.stride,
                padding=operands.padding,
                dilation=operands.dilation,
                groups=operands.groups,
                relu=operands.relu,
                fine_quant_max=fine_endpoint,
                coarse_scale_quant_max=coarse_endpoint,
            )
            fixed_node.args = (
                fixed_node.args[0],
                operands.input_value,
                operands.weight_value,
            )
        configured += 1
        if reset_output_observers:
            isolated = _isolate_exact_output_observer(
                model,
                operands.output_observer_node,
                operands.output_observer,
            )
            modules[str(operands.output_observer_node.target)] = isolated
    if configured and recompile:
        model.graph.lint()
        model.recompile()
    return configured


def _configure_nmp_calibration_epilogues(
    model: GraphModule,
    *,
    configured_only: bool,
    reset_output_observers: bool,
    recompile: bool = True,
) -> int:
    modules = dict(model.named_modules(remove_duplicate=False))
    configured = 0
    for fixed_node in model.graph.nodes:
        if fixed_node.op != "call_module":
            continue
        fixed_module = modules.get(str(fixed_node.target))
        if not isinstance(fixed_module, AlphaNMPCalibrationEpilogue):
            continue
        if configured_only and not fixed_module.exact_enabled:
            continue
        operands = _prepared_exact_nmp_operands(model, fixed_node, modules)
        if operands is None:
            continue
        fine_endpoint, coarse_endpoint = (
            AlphaCalibrationEpilogue._observer_endpoints(
                operands.output_observer
            )
        )
        fixed_module.configure_exact(
            operands.input_observer,
            operands.weight_observer,
            operands.real_bias,
            operands.output_scale,
            output_precision=operands.output_precision,
            uses_pvs=operands.uses_pvs,
            relu=operands.relu,
            fine_quant_max=fine_endpoint,
            coarse_scale_quant_max=coarse_endpoint,
        )
        configured += 1
        if reset_output_observers:
            isolated = _isolate_exact_output_observer(
                model,
                operands.output_observer_node,
                operands.output_observer,
            )
            modules[str(operands.output_observer_node.target)] = isolated
    if configured and recompile:
        model.graph.lint()
        model.recompile()
    return configured


_ALPHA_PREPARED_PVS_SIDECAR = "magnet_alpha_prepared_pvs_sidecar"
_ALPHA_PREPARED_PVS_COMPONENT = "magnet_alpha_prepared_pvs_component"


@dataclass(frozen=True)
class _PreparedPVSValue:
    """Prepared-time physical O4 components and their semantic contract."""

    coarse: Node
    fine: Node
    payload: Node
    coarse_value: torch.Tensor
    logical_coarse_value: torch.Tensor
    contract: PreparedQuantizationContract
    relu_applied: bool


def _prepared_pvs_contract(observer):
    coarse = getattr(observer, "coarse", {})
    if (
        observer.dtype != "int4"
        or observer.block_size != OUTPUT_LANES
        or observer.scale_dtype != "uint8"
        or coarse.get("dtype") != "int8"
    ):
        return None
    return PreparedQuantizationContract.from_observer(observer)


def _prepared_pvs_coarse_value(module) -> torch.Tensor:
    value = (
        module.exact_output_scale
        if isinstance(module, AlphaCalibrationEpilogue)
        else module.output_scale
    )
    return value.detach().reshape(-1).clone()


def _matching_prepared_pvs_sidecars(model: GraphModule):
    sidecars = {}
    for node in model.graph.nodes:
        source_name = node.meta.get(_ALPHA_PREPARED_PVS_SIDECAR)
        if source_name is None:
            continue
        if source_name in sidecars:
            raise RuntimeError(f"{source_name}: duplicate PVS sidecars")
        sidecars[source_name] = node
    return sidecars


def _mark_prepared_pvs_component(node: Node, dtype: str) -> None:
    node.meta[_ALPHA_PREPARED_PVS_COMPONENT] = True
    node.meta["dtype"] = dtype


def _prepared_pvs_component_nodes(graph, sidecar: Node):
    components = {
        int(user.args[1]): user
        for user in sidecar.users
        if (
            user.target == operator.getitem
            and len(user.args) == 2
            and isinstance(user.args[1], int)
        )
    }
    if components:
        if set(components) != {0, 1, 2}:
            raise RuntimeError(f"{sidecar.name}: malformed PVS components")
        return components[0], components[1], components[2]

    created = []
    previous = sidecar
    for index in range(3):
        with graph.inserting_after(previous):
            component = graph.call_function(
                operator.getitem, (sidecar, index)
            )
        _mark_prepared_pvs_component(
            component, (None, "uint8", "int4")[index]
        )
        created.append(component)
        previous = component
    return tuple(created)


def _get_or_create_prepared_pvs_sidecar(
    model: GraphModule,
    fixed_node: Node,
    sidecars,
):
    sidecar = sidecars.get(fixed_node.name)
    kwargs = dict(fixed_node.kwargs)
    kwargs["return_components"] = True
    if sidecar is None:
        with model.graph.inserting_after(fixed_node):
            sidecar = model.graph.call_module(
                str(fixed_node.target), fixed_node.args, kwargs
            )
        sidecar.meta = {
            _ALPHA_PREPARED_PVS_SIDECAR: fixed_node.name,
            "dtype": (None, "uint8", "int4"),
        }
        sidecars[fixed_node.name] = sidecar
    else:
        sidecar.args = fixed_node.args
        sidecar.kwargs = kwargs
    return _prepared_pvs_component_nodes(model.graph, sidecar)


def _clone_prepared_pvs_unary(
    graph,
    semantic_node: Node,
    component: Node,
    *,
    dtype: str,
):
    args = list(semantic_node.args)
    args[0] = component
    with graph.inserting_before(semantic_node):
        cloned = graph.call_function(
            semantic_node.target, tuple(args), dict(semantic_node.kwargs)
        )
    _mark_prepared_pvs_component(cloned, dtype)
    return cloned


def _same_prepared_pvs_scale(left, right) -> bool:
    return left.shape == right.shape and torch.allclose(
        left.to(torch.float64),
        right.to(torch.float64),
        rtol=1e-6,
        atol=0.0,
    )


def _slice_kind(node: Node):
    """Classify the only slices whose PVS components stay addressable."""
    if node.target != torch.ops.aten.slice.Tensor or not node.args:
        return None
    source = node.args[0]
    if not isinstance(source, Node):
        return None
    step = get_arg_value(node, 4, "step", 1)
    if step not in (None, 1):
        return None
    source_shape = _node_shape(source)
    result_shape = _node_shape(node)
    if not source_shape or len(source_shape) != len(result_shape):
        return None
    rank = len(source_shape)
    dim = int(get_arg_value(node, 1, "dim", 0)) % rank
    start = get_arg_value(node, 2, "start", 0)
    end = get_arg_value(node, 3, "end", source_shape[dim])
    start = 0 if start is None else int(start)
    end = source_shape[dim] if end is None else int(end)
    if start == 0 and end >= source_shape[dim] and result_shape == source_shape:
        return "identity"
    if (
        dim == 1
        and start == 0
        and result_shape[1] <= source_shape[1]
        and all(
            result_shape[index] == source_shape[index]
            for index in range(rank)
            if index != 1
        )
    ):
        return "leading_channel"
    return None


def _join_prepared_pvs_values(graph, join: Node, values):
    reference = values[0]
    if any(value.contract != reference.contract for value in values[1:]):
        raise ValueError(
            f"{join.name}: incompatible prepared PVS observer contracts"
        )
    if any(
        not _same_prepared_pvs_scale(
            value.coarse_value, reference.coarse_value
        )
        or not _same_prepared_pvs_scale(
            value.logical_coarse_value, reference.logical_coarse_value
        )
        for value in values[1:]
    ):
        raise ValueError(
            f"{join.name}: prepared PVS inputs do not share one coarse scale"
        )
    input_shapes = tuple(_node_shape(node) for node in join.args[0])
    if not input_shapes or not input_shapes[0]:
        raise ValueError(f"{join.name}: prepared PVS input shape is absent")
    rank = len(input_shapes[0])
    channel_axis = int(reference.contract.ch_axis) % rank
    join_axis = int(get_arg_value(join, 1, "dim", 0)) % rank
    block_size = int(reference.contract.block_size)
    if join_axis != channel_axis:
        raise ValueError(
            f"{join.name}: prepared PVS join must concatenate blocks"
        )
    if any(
        len(shape) != rank or shape[channel_axis] % block_size
        for shape in input_shapes
    ):
        raise ValueError(
            f"{join.name}: prepared PVS inputs must be block aligned"
        )
    args = list(join.args)
    args[0] = tuple(value.payload for value in values)
    with graph.inserting_before(join):
        payload = graph.call_function(
            join.target, tuple(args), dict(join.kwargs)
        )
    args[0] = tuple(value.fine for value in values)
    with graph.inserting_before(join):
        fine = graph.call_function(
            join.target, tuple(args), dict(join.kwargs)
        )
    _mark_prepared_pvs_component(payload, "int4")
    _mark_prepared_pvs_component(fine, "uint8")
    return _PreparedPVSValue(
        coarse=reference.coarse,
        fine=fine,
        payload=payload,
        coarse_value=reference.coarse_value,
        logical_coarse_value=reference.logical_coarse_value,
        contract=reference.contract,
        relu_applied=all(value.relu_applied for value in values),
    )


def _slice_prepared_pvs_value(graph, node: Node, value):
    kind = _slice_kind(node)
    if kind is None:
        raise ValueError(f"{node.name}: unsupported prepared PVS slice")
    payload = _clone_prepared_pvs_unary(
        graph, node, value.payload, dtype="int4"
    )
    fine = value.fine
    if kind == "leading_channel":
        result_shape = _node_shape(node)
        if value.contract.ch_axis is None:
            raise ValueError(f"{node.name}: PVS block axis is absent")
        axis = int(value.contract.ch_axis) % len(result_shape)
        if axis != 1:
            raise ValueError(f"{node.name}: unsupported PVS block axis")
        fine_channels = math.ceil(
            result_shape[1] / int(value.contract.block_size)
        )
        with graph.inserting_before(node):
            fine = graph.call_function(
                torch.ops.aten.slice.Tensor,
                (value.fine, 1, 0, fine_channels, 1),
            )
        _mark_prepared_pvs_component(fine, "uint8")
    return _PreparedPVSValue(
        coarse=value.coarse,
        fine=fine,
        payload=payload,
        coarse_value=value.coarse_value,
        logical_coarse_value=value.logical_coarse_value,
        contract=value.contract,
        relu_applied=value.relu_applied,
    )


def _validate_prepared_pvs_input(fixed_node, observer, value) -> None:
    if _prepared_pvs_contract(observer) != value.contract:
        raise ValueError(
            f"{fixed_node.name}: VSQ input contract differs from PVS producer"
        )
    logical_scale = observer.calculate_qparams().detach().reshape(-1)
    if not _same_prepared_pvs_scale(
        logical_scale, value.logical_coarse_value
    ):
        raise ValueError(
            f"{fixed_node.name}: VSQ coarse scale differs from PVS producer"
        )


def _clear_prepared_pvs_transfers(model: GraphModule) -> None:
    changed = False
    for node in model.graph.nodes:
        if "input_components" in node.kwargs:
            kwargs = dict(node.kwargs)
            kwargs.pop("input_components")
            node.kwargs = kwargs
            changed = True
    sidecars = {
        node
        for node in model.graph.nodes
        if node.meta.get(_ALPHA_PREPARED_PVS_SIDECAR)
    }
    direct = {
        node
        for node in model.graph.nodes
        if (
            node.meta.get(_ALPHA_PREPARED_PVS_COMPONENT)
            and node.target == operator.getitem
            and isinstance(node.args[0], Node)
            and node.args[0] in sidecars
        )
    }
    derived = {
        node
        for node in model.graph.nodes
        if node.meta.get(_ALPHA_PREPARED_PVS_COMPONENT) and node not in direct
    }
    for node in reversed(tuple(model.graph.nodes)):
        if node in derived and not node.users:
            model.graph.erase_node(node)
            changed = True
    dangling = [node.name for node in derived if node in model.graph.nodes]
    if dangling:
        raise RuntimeError(
            "prepared PVS transfers were not removable: "
            + ", ".join(dangling)
        )
    if changed:
        model.graph.lint()
        model.recompile()


def _prepared_pvs_transfer_rules() -> PreparedDomainRuleSet:
    rules = [
        PreparedDomainTransferRule(
            target=target,
            input_format=PreparedDomainInputFormat.FIRST,
            relation=PreparedDomainRelation.MAP_COMPONENTS,
        )
        for target in (
            torch.ops.aten.clone.default,
            torch.ops.aten.contiguous.default,
            torch.ops.aten.upsample_nearest2d.vec,
        )
    ]
    rules.extend(
        (
            PreparedDomainTransferRule(
                target=torch.ops.aten.relu.default,
                input_format=PreparedDomainInputFormat.FIRST,
                relation=PreparedDomainRelation.ALIAS,
            ),
            PreparedDomainTransferRule(
                target=torch.ops.aten.slice.Tensor,
                input_format=PreparedDomainInputFormat.FIRST,
                relation=PreparedDomainRelation.MAP_COMPONENTS,
                predicate=lambda node: _slice_kind(node) is not None,
            ),
            PreparedDomainTransferRule(
                target=torch.ops.aten.cat.default,
                input_format=PreparedDomainInputFormat.SEQUENCE,
                relation=PreparedDomainRelation.JOIN,
            ),
        )
    )
    return PreparedDomainRuleSet(tuple(rules))


def _materialize_prepared_pvs_values(model: GraphModule) -> None:
    """Carry O4 components through prepared movement into VSQ consumers."""
    _clear_prepared_pvs_transfers(model)
    graph = model.graph
    modules = dict(model.named_modules(remove_duplicate=False))
    sidecars = _matching_prepared_pvs_sidecars(model)
    original_nodes = tuple(
        node
        for node in graph.nodes
        if not node.meta.get(_ALPHA_PREPARED_PVS_COMPONENT)
        and not node.meta.get(_ALPHA_PREPARED_PVS_SIDECAR)
    )
    rules = _prepared_pvs_transfer_rules()
    values = {}

    for node in original_nodes:
        module = (
            modules.get(str(node.target)) if node.op == "call_module" else None
        )
        producer_module = None
        if isinstance(module, AlphaCalibrationEpilogue):
            if not module.exact_enabled:
                continue
            if module.exact_mode.startswith("vsq_"):
                input_source = node.args[1] if len(node.args) >= 2 else None
                input_value = values.get(input_source)
                kwargs = dict(node.kwargs)
                if input_value is None:
                    kwargs.pop("input_components", None)
                else:
                    conv = node.args[0]
                    input_observer = _prepared_vsq_observer(
                        get_arg_value(conv, 0, "input"), modules
                    )
                    if input_observer is None:
                        raise ValueError(
                            f"{node.name}: encoded input has no VSQ observer"
                        )
                    _validate_prepared_pvs_input(
                        node, input_observer, input_value
                    )
                    kwargs["input_components"] = (
                        input_value.coarse,
                        input_value.fine,
                        input_value.payload,
                    )
                node.kwargs = kwargs
            if module.uses_pvs:
                producer_module = module
        elif isinstance(module, AlphaNMPCalibrationEpilogue):
            if module.exact_enabled and module.uses_pvs:
                producer_module = module

        if producer_module is not None:
            output = _prepared_exact_output(node, modules)
            if output is None:
                raise ValueError(f"{node.name}: PVS producer has no observer")
            _output_node, observer, relu = output
            contract = _prepared_pvs_contract(observer)
            if contract is None:
                raise ValueError(f"{node.name}: PVS contract is absent")
            coarse, fine, payload = _get_or_create_prepared_pvs_sidecar(
                model, node, sidecars
            )
            values[node] = _PreparedPVSValue(
                coarse=coarse,
                fine=fine,
                payload=payload,
                coarse_value=_prepared_pvs_coarse_value(producer_module),
                logical_coarse_value=(
                    observer.calculate_qparams()
                    .detach()
                    .reshape(-1)
                    .clone()
                ),
                contract=contract,
                relu_applied=relu,
            )
            continue

        # Current transfer rules deliberately describe call_function nodes.
        # Prepared observers are call_module aliases and are handled here
        # explicitly rather than weakening the generic rule vocabulary.
        if isinstance(module, FusedAmaxObsFakeQuantize):
            source = node.args[0] if node.args else None
            source_value = values.get(source)
            if source_value is None:
                continue
            contract = _prepared_pvs_contract(module)
            if contract != source_value.contract:
                raise ValueError(
                    f"{node.name}: prepared PVS observer contract changed"
                )
            logical_scale = module.calculate_qparams().detach().reshape(-1)
            if not _same_prepared_pvs_scale(
                logical_scale, source_value.logical_coarse_value
            ):
                raise ValueError(
                    f"{node.name}: prepared PVS observer changed coarse scale"
                )
            values[node] = source_value
            continue

        transfer = rules.resolve(node)
        if transfer is None:
            continue
        input_values = tuple(values.get(item) for item in transfer.inputs)
        present = tuple(value for value in input_values if value is not None)
        if not present:
            continue
        if len(present) != len(input_values):
            if transfer.rule.relation is PreparedDomainRelation.JOIN:
                raise ValueError(
                    f"{node.name}: cannot mix encoded and semantic PVS inputs"
                )
            continue
        source_value = present[0]
        if node.target == torch.ops.aten.relu.default:
            if not source_value.relu_applied:
                raise ValueError(f"{node.name}: PVS ReLU was not encoded")
            values[node] = source_value
        elif node.target == torch.ops.aten.slice.Tensor:
            values[node] = _slice_prepared_pvs_value(
                graph, node, source_value
            )
        elif transfer.rule.relation is PreparedDomainRelation.JOIN:
            values[node] = _join_prepared_pvs_values(
                graph, node, present
            )
        elif transfer.rule.relation is PreparedDomainRelation.MAP_COMPONENTS:
            values[node] = _PreparedPVSValue(
                coarse=source_value.coarse,
                fine=_clone_prepared_pvs_unary(
                    graph, node, source_value.fine, dtype="uint8"
                ),
                payload=_clone_prepared_pvs_unary(
                    graph, node, source_value.payload, dtype="int4"
                ),
                coarse_value=source_value.coarse_value,
                logical_coarse_value=source_value.logical_coarse_value,
                contract=source_value.contract,
                relu_applied=source_value.relu_applied,
            )
        else:
            values[node] = source_value

    graph.lint()
    model.recompile()


def _remove_prepared_pvs_values(model: GraphModule) -> None:
    """Remove calibration-only component edges before target lowering."""
    for node in model.graph.nodes:
        if "input_components" in node.kwargs:
            kwargs = dict(node.kwargs)
            kwargs.pop("input_components")
            node.kwargs = kwargs
    removable = {
        node
        for node in model.graph.nodes
        if node.meta.get(_ALPHA_PREPARED_PVS_COMPONENT)
        or node.meta.get(_ALPHA_PREPARED_PVS_SIDECAR)
    }
    for node in reversed(tuple(model.graph.nodes)):
        if node not in removable:
            continue
        remaining = [user for user in node.users if user not in removable]
        if remaining:
            names = ", ".join(user.name for user in remaining)
            raise RuntimeError(f"{node.name}: PVS sidecar still feeds {names}")
        if not node.users:
            model.graph.erase_node(node)
    dangling = [node.name for node in removable if node in model.graph.nodes]
    if dangling:
        raise RuntimeError(
            "prepared PVS sidecars were not removable: "
            + ", ".join(dangling)
        )
    model.graph.lint()
    model.recompile()


def prepare_quantization_recalibration(
    model: GraphModule,
    *,
    reset_observers: bool,
) -> int:
    """Configure exact PE/NMP execution for the second PTQ replay."""
    from voyager_compiler.quantization.regions import (
        finalize_quantization_regions,
    )

    finalize_quantization_regions(model)
    pe_count = _configure_pe_calibration_epilogues(
        model,
        configured_only=False,
        reset_output_observers=reset_observers,
    )
    nmp_count = _configure_nmp_calibration_epilogues(
        model,
        configured_only=False,
        reset_output_observers=reset_observers,
    )
    _materialize_prepared_pvs_values(model)
    return pe_count + nmp_count


def unwrap_calibration_epilogues(model: GraphModule) -> int:
    """Remove every reference. Returns how many.

    Run before conversion. The reference is an instrument: it exists so the
    observers see the right range, and once they have it the graph must be
    the one the lowering expects, with the convolution's result reaching its
    consumer directly.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    removed = 0
    for node in list(model.graph.nodes):
        if node.op != "call_module":
            continue
        if not isinstance(
            modules.get(str(node.target)),
            (AlphaCalibrationEpilogue, AlphaNMPCalibrationEpilogue),
        ):
            continue
        source = node.args[0] if node.args else None
        if not isinstance(source, Node):
            raise ValueError(f"{node.name}: a reference with no input")
        node.replace_all_uses_with(source)
        model.graph.erase_node(node)
        removed += 1
    if removed:
        model.graph.lint()
        model.graph.eliminate_dead_code()
        model.recompile()
        model.delete_all_unused_submodules()
    LOGGER.info("removed %d Alpha calibration references", removed)
    return removed


def cleanup_quantization_preparation(model: GraphModule) -> GraphModule:
    """Remove Alpha's temporary PTQ execution references."""
    _remove_prepared_pvs_values(model)
    unwrap_calibration_epilogues(model)
    return model


def finalize_quantization_calibration(model: GraphModule) -> GraphModule:
    """Merge ranges belonging to one Alpha physical domain."""
    from voyager_compiler.quantization.regions import (
        finalize_quantization_regions,
    )

    finalize_quantization_regions(model)
    _configure_pe_calibration_epilogues(
        model,
        configured_only=True,
        reset_output_observers=False,
    )
    _configure_nmp_calibration_epilogues(
        model,
        configured_only=True,
        reset_output_observers=False,
    )
    return model


def _is_prepared_vsq_join_boundary(node: Node) -> bool:
    """Whether ``node`` is the encoded output of one Alpha PE invocation."""
    current = node
    visited = set()
    while isinstance(current, Node) and current not in visited:
        visited.add(current)
        source = current.args[0] if current.args else None
        if (
            current.op == "call_function"
            and current.target in _ACTIVATION_EPILOGUES
            and isinstance(source, Node)
        ):
            current = source
            continue
        if (
            current.op == "call_function"
            and isinstance(source, Node)
            and _transparent_view(current, source)
        ):
            current = source
            continue
        if current.op == "call_module" and current.meta.get(
            ALPHA_CALIBRATION_REFERENCE
        ):
            if not isinstance(source, Node):
                return False
            current = source
            continue
        return (
            current.op == "call_function"
            and current.target in _MATRIX_SOURCES
            and int(get_arg_value(current, 6, "groups", 1)) == 1
        )
    return False


def prepare_quantization(
    model: GraphModule,
    *,
    region_qspec=None,
    domain_rules=None,
) -> GraphModule:
    """Alpha's ordering around the shared region pass.

    Output observers move first, to the value Alpha truncates: an absorbed
    activation is upstream of truncation, not a separately stored value.
    References then bracket region formation, because propagation can expose
    a boundary that was not explicit before it installed an entry observer.
    """
    from voyager_compiler.quantization.regions import (
        move_output_observers_after_epilogues,
        prepare_quantization_regions,
        prepare_vsq_join_regions,
    )

    if domain_rules is None:
        # Runtime import avoids the calibration/traits module cycle while
        # still making the target hook complete for direct API callers.
        from voyager_compiler.codegen.targets.magnet_alpha.traits import (
            prepared_domain_rules,
        )

        domain_rules = prepared_domain_rules()

    move_output_observers_after_epilogues(model)
    insert_calibration_epilogues(model)
    insert_nmp_calibration_epilogues(model)
    prepare_quantization_regions(
        model, region_qspec=region_qspec, rule_set=domain_rules
    )
    prepare_vsq_join_regions(
        model,
        is_output_boundary=_is_prepared_vsq_join_boundary,
        rule_set=domain_rules,
    )
    insert_calibration_epilogues(model)
    insert_nmp_calibration_epilogues(model)
    return model
