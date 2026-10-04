"""Training with fake-quantized GEMMs, forward only or backward too.

``prepare_from_args`` picks the mode from the flags.  Without an error spec
it is QAT: the export is prepared as ``prepare_qat_pt2e`` prepares it, and
only the forward pass is quantized.  With one, ``prepare_training``
quantizes the backward pass too.

``capture_training`` hands ``prepare_training`` AOTAutograd's joint graph of
the forward and backward passes before splitting it.  AOTAutograd traces it
with the ops broken into the ones the compiler lowers, and each GEMM is
renamed back to the op the model wrote -- a forward GEMM to its ``linear``
or ``matmul``, a backward one to ``matmul``.  Each GEMM operand is tagged
with its type: an error is computed from an incoming gradient, a weight
views a parameter, and anything else is an activation.  The quantizer
annotates the joint graph as it annotates an export, a tagged operand
taking its type's spec blocked along its GEMM's contraction axis, and
``prepare`` inserts the fake-quants.  Everything but the GEMM operands stays
in high precision, the bias gradient included.

A tensor several GEMMs read, as it is or transposed, with the same spec
whose values do not depend on the axis -- per-tensor, a plain cast, or
blocks spanning both axes -- is quantized once: every such GEMM reads one
fake-quant's output, which the split saves when the backward reads it.

For the compiler, a training step is two graphs: ``gradient_program`` runs
the forward, the loss and the backward, and ``update_program`` the
optimizer's update.  Each writes the tensors it updates in place.
"""

import copy
import types
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from torch._decomp import core_aten_decompositions
from torch._functorch import config as functorch_config
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.descriptors import (
    BufferAOTInput,
    GradAOTOutput,
    ParamAOTInput,
    PlainAOTInput,
    TangentAOTInput,
)
from torch._functorch.aot_autograd import (
    aot_module_simplified,
    make_boxed_func,
)
from torch._functorch.partitioners import default_partition
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.export._unlift import _check_inputs_match
from torch.fx import CodeGen, Graph, GraphModule, Node
from torch.utils import _pytree as pytree
from torchao.quantization.pt2e import FakeQuantizeBase
from torchao.quantization.pt2e import prepare as torchao_prepare
from torchao.quantization.pt2e.quantizer import QuantizationAnnotation

from voyager_compiler.export_utils import (
    create_getattr_from_value,
    export_model,
)
from voyager_compiler.quantization.fake_quantize import (
    FusedAmaxObsFakeQuantize,
    MXFakeQuantize,
    RandomHadamardTransform,
    _DerivedObserverOrFakeQuantize,
)
from voyager_compiler.quantization.quantize_pt2e import (
    _get_obs_or_fq_map,
    _replace_observer_with_quantize_mx_node_decomposed,
    get_default_quantizer,
    prepare_qat_pt2e,
    set_training,
)
from voyager_compiler.quantization.quantizer.quantizer import (
    QScheme,
    QuantizationSpec,
)
from voyager_compiler.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
)

__all__ = [
    "TrainingQuantizers",
    "capture_training",
    "disable_observers",
    "gradient_program",
    "prepare_from_args",
    "prepare_training",
    "update_program",
]

aten = torch.ops.aten

# The GEMMs AOTAutograd lowers linear and matmul to.
_GEMMS = (aten.mm.default, aten.addmm.default, aten.bmm.default)
# The ops the quantizer annotates as GEMMs.
_OPS = (aten.linear.default, aten.matmul.default)
# Ops that only view a tensor: an operand read through them is still the
# tensor they start from.
_VIEWS = {
    aten.view.default,
    aten._unsafe_view.default,
    aten.reshape.default,
    aten.t.default,
    aten.transpose.int,
    aten.permute.default,
    aten.expand.default,
    aten.alias.default,
    aten.unsqueeze.default,
    aten.squeeze.dim,
}
# Schemes whose scales belong to blocks along an axis.
_BLOCKS = (QScheme.MICROSCALING, QScheme.GROUP_WISE_AFFINE)


def _quantized_once(spec: QuantizationSpec) -> bool:
    """Whether ``spec`` gives a tensor the same values whatever the axis.

    A per-tensor scale, a plain cast, or blocks spanning both of a matrix's
    axes quantize a matrix and its transpose alike, so one quantization
    serves every GEMM reading the tensor -- unless it is rotated along an
    axis first.
    """
    # A bias's derived spec has no rotation.
    if getattr(spec, "rht_axis", None) is not None:
        return False
    if spec.qscheme in _BLOCKS:
        return isinstance(spec.ch_axis, tuple) and len(spec.ch_axis) > 1
    return spec.qscheme in (None, QScheme.PER_TENSOR_SYMMETRIC)


def _fake_quantize(fake_quant: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply ``fake_quant`` to ``x``.

    The graph node calling a fake-quant: AOTAutograd's split keeps a
    ``call_function``, where it drops a ``call_module``.
    """
    return fake_quant(x)


class TrainingQuantizers(nn.Module):
    """The fake-quants of a quantized training run.

    Held by the model, the fake-quants move with it and are saved in its
    state dict.  ``disable_observers`` leaves the ones named in
    ``gradients``, which quantize gradients, observing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gradients = set()


def _fold_permutes(joint: GraphModule) -> None:
    """Fold each permute of a permute into one permute of the source.

    Autograd transposes a transpose -- the backward reads a linear's weight
    through two permutes, and a weight gradient leaves through two -- so a
    pair that cancels becomes the tensor itself, and a GEMM operand reads
    the tensor it transposes.

    Args:
        joint: AOTAutograd's joint graph, rewritten in place.
    """
    graph = joint.graph
    for node in list(graph.nodes):
        if (
            node.target is not aten.permute.default
            or node.args[0].target is not aten.permute.default
        ):
            continue
        source, inner_dims = node.args[0].args
        ndim = len(inner_dims)
        dims = [inner_dims[d] % ndim for d in node.args[1]]
        if dims == list(range(ndim)):
            node.replace_all_uses_with(source)
            graph.erase_node(node)
        else:
            node.args = (source, dims)
    joint.recompile()


def _rename_gemms(joint: GraphModule, export: GraphModule) -> None:
    """Rename the joint graph's GEMMs back to the ops the model wrote.

    A forward GEMM becomes the ``linear`` or ``matmul`` of the export it was
    traced from; a backward one becomes ``matmul``.  A forward GEMM traced
    from any other op, such as an einsum, is left as it is.

    Args:
        joint: AOTAutograd's joint graph, rewritten in place.
        export: The graph AOTAutograd traced.
    """
    ops = {n.name: n.target for n in export.graph.nodes}
    graph = joint.graph
    for gemm in list(graph.nodes):
        if gemm.op != "call_function" or gemm.target not in _GEMMS:
            continue
        op = aten.matmul.default
        if gemm.meta.get("partitioner_tag") != "is_backward":
            op = ops.get(gemm.meta["from_node"][-1].name)
        if op not in _OPS:
            continue
        if gemm.target is aten.addmm.default:
            bias, a, b = gemm.args
        else:
            (a, b), bias = gemm.args, None
        if op is aten.linear.default and gemm.target is not aten.bmm.default:
            # A linear's lowering reads its weight transposed.
            args = (a, b.args[0]) if bias is None else (a, b.args[0], bias)
        else:
            op, args = aten.matmul.default, (a, b)
        with graph.inserting_before(gemm):
            node = graph.call_function(op, args)
        node.meta = gemm.meta
        gemm.replace_all_uses_with(node)
        graph.erase_node(gemm)
    joint.recompile()


def _unflatten_gemms(joint: GraphModule) -> None:
    """Give each GEMM the operands before AOTAutograd flattened them.

    AOTAutograd traces a ``linear`` or ``matmul`` of N-d tensors as a 2-d or
    3-d GEMM of views that flatten their leading dimensions, and views the
    result back.  Where the result is viewed back, the GEMM reads the
    tensors those views flatten -- transposed, if it read the view
    transposed, and past any ``expand`` that changes nothing -- and gives
    the N-d result, as in the model.  A GEMM contracting over the flattened
    dimensions, as a weight gradient does, keeps its views.

    Args:
        joint: AOTAutograd's joint graph, rewritten in place.
    """
    graph = joint.graph

    def unflattened(operand, gemm, lead):
        """The tensor whose ``lead`` dimensions ``operand`` flattens, or
        ``None``."""
        ndim = operand.meta["val"].ndim
        swap = [*range(ndim - 2), ndim - 1, ndim - 2]
        transposed = (
            operand.target is aten.permute.default
            and list(operand.args[1]) == swap
        )
        view = operand.args[0] if transposed else operand
        if view.target is not aten.view.default:
            return None
        tensor = view.args[0]
        while (
            tensor.target is aten.expand.default
            and tensor.meta["val"].shape == tensor.args[0].meta["val"].shape
        ):
            tensor = tensor.args[0]
        if tensor.meta["val"].shape != (*lead, *view.meta["val"].shape[1:]):
            return None
        if not transposed:
            return tensor
        n = len(tensor.meta["val"].shape)
        with graph.inserting_before(gemm):
            node = graph.call_function(
                aten.permute.default, (tensor, [*range(n - 2), n - 1, n - 2])
            )
        node.meta = {**operand.meta, "val": tensor.meta["val"].mT}
        return node

    for gemm in list(graph.nodes):
        if gemm.target not in _OPS or len(gemm.users) != 1:
            continue
        (view,) = gemm.users
        if view.target is not aten.view.default:
            continue
        _, *rest = gemm.meta["val"].shape
        lead = tuple(view.meta["val"].shape[: -len(rest)])
        if len(lead) < 2 or view.meta["val"].shape != (*lead, *rest):
            continue
        # A 3-d GEMM flattens both operands; a 2-d one, only the rows.
        count = 2 if len(rest) == 2 else 1
        operands = [unflattened(a, gemm, lead) for a in gemm.args[:count]]
        if None in operands:
            continue
        gemm.args = (*operands, *gemm.args[count:])
        gemm.meta["val"] = view.meta["val"]
        view.replace_all_uses_with(gemm)
        graph.erase_node(view)
    graph.eliminate_dead_code()
    joint.recompile()


def _tag_operands(joint: GraphModule) -> Set[Node]:
    """Tag each linear and matmul of the joint graph with its operand types
    and its kind.

    ``meta["operand_types"]`` holds a GEMM's two operand types: an error is
    computed from an incoming gradient (a tangent input), a weight views a
    parameter, and anything else is an activation.  ``meta["gemm_kind"]``
    is ``"fprop"`` for a forward GEMM, ``"wgrad"`` for a backward GEMM whose
    result reaches a parameter's gradient through views and sums only, and
    ``"dgrad"`` for any other backward GEMM.

    Args:
        joint: AOTAutograd's joint graph, tagged in place.

    Returns:
        The nodes computed from incoming gradients.
    """
    errors = set()
    for node in joint.graph.nodes:
        if isinstance(
            node.meta.get("desc"), TangentAOTInput
        ) or not errors.isdisjoint(node.all_input_nodes):
            errors.add(node)
    output = joint.graph.output_node()
    pending = [
        value
        for value, desc in zip(output.args[0], output.meta["desc"])
        if isinstance(desc, GradAOTOutput) and value is not None
    ]
    wgrads = set()
    while pending:
        node = pending.pop()
        if node.target in _OPS:
            wgrads.add(node)
        elif node.target in _VIEWS or node.target is aten.add.Tensor:
            pending.extend(node.all_input_nodes)
    for gemm in joint.graph.nodes:
        if gemm.target not in _OPS:
            continue
        if gemm.meta.get("partitioner_tag") != "is_backward":
            gemm.meta["gemm_kind"] = "fprop"
        elif gemm in wgrads:
            gemm.meta["gemm_kind"] = "wgrad"
        else:
            gemm.meta["gemm_kind"] = "dgrad"
        kinds = []
        for operand in gemm.args[:2]:
            source = operand
            while source.target in _VIEWS:
                source = source.args[0]
            if operand in errors:
                kinds.append("error")
            elif isinstance(source.meta.get("desc"), ParamAOTInput):
                kinds.append("weight")
            else:
                kinds.append("activation")
        gemm.meta["operand_types"] = tuple(kinds)
    return errors


def _is_transpose(node: Node) -> bool:
    """Whether ``node`` swaps the last two axes of its input."""
    if node.op != "call_function":
        return False
    if node.target is aten.t.default:
        return True
    if node.target is aten.permute.default:
        ndim = node.meta["val"].dim()
        dims = [d % ndim for d in node.args[1]]
        return dims == list(range(ndim - 2)) + [ndim - 1, ndim - 2]
    if node.target is not aten.transpose.int:
        return False
    ndim = node.meta["val"].dim()
    return {d % ndim for d in node.args[1:3]} == {ndim - 2, ndim - 1}


def _share_quantized_transposes(joint: GraphModule) -> None:
    """Quantize once a tensor that one GEMM reads and another transposes.

    ``prepare`` gives a tensor's readers with the same spec one fake-quant.
    A GEMM operand that transposes a tensor another GEMM reads, both with
    the same spec quantizing once, becomes such a reader: its transposes
    are rebuilt to take the tensor through that fake-quant.

    Args:
        joint: Annotated joint graph, rewritten in place.
    """
    graph = joint.graph
    for gemm in list(graph.nodes):
        annotation = gemm.meta.get("quantization_annotation")
        if annotation is None:
            continue
        for operand, spec in list(annotation.input_qspec_map.items()):
            if spec is None or not _quantized_once(spec):
                continue
            chain, source, shared = [], operand, False
            while not shared and _is_transpose(source):
                chain.append(source)
                source = source.args[0]
                shared = any(
                    user.meta["quantization_annotation"].input_qspec_map.get(
                        source
                    )
                    == spec
                    for user in source.users
                    if "quantization_annotation" in user.meta
                )
            if not shared:
                continue
            copies = {source: source}
            with graph.inserting_before(gemm):
                for node in reversed(chain):
                    copies[node] = graph.node_copy(node, copies.__getitem__)
                    # The split places a node by its tag.
                    copies[node].meta["partitioner_tag"] = gemm.meta[
                        "partitioner_tag"
                    ]
            copies[chain[-1]].meta["quantization_annotation"] = (
                QuantizationAnnotation(
                    input_qspec_map={source: spec}, _annotated=True
                )
            )
            gemm.replace_input_with(operand, copies[operand])
            del annotation.input_qspec_map[operand]
    joint.recompile()


def capture_training(
    model: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    dynamic_shapes: Optional[Dict[str, Any]],
    make_transform: Optional[
        Callable[[GraphModule], Callable[[GraphModule], GraphModule]]
    ] = None,
    hidden: Sequence[str] = (),
) -> None:
    """Run ``model`` through the graphs AOTAutograd derives from its export.

    ``model``, in training mode, is exported once with the example inputs.
    AOTAutograd traces the export's forward and backward into one joint
    graph, with the ops broken into the ones the compiler lowers and the
    GEMMs named as the model wrote them.  ``make_transform(export)``, when
    given, rewrites the joint graph before AOTAutograd splits it into the
    forward graph and the backward graph that differentiates it.
    ``model.forward.graphs`` holds the ``"joint"``, ``"forward"`` and
    ``"backward"`` graphs, and ``model``'s forward is replaced to run the
    last two.  Parameters stay the
    model's own, so optimizers, checkpoints and ``train()`` / ``eval()``
    work as on the eager model: ``eval()`` switches the graphs' dropouts
    off, and under ``torch.no_grad()`` the outputs come back detached.
    Moving or casting the model, and choosing which parameters require
    grad, must come before this call.  If capture fails, ``model`` is left
    as it was.  A deep copy of a captured model still runs the original's
    graphs, and pickling one fails.

    Args:
        model: Module to capture, in training mode.
        args: Positional example inputs.
        kwargs: Keyword example inputs.
        dynamic_shapes: Dynamic-shape spec forwarded to ``torch.export``.
        make_transform: Called with the export graph; returns the rewrite of
            the joint graph.  None leaves the joint graph as traced.
        hidden: Names of ``model``'s submodules to detach while exporting,
            so that state the forward never reads stays out of the graphs.

    Raises:
        ValueError: ``model`` is in eval mode.
        NotImplementedError: The forward graph has a batch norm or a fused
            attention dropout, which cannot switch between training and
            eval in place.
    """
    if not model.training:
        raise ValueError("Capture the model in training mode")
    detached = {name: model._modules.pop(name) for name in hidden}
    try:
        gm = export_model(model, args, kwargs, dynamic_shapes=dynamic_shapes)
    finally:
        model._modules.update(detached)
    transform = None if make_transform is None else make_transform(gm)
    in_spec, out_spec = gm._in_spec, gm._out_spec
    inputs = [n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder"]
    # AOTAutograd takes flat inputs and a tuple of outputs.
    gm.graph._codegen = CodeGen()
    output = next(n for n in gm.graph.nodes if n.op == "output")
    output.args = (tuple(output.args[0]),)
    gm.recompile()
    graphs = {}

    def partition(joint, joint_inputs, **kwargs):
        _rename_gemms(joint, gm)
        _unflatten_gemms(joint)
        _fold_permutes(joint)
        graphs["joint"] = joint if transform is None else transform(joint)
        return default_partition(graphs["joint"], joint_inputs, **kwargs)

    def compile_forward(graph, _):
        graphs["forward"] = graph
        return make_boxed_func(graph)

    def compile_backward(graph, _):
        # AOTAutograd hands over a deep copy; the backward calls the joint
        # graph's submodules, as the forward does.
        for name, _ in list(graph.named_children()):
            setattr(graph, name, getattr(graphs["joint"], name))
        # Drop the partitioner's dead copies of random forward ops, which
        # dead-code elimination otherwise keeps as impure.
        graph.graph.eliminate_dead_code(
            is_impure_node=lambda n: n.is_impure(impure_random=False)
        )
        graph.recompile()
        graphs["backward"] = graph
        return make_boxed_func(graph)

    # The backward is compiled now, not at the first backward call.
    with (
        torch.enable_grad(),
        functorch_config.patch(force_non_lazy_backward_lowering=True),
    ):
        run = aot_module_simplified(
            gm,
            inputs,
            fw_compiler=compile_forward,
            bw_compiler=compile_backward,
            partition_fn=partition,
            # Backward ops break into the smaller ones the compiler lowers.
            decompositions=core_aten_decompositions(),
        )
    for node in graphs["forward"].graph.nodes:
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            continue
        names = [a.name for a in schema.arguments]
        dropout_p = 0.0
        if "dropout_p" in names:
            index = names.index("dropout_p")
            dropout_p = (
                node.args[index]
                if index < len(node.args)
                else node.kwargs.get("dropout_p", 0.0)
            )
        if "batch_norm" in schema.name or dropout_p:
            raise NotImplementedError(
                f"{node.target} cannot switch between training and eval"
            )

    def visible(name):
        return name.split(".")[0] not in hidden

    requires_grad = [
        p.requires_grad for n, p in model.named_parameters() if visible(n)
    ]
    training = [True]

    def forward(*call_args, **call_kwargs):
        if [
            p.requires_grad for n, p in model.named_parameters() if visible(n)
        ] != requires_grad:
            raise RuntimeError(
                "Which parameters require grad changed after capture"
            )
        if model.training != training[0]:
            training[0] = model.training
            # Out of training a dropout passes its input through.
            for graph in graphs.values():
                for node in graph.graph.nodes:
                    if node.target == torch.ops.aten.native_dropout.default:
                        node.update_arg(2, model.training)
                graph.recompile()
        flat = _check_inputs_match(call_args, call_kwargs, in_spec)
        outputs = list(run(*(leaf for _, leaf in flat)))
        # AOTAutograd records the backward even under no_grad.
        if not torch.is_grad_enabled():
            outputs = [
                o.detach() if isinstance(o, torch.Tensor) else o
                for o in outputs
            ]
        return pytree.tree_unflatten(outputs, out_spec)

    forward.graphs = graphs
    model.forward = forward


def prepare_training(
    model: nn.Module,
    quantizer: XNNPACKQuantizer,
    example_args: Tuple[Any, ...],
    example_kwargs: Optional[Dict[str, Any]],
    dynamic_shapes,
) -> nn.Module:
    """Quantize ``model``'s GEMMs in the forward and the backward pass.

    ``model`` keeps its parameters, names and ``train()`` / ``eval()``; its
    forward runs the graphs ``capture_training`` captures.  The fake-quants
    are held by ``model.training_quantizers``.

    Args:
        model: Float model, in training mode.
        quantizer: Annotates which GEMMs are quantized and how; its GEMM
            configs carry the error spec.
        example_args: Positional inputs to export with.
        example_kwargs: Keyword inputs to export with.
        dynamic_shapes: Dynamic dimensions of the inputs.

    Returns:
        ``model``.

    Raises:
        NotImplementedError: The quantizer would quantize an op other than
            a linear or matmul, or a GEMM output; or the model has a batch
            norm or a fused attention dropout.
        ValueError: A spec is per-channel, or ``model`` is in eval mode.
    """
    model.training_quantizers = quantizers = TrainingQuantizers()

    def make_transform(export):
        quantizer.annotate(export)
        for node in export.graph.nodes:
            annotation = node.meta.pop("quantization_annotation", None)
            if (
                annotation is not None
                and node.target not in _OPS
                and any(
                    spec is not None
                    for spec in annotation.input_qspec_map.values()
                )
            ):
                raise NotImplementedError(
                    f"Quantized training does not support {node.target} yet"
                )

        def transform(joint):
            errors = _tag_operands(joint)
            quantizer.annotate(joint)
            for node in joint.graph.nodes:
                annotation = node.meta.get("quantization_annotation")
                if annotation is None:
                    continue
                if annotation.output_qspec is not None:
                    raise NotImplementedError(
                        "GEMM outputs are not quantized in quantized training"
                    )
                if any(
                    getattr(spec, "qscheme", None)
                    == QScheme.PER_CHANNEL_SYMMETRIC
                    for spec in annotation.input_qspec_map.values()
                ):
                    raise ValueError(
                        "Per-channel specs have no contraction axis to "
                        "quantize along; use per-tensor or block specs"
                    )
            _share_quantized_transposes(joint)
            torchao_prepare._get_obs_or_fq_map = _get_obs_or_fq_map
            with unset_fake_temporarily():
                prepared = torchao_prepare.prepare(joint, {}, True)
            graph = prepared.graph
            for node in list(graph.nodes):
                if node.op != "call_module":
                    continue
                (operand,) = node.args
                fake_quant = prepared.get_submodule(node.target)
                with unset_fake_temporarily():
                    fake_quant.to(operand.meta["val"].device)
                quantizers.add_module(node.target, fake_quant)
                if operand in errors:
                    quantizers.gradients.add(node.target)
                with graph.inserting_before(node):
                    call = graph.call_function(
                        _fake_quantize, (graph.get_attr(node.target), operand)
                    )
                call.meta["val"] = operand.meta["val"]
                node.replace_all_uses_with(call)
                graph.erase_node(node)
            prepared.recompile()
            return prepared

        return transform

    try:
        capture_training(
            model,
            example_args,
            example_kwargs,
            dynamic_shapes,
            make_transform,
            hidden=("training_quantizers",),
        )
    except Exception:
        del model.training_quantizers
        raise
    return model


def _parameter_name(target: str) -> str:
    """The graph name of the parameter at module path ``target``."""
    return target.replace(".", "_")


def _delayed_scale(
    graph: Graph,
    operand: Node,
    history: Node,
    scale: Node,
    fake_quant: FusedAmaxObsFakeQuantize,
) -> Node:
    """Build the scale update a delayed-scaling fake-quant makes per call.

    The new scale is the largest amax in ``history`` over ``quant_max``,
    or the old ``scale`` when that amax is zero or not finite; ``operand``'s
    own amax then joins the history in place of its oldest.  Both state
    tensors are written in place.

    Args:
        graph: Graph to build in, at its insertion point.
        operand: The tensor the fake-quant quantizes.
        history: ``fake_quant``'s amax history.
        scale: ``fake_quant``'s scale.
        fake_quant: The per-tensor fake-quant being lowered.

    Returns:
        The new scale.
    """
    call = graph.call_function
    magnitude = call(aten.abs.default, (operand,))
    newest = call(aten.amax.default, (magnitude, []))
    amax = call(aten.amax.default, (history, [0]))
    # The update rolls the history back by one and overwrites its first.
    length = fake_quant.amax_history_len
    entries = [call(aten.view.default, (newest, [1]))]
    if length > 1:
        entries.append(call(aten.slice.Tensor, (history, 0, 2, length)))
        entries.append(call(aten.slice.Tensor, (history, 0, 0, 1)))
    updated = call(aten.cat.default, (entries,))
    call(aten.copy_.default, (history, updated))

    new = call(aten.div.Tensor, (amax, fake_quant.quant_max))
    positive = call(aten.gt.Scalar, (amax, 0.0))
    new = call(aten.where.self, (positive, new, scale))
    # An amax is never negative, so below infinity is finite.
    finite = call(aten.lt.Scalar, (amax, float("inf")))
    new = call(aten.where.self, (finite, new, scale))
    if fake_quant.power_2_scale:
        exponent = call(aten.log2.default, (new,))
        exponent = call(aten.ceil.default, (exponent,))
        new = call(aten.pow.Scalar, (2.0, exponent))
    call(aten.copy_.default, (scale, new))
    return new


def _dequantize_gemms(graph: Graph, scales: Dict[Node, Node]) -> None:
    """Scale each GEMM of per-tensor quantized operands back after it.

    The GEMM reads the quantized values themselves, so its result is scaled
    by the product of its operands' scales, and its bias, which is not in
    that domain, is added after.

    Args:
        graph: Graph rewritten in place.
        scales: Each per-tensor ``quantize`` -> the scale it divides by.

    Raises:
        NotImplementedError: A quantized tensor is read by an op other than
            a GEMM.
    """
    for quantized in scales:
        stack = list(quantized.users)
        while stack:
            user = stack.pop()
            if user.target in _VIEWS:
                stack.extend(user.users)
            elif user.target not in _OPS:
                raise NotImplementedError(
                    f"{user.target} reads the quantized {quantized.name}"
                )
    for gemm in list(graph.nodes):
        if gemm.target not in _OPS:
            continue
        found = []
        for operand in gemm.args[:2]:
            while operand.target in _VIEWS:
                operand = operand.args[0]
            if operand in scales:
                found.append(scales[operand])
        if not found:
            continue
        bias = gemm.args[2] if len(gemm.args) > 2 else None
        gemm.args = gemm.args[:2]
        users = list(gemm.users)
        with graph.inserting_before(gemm.next):
            product = found[0]
            for other in found[1:]:
                product = graph.call_function(aten.mul.Tensor, (product, other))
            result = graph.call_function(
                torch.ops.quantized_ops.dequantize.default, (gemm, product)
            )
            if bias is not None:
                result = graph.call_function(aten.add.Tensor, (result, bias))
        for user in users:
            user.replace_input_with(gemm, result)


def _lower_fake_quants(program: GraphModule, names: Dict[str, str]) -> None:
    """Replace each fake-quant of ``program`` with the ops that quantize.

    A microscaling fake-quant becomes a ``quantize_mx`` and the GEMMs
    reading it their ``_mx`` twins, as ``convert_pt2e`` lowers an
    activation's.  A weight is a program input that changes every step, so
    it is quantized in the graph like an activation rather than baked.

    A per-tensor fake-quant scales by its delayed-scaling state: its amax
    history and scale become program inputs, named after the fake-quant's
    buffers in the model, that the program updates as the fake-quant does
    (``_delayed_scale``).  Its GEMMs read the quantized values and are
    scaled back after (``_dequantize_gemms``).  A bias fake-quant, derived
    from its GEMM's operand scales as they stand where it runs, rounds the
    bias as the fake-quant does.

    Args:
        program: A gradient program, rewritten in place.
        names: Each fake-quant's attribute in ``program`` -> its module path
            in the model.

    Raises:
        NotImplementedError: A fake-quant's scheme or random Hadamard
            transform has no lowering yet, or a per-tensor quantized tensor
            is read by an op other than a GEMM.
    """
    graph = program.graph
    calls = [n for n in graph.nodes if n.target is _fake_quantize]
    attributes = {id(m): name for name, m in program.named_children()}

    # The scale each per-tensor fake-quant holds where the graph now stands:
    # its state until it runs, the scale it computes after.
    current = {}
    states = {}
    # ``prepare`` puts a parameter's fake-quant right after it; moving the
    # inputs ahead of every op lets the state inputs follow them.
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    first = next(n for n in graph.nodes if n.op != "placeholder")
    for placeholder in placeholders:
        first.prepend(placeholder)
    last = placeholders[-1]
    for node in calls:
        target = node.args[0].target
        fake_quant = program.get_submodule(target)
        if not isinstance(fake_quant, FusedAmaxObsFakeQuantize):
            continue
        state = []
        for buffer in ("amax_history", "scale"):
            with graph.inserting_after(last):
                last = graph.placeholder(
                    _parameter_name(f"{names[target]}.{buffer}")
                )
            last.meta["val"] = torch.empty_like(getattr(fake_quant, buffer))
            state.append(last)
        states[target] = state
        current[target] = state[1]

    scales = {}
    for node in calls:
        fake_quant_attr, operand = node.args
        target = fake_quant_attr.target
        fake_quant = program.get_submodule(target)
        if not fake_quant.fake_quant_on():
            node.replace_all_uses_with(operand)
            graph.erase_node(node)
            continue
        if (
            isinstance(fake_quant, RandomHadamardTransform)
            and fake_quant.rht_axis is not None
        ):
            raise NotImplementedError(
                "The random Hadamard transform has no lowering yet"
            )
        if isinstance(fake_quant, MXFakeQuantize):
            # The lowering replaces the call_module form ``prepare`` leaves.
            with graph.inserting_before(node):
                call = graph.call_module(target, (operand,))
            call.meta = node.meta
            node.replace_all_uses_with(call)
            graph.erase_node(node)
            modules = dict(program.named_modules(remove_duplicate=False))
            _replace_observer_with_quantize_mx_node_decomposed(
                program, call, modules
            )
            continue
        if isinstance(fake_quant, FusedAmaxObsFakeQuantize):
            if fake_quant.is_per_channel:
                raise NotImplementedError(
                    "Per-channel scales have no lowering for training"
                )
            with graph.inserting_before(node):
                if fake_quant.observer_on():
                    current[target] = _delayed_scale(
                        graph, operand, *states[target], fake_quant
                    )
        elif isinstance(fake_quant, _DerivedObserverOrFakeQuantize):
            act, weight = (
                current[attributes[id(m)]] for m in fake_quant.obs_or_fqs
            )
            with graph.inserting_before(node):
                weight = graph.call_function(aten.flatten.using_ints, (weight,))
                current[target] = graph.call_function(
                    aten.mul.Tensor, (act, weight)
                )
        else:
            raise NotImplementedError(
                f"{type(fake_quant).__name__} has no lowering for training"
            )
        with graph.inserting_before(node):
            # The fake-quant scales in its operand's dtype.
            scale = graph.call_function(
                aten._to_copy.default,
                (current[target],),
                {"dtype": operand.meta["val"].dtype},
            )
            qmap = create_getattr_from_value(
                program, graph, "qmap", fake_quant.qmap
            )
            quantized = graph.call_function(
                torch.ops.quantized_ops.quantize.default,
                (operand, scale, None, None, None, qmap),
            )
            quantized.meta["dtype"] = fake_quant.dtype
            if isinstance(fake_quant, _DerivedObserverOrFakeQuantize):
                # A bias is added in its own dtype, rounded as the fake-quant
                # rounds it.
                quantized = graph.call_function(
                    torch.ops.quantized_ops.dequantize.default,
                    (quantized, scale),
                )
            else:
                scales[quantized] = scale
        node.replace_all_uses_with(quantized)
        graph.erase_node(node)
    _dequantize_gemms(graph, scales)
    graph.eliminate_dead_code()
    program.delete_all_unused_submodules()


def gradient_program(model: nn.Module) -> GraphModule:
    """The forward, the loss and the backward of a training step, as a graph.

    Built from the joint graph ``capture_training`` captured from ``model``,
    whose forward returns only the loss.  The loss's incoming gradient is 1,
    and each parameter's gradient is written in place into a buffer of its
    own.  A parameter is named after its module path (``fc1.weight`` becomes
    ``fc1_weight``), a tied one after its first, and its gradient buffer
    after it, with ``_grad``.  The fake-quants of a model
    ``prepare_training`` prepared become the ops that quantize.

    Args:
        model: Model captured by ``capture_training``.

    Returns:
        A graph taking the parameters, the model's inputs and one gradient
        buffer per trainable parameter, in that order, and returning the loss.

    Raises:
        ValueError: ``model``'s forward returns more than the loss.
    """
    joint = model.forward.graphs["joint"]
    paths = {id(m): name for name, m in model.named_modules()}
    names = {}
    for node in joint.graph.find_nodes(
        op="call_function", target=_fake_quantize
    ):
        fake_quant_attr, operand = node.args
        module = joint.get_submodule(fake_quant_attr.target)
        names[fake_quant_attr.target] = paths[id(module)]
        if (
            isinstance(module, FusedAmaxObsFakeQuantize)
            and module.amax_history.numel() == 0
        ):
            # As the fake-quant's first call starts it.
            module.start_history((), operand.meta["val"].dtype)
    program = copy.deepcopy(joint)
    graph = program.graph
    # The joint graph takes its inputs as two lists, primals and tangents.
    graph._codegen = CodeGen()
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    tangents = [
        n for n in placeholders if isinstance(n.meta["desc"], TangentAOTInput)
    ]
    if len(tangents) != 1:
        raise ValueError("The model's forward must return only the loss")
    # A tensor under several names, such as tied embeddings, takes the first,
    # as ``named_parameters`` and so ``update_program`` name it.
    tensors = [
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ]
    first = {}
    for name, tensor in tensors:
        first.setdefault(tensor, name)
    canonical = {name: first[tensor] for name, tensor in tensors}
    named = {}
    for node in placeholders:
        desc = node.meta["desc"]
        if not isinstance(desc, (ParamAOTInput, BufferAOTInput)):
            continue
        name = _parameter_name(canonical[desc.target])
        if name in named:
            # The joint graph takes a tied tensor once per name.
            node.replace_all_uses_with(named[name])
            graph.erase_node(node)
        else:
            named[name] = node
            node._rename(name)
            node.target = node.name

    (tangent,) = tangents
    last = placeholders[-1]
    with graph.inserting_after(last):
        one = graph.call_function(
            aten.full.default,
            ([], 1.0),
            {"dtype": tangent.meta["val"].dtype},
        )
    one.meta["val"] = tangent.meta["val"]
    tangent.replace_all_uses_with(one)
    graph.erase_node(tangent)

    output = graph.output_node()
    loss = []
    grads = {}
    for value, desc in zip(output.args[0], output.meta["desc"]):
        if value is None:
            continue
        if not isinstance(desc, GradAOTOutput):
            loss.append(value)
        else:
            name = _parameter_name(canonical[desc.grad_of.target])
            grads.setdefault(name, []).append(value)
    for name, values in grads.items():
        with graph.inserting_before(one):
            buffer = graph.placeholder(f"{name}_grad")
        # A new placeholder's name is snake-cased; keep the parameter's.
        buffer._rename(f"{name}_grad")
        buffer.target = buffer.name
        buffer.meta["val"] = values[0].meta["val"]
        # A tied tensor's gradient sums the ones through each of its names.
        total = values[0]
        with graph.inserting_before(output):
            for value in values[1:]:
                total = graph.call_function(aten.add.Tensor, (total, value))
                total.meta["val"] = values[0].meta["val"]
            graph.call_function(aten.copy_.default, (buffer, total))
    output.args = (tuple(loss),)
    graph.eliminate_dead_code()
    _lower_fake_quants(program, names)
    program.recompile()
    return program


def update_program(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> GraphModule:
    """``optimizer``'s update of ``model``'s parameters, as a graph.

    Captured from ``optimizer.step`` itself, so it is whatever update the
    optimizer makes.  An optimizer of the same class and groups steps meta
    copies of the parameters twice under dynamo: the first step creates its
    state, and the second step's graph is the program.  It is capturable
    where the optimizer offers that, so its step count and other scalars
    stay tensors instead of Python numbers.  Every tensor that step reads
    is an input -- the parameters, their gradients, the optimizer's state
    and each group's learning rate, a 0-d tensor so a schedule can change
    it -- and every tensor it updates is written in place.

    Args:
        model: Model whose parameters ``optimizer`` updates.
        optimizer: The optimizer training ``model``.

    Returns:
        A graph whose inputs are named after the parameters as
        ``gradient_program`` names them: ``fc1_weight``, its gradient
        ``fc1_weight_grad`` and each state tensor under its key, such as
        ``fc1_weight_exp_avg``; each group's learning rate is ``lr_<i>``.
    """
    names = {p: _parameter_name(n) for n, p in model.named_parameters()}
    inputs = {}
    groups = []
    for i, group in enumerate(optimizer.param_groups):
        params = []
        for param in group["params"]:
            meta_param = nn.Parameter(
                torch.empty_like(param, device="meta"), param.requires_grad
            )
            inputs[meta_param] = names[param]
            if param.requires_grad:
                meta_param.grad = torch.empty_like(meta_param)
                inputs[meta_param.grad] = f"{names[param]}_grad"
            params.append(meta_param)
        lr = torch.tensor(float(group["lr"]))
        inputs[lr] = f"lr_{i}"
        groups.append({**group, "params": params, "lr": lr})
        if "capturable" in group:
            groups[-1]["capturable"] = True
    stand_in = type(optimizer)(groups)

    captured = []

    def backend(graph, example_inputs):
        attributes = dict(graph.named_parameters())
        attributes.update(graph.named_buffers())

        def keep(program, _):
            captured.append((program, attributes, example_inputs))
            return program

        return aot_autograd(
            fw_compiler=keep, keep_inference_input_mutations=True
        )(graph, example_inputs)

    step = torch.compile(stand_in.step, backend=backend)
    step()
    first = len(captured)
    step()
    # The first step creates the optimizer's state; the second retraces
    # unless that left the step unchanged.
    ((program, attributes, example_inputs),) = captured[first:] or captured
    # Named after the compiled step, which may replace a state tensor, such
    # as moving a step count to the parameters' device.
    for meta_param, state in stand_in.state.items():
        for key, value in state.items():
            inputs[value] = f"{inputs[meta_param]}_{key}"
    for node in program.graph.nodes:
        if node.op != "placeholder":
            continue
        desc = node.meta["desc"]
        tensor = (
            example_inputs[desc.idx]
            if isinstance(desc, PlainAOTInput)
            else attributes[desc.target]
        )
        node._rename(inputs[tensor])
        node.target = node.name
    # Dynamo writes an update as ``copy_(x, copy(x, value))``, where the
    # inner copy only gives ``value`` the shape and dtype it already has.
    graph = program.graph
    for node in list(graph.nodes):
        if node.target is not aten.copy.default:
            continue
        destination, value = (a.meta["val"] for a in node.args)
        if (destination.shape, destination.dtype) == (value.shape, value.dtype):
            node.replace_all_uses_with(node.args[1])
            graph.erase_node(node)
    # A tensor the step creates, such as ``where``'s scalar operand, goes on
    # the parameters' device, not the stand-in's.
    device = next(model.parameters()).device
    for node in graph.nodes:
        if node.kwargs.get("device") == torch.device("meta"):
            node.update_kwarg("device", device)
    program.recompile()
    return program


def disable_observers(model: nn.Module) -> None:
    """Fix every fake-quant's scale except the gradients', which keep moving.

    Args:
        model: Prepared graph, or a model prepared by ``prepare_training``.
    """
    for module in model.modules():
        if isinstance(module, FakeQuantizeBase):
            module.disable_observer()
    for module in model.modules():
        if isinstance(module, TrainingQuantizers):
            for name in module.gradients:
                getattr(module, name).enable_observer()


def prepare_from_args(
    model: nn.Module,
    args,
    example_args: Tuple[Any, ...] = (),
    example_kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes=None,
    quantizer: Optional[XNNPACKQuantizer] = None,
) -> nn.Module:
    """Prepare ``model`` for training as the quantization flags ask.

    The flags are the ones ``add_quantization_args`` defines.  With
    ``--error``, ``prepare_training`` quantizes the forward and backward
    passes.  Without it the model is exported in training mode and
    prepared for QAT: each conv-BN pair trains through the simulated fold,
    and ``train()`` and ``eval()`` switch the graph's dropouts and batch
    norms.

    Args:
        model: Float model.
        args: Parsed command-line arguments.
        example_args: Positional inputs to export with.
        example_kwargs: Keyword inputs to export with.
        dynamic_shapes: Dynamic dimensions of the inputs.
        quantizer: Quantizer to prepare with instead of the one the flags
            build, e.g. one that leaves some GEMMs unquantized.

    Returns:
        The prepared model: ``model`` itself with ``--error``, otherwise
        the prepared graph.
    """
    if args.bf16:
        model.bfloat16()
    if quantizer is None:
        quantizer = get_default_quantizer(
            input_activation=args.activation,
            weight=args.weight,
            bias=args.bias,
            error=args.error,
            random_hadamard_transform=args.random_hadamard_transform,
        )
    if args.error is not None:
        return prepare_training(
            model.train(),
            quantizer,
            example_args,
            example_kwargs,
            dynamic_shapes,
        )
    exported = export_model(
        model.train(),
        example_args,
        example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    model = prepare_qat_pt2e(exported, quantizer)

    def train(self, mode=True):
        set_training(self, mode)
        return self

    model.train = types.MethodType(train, model)
    model.eval = types.MethodType(nn.Module.eval, model)
    return model
