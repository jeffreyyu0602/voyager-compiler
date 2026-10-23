# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Rewriting the reference graph into the operations Alpha executes.

A rewrite here is not an optimization. Each one replaces a form the endpoint
has no instruction for with the single operation it *does* -- and the test of
one is not that the numbers still agree, but that the record it produces is a
record the converter will emit. `operators.py` says which those are.

The executable reference needs one representation-only movement rewrite too:
CPU ATen cannot run nearest-neighbour upsample on the signed-INT8 container
Alpha stores.  The target operation keeps the same record name and arithmetic
while providing an exact unsigned-byte execution bridge.
"""

import copy
import logging
import operator
from typing import Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.quantization.contracts import VSQ_SCALE_ENDPOINTS
from voyager_compiler.shape_prop import propagate_shape
from voyager_compiler.codegen.node_info import (
    get_arg_value,
    get_hardware_logical_shape,
    inherit_hardware_logical_shape,
    is_hardware_channel_pad,
    node_operand_bits,
)
from voyager_compiler.codegen.targets.magnet_alpha.traits import (
    NON_ALLOCATING_OPERAND_META,
    prepared_domain_rules,
)
from voyager_compiler.codegen.targets.magnet_alpha.hardware import (
    VSQ_BLOCK_VALUES,
)
from voyager_compiler.codegen.targets.magnet_alpha.ops import (
    ACCUMULATOR_DTYPE,
    ACTIVATION_DTYPE,
)
from voyager_compiler.codegen.targets.magnet_alpha.fixed_point import (
    ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    AlphaOutputDomain,
    rebalance_alpha_vsq_coarse_scale,
    encode_alpha_nmp_pipeline,
    encode_alpha_pe_pipeline,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HEAD_META",
    "INT8_HEAD",
    "VSQ_HEAD",
    "apply_graph_rewrites",
    "apply_memory_rewrites",
    "insert_shuffle_copies",
    "rewrite_channel_shuffles",
    "rewrite_integer_upsamples",
    "rewrite_pe_convolutions",
]

_CAT = torch.ops.aten.cat.default
_VIEW = torch.ops.aten.view.default
_RESHAPE = torch.ops.aten.reshape.default
_TRANSPOSE = torch.ops.aten.transpose.int
_CONTIGUOUS = torch.ops.aten.contiguous.default
_SHUFFLE = torch.ops.magnet_alpha.channel_shuffle.default
_SHUFFLE_COPY = torch.ops.magnet_alpha.channel_shuffle_copy.default
_UPSAMPLE = torch.ops.aten.upsample_nearest2d.vec
_ALPHA_UPSAMPLE = torch.ops.magnet_alpha.upsample_nearest2d.default
_ALPHA_META = "magnet_alpha"
_PHYSICAL_COPY_META = "physical_copy"
_SHUFFLE_KIND = "channel_shuffle"
_SHUFFLE_COPY_KIND = "channel_shuffle_copy"
_CONV2D = torch.ops.aten.conv2d.default
_CONV2D_MX = torch.ops.quantized_ops.conv2d_mx.default

#: What can head a matrix call. The two differ in how their operands are
#: scaled and in nothing else: the schemas share their first seven positions
#: exactly -- ``(input, weight, bias, stride, padding, dilation, groups)`` --
#: and ``conv2d_mx`` adds only keyword-only arguments after them, so every
#: ``get_arg_value(conv, N, ...)`` below reads the same operand from either.
_MATRIX_SOURCES = (_CONV2D, _CONV2D_MX)
_RELU = torch.ops.aten.relu.default
_QUANTIZE = torch.ops.quantized_ops.quantize.default
_DEQUANTIZE = torch.ops.quantized_ops.dequantize.default
_ACCUMULATE = torch.ops.magnet_alpha.conv2d_int8_accumulator.default
_ACCUMULATE_VSQ = torch.ops.magnet_alpha.conv2d_vsq.default
_EPILOGUE = torch.ops.magnet_alpha.ppu_fixed_relu.default
_NMP_EPILOGUE = torch.ops.magnet_alpha.nmp_fixed_requantize.default
_PVS = torch.ops.magnet_alpha.pvs_quantize.default

#: The output width that is per-vector scaled. A 4-bit output has no single
#: scale the epilogue could normalize into -- each block carries its own -- so
#: the record ends in a ``pvs_quantize`` that produces the payload and that
#: scale together.
#:
#: The endpoint's converter says the same thing in both directions, and
#: loudly: a 4-bit near-memory record without a PVS quantizer and companion
#: scale is refused as ``missing-explicit-pvs-output``, and an 8-bit record
#: *with* one is refused as ``unexpected-o8-companion-scale``. So the width
#: decides the sequence, and neither may be assumed.
PVS_OUTPUT_BITS = 4

#: The widths Alpha writes. Stated so a boundary asking for anything else is
#: declined and *counted*, rather than raising from inside the requantization
#: encoder where the message names a dataclass and not the layer.
OUTPUT_PRECISIONS = frozenset({4, 8})

#: What can feed the accumulator: a quantize, or an epilogue this pass has
#: *already* produced.
#:
#: The second is not a nicety. Rewriting is sequential over a chain of
#: convolutions, and the moment boundary N is lowered, boundary N+1's
#: activation stops being a ``quantize`` and becomes the epilogue just built.
#: Accepting only a quantize therefore declines roughly every second
#: convolution -- silently, since a graph with half its boundaries lowered
#: still compiles. On the reference workload it converted 62 of 85 and the
#: missing 23 looked like an unrelated pattern gap.
#:
#: It is also simply true: an Alpha epilogue emits the same int8 codes a
#: quantize does, which is what the accumulator reads.
#:
#: ``pvs_quantize`` is here for exactly the same reason, and omitting it cost
#: the same way: it is the stage that produces a 4-bit output, so the moment a
#: per-vector scaled boundary is lowered, the *next* boundary's activation is
#: a ``pvs_quantize`` rather than a ``quantize``. Leaving it out declined 22
#: of 85 convolutions -- and declined them silently, since a graph with
#: three-quarters of its boundaries lowered still compiles.
_QUANTIZED_SOURCES = (_QUANTIZE, _EPILOGUE, _NMP_EPILOGUE, _PVS)

#: ``quantize_vsq`` returns ``(coarse_scale, fine_scale, payload)`` and only
#: the third is the codes the array reads. Accepting a ``getitem`` blind to
#: its index would hand the accumulator a *scale* as its operand -- a graph
#: that type-checks, converts, and computes something unrelated -- so the
#: index is named rather than inferred.
_QUANTIZE_VSQ = torch.ops.quantized_ops.quantize_vsq.default
_VSQ_COARSE_INDEX = 0
_VSQ_FINE_INDEX = 1
_VSQ_PAYLOAD_INDEX = 2

#: The two operators that return a vector-scaled value as one tuple, in the
#: same order. ``pvs_quantize`` is Alpha's own stage and it *replaces*
#: ``quantize_vsq`` in the graph rather than following it, which is why the
#: indices above have to mean the same thing for both: a consumer's
#: ``getitem`` is left exactly where it was.
_VSQ_TUPLE_SOURCES = (_QUANTIZE_VSQ, _PVS)

#: Movement the accumulator must not look through, whatever the domain rules
#: say about it. A pad preserves the encoding and still changes *which tensor*
#: the array reads, so accepting the wrong one declares a channel extent the
#: converter checks against.
#:
#: The compiler's *own* channel pad is the exception, and it is the exception
#: because it is not a different tensor: widening an operand to the extent the
#: datapath holds is how the resident form is materialized, and the resident
#: form is precisely what the array reads. A reference compile agrees --
#: ``model_0_stem_1_conv`` names a 16-channel operand for a layer whose
#: logical input is 9. ``is_hardware_channel_pad`` is what says so, and it
#: asks for both halves: that the padding pass wrote the pad, and that the pad
#: is a trailing channel one rather than a bias or a spatial tail.
_NOT_THE_OPERAND = (torch.ops.aten.pad.default,)

#: The reshaping operators the source form may use interchangeably.
_REGROUPING = (_VIEW, _RESHAPE)


def _producer(node: Node, targets) -> Optional[Node]:
    """The node feeding ``node``, when that node is one of ``targets``.

    Returns the *producer*, not the consumer -- named so because a helper that
    returned "the input" invited reading each step's result as the node just
    matched, which is an off-by-one that still walks a chain of the right
    length and then asks the wrong node for its axes.
    """
    if not isinstance(node, Node) or not node.args:
        return None
    operand = node.args[0]
    if not isinstance(operand, Node) or operand.op != "call_function":
        return None
    return operand if operand.target in targets else None


def _channel_shuffle_source(node: Node):
    """``(left, right, groups)`` if ``node`` ends a channel shuffle.

    The form, as the model writes it::

        out = cat((left, right), 1)
        x   = out.view(N, groups, C // groups, H, W)
        x   = x.transpose(1, 2).contiguous()
        x   = x.view(N, -1, H, W)          <- node

    Matched from the end backwards because the last ``view`` is what the rest
    of the graph reads: starting at the ``cat`` would find the pattern but
    leave the question of which node to replace.

    ``groups`` is read off the five-dimensional regrouping rather than assumed
    to be two. The model only ever uses two, and a rewrite that hard-coded it
    would silently mis-shuffle the first graph that does not.
    """
    if node.op != "call_function" or node.target not in _REGROUPING:
        return None
    contiguous = _producer(node, (_CONTIGUOUS,))
    if contiguous is None:
        return None
    transpose = _producer(contiguous, (_TRANSPOSE,))
    if transpose is None:
        return None
    # It must be the transpose that crosses group with channel; any other
    # pair is a differently shaped operation that happens to look like this.
    if tuple(transpose.args[1:3]) != (1, 2):
        return None
    regrouped = _producer(transpose, _REGROUPING)
    if regrouped is None:
        return None
    shape = regrouped.args[1] if len(regrouped.args) > 1 else None
    if not isinstance(shape, (list, tuple)) or len(shape) != 5:
        return None
    joined = _producer(regrouped, (_CAT,))
    if joined is None:
        return None
    branches = joined.args[0]
    if not isinstance(branches, (list, tuple)) or len(branches) != 2:
        return None
    if not all(isinstance(branch, Node) for branch in branches):
        return None
    # Concatenated along the channel dimension, or it is not this pattern.
    axis = joined.args[1] if len(joined.args) > 1 else 0
    if axis != 1:
        return None
    groups = shape[1]
    if not isinstance(groups, int) or groups < 1:
        return None
    return branches[0], branches[1], groups


def rewrite_channel_shuffles(model: GraphModule) -> int:
    """Replace each channel-shuffle chain with the operation Alpha issues.

    Returns how many were rewritten, so a caller can tell "none present" from
    "none matched" -- on this workload those mean very different things.

    The ``cat`` is absorbed: the operator takes both branches, because on
    Alpha the join is where the branches are *written*, not a movement of its
    own. The following ``chunk``, if any, is left alone -- a reference compile
    emits it as its own record.
    """
    rewritten = 0
    for node in list(model.graph.nodes):
        matched = _channel_shuffle_source(node)
        if matched is None:
            continue
        left, right, groups = matched
        node_name = _channel_shuffle_name(node)
        with model.graph.inserting_before(node):
            shuffled = model.graph.create_node(
                "call_function",
                _SHUFFLE,
                (left, right, groups),
                name=node_name,
            )
        shuffled.meta.update(node.meta)
        # This is a target declaration, not serializer scratch space.  The
        # terminal memory rewrite and storage policy validate it before they
        # turn the logical shuffle into two physical output backings.
        shuffled.meta[_ALPHA_META] = {
            "groups": groups,
            "kind": _SHUFFLE_KIND,
        }
        _adopt_value(shuffled, node)
        node.replace_all_uses_with(shuffled)
        rewritten += 1
        logger.debug(
            "channel shuffle at %s -> %s (groups=%d)",
            node.name,
            shuffled.name,
            groups,
        )
    if rewritten:
        model.graph.eliminate_dead_code()
        model.recompile()
    return rewritten


def _channel_shuffle_name(output: Node) -> str:
    """Preserve the model-scoped Alpha shuffle name when export recorded it.

    Alpha has one channel-shuffle address exception keyed by this stable model
    scope. The generated FX name is not an equivalent ABI: it loses the
    identity before serialization can apply that rule.
    """
    module_scope = None
    for path, module_type in output.meta.get("nn_module_stack", {}).values():
        if "ShuffleV2Block" in str(module_type):
            module_scope = path
    if module_scope is None:
        return f"{output.name}_shuffle"
    if module_scope.startswith("L['self']."):
        module_scope = module_scope[len("L['self'].") :]
    return f"{module_scope.replace('.', '_')}_shuffle"


def _static_tensor(model: GraphModule, node) -> Optional[torch.Tensor]:
    """The tensor a ``get_attr`` names, or ``None`` for anything else.

    A scale that is not a frozen attribute is a scale that can still change,
    and encoding one would bake in a value the graph may not keep.
    """
    if not isinstance(node, Node) or node.op != "get_attr":
        return None
    owner, _, attribute = str(node.target).rpartition(".")
    module = model.get_submodule(owner) if owner else model
    value = getattr(module, attribute, None)
    return value if isinstance(value, torch.Tensor) else None


def _sole_user(node: Node, target) -> Optional[Node]:
    """``node``'s only user when it is ``target``.

    *Only* matters: a value read twice cannot be folded into an epilogue that
    consumes it, because the second reader would lose its operand.
    """
    users = tuple(node.users)
    if len(users) != 1:
        return None
    return users[0] if users[0].target is target else None


def _adopt_value(new: Node, source: Node, dtype=None) -> None:
    """Give a freshly built node the traced value its predecessor had.

    ``ShapeProp`` runs at the *start* of ``transform``; these rewrites run at
    the end, and the adapter fuses before ``compile`` propagates again. So a
    node built here reaches operator fusion with no ``.value`` at all, and
    fusion reads ``.value`` -- ``AttributeError: 'Node' object has no
    attribute 'value'``, from a pass that has nothing to do with the mistake.

    Copying it is right rather than expedient: these rewrites do not change
    any shape. The accumulator differs from its convolution only in dtype,
    which is what ``dtype`` is for.
    """
    value = getattr(source, "value", None)
    if value is None:
        value = source.meta.get("val")
    if value is None or not hasattr(value, "to"):
        return
    adopted = value.to(dtype) if dtype is not None else value
    new.value = adopted
    new.meta.setdefault("val", adopted)


def _frozen_operand(model: GraphModule, name: str, value: torch.Tensor) -> Node:
    """Register ``value`` on the model and return a ``get_attr`` naming it.

    The encoded multiplier and bias are *vectors* -- one entry per output
    channel -- so they cannot ride as operator arguments the way a shift can.
    They become graph operands, which is also what lets the serializer find
    them: a reference record names them ``*_ppu_primary_multiplier`` and
    ``*_ppu_bias``, and those names are kept.

    They allocate nothing. A float parameter of this kind is folded into an
    instruction field rather than staged, which is why the storage plan
    neither loads nor allocates one -- see ``codegen/storage``.
    """
    attribute = name.replace(".", "_")
    model.register_buffer(attribute, value)
    # Placed just after the placeholders, not at the caller's insertion point.
    #
    # A constant has no inputs, so the top of the graph is always
    # topologically legal for it -- and the alternative is not: a caller that
    # builds one *outside* an ``inserting_before`` context gets it at the
    # graph's default point, which is the end, after the very node that reads
    # it. That is a use-before-definition the graph only reports at the next
    # ``lint()``, sometimes a pass later. It has happened twice.
    anchor = None
    for candidate in model.graph.nodes:
        if candidate.op == "placeholder":
            anchor = candidate
        else:
            break
    context = (
        model.graph.inserting_after(anchor)
        if anchor is not None
        else model.graph.inserting_before(None)
    )
    with context:
        node = model.graph.get_attr(attribute)
    node.meta["val"] = value
    # ``.value`` as well as ``meta['val']``, and they are not the same thing
    # to every reader: ``create_and_insert_subgraph`` clones ``n.value`` for
    # each operand it lifts into a submodule, and raises ``AttributeError``
    # on a node that has only the meta. A frozen operand is an operand like
    # any other the moment its consumer becomes one instruction.
    node.value = value
    # And say so, rather than leaving it to a docstring. The endpoint folds
    # this into the instruction that reads it, so it occupies no storage and
    # must not be listed among the model's parameters; the reference lists
    # none of them.
    node.meta[NON_ALLOCATING_OPERAND_META[0]] = True
    return node


def _is_vsq_payload(node) -> bool:
    """Whether ``node`` is the narrow payload a vector-scaled quantize made.

    Index-aware on purpose. ``quantize_vsq`` returns three tensors and two of
    them are scales; a ``getitem`` accepted blind would offer the accumulator
    a uint8 scale where its operand belongs, and every shape downstream would
    still check out.
    """
    return (
        getattr(node, "target", None) is operator.getitem
        and isinstance(node.args[0], Node)
        and node.args[0].target in _VSQ_TUPLE_SOURCES
        and node.args[1] == _VSQ_PAYLOAD_INDEX
    )


def _carries_quantized_codes(node, rules, seen=None) -> bool:
    """Whether ``node`` hands the accumulator the codes a boundary produced.

    An accumulator reads int8 codes. A ``quantize`` produces them, and so does
    an Alpha epilogue -- that much ``_QUANTIZED_SOURCES`` already says. What it
    does not say is that *moving* those codes leaves them codes: a
    concatenation of two quantized branches, a slice of one, a view of one, a
    zero pad of one, are all still the same encoding, and on this machine most
    of them are address operations rather than work.

    Which operations preserve a domain is **already declared**, once, as the
    quantization-domain rules the quantizer itself used to form its regions --
    ``ALIAS`` for a passthrough, ``JOIN`` for a concatenation whose "result is
    literally the inputs' bytes adjacent", ``MAP_COMPONENTS`` for a zero pad.
    So this walks that declaration rather than restating it as a list of
    targets here. A second list would be a second opinion about the same
    question, and the two would drift.

    It also means the scale question is already settled and must not be
    re-litigated: a ``JOIN`` *is* the statement that every input and the result
    share one domain, and regions are finalized before conversion. Were that
    not so, concatenating two differently-scaled tensors would produce codes
    with no single valid interpretation -- which is exactly the record this
    would otherwise build.

    ``_NOT_THE_OPERAND`` is the exception, and it is not a domain question at
    all. A zero pad *does* preserve the encoding -- that is exactly why the
    domain rules admit it -- but the accumulator needs a different fact: which
    tensor the array will actually read. Domain-preserving and
    operand-identical are two questions, and only the first is declared.

    Which makes the compiler's own channel pad the one pad to look through,
    for the same reason the others are refused. It widens an operand to the
    extent the datapath *holds*; the array reads exactly that, so the padded
    tensor is not a different tensor from the operand -- it is the operand,
    materialized. Refusing it costs the record entirely, and a reference
    compile emits one. ``is_hardware_channel_pad`` is the mark, and it asks
    two things: that the padding pass wrote this pad -- a model's own pad is
    the same operator with the same arguments and nothing in the node
    distinguishes them -- and that what it wrote is a trailing *channel* pad,
    since that pass also widens biases and vector-op tails and neither of
    those is the tensor the array reads.

    ``seen`` is a **memo**, not a visited set, and the difference is a bug.
    A set marks a node the moment the walk enters it and never unmarks it, so
    two siblings reading one value -- ``cat(slice(v), slice(v))`` above a
    quantize -- make the second branch find ``v`` already marked and call it a
    cycle. Recording the *verdict* instead answers both questions: a node
    still on the stack has no verdict yet and is a genuine cycle, and one that
    finished has an answer worth reusing rather than re-deriving per branch.
    """
    if not isinstance(node, Node):
        return False
    if node.target in _QUANTIZED_SOURCES or _is_vsq_payload(node):
        return True
    if node.target in _NOT_THE_OPERAND and not is_hardware_channel_pad(node):
        return False
    seen = {} if seen is None else seen
    if node in seen:
        # ``None`` is "still on this walk's stack" -- a real cycle, which is
        # not quantized codes. Anything else is a finished verdict.
        recorded = seen[node]
        return False if recorded is None else recorded
    seen[node] = None
    transfer = rules.transfer(node)
    if transfer is None:
        seen[node] = False
        return False
    operands = [n for n in transfer.inputs if isinstance(n, Node)]
    # Every operand, not any: a join whose branches are not all quantized
    # produces a tensor that is only partly codes.
    result = bool(operands) and all(
        _carries_quantized_codes(operand, rules, seen) for operand in operands
    )
    seen[node] = result
    return result


def rewrite_integer_upsamples(model: GraphModule) -> int:
    """Give physical INT8 nearest upsample an executable Alpha target op.

    This is deliberately narrower than an ATen legalization: only the
    ``.vec`` form used by Alpha's prepared movement rules is replaced, and
    only when the existing domain declarations prove its input is quantized
    codes whose stored width is exactly eight bits.  Float resize and INT4
    PVS payload movement retain their original operators.
    """
    rules = prepared_domain_rules()
    rewritten = 0
    for node in list(model.graph.nodes):
        if node.op != "call_function" or node.target is not _UPSAMPLE:
            continue
        source = get_arg_value(node, 0, "input")
        if not isinstance(source, Node):
            continue
        if not _carries_quantized_codes(source, rules):
            continue
        try:
            bits = node_operand_bits(source)
        except (TypeError, ValueError):
            continue
        if bits != 8:
            continue

        with model.graph.inserting_before(node):
            replacement = model.graph.call_function(
                _ALPHA_UPSAMPLE,
                node.args,
                node.kwargs,
            )
        replacement.meta = copy.copy(node.meta)
        replacement.meta["dtype"] = "int8"
        # Target instruction fusion consumes ``.value`` before the next whole
        # graph ShapeProp.  Evaluate this representation adapter immediately
        # so a downstream fused placeholder receives the physical INT8 value.
        propagate_shape(replacement)
        node.replace_all_uses_with(replacement)
        model.graph.erase_node(node)
        rewritten += 1

    if rewritten:
        model.graph.lint()
        model.recompile()
    return rewritten


def _quantizer_endpoints(quantizer) -> Tuple[float, float]:
    """The pair one ``quantize_vsq`` node split its value against.

    ``fine_quant_max`` at argument 6 and ``coarse_scale_quant_max`` at 10.
    Read positionally *and* by keyword, because a graph built by hand names
    them and one built by the converter does not.

    The coarse endpoint is defaulted twice on purpose: the argument may be
    absent, and it may be present and ``None``, which is what the converter
    writes when the observer did not carry one. ``get_arg_value``'s default
    only answers the first.
    """
    return (
        float(
            get_arg_value(
                quantizer, 6, "fine_quant_max",
                ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
            )
        ),
        float(
            get_arg_value(
                quantizer, 10, "coarse_scale_quant_max",
                ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
            )
            or ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT
        ),
    )


def _vsq_endpoints(node) -> Tuple[float, float]:
    """The fine/coarse endpoints one vector-scaled value was split against.

    ``(sqmax, csqmax)`` -- the two numbers that fix the rebalance between the
    stored mantissa and the coarse scale, and therefore the only two a target
    needs to undo the quantizer's split.

    **Read from the operator, not from metadata.** ``quantize_vsq`` takes both
    as arguments -- ``fine_quant_max`` at 6 and ``coarse_scale_quant_max`` at
    10 -- so an activation's are already in the graph as call arguments, which
    survive every transformation a meta key does not. This module briefly
    carried its own meta key for them before that was noticed; the operator
    was already saying it.

    A *weight* has no quantizer node at all: the converter knew the parameter
    and precomputed all three components into ``get_attr``s, erasing the only
    thing that recorded the split. So the converter writes the pair onto the
    coarse scale it creates, and that is read second.

    The declared defaults are the last resort and not a safe one. They are
    right for a pairing that uses them -- this workload's weight side declares
    exactly 7.0/127.0 -- and *silently* wrong for one that does not: the
    endpoints fix the rebalance, so a wrong pair scales every channel by a
    constant and nothing downstream can tell. Hence the warning: reaching here
    means neither the operator nor the converter said, and the answer is a
    guess that happens to be usually right.
    """
    if node is None:
        # No operand, so no split to describe. That is not the case the
        # warning below is about -- it is about an operand that *is* vector
        # scaled and whose endpoints nothing recorded, which is a real defect
        # and a silent one. Warning here as well drowns it: this fires once
        # per non-vector-scaled operand on every compile.
        return (
            ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
            ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
        )
    recorded = getattr(node, "meta", {}).get(VSQ_SCALE_ENDPOINTS)
    if recorded:
        return float(recorded[0]), float(recorded[1])
    if _is_vsq_coarse_scale(node):
        return _quantizer_endpoints(node.args[0])
    if _is_alpha_coarse_scale(node):
        recorded = get_arg_value(node.args[0], 1, "coarse_scale")
        pair = getattr(recorded, "meta", {}).get(VSQ_SCALE_ENDPOINTS)
        if pair:
            return float(pair[0]), float(pair[1])
    logger.warning(
        "%s: no VSQ endpoints on the operand or its quantizer; assuming the "
        "machine's declared %s/%s, which is a guess the collateral cannot "
        "record",
        getattr(node, "name", node),
        ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
        ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    )
    return (
        ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
        ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    )


def _coarse_scale_tensor(model, node):
    """One coarse scale's value, whichever shape the converter left it in.

    A *weight*'s coarse scale is a ``get_attr``: the parameter was known at
    conversion, so all three of its VSQ components were computed and stored.
    An *activation*'s is not -- it is ``getitem(quantize_vsq, 0)``, because
    the payload is produced at runtime while the coarse scale it was
    calibrated against is a constant the operator was handed. Both are the
    same number; only one of them is a node holding it.

    Resolved through the ``getitem`` by index rather than by taking whatever
    the operator's first argument happens to be: the two other outputs are the
    fine scale and the payload, and folding either into an accumulator scale
    would be a silent factor of the wrong size.
    """
    if _is_vsq_coarse_scale(node):
        return _static_tensor(
            model, get_arg_value(node.args[0], 1, "coarse_scale")
        )
    return _static_tensor(model, node)


def _is_vsq_coarse_scale(node) -> bool:
    """Whether ``node`` is a vector-scaled *quantize*'s coarse-scale output.

    The quantizer's, and deliberately not the per-vector stage's. Both return
    a coarse scale at index 0 and the two are different numbers -- the
    quantizer's is semantic, the stage's has the rebalance already divided
    out -- so a predicate covering both would let a caller rebalance a value
    that is already rebalanced. :func:`_is_alpha_coarse_scale` is the other
    one, and :func:`_physical_coarse_scale` is where the distinction is used.
    """
    return (
        getattr(node, "target", None) is operator.getitem
        and isinstance(node.args[0], Node)
        and node.args[0].target is _QUANTIZE_VSQ
        and node.args[1] == _VSQ_COARSE_INDEX
    )


def _is_alpha_coarse_scale(node) -> bool:
    """Whether ``node`` is a per-vector stage's coarse-scale output.

    Which is *already physical*: the stage was built with the rebalanced
    scale as its operand and hands it back unchanged.
    """
    return (
        getattr(node, "target", None) is operator.getitem
        and isinstance(node.args[0], Node)
        and node.args[0].target is _PVS
        and node.args[1] == _VSQ_COARSE_INDEX
    )


def _physical_coarse_scale(model, node):
    """One operand's coarse scale as Alpha stores it, or ``None``.

    "As Alpha stores it" means the rebalance is already divided out. Where
    that division happens depends on which stage produced the operand, and
    getting it wrong is a factor of ~16 on every channel with nothing
    downstream able to notice:

    * a **quantizer**'s coarse scale is semantic -- what the value is worth --
      so it is rebalanced here, under the endpoints that operand was split
      against;
    * a **per-vector stage**'s was rebalanced when the stage was built, and
      rebalancing it again would apply the factor twice.

    The second case is not hypothetical and is why this exists: once a
    producing boundary is lowered, the consumer's ``input_coarse_scale``
    stops being ``getitem(quantize_vsq, 0)`` and becomes
    ``getitem(pvs_quantize, 0)``. Reading only the first shape silently
    declined 28 of the 36 vector-scaled boundaries -- every one whose input
    came from a boundary lowered before it.
    """
    if _is_alpha_coarse_scale(node):
        return _static_tensor(model, get_arg_value(node.args[0], 1, "coarse_scale"))
    values = _coarse_scale_tensor(model, node)
    if values is None:
        return None
    fine_quant_max, coarse_scale_quant_max = _vsq_endpoints(node)
    return rebalance_alpha_vsq_coarse_scale(
        values.reshape(-1).to(torch.float64),
        fine_quant_max=fine_quant_max,
        coarse_scale_quant_max=coarse_scale_quant_max,
    )


def _ensure_traced_value(model, node):
    """Give an existing node the ``.value`` fusion will ask it for.

    ``create_and_insert_subgraph`` clones ``n.value`` for every operand it
    lifts into a fused submodule and raises ``AttributeError`` on a node that
    carries only ``meta['val']``. The two are not the same attribute and
    plenty of nodes have only the second -- ``create_getattr_from_value``,
    which is how the converter materializes every scale, sets the meta alone.

    That never mattered while those nodes were read by passes rather than by
    records. It matters the moment one becomes a record's operand, which is
    what carrying the semantic coarse scales did. Filling it in here rather
    than at each call site: the requirement belongs to "is an operand", not to
    any one operand.
    """
    if node is None or hasattr(node, "value"):
        return node
    value = node.meta.get("val")
    if value is None:
        value = _static_tensor(model, node)
    if value is not None:
        node.value = value
    return node


def _fine_scale_endpoints(scale, coarse):
    """The endpoints *this fine scale* was split against.

    Read from the operator that produced it wherever one exists, and only
    then from the operand's coarse scale. The order matters and getting it
    backwards is not a near miss:

    An activation's fine scale is ``getitem(quantize_vsq, 1)``, and that
    quantizer carries its own pair -- 8.0/126.0 for this workload's
    activations against 7.0/127.0 for its weights. Taking the pair from the
    sibling ``input_coarse_scale`` fails when that operand is absent, and the
    declared default is the *weight* one. *Measured:* an activation scale of
    10.067647 then decodes to 141.502 rather than an integer, because 163 is
    what it is under 8.0/126.0.

    The encoder refuses that rather than rounding it, which is the whole
    reason it refuses: 141 or 142 would both have "worked" and scaled every
    block by a constant nothing downstream could detect.
    """
    if (
        getattr(scale, "target", None) is operator.getitem
        and isinstance(scale.args[0], Node)
        and scale.args[0].target is _QUANTIZE_VSQ
        and scale.args[1] == _VSQ_FINE_INDEX
    ):
        return _quantizer_endpoints(scale.args[0])
    return _vsq_endpoints(coarse)


def _semantic_coarse_scale(model, node):
    """The node holding one operand's coarse scale *before* the rebalance.

    ``None`` when the graph no longer has one, which is the ordinary case for
    an activation whose producer has been lowered: the boundary's coarse
    scale is then the per-vector stage's, already physical, and the raw
    parameter that fed the quantizer is dead.

    Absent rather than substituted. Handing the *physical* scale over under a
    name that says ``semantic`` would be a field that reads correctly and
    means the other thing -- the same mistake as labelling a float operand
    ``uint8``. The converter does not read these, so an absent one costs
    nothing; a wrong one would cost a reader believing it.
    """
    if _is_vsq_coarse_scale(node):
        return _ensure_traced_value(
            model, get_arg_value(node.args[0], 1, "coarse_scale")
        )
    if _is_alpha_coarse_scale(node):
        # The per-vector stage kept a reference to the quantizer's own coarse
        # scale when it replaced it. Naming that node here is what keeps it
        # alive -- it has no other reader once the quantizer is gone.
        operand = get_arg_value(node.args[0], 1, "coarse_scale")
        source = getattr(operand, "meta", {}).get(SEMANTIC_COARSE_SCALE_META)
        return _ensure_traced_value(model, source) if source else None
    if getattr(node, "op", None) == "get_attr":
        return _ensure_traced_value(model, node)
    return None


def _folded_coarse_scale(model, conv):
    """One vector-scaled call's accumulator scale, or ``None``.

    A vector-scaled convolution has no ``dequantize`` to read it from. The
    converter moved both scales onto the operation itself, so the value the
    accumulator counts in is the *product of the two coarse levels*, each with
    its rebalance divided out -- which is exactly the ``folded_coarse_scale``
    a reference record carries, and the identity
    ``encode_alpha_vsq_scales`` was checked against on all 36 of them.

    The fine scales do not appear here. They are already multiplied into the
    accumulator by the array, which is what ``vsq_operands=True`` undoes in
    the first requantization stage; folding them in twice would scale the
    result by 65536.

    **Computed in float64, deliberately, and not to match the fork.** The
    reference compile carries this out in float32: *measured*, its 36
    vector-scaled records are reproduced bit for bit by float32 arithmetic and
    by none of them in float64. That is a fact about the fork and not a
    specification. This value is an intermediate on the way to the integers
    the endpoint actually executes -- the primary multiplier and its shift --
    and bit-accuracy belongs *there*, at the encoders, where a real number
    becomes a hardware code. Reproducing another lowering's float32 rounding
    here would make the compiler less accurate in order to be more similar,
    and numerical deviation between lowerings is permitted.

    ``None`` rather than a guess when either operand is missing its scale or
    the endpoints it was split against: a rebalance derived from assumed
    endpoints is wrong by a constant factor on every channel, and a boundary
    that cannot be encoded is one this pass declines and the inventory counts.
    """
    folded = None
    for name in ("input_coarse_scale", "weight_coarse_scale"):
        rebalanced = _physical_coarse_scale(model, conv.kwargs.get(name))
        if rebalanced is None:
            return None
        folded = rebalanced if folded is None else folded * rebalanced
    return folded


def _pe_boundary(model: GraphModule, conv: Node, rules=None):
    """The pieces of one matrix call, or ``None``.

    The shape a converted graph actually has -- confirmed against a real
    compile rather than assumed::

        x_q  = quantize(x, input_scale, ..., qmap)
        conv = aten.conv2d(x_q, weight, bias, stride, padding, ...)
        deq  = dequantize(conv, accumulator_scale, ..., qmap)
        relu = aten.relu(deq)                       # optional
        out  = quantize(relu, output_scale, ..., qmap)

    The **accumulator scale is already in the graph**, as the dequantize's
    scale argument -- it is ``input_scale * weight_scale`` by construction.
    Recomputing it from the weight would duplicate a decision the quantizer
    already made, and the two could disagree.

    The weight stays a float ``get_attr``. It becomes int8 at serialization,
    which is where its own scale is recorded; a rewrite that quantized it here
    would be a second opinion about the same tensor.

    The output quantizer's own width comes back with the rest of it. It is
    part of what the boundary *is* -- a 4-bit output is per-vector scaled and
    takes a different record sequence from an 8-bit one -- and reading it here
    keeps the one place that decides what a boundary is from having a caller
    that decides part of it separately.
    """
    activation = get_arg_value(conv, 0, "input")
    if not isinstance(activation, Node):
        return None
    if rules is None:
        rules = prepared_domain_rules()
    if not _carries_quantized_codes(activation, rules):
        return None
    if conv.target is _CONV2D_MX:
        # The scales rode onto the operation, so the converter left no
        # dequantize behind to carry the accumulator's. It is the two coarse
        # levels, folded.
        dequantize = None
        accumulator_scale = _folded_coarse_scale(model, conv)
        source = conv
    else:
        dequantize = _sole_user(conv, _DEQUANTIZE)
        if dequantize is None:
            return None
        accumulator_scale = _static_tensor(
            model, get_arg_value(dequantize, 1, "scale")
        )
        source = dequantize
    if accumulator_scale is None:
        return None
    relu = _sole_user(source, _RELU)
    boundary = relu if relu is not None else source
    output = _sole_user(boundary, _QUANTIZE)
    if output is None:
        # A per-vector scaled output. The quantizer belongs to *this* record --
        # it is the stage that narrows the epilogue's result and emits the
        # companion scale beside it -- not to whatever reads the payload next.
        # The reference agrees: it lowers exactly this node as the producing
        # boundary's ``pvs_quantize`` member.
        output = _sole_user(boundary, _QUANTIZE_VSQ)
    if output is None:
        return None
    scale_argument = (
        "coarse_scale" if output.target is _QUANTIZE_VSQ else "scale"
    )
    output_scale = _static_tensor(
        model, get_arg_value(output, 1, scale_argument)
    )
    if output_scale is None:
        return None
    if output.target is _QUANTIZE_VSQ:
        # The quantizer's coarse scale is the *semantic* one -- what the value
        # is worth. Alpha does not store that: it stores an 8-bit mantissa
        # beside the payload and keeps the discarded range in the coarse
        # scale, so what the epilogue normalizes into, and what the per-vector
        # stage is programmed with, is the coarse scale with the rebalance
        # divided out.
        #
        # *Measured* on a reference compile: the epilogue's ``output_scale``
        # and the per-vector stage's ``coarse_scale`` are the same number on
        # all 38 records that have one, and that number is the semantic
        # coarse scale over this quantizer's own rebalance. Passing the
        # semantic scale instead is wrong by that factor -- 16.19 for the
        # activation endpoints this workload declares -- on every per-vector
        # boundary, and wrong silently: the record converts and runs.
        fine_quant_max, coarse_scale_quant_max = _quantizer_endpoints(output)
        output_scale = rebalance_alpha_vsq_coarse_scale(
            output_scale,
            fine_quant_max=fine_quant_max,
            coarse_scale_quant_max=coarse_scale_quant_max,
        )
    try:
        bits = node_operand_bits(output)
    except ValueError:
        # A quantizer that does not say what it produces is not a boundary
        # this pass can lower. Refusing counts it; assuming eight bits would
        # emit a record that converts, runs, and truncates.
        return None
    if bits not in OUTPUT_PRECISIONS:
        # Refused here rather than three calls deeper in the encoder, so it
        # arrives as a boundary this pass declined -- which the inventory
        # counts -- instead of an exception naming a dataclass. Said out loud
        # too: a declined boundary is one missing record, and a graph with a
        # missing record still compiles.
        logger.warning(
            "%s: declining a boundary whose output asks for %d bits; "
            "this machine writes %s",
            conv.name, bits, sorted(OUTPUT_PRECISIONS),
        )
        return None
    return (
        activation,
        dequantize,
        relu,
        output,
        accumulator_scale,
        output_scale,
        bits,
    )


#: Stamped on every accumulator this pass builds: which head the record should
#: carry. Read at serialization, where emitting the wrong one is a record that
#: converts and computes something else.
#: The quantizer's own coarse scale, kept beside the rebalanced one a
#: per-vector stage carries. Two different numbers for the same operand, and
#: a record names both -- ``semantic_*_coarse_scale`` is what the value is
#: worth, ``folded_coarse_scale`` is what the datapath applies.
SEMANTIC_COARSE_SCALE_META = "magnet_alpha_semantic_coarse_scale"

HEAD_META = "magnet_alpha_head"
VSQ_HEAD = "vsq"
INT8_HEAD = "int8"


def _conv_head(conv, weight=None, activation=None) -> str:
    """Which convolution head this boundary needs, from what it *is*.

    Structural, both ways round, and that is the whole point:

    * a vector-scaled convolution reaches this pass as
      ``quantized_ops.conv2d_mx``, because a plain ``aten.conv2d`` has nowhere
      to carry a fine scale;
    * a boundary reading a per-vector payload -- ``getitem(stage, 2)``, from
      either the quantizer or the ``pvs_quantize`` that replaced it -- reads
      4-bit codes under a fine scale the array has to multiply in, which is
      what ``conv2d_vsq`` does.

    A reference compile agrees exactly -- *measured*, all 49 of its
    ``conv2d_int8_accumulator`` records read an int8 input and all 36
    ``conv2d_vsq`` records read an int4 one, with no record crossing over.

    This replaced reading ``meta['quantization_annotation']``, and the reason
    is worth keeping. That map is keyed by the weight *node*; channel padding
    replaces a padded weight with a new ``get_attr``, the lookup then misses,
    and the fallback answered ``int8``. On the reference workload that misread
    exactly the three vector-scaled convolutions whose weights need padding --
    *silently*, not even appearing in ``mishead_count``, which is why it read
    33 against the 36 the quantizer chose.

    ``weight`` is accepted and unused: callers pass it, and dropping the
    parameter would be a second edit at every call site for no gain.
    """
    del weight
    if conv.target is _CONV2D_MX:
        return VSQ_HEAD
    if _is_vsq_payload(activation):
        return VSQ_HEAD
    return INT8_HEAD


def _block_size(output) -> int:
    """How many values share one per-vector scale.

    The machine's declaration is the answer, and the graph's is *checked*
    against it rather than preferred. They are not two opinions about the same
    thing: the block size is a fact about the datapath, and a quantizer that
    blocked at some other size produced a payload this machine cannot read at
    all. Preferring the graph's value would emit a record whose scale covers
    the wrong number of values -- silently, since the operator's own
    implementation does not consult it.
    """
    stated = get_arg_value(output, 4, "block_size")
    if (
        isinstance(stated, int)
        and not isinstance(stated, bool)
        and stated != VSQ_BLOCK_VALUES
    ):
        raise ValueError(
            f"{output.name}: quantized in blocks of {stated}, but this "
            f"machine reads {VSQ_BLOCK_VALUES}; the payload is not one it "
            "can interpret"
        )
    return VSQ_BLOCK_VALUES


def _replace_boundary_result(output, produced):
    """Hand ``produced`` to whatever was reading the boundary's result.

    For a plain quantize that is the node itself.

    For a vector-scaled one the whole **tuple** is replaced, because the
    per-vector stage returns the same three values in the same order. Every
    consumer's ``getitem`` therefore keeps its index and its meaning, the
    quantizer becomes dead, and the pre-lowering float chain behind it dies
    with it.

    That last clause is the point. While the stage returned only the payload,
    the quantizer had to stay to supply the next convolution's
    ``input_scale`` -- and it read the *dequantize* and the *ReLU*, so those
    stayed too, and the convolution under them. *Measured*: 30 convolutions
    survived a pass that reported all 85 rewritten, and fusion then met an
    int8 activation and a float bias.

    The payload-only replacement is kept for the case that cannot arise
    today and would be silent if it did: a vector-scaled output this pass
    lowered without building a per-vector stage. Substituting a tensor for a
    tuple there would rewire a consumer's ``getitem`` onto a tensor.
    """
    if (
        output.target is _QUANTIZE_VSQ
        and getattr(produced, "target", None) is _PVS
    ):
        output.replace_all_uses_with(produced)
        # The consumers' ``getitem`` nodes still describe the quantizer they
        # used to index -- both its ``dtype`` and its traced value. Neither is
        # inert: the serializer prefers ``meta['dtype']``, and *fusion clones
        # ``.value``* onto the submodule placeholder that becomes the record's
        # operand. Left alone, a companion scale that really is ``uint8``
        # serializes as ``float32`` on every vector-scaled record.
        #
        # Refreshed from the stage's own tuple rather than merely cleared.
        # The value alone is not the whole storage contract: PVS carries its
        # narrow payload in an int8 tensor because PyTorch has no int4 tensor
        # dtype.  The fork projects the same physical result declarations at
        # this replacement boundary -- index 1 is the uint8 mantissa and
        # index 2 is the int4 payload -- while index 0 remains the traced
        # floating-point coarse scale.
        produced_value = getattr(produced, "value", None)
        for user in list(produced.users):
            if user.target is not operator.getitem:
                continue
            user.meta.pop("dtype", None)
            index = user.args[1]
            if index == _VSQ_FINE_INDEX:
                user.meta["dtype"] = "uint8"
            elif index == _VSQ_PAYLOAD_INDEX:
                user.meta["dtype"] = "int4"
            if produced_value is not None and isinstance(index, int):
                user.value = produced_value[index]
                user.meta["val"] = user.value
        return
    value = _boundary_value(output)
    if value is not None:
        value.replace_all_uses_with(produced)


def _boundary_value(output):
    """The node holding the boundary's result, which is not always ``output``.

    A plain ``quantize`` *is* the value. A ``quantize_vsq`` returns three, and
    only the payload is the one a consumer reads as codes -- the other two are
    scales. Replacing the tuple node's uses would rewire the fine scale to
    index a tensor, which is a shape error at best and a silent scale swap at
    worst.
    """
    if output.target is not _QUANTIZE_VSQ:
        return output
    return next((user for user in output.users if _is_vsq_payload(user)), None)


def _quantize_per_vector(model, epilogue, output, output_scale, bits):
    """Append the stage that produces a per-vector scaled output.

    Its own record member, not a field on the epilogue: the converter reads
    the *sequence*, and ``ppu_fixed_relu -> pvs_quantize`` is a different
    layer from ``ppu_fixed_relu`` alone. The scale it divides by is the
    output domain's rebalanced coarse scale, which is the same value the
    epilogue was encoded against -- one number, read once, used twice, rather
    than two readings that could disagree.

    It is built as a **tuple-valued** call, matching the quantizer it
    replaces, and the caller substitutes it for that quantizer wholesale. See
    :func:`_replace_boundary_result` for why the alternative does not work.

    The *epilogue* stops claiming the boundary's output dtype here, and that
    matters for the record rather than for the graph. Only this stage
    produces the narrow payload; the epilogue hands it an intermediate, and a
    reference compile records exactly that -- ``pvs_quantize``'s input is
    ``int64`` where its output is ``int4``. Leaving the copied ``int4`` on the
    epilogue would serialize an operand narrower than the value that flows.
    """
    coarse_scale = _frozen_operand(
        model,
        f"{output.name}_alpha_coarse_scale",
        output_scale.reshape(-1).detach().clone(),
    )
    with model.graph.inserting_after(epilogue):
        quantized = model.graph.call_function(
            _PVS,
            (epilogue, coarse_scale),
            {"output_precision": bits, "block_size": _block_size(output)},
        )
    epilogue.meta.pop("dtype", None)
    # And its traced value, not only its dtype. ``_adopt_value`` gave the
    # epilogue the boundary output's tensor a moment ago, on the assumption
    # that the epilogue *was* the output; it is not once this stage exists.
    # Operator fusion reads ``.value``, so a stale one is not inert.
    value = _boundary_value(output)
    if value is not None:
        _adopt_value(epilogue, value, torch.int64)
    # The stage returns what the quantizer returned, so it takes that node's
    # traced tuple rather than the payload's tensor: a consumer reaching it
    # through ``getitem`` needs all three, and the payload alone would make
    # index 1 read a channel of the payload.
    # What the quantizer knew and the stage would otherwise lose.
    #
    # A consumer reaches this stage's coarse scale as ``getitem(pvs, 0)``,
    # and that node has no endpoints on it -- so a consumer asking which pair
    # its activation was split against falls back to the *declared* default,
    # which is the weight's 7.0/127.0 and wrong for an activation.
    # *Measured:* 32 of 36 vector-scaled records carried 7.0/127.0 where the
    # reference carries 8.0/126.0.
    #
    # Recorded on the coarse-scale operand rather than on the stage, because
    # that is the node a consumer holds.
    coarse_scale.meta[VSQ_SCALE_ENDPOINTS] = _quantizer_endpoints(output)
    semantic = get_arg_value(output, 1, "coarse_scale")
    if isinstance(semantic, Node):
        coarse_scale.meta[SEMANTIC_COARSE_SCALE_META] = semantic
    quantized.meta.update(output.meta)
    # This target operation owns a heterogeneous physical result.  Do not
    # inherit the surrounding payload domain as though it applied to all
    # three components: output 0 is the pass-through logical coarse scale,
    # output 1 is the uint8 mantissa, and output 2 is the narrow payload
    # carried in an int8 PyTorch tensor because PyTorch has no int4 dtype.
    quantized.meta["dtype"] = (None, "uint8", f"int{bits}")
    # The traced tuple, **run rather than inherited**.
    #
    # Taking the quantizer's is the obvious move and it is wrong: its second
    # element is that quantizer's *float* fine scale, where this stage emits a
    # uint8 mantissa. A consumer then serializes ``float32`` for an operand
    # the endpoint reads as ``uint8`` -- on all 36 vector-scaled records --
    # and clearing a stale ``meta['dtype']`` does not fix it, because the
    # value itself is the wrong one and fusion clones exactly that.
    #
    # So the stage is executed on the epilogue's own traced value: what it
    # will produce, by construction rather than by assertion.
    epilogue_value = getattr(epilogue, "value", None)
    tuple_value = None
    if epilogue_value is not None:
        tuple_value = _PVS(
            epilogue_value,
            coarse_scale.value,
            output_precision=bits,
            block_size=_block_size(output),
        )
    elif getattr(output, "value", None) is not None:
        # Nothing traced the epilogue, so only the quantizer's shapes are
        # available. Better than no value; the widths are its, not ours.
        tuple_value = output.value
    if tuple_value is not None:
        quantized.value = tuple_value
        quantized.meta["val"] = tuple_value
    return quantized


def _lower_near_memory(
    model, conv, activation, weight, stride, padding, dilation, groups,
    bias, relu, output, accumulator_scale, output_scale, bits, head,
) -> None:
    """A depthwise convolution and its near-memory epilogue.

    The same accumulator operator as the matrix unit -- the reference emits
    ``conv2d_int8_accumulator`` for both -- with the other epilogue. One
    scalar multiplier, one scalar bias: the NMP broadcasts across lanes, which
    is why a per-channel bias must already have been projected onto a single
    value before it reaches here.
    """
    multiplier, encoded_bias = encode_alpha_nmp_pipeline(
        accumulator_scale=float(accumulator_scale.reshape(-1)[0]),
        output_scale=float(output_scale.reshape(-1)[0]),
        output_bits=bits,
        bias=_scalar_bias(model, bias),
        relu=relu is not None,
    )
    with model.graph.inserting_before(output):
        accumulated = model.graph.call_function(
            _ACCUMULATE,
            (activation, weight, stride, padding, dilation, groups),
        )
        requantized = model.graph.call_function(
            _NMP_EPILOGUE,
            (
                accumulated,
                int(multiplier.multiplier),
                int(multiplier.shift),
                int(encoded_bias),
                relu is not None,
                bits,
            ),
        )
    accumulated.meta.update(conv.meta)
    accumulated.meta[HEAD_META] = head
    _adopt_value(accumulated, conv, ACCUMULATOR_DTYPE)
    requantized.meta.update(output.meta)
    _adopt_value(requantized, output)
    produced = requantized
    if bits == PVS_OUTPUT_BITS:
        produced = _quantize_per_vector(
            model, requantized, output, output_scale, bits
        )
    _replace_boundary_result(output, produced)


def _scalar_bias(model, bias):
    """The one value the near-memory unit's bias field can hold.

    A depthwise bias is per channel in the model and scalar on the machine.
    The projection is the *adapter's* -- it folds the per-channel bias onto one
    value while loading, and records the residual it costs -- so by here a
    bias that is still a vector of differing values is a bug upstream of this,
    not something to average away quietly.

    Read at the *logical* extent, because channel padding widens it: a bias
    ``[c, c, ..., c]`` grown to the datapath's lane count becomes
    ``[c, ..., c, 0, ..., 0]``, which is a vector of differing values by every
    test that matters here and is still one value broadcast across the lanes
    the layer owns. The zeros are the compiler's, not the model's, and the
    extent recorded before the pad is what says where they start.

    The extent is *checked*, not trusted. A recorded shape says how long the
    value was, and nothing about which end grew or what filled it; a bias
    padded at the front, or reordered, would have the same record and the
    wrong prefix. So the discarded tail has to be the zeros this compiler
    writes -- and where it is not, the whole vector is read and the vector's
    own disagreement is reported, which is the answer that was right before
    any of this existed.
    """
    values = _static_tensor(model, bias)
    if values is None:
        return None
    flat = values.reshape(-1)
    logical = get_hardware_logical_shape(bias)
    if (
        len(logical) == 1
        and 0 < logical[0] < flat.numel()
        and bool(torch.all(flat[logical[0]:] == 0))
    ):
        flat = flat[: logical[0]]
    if flat.numel() == 0:
        return None
    if flat.numel() > 1 and not bool(torch.all(flat == flat[0])):
        raise ValueError(
            f"{getattr(bias, 'name', bias)}: a near-memory bias is one value "
            f"broadcast across lanes, but this one has {flat.numel()} "
            f"differing entries; it should have been projected while loading"
        )
    return float(flat[0])


def rewrite_pe_convolutions(model: GraphModule) -> int:
    """Lower each matrix convolution into an accumulator and an epilogue.

    The pair is what the endpoint executes and what the converter emits as one
    ``gemm_convolution`` record; see ``operators.py``. Everything between --
    the dequantize, the ReLU, the output quantize -- is *absorbed*, because on
    this machine they are not operations but the epilogue's stages.

    **Only the int8 head so far.** A reference compile's 66 matrix records are
    30 ``conv2d_int8_accumulator`` and 36 ``conv2d_vsq``, and a vector-scaled
    convolution needs its fine-scale operand and ``vsq_operands=True`` in the
    encoding, which changes the primary multiplier's storage factor by 256.
    Emitting the int8 head for a VSQ boundary would produce a record that
    converts and computes the wrong thing, so this refuses those instead:
    a boundary whose activation carries a block size is left alone, and the
    count this returns is what says so.
    """
    rewritten = 0
    # Resolved once: the rule set is a property of the machine, not of a node,
    # and rebuilding it per convolution would compose the same lists 85 times.
    rules = prepared_domain_rules()
    for conv in list(model.graph.nodes):
        if conv.op != "call_function" or conv.target not in _MATRIX_SOURCES:
            continue
        matched = _pe_boundary(model, conv, rules)
        if matched is None:
            continue
        (
            activation,
            _dequantize,
            relu,
            output,
            accumulator_scale,
            output_scale,
            bits,
        ) = matched
        weight = get_arg_value(conv, 1, "weight")
        head = _conv_head(conv, weight, activation)
        bias = get_arg_value(conv, 2, "bias")
        stride = list(get_arg_value(conv, 3, "stride", [1, 1]))
        padding = list(get_arg_value(conv, 4, "padding", [0, 0]))
        dilation = list(get_arg_value(conv, 5, "dilation", [1, 1]))
        groups = get_arg_value(conv, 6, "groups", 1)

        if accumulator_scale.numel() == 1:
            # The near-memory unit's, and the accumulator scale is what says
            # so: it has one scalar multiplier and one scalar bias broadcast
            # across the lanes, so a boundary the quantizer gave a *scalar*
            # accumulator scale is one that unit can run. Measured, not
            # guessed -- on the reference workload this splits 85 convolutions
            # into exactly the 19 near-memory and 66 matrix records a
            # reference compile emits.
            _lower_near_memory(
                model, conv, activation, weight, stride, padding, dilation,
                groups, bias, relu, output, accumulator_scale, output_scale,
                bits, head,
            )
            rewritten += 1
            continue
        vsq = head == VSQ_HEAD
        uses_pvs = bits == PVS_OUTPUT_BITS
        pipeline = encode_alpha_pe_pipeline(
            accumulator_scale=accumulator_scale.reshape(-1),
            output=AlphaOutputDomain(
                scale=float(output_scale.reshape(-1)[0]),
                bits=bits,
                uses_pvs=uses_pvs,
            ),
            bias=_static_tensor(model, bias),
            relu=relu is not None,
            # Both operands carry a uint8 fine scale the array has already
            # multiplied into the accumulator, so it stands 65536x larger than
            # the same computation on plain 8-bit operands would leave it, and
            # the first stage's factor undoes exactly that. A fact about the
            # head, so it is read from the head.
            vsq_operands=vsq,
        )

        with model.graph.inserting_before(output):
            if vsq:
                input_coarse = conv.kwargs.get("input_coarse_scale")
                weight_coarse = conv.kwargs.get("weight_coarse_scale")
                input_sqmax, input_csqmax = _vsq_endpoints(input_coarse)
                weight_sqmax, weight_csqmax = _vsq_endpoints(weight_coarse)
                # The record carries the folded scale as an operand as well as
                # having it folded into the epilogue's multiplier. Both, and
                # deliberately: the epilogue is what applies it, and the head
                # states what was applied, which is what makes the record
                # readable without re-deriving the split.
                folded = _frozen_operand(
                    model,
                    f"{conv.name}_folded_coarse_scale",
                    accumulator_scale.reshape(-1)
                    .detach()
                    .clone()
                    .to(torch.float32),
                )
                operands = {
                    # Read from kwargs directly: the scales are keyword-only
                    # on ``conv2d_mx``, so they are never in ``args`` and a
                    # positional index has no meaning.
                    "input_scale": conv.kwargs.get("input_scale"),
                    "block_size": conv.kwargs.get("block_size"),
                    "folded_coarse_scale": folded,
                    "input_fine_quant_max": input_sqmax,
                    "input_coarse_scale_quant_max": input_csqmax,
                    "weight_fine_quant_max": weight_sqmax,
                    "weight_coarse_scale_quant_max": weight_csqmax,
                }
                for key, source in (
                    ("semantic_input_coarse_scale", input_coarse),
                    ("semantic_weight_coarse_scale", weight_coarse),
                ):
                    semantic = _semantic_coarse_scale(model, source)
                    if semantic is not None:
                        operands[key] = semantic
                accumulated = model.graph.call_function(
                    _ACCUMULATE_VSQ,
                    (
                        activation,
                        weight,
                        conv.kwargs.get("weight_scale"),
                        stride,
                        padding,
                        dilation,
                        groups,
                    ),
                    operands,
                )
            else:
                accumulated = model.graph.call_function(
                    _ACCUMULATE,
                    (activation, weight, stride, padding, dilation, groups),
                )
            requantized = model.graph.call_function(
                _EPILOGUE,
                (
                    accumulated,
                    _frozen_operand(
                        model,
                        f"{conv.name}_ppu_primary_multiplier",
                        torch.tensor(
                            pipeline.primary.values, dtype=torch.float32
                        ),
                    ),
                    int(pipeline.primary.shift),
                    _frozen_operand(
                        model,
                        f"{conv.name}_ppu_bias",
                        torch.tensor(pipeline.bias, dtype=torch.float32),
                    )
                    if pipeline.bias
                    else None,
                    int(pipeline.secondary.multiplier),
                    int(pipeline.secondary.shift),
                    relu is not None,
                    bits,
                ),
                {"integer_accumulator": True},
            )
        accumulated.meta.update(conv.meta)
        accumulated.meta[HEAD_META] = head
        _adopt_value(accumulated, conv, ACCUMULATOR_DTYPE)
        requantized.meta.update(output.meta)
        _adopt_value(requantized, output)
        produced = requantized
        if uses_pvs:
            produced = _quantize_per_vector(
                model, requantized, output, output_scale, bits
            )
        _replace_boundary_result(output, produced)
        rewritten += 1
    if rewritten:
        model.graph.eliminate_dead_code()
        model.recompile()
    return rewritten


_VSQ_SCALE_ENCODE = torch.ops.magnet_alpha.vsq_scale_encode.default

#: Marks a fine scale that is already the uint8 mantissa the datapath reads,
#: so the finalizer does not encode it a second time.
PHYSICAL_VSQ_SCALE_META = "magnet_alpha_vsq_physical_scale"


def _is_physical_vsq_scale(node, visited=None) -> bool:
    """Whether this fine scale is already a code rather than a fraction.

    Three ways it can be, and all three are structural:

    * it was marked when this pass encoded it;
    * it *is* an encode call;
    * it is the companion scale a per-vector stage produced --
      ``getitem(pvs_quantize, 1)`` -- which ``alpha_pvs_quantize`` returns as
      ``uint8`` by construction.

    The third is why a boundary reading a lowered producer needs nothing done
    to it: the producer already emitted the mantissa.
    """
    if not isinstance(node, Node):
        return False
    if visited is None:
        visited = set()
    if node in visited:
        return False
    visited.add(node)
    if node.meta.get(PHYSICAL_VSQ_SCALE_META):
        return True
    if node.target is _VSQ_SCALE_ENCODE:
        return True
    if (
        node.target is operator.getitem
        and isinstance(node.args[0], Node)
        and node.args[0].target is _PVS
        and node.args[1] == _VSQ_FINE_INDEX
    ):
        return True
    if node.target is _CAT:
        inputs = node.args[0] if node.args else ()
        return isinstance(inputs, (tuple, list)) and bool(inputs) and all(
            _is_physical_vsq_scale(value, visited.copy()) for value in inputs
        )
    return False


def _refresh_physical_vsq_scale_value(node) -> None:
    """Refresh a composed mantissa after its PVS leaves were rewritten.

    Target lowering runs after the shared graph's initial shape propagation.
    Replacing a prepared VSQ producer with ``pvs_quantize`` refreshes each
    component ``getitem``, but an already-existing component join still holds
    its pre-lowering float value. Re-evaluate only the composition whose
    physical leaves were proved by :func:`_is_physical_vsq_scale`.
    """
    if node.target is not _CAT:
        return
    inputs = node.args[0] if node.args else ()
    missing = [
        value.name
        for value in inputs
        if not isinstance(value, Node) or getattr(value, "value", None) is None
    ]
    if missing:
        raise ValueError(
            f"{node.name}: physical VSQ scale inputs have no traced values: "
            + ", ".join(missing)
        )
    propagate_shape(node)


def finalize_alpha_vsq_operands(model: GraphModule) -> int:
    """Materialize each physical Alpha fine-scale operand exactly once.

    Alpha does not read a float fine scale. It reads an **unsigned 8-bit
    mantissa** and keeps the discarded range in the coarse scale, which is
    what ``folded_coarse_scale`` already carries. So the operand a
    ``conv2d_vsq`` names has to *be* that mantissa.

    ``convert_pt2e`` cannot produce it: it knows the parameter and writes the
    three VSQ components as floats, under names that say nothing about
    Alpha's split. This pass encodes the weight's -- once per distinct
    ``(scale, endpoints)`` pair, since several convolutions can share one.

    The float it replaces then has no reader and is collected, which is what
    the reference does as well: its ``tensor_files/`` carries the mantissa and
    no ``*_weight_scale.bin`` at all. It is marked *semantic* first anyway --
    cheap, and it is what keeps the serializer from counting it as storage in
    a graph where something else still reads it.

    An activation's is left alone, because it already is one: the producing
    boundary's per-vector stage emits ``uint8``. Checking that structurally
    rather than by dtype matters -- a dtype can be relabelled, and was: the
    record declared ``uint8`` over a float payload before this existed, which
    is a collateral that converts and stages a tensor file contradicting its
    own field.

    Returns:
        How many mantissas were materialized. Operands it could not encode
        are counted in the log rather than here: this number is what the pass
        *did*, and mixing the two would let a compile that encoded nothing
        look the same as one with nothing to encode.
    """
    encoded_by_source = {}
    count = 0
    unencoded = 0
    for conv in list(model.graph.nodes):
        if conv.op != "call_function" or conv.target is not _ACCUMULATE_VSQ:
            continue
        kwargs = dict(conv.kwargs)
        weight_scale = get_arg_value(conv, 2, "weight_scale")
        for role, scale in (
            ("weight_scale", weight_scale),
            ("input_scale", kwargs.get("input_scale")),
        ):
            if not isinstance(scale, Node):
                continue
            if _is_physical_vsq_scale(scale):
                _refresh_physical_vsq_scale_value(scale)
                scale.meta[PHYSICAL_VSQ_SCALE_META] = True
                continue
            fine_quant_max, coarse_scale_quant_max = _fine_scale_endpoints(
                scale,
                conv.kwargs.get(
                    "weight_coarse_scale"
                    if role == "weight_scale"
                    else "input_coarse_scale"
                ),
            )
            key = (scale, fine_quant_max, coarse_scale_quant_max)
            encoded = encoded_by_source.get(key)
            if encoded is None:
                values = _static_tensor(model, scale)
                if values is None:
                    # A scale this pass cannot bake, left exactly as it is.
                    #
                    # The tempting thing is to encode it with an operation --
                    # the fork has such a branch -- and it does not work here.
                    # ``vsq_scale_encode`` is not a record: the endpoint would
                    # never perform it, so the collateral would be a layer
                    # short while every count stayed green. That trades a
                    # structural gap you can see for a numerical one you
                    # cannot.
                    #
                    # It is also not the real problem. A reference compile
                    # never reaches here: all 36 of its vector-scaled records
                    # read a scale a *record* produced -- a per-vector stage's
                    # second output. Reaching here means this activation's
                    # producer was not lowered, and that is the thing to fix.
                    #
                    # So the float stays. The record then reports ``float32``
                    # for an operand the endpoint reads as ``uint8``, and the
                    # converter refuses it by name. Loud and locatable beats
                    # quiet and wrong.
                    logger.warning(
                        "%s: %s is not a constant and not a record's output, "
                        "so no mantissa can be materialized for it; its "
                        "producing boundary was not lowered",
                        conv.name,
                        getattr(scale, "name", scale),
                    )
                    unencoded += 1
                    continue
                encoded = _frozen_operand(
                    model,
                    f"{scale.name}_alpha_mantissa",
                    _VSQ_SCALE_ENCODE(
                        values, fine_quant_max, coarse_scale_quant_max
                    ),
                )
                # A mantissa is *storage*, not an instruction field: the
                # array fetches it per block. ``_frozen_operand`` marks
                # everything it makes as non-allocating, which is right for a
                # multiplier folded into an instruction and wrong here.
                encoded.meta.pop(NON_ALLOCATING_OPERAND_META[0], None)
                encoded.meta["dtype"] = "uint8"
                encoded.meta[PHYSICAL_VSQ_SCALE_META] = True
                inherit_hardware_logical_shape(encoded, scale)
                # The float it replaces is kept for reference and issued by
                # nothing, which is exactly what the semantic mark means.
                scale.meta[NON_ALLOCATING_OPERAND_META[1]] = True
                encoded_by_source[key] = encoded
                count += 1
            if role == "weight_scale":
                conv.update_arg(2, encoded)
            else:
                kwargs[role] = encoded
        conv.kwargs = kwargs
    if unencoded:
        logger.warning(
            "%d vector-scaled operands kept their float scale; each is a "
            "record the endpoint will misread, and each names a boundary "
            "that did not lower",
            unencoded,
        )
    if count:
        model.graph.eliminate_dead_code()
        model.recompile()
    return count


def _nmp_lanes(config) -> int:
    """The near-memory unit's channel parallelism, from the declared machine.

    Read from the config rather than imported as a constant, and through
    ``compute_unit_bindings`` rather than by name, so a machine that binds a
    different unit is planned against that one. The fallback is the module's
    own declaration, for a caller with no config -- which is only the tests.
    """
    unit = config.unit("nmp") if config is not None else None
    if unit is not None:
        return unit.unroll("CHANNEL") or unit.lanes
    from voyager_compiler.codegen.targets.magnet_alpha.hardware import NMP_UNIT

    return NMP_UNIT.unroll("CHANNEL")


def insert_shuffle_copies(model: GraphModule, config) -> int:
    """The block movements a two-way shuffle leaves for the near-memory unit.

    Alpha's shuffle writes its two branches interleaved, but only the first
    and last channel blocks land where they belong; every block between them
    has to be moved, and the endpoint issues each move as its own operation.
    A reference compile emits them as ``nmp_copy`` records named
    ``<shuffle>_cpy1``, ``_cpy2`` -- 40 of them for this workload -- so a
    graph without them is a graph missing 40 records.

    The count is a property of the machine and the shape, not a choice:
    ``channel_blocks - 2``, where a block is the near-memory unit's channel
    parallelism. The two that need no move are the pair already at the ends.
    Checked against the reference: its three shuffle widths, 64 / 128 / 256
    channels over 32 lanes, give 2 / 4 / 8 blocks and therefore 0 / 2 / 6
    copies, and its sixteen shuffles sum to exactly 40.

    They form a *chain* -- each copy reads the previous one -- and the
    shuffle's original readers are repointed at the last. That is what makes
    them ordered: the moves overlap in memory, so a reader that took the
    shuffle directly would see blocks that have not moved yet.

    Args:
        model: The graph, after ``rewrite_channel_shuffles``.
        config: The machine, for its near-memory channel parallelism.

    Returns:
        How many copy operations were inserted.

    Raises:
        ValueError: If a shuffle's channel count is not a whole, even number
            of blocks. The movement pattern is defined over block *pairs*, so
            an odd or partial count has no answer -- and silently emitting
            the wrong number of copies would be a record miscount nothing
            downstream would catch.
    """
    lanes = _nmp_lanes(config)
    if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes <= 0:
        raise ValueError(
            f"Alpha's NMP channel parallelism must be a positive integer, "
            f"got {lanes!r}"
        )
    inserted = 0
    for shuffle in list(model.graph.nodes):
        if shuffle.op != "call_function" or shuffle.target is not _SHUFFLE:
            continue
        if any(user.target is _SHUFFLE_COPY for user in shuffle.users):
            continue  # idempotent: already expanded
        value = getattr(shuffle, "value", None)
        if value is None or getattr(value, "ndim", 0) != 4:
            raise ValueError(
                f"{shuffle.name}: a channel shuffle needs a rank-4 value to "
                f"count its blocks; got {value!r}"
            )
        channels = int(value.shape[1])
        groups = get_arg_value(shuffle, 2, "groups")
        if groups != 2:
            raise ValueError(
                f"{shuffle.name}: Alpha's physical shuffle movement is "
                f"defined for two groups, got {groups!r}"
            )
        blocks, remainder = divmod(channels, lanes)
        if remainder or blocks < 2 or blocks % 2:
            raise ValueError(
                f"{shuffle.name}: {channels} channels is not a whole, even "
                f"number of {lanes}-channel blocks, so its shuffle has no "
                f"defined block movement"
            )
        copies = blocks - 2
        if not copies:
            continue

        readers = list(shuffle.users)
        previous = shuffle
        for index in range(1, copies + 1):
            with model.graph.inserting_after(previous):
                copy = model.graph.create_node(
                    "call_function",
                    _SHUFFLE_COPY,
                    (previous,),
                    name=f"{shuffle.name}_cpy{index}",
                )
            # A late rewrite must not inherit a stale schedule, placement, or
            # codegen projection if it is applied to a previously inspected
            # graph.  These are the only transparent source facts the fork
            # carries across; final shape propagation fills the value facts.
            for key in ("dtype", "nn_module_stack"):
                if key in shuffle.meta:
                    copy.meta[key] = shuffle.meta[key]
            # Match the fork's memory-rewrite contract.  These fields state
            # which final FX owner this is; the subsequent storage pass fills
            # the precise source/destination views and backing identities.
            copy.meta[_PHYSICAL_COPY_META] = True
            copy.meta[_ALPHA_META] = {
                "block_channels": lanes,
                "copy_index": index,
                "kind": _SHUFFLE_COPY_KIND,
                "logical_parent": shuffle.name,
            }
            _adopt_value(copy, shuffle)
            previous = copy
        # After the chain is built, not during: repointing as we go would
        # make each copy read the one before it *and* be its own reader.
        for reader in readers:
            reader.replace_input_with(shuffle, previous)
        inserted += copies
    if inserted:
        model.graph.lint()
        model.recompile()
    return inserted


def apply_graph_rewrites(model: GraphModule, config) -> None:
    """Apply Alpha's semantic rewrites after target-neutral transforms.

    Called last, after the target-neutral passes have settled the graph.
    Physical movement is intentionally deferred to
    :func:`apply_memory_rewrites`, at the terminal compile boundary where
    storage and schedules are about to be frozen.
    """
    # Convolutions first. Their epilogues absorb the ReLU and the output
    # quantize, so running the shuffle rewrite first would leave it matching a
    # graph that is about to change under it -- and the shuffle's operands are
    # convolution results, which is exactly what the epilogue replaces.
    convolutions = rewrite_pe_convolutions(model)
    # Immediately after, and before anything else looks at the graph: until
    # this runs a vector-scaled call names a float where the datapath reads a
    # code, and every later pass that reads the operand reads the wrong one.
    mantissas = finalize_alpha_vsq_operands(model)
    upsamples = rewrite_integer_upsamples(model)
    shuffles = rewrite_channel_shuffles(model)
    # Logged, not silent: "none matched" and "none present" are different
    # answers on this workload, and a lane that rewrote nothing still
    # compiles -- it just compiles the wrong graph.
    logger.info(
        "magnet-alpha graph rewrites: %d convolutions, %d VSQ mantissas, "
        "%d integer upsamples, %d channel shuffles",
        convolutions,
        mantissas,
        upsamples,
        shuffles,
    )


def apply_memory_rewrites(model: GraphModule, config) -> int:
    """Materialize Alpha's physical movement operations before planning.

    The semantic graph exposes a channel-shuffle operation.  Its residual
    block movements are physical work, so—matching the fork—they enter only
    at the terminal memory-planning boundary.  The inserted copies are then
    ordinary final FX owners: shape propagation, local scheduling, storage
    relations, DP ordering, allocation, and serialization all see them.
    """
    copies = insert_shuffle_copies(model, config)
    logger.info("magnet-alpha memory rewrites: %d shuffle copies", copies)
    return copies
