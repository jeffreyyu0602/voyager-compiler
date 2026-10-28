# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
import math
import operator
from typing import List

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

from voyager_compiler.hardware_config import (
    COMMON_IMAGE_INPUT_CHANNELS,
)
from voyager_compiler.codegen.aten_classifier import is_elementwise_op
from voyager_compiler.codegen.node_info import (
    _BROADCAST_OPS,
    get_arg_value,
    HARDWARE_PADDING_PAD,
    HARDWARE_PADDING_SLICE,
    inherit_hardware_logical_shape,
    is_conv2d,
    is_depthwise_conv,
    is_gemm_op,
    is_matmul,
    node_operand_bits,
    preserve_hardware_logical_shape,
)
from voyager_compiler.export_utils import (
    create_getattr_from_value,
    get_aten_graph_module,
)
from voyager_compiler.shape_prop import fetch_attr, propagate_shape

logger = logging.getLogger(__name__)

__all__ = [
    "pad_matrix_op_dimensions",
    "pad_vector_op_dimensions",
    "pad_vit_embeddings_output",
]


def slicing_and_padding_cancel_out(shape, slice_dim, start, end, pad):
    ndim = len(shape)
    k = len(pad) // 2

    pad_pairs = list(reversed(list(zip(pad[0::2], pad[1::2]))))
    full_pad_pairs = [(0, 0)] * (ndim - k) + pad_pairs

    for dim, (left, right) in enumerate(full_pad_pairs):
        if dim != slice_dim:
            if left != 0 or right != 0:
                return False
        else:
            if left != start or right != shape[slice_dim] - end:
                return False

    return True


# Relayout ops a pad is immune to: each leaves the padded dim's extent -- and
# its position from the end -- alone, so a slice above them cancels a pad below
# them exactly as an adjacent one would.  ``repeat_kv`` is why this earns its
# keep: its unsqueeze / expand / reshape sit between the two.
_PAD_IMMUNE_OPS = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
)

# The immune ops that *name* their output extents, so the padded dim's size has
# to be restated on them once the slice below is gone.
_SIZED_OPS = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten.expand.default,
)


def _padded_dim(pad):
    """The one dim (counted from the end) ``pad`` grows, or None if it grows
    several -- no single slice cancels that."""
    dims = [
        -(k + 1) for k in range(len(pad) // 2) if pad[2 * k] or pad[2 * k + 1]
    ]
    return dims[0] if len(dims) == 1 else None


def _pad_for_dim(pad, dim, new_dim):
    """``pad``'s one non-zero pair, restated to grow ``new_dim`` rather than
    ``dim`` -- both counted from the end, as ``F.pad`` orders its pairs."""
    j = -dim - 1
    out = [0] * (2 * -new_dim)
    out[-2 * new_dim - 2] = pad[2 * j]
    out[-2 * new_dim - 1] = pad[2 * j + 1]
    return out


def _dim_above(node, dim):
    """``dim`` -- counted from the end of ``node``'s output -- restated against
    its input, or ``None`` if ``node`` does not hand that dim through intact.

    A transpose or permute genuinely moves the dim, so it is remapped; the rest
    of ``_PAD_IMMUNE_OPS`` only touch dims to its left and leave it where it is.
    """
    src = node.args[0]
    if not isinstance(src, Node) or not hasattr(src, "value"):
        return None
    out, inp = tuple(node.value.shape), tuple(src.value.shape)
    rank = len(out)

    if node.target is torch.ops.aten.transpose.int:
        a, b = (int(d) % rank - rank for d in node.args[1:3])
        above = {a: b, b: a}.get(dim, dim)
    elif node.target is torch.ops.aten.permute.default:
        perm = [int(p) % rank for p in node.args[1]]
        above = perm[dim + rank] - rank
    else:
        above = dim

    if len(inp) < -above or inp[above] != out[dim]:
        return None
    return above


def _relayout_chain_above(node, dim):
    """``(chain, src, src_dim, dims)`` -- the immune relayout ops feeding
    ``node`` (nearest first), the tensor they re-address, ``dim`` restated
    against it, and ``dim`` as each chain op's own output sees it.  Walks while
    the op hands the dim through and nobody else reads it: restating an op's
    extents must disturb no one."""
    chain, dims = [], []
    curr = node
    while curr.target in _PAD_IMMUNE_OPS and len(curr.users) == 1:
        above = _dim_above(curr, dim)
        if above is None:
            break
        chain.append(curr)
        dims.append(dim)
        dim = above
        curr = curr.args[0]
    return chain, curr, dim, dims


def _find_cancelling_slice(node, pad):
    """``(slice, chain, dims)`` for the slice above ``node`` that ``pad``
    exactly undoes, or ``(None, [], [])``."""
    dim = _padded_dim(pad)
    if dim is None:
        return None, [], []

    chain, curr, src_dim, dims = _relayout_chain_above(node, dim)
    if src_dim != dim:
        # The pad and the slice name different dims; they cannot be compared
        # without rebuilding the pad, which the cancel check does not do.
        return None, [], []
    if curr.target is not torch.ops.aten.slice.Tensor:
        return None, [], []
    if not slicing_and_padding_cancel_out(
        curr.args[0].value.shape, *curr.args[1:], pad
    ):
        return None, [], []
    return curr, chain, dims


def _restate_chain(chain, dims, size):
    """Restate the extent each op in ``chain`` names for the padded dim -- at
    that op's own view of it -- so a tensor grown to ``size`` flows through."""
    for n, d in zip(reversed(chain), reversed(dims)):
        if n.target in _SIZED_OPS:
            sizes = list(n.args[1])
            i = d + len(sizes)
            if sizes[i] != -1:
                sizes[i] = size
                n.update_arg(1, sizes)
        propagate_shape(n)


def _drop_slice_through_chain(cancel, chain, dims):
    """Re-point ``chain`` at what ``cancel`` sliced and restate the extents the
    relayout ops name, so the full-width tensor flows through them and the pad
    below is unnecessary."""
    full = cancel.args[0]
    chain[-1].replace_input_with(cancel, full)
    _restate_chain(chain, dims, full.value.shape[dims[-1]])


_INDEX_COPY = torch.ops.aten.index_copy_.default

# Whether a pad on a KV-cache write is taken into the buffer rather than run
# over the whole cache every step (``_fold_pad_into_cache``).  It rewrites the
# buffer's contents and its dtype, so it is worth being able to turn off.
FOLD_PAD_INTO_CACHE = True


def _fold_pad_into_cache(model, idx, pad, pad_value, fold_cache):
    """Pad the KV cache buffer itself, and the token written into it, rather
    than the whole cache on every step.

    The write puts one token into a buffer the array wants wider.  Widening the
    buffer is a compile-time edit of its contents, and the token then arrives
    already wide -- a pad on one position, which cancels against the slice its
    projection left on it.  So the cache pays nothing.

    Not when the pad grows the axis the write indexes: that would move every
    position the cache holds.
    """
    dim = _padded_dim(pad)
    if dim is None:
        return False

    cache = idx.args[0]
    if not isinstance(cache, Node) or cache.op != "get_attr":
        return False
    if idx.args[1] % len(idx.value.shape) == dim % len(idx.value.shape):
        return False

    baked = F.pad(fetch_attr(model, cache.target), pad, "constant", pad_value)
    with model.graph.inserting_before(idx):
        wide = create_getattr_from_value(
            model, model.graph, cache.target + "_padded", baked
        )
        propagate_shape(wide, model)

    # The token has to arrive wide, one position's worth.
    token = _insert_pad(model, idx.args[3], pad, pad_value, fold_cache)

    wide.meta["dtype"] = cache.meta.get("dtype")
    idx.update_arg(0, wide)
    idx.update_arg(3, token)
    propagate_shape(idx)
    return True


def _insert_pad(model, node, pad, pad_value, fold_cache=False):
    """``node`` grown by ``pad``: a pad node, or -- where a slice above it left
    exactly what the pad puts back -- what that slice was hiding, with neither
    op left behind.

    Every pad goes in through here, so each way of not paying for one is tried
    in turn: a slice above that already left room, then a cache write that can
    take the pad into its buffer, then a buffer that is simply padded as it
    stands.  The round-trip sweep would reach the same graph on its own, but
    only after L2 tiling and the layout pass have read a pad that was never
    needed, and tiling is what it would mislead.
    """
    cancel, chain, dims = _find_cancelling_slice(node, pad)
    if cancel is not None:
        logger.info(f"Padding {node} cancels the slice {cancel}")
        if not chain:
            return cancel.args[0]
        _drop_slice_through_chain(cancel, chain, dims)
        return node

    if (
        fold_cache
        and node.target is _INDEX_COPY
        and _fold_pad_into_cache(model, node, pad, pad_value, fold_cache)
    ):
        return node

    if node.op == "get_attr":
        # Nothing writes it, so its padded form is known now: widen the buffer
        # once here rather than the tensor it holds on every step.
        baked = F.pad(
            fetch_attr(model, node.target), pad, "constant", pad_value
        )
        with model.graph.inserting_after(node):
            wide = create_getattr_from_value(
                model, model.graph, node.target + "_padded", baked
            )
        propagate_shape(wide, model)
        wide.meta["dtype"] = node.meta.get("dtype")
        # The extent is gone from the tensor the moment it is baked: the added
        # channels are zeros and a parameter does not explain which of its
        # zeros were always there. The caller recorded it on the original; a
        # *replacement* has to be handed it, or the record one node later
        # reports the padded count as the layer's own.
        inherit_hardware_logical_shape(wide, node)
        wide.meta[HARDWARE_PADDING_PAD] = tuple(pad)
        return wide

    with model.graph.inserting_after(node):
        padded = model.graph.call_function(
            torch.ops.aten.pad.default, (node, pad, "constant", pad_value)
        )
    propagate_shape(padded)
    padded.meta["dtype"] = node.meta.get("dtype")
    # This pad is the compiler widening a value to the extent its hardware
    # holds, not something the model asked for -- and the two are the same
    # operator with the same arguments, so the difference has to be said
    # rather than inferred. Tagged here, at the point of creation, for the
    # same reason the restoring slice is: every pad this module makes comes
    # through here, and a caller tagging afterwards would miss the ones that
    # were hoisted onto a producer.
    padded.meta[HARDWARE_PADDING_PAD] = tuple(pad)
    inherit_hardware_logical_shape(padded, node)
    return padded


def _hoist_pad_above_repeat(model, node, pad, pad_value, fold_cache):
    """Pad what the relayout ops feeding ``node`` re-address, rather than what
    they hand on, and let the pad flow back down through them.

    ``repeat_kv`` is the case that matters: padding its output fills the 32
    heads it broadcast, where padding its input fills the 8 the cache holds --
    the same fill on a quarter of the data, and the broadcast still folds into
    the consumer's addressing.  Only worth it when the chain actually repeats;
    over a plain relayout the pad is the same size either side of it.  A
    transpose on the way up moves the dim the pad grows, so the pad is rebuilt
    against the dim the source holds it in.

    Returns the tensor the op should read: ``node`` itself where the pad was
    hoisted (its chain now carries the wide tensor), and otherwise whatever
    padding ``node`` directly comes to.
    """
    dim = _padded_dim(pad)
    if dim is None:
        return _insert_pad(model, node, pad, pad_value, fold_cache)

    chain, src, src_dim, dims = _relayout_chain_above(node, dim)
    if not any(n.target in _BROADCAST_OPS for n in chain):
        return _insert_pad(model, node, pad, pad_value, fold_cache)

    padded = _insert_pad(
        model, src, _pad_for_dim(pad, dim, src_dim), pad_value, fold_cache
    )
    chain[-1].replace_input_with(src, padded)
    _restate_chain(chain, dims, padded.value.shape[src_dim])
    return node


def pad_input_node(model, node, is_weight, pad, scale_pad, fold_cache):
    if is_weight:
        input = node.args[1]
        scale = node.kwargs.get("weight_scale")
        code = node.kwargs.get("weight_code")
    else:
        input = node.args[0]
        scale = node.kwargs.get("input_scale")
        code = node.kwargs.get("input_code")

    pad_quantize_mx_input = (
        scale is not None
        and input.target == operator.getitem
        and scale.target == operator.getitem
        and input.args[0] == scale.args[0]
        and input.args[0].target == torch.ops.quantized_ops.quantize_mx.default
    )

    node_to_pad = input.args[0].args[0] if pad_quantize_mx_input else input

    # Padding the pre-quantize float contributes 0 with a plain 0 fill; padding
    # the quantized value pads *codebook indices*, so it must fill with the
    # index closest to 0 (NF4 keeps 0 at index 7, not 0 -- index 0 is -1.0), or
    # the padded contraction adds a nonzero term.
    pad_value = 0
    if not pad_quantize_mx_input and code is not None:
        pad_value = int(fetch_attr(model, code.target).abs().argmin())

    new_input = _hoist_pad_above_repeat(
        model, node_to_pad, pad, pad_value, fold_cache
    )

    if pad_quantize_mx_input:
        input.args[0].replace_input_with(node_to_pad, new_input)
        propagate_shape(input.args[0])
        propagate_shape(input)
        propagate_shape(scale)
    else:
        node.replace_input_with(node_to_pad, new_input)

        if scale is not None and any(x for x in scale_pad):
            node.replace_input_with(
                scale,
                _hoist_pad_above_repeat(model, scale, scale_pad, 0, fold_cache),
            )


#: Keyword operands of a matrix operation that carry one value per *output*
#: channel. Padding K widens that axis, and an operand describing it has to be
#: widened with it or the operation is handed a scale shorter than the tensor
#: it scales -- loudly, at the next shape propagation, and it was: a
#: vector-scaled convolution whose K went 48 -> 64 met a 48-entry coarse scale.
#:
#: Named rather than inferred from shape. An operand that happens to be K long
#: is not necessarily per output channel, and padding one that is not would be
#: silent where this is loud.
PER_OUTPUT_CHANNEL_OPERANDS = ("weight_coarse_scale",)


# Shape-preserving stages which may sit between a matrix result and a
# model-authored channel-tail pad on the pre-target graph.  Keep this list
# deliberately narrow: folding a pad through an operation is a physical
# ownership claim, not merely a claim that the two tensors have equal shapes.
_MATRIX_OUTPUT_PADDING_EPILOGUES = (
    torch.ops.quantized_ops.dequantize.default,
    torch.ops.aten.relu.default,
    torch.ops.quantized_ops.quantize.default,
)


def _constant_zero_channel_tail_pad(node):
    """Parse one model-authored rank-four NCHW zero tail pad."""
    if (
        node.target is not torch.ops.aten.pad.default
        or HARDWARE_PADDING_PAD in node.meta
    ):
        return None

    source = get_arg_value(node, 0, "input")
    pad = get_arg_value(node, 1, "pad")
    mode = get_arg_value(node, 2, "mode", "constant")
    value = get_arg_value(node, 3, "value")
    if not isinstance(source, Node):
        return None
    if mode not in (None, "constant"):
        return None
    if value is not None and (
        not isinstance(value, (int, float)) or value != 0
    ):
        return None

    source_shape = tuple(getattr(source, "shape", ()) or ())
    output_shape = tuple(getattr(node, "shape", ()) or ())
    if len(source_shape) != 4 or len(output_shape) != 4:
        return None
    if (
        not isinstance(pad, (tuple, list))
        or len(pad) % 2
        or len(pad) > 2 * len(source_shape)
    ):
        return None

    # Aten lists dimensions from the last toward the first.  NCHW's channel
    # pair is therefore the third pair in the list.
    channel_right_index = 2 * (len(source_shape) - 2) + 1
    if len(pad) <= channel_right_index:
        return None
    if any(
        amount != 0
        for index, amount in enumerate(pad)
        if index != channel_right_index
    ):
        return None

    tail = pad[channel_right_index]
    if isinstance(tail, bool) or not isinstance(tail, int) or tail <= 0:
        return None
    expected_shape = list(source_shape)
    expected_shape[1] += tail
    if tuple(expected_shape) != output_shape:
        return None
    return source, tail


def _matrix_source_for_output_pad(model, pad):
    """Trace a safe pre-target epilogue chain to one matrix producer."""
    parsed = _constant_zero_channel_tail_pad(pad)
    if parsed is None:
        return None
    current, _tail = parsed
    visited = set()
    while current not in visited:
        visited.add(current)
        if is_gemm_op(current) and not is_depthwise_conv(current):
            return current
        if current.target not in _MATRIX_OUTPUT_PADDING_EPILOGUES:
            return None
        source = get_arg_value(current, 0, "input")
        if not isinstance(source, Node):
            return None
        current_shape = tuple(getattr(current, "shape", ()) or ())
        source_shape = tuple(getattr(source, "shape", ()) or ())
        if (
            len(current_shape) != 4
            or current_shape != source_shape
        ):
            return None
        if current.target in {
            torch.ops.quantized_ops.dequantize.default,
            torch.ops.quantized_ops.quantize.default,
        }:
            if get_arg_value(current, 4, "block_size") is not None:
                return None
            if not _scale_is_per_tensor(current):
                # The padding pass can widen only a direct, static
                # per-output-channel dequantization scale.  Admitting any
                # other varying scale here would select a fold that
                # ``slice_output`` correctly refuses later.
                scale = get_arg_value(current, 1, "scale")
                if (
                    current.target
                    is not torch.ops.quantized_ops.dequantize.default
                    or not is_gemm_op(source)
                    or not isinstance(scale, Node)
                    or scale.op != "get_attr"
                    or not _only_read_by(model, scale, source)
                ):
                    return None
                scale_value = fetch_attr(model, scale.target)
                if (
                    scale_value.numel() == 1
                    or scale_value.ndim == 0
                    or scale_value.shape[0] != source_shape[1]
                ):
                    return None
        current = source
    return None


def _foldable_matrix_output_pads(model, config):
    """Select unambiguous authored pads a physical matrix output can own."""
    # Dtype-aware resident extents are the declaration that makes this
    # transform meaningful.  An ordinary flat config keeps the historical
    # padding path byte-for-byte unchanged.
    if config is None or not getattr(config, "dtype_modes", ()):
        return {}, set()

    candidates = {}
    for pad in model.graph.nodes:
        parsed = _constant_zero_channel_tail_pad(pad)
        if parsed is None:
            continue
        source, tail = parsed
        producer = _matrix_source_for_output_pad(model, pad)
        if producer is None:
            continue

        source_shape = tuple(getattr(source, "shape", ()) or ())
        producer_shape = tuple(getattr(producer, "shape", ()) or ())
        if source_shape != producer_shape or len(producer_shape) != 4:
            continue
        output_channels = producer_shape[1]
        resident_channels = output_channels + tail
        input_node = get_arg_value(producer, 0, "input")
        if not isinstance(input_node, Node):
            continue
        operand_bits = node_operand_bits(input_node)
        if (
            config.matrix_output_resident_channels(
                resident_channels,
                operand_bits,
            )
            != resident_channels
        ):
            continue
        default_resident = config.matrix_output_resident_channels(
            output_channels,
            operand_bits,
        )
        if default_resident > resident_channels:
            continue
        candidates.setdefault(producer, {}).setdefault(
            resident_channels,
            set(),
        ).add(pad)

    resident_overrides = {}
    folded_pads = set()
    for producer, by_resident in candidates.items():
        # Conflicting authored tails do not give the producer one physical
        # extent it can truthfully own, so leave every pad explicit.
        if len(by_resident) != 1:
            continue
        resident_channels, pads = next(iter(by_resident.items()))
        resident_overrides[producer] = resident_channels
        folded_pads.update(pads)
    return resident_overrides, folded_pads


def _fold_output_pad_as_view(pad, physical_source):
    """Turn an absorbed authored pad into a view of the wide producer."""
    logical_shape = tuple(getattr(pad, "shape", ()) or ())
    physical_shape = tuple(getattr(physical_source, "shape", ()) or ())
    if not logical_shape or logical_shape != physical_shape:
        raise ValueError(
            f"{pad.name}: folded output pad shape {logical_shape} does not "
            f"match physical producer shape {physical_shape}"
        )

    preserve_hardware_logical_shape(pad)
    pad.target = torch.ops.aten.view.default
    pad.args = (physical_source, physical_shape)
    pad.kwargs = {}
    propagate_shape(pad)
    pad.meta["dtype"] = physical_source.meta.get("dtype")


def _pad_per_output_channel_operands(model, node, output_channels, pad_K):
    """Zero-extend the matrix operation's per-output-channel scale operands.

    Zero for the added lanes, and the value is load-bearing rather than a
    filler: a channel the alignment invented is worth nothing, its weight is
    already zero, and a requantization encoder reads a zero factor as
    "nothing here".

    Only a static ``get_attr`` whose leading extent is the one being padded.
    Anything else is left alone and reports itself at the next shape
    propagation rather than being reshaped on a guess.
    """
    for name in PER_OUTPUT_CHANNEL_OPERANDS:
        operand = node.kwargs.get(name)
        if not isinstance(operand, Node) or operand.op != "get_attr":
            continue
        param = fetch_attr(model, operand.target)
        if param.numel() == 1 or param.shape[0] != output_channels:
            continue
        preserve_hardware_logical_shape(operand)
        tail = [0, 0] * (param.ndim - 1) + [0, pad_K]
        param.data = F.pad(param.data, tail, value=0.0)
        propagate_shape(operand, model)


def _only_read_by(model, parameter, producer) -> bool:
    """Whether ``parameter`` is read only by operations consuming ``producer``.

    A parameter is widened *in place* -- ``param.data = F.pad(...)`` -- so the
    buffer every ``get_attr`` for that target names changes at once. If some
    other operation also reads it and is not being widened, that operation
    silently acquires a longer scale than the tensor it scales.

    Checked over targets rather than nodes: two ``get_attr`` nodes may name
    one buffer, and mutating through either is the same mutation.
    """
    aliases = {
        n
        for n in model.graph.nodes
        if n.op == "get_attr" and n.target == parameter.target
    }
    consumers = set(producer.users)
    for alias in aliases:
        for reader in alias.users:
            if reader not in consumers:
                return False
    return True


def _pad_output_dequantize_scales(model, node, output_channels, pad_K):
    """Zero-fill a static per-channel dequantize scale for inactive lanes.

    Padding a matrix op's output channels leaves its consumers reading a
    tensor wider than the scale that describes it. The scale is per *channel*
    along exactly the axis that grew, so the two now disagree, and every
    reader downstream has to either be given a narrower tensor -- the
    restoring slice -- or a scale that covers what it will actually see.

    Zero for the added lanes, and that value is load-bearing rather than a
    filler: a channel the array computes only because the alignment required
    it is worth nothing, and a target's requantization encoder reads a zero
    factor as "nothing here". Any other fill would scale garbage.

    Only a static ``get_attr`` scale, and only one whose leading dimension is
    the extent that was padded. A scale this cannot place is left alone, and
    the restoring slice below stays where it is -- which is exactly today's
    behaviour, so an unfamiliar scale costs a slice and never a wrong number.
    """
    padded = set()
    for user in node.users:
        if user.target is not torch.ops.quantized_ops.dequantize.default:
            continue
        scale = get_arg_value(user, 1, "scale")
        if not isinstance(scale, Node) or scale.op != "get_attr":
            continue
        if scale.target in padded:
            continue
        param = fetch_attr(model, scale.target)
        if param.numel() == 1:
            continue
        if param.shape[0] == output_channels + pad_K:
            padded.add(scale.target)
            continue
        if param.shape[0] != output_channels:
            logger.warning(
                "%s: leaving %s unpadded -- leading dimension %d is neither "
                "the %d channels being padded nor the %d they become",
                node.name, scale.target, param.shape[0], output_channels,
                output_channels + pad_K,
            )
            continue
        if not _only_read_by(model, scale, node):
            logger.warning(
                "%s: leaving %s unpadded -- it is read by an operation this "
                "padding does not widen, and the buffer is shared",
                node.name, scale.target,
            )
            continue
        aliases = tuple(
            candidate
            for candidate in model.graph.nodes
            if candidate.op == "get_attr" and candidate.target == scale.target
        )
        for alias in aliases:
            preserve_hardware_logical_shape(alias)
        tail = [0, 0] * (param.ndim - 1) + [0, pad_K]
        param.data = F.pad(param.data, tail, value=0.0)
        # Every get_attr names the same mutated buffer. Refresh every alias;
        # leaving one at its old shape makes a later reader see a stale K16
        # node over a K32 tensor.
        for alias in aliases:
            propagate_shape(alias, model)
        padded.add(scale.target)


def _scale_is_per_tensor(user) -> bool:
    """Whether one quantize's scale is a single value for the whole tensor.

    Conservative on purpose: a scale whose value cannot be read is treated as
    varying.  The cost of refusing is one slice node that a later pass may
    fold; the cost of crossing wrongly is a scale applied to the wrong
    channels, which no later pass can detect.
    """
    scale = get_arg_value(user, 1, "scale")
    if scale is None:
        return True
    if isinstance(scale, (int, float)):
        return True  # a plain number is one value for the whole tensor
    value = getattr(scale, "value", None)
    if value is None and hasattr(scale, "meta"):
        value = scale.meta.get("val")
    if value is None or not hasattr(value, "numel"):
        return False
    return value.numel() == 1


def _scale_covers(node, user, slice_args) -> bool:
    """Whether ``user``'s per-channel scale already spans the unsliced tensor.

    The companion to ``_scale_is_per_tensor``, and the other way an op can be
    indifferent to the lanes a pad added. A per-tensor scale is indifferent
    because it has nothing to line up; a per-channel scale is indifferent once
    it has been *given* an entry per lane, which is what
    ``_pad_output_dequantize_scales`` does immediately above the slice that
    would otherwise be needed. The added entries are zero, so the lanes they
    describe come out zero and the slice below removes them unchanged.

    Measured against the axis ``slice_args`` names, not against whichever
    dimension happens to match. Those are different questions: a scale as long
    as some *other* axis says nothing about the one being cut, and crossing on
    that coincidence would leave a scale lined up with the wrong dimension --
    silently, since the shapes would still broadcast.
    """
    scale = get_arg_value(user, 1, "scale")
    if not isinstance(scale, Node):
        return False
    value = getattr(scale, "value", None)
    if value is None and hasattr(scale, "meta"):
        value = scale.meta.get("val")
    if value is None or not hasattr(value, "shape") or not len(value.shape):
        return False
    resident = tuple(getattr(node, "shape", ()) or ())
    dim = slice_args[0]
    if not resident or not -len(resident) <= dim < len(resident):
        return False
    return int(value.shape[0]) == int(resident[dim])


def _crossable(node, user, slice_args):
    """Whether ``user`` can read ``node`` unsliced -- and widen it to, if so.

    An op that keeps every lane to itself does not care that the pad left extra
    ones, so the slice can sink past it and be spent nearer a pad that cancels
    it.  A per-tensor quantize is such an op; a blocked one is not, a slice
    below it cutting the blocks it scaled.  A second operand agrees only if it
    is already sliced the same way -- and stripping those slices is what widens
    the op.

    A **per-channel** quantize is not such an op either, and testing only for
    blocking missed that: its scale has one entry per channel along exactly
    the dimension the slice cuts, so sinking the slice past it leaves an
    8-element scale multiplying a 32-channel tensor.  Reachable only by a
    matrix op narrower than the array's output width *and* a per-channel
    weight scale -- ResNet-18 has the second and not the first, YOLOv5n-0.5's
    8-channel stem has both.
    """
    # Only support non-MX quantization
    if user.target in [
        torch.ops.quantized_ops.dequantize.default,
        torch.ops.quantized_ops.quantize.default,
    ]:
        if get_arg_value(user, 4, "block_size") is not None:
            return False
        can_cross = _scale_is_per_tensor(user) or _scale_covers(
            node, user, slice_args
        )
        if can_cross:
            preserve_hardware_logical_shape(user)
            propagate_shape(user)
        return can_cross

    if not is_elementwise_op(user):
        return False

    if len(user.all_input_nodes) != 1:
        if not all(
            n == node
            or (
                n.target == torch.ops.aten.slice.Tensor
                and n.args[1:] == slice_args
            )
            for n in user.all_input_nodes
        ):
            return False

        for n in user.all_input_nodes:
            if n.target == torch.ops.aten.slice.Tensor:
                user.replace_input_with(n, n.args[0])

    preserve_hardware_logical_shape(user)
    propagate_shape(user)
    return True


def slice_output(model, node, slice_args, folded_output_pads=None):
    if folded_output_pads is None:
        folded_output_pads = set()
    # The loop rebinds ``node`` as it sinks; the logical extent belongs to
    # where the walk started.
    origin = node
    frontier = [node]

    while frontier:
        node = frontier.pop()
        needs_slice = []
        for user in list(node.users.keys()):
            if user in folded_output_pads:
                _fold_output_pad_as_view(user, node)
                continue
            if _crossable(node, user, slice_args):
                frontier.append(user)
            else:
                needs_slice.append(user)
        if not needs_slice:
            continue

        with model.graph.inserting_after(node):
            slice_node = model.graph.call_function(
                torch.ops.aten.slice.Tensor, (node, *slice_args)
            )

        propagate_shape(slice_node)
        slice_node.meta["dtype"] = node.meta.get("dtype")
        # This slice is the compiler undoing its own widening, not something
        # the model asked for -- and the two are the same operator with the
        # same arguments, so the difference has to be said rather than
        # inferred. Tagged here, at the point of creation, because this walk
        # sinks past crossable users and may insert several, on descendants
        # rather than on the origin.
        slice_node.meta[HARDWARE_PADDING_SLICE] = True
        inherit_hardware_logical_shape(slice_node, origin)

        for n in needs_slice:
            n.replace_input_with(node, slice_node)


def _matrix_unrolls(config, input_node, C_unroll, K_unroll, is_matrix=True):
    """The unrolling one operation actually meets, and at what operand width.

    A narrower operand packs more reduction lanes into the same array and can
    change the output-channel granularity the datapath requires, so the pair a
    layer is padded to is a function of the operand's width and not of the
    machine alone. A config that declares ``dtype_modes`` says what those are;
    one that declares none has a single implicit mode, and the flat pair the
    caller passed *is* that mode.

    ``is_matrix`` is false for a depthwise convolution, which on a machine
    with two units is not the matrix unit's work at all -- Alpha runs it on
    the near-memory processor, whose channel parallelism is its own and is not
    the reduction lanes of an array it never enters. Answering with the matrix
    modes there would be describing the wrong unit, so the flat pair stands
    until the unit that runs it is the one asked.

    The flat path is taken without reading the node at all. That is the point:
    every configuration on the shared path declares no modes, so this pass
    neither consults the config nor probes a dtype there, and cannot fail on
    an operand whose width it would otherwise have had to name.

    Returns:
        ``(C_unroll, K_unroll, operand_bits)``, with ``operand_bits`` ``None``
        on the flat path -- which is what tells the caller the config has no
        opinion about the resident extent either.
    """
    declares_modes = config is not None and getattr(
        config, "dtype_modes", ()
    )
    if not is_matrix or not declares_modes:
        return C_unroll, K_unroll, None
    operand_bits = node_operand_bits(input_node)
    return (
        config.input_lanes(operand_bits),
        config.output_lanes(operand_bits),
        operand_bits,
    )


def pad_matrix_op_dimensions(
    model: GraphModule,
    C_unroll,
    K_unroll,
    fold_cache: bool = FOLD_PAD_INTO_CACHE,
    config=None,
) -> GraphModule:
    """
    Pad inputs and weights to conv2d nodes in a torch.fx.GraphModule so that
    the input channels (C) and output channels (K) are multiples of the
    provided unroll factors.

    Parameters:
        model (torch.fx.GraphModule): The FX graph module to transform.
        C_unroll (int): Unroll factor for the input channels (C_in). Used
            as given unless ``config`` declares dtype modes, in which case the
            machine's unrolling at the operand's own width takes over.
        K_unroll (int): Unroll factor for the output channels (C_out), on the
            same terms.
        config: Accelerator config, consulted for whether the matrix unit can
            mask unused input lanes, and -- when it declares dtype modes --
            for the unrolling and the resident output extent at the operand's
            width. Without one the historical rule stands: never pad a 1- or
            3-channel first layer, and pad to the flat factors given.

    Returns:
        torch.fx.GraphModule: The transformed FX graph module.
    """
    resident_overrides, folded_output_pads = (
        _foldable_matrix_output_pads(model, config)
    )
    for node in list(model.graph.nodes):
        if not is_gemm_op(node):
            continue

        is_dw = is_depthwise_conv(node)
        is_conv = is_conv2d(node)
        is_mm = is_matmul(node)

        input = node.args[0]
        C_in = input.shape[1] if is_conv else input.shape[-1]

        # What this operation's operand width makes of the array. Resolved per
        # node rather than once per graph because two layers of one model may
        # run at different widths on the same machine.
        node_C_unroll, node_K_unroll, operand_bits = _matrix_unrolls(
            config, input, C_unroll, K_unroll, is_matrix=not is_dw
        )

        # A narrow first layer is only left unpadded when the matrix unit can
        # mask the lanes it does not use.  A unit with no partial-input path
        # would read those lanes as garbage, so its operand is padded up to
        # the array width instead -- which is also what lets the layer reach
        # the tile search at all, since a 3-deep reduction on a 16-wide array
        # is not a mapping the scheduler will accept.
        #
        # It exempts the *reduction* axis and nothing else. Skipping the whole
        # operation was the earlier reading and it is wrong on a machine whose
        # output alignment is its own number: a 3-channel layer with 8 output
        # channels needs no input padding and still needs its K rounded to a
        # resident extent the datapath can hold. The two axes are padded for
        # unrelated reasons and a masked input says nothing about the output.
        if is_conv and config is not None:
            partial_input = config.supports_partial_matrix_input(C_in)
        else:
            partial_input = is_conv and C_in in COMMON_IMAGE_INPUT_CHANNELS

        pad_C = 0 if partial_input else (-C_in) % node_C_unroll

        bs = node.kwargs.get("block_size", 1)

        # Pad input along input channel dimension
        if pad_C:
            lead = [0, 0, 0, 0] if is_conv else []
            in_pad = lead + [0, pad_C]
            in_scale_pad = lead + [0, pad_C // bs]
            # Before, not after: pad_input_node may bake the widened tensor
            # into a new parameter, and the extent is gone the moment it does.
            activation = get_arg_value(node, 0, "input")
            if isinstance(activation, Node):
                preserve_hardware_logical_shape(activation)
            pad_input_node(model, node, False, in_pad, in_scale_pad, fold_cache)

        weight = node.args[1]
        C_in = weight.shape[-2] if is_mm else weight.shape[1]
        C_out = weight.shape[-1] if is_mm else weight.shape[0]

        if is_dw:
            C_in *= node.args[6]
            node.args = node.args[:-1] + (node.args[-1] + pad_C,)

        pad_C = 0 if partial_input else (-C_in) % node_C_unroll
        if operand_bits is None:
            pad_K = (-C_out) % node_K_unroll
        else:
            # Not ``(-C_out) % lanes``: a machine may accept a resident K
            # narrower than its datapath by masking the lanes above it, and
            # rounding to the lane count would emit a weight wider than the
            # one the endpoint reads. The machine answers what extent is
            # legal; this only takes the difference.
            pad_K = (
                resident_overrides.get(
                    node,
                    config.matrix_output_resident_channels(
                        C_out,
                        operand_bits,
                    ),
                )
                - C_out
            )

        if is_dw:
            pad_K = pad_C

        # Pad weight along input and output channel dimensions
        if pad_C or pad_K:
            if is_dw:
                w_pad = [0, 0, 0, 0, 0, 0, 0, pad_K]
                ws_pad = [0, 0, 0, 0, 0, 0, 0, pad_K]
            elif is_conv:
                w_pad = [0, 0, 0, 0, 0, pad_C, 0, pad_K]
                ws_pad = [0, 0, 0, 0, 0, pad_C // bs, 0, pad_K]
            elif is_mm:
                w_pad = [0, pad_K, 0, pad_C]
                ws_pad = [0, pad_K, 0, pad_C // bs]
            else:
                w_pad = [0, pad_C, 0, pad_K]
                ws_pad = [0, pad_C // bs, 0, pad_K]
            weight_scale = node.kwargs.get("weight_scale")
            preserve_hardware_logical_shape(weight)
            if isinstance(weight_scale, Node):
                preserve_hardware_logical_shape(weight_scale)
            pad_input_node(model, node, True, w_pad, ws_pad, fold_cache)
            if pad_K:
                # The fine scale rode along above, blocked with the weight. A
                # coarse one is per output channel and has to be widened on
                # its own axis.
                _pad_per_output_channel_operands(model, node, C_out, pad_K)

        bias = get_arg_value(node, 2, "bias")
        if pad_K and bias is not None:
            preserve_hardware_logical_shape(bias)
            new_bias = _insert_pad(model, bias, [0, pad_K], 0)
            # The replacement is the padded value; it has to carry the extent
            # forward or the record loses it one node later.
            inherit_hardware_logical_shape(new_bias, bias)
            node.replace_input_with(bias, new_bias)

        # The operation's own extent, recorded before its shape is
        # re-propagated against the widened operands. This is the one a
        # target reads to say how many of its resident channels are real.
        preserve_hardware_logical_shape(node)
        propagate_shape(node)
        if pad_K:
            # Before the slice, not after: whether the slice is needed at all
            # depends on whether the consumers can read the wider tensor, and
            # a per-channel scale is what decides that.
            _pad_output_dequantize_scales(model, node, C_out, pad_K)
            slice_dim = 1 if is_conv else -1
            slice_output(
                model,
                node,
                (slice_dim, 0, C_out),
                folded_output_pads,
            )

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def _pad_layer_norm(
    model: GraphModule,
    node: Node,
    unroll: int,
) -> GraphModule:
    input = node.args[0]
    normalize_shape = node.args[1]
    weight = node.args[2]
    bias = node.args[3] if len(node.args) > 3 else None

    orig_k = input.shape[-1]
    pad_k = (-orig_k) % unroll
    if pad_k == 0:
        return model

    logger.info(f"Padding layer_norm {node} last dimension with {pad_k}")

    def pad_param(attr_node: Node):
        if attr_node.op == "get_attr":
            param = fetch_attr(model, attr_node.target)
            new_param = F.pad(param, [0, pad_k])
            with model.graph.inserting_after(attr_node):
                new_attr = create_getattr_from_value(
                    model, model.graph, f"{attr_node.target}_padded", new_param
                )
        else:
            with model.graph.inserting_after(attr_node):
                new_attr = model.graph.call_function(
                    torch.ops.aten.pad.default,
                    (attr_node, [0, pad_k]),
                )
        propagate_shape(new_attr, model)
        return new_attr

    new_weight = pad_param(weight)
    new_bias = pad_param(bias) if bias is not None else None

    with model.graph.inserting_before(node):
        new_input = model.graph.call_function(
            torch.ops.aten.pad.default,
            (input, [0, pad_k]),
        )
        layer_norm = model.graph.call_function(
            torch.ops.quantized_ops.layer_norm.default,
            (new_input, normalize_shape, new_weight, new_bias) + node.args[4:],
        )
        slice_node = model.graph.call_function(
            torch.ops.aten.slice.Tensor,
            (layer_norm, -1, 0, orig_k),
        )

    propagate_shape(new_input)
    new_input.meta["dtype"] = input.meta.get("dtype")

    propagate_shape(layer_norm)

    propagate_shape(slice_node)
    node.replace_all_uses_with(slice_node)
    model.graph.erase_node(node)


def _pad_quantize_mx(model, node, unroll, fold_cache):
    input = node.args[0]
    axes = get_arg_value(node, 2, "axes")
    block_size = get_arg_value(node, 3, "block_size")
    ndim = len(input.shape)
    axes = {a % ndim for a in axes}

    # A tile boundary on the last dim must land on a hardware-unroll multiple,
    # and no quantization block may straddle two tiles, so each quant axis must
    # land on a block multiple.  A dim that is both must satisfy their lcm.  We
    # do not rely on the GEMM input padding to have aligned the last dim.
    pad_dims = {}
    for i in range(ndim):
        multiple = 1
        if i == ndim - 1:
            multiple = unroll
        if i in axes:
            multiple = math.lcm(multiple, block_size)
        if multiple > 1 and input.shape[i] % multiple:
            pad_dims[i] = (-input.shape[i]) % multiple

    if not pad_dims:
        return

    logger.info(f"Padding quantize_mx {node} with {pad_dims}")

    getitems = list(node.users)
    orig_shapes = {g: tuple(g.shape) for g in getitems}

    min_pad_dim = min(pad_dims)
    pad_tuple = []
    for dim in range(ndim - 1, min_pad_dim - 1, -1):
        pad_tuple.extend([0, pad_dims.get(dim, 0)])

    new_input = _insert_pad(model, input, pad_tuple, 0, fold_cache)
    node.replace_input_with(input, new_input)
    propagate_shape(node)

    # Slice each output back to its pre-pad shape, but only along dims that
    # actually grew: the quantized value carries every padded dim, whereas the
    # scale grows only where the pad added whole blocks or non-quant columns.
    for g in getitems:
        propagate_shape(g)
        orig = orig_shapes[g]
        users = list(g.users)
        sliced = g
        for d in range(len(orig)):
            if g.shape[d] != orig[d]:
                with model.graph.inserting_after(sliced):
                    sliced = model.graph.call_function(
                        torch.ops.aten.slice.Tensor,
                        (sliced, d, 0, orig[d]),
                    )
                propagate_shape(sliced)
                sliced.meta["dtype"] = g.meta.get("dtype")
        if sliced is not g:
            for u in users:
                u.replace_input_with(g, sliced)


def _pad_softmax(model, node, unroll):
    input = node.args[0]

    # The hardware fetches in units of ``unroll`` elements along the last
    # dim, so pad it to a multiple of ``unroll`` with -inf and slice back.
    orig = input.shape[-1]
    pad_k = (-orig) % unroll
    if pad_k == 0:
        return

    logger.info(f"Padding softmax {node} with {pad_k}")

    with model.graph.inserting_after(input):
        new_input = model.graph.call_function(
            torch.ops.aten.pad.default,
            (input, [0, pad_k], "constant", float("-inf")),
        )
    propagate_shape(new_input)
    new_input.meta["dtype"] = input.meta.get("dtype")
    node.replace_input_with(input, new_input)
    propagate_shape(node)

    with model.graph.inserting_after(node):
        slice_node = model.graph.call_function(
            torch.ops.aten.slice.Tensor,
            (node, -1, 0, orig),
        )
    node.replace_all_uses_with(slice_node)
    slice_node.replace_input_with(slice_node, node)
    propagate_shape(slice_node)
    slice_node.meta["dtype"] = node.meta.get("dtype")


def pad_vector_op_dimensions(
    model: GraphModule,
    K_unroll,
    fold_cache: bool = FOLD_PAD_INTO_CACHE,
) -> GraphModule:
    """
    Pad inputs to vector operations to multiples of the hardware unroll size.
    Only support softmax operation for now.

    Parameters:
        model (torch.fx.GraphModule): The FX graph module to transform.
        K_unroll (int): Unroll factor for the output channels.

    Returns:
        torch.fx.GraphModule: The transformed FX graph module.
    """
    for node in list(model.graph.nodes):
        if node.target == torch.ops.aten.layer_norm.default:
            _pad_layer_norm(model, node, K_unroll)
        elif node.target == torch.ops.aten.softmax.int:
            _pad_softmax(model, node, K_unroll)
        elif node.target == torch.ops.quantized_ops.quantize_mx.default:
            _pad_quantize_mx(model, node, K_unroll, fold_cache)

    for node in list(model.graph.nodes):
        if node.target is not torch.ops.aten.pad.default:
            continue

        pad = get_arg_value(node, 1, "pad")
        cancel, chain, dims = _find_cancelling_slice(node.args[0], pad)
        if cancel is None:
            continue

        logger.info(f"Eliminating slice / pad round-trip: {cancel} and {node}")
        if chain:
            _drop_slice_through_chain(cancel, chain, dims)
            node.replace_all_uses_with(node.args[0])
        else:
            node.replace_all_uses_with(cancel.args[0])

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def pad_vit_embeddings_output(
    model: GraphModule,
    embeddings,
    example_inputs,
    dynamic_shapes=None,
    unroll=32,
):
    original_graph = model.graph

    pattern = get_aten_graph_module(
        embeddings, example_inputs, dynamic_shapes=dynamic_shapes
    )
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

    if not _matches:
        return model

    vit_embed_out = _matches[0].returning_nodes[0]
    orig_dim = vit_embed_out.meta["val"].shape[-2]
    pad = (unroll - (orig_dim % unroll)) % unroll
    logger.info(f"Padding {vit_embed_out} with {pad}")

    with model.graph.inserting_after(vit_embed_out):
        pad_node = model.graph.call_function(
            torch.ops.aten.pad.default,
            (vit_embed_out, [0, 0, 0, pad]),
        )

    propagate_shape(pad_node)
    pad_node.meta["val"] = pad_node.value

    for user in list(vit_embed_out.users):
        if id(user) != id(pad_node):
            user.replace_input_with(vit_embed_out, pad_node)

    for node in model.graph.nodes:
        if node.target in [
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
        ]:
            new_size = [x if x != orig_dim else x + pad for x in node.args[1]]
            node.args = (node.args[0], new_size)

    model.graph.lint()
    model.recompile()
    return model
