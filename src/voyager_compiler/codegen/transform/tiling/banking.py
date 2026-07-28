from __future__ import annotations

import logging
import math

import torch

from ...node_info import (
    _align_size,
    get_anchor_node,
    get_node_to_key_map,
    is_gemm_op,
    quant_param_arg_nodes,
)
from ....pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)


def _find_user_target(node: torch.fx.Node, targets):
    for user in node.users:
        if user.target in targets and user.args[0] == node:
            return user.target

        # Check for users of fused dequantization nodes
        if (
            user.target == torch.ops.quantized_ops.dequantize.default
            and user.meta.get("fused") is True
        ):
            found = _find_user_target(user, targets)
            if found is not None:
                return found

        if user.op == "call_module":
            gm = user.meta["submodule"]
            placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
            idx = next(i for i, arg in enumerate(user.args) if arg is node)
            found = _find_user_target(placeholders[idx], targets)
            if found is not None:
                return found

    return None


ALLOC_TARGETS = (
    torch.ops.aten.conv2d.default,
    torch.ops.quantized_ops.conv2d.default,
    torch.ops.aten.softmax.int,
    torch.ops.aten.layer_norm.default,
)


def compute_tensor_size(
    node, shape, is_output=False, bank_width=None, vector_lanes=None
):
    val = node.value
    if isinstance(val, torch.Tensor):
        dtype = node.meta.get("dtype") or val.dtype
        numel = math.prod(shape) if shape is not None else val.numel()
        tensor_size = numel * dtype_byte_size(dtype)

        target = _find_user_target(node, ALLOC_TARGETS)

        # Extra space for a conv2d's replicated input, and for the
        # intermediates a softmax / layer_norm keeps beside its own input.
        if target in ALLOC_TARGETS[:2]:
            dim = 1 if target == ALLOC_TARGETS[0] else -1
            if val.shape[dim] == 3:
                logger.debug(f"Increase space for conv2d input {node}")
                tensor_size *= 3
        elif not is_output:
            if target == ALLOC_TARGETS[2]:
                logger.debug(f"Increase space for softmax input {node}")
                tensor_size = tensor_size * 2
            elif target == ALLOC_TARGETS[3]:
                logger.debug(f"Increase space for layer_norm input {node}")
                tensor_size = (tensor_size + numel) * 2

        return _align_size(tensor_size, bank_width)

    if isinstance(val, (tuple, list)):
        if shape is not None:
            numel = [math.prod(s) for s in shape]
        else:
            numel = [t.numel() for t in val]

        # Sparse outputs need to be aligned with hardware unroll dimension
        if vector_lanes is not None:
            numel = [_align_size(s, vector_lanes) for s in numel]

        dtypes = node.meta.get("dtype") or [None for _ in val]
        sizes = [
            _align_size(n * dtype_byte_size(dt or t.dtype), bank_width)
            for t, n, dt in zip(val, numel, dtypes)
        ]

        return sum(sizes)

    logger.warning(f"Node {node} has a non-tensor output")
    return None


def require_allocation(node: torch.fx.Node) -> bool:
    # A consumer's quantization lookup table is resident, not a streamed tile.
    # Ask every op that reads it: in a fused kernel the table belongs to the
    # tail's ``quantize_mx`` as often as to the anchor, and its params are
    # placeholders that resolve back through ``meta['source_node']``.
    for user in node.users:
        gm = user.meta.get("submodule")
        for op in gm.graph.nodes if gm is not None else [user]:
            params = quant_param_arg_nodes(op)
            if any(p.meta.get("source_node", p) is node for p in params):
                return False

    if (val := getattr(node, "value", None)) is None:
        return True

    if not isinstance(val, torch.Tensor):
        return False

    if node.op in ["placeholder", "get_attr"] and val.numel() == 1:
        return False

    return True


# One bank per operand *group*, grouped by the read bandwidth each needs.  A
# GEMV streams its weight -- a fresh element per MAC, nothing reused -- so it
# takes a bank of its own.  The input and its scales are reused across
# ``vector_lanes`` outputs, so one fetch feeds a whole lane group and they need
# a fraction of the weight's bandwidth: nothing like a GEMM, where the input
# streams too.  The weight scales do matter, one per block, but the output is
# only written on the step that finishes a tile, so the two rarely contend.
GEMV_BANK_GROUPS = (
    ("input", "input_scale"),
    ("weight", "other"),
    ("output", "weight_scale", "bias"),
    ("A_data", "A_indices", "A_indptr"),
)


def operand_roles(node) -> dict:
    """Each operand FX node -> the role it plays, for grouping into banks.

    A fused ``call_module``'s operands have no roles of their own, so read them
    off the anchor: ``get_node_to_key_map`` traces each placeholder back through
    ``meta['source_node']`` to the outer node the shape map is keyed by.  What
    the tail brought of its own the anchor never reads, so it stays unnamed and
    takes a bank of its own.  A bare node is its own anchor.
    """
    anchor = get_anchor_node(node)
    if not is_gemm_op(anchor):
        return {}
    roles = get_node_to_key_map(anchor)
    # The output is named on the outer node -- what the shape map keys it by --
    # in place of the anchor's own entry, which is inside the submodule.
    del roles[anchor]
    roles[node] = "output"
    return roles


def scratchpad_bytes(
    node,
    tiled_shapes,
    bank_width,
    bank_size,
    num_banks,
    extra_sharing=0,
    vector_lanes=None,
):
    """Scratchpad bytes one candidate tile occupies.

    Every group takes a whole bank, since a bank cannot be split, and the two
    smallest merge while the groups outnumber the banks.
    ``GEMV_BANK_GROUPS`` names one op's roles, so an operand it misses -- a
    ``where`` mask, what a fused tail brings of its own -- takes a bank of its
    own.

    Args:
        node: The op being tiled; its own entry is the output.
        tiled_shapes: Operand FX node -> tiled shape.
        bank_width: Per-tensor alignment, bytes.
        bank_size: Bytes per bank; ``None`` is unbanked, so just sum.
        num_banks: Banks available to merge down to.
        extra_sharing: Merge this many groups further.  ``G`` groups floor the
            footprint at ``G * bank_size``, so at ``G == num_banks`` no tile
            fits however small and the caller has to raise it.
        vector_lanes: Sparse-output alignment, elements.

    Returns:
        Bytes the tile needs with every group rounded to whole banks.
    """
    # ``groups`` is the bank layout, one entry per bank: the groups that match,
    # then any operand they do not name on its own.  A ``where`` lays out as
    # ("input",) | ("other", ...) | ("output", ...) | ("condition",).
    roles = operand_roles(node)
    groups = [
        [n for n in tiled_shapes if roles.get(n) in group]
        for group in GEMV_BANK_GROUPS
    ]
    grouped = {n for group in groups for n in group}
    groups += [[n] for n in tiled_shapes if n not in grouped]

    sizes = []
    for group in groups:
        total = 0
        for n in group:
            shape = tiled_shapes[n]
            if shape is None or not require_allocation(n):
                continue
            total += compute_tensor_size(
                n, shape, n is node, bank_width, vector_lanes
            )
        if total:
            sizes.append(total)

    if not sizes:
        return 0
    if not bank_size:
        return sum(sizes)

    target = len(sizes)
    if num_banks:
        target = num_banks
    target = max(1, target - extra_sharing)
    while len(sizes) > target:
        sizes.sort()
        sizes = [sizes[0] + sizes[1]] + sizes[2:]

    return sum(math.ceil(s / bank_size) * bank_size for s in sizes)
