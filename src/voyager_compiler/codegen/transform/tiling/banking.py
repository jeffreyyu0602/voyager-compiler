from __future__ import annotations

import logging
import math
import re

import torch

from ...node_info import _align_size
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
    if re.fullmatch(r"(code|qmap)(_\d+)?", node.name):
        return False

    val = getattr(node, "value", None)
    if val is None:
        return True

    if not isinstance(val, torch.Tensor):
        return False

    if node.op == "get_attr" and val.numel() == 1:
        return False

    return True


# One bank per operand *group*: a tensor shares a bank with the metadata that
# is only ever read alongside it -- a weight with its block scales and bias,
# an output with its output scales, a CSR matrix with its index arrays.  The
# activation and its block scales stay apart: the vector unit reads them on
# separate ports.
BANK_GROUPS = (
    ("input",),
    ("input_scale",),
    ("weight", "other", "weight_scale", "bias"),
    ("A_data", "A_indices", "A_indptr"),
    ("output", "output_scale"),
)


def scratchpad_bytes(
    key_to_node,
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
    smallest merge while the groups outnumber the banks.  ``BANK_GROUPS`` names
    one op's roles, so a role it misses -- a ``where`` mask, a fused group's
    extra inputs -- takes a bank of its own.

    Args:
        key_to_node: Operand role -> FX node, which carries the dtype.
        node: The op being tiled; its own entry is the output.
        tiled_shapes: Operand role -> tiled shape.
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
    # ``groups`` is the bank layout, one entry per bank: the BANK_GROUPS that
    # match, then any unnamed role on its own.  A ``where`` lays out as
    # ("input",) | ("other", ...) | ("output", ...) | ("condition",).
    named = {key for group in BANK_GROUPS for key in group}
    groups = BANK_GROUPS + tuple(
        (key,) for key in tiled_shapes if key not in named
    )

    sizes = []
    for group in groups:
        total = 0
        for key in group:
            shape = tiled_shapes.get(key)
            n = key_to_node.get(key)
            if shape is None or n is None or not require_allocation(n):
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
