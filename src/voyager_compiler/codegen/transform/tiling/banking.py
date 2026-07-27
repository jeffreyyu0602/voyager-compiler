from __future__ import annotations

import logging
import math
import re

import torch

from ...node_info import _align_size
from ....pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)


def _find_user_with_target(node: torch.fx.Node, targets):
    if not isinstance(targets, set):
        if isinstance(targets, (list, tuple)):
            targets = set(targets)
        else:
            targets = {targets}

    for user in node.users:
        if user.target in targets and user.args[0] == node:
            return user

        # Check for users of fused dequantization nodes
        if (
            user.target == torch.ops.quantized_ops.dequantize.default
            and user.meta.get("fused") is True
        ):
            found = _find_user_with_target(user, targets)
            if found is not None:
                return found

        if user.op == "call_module":
            # Map this node to the submodule placeholder by argument
            # position, not by name: a producer can be renamed in the parent
            # graph after fusion (e.g. by extract_input_preprocessor) while
            # the submodule placeholder keeps its original name.
            gm = user.meta["submodule"]
            placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
            idx = next(i for i, arg in enumerate(user.args) if arg is node)
            found = _find_user_with_target(placeholders[idx], targets)
            if found is not None:
                return found

    return None


def compute_tensor_size(
    node,
    shape=None,
    is_scratchpad_output=False,
    bank_width=None,
    unroll_dim=None,
):
    val = node.value
    if isinstance(val, torch.Tensor):
        dtype = node.meta.get("dtype") or val.dtype
        numel = math.prod(shape) if shape is not None else val.numel()
        tensor_size = numel * dtype_byte_size(dtype)

        conv_targets = (
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_ops.conv2d.default,
        )
        conv_user = _find_user_with_target(node, conv_targets)

        if conv_user is not None:
            dim = 1 if conv_user.target == torch.ops.aten.conv2d.default else -1
            if val.shape[dim] == 3:
                logger.debug(f"Increase memory for conv2d input {node} by 3x")
                tensor_size *= 3

        # Allocate extra memory for intermediate results like mean and variance.
        # TODO: Should only do this when allocating scratchpad memory for the
        # specific operation. E.g. if a node is consumed by both a softmax and
        # an add node, we shouldn't increase the size for the add path.
        if not is_scratchpad_output:
            if _find_user_with_target(node, torch.ops.aten.softmax.int):
                logger.debug(f"Increase memory for softmax input {node} by 2x")
                tensor_size = tensor_size * 2

            if _find_user_with_target(node, torch.ops.aten.layer_norm.default):
                logger.debug(
                    f"Increase memory for layer_norm input {node} by 2x"
                )
                tensor_size = (tensor_size + numel) * 2

        return _align_size(tensor_size, bank_width)

    if isinstance(val, (tuple, list)):
        if shape is not None:
            key = "tiled_output_sizes"
            numel = [math.prod(s) for s in shape]
        else:
            key = "output_sizes"
            numel = [t.numel() for t in val]

        # Sparse outputs need to be aligned with hardware unroll dimension
        if unroll_dim is not None:
            numel = [_align_size(s, unroll_dim) for s in numel]

        dtypes = node.meta.get("dtype") or [None for _ in val]

        sizes = [
            _align_size(n * dtype_byte_size(dt or t.dtype), bank_width)
            for t, n, dt in zip(val, numel, dtypes)
        ]

        node.meta[key] = tuple(sizes)
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
    unroll_dim=None,
):
    """Scratchpad bytes one candidate tile occupies.

    Each group of ``BANK_GROUPS`` that has any tensor gets a bank of its own
    and rounds up to a whole one, since a bank cannot be split between groups.
    With more groups than banks the two *smallest* are merged and the merge
    repeats until they fit -- a few hundred bytes of block scales would
    otherwise cost a whole bank while a real operand had none.

    ``extra_sharing`` merges that many groups further.  It is not a nicety:
    every group costs a whole bank, so ``G`` groups put a floor of
    ``G * bank_size`` on the footprint, and at ``G == num_banks`` that floor
    is the entire scratchpad -- no tile fits, however small.  The caller
    raises ``extra_sharing`` until something maps.

    ``tiled_shapes`` is keyed by operand role (what ``shape_func`` returns);
    ``key_to_node`` maps a role to its FX node, which carries the dtype and
    the sizing allowances ``compute_tensor_size`` applies.  ``bank_size`` of
    ``None`` means the design is unbanked: just sum the tiles.
    """
    sizes = []
    for group in BANK_GROUPS:
        total = 0
        for key in group:
            shape = tiled_shapes.get(key)
            n = key_to_node.get(key)
            if shape is None or n is None or not require_allocation(n):
                continue
            total += compute_tensor_size(
                n, shape, n is node, bank_width, unroll_dim
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
