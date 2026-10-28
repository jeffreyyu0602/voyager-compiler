# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Target-owned recognition of Magnet Alpha source graph forms."""

from typing import Optional

import torch
from torch.fx import Node

from voyager_compiler.codegen.node_info import get_arg_value

__all__ = ["is_channel_shuffle_cat"]


def _resolved_view_shape(node: Node) -> Optional[tuple[int, ...]]:
    """Return an exact view shape, resolving PyTorch's single ``-1``."""
    declared = get_arg_value(node, 1, "size")
    if not isinstance(declared, (tuple, list)):
        return None
    declared = tuple(declared)
    if not all(
        isinstance(extent, int) and not isinstance(extent, bool)
        for extent in declared
    ):
        return None
    if -1 not in declared:
        return declared
    if declared.count(-1) != 1:
        return None

    inferred = getattr(node, "shape", None)
    if inferred is None:
        inferred = getattr(getattr(node, "value", None), "shape", None)
    if inferred is None:
        inferred = getattr(node.meta.get("val"), "shape", None)
    if inferred is None:
        inferred = getattr(node.meta.get("tensor_meta"), "shape", None)
    if inferred is None:
        return None
    inferred = tuple(inferred)
    if len(inferred) != len(declared) or not all(
        isinstance(extent, int)
        and not isinstance(extent, bool)
        and extent >= 0
        for extent in inferred
    ):
        return None
    if any(
        expected != -1 and expected != actual
        for expected, actual in zip(declared, inferred)
    ):
        return None
    return inferred


def _match_channel_shuffle(output: Node):
    """Match one exclusive two-input, two-group channel-shuffle spine."""
    view_targets = {
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
    }
    if output.target not in view_targets:
        return None

    contiguous = output.args[0]
    if not isinstance(contiguous, Node) or (
        contiguous.target is not torch.ops.aten.contiguous.default
    ):
        return None

    transpose = contiguous.args[0]
    if not isinstance(transpose, Node) or (
        transpose.target is not torch.ops.aten.transpose.int
    ):
        return None
    if (
        get_arg_value(transpose, 1, "dim0") != 1
        or get_arg_value(transpose, 2, "dim1") != 2
    ):
        return None

    grouped_view = transpose.args[0]
    if (
        not isinstance(grouped_view, Node)
        or grouped_view.target not in view_targets
    ):
        return None

    grouped_shape = _resolved_view_shape(grouped_view)
    if grouped_shape is None or len(grouped_shape) != 5:
        return None
    groups = grouped_shape[1]
    if groups != 2:
        return None
    output_shape = _resolved_view_shape(output)
    if output_shape is None or len(output_shape) != 4:
        return None
    expected_output_shape = (
        grouped_shape[0],
        grouped_shape[2] * groups,
        grouped_shape[3],
        grouped_shape[4],
    )
    if tuple(output_shape) != expected_output_shape:
        return None

    cat = grouped_view.args[0]
    if not isinstance(cat, Node) or (
        cat.target is not torch.ops.aten.cat.default
    ):
        return None
    if get_arg_value(cat, 1, "dim", 0) != 1:
        return None

    inputs = get_arg_value(cat, 0, "tensors")
    if (
        not isinstance(inputs, (tuple, list))
        or len(inputs) != 2
        or not all(isinstance(node, Node) for node in inputs)
    ):
        return None

    spine = (cat, grouped_view, transpose, contiguous, output)
    if any(
        set(node.users) != {spine[index + 1]}
        for index, node in enumerate(spine[:-1])
    ):
        return None
    return tuple(inputs), groups, spine


def _match_channel_shuffle_cat(cat: Node):
    """Return the complete shuffle match rooted at ``cat``, if present."""
    if cat.target is not torch.ops.aten.cat.default:
        return None

    output = cat
    for _ in range(4):
        if len(output.users) != 1:
            return None
        output = next(iter(output.users))

    match = _match_channel_shuffle(output)
    if match is None or match[2][0] is not cat:
        return None
    return match


def is_channel_shuffle_cat(node: Node) -> bool:
    """Return whether ``node`` starts one complete Alpha shuffle spine."""
    return _match_channel_shuffle_cat(node) is not None
