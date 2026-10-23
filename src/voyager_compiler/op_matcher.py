# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Matching one graph node against a named operation.

A leaf module: it imports ``torch`` and nothing from this package. That is the
whole reason it exists separately. ``voyager_compiler/__init__`` imports
``codegen.targets`` while it is still initialising, so a target that wanted an
``OpMatcher`` at module scope got

    ImportError: cannot import name 'OpMatcher' from partially initialized
                 module 'voyager_compiler'

and a target must be able to describe what it fuses. Importing from here
instead breaks the cycle. ``voyager_compiler`` re-exports both names, so every
existing ``from voyager_compiler import OpMatcher`` keeps working.
"""

from typing import Callable, Optional, Tuple

import torch
from torch.fx import Node

__all__ = ["OpMatcher", "get_op_overload"]


def get_op_overload(op_name: str):
    """Every overload a symbolic operation name resolves to.

    Both libraries are searched, and the in-place spelling as well, so
    ``"add"`` reaches ``aten.add.*`` and ``aten.add_.*`` alike. Note how wide
    that is -- ``"add"`` alone resolves nineteen overloads, ``add.Scalar`` and
    ``add.out`` among them. A predicate that means to refuse an operation has
    to refuse the *packet*, not a handful of overloads it thought of.
    """
    all_overloads = []
    for lib in [torch.ops.aten, torch.ops.quantized_ops]:
        # Also check inplace version of the op (e.g., "add_" for "add")
        for name in [op_name, f"{op_name}_"]:
            if (packet := getattr(lib, name, None)) is None:
                continue
            all_overloads.extend(
                [getattr(packet, name) for name in packet.overloads()]
            )
    return all_overloads


class OpMatcher:
    """One stage of a fusion pattern: which operations it admits.

    ``targets`` keeps the order the overloads resolved in, because that order
    is observable -- ``find_sequential_nodes_`` walks it and takes the first
    sequential candidate, so a set here would make fusion depend on hash
    order.
    """

    targets: Tuple[torch._ops.OpOverload]
    predicate: Optional[Callable[[Node], bool]] = None

    def __init__(self, *ops, predicate=None):
        self.predicate = predicate

        # Resolve symbolic ops. An op given as an overload is taken as-is: a
        # target naming an exact overload means that one and not its
        # siblings, which a symbolic name cannot express.
        targets = []
        for op in ops:
            if isinstance(op, torch._ops.OpOverload):
                targets.append(op)
            else:
                targets.extend(get_op_overload(op))

        # Freeze resolved targets
        self.targets = tuple(targets)

    def matches(self, node: Node) -> bool:
        if node.target not in self.targets:
            return False

        return self.predicate(node) if self.predicate else True
