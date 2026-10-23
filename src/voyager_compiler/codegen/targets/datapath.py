# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Turning a machine's datapath into the fusion patterns the compiler matches.

A fused group is a claim that the datapath runs those operations as one
instruction. That claim is decidable from the hardware: start at a unit that
issues a call, follow the wires it drains through, and collect the blocks
along the way. The result is a stage list -- exactly what ``fuse_operator``
consumes -- with the difference that it is a *consequence* of the machine
description rather than a second, hand-maintained statement of it.

Imports :mod:`voyager_compiler.op_matcher`, the leaf, not ``voyager_compiler``
itself: this module is reached while that package is still initialising.
"""

from functools import lru_cache
from typing import List, Optional, Tuple

from voyager_compiler.hardware_config import Block, Datapath, Unit
from voyager_compiler.op_matcher import OpMatcher, get_op_overload

__all__ = ["derive_patterns", "validate_datapath", "walks"]


def walks(datapath: Datapath) -> Tuple[Tuple[Unit, ...], ...]:
    """Every path from an issuing unit to a sink, as a tuple of units.

    Depth-first over ``links``. A machine with one drain chain yields one
    walk; a machine whose post-processing forks -- Alpha's, eventually --
    yields one per distinct path, and each becomes its own fusion pattern.
    """
    outgoing = {}
    for link in datapath.links:
        outgoing.setdefault(link.source, []).append(link.destination)

    sinks = set(datapath.sinks)
    found: List[Tuple[Unit, ...]] = []

    def extend(path: Tuple[Unit, ...], seen: frozenset) -> None:
        last = path[-1]
        successors = outgoing.get(last.name, ())
        # A path that can go no further has arrived, whether or not anyone
        # declared its end a sink -- a machine that forgot to name one should
        # still describe what it can fuse.
        if last.name in sinks or not successors:
            found.append(path)
            return
        for name in successors:
            if name in seen:
                continue  # a cycle is not a pipeline; stop rather than spin
            extend(path + (datapath.unit(name),), seen | {name})

    for unit in datapath.units:
        if unit.issue is not None:
            extend((unit,), frozenset({unit.name}))
    return tuple(found)


def _blocks_of(path: Tuple[Unit, ...]) -> Tuple[Block, ...]:
    """The blocks a walk passes through, issue block first."""
    blocks: List[Block] = []
    for position, unit in enumerate(path):
        if position == 0 and unit.issue is not None:
            blocks.append(unit.issue)
        blocks.extend(unit.blocks)
    return tuple(blocks)


def validate_datapath(datapath: Datapath) -> None:
    """Check every block against what it says about itself.

    The load-bearing check is on ``virtual_ops``. A block may name an
    operation so a virtual one -- the cross-tile accumulator add, which has no
    FX node -- can be found by target membership, provided the predicate then
    refuses every *real* node carrying it. Getting that wrong admits the
    operation for real, silently, on a machine whose whole point is that it
    cannot do it. So probe it: resolve every overload the name reaches and
    require the predicate to reject each one.

    Raises:
        ValueError: if a block admits a real node it declared virtual, or
            names an operation that resolves to nothing.
    """
    import torch

    for unit in datapath.units:
        for block in ((unit.issue,) if unit.issue else ()) + unit.blocks:
            for name in block.ops:
                if not get_op_overload(name):
                    raise ValueError(
                        f"{unit.name}/{block.name}: '{name}' resolves to no "
                        "overload; it would silently contribute nothing"
                    )
            for name in block.virtual_ops:
                for overload in get_op_overload(name):
                    graph = torch.fx.Graph()
                    operand = graph.placeholder("x")
                    try:
                        node = graph.call_function(overload, (operand, operand))
                    except Exception:  # not callable with two operands
                        continue
                    if block.predicate is None or block.predicate(node):
                        raise ValueError(
                            f"{unit.name}/{block.name}: '{name}' is declared "
                            f"virtual but the predicate admits {overload}. A "
                            "virtual operation must be named for the stage and "
                            "refused for every real node."
                        )


def _stage(block: Block) -> OpMatcher:
    return OpMatcher(*block.ops, predicate=block.predicate)


@lru_cache(maxsize=None)
def derive_patterns(datapath: Optional[Datapath]) -> Tuple[Tuple[OpMatcher, ...], ...]:
    """The fusion patterns this machine's wiring implies.

    Cached on the datapath, which is hashable by construction, so repeated
    calls hand back an identical object -- the same identity a module-level
    table has, and what keeps a caller from rebuilding matchers per compile.

    ``None`` -- an uninventoried machine -- derives nothing, and the caller
    falls back to whatever the target authored. That is deliberately different
    from a machine that describes itself as fusing nothing.
    """
    if datapath is None:
        return ()
    validate_datapath(datapath)
    patterns = []
    for path in walks(datapath):
        stages = tuple(
            _stage(block) for block in _blocks_of(path) if block.ops
        )
        if len(stages) > 1:  # a lone issue block is a call, not a fusion
            patterns.append(stages)
    return tuple(patterns)
