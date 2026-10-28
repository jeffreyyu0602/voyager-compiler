# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The vocabulary a local scheduler's result is expressed in.

A local scheduler decides how one operation's loop nest is blocked, ordered
and spread across the memory hierarchy. *What it decided* is the same fact
whichever mapper decided it, so it is described here, once, and every
scheduler produces this.

This module is **stdlib-only and mapper-agnostic on purpose**. It knows
nothing about a specific mapper or the compiler; converting a mapper's output
into this vocabulary is an adopter's job, not this module's (see
``scheduler.interstellar_adapter``).

Indexing convention
-------------------
Matrices are ``[loop][level]``. Levels run from the innermost (PE) outward to
the outermost memory; code generation deliberately skips level 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Optional
from enum import IntEnum
from typing import Iterable


class Loop(IntEnum):
    """Canonical GEMM/convolution loop order shared by all schedulers."""

    FX = 0  # filter width
    FY = 1  # filter height
    OX = 2  # output width
    OY = 3  # output height
    OC = 4  # output channel
    IC = 5  # input channel
    ON = 6  # batch


NUM_LOOPS = len(Loop)


def _freeze_matrix(
    values: Iterable[Iterable[int]],
    field_name: str,
) -> tuple[tuple[int, ...], ...]:
    """Validate one ``[loop][level]`` matrix and freeze it to nested tuples."""
    result = tuple(tuple(int(value) for value in row) for row in values)
    if len(result) != NUM_LOOPS:
        raise ValueError(
            f"{field_name} must contain one row per canonical loop "
            f"({NUM_LOOPS}), got {len(result)}"
        )
    if not result[0]:
        raise ValueError(f"{field_name} must contain at least one level")
    num_levels = len(result[0])
    if any(len(row) != num_levels for row in result):
        raise ValueError(f"{field_name} rows must have equal length")
    return result


@dataclass(frozen=True)
class MappingPoint:
    """A complete local mapping, independent of the scheduler that found it."""

    loop_orders: tuple[tuple[int, ...], ...]
    loop_blockings: tuple[tuple[int, ...], ...]
    loop_partitionings: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        orders = _freeze_matrix(self.loop_orders, "loop_orders")
        blockings = _freeze_matrix(self.loop_blockings, "loop_blockings")
        partitionings = _freeze_matrix(
            self.loop_partitionings, "loop_partitionings"
        )
        num_levels = len(orders[0])
        if len(blockings[0]) != num_levels:
            raise ValueError("loop_blockings level count does not match")
        if len(partitionings[0]) != num_levels:
            raise ValueError("loop_partitionings level count does not match")
        if any(value <= 0 for row in blockings for value in row):
            raise ValueError("loop blocking factors must be positive")
        if any(value <= 0 for row in partitionings for value in row):
            raise ValueError("loop partitioning factors must be positive")
        if any(value < 0 for row in orders for value in row):
            raise ValueError("loop orders must be non-negative")
        object.__setattr__(self, "loop_orders", orders)
        object.__setattr__(self, "loop_blockings", blockings)
        object.__setattr__(self, "loop_partitionings", partitionings)

    @property
    def num_levels(self) -> int:
        return len(self.loop_orders[0])

    def loop_coverage(self, loop: Loop | int) -> int:
        """Total temporal and spatial coverage of one canonical loop."""
        index = int(loop)
        return math.prod(self.loop_blockings[index]) * math.prod(
            self.loop_partitionings[index]
        )


@dataclass(frozen=True)
class LevelAccessCounts:
    """Operand accesses at one memory level.

    Field order is the ``(input, output, weight)`` triple the emitter indexes
    positionally, so :meth:`as_triple` round-trips.
    """

    input: int = 0
    output: int = 0
    weight: int = 0

    def __post_init__(self):
        if min(self.input, self.output, self.weight) < 0:
            raise ValueError("access counts must be non-negative")

    def as_triple(self) -> tuple[int, int, int]:
        return (self.input, self.output, self.weight)


__all__ = [
    "GemmWorkload",
    "LevelAccessCounts",
    "LocalScheduleResult",
    "Loop",
    "MappingPoint",
    "NUM_LOOPS",
    "OperationClassification",
    "OperationOperand",
    "OperationWorkload",
    "SchedulerProviderSet",
]

# -----------------------------------------------------------------------------
# What a scheduler is asked, and what it answers
# -----------------------------------------------------------------------------
#
# The types above describe a *mapping*. These describe the decision that
# produced it: one operation in, one result out.
#
# Why per-operation rather than per-compile. A provider that returns a context
# the compiler then maps every layer against can only answer "how should this
# GEMM be tiled". It cannot answer "this operation is mine and is never tiled",
# which is what a machine with a second unit needs to say about the work that
# unit owns. Nor can it return a runtime, an access count, or the target's own
# names for the factors it chose -- all of which have to reach serialization,
# and none of which fits in a tiler context.


@dataclass(frozen=True)
class GemmWorkload:
    """The loop extents of one convolution or matrix multiplication.

    A matrix multiply is a 1x1 filter over a flattened output space, so one
    shape serves both. Operand *widths* are deliberately absent: they are a
    property of the invocation, not of the loops, and a machine that reduces
    over 32 input channels at 4 bits and 16 at 8 would otherwise need two
    workloads for one layer.
    """

    nifm: int
    nofm: int
    wofm: int
    hofm: int
    wfil: int
    hfil: int
    nimg: int = 1
    wstd: int = 1
    hstd: int = 1

    def __post_init__(self):
        for name in (
            "nifm",
            "nofm",
            "wofm",
            "hofm",
            "wfil",
            "hfil",
            "nimg",
            "wstd",
            "hstd",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))

    @property
    def wifm(self) -> int:
        return self.wfil + (self.wofm - 1) * self.wstd

    @property
    def hifm(self) -> int:
        return self.hfil + (self.hofm - 1) * self.hstd

    @property
    def sizes(self) -> tuple:
        """Extents in the canonical ``Loop`` order."""
        return (
            self.wfil,
            self.hfil,
            self.wofm,
            self.hofm,
            self.nofm,
            self.nifm,
            self.nimg,
        )


@dataclass(frozen=True)
class OperationOperand:
    """One operand, as the scheduler sees it.

    ``port`` is the operator schema's own keyword, so a scheduler reads
    ``"weight"`` rather than "argument 1". ``occurrence`` distinguishes
    repeated leaves -- the two halves of a concatenated input -- without
    encoding a path into the port name.
    """

    port: str
    shape: Optional[tuple] = None
    bits: Optional[int] = None
    occurrence: tuple = ()

    def __post_init__(self):
        if not isinstance(self.port, str) or not self.port:
            raise ValueError("an operand needs a port name")
        if self.shape is not None:
            shape = tuple(int(dimension) for dimension in self.shape)
            if any(dimension < 0 for dimension in shape):
                raise ValueError(f"{self.port}: dimensions must be non-negative")
            object.__setattr__(self, "shape", shape)
        if self.bits is not None:
            if isinstance(self.bits, bool) or int(self.bits) <= 0:
                raise ValueError(f"{self.port}: operand width must be positive")
            object.__setattr__(self, "bits", int(self.bits))
        object.__setattr__(self, "occurrence", tuple(self.occurrence))

    @property
    def identity(self) -> tuple:
        return (self.port, self.occurrence)


def _freeze_attribute(value, name: str):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_attribute(item, name) for item in value)
    raise TypeError(f"{name} must hold only scalars and tuples")


@dataclass(frozen=True)
class OperationClassification:
    """What kind of operation this is, and the scalar facts that qualify it.

    The kind is the target's vocabulary -- ``"gemm"``, ``"depthwise_conv2d"``,
    ``"channel_shuffle"`` -- because which unit owns an operation is a fact
    about the machine. The attributes are frozen scalars so a result can be
    memoized on the workload that produced it.
    """

    kind: str
    attributes: tuple = ()

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("an operation classification needs a kind")
        items = (
            self.attributes.items()
            if isinstance(self.attributes, Mapping)
            else self.attributes
        )
        attributes = tuple(
            sorted(
                (
                    (str(name), _freeze_attribute(value, f"attribute {name!r}"))
                    for name, value in items
                ),
                key=lambda item: item[0],
            )
        )
        names = tuple(name for name, _ in attributes)
        if len(set(names)) != len(names):
            raise ValueError("operation attribute names must be unique")
        object.__setattr__(self, "attributes", attributes)

    def attribute(self, name: str, default=None):
        return next(
            (value for key, value in self.attributes if key == name), default
        )


@dataclass(frozen=True)
class OperationWorkload:
    """One scheduling decision's input, with the graph already resolved.

    Everything a provider needs is frozen here before dispatch, so a provider
    never inspects an FX node. That is what lets a scheduler live in its own
    package and be tested without a compiler.
    """

    operands: tuple = ()
    output: Optional[OperationOperand] = None
    operation: Optional[OperationClassification] = None
    gemm: Optional[GemmWorkload] = None
    has_high_precision_vector_input: bool = False
    has_sparse_op: bool = False

    def __post_init__(self):
        operands = tuple(self.operands)
        if any(not isinstance(item, OperationOperand) for item in operands):
            raise TypeError("operands must be OperationOperand values")
        identities = tuple(operand.identity for operand in operands)
        if len(set(identities)) != len(identities):
            raise ValueError("operand port/occurrence identities must be unique")
        if any(operand.port == "output" for operand in operands):
            raise ValueError("the output has its own field")
        object.__setattr__(self, "operands", operands)

        if self.output is None:
            raise ValueError("an operation workload needs a resolved output")
        if not isinstance(self.output, OperationOperand):
            raise TypeError("the output must be an OperationOperand")
        if self.output.port != "output":
            raise ValueError("the output operand uses port 'output'")
        if self.output.shape is None:
            raise ValueError("the output shape must be resolved")

        if self.operation is not None and not isinstance(
            self.operation, OperationClassification
        ):
            raise TypeError(
                "operation must be an OperationClassification or None"
            )
        if self.gemm is not None and not isinstance(self.gemm, GemmWorkload):
            raise TypeError("gemm must be a GemmWorkload or None")
        for name in ("has_high_precision_vector_input", "has_sparse_op"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")

        object.__setattr__(
            self,
            "operands",
            tuple(
                sorted(
                    operands,
                    key=lambda operand: (
                        operand.port,
                        repr(operand.occurrence),
                    ),
                )
            ),
        )

    def operands_for_port(self, port: str) -> tuple[OperationOperand, ...]:
        """Every occurrence bound to one schema port."""
        return tuple(
            operand for operand in self.operands if operand.port == port
        )

    def operand(self, port: str, occurrence=None):
        """One named occurrence, or the sole unambiguous one for ``port``."""
        matches = self.operands_for_port(port)
        if occurrence is not None:
            occurrence = tuple(occurrence)
            return next(
                (
                    operand
                    for operand in matches
                    if operand.occurrence == occurrence
                ),
                None,
            )
        if len(matches) > 1:
            raise ValueError(
                f"operation workload has multiple {port!r} occurrences"
            )
        return matches[0] if matches else None

    def require_operand(self, port: str, occurrence=None) -> OperationOperand:
        """Return a required operand, naming a missing binding immediately."""
        operand = self.operand(port, occurrence)
        if operand is None:
            raise ValueError(f"operation workload has no {port!r} operand")
        return operand


@dataclass(frozen=True)
class LocalScheduleResult:
    """What one scheduler decided, in vocabulary no engine owns.

    ``named_factors`` is the part that is not a convenience. Serialization and
    collateral conversion consume the *target's* names for the factors a
    schedule chose -- ``p1``, ``c0``, ``k2`` -- and those names are a
    contract, not a debug dictionary: the endpoint's schedule file has a field
    per name. Deriving them at serialization time from whichever engine's
    metadata happened to be around is how two components end up disagreeing
    about what the hardware will run.

    The keys are validated as lowercase identifiers for that reason. A typo in
    a factor name is a silently missing field in the emitted schedule.
    """

    mapping: MappingPoint
    backend: str
    access_counts: tuple = ()
    runtime: Optional[float] = None
    energy_pj: Optional[float] = None
    named_factors: tuple = ()

    def __post_init__(self):
        if not isinstance(self.mapping, MappingPoint):
            raise TypeError(
                "a local schedule mapping must be scheduler.types.MappingPoint"
            )
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("a schedule result names the engine that made it")

        items = (
            self.named_factors.items()
            if isinstance(self.named_factors, Mapping)
            else self.named_factors
        )
        factors = []
        for name, value in items:
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or name != name.lower()
            ):
                raise ValueError(
                    f"schedule factor names are lowercase identifiers: {name!r}"
                )
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"schedule factor {name!r} must be a positive integer"
                )
            factors.append((name, value))
        factors.sort(key=lambda item: item[0])
        names = tuple(name for name, _ in factors)
        if len(set(names)) != len(names):
            raise ValueError("schedule factor names must be unique")
        object.__setattr__(self, "named_factors", tuple(factors))

        counts = tuple(self.access_counts)
        if any(not isinstance(count, LevelAccessCounts) for count in counts):
            raise TypeError(
                "access counts must contain LevelAccessCounts values"
            )
        if counts and len(counts) != self.mapping.num_levels:
            raise ValueError(
                "access counts must have one entry per mapping level"
            )
        object.__setattr__(self, "access_counts", counts)

        for name in ("runtime", "energy_pj"):
            value = getattr(self, name)
            if value is not None and (value < 0 or value != value):
                raise ValueError(f"{name} must be a non-negative number")

    def factor(self, name: str, default=None):
        """One named schedule factor, or ``default``."""
        return next(
            (value for key, value in self.named_factors if key == name), default
        )



@dataclass(frozen=True)
class SchedulerProviderSet:
    """The scheduling engines a target may use, and which one it picks.

    Compilation-target identity deliberately does not appear here: a target
    selects one immutable provider set, and dispatch only resolves an engine
    name within it. That keeps "which engine" separable from "which machine" --
    two targets can share an engine while describing different hardware, and
    one target can carry several engines to choose between.

    A provider builds the scheduler's view of a machine. It is the seam a
    target uses to *constrain* an engine (a datapath whose loop placement
    differs from the generic one) or to *replace* it (an accelerator the
    generic engine cannot express at all). Expressing that as a provider rather
    than as a hook that edits an engine's hint keeps the engine's internals
    private to the engine.

    There are two seams here, and they answer different questions.

    ``providers`` is per *compile*: it builds the scheduler's view of a
    machine once, and the compiler maps every layer against it. That is
    enough for a target whose every operation is a GEMM the shared engine can
    tile, which is what every lane did before Alpha.

    ``gemm_providers`` and ``operation_provider`` are per *operation*, and
    the second is the one a two-unit machine cannot do without. A provider
    that only ever returns a tiler context can say how a GEMM should be
    blocked; it has no way to say "this operation is mine, it is not a GEMM,
    and it is never tiled" -- which is the whole of what a near-memory unit
    does. So the target's operation provider gets *first refusal* on every
    operation, and only what it declines, and that has a canonical GEMM
    projection, reaches an engine.

    Attributes:
        default_backend: Which engine to use when none is named.
        providers: Engine name -> ``(config, **options) -> TilerContext``.
        gemm_providers: Engine name -> ``(workload, config, cache, options)
            -> LocalScheduleResult | None``.
        operation_provider: ``(workload, config) -> LocalScheduleResult |
            None``. Consulted before any engine.
        gemm_tiling_candidate_fits: A target's legality filter, offered to
            graph tiling so a tile no native call could issue is never
            chosen. Exposing the same rule the scheduler enforces means the
            search does not spend its time on mappings that will be refused.
    """

    default_backend: str
    providers: Mapping[str, Callable] = field(default_factory=dict)
    gemm_providers: Mapping[str, Callable] = field(default_factory=dict)
    operation_provider: Optional[Callable] = None
    gemm_tiling_candidate_fits: Optional[Callable] = None

    def __post_init__(self):
        if not isinstance(self.default_backend, str) or not self.default_backend:
            raise ValueError("default scheduler backend must be non-empty")

        registered = dict(self.providers)
        per_operation = dict(self.gemm_providers)
        if not registered and not per_operation:
            raise ValueError("at least one scheduler provider is required")
        for engines in (registered, per_operation):
            for name, provider in engines.items():
                if not isinstance(name, str) or not name:
                    raise TypeError("scheduler engine names must be non-empty")
                if not callable(provider):
                    raise TypeError(f"scheduler engine {name!r} is not callable")
        # The default has to name an engine in whichever seam is declared,
        # because that is the name the command line resolves against.
        declared = set(registered) | set(per_operation)
        if self.default_backend not in declared:
            raise ValueError(
                f"default backend {self.default_backend!r} has no registered "
                f"provider; declared: {sorted(declared)}"
            )
        for name in ("operation_provider", "gemm_tiling_candidate_fits"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")
        object.__setattr__(self, "providers", MappingProxyType(registered))
        object.__setattr__(
            self, "gemm_providers", MappingProxyType(per_operation)
        )

    def resolve(self, backend: Optional[str] = None) -> Callable:
        """The named per-compile provider, or the default one."""
        name = backend or self.default_backend
        try:
            return self.providers[name]
        except KeyError:
            raise ValueError(
                f"no scheduler provider {name!r}; declared: "
                f"{sorted(self.providers)}"
            ) from None

    def resolve_gemm(self, backend: Optional[str] = None) -> Callable:
        """The named per-operation GEMM engine, or the default one."""
        name = backend or self.default_backend
        try:
            return self.gemm_providers[name]
        except KeyError:
            raise ValueError(
                f"no GEMM engine {name!r}; declared: "
                f"{sorted(self.gemm_providers)}"
            ) from None
