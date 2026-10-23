# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Persistent DRAM placement for Magnet Alpha's flat compilation path.

The shared storage plan decides which physical values cross from DRAM into
Alpha's global memories.  This module gives those source backings stable
off-chip addresses; it does not discover operands, infer aliases, or allocate
computed intermediates.  Keeping the input as a finalized :class:`StoragePlan`
makes serialization a projection of decisions already made upstream.

The reference compiler uses a linear, non-reusing allocator for model inputs
and parameters.  Allocation order is the storage plan's transfer order, which
is also the final graph order from which those loads were derived.  Multiple
logical transfers may name one physical source, so roots are de-duplicated
without reordering them.
"""

from dataclasses import dataclass
from typing import Iterable

from voyager_compiler.codegen.storage import (
    PhysicalTensorId,
    StoragePlan,
    StoragePlane,
)
from voyager_compiler.codegen.targets.magnet_alpha.hardware import (
    INPUT_WORD_BITS,
)


ALPHA_DRAM_ALIGNMENT_BYTES = INPUT_WORD_BITS // 8
ALPHA_DRAM_MEMORY_SPACE = 0
ALPHA_DRAM_INSTANCE = "DRAM"


class AlphaDramAllocationError(RuntimeError):
    """Raised when a finalized DRAM transfer set cannot be placed exactly."""


@dataclass(frozen=True)
class AlphaDramPlacement:
    """One immutable linear placement for a persistent DRAM backing."""

    allocation_id: int
    physical_tensor: PhysicalTensorId
    size_bytes: int
    memory_space: int
    start: int
    end: int
    address: int
    memory_instance: str

    def __post_init__(self) -> None:
        if self.physical_tensor.plane != StoragePlane.DRAM.value:
            raise ValueError(
                f"{self.physical_tensor.name}: a DRAM placement requires a "
                "DRAM physical tensor"
            )
        if self.allocation_id < 0:
            raise ValueError("a DRAM allocation id must not be negative")
        if self.size_bytes <= 0:
            raise ValueError(
                f"{self.physical_tensor.name}: DRAM size must be positive"
            )
        if self.memory_space != ALPHA_DRAM_MEMORY_SPACE:
            raise ValueError(
                f"{self.physical_tensor.name}: Alpha DRAM partition must be "
                f"{ALPHA_DRAM_MEMORY_SPACE}"
            )
        if self.memory_instance != ALPHA_DRAM_INSTANCE:
            raise ValueError(
                f"{self.physical_tensor.name}: Alpha DRAM instance must be "
                f"{ALPHA_DRAM_INSTANCE!r}"
            )
        if self.start < 0 or self.end <= self.start:
            raise ValueError(
                f"{self.physical_tensor.name}: invalid DRAM byte range "
                f"[{self.start}, {self.end})"
            )
        if self.address != self.start:
            raise ValueError(
                f"{self.physical_tensor.name}: linear DRAM address must equal "
                "its byte offset"
            )
        if self.end - self.start < self.size_bytes:
            raise ValueError(
                f"{self.physical_tensor.name}: DRAM placement is smaller than "
                "its backing"
            )


def alpha_dram_roots(plan: StoragePlan) -> tuple[PhysicalTensorId, ...]:
    """Return distinct persistent DRAM sources in finalized transfer order."""
    dram = StoragePlane.DRAM.value
    roots = []
    seen = set()
    for transfer in plan.transfers:
        source = transfer.source
        if source.plane != dram:
            continue
        try:
            source = plan.storage_key(source, StoragePlane.DRAM)
        except (KeyError, ValueError) as error:
            raise AlphaDramAllocationError(
                f"{source.name}: transfer source is not a finalized DRAM "
                "backing"
            ) from error
        if source not in seen:
            roots.append(source)
            seen.add(source)
    return tuple(roots)


def validate_alpha_dram_placements(
    plan: StoragePlan,
    placements: Iterable[AlphaDramPlacement],
) -> tuple[AlphaDramPlacement, ...]:
    """Validate one placement per required transfer-source root, no more."""
    placements = tuple(placements)
    required = alpha_dram_roots(plan)
    actual = tuple(placement.physical_tensor for placement in placements)

    duplicate_roots = sorted(
        physical.name
        for physical in set(actual)
        if actual.count(physical) > 1
    )
    if duplicate_roots:
        raise AlphaDramAllocationError(
            "Alpha DRAM placements repeat physical tensors: "
            + ", ".join(duplicate_roots)
        )

    required_set = set(required)
    actual_set = set(actual)
    if required_set != actual_set:
        missing = sorted(root.name for root in required_set - actual_set)
        unexpected = sorted(root.name for root in actual_set - required_set)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise AlphaDramAllocationError(
            "Alpha DRAM placement roots do not match transfer sources: "
            + "; ".join(details)
        )

    by_root = {placement.physical_tensor: placement for placement in placements}
    for root in required:
        placement = by_root[root]
        try:
            backing_size = plan.backing_size_bytes(root, StoragePlane.DRAM)
        except (KeyError, ValueError) as error:
            raise AlphaDramAllocationError(
                f"{root.name}: persistent DRAM backing size is unknown"
            ) from error
        if backing_size <= 0:
            raise AlphaDramAllocationError(
                f"{root.name}: persistent DRAM backing size must be positive"
            )
        if placement.size_bytes != backing_size:
            raise AlphaDramAllocationError(
                f"{root.name}: placement records {placement.size_bytes} B, "
                f"but the storage plan requires {backing_size} B"
            )
    return placements


class AlphaDramAllocator:
    """Deterministically pack Alpha's persistent DRAM sources from byte zero."""

    def __init__(self, alignment_bytes: int = ALPHA_DRAM_ALIGNMENT_BYTES):
        alignment_bytes = int(alignment_bytes)
        if alignment_bytes <= 0:
            raise ValueError("DRAM alignment must be positive")
        self.alignment_bytes = alignment_bytes

    def _align(self, size_bytes: int) -> int:
        return (
            (size_bytes + self.alignment_bytes - 1)
            // self.alignment_bytes
            * self.alignment_bytes
        )

    def allocate(self, plan: StoragePlan) -> tuple[AlphaDramPlacement, ...]:
        """Place precisely the plan's persistent DRAM transfer sources."""
        placements = []
        next_address = 0
        for allocation_id, physical in enumerate(alpha_dram_roots(plan)):
            try:
                size_bytes = plan.backing_size_bytes(
                    physical, StoragePlane.DRAM
                )
            except (KeyError, ValueError) as error:
                raise AlphaDramAllocationError(
                    f"{physical.name}: persistent DRAM backing size is unknown"
                ) from error
            if size_bytes <= 0:
                raise AlphaDramAllocationError(
                    f"{physical.name}: persistent DRAM backing size must be "
                    "positive"
                )
            end = next_address + self._align(size_bytes)
            placements.append(
                AlphaDramPlacement(
                    allocation_id=allocation_id,
                    physical_tensor=physical,
                    size_bytes=size_bytes,
                    memory_space=ALPHA_DRAM_MEMORY_SPACE,
                    start=next_address,
                    end=end,
                    address=next_address,
                    memory_instance=ALPHA_DRAM_INSTANCE,
                )
            )
            next_address = end

        return validate_alpha_dram_placements(plan, placements)


__all__ = [
    "ALPHA_DRAM_ALIGNMENT_BYTES",
    "ALPHA_DRAM_INSTANCE",
    "ALPHA_DRAM_MEMORY_SPACE",
    "AlphaDramAllocationError",
    "AlphaDramAllocator",
    "AlphaDramPlacement",
    "alpha_dram_roots",
    "validate_alpha_dram_placements",
]
