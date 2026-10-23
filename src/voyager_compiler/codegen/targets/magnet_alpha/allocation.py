# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Validated two-instance global-memory placement for Magnet Alpha.

This module knows physical banks, entries, reservations, and Alpha's encoded
address.  It deliberately does not inspect FX or choose storage identity: its
inputs are finalized physical tensors with roles, lifetimes, and target
constraints supplied by :mod:`.storage`.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import combinations
from typing import TYPE_CHECKING, Iterable

from torch.fx import Node

from voyager_compiler.codegen.storage import (
    PhysicalTensorId,
    StorageInterval,
    StoragePlane,
    ValueRef,
)
from voyager_compiler.codegen.artifacts import set_supplemental_artifact
from voyager_compiler.codegen.targets.magnet_alpha.address import (
    ALPHA_MEMORY_BASES,
    AlphaAddress,
    alpha_entry_bytes,
    alpha_place,
)
from voyager_compiler.codegen.targets.magnet_alpha.hardware import (
    GLOBAL,
    MATRIX_UNIT,
    operand_mode,
)
from voyager_compiler.hardware_config import MemoryInstance, TensorRole

if TYPE_CHECKING:
    from voyager_compiler.codegen.targets.magnet_alpha.memory_plan import (
        AlphaPreparedStorage,
    )


ALPHA_BANK_CLUSTERS = frozenset({1, 2, 4})
ALPHA_RESERVED_BANKS = {
    "global_activation": frozenset({0, 1}),
    "global_weight": frozenset(),
}


class AlphaAllocationError(RuntimeError):
    """Raised when finalized Alpha storage cannot be placed legally."""


@dataclass(frozen=True)
class AlphaAllocationRequest:
    """One finalized physical backing awaiting a target placement."""

    physical_tensor: PhysicalTensorId
    size_bytes: int
    role: TensorRole
    memory_instance: str
    cluster: int
    operand_bits: int
    interval: StorageInterval
    compute_unit: str | None = None
    accesses: frozenset[tuple[TensorRole, str | None]] = frozenset()
    conflicts: frozenset[PhysicalTensorId] = frozenset()
    primary: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", TensorRole(self.role))
        object.__setattr__(
            self,
            "accesses",
            frozenset(
                (TensorRole(role), compute_unit)
                for role, compute_unit in self.accesses
            ),
        )
        object.__setattr__(self, "conflicts", frozenset(self.conflicts))
        if self.physical_tensor.plane != StoragePlane.SCRATCHPAD.value:
            raise ValueError(
                f"{self.physical_tensor.name}: Alpha allocates scratchpad "
                "physical tensors only"
            )
        if self.interval.physical_tensor != self.physical_tensor:
            raise ValueError(
                f"{self.physical_tensor.name}: allocation interval belongs "
                f"to {self.interval.physical_tensor.name}"
            )
        if self.size_bytes <= 0:
            raise ValueError(
                f"{self.physical_tensor.name}: allocation size must be positive"
            )
        if self.cluster not in ALPHA_BANK_CLUSTERS:
            raise ValueError(
                f"{self.physical_tensor.name}: Alpha cluster must be one of "
                f"{sorted(ALPHA_BANK_CLUSTERS)}, not {self.cluster}"
            )
        if self.operand_bits not in {4, 8}:
            raise ValueError(
                f"{self.physical_tensor.name}: Alpha address width must be "
                "4 or 8 bits"
            )


@dataclass(frozen=True)
class AlphaPlacement:
    """One immutable, address-validated physical placement."""

    allocation_id: int
    request: AlphaAllocationRequest
    memory_space: int
    start: int
    end: int
    address: int
    cells: tuple[tuple[int, int], ...]
    banks: frozenset[int]

    @property
    def physical_tensor(self) -> PhysicalTensorId:
        return self.request.physical_tensor

    @property
    def memory_instance(self) -> str:
        return self.request.memory_instance

    @property
    def interval(self) -> StorageInterval:
        return self.request.interval

    @property
    def cluster(self) -> int:
        return self.request.cluster

    @property
    def operand_bits(self) -> int:
        return self.request.operand_bits

    @property
    def role(self) -> TensorRole:
        return self.request.role

    @property
    def primary(self) -> bool:
        return self.request.primary


_ROLE_PRIORITY = (
    TensorRole.WEIGHT,
    TensorRole.ACTIVATION,
    TensorRole.OUTPUT,
    TensorRole.BIAS,
    TensorRole.SCALE,
    TensorRole.PARTIAL_SUM,
)


def _duplicates(values):
    values = tuple(values)
    seen = set()
    repeated = []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return tuple(repeated)


def _record_constraint_map(prepared):
    """Map every standalone/fused record member to one exact constraint."""
    from voyager_compiler.codegen.targets.magnet_alpha.memory_plan import (
        AlphaOperationConstraint,
    )

    constraints = tuple(prepared.operation_constraints)
    if not all(
        isinstance(constraint, AlphaOperationConstraint)
        for constraint in constraints
    ):
        raise TypeError(
            "Alpha allocation needs AlphaOperationConstraint values"
        )
    owners = tuple(prepared.executable_owners)
    if not all(isinstance(owner, Node) for owner in owners):
        raise TypeError("Alpha executable owners must be FX nodes")
    duplicate_owners = _duplicates(owners)
    if duplicate_owners:
        raise AlphaAllocationError(
            "Alpha executable owners repeat: "
            + ", ".join(owner.name for owner in duplicate_owners)
        )
    constraint_owners = tuple(
        constraint.owner for constraint in constraints
    )
    duplicate_constraints = _duplicates(constraint_owners)
    if duplicate_constraints:
        raise AlphaAllocationError(
            "Alpha operation constraints repeat owners: "
            + ", ".join(owner.name for owner in duplicate_constraints)
        )
    if set(constraint_owners) != set(owners):
        missing = sorted(
            owner.name for owner in set(owners) - set(constraint_owners)
        )
        unexpected = sorted(
            owner.name for owner in set(constraint_owners) - set(owners)
        )
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise AlphaAllocationError(
            "Alpha operation constraints do not cover executable owners: "
            + "; ".join(details)
        )

    by_operation = {}

    def bind(operation, constraint):
        previous = by_operation.get(operation)
        if previous is not None and previous is not constraint:
            raise AlphaAllocationError(
                f"{operation.name}: ambiguous Alpha executable owner"
            )
        by_operation[operation] = constraint

    model = prepared.graph.model
    for constraint in constraints:
        if not isinstance(constraint.implementation, Node):
            raise TypeError("Alpha constraint implementation must be FX")
        bind(constraint.owner, constraint)
        bind(constraint.implementation, constraint)
        if constraint.owner.op != "call_module":
            continue
        try:
            submodule = model.get_submodule(str(constraint.owner.target))
        except (AttributeError, KeyError) as error:
            raise AlphaAllocationError(
                f"{constraint.owner.name}: fused owner has no graph module"
            ) from error
        graph = getattr(submodule, "graph", None)
        if graph is None:
            raise AlphaAllocationError(
                f"{constraint.owner.name}: fused owner has no FX graph"
            )
        for member in graph.nodes:
            bind(member, constraint)
    return by_operation


def _resolve_cluster_constraints(prepared, constraints):
    if constraints is None:
        return {}
    if not isinstance(constraints, Mapping):
        raise TypeError("Alpha cluster constraints must be a mapping")
    plan = prepared.plan
    scratchpad = StoragePlane.SCRATCHPAD
    residents = set(prepared.resident_physical_ids)
    normalized = {}
    for value, cluster in constraints.items():
        if isinstance(value, PhysicalTensorId):
            try:
                physical = plan.storage_key(value, scratchpad)
            except (KeyError, TypeError, ValueError) as error:
                raise AlphaAllocationError(
                    f"{value.name}: cluster constraint is not in storage"
                ) from error
        elif isinstance(value, (Node, ValueRef)):
            try:
                physical = plan.physical_id(scratchpad, plan.ref(value))
            except (KeyError, TypeError, ValueError) as error:
                name = getattr(value, "name", repr(value))
                raise AlphaAllocationError(
                    f"{name}: cluster constraint has no exact physical ID"
                ) from error
        else:
            raise TypeError(
                "Alpha cluster constraint keys must be FX, logical, or "
                "physical values"
            )
        if physical not in residents:
            raise AlphaAllocationError(
                f"{physical.name}: cluster constraint is not resident"
            )
        if (
            isinstance(cluster, bool)
            or not isinstance(cluster, int)
            or cluster not in ALPHA_BANK_CLUSTERS
        ):
            raise AlphaAllocationError(
                f"{physical.name}: unsupported Alpha bank cluster {cluster!r}"
            )
        previous = normalized.get(physical)
        if previous is not None and previous != cluster:
            raise AlphaAllocationError(
                f"{physical.name}: conflicting Alpha bank clusters "
                f"{previous} and {cluster}"
            )
        normalized[physical] = cluster
    return normalized


def _activation_cluster(config, operand_bits: int) -> int:
    cluster = None
    if config is not None:
        dtype_mode = getattr(config, "dtype_mode", None)
        if not callable(dtype_mode):
            raise TypeError("Alpha config must provide dtype_mode(bits)")
        try:
            mode = dtype_mode(operand_bits)
        except (TypeError, ValueError) as error:
            raise AlphaAllocationError(
                f"Alpha config has no {operand_bits}-bit dtype mode"
            ) from error
        if mode is not None:
            cluster = getattr(mode, "banks_per_value", None)
    if cluster is None:
        try:
            cluster = operand_mode(operand_bits)["banks_per_value"]
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaAllocationError(
                f"Alpha has no {operand_bits}-bit placement mode"
            ) from error
    if (
        isinstance(cluster, bool)
        or not isinstance(cluster, int)
        or cluster not in ALPHA_BANK_CLUSTERS
    ):
        raise AlphaAllocationError(
            f"unsupported Alpha bank cluster {cluster!r} for "
            f"{operand_bits}-bit payloads"
        )
    return cluster


def _allocation_role(accesses, physical):
    roles = {role for role, _unit in accesses}
    for role in _ROLE_PRIORITY:
        if role in roles:
            return role
    raise AlphaAllocationError(
        f"{physical.name}: physical backing has no allocation role"
    )


def build_alpha_allocation_requests(
    prepared: "AlphaPreparedStorage",
    config=None,
    cluster_constraints=None,
) -> "AlphaPreparedStorage":
    """Derive one exact Alpha allocation request per resident backing.

    Storage has already fixed identity, liveness, and every operation's
    physical operands.  This step only projects those decisions into Alpha's
    two global memories and symmetric bank-exclusion constraints.
    """
    # Local import is deliberate: memory_plan owns the immutable handoff but
    # imports AlphaAllocationRequest as one of its field types.
    from voyager_compiler.codegen.targets.magnet_alpha.memory_plan import (
        AlphaClusterConstraint,
        AlphaEgressBoundary,
        AlphaOperandBinding,
        AlphaPreparedStorage,
    )

    if not isinstance(prepared, AlphaPreparedStorage):
        raise TypeError("Alpha allocation needs AlphaPreparedStorage")
    plan = prepared.plan
    scratchpad = StoragePlane.SCRATCHPAD
    residents = tuple(prepared.resident_physical_ids)
    if not all(isinstance(item, PhysicalTensorId) for item in residents):
        raise TypeError("Alpha resident IDs must be physical tensor IDs")
    duplicates = _duplicates(residents)
    if duplicates:
        raise AlphaAllocationError(
            "Alpha resident physical IDs repeat: "
            + ", ".join(physical.name for physical in duplicates)
        )
    resident_set = set(residents)
    for physical in residents:
        if physical.plane != scratchpad.value:
            raise AlphaAllocationError(
                f"{physical.name}: Alpha resident is not in scratchpad"
            )
        try:
            plan.storage_key(physical, scratchpad)
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaAllocationError(
                f"{physical.name}: resident is missing from final storage"
            ) from error

    constraint_by_operation = _record_constraint_map(prepared)
    accesses = {physical: set() for physical in residents}
    working_sets = {
        owner: set() for owner in prepared.executable_owners
    }

    def require_resident(logical, owner, label):
        try:
            physical = plan.physical_id(scratchpad, plan.ref(logical))
        except (KeyError, TypeError, ValueError) as error:
            name = getattr(logical, "name", repr(logical))
            raise AlphaAllocationError(
                f"{owner.name}: {label} {name} has no physical backing"
            ) from error
        if physical not in resident_set:
            raise AlphaAllocationError(
                f"{owner.name}: {physical.name} is missing from residents"
            )
        working_sets[owner].add(physical)
        return physical

    for constraint in prepared.operation_constraints:
        unit = constraint.unit
        if not isinstance(unit, str) or not unit:
            raise AlphaAllocationError(
                f"{constraint.owner.name}: operation has no compute unit"
            )
        for operand in constraint.operands:
            if not isinstance(operand, AlphaOperandBinding):
                raise TypeError(
                    "Alpha constraint operands must be AlphaOperandBinding"
                )
            if (
                not isinstance(operand.compute_unit, str)
                or not operand.compute_unit
            ):
                raise AlphaAllocationError(
                    f"{constraint.owner.name}: {operand.port} has no unit"
                )
            try:
                role = TensorRole(operand.role)
            except ValueError as error:
                raise AlphaAllocationError(
                    f"{constraint.owner.name}: {operand.port} has no role"
                ) from error
            physical = require_resident(
                operand.logical_tensor,
                constraint.owner,
                operand.port,
            )
            accesses[physical].add((role, operand.compute_unit))

        output = require_resident(
            constraint.output,
            constraint.owner,
            "output",
        )
        accesses[output].add((TensorRole.OUTPUT, unit))

    egress_physical_ids = set()
    executable_set = set(prepared.executable_owners)
    for boundary in prepared.egress_boundaries:
        if not isinstance(boundary, AlphaEgressBoundary):
            raise TypeError(
                "prepared Alpha egress boundaries must be "
                "AlphaEgressBoundary records"
            )
        if not boundary.owners <= executable_set:
            raise AlphaAllocationError(
                f"{boundary.dequantize.name}: Alpha egress source has no "
                "executable owner"
            )
        for node in boundary.boundary_nodes:
            try:
                physical = plan.physical_id(scratchpad, plan.ref(node))
            except (KeyError, TypeError, ValueError) as error:
                raise AlphaAllocationError(
                    f"{node.name}: Alpha egress materialization has no "
                    "physical backing"
                ) from error
            if physical not in resident_set:
                raise AlphaAllocationError(
                    f"{node.name}: Alpha egress materialization is not "
                    "resident"
                )
            if physical in egress_physical_ids:
                raise AlphaAllocationError(
                    f"{physical.name}: Alpha egress materialization repeats"
                )
            egress_physical_ids.add(physical)
            accesses[physical].add((TensorRole.OUTPUT, None))

    try:
        physical_tensors = plan.physical_tensors[scratchpad.value]
    except KeyError as error:
        raise AlphaAllocationError(
            "final Alpha storage has no scratchpad plane"
        ) from error
    for physical, tensor in physical_tensors.items():
        for access in tensor.accesses:
            if physical not in resident_set:
                raise AlphaAllocationError(
                    f"{physical.name}: hidden access is not resident"
                )
            constraint = constraint_by_operation.get(access.operation)
            if constraint is None:
                raise AlphaAllocationError(
                    f"{physical.name}: hidden access by "
                    f"{access.operation.name} has no execution unit"
                )
            try:
                role = TensorRole(access.role)
            except ValueError as error:
                raise AlphaAllocationError(
                    f"{physical.name}: hidden access has no role"
                ) from error
            accesses[physical].add((role, constraint.unit))
            working_sets[constraint.owner].add(physical)

    conflicts = {physical: set() for physical in residents}
    for working_set in working_sets.values():
        for left, right in combinations(
            sorted(working_set, key=lambda item: item.name),
            2,
        ):
            conflicts[left].add(right)
            conflicts[right].add(left)

    auxiliary_ids = tuple(
        shuffle.auxiliary for shuffle in prepared.shuffle_plans
    )
    duplicate_auxiliaries = _duplicates(auxiliary_ids)
    if duplicate_auxiliaries:
        raise AlphaAllocationError(
            "Alpha shuffle auxiliary IDs repeat: "
            + ", ".join(item.name for item in duplicate_auxiliaries)
        )
    auxiliary_set = set(auxiliary_ids)
    shuffle_clusters = {}
    activation_entry_bits = int(
        GLOBAL.instance(GLOBAL.output_memory).bits_per_bank_entry
    )
    for shuffle in prepared.shuffle_plans:
        logical_entry_bits = int(shuffle.block_channels) * 8
        cluster, remainder = divmod(
            logical_entry_bits,
            activation_entry_bits,
        )
        if remainder or cluster not in ALPHA_BANK_CLUSTERS:
            raise AlphaAllocationError(
                f"{shuffle.shuffle.name}: {shuffle.block_channels}-channel "
                "INT8 shuffle blocks do not fit an Alpha bank cluster"
            )
        for physical in (shuffle.primary, shuffle.auxiliary):
            previous = shuffle_clusters.get(physical)
            if previous is not None and previous != cluster:
                raise AlphaAllocationError(
                    f"{physical.name}: shuffle plans require conflicting "
                    "Alpha bank clusters"
                )
            shuffle_clusters[physical] = cluster
    unclassified_auxiliaries = {
        physical
        for physical in residents
        if physical.component != "primary" and physical not in auxiliary_set
    }
    if unclassified_auxiliaries:
        raise AlphaAllocationError(
            "Alpha resident auxiliaries lack a target storage plan: "
            + ", ".join(
                sorted(item.name for item in unclassified_auxiliaries)
            )
        )
    missing_auxiliaries = auxiliary_set - resident_set
    if missing_auxiliaries:
        raise AlphaAllocationError(
            "Alpha shuffle auxiliaries are missing from residents: "
            + ", ".join(sorted(item.name for item in missing_auxiliaries))
        )

    prepared_cluster_map = {}
    for constraint in prepared.cluster_constraints:
        if not isinstance(constraint, AlphaClusterConstraint):
            raise TypeError(
                "prepared Alpha cluster constraints must be "
                "AlphaClusterConstraint records"
            )
        physical = constraint.physical_tensor
        if physical in prepared_cluster_map:
            raise AlphaAllocationError(
                f"{physical.name}: prepared Alpha cluster constraint repeats"
            )
        prepared_cluster_map[physical] = constraint.cluster
    overrides = _resolve_cluster_constraints(
        prepared,
        prepared_cluster_map,
    )
    external_overrides = _resolve_cluster_constraints(
        prepared,
        cluster_constraints,
    )
    for physical, cluster in external_overrides.items():
        previous = overrides.get(physical)
        if previous is not None and previous != cluster:
            raise AlphaAllocationError(
                f"{physical.name}: external Alpha bank cluster {cluster} "
                f"conflicts with prepared cluster {previous}"
            )
        overrides[physical] = cluster
    requests = []
    for physical in residents:
        requirements = frozenset(accesses[physical])
        if not requirements:
            aliases = sorted(
                ref.name
                for ref, backing in plan.physical_of[scratchpad.value].items()
                if backing == physical
            )
            raise AlphaAllocationError(
                f"{physical.name}: physical backing has no exact access role; "
                f"logical aliases: {', '.join(aliases)}"
            )
        try:
            tensor = plan.physical_tensor(scratchpad, physical)
            component = tensor.component("primary")
            size_bytes = plan.backing_size_bytes(physical, scratchpad)
            interval = plan.intervals[physical]
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaAllocationError(
                f"{physical.name}: missing size, interval, or primary payload"
            ) from error
        payload_bits = component.bits
        if (
            isinstance(payload_bits, bool)
            or not isinstance(payload_bits, int)
            or payload_bits <= 0
        ):
            raise AlphaAllocationError(
                f"{physical.name}: unsupported Alpha payload width "
                f"{payload_bits!r}"
            )
        if physical in egress_physical_ids:
            # The egress remains a true FP32 backing.  Alpha's address has a
            # one-bit payload-width mode, so the fork encodes every width
            # greater than four through the 8-bit mode without shrinking it.
            operand_bits = 4 if payload_bits <= 4 else 8
        elif payload_bits in {4, 8}:
            operand_bits = payload_bits
        else:
            raise AlphaAllocationError(
                f"{physical.name}: unsupported Alpha payload width "
                f"{payload_bits!r}"
            )
        if not isinstance(interval, StorageInterval):
            raise AlphaAllocationError(
                f"{physical.name}: final storage interval is invalid"
            )
        role = _allocation_role(requirements, physical)
        pe_weight = all(
            access_role is TensorRole.WEIGHT
            and unit == MATRIX_UNIT.name
            for access_role, unit in requirements
        )
        memory_instance = (
            GLOBAL.weight_memory if pe_weight else GLOBAL.output_memory
        )
        override = overrides.get(physical)
        if memory_instance == GLOBAL.weight_memory:
            if override is not None and override != 1:
                raise AlphaAllocationError(
                    f"{physical.name}: Alpha weight memory requires cluster 1"
                )
            cluster = 1
        else:
            cluster = (
                override
                if override is not None
                else _activation_cluster(config, operand_bits)
            )
        if physical in shuffle_clusters:
            shuffle_cluster = shuffle_clusters[physical]
            if operand_bits != 8 or cluster != shuffle_cluster:
                raise AlphaAllocationError(
                    f"{physical.name}: Alpha shuffle backing requires "
                    f"8-bit cluster {shuffle_cluster}, not "
                    f"{operand_bits}-bit cluster {cluster}"
                )
        units = {unit for _role, unit in requirements}
        compute_unit = next(iter(units)) if len(units) == 1 else None
        request = AlphaAllocationRequest(
            physical_tensor=physical,
            size_bytes=size_bytes,
            role=role,
            memory_instance=memory_instance,
            cluster=cluster,
            operand_bits=operand_bits,
            interval=interval,
            compute_unit=compute_unit,
            accesses=requirements,
            conflicts=frozenset(conflicts[physical]),
            primary=physical not in auxiliary_set,
        )
        memory = GLOBAL.instance(memory_instance)
        if not AlphaGlobalAllocator._accepts(memory, request):
            rendered = sorted(
                (access_role.value, unit)
                for access_role, unit in requirements
            )
            raise AlphaAllocationError(
                f"{physical.name}: {memory_instance} does not accept "
                f"{rendered}"
            )
        requests.append(request)

    requests = tuple(requests)
    existing = tuple(prepared.allocation_requests)
    if existing and existing != requests:
        raise AlphaAllocationError(
            "prepared Alpha allocation requests are stale"
        )
    return replace(prepared, allocation_requests=requests)


class AlphaGlobalAllocator:
    """First-fit interval allocation over Alpha's physical bank cells.

    Offsets from different cluster widths are never compared as byte ranges.
    A cluster stripes one logical entry over several physical banks, so
    overlap is decided on exact ``(bank, entry)`` cells and half-open
    lifetimes.  This is the same geometry the endpoint decodes.
    """

    def __init__(self, memory_level=GLOBAL):
        self.memory_level = memory_level
        self._memories = {
            memory.name: memory for memory in memory_level.instances
        }
        if set(self._memories) != set(ALPHA_MEMORY_BASES):
            raise ValueError(
                "Alpha requires exactly its activation and weight global "
                f"memories; got {sorted(self._memories)}"
            )

    @staticmethod
    def _accepts(memory: MemoryInstance, request: AlphaAllocationRequest) -> bool:
        requirements = request.accesses or frozenset(
            {(request.role, request.compute_unit)}
        )
        if not memory.accepted_accesses:
            return not memory.accepted_roles or all(
                role in memory.accepted_roles for role, _unit in requirements
            )
        return all(
            any(
                access.role is role
                and (
                    access.compute_unit is None
                    or access.compute_unit == compute_unit
                )
                for access in memory.accepted_accesses
            )
            for role, compute_unit in requirements
        )

    @staticmethod
    def _candidate_cells(
        memory: MemoryInstance,
        *,
        cluster: int,
        linear_entry: int,
        number_of_entries: int,
    ) -> tuple[tuple[int, int], ...]:
        depth = int(memory.depth_per_bank)
        cells = []
        for offset in range(number_of_entries):
            logical = linear_entry + offset
            virtual_bank, entry = divmod(logical, depth)
            first_bank = virtual_bank * cluster
            cells.extend(
                (bank, entry)
                for bank in range(first_bank, first_bank + cluster)
            )
        return tuple(cells)

    @staticmethod
    def _overlap(left: StorageInterval, right: StorageInterval) -> bool:
        return left.birth < right.death and right.birth < left.death

    def _find_open_block(
        self,
        request: AlphaAllocationRequest,
        placements: tuple[AlphaPlacement, ...],
    ) -> tuple[int, tuple[tuple[int, int], ...]] | None:
        memory = self._memories[request.memory_instance]
        cluster = request.cluster
        if int(memory.bank_count) % cluster:
            raise AlphaAllocationError(
                f"{memory.name}: {memory.bank_count} banks cannot form "
                f"clusters of {cluster}"
            )
        logical_entry_bytes = alpha_entry_bytes(memory, cluster)
        number_of_entries = math.ceil(request.size_bytes / logical_entry_bytes)
        capacity_entries = (
            int(memory.bank_count) // cluster * int(memory.depth_per_bank)
        )
        if number_of_entries > capacity_entries:
            return None

        reserved = ALPHA_RESERVED_BANKS[memory.name]
        usable_entries = sum(
            1
            for first_bank in range(0, int(memory.bank_count), cluster)
            if not reserved.intersection(
                range(first_bank, first_bank + cluster)
            )
        ) * int(memory.depth_per_bank)
        if number_of_entries > usable_entries:
            return None
        simultaneous = tuple(
            placement
            for placement in placements
            if placement.memory_instance == memory.name
            and self._overlap(request.interval, placement.interval)
        )
        conflicts = tuple(
            placement
            for placement in placements
            if placement.memory_instance == memory.name
            and (
                placement.physical_tensor in request.conflicts
                or request.physical_tensor in placement.request.conflicts
            )
        )

        for linear_entry in range(capacity_entries - number_of_entries + 1):
            cells = self._candidate_cells(
                memory,
                cluster=cluster,
                linear_entry=linear_entry,
                number_of_entries=number_of_entries,
            )
            banks = frozenset(bank for bank, _entry in cells)
            if banks & reserved:
                continue
            if any(
                banks & placement.banks
                for placement in conflicts
            ):
                continue
            candidate = frozenset(cells)
            if any(candidate.intersection(placement.cells) for placement in simultaneous):
                continue
            return linear_entry, cells
        return None

    def allocate(
        self,
        requests: Iterable[AlphaAllocationRequest],
    ) -> tuple[AlphaPlacement, ...]:
        """Place every request or fail without returning a partial plan."""
        requests = tuple(requests)
        physicals = [request.physical_tensor for request in requests]
        if len(set(physicals)) != len(physicals):
            duplicates = sorted(
                physical.name
                for physical in set(physicals)
                if physicals.count(physical) > 1
            )
            raise AlphaAllocationError(
                "Alpha allocation requests repeat physical tensors: "
                + ", ".join(duplicates)
            )

        ordered = sorted(
            requests,
            key=lambda request: (
                request.interval.birth,
                request.interval.death,
                request.memory_instance,
                request.role.value,
                request.physical_tensor.name,
            ),
        )
        placements: list[AlphaPlacement] = []
        for request in ordered:
            memory = self._memories.get(request.memory_instance)
            if memory is None:
                raise AlphaAllocationError(
                    f"{request.physical_tensor.name}: unknown Alpha memory "
                    f"{request.memory_instance!r}"
                )
            if not self._accepts(memory, request):
                raise AlphaAllocationError(
                    f"{request.physical_tensor.name}: {memory.name} does not "
                    f"accept {request.role.value} for {request.compute_unit!r}"
                )
            found = self._find_open_block(request, tuple(placements))
            if found is None:
                raise AlphaAllocationError(
                    f"{request.physical_tensor.name}: no legal "
                    f"{memory.name} placement for {request.size_bytes} bytes "
                    f"during [{request.interval.birth}, "
                    f"{request.interval.death})"
                )
            linear_entry, cells = found
            logical_entry_bytes = alpha_entry_bytes(memory, request.cluster)
            start = linear_entry * logical_entry_bytes
            end = start + (
                len(cells) // request.cluster * logical_entry_bytes
            )
            address = alpha_place(
                memory,
                start,
                cluster=request.cluster,
                operand_bits=request.operand_bits,
            ).encode()
            placement = AlphaPlacement(
                allocation_id=len(placements),
                request=request,
                memory_space=self.memory_level.instances.index(memory),
                start=start,
                end=end,
                address=address,
                cells=cells,
                banks=frozenset(bank for bank, _entry in cells),
            )
            self._validate_placement(placement, memory)
            placements.append(placement)
        return tuple(placements)

    @staticmethod
    def _validate_placement(
        placement: AlphaPlacement,
        memory: MemoryInstance,
    ) -> None:
        request = placement.request
        if placement.end - placement.start < request.size_bytes:
            raise AlphaAllocationError(
                f"{request.physical_tensor.name}: placement is smaller than "
                "its backing"
            )
        decoded = AlphaAddress.decode(
            placement.address,
            ALPHA_MEMORY_BASES[memory.name],
        )
        expected_entry = placement.start // alpha_entry_bytes(
            memory, request.cluster
        )
        expected_virtual, expected_offset = divmod(
            expected_entry, int(memory.depth_per_bank)
        )
        if (
            decoded.cluster != request.cluster
            or decoded.operand_bits != request.operand_bits
            or decoded.virtual_bank != expected_virtual
            or decoded.entry != expected_offset
        ):
            raise AlphaAllocationError(
                f"{request.physical_tensor.name}: encoded address does not "
                "round-trip to its placement"
            )
        if placement.banks & ALPHA_RESERVED_BANKS[memory.name]:
            raise AlphaAllocationError(
                f"{request.physical_tensor.name}: placement uses a reserved bank"
            )
        if any(
            bank < 0
            or bank >= int(memory.bank_count)
            or entry < 0
            or entry >= int(memory.depth_per_bank)
            for bank, entry in placement.cells
        ):
            raise AlphaAllocationError(
                f"{request.physical_tensor.name}: placement exceeds {memory.name}"
            )


def alpha_bank_placement_payload(
    prepared: "AlphaPreparedStorage",
    placements: Iterable[AlphaPlacement],
) -> dict:
    """Project finalized Alpha placements into the converter's manifest.

    The payload deliberately contains no allocation logic.  Every range,
    address, lifetime, role, and physical identity must already be present in
    ``AlphaPlacement`` and the finalized storage plan.
    """
    from voyager_compiler.codegen.targets.magnet_alpha.memory_plan import (
        AlphaPreparedStorage,
    )

    if not isinstance(prepared, AlphaPreparedStorage):
        raise TypeError("Alpha bank placements need prepared storage")
    placements = tuple(placements)
    if not all(isinstance(item, AlphaPlacement) for item in placements):
        raise TypeError("Alpha bank placements need AlphaPlacement values")
    physicals = tuple(item.physical_tensor for item in placements)
    duplicates = _duplicates(physicals)
    if duplicates:
        raise AlphaAllocationError(
            "Alpha bank placements repeat physical tensors: "
            + ", ".join(physical.name for physical in duplicates)
        )
    requests = tuple(prepared.allocation_requests)
    if set(physicals) != {request.physical_tensor for request in requests}:
        raise AlphaAllocationError(
            "Alpha bank placements do not cover the prepared allocation "
            "requests"
        )

    allocations = []
    for placement in sorted(placements, key=lambda item: item.allocation_id):
        try:
            request = prepared.request_for(placement.physical_tensor)
            physical = prepared.plan.physical_tensor(
                StoragePlane.SCRATCHPAD,
                placement.physical_tensor,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaAllocationError(
                f"{placement.physical_tensor.name}: placement is absent "
                "from prepared Alpha storage"
            ) from error
        if placement.request is not request:
            raise AlphaAllocationError(
                f"{placement.physical_tensor.name}: placement has a stale "
                "allocation request"
            )
        allocations.append(
            {
                "allocation_id": int(placement.allocation_id),
                "physical_tensor": placement.physical_tensor.name,
                "logical_tensors": sorted(
                    logical.name for logical in physical.logical_tensors
                ),
                "role": placement.role.value,
                "memory_name": placement.memory_instance,
                "memory_space": int(placement.memory_space),
                "start": int(placement.start),
                "end": int(placement.end),
                "address": int(placement.address),
                "birth_position": int(placement.interval.birth),
                "death_position": int(placement.interval.death),
                "primary": bool(placement.primary),
                "banks": sorted(int(bank) for bank in placement.banks),
            }
        )
    return {
        "schema_version": 1,
        "target": "magnet-alpha",
        "position_semantics": "zero_based_execution_order_index",
        "interval_semantics": "birth_inclusive_death_exclusive",
        "execution_order": [
            node.name for node in prepared.execution_order.nodes
        ],
        "allocations": allocations,
    }


def publish_alpha_bank_placements(
    model,
    prepared: "AlphaPreparedStorage",
    placements: Iterable[AlphaPlacement],
) -> None:
    """Publish the already-decided Alpha bank manifest for artifact dumping."""
    set_supplemental_artifact(
        model,
        "magnet_alpha_bank_placements",
        alpha_bank_placement_payload(prepared, placements),
    )


__all__ = [
    "ALPHA_BANK_CLUSTERS",
    "ALPHA_RESERVED_BANKS",
    "AlphaAllocationError",
    "AlphaAllocationRequest",
    "AlphaGlobalAllocator",
    "AlphaPlacement",
    "alpha_bank_placement_payload",
    "build_alpha_allocation_requests",
    "publish_alpha_bank_placements",
]
