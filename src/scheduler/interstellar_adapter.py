# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Adapter between the compiler and the vendored Interstellar engine.

Interstellar describes a machine as a fixed list of buffer levels with element
or byte capacities, an operand partition per level, and a schedule hint saying
where each loop may be blocked. The compiler describes one as an
``AcceleratorConfig``: declared memory levels, compute units, operand widths.
Translating between the two is this module's whole job, in both directions:
describing a machine to the engine, and adopting the engine's answer back into
the shared vocabulary the rest of the compiler speaks.

It sits *beside* the engine rather than inside it because the engine is
vendored -- a Voyager-specific file under ``interstellar/`` would be lost at the
next refresh. It sits outside the compiler because nothing here reads an FX
graph: give it a machine, it returns the engine's view of that machine.

One module rather than two: the fork keeps the same pair together in
``interstellar/voyager.py`` ("Interstellar provider adapter and its runtime
models"), and splitting them here bought nothing but a second file whose name
differed by one letter.

What deliberately stays in the compiler (``codegen/transform/tiling/``): the
size function, the runtime model, bank packing, and the search driver. Those
describe how *this* compiler's bufferized kernels behave rather than what the
engine needs to know, and they read FX nodes.
"""

from scheduler import interstellar
from scheduler.interstellar import loop_enum as le
from scheduler.types import NUM_LOOPS, LevelAccessCounts, Loop, MappingPoint
from voyager_compiler.hardware_config import TensorRole

#: The engine's hierarchy depth: PE, L1, L2, DRAM.
_SCHEDULE_LEVELS = 4

__all__ = [
    "adopt_access_counts",
    "adopt_mapping",
    "build_architecture",
    "build_schedule",
]


# ``Loop`` restates this numbering so ``scheduler.types`` needs no mapper
# import. That is two sources of truth, so check them here -- the one place
# that already imports both -- rather than trusting they stay aligned.
if NUM_LOOPS != le.NUM or any(
    int(loop) != le.loop_table[loop.name] for loop in Loop
):
    raise ImportError(
        "scheduler.types.Loop and interstellar.loop_enum disagree on the "
        "canonical loop numbering"
    )


def adopt_mapping(point) -> MappingPoint:
    """Convert one Interstellar ``MappingPoint`` into the shared type.

    Duck-typed on the three factor matrices rather than the concrete class, so
    an Interstellar fork or a stub in a test is adopted the same way.
    """
    return MappingPoint(
        loop_orders=point.loop_orders,
        loop_blockings=point.loop_blockings,
        loop_partitionings=point.loop_partitionings,
    )


def adopt_access_counts(access_list) -> tuple[LevelAccessCounts, ...]:
    """Convert Interstellar's per-level byte triples into the shared type."""
    return tuple(
        LevelAccessCounts(*(int(value) for value in level[:3]))
        for level in access_list
    )


def _declared_architecture(config, levels, ic_dim, oc_dim, dram_access_cost):
    """An interstellar ``Resource`` shaped by the levels a target declares.

    Mirrors how the fork translates its typed profile: one entry per declared
    level, each carrying its own instance capacities and its own operand
    partition, so a role-split level stays split and a unified one stays
    unified. Capacities are element counts -- that is what interstellar checks
    L0/L1 against, and sizing every declared level the same way keeps one unit
    across the hierarchy instead of switching to bytes partway up.

    Off-chip memory is appended only when the config has some, so a machine
    whose storage ends at its SRAMs is described with no level above them
    rather than a pool summed from the bytes it already spent.
    """
    widths = _operand_slot_bits(config)
    capacities = [[1, 1, 1]]
    access_costs = [[1, 1, 1]]
    partitions = [[0, 1, 2]]
    for index, level in enumerate(levels):
        # A memory is *described* in bits; which unit it is *checked* in is
        # interstellar's choice and differs by depth.  L0/L1 are slot arrays,
        # where an element occupies a fixed-width slot whatever its dtype, so
        # they are checked in element counts; L2 and above are byte pools
        # (``cost_model.get_bank_size`` branches on exactly this).  Converting
        # here, at the one interface, keeps the declaration in one unit and
        # still hands each level the unit it is compared against -- feeding
        # elements to a byte pool makes nothing fit at any tile size.
        depth = index + 1
        if depth <= 1:
            capacities.append(list(level.capacity_elements(widths)))
        else:
            capacities.append(
                [instance.size_bytes for instance in level.instances]
            )
        access_costs.append([10 * depth] * len(level.instances))
        partitions.append(list(level.operand_partitions))

    dram = config.external_memory
    if dram is not None and dram.size_gb:
        capacities.append([int(dram.size_gb * 1024**3)])
        access_costs.append([dram_access_cost])
        partitions.append([0, 0, 0])

    depth = len(capacities)
    return interstellar.Resource(
        buf_capacity_list=capacities,
        buf_access_cost_list=access_costs,
        buf_unit_static_cost_list=[[0] * len(c) for c in capacities],
        para_count_list=[ic_dim * oc_dim, *([1] * (depth - 1))],
        memory_partitions=partitions,
        mac_capacity=0,
        partition_mode=[0] * depth,
        invalid_underutilized=False,
        bank_size_list=[None] * depth,
    )


def _operand_slot_bits(config):
    """``(input, output, weight)`` operand widths, for sizing a level in slots.

    The output is a partial sum, so it is the accumulator's width rather than
    the operand's; without declared modes there is nothing to convert with and
    a level reports its own slot widths instead.
    """
    if not config.dtype_modes:
        return (None, None, None)
    operand = max(mode.operand_bits for mode in config.dtype_modes)
    accumulator = None
    if config.local_memory is not None:
        served = config.local_memory.for_role(TensorRole.OUTPUT)
        accumulator = served.slot_bits if served is not None else None
    return (operand, accumulator or operand, operand)


def build_architecture(config, dram_access_cost=1000):
    """The engine's view of ``config``'s memory hierarchy and parallelism.

    A target that describes its levels gets the hierarchy it declared, one
    entry per level with its own instance capacities and operand partition. One
    that describes none gets the fixed four-level shape (PE / L1 / L2 / DRAM)
    the compiler has always assumed.

    ``config.pe_array_size`` is ``(ic_dim, oc_dim)``. DRAM transfer accounting
    is in absolute bytes (bandwidth as ``config.bytes_per_cycle`` is read at run
    time), so it is not normalized by any element width.

    L0/L1 capacities are slot arrays: one fixed-width slot per element, so they
    are element counts and the fit check is dtype-independent. The flat L2/L3
    byte pools let sub-byte operands pack, so those stay in bytes.

    L2 is described the way ``plan_memory`` allocates it: the capacity is
    ``usable_scratchpad_size``, and a ping-ponged source is charged one bank per
    slot rather than the capacity being halved -- halving cannot express a
    buffer allocated only once, and that omission let the tiler hand back
    tilings the allocator could not place.

    Args:
        config: The machine to describe.
        dram_access_cost: Relative cost the engine charges an off-chip access.

    Returns:
        The engine's ``Resource``.
    """
    ic_dim, oc_dim = config.pe_array_size
    ic_dim, oc_dim = config.pe_array_size

    declared = [
        level
        for name, level in (
            ("local", config.local_memory),
            ("global", config.global_memory),
        )
        if level is not None
        and config.has_memory_level(name)
        and config.describes_memory_level(name)
    ]
    if declared:
        # A target that describes its memory gets the hierarchy it declared:
        # one level per description, each with its own instance count and its
        # own operand partition, and DRAM only if it has one.  The fixed
        # four-level shape below cannot say "three role-specific SRAMs and
        # nothing above them", so it would have to invent a pool -- which is
        # how the same bytes ended up counted at two levels.
        architecture = _declared_architecture(
            config, declared, ic_dim, oc_dim, dram_access_cost
        )
    else:
        architecture = interstellar.Resource(
    
            buf_capacity_list=[
                [1, 1, 1],
                [
                    config.input_buffer_size * ic_dim,
                    config.accum_buffer_size * oc_dim,
                    config.weight_buffer_size * oc_dim,
                ],
                # A machine that declares no unified pool above its per-role L1
                # gets an unconstrained level here, not one sized from the same
                # bytes L1 already spends -- that would let the search place a
                # tile at L1 *and* believe an independent pool exists to stage it
                # from, and every mapping would be solved against twice the
                # storage the machine has.
                [
                    config.usable_scratchpad_size
                    if config.has_memory_level("global")
                    else float("inf")
                ],
                [config.dram_size * 1024**3],  # GB -> bytes
            ],
            buf_access_cost_list=[
                [1, 1, 1],
                [10, 10, 10],
                [100],
                [dram_access_cost],
            ],
            buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0], [0]],
            para_count_list=[ic_dim * oc_dim, 1, 1, 1],
            memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
            mac_capacity=0,
            partition_mode=[0, 0, 0, 0],
            invalid_underutilized=False,
            bank_size_list=[
                None,
                None,
                config.bank_size if config.has_memory_level("global") else None,
                None,
            ],
        )

    return architecture


def build_schedule(config, constraint=None):
    """The engine's schedule hint: where each loop may be blocked and ordered.

    The hint is the engine's own vocabulary and stays private to this module. A
    target whose datapath places loops differently supplies ``constraint`` -- a
    ``(hint, config) -> hint`` narrowing of the hint built here -- so it narrows
    a placement it can see rather than restating one it cannot.

    Args:
        config: The machine being scheduled.
        constraint: Optional narrowing; ``None`` leaves the hint as built.

    Returns:
        The engine's ``Schedule``.
    """
    ic_dim, oc_dim = config.pe_array_size

    # The L1 order is pinned to ... > FY > FX > OY > OX (IC/OC free above).
    # OX/OY innermost is never slower -- the L1 sweep costs
    # ``max(loading, reused_tile) * remaining``, monotone in the reused
    # tile -- and the arrangement of the loops above them ties on both
    # runtime and energy, so FX/FY are fixed to the representative the
    # search's first-seen tie-break picked anyway.  FX=2/FY=3 assumes a
    # square kernel (equal FX/FY blocking); a non-square model may prefer
    # them swapped.
    generic_constraint = {
        "schedule_hint": {
            "IC": {
                "level0": {"order": 1, "partitioning_size": ic_dim},
                "level1": {"order": -1},
                "level2": {"order": 0},
                "level3": {"order": 0},
            },
            "OC": {
                "level0": {"order": 0, "partitioning_size": oc_dim},
            },
            "OX": {
                "level1": {"order": 0},
            },
            "OY": {
                "level1": {"order": 1},
            },
            "FX": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level1": {"order": 2},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
                "level3": {"blocking_size": 1, "partitioning_size": 1},
            },
            "FY": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level1": {"order": 3},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
                "level3": {"blocking_size": 1, "partitioning_size": 1},
            },
        }
    }

    # A provider may hand back a constrained view of the generic hint; the
    # hint's shape stays this engine's business, and a target that has no
    # opinion gets it unchanged.
    resolved = (
        generic_constraint
        if constraint is None
        else constraint(generic_constraint, config)
    )
    schedule_data = interstellar.extract_input.extract_schedule_info(
        resolved, _SCHEDULE_LEVELS
    )
    schedule = interstellar.Schedule(
        schedule_data["schedule_hint"],
        schedule_data["partition_loops"],
    )
    return schedule
