# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The Magnet Alpha machine: what the silicon is, independent of any compile.

Alpha is two units behind one global SRAM. A 32-lane vector MAC runs
GEMM-shaped work; a narrower near-memory processor (the NMP) owns everything
the MAC cannot express -- depthwise convolution, pooling, softmax, nearest
upsampling, channel shuffle. Which unit runs an operation is a hardware fact,
so it is declared rather than inferred from the operation's shape.

Two things about this machine do not fit a single-unit description, and both
are why the config grew the fields it did:

* **The reduction unrolling depends on the operand width.** Two INT4 operands
  occupy the slot one INT8 operand does, so the array reduces over 32 input
  channels at 4 bits and 16 at 8. That is a property of the ``IC`` dimension,
  not of the unit, and it is stated there.
* **A memory's role is not enough to say who may use it.** The activation
  SRAM holds weights *for the NMP* and activations for both units; the weight
  SRAM holds weights for the matrix unit only, and is never written by the
  datapath. A role-only claim collapses those into "both hold weights".

Stated once here. The test-side profile imports these rather than restating
them, so a number cannot drift between the machine and the way it is invoked.
"""

from voyager_compiler.hardware_config import (
    ComputeUnit,
    DTypeMode,
    HierarchyBandwidth,
    MemoryAccess,
    MemoryInstance,
    MemoryLevelConfig,
    OutputCollector,
    TensorRole,
    UnrollDimension,
)

# --- the datapath's shape ----------------------------------------------------

#: Output-channel lanes of the vector MAC, and the NMP's channel parallelism.
OUTPUT_LANES = 32
#: Reduction lanes, per operand width. Narrow operands pack denser.
INPUT_LANES = {4: 32, 8: 16}
#: Width of one activation word delivered to the MAC.
INPUT_WORD_BITS = 128
#: Width of one scalar -- a per-channel bias, a multiplier -- read beside the
#: operands. Narrower than the accumulator on purpose: the PPU adds its bias
#: after the multiplier and shift, in the output's units, not the sum's.
SCALAR_BITS = 8
#: The accumulator the MAC drains into.
ACCUMULATOR_BITS = 32
#: Values sharing one VSQ fine scale.
VSQ_BLOCK_VALUES = 32
#: Process node, for an energy model that asks.
TECHNOLOGY_NM = 5

#: Operand widths the MAC accepts, and what each implies. ``banks_per_value``
#: is the physical banking an 8-bit value needs; the output-channel alignment
#: is the granularity the datapath requires at that width.
OPERAND_MODES = {
    4: {"channel_alignment": 32, "banks_per_value": 1},
    8: {"channel_alignment": 16, "banks_per_value": 2},
}
ACCEPTED_OPERAND_BITS = frozenset(OPERAND_MODES)

#: Operations the NMP owns. The MAC handles GEMM-shaped work only, so an
#: operation in neither set has no unit on this machine.
NMP_OPERATIONS = frozenset(
    {
        "channel_shuffle",
        "channel_shuffle_copy",
        "depthwise_conv2d",
        "group_norm",
        "pool2d",
        "softmax",
        "upsample_nearest2d",
    }
)

#: Input-channel counts the MAC runs by masking its unused lanes rather than
#: being handed a padded operand. A grayscale stem is one channel; the
#: three-channel case is the ordinary colour first layer.
PARTIAL_MATRIX_INPUT_CHANNELS = frozenset({1, 3})

DTYPE_MODES = tuple(
    DTypeMode(
        operand_bits=bits,
        input_lanes=INPUT_LANES[bits],
        channel_alignment=mode["channel_alignment"],
        banks_per_value=mode["banks_per_value"],
    )
    for bits, mode in sorted(OPERAND_MODES.items())
)


def operand_mode(operand_bits: int) -> dict:
    """What the datapath does at one operand width.

    Fail closed: a width Alpha cannot run is a compile-time error rather than
    a silently substituted default, because the substitution would be a legal
    schedule for a machine that is not this one.
    """
    try:
        return OPERAND_MODES[operand_bits]
    except KeyError:
        raise ValueError(
            f"Magnet Alpha does not accept {operand_bits}-bit operands; "
            f"accepted: {sorted(ACCEPTED_OPERAND_BITS)}"
        ) from None


def input_lanes(operand_bits: int) -> int:
    """Reduction unrolling at one operand width."""
    operand_mode(operand_bits)
    return INPUT_LANES[operand_bits]


# --- the compute units -------------------------------------------------------

MATRIX_UNIT = ComputeUnit(
    name="vector_mac",
    lanes=OUTPUT_LANES,
    accepted_operations=frozenset({"gemm"}),
    accepted_operand_bits=ACCEPTED_OPERAND_BITS,
    unroll_dimensions=(
        # The declared extent is the 4-bit one, with the 8-bit case named as
        # the exception. Stating both as exceptions would leave no answer for
        # a caller that has no width in hand -- the padding pass, for one.
        UnrollDimension("IC", INPUT_LANES[4], dtype_extents=((8, INPUT_LANES[8]),)),
        UnrollDimension("OC", OUTPUT_LANES),
    ),
    input_word_bits=INPUT_WORD_BITS,
    scalar_bits=SCALAR_BITS,
)

NMP_UNIT = ComputeUnit(
    name="nmp",
    lanes=OUTPUT_LANES,
    accepted_operations=NMP_OPERATIONS,
    accepted_operand_bits=frozenset({8}),
    unroll_dimensions=(UnrollDimension("CHANNEL", OUTPUT_LANES),),
)

COMPUTE_UNITS = (MATRIX_UNIT, NMP_UNIT)

#: Which declared unit performs a named kind of work. Serialization and
#: conversion resolve a unit through this rather than by matching its name,
#: so renaming the silicon does not change the collateral's vocabulary.
COMPUTE_UNIT_BINDINGS = (
    ("matrix", MATRIX_UNIT.name),
    ("nmp", NMP_UNIT.name),
)


# --- the memory hierarchy ----------------------------------------------------

#: Relative access cost, in the scheduler's units: on-chip local storage
#: against the global SRAM an order of magnitude further away.
LOCAL_ACCESS_COST = 10
GLOBAL_ACCESS_COST = 100

#: L1: the buffers at the datapath. Every one of them serves the matrix unit
#: alone -- the NMP reads and writes the global SRAM directly, which is what
#: "near-memory" means here.
PE_LOCAL = MemoryLevelConfig(
    name="pe_local",
    instances=(
        MemoryInstance(
            name="input_buffer",
            bank_count=1,
            depth_per_bank=256,
            bits_per_bank_entry=INPUT_WORD_BITS,
            accepted_accesses=(
                MemoryAccess(TensorRole.ACTIVATION, MATRIX_UNIT.name),
            ),
            access_cost=LOCAL_ACCESS_COST,
        ),
        MemoryInstance(
            name="output_collector",
            bank_count=1,
            depth_per_bank=32,
            bits_per_bank_entry=OUTPUT_LANES * ACCUMULATOR_BITS,
            accepted_accesses=(
                MemoryAccess(TensorRole.OUTPUT, MATRIX_UNIT.name),
                MemoryAccess(TensorRole.PARTIAL_SUM, MATRIX_UNIT.name),
            ),
            access_cost=LOCAL_ACCESS_COST,
        ),
        MemoryInstance(
            name="weight_buffer",
            bank_count=32,
            depth_per_bank=64,
            bits_per_bank_entry=INPUT_WORD_BITS,
            accepted_accesses=(
                MemoryAccess(TensorRole.WEIGHT, MATRIX_UNIT.name),
            ),
            access_cost=LOCAL_ACCESS_COST,
        ),
    ),
    input_memory="input_buffer",
    output_memory="output_collector",
    weight_memory="weight_buffer",
)

#: L2: the global SRAM, split by role. The activation memory is the machine's
#: working set and the only thing the NMP touches; the weight memory feeds the
#: MAC and is not written by the datapath at all, which is why its write
#: bandwidth is a declared zero rather than an absent number.
GLOBAL = MemoryLevelConfig(
    name="global",
    instances=(
        MemoryInstance(
            name="global_activation",
            bank_count=64,
            depth_per_bank=1024,
            bits_per_bank_entry=INPUT_WORD_BITS,
            accepted_accesses=(
                MemoryAccess(TensorRole.ACTIVATION, NMP_UNIT.name),
                MemoryAccess(TensorRole.ACTIVATION, MATRIX_UNIT.name),
                MemoryAccess(TensorRole.BIAS, None),
                MemoryAccess(TensorRole.OUTPUT, None),
                MemoryAccess(TensorRole.SCALE, None),
                MemoryAccess(TensorRole.WEIGHT, NMP_UNIT.name),
                MemoryAccess(TensorRole.WEIGHT, MATRIX_UNIT.name),
            ),
            access_cost=GLOBAL_ACCESS_COST,
            access_granularity_bits=512,
            read_bandwidth=16.0,
            write_bandwidth=32.0,
        ),
        MemoryInstance(
            name="global_weight",
            bank_count=16,
            depth_per_bank=1024,
            bits_per_bank_entry=512,
            accepted_accesses=(
                MemoryAccess(TensorRole.BIAS, None),
                MemoryAccess(TensorRole.WEIGHT, MATRIX_UNIT.name),
            ),
            access_cost=GLOBAL_ACCESS_COST,
            access_granularity_bits=512,
            read_bandwidth=64.0,
            write_bandwidth=0,
        ),
    ),
    input_memory="global_activation",
    output_memory="global_activation",
    weight_memory="global_weight",
)

#: The collector the MAC drains through. It is also declared as an L1 memory
#: above, because the two describe different things: there it is capacity the
#: allocator sees, here it is the shape a single call's output tile is bounded
#: by. Deriving one from the other would lose the lane count.
OUTPUT_COLLECTOR = OutputCollector(
    name="output_collector",
    depth_words=32,
    lanes_per_word=OUTPUT_LANES,
    accumulator_bits=ACCUMULATOR_BITS,
)

#: The routes between the levels, per direction and per role. One number per
#: level cannot say that the weight path into the PE is four times the
#: activation path, or that the NMP's paths are symmetric while the MAC's are
#: not.
HIERARCHY_BANDWIDTHS = (
    HierarchyBandwidth(
        name="activation_to_pe_local",
        source="global",
        destination="pe_local",
        bandwidth_bits_per_cycle=128,
        roles=frozenset({TensorRole.ACTIVATION}),
    ),
    HierarchyBandwidth(
        name="weight_to_pe_local",
        source="global",
        destination="pe_local",
        bandwidth_bits_per_cycle=512,
        roles=frozenset({TensorRole.WEIGHT}),
    ),
    HierarchyBandwidth(
        name="pe_local_to_activation",
        source="pe_local",
        destination="global",
        bandwidth_bits_per_cycle=256,
        roles=frozenset({TensorRole.OUTPUT}),
    ),
    HierarchyBandwidth(
        name="activation_to_nmp",
        source="global",
        destination="nmp",
        bandwidth_bits_per_cycle=256,
        roles=frozenset({TensorRole.ACTIVATION}),
    ),
    HierarchyBandwidth(
        name="weight_to_nmp",
        source="global",
        destination="nmp",
        bandwidth_bits_per_cycle=256,
        roles=frozenset({TensorRole.WEIGHT}),
    ),
    HierarchyBandwidth(
        name="nmp_to_activation",
        source="nmp",
        destination="global",
        bandwidth_bits_per_cycle=256,
        roles=frozenset({TensorRole.OUTPUT}),
    ),
)

#: The whole global SRAM, as one number, for the passes that address L2 as a
#: pool. Both instances together: the split is real, but a search that reads
#: one capacity and is given only the activation memory would refuse every
#: layer whose weights are most of its footprint.
GLOBAL_MEMORY_BYTES = sum(
    instance.size_bytes for instance in GLOBAL.instances
)

__all__ = [
    "ACCEPTED_OPERAND_BITS",
    "ACCUMULATOR_BITS",
    "COMPUTE_UNITS",
    "COMPUTE_UNIT_BINDINGS",
    "DTYPE_MODES",
    "GLOBAL",
    "GLOBAL_MEMORY_BYTES",
    "HIERARCHY_BANDWIDTHS",
    "INPUT_LANES",
    "INPUT_WORD_BITS",
    "MATRIX_UNIT",
    "NMP_OPERATIONS",
    "NMP_UNIT",
    "OPERAND_MODES",
    "OUTPUT_COLLECTOR",
    "OUTPUT_LANES",
    "PARTIAL_MATRIX_INPUT_CHANNELS",
    "PE_LOCAL",
    "SCALAR_BITS",
    "TECHNOLOGY_NM",
    "VSQ_BLOCK_VALUES",
    "input_lanes",
    "operand_mode",
]
