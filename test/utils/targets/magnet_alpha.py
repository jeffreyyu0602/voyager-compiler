# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The Magnet Alpha target.

Alpha is a 32-lane vector MAC with a narrower unit behind it -- the NMP --
for the operations the MAC cannot express: depthwise convolution, pooling,
softmax, nearest upsampling, channel shuffle.

On where the machine lives
--------------------------
It used to live here, as plain module data, because the compiler's config had
no field for most of it. It does now, so the machine has moved to
``voyager_compiler.codegen.targets.magnet_alpha.hardware`` -- one home for a
fact about the silicon -- and this module imports it. What stays here is what
a *profile* owns: which of those numbers the command line may override, and
which quantizer and fusion table the target is invoked with.

The distinction matters because the two are read at different times. The
machine is read during compilation, with the graph in hand; the profile is
read before it, to decide how the model is prepared.
"""

from voyager_compiler.codegen.targets.magnet_alpha.hardware import (
    ACCEPTED_OPERAND_BITS,
    ACCUMULATOR_BITS,
    COMPUTE_UNITS,
    COMPUTE_UNIT_BINDINGS,
    DTYPE_MODES,
    GLOBAL,
    GLOBAL_MEMORY_BYTES,
    HIERARCHY_BANDWIDTHS,
    INPUT_WORD_BITS,
    MATRIX_UNIT,
    NMP_OPERATIONS,
    OUTPUT_COLLECTOR,
    OUTPUT_LANES,
    PARTIAL_MATRIX_INPUT_CHANNELS,
    PE_LOCAL,
    TECHNOLOGY_NM,
    VSQ_BLOCK_VALUES,
    input_lanes,
    operand_mode,
)

from . import generic

TARGET = "magnet-alpha"

#: One bank per operand group the tile search recognises. Alpha's global SRAM
#: is two role-split memories, and the search addresses L2 as one banked pool,
#: so this is the modelling stand-in rather than a fact about the silicon --
#: which is exactly why it is a permitted override below.
MAPPING_NUM_BANKS = 4
MAPPING_BANK_WIDTH = INPUT_WORD_BITS // 8

#: The structural machine. ``build_config`` derives the flat fields each of
#: these owns, so ``pe_array_size`` and ``scratchpad_size`` are not restated:
#: they follow from the matrix unit's unrolling and the global memory's size.
HARDWARE_PROFILE = {
    "compute": MATRIX_UNIT,
    "local_memory": PE_LOCAL,
    "global_memory": GLOBAL,
    # Alpha's characterised hierarchy stops at the global SRAM. Tiles are
    # still staged from DRAM, so the default external memory stands in as the
    # staging source rather than being declared absent.
}

#: What the split global level cannot supply on its own. The tile search reads
#: one capacity and one banking, and a target that declares neither is
#: invisible to it.
MONOLITHIC_STAND_IN = {
    "scratchpad_size": GLOBAL_MEMORY_BYTES,
    "num_banks": MAPPING_NUM_BANKS,
    "bank_width": MAPPING_BANK_WIDTH,
}

HARDWARE_DEFAULTS = {**HARDWARE_PROFILE, **MONOLITHIC_STAND_IN}

#: Machine fields the command line may override. The two mapping stand-ins
#: are negotiable precisely because they are modelling choices; the declared
#: geometry is not.
PERMITTED_OVERRIDES = frozenset({"scratchpad_size", "num_banks", "bank_width"})

#: Capability the CLI has no say over: what the units physically accept, and
#: how the machine is wired.
_CAPABILITY = {
    "compute_units": COMPUTE_UNITS,
    "compute_unit_bindings": COMPUTE_UNIT_BINDINGS,
    "dtype_modes": DTYPE_MODES,
    "output_collector": OUTPUT_COLLECTOR,
    "hierarchy_bandwidths": HIERARCHY_BANDWIDTHS,
    "technology_nm": TECHNOLOGY_NM,
    # The NMP reads a whole 32-channel word, so a depthwise convolution whose
    # channel count is not a multiple of that is padded rather than masked.
    "pad_depthwise_channels": True,
    # The MAC can mask unused reduction lanes for a 1- or 3-channel first
    # layer, so those are run partially rather than padded up to 32.
    "partial_matrix_input_channels": PARTIAL_MATRIX_INPUT_CHANNELS,
}


def apply_defaults(args):
    """No flat machine values to fill; the structure supplies them.

    ``build_config`` derives ``pe_array_size`` and the buffer sizes from the
    declared unit and memories, so filling ``args`` first would state the same
    numbers twice and give the copy here the chance to be the stale one.
    """
    return args


def accelerator_config(args):
    """Alpha's resources, from the profile, with a named CLI value winning."""
    from . import build_config

    return build_config(args, HARDWARE_DEFAULTS, extra=_CAPABILITY)


def vector_pipeline(args):
    """Alpha fuses the same operator sequences the default target does.

    An explicit delegation, not a copy. Alpha's real fusion table is a
    consequence of its PE and NMP datapaths, and neither the operators nor the
    datapath exist yet; authoring the table before them would be asserting a
    pipeline rather than deriving one.
    """
    return generic.vector_pipeline(args)


def quantizer(args):
    """Alpha's quantizer.

    Delegates until the mixed W4A4-VSQ / W8A8 operand specs are adopted. The
    workload's own policy is the thing that varies here, and it belongs to the
    ``(workload, target)`` pair rather than to the target alone.
    """
    return generic.quantizer(args)


__all__ = [
    "ACCEPTED_OPERAND_BITS",
    "ACCUMULATOR_BITS",
    "COMPUTE_UNITS",
    "COMPUTE_UNIT_BINDINGS",
    "DTYPE_MODES",
    "GLOBAL_MEMORY_BYTES",
    "HARDWARE_DEFAULTS",
    "HARDWARE_PROFILE",
    "MAPPING_BANK_WIDTH",
    "MAPPING_NUM_BANKS",
    "NMP_OPERATIONS",
    "OUTPUT_LANES",
    "PERMITTED_OVERRIDES",
    "TARGET",
    "VSQ_BLOCK_VALUES",
    "accelerator_config",
    "apply_defaults",
    "input_lanes",
    "operand_mode",
    "quantizer",
    "vector_pipeline",
]
