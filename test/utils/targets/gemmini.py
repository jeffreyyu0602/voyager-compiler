# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The Gemmini target.

A 16x16 systolic array with signed INT8 operands, a banked scratchpad and a
separate accumulator SRAM. It needs no block-scaled quantization:
shared per-tensor signed INT8 with an INT32 bias.
"""

import copy

from voyager_compiler.hardware_config import (
    AcceleratorConfig,
    ComputeUnit,
    DTypeMode,
    MemoryInstance,
    MemoryLevelConfig,
    TensorRole,
    UnrollDimension,
)

from . import generic

TARGET = "gemmini"


# --- the machine ------------------------------------------------------------

#: Systolic array edge; the array is DIM x DIM.
DIM = 16
INPUT_BITS = 8
OPERANDS_SIGNED = True
#: Width of a spatial output before requantization, and of the accumulator.
SPATIAL_OUTPUT_BITS = 20
ACCUMULATOR_BITS = 32

SCRATCHPAD_BANKS = 4
SCRATCHPAD_BYTES = 256 * 1024
#: One scratchpad row holds a whole input beat: DIM operands.
SCRATCHPAD_ROW_BITS = DIM * INPUT_BITS

ACCUMULATOR_BANKS = 2
ACCUMULATOR_BYTES = 64 * 1024

#: One operand width, and one unit that runs GEMM-shaped work.
DTYPE_MODES = (
    DTypeMode(
        operand_bits=INPUT_BITS,
        input_lanes=DIM,
        channel_alignment=DIM,
    ),
)
#: The helper API exposes pooling and nonlinear activations alongside the
#: systolic array, so those are accelerator work rather than boundaries.
HAS_MAX_POOL = True
HAS_NONLINEAR_ACTIVATIONS = True
HAS_DEPTHWISE_CONVS = True

SYSTOLIC_UNIT = ComputeUnit(
    name="systolic_array",
    lanes=DIM,
    accepted_operations=frozenset(
        {"gemm"}
        | ({"pool2d"} if HAS_MAX_POOL else set())
        | ({"depthwise_conv2d"} if HAS_DEPTHWISE_CONVS else set())
    ),
    accepted_operand_bits=frozenset({INPUT_BITS}),
    unroll_dimensions=(
        UnrollDimension("IC", DIM),
        UnrollDimension("OC", DIM),
    ),
)
COMPUTE_UNITS = (SYSTOLIC_UNIT,)


#: L2 (
#: unified): one addressable banked SRAM serving every operand -- the shape
#: the compiler's scratchpad model was written for, so the flat
#: ``scratchpad_size`` / ``num_banks`` / ``bank_width`` all derive from it.
SCRATCHPAD = MemoryLevelConfig(
    name="gemmini_scratchpad",
    instances=(
        MemoryInstance(
            name="scratchpad",
            bank_count=SCRATCHPAD_BANKS,
            depth_per_bank=(
                SCRATCHPAD_BYTES * 8 // (SCRATCHPAD_BANKS * SCRATCHPAD_ROW_BITS)
            ),
            bits_per_bank_entry=SCRATCHPAD_ROW_BITS,
            accepted_roles=frozenset(
                {
                    TensorRole.ACTIVATION,
                    TensorRole.WEIGHT,
                    TensorRole.OUTPUT,
                    TensorRole.PARTIAL_SUM,
                }
            ),
        ),
    ),
)

#: The authoritative machine profile.  No ``local_memory``: Gemmini's
#: accumulator SRAM is not modelled here, so the L1 triple keeps the
#: compiler's generic defaults rather than being given invented geometry.
HARDWARE_PROFILE = {
    "compute": SYSTOLIC_UNIT,
    "global_memory": SCRATCHPAD,
}

HARDWARE_DEFAULTS = HARDWARE_PROFILE


def _derived_arg_defaults():
    """The flat values the structures imply, for callers that read ``args``.

    The workload adapters build their own config straight from ``args``, so the
    structural profile has to be reducible to plain fields for them.  Building
    one config and reading it back keeps the numbers stated once.
    """
    resolved = AcceleratorConfig(
        compute=SYSTOLIC_UNIT,
        global_memory=SCRATCHPAD,
        pe_array_size=None,
        scratchpad_size=None,
        num_banks=None,
        bank_width=None,
    )
    return {
        field: getattr(resolved, field)
        for field in (
            "pe_array_size",
            "scratchpad_size",
            "num_banks",
            "bank_width",
        )
    }


_ARG_DEFAULTS = _derived_arg_defaults()

#: Machine fields the command line may override.  Gemmini's scratchpad is a
#: real addressable SRAM, so overriding it is a deliberate experiment rather
#: than a modelling stand-in -- ResNet-18 needs 969,808 B and does not fit the
#: declared 256 KiB, so the lanes pass an explicit larger value.
PERMITTED_OVERRIDES = frozenset({"scratchpad_size"})

#: Capability the CLI has no say over: what the units physically accept.
_CAPABILITY = {"dtype_modes": DTYPE_MODES, "compute_units": COMPUTE_UNITS}


def apply_defaults(args):
    """Fill this target's machine values into ``args``, CLI values winning.

    The workload adapters build their own ``AcceleratorConfig`` from ``args``
    (``models/utils.py``), so a target that only returned a config would be
    ignored. Filling ``args`` instead means every existing caller picks the
    machine up without being rewritten, and ``AcceleratorConfig.from_args``
    stays the single translation point.

    Returns a copy; the caller's namespace is not mutated.
    """
    updated = copy.copy(args)
    for field, value in _ARG_DEFAULTS.items():
        if getattr(updated, field, None) is None:
            setattr(updated, field, value)
    return updated


def accelerator_config(args):
    """Gemmini's resources, from the profile, with a named CLI value winning."""
    from . import build_config

    return build_config(args, HARDWARE_DEFAULTS, extra=_CAPABILITY)


def vector_pipeline(args):
    """Gemmini fuses the same sequences the default target does."""
    return generic.vector_pipeline(args)


def quantizer(args):
    """Gemmini quantizes with the ordinary per-tensor path."""
    return generic.quantizer(args)


__all__ = [
    "apply_defaults",
    "ACCUMULATOR_BANKS",
    "COMPUTE_UNITS",
    "DTYPE_MODES",
    "ACCUMULATOR_BITS",
    "ACCUMULATOR_BYTES",
    "DIM",
    "INPUT_BITS",
    "OPERANDS_SIGNED",
    "SCRATCHPAD_BANKS",
    "SCRATCHPAD_BYTES",
    "SCRATCHPAD_ROW_BITS",
    "SPATIAL_OUTPUT_BITS",
    "TARGET",
    "accelerator_config",
    "quantizer",
    "vector_pipeline",
]
