# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compiler-side backend for the ``magnet-alpha`` target.

Alpha is a 32-lane vector MAC paired with a near-memory processor. The machine
itself is in :mod:`.hardware`; what is here is what the *target* decides.

Today that is the machine, the address encoding, the fixed-point encoders and
the **operator set**: importing this module registers the ``magnet_alpha``
torch.library namespace, so the operators exist as soon as the target does.
That import is a side effect and is therefore done here rather than left to
whichever pass happens to need it first -- an operator that registers only
when someone imports the right module is one that is missing exactly when a
rewrite tries to build it.

It also rewrites: ``apply_graph_rewrites`` is :mod:`.lowering`'s, and it
turns each quantized convolution boundary into Alpha's own accumulator plus
its PE or near-memory epilogue, and each channel shuffle -- ``cat``, regroup,
transpose, ``contiguous``, flatten -- into the single operation the endpoint
issues for it. So ``--target magnet-alpha`` does not produce the same graph
as ``generic``.

That hook is *re-exported*, not redefined. It was previously shadowed by a
no-op of the same name declared further down this module, written when the
operators did not exist; Python resolved the later definition, so the target
silently rewrote nothing and compiled a generic graph. A stub that shadows
the thing it stands in for is worse than no stub, because the seam still
looks bound.

What arrives here later, in the order it is needed:

* ``unsupported_policy`` -- and only where no legal placement exists, which
  for this workload is nowhere.
"""

from scheduler.magnet_alpha_scheduler import (
    MAGNET_ALPHA_SCHEDULER_PROVIDERS,
)

# Imported for its registration side effect: defining the operators.
from voyager_compiler.codegen.targets.magnet_alpha import ops  # noqa: F401
from voyager_compiler.codegen.targets.magnet_alpha.lowering import (
    apply_graph_rewrites,
    apply_memory_rewrites,
)
from voyager_compiler.codegen.targets.magnet_alpha.calibration import (
    cleanup_quantization_preparation,
    finalize_quantization_calibration,
    prepare_quantization,
    prepare_quantization_recalibration,
)
from voyager_compiler.codegen.targets.magnet_alpha.pipeline import gen_code
from voyager_compiler.codegen.targets.magnet_alpha.serialization import (
    layer_table,
)
from voyager_compiler.codegen.targets.magnet_alpha.storage import (
    EXECUTION_MODES,
)
from voyager_compiler.codegen.targets.magnet_alpha.traits import (
    EXECUTION_FUSION_PATTERNS,
    prepared_domain_rules,
)
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
    NMP_UNIT,
    OUTPUT_COLLECTOR,
    OUTPUT_LANES,
    PARTIAL_MATRIX_INPUT_CHANNELS,
    PE_LOCAL,
    SCALAR_BITS,
    TECHNOLOGY_NM,
    VSQ_BLOCK_VALUES,
    input_lanes,
    operand_mode,
)

TARGET = "magnet-alpha"
SCHEDULER_PROVIDERS = MAGNET_ALPHA_SCHEDULER_PROVIDERS

__all__ = [
    "ACCEPTED_OPERAND_BITS",
    "EXECUTION_FUSION_PATTERNS",
    "EXECUTION_MODES",
    "ACCUMULATOR_BITS",
    "COMPUTE_UNITS",
    "cleanup_quantization_preparation",
    "finalize_quantization_calibration",
    "COMPUTE_UNIT_BINDINGS",
    "DTYPE_MODES",
    "GLOBAL",
    "GLOBAL_MEMORY_BYTES",
    "HIERARCHY_BANDWIDTHS",
    "INPUT_WORD_BITS",
    "MATRIX_UNIT",
    "NMP_OPERATIONS",
    "NMP_UNIT",
    "OUTPUT_COLLECTOR",
    "OUTPUT_LANES",
    "PARTIAL_MATRIX_INPUT_CHANNELS",
    "PE_LOCAL",
    "SCALAR_BITS",
    "SCHEDULER_PROVIDERS",
    "TARGET",
    "TECHNOLOGY_NM",
    "VSQ_BLOCK_VALUES",
    "apply_graph_rewrites",
    "apply_memory_rewrites",
    "gen_code",
    "input_lanes",
    "layer_table",
    "operand_mode",
    "prepared_domain_rules",
    "prepare_quantization",
    "prepare_quantization_recalibration",
]
