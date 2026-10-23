# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compiler-side backend for the ``gemmini`` target.

The machine is in :mod:`.hardware`; what is here is what the *target*
decides.
"""

from voyager_compiler.codegen.targets.gemmini.hardware import (
    COMPUTE_UNITS,
    DIM,
    DTYPE_MODES,
    INPUT_BITS,
)

TARGET = "gemmini"

__all__ = [
    "COMPUTE_UNITS",
    "DIM",
    "DTYPE_MODES",
    "INPUT_BITS",
    "TARGET",
    "apply_graph_rewrites",
]


def apply_graph_rewrites(model, config) -> None:
    """No graph rewrite of its own.

    Gemmini's units cover every operation in the exercised workloads.

    An operation no declared unit accepts is refused by
    ``assign_compute_units``, which reads ``COMPUTE_UNITS`` directly. An
    earlier note here said such a scan was wrong because the reference
    collateral records pooling as ordinary accelerator work. That
    collateral marks only ``pad`` unsupported -- but the lane it came from
    converted a single call, so what it shows is *candidate assignment*,
    not a capability audit. It cannot settle what a unit can issue.
    """
    del model, config
