# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Project a canonical mapping into Magnet Alpha's endpoint factor names.

This module deliberately contains no mapper integration. The rule-based
scheduler's three-level :class:`~scheduler.types.MappingPoint` is the source,
and these names are the Alpha ABI carried by ``LocalScheduleResult``.
"""

from scheduler.types import Loop, MappingPoint


def mapping_point_to_alpha_factors(mapping: MappingPoint) -> dict[str, int]:
    """Return the complete Alpha GEMM factor record for a three-level map."""
    if not isinstance(mapping, MappingPoint):
        raise TypeError("Alpha factor projection requires a MappingPoint")
    if mapping.num_levels != 3:
        raise ValueError(
            "the rule-based Alpha GEMM projection requires three levels"
        )

    blockings = mapping.loop_blockings
    partitionings = mapping.loop_partitionings
    return {
        "c0": partitionings[Loop.IC][0],
        "c1": blockings[Loop.IC][1],
        "c2": blockings[Loop.IC][2],
        "c3": 1,
        "k0": partitionings[Loop.OC][0],
        "k1": blockings[Loop.OC][1],
        "k2": blockings[Loop.OC][2],
        "k3": 1,
        "p0": blockings[Loop.OY][1],
        "p1": blockings[Loop.OY][1],
        "p2": blockings[Loop.OY][2],
        "p2s": 1,
        "p3": 1,
        "q0": blockings[Loop.OX][1],
        "q1": blockings[Loop.OX][1],
        "q2": blockings[Loop.OX][2],
        "q2s": 1,
        "q3": 1,
        "r": blockings[Loop.FY][1],
        "s": blockings[Loop.FX][1],
        "batch_size": blockings[Loop.ON][2],
    }


__all__ = ["mapping_point_to_alpha_factors"]
