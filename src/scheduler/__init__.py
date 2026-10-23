# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Voyager scheduling.

A scheduler decides how one operation is mapped onto a machine: loop order,
blocking, and what that costs. Which scheduler runs is a target's choice, so
this package holds the engines and the seam that selects between them, and the
compiler holds what it does with the answer.

``types``                  the vocabulary a scheduler's result is expressed in,
                           and ``SchedulerProviderSet``, the seam a target uses
                           to constrain an engine or supply its own
``providers``              the engines shipped with the compiler
``interstellar``           vendored external mapper; do not edit here
``interstellar_adapter``   translates between a machine description and what
                           that mapper needs, and adopts its answer back
"""
