# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compiler-side backend for the default target.

The untargeted compiler, named so every caller can resolve a target rather
than special-casing ``None``. It defines no hooks: its behaviour *is* the
target-neutral pipeline.
"""

TARGET = "generic"

__all__ = ["TARGET"]
