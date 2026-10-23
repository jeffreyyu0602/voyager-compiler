<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Magnet Alpha conversion instructions

This directory owns conversion of serialized Magnet Alpha compiler output.
Keep the converter independent from compiler implementation modules: it reads
the checked-in protobuf schema, `model.txt`, hardware metadata, schedules, and
tensor files from a compiler output directory.

Use repository-relative paths in documentation and tests. Do not add target
runtime launchers, farm-specific commands, private collateral locations, or
claims of hardware execution to this directory.

Run `test_convert.py` after changing converter behavior. Conversion validates
the structure and consistency of the emitted C-model-format data; it does not
establish numerical parity or successful execution on a target runtime.
