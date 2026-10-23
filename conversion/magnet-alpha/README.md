<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Voyager dump to Magnet Alpha C-model format

`convert.py` converts a serialized Magnet Alpha compiler output directory into
the layer-oriented format consumed by the Alpha C-model interface. It operates
only on compiler artifacts and does not launch a target runtime.

The input directory must contain `model.txt`, `hardware_config.json`, and the
tensor files referenced by the serialized operations. Use a freshly generated
dump from the same source revision as the converter.

From the repository root:

```bash
python conversion/magnet-alpha/convert.py PATH_TO_MODEL_DIRECTORY
```

Use `--output` to select a destination, `--no-tensors` for metadata-only
validation, and `--require-all-supported` to reject unsupported executable
operations. Run `python -m unittest discover -s conversion/magnet-alpha -p
'test_convert.py'` to exercise the converter tests.

Successful conversion means the generated metadata and tensor encodings are
internally consistent with the converter's supported ABI. It is not evidence
of numerical equivalence or successful execution on external hardware or a
runtime model.
