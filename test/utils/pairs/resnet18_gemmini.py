# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""ResNet-18 x Gemmini.

One characterized candidate: shared per-tensor signed INT8 on activations and
weights with an INT32 bias. "Shared" means both operands use the same spec,
which is what the Gemmini helper API expects.
"""

WORKLOAD = "resnet18"
TARGET = "gemmini"

QSPEC = "int8,qs=per_tensor_symmetric"
BIAS_QSPEC = "int32"

DEFAULT_CANDIDATE = "gemmini-w8a8"
CANDIDATES = {
    DEFAULT_CANDIDATE: {
        "activation": QSPEC,
        "output_activation": QSPEC,
        "weight": QSPEC,
        "bias": BIAS_QSPEC,
    },
}

ACCEPTANCE = {
    "minimum_cosine": 0.90,
    "minimum_sqnr_db": 5.0,
}

__all__ = [
    "ACCEPTANCE",
    "BIAS_QSPEC",
    "CANDIDATES",
    "DEFAULT_CANDIDATE",
    "QSPEC",
    "TARGET",
    "WORKLOAD",
]
