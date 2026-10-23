# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""YOLOv5n-0.5 policy required by the Magnet Alpha datapath.

Most matrix convolutions use Alpha's W4A4 vector-scale format. Layers whose
operator or channel geometry requires the eight-bit path are named explicitly;
the depthwise subset uses the NMP's per-tensor weight scale. This is a compile
policy, not an accuracy claim.
"""

WORKLOAD = "yolo5n-0.5"
TARGET = "magnet-alpha"

W4A4_ACTIVATION_QSPEC = (
    "int4,qs=vsq,coarse=int8,cqs=per_tensor_symmetric,"
    "bs=32,scale=uint8,sqmax=8,csqmax=126"
)
W4A4_WEIGHT_QSPEC = (
    "int4,qs=vsq,coarse=int8,cqs=per_channel_symmetric,cax=0,"
    "bs=32,scale=uint8,qmin=-7,qmax=7"
)
W8A8_ACTIVATION_QSPEC = "int8,qs=per_tensor_symmetric,sqmax=126,rnd=floor"
W8_WEIGHT_QSPEC = "int8,qs=per_channel_symmetric,ax=0"
DEPTHWISE_W8_WEIGHT_QSPEC = "int8,qs=per_tensor_symmetric"
NONGEMM_QSPEC = "int8,qs=per_tensor_symmetric,sqmax=126,rnd=floor"

DEFAULT_CANDIDATE = "alpha-mixed-w4a4-v1"
CANDIDATES = {
    DEFAULT_CANDIDATE: {
        "activation": W4A4_ACTIVATION_QSPEC,
        "output_activation": None,
        "weight": W4A4_WEIGHT_QSPEC,
        "bias": None,
    },
}

W8_LAYERS = (
    "model.0.stem_1.conv",
    "model.0.stem_2a.conv",
    "model.0.stem_2b.conv",
    "model.0.stem_3.conv",
    "model.1.branch1.0",
    "model.1.branch1.2",
    "model.1.branch2.0",
    "model.1.branch2.3",
    "model.2.0.branch2.0",
    "model.2.0.branch2.3",
    "model.2.1.branch2.0",
    "model.2.1.branch2.3",
    "model.2.2.branch2.0",
    "model.2.2.branch2.3",
    "model.3.branch1.0",
    "model.3.branch2.0",
    "model.3.branch2.3",
    "model.4.0.branch2.0",
    "model.4.0.branch2.3",
    "model.4.1.branch2.0",
    "model.4.1.branch2.3",
    "model.4.2.branch2.0",
    "model.4.2.branch2.3",
    "model.4.3.branch2.0",
    "model.4.3.branch2.3",
    "model.4.4.branch2.0",
    "model.4.4.branch2.3",
    "model.4.5.branch2.0",
    "model.4.5.branch2.3",
    "model.4.6.branch2.0",
    "model.4.6.branch2.3",
    "model.5.branch1.0",
    "model.5.branch2.0",
    "model.5.branch2.3",
    "model.6.0.branch2.0",
    "model.6.0.branch2.3",
    "model.6.1.branch2.0",
    "model.6.1.branch2.3",
    "model.6.2.branch2.0",
    "model.6.2.branch2.3",
    "model.7.conv",
    "model.10.cv1.conv",
    "model.10.cv2.conv",
    "model.14.cv1.conv",
    "model.14.cv2.conv",
    "model.17.cv1.conv",
    "model.17.cv2.conv",
    "model.20.cv1.conv",
    "model.20.cv2.conv",
)

DEPTHWISE_W8_LAYERS = (
    "model.1.branch1.0",
    "model.1.branch2.3",
    "model.2.0.branch2.3",
    "model.2.1.branch2.3",
    "model.2.2.branch2.3",
    "model.3.branch1.0",
    "model.3.branch2.3",
    "model.4.0.branch2.3",
    "model.4.1.branch2.3",
    "model.4.2.branch2.3",
    "model.4.3.branch2.3",
    "model.4.4.branch2.3",
    "model.4.5.branch2.3",
    "model.4.6.branch2.3",
    "model.5.branch1.0",
    "model.5.branch2.3",
    "model.6.0.branch2.3",
    "model.6.1.branch2.3",
    "model.6.2.branch2.3",
)

HEAD_LAYERS = (
    "model.21.m.0",
    "model.21.m.1",
    "model.21.m.2",
)

INPUT_CHANNEL_PADDING = (
    ("model.1.branch1.2", 32),
    ("model.1.branch2.0", 32),
)

__all__ = [
    "CANDIDATES",
    "DEFAULT_CANDIDATE",
    "DEPTHWISE_W8_LAYERS",
    "DEPTHWISE_W8_WEIGHT_QSPEC",
    "HEAD_LAYERS",
    "INPUT_CHANNEL_PADDING",
    "NONGEMM_QSPEC",
    "TARGET",
    "W8A8_ACTIVATION_QSPEC",
    "W8_LAYERS",
    "W8_WEIGHT_QSPEC",
    "WORKLOAD",
]
