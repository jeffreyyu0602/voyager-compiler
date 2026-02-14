from typing import Dict, Any, Callable

import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import peft.tuners.lora as lora

from transformers.activations import GELUActivation
from transformers.models import llama, mobilebert
from transformers.pytorch_utils import Conv1D

import voyager_compiler.modules.qat as nnqat


DEFAULT_QAT_MODULE_MAPPINGS: Dict[Callable, Any] = {
    nn.Conv2d: nnqat.Conv2d,
    nn.Conv3d: nnqat.Conv3d,
    nn.Linear: nnqat.Linear,
    lora.Linear: nnqat.LoraLinear,
    # Intrinsic modules:
    nni.ConvBn1d: nnqat.ConvBn1d,
    nni.ConvBn2d: nnqat.ConvBn2d,
    nni.ConvBn3d: nnqat.ConvBn3d,
}

QCONFIG_PROPAGATE_MODULE_CLASS_LIST = {
    'activation': [
        nn.ReLU,
        nn.GELU,
        nn.Softmax,
        GELUActivation,
    ],
    'gemm': [
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.Linear,
        Conv1D,
    ],
    'layernorm': [
        nn.LayerNorm,
        llama.modeling_llama.LlamaRMSNorm,
        mobilebert.modeling_mobilebert.NoNorm,
    ]
}
