"""Modules swapped into a float model before or during quantization."""

from voyager_compiler.quantization.modules.llama_attention_kivi import (
    LlamaAttentionKIVI,
    swap_llama_attention,
)
from voyager_compiler.quantization.modules.posit_softmax import Softmax

__all__ = [
    "LlamaAttentionKIVI",
    "Softmax",
    "swap_llama_attention",
]
