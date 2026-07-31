"""Modules swapped into a float model before or during quantization."""

from .llama_attention_kivi import LlamaAttentionKIVI, swap_llama_attention
from .posit_softmax import Softmax

__all__ = [
    "LlamaAttentionKIVI",
    "Softmax",
    "swap_llama_attention",
]
