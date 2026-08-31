"""Modules swapped into a float model before or during quantization."""

from voyager_compiler.quantization.modules.posit_softmax import Softmax

__all__ = ["Softmax"]
