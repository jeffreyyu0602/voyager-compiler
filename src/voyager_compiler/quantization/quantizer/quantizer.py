from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

from torch import Tensor
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.fx import Node
from torchao.quantization.pt2e import ObserverOrFakeQuantize
from torchao.quantization.pt2e.quantizer.quantizer import QuantizationSpecBase

from voyager_compiler.quantization.fake_quantize import FusedAmaxObsFakeQuantize
from voyager_compiler.quantization.qspec import QScheme, parse_spec_fields

__all__ = [
    "QuantizationSpec",
]


@dataclass(eq=True)
class QuantizationSpec(QuantizationSpecBase):
    """Quantization spec for common operators that allows user to specify how to
    quantize a Tensor, this includes dtype, qscheme, quant_max etc.
    """

    dtype: str
    observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        FusedAmaxObsFakeQuantize
    )
    quant_min: Optional[float] = None
    quant_max: Optional[float] = None
    qscheme: Optional[QScheme] = None
    amax_history_len: Optional[int] = None
    ch_axis: Optional[Union[int, List[int]]] = None
    block_size: Optional[Union[int, List[int]]] = None
    scale_dtype: Optional[str] = None
    outlier_threshold: Optional[float] = None
    outlier_pct: Optional[float] = None
    is_dynamic: bool = False  # required by sharing nodes

    @staticmethod
    def from_str(s):
        """Build a spec from a comma-separated spec string."""
        return QuantizationSpec(**parse_spec_fields(s))

    def __post_init__(self):
        if self.qscheme is not None and self.quant_max is None:
            raise ValueError("quant_max is required for quantization.")

        if (
            self.qscheme in [QScheme.MICROSCALING, QScheme.GROUP_WISE_AFFINE]
            and self.block_size is None
        ):
            raise ValueError("block_size is required for microscaling.")


EdgeOrNode = Union[Tuple[Node, Node], Node]


@dataclass(eq=True)
class DerivedQuantizationSpec(QuantizationSpecBase):
    """quantization spec for the Tensors whose quantization parameters are
    derived from other Tensors
    """

    derived_from: List[EdgeOrNode]
    derive_qparams_fn: Callable[
        [List[ObserverOrFakeQuantize]], Tuple[Tensor, Tensor]
    ]
    dtype: str
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    qscheme: Optional[QScheme] = None
