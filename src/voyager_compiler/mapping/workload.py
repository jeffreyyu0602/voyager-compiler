# Describe matrix and convolution work independently of compiler formats
from dataclasses import dataclass, asdict
from math import prod
from typing import Tuple
from .timing.transfer import ceil_div

# Describe a padded dense matrix or unit-dilation convolution
@dataclass(frozen=True)
class Workload:
    input_x: int
    input_y: int
    input_channels: int
    output_channels: int
    filter_x: int = 1
    filter_y: int = 1
    stride: int = 1
    padding: int = 0
    has_bias: bool = False
    output_to_memory: bool = True
    input_transpose: bool = False
    weight_transpose: bool = False
    logical_channels: Tuple[int, int] = ()
    padded_channels: Tuple[int, int] = ()
    input_bits: int = 8
    weight_bits: int = 8

    # Require explicit tensor extents and unambiguous command flags
    def __post_init__(self):
        for name in ("logical_channels", "padded_channels"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if bool(self.logical_channels) != bool(self.padded_channels):
            raise ValueError("logical and padded channels must be supplied together")
        if self.logical_channels and (len(self.logical_channels) != 2 or len(self.padded_channels) != 2
                or any(type(v) is not int or v < 0 for v in self.logical_channels)
                or any(type(v) is not int or v <= 0 for v in self.padded_channels)
                or any(a > b for a, b in zip(self.logical_channels, self.padded_channels))):
            raise ValueError("logical channels must fit the padded channel extents")
        for name, value in asdict(self).items():
            if name in ("logical_channels", "padded_channels"):
                continue
            if name in ("has_bias", "output_to_memory", "input_transpose", "weight_transpose"):
                if type(value) is not bool:
                    raise ValueError(f"{name} must be boolean")
            elif type(value) is not int or value < (0 if name == "padding" else 1):
                raise ValueError(f"{name} has an invalid dimension")

    # Average useful channel fraction across the operation's repeated L2 slices
    @property
    def channel_useful_fraction(self):
        return prod(self.logical_channels) / prod(self.padded_channels) if self.logical_channels else 1.0

    # Derive the output width under the compiler's symmetric-padding convention
    @property
    def output_x(self) -> int:
        return (self.input_x + 2 * self.padding - self.filter_x) // self.stride + 1

    # Derive the output height under the compiler's symmetric-padding convention
    @property
    def output_y(self) -> int:
        return (self.input_y + 2 * self.padding - self.filter_y) // self.stride + 1

    # Count valid convolution terms with work bounded by filter extent
    @property
    def useful_positions(self):
        def axis(inputs, outputs, kernel):
            return sum(max(0, min(outputs, (inputs - 1 + self.padding - k) // self.stride + 1)
                       - max(0, ceil_div(self.padding - k, self.stride))) for k in range(kernel))
        return axis(self.input_x, self.output_x, self.filter_x) * axis(self.input_y, self.output_y, self.filter_y)

    @property
    def useful_work_fraction(self):
        return self.useful_positions / (self.output_x * self.output_y * self.filter_x * self.filter_y) * self.channel_useful_fraction
