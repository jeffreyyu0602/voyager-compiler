"""The accelerator hardware description.

``AcceleratorConfig`` bundles every hardware knob the compiler needs — the PE
array, the on-chip L1 systolic buffers, the L2 scratchpad, DRAM, and the clock —
into one frozen object that ``transform()`` / ``compile()`` and their callees
pass around, instead of threading a dozen loose, drift-prone arguments through
every layer.

Physical units: ``dram_bandwidth`` is GB/s, ``dram_access_latency`` ns,
``frequency`` GHz, so bytes/cycle is ``dram_bandwidth / frequency`` and
per-transfer latency in cycles is ``dram_access_latency * frequency``.  The
reporting model reads this object directly as its cost knobs.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

DEFAULT_PE_ARRAY_SIZE = (32, 32)
DEFAULT_FREQUENCY_GHZ = 1.0
DEFAULT_INPUT_BUFFER_SIZE = 1024
DEFAULT_WEIGHT_BUFFER_SIZE = 1024
DEFAULT_ACCUM_BUFFER_SIZE = 1024
DEFAULT_SCRATCHPAD_OFFSET = 0
DEFAULT_DOUBLE_BUFFERED_L2 = True
DEFAULT_DRAM_SIZE_GB = 16.0
DEFAULT_DRAM_BANDWIDTH_GBS = 64.0
DEFAULT_DRAM_ACCESS_LATENCY_NS = 100.0


@dataclass(frozen=True)
class AcceleratorConfig:
    """Compiler-visible hardware parameters for Voyager accelerators.

    Voyager couples two programmable compute engines: a weight-stationary
    2-D systolic array for convolution and GEMM, and a multi-stage vector
    unit for elementwise operations, reductions, nonlinear activations,
    normalization, and quantization. During a matrix tile, weights remain
    in the processing elements, activations flow horizontally, and partial
    sums propagate vertically. The matrix output can stream directly into
    the vector unit for operator fusion or pass through a double-buffered
    accumulation buffer to decouple the two engines.

    The systolic array uses dedicated input, weight, and accumulation
    buffers backed by a banked L2 scratchpad. This configuration describes
    the accelerator's compute parallelism, clock frequency, and on-chip
    memory hierarchy. The optional DRAM capacity, bandwidth, and latency
    parameters are later system-modeling extensions, rather than parameters
    of the Voyager accelerator template described in the paper.
    """

    # Compute
    pe_array_size: Tuple[int, int] = DEFAULT_PE_ARRAY_SIZE
    vector_unit_width: Optional[int] = None  # None -> pe_array_size[1]
    frequency: float = DEFAULT_FREQUENCY_GHZ  # accelerator clock
    # L1 systolic buffers (# elements)
    input_buffer_size: Optional[int] = DEFAULT_INPUT_BUFFER_SIZE
    weight_buffer_size: Optional[int] = DEFAULT_WEIGHT_BUFFER_SIZE
    accum_buffer_size: Optional[int] = DEFAULT_ACCUM_BUFFER_SIZE
    double_buffered_accum_buffer: bool = False
    # L2 scratchpad
    scratchpad_size: Optional[int] = None
    scratchpad_offset: int = DEFAULT_SCRATCHPAD_OFFSET  # bytes at the base
    num_banks: Optional[int] = None
    bank_width: Optional[int] = None
    double_buffered_l2: bool = DEFAULT_DOUBLE_BUFFERED_L2
    # L3 DRAM
    dram_size: Optional[float] = DEFAULT_DRAM_SIZE_GB
    dram_bandwidth: Optional[float] = DEFAULT_DRAM_BANDWIDTH_GBS
    dram_access_latency: Optional[float] = DEFAULT_DRAM_ACCESS_LATENCY_NS

    def __post_init__(self):
        """Reject a reservation the rest of the compiler could not honour.

        Inert at the default of 0, so a config that names no scratchpad at
        all still constructs.
        """
        if self.scratchpad_offset < 0:
            raise ValueError(
                f"scratchpad_offset {self.scratchpad_offset} is negative"
            )
        if not self.scratchpad_offset:
            return
        if self.scratchpad_size is None:
            raise ValueError(
                "scratchpad_offset needs a scratchpad_size to reserve from"
            )
        if self.scratchpad_offset >= self.scratchpad_size:
            raise ValueError(
                f"scratchpad_offset {self.scratchpad_offset} leaves nothing of "
                f"scratchpad_size {self.scratchpad_size}"
            )
        # A reservation that split a bank would put the planner's bank-aligned
        # groups and the tile search's bank budget on different geometries.
        bank = self.bank_size
        if bank is not None and self.scratchpad_offset % bank:
            raise ValueError(
                f"scratchpad_offset {self.scratchpad_offset} is not a multiple "
                f"of the {bank} B bank size"
            )

    @property
    def vector_lanes(self) -> int:
        """Vector-unit lane count: its own width, else the PE array columns."""
        if self.vector_unit_width is not None:
            return self.vector_unit_width
        return self.pe_array_size[1]

    @property
    def bytes_per_cycle(self) -> float:
        return self.dram_bandwidth / self.frequency

    @property
    def access_latency_cycles(self) -> float:
        return self.dram_access_latency * self.frequency

    @property
    def num_slots(self) -> int:
        """Banks one buffer occupies: two when it is ping-ponged, else one."""
        return 2 if self.double_buffered_l2 else 1

    @property
    def bank_size(self) -> Optional[int]:
        if self.num_banks is None:
            return None
        return self.scratchpad_size // self.num_banks

    @property
    def usable_scratchpad_size(self) -> Optional[int]:
        """What the plan may spend: the SRAM above ``scratchpad_offset``."""
        if self.scratchpad_size is None:
            return None
        return self.scratchpad_size - self.scratchpad_offset

    @property
    def usable_banks(self) -> Optional[int]:
        """Banks above the reservation.

        Counted off ``num_banks`` rather than divided out of the usable
        bytes, so it is exactly ``num_banks`` at an offset of 0 even when the
        banks do not divide the SRAM evenly.
        """
        if self.num_banks is None or self.bank_size is None:
            return None
        return self.num_banks - self.scratchpad_offset // self.bank_size

    @classmethod
    def from_args(cls, args) -> "AcceleratorConfig":
        """Build the config from parsed CLI args (``add_compile_args``)."""
        return cls(
            pe_array_size=args.pe_array_size,
            vector_unit_width=args.vector_unit_width,
            frequency=args.frequency,
            input_buffer_size=args.input_buffer_size,
            weight_buffer_size=args.weight_buffer_size,
            accum_buffer_size=args.accum_buffer_size,
            double_buffered_accum_buffer=args.double_buffered_accum_buffer,
            scratchpad_size=args.scratchpad_size,
            scratchpad_offset=args.scratchpad_offset,
            num_banks=args.num_banks,
            bank_width=args.bank_width,
            double_buffered_l2=args.double_buffered_l2,
            dram_size=args.dram_size,
            dram_bandwidth=args.dram_bandwidth,
            dram_access_latency=args.dram_access_latency,
        )
