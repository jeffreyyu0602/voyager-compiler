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

DEFAULT_DRAM_SIZE_GB = 16.0
DEFAULT_DRAM_BANDWIDTH_GBS = 64.0
DEFAULT_DRAM_ACCESS_LATENCY_NS = 100.0


@dataclass(frozen=True)
class AcceleratorConfig:
    """Full accelerator hardware architecture.

    ``scratchpad_size`` is the **per-buffer** L2 budget: with
    ``double_buffered_l2`` the physical SRAM is ``scratchpad_size * 2`` (two
    ping-pong buffers), and a tile is sized to fill one buffer, so tiling
    against ``scratchpad_size`` needs no halving.

    Concrete defaults live in ``cli_args.py`` (the single source of truth); the
    fields here default to ``None`` and just carry whatever the entry point was
    given.  ``vector_unit_width`` of ``None`` means the vector unit is as wide
    as the PE array's column count.
    """

    # Compute
    pe_array_size: Tuple[int, int]  # (rows, cols); systolic array unrolling
    vector_unit_width: Optional[int] = None  # None -> pe_array_size[1]
    frequency: float = 1.0  # GHz (accelerator clock)
    # L1 systolic buffers (# elements)
    input_buffer_size: Optional[int] = None
    weight_buffer_size: Optional[int] = None
    accum_buffer_size: Optional[int] = None
    double_buffered_accum_buffer: bool = False
    # L2 scratchpad
    scratchpad_size: Optional[int] = None  # bytes (per-buffer; --cache_size)
    num_banks: Optional[int] = None
    bank_width: Optional[int] = None
    double_buffered_l2: bool = False
    # DRAM
    dram_size: Optional[float] = None  # GB
    dram_bandwidth: Optional[float] = None  # GB/s
    dram_access_latency: Optional[float] = None  # ns

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
    def bank_size(self) -> Optional[int]:
        if self.num_banks is None:
            return None
        return self.scratchpad_size // self.num_banks

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
            scratchpad_size=args.cache_size,
            num_banks=args.num_banks,
            bank_width=args.bank_width,
            double_buffered_l2=args.double_buffered_l2,
            dram_size=args.dram_size,
            dram_bandwidth=args.dram_bandwidth,
            dram_access_latency=args.dram_access_latency,
        )
