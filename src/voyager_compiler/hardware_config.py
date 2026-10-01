"""The accelerator hardware description.

``AcceleratorConfig`` bundles every hardware knob the compiler needs — the PE
array, the on-chip L1 systolic buffers, the L2 scratchpad, DRAM, and the clock —
into one frozen object that ``transform()`` / ``compile()`` and their callees
pass around, instead of threading a dozen loose, drift-prone arguments through
every layer.

Physical units: ``dram_bandwidth`` is GB/s, ``dram_access_latency`` ns,
``frequency`` GHz, so bytes/cycle is ``dram_bandwidth / frequency`` and
per-transfer latency in cycles is ``dram_access_latency * frequency``.
``dram_energy_per_bit`` is pJ/bit, which ``dram_energy_per_byte`` turns into
J/B for the reporting energy columns.  The reporting model reads this object
directly as its cost knobs.
"""

from dataclasses import dataclass, fields
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
DEFAULT_DRAM_ENERGY_PJ_PER_BIT = 6.25


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

    ``matrix_backend=1`` describes the native INT8 CIM replacement for the
    systolic engine. Its configuration can be validated here and used by the
    standalone tiling search. Full graph transformation and instruction
    lowering for that backend are not implemented yet.
    """

    # Compute
    pe_array_size: Tuple[int, int] = DEFAULT_PE_ARRAY_SIZE
    vector_unit_width: Optional[int] = None  # None -> pe_array_size[1]
    matrix_vector_unit_width: Optional[int] = None  # None -> pe_array_size[1]
    accumulator_width: Optional[int] = None  # None -> vector_lanes
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
    dram_energy_per_bit: float = DEFAULT_DRAM_ENERGY_PJ_PER_BIT  # pJ/bit

    # Matrix backend: matches MATRIX_BACKEND in the accelerator build.
    # pe_array_size remains the complete (input, output) lane count for both.
    matrix_backend: int = 0  # 0 = systolic, 1 = CIM
    # Native INT8 CIM geometry; defaults match src/cim/CIMConfig.h.
    cim_macro_input_lanes: int = 64
    cim_macro_output_lanes: int = 8
    cim_weight_sets: int = 18
    cim_base_a_width: int = 4
    cim_base_b_width: int = 4
    cim_base_c_width: int = 20
    cim_macro_write_input_lanes: int = 1
    cim_mac_latency: int = 1
    cim_mode: int = 0
    cim_signed: bool = True
    cim_tile_input_axis_elements: int = 1
    cim_tile_output_axis_elements: int = 4
    cim_input_axis_tiles: int = 1
    cim_output_axis_tiles: int = 1
    # None means the full corresponding axis, as in the hardware defaults.
    cim_a_port_tiles: Optional[int] = None
    cim_b_port_tiles: Optional[int] = None
    cim_c_port_tiles: Optional[int] = None
    cim_c_beat_layout: int = 1
    cim_array_result_slots: Optional[int] = None
    cim_local_accum_contexts: int = 4

    def __post_init__(self):
        """Reject a reservation the rest of the compiler could not honour.

        Inert at the default of 0, so a config that names no scratchpad at
        all still constructs.

        Also validate the selected matrix backend and its CIM configuration.
        """
        if type(self.matrix_backend) is not int or self.matrix_backend not in (
            0,
            1,
        ):
            raise ValueError("matrix_backend must be 0 (systolic) or 1 (CIM)")
        if self.matrix_backend == 1:
            self._validate_cim()
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

    def _validate_cim(self):
        """Validate the native INT8 contract of CIMProcessor and CIMArray."""
        for name, axis in (
            ("cim_a_port_tiles", self.cim_input_axis_tiles),
            ("cim_b_port_tiles", self.cim_output_axis_tiles),
            ("cim_c_port_tiles", self.cim_output_axis_tiles),
            ("cim_array_result_slots", self.cim_input_axis_tiles),
        ):
            if getattr(self, name) is None:
                object.__setattr__(self, name, axis)

        for field in fields(self):
            if not field.name.startswith("cim_") or field.name in (
                "cim_mode",
                "cim_signed",
                "cim_c_beat_layout",
            ):
                continue
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer")
        if type(self.cim_mode) is not int or self.cim_mode not in (0, 1):
            raise ValueError("cim_mode must be 0 (parallel) or 1 (serial)")
        if self.cim_signed is not True:
            raise ValueError("CIMProcessor requires signed native INT8")
        if type(self.cim_c_beat_layout) is not int or self.cim_c_beat_layout != 1:
            raise ValueError("CIMProcessor requires output-major C beats (1)")
        if self.cim_macro_write_input_lanes != 1:
            raise ValueError("CIMProcessor writes one input lane per request")
        if 8 % self.cim_base_b_width:
            raise ValueError("cim_base_b_width must divide the INT8 weight")
        weight_slices = 8 // self.cim_base_b_width
        if self.cim_macro_output_lanes % weight_slices:
            raise ValueError(
                "cim_macro_output_lanes must be divisible by INT8 weight slices"
            )
        guard_width = max(1, (self.cim_macro_input_lanes - 1).bit_length())
        if (
            self.cim_mode == 1
            and self.cim_base_c_width <= self.cim_base_b_width + guard_width
        ):
            raise ValueError("cim_base_c_width cannot hold a bit-serial slice")
        if self.cim_input_lanes not in (4, 8, 16, 32, 64):
            raise ValueError("CIM input lanes must be 4, 8, 16, 32, or 64")
        if self.pe_array_size != (self.cim_input_lanes, self.cim_output_lanes):
            raise ValueError(
                "pe_array_size must match the CIM geometry: "
                f"{self.cim_input_lanes},{self.cim_output_lanes}"
            )
        if self.cim_a_port_tiles != self.cim_input_axis_tiles:
            raise ValueError("CIMProcessor requires the full input axis A port")
        if self.cim_c_port_tiles != self.cim_output_axis_tiles:
            raise ValueError("CIMProcessor requires a full output axis C port")
        if self.cim_output_axis_tiles % self.cim_b_port_tiles:
            raise ValueError(
                "cim_b_port_tiles must divide cim_output_axis_tiles"
            )
        if self.cim_array_result_slots < self.cim_input_axis_tiles:
            raise ValueError(
                "CIM needs at least one result slot per input tile"
            )
        if self.cim_weight_sets > 0xFFFF:
            raise ValueError(
                "cim_weight_sets exceeds the 16-bit schedule limit"
            )
        if (
            type(self.input_buffer_size) is not int
            or self.input_buffer_size <= 0
        ):
            raise ValueError("CIM requires a positive input_buffer_size")
        if (
            type(self.accum_buffer_size) is not int
            or self.accum_buffer_size < self.cim_local_accum_contexts
        ):
            raise ValueError("CIM local contexts must fit in the accum buffer")

    @property
    def cim_input_lanes(self) -> int:
        """Complete CIM input axis, in native INT8 lanes."""
        return (
            self.cim_macro_input_lanes
            * self.cim_tile_input_axis_elements
            * self.cim_input_axis_tiles
        )

    @property
    def cim_output_lanes(self) -> int:
        """Logical outputs after allocating macro lanes to weight slices."""
        return (
            self.cim_macro_output_lanes
            // (8 // self.cim_base_b_width)
            * self.cim_tile_output_axis_elements
            * self.cim_output_axis_tiles
        )

    def require_systolic_mapping(self):
        """Keep CIM configs out of the existing systolic lowering path."""
        if self.matrix_backend == 1:
            raise NotImplementedError(
                "CIM tiling is supported, but CIM instruction lowering "
                "is not implemented yet"
            )

    @property
    def vector_lanes(self) -> int:
        """Vector-unit lane count: its own width, else the PE array columns."""
        if self.vector_unit_width is not None:
            return self.vector_unit_width
        return self.pe_array_size[1]

    @property
    def matrix_vector_lanes(self) -> int:
        """Matrix-vector unit width in elements: its own, else the PE array
        columns."""
        if self.matrix_vector_unit_width is not None:
            return self.matrix_vector_unit_width
        return self.pe_array_size[1]

    @property
    def accumulator_lanes(self) -> int:
        """Channels one vector-unit fetch of a pool covers (the accelerator's
        ACCUMULATOR_WIDTH): its own width, else the vector unit's lanes."""
        if self.accumulator_width is not None:
            return self.accumulator_width
        return self.vector_lanes

    @property
    def dram_energy_per_byte(self) -> float:
        """DRAM access energy in joules per byte, from the pJ/bit figure."""
        return self.dram_energy_per_bit * 8 * 1e-12

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
            matrix_vector_unit_width=args.matrix_vector_unit_width,
            accumulator_width=args.accumulator_width,
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
            dram_energy_per_bit=args.dram_energy_per_bit,
            matrix_backend=args.matrix_backend,
            **{
                field.name: getattr(args, field.name)
                for field in fields(cls)
                if field.name.startswith("cim_")
            },
        )
