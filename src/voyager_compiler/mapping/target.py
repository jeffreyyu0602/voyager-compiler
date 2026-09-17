# Load resolved hardware targets through one shared configuration contract
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from .timing.transfer import ceil_div

ACCUMULATION_POLICY = "static-output-address-prefix"


# Share physical widths, buffers, interfaces, and vector geometry between matrix backends
@dataclass(frozen=True, kw_only=True)
class MappingTarget:
    datatype: str
    input_bits: int
    weight_bits: int
    accum_bits: int  # Accumulation-buffer storage width, not an inferred compute precision
    input_buffer_words: int
    accum_buffer_words: int
    double_buffered_accum: bool
    ic_port_bits: int
    oc_port_bits: int
    hardware_options: dict = field(default_factory=dict)

    # Validate shared hardware facts before backend-specific constraints
    def __post_init__(self):
        self._validate_positive("input_bits", "weight_bits", "accum_bits", "input_buffer_words",
                                "accum_buffer_words", "ic_port_bits", "oc_port_bits")
        if type(self.double_buffered_accum) is not bool:
            raise ValueError("double_buffered_accum must be boolean")

    # Apply the same positive-integer rule to shared and backend-specific dimensions
    def _validate_positive(self, *names):
        for name in names:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    # Compare the actual parameters rather than an opaque identifier
    @property
    def configuration_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    # Preserve the flat configuration consumed by Python and C++
    def to_dict(self):
        return asdict(self)

    # Load and validate the supplied target configuration
    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


# Extend the common hardware configuration with SA geometry and weight storage
@dataclass(frozen=True)
class SATarget(MappingTarget):
    k: int
    n: int
    weight_buffer_words: int
    backend: str = "sa"

    # Add only SA-specific checks to the shared target validation
    def __post_init__(self):
        super().__post_init__()
        if self.backend != "sa":
            raise ValueError("SATarget requires backend=sa")
        self._validate_positive("k", "n", "weight_buffer_words")


# Dispatch a flat exported configuration to its backend-specific target
def load_target(path):
    values = json.loads(Path(path).read_text())
    if values.get("backend") == "sa":
        return SATarget(**values)
    if values.get("backend") == "cim":
        return CIMTarget(**values)
    raise ValueError("mapping target must identify backend=sa or backend=cim")

# Describe one effective current-hardware instance in resolved units
@dataclass(frozen=True)
class CIMTarget(MappingTarget):
    ch_in: int
    ch_out: int
    b_sets: int
    base_a_width: int
    base_b_width: int
    base_c_width: int
    write_ch_in: int
    mac_latency: int
    mode: int
    tile_input_axis_elements: int
    tile_output_axis_elements: int
    input_axis_tiles: int
    output_axis_tiles: int
    a_port_tiles: int
    b_port_tiles: int
    c_port_tiles: int
    c_beat_layout: int
    result_slots_per_output_lane: int
    local_accum_contexts: int
    b_store_scope: str = "L1-compute-resident"
    weight_policy: str = "fitting-sequence-replay-else-singleton-refetch"
    a_policy: str = "fresh-beat-per-MAC-full-array-multicast-reduce"
    accumulation_policy: str = ACCUMULATION_POLICY
    backend: str = "cim"

    # Reject descriptors outside the implemented processor and macro contracts
    def __post_init__(self):
        super().__post_init__()
        if self.backend != "cim":
            raise ValueError("CIMTarget requires backend=cim")
        if self.datatype not in ("INT8", "INT8_32"):
            raise ValueError("current CIM requires signed INT8 operands")
        if (self.input_bits, self.weight_bits) != (8, 8):
            raise ValueError("current CIM requires exported 8-bit input and weight widths")
        self._validate_positive("ch_in", "ch_out", "b_sets", "base_a_width", "base_b_width",
                                "base_c_width", "write_ch_in", "mac_latency", "tile_input_axis_elements",
                                "tile_output_axis_elements", "input_axis_tiles", "output_axis_tiles",
                                "a_port_tiles", "b_port_tiles", "c_port_tiles",
                                "result_slots_per_output_lane", "local_accum_contexts")
        if self.mode not in (0, 1) or type(self.mode) is not int:
            raise ValueError("mode must be bit-parallel 0 or bit-serial 1")
        if self.c_beat_layout != 1 or type(self.c_beat_layout) is not int:
            raise ValueError("current CIM requires output-major C beats")
        if self.weight_bits % self.base_b_width or self.ch_out % (self.weight_bits // self.base_b_width):
            raise ValueError("INT8 B slices must exactly divide CH_OUT")
        if self.k not in (4, 8, 16, 32, 64):
            raise ValueError("input extent is unsupported by current InputController")
        if self.write_ch_in != 1:
            raise ValueError("current CIM requires one weight row per write request")
        if self.a_port_tiles != self.input_axis_tiles or self.c_port_tiles != self.output_axis_tiles:
            raise ValueError("current CIM requires complete A and reduced C beats")
        if self.output_axis_tiles % self.b_port_tiles:
            raise ValueError("B ports must evenly divide the output axis")
        if self.result_slots_per_output_lane < self.input_axis_tiles:
            raise ValueError("each output lane must hold one unreduced result")
        if not 1 <= self.local_accum_contexts <= self.accum_buffer_words <= 65536:
            raise ValueError("local contexts and accumulation addresses exceed buffer capacity")
        if self.b_sets > 65535:
            raise ValueError("resident set count exceeds descriptor width")
        required_c = self.input_bits + self.weight_bits + max(1, (self.ch_in - 1).bit_length())
        required_c += (self.tile_input_axis_elements - 1).bit_length()
        required_c += (self.input_axis_tiles - 1).bit_length()
        if self.accum_bits < required_c:
            raise ValueError("accumulation datatype cannot hold the reduced array result")
        if self.mode == 1 and self.base_c_width <= self.base_b_width + max(1, (self.ch_in - 1).bit_length()):
            raise ValueError("bit-serial macro has no usable A slice")
        if self.ic_port_bits % 8 or self.oc_port_bits % 8:
            raise ValueError("external ports must carry whole bytes")
        expected = ("L1-compute-resident", "fitting-sequence-replay-else-singleton-refetch",
                    "fresh-beat-per-MAC-full-array-multicast-reduce", ACCUMULATION_POLICY)
        if (self.b_store_scope, self.weight_policy, self.a_policy, self.accumulation_policy) != expected:
            raise ValueError("descriptor requests a policy not implemented by current CIM")

    # Derive the fixed spatial input-lane extent
    @property
    def k(self) -> int:
        return self.ch_in * self.tile_input_axis_elements * self.input_axis_tiles

    # Derive the fixed spatial output-lane extent after B slicing
    @property
    def n(self) -> int:
        return self.ch_out // (self.weight_bits // self.base_b_width) * self.tile_output_axis_elements * self.output_axis_tiles

    # Count one full-set load's accepted B-port transfers
    @property
    def b_writes_per_set(self) -> int:
        return self.k * (self.output_axis_tiles // self.b_port_tiles)

    # Return the byte payload of one accepted B-port transfer
    @property
    def b_beat_bytes(self) -> int:
        return self.n * self.weight_bits * self.b_port_tiles // (8 * self.output_axis_tiles)

    # Match CIMElement::issue_window for the configured macro mode
    @property
    def issue_interval(self) -> int:
        if self.mode == 0:
            return ceil_div(self.input_bits, self.base_a_width)
        slice_width = min(self.input_bits, self.base_c_width - self.base_b_width - max(1, (self.ch_in - 1).bit_length()))
        return ceil_div(self.input_bits, slice_width) * ceil_div(slice_width, self.base_a_width) * self.base_a_width

    # Separate macro result latency from the physical operand-use window
    @property
    def macro_result_latency(self) -> int:
        return self.issue_interval + self.mac_latency - 1
