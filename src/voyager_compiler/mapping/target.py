# Load resolved hardware targets through one shared configuration contract
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


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
    raise ValueError("mapping target must identify backend=sa or backend=cim")
