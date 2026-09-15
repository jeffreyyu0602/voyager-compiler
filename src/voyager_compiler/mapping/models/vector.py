# Describe vector passes and physical transfer demand independently of the matrix backend
from dataclasses import dataclass
import re
from ..timing.transfer import ceil_div


# Resolve scalar precision rather than treating every vector as the same number of bits
def dtype_bits(dtype):
    match = re.search(r'[^\d](\d+)(?:_.*)?$', str(dtype))
    if match is None or int(match[1]) <= 0:
        raise ValueError(f'unsupported vector datatype: {dtype}')
    return int(match[1])


# Specify vector geometry and initiation intervals rather than operation latencies
@dataclass(frozen=True)
class VectorHardware:
    lanes: int
    port_bits: int
    stage_intervals: tuple = (1, 1, 1, 1)

    # Freeze and validate the four physical pipeline stages
    def __post_init__(self):
        object.__setattr__(self, 'stage_intervals', tuple(self.stage_intervals))
        if len(self.stage_intervals) != 4 or any(type(n) is not int or n <= 0 for n in self.stage_intervals):
            raise ValueError('four stage intervals must be positive integers')

    # Read vector lanes and interface width from the exported target
    @classmethod
    def from_target(cls, target):
        return cls(target.vector_config['lanes'], target.oc_port_bits)


# Count one resource's physical traffic per result vector
@dataclass(frozen=True)
class Transfer:
    resource: str
    elements: int
    element_bits: int

    # Retain conversion widths at each interface
    @property
    def bits(self):
        return self.elements * self.element_bits


# Describe a scheduled pass rather than counting mathematical epilogue operations
@dataclass(frozen=True)
class VectorPass:
    source: str
    stages: tuple = ('', '', '', '')
    transfers: tuple = ()
    dequantize: bool = False

    # Preserve immutable pass descriptors for repeated candidate evaluation
    def __post_init__(self):
        object.__setattr__(self, 'stages', tuple(self.stages))
        object.__setattr__(self, 'transfers', tuple(self.transfers))
        if self.source not in ('matrix', 'intermediate', 'reducer', 'accumulator') or len(self.stages) != 4:
            raise ValueError('vector passes require a supported source and four stage assignments')


# Retain per-pass consumption phases and the shared-resource throughput bound
@dataclass(frozen=True)
class VectorTiming:
    elements_per_vector: int
    input_element_bits: int
    output_element_bits: int
    direct: bool
    passes: tuple
    pass_cycles_per_vector: tuple
    resource_cycles_per_vector: tuple

    # Bound throughput without summing independently pipelined resources
    @property
    def cycles_per_vector(self):
        return max((cycles for _, cycles in self.resource_cycles_per_vector), default=1)

    # Match the hardware's bandwidth-based accumulation banking decision
    @property
    def port_cycles_per_vector(self):
        return max((cycles for name, cycles in self.resource_cycles_per_vector if name != 'vector'), default=1)

    # Emit readable phase demand without creating version or provenance fields
    def report(self):
        return dict(elements_per_vector=self.elements_per_vector, input_element_bits=self.input_element_bits,
                    output_element_bits=self.output_element_bits, direct=self.direct,
                    cycles_per_vector=self.cycles_per_vector,
                    resources=dict(self.resource_cycles_per_vector),
                    passes=[dict(source=p.source, stages=list(p.stages), dequantize=p.dequantize,
                                 cycles_per_vector=cycles,
                                 transfers=[dict(resource=t.resource, elements=t.elements,
                                                 element_bits=t.element_bits, bits=t.bits) for t in p.transfers])
                            for p, cycles in zip(self.passes, self.pass_cycles_per_vector)])


# Sum reuse on shared resources while preserving each pass's matrix-consumption phase
def evaluate_passes(hardware, elements_per_vector, input_bits, output_bits, passes, *, direct=False):
    passes = tuple(passes)
    if not passes:
        raise ValueError('vector timing requires at least one pass')
    resources, intervals = {}, []
    for stage_pass in passes:
        demand = {} if direct else dict(vector=ceil_div(elements_per_vector, hardware.lanes)
                                       * max(hardware.stage_intervals))
        for transfer in stage_pass.transfers:
            demand[transfer.resource] = demand.get(transfer.resource, 0) + ceil_div(transfer.bits, hardware.port_bits)
        intervals.append(max(demand.values(), default=1))
        for resource, cycles in demand.items():
            resources[resource] = resources.get(resource, 0) + cycles
    return VectorTiming(elements_per_vector, input_bits, output_bits, direct, passes,
                         tuple(intervals), tuple(sorted(resources.items())))
