# Project matrix loops into an analytical bandwidth-and-capacity model shared by SA and CIM
from dataclasses import dataclass, replace
from functools import lru_cache
from ..timing.backpressure import burst, idle


# Describe serial payload storage using elements or a named exported target capacity
@dataclass(frozen=True)
class OutputStage:
    name: str
    elements: int | str
    forward_cycles: int
    feedback_cycles: int = 0

    # Restrict capacity references to final-output storage
    def __post_init__(self):
        if type(self.elements) is int and self.elements >= 0:
            return
        if self.elements not in ('matrix_output', 'vector_pipeline'):
            raise ValueError('stage capacity must be nonnegative elements or exported final-output storage')


# Keep synthesis-dependent buffering separate from the physical mapping target
@dataclass(frozen=True)
class OutputPipeline:
    elements_per_vector: int
    vector_lanes: int
    accum_bits: int
    output_bits: int
    port_bits: int
    stages: tuple

    # Freeze JSON inputs so profiles remain usable in per-search memoization
    def __post_init__(self):
        object.__setattr__(self, 'stages', tuple(OutputStage(**s) if isinstance(s, dict) else s
                                               for s in self.stages))
        if any(type(n) is not int or n <= 0 for n in
               (self.elements_per_vector, self.vector_lanes, self.accum_bits, self.output_bits, self.port_bits)):
            raise ValueError('output profile geometry and widths must be positive integers')
        if not self.stages or any(not isinstance(s, OutputStage) for s in self.stages):
            raise ValueError('output profile requires stage descriptions')
        if len({s.name for s in self.stages}) != len(self.stages):
            raise ValueError('output profile stage names must be distinct')
        references = [s.elements for s in self.stages if isinstance(s.elements, str)]
        if sorted(references) != ['matrix_output', 'vector_pipeline']:
            raise ValueError('output profile must include each exported final-output capacity once')

    # Prevent a timing profile for one synthesized datapath from silently fitting another
    def validate_target(self, target):
        if (self.elements_per_vector, self.vector_lanes, self.accum_bits, self.port_bits) != (
                target.n, target.vector_config['lanes'], target.accum_bits, target.oc_port_bits):
            raise ValueError('output timing profile does not match target geometry and widths')


# Supply optional physical timing without inventing HLS registers in the default model
@dataclass(frozen=True)
class OutputOptions:
    output_pipeline: OutputPipeline | None = None

    # Accept the same JSON profile for the SA and CIM evaluators
    def __post_init__(self):
        if isinstance(self.output_pipeline, dict):
            object.__setattr__(self, 'output_pipeline', OutputPipeline(**self.output_pipeline))
        if self.output_pipeline is not None and not isinstance(self.output_pipeline, OutputPipeline):
            raise ValueError('output_pipeline must be a timing profile')

    # Unprofiled routes retain their bandwidth model instead of rejecting new epilogues
    def for_vector(self, target, vector_timing):
        if self.output_pipeline is None:
            return self
        self.output_pipeline.validate_target(target)
        if (vector_timing.direct or len(vector_timing.passes) != 1 or vector_timing.passes[0].source != 'matrix'
                or vector_timing.output_element_bits != self.output_pipeline.output_bits):
            return replace(self, output_pipeline=None)
        return self


# Preserve reduction gaps and output bursts by composing summaries, never individual tiles
@lru_cache(maxsize=4096)
def completion_timing(loops, cycles_per_vector, capacity_vectors, interval=1, prefix_at=None, prefix_cycles=0,
                      credit_delay_cycles=0):
    summary = burst(1, interval, cycles_per_vector, capacity_vectors, credit_delay_cycles=credit_delay_cycles)
    for index, (loop, bound) in enumerate(loops):
        if index == prefix_at:
            summary = idle(prefix_cycles).then(summary)
        if loop in ('IC', 'FX', 'FY'):
            summary = idle((bound - 1) * summary.work_cycles).then(summary)
        else:
            summary = summary.repeated(bound)
    return summary.timing()


# Normalize final-output storage at the accumulation-output boundary, shared by SA and CIM
# Intermediate-reduction storage is used for accumulation and cannot hold final-output bursts
def output_timing(target, loops, cycles_per_vector, *, direct=False, interval=1,
                  prefix_at=None, prefix_cycles=0, options=None):
    storage = {name: elements for name, elements in target.output_storage.items()
               if name == 'matrix_output' or (name == 'vector_pipeline' and not direct)}
    profile = None if direct else (options or OutputOptions()).output_pipeline
    forward = feedback = 0
    if profile is not None:
        storage = {s.name: target.output_storage[s.elements] if isinstance(s.elements, str) else s.elements
                   for s in profile.stages}
        forward = sum(s.forward_cycles for s in profile.stages)
        feedback = sum(s.feedback_cycles for s in profile.stages)
    capacity = sum(storage.values()) // target.n
    result = completion_timing(loops, cycles_per_vector, capacity, interval, prefix_at, prefix_cycles,
                               forward + feedback)
    # Add forward transit once to the final completion time
    result = replace(result, consumer_cycles=result.consumer_cycles + forward,
                     backlog_cycles=result.backlog_cycles + forward)
    return result, dict(output_stall_cycles=result.stall_cycles,
                        output_consumer_finish_cycles=result.consumer_cycles,
                        output_backlog_cycles=result.backlog_cycles,
                        output_capacity_vectors=capacity,
                        output_elements_per_vector=target.n,
                        output_cycles_per_vector=cycles_per_vector,
                        output_pipeline_profiled=profile is not None,
                        output_forward_cycles=forward, output_credit_delay_cycles=forward + feedback,
                        output_headroom_cycles=max(0, capacity * cycles_per_vector - forward - feedback),
                        output_storage_elements=storage)
