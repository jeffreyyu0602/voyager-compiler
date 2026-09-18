# Mapping and hardware models

The mapping workflow converts compiler operations and search candidates into
hardware-independent data, evaluates them, and writes the selected tilings.

## Data flow

1. `operations.parse_operation` converts a compiler operation to `Workload` and
   `VectorTiming`. `target.load_target` reads the resolved hardware settings.
2. `search.prepare_search` builds Interstellar constraints and one hardware model.
3. The search callback converts each candidate to `Schedule` and calls the model.
4. The model returns `Evaluation`: legality, traffic, timing, and diagnostics.
5. `results.serialize` uses the retained schedule and result. `write_tilings`
   writes completed results; it does not run a search.

## Modules

| Location | Responsibility |
|---|---|
| `driver.py` | Command line, operation traversal, search invocation |
| `operations.py` | Compiler-format conversion and supported operation checks |
| `target.py`, `workload.py`, `schedule.py` | Hardware settings, operation dimensions, loop schedules |
| `search.py` | Interstellar setup, candidate conversion, callbacks, statistics |
| `results.py`, `reporting.py` | Selected-tiling serialization and report output |
| `models/sa.py`, `models/cim.py` | Backend legality, work counts, complete evaluation |
| `models/cim_weight_policy.py` | Resident-weight loading, reuse, and descriptors |
| `models/cim_timing.py` | CIM runtime, reductions, replay, and burst traversal |
| `models/input.py`, `vector.py`, `output.py`, `bias.py` | Shared hardware-path calculations |
| `timing/transfer.py` | Port transfers and channel packing |
| `timing/buffers.py` | Repeated storage fill, use, and release |
| `timing/backpressure.py` | Producer stalls and output backlog |
| `timing/overlap.py` | Interaction of operand readiness and output backpressure |

Hardware models and timing calculations have no compiler protobuf or Interstellar
imports. The SA memory-access calculation remains in the search adapter because
Interstellar owns that calculation. Each selected candidate retains its evaluation.

## Units

A result vector contains `target.n` scalar results. Its width can differ from the
vector unit's lane count. `cycles_per_vector` is the steady processing interval;
it is not the latency through all functional-unit stages.

Use `_cycles` for durations, `_at` for absolute timestamps, and explicit `_bits`,
`_bytes`, `_elements`, or `_vectors` for quantities. `resource_cycles` contains
total work per resource. These totals overlap and must not be summed as runtime.
`resource_cycles_per_vector` contains vector-path work for one result vector.
A buffer sequence and a resident-weight sequence are separate from result vectors.

Timing calculations use bounded repetition. The CIM model reports a conservative
fallback when interacting waits exceed the explicit timing budget. Model input
values describe hardware rates and delays; detailed RTL handshakes are outside
this analytical model.

Output timing profiles use `elements_per_vector` and a list of stages with
`elements`, `forward_cycles`, and `feedback_cycles`. Report fields use
`output_capacity_vectors` and `output_cycles_per_vector`. Hardware target JSON
and the tiling protobuf retain their existing fields.
