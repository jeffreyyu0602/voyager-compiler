# Mapping service and buffering

`mapping/models/vector.py` describes vector passes independently of the matrix backend. A logical group contains the target's output-lane count in elements. Each transfer records its own element count, scalar width, and port; INT24 input and BF16 output are different physical payloads for the same logical elements.

The compiler's `lower_epilogue()` selects vector stages and operand-fetch ports for fused matrix epilogues. Command generation uses those selections to produce instructions; `ExportEpilogue` exports them as execution descriptions for mapping. Dequantization does not consume a stage. Supported pointwise operations share one pipeline traversal. An operation that does not fit must be lowered into another command; the model does not invent a hardware pass. Microscaled output is unsupported. `evaluate_passes` accepts explicit multiple-pass descriptors, preserves each pass's source and demand, and sums work on reused resources. Its maximum resource demand is a throughput bound, not the elapsed time of dependent passes. The dense mapping adapters consume the single-pass epilogue descriptions emitted by the matrix toolchain.

`mapping/timing/backpressure.py` models bandwidth mismatch with one backlog. Work uses logical result groups; `s` is consumer cycles per group, `B` is burst size, `T` is unstalled production time, and `E` is final-output storage capacity in groups. The backlog `q` is measured in remaining consumer cycles. An applicable timing profile supplies forward and credit-return delay `L`; without a profile, `L` is zero:

```text
headroom = max(0, E*s - L)
stall = max(0, q + B*s - T - headroom)
q_after = max(0, q + B*s - T - stall)
q_next = max(0, q_after - gap)
```

A summary carries two clock equations: producer elapsed time and consumer busy-until time. Composition substitutes those equations; repeated squaring handles a loop bound in logarithmic work. The summary has constant size and retains backlog across burst boundaries. Equivalent loop summaries share a bounded in-process cache.

`mapping/models/output.py` projects final-reduction traffic from inner-to-outer temporal loops. Partial reductions become gaps; output loops repeat the same summary. SA includes its weight-reuse startup gaps. Both backends use the shared burst equation for streaming results. Banked output uses the backend's buffer-bank readiness model.

## Search and evaluation boundary

| Owner | Responsibility |
|---|---|
| Interstellar | Generic factor and loop-order enumeration, capacity hooks, candidate validation, ranking |
| Voyager SA/CIM evaluators | Hardware legality, timing, traffic, and utilization for each candidate |
| Shared Voyager modules | Vector service, finite buffering, operation traversal, target configuration, cache, and reports |

Both backends provide a complete `CandidateEvaluation`. Search uses its runtime and retains the winning report without rescoring. Ranking is lowest cycles; exact ties retain the first enumerated candidate. SA uses generic capacity checks and logical access-count helpers; CIM supplies its hardware capacity filter and detailed traffic counters.

Voyager does not assign energy costs or use energy to break ties. Reports show `n/a`, never zero, for uncharacterized energy. A future characterized energy evaluator belongs with Voyager hardware models and must match precision, SRAM geometry, banking/ports, and CIM configuration. Interstellar accepts a supplied cost and unit without interpreting the hardware. Its standalone generic weighted-cost evaluator remains available for resources with explicit energy costs.

SA traffic counts are scalar element accesses, ordered input/output/weight by memory level. They are not physical SRAM transactions or bus beats. CIM reports named physical counters with their respective units; its native INT8 byte counts are not interchangeable with wider-element access counts.

Timing JSON is decoded by the selected backend: SA accepts shared `OutputOptions`; CIM accepts `TimingOptions`, which includes those shared fields. Unsupported fields are rejected. No energy-cost file is accepted by the Voyager driver.

## Shared transfer and buffer timing

`mapping/timing/transfer.py` expresses payloads in bits, counts port beats, and applies the controller's all-or-one packing rule. A streaming reader/writer has separate first-fill and steady-service costs; a blocking transpose adds its read and write phases.

`mapping/models/input.py` describes the dense input controller independently of the matrix backend. It groups equivalent boundary tiles and counts valid external requests, packed transfers, halo storage, and writes of zero padding. SA uses the input tensor's scalar precision; CIM supplies its native INT8 precision. Packing reduces external requests but does not remove buffer-word writes. Sustained fill service overlaps memory transfers with unpacking and writing. The first fill includes pipeline latency once; later fills keep streaming across bank boundaries.

`mapping/timing/buffers.py` tracks loading, consumption, and final-use release of reusable slots. It skips repeated normalized states and limits explicit work to 96 groups and 64 slots. SA and CIM input banks use two slots. CIM also uses it for resident weight sets and banked output. Uniform fills use their actual service; variable boundary fills use a reported maximum-fill bound.

SA caches input summaries by temporal factors. Its input model requires materialized input transposes/head permutations and batch-one, unit-dilation convolutions. Halo capacity and input packing limits reject unsupported candidates. The report separates external beats from buffer-word writes and shows input readiness waits and active bounds.

SA weight timing uses buffer-transfer demand and array loading/reuse. CIM weight timing uses streaming row assembly and resident-set lifetimes. These backend rules supply service and release times to shared primitives; neither backend borrows the other's storage policy.

## Hardware configuration

`vector_config.lanes` and `output_storage` are exported hardware facts. Storage values count logical result elements, excluding control metadata and parallel operands attached to the same result. The exported fields describe distinct storage locations:

| Storage | CIM | SA |
|---|---|---|
| Matrix results | Result slots per output lane; parallel lanes jointly form one group | Common minimum of the per-lane result-skewer FIFOs; additional depth aligns lane skew |
| Accumulation metadata | Two entries carrying result payloads | None |
| Accumulation writeback | Declared accumulation-to-writeback FIFO | Same |
| Matrix output | Declared output FIFO | Same |
| Vector pipeline | Conditional stage-3 FIFO when microscaling split mode is compiled | Same |

The default final-output model counts `matrix_output` and, for vector output, `vector_pipeline`. It divides their element counts by the matrix output width and retains whole result groups. Direct matrix output excludes `vector_pipeline`. Matrix result slots, accumulation metadata, accumulation writeback, and local partial-sum contexts do not add final-output credit. Descriptor queues name results that are already counted. The active serializer or writer is covered by service demand. The default model omits HLS-inserted registers and stage delays.

An optional `OutputOptions.output_pipeline` profile describes stage capacity and forward and credit-return delays for a matching single-pass matrix epilogue. It includes the exported `matrix_output` and `vector_pipeline` capacities once each; other stage capacities use explicit element counts. Target geometry and widths must match the profile. Direct output or a different pass or output width uses the default model. A profile reduces usable burst headroom by its delays and adds forward transit to final drain. It is an analytical timing description, not a replay of internal ready/valid signals.

Storage depths are shared constants used by hardware declarations and the host exporter. Regenerate the target when these constants or other exported hardware settings change. The mapper does not infer extra storage or delays from synthesis. Known macro latency comes from the target; external request latency and optional SRAM feedback constraints remain explicit timing inputs.

From the accelerator checkout:

```bash
make network-proto "${MAPPING_ARGS[@]}" MAPPING_VERBOSE=2
```

The shared vector module can also be used directly with `VectorHardware`, `Transfer`, `VectorPass`, and `evaluate_passes`, without a CIM target or Interstellar search.

## Reports and scope

Per-layer reports show logical group size, input/output bit widths, scheduled vector passes, per-resource service demand, explicit storage capacity, producer stalls, and remaining consumer work. Consumer completion is reported separately from matrix-side producer completion.

Matrix runtime includes burst backpressure. SA combines input and output readiness bounds. CIM composes interacting operand waits and output backlog at burst and loop boundaries; an unrecognized long transient uses a reported conservative bound. The default output model omits pipeline fill and handshake edge delays; an applicable profile adds stated forward and credit-return delays. Final vector drain is not a complete end-to-end fused-operator latency estimate. Energy is unavailable without hardware characterization.

Small checks cover vector resource service, explicit multiple-pass demand, the backlog equation, partially drained bursts, large loop bounds, the historical 32-versus-128-group burst comparison, and complete 8-lane CIM/SA searches.
