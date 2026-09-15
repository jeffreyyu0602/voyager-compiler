# Mapping service and buffering

`mapping/vector.py` describes vector passes independently of the matrix backend. A logical group contains the target's output-lane count in elements. Each transfer records its own element count, scalar width, and port; INT24 input and BF16 output are different physical payloads for the same logical elements.

`plan_epilogue` follows the four-stage assignment in `test/toolchain/MatrixOps.h` and `Common.h`. Dequantization does not consume a stage. Supported pointwise operations share one pipeline traversal. An operation that does not fit must be lowered into another command; the model does not invent a hardware pass. Microscaled output is unsupported. `evaluate_program` accepts explicit multiple-pass descriptors, preserves each pass's source and demand, and sums work on reused resources. Its maximum resource demand is a throughput bound, not the elapsed time of dependent passes. The dense mapping adapters consume the single-pass programs emitted by the current matrix toolchain.

`mapping/stream.py` models finite packet credits with independent producer and consumer clocks, forward latency, and credit-return delay. A packet's credit is reusable on the edge after its return. Repeated normalized timing states are skipped, both within bursts and across nested loop repetitions. Work is bounded by 4,096 explicitly processed packets; exceeding that budget reports an unsupported timing sequence rather than inserting a pessimistic penalty. Equivalent traffic patterns share a bounded cache.

`mapping/output.py` projects final-reduction traffic from inner-to-outer temporal loops. Partial-reduction work advances the producer clock without emitting a packet. Backlog and delayed credits survive burst boundaries. CIM and SA use the same engine. SA also includes its existing weight-reuse startup gaps. Banked output uses the backend's buffer-bank readiness model.

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

`mapping/transfer.py` expresses payloads in bits, counts port beats, and applies the controller's all-or-one packing rule. A streaming reader/writer has separate first-fill and steady-service costs; a blocking transpose adds its read and write phases.

`mapping/input.py` describes the dense input controller independently of the matrix backend. It groups equivalent boundary tiles and counts valid external requests, packed transfers, halo storage, and writes of zero padding. SA uses the input tensor's scalar precision; CIM supplies its native INT8 precision. Packing reduces external requests but does not remove buffer-word writes. Fill service overlaps memory transfers with unpacking and writing, including the first transfer's pipeline tail.

`mapping/buffering.py` tracks loading, consumption, and final-use release of reusable slots. It skips repeated normalized states and limits explicit work to 96 groups and 64 slots. SA and CIM input banks use two slots. CIM also uses it for resident weight sets and banked output. Uniform fills use their actual service; variable boundary fills use a reported maximum-fill bound.

SA caches input summaries by temporal factors. Its input model requires materialized input transposes/head permutations and batch-one, unit-dilation convolutions. Halo capacity and input packing limits reject unsupported candidates. The report separates external beats from buffer-word writes and shows input readiness waits and active bounds.

SA weight timing uses buffer-transfer demand and array loading/reuse. CIM weight timing uses streaming row assembly and resident-set lifetimes. These backend rules supply service and release times to shared primitives; neither backend borrows the other's storage policy.

## Hardware configuration

`vector_config.lanes` and `vector_config.output_fifo_packets` come from the target exporter. Targets without these fields use full-width vector lanes and the hardware's eight-entry final-result FIFO.

Optional `TimingOptions.output_stages` describe pipeline holding capacity and forward/feedback timing in logical-group units. `output_profile_elements` identifies that group's element count. Profiles with incompatible lane widths are rejected. FIFO capacity and stage holding capacities form a pooled credit envelope; this is an analytical pipeline model, not a replay of every internal ready/valid connection.

Without a profile, only the physical FIFO receives storage credit. This can overestimate stalls when HLS inserts additional elastic stages. `examples/timing/vector-64-bf16.json` describes the characterized 64-lane INT24-to-BF16 route from the 2 ns HLS build. It is an explicit configuration, not a universal pipeline default. Check the profile when changing the active vector route or synthesis schedule.

From the accelerator checkout:

```bash
make network-proto "${MAPPING_ARGS[@]}" \
  MAPPING_TIMING_OPTIONS=voyager-compiler/examples/timing/vector-64-bf16.json \
  MAPPING_FORCE_SEARCH=1 MAPPING_VERBOSE=2
```

The shared vector module can also be used directly with `VectorHardware`, `Transfer`, `VectorPass`, and `evaluate_program`, without a CIM target or Interstellar search.

## Reports and scope

Per-layer reports show logical group size, input/output bit widths, scheduled vector passes, per-resource service demand, producer stall cycles, and whether pipeline elasticity was supplied. The explicit packet/repetition counts expose the amount of work performed by the timing model. Consumer completion is reported separately from matrix-side producer completion.

Matrix runtime includes finite-output-credit backpressure. Input and weight readiness remain independent timing envelopes combined by a maximum; their detailed interaction with output stalls is approximate. Final vector drain is not a complete end-to-end fused-operator latency estimate. Energy is unavailable without hardware characterization.

Small checks cover shared stage scheduling, independent and shared ports, explicit multiple-pass demand, compressed timing against uncompressed reference events, the historical 32-versus-128-result burst comparison, and complete 8-lane CIM/SA searches.
