# Memory interface timing

The standalone SystemC and RTL harnesses support banked arbitration and
independent external interfaces for both the systolic array and CIM.

With scratchpad size, bank count and bank width supplied, the compiler
defaults to shared-bank timing and writes `mode: BANKED` in
`memory_config.txt`. Without bank geometry, it writes `mode: INDEPENDENT`.

Set `AcceleratorConfig(independent_memory_ports=True, ...)`, or pass
`--independent_memory_ports`, to estimate independent interfaces while
retaining the configured bank geometry for tensor allocation. The shared
runtime model then omits contention between interfaces and SoC bank-switch
delays. Per-interface transfer bandwidth and matrix/controller costs remain.
`compile()` writes `mode: INDEPENDENT` with the same allocation geometry, so
the harness selects the matching mode without an environment override.

This setting does not change DRAM bandwidth or latency. The harness's host
DMA is untimed; matching compiler searches must separately use infinite DRAM
bandwidth and zero DRAM latency. Compiler estimates are not exact RTL cycles.

`SCRATCHPAD_MODEL=banked` or `SCRATCHPAD_MODEL=independent` can override the
harness mode for diagnostic replays. Such an override must be reflected in
the compiler configuration when comparing mapping estimates with those runs.
