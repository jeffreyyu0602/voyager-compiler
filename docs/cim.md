# CIM matrix backend

## Configuration

`matrix_backend=1` (`--matrix_backend 1`) selects CIM; `0`, the default,
selects the systolic array. Compiler settings do not change the hardware
build settings, so supply the same values to both.

The `cim_*` fields use the hardware parameter names in lowercase.
`cim_base_a_width`, `cim_base_b_width` and `cim_base_c_width` are the macro
input, weight and accumulation widths; wider operands use slices.
`pe_array_size` gives the total input and output lanes. Each weight slice
uses one macro output lane. For the default macro and 8-bit weights, use
`AcceleratorConfig(matrix_backend=1, pe_array_size=(64, 16))`.
`cim_a_port_tiles` and `cim_array_result_slots` default to
`cim_input_axis_tiles`; `cim_b_port_tiles` and `cim_c_port_tiles` default to
`cim_output_axis_tiles`.

## Supported operations

- Signed or unsigned integer GEMMs and dense convolutions with dilation 1.
  `cim_signed` must agree with both operand datatypes.
- Equal strides in both dimensions. The shared input controller does not
  support strided 1×N or N×1 filters.
- GEMV uses the vector tiler.
- Floating-point and microscaling formats, codebook, depthwise and sparse
  matrix operators, and `accumulate_fp32` are rejected.
- CIM needs the complete schedules of the tiling search. Explicit
  `l2_tiling` counts and the C++ fallback (`MANUAL_TILING=1`) are rejected.

## Mapping search

FX stays at L1. FY runs at L1, or at L2 when its L1 factor is 1. The search
checks the input buffer with convolution halos, the accumulation capacity and
the controller limits. It selects the lowest modeled access energy within
`runtime_tolerance` (default 2%) of the fastest estimate;
`--runtime_tolerance 0` keeps only the fastest. CIM energy costs are not
calibrated. The selected weight policy and operation counts are in
`anchor.meta["tiling"]["cim_evaluation"]`. The C++ instruction mapper keeps
the selected L1/L2 loop order.

## Runtime model

The estimate ranks mappings; it does not give exact RTL cycles.

- Weight sequences that fit stay resident; larger sequences reload.
- Local accumulation registers hold the largest inner reduction whose live
  outputs fit. Other partial sums use the accumulation SRAM. Both take one
  accumulation cycle; the SRAM accesses add energy, not time.
- Input and weight prefetch overlap computation. Interacting waits are added,
  which can overestimate the runtime.
- A read stream that changes scratchpad bank pays the SoC round trip, as in
  the systolic estimate. Independent interfaces disable shared-bank timing
  for both backends; see [Memory interface timing](memory.md).
- The estimate includes the work on padding. It does not include command
  setup or unprofiled HLS pipeline stages.

### Output-burst ranking limitation

The model can select a slower mapping because it cannot predict the extra
final-output storage introduced by HLS scheduling. This was observed on the
N7/1 ns build with one accumulation SRAM, independent external interfaces,
untimed DMA, and a 64-lane BF16 vector output on a 512-bit port. Each final
vector takes two port cycles, so a finished reduction's burst can stall the
CIM issue stream unless the output path has enough storage.

The current estimate counts the CIM final-output FIFO (8 vectors). Adding
one intended register for each of the 8 output processes gives 16 vectors,
but still does not reproduce the observed mapping rankings. In the measured
build, HLS scheduled the router and four arithmetic stages with 6, 6, 7, 6
and 9 registers. The additional storage absorbs bursts that the intended
model expects to stall, changing the relative cost of spatial blocking.

The following are full-operation RTL cycles for GEMM M1024/N128 with a fused
BF16 output, on the same build and harness. All listed runs passed gold and
operation-counter checks. These are re-simulated mappings, not historical
workbook cycle measurements or compiler estimates.

| K | Workbook Current mapping | Compiler-selected mapping | Best tested mapping |
|---|---|---|---|
| 128 | OX32: 4,412 | OX16: 4,444 | OX32: 4,412 |
| 512 | OX32: 17,120 | OX16: 17,344 | OX64: 16,873 |
| 576 | OX64: 19,595 | OX128: 19,812 | OX64: 19,595 |

Diagnostic experiments with a manually supplied equivalent capacity of
about 35 vectors and 9 clocks of transit improved these selections. That
capacity describes this synthesized build; it is not an intended hardware
parameter or a generic FIFO depth. The compiler does not derive it
automatically. A different clock, technology or output configuration would
require new characterization and manual input, making this an unsustainable
workaround. It is not adopted as a compiler fix.

The three ranking errors remain unresolved under the intended-values-only
model. The tested candidates establish this limitation, not global mapping
optimality. A sustainable correction would require a declared output-path
storage/latency contract that the implementation meets; that change has not
been implemented.

## Outputs

`compile()` writes `memory_config.txt` beside `model.txt`: a `MemoryConfig`
protobuf with the scratchpad bank geometry and the compiler frequency, for
the standalone SystemC and RTL harnesses. Matrix reports add the useful-work
fraction and the effective utilization, which exclude padded channels and
convolution borders.
