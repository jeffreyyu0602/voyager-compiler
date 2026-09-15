# Voyager Compiler

The Voyager Compiler is a hardware–software co-designed machine learning (ML) compiler that efficiently maps PyTorch models onto Voyager-generated deep neural network (DNN) accelerators.

The compiler ingests PyTorch models, extracts a static compute graph using [PyTorch 2 Export (PT2E)](https://docs.pytorch.org/docs/stable/export.html), and applies [PT2E-based quantization](https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quant_qat.html). It incorporates a custom quantization framework that provides fine-grained control over data types—including low-bitwidth integers, floating point, posit, and NormalFloat—as well as advanced techniques such as mixed-precision and codebook quantization.

After quantization, the compiler lowers models through hardware-aware operator fusion, architecture-specific optimizations, scheduling, and accelerator instruction generation. Voyager produces a hardware-oriented intermediate representation (IR) built on the PyTorch FX graph, which is serialized via [Protocol Buffers](https://github.com/protocolbuffers/protobuf) and consumed by a C-based backend to generate the final accelerator instruction bitstream.

## Getting Started

End-to-end examples of using the Voyager Compiler can be found in test/test_codegen.py, which demonstrates model ingestion, quantization, compilation, and instruction generation.

## CIM timing

CIM mapping estimates include resident-set readiness, first-pass loading, replay,
and final-use release. A repeated timing-state cache skips whole sequences;
evaluation visits at most 96 sequences with at most 64 resident states, regardless
of layer size. Larger state spaces or unresolved transients use a reported
serialized upper bound. Weight service is derived from geometry and port width;
non-transposed streams overlap transfers while blocking transposes retain their
load/emit phases.

Input fills and banked output draining use the same bounded two-bank recurrence.
Uniform input fills use their actual service time; variable boundary fills use a
reported maximum-fill bound. Independent readiness envelopes combine by maximum,
not by adding stalls; detailed cross-interface phasing remains approximate.

The evaluator reports minimum SRAM feedback spacing from loop order. Supply
`TimingOptions(accumulation_feedback_cycles=...)` with the applicable HLS latency
to conservatively exclude schedules whose unstalled spacing is too short. The
default zero leaves feedback safety unvalidated; it does not invent hardware
stalls or certify read-after-write correctness. Capacity checks and traffic counts are separate from timing. Energy is
unavailable without hardware characterization.

`mapping-report.md` explains readiness waits and active bounds. Cached search
results track timing-source changes automatically. Focused checks:

```bash
python -m unittest discover -s test -p 'test_cim*.py'
python -m unittest discover -s test -p test_mapping_reporting.py
```

## Verified Models

Validated on the following ML models:
- CNNs (ResNet, MobileNet, EfficientNet)
- Transformers (BERT, ViT, MobileBERT)
- LLMs (GPT-2, LLaMA)
- Detection and sequence models (YOLO, Mamba)

## Citation

If you use Voyager Compiler in your research, please cite:

```bibtex
@misc{prabhu2025voyagerendtoendframeworkdesignspace,
  title={Voyager: An End-to-End Framework for Design-Space Exploration and Generation of DNN Accelerators},
  author={Kartik Prabhu and Jeffrey Yu and Xinyuan Allen Pan and Zhouhua Xie and Abigail Aleshire and Zihan Chen and Ammar Ali Ratnani and Priyanka Raina},
  year={2025},
  eprint={2509.15205},
  archivePrefix={arXiv},
  primaryClass={cs.AR}
}
```
