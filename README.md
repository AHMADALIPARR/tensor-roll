# TENSOR ROLL — Recursive CUDA-Q Model Quantizer

Copyright (C) 2026 SnapKitty Collective — AGPLv3 (see `LICENSE`).

```
GLIMMER 30B -> TENSOR ROLL -> CUDA -> CUDA-Q -> Q# -> 8B NANO MODEL
```

TENSOR ROLL recursively partitions a large teacher model's tensors with
the operator `TensorRoll(T, axis, depth, rank, quantum_policy)`, measures
candidate representations per partition (cost, retained energy,
reconstruction error), searches a global representation plan under a hard
parameter budget (**P ≤ 8.5B**), distills a student with 7 loss terms, and
exports FP16 / BF16 / INT8 / INT4 / `.trq` artifacts.

No wrappers: the tensor core, transformer, distillation, quantizers,
`.trq` codec, OS sandbox, and CLI are written from scratch on NumPy.
No fake kernels, no fabricated benchmarks, no quantum-execution claims —
every report carries an honest backend label (`cpu-numpy`,
`qsim-classical`, or `unavailable`).

## Install

```bash
pip install --break-system-packages -e .
tensor-roll --help          # 14 commands
```

Requires Python 3.10+ and NumPy. No GPU needed for the validation path.

## 60-second validation

```bash
tensor-roll --workdir /tmp/trdemo demo --fast
```

Runs the full pipeline on a tiny from-scratch teacher (~95 s on 2 CPUs):
ingest → roll → qkernel → compress → train (stages 0–7) → finetune →
quantize → evaluate → benchmark → sandbox → streaming inference.
Every number printed was measured on your machine.

Individual commands:

```bash
tensor-roll --workdir DIR ingest --scale tiny
tensor-roll --workdir DIR map
tensor-roll --workdir DIR roll --depth 2 --quantum-policy explore
tensor-roll --workdir DIR qkernel --shots 1024
tensor-roll --workdir DIR compress
tensor-roll --workdir DIR train --steps 40
tensor-roll --workdir DIR finetune --steps 40
tensor-roll --workdir DIR quantize
tensor-roll --workdir DIR evaluate
tensor-roll --workdir DIR benchmark
tensor-roll --workdir DIR chat
tensor-roll --workdir DIR sandbox --demo
```

## Layout

```
tensor_roll/        Python package (core, qsim, model, search, distill,
                    quant, trq, sandbox, boundary, cli)
cudaq/              real CUDA-Q kernels (need the CUDA-Q toolchain)
qsharp/             real Q# operations (need the Q# toolchain)
cuda/               (reserved: CUDA GEMM sources when GPU HW exists)
docs/               ARCHITECTURE.md, BOUNDARIES.md, tensor-roll-demo.png
```

## The honesty contract

- `unavailable` means not executed here. It is never replaced by an
  invented number.
- `qsim-classical` is a classical statevector simulator. It is never
  called a QPU.
- The CUDA-Q ↔ Q# boundary is a checksummed serialized trace
  (`qkernel/boundary.json`), not an invented in-memory API.
- Failed experiments are reported as failures. See `docs/ARCHITECTURE.md`
  for what ran, what didn't, and the hardware each needs.

## License

AGPL-3.0-or-later. Every source file carries the SPDX header.
