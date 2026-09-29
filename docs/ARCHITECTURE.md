# Tensor Roll — architecture

Copyright (C) 2026 SnapKitty Collective — AGPLv3.

## What this is

TENSOR ROLL is a recursive model quantizer/compressor. A large teacher
model's tensors are recursively partitioned by the operator

```
TensorRoll(T, axis, depth, rank, quantum_policy)
```

every leaf's candidate representations are *measured* (cost in parameters,
retained energy, reconstruction error), a global knapsack search picks the
winning representation per leaf under a hard parameter budget, the student
architecture is searched to fit the same budget, and the student is trained
by recursive distillation (7 loss terms), fine-tuned, quantized, and
exported to FP16 / BF16 / INT8 / INT4 / `.trq`.

Nothing here wraps PyTorch, Transformers, llama.cpp, ONNX, or an existing
quantizer. The tensor core, transformer, distillation, quantizers, `.trq`
codec, sandbox, and CLI are written from scratch on NumPy.

## Pipeline stages (0–7) and CLI commands

| Stage | Name | CLI |
|---|---|---|
| ingest | build/load teacher | `tensor-roll ingest [--scale tiny\|30b]` |
| map | tensor inventory + metrics | `tensor-roll map` |
| roll | `TensorRoll` operator + global search | `tensor-roll roll` |
| qkernel | kernel family on a real block | `tensor-roll qkernel` |
| compress | student architecture search | `tensor-roll compress` |
| 0 | architecture-init (checkpoint) | `tensor-roll train --stage 0` |
| 1 | tensor-reconstruction | `tensor-roll train --stage 1` |
| 2 | hidden-state-matching | `tensor-roll train --stage 2` |
| 3 | logit-distillation | `tensor-roll train --stage 3` |
| 4 | language-model | `tensor-roll train --stage 4` |
| 5 | instruction-tuning | `tensor-roll train --stage 5` |
| 6 | quantization-aware | `tensor-roll train --stage 6` |
| 7 | regression-eval | `tensor-roll train --stage 7` |
| — | task fine-tuning | `tensor-roll finetune` |
| — | FP16/BF16/INT8/INT4 + `.trq` | `tensor-roll quantize` |
| — | held-out eval + recursive metrics | `tensor-roll evaluate` |
| — | benchmark table | `tensor-roll benchmark` |
| — | REPL generation | `tensor-roll chat` |
| — | sandbox supervisor | `tensor-roll sandbox` |
| — | full end-to-end run | `tensor-roll demo` |

Checkpoints are deterministic and resumable: `checkpoints/stage{N}.npz`
plus `checkpoints/meta.json`; re-running a stage overwrites only that
stage's checkpoint.

## The recursive operator

`TensorRoll(T, axis, depth, rank, quantum_policy)` (`tensor_roll/core.py`):

1. `roll_tree` partitions `T` along `axis` to `depth`, halving each level.
   Every node records: shape, parameter count, Frobenius norm, variance,
   SVD rank, Shannon entropy of the normalized singular spectrum, and
   spectral contribution (top singular value² / total energy).
2. Each leaf evaluates candidate representations with **measured**
   triples `(cost_params, retained_energy, reconstruction_error)`:
   - `PRESERVE` — bit-exact, full cost.
   - `QUANTIZE` — real INT8 round-trip, plus codebook VQ (from-scratch
     k-means, k=16).
   - `FACTORIZE` — truncated SVD at measured fractions and at the
     requested `rank`.
   - `MERGED` — 1-D tensors folded into the parent op (exact).
   - `ROUTED` — block routed to the qsim measurement path (exploration).
   - `RECONSTRUCTED` — recursive-measurement classical reconstruction.
   - `PRUNED` — magnitude pruning at 25%/50% with a sparse cost model.
   - `QUANTUM-ENCODED` — angle/amplitude encode → rotate → entangle →
     measure → reconstruct through the real statevector simulator
     (`tensor_roll/qsim.py`), gated by `quantum_policy` and an 8-qubit
     eligibility cap. Labeled `qsim-classical`, never a QPU.
3. `select_plan` runs a greedy knapsack: start every leaf at its cheapest
   option, then repeatedly apply the highest energy-per-cost upgrade that
   keeps total cost under the hard budget `P ≤ 8.5B` (scaled by
   `BUDGET_RATIO = 8.5/30` at tiny scale).

No decision is hard-coded; the policy that emerges is a function of the
measured numbers.

## Distillation (7 loss terms, all real)

`tensor_roll/distill.py::distill_step` computes task cross-entropy,
logit KL (temperature-scaled), hidden-state MSE, attention-map MSE,
embedding MSE, reconstruction MSE, and the roll loss
(`1 - mean leaf retained energy`), each with a measured value and an
analytic gradient through the from-scratch backward pass. Recursive
metrics (`recursive_metrics`) report per model / layer-group / tensor /
sub-block: relative error, cosine similarity, KL divergence, retained
energy.

## `.trq` format

Magic `TRQ1`, versioned JSON header (architecture, tensor index, shapes,
dtypes, quantization metadata incl. codebooks/scales/zero-points, roll
metadata, provenance), per-tensor records with SHA-256, whole-file
SHA-256. The reader refuses corrupted files (tested). Packed INT4 counts
parameters from recorded shapes, not bytes.

## Measured results (tiny validation, this machine, 2026-09-29)

Machine: 2 CPUs, ~7 GB RAM, no GPU (`nvidia-smi` absent), Python 3.12,
NumPy 1.26. `tensor-roll demo --fast` completes in ~95 s.

| Artifact | Params | Size | ppl | acc |
|---|---|---|---|---|
| tiny teacher (fp32) | 41.1K | 160.62KB | 1.87 | 0.851 |
| student (fp32/trq) | 11.4K | 57.07KB | 8.32 | 0.345 |
| student (int8/trq) | 11.4K | 24.88KB | 8.39 | 0.342 |
| student (int4/trq) | 11.4K | 36.15KB | 15.13 | 0.221 |

Student = 27.8% of teacher params (target ratio 8.5/30 = 28.3%).
INT4 degrades gracefully (ppl 8.3 → 15.1), as honestly expected.
qsim recursive reconstruction of an 8×8 block: rel err 0.168 at 512 shots.
Teacher/student token agreement on 24 generated tokens: 3/24 — the
student is a weak model at this scale; the pipeline, not the perplexity,
is what is being validated.

## What was NOT run here (and what it needs)

- Glimmer 30B ingestion: needs the 30B checkpoint shards + a machine with
  ~60 GB+ RAM (fp16) or multi-GPU. `ingest --scale 30b` checks for
  `nvidia-smi`/CUDA and exits `unavailable` instead of faking it.
- CUDA-Q kernels: need `pip install cudaq` + a target (`qpp-cpu` sim or
  QPU/GPU). Absent here; `cudaq/tensor_roll_kernels.py` prints
  `unavailable`.
- Q# operations: need the Q# toolchain. `qsharp/TensorRoll.qs` is genuine
  source, uncompiled here.
- CUDA GEMM: needs a CUDA GPU; `dispatch_matmul` reports `cpu-numpy`.

The 30B path is the same code path: ingest shards → same `TensorRoll`
operator → same search → same stages. Only the tensor source and the
hardware change.
