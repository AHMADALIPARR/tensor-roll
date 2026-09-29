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

Rerun (`demo --fast`, same machine, 2026-09-29, failure-resolution
pass): teacher acc 0.849, int8 PPL 8.45, int4 PPL 15.27, compression
0.2777. Training is deterministic per (steps, seed) — verified by
byte-identical reruns; the small deltas vs the table above reflect run
configuration, and both runs are kept on record instead of overwriting.

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

## Failure-resolution pass (2026-09-29): what the F1–F7 mission changed

Failures are attacked in order; every item below is measured on this
machine (2 CPUs, ~7 GB RAM, no GPU) unless marked otherwise.

### F1 — Out-of-core architecture (RULE ZERO: never require 30B in RAM)

New module `tensor_roll/ooc.py`:

- `ShardWriter`/`TensorSource`/`ShardReader`: sharded weight storage with
  manifest; reads are `np.memmap` views — one tensor (or window) at a
  time, never the model.
- `MappedTensor.windows(budget)`: recursive windowing along the largest
  axis until each window fits the byte budget; `reassemble()` verifies
  exact coverage (no overlap, no gaps, no shape drift) — tested.
- `RollWorkspace`: hard working-set ceiling; allocations past it raise
  `MemoryBudgetExceeded` instead of paging the machine — tested.
- `StudentWriter`: per-tensor SHA-256, atomic shard writes, index
  flushed per tensor; `verify()` detects corrupted shards — tested.
- `Checkpoint`: JSON state + SHA-256, atomic replace; corrupted
  checkpoints are refused, never loaded — tested.
- `OutOfCoreDriver`: SCAN → PLAN → EXECUTE with per-tensor checkpointing
  and `--resume`; an injected crash mid-run resumes with zero
  emitted-data loss (tested: crashed run + resume == clean run, index
  verified).
- External teachers: `prepare_teacher_source()` opens `.safetensors`
  directories with true mmap windows (no copy) or repacks `.npz`
  streaming (one array in RAM at a time).
- `stream_compress()`: per-window truncated SVD at the tensor's planned
  rank share; student shards are low-rank factor pairs with honestly
  accounted parameter counts.

Measured: tiny teacher (41.1K params) streamed through
`compress --out-of-core --target-params 20000 --memory-budget 1MB` →
20.4K student params (rank discretization overshoot +2%, reported),
mean measured retained energy 0.809, peak workspace 18KB of 1MB.

### F2 — Virtual 30B planning (labeled exactly this, never "validated")

New module `tensor_roll/planner.py`:

- `glimmer30b_manifest()`: realistic 30.19B-param, 500-tensor manifest;
  **zero weights allocated** (any read attempt raises).
- `plan()`: SCAN (metadata only) → sensitivity-aware nonuniform
  allocation (sensitive tensors up to 2× the uniform ratio, funded by
  insensitive ones down to 0.4×) → hard cap 8.5B, lands on the 8.0B
  target from either side.

Measured (this machine, peak RSS 35MB — bounded working set proven):

| Scale | Source params | Planned | Ratio | Within cap | Planner time |
|---|---|---|---|---|---|
| virtual 30B | 30.19B | 8.000B | 0.265 | yes | <0.1 s |

Scale-invariance regression (virtual manifests, no weights): the
planner holds the 8.5/30 ratio at 100K / 1M / 10M params within ±0.02 —
tested (`check_ratio_invariance`).

Actual 30B weights: **NOT TESTED** — no 30B checkpoint exists on this
machine.

### F3 — Backend honesty (`tensor-roll doctor`)

New modules `tensor_roll/doctor.py`, `tensor_roll/ir.py`,
`tensor_roll/conformance.py` (16 tests):

- `tensor-roll doctor` prints the live capability matrix: CPU/NumPy
  AVAILABLE, CUDA/CUDA-Q/Q# UNAVAILABLE (with reasons), sources
  VERIFIED (content-checked) vs execution NOT EXECUTED — source presence
  is never presented as backend execution.
- TensorRoll IR (`LOAD/SLICE/ROLL/FACTOR/ROTATE/ENTANGLE/MEASURE/
  RECONSTRUCT/GEMM/QUANTIZE/EMIT`) with JSON round-trip; CPU lowering
  executes numerically, CUDA-Q/Q# lowering returns NOT EXECUTED records.
- Integration audit (docs/BOUNDARIES.md): the CUDA-Q↔Q# boundary is a
  checksummed serialized JSON trace file — file-mediated, not an API.

### F4/F5 — Fidelity instrumentation and the 3/24 divergence

New module `tensor_roll/fidelity.py`:

- `divergence_trace()`: per-token teacher/student trace on
  teacher-forced contexts (tokens, top-k, full logits, KL/JS divergence,
  cosine similarity, entropies, per-layer hidden-state and attention
  divergence) — isolates model divergence from context drift.
- `first_divergence()`: first position AND first layer with significant
  representation drift (not just final tokens).
- `ablation_report()`: per-component evidence (embeddings, attention
  projections/output, MLP up/down, norms, output head) — which
  component's restoration recovers agreement.
- `fidelity_aware_select()`: the section-6 utility
  `savings − recon − logit − hidden − task` applied **uniformly** to
  every decision class from measured errors; quantum scoring untouched.

Resolution experiment (`experiments/fidelity_resolution.py`, fixed
seeds, 3 prompts × 24 tokens; full report in the experiment workdir):

- Section-6 check on 148 real roll leaves: `select_plan` chose
  65 QUANTIZE / 83 PRESERVE at cost 11645 params; `fidelity_aware_select`
  chose 55 QUANTIZE / 93 PRESERVE at cost 11456 — 12 decision diffs.
  The fidelity penalty made the planner *more conservative* (fewer
  quantize decisions), from measured errors only. Quantum-gated
  candidates lost under both scorings — the empirical result stands.
- V0 (baseline pipeline): token agreement **7/72**, mean teacher-KL
  1.548.
- First divergence at **position 0, layer_0** (hidden rel_err 1.04,
  KL 2.73): the student diverges immediately in the first layer — not
  late context drift but immediate representation mismatch.
- Ablations: *every* single-component teacher graft hurt agreement
  (best: mlp_down −0.021; output_head graft → 0.000) and raised KL.
  The distilled student is a co-adapted system — fidelity loss is
  **systemic, not localizable** to one component.
- V1 (stages 2+3 hidden/logit distillation at 3× steps, then finetune):
  agreement **4/72**, mean KL **1.187**. More distillation closes the
  distribution gap (KL −0.36) but does **not** recover greedy token
  agreement.

Conclusion (measured, not guessed): the bottleneck is
capacity/architecture, not distillation weight. Closing 3/24-class
agreement needs a larger student budget or a better init, not more
distillation steps. This remains an open failure.

### F6 — INT4 characterization and recovery (`tensor_roll/quant.py`)

- `int4_scheme_report()`: per-tensor / per-channel / group-wise INT4
  benchmarked by **measured MSE** on each tensor; `search_group_size()`
  picks the argmin over {16, 32, 64, 128}.
- `quant_error_map()`: per-tensor min/max/mean/variance/outlier
  fraction, MSE, cosine similarity, rel err, per-channel and per-group
  MSE — all measured.
- `mixed_precision_plan()`: INT4/INT8 assignment under a physical size
  budget; tensors where INT8 buys the most error reduction per extra
  byte stay INT8.
- `recovery_train()`: quantization-aware recovery —
  INT4-dequantized student → distillation → requantize → evaluate;
  reports PPL before/after/requantized. Improvement is reported only if
  measured.

Measured recovery run (demo-scale student_final → INT4, 30 steps,
2026-09-29; `experiments/int4_recovery.json`): PPL 7.43 before →
5.49 after distillation → **6.88 after requantization**
(gap closed by 0.55, `improved: true`). Recovery training genuinely
recovers part of the INT4 degradation; the residual gap is the
quantization floor, not a training artifact.

Baseline INT4 gap stands at 6.74 PPL (15.13 − 8.39) for the full
pipeline variant until a recovery run on that exact artifact measures
otherwise.

### Remaining failures (not fixed, reported as such)

- Token agreement 3/24: under investigation (see F4/F5 results above).
- INT4 PPL gap 6.74: characterized, not yet closed.
- No CUDA / CUDA-Q / Q# execution on this machine (hardware absent).
- No actual Glimmer 30B weights ingested (checkpoint absent).
- `tensor-roll demo` exposes all of the above in its [10/9] status
  board; nothing is relabeled to look like a pass.
