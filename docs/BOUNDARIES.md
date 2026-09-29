# Tensor Roll — process / IR boundaries

Copyright (C) 2026 SnapKitty Collective — AGPLv3.

This document states exactly where one execution domain ends and another
begins. There are three boundaries that matter; none of them is a wrapper
around an existing framework, and none of them is invented.

## 1. Classical dispatch boundary: `TensorRoll` → GEMM vs quantum candidate

Location: `tensor_roll/core.py` — `dispatch_matmul()` and
`evaluate_options()` / `qsim_encode_for_roll`.

The recursive operator `TensorRoll(T, axis, depth, rank, quantum_policy)`
partitions a tensor and evaluates candidate representations per leaf.
Ordinary candidates (PRESERVE / QUANTIZE / FACTORIZE / MERGED / ROUTED /
RECONSTRUCTED / PRUNED) execute as NumPy on CPU and every report labels
them `backend: cpu-numpy`.

A leaf is *eligible* for the quantum path only when all of these hold:

- `quantum_policy != "off"` (CLI: `roll --quantum-policy`,
  default for `demo` is `explore`),
- the leaf block has at most `QUANTUM_ELIGIBLE_MAX = 256` elements
  (8 qubits — an honest near-term register size),
- a qsim encoder is supplied.

Eligible leaves gain QUANTUM-ENCODED / RECONSTRUCTED / ROUTED candidates
whose numbers come from the real statevector simulator in
`tensor_roll/qsim.py`. Their reports are labeled
`backend: qsim-classical`, `device: cpu (statevector simulator)`.
That label is the boundary: it is classical linear algebra (complex128
statevectors, explicit unitary application, Born-rule sampling), never a
QPU, and it never claims to be one.

On a machine with the NVIDIA CUDA-Q toolchain, the same kernel family in
`cudaq/tensor_roll_kernels.py` can execute for real
(`run_smoke()` targets `qpp-cpu`); on this machine the import fails and the
module prints `unavailable` instead of pretending.

## 2. CUDA-Q ↔ Q# boundary: serialized trace artifact (real, not an API)

There is no direct CUDA-Q ↔ Q# linkage — the two toolchains do not share
an in-process API, and inventing one would be fabrication. The real
boundary is a serialized artifact produced by `tensor_roll/boundary.py`:

- `trace_for_kernel(kernel, n_qubits, angles, depth)` builds the gate-level
  IR: the exact gate sequence (`ry` / `cnot` / `measure` / `reset`) that
  both `cudaq/tensor_roll_kernels.py` and `qsharp/TensorRoll.qs` implement.
  Each gate carries its QIR op name (`__quantum__qis__ry__body`, …) so the
  trace maps 1:1 onto QIR when both vendor toolchains are present.
- `write_trace(path, ...)` writes `{sha256, trace}` JSON: schema version,
  kernel name, qubit count, shots, the gate list, the measurement counts,
  producer label, and timestamp.
- `read_trace(path)` refuses the file if the SHA-256 does not match or the
  schema is wrong.
- `counts_to_amplitudes(trace)` is the shared classical reconstruction
  estimator (amplitude = sqrt(count/shots)) — classical by construction on
  every side of the boundary.

Data flow:

```
CUDA-Q process (cudaq/tensor_roll_kernels.py)
        │  executes kernels (needs CUDA-Q toolchain + target)
        ▼
boundary.json  {gates, counts, sha256}     <-- the boundary: a file
        ▲
Q# toolchain (qsharp/TensorRoll.qs)
        │  implements the same gate sequence; reads/writes the artifact
```

`tensor-roll qkernel` writes `qkernel/boundary.json` on every run and
immediately re-reads it to prove the round trip. Corrupt the file and the
reader rejects it — verified by the test suite.

What is NOT claimed: at no point does tensor-roll convert CUDA-Q objects
to Q# objects in memory. The handoff is serialization, honestly labeled.

## 3. Sandbox boundary: supervisor ↔ untrusted child

Location: `tensor_roll/sandbox.py`, exercised by `tensor-roll sandbox`.

- The child runs in its own session/process group (`start_new_session`),
  with rlimits (address space, CPU, file size, process count), a filtered
  environment (only `PATH`/`LANG`/`TZ`/`HOME` survive), and `cwd` pinned to
  a jail directory.
- Policy: absolute paths must resolve inside the jail; the binary must be
  on an allowlist or not on the denylist (deny wins). Network namespaces
  are entered via `unshare(CLONE_NEWNET)` when the kernel permits it —
  probed at startup and reported (`netns_available=True/False`); on this
  machine the probe passes, so `qkernel`-adjacent untrusted code genuinely
  loses network access.
- Every spawn, policy verdict, timeout, and signal is appended as JSONL to
  `sandbox/audit.log`.

Signals: on timeout the supervisor kills the whole process group
(`killpg(SIGKILL)`), then reaps strays the same way. The audit log records
which.

## 4. What runs where (summary table)

| Component | Executes on | Label in reports |
|---|---|---|
| Tensor math, training, search | NumPy / CPU | `cpu-numpy` |
| Quantum kernel candidates | classical statevector sim | `qsim-classical` |
| CUDA-Q kernels | NVIDIA toolchain (absent here) | `unavailable` here |
| Q# operations | Q# toolchain (absent here) | `unavailable` here |
| CUDA-Q ↔ Q# handoff | serialized `boundary.json` | sha256-verified |
| Untrusted subprocesses | sandboxed child | audit-logged |

If a row says `unavailable`, no number was invented for it — the CLI
prints `unavailable` and the docs say what hardware would change that
(see `docs/ARCHITECTURE.md`).

## Integration state (failure-resolution audit, 2026-09-29)

**Finding: file-mediated.** The CUDA-Q ↔ Q# handoff in this repository is a
checksummed serialized JSON trace artifact on disk — not a direct in-process
API, not live QIR/MLIR bitcode exchange, and not process-to-process RPC.
Vendor-toolchain execution on this machine is additionally **source-only**:
CUDA, CUDA-Q, and Q# all probe UNAVAILABLE here, so no vendor kernel has
ever executed on this box. The file artifact itself round-trips for real
(`tensor-roll qkernel` writes `qkernel/boundary.json` and re-reads it;
`test_trace_roundtrip` corrupts it and watches the reader refuse).

Candidates considered and rejected:

- **direct** — rejected. `tensor_roll/boundary.py` states the two toolchains
  "do not share an in-process API", and no code path converts CUDA-Q
  objects to Q# objects in memory.
- **IR-mediated** (true QIR/MLIR bitcode) — rejected for this machine.
  `boundary.py` notes "True QIR/MLIR bitcode exchange requires both vendor
  toolchains installed"; neither is installed, so the QIR op names carried
  on each gate (`__quantum__qis__ry__body`, …) are a mapping specification,
  not an executed exchange.
- **process-mediated** — rejected. No live RPC exists between a CUDA-Q
  process and a Q# process; the handoff is a file both sides can write/read.
- **file-mediated** — ACCEPTED. Evidence below.
- **source-only** — true of *execution* on this machine (see probes), but
  the integration mechanism itself is richer than source-only: the trace
  artifact is really written, checksummed, read, and verified on every
  `qkernel` run via the `qsim-classical` path.

File/line evidence (from `tensor_roll/doctor.py::integration_state()`):

- `tensor_roll/boundary.py:7` — "an in-process API. The real boundary used
  by tensor-roll is a serialized," [no direct linkage; artifact, not API]
- `tensor_roll/boundary.py:8` — "checksummed, versioned artifact
  containing:" [integrity-checked artifact]
- `tensor_roll/boundary.py:20` — "True QIR/MLIR bitcode exchange requires
  both vendor toolchains installed;" [true QIR/MLIR exchange needs
  toolchains]
- `tensor_roll/boundary.py:57` — `def trace_for_kernel(...)` [gate IR
  constructor: the single source of truth both kernel families implement]
- `tensor_roll/boundary.py:102` — `def write_trace(...)` [trace is WRITTEN
  to disk as `{sha256, trace}` JSON]
- `tensor_roll/boundary.py:130` — `def read_trace(...)` [trace is READ from
  disk; checksum/schema mismatch raises]
- `tensor_roll/cli.py:291` — `bpath = os.path.join(wdir, "qkernel",
  "boundary.json")` [CLI writes the file artifact]
- `tensor_roll/cli.py:296` — `back = BD.read_trace(bpath)` [CLI verifies
  the round trip immediately]
- `cudaq/tensor_roll_kernels.py:9` — "These kernels are NOT executed by the
  tensor-roll CLI on machines without" [the CUDA-Q toolchain]
- `cudaq/tensor_roll_kernels.py:28` — `_HAS_CUDAQ = False`
  [toolchain-absent fallback path]
- `qsharp/TensorRoll.qs:13` — "NOTE: not compiled or executed in this
  repository's test environment" [source-only on this machine]

Live probes on this machine (`tensor-roll doctor`, 2026-09-29):

- CUDA: UNAVAILABLE — no nvidia-smi on PATH, no `/dev/nvidia*` devices.
- CUDA-Q: UNAVAILABLE — no toolchain (`ModuleNotFoundError` from a neutral
  directory; the repo-local `cudaq/` directory is a namespace package of
  kernel *source*, not the NVIDIA toolchain).
- Q#: UNAVAILABLE — no toolchain and no `dotnet`; the repo-local `qsharp/`
  directory is operation *source*, not an installed toolchain.
- GPU: NONE. CPU (NumPy) executes for real and passes.

What would change this determination: installing the NVIDIA CUDA-Q
toolchain and a Q# toolchain would move the *execution* rows from
source-only to executed, but the integration state would remain
file-mediated until a real QIR/MLIR bitcode handoff (not just a JSON trace
carrying QIR op names) is demonstrated between the two toolchains.

## Failure-resolution modules (2026-09-29)

### Out-of-core boundary (`tensor_roll/ooc.py`)

The streaming path never confuses "addressable" with "in RAM":

- A `VirtualManifest` carries shapes, dtypes, and byte offsets only.
  `reader()` raises `RuntimeError` — there is no silent path from a
  virtual manifest to weight data.
- `MappedTensor.view` is an `np.memmap` view; `TensorWindow.data`
  materializes only the window. `RollWorkspace` accounts the logical
  working set against an explicit ceiling and raises
  `MemoryBudgetExceeded` past it. Measured: virtual 30B planning peaks
  at 35MB RSS — the bound is demonstrated, not asserted.
- `StudentWriter` checksums every emitted tensor (SHA-256); `verify()`
  re-hashes on resume and refuses mismatches. Checkpoints are JSON +
  SHA-256 with atomic replace; corruption is refused, never loaded.
- The `.npz` repack path reads one array at a time into the shard
  format; `.safetensors` is windowed in place via mmap with zero copy.

### Planner boundary (`tensor_roll/planner.py`)

- `plan()` lands the allocation on the requested target (≤ 8.0B by
  default) and never above the 8.5B hard cap. Sensitivity is measured
  from sampled windows (`calibrate_sensitivity`) or documented as
  uniform 0.5 when no data exists — the planner degrades to a uniform
  ratio rather than inventing sensitivity.
- The 30B STRUCTURAL SCALE TEST label is load-bearing: it certifies
  storage addressing and planning under bounded RAM, and explicitly
  does **not** certify model quality. Actual 30B weights: NOT TESTED.

### Fidelity boundary (`tensor_roll/fidelity.py`)

- `divergence_trace()` measures on teacher-forced contexts so model
  divergence is isolated from context drift; free-run agreement
  (the 3/24 number) is reported separately and never mixed with it.
- `fidelity_aware_select()` applies the divergence penalty uniformly to
  every decision class from measured (sensitivity, reconstruction
  error). Quantum-gated options are scored from their own qsim-measured
  errors — the penalty cannot prefer or punish a backend beyond what
  the measurements say.
- Ablations replace student components with shape-aligned teacher
  weights and report Δagreement/ΔKL; components that cannot be aligned
  are skipped and reported as skipped, not zeroed.

### INT4 boundary (`tensor_roll/quant.py`)

- Scheme choice (per-tensor / per-channel / group-wise) is the argmin
  of measured MSE per tensor, never a hard-coded default.
- `recovery_train()` reports PPL before training, after training, and
  after requantization; "improved" is a measured boolean, not a claim.
