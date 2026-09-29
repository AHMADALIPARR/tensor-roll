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
