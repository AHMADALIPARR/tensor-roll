# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""TensorRoll IR: the backend-neutral plan representation.

A roll plan ({tensor_name: {decision, cost_params, ...}}) compiles to a
list of IROps. ``lower(ops, backend)`` then maps the plan onto a concrete
backend:

  * ``cpu``    — REALLY executes the classical ops (LOAD/SLICE/GEMM/
                 QUANTIZE/FACTOR) with NumPy, and the quantum-family ops
                 (ROLL/ROTATE/ENTANGLE/MEASURE/RECONSTRUCT) with the real
                 qsim-classical statevector simulator. Execution records
                 carry measured milliseconds and output digests.
  * ``cuda`` / ``cuda-q`` / ``qsharp`` — return per-op records with state
                 NOT EXECUTED, a reason, and the source-verification state
                 (VERIFIED/PRESENT) of the corresponding kernel source.
                 For ``cuda-q``/``qsharp``, ROTATE/ENTANGLE/MEASURE records
                 reference the real gate-trace schema from
                 ``tensor_roll/boundary.py``. This is IR-specification
                 only: no in-memory ABI between this IR and any vendor
                 kernel is claimed — exchange happens via the checksummed
                 file artifact until a toolchain executes the kernels.

The IR never invents numbers for backends it cannot run.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import boundary as BD
from . import quant as QT
from . import qsim as QSIM
from .doctor import verify_sources

IR_VERSION = 1

VALID_OPS = ("LOAD", "SLICE", "ROLL", "FACTOR", "ROTATE", "ENTANGLE",
             "MEASURE", "RECONSTRUCT", "GEMM", "QUANTIZE", "EMIT")

_CPU_BACKENDS = ("cpu-numpy", "qsim-classical")

#: IROp -> boundary kernel name for the quantum-family ops (used when
#: lowering to cuda-q / qsharp so records reference the real gate schema).
_OP_KERNEL = {
    "ROTATE": "tensor_roll_rotate",
    "ENTANGLE": "tensor_roll_entangle",
    "MEASURE": "tensor_roll_measure",
    "RECONSTRUCT": "tensor_roll_reconstruct",
    "ROLL": "tensor_roll_recursive",
}

#: which audited source file backs each non-cpu backend
_BACKEND_SOURCE = {"cuda": "CUDA source", "cuda-q": "CUDA-Q source",
                   "qsharp": "Q# source"}

_NOT_EXECUTED_REASON = {
    "cuda": ("no CUDA device, no nvidia-smi, no nvcc on this machine; "
             "lowering to CUDA is a plan only — cuda/tensor_roll_gemm.cu "
             "was not compiled or executed"),
    "cuda-q": ("no CUDA-Q toolchain on this machine; lowering to cuda-q is "
               "IR-specification only until a toolchain executes "
               "cudaq/tensor_roll_kernels.py"),
    "qsharp": ("no Q# toolchain (and no dotnet) on this machine; lowering "
               "to qsharp is IR-specification only until a toolchain "
               "executes qsharp/TensorRoll.qs"),
}


@dataclass
class IROp:
    op: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.op not in VALID_OPS:
            raise ValueError(f"unknown IR op {self.op!r}; valid: {VALID_OPS}")
        self.inputs = list(self.inputs)
        self.outputs = list(self.outputs)
        self.params = dict(self.params)
        self.meta = dict(self.meta)


# ---------------------------------------------------------------------------
# Plan -> IR
# ---------------------------------------------------------------------------

def _decision_ops(name: str, cur: str, decision: str,
                  cost: Dict[str, Any]) -> Tuple[List[IROp], str]:
    """Per-decision op sequence. Returns (ops, current_value_name)."""
    ops: List[IROp] = []
    if decision == "QUANTIZE":
        ops.append(IROp("QUANTIZE", [cur], [f"q:{name}"],
                        {"bits": int(cost.get("bits", 8))},
                        {"decision": decision}))
        cur = f"q:{name}"
    elif decision == "FACTORIZE":
        ops.append(IROp("FACTOR", [cur], [f"f:{name}"],
                        {"rank": int(cost.get("rank", 4))},
                        {"decision": decision}))
        cur = f"f:{name}"
    elif decision == "QUANTUM-ENCODED":
        nq = int(cost.get("n_qubits", 3))
        angles = [float(a) for a in
                  cost.get("angles", [0.3] * nq)]
        ops.append(IROp("ROTATE", [cur], [f"qr:{name}"],
                        {"angles": angles, "n_qubits": nq},
                        {"decision": decision,
                         "backend": "qsim-classical"}))
        ops.append(IROp("ENTANGLE", [f"qr:{name}"], [f"qe:{name}"], {},
                        {"decision": decision}))
        ops.append(IROp("MEASURE", [f"qe:{name}"], [f"qm:{name}"],
                        {"shots": int(cost.get("shots", 256)),
                         "seed": int(cost.get("seed", 0))},
                        {"decision": decision}))
        ops.append(IROp("RECONSTRUCT", [f"qm:{name}"], [f"qc:{name}"],
                        {}, {"decision": decision,
                             "note": "classical reconstruction by construction"}))
        cur = f"qc:{name}"
    else:
        # PRESERVE / MERGED / ROUTED / RECONSTRUCTED / PRUNED and anything
        # else: no transform op invented; the decision rides in meta.
        ops.append(IROp("EMIT", [cur], [f"out:{name}"],
                        {"artifact": f"{name}.json"},
                        {"decision": decision,
                         "note": "pass-through: no transform op for this decision"}))
        return ops, f"out:{name}"
    ops.append(IROp("EMIT", [cur], [f"out:{name}"],
                    {"artifact": f"{name}.json"},
                    {"decision": decision}))
    return ops, f"out:{name}"


def build_ir(roll_plan: Dict[str, Dict[str, Any]]) -> List[IROp]:
    """Compile a roll plan ({tensor: {decision, cost_params, ...}}) to IR.

    Per tensor: LOAD -> ROLL -> per-decision op(s) -> EMIT.
    """
    ops: List[IROp] = []
    for name, spec in roll_plan.items():
        spec = spec or {}
        decision = str(spec.get("decision", "PRESERVE"))
        cost = dict(spec.get("cost_params", {}))
        ops.append(IROp("LOAD", [], [f"t:{name}"],
                        {"tensor": name}, {"source": "roll_plan"}))
        ops.append(IROp("ROLL", [f"t:{name}"], [f"r:{name}"],
                        {"depth": int(cost.get("depth", 1))},
                        {"note": "recursive partition/transform step"}))
        dops, _ = _decision_ops(name, f"r:{name}", decision, cost)
        ops.extend(dops)
    return ops


def to_json(ops: List[IROp]) -> Dict[str, Any]:
    return {"ir_version": IR_VERSION,
            "ops": [asdict(op) for op in ops]}


def from_json(doc: Dict[str, Any]) -> List[IROp]:
    if doc.get("ir_version") != IR_VERSION:
        raise ValueError(f"unsupported IR version {doc.get('ir_version')!r}")
    return [IROp(**item) for item in doc["ops"]]


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------

def _arr_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _lower_cpu(ops: List[IROp]) -> List[Dict[str, Any]]:
    vals: Dict[str, np.ndarray] = {}
    payloads: Dict[str, Dict] = {}
    records: List[Dict[str, Any]] = []

    def payload_for(name: str) -> Dict:
        if name in payloads:
            return payloads[name]
        arr = vals.get(name)
        if arr is None:
            raise KeyError(f"no array or payload named {name!r}")
        if arr.size > 256:
            raise ValueError(
                f"quantum-family op on {arr.size} elements exceeds the "
                "256-element eligibility cap; refusing")
        p, _ = QSIM.tensor_roll_encode(arr, encoding="amplitude")
        return p

    for seq, op in enumerate(ops):
        t0 = time.perf_counter()
        rec: Dict[str, Any] = {"seq": seq, "op": op.op,
                               "inputs": op.inputs, "outputs": op.outputs}
        try:
            if op.op == "LOAD":
                data = op.params.get("data")
                if data is None:
                    raise ValueError(
                        "LOAD has no params['data'] — nothing to execute")
                arr = np.asarray(data, dtype=np.float64)
                if "shape" in op.params:
                    arr = arr.reshape(op.params["shape"])
                vals[op.outputs[0]] = arr
                rec.update(state="EXECUTED", backend="cpu-numpy",
                           detail={"shape": list(arr.shape)})
            elif op.op == "SLICE":
                src = vals[op.inputs[0]]
                # params["slices"]: per-dim [lo, hi] pair, or null for full dim
                sel = tuple(slice(p[0], p[1]) if isinstance(p, (list, tuple))
                            else slice(None)
                            for p in op.params["slices"])
                vals[op.outputs[0]] = src[sel]
                rec.update(state="EXECUTED", backend="cpu-numpy",
                           detail={"shape": list(vals[op.outputs[0]].shape)})
            elif op.op == "GEMM":
                C = vals[op.inputs[0]] @ vals[op.inputs[1]]
                vals[op.outputs[0]] = C
                rec.update(state="EXECUTED", backend="cpu-numpy",
                           detail={"shape": list(C.shape),
                                   "checksum": float(C.sum()),
                                   "sha256": _arr_sha256(C)})
            elif op.op == "QUANTIZE":
                q, meta = QT.quantize_int8(vals[op.inputs[0]])
                vals[op.outputs[0]] = q.astype(np.float64)
                rec.update(state="EXECUTED", backend="cpu-numpy",
                           detail={"fmt": meta["fmt"], "scale": meta["scale"],
                                   "codes_sha256": _arr_sha256(q)})
            elif op.op == "FACTOR":
                W = vals[op.inputs[0]]
                r = int(op.params.get("rank", 2))
                U, S, Vt = np.linalg.svd(W.astype(np.float64),
                                         full_matrices=False)
                What = (U[:, :r] * S[:r]) @ Vt[:r, :]
                vals[op.outputs[0]] = What
                denom = float(np.linalg.norm(W))
                err = float(np.linalg.norm(W - What) / denom) if denom else 0.0
                rec.update(state="EXECUTED", backend="cpu-numpy",
                           detail={"rank": r,
                                   "relative_error": err,
                                   "sha256": _arr_sha256(What)})
            elif op.op == "ROLL":
                arr = vals[op.inputs[0]]
                if arr.size > 256:
                    raise ValueError(
                        f"ROLL on {arr.size} elements exceeds the 256-element "
                        "quantum-eligibility cap; refusing")
                What, rep = QSIM.tensor_roll_recursive(
                    arr, depth=int(op.params.get("depth", 1)),
                    shots=int(op.params.get("shots", 256)),
                    seed=int(op.params.get("seed", 0)))
                vals[op.outputs[0]] = What
                rec.update(state="EXECUTED", backend="qsim-classical",
                           detail={"kernel": "tensor_roll_recursive",
                                   "reconstruction_error":
                                       rep.get("reconstruction_error"),
                                   "sha256": _arr_sha256(What)})
            elif op.op == "ROTATE":
                p = payload_for(op.inputs[0])
                p2, rep = QSIM.tensor_roll_rotate(
                    p, angles=op.params["angles"])
                payloads[op.outputs[0]] = p2
                rec.update(state="EXECUTED", backend="qsim-classical",
                           detail={"kernel": "tensor_roll_rotate"})
            elif op.op == "ENTANGLE":
                p = payload_for(op.inputs[0])
                p2, _ = QSIM.tensor_roll_entangle(p)
                payloads[op.outputs[0]] = p2
                rec.update(state="EXECUTED", backend="qsim-classical",
                           detail={"kernel": "tensor_roll_entangle"})
            elif op.op == "MEASURE":
                p = payload_for(op.inputs[0])
                p2, rep = QSIM.tensor_roll_measure(
                    p, shots=int(op.params.get("shots", 256)),
                    seed=int(op.params.get("seed", 0)))
                payloads[op.outputs[0]] = p2
                stats = rep.get("measurement_stats") or {}
                rec.update(state="EXECUTED", backend="qsim-classical",
                           detail={"kernel": "tensor_roll_measure",
                                   "shots": int(op.params.get("shots", 256)),
                                   "n_outcomes": stats.get("n_outcomes"),
                                   "top_p": stats.get("top_p")})
            elif op.op == "RECONSTRUCT":
                p = payload_for(op.inputs[0])
                if "counts" not in p:
                    raise ValueError(
                        "RECONSTRUCT needs a measured payload (run MEASURE first)")
                What, _ = QSIM.tensor_roll_reconstruct(p)
                vals[op.outputs[0]] = What
                rec.update(state="EXECUTED", backend="qsim-classical",
                           detail={"kernel": "tensor_roll_reconstruct",
                                   "shape": list(What.shape),
                                   "sha256": _arr_sha256(What)})
            elif op.op == "EMIT":
                arr = vals.get(op.inputs[0])
                if arr is None:
                    raise ValueError(
                        f"EMIT input {op.inputs[0]!r} was never produced")
                vals[op.outputs[0]] = arr
                rec.update(state="EXECUTED", backend="cpu-numpy",
                           detail={"artifact": op.params.get("artifact"),
                                   "shape": list(arr.shape),
                                   "sha256": _arr_sha256(arr)})
            else:  # pragma: no cover - constructor guards VALID_OPS
                raise ValueError(f"no cpu lowering for {op.op}")
        except Exception as exc:  # noqa: BLE001 - record, don't crash
            rec.update(state="FAILED",
                       reason=f"{type(exc).__name__}: {exc}")
        rec["ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        records.append(rec)
    return records


def _lower_not_executed(ops: List[IROp], backend: str) -> List[Dict[str, Any]]:
    sources = verify_sources()
    src_state = sources[_BACKEND_SOURCE[backend]]["state"]
    records: List[Dict[str, Any]] = []
    for seq, op in enumerate(ops):
        rec: Dict[str, Any] = {
            "seq": seq, "op": op.op, "inputs": op.inputs,
            "outputs": op.outputs, "backend": backend,
            "state": "NOT EXECUTED",
            "reason": _NOT_EXECUTED_REASON[backend],
            "source": src_state,
            "source_path": sources[_BACKEND_SOURCE[backend]]["path"],
        }
        if backend in ("cuda-q", "qsharp") and op.op in _OP_KERNEL:
            # Reference the REAL gate-trace schema: this is what the vendor
            # kernel would have to implement. IR-specification only — no
            # ABI compatibility is asserted (see module docstring).
            kernel = _OP_KERNEL[op.op]
            nq = int(op.params.get("n_qubits",
                                   len(op.params.get("angles", [0.3, 0.3]))))
            rec["gate_trace_schema"] = BD.TRACE_SCHEMA
            rec["gate_trace"] = BD.trace_for_kernel(kernel, max(1, nq))
            rec["note"] = ("IR-specification only: references the boundary "
                           "gate-trace schema; asserts no in-memory ABI "
                           "compatibility with vendor kernels. Exchange "
                           "happens via the checksummed file artifact "
                           "(tensor_roll/boundary.py) until a toolchain "
                           "executes the kernels.")
        records.append(rec)
    return records


def lower(ops: List[IROp], backend: str) -> List[Dict[str, Any]]:
    """Lower IR ops to *backend*. Returns per-op execution records.

    ``cpu`` executes for real (NumPy + qsim-classical). ``cuda``,
    ``cuda-q`` and ``qsharp`` return NOT EXECUTED records with reasons —
    a plan, not a fabrication.
    """
    if backend == "cpu":
        return _lower_cpu(ops)
    if backend in ("cuda", "cuda-q", "qsharp"):
        return _lower_not_executed(ops, backend)
    raise ValueError(f"unknown backend {backend!r}; "
                     "expected 'cpu', 'cuda', 'cuda-q' or 'qsharp'")
