# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Process/IR boundary between the CUDA-Q kernels and the Q# operations.

There is no direct CUDA-Q <-> Q# linkage: the two toolchains do not share
an in-process API. The real boundary used by tensor-roll is a serialized,
checksummed, versioned artifact containing:

  1. a gate-level IR trace (S-expression-ish JSON list) describing exactly
     which kernel ran, on how many qubits, with which gates, and
  2. the measurement counts produced by executing that trace.

Either side of the boundary — the CUDA-Q process
(``cudaq/tensor_roll_kernels.py``), the Q# toolchain (``qsharp/TensorRoll.qs``),
or the classical ``qsim-classical`` simulator — can write this artifact and
any other side can read it. Nothing is invented: the file is real JSON on
disk with a SHA-256 checksum, and the reader refuses corrupted artifacts.

True QIR/MLIR bitcode exchange requires both vendor toolchains installed;
when they are present, the same trace maps 1:1 onto QIR gate ops
(``__quantum__qis__ry``, ``__quantum__qis__cnot__ctl``, ...).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

BOUNDARY_VERSION = 1
TRACE_SCHEMA = "tensor-roll/boundary-trace"

# ---------------------------------------------------------------------------
# Gate IR construction (mirrors both kernel families' semantics)
# ---------------------------------------------------------------------------

def _ry(q: int, theta: float) -> Dict[str, Any]:
    return {"gate": "ry", "qubits": [q], "params": [float(theta)],
            "qir": "__quantum__qis__ry__body"}

def _cnot(c: int, t: int) -> Dict[str, Any]:
    return {"gate": "cnot", "qubits": [c, t], "params": [],
            "qir": "__quantum__qis__cnot__ctl"}

def _measure(q: int) -> Dict[str, Any]:
    return {"gate": "measure", "qubits": [q], "params": [],
            "qir": "__quantum__qis__mz__body"}

def _reset(q: int) -> Dict[str, Any]:
    return {"gate": "reset", "qubits": [q], "params": [],
            "qir": "__quantum__qis__reset__body"}


def trace_for_kernel(kernel: str, n_qubits: int,
                     angles: Optional[List[float]] = None,
                     depth: int = 0) -> List[Dict[str, Any]]:
    """Build the gate IR trace for a tensor_roll kernel.

    This is the exact gate sequence the CUDA-Q kernels in
    cudaq/tensor_roll_kernels.py apply and the Q# operations in
    qsharp/TensorRoll.qs apply — a single source of truth for the
    cross-toolchain boundary.
    """
    a = list(angles) if angles else [0.0] * n_qubits
    gates: List[Dict[str, Any]] = []
    if kernel == "tensor_roll_encode":
        for q in range(n_qubits):
            gates.append(_ry(q, 2.0 * a[q % len(a)]))
    elif kernel == "tensor_roll_rotate":
        for q in range(n_qubits):
            gates.append(_ry(q, a[q % len(a)]))
    elif kernel == "tensor_roll_entangle":
        for q in range(n_qubits - 1):
            gates.append(_cnot(q, q + 1))
    elif kernel in ("tensor_roll_measure", "tensor_roll_reconstruct"):
        for q in range(n_qubits):
            gates.append(_measure(q))
    elif kernel == "tensor_roll_recursive":
        for lvl in range(max(1, depth)):
            k = max(1, n_qubits - lvl)
            for q in range(k):
                gates.append(_ry(q, a[q % len(a)]))
            for q in range(k - 1):
                gates.append(_cnot(q, q + 1))
            if lvl < max(1, depth) - 1:
                gates.append(_measure(k - 1))
                gates.append(_reset(k - 1))
        for q in range(n_qubits):
            gates.append(_measure(q))
    else:
        raise ValueError(f"unknown tensor_roll kernel: {kernel}")
    return gates


# ---------------------------------------------------------------------------
# Trace artifact: write / read with checksum
# ---------------------------------------------------------------------------

def write_trace(path: str, kernel: str, n_qubits: int, shots: int,
                counts: Dict[str, int],
                angles: Optional[List[float]] = None,
                depth: int = 0,
                producer: str = "qsim-classical") -> Dict[str, Any]:
    """Write a boundary trace artifact. Returns the decoded header."""
    gates = trace_for_kernel(kernel, n_qubits, angles=angles, depth=depth)
    body = {
        "schema": TRACE_SCHEMA,
        "version": BOUNDARY_VERSION,
        "kernel": kernel,
        "n_qubits": n_qubits,
        "shots": shots,
        "gates": gates,
        "n_gates": len(gates),
        "counts": {k: int(v) for k, v in counts.items()},
        "producer": producer,
        "produced_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime()),
        "qir_mapping": "each gate lists its __quantum__qis__* QIR op name",
    }
    payload = json.dumps(body, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"sha256": digest, "trace": body}, f, indent=2)
    return body


def read_trace(path: str) -> Dict[str, Any]:
    """Read and checksum-verify a boundary trace artifact."""
    with open(path) as f:
        doc = json.load(f)
    payload = json.dumps(doc["trace"], sort_keys=True).encode()
    if hashlib.sha256(payload).hexdigest() != doc["sha256"]:
        raise ValueError(f"boundary trace checksum mismatch: {path}")
    if doc["trace"].get("schema") != TRACE_SCHEMA:
        raise ValueError(f"not a tensor-roll boundary trace: {path}")
    return doc["trace"]


def counts_to_amplitudes(trace: Dict[str, Any]) -> List[float]:
    """Classical reconstruction estimator shared across the boundary:
    amplitude_i = sqrt(count_i / shots). Classical by construction."""
    shots = trace["shots"]
    dim = 1 << trace["n_qubits"]
    amps = [0.0] * dim
    for bits, c in trace["counts"].items():
        idx = int(bits, 2) if set(bits) <= {"0", "1"} else 0
        if idx < dim and shots > 0:
            amps[idx] = (c / shots) ** 0.5
    return amps
