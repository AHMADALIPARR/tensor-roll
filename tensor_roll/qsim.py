# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classical state-vector simulator for the tensor_roll kernel family.

This is NOT a quantum computer. Every invocation report from this module
carries backend="qsim-classical" and device="cpu (statevector simulator)".
It implements genuine quantum-circuit linear algebra (unitary evolution +
Born-rule measurement) so the encode -> transform -> measure -> reconstruct
pipeline is empirically measurable, including its reconstruction error.

The real CUDA-Q kernels live in cudaq/tensor_roll_kernels.py and the real
Q# operations in qsharp/TensorRoll.qs. Neither can execute on this machine
(no QPU, no CUDA-Q toolchain, no Q# toolchain); see docs/BOUNDARIES.md.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from .core import QSIM_BACKEND, new_report

MAX_QUBITS = 10  # 2**10 amplitudes; hard cap for honesty about cost


# ---------------------------------------------------------------------------
# Low-level statevector machinery (real linear algebra)
# ---------------------------------------------------------------------------

def _n_qubits_for(size: int) -> int:
    n = max(1, int(np.ceil(np.log2(size))))
    if n > MAX_QUBITS:
        raise ValueError(
            f"block of {size} elements needs {n} qubits > cap {MAX_QUBITS}; "
            "refusing rather than silently approximating")
    return n


def _apply_1q(sv: np.ndarray, U: np.ndarray, qubit: int, n: int) -> np.ndarray:
    t = sv.reshape([2] * n)
    ax = n - 1 - qubit
    t = np.tensordot(U, t, axes=([1], [ax]))
    t = np.moveaxis(t, 0, ax)
    return t.reshape(2 ** n)


def _apply_cnot(sv: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    mask = ((idx >> control) & 1).astype(bool)
    flipped = idx ^ (1 << target)
    out = sv.copy()
    out[mask] = sv[flipped[mask]]
    return out


def _rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]],
                    dtype=np.complex128)


def _measure_counts(sv: np.ndarray, shots: int, seed: int) -> Dict[str, int]:
    probs = np.abs(sv) ** 2
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(sv), size=shots, p=probs)
    counts: Dict[str, int] = {}
    n = int(np.log2(len(sv)))
    for d in draws:
        key = format(int(d), f"0{n}b")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _dist_stats(counts: Dict[str, int], shots: int) -> Dict:
    p = np.array(sorted(counts.values()), dtype=np.float64) / shots
    ent = float(-(p * np.log2(p + 1e-300)).sum())
    return {
        "n_outcomes": len(counts),
        "top_p": float(p.max()) if p.size else 0.0,
        "dist_entropy_bits": ent,
    }


# ---------------------------------------------------------------------------
# tensor_roll kernel family (simulated)
# ---------------------------------------------------------------------------

def tensor_roll_encode(block: np.ndarray, *, encoding: str = "amplitude",
                       seed: int = 0) -> Tuple[np.ndarray, Dict]:
    """Encode a tensor block into a simulated n-qubit register.

    amplitude: |psi> = sum_i (x_i/||x||) |i>; signs are stored classically
               (documented hybrid classical/quantum encoding — amplitude
               encoding cannot carry sign in measurement probabilities).
    angle:     one qubit per element, RY(2*atan(x_i)) rotations.
    """
    t0 = time.perf_counter()
    W = np.asarray(block, dtype=np.float64)
    flat = W.ravel()
    mem = float(W.nbytes)
    shots = 0
    stats = None
    if encoding == "amplitude":
        n = _n_qubits_for(flat.size)
        padded = np.zeros(2 ** n)
        padded[:flat.size] = flat
        norm = float(np.linalg.norm(padded))
        signs = np.sign(padded)
        sv = (padded / norm).astype(np.complex128) if norm > 0 else np.zeros(2 ** n, dtype=np.complex128)
        if norm == 0:
            sv[0] = 1.0
        payload = {"sv": sv, "n": n, "norm": norm, "signs": signs,
                   "shape": W.shape, "encoding": encoding}
        mem += float(sv.nbytes)
    elif encoding == "angle":
        if flat.size > MAX_QUBITS:
            raise ValueError(f"angle encoding needs 1 qubit/element: {flat.size} > {MAX_QUBITS}")
        n = flat.size
        sv = np.ones(2 ** n, dtype=np.complex128) / np.sqrt(2 ** n)
        for q, x in enumerate(flat):
            sv = _apply_1q(sv, _ry(2 * np.arctan(x)), q, n)
        payload = {"sv": sv, "n": n, "shape": W.shape, "encoding": encoding}
        mem += float(sv.nbytes)
    else:
        raise ValueError(f"unknown encoding {encoding!r}")
    dt = time.perf_counter() - t0
    rep = new_report(backend=QSIM_BACKEND, device="cpu (statevector simulator)",
                     kernel="tensor_roll_encode",
                     tensor_dims=[W.shape, (2 ** payload["n"],)], precision="complex128",
                     exec_time_s=dt, memory_bytes=mem, recursion_depth=0,
                     shots=shots, measurement_stats=stats,
                     reconstruction_error=None,
                     extra={"encoding": encoding, "n_qubits": payload["n"]})
    return payload, rep


def tensor_roll_rotate(payload: Dict, *, angles, seed: int = 0) -> Tuple[Dict, Dict]:
    """Apply RY rotations (one angle per qubit) to the register."""
    t0 = time.perf_counter()
    sv, n = payload["sv"].copy(), payload["n"]
    angles = np.asarray(angles, dtype=float)
    if angles.size != n:
        raise ValueError(f"need {n} angles, got {angles.size}")
    for q in range(n):
        sv = _apply_1q(sv, _ry(float(angles[q])), q, n)
    payload = dict(payload, sv=sv)
    dt = time.perf_counter() - t0
    rep = new_report(backend=QSIM_BACKEND, device="cpu (statevector simulator)",
                     kernel="tensor_roll_rotate",
                     tensor_dims=[(2 ** n,)], precision="complex128",
                     exec_time_s=dt, memory_bytes=float(sv.nbytes),
                     recursion_depth=0, reconstruction_error=None)
    return payload, rep


def tensor_roll_entangle(payload: Dict, *, seed: int = 0) -> Tuple[Dict, Dict]:
    """CNOT ladder across the register (nearest-neighbour entanglement)."""
    t0 = time.perf_counter()
    sv, n = payload["sv"].copy(), payload["n"]
    for q in range(n - 1):
        sv = _apply_cnot(sv, q, q + 1, n)
    payload = dict(payload, sv=sv)
    dt = time.perf_counter() - t0
    rep = new_report(backend=QSIM_BACKEND, device="cpu (statevector simulator)",
                     kernel="tensor_roll_entangle",
                     tensor_dims=[(2 ** n,)], precision="complex128",
                     exec_time_s=dt, memory_bytes=float(sv.nbytes),
                     recursion_depth=0, reconstruction_error=None)
    return payload, rep


def tensor_roll_measure(payload: Dict, *, shots: int = 1024,
                        seed: int = 0) -> Tuple[Dict, Dict]:
    """Born-rule measurement of the register -> counts."""
    t0 = time.perf_counter()
    sv, n = payload["sv"], payload["n"]
    counts = _measure_counts(sv, shots, seed)
    stats = _dist_stats(counts, shots)
    payload = dict(payload, counts=counts, shots=shots)
    dt = time.perf_counter() - t0
    rep = new_report(backend=QSIM_BACKEND, device="cpu (statevector simulator)",
                     kernel="tensor_roll_measure",
                     tensor_dims=[(2 ** n,)], precision="complex128",
                     exec_time_s=dt, memory_bytes=float(sv.nbytes),
                     recursion_depth=0, shots=shots,
                     measurement_stats=stats, reconstruction_error=None)
    return payload, rep


def tensor_roll_reconstruct(payload: Dict) -> Tuple[np.ndarray, Dict]:
    """Reconstruct the classical block from measurement counts.

    Amplitude encoding: x_i_hat = sign_i * sqrt(count_i/shots) * ||x||.
    This is a statistical estimator — its error is measured and reported,
    never hidden.
    """
    t0 = time.perf_counter()
    if payload.get("encoding") != "amplitude" or "counts" not in payload:
        raise ValueError("reconstruct needs amplitude encoding + measured counts")
    n, shots = payload["n"], payload["shots"]
    norm, signs, shape = payload["norm"], payload["signs"], payload["shape"]
    probs = np.zeros(2 ** n)
    for bitstr, c in payload["counts"].items():
        probs[int(bitstr, 2)] = c / shots
    flat_hat = signs[:np.prod(shape)] * np.sqrt(probs[:int(np.prod(shape))]) * norm
    What = flat_hat.reshape(shape)
    dt = time.perf_counter() - t0
    rep = new_report(backend=QSIM_BACKEND, device="cpu (statevector simulator)",
                     kernel="tensor_roll_reconstruct",
                     tensor_dims=[(2 ** n,), tuple(shape)], precision="float64",
                     exec_time_s=dt, memory_bytes=float(What.nbytes),
                     recursion_depth=0, shots=shots,
                     measurement_stats=_dist_stats(payload["counts"], shots),
                     reconstruction_error=None)
    return What, rep


def tensor_roll_recursive(block: np.ndarray, *, depth: int = 2,
                          shots: int = 1024, seed: int = 0,
                          encoding: str = "amplitude") -> Tuple[np.ndarray, Dict]:
    """Recursive kernel: at each level, rotate by block-derived angles,
    entangle, measure a shrinking qubit subset, record per-level stats."""
    t0 = time.perf_counter()
    payload, reps = tensor_roll_encode(block, encoding=encoding, seed=seed)
    reps_list = [reps]
    n = payload["n"]
    levels = []
    for lvl in range(depth):
        lt0 = time.perf_counter()
        k = max(1, n - lvl)  # shrinking active subset
        angles = np.linspace(0.1, 0.9, k) * float(np.mean(np.abs(block)) + 1e-6)
        sub = dict(payload)
        sv = payload["sv"].copy()
        for q in range(k):
            sv = _apply_1q(sv, _ry(float(angles[q])), q, n)
        for q in range(k - 1):
            sv = _apply_cnot(sv, q, q + 1, n)
        sub["sv"] = sv
        counts = _measure_counts(sv, shots, seed + lvl)
        levels.append({"level": lvl, "active_qubits": k,
                       "stats": _dist_stats(counts, shots)})
        reps_list.append(new_report(
            backend=QSIM_BACKEND, device="cpu (statevector simulator)",
            kernel="tensor_roll_recursive", tensor_dims=[(2 ** n,)],
            precision="complex128", exec_time_s=time.perf_counter() - lt0,
            memory_bytes=float(sv.nbytes), recursion_depth=lvl + 1,
            shots=shots, measurement_stats=levels[-1]["stats"],
            reconstruction_error=None))
    payload, rep_m = tensor_roll_measure(payload, shots=shots, seed=seed)
    What, rep_r = tensor_roll_reconstruct(payload)
    dt = time.perf_counter() - t0
    from .core import relative_error
    err = relative_error(np.asarray(block, dtype=np.float64), What)
    summary = new_report(
        backend=QSIM_BACKEND, device="cpu (statevector simulator)",
        kernel="tensor_roll_recursive[summary]",
        tensor_dims=[tuple(np.asarray(block).shape), tuple(What.shape)],
        precision="float64", exec_time_s=dt,
        memory_bytes=float(np.asarray(block).nbytes + What.nbytes),
        recursion_depth=depth, shots=shots,
        measurement_stats={"levels": levels},
        reconstruction_error=err)
    # final measurement counts + qubit count: payload of the real
    # process/IR boundary artifact (see tensor_roll/boundary.py)
    summary["counts"] = {k: int(v) for k, v in payload["counts"].items()}
    summary["n_qubits"] = n
    return What, summary


def qsim_encode_for_roll(W: np.ndarray, *, shots: int = 2048,
                         seed: int = 0):
    """Adapter used by core.evaluate_options: full qsim pipeline on a block,
    returning (reconstructed_block, summary_report)."""
    return tensor_roll_recursive(np.asarray(W, dtype=np.float64),
                                 depth=1, shots=shots, seed=seed)
