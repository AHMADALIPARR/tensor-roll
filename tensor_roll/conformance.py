# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Backend conformance contract for tensor-roll.

For each backend in [numpy, cuda, cuda-q, qsharp] this module either
EXECUTES a fixed test vector and compares against a precomputed expected
output, or records NOT EXECUTED with the probe reason. It never invents
numbers for a backend it cannot run.

Comparison semantics:
  * Classical backends (numpy): EXACT floating-point equality. The test
    vectors are fixed literals (inputs and expected outputs below were
    generated once by the NumPy path with numpy 1.26.4 and hard-coded, so
    conformance never depends on regenerating them). Re-executing A@B in
    the same process/BLAS is bit-identical, so ``match`` is a strict
    equality check.
  * Quantum measurement (cuda-q, if a toolchain is ever present): exact
    equality is MEANINGLESS for sampled counts. Comparison is
    distributional — chi-square goodness-of-fit of the sampled counts
    against the exact Born distribution computed by an independent
    classical statevector re-implementation of the same circuit, plus
    top-outcome agreement. Both are computed for real when executed.
    Bit-ordering assumption: cudaq.sample is assumed to print qubit n-1
    leftmost (matching qsim._measure_counts); the raw counts are preserved
    in the record so the comparison can be re-done if a real toolchain
    disagrees.

``run_conformance(workdir)`` writes machine-readable JSON to
``<workdir>/conformance.json``: per backend {state, reason, outputs (when
executed), expected, match (bool or null), measured fields}.
"""

from __future__ import annotations

import importlib.util
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import qsim as _qsim
from . import quant as QT
from .doctor import cudaq_toolchain, qsharp_toolchain, verify_sources
from .util import utc_now_iso, write_json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Fixed test vectors. Inputs AND expected outputs were generated once by the
# NumPy path (numpy 1.26.4) and hard-coded here; conformance re-executes the
# inputs and demands exact equality with these literals.
# ---------------------------------------------------------------------------

_VEC_MATMUL_A = [
    [0.3502483584951709, 0.7847217336600212, 0.514335740326966,
     0.08997722942380026],
    [0.000875639493777336, 0.5394009336088772, 0.3630734994591833,
     0.6561419338727186],
    [0.6857523669661448, 0.28778351696528304, 0.31958621893475014,
     0.5151307753113961],
    [0.6711942461900504, 0.18942041616006677, 0.8017567034768246,
     0.29579706764196234],
]
_VEC_MATMUL_B = [
    [0.2863552420503723, 0.9336882487158795, 0.8979195228933917,
     0.3228591430362756],
    [0.4453653183959694, 0.20672814313855958, 0.14869990691155455,
     0.06988183293319483],
    [0.7603953595694858, 0.9773212481372078, 0.5499040814050973,
     0.18811956769502203],
    [0.4823260186818574, 0.922115503183646, 0.22132889167532976,
     0.2948266793901039],
]
_EXPECT_MATMUL_C = [
    [0.8842801672837294, 1.0748874892220983, 0.7339327709835938,
     0.29120298285643825],
    [0.8330349433254504, 1.0722050228875455, 0.42587388862051745,
     0.29972641147510964],
    [0.8160104365083471, 1.4871203551119165, 0.9482989100732389,
     0.4535069784712827],
    [1.0288839735854158, 1.7221776351919256, 1.1372029362357199,
     0.4679732366928886],
]

_VEC_QUANT_X = [0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.75, 2.0]
_EXPECT_QUANT_CODES = [-43, 0, -86, 42, -128, -22, -107, 127]
_EXPECT_QUANT_SCALE = 0.011764705882352941
_EXPECT_QUANT_ZERO = -1.0

_VEC_ENCODE_BLOCK = [0.1, 0.4, 0.2, 0.8, -0.3, 0.6, -0.1, 0.5]
# Amplitude-encoding magnitudes: x / ||x|| (signs are stored classically
# in the hybrid encoding; see qsim.tensor_roll_encode).
_EXPECT_ENCODE_AMPS = [
    0.08006407690254357, 0.32025630761017426, 0.16012815380508713,
    0.6405126152203485, -0.24019223070763068, 0.48038446141526137,
    -0.08006407690254357, 0.40032038451271784,
]


def test_vectors() -> Dict[str, Any]:
    """The fixed deterministic input vectors (no expected values)."""
    return {
        "matmul": {"A": [row[:] for row in _VEC_MATMUL_A],
                   "B": [row[:] for row in _VEC_MATMUL_B]},
        "quantize": {"x": list(_VEC_QUANT_X)},
        "encode": {"block": list(_VEC_ENCODE_BLOCK)},
    }


def expected_outputs() -> Dict[str, Any]:
    """The hard-coded expected outputs for the test vectors."""
    return {
        "matmul": {"C": [row[:] for row in _EXPECT_MATMUL_C]},
        "quantize": {"codes": list(_EXPECT_QUANT_CODES),
                     "scale": _EXPECT_QUANT_SCALE,
                     "zero": _EXPECT_QUANT_ZERO},
        "encode": {"amplitudes": list(_EXPECT_ENCODE_AMPS)},
    }


# ---------------------------------------------------------------------------
# numpy backend: executes for real, exact comparison
# ---------------------------------------------------------------------------

def _run_numpy(vecs: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    A = np.array(vecs["matmul"]["A"])
    B = np.array(vecs["matmul"]["B"])
    C = A @ B
    matmul_ms = (time.perf_counter() - t0) * 1000.0

    x = np.array(vecs["quantize"]["x"])
    q, meta = QT.quantize_int8(x)

    blk = np.array(vecs["encode"]["block"])
    amps = blk / np.linalg.norm(blk)

    exp = expected_outputs()
    outputs = {
        "matmul": {"C": C.tolist()},
        "quantize": {"codes": [int(v) for v in q.ravel()],
                     "scale": meta["scale"], "zero": meta["zero"]},
        "encode": {"amplitudes": amps.tolist()},
    }
    checks = {
        "matmul_exact": outputs["matmul"]["C"] == exp["matmul"]["C"],
        "quantize_exact": (
            outputs["quantize"]["codes"] == exp["quantize"]["codes"]
            and outputs["quantize"]["scale"] == exp["quantize"]["scale"]
            and outputs["quantize"]["zero"] == exp["quantize"]["zero"]),
        "encode_exact": (outputs["encode"]["amplitudes"]
                         == exp["encode"]["amplitudes"]),
    }
    return {
        "backend": "numpy",
        "state": "EXECUTED",
        "reason": None,
        "outputs": outputs,
        "expected": exp,
        "match": all(checks.values()),
        "checks": checks,
        "measured": {"matmul_ms": round(matmul_ms, 3),
                     "numpy_version": np.__version__,
                     "comparison": "exact fp equality"},
    }


# ---------------------------------------------------------------------------
# cuda backend: probe only
# ---------------------------------------------------------------------------

def _run_cuda() -> Dict[str, Any]:
    src = verify_sources()["CUDA source"]
    return {
        "backend": "cuda",
        "state": "NOT EXECUTED",
        "reason": ("probe only: no CUDA device on this machine (nvidia-smi "
                   "absent, no /dev/nvidia*); cuda/tensor_roll_gemm.cu was "
                   "not compiled and no measurement was taken"),
        "outputs": None,
        "expected": None,
        "match": None,
        "source": src["state"],
        "measured": {},
    }


# ---------------------------------------------------------------------------
# cuda-q backend: run run_smoke for real if the toolchain exists
# ---------------------------------------------------------------------------

def _load_kernels_module():
    path = os.path.join(REPO_ROOT, "cudaq", "tensor_roll_kernels.py")
    spec = importlib.util.spec_from_file_location(
        "tensor_roll_cudaq_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _smoke_reference_distribution() -> Dict[str, float]:
    """Exact Born distribution of cudaq/tensor_roll_kernels.py::run_smoke's
    circuit (n_qubits=4, depth=2), via branch enumeration of the single
    mid-circuit measurement, using qsim's own unitary primitives.

    Circuit replicated exactly:
      encode:  RY(2*a[i]) on qubit i,  a[i] = 2*atan(0.1*(i+1))
      level 0: RY(a[q]) q=0..3; CNOT ladder q=0..2; measure+reset qubit 3
      level 1: RY(a[q]) q=0..2; CNOT ladder q=0..1
      final:   measure all
    """
    n = 4
    angles = [2.0 * math.atan(0.1 * (i + 1)) for i in range(n)]

    def branch_final(sv: np.ndarray) -> Dict[str, float]:
        probs: Dict[str, float] = {}
        for d, amp in enumerate(sv):
            p = float(abs(amp) ** 2)
            if p > 0:
                probs[format(d, "04b")] = probs.get(format(d, "04b"), 0.0) + p
        return probs

    sv = np.zeros(2 ** n, dtype=np.complex128)
    sv[0] = 1.0
    for q in range(n):
        sv = _qsim._apply_1q(sv, _qsim._ry(2.0 * angles[q]), q, n)
    # level 0
    for q in range(n):
        sv = _qsim._apply_1q(sv, _qsim._ry(angles[q]), q, n)
    for q in range(n - 1):
        sv = _qsim._apply_cnot(sv, q, q + 1, n)
    # mid-circuit measurement of qubit 3 -> two branches; reset to |0>
    branches = []
    for b in (0, 1):
        p = sum(float(abs(sv[d]) ** 2) for d in range(2 ** n)
                if (d >> 3) & 1 == b)
        if p <= 0:
            continue
        t = np.zeros(2 ** n, dtype=np.complex128)
        for d in range(2 ** n):
            if (d >> 3) & 1 == b:
                t[d & ~(1 << 3)] += sv[d] / math.sqrt(p)
        branches.append((p, t))
    dist: Dict[str, float] = {}
    for p_branch, t in branches:
        # level 1
        for q in range(n - 1):
            t = _qsim._apply_1q(t, _qsim._ry(angles[q]), q, n)
        for q in range(n - 2):
            t = _qsim._apply_cnot(t, q, q + 1, n)
        for bits, p in branch_final(t).items():
            dist[bits] = dist.get(bits, 0.0) + p_branch * p
    return dist


def _run_cudaq() -> Dict[str, Any]:
    ok, detail = cudaq_toolchain()
    src = verify_sources()["CUDA-Q source"]
    if not ok:
        return {
            "backend": "cuda-q",
            "state": "NOT EXECUTED",
            "reason": ("CUDA-Q kernels not executed: " + detail),
            "outputs": None,
            "expected": None,
            "match": None,
            "source": src["state"],
            "measured": {},
        }
    try:
        mod = _load_kernels_module()
        t0 = time.perf_counter()
        result = mod.run_smoke(n_qubits=4, shots=1024)
        ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001
        return {
            "backend": "cuda-q",
            "state": "FAILED",
            "reason": f"toolchain present but run_smoke raised: {exc!r}",
            "outputs": None, "expected": None, "match": None,
            "source": src["state"], "measured": {},
        }
    counts = result["counts"]
    shots = result["shots"]
    ref = _smoke_reference_distribution()
    # chi-square vs the exact Born distribution + top-outcome agreement.
    # (Bit-ordering assumption documented in the module docstring.)
    keys = set(counts) | set(ref)
    chi2 = sum((counts.get(k, 0) - shots * ref.get(k, 0.0)) ** 2
               / max(shots * ref.get(k, 0.0), 1e-12) for k in keys)
    top_sampled = max(counts, key=counts.get)
    top_exact = max(ref, key=ref.get)
    l1 = sum(abs(counts.get(k, 0) / shots - ref.get(k, 0.0)) for k in keys)
    return {
        "backend": "cuda-q",
        "state": "EXECUTED",
        "reason": None,
        "outputs": {"counts": counts, "shots": shots,
                    "target": result.get("target")},
        "expected": {"distribution": ref,
                     "note": ("exact Born distribution of the smoke circuit; "
                              "comparison is distributional (chi-square / "
                              "top-outcome), never exact count equality")},
        "match": top_sampled == top_exact,
        "checks": {"top_outcome_agreement": top_sampled == top_exact,
                   "top_sampled": top_sampled, "top_exact": top_exact},
        "measured": {"smoke_ms": round(ms, 3),
                     "chi_square": chi2,
                     "l1_distance": l1,
                     "comparison": "distributional (chi-square + top-outcome)",
                     "bit_order_assumption": ("cudaq.sample prints qubit n-1 "
                                              "leftmost, as qsim does")},
        "source": src["state"],
    }


# ---------------------------------------------------------------------------
# qsharp backend: run a minimal snippet for real if the toolchain exists
# ---------------------------------------------------------------------------

def _run_qsharp() -> Dict[str, Any]:
    ok, detail = qsharp_toolchain()
    src = verify_sources()["Q# source"]
    if not ok:
        return {
            "backend": "qsharp",
            "state": "NOT EXECUTED",
            "reason": ("Q# operations not executed: " + detail),
            "outputs": None,
            "expected": None,
            "match": None,
            "source": src["state"],
            "measured": {},
        }
    try:
        import qsharp
        t0 = time.perf_counter()
        # Minimal real execution: X then measure must return One.
        res = qsharp.eval("use q = Qubit(); X(q); MResetZ(q)")
        ms = (time.perf_counter() - t0) * 1000.0
        got = str(res)
    except Exception as exc:  # noqa: BLE001
        return {
            "backend": "qsharp",
            "state": "FAILED",
            "reason": f"toolchain present but qsharp.eval raised: {exc!r}",
            "outputs": None, "expected": None, "match": None,
            "source": src["state"], "measured": {},
        }
    return {
        "backend": "qsharp",
        "state": "EXECUTED",
        "reason": None,
        "outputs": {"x_then_measure": got},
        "expected": {"x_then_measure": "One"},
        "match": got == "One",
        "measured": {"eval_ms": round(ms, 3),
                     "comparison": "exact (single deterministic outcome)",
                     "note": ("toolchain-level smoke via qsharp.eval; the "
                              "TensorRoll.qs operations themselves require a "
                              "Q# project harness and were not invoked")},
        "source": src["state"],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_conformance(workdir: str) -> Dict[str, Any]:
    """Run the conformance contract; write <workdir>/conformance.json."""
    os.makedirs(workdir, exist_ok=True)
    vecs = test_vectors()
    backends = {
        "numpy": _run_numpy(vecs),
        "cuda": _run_cuda(),
        "cuda-q": _run_cudaq(),
        "qsharp": _run_qsharp(),
    }
    doc = {
        "tool": "tensor-roll conformance",
        "generated_at": utc_now_iso(),
        "backends": backends,
    }
    write_json(os.path.join(workdir, "conformance.json"), doc)
    return doc
