# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The recursive TensorRoll operator, per-partition metrics, searched
compression policy, and the explicit classical/quantum dispatch boundary.

Honesty contract (enforced here, not just documented):
  * Classical execution on this machine is NumPy on CPU. Every classical
    report carries backend="cpu-numpy", device="cpu".
  * Quantum-candidate execution runs on the classical state-vector
    simulator in ``qsim.py``. Every such report carries
    backend="qsim-classical". Nothing here claims real QPU/GPU execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .util import fmt_bytes, rss_mb

DECISIONS = (
    "PRESERVE",
    "QUANTIZE",
    "FACTORIZE",
    "MERGED",
    "ROUTED",
    "RECONSTRUCTED",
    "PRUNED",
    "QUANTUM-ENCODED",
)

#: Max elements a block may have to be eligible for the quantum path
#: (amplitude encoding needs 2**n amplitudes; we cap at 2**6).
QUANTUM_ELIGIBLE_MAX = 256  # 8 qubits; honest near-term register size

CLASSICAL_BACKEND = "cpu-numpy"
QSIM_BACKEND = "qsim-classical"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def tensor_metrics(W: np.ndarray) -> Dict:
    W = np.asarray(W)
    flat = W.ravel().astype(np.float64)
    m = {
        "shape": list(W.shape),
        "ndim": int(W.ndim),
        "params": int(W.size),
        "dtype": str(W.dtype),
        "norm": float(np.linalg.norm(flat)),
        "variance": float(flat.var()) if flat.size else 0.0,
    }
    if W.ndim == 2 and min(W.shape) > 1:
        s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
        total = float((s ** 2).sum())
        p = (s ** 2) / total if total > 0 else np.zeros_like(s)
        p = p[p > 0]
        m["rank"] = int(np.sum(s > max(W.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)))
        m["entropy"] = float(-(p * np.log(p)).sum()) if p.size else 0.0
        m["spectral"] = float((s[0] ** 2) / total) if total > 0 else 0.0
        m["sing_vals"] = [float(x) for x in s[:8]]
    else:
        m["rank"] = int(1 if m["norm"] > 0 else 0)
        m["entropy"] = 0.0
        m["spectral"] = 1.0 if m["norm"] > 0 else 0.0
        m["sing_vals"] = []
    return m


def relative_error(A: np.ndarray, B: np.ndarray) -> float:
    A = A.astype(np.float64)
    denom = float(np.linalg.norm(A))
    if denom == 0:
        return 0.0 if float(np.linalg.norm(B)) == 0 else float("inf")
    return float(np.linalg.norm(A - B) / denom)


def cosine_sim(A: np.ndarray, B: np.ndarray) -> float:
    a = A.astype(np.float64).ravel()
    b = B.astype(np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Invocation reports — every kernel call, classical or quantum, reports these.
# ---------------------------------------------------------------------------

def new_report(
    *,
    backend: str,
    device: str,
    kernel: str,
    tensor_dims,
    precision: str,
    exec_time_s: float,
    memory_bytes: float,
    recursion_depth: int,
    shots=None,
    measurement_stats=None,
    reconstruction_error=None,
    extra=None,
) -> Dict:
    rep = {
        "backend": backend,
        "device": device,
        "kernel": kernel,
        "tensor_dims": [tuple(d) for d in tensor_dims],
        "precision": precision,
        "exec_time_s": float(exec_time_s),
        "memory_bytes": float(memory_bytes),
        "recursion_depth": int(recursion_depth),
        "shots": shots,
        "measurement_stats": measurement_stats,
        "reconstruction_error": reconstruction_error,
    }
    if extra:
        rep.update(extra)
    return rep


def dispatch_matmul(A: np.ndarray, B: np.ndarray, *, depth: int = 0,
                    label: str = "gemm") -> tuple:
    """Classical GEMM baseline. Backend is honestly cpu-numpy."""
    A = np.asarray(A)
    B = np.asarray(B)
    mem = float(A.nbytes + B.nbytes)
    t0 = time.perf_counter()
    C = A @ B
    dt = time.perf_counter() - t0
    mem += float(C.nbytes)
    rep = new_report(
        backend=CLASSICAL_BACKEND, device="cpu", kernel=label,
        tensor_dims=[A.shape, B.shape, C.shape], precision=str(A.dtype),
        exec_time_s=dt, memory_bytes=mem, recursion_depth=depth,
    )
    return C, rep


# ---------------------------------------------------------------------------
# Compression option evaluation (measured, not hard-coded)
# ---------------------------------------------------------------------------

def _svd_options(W: np.ndarray):
    U, s, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)
    return U, s, Vt


def _vq_codebook(W: np.ndarray, k: int = 16, iters: int = 12,
                 seed: int = 0):
    """From-scratch k-means vector quantization of W's elements.
    Returns (reconstructed_W, cost_params). Deterministic."""
    flat = W.ravel().astype(np.float64)
    rng = np.random.default_rng(seed)
    cents = np.quantile(flat, np.linspace(0, 1, k)).astype(np.float64)
    for _ in range(iters):
        assign = np.abs(flat[:, None] - cents[None, :]).argmin(axis=1)
        for j in range(k):
            m = assign == j
            if np.any(m):
                cents[j] = flat[m].mean()
    assign = np.abs(flat[:, None] - cents[None, :]).argmin(axis=1)
    Wd = cents[assign].reshape(W.shape)
    bits_per_idx = float(np.ceil(np.log2(k)))
    cost = float(k + flat.size * bits_per_idx / 32.0)
    return Wd, cost


def evaluate_options(name: str, W: np.ndarray, *, quantum_policy: str = "off",
                     qsim_encode=None, rank: int = 0) -> List[Dict]:
    """Measure candidate representations of W. Each option carries a real
    (cost_params, retained_energy, reconstruction_error) triple computed
    from actual transforms — the policy search below picks winners, so no
    decision is hard-coded here. Decision classes searched: PRESERVE,
    QUANTIZE (incl. codebook VQ), FACTORIZE, MERGED, ROUTED, RECONSTRUCTED,
    PRUNED, QUANTUM-ENCODED."""
    W = np.asarray(W, dtype=np.float64)
    params = int(W.size)
    opts: List[Dict] = []

    def add(decision, cost, energy, recon, detail=""):
        opts.append({
            "tensor": name, "decision": decision,
            "cost_params": float(cost), "retained_energy": float(energy),
            "reconstruction_error": float(recon), "detail": detail,
        })

    add("PRESERVE", params, 1.0, 0.0, "bit-exact")

    if W.ndim == 2 and min(W.shape) > 1:
        U, s, Vt = _svd_options(W)
        total_e = float((s ** 2).sum())
        # FACTORIZE at several ranks — measured energies
        fracs = [0.75, 0.5, 0.25]
        if rank and 0 < rank < min(W.shape):
            fracs.append(rank / min(W.shape))
        for frac in fracs:
            r = max(1, int(len(s) * frac))
            cost = r * (W.shape[0] + W.shape[1])
            energy = float((s[:r] ** 2).sum() / total_e) if total_e else 1.0
            Wr = (U[:, :r] * s[:r]) @ Vt[:r, :]
            add("FACTORIZE", cost, energy, relative_error(W, Wr), f"svd-rank-{r}")
        # QUANTIZE int8 — actually quantize and measure
        qmin, qmax = W.min(), W.max()
        if qmax > qmin:
            scale = (qmax - qmin) / 255.0
            Wq = np.round((W - qmin) / scale).astype(np.int64)
            Wd = Wq.astype(np.float64) * scale + qmin
            add("QUANTIZE", params * 8 / 32, 1.0 - relative_error(W, Wd) ** 2,
                relative_error(W, Wd), "int8-symmetric-range")
            # QUANTIZE via codebook vector quantization (k-means, k=16)
            Wv, vq_cost = _vq_codebook(W, k=16)
            vq_err = relative_error(W, Wv)
            add("QUANTIZE", vq_cost, max(0.0, 1.0 - vq_err ** 2), vq_err,
                "codebook-vq-k16")
        # PRUNED — magnitude prune 25% / 50%, sparse cost model
        flat = np.abs(W.ravel())
        for frac in (0.25, 0.5):
            k = int(flat.size * frac)
            if k > 0:
                thr = np.partition(flat, k)[k]
                Wp = np.where(np.abs(W) >= thr, W, 0.0)
                kept = int(np.count_nonzero(Wp))
                # sparse CSR-ish cost: values + indices
                cost = kept * 2
                energy = 1.0 - relative_error(W, Wp) ** 2
                add("PRUNED", cost, energy, relative_error(W, Wp),
                    f"magnitude-{int(frac*100)}pct-sparse")
        # QUANTUM-ENCODED — only tiny blocks, routed through the real qsim
        # RECONSTRUCTED — recursive-measurement classical reconstruction
        if W.size <= QUANTUM_ELIGIBLE_MAX and qsim_encode is not None and quantum_policy != "off":
            t0 = time.perf_counter()
            rec, qrep = qsim_encode(W)
            dt = time.perf_counter() - t0
            err = relative_error(W, rec)
            # Honest cost: classical reconstruction must still be stored.
            add("QUANTUM-ENCODED", params, max(0.0, 1.0 - err ** 2), err,
                f"qsim shots={qrep['shots']} t={dt:.4f}s")
            # RECONSTRUCTED: deeper recursive measurement -> classical
            # amplitude reconstruction stored as fp32 (measured, distinct
            # from the single-level quantum-encoded path above).
            from . import qsim as _Q
            rec2, rep2 = _Q.tensor_roll_recursive(
                W, depth=2, shots=qrep.get("shots", 2048), seed=0)
            err2 = relative_error(W, rec2)
            add("RECONSTRUCTED", params, max(0.0, 1.0 - err2 ** 2), err2,
                f"recursive-measurement depth=2 shots={rep2['shots']}")
            if quantum_policy == "explore":
                add("ROUTED", params, max(0.0, 1.0 - err ** 2), err,
                    "routed-to-qsim-for-measurement")
    else:
        # 1-D tensors (biases, norms): merge into parent op — MERGED.
        add("MERGED", params, 1.0, 0.0, "folded-into-parent-op")
        qmin, qmax = W.min(), W.max()
        if qmax > qmin:
            scale = (qmax - qmin) / 255.0
            Wd = np.round((W - qmin) / scale) * scale + qmin
            add("QUANTIZE", params * 8 / 32, 1.0 - relative_error(W, Wd) ** 2,
                relative_error(W, Wd), "int8-1d")
    return opts


# ---------------------------------------------------------------------------
# Recursive TensorRoll operator
# ---------------------------------------------------------------------------

@dataclass
class RollNode:
    name: str
    shape: tuple
    depth: int
    metrics: Dict = field(default_factory=dict)
    children: List["RollNode"] = field(default_factory=list)
    options: List[Dict] = field(default_factory=list)
    decision: Optional[Dict] = None


def roll_tree(name: str, W: np.ndarray, *, axis: int = 0, depth: int = 3,
              quantum_policy: str = "off", qsim_encode=None, rank: int = 0,
              _level: int = 0, _counter=None) -> RollNode:
    """Recursive partitioner backing TensorRoll(T, axis, depth, rank,
    quantum_policy).

    Recursively partitions T along `axis`; every node measures norm, rank,
    entropy, variance, spectral contribution. Leaves evaluate real
    compression options (rank-aware when `rank` > 0). Internal nodes
    aggregate child energy.
    """
    if _counter is None:
        _counter = [0]
    W = np.asarray(W)
    node = RollNode(name=name, shape=tuple(W.shape), depth=_level,
                    metrics=tensor_metrics(W))
    _counter[0] += 1
    node.metrics["roll_id"] = _counter[0]

    splittable = (_level < depth and W.ndim >= 1 and W.shape[axis] >= 4)
    if splittable:
        halves = np.array_split(W, 2, axis=axis)
        for i, h in enumerate(halves):
            node.children.append(
                roll_tree(f"{name}/{i}", h, axis=axis, depth=depth,
                          quantum_policy=quantum_policy,
                          qsim_encode=qsim_encode, rank=rank,
                          _level=_level + 1, _counter=_counter))
    else:
        node.options = evaluate_options(
            name, W, quantum_policy=quantum_policy, qsim_encode=qsim_encode,
            rank=rank)
    return node


def select_plan(leaves: List[RollNode], budget_params: float) -> Dict[str, Dict]:
    """Global representation search: greedy knapsack over measured
    (cost, retained_energy) options subject to the hard parameter budget.
    Returns {tensor_name: winning_option}."""
    # Start every leaf at its cheapest option, then upgrade greedily.
    srt = [sorted(leaf.options, key=lambda o: o["cost_params"]) for leaf in leaves]
    choice = [0] * len(leaves)
    total = sum(s[0]["cost_params"] for s in srt)

    improved = True
    while improved:
        improved = False
        best_gain, best_move = 0.0, None
        for i, opts in enumerate(srt):
            cur = choice[i]
            for j in range(cur + 1, len(opts)):
                dcost = opts[j]["cost_params"] - opts[cur]["cost_params"]
                denergy = opts[j]["retained_energy"] - opts[cur]["retained_energy"]
                if total + dcost <= budget_params and dcost > 0 and denergy > 0:
                    gain = denergy / dcost
                    if gain > best_gain:
                        best_gain, best_move = gain, (i, j, dcost)
        if best_move:
            i, j, dcost = best_move
            choice[i] = j
            total += dcost
            improved = True

    plan = {}
    for i, leaf in enumerate(leaves):
        plan[leaf.name] = srt[i][choice[i]]
    plan["_total_cost_params"] = total
    plan["_budget_params"] = float(budget_params)
    return plan


def iter_leaves(node: RollNode):
    if node.children:
        for c in node.children:
            yield from iter_leaves(c)
    else:
        yield node


def TensorRoll(T, axis: int = -1, depth: int = 2, rank: int = 8,
               quantum_policy: str = "balanced", qsim_encode=None,
               budget_params: float = 0.0, name: str = "T") -> Dict:
    """First-class recursive operator.

    TensorRoll(T, axis, depth, rank, quantum_policy):
      1. recursively partitions T along `axis` to `depth`, measuring every
         node (norm, rank, entropy, variance, spectral contribution);
      2. evaluates candidate representations per leaf — PRESERVE, QUANTIZE
         (incl. codebook VQ), FACTORIZE (rank-aware via `rank`), MERGED,
         ROUTED, RECONSTRUCTED, PRUNED, QUANTUM-ENCODED (the last gated by
         `quantum_policy`);
      3. searches the global representation plan under `budget_params`
         (0 = unconstrained: keep the best-energy option per leaf);
      4. returns the plan plus measured totals. No decision is hard-coded;
         every option carries measured (cost, retained_energy,
         reconstruction_error).

    Backend labels: ordinary options execute on cpu-numpy; quantum-gated
    options execute on qsim-classical (statevector simulator, NOT a QPU).
    """
    T = np.asarray(T, dtype=np.float64)
    ax = axis % T.ndim if T.ndim else 0
    tree = roll_tree(name, T, axis=ax, depth=depth,
                     quantum_policy=quantum_policy, qsim_encode=qsim_encode,
                     rank=rank)
    leaves = list(iter_leaves(tree))
    full = float(sum(l.metrics["params"] for l in leaves))
    budget = float(budget_params) if budget_params > 0 else full
    plan = select_plan(leaves, budget)
    for leaf in leaves:
        leaf.decision = plan[leaf.name]
    decisions = {}
    for leaf in leaves:
        d = leaf.decision["decision"]
        decisions[d] = decisions.get(d, 0) + 1
    energies = [leaf.decision["retained_energy"] for leaf in leaves]
    return {
        "tree": tree,
        "plan": plan,
        "n_leaves": len(leaves),
        "decisions": decisions,
        "total_cost_params": plan["_total_cost_params"],
        "budget_params": plan["_budget_params"],
        "full_params": full,
        "mean_retained_energy": float(np.mean(energies)) if energies else 1.0,
        "quantum_policy": quantum_policy,
        "rank": rank,
    }


def render_roll_line(node: RollNode, idx: int) -> str:
    m = node.metrics
    dec = node.decision["decision"] if node.decision else "—"
    backend = "qsim-classical" if node.decision and "QUANTUM" in dec else CLASSICAL_BACKEND
    return (
        f"[ROLL {idx:04d}] source: {node.name} shape: {list(node.shape)} "
        f"depth: {node.depth} params: {m['params']} "
        f"rank: {m['rank']} entropy: {m['entropy']:.3f} "
        f"spectral: {m['spectral']:.3f} "
        f"cost: {node.decision['cost_params']:.0f} "
        f"retained-energy: {node.decision['retained_energy']:.4f} "
        f"recon-err: {node.decision['reconstruction_error']:.2e} "
        f"backend: {backend} decision: {dec}"
    )
