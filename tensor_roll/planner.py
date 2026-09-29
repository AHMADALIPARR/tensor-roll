# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Global planner: SCAN -> PLAN -> EXECUTE.

Streaming introduces a global problem: local per-tensor decisions must
satisfy a global ~8B budget. The planner solves allocation in three
passes:

  SCAN:    lightweight per-tensor statistics WITHOUT materializing the
           network (manifest metadata only; optional measured calibration
           on sampled windows for sensitivity).
  PLAN:    solve the allocation  Σ target_params <= 8.5B  (prefer ~8.0B).
           Budgets are NONUNIFORM and sensitivity-aware: tensors measured
           as sensitive receive more budget, funded by less-sensitive
           tensors. The only global invariant is the final budget.
  EXECUTE: the expensive transforms run per the plan (see ooc.py).

The 27.8% ≈ 28.3% small-scale ratio is an explicit regression target:
virtual manifests at 100K/1M/10M/100M params verify the planner holds the
requested compression ratio as scale grows (section 4).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from . import core as C

CAP_PARAMS = 8_500_000_000      # hard architectural budget
TARGET_PARAMS = 8_000_000_000   # preferred target
BUDGET_RATIO = 8.5 / 30.0       # scale-invariant ratio (regression target)


# ---------------------------------------------------------------------------
# SCAN
# ---------------------------------------------------------------------------

def scan_rows(specs) -> List[dict]:
    """Lightweight per-tensor stats from manifest specs only. No weights."""
    rows = []
    for s in specs:
        shape = tuple(s.shape)
        rows.append({
            "tensor": s.name,
            "shape": list(shape),
            "dtype": s.dtype,
            "params": int(s.params),
            "bytes": int(s.nbytes),
            "shard": s.shard,
            "is_matrix": len(shape) == 2 and min(shape) > 1,
        })
    return rows


def calibrate_sensitivity(rows: List[dict], sample_fn=None) -> Dict[str, float]:
    """Measured sensitivity per tensor in [0, 1].

    `sample_fn(name)` returns a small (<=4096-elem) sample window of the
    tensor's data, or None when unavailable (virtual manifests). From the
    sample we measure spectral concentration and entropy via
    core.tensor_metrics; high spectral concentration + low entropy =>
    low-rank-friendly => LOW sensitivity to compression. Tensors whose
    energy is spread across many singular values are HIGH sensitivity.

    When no data is available the sensitivity is uniform 0.5 (documented),
    and the planner degrades gracefully to a uniform-ratio allocation.
    """
    sens = {}
    for r in rows:
        s = 0.5
        if sample_fn is not None:
            try:
                w = sample_fn(r["tensor"])
            except Exception:
                w = None
            if w is not None:
                w = np.asarray(w, dtype=np.float64)
                if w.ndim == 2 and min(w.shape) > 1:
                    m = C.tensor_metrics(w)
                    # spectral in [0,1]: concentrated => compressible
                    # entropy normalized by log(rank): spread => sensitive
                    rank = max(1, m["rank"])
                    h_norm = m["entropy"] / math.log(rank + 1e-9) \
                        if rank > 1 else 0.0
                    s = float(np.clip(0.35 * (1.0 - m["spectral"])
                                      + 0.65 * min(1.0, h_norm), 0.0, 1.0))
                else:
                    # 1-D params (norms/biases): cheap, keep them
                    s = 0.15
        sens[r["tensor"]] = s
    return sens


# ---------------------------------------------------------------------------
# PLAN — nonuniform, sensitivity-aware, hard-capped
# ---------------------------------------------------------------------------

def plan(rows: List[dict], *, target: float = TARGET_PARAMS,
         cap: float = CAP_PARAMS,
         sensitivity: Optional[Dict[str, float]] = None,
         ratio: Optional[float] = None) -> dict:
    """Allocate per-tensor target parameter counts.

    Starts from a uniform ratio (target/total, or `ratio` when given for
    scale-invariant small-scale runs), then reallocates from
    low-sensitivity to high-sensitivity tensors: sensitive tensors may
    receive up to 2x the uniform ratio, funded by insensitive tensors down
    to 0.4x. Iterates until Σ targets <= cap (hard) and Σ targets is as
    close to `target` as the bounds allow.

    Returns {"tensors": {name: {"target_params": int, "ratio": float,
    "sensitivity": float}}, "total_target_params": int, "cap": cap,
    "within_cap": bool}.
    """
    total = sum(r["params"] for r in rows)
    if total == 0:
        raise ValueError("planner: empty model")
    if ratio is not None:
        # scale-invariant mode: the target IS the ratio (regression target)
        target = float(ratio) * total
    base_ratio = min(1.0, target / total)
    sens = sensitivity or {r["tensor"]: 0.5 for r in rows}

    alloc = {}
    for r in rows:
        s = float(np.clip(sens.get(r["tensor"], 0.5), 0.0, 1.0))
        # sensitivity -> multiplier in [0.4, 2.0]:
        # s=0 -> 0.4, s=0.5 -> 1.2, s=1 -> 2.0 (recentered below)
        mult = 0.4 + 1.6 * s
        # recenter so the mean multiplier is ~1: divide by mean
        alloc[r["tensor"]] = {"params": r["params"], "s": s, "mult": mult}

    mean_mult = sum(v["mult"] for v in alloc.values()) / len(alloc)
    for v in alloc.values():
        v["mult"] /= mean_mult

    def totals():
        t = {k: max(1, int(v["params"] * base_ratio * v["mult"]))
             for k, v in alloc.items()}
        return t, sum(t.values())

    targets, tot = totals()
    # hard cap: scale down proportionally (preserves nonuniform shape)
    if tot > cap:
        f = cap / tot
        targets = {k: max(1, int(v * f)) for k, v in targets.items()}
        tot = sum(targets.values())
    # land on the target from either side (target <= cap by contract; if a
    # caller passes target > cap, the cap wins). The sensitivity-driven
    # nonuniformity is param-correlated, so without this the plan can
    # overshoot the target while staying under the cap.
    goal = min(target, cap)
    if tot > goal:
        f = goal / tot
        targets = {k: max(1, int(v * f)) for k, v in targets.items()}
        tot = sum(targets.values())
    elif tot < goal:
        f = goal / tot
        targets = {k: max(1, int(v * f)) for k, v in targets.items()}
        tot = sum(targets.values())

    out = {"tensors": {
        k: {"target_params": targets[k],
            "ratio": targets[k] / alloc[k]["params"],
            "sensitivity": alloc[k]["s"]}
        for k in targets},
        "total_target_params": tot,
        "total_source_params": total,
        "achieved_ratio": tot / total,
        "cap": float(cap),
        "within_cap": tot <= cap}
    return out


# ---------------------------------------------------------------------------
# Scale-invariance regression (section 4)
# ---------------------------------------------------------------------------

def virtual_manifest_for_params(total_params: int, seed: int = 0) -> List[dict]:
    """Build a virtual tensor manifest totaling ~total_params WITHOUT
    allocating weights. Mixes matrix shapes like a transformer
    (attention projections, MLP up/down, embeddings, norms)."""
    rng = np.random.default_rng(seed)
    rows = []
    remaining = int(total_params)
    i = 0
    # block templates (fractions of a "layer")
    while remaining > 0:
        d = int(rng.choice([64, 128, 256, 512, 1024, 2048, 4096, 8192]))
        kind = rng.integers(0, 5)
        if kind == 0:
            shape = (d, d)
        elif kind == 1:
            shape = (d, 4 * d)
        elif kind == 2:
            shape = (4 * d, d)
        elif kind == 3:
            shape = (32768, d)
        else:
            shape = (d,)
        p = int(np.prod(shape))
        if p > remaining and remaining < 10_000:
            shape = (remaining,)
            p = remaining
        if p > remaining:
            continue
        rows.append({"tensor": f"blk{i}", "shape": list(shape),
                     "dtype": "float16", "params": p,
                     "bytes": p * 2, "shard": "virtual.safetensors",
                     "is_matrix": len(shape) == 2})
        remaining -= p
        i += 1
        if i > 20000:
            break
    return rows


def check_ratio_invariance(total_params: int, *, ratio: float = BUDGET_RATIO,
                           tol: float = 0.02, seed: int = 0) -> dict:
    """Regression: the planner must hold the requested compression ratio
    as scale grows, using virtual manifests (no weight allocation)."""
    rows = virtual_manifest_for_params(total_params, seed=seed)
    src = sum(r["params"] for r in rows)
    p = plan(rows, ratio=ratio)
    achieved = p["achieved_ratio"]
    return {"source_params": src, "requested_ratio": ratio,
            "achieved_ratio": achieved,
            "within_tol": abs(achieved - ratio) <= tol,
            "plan_total": p["total_target_params"]}
