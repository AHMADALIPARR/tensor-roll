# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Representation search: 30B -> 8B (parameterized).

Pipeline:
  1. Per-tensor TensorRoll analysis (core.roll_tree) -> measured options.
  2. Global greedy selection under the hard parameter budget.
  3. Student *architecture* search over (d, heads, dff, layers): each
     candidate is scored by projected retained energy, where every teacher
     tensor is mapped to the candidate shape by truncated-SVD projection
     and its retained singular energy is measured. No hard-coded answer —
     the argmax under the budget wins.
  4. Student weight init from the teacher via the winning projections
     (Xavier where no projection exists), with per-tensor provenance.

The budget is P_target <= 8.5B for the real 30B path. At small scale the
same ratio applies: budget = teacher_params * (8.5/30).
"""

from __future__ import annotations

import itertools

import numpy as np

from . import core as C
from .model import Config, count_params, init_params
from . import qsim as Q

BUDGET_RATIO = 8.5 / 30.0  # hard architectural budget, same at every scale
ABS_BUDGET_CAP = 8_500_000_000


def _svd_cache(params: dict) -> dict:
    cache = {}
    for k, v in params.items():
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 2 and min(v.shape) > 1:
            U, s, Vt = np.linalg.svd(v, full_matrices=False)
            cache[k] = (U, s, Vt)
    return cache


def project_matrix(W: np.ndarray, out_shape: tuple, svd=None):
    """Truncated-SVD projection of W to out_shape (both dims <= input dims).
    Returns (W_proj, retained_energy_fraction). Optimal rank-k approximation,
    cropped — the classical baseline every exotic option is measured against.
    """
    W = np.asarray(W, dtype=np.float64)
    mo, no = out_shape
    m, n = W.shape
    assert mo <= m and no <= n, f"projection only shrinks: {W.shape} -> {out_shape}"
    if svd is None:
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
    else:
        U, s, Vt = svd
    k = min(mo, no)
    Wp = (U[:mo, :k] * s[:k]) @ Vt[:k, :no]
    total = float((s ** 2).sum())
    energy = float((s[:k] ** 2).sum() / total) if total > 0 else 1.0
    return Wp.astype(np.float32), energy


def _candidate_configs(teacher_cfg: Config):
    """Small grid of student architectures to search (real 30B path uses a
    wider grid over the same code)."""
    ds = sorted({teacher_cfg.d // 2, max(8, teacher_cfg.d * 3 // 4)})
    hs = sorted({max(1, teacher_cfg.heads - 1), teacher_cfg.heads})
    dffs = sorted({teacher_cfg.dff // 2, max(16, teacher_cfg.dff * 3 // 4)})
    ls = sorted({max(1, teacher_cfg.layers - 1), teacher_cfg.layers})
    for d, h, dff, l in itertools.product(ds, hs, dffs, ls):
        if d % h == 0:
            yield Config(vocab=teacher_cfg.vocab, d=d, heads=h,
                         layers=l, dff=dff, seq=teacher_cfg.seq)


def _config_param_count(cfg: Config) -> int:
    # mirrors init_params structure without allocating
    n = cfg.vocab * cfg.d
    per_layer = (4 * cfg.d * cfg.d + 4 * cfg.d      # q,k,v,o + biases
                 + 4 * cfg.d                        # two layernorms
                 + 2 * cfg.d * cfg.dff + cfg.dff + cfg.d)  # mlp
    n += cfg.layers * per_layer
    n += 2 * cfg.d + cfg.d * cfg.vocab + cfg.vocab
    return n


def search_student_config(teacher_params: dict, teacher_cfg: Config,
                          budget_params: float):
    """Returns (best_cfg, scored_list). scored_list entries are
    (cfg, est_energy, param_count), sorted by energy desc."""
    svd = _svd_cache(teacher_params)
    scored = []
    for cfg in _candidate_configs(teacher_cfg):
        pc = _config_param_count(cfg)
        if pc > budget_params:
            continue
        energies, weights = [], []
        for k, v in teacher_params.items():
            v = np.asarray(v)
            if v.ndim != 2 or k not in svd:
                continue
            # map teacher key -> student key shape
            sk = _student_key(k, cfg, teacher_cfg)
            if sk is None:
                continue
            sshape = _student_shape(k, v.shape, cfg, teacher_cfg)
            if sshape is None or sshape[0] > v.shape[0] or sshape[1] > v.shape[1]:
                continue
            _, e = project_matrix(v, sshape, svd[k])
            energies.append(e)
            weights.append(v.size)
        est = float(np.average(energies, weights=weights)) if energies else 0.0
        scored.append({"cfg": cfg.to_dict(), "est_energy": est, "params": pc})
    scored.sort(key=lambda s: -s["est_energy"])
    if not scored:
        raise RuntimeError("no student config fits the budget; cannot proceed honestly")
    best = Config.from_dict(scored[0]["cfg"])
    return best, scored


def _student_key(k: str, cfg: Config, tcfg: Config):
    # student uses identical key names; layer count may shrink
    if k.startswith("L"):
        l = int(k[1:].split(".")[0])
        if l >= cfg.layers:
            return None
    return k


def _student_shape(k: str, tshape: tuple, cfg: Config, tcfg: Config):
    d, td = cfg.d, tcfg.d
    if k == "emb":
        return (cfg.vocab, d)
    if k in ("lnfg", "lnfb"):
        return (d,)
    if k == "Whead":
        return (d, cfg.vocab)
    if k == "bhead":
        return (cfg.vocab,)
    if k.startswith("L"):
        rest = k.split(".", 1)[1]
        if rest in ("ln1g", "ln1b", "ln2g", "ln2b"):
            return (d,)
        if rest in ("Wq", "Wk", "Wv", "Wo"):
            return (d, d)
        if rest in ("bq", "bk", "bv", "bo"):
            return (d,)
        if rest == "W1":
            return (d, cfg.dff)
        if rest == "b1":
            return (cfg.dff,)
        if rest == "W2":
            return (cfg.dff, d)
        if rest == "b2":
            return (d,)
    return None


def init_student_from_teacher(teacher_params: dict, teacher_cfg: Config,
                              student_cfg: Config, rng: np.random.Generator):
    """Project teacher weights into the student shapes; Xavier elsewhere.
    Returns (student_params, provenance, achieved_energies)."""
    student = init_params(rng, student_cfg)  # Xavier baseline everywhere
    svd = _svd_cache(teacher_params)
    provenance, energies = {}, {}
    for k, v in teacher_params.items():
        sk = _student_key(k, student_cfg, teacher_cfg)
        if sk is None or sk not in student:
            provenance[k] = "dropped-by-architecture-search"
            continue
        sv = student[sk]
        v = np.asarray(v)
        if (v.ndim == 2 and sv.ndim == 2 and k in svd
                and sv.shape[0] <= v.shape[0] and sv.shape[1] <= v.shape[1]):
            proj, e = project_matrix(v, sv.shape, svd[k])
            student[sk] = proj
            provenance[k] = f"truncated-svd-projection energy={e:.4f}"
            energies[k] = e
        elif v.shape == sv.shape:
            student[sk] = v.astype(np.float32)
            provenance[k] = "copied-exact"
            energies[k] = 1.0
        else:
            provenance[k] = "xavier-init(shape-mismatch)"
            energies[k] = 0.0
    return student, provenance, energies


def head_consolidation_report(teacher_params: dict, teacher_cfg: Config,
                              student_cfg: Config) -> dict:
    """Which attention heads survived consolidation, by measured spectral
    energy of each head's QK block."""
    report = {}
    for l in range(teacher_cfg.layers):
        if l >= student_cfg.layers:
            continue
        Wq = np.asarray(teacher_params[f"L{l}.Wq"], dtype=np.float64)
        dh = teacher_cfg.dh
        energies = []
        for h in range(teacher_cfg.heads):
            blk = Wq[:, h * dh:(h + 1) * dh]
            s = np.linalg.svd(blk, compute_uv=False)
            energies.append(float((s ** 2).sum()))
        order = sorted(range(len(energies)), key=lambda h: -energies[h])
        kept = sorted(order[:student_cfg.heads])
        report[f"layer_{l}"] = {
            "head_energies": [round(e, 4) for e in energies],
            "kept_heads": kept,
            "dropped_heads": sorted(order[student_cfg.heads:]),
        }
    return report


def run_roll_analysis(teacher_params: dict, *, depth: int = 3,
                      quantum_policy: str = "off", rank: int = 0,
                      stats: dict = None):
    """Run the first-class TensorRoll(T, axis, depth, rank, quantum_policy)
    operator over every teacher tensor. Returns (trees, leaves).

    `stats`: optional {tensor_name: {"act_rms", "grad_norm", "grad_rel"}}
    from model.measure_tensor_stats; attached to each leaf's metrics so the
    search sees measured activation/gradient contributions.
    """
    trees, leaves = {}, []
    for k in sorted(teacher_params.keys()):
        W = np.asarray(teacher_params[k])
        res = C.TensorRoll(W, 0, depth, rank, quantum_policy,
                           qsim_encode=Q.qsim_encode_for_roll
                           if quantum_policy != "off" else None,
                           name=k)
        trees[k] = res["tree"]
        for leaf in C.iter_leaves(res["tree"]):
            if stats and k in stats:
                leaf.metrics["act_rms"] = stats[k]["act_rms"]
                leaf.metrics["grad_norm"] = stats[k]["grad_norm"]
                leaf.metrics["grad_rel"] = stats[k]["grad_rel"]
            leaves.append(leaf)
    return trees, leaves
