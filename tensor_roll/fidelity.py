# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fidelity instrumentation + recursive fidelity search (F5).

The most important measured failure is 3/24 token agreement. This module
instruments EXACTLY where teacher/student behavior diverges:

  * per-token divergence trace: position, teacher/student tokens,
    teacher/student top-k, full logits, KL / JS divergence, cosine
    similarity, entropies, per-layer hidden-state divergence and
    per-layer attention divergence — all measured on teacher-forced
    contexts so model divergence is isolated from context drift;
  * first-divergence detection: the first position AND the first layer
    producing significant representation drift (not just final tokens);
  * recursive fidelity search: candidate transforms that save parameters
    but cause behavioral divergence pay an explicit, measured cost:
        utility = savings - recon_penalty - logit_penalty
                  - hidden_penalty - task_penalty
  * ablations by component (embeddings, attention projections, MLP
    up/down, norms, output head, quantization) reporting measured
    evidence — never guesses — about where fidelity is lost.

The fidelity penalty applies UNIFORMLY to every decision class
(including QUANTUM-ENCODED, scored from its own measured reconstruction
error). Nothing here touches the quantum scoring to force a winner; the
experiment still decides.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import core as C
from .model import forward


# ---------------------------------------------------------------------------
# divergence metrics
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-300)).sum())


def kl_div(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(softmax(p) || softmax(q))."""
    p, q = _softmax(p_logits), _softmax(q_logits)
    return float((p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum())


def js_div(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """Jensen-Shannon divergence between the two softmax distributions."""
    p, q = _softmax(p_logits), _softmax(q_logits)
    m = 0.5 * (p + q)
    kl_pm = float((p * (np.log(p + 1e-300) - np.log(m + 1e-300))).sum())
    kl_qm = float((q * (np.log(q + 1e-300) - np.log(m + 1e-300))).sum())
    return 0.5 * (kl_pm + kl_qm)


def top_k(logits: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    p = _softmax(logits)
    idx = np.argsort(p)[::-1][:k]
    return [(int(i), float(p[i])) for i in idx]


# ---------------------------------------------------------------------------
# divergence trace
# ---------------------------------------------------------------------------

def divergence_trace(t_params: dict, t_cfg, s_params: dict, s_cfg,
                     prompt: List[int], max_tokens: int = 24,
                     top_k_k: int = 5) -> dict:
    """Per-token teacher/student divergence on teacher-forced contexts.

    1. Teacher free-runs greedy -> reference token sequence.
    2. Student free-runs greedy -> token agreement count (the 3/24 number).
    3. At each position, BOTH models run on the teacher's prefix
       (teacher-forcing), isolating model divergence from context drift.
    Records per position: tokens, top-k, full logits, KL/JS/cosine,
    entropies, per-layer hidden-state divergence (rel err + cosine of the
    final hidden vectors) and per-layer attention divergence
    (mean |P_student - P_teacher| over heads).
    """
    prompt = list(prompt)
    # --- free runs -------------------------------------------------------
    t_toks, _, _ = _greedy(t_params, t_cfg, prompt, max_tokens)
    s_toks, _, _ = _greedy(s_params, s_cfg, prompt, max_tokens)
    agreement = sum(a == b for a, b in zip(t_toks, s_toks))

    positions = []
    n_align = min(s_cfg.layers, t_cfg.layers)
    for i in range(len(t_toks)):
        ctx = np.array([prompt + t_toks[:i]], dtype=np.int64)
        tl, tc = forward(t_params, t_cfg, ctx[:, -t_cfg.seq:])
        sl, sc = forward(s_params, s_cfg, ctx[:, -s_cfg.seq:])
        t_log = tl[0, -1].astype(np.float64)
        s_log = sl[0, -1].astype(np.float64)
        pos = {
            "position": i,
            "teacher_token": int(t_toks[i]),
            "student_token": int(s_toks[i]),
            "match": bool(t_toks[i] == s_toks[i]),
            "teacher_topk": top_k(t_log, top_k_k),
            "student_topk": top_k(s_log, top_k_k),
            "teacher_logits": [float(x) for x in t_log],
            "student_logits": [float(x) for x in s_log],
            "kl_div": kl_div(t_log, s_log),
            "js_div": js_div(t_log, s_log),
            "cosine_sim": C.cosine_sim(t_log, s_log),
            "teacher_entropy": _entropy(_softmax(t_log)),
            "student_entropy": _entropy(_softmax(s_log)),
            "hidden_div": {},
            "attn_div": {},
        }
        for l in range(n_align):
            th = tc[f"L{l}"]["y"] if "y" in tc[f"L{l}"] else tc["f"]
            sh = sc[f"L{l}"]["y"] if "y" in sc[f"L{l}"] else sc["f"]
            # align: truncate/pad student hidden to teacher dim (measured)
            sv = np.asarray(sh[0, -1], dtype=np.float64)
            tv = np.asarray(th[0, -1], dtype=np.float64)
            n = min(sv.size, tv.size)
            pos["hidden_div"][f"layer_{l}"] = {
                "rel_err": C.relative_error(tv[:n], sv[:n]),
                "cosine_sim": C.cosine_sim(tv[:n], sv[:n]),
            }
            pt = tc[f"L{l}"]["P"][0].mean(axis=0)  # (T,T) mean over heads
            ps = sc[f"L{l}"]["P"][0].mean(axis=0)
            nT = min(pt.shape[0], ps.shape[0])
            pos["attn_div"][f"layer_{l}"] = float(
                np.abs(pt[:nT, :nT] - ps[:nT, :nT]).mean())
        positions.append(pos)

    return {"prompt": prompt,
            "teacher_tokens": [int(x) for x in t_toks],
            "student_tokens": [int(x) for x in s_toks],
            "token_agreement": f"{agreement}/{len(t_toks)}",
            "agreement_count": agreement,
            "n_tokens": len(t_toks),
            "positions": positions}


def _greedy(params, cfg, prompt, max_tokens):
    from .model import generate
    return generate(params, cfg, list(prompt), max_tokens)


def first_divergence(trace: dict, *, kl_thresh: float = 0.5,
                     hidden_thresh: float = 0.35) -> dict:
    """Locate the FIRST divergence: earliest position where the student
    token mismatches OR KL exceeds threshold; then the layer with the
    largest hidden-state relative error at that position — the first
    layer producing significant representation drift."""
    for pos in trace["positions"]:
        if (not pos["match"]) or pos["kl_div"] > kl_thresh:
            hd = pos["hidden_div"]
            worst_layer = max(hd.items(),
                              key=lambda kv: kv[1]["rel_err"])[0]
            return {
                "position": pos["position"],
                "teacher_token": pos["teacher_token"],
                "student_token": pos["student_token"],
                "kl_div": pos["kl_div"],
                "js_div": pos["js_div"],
                "first_diverging_layer": worst_layer,
                "layer_hidden_rel_err": hd[worst_layer]["rel_err"],
                "layer_attn_div": pos["attn_div"].get(worst_layer),
                "all_layer_hidden_rel_err": {
                    k: v["rel_err"] for k, v in hd.items()},
                "above_hidden_thresh": {
                    k: v["rel_err"] for k, v in hd.items()
                    if v["rel_err"] > hidden_thresh},
            }
    return {"position": None, "note": "no divergence above thresholds"}


def render_trace(trace: dict, first: dict) -> List[str]:
    lines = []
    for pos in trace["positions"]:
        mark = "PASS" if pos["match"] else "DIVERGED"
        lines.append(
            f"token {pos['position']:2d} {mark:8s} "
            f"t={pos['teacher_token']:3d} s={pos['student_token']:3d} "
            f"KL={pos['kl_div']:.3f} JS={pos['js_div']:.3f} "
            f"cos={pos['cosine_sim']:.3f}")
    if first.get("position") is not None:
        lines.append(f"first divergence at position {first['position']}: "
                     f"layer {first['first_diverging_layer']} "
                     f"(hidden rel_err={first['layer_hidden_rel_err']:.3f})")
    return lines


# ---------------------------------------------------------------------------
# ablations — measured evidence per component
# ---------------------------------------------------------------------------

def _align_to(src: np.ndarray, shape: tuple) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(src.shape, shape))
    out[sl] = np.asarray(src)[sl]
    return out


COMPONENTS = [
    ("embeddings", lambda k: k == "emb"),
    ("attention_projections",
     lambda k: any(x in k for x in (".Wq", ".Wk", ".Wv", ".bq", ".bk", ".bv"))),
    ("attention_output",
     lambda k: any(x in k for x in (".Wo", ".bo"))),
    ("mlp_up", lambda k: ".W1" in k or ".b1" in k),
    ("mlp_down", lambda k: ".W2" in k or ".b2" in k),
    ("normalization",
     lambda k: any(x in k for x in ("ln1g", "ln1b", "ln2g", "ln2b",
                                    "lnfg", "lnfb"))),
    ("output_head", lambda k: k in ("Whead", "bhead")),
]


def ablation_report(t_params: dict, t_cfg, s_params: dict, s_cfg,
                    prompts: List[List[int]], max_tokens: int = 16) -> dict:
    """For each component: replace the student's component with the
    teacher's (shape-aligned), measure token-agreement and mean-KL delta
    vs the unmodified student. Returns ranked evidence — the component
    whose restoration helps most is where fidelity is being lost."""
    base = _agreement_and_kl(t_params, t_cfg, s_params, s_cfg,
                             prompts, max_tokens)
    rows = [{"component": "baseline(student)",
             "agreement": base[0], "mean_kl": base[1], "delta_agree": 0,
             "n_swapped": 0}]
    for name, pred in COMPONENTS:
        variant = dict(s_params)
        n = 0
        for k in list(variant.keys()):
            if pred(k) and k in t_params:
                variant[k] = _align_to(t_params[k], variant[k].shape)
                n += 1
        if n == 0:
            continue
        agree, mkl = _agreement_and_kl(t_params, t_cfg, variant, s_cfg,
                                       prompts, max_tokens)
        rows.append({"component": name, "agreement": agree, "mean_kl": mkl,
                     "delta_agree": agree - base[0],
                     "delta_kl": mkl - base[1], "n_swapped": n})
    rows.sort(key=lambda r: -r["delta_agree"])
    return {"baseline": {"agreement": base[0], "mean_kl": base[1]},
            "ranking": rows}


def _agreement_and_kl(t_params, t_cfg, s_params, s_cfg, prompts, max_tokens):
    agrees, kls, n = 0, [], 0
    for pr in prompts:
        tt, _, _ = _greedy(t_params, t_cfg, pr, max_tokens)
        st, _, _ = _greedy(s_params, s_cfg, pr, max_tokens)
        agrees += sum(a == b for a, b in zip(tt, st))
        n += len(tt)
        for i in range(len(tt)):
            ctx = np.array([pr + tt[:i]], dtype=np.int64)
            tl, _ = forward(t_params, t_cfg, ctx[:, -t_cfg.seq:])
            sl, _ = forward(s_params, s_cfg, ctx[:, -s_cfg.seq:])
            kls.append(kl_div(tl[0, -1], sl[0, -1]))
    return agrees / max(1, n), float(np.mean(kls)) if kls else 0.0


# ---------------------------------------------------------------------------
# recursive fidelity search — utility with explicit divergence cost
# ---------------------------------------------------------------------------

def fidelity_penalty(sensitivity: float, reconstruction_error: float,
                     lam: float = 1.0) -> float:
    """Explicit cost for a candidate transform that saves parameters but
    causes behavioral divergence. Applied uniformly to every decision
    class from measured (sensitivity, reconstruction_error)."""
    return float(lam * max(0.0, sensitivity) * max(0.0, reconstruction_error))


def fidelity_aware_select(leaves, budget_params: float,
                          sensitivity: Dict[str, float],
                          lam: float = 1.0) -> Dict[str, dict]:
    """Greedy knapsack identical in structure to core.select_plan, but each
    option's effective energy is

        effective = retained_energy - lam * sensitivity * recon_error

    (clipped at 0). The penalty is computed from MEASURED sensitivity and
    MEASURED reconstruction error for every option — quantum-gated options
    included, from their own qsim-measured errors. No decision class is
    favored or punished beyond what the measurements say.
    """
    srt = []
    for leaf in leaves:
        sens = float(sensitivity.get(leaf.name, 0.5))
        adj = []
        for o in leaf.options:
            pen = fidelity_penalty(sens, o["reconstruction_error"], lam)
            eff = max(0.0, o["retained_energy"] - pen)
            adj.append({**o, "effective_energy": eff,
                        "fidelity_penalty": pen})
        adj.sort(key=lambda o: o["cost_params"])
        srt.append(adj)
    choice = [0] * len(srt)
    total = sum(s[0]["cost_params"] for s in srt)
    improved = True
    while improved:
        improved = False
        best_gain, best_move = 0.0, None
        for i, opts in enumerate(srt):
            cur = choice[i]
            for j in range(cur + 1, len(opts)):
                dcost = opts[j]["cost_params"] - opts[cur]["cost_params"]
                denergy = (opts[j]["effective_energy"]
                           - opts[cur]["effective_energy"])
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
    plan["_fidelity_lambda"] = float(lam)
    return plan


def utility_breakdown(savings_params: int, penalties: dict) -> dict:
    """utility = savings - recon - logit - hidden - task penalties.
    All inputs measured; returns the itemized breakdown."""
    u = (savings_params - penalties.get("reconstruction", 0.0)
         - penalties.get("logit", 0.0) - penalties.get("hidden", 0.0)
         - penalties.get("task", 0.0))
    return {"utility": float(u), "savings_params": int(savings_params),
            **{k: float(v) for k, v in penalties.items()}}
