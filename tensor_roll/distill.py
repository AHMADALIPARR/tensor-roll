# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recursive distillation with the multi-objective loss.

L = l1*L_logits + l2*L_hidden + l3*L_attention + l4*L_embedding
  + l5*L_task + l6*L_reconstruction + l7*L_roll

All seven terms are real computed quantities (no placeholders):
  logits      : tau^2 * KL(softmax(teacher/tau) || softmax(student/tau))
  hidden      : MSE(student_final @ Ph, teacher_final), Ph frozen random proj
  attention   : MSE(mean-head attention probs), layer-aligned
  embedding   : MSE(student_emb @ Pe, teacher_emb), Pe frozen random proj
  task        : cross-entropy on the synthetic grammar
  reconstruction: MSE(student weights, projected-teacher init targets)
  roll        : MSE(achieved per-tensor energy, roll-plan energy) — monitored,
                gradient detached (documented; proxy grad via reconstruction)

Recursive distillation metrics (spec section 8) are computed per recursion
level: model -> layer group -> tensor -> sub-block, each reporting teacher
tensor, student tensor, relative error, cosine similarity, KL divergence,
retained energy.
"""

from __future__ import annotations

import numpy as np

from . import core as C
from .model import backward, ce_loss, forward, kl_loss

DEFAULT_LAMBDAS = {
    "logits": 1.0,
    "hidden": 0.3,
    "attention": 0.3,
    "embedding": 0.1,
    "task": 1.0,
    "reconstruction": 0.1,
    "roll": 0.1,
}


def _align(a: np.ndarray, shape: tuple) -> np.ndarray:
    """Shape alignment for comparison only: truncate or zero-pad. This is a
    measurement convenience, documented here — not a claim about the model."""
    a = np.asarray(a, dtype=np.float64)
    out = np.zeros(shape, dtype=np.float64)
    sl = tuple(slice(0, min(x, y)) for x, y in zip(a.shape, shape))
    out[sl] = a[sl]
    return out


def distill_step(s_params: dict, s_cfg, t_params: dict, t_cfg,
                 ids: np.ndarray, *, lambdas=None, tau: float = 2.0,
                 proj_h=None, proj_e=None, recon_targets=None,
                 roll_plan_energies=None, achieved_energies=None):
    """One recursive-distillation step. Returns (total_loss, terms, grads)."""
    lambdas = {**DEFAULT_LAMBDAS, **(lambdas or {})}
    B = ids.shape[0]
    inputs, targets = ids[:, :-1], ids[:, 1:]

    t_logits, t_cache = forward(t_params, t_cfg, inputs)
    s_logits, s_cache = forward(s_params, s_cfg, inputs)
    terms = {}

    # L_task + L_logits -> dlogits
    terms["task"], dl_task = ce_loss(s_logits, targets)
    terms["logits"], dl_kl = kl_loss(s_logits, t_logits, tau)
    dlogits = lambdas["task"] * dl_task + lambdas["logits"] * dl_kl

    # L_attention: mean-over-heads, layer-aligned (first min(L) layers)
    d_attn = {}
    n_align = min(s_cfg.layers, t_cfg.layers)
    attn_num = 0.0
    for l in range(n_align):
        ps = s_cache[f"L{l}"]["P"].mean(axis=1)  # (B,T,T)
        pt = t_cache[f"L{l}"]["P"].mean(axis=1)
        diff = ps - pt
        terms[f"attention_l{l}"] = float((diff ** 2).mean())
        attn_num += terms[f"attention_l{l}"]
        dmean = (2.0 * diff / diff.size) * lambdas["attention"] / max(n_align, 1)
        d_attn[l] = np.broadcast_to(
            dmean[:, None, :, :], s_cache[f"L{l}"]["P"].shape).copy()
    terms["attention"] = attn_num / max(n_align, 1)

    # L_hidden: final hidden, frozen random projection to teacher dim
    d_hiddens = {}
    if proj_h is not None:
        fs = s_cache["f"]                      # (B,T,ds)
        ft = t_cache["f"]                      # (B,T,dt)
        pred = fs @ proj_h
        diff = pred - ft
        terms["hidden"] = float((diff ** 2).mean())
        d_hiddens[s_cfg.layers - 1] = (
            (2.0 * diff / diff.size) @ proj_h.T) * lambdas["hidden"]
    else:
        terms["hidden"] = 0.0

    grads = backward(s_params, s_cfg, s_cache, dlogits,
                     d_hiddens=d_hiddens or None, d_attn=d_attn or None)

    # L_embedding
    if proj_e is not None:
        es = s_params["emb"].astype(np.float64)
        et = t_params["emb"].astype(np.float64)
        diff = es @ proj_e - et
        terms["embedding"] = float((diff ** 2).mean())
        grads["emb"] = grads["emb"] + (
            (2.0 * diff / diff.size) @ proj_e.T * lambdas["embedding"]
        ).astype(np.float32)
    else:
        terms["embedding"] = 0.0

    # L_reconstruction: stay near the projected-teacher init targets
    if recon_targets:
        rn, rv = 0.0, 0
        for k, tgt in recon_targets.items():
            if k in grads and grads[k].shape == tgt.shape:
                w = s_params[k].astype(np.float64)
                diff = w - tgt.astype(np.float64)
                rn += float((diff ** 2).mean())
                rv += 1
                grads[k] = grads[k] + (
                    2.0 * diff / diff.size * lambdas["reconstruction"]).astype(np.float32)
        terms["reconstruction"] = rn / max(rv, 1)
    else:
        terms["reconstruction"] = 0.0

    # L_roll: achieved vs planned per-tensor retained energy (monitored)
    if roll_plan_energies and achieved_energies:
        vals = [(achieved_energies[k] - roll_plan_energies[k]) ** 2
                for k in roll_plan_energies if k in achieved_energies]
        terms["roll"] = float(np.mean(vals)) if vals else 0.0
    else:
        terms["roll"] = 0.0

    total = sum(lambdas[k] * terms[k] for k in
                ("logits", "hidden", "attention", "embedding", "task",
                 "reconstruction", "roll"))
    terms["total"] = float(total)
    return float(total), terms, grads


def _kl_vec(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    p = p_logits.astype(np.float64).ravel()
    q = q_logits.astype(np.float64).ravel()
    p = p - p.max(); q = q - q.max()
    lp = p - np.log(np.exp(p).sum()); lq = q - np.log(np.exp(q).sum())
    ep = np.exp(lp)
    return float((ep * (lp - lq)).sum())


def recursive_metrics(t_params: dict, s_params: dict) -> dict:
    """Per-recursion-level teacher/student comparison.

    Levels: model -> layer_group -> tensor -> subblock. Student tensors are
    shape-aligned to the teacher by truncate/zero-pad (comparison only).
    Each level reports: rel_err, cosine_sim, kl_div, retained_energy,
    teacher_params, student_params.
    """
    out = {}

    def pack(params, keys):
        return np.concatenate([np.asarray(params[k]).ravel() for k in keys])

    def level(name, t_keys, s_keys):
        tv = pack(t_params, t_keys).astype(np.float64)
        sv_raw = pack(s_params, s_keys).astype(np.float64)
        sv = _align(sv_raw, tv.shape)
        out[name] = {
            "teacher_params": int(tv.size),
            "student_params": int(sv_raw.size),
            "rel_err": C.relative_error(tv, sv),
            "cosine_sim": C.cosine_sim(tv, sv),
            "kl_div": _kl_vec(tv / (np.abs(tv).max() + 1e-9),
                              sv / (np.abs(sv).max() + 1e-9)),
            "retained_energy": float((sv ** 2).sum() / ((tv ** 2).sum() + 1e-300)),
        }

    t_keys = sorted(t_params.keys())
    s_keys = sorted(s_params.keys())
    level("model", t_keys, s_keys)

    # layer groups
    layers = sorted({k.split(".")[0] for k in t_keys if k.startswith("L")})
    for i, lg in enumerate(layers):
        tk = [k for k in t_keys if k.startswith(lg + ".")]
        sk = [k for k in s_keys if k.startswith(lg + ".")]
        if tk and sk:
            level(f"layer_group_{i}", tk, sk)

    # tensors (sample up to 12 for readability; full data in JSON)
    for k in t_keys[:12]:
        sk = [x for x in s_keys if x == k]
        if sk:
            level(f"tensor:{k}", [k], sk)

    # sub-blocks of the largest teacher tensor
    big = max(t_keys, key=lambda k: t_params[k].size)
    W = np.asarray(t_params[big], dtype=np.float64).ravel()
    sm = [x for x in s_keys if x == big]
    if sm:
        S = np.asarray(s_params[sm[0]], dtype=np.float64).ravel()
        for b, (ws, ss) in enumerate(zip(np.array_split(W, 4),
                                         np.array_split(S, 4))):
            sa = _align(ss, ws.shape)
            out[f"subblock:{big}[{b}]"] = {
                "teacher_params": int(ws.size),
                "student_params": int(ss.size),
                "rel_err": C.relative_error(ws, sa),
                "cosine_sim": C.cosine_sim(ws, sa),
                "kl_div": _kl_vec(ws / (np.abs(ws).max() + 1e-9),
                                  sa / (np.abs(sa).max() + 1e-9)),
                "retained_energy": float((sa ** 2).sum() / ((ws ** 2).sum() + 1e-300)),
            }
    return out
