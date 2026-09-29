# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quantization formats: FP16 / BF16 / INT8 / INT4.

All quantizers are implemented here from first principles (no external
quantizer libraries). Every quantize() returns (payload, meta) and every
meta carries enough information for an exact dequantize() round-trip.
"""

from __future__ import annotations

import numpy as np


def to_fp16(W: np.ndarray):
    W = np.asarray(W, dtype=np.float32)
    return W.astype(np.float16), {"fmt": "fp16"}


def from_fp16(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float32)


def to_bf16(W: np.ndarray):
    """Round-to-nearest-even float32 -> bfloat16 (stored as uint16)."""
    a = np.asarray(W, dtype=np.float32)
    i = a.view(np.int32).copy()
    finite = np.isfinite(a)
    lsb = (i >> 16) & 1
    i[finite] += (0x7FFF + lsb[finite]).astype(np.int32)
    return (i >> 16).astype(np.uint16), {"fmt": "bf16"}


def from_bf16(q: np.ndarray) -> np.ndarray:
    return (np.asarray(q, dtype=np.int32) << 16).view(np.float32)


def quantize_int8(W: np.ndarray):
    """Asymmetric per-tensor INT8."""
    W = np.asarray(W, dtype=np.float64)
    wmin, wmax = float(W.min()), float(W.max())
    if wmax == wmin:
        return np.zeros(W.shape, dtype=np.int8), {
            "fmt": "int8", "scale": 1.0, "zero": 0.0}
    scale = (wmax - wmin) / 255.0
    q = np.clip(np.round((W - wmin) / scale), 0, 255).astype(np.int64) - 128
    return q.astype(np.int8), {"fmt": "int8", "scale": scale, "zero": wmin}


def dequantize_int8(q: np.ndarray, meta: dict) -> np.ndarray:
    return (q.astype(np.float64) + 128.0) * meta["scale"] + meta["zero"]


def quantize_int4(W: np.ndarray, group: int = 32):
    """Group-wise asymmetric INT4, two nibbles packed per byte (low nibble
    first). Groups run along the last axis; ragged tail is zero-padded and
    the pad is recorded so dequantize() restores the exact shape."""
    W = np.asarray(W, dtype=np.float64)
    shape = list(W.shape)
    last = shape[-1]
    rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    flat = W.reshape(rows, last)
    pad = (-last) % group
    if pad:
        flat = np.concatenate([flat, np.zeros((rows, pad))], axis=1)
    G = flat.reshape(-1, group)
    gmin = G.min(axis=1)
    gmax = G.max(axis=1)
    scale = np.maximum(gmax - gmin, 1e-300) / 15.0
    q = np.clip(np.round((G - gmin[:, None]) / scale[:, None]), 0, 15).astype(np.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    meta = {"fmt": "int4", "group": group, "shape": shape,
            "scales": scale.astype(np.float64), "zeros": gmin.astype(np.float64),
            "rows": rows, "last": last}
    return packed, meta


def dequantize_int4(packed: np.ndarray, meta: dict) -> np.ndarray:
    group, rows, last = meta["group"], meta["rows"], meta["last"]
    p = np.asarray(packed).reshape(-1, group // 2)
    q = np.empty((p.shape[0], group), dtype=np.float64)
    q[:, 0::2] = p & 0x0F
    q[:, 1::2] = (p >> 4) & 0x0F
    G = q * np.asarray(meta["scales"])[:, None] + np.asarray(meta["zeros"])[:, None]
    flat = G.reshape(rows, -1)[:, :last]
    return flat.reshape(meta["shape"]).astype(np.float32)


def quantize_tensor(W: np.ndarray, fmt: str):
    """Dispatch: fmt in {fp32, fp16, bf16, int8, int4}."""
    if fmt == "fp32":
        return np.asarray(W, dtype=np.float32), {"fmt": "fp32"}
    if fmt == "fp16":
        return to_fp16(W)
    if fmt == "bf16":
        return to_bf16(W)
    if fmt == "int8":
        return quantize_int8(W)
    if fmt == "int4":
        return quantize_int4(W)
    raise ValueError(f"unknown format {fmt!r}")


def dequantize_tensor(q: np.ndarray, meta: dict) -> np.ndarray:
    fmt = meta["fmt"]
    if fmt == "fp32":
        return np.asarray(q, dtype=np.float32)
    if fmt == "fp16":
        return from_fp16(q)
    if fmt == "bf16":
        return from_bf16(q)
    if fmt == "int8":
        return dequantize_int8(q, meta).astype(np.float32)
    if fmt == "int4":
        return dequantize_int4(q, meta)
    raise ValueError(f"unknown format {fmt!r}")


def payload_nbytes(q: np.ndarray, fmt: str) -> int:
    return int(np.asarray(q).nbytes)


# ---------------------------------------------------------------------------
# INT4 failure resolution (F6): per-tensor error instrumentation, advanced
# schemes, mixed precision, and quantization-aware recovery.
# ---------------------------------------------------------------------------

def quantize_int4_per_tensor(W: np.ndarray):
    """Whole-tensor asymmetric INT4: ONE scale/zero over the entire tensor."""
    W = np.asarray(W, dtype=np.float64)
    shape = list(W.shape)
    flat = W.ravel()
    gmin, gmax = float(flat.min()), float(flat.max())
    scale = max(gmax - gmin, 1e-300) / 15.0
    q = np.clip(np.round((flat - gmin) / scale), 0, 15).astype(np.uint8)
    if q.size % 2:
        q = np.concatenate([q, np.zeros(1, dtype=np.uint8)])
    packed = (q[0::2] | (q[1::2] << 4)).astype(np.uint8)
    meta = {"fmt": "int4", "scheme": "per-tensor", "shape": shape,
            "scales": np.array([scale]), "zeros": np.array([gmin]),
            "size": flat.size}
    return packed, meta


def dequantize_int4_per_tensor(packed: np.ndarray, meta: dict) -> np.ndarray:
    q = np.asarray(packed)
    flat = np.empty(q.size * 2, dtype=np.float64)
    flat[0::2] = q & 0x0F
    flat[1::2] = (q >> 4) & 0x0F
    flat = flat[:meta["size"]]
    W = flat * float(meta["scales"][0]) + float(meta["zeros"][0])
    return W.reshape(meta["shape"]).astype(np.float32)


def quantize_int4_per_channel(W: np.ndarray):
    """Per-channel (per-row) asymmetric INT4: one scale/zero per row instead
    of per group-of-32. Returns (packed, meta) with the same nibble layout
    as quantize_int4 so dequantize_int4-style unpacking applies."""
    W = np.asarray(W, dtype=np.float64)
    shape = list(W.shape)
    last = shape[-1]
    rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    flat = W.reshape(rows, last)
    gmin = flat.min(axis=1)
    gmax = flat.max(axis=1)
    scale = np.maximum(gmax - gmin, 1e-300) / 15.0
    q = np.clip(np.round((flat - gmin[:, None]) / scale[:, None]), 0, 15)
    q = q.astype(np.uint8)
    # pad last dim to even for nibble packing
    if last % 2:
        q = np.concatenate([q, np.zeros((rows, 1), dtype=np.uint8)], axis=1)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    meta = {"fmt": "int4", "scheme": "per-channel", "shape": shape,
            "scales": scale.astype(np.float64),
            "zeros": gmin.astype(np.float64),
            "rows": rows, "last": last}
    return packed, meta


def dequantize_int4_per_channel(packed: np.ndarray, meta: dict) -> np.ndarray:
    rows, last = meta["rows"], meta["last"]
    p = np.asarray(packed).reshape(rows, -1)
    q = np.empty((rows, p.shape[1] * 2), dtype=np.float64)
    q[:, 0::2] = p & 0x0F
    q[:, 1::2] = (p >> 4) & 0x0F
    q = q[:, :last]
    W = q * np.asarray(meta["scales"])[:, None] + np.asarray(meta["zeros"])[:, None]
    return W.reshape(meta["shape"]).astype(np.float32)


def search_group_size(W: np.ndarray,
                      groups: tuple = (16, 32, 64, 128)) -> dict:
    """Search the INT4 group size by MEASURED reconstruction MSE — the
    winner is the argmin of measured error, never a hard-coded default."""
    W = np.asarray(W, dtype=np.float64)
    results = []
    for g in groups:
        if g < 2:
            continue
        q, meta = quantize_int4(W, group=g)
        back = dequantize_int4(q, meta).astype(np.float64)
        err = float(np.mean((W - back) ** 2))
        results.append({"group": g, "mse": err,
                        "bytes": int(np.asarray(q).nbytes)})
    results.sort(key=lambda r: r["mse"])
    return {"results": results, "best_group": results[0]["group"],
            "best_mse": results[0]["mse"]}


def quant_error_map(W: np.ndarray, Wd: np.ndarray) -> dict:
    """Instrument quantization error per tensor. All values measured."""
    W = np.asarray(W, dtype=np.float64)
    Wd = np.asarray(Wd, dtype=np.float64)
    err = W - Wd
    flat = err.ravel()
    sd = float(flat.std()) if flat.size else 0.0
    per_channel = []
    if W.ndim == 2:
        for r in range(W.shape[0]):
            e = err[r]
            per_channel.append(float(np.mean(e ** 2)))
    # per-group error at group=32 along the last axis
    per_group = []
    if W.ndim >= 1 and W.shape[-1] >= 32:
        G = err.reshape(-1, 32)
        per_group = [float(np.mean(g ** 2)) for g in G]
    denom = float(np.linalg.norm(W))
    return {
        "min": float(err.min()) if err.size else 0.0,
        "max": float(err.max()) if err.size else 0.0,
        "mean": float(err.mean()) if err.size else 0.0,
        "variance": float(err.var()) if err.size else 0.0,
        "std": sd,
        "outlier_frac": float(np.mean(np.abs(flat) > 3 * sd)) if sd > 0 else 0.0,
        "mse": float(np.mean(flat ** 2)) if flat.size else 0.0,
        "cosine_sim": float(np.dot(W.ravel(), Wd.ravel())
                            / (np.linalg.norm(W.ravel())
                               * np.linalg.norm(Wd.ravel()) + 1e-300)),
        "rel_err": float(np.linalg.norm(err) / (denom + 1e-300)),
        "per_channel_mse": per_channel,
        "per_group_mse": per_group,
    }


def int4_scheme_report(W: np.ndarray) -> dict:
    """Benchmark per-tensor / per-channel / group-wise INT4 on one tensor.

    Returns measured MSE + byte size for each scheme; the caller picks the
    winner from measurements."""
    W = np.asarray(W, dtype=np.float64)
    out = {}
    qt, mt = quantize_int4_per_tensor(W)
    backt = dequantize_int4_per_tensor(qt, mt).astype(np.float64)
    out["per-tensor"] = {"mse": float(np.mean((W - backt) ** 2)),
                         "bytes": int(np.asarray(qt).nbytes),
                         "meta_bytes": 16}
    qp, mp = quantize_int4_per_channel(W)
    backp = dequantize_int4_per_channel(qp, mp).astype(np.float64)
    out["per-channel"] = {"mse": float(np.mean((W - backp) ** 2)),
                          "bytes": int(np.asarray(qp).nbytes),
                          "meta_bytes": int(mp["scales"].nbytes
                                            + mp["zeros"].nbytes)}
    gs = search_group_size(W)
    out["group-wise"] = {"mse": gs["best_mse"],
                         "bytes": next(r["bytes"] for r in gs["results"]
                                       if r["group"] == gs["best_group"]),
                         "best_group": gs["best_group"],
                         "all": gs["results"]}
    # int8 reference for the mixed-precision gain computation
    qi, mi = quantize_int8(W)
    backi = dequantize_int8(qi, mi).astype(np.float64)
    out["int8"] = {"mse": float(np.mean((W - backi) ** 2)),
                   "bytes": int(np.asarray(qi).nbytes)}
    return out


def mixed_precision_plan(params: dict, size_budget_bytes: int) -> dict:
    """Assign INT4 vs INT8 per tensor under a physical size budget.

    Tensors where INT8 buys the most error reduction per extra byte stay
    INT8; the rest go INT4 (best measured scheme per tensor). Sensitive
    tensors are thereby protected by measurement, not by fiat.
    """
    rows = []
    for k, v in params.items():
        v = np.asarray(v)
        if v.ndim != 2 or v.size < 64:
            rows.append({"tensor": k, "fmt": "fp32",
                         "bytes": int(v.nbytes), "gain_per_byte": 0.0,
                         "int4_mse": 0.0, "int8_mse": 0.0})
            continue
        rep = int4_scheme_report(v)
        best4 = min(rep["per-tensor"], rep["per-channel"],
                    rep["group-wise"], key=lambda r: r["mse"])
        b4 = best4["bytes"] + best4.get("meta_bytes", 0)
        b8 = rep["int8"]["bytes"]
        gain = max(0.0, best4["mse"] - rep["int8"]["mse"])
        gpb = gain / max(1, b8 - b4)
        rows.append({"tensor": k, "fmt": "int4", "bytes": b4,
                     "gain_per_byte": gpb, "int4_mse": best4["mse"],
                     "int8_mse": rep["int8"]["mse"],
                     "int4_scheme": "per-tensor"
                     if best4 is rep["per-tensor"] else
                     ("per-channel" if best4 is rep["per-channel"]
                      else f"group-{rep['group-wise']['best_group']}")})
    # start all-int4, upgrade highest gain/byte to int8 while budget allows
    total4 = sum(r["bytes"] for r in rows)
    cands = sorted([r for r in rows if r["fmt"] == "int4"
                    and r["gain_per_byte"] > 0],
                   key=lambda r: -r["gain_per_byte"])
    plan, used = {}, total4
    for r in rows:
        plan[r["tensor"]] = "int4"
    for r in cands:
        v = np.asarray(params[r["tensor"]])
        qi, _ = quantize_int8(v)
        b8 = int(np.asarray(qi).nbytes)
        if used - r["bytes"] + b8 <= size_budget_bytes:
            plan[r["tensor"]] = "int8"
            used = used - r["bytes"] + b8
    return {"plan": plan, "total_bytes": used,
            "budget_bytes": int(size_budget_bytes),
            "within_budget": used <= size_budget_bytes,
            "n_int8": sum(1 for v in plan.values() if v == "int8"),
            "n_int4": sum(1 for v in plan.values() if v == "int4"),
            "rows": rows}


def dequantize_dispatch(q: np.ndarray, meta: dict) -> np.ndarray:
    """Dequantize honoring per-tensor / per-channel int4 metadata."""
    if meta.get("fmt") == "int4" and meta.get("scheme") == "per-tensor":
        return dequantize_int4_per_tensor(q, meta)
    if meta.get("fmt") == "int4" and meta.get("scheme") == "per-channel":
        return dequantize_int4_per_channel(q, meta)
    return dequantize_tensor(q, meta)


def recovery_train(s_qparams: dict, s_cfg, t_params: dict, t_cfg,
                   *, steps: int = 30, batch: int = 64, lr: float = 1e-3,
                   seed: int = 0) -> dict:
    """Quantization-aware recovery (spec section 9):

        INT4-dequantized student -> distillation (task + logit KL)
        vs the teacher -> requantize INT4 -> evaluate.

    Returns measured ppl before/after and the gap delta. Improvement is
    reported ONLY if measured; otherwise the (non-)result stands.
    """
    from .model import (Adam, backward, ce_loss, eval_metrics, forward,
                        gen_data, kl_loss)
    rng = np.random.default_rng(seed)
    # start from the dequantized INT4 weights (float working copy)
    params = {k: v.astype(np.float32) for k, v in s_qparams.items()}
    before = eval_metrics(params, s_cfg, batches=4, batch=batch, seed=911)
    opt = Adam(params, lr=lr)
    for _ in range(steps):
        ids = gen_data(rng, batch, s_cfg)
        tl, _ = forward(t_params, t_cfg, ids[:, :-1])
        sl, cache = forward(params, s_cfg, ids[:, :-1])
        lt, dl_t = ce_loss(sl, ids[:, 1:])
        lk, dl_k = kl_loss(sl, tl, tau=2.0)
        grads = backward(params, s_cfg, cache, dl_t + 0.5 * dl_k)
        opt.step(params, grads)
    after = eval_metrics(params, s_cfg, batches=4, batch=batch, seed=911)
    # requantize and measure the real INT4 artifact quality
    re_q = {}
    for k, v in params.items():
        if v.ndim == 2 and v.size >= 64:
            q, m = quantize_int4(v)
            re_q[k] = dequantize_int4(q, m).astype(np.float32)
        else:
            re_q[k] = v
    re_eval = eval_metrics(re_q, s_cfg, batches=4, batch=batch, seed=911)
    return {
        "steps": steps,
        "ppl_before": float(np.exp(before["loss"])),
        "ppl_after_train": float(np.exp(after["loss"])),
        "ppl_after_requant": float(np.exp(re_eval["loss"])),
        "gap_before": float(np.exp(before["loss"]) - np.exp(after["loss"])),
        "gap_delta": float(np.exp(before["loss"])
                           - np.exp(re_eval["loss"])),
        "improved": bool(np.exp(re_eval["loss"]) < np.exp(before["loss"])),
    }
