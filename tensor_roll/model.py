# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tiny transformer implemented from scratch in NumPy (forward + backward).

This is the empirical validation vehicle: a small teacher is trained on
synthetic data, then the ENTIRE Tensor Roll pipeline (roll -> compress ->
distill -> finetune -> quantize -> .trq -> sandboxed inference) runs on it.
Same code path as the 30B scale-up; only the config and data change.
No torch / transformers / llama.cpp anywhere in this file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

DTYPE = np.float32
EPS = 1e-5


@dataclass
class Config:
    vocab: int = 32
    d: int = 48
    heads: int = 3
    layers: int = 2
    dff: int = 96
    seq: int = 16

    @property
    def dh(self) -> int:
        assert self.d % self.heads == 0
        return self.d // self.heads

    def to_dict(self):
        return {"vocab": self.vocab, "d": self.d, "heads": self.heads,
                "layers": self.layers, "dff": self.dff, "seq": self.seq}

    @staticmethod
    def from_dict(d):
        return Config(**d)


def count_params(params: dict) -> int:
    return int(sum(v.size for v in params.values()))


def init_params(rng: np.random.Generator, cfg: Config) -> dict:
    def xavier(shape):
        fan = shape[0] + shape[1] if len(shape) == 2 else shape[0]
        lim = np.sqrt(6.0 / fan)
        return rng.uniform(-lim, lim, size=shape).astype(DTYPE)

    p = {"emb": xavier((cfg.vocab, cfg.d))}
    for l in range(cfg.layers):
        p[f"L{l}.ln1g"] = np.ones(cfg.d, dtype=DTYPE)
        p[f"L{l}.ln1b"] = np.zeros(cfg.d, dtype=DTYPE)
        for nm in ("Wq", "Wk", "Wv", "Wo"):
            p[f"L{l}.{nm}"] = xavier((cfg.d, cfg.d))
            p[f"L{l}.b{nm[1:]}"] = np.zeros(cfg.d, dtype=DTYPE)
        p[f"L{l}.ln2g"] = np.ones(cfg.d, dtype=DTYPE)
        p[f"L{l}.ln2b"] = np.zeros(cfg.d, dtype=DTYPE)
        p[f"L{l}.W1"] = xavier((cfg.d, cfg.dff))
        p[f"L{l}.b1"] = np.zeros(cfg.dff, dtype=DTYPE)
        p[f"L{l}.W2"] = xavier((cfg.dff, cfg.d))
        p[f"L{l}.b2"] = np.zeros(cfg.d, dtype=DTYPE)
    p["lnfg"] = np.ones(cfg.d, dtype=DTYPE)
    p["lnfb"] = np.zeros(cfg.d, dtype=DTYPE)
    p["Whead"] = xavier((cfg.d, cfg.vocab))
    p["bhead"] = np.zeros(cfg.vocab, dtype=DTYPE)
    return p


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def _ln_forward(x, g, b):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std = np.sqrt(var + EPS)
    xh = (x - mean) / std
    return g * xh + b, (x, xh, std, g)


def _ln_backward(dy, cache):
    x, xh, std, g = cache
    n_feat = x.shape[-1]
    dxh = dy * g
    dvar = np.sum(dxh * (x - x.mean(axis=-1, keepdims=True))
                  * -0.5 * (std ** -3), axis=-1, keepdims=True)
    dmean = (np.sum(dxh * -1.0 / std, axis=-1, keepdims=True)
             + dvar * np.mean(-2.0 * (x - x.mean(axis=-1, keepdims=True)), axis=-1, keepdims=True))
    dx = dxh / std + dvar * 2.0 * (x - x.mean(axis=-1, keepdims=True)) / n_feat + dmean / n_feat
    dg = np.sum(dy * xh, axis=(0, 1))
    db = np.sum(dy, axis=(0, 1))
    return dx, dg, db


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def forward(params: dict, cfg: Config, ids: np.ndarray):
    """Returns (logits, cache). ids: (B,T) int."""
    B, T = ids.shape
    H, dh = cfg.heads, cfg.dh
    cache: dict = {"ids": ids}
    x = params["emb"][ids]  # (B,T,d)

    cmask = np.zeros((T, T), dtype=DTYPE)
    cmask[np.triu_indices(T, k=1)] = -1e9

    for l in range(cfg.layers):
        c: dict = {"x_prev": x}
        h, c["ln1"] = _ln_forward(x, params[f"L{l}.ln1g"], params[f"L{l}.ln1b"])
        Q = (h @ params[f"L{l}.Wq"] + params[f"L{l}.bq"]).reshape(B, T, H, dh)
        K = (h @ params[f"L{l}.Wk"] + params[f"L{l}.bk"]).reshape(B, T, H, dh)
        V = (h @ params[f"L{l}.Wv"] + params[f"L{l}.bv"]).reshape(B, T, H, dh)
        s = np.einsum("bthd,bshd->bhts", Q, K) / np.sqrt(dh) + cmask[None, None, :, :]
        P = _softmax(s)
        A = np.einsum("bhts,bshd->bthd", P, V).reshape(B, T, cfg.d)
        O = A @ params[f"L{l}.Wo"] + params[f"L{l}.bo"]
        y = x + O
        z, c["ln2"] = _ln_forward(y, params[f"L{l}.ln2g"], params[f"L{l}.ln2b"])
        u = z @ params[f"L{l}.W1"] + params[f"L{l}.b1"]
        a = np.maximum(u, 0)
        m_ = a @ params[f"L{l}.W2"] + params[f"L{l}.b2"]
        x = y + m_
        c.update({"h": h, "Q": Q, "K": K, "V": V, "P": P, "A": A,
                  "y": y, "z": z, "u": u, "a": a})
        cache[f"L{l}"] = c
    f, cache["lnf"] = _ln_forward(x, params["lnfg"], params["lnfb"])
    cache["f"] = f
    cache["x_pre_lnf"] = x
    logits = f @ params["Whead"] + params["bhead"]
    return logits, cache


def backward(params: dict, cfg: Config, cache: dict, dlogits: np.ndarray,
             d_hiddens: dict | None = None, d_attn: dict | None = None) -> dict:
    """dlogits: (B,T,V). d_hiddens: {layer_idx: (B,T,d) upstream grad added
    at that layer's output} for hidden-state matching. d_attn: {layer_idx:
    (B,H,T,T) upstream grad added to attention probs} for attention matching."""
    B, T = cache["ids"].shape
    grads = {}
    f = cache["f"]
    grads["Whead"] = np.einsum("btd,btv->dv", f, dlogits)
    grads["bhead"] = dlogits.sum(axis=(0, 1))
    df = dlogits @ params["Whead"].T
    dx, dg, db = _ln_backward(df, cache["lnf"])
    grads["lnfg"], grads["lnfb"] = dg, db

    for l in reversed(range(cfg.layers)):
        c = cache[f"L{l}"]
        if d_hiddens and l in d_hiddens:
            dx = dx + d_hiddens[l]
        # x = y + m ; m = a@W2+b2
        dy = dx.copy()
        dm = dx
        A = c["A"]
        grads[f"L{l}.W2"] = np.einsum("bti,btj->ij", c["a"], dm)
        grads[f"L{l}.b2"] = dm.sum(axis=(0, 1))
        da = dm @ params[f"L{l}.W2"].T
        du = da * (c["u"] > 0)
        grads[f"L{l}.W1"] = np.einsum("bti,btj->ij", c["z"], du)
        grads[f"L{l}.b1"] = du.sum(axis=(0, 1))
        dz = du @ params[f"L{l}.W1"].T
        dy2, dg2, db2 = _ln_backward(dz, c["ln2"])
        grads[f"L{l}.ln2g"], grads[f"L{l}.ln2b"] = dg2, db2
        dy = dy + dy2
        # y = x_prev + O ; O = A@Wo+bo
        dx_prev = dy.copy()
        dO = dy
        grads[f"L{l}.Wo"] = np.einsum("bti,btj->ij", A, dO)
        grads[f"L{l}.bo"] = dO.sum(axis=(0, 1))
        dA = (dO @ params[f"L{l}.Wo"].T).reshape(B, T, cfg.heads, cfg.dh)
        Q, K, V, P = c["Q"], c["K"], c["V"], c["P"]
        dP = np.einsum("bthd,bshd->bhts", dA, V)
        if d_attn is not None and l in d_attn:
            dP = dP + d_attn[l]
        dV = np.einsum("bhts,bthd->bshd", P, dA)
        dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True))
        inv = 1.0 / np.sqrt(cfg.dh)
        dQ = np.einsum("bhts,bshd->bthd", dS, K) * inv
        dK = np.einsum("bhts,bthd->bshd", dS, Q) * inv
        h = c["h"]
        for nm, dM in (("Wq", dQ), ("Wk", dK), ("Wv", dV)):
            dM2 = dM.reshape(B, T, cfg.d)
            grads[f"L{l}.{nm}"] = np.einsum("bti,btj->ij", h, dM2)
            grads[f"L{l}.b{nm[1:]}"] = dM2.sum(axis=(0, 1))
            dx_prev = dx_prev + dM2 @ params[f"L{l}.{nm}"].T
        dh, dg1, db1 = _ln_backward(dx_prev, c["ln1"])
        grads[f"L{l}.ln1g"], grads[f"L{l}.ln1b"] = dg1, db1
        dx = dh  # grad w.r.t. this layer's input = next (previous) layer's output grad
    dEmb = np.zeros_like(params["emb"])
    np.add.at(dEmb, cache["ids"], dx)
    grads["emb"] = dEmb
    return grads


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def ce_loss(logits: np.ndarray, targets: np.ndarray):
    """Mean softmax cross-entropy. Returns (loss, dlogits)."""
    B, T, V = logits.shape
    x = logits.reshape(-1, V).astype(np.float64)
    t = targets.reshape(-1)
    x = x - x.max(axis=1, keepdims=True)
    logp = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    loss = float(-logp[np.arange(t.size), t].mean())
    d = np.exp(logp)
    d[np.arange(t.size), t] -= 1.0
    d /= t.size
    return loss, d.reshape(B, T, V).astype(DTYPE)


def kl_loss(s_logits: np.ndarray, t_logits: np.ndarray, tau: float = 2.0):
    """tau^2 * mean KL(softmax(t/tau) || softmax(s/tau)). Returns (loss, dlogits_s)."""
    B, T, V = s_logits.shape
    s = (s_logits / tau).reshape(-1, V).astype(np.float64)
    t = (t_logits / tau).reshape(-1, V).astype(np.float64)
    s = s - s.max(1, keepdims=True)
    t = t - t.max(1, keepdims=True)
    log_q = s - np.log(np.exp(s).sum(1, keepdims=True))
    log_p = t - np.log(np.exp(t).sum(1, keepdims=True))
    p = np.exp(log_p)
    kl = (p * (log_p - log_q)).sum(1).mean()
    loss = float(tau * tau * kl)
    d = (np.exp(log_q) - p) * tau / p.shape[0]
    return loss, d.reshape(B, T, V).astype(DTYPE)


def kl_divergence_mean(t_logits: np.ndarray, s_logits: np.ndarray) -> float:
    """Mean KL(teacher || student) at tau=1 — teacher-divergence metric."""
    t = t_logits.reshape(-1, t_logits.shape[-1]).astype(np.float64)
    s = s_logits.reshape(-1, s_logits.shape[-1]).astype(np.float64)
    t = t - t.max(1, keepdims=True)
    s = s - s.max(1, keepdims=True)
    log_p = t - np.log(np.exp(t).sum(1, keepdims=True))
    log_q = s - np.log(np.exp(s).sum(1, keepdims=True))
    p = np.exp(log_p)
    return float((p * (log_p - log_q)).sum(1).mean())


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, params: dict, lr: float = 3e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            g = grads[k].astype(np.float64)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = (params[k].astype(np.float64)
                         - self.lr * mh / (np.sqrt(vh) + self.eps)).astype(DTYPE)


# ---------------------------------------------------------------------------
# Synthetic data + training
# ---------------------------------------------------------------------------

def gen_data(rng: np.random.Generator, n: int, cfg: Config) -> np.ndarray:
    """Deterministic grammar: x[t] = (3*x[t-1] + 5*x[t-2] + 7) mod V."""
    ids = np.zeros((n, cfg.seq), dtype=np.int64)
    ids[:, 0] = rng.integers(0, cfg.vocab, size=n)
    ids[:, 1] = rng.integers(0, cfg.vocab, size=n)
    for t in range(2, cfg.seq):
        ids[:, t] = (3 * ids[:, t - 1] + 5 * ids[:, t - 2] + 7) % cfg.vocab
    return ids


def train_teacher(cfg: Config, *, steps: int, batch: int, lr: float,
                  seed: int, log_every: int = 100):
    rng = np.random.default_rng(seed)
    params = init_params(rng, cfg)
    opt = Adam(params, lr=lr)
    history = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        ids = gen_data(rng, batch, cfg)
        logits, cache = forward(params, cfg, ids[:, :-1])
        loss, dl = ce_loss(logits, ids[:, 1:])
        grads = backward(params, cfg, cache, dl)
        opt.step(params, grads)
        if step % log_every == 0 or step == steps:
            history.append({"step": step, "loss": loss,
                            "elapsed_s": time.perf_counter() - t0})
    return params, history


def eval_metrics(params: dict, cfg: Config, *, batches: int = 8,
                 batch: int = 256, seed: int = 999):
    """Real held-out accuracy + loss on the synthetic grammar."""
    rng = np.random.default_rng(seed)
    tot_loss, tot_correct, tot_n = 0.0, 0, 0
    for _ in range(batches):
        ids = gen_data(rng, batch, cfg)
        logits, _ = forward(params, cfg, ids[:, :-1])
        loss, _ = ce_loss(logits, ids[:, 1:])
        pred = logits.argmax(axis=-1)
        tot_loss += loss
        tot_correct += int((pred == ids[:, 1:]).sum())
        tot_n += batch
    return {"loss": tot_loss / batches, "accuracy": tot_correct / (tot_n * (cfg.seq - 1))}


def _act_input(name: str, cache: dict, cfg: Config):
    """Input activation feeding the op that owns tensor `name`, taken from
    a real forward cache. Returns None when not attributable."""
    if name == "emb":
        return cache["L0"]["x_prev"]
    if name in ("Whead", "bhead"):
        return cache["f"]
    if name in ("lnfg", "lnfb"):
        return cache["x_pre_lnf"]
    if name.startswith("L"):
        _, rest = name.split(".", 1)
        l = name[1:].split(".")[0]
        c = cache[f"L{l}"]
        if rest in ("Wq", "Wk", "Wv", "bq", "bk", "bv"):
            return c["h"]
        if rest in ("Wo", "bo"):
            return c["A"]
        if rest in ("W1", "b1"):
            return c["z"]
        if rest in ("W2", "b2"):
            return c["a"]
        if rest in ("ln1g", "ln1b"):
            return c["x_prev"]
        if rest in ("ln2g", "ln2b"):
            return c["y"]
    return None


def measure_tensor_stats(params: dict, cfg: Config, batches: int = 4,
                         batch: int = 64, seed: int = 0) -> dict:
    """Measured per-tensor contributions over real forward/backward passes.

    activation contribution: RMS of the input activation feeding each
        tensor's op, averaged over batches (from the forward cache).
    gradient contribution: mean Frobenius norm of the tensor's gradient,
        plus the relative ||grad|| / ||W||.
    Returns {tensor_name: {"act_rms": float, "grad_norm": float,
                           "grad_rel": float}}.
    """
    rng = np.random.default_rng(seed)
    acc_a, acc_g, n = {}, {}, 0
    for _ in range(batches):
        ids = gen_data(rng, batch, cfg)
        logits, cache = forward(params, cfg, ids[:, :-1])
        _, dlogits = ce_loss(logits, ids[:, 1:])
        grads = backward(params, cfg, cache, dlogits)
        n += 1
        for k, W in params.items():
            ai = _act_input(k, cache, cfg)
            if ai is not None:
                v = float(np.sqrt(np.mean(np.asarray(ai, dtype=np.float64) ** 2)))
                acc_a[k] = acc_a.get(k, 0.0) + v
            g = grads.get(k)
            if g is not None:
                acc_g[k] = acc_g.get(k, 0.0) + float(np.linalg.norm(g))
    stats = {}
    for k, W in params.items():
        wn = float(np.linalg.norm(W))
        gn = acc_g.get(k, 0.0) / n
        stats[k] = {
            "act_rms": acc_a.get(k, 0.0) / n if k in acc_a else None,
            "grad_norm": gn,
            "grad_rel": gn / (wn + 1e-12),
        }
    return stats


# ---------------------------------------------------------------------------
# Generation (greedy; streaming-capable)
# ---------------------------------------------------------------------------

def generate(params: dict, cfg: Config, prompt, max_tokens: int,
             seed: int = 0):
    """Greedy generation. Returns (tokens, tokens_per_sec, elapsed_s)."""
    return _generate_impl(params, cfg, prompt, max_tokens, on_token=None)


def generate_stream(params: dict, cfg: Config, prompt, max_tokens: int,
                    seed: int = 0, on_token=None):
    """Greedy generation, yielding each token as produced (true streaming).
    Returns (tokens, tokens_per_sec, elapsed_s)."""
    return _generate_impl(params, cfg, prompt, max_tokens, on_token=on_token)


def _generate_impl(params: dict, cfg: Config, prompt, max_tokens: int,
                   on_token):
    ids = np.array([list(prompt)], dtype=np.int64)
    out = []
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        ctx = ids[:, -cfg.seq:]
        logits, _ = forward(params, cfg, ctx)
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if on_token is not None:
            on_token(nxt)
        ids = np.concatenate([ids, [[nxt]]], axis=1)
    dt = time.perf_counter() - t0
    tps = len(out) / dt if dt > 0 else 0.0
    return out, tps, dt
