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
