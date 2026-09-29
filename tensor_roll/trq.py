# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The .trq container format (spec section 10) — a real binary codec.

Layout (all integers little-endian):
  magic            4 bytes  b'TRQ1'
  header_len       u32
  header           JSON {version, arch, n_tensors, provenance, roll_meta, created}
  per tensor:
    name_len       u16
    name           bytes
    ndim           u8
    shape          ndim x u64
    dtype_code     u8   0=fp32 1=fp16 2=bf16 3=int8 4=int4-packed
    quant_code     u8   0=none 1=int8-asym 2=int4-group32 3=fp16-cast 4=bf16-cast
    meta_len       u32
    meta           JSON {quantization metadata: scales/zeros as base64 f64,
                         roll metadata, codebooks}
    data_len       u64
    data           bytes
    checksum       32 bytes sha256(data)
  footer           32 bytes sha256(all preceding bytes)

read_trq() verifies the file checksum and every per-tensor checksum and
raises on any mismatch. Corrupt files never load silently.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct

import numpy as np

from .util import utc_now_iso

MAGIC = b"TRQ1"
VERSION = 1

_DTYPE_CODES = {"fp32": 0, "fp16": 1, "bf16": 2, "int8": 3, "int4": 4}
_DTYPE_NAMES = {v: k for k, v in _DTYPE_CODES.items()}
_NP_DTYPES = {"fp32": np.float32, "fp16": np.float16, "bf16": np.uint16,
              "int8": np.int8, "int4": np.uint8}
_QUANT_CODES = {"none": 0, "int8-asym": 1, "int4-group32": 2,
                "fp16-cast": 3, "bf16-cast": 4}
_QUANT_NAMES = {v: k for k, v in _QUANT_CODES.items()}


def _b64(arr) -> str:
    return base64.b64encode(np.asarray(arr, dtype=np.float64).tobytes()).decode()


def _unb64(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s.encode()), dtype=np.float64)


def _meta_to_json(meta: dict) -> dict:
    m = dict(meta)
    for k, v in list(m.items()):
        if isinstance(v, np.ndarray):
            # codebooks, scales, zero-points, or any ndarray metadata
            m[k] = {"__b64_f64__": _b64(np.asarray(v, dtype=np.float64)),
                    "__shape__": list(v.shape), "__key__": k}
    return m


def _meta_from_json(m: dict) -> dict:
    m = dict(m)
    for k, v in list(m.items()):
        if isinstance(v, dict) and "__b64_f64__" in v:
            arr = _unb64(v["__b64_f64__"])
            if "__shape__" in v:
                arr = arr.reshape(v["__shape__"])
            m[k] = arr
    return m


class TrqError(ValueError):
    pass


def write_trq(path: str, *, arch: dict, tensors: dict, provenance: dict,
              roll_meta: dict = None) -> dict:
    """tensors: {name: (payload_array, fmt_name, quant_meta_dict)}.
    Returns {"path":..., "bytes":..., "sha256":...}."""
    buf = bytearray()
    buf += MAGIC
    header = {
        "version": VERSION,
        "arch": arch,
        "n_tensors": len(tensors),
        "provenance": provenance,
        "roll_meta": roll_meta or {},
        "created": utc_now_iso(),
    }
    hbytes = json.dumps(header).encode()
    buf += struct.pack("<I", len(hbytes)) + hbytes

    for name, (payload, fmt, qmeta) in tensors.items():
        payload = np.ascontiguousarray(payload)
        nbytes = payload.tobytes()
        name_b = name.encode()
        buf += struct.pack("<H", len(name_b)) + name_b
        buf += struct.pack("<B", payload.ndim)
        buf += struct.pack(f"<{payload.ndim}Q", *payload.shape)
        buf += struct.pack("<B", _DTYPE_CODES[fmt])
        quant_name = {"fp32": "none", "fp16": "fp16-cast", "bf16": "bf16-cast",
                      "int8": "int8-asym", "int4": "int4-group32"}[fmt]
        buf += struct.pack("<B", _QUANT_CODES[quant_name])
        meta = _meta_to_json({"fmt": fmt, **qmeta,
                              "roll": (roll_meta or {}).get(name, {})})
        mbytes = json.dumps(meta).encode()
        buf += struct.pack("<I", len(mbytes)) + mbytes
        buf += struct.pack("<Q", len(nbytes)) + nbytes
        buf += hashlib.sha256(nbytes).digest()

    digest = hashlib.sha256(bytes(buf)).digest()
    buf += digest
    with open(path, "wb") as f:
        f.write(bytes(buf))
    return {"path": path, "bytes": len(buf), "sha256": digest.hex()}


def read_trq(path: str):
    """Returns (header, tensors) with all checksums verified.
    tensors: {name: (payload_array, meta_dict)}. Raises TrqError on corruption."""
    with open(path, "rb") as f:
        blob = f.read()
    if len(blob) < 4 + 32 or blob[:4] != MAGIC:
        raise TrqError("not a .trq file (bad magic)")
    file_sum, body = blob[-32:], blob[:-32]
    if hashlib.sha256(body).digest() != file_sum:
        raise TrqError("file checksum mismatch — refusing to load")
    off = 4
    hlen = struct.unpack_from("<I", body, off)[0]
    off += 4
    header = json.loads(body[off:off + hlen])
    off += hlen
    tensors = {}
    for _ in range(header["n_tensors"]):
        nlen = struct.unpack_from("<H", body, off)[0]
        off += 2
        name = body[off:off + nlen].decode()
        off += nlen
        ndim = struct.unpack_from("<B", body, off)[0]
        off += 1
        shape = struct.unpack_from(f"<{ndim}Q", body, off)
        off += 8 * ndim
        dtype_code = struct.unpack_from("<B", body, off)[0]
        off += 1
        quant_code = struct.unpack_from("<B", body, off)[0]
        off += 1
        mlen = struct.unpack_from("<I", body, off)[0]
        off += 4
        meta = _meta_from_json(json.loads(body[off:off + mlen]))
        off += mlen
        dlen = struct.unpack_from("<Q", body, off)[0]
        off += 8
        data = body[off:off + dlen]
        off += dlen
        cksum = body[off:off + 32]
        off += 32
        if hashlib.sha256(data).digest() != cksum:
            raise TrqError(f"tensor {name!r} checksum mismatch — refusing to load")
        fmt = _DTYPE_NAMES[dtype_code]
        arr = np.frombuffer(data, dtype=_NP_DTYPES[fmt])
        if fmt == "int4":
            arr = arr.reshape(-1)  # packed nibbles stay flat; dequantize reshapes
        else:
            arr = arr.reshape(shape)
        tensors[name] = (arr, meta)
    return header, tensors


def total_params(tensors: dict) -> int:
    """Total logical parameters across tensors (uses recorded shapes, so
    packed INT4 counts its true parameters, not its packed bytes)."""
    total = 0
    for _name, (arr, meta) in tensors.items():
        shape = meta.get("shape") or list(arr.shape)
        total += int(np.prod(shape))
    return total


def param_budget_ok(tensors: dict, cap: float = 8_500_000_000) -> tuple:
    """Verify the exported artifact obeys the hard parameter budget.
    Returns (total_params, ok)."""
    total = total_params(tensors)
    return total, total <= cap
