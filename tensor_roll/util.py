# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared utilities: timing, memory accounting, formatting, audit logging."""

from __future__ import annotations

import json
import os
import resource
import time
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def rss_mb() -> float:
    """Current process resident set size in MiB (Linux /proc)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}TB"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


class AuditLog:
    """Append-only JSON-lines audit log."""

    def __init__(self, path: str):
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        self.path = path

    def record(self, **fields) -> None:
        fields.setdefault("ts", utc_now_iso())
        with open(self.path, "a") as f:
            f.write(json.dumps(fields) + "\n")
