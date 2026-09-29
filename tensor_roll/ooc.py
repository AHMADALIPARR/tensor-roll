# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Out-of-core execution — RULE ZERO: never require the teacher in RAM.

The transformation must operate incrementally over a model larger than
system RAM (and larger than RAM+VRAM combined):

    30B MODEL STORAGE -> MANIFEST READER -> SHARD INDEX -> MMAP READER
      -> ONE TENSOR / BLOCK -> TENSOR ROLL -> 8B TARGET SHARD
      -> release source memory -> NEXT BLOCK

New components (spec section 1):
  TensorSource   manifest + shard index; never materializes the model
  ShardReader    lazy, memory-mapped tensor reads
  MappedTensor   memmap-backed tensor with exact shape/dtype
  TensorWindow   recursive windowing of tensors larger than the budget
  RollWorkspace  bounded working-set accounting with a hard ceiling
  StudentWriter  incremental 8B-shard emission + verified shard index

At no point does ingestion call load_model_30b() and hold all weights.
The process respects --memory-budget within allocator overhead; the
RollWorkspace refuses allocations that would exceed the ceiling.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .util import ensure_dir, utc_now_iso


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_bytes(s) -> int:
    """'5G' -> bytes. Accepts int (bytes) or strings like '512M', '2G'."""
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip().upper()
    for suffix, mult in (("TB", 1024 ** 4), ("GB", 1024 ** 3), ("G", 1024 ** 3),
                         ("MB", 1024 ** 2), ("M", 1024 ** 2),
                         ("KB", 1024), ("K", 1024), ("B", 1)):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(float(s))


_DTYPE_ITEMSIZE = {"float32": 4, "float64": 8, "float16": 2, "int8": 1,
                   "int32": 4, "int64": 8, "uint8": 1, "uint16": 2}


def dtype_itemsize(dtype: str) -> int:
    return _DTYPE_ITEMSIZE[np.dtype(dtype).name]


class MemoryBudgetExceeded(MemoryError):
    pass


# ---------------------------------------------------------------------------
# Manifests — the model is described, never loaded
# ---------------------------------------------------------------------------

@dataclass
class TensorSpec:
    """One tensor's address in storage. No data attached."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    nbytes: int
    shard: str          # shard file name (relative to manifest dir)
    offset: int         # byte offset inside the shard

    @property
    def params(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def to_dict(self) -> dict:
        return {"name": self.name, "shape": list(self.shape),
                "dtype": self.dtype, "nbytes": self.nbytes,
                "shard": self.shard, "offset": self.offset}

    @staticmethod
    def from_dict(d: dict) -> "TensorSpec":
        return TensorSpec(d["name"], tuple(d["shape"]), d["dtype"],
                          d["nbytes"], d["shard"], d["offset"])


class VirtualManifest:
    """A model manifest with NO weight data — for structural scale tests.

    Lets the SCAN/PLAN machinery address a ~30B model without allocating
    a single weight. Any attempt to read data raises; only metadata flows.
    """

    def __init__(self, specs: List[TensorSpec], *, model_name: str = "virtual",
                 provenance: dict = None):
        self.specs = specs
        self.model_name = model_name
        self.provenance = provenance or {}

    @property
    def total_params(self) -> int:
        return sum(s.params for s in self.specs)

    @property
    def total_bytes(self) -> int:
        return sum(s.nbytes for s in self.specs)

    def to_dict(self) -> dict:
        return {"model": self.model_name, "virtual": True,
                "total_params": self.total_params,
                "total_bytes": self.total_bytes,
                "created": utc_now_iso(),
                "provenance": self.provenance,
                "tensors": [s.to_dict() for s in self.specs]}

    def write(self, path: str) -> dict:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return {"path": path, "tensors": len(self.specs),
                "total_params": self.total_params}

    @staticmethod
    def read(path: str) -> "VirtualManifest":
        with open(path) as f:
            d = json.load(f)
        return VirtualManifest(
            [TensorSpec.from_dict(s) for s in d["tensors"]],
            model_name=d.get("model", "virtual"),
            provenance=d.get("provenance", {}))

    def reader(self):
        raise RuntimeError(
            "virtual manifest has no weight data to read — "
            "metadata only (30B STRUCTURAL SCALE TEST)")


def glimmer30b_manifest() -> VirtualManifest:
    """Realistic ~30B dense-transformer manifest (decoder-heavy, like the
    published Glimmer 30B: ~2B perception encoder + ~28B text decoder).

    Shapes are realistic transformer dimensions; NO weights are allocated.
    Used exclusively by the 30B STRUCTURAL SCALE TEST — this does not
    validate model quality, only that storage/planning address 30B under
    bounded RAM.
    """
    specs: List[TensorSpec] = []
    off = 0

    def add(name, shape, dtype="float16", shard="decoder.safetensors"):
        nonlocal off
        nbytes = int(np.prod(shape)) * dtype_itemsize(dtype)
        specs.append(TensorSpec(name, tuple(shape), dtype, nbytes, shard, off))
        off += nbytes

    # --- perception encoder (~2B): a few large projection blocks --------
    d_v, n_v = 2048, 24
    for l in range(n_v):
        add(f"vision.L{l}.Wqkv", (d_v, 3 * d_v))
        add(f"vision.L{l}.Wo", (d_v, d_v))
        add(f"vision.L{l}.W1", (d_v, 4 * d_v))
        add(f"vision.L{l}.W2", (4 * d_v, d_v))
        add(f"vision.L{l}.ln1", (d_v,), shard="decoder.safetensors")
        add(f"vision.L{l}.ln2", (d_v,), shard="decoder.safetensors")
    add("vision.proj", (d_v, 7168))

    # --- text decoder (~28B): 44 layers, d=7168 --------------------------
    d, dff, layers, vocab = 7168, 28672, 44, 128256
    for s_idx in range(6):
        shard = f"decoder-{s_idx:05d}.safetensors"
        l0, l1 = s_idx * 8, min(layers, (s_idx + 1) * 8)
        for l in range(l0, l1):
            add(f"L{l}.Wq", (d, d), shard=shard)
            add(f"L{l}.Wk", (d, d), shard=shard)
            add(f"L{l}.Wv", (d, d), shard=shard)
            add(f"L{l}.Wo", (d, d), shard=shard)
            add(f"L{l}.W1", (d, dff), shard=shard)
            add(f"L{l}.W2", (dff, d), shard=shard)
            add(f"L{l}.ln1", (d,), shard=shard)
            add(f"L{l}.ln2", (d,), shard=shard)
    add("emb", (vocab, d), shard="decoder-00000.safetensors")
    add("Whead", (d, vocab), shard="decoder-00005.safetensors")
    add("lnf", (d,), shard="decoder-00005.safetensors")

    total = sum(s.params for s in specs)
    return VirtualManifest(
        specs, model_name="glimmer-30b-virtual",
        provenance={"note": "30B STRUCTURAL SCALE TEST manifest — shapes "
                            "only, no weights allocated",
                    "total_params": total})


# ---------------------------------------------------------------------------
# Real shard storage: write once, read via mmap
# ---------------------------------------------------------------------------

class ShardWriter:
    """Writes tensors as raw little-endian blobs into shard files and emits
    a manifest.json describing every tensor's address."""

    def __init__(self, root: str, shard_size_bytes: int = 256 * 1024 * 1024,
                 dtype: str = "float32"):
        self.root = ensure_dir(root)
        self.shard_size = int(shard_size_bytes)
        self.specs: List[TensorSpec] = []
        self._cur: Optional[str] = None
        self._fh = None
        self._off = 0

    def _rotate(self):
        if self._fh:
            self._fh.close()
        idx = len([s for s in self.specs])  # unique shard index
        self._cur = f"shard-{idx:05d}.bin"
        self._fh = open(os.path.join(self.root, self._cur), "wb")
        self._off = 0

    def add(self, name: str, arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        blob = arr.tobytes()
        if self._fh is None or self._off + len(blob) > self.shard_size:
            self._rotate()
        self._fh.write(blob)
        self.specs.append(TensorSpec(
            name, tuple(arr.shape), str(arr.dtype), len(blob),
            self._cur, self._off))
        self._off += len(blob)

    def finalize(self, model_name: str = "teacher") -> str:
        if self._fh:
            self._fh.close()
            self._fh = None
        man = {"model": model_name, "virtual": False,
               "total_params": sum(s.params for s in self.specs),
               "total_bytes": sum(s.nbytes for s in self.specs),
               "created": utc_now_iso(),
               "tensors": [s.to_dict() for s in self.specs]}
        path = os.path.join(self.root, "manifest.json")
        with open(path, "w") as f:
            json.dump(man, f, indent=2)
        return path


class TensorSource:
    """A model addressable tensor-by-tensor. Backed by real shard files
    (mmap reads) or by a virtual manifest (metadata only)."""

    def __init__(self, root: str):
        self.root = root
        man_path = os.path.join(root, "manifest.json")
        with open(man_path) as f:
            man = json.load(f)
        self.virtual = bool(man.get("virtual"))
        self.model_name = man.get("model", "?")
        self.specs = [TensorSpec.from_dict(s) for s in man["tensors"]]

    @property
    def total_params(self) -> int:
        return sum(s.params for s in self.specs)

    def names(self) -> List[str]:
        return [s.name for s in self.specs]

    def spec(self, name: str) -> TensorSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise KeyError(name)

    def reader(self) -> "ShardReader":
        if self.virtual:
            raise RuntimeError("virtual manifest has no weight data to read")
        return ShardReader(self)


class ShardReader:
    """Lazy memory-mapped tensor reads. One tensor (or window) at a time."""

    def __init__(self, source: TensorSource):
        self.source = source
        self._mmaps: Dict[str, np.memmap] = {}

    def _mmap(self, spec: TensorSpec) -> np.memmap:
        key = spec.shard
        if key not in self._mmaps:
            path = os.path.join(self.source.root, spec.shard)
            # map the whole shard file once; tensors are views into it
            size = os.path.getsize(path)
            self._mmaps[key] = np.memmap(path, dtype=np.uint8, mode="r",
                                         shape=(size,))
        return self._mmaps[key]

    def read(self, name: str) -> "MappedTensor":
        spec = self.source.spec(name)
        raw = self._mmap(spec)
        view = raw[spec.offset:spec.offset + spec.nbytes].view(
            np.dtype(spec.dtype)).reshape(spec.shape)
        return MappedTensor(name, view, spec)

    def close(self):
        self._mmaps = {}


class MappedTensor:
    """A memmap-backed tensor. `.window(slices)` yields TensorWindows."""

    def __init__(self, name: str, view: np.ndarray, spec: TensorSpec):
        self.name = name
        self.view = view  # memmap view — no copy
        self.spec = spec

    @property
    def shape(self):
        return self.view.shape

    def window(self, slices: Tuple[slice, ...]) -> "TensorWindow":
        return TensorWindow(self, slices)

    def windows(self, budget_bytes: int) -> Iterator["TensorWindow"]:
        """Recursive windowing until each window fits the budget."""
        for sl in plan_windows(self.view.shape,
                               self.view.dtype.itemsize, budget_bytes):
            yield TensorWindow(self, sl)


@dataclass
class TensorWindow:
    """One bounded piece of a (possibly huge) tensor, with exact
    reconstruction metadata. No silent shape corruption: the window
    records its slices, shape, and global offsets."""
    parent: MappedTensor
    slices: Tuple[slice, ...]

    @property
    def name(self) -> str:
        idx = ",".join(f"{s.start}:{s.stop}" for s in self.slices)
        return f"{self.parent.name}@[{idx}]"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def nbytes(self) -> int:
        return self.data.nbytes

    @property
    def data(self) -> np.ndarray:
        # materialize ONLY this window (bounded by the budget)
        return np.asarray(self.parent.view[self.slices])

    def meta(self) -> dict:
        return {"tensor": self.parent.name,
                "slices": [[s.start, s.stop] for s in self.slices],
                "shape": list(self.shape),
                "global_shape": list(self.parent.shape)}


def plan_windows(shape: Tuple[int, ...], itemsize: int,
                 budget_bytes: int) -> List[Tuple[slice, ...]]:
    """Recursively split along the largest axis until every chunk's
    byte size <= budget_bytes. Returns exact slice tuples covering the
    tensor with no overlap and no gaps."""
    total = int(np.prod(shape)) * itemsize
    if total <= budget_bytes or len(shape) == 0:
        return [tuple(slice(0, d) for d in shape)]
    # split the largest axis in half, recurse on both halves
    ax = int(np.argmax(shape))
    mid = shape[ax] // 2
    if mid == 0:
        raise MemoryBudgetExceeded(
            f"cannot window shape {shape}: axis {ax} has size 1 but a "
            f"single element ({itemsize}B) exceeds the budget")
    out = []
    for lo, hi in ((0, mid), (mid, shape[ax])):
        sub = list(shape)
        sub[ax] = hi - lo
        for sl in plan_windows(tuple(sub), itemsize, budget_bytes):
            full = list(sl)
            s = full[ax]
            full[ax] = slice(s.start + lo, s.stop + lo)
            out.append(tuple(full))
    return out


def reassemble(windows: List[TensorWindow], shape: Tuple[int, ...],
               dtype=np.float64) -> np.ndarray:
    """Rebuild the full tensor from windows; verifies exact coverage."""
    out = np.zeros(shape, dtype=dtype)
    seen = np.zeros(shape, dtype=bool)
    for w in windows:
        m = w.meta()
        sl = tuple(slice(a, b) for a, b in m["slices"])
        assert w.data.shape == tuple(b - a for a, b in m["slices"]), \
            f"window shape mismatch for {w.name}"
        out[sl] = w.data
        assert not seen[sl].any(), f"overlapping windows at {w.name}"
        seen[sl] = True
    assert seen.all(), "windows do not cover the tensor exactly"
    return out


# ---------------------------------------------------------------------------
# Bounded working set
# ---------------------------------------------------------------------------

class RollWorkspace:
    """Tracks the live working set against a hard memory ceiling.

    Every materialized window/buffer must be allocated through here.
    Allocations that would exceed the ceiling raise MemoryBudgetExceeded
    instead of silently paging the machine to death.
    """

    def __init__(self, ceiling_bytes: int):
        self.ceiling = int(ceiling_bytes)
        self.live: Dict[str, int] = {}
        self.peak = 0
        self.allocations = 0

    @property
    def used(self) -> int:
        return sum(self.live.values())

    def alloc(self, tag: str, nbytes: int):
        nbytes = int(nbytes)
        if self.used + nbytes > self.ceiling:
            raise MemoryBudgetExceeded(
                f"workspace ceiling {self.ceiling}B exceeded: "
                f"used={self.used}B + {nbytes}B ({tag})")
        self.live[tag] = self.live.get(tag, 0) + nbytes
        self.peak = max(self.peak, self.used)
        self.allocations += 1

    def free(self, tag: str):
        self.live.pop(tag, None)

    def report(self) -> dict:
        return {"ceiling_bytes": self.ceiling, "peak_bytes": self.peak,
                "current_bytes": self.used,
                "allocations": self.allocations,
                "within_budget": self.peak <= self.ceiling}


# ---------------------------------------------------------------------------
# Incremental student emission
# ---------------------------------------------------------------------------

class StudentWriter:
    """Writes student shards incrementally + a verified shard index.

    Each emitted tensor is checksummed; finalize() writes shard_index.json
    with per-tensor sha256 and a total parameter count.

    Crash safety: flush_index() rewrites shard_index.json after every
    tensor (atomic replace). On resume, the writer reloads the existing
    index so previously emitted shards are kept and file names continue
    without collision.
    """

    INDEX_NAME = "shard_index.json"

    def __init__(self, root: str, resume: bool = False):
        self.root = ensure_dir(root)
        self.entries: List[dict] = []
        if resume:
            idx_path = os.path.join(root, self.INDEX_NAME)
            if os.path.exists(idx_path):
                with open(idx_path) as f:
                    index = json.load(f)
                self.entries = index.get("tensors", [])

    def emit(self, name: str, arr: np.ndarray, meta: dict = None) -> dict:
        arr = np.ascontiguousarray(arr)
        fname = f"student-{len(self.entries):05d}.npy"
        path = os.path.join(self.root, fname)
        # np.save appends .npy when missing, so the temp name keeps it
        tmp = path + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, path)  # atomic: no half-written shards
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        entry = {"name": name, "file": fname, "shape": list(arr.shape),
                 "dtype": str(arr.dtype), "params": int(arr.size),
                 "sha256": digest, "meta": meta or {}}
        self.entries.append(entry)
        return entry

    def flush_index(self) -> dict:
        """Rewrite the index now (per-tensor checkpointing)."""
        return self.finalize()

    def finalize(self) -> dict:
        index = {"created": utc_now_iso(),
                 "n_tensors": len(self.entries),
                 "total_params": sum(e["params"] for e in self.entries),
                 "tensors": self.entries}
        payload = json.dumps(index, sort_keys=True).encode()
        index["index_sha256"] = hashlib.sha256(payload).hexdigest()
        tmp = os.path.join(self.root, self.INDEX_NAME + ".tmp")
        with open(tmp, "w") as f:
            json.dump(index, f, indent=2)
        os.replace(tmp, os.path.join(self.root, self.INDEX_NAME))
        return index

    @staticmethod
    def verify(root: str) -> dict:
        with open(os.path.join(root, "shard_index.json")) as f:
            index = json.load(f)
        for e in index["tensors"]:
            arr = np.load(os.path.join(root, e["file"]))
            assert list(arr.shape) == e["shape"], f"shape drift: {e['name']}"
            assert (hashlib.sha256(
                np.ascontiguousarray(arr).tobytes()).hexdigest()
                    == e["sha256"]), f"checksum mismatch: {e['name']}"
        return {"tensors": index["n_tensors"],
                "total_params": index["total_params"], "ok": True}


# ---------------------------------------------------------------------------
# Checkpoints — crash-safe, corruption-rejecting
# ---------------------------------------------------------------------------

class Checkpoint:
    """JSON state + sha256. Corrupted checkpoints are refused, never loaded."""

    def __init__(self, path: str):
        self.path = path

    def save(self, state: dict) -> dict:
        payload = json.dumps(state, sort_keys=True).encode()
        digest = hashlib.sha256(payload).hexdigest()
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"sha256": digest, "state": state}, f)
        os.replace(tmp, self.path)  # atomic
        return {"path": self.path, "sha256": digest}

    def load(self) -> dict:
        with open(self.path) as f:
            doc = json.load(f)
        payload = json.dumps(doc["state"], sort_keys=True).encode()
        if hashlib.sha256(payload).hexdigest() != doc["sha256"]:
            raise ValueError(
                f"checkpoint checksum mismatch: {self.path} — refusing")
        return doc["state"]

    def exists(self) -> bool:
        return os.path.exists(self.path)


# ---------------------------------------------------------------------------
# Out-of-core driver: SCAN -> PLAN -> EXECUTE with resume
# ---------------------------------------------------------------------------

class OutOfCoreDriver:
    """Streams a teacher through TensorRoll without ever holding it in RAM.

    scan:    lightweight per-tensor stats from the manifest (no weights).
    plan:    global budget allocation (delegates to planner.plan).
    execute: per tensor (windowed as needed): mmap -> roll -> transform ->
             emit student shard -> release. Checkpointed per tensor so
             --resume continues after a crash.
    """

    def __init__(self, source: TensorSource, workdir: str,
                 memory_budget: str = "5G", target_params: float = 8.0e9,
                 cap_params: float = 8.5e9):
        self.source = source
        self.workdir = ensure_dir(workdir)
        self.ws = RollWorkspace(parse_bytes(memory_budget))
        self.target = float(target_params)
        self.cap = float(cap_params)
        self.ckpt = Checkpoint(os.path.join(workdir, "ooc_checkpoint.json"))
        self.student_dir = ensure_dir(os.path.join(workdir, "student_shards"))

    def scan(self) -> List[dict]:
        """Lightweight stats per tensor — manifest only, no weight data."""
        from . import core as C  # deferred: keeps import graph light
        rows = []
        for s in self.source.specs:
            rows.append({
                "tensor": s.name, "shape": list(s.shape), "dtype": s.dtype,
                "params": s.params, "bytes": s.nbytes, "shard": s.shard,
                # shape-derived proxies, measured from metadata only
                "is_matrix": len(s.shape) == 2 and min(s.shape) > 1,
                "aspect": (max(s.shape) / max(1, min(s.shape))
                           if s.shape else 1.0),
            })
        return rows

    def plan(self, rows: List[dict]) -> dict:
        from . import planner as P
        return P.plan(rows, target=self.target, cap=self.cap)

    def execute(self, plan: dict, transform, *,
                resume: bool = False, fault_after: int = -1) -> dict:
        """Run the plan. `transform(name, window_data, w_target, ws, meta)`
        is the caller-supplied per-window transform returning either one
        student array or a dict {suffix: array} (e.g. factor pairs).
        Checkpointed per tensor; the student index is flushed per tensor
        too, so `fault_after` crash tests resume with zero emitted-data
        loss. `fault_after` injects a crash (tests only, never production).
        """
        state = self.ckpt.load() if (resume and self.ckpt.exists()) else {
            "done": [], "emitted_params": 0}
        done = set(state["done"])
        writer = StudentWriter(self.student_dir, resume=resume)
        # re-verify previously emitted shards on resume
        if resume and done:
            StudentWriter.verify(self.student_dir)
        reader = self.source.reader()
        processed, emitted = 0, state["emitted_params"]
        try:
            for spec in self.source.specs:
                if spec.name in done:
                    continue
                mt = reader.read(spec.name)
                for w in mt.windows(self.ws.ceiling):
                    tag = f"win:{w.name}"
                    self.ws.alloc(tag, w.nbytes)
                    try:
                        data = w.data
                        target = plan["tensors"][spec.name]["target_params"]
                        # split the tensor's target budget across windows
                        # proportionally to window params
                        w_target = max(
                            1, int(target * w.data.size / spec.params))
                        out = transform(spec.name, data, w_target, self.ws,
                                        w.meta())
                    finally:
                        self.ws.free(tag)
                    parts = out if isinstance(out, dict) else {"": out}
                    for suffix, arr in parts.items():
                        arr = np.asarray(arr)
                        ename = w.name + (f".{suffix}" if suffix else "")
                        writer.emit(ename, arr,
                                    {"parent": spec.name, **w.meta()})
                        emitted += int(arr.size)
                # release the memmap view for this tensor before next
                del mt
                processed += 1
                done.add(spec.name)
                state = {"done": sorted(done), "emitted_params": emitted}
                self.ckpt.save(state)
                writer.flush_index()
                if fault_after >= 0 and processed > fault_after:
                    raise RuntimeError("injected crash (fault_after test)")
        finally:
            reader.close()
        index = writer.finalize()
        return {"processed_tensors": processed,
                "emitted_params": emitted,
                "index_total_params": index["total_params"],
                "workspace": self.ws.report(),
                "budget_ok": index["total_params"] <= self.cap}


# ---------------------------------------------------------------------------
# External teacher formats: safetensors (true mmap) and npz (streaming repack)
# ---------------------------------------------------------------------------

_SAFETENSORS_DTYPES = {
    "F32": "float32", "F16": "float16",
    "I32": "int32", "I64": "int64",
    "U8": "uint8", "I8": "int8", "BOOL": "bool",
}


def _read_safetensors_header(path: str) -> Tuple[dict, int]:
    """Returns (header_dict, data_start_offset). Metadata only."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    return json.loads(raw.decode("utf-8")), 8 + n


class SafetensorsSource:
    """A directory of .safetensors files. SCAN reads only JSON headers;
    EXECUTE windows tensors through np.memmap — the 30B weights are never
    fully in RAM."""

    def __init__(self, path: str):
        self.path = path
        self.specs: List[TensorSpec] = []
        self._files: Dict[str, str] = {}  # shard name -> abs path
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".safetensors"):
                continue
            fpath = os.path.join(path, fname)
            header, data_start = _read_safetensors_header(fpath)
            self._files[fname] = fpath
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                dt = info["dtype"]
                if dt == "BF16":
                    try:
                        import ml_dtypes  # noqa
                        np_dtype = "bfloat16"
                    except ImportError:
                        raise RuntimeError(
                            f"tensor {name}: BF16 needs the ml_dtypes "
                            f"package for CPU decode; refusing to guess")
                elif dt not in _SAFETENSORS_DTYPES:
                    raise RuntimeError(
                        f"tensor {name}: unsupported dtype {dt}")
                else:
                    np_dtype = _SAFETENSORS_DTYPES[dt]
                shape = tuple(info["shape"])
                start, end = info["data_offsets"]
                nbytes = end - start
                self.specs.append(TensorSpec(
                    f"{fname}:{name}", shape, np_dtype, nbytes, fname,
                    data_start + start))
        if not self.specs:
            raise ValueError(f"no safetensors tensors found in {path}")

    @property
    def total_params(self) -> int:
        return sum(s.params for s in self.specs)

    def names(self) -> List[str]:
        return [s.name for s in self.specs]

    def spec(self, name: str) -> TensorSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise KeyError(name)

    def reader(self) -> "SafetensorsReader":
        return SafetensorsReader(self)


class SafetensorsReader:
    """Lazy memory-mapped reads over safetensors files."""

    def __init__(self, source: SafetensorsSource):
        self.source = source
        self._mmaps: Dict[str, np.memmap] = {}

    def _mmap(self, spec: TensorSpec) -> np.memmap:
        if spec.shard not in self._mmaps:
            path = self.source._files[spec.shard]
            size = os.path.getsize(path)
            self._mmaps[spec.shard] = np.memmap(
                path, dtype=np.uint8, mode="r", shape=(size,))
        return self._mmaps[spec.shard]

    def read(self, name: str) -> MappedTensor:
        spec = self.source.spec(name)
        raw = self._mmap(spec)
        view = raw[spec.offset:spec.offset + spec.nbytes].view(
            np.dtype(spec.dtype)).reshape(spec.shape)
        return MappedTensor(name, view, spec)

    def close(self):
        self._mmaps = {}


def _scan_npz(path: str) -> List[Tuple[str, tuple, str, int]]:
    """Metadata-only scan of one .npz (zip headers, no array data)."""
    import zipfile
    from numpy.lib import format as npfmt
    specs = []
    with zipfile.ZipFile(path) as zf:
        for entry in zf.namelist():
            if not entry.endswith(".npy"):
                continue
            with zf.open(entry) as f:
                npfmt.read_magic(f)
                shape, _fortran, dtype = npfmt.read_array_header_1_0(f)
            name = entry[:-4]
            specs.append((name, tuple(shape), np.dtype(dtype).name,
                          int(np.prod(shape, dtype=np.int64))
                          * np.dtype(dtype).itemsize))
    return specs


def prepare_teacher_source(teacher_dir: str, cache_dir: str):
    """Open a teacher weight directory as a streaming source.

    .safetensors -> true mmap source (no copy). .npz -> one-time streaming
    repack into the shard format (one array in RAM at a time), then a
    TensorSource over the repacked shards. Raises informatively when the
    directory holds neither.
    """
    teacher_dir = os.path.abspath(teacher_dir)
    st_files = sorted(f for f in os.listdir(teacher_dir)
                      if f.endswith(".safetensors"))
    npz_files = sorted(f for f in os.listdir(teacher_dir)
                       if f.endswith(".npz"))
    if st_files:
        return SafetensorsSource(teacher_dir)
    if npz_files:
        shard_root = ensure_dir(os.path.join(cache_dir, "repacked_shards"))
        man_path = os.path.join(shard_root, "manifest.json")
        if not os.path.exists(man_path):
            import zipfile
            writer = ShardWriter(shard_root)
            for fname in npz_files:
                specs = _scan_npz(os.path.join(teacher_dir, fname))
                zf = zipfile.ZipFile(os.path.join(teacher_dir, fname))
                for name, _shape, _dtype, _nbytes in specs:
                    # one array in RAM at a time — the streaming invariant
                    with zf.open(name + ".npy") as f:
                        buf = f.read()
                    import io
                    arr = np.load(io.BytesIO(buf))
                    writer.add(f"{fname}:{name}",
                               np.ascontiguousarray(arr))
                    del arr, buf
                zf.close()
            writer.finalize("teacher-repacked")
        return TensorSource(shard_root)
    raise ValueError(
        f"no .safetensors or .npz weights in {teacher_dir}")


# ---------------------------------------------------------------------------
# Streaming compression: per-window truncated SVD under the planned budget
# ---------------------------------------------------------------------------

def stream_compress(source, plan: dict, memory_budget: str, out_dir: str, *,
                    resume: bool = False, fault_after: int = -1,
                    dtype=np.float32) -> dict:
    """End-to-end out-of-core compression.

    For each tensor: mmap -> recursive windows (each <= budget) ->
    truncated SVD at the tensor's planned rank share -> emit low-rank
    factor pairs (A = U_r S_r, B = V_r^T) as student shards. Student
    parameter count = sum of factor params, honestly accounted against
    the plan. Per-tensor checkpoint + per-tensor index flush; resume
    continues after a crash with zero emitted-data loss.
    """
    workdir = ensure_dir(out_dir)
    driver = OutOfCoreDriver(
        source, workdir, memory_budget=memory_budget,
        target_params=plan.get("total_target_params", 8.0e9),
        cap_params=plan.get("cap", 8.5e9))
    stats: Dict[str, dict] = {}
    rep_path = os.path.join(workdir, "stream_compress_report.json")
    # on resume, keep the previous run's per-tensor stats for tensors we
    # skip this time (otherwise a resume-noop would report empty stats)
    prior_stats: Dict[str, dict] = {}
    if resume and os.path.exists(rep_path):
        try:
            with open(rep_path) as f:
                prior_stats = json.load(f).get("tensor_stats", {})
        except (json.JSONDecodeError, OSError):
            prior_stats = {}

    def transform(name, data, w_target, ws, meta):
        parent = meta["tensor"]
        W = np.asarray(data)
        st = stats.setdefault(parent, {"windows": 0, "e_num": 0.0,
                                       "e_den": 0.0, "rank": 0})
        if W.ndim == 2 and min(W.shape) > 1:
            m, n = W.shape
            R = max(1, int(w_target // (m + n)))
            st["rank"] = max(st["rank"], R)
            Wd = W.astype(np.float64)
            U, s, Vt = np.linalg.svd(Wd, full_matrices=False)
            r = max(1, min(m, n, R, s.size))
            A = (U[:, :r] * s[:r]).astype(dtype)
            B = Vt[:r, :].astype(dtype)
            st["e_num"] += float((s[:r] ** 2).sum())
            st["e_den"] += float((s ** 2).sum())
            st["windows"] += 1
            return {"A": A, "B": B}
        st["windows"] += 1
        return {"raw": np.asarray(W).astype(dtype)}

    t0 = time.perf_counter()
    res = driver.execute(plan, transform, resume=resume,
                         fault_after=fault_after)
    dt = time.perf_counter() - t0
    # per-tensor retained energy (measured); re-run tensors overwrite
    for _name, st in stats.items():
        st["retained_energy"] = (st["e_num"] / st["e_den"]
                                 if st["e_den"] > 0 else 1.0)
        del st["e_num"], st["e_den"]
    merged_stats = {**prior_stats, **stats}
    res.update({"elapsed_s": dt, "tensor_stats": merged_stats,
                "source_params": source.total_params,
                "compression_ratio": (res["index_total_params"]
                                      / max(1, source.total_params))})
    rep_path = os.path.join(workdir, "stream_compress_report.json")
    with open(rep_path, "w") as f:
        json.dump(res, f, indent=2, default=float)
    res["report"] = rep_path
    return res
