# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Failure-resolution tests (F1-F6): out-of-core streaming, global planner,
fidelity instrumentation, INT4 characterization, CLI wiring.

Every test measures; none fabricates. The 31 pre-existing tests in
test_tensor_roll.py are untouched and must stay green alongside these.
"""

import io
import json
import os
import struct
import tempfile
import unittest

import numpy as np

from tensor_roll import fidelity as F
from tensor_roll import model as M
from tensor_roll import ooc
from tensor_roll import planner
from tensor_roll import quant as QT


def _tiny_cfg():
    return M.Config(vocab=16, d=16, heads=2, layers=1, dff=32, seq=8)


def _write_safetensors(path, tensors):
    """Minimal honest safetensors writer for tests (F32/F16 only)."""
    header = {}
    blobs = []
    off = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        dt = {np.dtype("float32"): "F32",
              np.dtype("float16"): "F16"}[arr.dtype]
        blob = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(blob)]}
        blobs.append(blob)
        off += len(blob)
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)


class TestShardIO(unittest.TestCase):
    def test_roundtrip_and_mmap(self):
        tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        W = rng.standard_normal((32, 24)).astype(np.float32)
        sw = ooc.ShardWriter(os.path.join(tmp, "sh"), shard_size_bytes=4096)
        sw.add("W", W)
        sw.finalize("t")
        src = ooc.TensorSource(os.path.join(tmp, "sh"))
        self.assertEqual(src.total_params, 32 * 24)
        rdr = src.reader()
        mt = rdr.read("W")
        self.assertIsInstance(mt.view, np.memmap)  # mmap, not RAM copy
        np.testing.assert_array_equal(np.asarray(mt.view), W)
        rdr.close()

    def test_windows_cover_exactly_within_budget(self):
        tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(1)
        W = rng.standard_normal((100, 7)).astype(np.float32)
        sw = ooc.ShardWriter(os.path.join(tmp, "sh"))
        sw.add("W", W)
        sw.finalize("t")
        src = ooc.TensorSource(os.path.join(tmp, "sh"))
        mt = src.reader().read("W")
        wins = list(mt.windows(1024))
        self.assertTrue(all(w.nbytes <= 1024 for w in wins))
        rec = ooc.reassemble(wins, W.shape, dtype=np.float32)
        np.testing.assert_array_equal(rec, W)
        # no-overlap check is inside reassemble (asserts); also count cells
        total = sum(w.data.size for w in wins)
        self.assertEqual(total, W.size)


class TestWorkspace(unittest.TestCase):
    def test_ceiling_enforced(self):
        ws = ooc.RollWorkspace(1000)
        ws.alloc("a", 600)
        with self.assertRaises(ooc.MemoryBudgetExceeded):
            ws.alloc("b", 500)
        ws.free("a")
        ws.alloc("b", 500)
        rep = ws.report()
        self.assertTrue(rep["within_budget"])
        self.assertEqual(rep["peak_bytes"], 600)


class TestVirtualManifest(unittest.TestCase):
    def test_30b_scale_without_weights(self):
        man = ooc.glimmer30b_manifest()
        self.assertGreaterEqual(man.total_params, 29e9)
        self.assertLess(man.total_params, 32e9)
        src = ooc.TensorSource.__new__(ooc.TensorSource)  # not needed
        with self.assertRaises(RuntimeError):
            man.reader()

    def test_virtual_manifest_reader_raises(self):
        man = ooc.glimmer30b_manifest()
        with self.assertRaises(RuntimeError):
            man.reader()


class TestStudentWriter(unittest.TestCase):
    def test_verify_and_corruption_detected(self):
        tmp = tempfile.mkdtemp()
        w = ooc.StudentWriter(os.path.join(tmp, "st"))
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        w.emit("t", a)
        w.finalize()
        self.assertTrue(ooc.StudentWriter.verify(os.path.join(tmp, "st"))["ok"])
        # corrupt one data byte on disk (last byte is always data)
        p = os.path.join(tmp, "st", "student-00000.npy")
        with open(p, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            f.write(b"\xff")
        with self.assertRaises(AssertionError):
            ooc.StudentWriter.verify(os.path.join(tmp, "st"))


class TestCheckpoint(unittest.TestCase):
    def test_roundtrip_and_corruption_rejected(self):
        tmp = tempfile.mkdtemp()
        cp = ooc.Checkpoint(os.path.join(tmp, "c.json"))
        cp.save({"done": ["a"], "n": 3})
        self.assertEqual(cp.load()["n"], 3)
        with open(cp.path) as f:
            doc = json.load(f)
        doc["state"]["n"] = 999
        with open(cp.path, "w") as f:
            json.dump(doc, f)
        with self.assertRaises(ValueError):
            cp.load()


def _tiny_source(tmp, tensors):
    sw = ooc.ShardWriter(os.path.join(tmp, "sh"))
    for k, v in tensors.items():
        sw.add(k, v)
    sw.finalize("t")
    return ooc.TensorSource(os.path.join(tmp, "sh"))


class TestDriverResume(unittest.TestCase):
    def test_crash_then_resume_matches_clean_run(self):
        rng = np.random.default_rng(2)
        tensors = {"a": rng.standard_normal((16, 12)).astype(np.float32),
                   "b": rng.standard_normal((8, 8)).astype(np.float32)}
        ident = lambda n, d, wt, ws, meta: np.asarray(d)  # noqa
        plan = {"tensors": {k: {"target_params": v.size}
                            for k, v in tensors.items()},
                "total_target_params": sum(v.size for v in tensors.values()),
                "cap": 8.5e9}
        # clean run
        tmp1 = tempfile.mkdtemp()
        drv1 = ooc.OutOfCoreDriver(_tiny_source(tmp1, tensors),
                                   os.path.join(tmp1, "w"),
                                   memory_budget="4KB")
        r1 = drv1.execute(plan, ident)
        # crashed run then resume
        tmp2 = tempfile.mkdtemp()
        drv2 = ooc.OutOfCoreDriver(_tiny_source(tmp2, tensors),
                                   os.path.join(tmp2, "w"),
                                   memory_budget="4KB")
        with self.assertRaises(RuntimeError):
            drv2.execute(plan, ident, fault_after=0)
        drv3 = ooc.OutOfCoreDriver(_tiny_source(tmp2, tensors),
                                   os.path.join(tmp2, "w"),
                                   memory_budget="4KB")
        r2 = drv3.execute(plan, ident, resume=True)
        self.assertEqual(r1["index_total_params"], r2["index_total_params"])
        self.assertEqual(r2["processed_tensors"], 1)  # only the remainder
        idx = ooc.StudentWriter.verify(os.path.join(tmp2, "w",
                                                    "student_shards"))
        self.assertTrue(idx["ok"])


class TestStreamCompress(unittest.TestCase):
    def test_safetensors_mmap_stream(self):
        tmp = tempfile.mkdtemp()
        tdir = os.path.join(tmp, "teacher")
        os.makedirs(tdir)
        rng = np.random.default_rng(3)
        W = rng.standard_normal((64, 48)).astype(np.float32)
        b = rng.standard_normal((48,)).astype(np.float32)
        _write_safetensors(os.path.join(tdir, "model.safetensors"),
                           {"W": W, "b": b})
        src = ooc.prepare_teacher_source(tdir, os.path.join(tmp, "cache"))
        self.assertEqual(src.total_params, 64 * 48 + 48)
        rows = planner.scan_rows(src.specs)
        plan = planner.plan(rows, target=2000)
        res = ooc.stream_compress(src, plan, "64KB",
                                  os.path.join(tmp, "out"))
        self.assertLessEqual(res["index_total_params"], plan["cap"])
        self.assertTrue(res["budget_ok"])
        for st in res["tensor_stats"].values():
            self.assertGreater(st["retained_energy"], 0.0)
            self.assertLessEqual(st["retained_energy"], 1.0 + 1e-9)
        self.assertLessEqual(res["workspace"]["peak_bytes"],
                             res["workspace"]["ceiling_bytes"])
        # student factors reconstruct approximately
        idx_path = os.path.join(tmp, "out", "student_shards",
                                "shard_index.json")
        with open(idx_path) as f:
            index = json.load(f)
        parts = {}
        for e in index["tensors"]:
            if e["name"].startswith("model.safetensors:W@"):
                arr = np.load(os.path.join(tmp, "out", "student_shards",
                                           e["file"]))
                parts[e["name"].rsplit(".", 1)[-1]] = arr
        rec = parts["A"] @ parts["B"]
        rel = np.linalg.norm(W - rec) / np.linalg.norm(W)
        self.assertLess(rel, 0.9)  # rank-17 approx of random 64x48


class TestPlanner(unittest.TestCase):
    def test_ratio_invariance(self):
        for n in (100_000, 1_000_000, 10_000_000):
            r = planner.check_ratio_invariance(n, tol=0.02, seed=7)
            self.assertTrue(r["within_tol"],
                            f"scale {n}: {r['achieved_ratio']}")

    def test_30b_plan_within_cap(self):
        import time
        man = ooc.glimmer30b_manifest()
        rows = planner.scan_rows(man.specs)
        t0 = time.perf_counter()
        p = planner.plan(rows)
        dt = time.perf_counter() - t0
        self.assertTrue(p["within_cap"])
        self.assertLessEqual(p["total_target_params"], planner.CAP_PARAMS)
        self.assertGreaterEqual(p["total_target_params"], 7e9)
        self.assertLess(dt, 60.0)

    def test_sensitivity_nonuniform(self):
        rows = [{"tensor": "sens", "shape": [64, 64], "dtype": "float32",
                 "params": 4096, "bytes": 16384, "shard": "x", "is_matrix": True},
                {"tensor": "insens", "shape": [64, 64], "dtype": "float32",
                 "params": 4096, "bytes": 16384, "shard": "x", "is_matrix": True}]
        p = planner.plan(rows, ratio=0.5,
                         sensitivity={"sens": 1.0, "insens": 0.0})
        rs = p["tensors"]["sens"]["ratio"]
        ri = p["tensors"]["insens"]["ratio"]
        self.assertGreater(rs, ri)

    def test_calibrate_sensitivity_measured(self):
        rng = np.random.default_rng(4)
        rows = [{"tensor": "m", "shape": [32, 32], "dtype": "float32",
                 "params": 1024, "bytes": 4096, "shard": "x", "is_matrix": True}]
        W = rng.standard_normal((32, 32))
        s = planner.calibrate_sensitivity(rows,
                                          sample_fn=lambda n: W[:8, :8])
        self.assertIn("m", s)
        self.assertGreaterEqual(s["m"], 0.0)
        self.assertLessEqual(s["m"], 1.0)


class TestFidelityMetrics(unittest.TestCase):
    def test_kl_js_zero_for_identical(self):
        x = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(F.kl_div(x, x), 0.0, places=9)
        self.assertAlmostEqual(F.js_div(x, x), 0.0, places=9)

    def test_kl_positive_js_bounded(self):
        x = np.array([3.0, 0.0, 0.0])
        y = np.array([0.0, 3.0, 0.0])
        self.assertGreater(F.kl_div(x, y), 0.0)
        j = F.js_div(x, y)
        self.assertGreater(j, 0.0)
        self.assertLessEqual(j, np.log(2) + 1e-9)

    def test_top_k(self):
        tk = F.top_k(np.array([0.0, 5.0, 1.0]), 2)
        self.assertEqual(tk[0][0], 1)
        self.assertEqual(len(tk), 2)


class TestDivergenceTrace(unittest.TestCase):
    def test_trace_structure_and_first_divergence(self):
        rng = np.random.default_rng(5)
        cfg = _tiny_cfg()
        tp = M.init_params(rng, cfg)
        sp = M.init_params(rng, cfg)
        tr = F.divergence_trace(tp, cfg, sp, cfg, [1, 2, 3], max_tokens=8)
        self.assertEqual(tr["n_tokens"], 8)
        self.assertEqual(len(tr["positions"]), 8)
        pos0 = tr["positions"][0]
        for k in ("teacher_token", "student_token", "teacher_topk",
                  "student_topk", "teacher_logits", "student_logits",
                  "kl_div", "js_div", "cosine_sim", "teacher_entropy",
                  "student_entropy", "hidden_div", "attn_div"):
            self.assertIn(k, pos0)
        a, b = tr["agreement_count"], tr["n_tokens"]
        self.assertEqual(tr["token_agreement"], f"{a}/{b}")
        first = F.first_divergence(tr)
        self.assertIn("position", first)
        lines = F.render_trace(tr, first)
        self.assertEqual(len(lines), 9)  # 8 tokens + summary


class _Leaf:
    def __init__(self, name, options):
        self.name = name
        self.options = options


class TestFidelitySelect(unittest.TestCase):
    def test_budget_and_uniform_penalty(self):
        leaves = [_Leaf("a", [
            {"cost_params": 100, "retained_energy": 0.9,
             "reconstruction_error": 0.01},
            {"cost_params": 200, "retained_energy": 0.99,
             "reconstruction_error": 0.001}]),
            _Leaf("b", [
                {"cost_params": 100, "retained_energy": 0.9,
                 "reconstruction_error": 0.01},
                {"cost_params": 200, "retained_energy": 0.95,
                 "reconstruction_error": 0.05}])]
        plan = F.fidelity_aware_select(leaves, 300,
                                       {"a": 1.0, "b": 1.0}, lam=1.0)
        self.assertLessEqual(plan["_total_cost_params"], 300)
        # penalty is uniform: same error+lambda -> same penalty class
        pa = plan["a"]["fidelity_penalty"]
        self.assertAlmostEqual(
            pa, F.fidelity_penalty(1.0, plan["a"]["reconstruction_error"]))
        u = F.utility_breakdown(1000, {"reconstruction": 10.0,
                                       "logit": 5.0, "hidden": 2.0,
                                       "task": 1.0})
        self.assertAlmostEqual(u["utility"], 982.0)


class TestInt4(unittest.TestCase):
    def test_schemes_roundtrip_and_search(self):
        rng = np.random.default_rng(6)
        W = (rng.standard_normal((48, 64)) * 3).astype(np.float32)
        rep = QT.int4_scheme_report(W)
        for k in ("per-tensor", "per-channel", "group-wise", "int8"):
            self.assertIn("mse", rep[k])
        # search picks the measured argmin
        mses = {r["group"]: r["mse"] for r in rep["group-wise"]["all"]}
        best = min(mses, key=mses.get)
        self.assertEqual(rep["group-wise"]["best_group"], best)
        # per-tensor round trip shape
        q, m = QT.quantize_int4_per_tensor(W)
        back = QT.dequantize_int4_per_tensor(q, m)
        self.assertEqual(back.shape, W.shape)
        # error map instrumentation
        em = QT.quant_error_map(W, back)
        for k in ("min", "max", "mean", "variance", "outlier_frac", "mse",
                  "cosine_sim", "rel_err", "per_channel_mse",
                  "per_group_mse"):
            self.assertIn(k, em)
        self.assertEqual(len(em["per_channel_mse"]), 48)

    def test_outlier_detection(self):
        rng = np.random.default_rng(9)
        clean = rng.standard_normal((8, 64)).astype(np.float32)
        dirty = clean.copy()
        dirty[0, 0] = 100.0  # planted outlier stretches the int4 scale
        qc, mc = QT.quantize_int4(clean)
        qd, md = QT.quantize_int4(dirty)
        em_c = QT.quant_error_map(clean, QT.dequantize_int4(qc, mc))
        em_d = QT.quant_error_map(dirty, QT.dequantize_int4(qd, md))
        # the outlier inflates measured error by an order of magnitude
        self.assertGreater(em_d["mse"], 10 * em_c["mse"])
        self.assertGreater(em_d["max"], 1.0)
        self.assertGreaterEqual(em_d["outlier_frac"], 0.0)

    def test_mixed_precision_plan_within_budget(self):
        rng = np.random.default_rng(7)
        params = {"W1": rng.standard_normal((32, 32)).astype(np.float32),
                  "W2": rng.standard_normal((16, 16)).astype(np.float32)}
        mp = QT.mixed_precision_plan(params, 10_000_000)
        self.assertTrue(mp["within_budget"])
        self.assertEqual(set(mp["plan"]), {"W1", "W2"})
        self.assertTrue(all(v in ("int4", "int8")
                            for v in mp["plan"].values()))

    def test_recovery_train_measured(self):
        rng = np.random.default_rng(8)
        cfg = _tiny_cfg()
        tp = M.init_params(rng, cfg)
        sp = {k: v.astype(np.float32) for k, v in
              M.init_params(rng, cfg).items()}
        rep = QT.recovery_train(sp, cfg, tp, cfg, steps=2, batch=8,
                                seed=0)
        for k in ("ppl_before", "ppl_after_train", "ppl_after_requant",
                  "improved"):
            self.assertIn(k, rep)
        self.assertIsInstance(rep["improved"], bool)
        self.assertTrue(np.isfinite(rep["ppl_after_requant"]))


class TestCLIWiring(unittest.TestCase):
    def test_doctor_and_compress_options(self):
        from tensor_roll import cli
        p = cli.build_parser()
        a = p.parse_args(["doctor"])
        self.assertEqual(a.cmd, "doctor")
        a = p.parse_args(["compress", "--out-of-core", "--resume",
                          "--teacher", "/tmp", "--target-params", "1000",
                          "--memory-budget", "64MB"])
        self.assertTrue(a.out_of_core)
        self.assertTrue(a.resume)
        self.assertEqual(a.teacher, "/tmp")
        self.assertEqual(a.target_params, 1000)
        self.assertEqual(a.memory_budget, "64MB")


if __name__ == "__main__":
    unittest.main()
