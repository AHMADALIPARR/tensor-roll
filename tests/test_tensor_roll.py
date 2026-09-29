# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test suite for tensor-roll. Every test asserts measured behavior —
no fabricated numbers, no stubbed calls. Run: python3 -m unittest discover -s tests"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tensor_roll import TensorRoll
from tensor_roll import core as C
from tensor_roll import qsim as Q
from tensor_roll import model as M
from tensor_roll import distill as D
from tensor_roll import quant as QT
from tensor_roll import trq
from tensor_roll import boundary as BD
from tensor_roll import sandbox as SB


class TestTensorRollCore(unittest.TestCase):
    def test_metrics_known_matrix(self):
        W = np.eye(4, dtype=np.float64) * 2.0
        m = C.tensor_metrics(W)
        self.assertEqual(m["params"], 16)
        self.assertAlmostEqual(m["norm"], 4.0)
        self.assertEqual(m["rank"], 4)
        self.assertAlmostEqual(m["spectral"], 0.25)  # top sv^2 / total
        self.assertGreaterEqual(m["entropy"], 0.0)

    def test_all_decision_classes_evaluated(self):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((16, 16))
        opts = C.evaluate_options("t", W, quantum_policy="explore",
                                  qsim_encode=Q.qsim_encode_for_roll, rank=4)
        classes = {o["decision"] for o in opts}
        for want in ("PRESERVE", "QUANTIZE", "FACTORIZE", "MERGED",
                     "ROUTED", "RECONSTRUCTED", "PRUNED", "QUANTUM-ENCODED"):
            # MERGED only applies to 1-D; the rest must appear for 2-D
            if want != "MERGED":
                self.assertIn(want, classes, f"missing {want}")
        # every option's energy/error is a measured finite number
        for o in opts:
            self.assertTrue(np.isfinite(o["retained_energy"]))
            self.assertTrue(np.isfinite(o["reconstruction_error"]))
            self.assertGreaterEqual(o["cost_params"], 0)

    def test_merged_for_1d(self):
        W = np.random.default_rng(1).standard_normal(24)
        opts = C.evaluate_options("b", W)
        self.assertIn("MERGED", {o["decision"] for o in opts})

    def test_hard_budget(self):
        rng = np.random.default_rng(2)
        W = rng.standard_normal((32, 16))
        res = TensorRoll(W, 0, 2, 4, "explore",
                         qsim_encode=Q.qsim_encode_for_roll,
                         budget_params=128.0)
        self.assertLessEqual(res["total_cost_params"], 128.0)
        self.assertEqual(res["n_leaves"], 4)
        self.assertTrue(res["decisions"])

    def test_first_class_signature(self):
        import inspect
        sig = inspect.signature(TensorRoll)
        params = list(sig.parameters)
        self.assertEqual(params[:5], ["T", "axis", "depth", "rank",
                                     "quantum_policy"])

    def test_dispatch_matmul_honest_label(self):
        A = np.random.default_rng(3).standard_normal((8, 8))
        B = np.random.default_rng(4).standard_normal((8, 8))
        Cc, rep = C.dispatch_matmul(A, B)
        self.assertTrue(np.allclose(Cc, A @ B))
        self.assertEqual(rep["backend"], "cpu-numpy")


class TestQsim(unittest.TestCase):
    def test_encode_normalized(self):
        rng = np.random.default_rng(5)
        block = rng.standard_normal((8, 8))
        payload, rep = Q.tensor_roll_encode(block, encoding="amplitude")
        self.assertAlmostEqual(float(np.linalg.norm(payload["sv"])), 1.0,
                               places=12)
        self.assertEqual(rep["backend"], "qsim-classical")
        self.assertIn("statevector", rep["device"])

    def test_measure_counts_sum_to_shots(self):
        rng = np.random.default_rng(6)
        block = rng.standard_normal((8, 8))
        payload, _ = Q.tensor_roll_encode(block)
        payload, rep = Q.tensor_roll_measure(payload, shots=512, seed=1)
        self.assertEqual(sum(payload["counts"].values()), 512)
        self.assertEqual(rep["measurement_stats"]["n_outcomes"],
                         len(payload["counts"]))

    def test_reconstruct_report(self):
        rng = np.random.default_rng(7)
        block = rng.standard_normal((8, 8))
        payload, _ = Q.tensor_roll_encode(block)
        payload, _ = Q.tensor_roll_measure(payload, shots=2048, seed=2)
        rec, rep = Q.tensor_roll_reconstruct(payload)
        self.assertEqual(rec.shape, (8, 8))
        self.assertIn("reconstruction_error", rep)
        self.assertTrue(np.isfinite(C.relative_error(block, rec)))

    def test_recursive_summary(self):
        rng = np.random.default_rng(8)
        block = rng.standard_normal((8, 8))
        What, summary = Q.tensor_roll_recursive(block, depth=2, shots=256,
                                                seed=3)
        self.assertEqual(summary["kernel"], "tensor_roll_recursive[summary]")
        self.assertEqual(len(summary["measurement_stats"]["levels"]), 2)
        self.assertIn("counts", summary)
        self.assertIn("n_qubits", summary)

    def test_angle_encoding_cap(self):
        block = np.zeros((8, 8))
        with self.assertRaises(ValueError):
            Q.tensor_roll_encode(block, encoding="angle")


class TestQuant(unittest.TestCase):
    def setUp(self):
        self.W = np.random.default_rng(9).standard_normal((16, 16)).astype(
            np.float32)

    def test_fp16_roundtrip(self):
        q, meta = QT.quantize_tensor(self.W, "fp16")
        Wd = QT.dequantize_tensor(q, meta)
        self.assertLess(C.relative_error(self.W, Wd), 2e-2)

    def test_bf16_roundtrip(self):
        q, meta = QT.quantize_tensor(self.W, "bf16")
        Wd = QT.dequantize_tensor(q, meta)
        self.assertLess(C.relative_error(self.W, Wd), 2e-2)

    def test_int8_roundtrip(self):
        q, meta = QT.quantize_tensor(self.W, "int8")
        Wd = QT.dequantize_tensor(q, meta)
        self.assertLess(C.relative_error(self.W, Wd), 5e-2)

    def test_int4_roundtrip(self):
        q, meta = QT.quantize_tensor(self.W, "int4")
        Wd = QT.dequantize_tensor(q, meta)
        self.assertLess(C.relative_error(self.W, Wd), 0.35)
        # packed nibbles must not be miscounted as parameters
        self.assertEqual(int(np.prod(meta["shape"])), self.W.size)


class TestTrq(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.W = np.random.default_rng(10).standard_normal((8, 8)).astype(
            np.float32)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _write(self, fmt):
        q, qmeta = QT.quantize_tensor(self.W, fmt)
        p = os.path.join(self.d, f"t_{fmt}.trq")
        trq.write_trq(p, arch={"name": "test", "format": fmt},
                      tensors={"w": (np.ascontiguousarray(q), fmt, qmeta)},
                      provenance={"test": True},
                      roll_meta={"w": {"decision": "QUANTIZE"}})
        return p

    def test_roundtrip_all_formats(self):
        for fmt in ("fp32", "fp16", "bf16", "int8", "int4"):
            p = self._write(fmt)
            header, tensors = trq.read_trq(p)
            arr, meta = tensors["w"]
            Wd = QT.dequantize_tensor(arr, meta)
            self.assertEqual(header["version"], trq.VERSION)
            self.assertTrue(np.allclose(Wd, self.W, atol=0.3),
                            f"roundtrip failed for {fmt}")

    def test_corruption_rejected(self):
        p = self._write("int8")
        with open(p, "r+b") as f:
            f.seek(200)
            b = f.read(1)
            f.seek(200)
            f.write(bytes([b[0] ^ 0xFF]))
        with self.assertRaises(ValueError):
            trq.read_trq(p)

    def test_codebook_roundtrip(self):
        # codebook-quantized tensor: indices payload + codebook in meta
        rng = np.random.default_rng(15)
        W = rng.standard_normal((8, 8)).astype(np.float32)
        # rebuild a codebook explicitly for the artifact
        flat = W.ravel().astype(np.float64)
        cents = np.quantile(flat, np.linspace(0, 1, 16))
        assign = np.abs(flat[:, None] - cents[None, :]).argmin(axis=1)
        idx = assign.astype(np.uint8).reshape(8, 8)
        p = os.path.join(self.d, "t_codebook.trq")
        trq.write_trq(p, arch={"name": "test"},
                      tensors={"w": (np.ascontiguousarray(idx), "int8",
                                     {"quant": "codebook-vq-k16",
                                      "codebook": cents})},
                      provenance={"test": True})
        header, tensors = trq.read_trq(p)
        arr, meta = tensors["w"]
        self.assertIn("codebook", meta)
        self.assertEqual(meta["codebook"].shape, (16,))
        Wd = meta["codebook"][arr]
        # the artifact genuinely carries a usable codebook
        self.assertLess(C.relative_error(W, Wd), 0.5)

    def test_header_contents(self):
        p = self._write("fp16")
        header, _ = trq.read_trq(p)
        for key in ("arch", "n_tensors", "provenance", "roll_meta", "version"):
            self.assertIn(key, header)
        self.assertEqual(header["n_tensors"], 1)


class TestModel(unittest.TestCase):
    def test_numeric_gradient(self):
        # finite-difference check of the from-scratch backward pass
        cfg = M.Config(vocab=16, d=16, heads=2, layers=1, dff=32, seq=8)
        rng = np.random.default_rng(11)
        params = M.init_params(rng, cfg)
        ids = M.gen_data(rng, 2, cfg)
        logits, cache = M.forward(params, cfg, ids[:, :-1])
        _, dlogits = M.ce_loss(logits, ids[:, 1:])
        ana = M.backward(params, cfg, cache, dlogits)
        eps = 1e-4
        max_rel = 0.0
        for k in ("emb", "L0.Wq"):
            p = params[k]
            idx = (0, 0) if p.ndim == 2 else (0,)
            orig = float(p[idx])
            p[idx] = orig + eps
            lp, _ = M.ce_loss(M.forward(params, cfg, ids[:, :-1])[0],
                              ids[:, 1:])
            p[idx] = orig - eps
            lm, _ = M.ce_loss(M.forward(params, cfg, ids[:, :-1])[0],
                              ids[:, 1:])
            p[idx] = orig
            num = (lp - lm) / (2 * eps)
            a = float(ana[k][idx])
            max_rel = max(max_rel,
                          abs(a - num) / (abs(a) + abs(num) + 1e-8))
        self.assertLess(max_rel, 0.05)

    def test_training_reduces_loss(self):
        cfg = M.Config(vocab=16, d=16, heads=2, layers=1, dff=32, seq=8)
        rng = np.random.default_rng(12)
        params = M.init_params(rng, cfg)
        opt = M.Adam(params)
        l0 = M.eval_metrics(params, cfg, batches=2, batch=32)["loss"]
        for _ in range(30):
            b = M.gen_data(rng, 16, cfg)
            logits, cache = M.forward(params, cfg, b[:, :-1])
            _, dlogits = M.ce_loss(logits, b[:, 1:])
            grads = M.backward(params, cfg, cache, dlogits)
            opt.step(params, grads)
        l1 = M.eval_metrics(params, cfg, batches=2, batch=32)["loss"]
        self.assertLess(l1, l0)


class TestDistill(unittest.TestCase):
    def test_all_losses_finite(self):
        cfg_t = M.Config(vocab=16, d=16, heads=2, layers=2, dff=32, seq=8)
        cfg_s = M.Config(vocab=16, d=16, heads=2, layers=1, dff=32, seq=8)
        rng = np.random.default_rng(13)
        tp = M.init_params(rng, cfg_t)
        sp = M.init_params(rng, cfg_s)
        ids = M.gen_data(rng, 8, cfg_t)
        total, terms, grads = D.distill_step(sp, cfg_s, tp, cfg_t, ids,
                                            lambdas=D.DEFAULT_LAMBDAS)
        self.assertTrue(np.isfinite(total))
        for k in ("task", "logits", "hidden", "attention", "embedding",
                  "reconstruction", "roll"):
            self.assertIn(k, terms)
            self.assertTrue(np.isfinite(terms[k]), k)

    def test_recursive_metrics(self):
        cfg_t = M.Config(vocab=16, d=16, heads=2, layers=1, dff=32, seq=8)
        rng = np.random.default_rng(14)
        tp = M.init_params(rng, cfg_t)
        sp = {k: v * 0.9 for k, v in tp.items()}
        m = D.recursive_metrics(sp, tp)
        self.assertIn("model", m)
        for key in ("rel_err", "cosine_sim", "kl_div", "retained_energy"):
            self.assertIn(key, m["model"])


class TestBoundary(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_trace_roundtrip(self):
        p = os.path.join(self.d, "b.json")
        counts = {"000": 300, "111": 212}
        BD.write_trace(p, "tensor_roll_recursive", 3, 512, counts, depth=2)
        back = BD.read_trace(p)
        self.assertEqual(back["counts"], counts)
        self.assertGreater(back["n_gates"], 0)
        amps = BD.counts_to_amplitudes(back)
        self.assertEqual(len(amps), 8)

    def test_corruption_rejected(self):
        p = os.path.join(self.d, "b.json")
        BD.write_trace(p, "tensor_roll_measure", 2, 100, {"00": 100})
        with open(p, "r") as f:
            doc = json.load(f)
        doc["trace"]["shots"] = 999
        with open(p, "w") as f:
            json.dump(doc, f)
        with self.assertRaises(ValueError):
            BD.read_trace(p)


class TestSandbox(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.sup = SB.SandboxSupervisor(os.path.join(self.d, "sb"))

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_allow_echo(self):
        rec = self.sup.run("echo hello-tensor-roll")
        self.assertTrue(rec["allowed"])
        self.assertIn("hello-tensor-roll", rec["stdout"])
        self.assertTrue(os.path.exists(self.sup.audit.path))

    def test_deny_shadow(self):
        rec = self.sup.run("cat /etc/shadow")
        self.assertFalse(rec["allowed"])

    def test_deny_curl(self):
        rec = self.sup.run("curl http://example.com")
        self.assertFalse(rec["allowed"])

    def test_audit_log_written(self):
        self.sup.run("echo x")
        with open(self.sup.audit.path) as f:
            lines = f.readlines()
        self.assertTrue(lines)
        json.loads(lines[-1])  # valid JSONL


class TestDemo(unittest.TestCase):
    def test_demo_fast_end_to_end(self):
        d = tempfile.mkdtemp(prefix="trdemo-test-")
        try:
            r = subprocess.run(
                [sys.executable, "-m", "tensor_roll.cli", "--workdir", d,
                 "demo", "--fast"],
                capture_output=True, text=True, timeout=600,
                cwd=os.path.join(os.path.dirname(__file__), ".."))
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            for f in ("student_fp32.trq", "student_int8.trq",
                      "student_int4.trq"):
                self.assertTrue(os.path.exists(os.path.join(d, "trq", f)),
                                f)
            self.assertIn("Demo complete", r.stdout)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
