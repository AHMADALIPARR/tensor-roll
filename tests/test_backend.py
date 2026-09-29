# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the backend-honesty chunk: doctor, IR, conformance, and the
integration-state audit. Every test asserts measured behavior — no
fabricated numbers. Fast: the whole file runs in seconds."""
import hashlib
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tensor_roll import doctor as DR
from tensor_roll import ir as IR
from tensor_roll import conformance as CF
from tensor_roll import boundary as BD


class TestDoctor(unittest.TestCase):
    def test_report_structure(self):
        rep = DR.doctor_report()
        for key in ("capabilities", "sources", "execution", "gpu",
                    "integration"):
            self.assertIn(key, rep)
        cap = rep["capabilities"]
        self.assertEqual(cap["CPU"]["state"], "AVAILABLE")
        self.assertEqual(cap["NumPy"]["state"], "AVAILABLE")
        # This machine: no GPU, no CUDA-Q toolchain, no Q# toolchain.
        self.assertEqual(cap["CUDA"]["state"], "UNAVAILABLE")
        self.assertEqual(cap["CUDA-Q"]["state"], "UNAVAILABLE")
        self.assertEqual(cap["Q#"]["state"], "UNAVAILABLE")
        self.assertEqual(rep["gpu"], "NONE")
        for label in ("CUDA", "CUDA-Q", "Q#"):
            self.assertTrue(cap[label].get("reason"),
                            f"{label} must carry a reason, not just a state")

    def test_cpu_execution_passes(self):
        rep = DR.doctor_report()
        cpu = rep["execution"]["CPU"]
        self.assertEqual(cpu["state"], "EXECUTED")
        self.assertEqual(cpu["result"], "PASS")
        self.assertGreater(cpu["matmul_ms"], 0)

    def test_others_not_executed(self):
        rep = DR.doctor_report()
        for label in ("CUDA", "CUDA-Q", "Q#"):
            rec = rep["execution"][label]
            self.assertEqual(rec["state"], "NOT EXECUTED")
            self.assertTrue(rec["reason"])
            self.assertIn(rec["source"], ("VERIFIED", "PRESENT"))

    def test_sources_verified(self):
        for label, rec in DR.verify_sources().items():
            self.assertIn(rec["state"], ("VERIFIED", "PRESENT"), label)
        # On this box all three sources carry the AGPL header and the
        # expected identifiers, so VERIFIED is the honest state.
        states = {l: r["state"] for l, r in DR.verify_sources().items()}
        self.assertEqual(states, {"CUDA source": "VERIFIED",
                                 "CUDA-Q source": "VERIFIED",
                                 "Q# source": "VERIFIED"})

    def test_render_shape(self):
        text = DR.render_doctor(DR.doctor_report())
        expected = "\n".join([
            "Tensor Roll Backend Report",
            "CPU            AVAILABLE",
            "NumPy          AVAILABLE",
            "CUDA           UNAVAILABLE",
            "CUDA-Q         UNAVAILABLE",
            "Q#             UNAVAILABLE",
            "GPU            NONE",
            "CUDA source    VERIFIED",
            "CUDA-Q source  VERIFIED",
            "Q# source      VERIFIED",
            "Execution:",
            "CPU            PASS",
            "CUDA           NOT EXECUTED",
            "CUDA-Q         NOT EXECUTED",
            "Q#             NOT EXECUTED",
            "",
        ])
        self.assertEqual(text, expected)


class TestIR(unittest.TestCase):
    def _plan(self):
        return {
            "w1": {"decision": "QUANTIZE",
                   "cost_params": {"bits": 8}},
            "w2": {"decision": "QUANTUM-ENCODED",
                   "cost_params": {"n_qubits": 2,
                                   "angles": [0.3, 0.6],
                                   "shots": 64, "seed": 1}},
            "w3": {"decision": "PRESERVE", "cost_params": {}},
        }

    def test_build_ir_structure(self):
        ops = IR.build_ir(self._plan())
        by_out = [o.op for o in ops]
        # w1: LOAD ROLL QUANTIZE EMIT
        self.assertEqual(by_out[:4], ["LOAD", "ROLL", "QUANTIZE", "EMIT"])
        # w2: LOAD ROLL ROTATE ENTANGLE MEASURE RECONSTRUCT EMIT
        self.assertEqual(by_out[4:11],
                         ["LOAD", "ROLL", "ROTATE", "ENTANGLE", "MEASURE",
                          "RECONSTRUCT", "EMIT"])
        # w3: LOAD ROLL EMIT (pass-through, decision in meta)
        self.assertEqual(by_out[11:], ["LOAD", "ROLL", "EMIT"])
        self.assertEqual(ops[-1].meta["decision"], "PRESERVE")

    def test_json_roundtrip(self):
        ops = IR.build_ir(self._plan())
        back = IR.from_json(IR.to_json(ops))
        self.assertEqual([o.op for o in back], [o.op for o in ops])
        self.assertEqual([o.params for o in back], [o.params for o in ops])
        self.assertEqual([o.meta for o in back], [o.meta for o in ops])
        self.assertEqual([o.inputs for o in back], [o.inputs for o in ops])
        self.assertEqual([o.outputs for o in back], [o.outputs for o in ops])

    def test_lower_cpu_gemm_numeric(self):
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        B = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ops = [
            IR.IROp("LOAD", [], ["A"], {"data": A, "shape": [2, 3]}, {}),
            IR.IROp("LOAD", [], ["B"], {"data": B, "shape": [3, 2]}, {}),
            IR.IROp("GEMM", ["A", "B"], ["C"], {}, {}),
            IR.IROp("EMIT", ["C"], ["out"], {}, {}),
        ]
        recs = IR.lower(ops, "cpu")
        self.assertTrue(all(r["state"] == "EXECUTED" for r in recs))
        gemm = recs[2]
        self.assertEqual(gemm["backend"], "cpu-numpy")
        want = np.array(A) @ np.array(B)
        self.assertEqual(gemm["detail"]["shape"], [2, 2])
        self.assertAlmostEqual(gemm["detail"]["checksum"], float(want.sum()))
        emit = recs[3]
        self.assertEqual(emit["detail"]["sha256"],
                         hashlib.sha256(want.tobytes()).hexdigest())

    def test_lower_cpu_quantum_family(self):
        ops = [
            IR.IROp("LOAD", [], ["x"],
                    {"data": [0.5, -0.25, 0.75, 0.1], "shape": [4]}, {}),
            IR.IROp("ROTATE", ["x"], ["xr"],
                    {"angles": [0.2, 0.4]}, {}),
            IR.IROp("ENTANGLE", ["xr"], ["xe"], {}, {}),
            IR.IROp("MEASURE", ["xe"], ["xm"],
                    {"shots": 128, "seed": 3}, {}),
            IR.IROp("RECONSTRUCT", ["xm"], ["xhat"], {}, {}),
            IR.IROp("EMIT", ["xhat"], ["out"], {}, {}),
        ]
        recs = IR.lower(ops, "cpu")
        self.assertTrue(all(r["state"] == "EXECUTED" for r in recs),
                        [r for r in recs if r["state"] != "EXECUTED"])
        self.assertEqual(recs[1]["backend"], "qsim-classical")
        self.assertEqual(recs[3]["detail"]["shots"], 128)

    def test_lower_cudaq_not_executed(self):
        ok, _ = DR.cudaq_toolchain()
        if ok:
            self.skipTest("real CUDA-Q toolchain present; "
                          "NOT-EXECUTED path does not apply")
        ops = IR.build_ir(self._plan())
        recs = IR.lower(ops, "cuda-q")
        self.assertTrue(all(r["state"] == "NOT EXECUTED" for r in recs))
        self.assertTrue(all(r["reason"] for r in recs))
        self.assertTrue(all(r["source"] in ("VERIFIED", "PRESENT")
                            for r in recs))
        rot = next(r for r in recs if r["op"] == "ROTATE")
        self.assertEqual(rot["gate_trace_schema"], BD.TRACE_SCHEMA)
        self.assertTrue(rot["gate_trace"])
        self.assertIn("IR-specification only", rot["note"])

    def test_lower_unknown_backend(self):
        with self.assertRaises(ValueError):
            IR.lower([], "tpu")


class TestConformance(unittest.TestCase):
    def test_numpy_executed_and_matches(self):
        with tempfile.TemporaryDirectory() as d:
            doc = CF.run_conformance(d)
            path = os.path.join(d, "conformance.json")
            self.assertTrue(os.path.isfile(path))
            with open(path) as f:
                back = json.load(f)["backends"]
            rec = back["numpy"]
            self.assertEqual(rec["state"], "EXECUTED")
            self.assertTrue(rec["match"])
            self.assertTrue(all(rec["checks"].values()))
            self.assertEqual(set(back), {"numpy", "cuda", "cuda-q", "qsharp"})

    def test_others_not_executed_with_reasons(self):
        with tempfile.TemporaryDirectory() as d:
            doc = CF.run_conformance(d)
            back = doc["backends"]
            for name in ("cuda", "cuda-q", "qsharp"):
                rec = back[name]
                self.assertEqual(rec["state"], "NOT EXECUTED", name)
                self.assertTrue(rec["reason"], name)
                self.assertIsNone(rec["match"], name)
                for key in ("state", "reason", "outputs", "expected",
                            "match", "measured"):
                    self.assertIn(key, rec, f"{name}.{key}")

    def test_vectors_deterministic(self):
        self.assertEqual(CF.test_vectors(), CF.test_vectors())
        exp = CF.expected_outputs()
        # expected C hard-coded matches a fresh NumPy computation
        v = CF.test_vectors()
        C = (np.array(v["matmul"]["A"]) @ np.array(v["matmul"]["B"])).tolist()
        self.assertEqual(C, exp["matmul"]["C"])


class TestIntegrationState(unittest.TestCase):
    def test_audited_value(self):
        res = DR.integration_state()
        self.assertEqual(res["state"], "file-mediated")
        self.assertTrue(res["evidence"])
        self.assertTrue(any("tensor_roll/boundary.py:" in e
                            for e in res["evidence"]))
        # the audit must record that vendor execution here is source-only
        self.assertTrue(any("source-only" in e for e in res["evidence"]))

    def test_boundary_trace_still_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.json")
            BD.write_trace(p, "tensor_roll_encode", 2, 64,
                           {"00": 40, "11": 24}, producer="test")
            back = BD.read_trace(p)
            self.assertEqual(back["counts"], {"00": 40, "11": 24})
            with open(p, "a") as f:
                f.write("corrupt")
            with self.assertRaises(ValueError):
                BD.read_trace(p)


if __name__ == "__main__":
    unittest.main()
