# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""F5 resolution experiment: diagnose the 3/24 token agreement, attempt a
principled fix, measure the result.

Protocol (all measured, fixed seeds, no cherry-picking):
  V0: baseline demo --fast pipeline (control).
  Diagnose: divergence trace + first divergence + component ablations on V0.
  Section-6 check: core.select_plan vs fidelity.fidelity_aware_select on the
      same roll leaves — do the decisions differ, and does the
      fidelity-aware utility change anything measurable?
  V1: same init, distillation stages 2 (hidden-state) and 3 (logit) rerun
      at 3x steps — tests whether distillation was underpowered vs whether
      capacity/init is the bottleneck.
  Report: agreement/24 and mean teacher-KL for V0 and V1.

Whatever the outcome — improvement, null, or regression — is reported.
Failed experiments are valid output.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_roll import cli as CLI
from tensor_roll import core as C
from tensor_roll import fidelity as F
from tensor_roll import model as M
from tensor_roll import planner as P
from tensor_roll import search as S
from tensor_roll.util import write_json

PROMPTS = [[3, 1, 4, 1], [7, 7, 2, 9], [0, 5, 5, 0]]
N_TOKENS = 24


def ns(**kw):
    a = argparse.Namespace()
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def measure_agreement_kl(t_params, t_cfg, s_params, s_cfg):
    agrees, n, kls = 0, 0, []
    for pr in PROMPTS:
        tt, _, _ = M.generate(t_params, t_cfg, pr, N_TOKENS)
        st, _, _ = M.generate(s_params, s_cfg, pr, N_TOKENS)
        agrees += sum(a == b for a, b in zip(tt, st))
        n += len(tt)
        rng = np.random.default_rng(4242)
        for _ in range(2):
            ids = M.gen_data(rng, 32, s_cfg)
            tl, _ = M.forward(t_params, t_cfg, ids[:, :-1])
            sl, _ = M.forward(s_params, s_cfg, ids[:, :-1])
            kls.append(float(M.kl_divergence_mean(tl, sl)))
    return {"agreement": f"{agrees}/{n}", "agree_frac": agrees / n,
            "mean_kl": float(np.mean(kls))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--steps-t", type=int, default=250)
    ap.add_argument("--steps-d", type=int, default=40)
    args = ap.parse_args()
    wdir = args.workdir
    os.makedirs(wdir, exist_ok=True)
    rep = {"protocol": "V0 baseline vs V1 focused-distillation",
           "prompts": PROMPTS, "n_tokens": N_TOKENS}

    # ---- teacher ------------------------------------------------------
    print("== teacher ==", flush=True)
    CLI.cmd_ingest(ns(workdir=wdir, scale="tiny", steps=args.steps_t,
                      batch=128, lr=3e-3, seed=11))
    t_params, t_cfg = CLI._teacher(wdir)

    # ---- roll: section-6 comparison -----------------------------------
    print("== roll: select_plan vs fidelity_aware_select ==", flush=True)
    stats = M.measure_tensor_stats(t_params, t_cfg)
    trees, leaves = S.run_roll_analysis(
        t_params, depth=2, quantum_policy="explore", stats=stats)
    budget = M.count_params(t_params) * S.BUDGET_RATIO
    plan0 = C.select_plan(leaves, budget)
    sens_rows = [{"tensor": l.name, "shape": [1], "dtype": "float32",
                 "params": 1, "bytes": 4, "shard": "x", "is_matrix": False}
                for l in leaves]
    sens = {l.name: float(np.clip(
        l.metrics.get("spectral", 0.5), 0, 1)) for l in leaves}
    plan1 = F.fidelity_aware_select(leaves, budget, sens, lam=1.0)
    diff = sum(1 for l in leaves
               if plan0[l.name]["decision"] != plan1[l.name]["decision"])
    rep["section6"] = {
        "n_leaves": len(leaves),
        "baseline_decisions": {d: sum(1 for l in leaves
                                     if plan0[l.name]["decision"] == d)
                               for d in set(plan0[l.name]["decision"]
                                            for l in leaves)},
        "fidelity_aware_decisions": {d: sum(
            1 for l in leaves if plan1[l.name]["decision"] == d)
            for d in set(plan1[l.name]["decision"] for l in leaves)},
        "n_decision_diffs": diff,
        "baseline_cost": plan0["_total_cost_params"],
        "fidelity_cost": plan1["_total_cost_params"],
    }
    print(f"  leaves={len(leaves)} decision_diffs={diff} "
          f"cost0={plan0['_total_cost_params']} "
          f"cost1={plan1['_total_cost_params']}", flush=True)
    for l in leaves:
        l.decision = plan0[l.name]  # baseline drives the pipeline

    # ---- compress + train + finetune (V0) ------------------------------
    print("== compress ==", flush=True)
    CLI.cmd_compress(ns(workdir=wdir, budget_ratio=S.BUDGET_RATIO, seed=11))
    print("== train (V0) ==", flush=True)
    CLI.cmd_train(ns(workdir=wdir, steps=args.steps_d, batch=128, lr=3e-3,
                     seed=12, stage="all"))
    print("== finetune (V0) ==", flush=True)
    CLI.cmd_finetune(ns(workdir=wdir, steps=args.steps_d, batch=128, seed=13))
    s_params = CLI._load_params(os.path.join(wdir, "student_final.npz"))
    s_cfg = M.Config.from_dict(
        CLI.read_json(os.path.join(wdir, "student_cfg.json")))
    rep["V0"] = measure_agreement_kl(t_params, t_cfg, s_params, s_cfg)
    print(f"  V0: {rep['V0']}", flush=True)

    # ---- diagnose V0 ----------------------------------------------------
    print("== divergence trace (V0) ==", flush=True)
    tr = F.divergence_trace(t_params, t_cfg, s_params, s_cfg,
                            PROMPTS[0], max_tokens=N_TOKENS)
    first = F.first_divergence(tr)
    rep["trace_V0"] = {
        "agreement": tr["token_agreement"],
        "first_divergence": first,
        "mean_kl": float(np.mean([p["kl_div"] for p in tr["positions"]])),
        "mean_js": float(np.mean([p["js_div"] for p in tr["positions"]])),
    }
    print(f"  first divergence: position={first.get('position')} "
          f"layer={first.get('first_diverging_layer')} "
          f"KL={first.get('kl_div'):.3f}", flush=True)
    print("== ablations (V0) ==", flush=True)
    abl = F.ablation_report(t_params, t_cfg, s_params, s_cfg,
                            PROMPTS, max_tokens=16)
    rep["ablation_V0"] = abl["ranking"]
    for r in abl["ranking"][:4]:
        print(f"  {r['component']:24s} d_agree={r['delta_agree']:+.3f} "
              f"d_kl={r.get('delta_kl', 0):+.3f}", flush=True)

    # ---- V1: focused extra distillation ---------------------------------
    print("== V1: extra hidden/logit distillation ==", flush=True)
    CLI.cmd_train(ns(workdir=wdir, steps=args.steps_d * 3, batch=128,
                     lr=3e-3, seed=12, stage="2"))
    CLI.cmd_train(ns(workdir=wdir, steps=args.steps_d * 3, batch=128,
                     lr=3e-3, seed=12, stage="3"))
    s_params1 = CLI._load_params(os.path.join(wdir, "student_distilled.npz"))
    rep["V1"] = measure_agreement_kl(t_params, t_cfg, s_params1, s_cfg)
    print(f"  V1: {rep['V1']}", flush=True)
    rep["delta_agree"] = rep["V1"]["agree_frac"] - rep["V0"]["agree_frac"]
    rep["delta_kl"] = rep["V1"]["mean_kl"] - rep["V0"]["mean_kl"]

    out = os.path.join(wdir, "fidelity_resolution_report.json")
    write_json(out, rep)
    print(f"wrote {out}", flush=True)
    print(json.dumps({k: rep[k] for k in ("V0", "V1", "section6")
                      if k in rep}, indent=2))


if __name__ == "__main__":
    main()
