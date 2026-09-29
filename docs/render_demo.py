# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render docs/tensor-roll-demo.png: dark-terminal demo picture.

Recursive tensor blocks folding into a compact core, the pipeline chain
GLIMMER 30B -> TENSOR ROLL -> CUDA -> CUDA-Q -> Q# -> 8B NANO MODEL,
and real measured numbers from the tiny validation run. No generic
AI-brain imagery; everything drawn is the actual architecture.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

BG = "#0a0e14"
PANEL = "#11161f"
FG = "#d8dee9"
DIM = "#7b8798"
ACC = "#5ccfe6"
GRN = "#bae67e"
YLW = "#ffd580"
MAG = "#f28779"

W, H = 1600, 900
fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# title
ax.text(W / 2, 848, "TENSOR ROLL  \u2014  Recursive CUDA-Q Model Quantizer",
        color=FG, fontsize=30, ha="center", family="monospace", weight="bold")
ax.text(W / 2, 808,
        "GLIMMER 30B  ->  TENSOR ROLL  ->  CUDA  ->  CUDA-Q  ->  Q#  ->  8B NANO MODEL",
        color=ACC, fontsize=18, ha="center", family="monospace")

# ---- left: recursive folding blocks ----
ax.text(60, 748, "$ TensorRoll(T, axis, depth, rank, quantum_policy)",
        color=DIM, fontsize=14, family="monospace")
cx, cy = 330, 430
sizes = [400, 292, 208, 140, 88]
cols = [MAG, YLW, ACC, GRN, ACC]
for i, (s, c) in enumerate(zip(sizes, cols)):
    x0, y0 = cx - s / 2, cy - s / 2
    ax.add_patch(Rectangle((x0, y0), s, s, facecolor="none",
                           edgecolor=c, linewidth=2.4, alpha=0.9))
    if s > 120:  # tensor cell grid
        for gx in np.linspace(x0 + 24, x0 + s - 24, 4):
            ax.plot([gx, gx], [y0 + 10, y0 + s - 10], color=c, alpha=0.16, lw=1)
        for gy in np.linspace(y0 + 24, y0 + s - 24, 4):
            ax.plot([x0 + 10, x0 + s - 10], [gy, gy], color=c, alpha=0.16, lw=1)
    ax.text(x0 + 12, y0 + s - 24, f"depth {i}", color=c, fontsize=12,
            family="monospace", alpha=0.9)
    if i < len(sizes) - 1:
        s2 = sizes[i + 1]
        ax.add_patch(FancyArrowPatch((x0 + s - 26, y0 + 30),
                                     (cx + s2 / 2 - 34, cy - s2 / 2 + 40),
                                     arrowstyle="->", color=c, linewidth=1.6,
                                     mutation_scale=14))
# compact core
core = 56
ax.add_patch(Rectangle((cx - core / 2, cy - core / 2), core, core,
                       facecolor=GRN, edgecolor="none"))
ax.text(cx, cy, "8B", color=BG, fontsize=20, ha="center", va="center",
        family="monospace", weight="bold")
ax.text(cx, 150, "148 leaves  \u00b7  knapsack under P \u2264 8.5B",
        color=DIM, fontsize=14, ha="center", family="monospace")
ax.text(cx, 118, "PRESERVE / QUANTIZE / FACTORIZE / MERGED /",
        color=DIM, fontsize=12.5, ha="center", family="monospace")
ax.text(cx, 94, "ROUTED / RECONSTRUCTED / PRUNED / QUANTUM-ENCODED",
        color=DIM, fontsize=12.5, ha="center", family="monospace")

# ---- right: pipeline chain ----
ax.text(860, 748, "pipeline  (measured on 2 CPUs, no GPU)",
        color=DIM, fontsize=14, family="monospace")
stages = [
    ("tensor-roll ingest", "tiny teacher \u00b7 41.1K params", GRN),
    ("tensor-roll roll", "TensorRoll() \u00b7 148 leaves \u00b7 budget OK", ACC),
    ("tensor-roll qkernel", "qsim-classical \u00b7 recon err 0.168", YLW),
    ("boundary.json", "gate trace + counts \u00b7 sha256 verified", ACC),
    ("tensor-roll compress", "student arch \u00b7 11.4K (27.8% of teacher)", GRN),
    ("tensor-roll train", "stages 0\u20137 \u00b7 7 distillation losses", ACC),
    ("tensor-roll quantize", "fp16 / bf16 / int8 / int4 \u00b7 .trq \u00d75", YLW),
    ("tensor-roll evaluate", "int8: ppl 8.39 \u00b7 acc 0.342", GRN),
    ("tensor-roll sandbox", "allow 2 / deny 3 \u00b7 netns isolated", MAG),
]
y = 706
box_w, box_h, gap = 700, 56, 8
for name, note, col in stages:
    ax.add_patch(Rectangle((860, y - box_h), box_w, box_h, facecolor=PANEL,
                           edgecolor=col, linewidth=1.5))
    ax.text(880, y - 20, f"$ {name}", color=FG, fontsize=15,
            family="monospace", va="center")
    ax.text(880, y - 44, note, color=DIM, fontsize=12.5,
            family="monospace", va="center")
    if y - box_h - gap > 120:
        ax.add_patch(FancyArrowPatch((1210, y - box_h - 2),
                                     (1210, y - box_h - gap + 2),
                                     arrowstyle="->", color=DIM,
                                     linewidth=1.2, mutation_scale=12))
    y -= box_h + gap

# ---- bottom honesty strip ----
ax.text(W / 2, 56, "backend labels:  cpu-numpy  \u00b7  qsim-classical (NOT a QPU)  \u00b7  "
        "unavailable = not executed here, never invented",
        color=DIM, fontsize=13, ha="center", family="monospace")
ax.text(W / 2, 28, "AGPLv3  \u00b7  Copyright (C) 2026 SnapKitty Collective",
        color="#3a4353", fontsize=12, ha="center", family="monospace")

plt.savefig("docs/tensor-roll-demo.png", facecolor=BG, dpi=110,
            bbox_inches="tight", pad_inches=0.15)
print("wrote docs/tensor-roll-demo.png")
