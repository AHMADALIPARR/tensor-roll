# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tensor Roll: recursive tensor transformation for model compression.

Classical execution on this machine is NumPy-on-CPU, honestly labeled
``cpu-numpy`` in every report. Quantum paths execute on a classical
state-vector simulator (``qsim-classical``) unless real CUDA-Q/Q#
hardware tooling is present.
"""

__version__ = "0.1.0"

from .core import TensorRoll, roll_tree, select_plan, iter_leaves  # noqa: E402

__all__ = ["__version__", "TensorRoll", "roll_tree", "select_plan",
           "iter_leaves"]
