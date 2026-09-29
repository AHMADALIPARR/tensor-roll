# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real CUDA-Q kernels for the tensor_roll family.

Target: NVIDIA CUDA-Q Python API (>= 0.8). Execute with a simulator
target (``cudaq.set_target("qpp-cpu")``) or on QPU/GPU hardware.

These kernels are NOT executed by the tensor-roll CLI on machines without
the CUDA-Q toolchain (see docs/BOUNDARIES.md). The classical state-vector
simulator in tensor_roll/qsim.py mirrors this family's semantics so the
pipeline is measurable everywhere; every qsim report is labeled
``qsim-classical`` and never claims QPU execution.

Conceptual mapping (classical -> quantum descent):
    C = A x B            (CUDA GEMM, classical baseline)
    TensorRoll(A, B)     (dispatch boundary in tensor_roll/core.py)
      |- classical region -> CUDA GEMM
      -- candidate region -> tensor_roll kernels below -> measurement
                             -> classical reconstruction
"""

try:
    import cudaq
    from cudaq import spin
    _HAS_CUDAQ = True
except ImportError:  # pragma: no cover - toolchain absent on most machines
    _HAS_CUDAQ = False


if _HAS_CUDAQ:

    @cudaq.kernel
    def tensor_roll_encode(qubits: cudaq.qview, angles: list[float]):
        """Angle-encode a tensor block: RY(2*atan(x_i)) per qubit."""
        for q in range(qubits.size()):
            ry(2.0 * angles[q], qubits[q])

    @cudaq.kernel
    def tensor_roll_rotate(qubits: cudaq.qview, angles: list[float]):
        """Per-qubit RY rotations (the 'roll' of the register)."""
        for q in range(qubits.size()):
            ry(angles[q], qubits[q])

    @cudaq.kernel
    def tensor_roll_entangle(qubits: cudaq.qview):
        """Nearest-neighbour CNOT ladder."""
        for q in range(qubits.size() - 1):
            x.ctrl(qubits[q], qubits[q + 1])

    @cudaq.kernel
    def tensor_roll_measure(qubits: cudaq.qview):
        """Measure the full register in the Z basis."""
        mz(qubits)

    @cudaq.kernel
    def tensor_roll_recursive(qubits: cudaq.qview, angles: list[float],
                              depth: int):
        """Recursive kernel: each level rotates + entangles a shrinking
        active qubit subset, then measures it. Implemented iteratively over
        levels (CUDA-Q kernels unroll statically); depth selects how many
        levels execute."""
        n = qubits.size()
        for level in range(depth):
            # active subset shrinks with depth: n, n-1, ...
            for q in range(n):
                if q < n - level:
                    ry(angles[q], qubits[q])
            for q in range(n - 1):
                if q < n - 1 - level:
                    x.ctrl(qubits[q], qubits[q + 1])
            if level < depth - 1:
                # mid-circuit measurement of the outer qubit, then reset
                # for the next recursion level
                mz(qubits[n - 1 - level])
                reset(qubits[n - 1 - level])
        mz(qubits)

    @cudaq.kernel
    def tensor_roll_reconstruct(counts_register: cudaq.qview):
        """Measurement kernel whose counts are classically post-processed
        into amplitude estimates (see tensor_roll/qsim.py for the estimator).
        Reconstruction itself is classical by construction."""
        mz(counts_register)


def run_smoke(n_qubits: int = 4, shots: int = 1024) -> dict:
    """Execute the kernel family on the qpp-cpu simulator target and return
    a real invocation summary. Requires the CUDA-Q toolchain."""
    if not _HAS_CUDAQ:
        raise RuntimeError(
            "CUDA-Q toolchain not installed (pip install cudaq). "
            "This machine cannot execute these kernels; use the "
            "qsim-classical path (tensor-roll qkernel) instead.")
    import math
    cudaq.set_target("qpp-cpu")
    angles = [2.0 * math.atan(0.1 * (i + 1)) for i in range(n_qubits)]

    @cudaq.kernel
    def _smoke_entry(n_qubits: int, angles: list[float], depth: int):
        qubits = cudaq.qvector(n_qubits)
        tensor_roll_encode(qubits, angles)
        tensor_roll_recursive(qubits, angles, depth)

    counts = cudaq.sample(_smoke_entry, n_qubits, angles, 2,
                          shots_count=shots)
    return {"counts": {k: int(v) for k, v in counts.items()},
            "shots": shots, "target": cudaq.get_target().name}


if __name__ == "__main__":
    try:
        print(run_smoke())
    except RuntimeError as e:
        print(f"unavailable: {e}")
