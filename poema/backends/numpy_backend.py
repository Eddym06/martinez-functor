"""
NumpyBackend — pure NumPy CPU backend.
No PyTorch, no GPU, no CUDA.  Works anywhere Python + NumPy run.
"""
from __future__ import annotations
from typing import Any, List, Tuple
import math

from .protocol import BackendCapabilities, BackendProtocol, BackendResult


class NumpyBackend(BackendProtocol):

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="numpy_cpu",
            supports_gpu=False,
            supports_cpu=True,
            supports_batched=True,
            supports_gradient=False,
            hardware_vendor="generic",
            precision_formats=["fp64", "fp32"],
            notes="Pure NumPy — zero GPU dependency. Ideal for CPU-only or ROCm-free deployments.",
        )

    def verify_available(self) -> bool:
        try:
            import numpy  # noqa: F401
            return True
        except ImportError:
            return False

    def compile(
        self,
        fma_sequence: List[Any],
        source_ast: Any,
        domain: Tuple[float, float] = (-1.0, 1.0),
        precision: str = "fp64",
        **kwargs,
    ) -> BackendResult:
        import numpy as np

        dtype = np.float64 if precision in ("fp64", "double") else np.float32
        instructions = [(instr.weight, instr.bias) for instr in fma_sequence]
        fma_count = len(instructions)

        # Build a closure that evaluates the FMA chain
        def _eval_numpy(x):
            arr = np.asarray(x, dtype=dtype)
            y = arr.copy()
            for w, b in instructions:
                y = dtype(w) * y + dtype(b)
            return y

        # Also emit readable C-like pseudocode for inspection
        lines = ["// Poema FMA sequence — NumPy backend",
                 f"// FMA depth: {fma_count}",
                 "double evaluate(double x) {",
                 "    double y = x;"]
        for i, (w, b) in enumerate(instructions):
            lines.append(f"    y = {w!r} * y + {b!r};  // FMA step {i}")
        lines += ["    return y;", "}"]
        code = "\n".join(lines)

        return BackendResult(
            callable_fn=_eval_numpy,
            emitted_code=code,
            fma_count=fma_count,
            epsilon_bound=0.0,
            backend_name="numpy_cpu",
        )
