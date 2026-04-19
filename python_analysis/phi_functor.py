from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class FMAOp:
    """Single Horner-stage operator y <- fma(y, x, b)."""

    w: torch.Tensor
    b: torch.Tensor

    def __call__(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # PyTorch FMA-style primitive: b + y * x
        return torch.addcmul(self.b, y, x)


class PhiFunctor:
    """Minimal, test-oriented Phi functor for exact polynomial -> FMA mapping."""

    @staticmethod
    def polynomial_exact(coeffs: Sequence[float], *, dtype: torch.dtype = torch.float32) -> List[FMAOp]:
        # Ascending coefficients: [a0, a1, ..., an]
        # Horner recurrence starts at seed an and consumes a_{n-1}..a0.
        return [
            FMAOp(torch.tensor(1.0, dtype=dtype), torch.tensor(c, dtype=dtype))
            for c in reversed(coeffs[:-1])
        ]

    @staticmethod
    def eval_sequence(ops: Sequence[FMAOp], x: torch.Tensor, seed: float) -> torch.Tensor:
        y = torch.zeros_like(x) + torch.tensor(seed, dtype=x.dtype, device=x.device)
        for op in ops:
            y = op(y, x)
        return y

    @staticmethod
    def polynomial_exact_from_chebyshev(approx) -> tuple[List[FMAOp], float]:
        """Convert Chebyshev approx on [a,b] to power basis and then to Horner-FMA."""

        cheb = np.polynomial.chebyshev.Chebyshev(
            np.array(approx.coeffs, dtype=np.float64),
            domain=[float(approx.domain_a), float(approx.domain_b)],
        )
        poly = cheb.convert(kind=np.polynomial.Polynomial)
        power_coeffs = poly.coef.tolist()
        target_len = int(getattr(approx, "degree", len(power_coeffs) - 1)) + 1
        if len(power_coeffs) < target_len:
            power_coeffs = power_coeffs + [0.0] * (target_len - len(power_coeffs))
        ops = PhiFunctor.polynomial_exact(power_coeffs, dtype=torch.float64)
        seed = float(power_coeffs[-1]) if power_coeffs else 0.0
        return ops, seed


def eval_generated(seq: Iterable[Tuple[float, float]], x: torch.Tensor, seed: float) -> torch.Tensor:
    """Evaluate generated (w, b) sequence using the same FMA semantics as PhiFunctor."""

    y = torch.zeros_like(x) + torch.tensor(seed, dtype=x.dtype, device=x.device)
    for _w, b in seq:
        y = torch.addcmul(torch.tensor(b, dtype=x.dtype, device=x.device), y, x)
    return y
