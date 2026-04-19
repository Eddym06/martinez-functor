from __future__ import annotations

import torch

from python_analysis.horner_generated import horner_fma_seq, horner_seed
from python_analysis.phi_functor import PhiFunctor, eval_generated


def test_lean_python_equivalence() -> None:
    """Verifica que el código extraído de Lean coincide con PhiFunctor manual."""

    coeffs = [1.0, -2.0, 3.0, -4.0, 5.0] * 20 + [7.0]  # grado 100

    seq_lean = horner_fma_seq(coeffs)
    seed_lean = horner_seed(coeffs)

    seq_manual = PhiFunctor.polynomial_exact(coeffs, dtype=torch.float64)

    assert len(seq_lean) == len(seq_manual)
    for (w1, b1), op in zip(seq_lean, seq_manual):
        assert abs(w1 - float(op.w.item())) < 1e-12
        assert abs(b1 - float(op.b.item())) < 1e-12

    x = torch.linspace(-1.0, 1.0, 1000, dtype=torch.float64)
    y_lean = eval_generated(seq_lean, x, seed_lean)
    y_manual = PhiFunctor.eval_sequence(seq_manual, x, coeffs[-1])

    assert torch.allclose(y_lean, y_manual, atol=1e-12)
