from __future__ import annotations

import torch

from python_analysis.phi_functor import PhiFunctor


def energy_fma_depth(fma_sequence) -> int:
    """Retorna la 'energía computacional' E(f) como profundidad FMA."""

    return len(fma_sequence)


def test_conservation_structural() -> None:
    """Verifica E(f) = E(Φ(f)) para un polinomio de grado 7."""

    coeffs = [0.0, 1.0, 0.0, -1.0 / 6.0, 0.0, 1.0 / 120.0, 0.0, -1.0 / 5040.0]
    phi_f = PhiFunctor.polynomial_exact(coeffs)

    e_f = len(coeffs) - 1
    e_phi = energy_fma_depth(phi_f)

    print(f"E(f) teórica: {e_f}")
    print(f"E(Φ(f)) medida: {e_phi}")
    print(f"Conservado: {e_f == e_phi}")

    assert e_f == e_phi, "Conservación estructural violada"


def test_conservation_numerical() -> None:
    """Verifica Φ(f)(x) ≈ f(x) con error de redondeo para el mismo polinomio."""

    coeffs = [0.0, 1.0, 0.0, -1.0 / 6.0, 0.0, 1.0 / 120.0, 0.0, -1.0 / 5040.0]

    def f_naive(x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        for i, c in enumerate(coeffs):
            y = y + torch.tensor(c, dtype=x.dtype, device=x.device) * torch.pow(x, i)
        return y

    x_test = torch.linspace(-0.5, 0.5, 1000, dtype=torch.float32)
    phi_f = PhiFunctor.polynomial_exact(coeffs, dtype=x_test.dtype)

    y_naive = f_naive(x_test)
    y_phi = PhiFunctor.eval_sequence(phi_f, x_test, coeffs[-1])

    error = torch.max(torch.abs(y_naive - y_phi)).item()
    epsilon_machine = 1e-7

    print(f"Error numérico max: {error:.2e}")
    print(f"Epsilon máquina (fp32): {epsilon_machine:.2e}")

    assert error < 10 * epsilon_machine, f"Error {error:.3e} demasiado grande"


if __name__ == "__main__":
    test_conservation_structural()
    test_conservation_numerical()
    print("Índice Afín α(f) (estructural): True")
    print("Índice Afín α(f) (numérico): True")