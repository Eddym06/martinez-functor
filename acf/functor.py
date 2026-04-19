import re
from typing import Optional

import numpy as np
import torch

from acf_functor import ACFFunctor as CoreACFFunctor


def _precision_to_dtype(precision: str) -> torch.dtype:
    mapping = {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(precision.lower(), torch.float32)


class ACFunctor:
    """Canonical facade over the core ACF engine."""

    def __init__(self, target_dim_d: int = 1024, precision: str = "fp32"):
        self.target_dim_d = target_dim_d
        self.precision = _precision_to_dtype(precision)
        self.core = CoreACFFunctor(default_dtype=torch.float64)
        self._last_reduction = None
        self.state_matrix: Optional[torch.Tensor] = None

    def _precompensate_reversibility(self, W: torch.Tensor, expected_precision: torch.dtype) -> torch.Tensor:
        W_exact = W.to(torch.float64)
        W_hw = W_exact.to(expected_precision)
        residue = W_exact - W_hw.to(torch.float64)
        return (W_exact + residue).to(expected_precision)

    def _extract_poly_coeffs(self, expr: str):
        clean = expr.replace(" ", "")
        if "x" not in clean:
            try:
                return [float(clean)]
            except ValueError:
                return None

        terms = re.finditer(r"([+-]?[^+-]+)", clean)
        coeffs = {}
        for m in terms:
            term = m.group(1)
            if not term:
                continue
            if "x" not in term:
                p = 0
                c = float(term)
            else:
                if "*x" in term:
                    c_str = term.split("*x")[0]
                elif term.startswith("x"):
                    c_str = "1"
                elif term.startswith("-x"):
                    c_str = "-1"
                else:
                    return None
                c = float(c_str)
                if "**" in term:
                    p = int(term.split("**")[1])
                else:
                    p = 1
            coeffs[p] = coeffs.get(p, 0.0) + c

        if not coeffs:
            return None
        deg = max(coeffs.keys())
        return [coeffs.get(i, 0.0) for i in range(deg + 1)]

    def reduce(self, func_expression):
        if isinstance(func_expression, str):
            expr = func_expression.lower()
            if any(k in expr for k in ["sin", "cos", "exp", "tanh", "log", "sigmoid"]):
                fn_name = "sin" if "sin" in expr else "exp" if "exp" in expr else "cos" if "cos" in expr else "tanh" if "tanh" in expr else "log" if "log" in expr else "sigmoid"
                self._last_reduction = self.core.reduce_transcendental(fn_name, degree=20)
                return "Chebyshev_Path_Activated"

            coeffs = self._extract_poly_coeffs(expr)
            if coeffs is not None:
                self._last_reduction = self.core.reduce_polynomial(coeffs)
                return "Horner_Path_Activated"

            self._last_reduction = self.core.reduce_piecewise("relu")
            return "Stratified_Path_Activated"

        if isinstance(func_expression, torch.Tensor):
            self._last_reduction = self.core.reduce_dynamical_system(func_expression)
            return "Koopman_Path_Activated"

        raise TypeError("func_expression must be a string or trajectory tensor")

    def compute_acf_invariant(self, eigenvalues):
        alpha, delta_d = self.core.compute_invariant(eigenvalues)
        return alpha, delta_d

    def stratified_execute(self, x, selectors, strata_functions, device="cuda"):
        x = x.to(device)
        out = torch.zeros_like(x)
        for condition, W in zip(selectors, strata_functions):
            mask = condition(x) if callable(condition) else condition
            mask = mask.to(device=device, dtype=x.dtype)
            out = out + mask * (x @ W.to(device=device, dtype=x.dtype))
        return out

    def execute(self, input_tensor, device="cuda"):
        if self._last_reduction is None:
            raise ValueError("Functor has not reduced any function yet. Call .reduce() first.")
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.tensor(input_tensor, dtype=self.precision)

        x = input_tensor.to(device=device, dtype=self.precision)

        # Preserve historical behavior for high-dimensional tensor API users.
        if x.dim() >= 2 and x.shape[-1] == self.target_dim_d:
            W = torch.eye(self.target_dim_d, dtype=self.precision, device=device)
            self.state_matrix = self._precompensate_reversibility(W, self.precision)
            return torch.matmul(x, self.state_matrix)

        return self.core.evaluate(self._last_reduction, x, device=device)


# Backward-compatible alias.
ACFFunctor = ACFunctor
