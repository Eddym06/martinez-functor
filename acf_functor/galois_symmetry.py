"""Galois-style symmetry analysis and compression (Evolution 11)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import math

import torch

from .core import ChebyshevReducer, HornerReducer, ReductionResult


class SymmetryType:
    EVEN = "even"
    ODD = "odd"
    PERIODIC = "periodic"
    SHIFT = "shift"
    SCALE = "scale"
    NONE = "none"


@dataclass
class DetectedSymmetry:
    symmetry_type: str
    parameters: Dict[str, float]
    confidence: float
    compression_ratio: float


@dataclass
class GaloisGroup:
    symmetries: List[DetectedSymmetry]
    order: int
    total_compression_ratio: float
    compressed_coefficients: Optional[torch.Tensor]
    original_degree: int
    effective_degree: int

    def __repr__(self) -> str:
        names = [s.symmetry_type for s in self.symmetries]
        return f"Gal_Phi(f): order={self.order}, symmetries={names}, compression={self.total_compression_ratio:.2f}x"


class GaloisAnalyzer:
    def __init__(
        self,
        symmetry_threshold: float = 1e-8,
        n_probe: int = 5000,
        dtype: torch.dtype = torch.float64,
    ):
        self.symmetry_threshold = symmetry_threshold
        self.n_probe = n_probe
        self.dtype = dtype

    def analyze(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        reduction: Optional[ReductionResult] = None,
    ) -> GaloisGroup:
        symmetries: List[DetectedSymmetry] = []

        even = self._test_even(f, domain)
        if even is not None:
            symmetries.append(even)
        odd = self._test_odd(f, domain)
        if odd is not None:
            symmetries.append(odd)
        periodic = self._test_periodic(f, domain)
        if periodic is not None:
            symmetries.append(periodic)
        shift = self._test_shift(f, domain)
        if shift is not None:
            symmetries.append(shift)

        order = 1
        for sym in symmetries:
            order *= max(1, int(sym.compression_ratio))

        total_compression = max([sym.compression_ratio for sym in symmetries], default=1.0)
        compressed = None
        original_degree = 0
        effective_degree = 0

        if reduction is not None:
            coeffs = reduction.metadata.get(
                "monomial_coefficients",
                reduction.metadata.get("coefficients", None),
            )
            if coeffs is not None:
                coeffs_t = torch.as_tensor(coeffs, dtype=self.dtype)
                original_degree = max(0, int(coeffs_t.numel() - 1))
                compressed, effective_degree = self._compress_coefficients(coeffs_t, symmetries)

        return GaloisGroup(
            symmetries=symmetries,
            order=order,
            total_compression_ratio=float(total_compression),
            compressed_coefficients=compressed,
            original_degree=original_degree,
            effective_degree=effective_degree,
        )

    def compress_reduction(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        degree: int = 24,
    ) -> Tuple[ReductionResult, GaloisGroup]:
        reduction = ChebyshevReducer.reduce(f, degree=degree, domain=domain, dtype=self.dtype)
        galois = self.analyze(f, domain, reduction)

        if (
            galois.compressed_coefficients is not None
            and galois.original_degree > 0
            and galois.effective_degree < galois.original_degree
        ):
            compressed_reduction = HornerReducer.reduce(galois.compressed_coefficients.tolist(), dtype=self.dtype)
            compressed_reduction.epsilon_bound = reduction.epsilon_bound
            compressed_reduction.domain = domain
            compressed_reduction.metadata["galois_compression"] = {
                "original_degree": galois.original_degree,
                "effective_degree": galois.effective_degree,
                "compression_ratio": galois.total_compression_ratio,
                "symmetries": [sym.symmetry_type for sym in galois.symmetries],
            }
            compressed_reduction.metadata["monomial_coefficients"] = galois.compressed_coefficients.tolist()
            return compressed_reduction, galois

        return reduction, galois

    def _test_even(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> Optional[DetectedSymmetry]:
        a, b = domain
        if abs(a + b) > abs(b - a) * 0.1:
            return None
        half = min(abs(a), abs(b))
        x = torch.linspace(0.0, half, self.n_probe, dtype=self.dtype)
        try:
            f_pos = f(x)
            f_neg = f(-x)
            diff = float(torch.max(torch.abs(f_pos - f_neg)).item())
            scale = float(torch.max(torch.abs(f_pos)).item()) + 1e-15
            rel = diff / scale
            if rel < self.symmetry_threshold:
                return DetectedSymmetry(
                    symmetry_type=SymmetryType.EVEN,
                    parameters={"relative_error": rel},
                    confidence=1.0 - min(rel * 1e6, 1.0),
                    compression_ratio=2.0,
                )
        except Exception:
            return None
        return None

    def _test_odd(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> Optional[DetectedSymmetry]:
        a, b = domain
        if abs(a + b) > abs(b - a) * 0.1:
            return None
        half = min(abs(a), abs(b))
        x = torch.linspace(0.01 * half, half, self.n_probe, dtype=self.dtype)
        try:
            f_pos = f(x)
            f_neg = f(-x)
            diff = float(torch.max(torch.abs(f_pos + f_neg)).item())
            scale = float(torch.max(torch.abs(f_pos)).item()) + 1e-15
            rel = diff / scale
            if rel < self.symmetry_threshold:
                return DetectedSymmetry(
                    symmetry_type=SymmetryType.ODD,
                    parameters={"relative_error": rel},
                    confidence=1.0 - min(rel * 1e6, 1.0),
                    compression_ratio=2.0,
                )
        except Exception:
            return None
        return None

    def _test_periodic(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> Optional[DetectedSymmetry]:
        a, b = domain
        width = b - a
        periods = [math.pi, 2.0 * math.pi, 1.0, 2.0, width / 2.0, width / 3.0]
        for period in periods:
            if period > width * 0.9:
                continue
            x = torch.linspace(a, b - period, self.n_probe, dtype=self.dtype)
            try:
                fx = f(x)
                fxp = f(x + period)
                diff = float(torch.max(torch.abs(fx - fxp)).item())
                scale = float(torch.max(torch.abs(fx)).item()) + 1e-15
                rel = diff / scale
                if rel < self.symmetry_threshold:
                    return DetectedSymmetry(
                        symmetry_type=SymmetryType.PERIODIC,
                        parameters={"period": float(period), "relative_error": rel},
                        confidence=1.0 - min(rel * 1e6, 1.0),
                        compression_ratio=max(1.0, width / max(period, 1e-12)),
                    )
            except Exception:
                continue
        return None

    def _test_shift(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> Optional[DetectedSymmetry]:
        a, b = domain
        width = b - a
        c = width * 0.1
        x = torch.linspace(a, b - c, self.n_probe, dtype=self.dtype)
        try:
            fx = f(x)
            fxc = f(x + c)
            diff = fxc - fx
            diff_std = float(torch.std(diff).item())
            diff_mean = float(torch.mean(diff).item())
            if diff_std < self.symmetry_threshold * abs(diff_mean + 1e-15):
                return DetectedSymmetry(
                    symmetry_type=SymmetryType.SHIFT,
                    parameters={"shift": float(c), "offset": float(diff_mean)},
                    confidence=1.0 - min(diff_std * 1e6, 1.0),
                    compression_ratio=1.5,
                )
        except Exception:
            return None
        return None

    def _compress_coefficients(
        self,
        coefficients: torch.Tensor,
        symmetries: List[DetectedSymmetry],
    ) -> Tuple[torch.Tensor, int]:
        compressed = coefficients.clone()
        for sym in symmetries:
            if sym.symmetry_type == SymmetryType.EVEN:
                for idx in range(1, compressed.numel(), 2):
                    compressed[idx] = 0.0
            elif sym.symmetry_type == SymmetryType.ODD:
                for idx in range(0, compressed.numel(), 2):
                    compressed[idx] = 0.0
        effective = int((torch.abs(compressed) > 1e-30).sum().item())
        return compressed, effective
