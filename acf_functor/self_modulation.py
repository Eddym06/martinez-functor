"""
Affine Collapse Functor (ACF) Phi - Topological Self-Modulation Engine
==========================================================
Intrinsic topological self-correction for reduction pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch

from .core import (
    ChebyshevReducer,
    EnrichedFunctor,
    HornerReducer,
    KoopmanReducer,
    ACFInvariant,
    ReductionPath,
    ReductionResult,
)


class TearType(Enum):
    NONE = auto()
    DISCONTINUITY = auto()
    HIGH_CURVATURE = auto()
    OSCILLATORY = auto()
    DIVERGENT = auto()
    COMPOSITIONAL = auto()


@dataclass
class TopologicalTear:
    tear_type: TearType
    location: float
    radius: float
    severity: float
    gradient_signature: float
    domain_interval: Tuple[float, float]

    @property
    def is_critical(self) -> bool:
        return self.severity > 0

    def __lt__(self, other: "TopologicalTear") -> bool:
        return self.severity > other.severity


@dataclass
class Stratum:
    domain: Tuple[float, float]
    reduction: ReductionResult
    selector_fn: Callable[[torch.Tensor], torch.Tensor]
    parent_tears: List[TearType] = field(default_factory=list)
    depth: int = 0


@dataclass
class SheafReduction(ReductionResult):
    strata: List[Stratum] = field(default_factory=list)
    tear_history: List[TopologicalTear] = field(default_factory=list)
    refinement_depth: int = 0
    convergence_achieved: bool = False

    def execute(self, x: torch.Tensor) -> torch.Tensor:
        if not self.strata:
            return super().execute(x)

        result = torch.zeros_like(x)
        covered = torch.zeros_like(x, dtype=torch.bool)

        for stratum in self.strata:
            mask = stratum.selector_fn(x)
            active = mask & ~covered
            if active.any():
                piece = _evaluate_reduction_on_domain(stratum.reduction, x)
                result = torch.where(active, piece, result)
                covered = covered | active

        if not covered.all():
            fallback = _evaluate_reduction_on_domain(
                getattr(self, "_global_fallback", self), x
            )
            result = torch.where(~covered, fallback, result)

        return result


class TopologicalResidue:
    def __init__(
        self,
        sensitivity_threshold: float = 1e-6,
        gradient_ratio_threshold: float = 10.0,
        oscillation_window: int = 7,
        min_tear_separation: float = 1e-4,
    ):
        self.sensitivity_threshold = sensitivity_threshold
        self.gradient_ratio_threshold = gradient_ratio_threshold
        self.oscillation_window = oscillation_window
        self.min_tear_separation = min_tear_separation

    def compute_residual_field(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        phi_f: ReductionResult,
        domain: Tuple[float, float],
        n_probe_points: int = 2000,
        dtype: torch.dtype = torch.float64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = domain
        x_probe = torch.linspace(a, b, n_probe_points, dtype=dtype)
        with torch.no_grad():
            f_exact = f(x_probe)
            f_approx = _evaluate_reduction_on_domain(phi_f, x_probe)
            residual = f_exact - f_approx
        return x_probe, residual

    def detect_tears(
        self,
        x_probe: torch.Tensor,
        residual: torch.Tensor,
        epsilon_bound: float,
    ) -> List[TopologicalTear]:
        abs_residual = torch.abs(residual)
        threshold = max(epsilon_bound, self.sensitivity_threshold)
        violations = abs_residual > threshold
        if not violations.any():
            return []

        dx = x_probe[1] - x_probe[0]
        grad_r = torch.zeros_like(residual)
        grad_r[1:-1] = (residual[2:] - residual[:-2]) / (2 * dx)
        grad_r[0] = (residual[1] - residual[0]) / dx
        grad_r[-1] = (residual[-1] - residual[-2]) / dx

        grad2_r = torch.zeros_like(residual)
        grad2_r[1:-1] = (residual[2:] - 2 * residual[1:-1] + residual[:-2]) / (dx**2)

        tear_regions = self._segment_violations(violations)

        tears: List[TopologicalTear] = []
        for start, end, mask in tear_regions:
            tear = self._classify_region(
                x_probe, residual, grad_r, grad2_r, start, end, mask, abs_residual
            )
            if tear is not None:
                tears.append(tear)

        tears = self._merge_nearby_tears(tears)
        tears.sort()
        return tears

    def _segment_violations(self, violations: torch.Tensor) -> List[Tuple[int, int, torch.Tensor]]:
        regions = []
        in_region = False
        start = 0

        for i in range(len(violations)):
            if violations[i] and not in_region:
                start = i
                in_region = True
            elif not violations[i] and in_region:
                mask = torch.zeros_like(violations)
                mask[start:i] = True
                regions.append((start, i, mask))
                in_region = False

        if in_region:
            mask = torch.zeros_like(violations)
            mask[start:] = True
            regions.append((start, len(violations), mask))

        return regions

    def _classify_region(
        self,
        x_probe: torch.Tensor,
        residual: torch.Tensor,
        grad_r: torch.Tensor,
        grad2_r: torch.Tensor,
        start: int,
        end: int,
        mask: torch.Tensor,
        abs_residual: torch.Tensor,
    ) -> Optional[TopologicalTear]:
        region_x = x_probe[start:end]
        region_r = residual[start:end]
        region_grad = grad_r[start:end]
        region_abs = abs_residual[start:end]

        if len(region_x) < 2:
            return None

        severity = region_abs.max().item()
        center = region_x[region_abs.argmax()].item()
        radius = (region_x[-1] - region_x[0]).item() / 2.0
        grad_max = torch.abs(region_grad).max().item()
        grad_mean = torch.abs(region_grad).mean().item()
        grad_ratio = grad_max / (grad_mean + 1e-30)

        sign_changes = torch.sum((region_r[1:] * region_r[:-1]) < 0).item()
        oscillation_density = sign_changes / max(len(region_r) - 1, 1)
        region_width = (region_x[-1] - region_x[0]).item()
        domain_width = (x_probe[-1] - x_probe[0]).item()
        jump_indicator = torch.abs(region_r[-1] - region_r[0]).item()

        if (
            grad_ratio > self.gradient_ratio_threshold
            or (jump_indicator > 0.4 * max(1.0, severity) and region_width < 0.1 * domain_width)
        ):
            tear_type = TearType.DISCONTINUITY
        elif oscillation_density > 0.3:
            tear_type = TearType.OSCILLATORY
        elif (region_abs[-1] > region_abs[0] * 2) or (region_abs[0] > region_abs[-1] * 2):
            tear_type = TearType.DIVERGENT
        else:
            tear_type = TearType.HIGH_CURVATURE

        domain_interval = (
            max(region_x[0].item() - radius * 0.2, x_probe[0].item()),
            min(region_x[-1].item() + radius * 0.2, x_probe[-1].item()),
        )

        return TopologicalTear(
            tear_type=tear_type,
            location=center,
            radius=max(radius, 1e-8),
            severity=severity,
            gradient_signature=grad_ratio,
            domain_interval=domain_interval,
        )

    def _merge_nearby_tears(self, tears: List[TopologicalTear]) -> List[TopologicalTear]:
        if len(tears) <= 1:
            return tears

        tears_sorted = sorted(tears, key=lambda t: t.location)
        merged = [tears_sorted[0]]

        for tear in tears_sorted[1:]:
            prev = merged[-1]
            gap = tear.domain_interval[0] - prev.domain_interval[1]
            if gap < self.min_tear_separation:
                merged[-1] = TopologicalTear(
                    tear_type=prev.tear_type if prev.severity >= tear.severity else tear.tear_type,
                    location=(prev.location + tear.location) / 2.0,
                    radius=(tear.domain_interval[1] - prev.domain_interval[0]) / 2.0,
                    severity=max(prev.severity, tear.severity),
                    gradient_signature=max(prev.gradient_signature, tear.gradient_signature),
                    domain_interval=(prev.domain_interval[0], tear.domain_interval[1]),
                )
            else:
                merged.append(tear)

        return merged


class SheafInjector:
    def __init__(
        self,
        transition_sharpness: float = 100.0,
        min_stratum_width: float = 1e-4,
        max_strata: int = 64,
    ):
        self.transition_sharpness = transition_sharpness
        self.min_stratum_width = min_stratum_width
        self.max_strata = max_strata

    def inject_at_tears(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        global_reduction: ReductionResult,
        tears: List[TopologicalTear],
        global_domain: Tuple[float, float],
        target_epsilon: float,
        degree: int = 20,
        dtype: torch.dtype = torch.float64,
    ) -> SheafReduction:
        if not tears:
            return self._wrap_as_sheaf(global_reduction, global_domain)

        split_points = self._compute_split_points(tears, global_domain)
        strata = self._build_strata(
            f, split_points, global_domain, target_epsilon, degree, dtype, tears
        )

        total_energy = sum(s.reduction.computational_energy for s in strata)
        max_eps = max(s.reduction.epsilon_bound for s in strata)

        sheaf = SheafReduction(
            path=ReductionPath.STRATIFIED,
            fma_sequence=global_reduction.fma_sequence,
            computational_energy=total_energy,
            epsilon_bound=max_eps,
            domain=global_domain,
            metadata={
                "method": "topological_sheaf_injection",
                "n_strata": len(strata),
                "n_tears_resolved": len(tears),
                "tear_types": [t.tear_type.name for t in tears],
                "split_points": split_points,
            },
            strata=strata,
            tear_history=tears,
            refinement_depth=1,
            convergence_achieved=(max_eps <= target_epsilon),
        )
        sheaf._global_fallback = global_reduction
        return sheaf

    def _compute_split_points(
        self,
        tears: List[TopologicalTear],
        global_domain: Tuple[float, float],
    ) -> List[float]:
        splits = set()
        a, b = global_domain

        for tear in tears:
            ta, tb = tear.domain_interval
            if tear.tear_type == TearType.DISCONTINUITY:
                splits.add(tear.location)
            elif tear.tear_type in (TearType.HIGH_CURVATURE, TearType.OSCILLATORY):
                if ta > a + self.min_stratum_width:
                    splits.add(ta)
                if tb < b - self.min_stratum_width:
                    splits.add(tb)
            elif tear.tear_type == TearType.DIVERGENT:
                margin = tear.radius * 0.5
                if ta - margin > a:
                    splits.add(ta - margin)
                if tb + margin < b:
                    splits.add(tb + margin)

        sorted_splits = sorted(splits)
        filtered = []
        for s in sorted_splits:
            if (s > a + self.min_stratum_width) and (s < b - self.min_stratum_width):
                if not filtered or s - filtered[-1] > self.min_stratum_width:
                    filtered.append(s)
        return filtered

    def _build_strata(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        split_points: List[float],
        global_domain: Tuple[float, float],
        target_epsilon: float,
        base_degree: int,
        dtype: torch.dtype,
        tears: List[TopologicalTear],
    ) -> List[Stratum]:
        a, b = global_domain
        boundaries = [a] + split_points + [b]
        strata: List[Stratum] = []

        for i in range(len(boundaries) - 1):
            s_a = boundaries[i]
            s_b = boundaries[i + 1]
            if s_b - s_a < self.min_stratum_width:
                continue

            local_degree = self._compute_local_degree(s_a, s_b, tears, base_degree)
            try:
                local_reduction = ChebyshevReducer.reduce(
                    f,
                    degree=local_degree,
                    domain=(s_a, s_b),
                    target_epsilon=target_epsilon,
                    dtype=dtype,
                )
            except Exception:
                local_reduction = ChebyshevReducer.reduce(
                    f,
                    degree=min(local_degree * 2, 200),
                    domain=(s_a, s_b),
                    dtype=dtype,
                )

            selector = self._build_selector(s_a, s_b, boundaries, i)
            local_tears = [
                t.tear_type
                for t in tears
                if (t.domain_interval[0] >= s_a - t.radius) and (t.domain_interval[1] <= s_b + t.radius)
            ]

            strata.append(
                Stratum(
                    domain=(s_a, s_b),
                    reduction=local_reduction,
                    selector_fn=selector,
                    parent_tears=local_tears,
                    depth=0,
                )
            )

        if len(strata) > self.max_strata:
            warnings.warn(
                f"Sheaf injection produced {len(strata)} strata, truncating to max_strata={self.max_strata}."
            )
            strata.sort(key=lambda s: s.reduction.epsilon_bound, reverse=True)
            strata = strata[: self.max_strata]

        return strata

    def _compute_local_degree(
        self,
        s_a: float,
        s_b: float,
        tears: List[TopologicalTear],
        base_degree: int,
    ) -> int:
        degree = base_degree
        for tear in tears:
            if s_a <= tear.location <= s_b:
                if tear.tear_type == TearType.HIGH_CURVATURE:
                    degree = max(degree, base_degree * 2)
                elif tear.tear_type == TearType.OSCILLATORY:
                    degree = max(degree, base_degree * 3)
                elif tear.tear_type == TearType.DIVERGENT:
                    degree = max(degree, base_degree * 2)
        return min(degree, 200)

    def _build_selector(
        self,
        s_a: float,
        s_b: float,
        all_boundaries: List[float],
        index: int,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        is_leftmost = index == 0
        is_rightmost = index == len(all_boundaries) - 2

        def selector(x, _left=s_a, _right=s_b, _lm=is_leftmost, _rm=is_rightmost):
            if _lm and _rm:
                return torch.ones_like(x, dtype=torch.bool)
            if _lm:
                return x < _right
            if _rm:
                return x >= _left
            return (x >= _left) & (x < _right)

        return selector

    def _wrap_as_sheaf(
        self,
        reduction: ReductionResult,
        domain: Tuple[float, float],
    ) -> SheafReduction:
        selector = lambda x: torch.ones_like(x, dtype=torch.bool)
        return SheafReduction(
            path=reduction.path,
            fma_sequence=reduction.fma_sequence,
            computational_energy=reduction.computational_energy,
            epsilon_bound=reduction.epsilon_bound,
            domain=domain,
            metadata={**reduction.metadata, "method": "trivial_sheaf_wrap", "n_strata": 1},
            strata=[Stratum(domain=domain, reduction=reduction, selector_fn=selector)],
            tear_history=[],
            refinement_depth=0,
            convergence_achieved=True,
        )


class ConvergenceMonad:
    def __init__(
        self,
        max_iterations: int = 8,
        convergence_ratio: float = 0.5,
        absolute_floor: float = 1e-15,
    ):
        self.max_iterations = max_iterations
        self.convergence_ratio = convergence_ratio
        self.absolute_floor = absolute_floor

    def iterate_to_fixpoint(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        initial_reduction: ReductionResult,
        domain: Tuple[float, float],
        target_epsilon: float,
        residue_analyzer: TopologicalResidue,
        sheaf_injector: SheafInjector,
        base_degree: int = 20,
        dtype: torch.dtype = torch.float64,
        n_probe: int = 2000,
    ) -> SheafReduction:
        current: ReductionResult = initial_reduction
        all_tears: List[TopologicalTear] = []
        history: List[Dict[str, Any]] = []
        prev_eps = float("inf")

        for it in range(self.max_iterations):
            x_probe, residual = residue_analyzer.compute_residual_field(
                f, current, domain, n_probe, dtype
            )
            max_err = torch.max(torch.abs(residual)).item()
            mean_err = torch.mean(torch.abs(residual)).item()
            history.append({"iteration": it, "max_error": max_err, "mean_error": mean_err})

            if max_err <= target_epsilon:
                if isinstance(current, SheafReduction):
                    current.convergence_achieved = True
                    current.refinement_depth = max(current.refinement_depth, it)
                    current.metadata["iteration_history"] = history
                    current.metadata["convergence"] = "target_achieved"
                    return current

                out = sheaf_injector._wrap_as_sheaf(current, domain)
                out.convergence_achieved = True
                out.refinement_depth = it
                out.metadata["iteration_history"] = history
                out.metadata["convergence"] = "target_achieved"
                return out

            tears = residue_analyzer.detect_tears(x_probe, residual, target_epsilon)
            all_tears.extend(tears)

            if tears:
                sheaf = sheaf_injector.inject_at_tears(
                    f,
                    current,
                    tears,
                    domain,
                    target_epsilon,
                    base_degree,
                    dtype,
                )
                current = sheaf

                _, new_res = residue_analyzer.compute_residual_field(f, sheaf, domain, n_probe, dtype)
                new_max = torch.max(torch.abs(new_res)).item()
                if new_max >= max_err * 0.99:
                    elevated = self._elevate_strata_degrees(f, sheaf, target_epsilon, dtype)
                    if elevated is not None:
                        current = elevated
            else:
                improved = self._reduce_residual(f, current, domain, base_degree, dtype)
                if improved is not None:
                    current = improved

            prev_eps = max_err
            if max_err <= self.absolute_floor:
                break
            if max_err >= prev_eps * self.convergence_ratio and it > 0:
                # keep iterating; sheaf strategy already handles non-monotonic regions
                pass

        if isinstance(current, SheafReduction):
            current.refinement_depth = len(history)
            current.tear_history = all_tears
            current.metadata["iteration_history"] = history
            current.metadata["convergence"] = "achieved" if history[-1]["max_error"] <= target_epsilon else "fixpoint"
            current.convergence_achieved = history[-1]["max_error"] <= target_epsilon
            return current

        out = sheaf_injector._wrap_as_sheaf(current, domain)
        out.refinement_depth = len(history)
        out.tear_history = all_tears
        out.metadata["iteration_history"] = history
        out.convergence_achieved = history[-1]["max_error"] <= target_epsilon
        return out

    def _reduce_residual(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        current: ReductionResult,
        domain: Tuple[float, float],
        base_degree: int,
        dtype: torch.dtype,
    ) -> Optional[ReductionResult]:
        current_degree = current.metadata.get("degree", base_degree)
        new_degree = min(current_degree + 10, 200)
        try:
            return ChebyshevReducer.reduce(f, degree=new_degree, domain=domain, dtype=dtype)
        except Exception:
            return None

    def _elevate_strata_degrees(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        sheaf: SheafReduction,
        target_epsilon: float,
        dtype: torch.dtype,
    ) -> Optional[SheafReduction]:
        improved_strata: List[Stratum] = []
        any_improved = False

        for stratum in sheaf.strata:
            s_a, s_b = stratum.domain
            current_degree = stratum.reduction.metadata.get("degree", 20)
            new_degree = min(current_degree + 10, 200)
            try:
                new_red = ChebyshevReducer.reduce(
                    f,
                    degree=new_degree,
                    domain=(s_a, s_b),
                    target_epsilon=target_epsilon,
                    dtype=dtype,
                )
                if new_red.epsilon_bound < stratum.reduction.epsilon_bound:
                    improved_strata.append(
                        Stratum(
                            domain=stratum.domain,
                            reduction=new_red,
                            selector_fn=stratum.selector_fn,
                            parent_tears=stratum.parent_tears,
                            depth=stratum.depth + 1,
                        )
                    )
                    any_improved = True
                else:
                    improved_strata.append(stratum)
            except Exception:
                improved_strata.append(stratum)

        if not any_improved:
            return None

        total_energy = sum(s.reduction.computational_energy for s in improved_strata)
        max_eps = max(s.reduction.epsilon_bound for s in improved_strata)
        out = SheafReduction(
            path=ReductionPath.STRATIFIED,
            fma_sequence=sheaf.fma_sequence,
            computational_energy=total_energy,
            epsilon_bound=max_eps,
            domain=sheaf.domain,
            metadata={**sheaf.metadata, "elevated": True},
            strata=improved_strata,
            tear_history=sheaf.tear_history,
            refinement_depth=sheaf.refinement_depth + 1,
        )
        out._global_fallback = sheaf
        return out


class AdaptiveReducer:
    def __init__(
        self,
        base_degree: int = 20,
        target_epsilon: float = 1e-8,
        sensitivity_threshold: float = 1e-8,
        gradient_ratio_threshold: float = 10.0,
        transition_sharpness: float = 100.0,
        max_strata: int = 64,
        max_iterations: int = 8,
        convergence_ratio: float = 0.5,
        n_probe_points: int = 2000,
    ):
        self.base_degree = base_degree
        self.target_epsilon = target_epsilon
        self.n_probe_points = n_probe_points

        self.residue_analyzer = TopologicalResidue(
            sensitivity_threshold=sensitivity_threshold,
            gradient_ratio_threshold=gradient_ratio_threshold,
        )
        self.sheaf_injector = SheafInjector(
            transition_sharpness=transition_sharpness,
            max_strata=max_strata,
        )
        self.convergence_monad = ConvergenceMonad(
            max_iterations=max_iterations,
            convergence_ratio=convergence_ratio,
        )

    def reduce(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        target_epsilon: Optional[float] = None,
        dtype: torch.dtype = torch.float64,
    ) -> SheafReduction:
        eps = target_epsilon or self.target_epsilon

        try:
            initial = ChebyshevReducer.reduce(f, degree=self.base_degree, domain=domain, dtype=dtype)
        except Exception as exc:
            warnings.warn(f"Initial reduction failed ({exc}); retrying lower degree.")
            initial = ChebyshevReducer.reduce(
                f,
                degree=max(self.base_degree // 2, 4),
                domain=domain,
                dtype=dtype,
            )

        _, residual = self.residue_analyzer.compute_residual_field(
            f, initial, domain, self.n_probe_points, dtype
        )
        max_error = torch.max(torch.abs(residual)).item()
        if max_error <= eps:
            return self.sheaf_injector._wrap_as_sheaf(initial, domain)

        return self.convergence_monad.iterate_to_fixpoint(
            f=f,
            initial_reduction=initial,
            domain=domain,
            target_epsilon=eps,
            residue_analyzer=self.residue_analyzer,
            sheaf_injector=self.sheaf_injector,
            base_degree=self.base_degree,
            dtype=dtype,
            n_probe=self.n_probe_points,
        )


def _evaluate_reduction_on_domain(reduction: ReductionResult, x: torch.Tensor) -> torch.Tensor:
    if isinstance(reduction, SheafReduction) and reduction.strata:
        return reduction.execute(x)

    cheb = reduction.metadata.get("chebyshev_coefficients")
    if cheb is not None and reduction.domain is not None:
        return ChebyshevReducer.evaluate_chebyshev_series(
            torch.tensor(cheb, dtype=x.dtype, device=x.device), x, reduction.domain
        )

    coeffs = reduction.metadata.get("monomial_coefficients", reduction.metadata.get("coefficients", None))
    if coeffs is not None:
        return HornerReducer.execute_horner(torch.tensor(coeffs, dtype=x.dtype, device=x.device), x)

    return reduction.execute(x)


class SelfModulatingFunctor:
    def __init__(
        self,
        default_dtype: torch.dtype = torch.float64,
        base_degree: int = 20,
        target_epsilon: float = 1e-8,
        max_iterations: int = 8,
        max_strata: int = 64,
        n_probe_points: int = 2000,
    ):
        self.dtype = default_dtype
        self.horner = HornerReducer()
        self.chebyshev = ChebyshevReducer()
        self.koopman = KoopmanReducer()
        self.invariant = ACFInvariant()
        self.enriched = EnrichedFunctor()

        self.adaptive = AdaptiveReducer(
            base_degree=base_degree,
            target_epsilon=target_epsilon,
            max_iterations=max_iterations,
            max_strata=max_strata,
            n_probe_points=n_probe_points,
        )

        self._reductions: Dict[str, ReductionResult] = {}

    def reduce_polynomial(self, coefficients: Union[List[float], torch.Tensor]) -> ReductionResult:
        result = self.horner.reduce(coefficients, self.dtype)
        self._reductions["last_polynomial"] = result
        return result

    def reduce_transcendental(
        self,
        func: Union[str, Callable],
        degree: int = 20,
        domain: Optional[Tuple[float, float]] = None,
        target_epsilon: Optional[float] = None,
    ) -> Union[ReductionResult, SheafReduction]:
        if isinstance(func, str):
            key = func.lower().strip()
            if key not in ChebyshevReducer.CANONICAL_FUNCTIONS:
                raise ValueError(f"Unknown canonical: {func}")
            fn = ChebyshevReducer.CANONICAL_FUNCTIONS[key]["generator"]
            if domain is None:
                domain = ChebyshevReducer.CANONICAL_FUNCTIONS[key]["domain"]
        elif callable(func):
            fn = func
            if domain is None:
                raise ValueError("Domain required for callable")
        else:
            raise TypeError("func must be string or callable")

        eps = target_epsilon or self.adaptive.target_epsilon
        old_degree = self.adaptive.base_degree
        self.adaptive.base_degree = max(2, degree)
        try:
            result = self.adaptive.reduce(fn, domain=domain, target_epsilon=eps, dtype=self.dtype)
        finally:
            self.adaptive.base_degree = old_degree

        self._reductions["last_transcendental"] = result
        return result

    def reduce_arbitrary(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        target_epsilon: Optional[float] = None,
    ) -> SheafReduction:
        eps = target_epsilon or self.adaptive.target_epsilon
        result = self.adaptive.reduce(f=f, domain=domain, target_epsilon=eps, dtype=self.dtype)
        self._reductions["last_arbitrary"] = result
        return result

    def reduce_dynamical_system(
        self,
        trajectory: torch.Tensor,
        observable_fn: Optional[Callable] = None,
        rank: Optional[int] = None,
    ) -> ReductionResult:
        result = self.koopman.reduce(trajectory, observable_fn=observable_fn, rank=rank, dtype=self.dtype)
        self._reductions["last_koopman"] = result
        return result

    def compose(self, phi_f: ReductionResult, phi_g: ReductionResult) -> ReductionResult:
        return self.enriched.compose(phi_f, phi_g)

    def compute_invariant(self, eigenvalues: Union[torch.Tensor, np.ndarray]) -> Tuple[float, float]:
        return self.invariant.compute_alpha(eigenvalues)

    def verify_conservation(self, f_energy: int, reduction: ReductionResult) -> bool:
        return self.enriched.verify_conservation(f_energy, reduction.computational_energy)

    def evaluate(self, reduction: ReductionResult, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        x_dev = x.to(device)
        if isinstance(reduction, SheafReduction) and reduction.strata:
            return reduction.execute(x_dev)
        return _evaluate_reduction_on_domain(reduction, x_dev)

    def diagnose(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        n_points: int = 2000,
    ) -> Dict[str, Any]:
        initial = ChebyshevReducer.reduce(
            f,
            degree=self.adaptive.base_degree,
            domain=domain,
            dtype=self.dtype,
        )

        x_probe, residual = self.adaptive.residue_analyzer.compute_residual_field(
            f, initial, domain, n_points, self.dtype
        )
        tears = self.adaptive.residue_analyzer.detect_tears(
            x_probe, residual, initial.epsilon_bound
        )

        max_err = torch.max(torch.abs(residual)).item()
        mean_err = torch.mean(torch.abs(residual)).item()

        return {
            "max_residual": max_err,
            "mean_residual": mean_err,
            "initial_epsilon": initial.epsilon_bound,
            "n_tears_detected": len(tears),
            "tear_details": [
                {
                    "type": t.tear_type.name,
                    "location": t.location,
                    "severity": t.severity,
                    "interval": t.domain_interval,
                }
                for t in tears
            ],
            "recommended_strategy": "EXACT_CHEBYSHEV" if not tears else "SHEAF_INJECTION",
            "estimated_strata": max(len(tears) + 1, 1),
        }
