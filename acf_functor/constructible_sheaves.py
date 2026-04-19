"""Constructible sheaves over local FMA reductions (Evolution 8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from .core import ChebyshevReducer, HornerReducer, ReductionPath, ReductionResult


@dataclass
class OpenSet:
    interval: Tuple[float, float]
    index: int
    overlap_left: Optional[float] = None
    overlap_right: Optional[float] = None

    @property
    def a(self) -> float:
        return self.interval[0]

    @property
    def b(self) -> float:
        return self.interval[1]

    @property
    def extended_interval(self) -> Tuple[float, float]:
        return (
            self.a - float(self.overlap_left or 0.0),
            self.b + float(self.overlap_right or 0.0),
        )

    def intersects(self, other: "OpenSet") -> bool:
        return self.a < other.b and other.a < self.b

    def intersection(self, other: "OpenSet") -> Optional[Tuple[float, float]]:
        if not self.intersects(other):
            return None
        return (max(self.a, other.a), min(self.b, other.b))


@dataclass
class SheafSection:
    open_set: OpenSet
    reduction: ReductionResult
    local_epsilon: float = 0.0

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        cheb_coeffs = self.reduction.metadata.get("chebyshev_coefficients", None)
        if cheb_coeffs is not None and self.reduction.domain is not None:
            return ChebyshevReducer.evaluate_chebyshev_series(
                torch.as_tensor(cheb_coeffs, dtype=x.dtype, device=x.device),
                x,
                self.reduction.domain,
            )

        coeffs = self.reduction.metadata.get(
            "monomial_coefficients",
            self.reduction.metadata.get("coefficients", None),
        )
        if coeffs is None:
            return self.reduction.execute(x)
        return HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=x.dtype, device=x.device), x)


@dataclass
class RestrictionMap:
    source_index: int
    target_index: int
    overlap_interval: Tuple[float, float]
    agreement_error: float
    is_compatible: bool


@dataclass
class GluingDefect:
    section_i: int
    section_j: int
    overlap: Tuple[float, float]
    max_discrepancy: float
    mean_discrepancy: float
    defect_type: str


@dataclass
class SheafCohomologyResult:
    h0_rank: int
    h1_rank: int
    global_section_exists: bool
    gluing_defects: List[GluingDefect]
    restriction_maps: List[RestrictionMap]
    euler_characteristic: int


class ConstructibleSheaf:
    """Builds local reductions and glues them with a partition of unity."""

    def __init__(
        self,
        overlap_fraction: float = 0.1,
        gluing_threshold: float = 1e-6,
        transition_sharpness: float = 50.0,
        max_refinements: int = 5,
        n_probe: int = 1000,
        dtype: torch.dtype = torch.float64,
    ):
        self.overlap_fraction = overlap_fraction
        self.gluing_threshold = gluing_threshold
        self.transition_sharpness = transition_sharpness
        self.max_refinements = max_refinements
        self.n_probe = n_probe
        self.dtype = dtype

    def construct(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        split_points: List[float],
        degree: int = 24,
    ) -> Tuple[List[SheafSection], SheafCohomologyResult]:
        a, b = domain
        cover = self._build_cover(a, b, split_points)
        sections = self._reduce_sections(f, cover, degree)
        restrictions = self._compute_restrictions(sections)
        defects = self._detect_defects(sections, restrictions)

        for _ in range(self.max_refinements):
            if not defects:
                break
            sections, cover = self._refine_at_defects(f, sections, cover, defects, degree)
            restrictions = self._compute_restrictions(sections)
            defects = self._detect_defects(sections, restrictions)

        cohomology = SheafCohomologyResult(
            h0_rank=len(sections),
            h1_rank=len(defects),
            global_section_exists=(len(defects) == 0),
            gluing_defects=defects,
            restriction_maps=restrictions,
            euler_characteristic=len(sections) - len(defects),
        )
        return sections, cohomology

    def evaluate_global_section(self, sections: List[SheafSection], x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        total_weight = torch.zeros_like(x)

        for section in sections:
            a, b = section.open_set.interval
            weight = self._partition_of_unity(
                x,
                a,
                b,
                float(section.open_set.overlap_left or 0.0),
                float(section.open_set.overlap_right or 0.0),
            )
            result = result + weight * section.evaluate(x)
            total_weight = total_weight + weight

        return result / torch.clamp(total_weight, min=1e-30)

    def to_reduction_result(
        self,
        sections: List[SheafSection],
        cohomology: SheafCohomologyResult,
        domain: Tuple[float, float],
    ) -> ReductionResult:
        all_fma = []
        total_energy = 0
        max_eps = 0.0
        for section in sections:
            all_fma.extend(section.reduction.fma_sequence)
            total_energy += int(section.reduction.computational_energy)
            max_eps = max(max_eps, float(section.local_epsilon))

        if cohomology.gluing_defects:
            max_eps = max(max_eps, max(d.max_discrepancy for d in cohomology.gluing_defects))

        return ReductionResult(
            path=ReductionPath.STRATIFIED,
            fma_sequence=all_fma,
            computational_energy=total_energy,
            epsilon_bound=max_eps,
            domain=domain,
            metadata={
                "method": "constructible_sheaf",
                "n_sections": len(sections),
                "h0_rank": cohomology.h0_rank,
                "h1_rank": cohomology.h1_rank,
                "global_section_exists": cohomology.global_section_exists,
                "euler_characteristic": cohomology.euler_characteristic,
                "section_domains": [s.open_set.interval for s in sections],
                "gluing_defects": [
                    {
                        "sections": (d.section_i, d.section_j),
                        "overlap": d.overlap,
                        "discrepancy": d.max_discrepancy,
                        "type": d.defect_type,
                    }
                    for d in cohomology.gluing_defects
                ],
            },
        )

    def _build_cover(self, a: float, b: float, split_points: List[float]) -> List[OpenSet]:
        boundaries = [a] + sorted([p for p in split_points if a < p < b]) + [b]
        cover: List[OpenSet] = []
        for i in range(len(boundaries) - 1):
            left = boundaries[i]
            right = boundaries[i + 1]
            width = max(right - left, 1e-12)
            overlap_l = width * self.overlap_fraction if i > 0 else 0.0
            overlap_r = width * self.overlap_fraction if i < len(boundaries) - 2 else 0.0
            cover.append(
                OpenSet(
                    interval=(left, right),
                    index=i,
                    overlap_left=overlap_l,
                    overlap_right=overlap_r,
                )
            )
        return cover

    def _reduce_sections(self, f: Callable[[torch.Tensor], torch.Tensor], cover: List[OpenSet], degree: int) -> List[SheafSection]:
        sections: List[SheafSection] = []
        for open_set in cover:
            ext_a, ext_b = open_set.extended_interval
            try:
                reduction = ChebyshevReducer.reduce(f, degree=degree, domain=(ext_a, ext_b), dtype=self.dtype)
            except Exception:
                reduction = ChebyshevReducer.reduce(
                    f,
                    degree=max(degree // 2, 4),
                    domain=(ext_a, ext_b),
                    dtype=self.dtype,
                )
            sections.append(
                SheafSection(
                    open_set=open_set,
                    reduction=reduction,
                    local_epsilon=float(reduction.epsilon_bound),
                )
            )
        return sections

    def _compute_restrictions(self, sections: List[SheafSection]) -> List[RestrictionMap]:
        restrictions: List[RestrictionMap] = []
        for i in range(len(sections)):
            for j in range(i + 1, len(sections)):
                overlap = sections[i].open_set.intersection(sections[j].open_set)
                if overlap is None:
                    continue
                oa, ob = overlap
                if ob - oa < 1e-10:
                    continue

                x = torch.linspace(oa, ob, self.n_probe, dtype=self.dtype)
                yi = sections[i].evaluate(x)
                yj = sections[j].evaluate(x)
                err = float(torch.max(torch.abs(yi - yj)).item())
                restrictions.append(
                    RestrictionMap(
                        source_index=i,
                        target_index=j,
                        overlap_interval=overlap,
                        agreement_error=err,
                        is_compatible=err < self.gluing_threshold,
                    )
                )
        return restrictions

    def _detect_defects(self, sections: List[SheafSection], restrictions: List[RestrictionMap]) -> List[GluingDefect]:
        defects: List[GluingDefect] = []
        for rmap in restrictions:
            if rmap.is_compatible:
                continue
            oa, ob = rmap.overlap_interval
            x = torch.linspace(oa, ob, self.n_probe, dtype=self.dtype)
            yi = sections[rmap.source_index].evaluate(x)
            yj = sections[rmap.target_index].evaluate(x)
            diff = yi - yj
            abs_diff = torch.abs(diff)
            sign_changes = int(((diff[1:] * diff[:-1]) < 0).sum().item())
            density = sign_changes / max(len(diff) - 1, 1)
            if density > 0.3:
                defect_type = "oscillatory"
            elif float(abs_diff.max().item()) > float(abs_diff.mean().item()) * 5.0:
                defect_type = "jump"
            else:
                defect_type = "smooth"
            defects.append(
                GluingDefect(
                    section_i=rmap.source_index,
                    section_j=rmap.target_index,
                    overlap=rmap.overlap_interval,
                    max_discrepancy=float(abs_diff.max().item()),
                    mean_discrepancy=float(abs_diff.mean().item()),
                    defect_type=defect_type,
                )
            )
        return defects

    def _refine_at_defects(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        sections: List[SheafSection],
        cover: List[OpenSet],
        defects: List[GluingDefect],
        degree: int,
    ) -> Tuple[List[SheafSection], List[OpenSet]]:
        refined = list(sections)
        for defect in defects:
            for idx in [defect.section_i, defect.section_j]:
                if idx >= len(refined):
                    continue
                section = refined[idx]
                ext_a, ext_b = section.open_set.extended_interval
                new_degree = min(degree * 2, 200)
                try:
                    candidate = ChebyshevReducer.reduce(
                        f,
                        degree=new_degree,
                        domain=(ext_a, ext_b),
                        dtype=self.dtype,
                    )
                    if candidate.epsilon_bound < section.local_epsilon:
                        refined[idx] = SheafSection(
                            open_set=section.open_set,
                            reduction=candidate,
                            local_epsilon=float(candidate.epsilon_bound),
                        )
                except Exception:
                    continue
        return refined, cover

    def _partition_of_unity(
        self,
        x: torch.Tensor,
        a: float,
        b: float,
        overlap_left: float,
        overlap_right: float,
    ) -> torch.Tensor:
        beta = self.transition_sharpness
        if overlap_left > 0:
            left = torch.sigmoid(beta * (x - a) / max(overlap_left, 1e-10))
        else:
            left = (x >= a).to(x.dtype)

        if overlap_right > 0:
            right = torch.sigmoid(beta * (b - x) / max(overlap_right, 1e-10))
        else:
            right = (x <= b).to(x.dtype)
        return left * right
