"""
Sheaf semantics for Poema programs (Evolution 17).

A program is considered globally well-typed when the inferred gluing
obstructions are absent (H1 rank == 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import torch

from .ast_nodes import (
    ASTNode,
    AffineNode,
    ComposeNode,
    ConstantNode,
    IdentityNode,
    InputNode,
    PolynomialNode,
    ScaleNode,
    ShiftNode,
    StratifiedNode,
    TranscendentalNode,
)


class SemanticDomain(Enum):
    SCALAR = auto()
    INTERVAL = auto()
    POLYNOMIAL = auto()
    TRANSCENDENTAL = auto()
    STRATIFIED = auto()


@dataclass
class SemanticSection:
    domain: Tuple[float, float]
    value_range: Tuple[float, float]
    precision: float
    semantic_type: SemanticDomain
    source_node: Optional[ASTNode] = None
    fma_depth: int = 0

    @property
    def is_exact(self) -> bool:
        return self.precision <= 0.0

    @property
    def truth_value(self) -> float:
        if self.precision <= 0:
            return 1.0
        if self.precision >= 1.0:
            return 0.0
        return 1.0 - self.precision


@dataclass
class RestrictionMap:
    source: SemanticSection
    target: SemanticSection
    overlap: Tuple[float, float]
    compatibility_error: float

    @property
    def is_compatible(self) -> bool:
        tolerance = max(self.source.precision + self.target.precision, 1e-10)
        return self.compatibility_error <= tolerance


@dataclass
class CohomologicalVerdict:
    h0_rank: int
    h1_rank: int
    is_correct: bool
    truth_value: float
    obstructions: List[str]
    sections: List[SemanticSection]
    restrictions: List[RestrictionMap]

    def summary(self) -> str:
        status = "CORRECT" if self.is_correct else "OBSTRUCTED"
        lines = [
            "Sheaf Semantic Verdict: {0}".format(status),
            "  H0 rank: {0}".format(self.h0_rank),
            "  H1 rank: {0}".format(self.h1_rank),
            "  truth: {0:.4f}".format(self.truth_value),
            "  sections: {0}".format(len(self.sections)),
        ]
        if self.obstructions:
            lines.append("  obstructions:")
            for obs in self.obstructions:
                lines.append("    - {0}".format(obs))
        return "\n".join(lines)


class SheafSemantics:
    def __init__(
        self,
        default_domain: Tuple[float, float] = (-10.0, 10.0),
        n_probe: int = 1000,
        dtype: torch.dtype = torch.float64,
    ):
        self.default_domain = default_domain
        self.n_probe = n_probe
        self.dtype = dtype

    def analyze(
        self,
        program: ASTNode,
        domain: Optional[Tuple[float, float]] = None,
    ) -> CohomologicalVerdict:
        dom = domain or self.default_domain
        sections = self._compute_sections(program, dom)
        restrictions = self._compute_restrictions(sections)
        h0_rank = self._compute_h0(sections, restrictions)
        h1_rank, obstructions = self._compute_h1(sections, restrictions)

        if not sections:
            truth = 0.0
        elif h1_rank > 0:
            truth = max(0.0, 1.0 - 0.2 * h1_rank - max(s.precision for s in sections))
        else:
            truth = min(s.truth_value for s in sections)

        return CohomologicalVerdict(
            h0_rank=h0_rank,
            h1_rank=h1_rank,
            is_correct=(h1_rank == 0),
            truth_value=truth,
            obstructions=obstructions,
            sections=sections,
            restrictions=restrictions,
        )

    def is_well_typed(
        self,
        program: ASTNode,
        domain: Optional[Tuple[float, float]] = None,
    ) -> bool:
        return self.analyze(program, domain).is_correct

    def _compute_sections(self, node: ASTNode, domain: Tuple[float, float]) -> List[SemanticSection]:
        sections: List[SemanticSection] = []

        if isinstance(node, InputNode):
            sections.append(
                SemanticSection(
                    domain=domain,
                    value_range=domain,
                    precision=0.0,
                    semantic_type=SemanticDomain.SCALAR,
                    source_node=node,
                    fma_depth=0,
                )
            )
            return sections

        if isinstance(node, IdentityNode):
            sections.append(
                SemanticSection(
                    domain=domain,
                    value_range=domain,
                    precision=0.0,
                    semantic_type=SemanticDomain.SCALAR,
                    source_node=node,
                    fma_depth=0,
                )
            )
            return sections

        if isinstance(node, ConstantNode):
            val = float(node.value.item())
            sections.append(
                SemanticSection(
                    domain=domain,
                    value_range=(val, val),
                    precision=0.0,
                    semantic_type=SemanticDomain.SCALAR,
                    source_node=node,
                    fma_depth=1,
                )
            )
            return sections

        if isinstance(node, ScaleNode):
            alpha = float(node.factor.item())
            child_sections = self._sections_from_children(node, domain)
            if child_sections:
                for cs in child_sections:
                    a, b = cs.value_range
                    sections.append(
                        SemanticSection(
                            domain=cs.domain,
                            value_range=(min(alpha * a, alpha * b), max(alpha * a, alpha * b)),
                            precision=abs(alpha) * cs.precision,
                            semantic_type=cs.semantic_type,
                            source_node=node,
                            fma_depth=cs.fma_depth + 1,
                        )
                    )
            else:
                a, b = domain
                sections.append(
                    SemanticSection(
                        domain=domain,
                        value_range=(min(alpha * a, alpha * b), max(alpha * a, alpha * b)),
                        precision=0.0,
                        semantic_type=SemanticDomain.SCALAR,
                        source_node=node,
                        fma_depth=1,
                    )
                )
            return sections

        if isinstance(node, ShiftNode):
            beta = float(node.value.item())
            child_sections = self._sections_from_children(node, domain)
            if child_sections:
                for cs in child_sections:
                    a, b = cs.value_range
                    sections.append(
                        SemanticSection(
                            domain=cs.domain,
                            value_range=(a + beta, b + beta),
                            precision=cs.precision,
                            semantic_type=cs.semantic_type,
                            source_node=node,
                            fma_depth=cs.fma_depth + 1,
                        )
                    )
            else:
                a, b = domain
                sections.append(
                    SemanticSection(
                        domain=domain,
                        value_range=(a + beta, b + beta),
                        precision=0.0,
                        semantic_type=SemanticDomain.SCALAR,
                        source_node=node,
                        fma_depth=1,
                    )
                )
            return sections

        if isinstance(node, AffineNode):
            alpha = float(node.scale_factor.item())
            beta = float(node.shift_value.item())
            child_sections = self._sections_from_children(node, domain)
            if child_sections:
                for cs in child_sections:
                    a, b = cs.value_range
                    vals = [alpha * a + beta, alpha * b + beta]
                    sections.append(
                        SemanticSection(
                            domain=cs.domain,
                            value_range=(min(vals), max(vals)),
                            precision=abs(alpha) * cs.precision,
                            semantic_type=cs.semantic_type,
                            source_node=node,
                            fma_depth=cs.fma_depth + 1,
                        )
                    )
            else:
                a, b = domain
                vals = [alpha * a + beta, alpha * b + beta]
                sections.append(
                    SemanticSection(
                        domain=domain,
                        value_range=(min(vals), max(vals)),
                        precision=0.0,
                        semantic_type=SemanticDomain.SCALAR,
                        source_node=node,
                        fma_depth=1,
                    )
                )
            return sections

        if isinstance(node, PolynomialNode):
            x = torch.linspace(domain[0], domain[1], self.n_probe, dtype=self.dtype)
            from acf_functor.core import HornerReducer

            y = HornerReducer.execute_horner(node.coefficients.to(self.dtype), x)
            sections.append(
                SemanticSection(
                    domain=domain,
                    value_range=(float(y.min().item()), float(y.max().item())),
                    precision=0.0,
                    semantic_type=SemanticDomain.POLYNOMIAL,
                    source_node=node,
                    fma_depth=max(0, int(node.coefficients.numel()) - 1),
                )
            )
            return sections

        if isinstance(node, TranscendentalNode):
            poly_sections = self._compute_sections(node.polynomial, domain)
            for ps in poly_sections:
                sections.append(
                    SemanticSection(
                        domain=ps.domain,
                        value_range=ps.value_range,
                        precision=float(node.certified_epsilon),
                        semantic_type=SemanticDomain.TRANSCENDENTAL,
                        source_node=node,
                        fma_depth=ps.fma_depth,
                    )
                )
            return sections

        if isinstance(node, ComposeNode):
            inner_sections = self._compute_sections(node.inner, domain)
            for isec in inner_sections:
                outer_sections = self._compute_sections(node.outer, isec.value_range)
                lipschitz = self._estimate_lipschitz(node.outer, isec.value_range)
                for osec in outer_sections:
                    sections.append(
                        SemanticSection(
                            domain=isec.domain,
                            value_range=osec.value_range,
                            precision=osec.precision + isec.precision * lipschitz,
                            semantic_type=osec.semantic_type,
                            source_node=node,
                            fma_depth=isec.fma_depth + osec.fma_depth,
                        )
                    )
            return sections

        if isinstance(node, StratifiedNode):
            for branch in node.branches:
                sections.extend(self._compute_sections(branch.body_ast, branch.domain))
            return sections

        sections.append(
            SemanticSection(
                domain=domain,
                value_range=(-float("inf"), float("inf")),
                precision=float("inf"),
                semantic_type=SemanticDomain.SCALAR,
                source_node=node,
                fma_depth=node.estimate_fma_cost(),
            )
        )
        return sections

    def _sections_from_children(self, node: ASTNode, domain: Tuple[float, float]) -> List[SemanticSection]:
        out: List[SemanticSection] = []
        for child in node.children:
            if isinstance(child, ASTNode):
                out.extend(self._compute_sections(child, domain))
        return out

    def _compute_restrictions(self, sections: List[SemanticSection]) -> List[RestrictionMap]:
        restrictions: List[RestrictionMap] = []
        for i in range(len(sections)):
            for j in range(i + 1, len(sections)):
                si = sections[i]
                sj = sections[j]
                overlap = (max(si.domain[0], sj.domain[0]), min(si.domain[1], sj.domain[1]))
                if overlap[0] >= overlap[1]:
                    continue
                range_diff = max(
                    abs(si.value_range[0] - sj.value_range[0]),
                    abs(si.value_range[1] - sj.value_range[1]),
                )
                restrictions.append(
                    RestrictionMap(
                        source=si,
                        target=sj,
                        overlap=overlap,
                        compatibility_error=range_diff,
                    )
                )
        return restrictions

    def _compute_h0(self, sections: List[SemanticSection], restrictions: List[RestrictionMap]) -> int:
        if not sections:
            return 0

        parent = list(range(len(sections)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px = find(x)
            py = find(y)
            if px != py:
                parent[px] = py

        for rmap in restrictions:
            if rmap.is_compatible:
                i = sections.index(rmap.source)
                j = sections.index(rmap.target)
                union(i, j)

        roots = {find(i) for i in range(len(sections))}
        return len(roots)

    def _compute_h1(
        self, sections: List[SemanticSection], restrictions: List[RestrictionMap]
    ) -> Tuple[int, List[str]]:
        _ = sections
        obstructions: List[str] = []
        for rmap in restrictions:
            if not rmap.is_compatible:
                obstructions.append(
                    "gluing failure between {0} and {1} on [{2:.4f}, {3:.4f}] with error {4:.2e}".format(
                        repr(rmap.source.source_node),
                        repr(rmap.target.source_node),
                        rmap.overlap[0],
                        rmap.overlap[1],
                        rmap.compatibility_error,
                    )
                )
        return len(obstructions), obstructions

    def _estimate_lipschitz(self, node: ASTNode, domain: Tuple[float, float]) -> float:
        if isinstance(node, ScaleNode):
            return abs(float(node.factor.item()))
        if isinstance(node, AffineNode):
            return abs(float(node.scale_factor.item()))
        if isinstance(node, (ShiftNode, IdentityNode)):
            return 1.0
        if isinstance(node, PolynomialNode):
            from acf_functor.composition import LipschitzEstimator
            from acf_functor.core import HornerReducer

            reduction = HornerReducer.reduce(node.coefficients.tolist(), dtype=self.dtype)
            reduction.domain = domain
            return float(
                LipschitzEstimator.estimate_from_reduction(reduction, domain, dtype=self.dtype)
            )
        return 10.0
