"""Affine Computability Topos (ACT) with graded truth values (Evolution 15)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import torch

from .core import ChebyshevReducer, HornerReducer, ReductionPath, ReductionResult


@dataclass
class TruthValue:
    value: float
    is_exact: bool
    is_impossible: bool
    epsilon_bound: float

    @staticmethod
    def top() -> "TruthValue":
        return TruthValue(1.0, True, False, 0.0)

    @staticmethod
    def bottom() -> "TruthValue":
        return TruthValue(0.0, False, True, float("inf"))

    @staticmethod
    def partial(epsilon: float, max_epsilon: float = 1.0) -> "TruthValue":
        level = max(0.0, 1.0 - epsilon / max(max_epsilon, 1e-30))
        return TruthValue(level, False, False, epsilon)

    def __and__(self, other: "TruthValue") -> "TruthValue":
        return TruthValue(
            value=min(self.value, other.value),
            is_exact=self.is_exact and other.is_exact,
            is_impossible=self.is_impossible or other.is_impossible,
            epsilon_bound=max(self.epsilon_bound, other.epsilon_bound),
        )

    def __or__(self, other: "TruthValue") -> "TruthValue":
        return TruthValue(
            value=max(self.value, other.value),
            is_exact=self.is_exact or other.is_exact,
            is_impossible=self.is_impossible and other.is_impossible,
            epsilon_bound=min(self.epsilon_bound, other.epsilon_bound),
        )

    def implies(self, other: "TruthValue") -> "TruthValue":
        impl_value = max(1.0 - self.value, other.value)
        return TruthValue(
            value=impl_value,
            is_exact=(not self.is_exact) or other.is_exact,
            is_impossible=False,
            epsilon_bound=other.epsilon_bound if self.value > 0.5 else float("inf"),
        )

    def __repr__(self) -> str:
        if self.is_exact:
            return "TOP"
        if self.is_impossible:
            return "BOTTOM"
        return f"[{self.value:.4f}, eps={self.epsilon_bound:.2e}]"


@dataclass
class ToposObject:
    reduction: ReductionResult
    truth: TruthValue
    name: str = ""

    @property
    def is_well_defined(self) -> bool:
        return not self.truth.is_impossible

    @property
    def is_exact(self) -> bool:
        return self.truth.is_exact


@dataclass
class ToposMorphism:
    source: ToposObject
    target: ToposObject
    transformation_epsilon: float
    is_isomorphism: bool

    @property
    def truth(self) -> TruthValue:
        return self.source.truth & self.target.truth


@dataclass
class ToposAnalysis:
    objects: List[ToposObject]
    morphisms: List[ToposMorphism]
    subobject_classifier: Dict[str, TruthValue]
    internal_logic_consistent: bool
    total_truth: TruthValue


class ACFTopos:
    def __init__(self, max_epsilon: float = 1.0, dtype: torch.dtype = torch.float64):
        self.max_epsilon = max_epsilon
        self.dtype = dtype
        self._objects: Dict[str, ToposObject] = {}
        self._morphisms: List[ToposMorphism] = []

    def internalize(self, name: str, reduction: ReductionResult) -> ToposObject:
        if reduction.path == ReductionPath.HORNER_EXACT or reduction.epsilon_bound <= 0:
            truth = TruthValue.top()
        else:
            truth = TruthValue.partial(float(reduction.epsilon_bound), self.max_epsilon)
        obj = ToposObject(reduction=reduction, truth=truth, name=name)
        self._objects[name] = obj
        return obj

    def morphism(self, source_name: str, target_name: str) -> Optional[ToposMorphism]:
        if source_name not in self._objects or target_name not in self._objects:
            return None
        source = self._objects[source_name]
        target = self._objects[target_name]
        trans_eps = float(source.reduction.epsilon_bound) + float(target.reduction.epsilon_bound)
        is_iso = (source.truth.is_exact and target.truth.is_exact) or (trans_eps < 1e-12)
        morph = ToposMorphism(
            source=source,
            target=target,
            transformation_epsilon=trans_eps,
            is_isomorphism=is_iso,
        )
        self._morphisms.append(morph)
        return morph

    def subobject_classifier(self) -> Dict[str, TruthValue]:
        omega: Dict[str, TruthValue] = {}
        for name, obj in self._objects.items():
            omega[f"obj:{name}"] = obj.truth
        for morph in self._morphisms:
            key = f"morph:{morph.source.name}->{morph.target.name}"
            omega[key] = morph.truth
        return omega

    def verify_internal_logic(self) -> bool:
        for morph in self._morphisms:
            if morph.source.truth.is_impossible:
                continue
            if morph.transformation_epsilon < 0:
                return False
        return True

    def analyze(self) -> ToposAnalysis:
        omega = self.subobject_classifier()
        consistent = self.verify_internal_logic()
        total = TruthValue.top()
        for obj in self._objects.values():
            total = total & obj.truth
        return ToposAnalysis(
            objects=list(self._objects.values()),
            morphisms=self._morphisms,
            subobject_classifier=omega,
            internal_logic_consistent=consistent,
            total_truth=total,
        )

    def judge(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        degree: int = 24,
    ) -> TruthValue:
        try:
            reduction = ChebyshevReducer.reduce(f, degree=degree, domain=domain, dtype=self.dtype)
            x = torch.linspace(domain[0], domain[1], 5000, dtype=self.dtype)
            coeffs = reduction.metadata.get(
                "monomial_coefficients",
                reduction.metadata.get("coefficients", []),
            )
            y = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=self.dtype), x)
            f_exact = f(x)
            eps = float(torch.max(torch.abs(f_exact - y)).item())
            if eps < 1e-14:
                return TruthValue.top()
            return TruthValue.partial(eps, self.max_epsilon)
        except Exception:
            return TruthValue.bottom()

    def certificate_of_impossibility(self, reason: str) -> TruthValue:
        _ = reason
        return TruthValue.bottom()


# Canonical ACT naming (preferred) with backward-compatible alias.
ACTTopos = ACFTopos
