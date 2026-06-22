"""
acf_neuron.py — Neurona ACF: La unidad fundamental de inteligencia funcional
============================================================================

Una neurona ACF no almacena pesos. Almacena su IDENTIDAD FUNCIONAL.
Cada neurona es una función f: R → R con representación FMA certificada,
dominio de validez geométrica, y capacidades innatas de:
  - Auto-conocimiento (sabe exactamente qué función es)
  - Auto-verificación (certifica su propia precisión)
  - Incertidumbre cuantificada (sabe cuánto no sabe)
  - Detección OOD (rechaza preguntas fuera de su dominio)
  - Conocimiento compartido (publica y consulta funciones en catálogo)

Autor: AXIOM-1 sobre fundamentos de Eddy Manuel Piantini
Fecha: Junio 2026
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Enums y tipos fundamentales
# ═════════════════════════════════════════════════════════════════════════════


class FunctionalForm(str, Enum):
    POLYNOMIAL = "polynomial"
    CHEBYSHEV = "chebyshev"
    KOOPMAN = "koopman"
    STRATIFIED = "stratified"


class NCClass(str, Enum):
    NC0 = "NC0"
    NC1 = "NC1"
    NC2 = "NC2"
    NC3 = "NC3"


class NeuronHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADING = "degrading"
    UNSTABLE = "unstable"
    OVERTAKEN = "overtaken"
    RETIRED = "retired"


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class FMAInstruction:
    weight: float
    bias: float
    source: str = "self"

    def evaluate(self, x: float) -> float:
        return self.weight * x + self.bias

    def to_tuple(self) -> Tuple[float, float]:
        return (self.weight, self.bias)


@dataclass
class IntervalContract:
    lo: float
    hi: float
    strict: bool = True

    def contains(self, x: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        if self.strict:
            return (x >= self.lo) & (x <= self.hi)
        return (x >= self.lo - 1e-10) & (x <= self.hi + 1e-10)

    def distance_to_boundary(self, x: float) -> float:
        return min(abs(x - self.lo), abs(x - self.hi))

    def project(self, x: float) -> float:
        return max(self.lo, min(self.hi, x))

    @property
    def scale(self) -> float:
        return self.hi - self.lo

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lo + self.hi)

    def sample(self, n: int) -> np.ndarray:
        return np.random.uniform(self.lo, self.hi, n)

    def boundary_sample(self, n: int) -> np.ndarray:
        half = n // 2
        return np.concatenate([
            np.linspace(self.lo, self.lo + self.scale * 0.01, half),
            np.linspace(self.hi - self.scale * 0.01, self.hi, n - half),
        ])

    def is_compatible_with(self, other: "IntervalContract") -> bool:
        return (other.lo >= self.lo and other.hi <= self.hi)

    def __repr__(self) -> str:
        return f"[{self.lo:.4g}, {self.hi:.4g}]"


@dataclass
class Observation:
    x: np.ndarray
    y_target: float
    timestamp: float = field(default_factory=time.time)
    weight: float = 1.0


@dataclass
class EvolutionStep:
    timestamp: float
    x: np.ndarray
    y_target: float
    y_pred: float
    error: float
    coefficients_snapshot: np.ndarray
    energy: int
    epsilon: float
    form_change: Optional["FormChange"] = None


@dataclass
class FormChange:
    old_form: FunctionalForm
    new_form: FunctionalForm
    reason: str
    alpha_A_before: float
    alpha_A_after: float


@dataclass
class VerificationRecord:
    timestamp: float
    checks: List["Check"]
    all_passed: bool
    neuron_state_hash: str


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class OODReport:
    is_ood: bool
    reason: str = ""
    closest_valid: Optional[float] = None
    recommended_action: str = ""


@dataclass
class UncertaintyBounds:
    prediction: float
    epsilon: float
    confidence_interval: Tuple[float, float]
    is_reliable: bool


@dataclass
class SelfKnowledge:
    functional_form: FunctionalForm
    expression: str
    energy: int
    alpha_A: float
    epsilon: float
    domain: IntervalContract
    nc_class: NCClass
    limitations: List[str]
    creation_time: float
    total_observations: int
    last_evolution: float


@dataclass
class FunctionFingerprint:
    fma_chain_hash: str
    alpha_A: float
    epsilon: float
    domain: IntervalContract
    energy: int
    functional_form: FunctionalForm
    nc_class: NCClass

    @property
    def hash(self) -> str:
        return self.fma_chain_hash

    def distance_to(self, other: "FunctionFingerprint") -> float:
        return (
            abs(self.alpha_A - other.alpha_A) * 10.0
            + abs(math.log10(max(self.epsilon, 1e-16)) - math.log10(max(other.epsilon, 1e-16)))
            + abs(self.energy - other.energy) * 0.01
        )


@dataclass
class FunctionEntry:
    name: str
    neuron_id: str
    fma_chain: List[FMAInstruction]
    fingerprint: FunctionFingerprint
    verified: bool
    parent_functions: List[str] = field(default_factory=list)
    published_at: float = field(default_factory=time.time)
    usage_count: int = 0


# ═════════════════════════════════════════════════════════════════════════════
# Catálogo de funciones
# ═════════════════════════════════════════════════════════════════════════════


class FunctionCatalog:
    def __init__(self):
        self._catalog: Dict[str, FunctionEntry] = {}
        self._fingerprint_index: Dict[str, str] = {}

    def publish(self, entry: FunctionEntry) -> None:
        self._catalog[entry.name] = entry
        self._fingerprint_index[entry.fingerprint.hash] = entry.name

    def get(self, name: str) -> Optional[FunctionEntry]:
        return self._catalog.get(name)

    def get_by_hash(self, fp_hash: str) -> Optional[FunctionEntry]:
        name = self._fingerprint_index.get(fp_hash)
        return self._catalog.get(name) if name else None

    def find_similar(self, target: FunctionFingerprint, tolerance: float = 0.15) -> List[FunctionEntry]:
        matches = [(e.fingerprint.distance_to(target), e) for e in self._catalog.values()]
        matches.sort(key=lambda x: x[0])
        return [m[1] for m in matches if m[0] < tolerance]

    def find_by_domain(self, domain: IntervalContract,
                       functional_form: Optional[FunctionalForm] = None) -> List[FunctionEntry]:
        matches = []
        for entry in self._catalog.values():
            if entry.fingerprint.domain.is_compatible_with(domain):
                if functional_form is None or entry.fingerprint.functional_form == functional_form:
                    matches.append(entry)
        return matches

    def list_all(self) -> List[FunctionEntry]:
        return list(self._catalog.values())

    def size(self) -> int:
        return len(self._catalog)

    def __contains__(self, name: str) -> bool:
        return name in self._catalog


# ═════════════════════════════════════════════════════════════════════════════
# Utilidades de cadenas FMA
# ═════════════════════════════════════════════════════════════════════════════


def evaluate_fma_chain(chain: List[FMAInstruction], x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    acc = x
    for instr in chain:
        acc = instr.weight * acc + instr.bias
    return acc


def fma_chain_to_horner_polynomial(chain: List[FMAInstruction]) -> np.ndarray:
    n = len(chain)
    coeffs = np.zeros(n + 1)
    if chain:
        coeffs[n] = chain[-1].weight
        for i in range(n - 1, -1, -1):
            coeffs[i] = chain[i].bias
    return coeffs


def compose_fma_chains(outer: List[FMAInstruction], inner: List[FMAInstruction]) -> List[FMAInstruction]:
    return [
        FMAInstruction(weight=i.weight, bias=i.bias, source=i.source)
        for i in (list(inner) + list(outer))
    ]


def simplify_fma_chain(chain: List[FMAInstruction]) -> List[FMAInstruction]:
    if not chain:
        return []
    simplified: List[FMAInstruction] = []
    i = 0
    while i < len(chain):
        instr = chain[i]
        if abs(instr.weight - 1.0) < 1e-15 and abs(instr.bias) < 1e-15:
            i += 1
            continue
        if i + 1 < len(chain):
            w1, b1 = instr.weight, instr.bias
            w2, b2 = chain[i + 1].weight, chain[i + 1].bias
            if abs(w1) < 100 and abs(b1) < 100:
                simplified.append(FMAInstruction(weight=w2 * w1, bias=w2 * b1 + b2, source="self"))
                i += 2
                continue
        simplified.append(instr)
        i += 1
    return simplified


def hash_fma_chain(chain: List[FMAInstruction]) -> str:
    data = "|".join(f"{i.weight:.15e}:{i.bias:.15e}" for i in chain)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


# ═════════════════════════════════════════════════════════════════════════════
# La Neurona ACF
# ═════════════════════════════════════════════════════════════════════════════


class ACFNeuron:
    """
    Neurona ACF — Unidad fundamental de inteligencia funcional.

    No almacena pesos. Almacena su IDENTIDAD FUNCIONAL como cadena FMA
    certificada. Sabe qué función es, en qué dominio, con qué error,
    y cuánto no sabe.

    Capacidades innatas:
      - self_knowledge  → sabe exactamente qué función es
      - evaluate(x)     → f(x) con cadena FMA certificada
      - uncertainty(x)  → ε(x) puntual con intervalo de confianza
      - detect_ood(x)   → rechaza entradas fuera de dominio
      - self_verify()   → verificación automática de precisión
      - publish_to_catalog() → conocimiento compartido
    """

    def __init__(
        self,
        name: str,
        functional_form: FunctionalForm = FunctionalForm.CHEBYSHEV,
        domain: Optional[IntervalContract] = None,
        max_degree: int = 32,
        catalog: Optional[FunctionCatalog] = None,
    ):
        self.neuron_id: str = f"acf_{uuid.uuid4().hex[:12]}"
        self.name: str = name
        self.functional_form: FunctionalForm = functional_form
        self.domain: IntervalContract = domain or IntervalContract(-1.0, 1.0)
        self.max_degree: int = max_degree

        self.fma_chain: List[FMAInstruction] = []
        self.coefficients: np.ndarray = np.zeros(max_degree + 1)
        self.energy: int = 0
        self.epsilon: float = float("inf")
        self.alpha_A: float = 0.0
        self.nc_class: NCClass = NCClass.NC0

        self.catalog: FunctionCatalog = catalog or FunctionCatalog()
        self.observation_buffer: Deque[Observation] = deque(maxlen=10000)
        self.evolution_history: List[EvolutionStep] = []
        self.total_observations: int = 0
        self.creation_time: float = time.time()
        self.last_evolution_time: float = self.creation_time

        self.incoming: Dict[str, Any] = {}
        self.outgoing: Dict[str, Any] = {}
        self.health: NeuronHealth = NeuronHealth.HEALTHY
        self.self_verification_log: List[VerificationRecord] = []

        self.learning_engine: Optional[Any] = None
        self._learning_enabled: bool = False

    # ── Propiedades innatas ────────────────────────────────────────

    @property
    def self_knowledge(self) -> SelfKnowledge:
        return SelfKnowledge(
            functional_form=self.functional_form,
            expression=self._function_expression(),
            energy=self.energy,
            alpha_A=self.alpha_A,
            epsilon=self.epsilon,
            domain=self.domain,
            nc_class=self.nc_class,
            limitations=self._known_limitations(),
            creation_time=self.creation_time,
            total_observations=self.total_observations,
            last_evolution=self.last_evolution_time,
        )

    def _function_expression(self) -> str:
        if self.functional_form in (FunctionalForm.POLYNOMIAL, FunctionalForm.CHEBYSHEV):
            terms = []
            for i, c in enumerate(self.coefficients):
                if abs(c) < 1e-15:
                    continue
                prefix = f"{c:.6g}"
                if i == 0:
                    terms.append(prefix)
                else:
                    var = "x" if self.functional_form == FunctionalForm.POLYNOMIAL else f"T{i}(x)"
                    terms.append(f"{prefix}·{var}" if i == 1 else f"{prefix}·{var}")
            return " + ".join(terms) if terms else "0"
        if self.functional_form == FunctionalForm.KOOPMAN:
            n = len(getattr(self, "koopman_modes", []))
            return f"Koopman({n} modos)"
        if self.functional_form == FunctionalForm.STRATIFIED:
            n = len(getattr(self, "regions", []))
            return f"Stratified({n} regiones)"
        return "indefinida"

    def _known_limitations(self) -> List[str]:
        lims = []
        if self.epsilon > 0.01:
            lims.append(f"Precisión limitada: ε={self.epsilon:.2e}")
        if self.alpha_A < 0.3:
            lims.append(f"Baja compresibilidad: α_A={self.alpha_A:.3f}")
        if self.functional_form == FunctionalForm.KOOPMAN:
            lims.append("Error por truncación de modos Koopman")
        if self.total_observations < 50:
            lims.append(f"Pocas observaciones: {self.total_observations}")
        return lims

    # ── Evaluación ─────────────────────────────────────────────────

    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluar la neurona: y = f(x).

        Usa chebval con normalización de dominio a [-1,1].
        """
        x_arr = np.asarray(x, dtype=np.float64)
        scalar_input = x_arr.ndim == 0
        x_val = float(x_arr) if scalar_input else np.asarray(x_arr, dtype=np.float64)

        # Proyectar a dominio
        if scalar_input:
            if not self.domain.contains(x_val):
                x_val = float(self.domain.project(x_val))
        else:
            x_val = np.clip(x_val, self.domain.lo, self.domain.hi)

        # Normalizar x ∈ [lo, hi] → t ∈ [-1, 1]
        lo, hi = self.domain.lo, self.domain.hi
        t = 2.0 * (x_val - lo) / (hi - lo) - 1.0
        t = np.clip(t, -1.0, 1.0) if not scalar_input else max(-1.0, min(1.0, t))

        d = self._effective_degree()
        if d < 0:
            result = 0.0
        else:
            result = np.polynomial.chebyshev.chebval(t, self.coefficients[:d + 1])

        return float(result) if scalar_input else result

    def __call__(self, x):
        return self.evaluate(x)

    # ── Incertidumbre y OOD ────────────────────────────────────────

    def uncertainty(self, x: float) -> UncertaintyBounds:
        y_pred = float(self.evaluate(x))
        eps = self.epsilon if self.epsilon != float("inf") else 0.1
        dist = self.domain.distance_to_boundary(x)
        boundary_factor = math.exp(-dist / max(self.domain.scale * 0.1, 1e-6))
        eps *= (1.0 + boundary_factor)
        curvature = self._estimate_curvature(x)
        eps *= (1.0 + curvature * 10.0)
        return UncertaintyBounds(
            prediction=y_pred,
            epsilon=float(eps),
            confidence_interval=(y_pred - 2 * eps, y_pred + 2 * eps),
            is_reliable=(eps < max(self.epsilon * 10, 0.01)),
        )

    def _estimate_curvature(self, x: float, h: float = 1e-4) -> float:
        fp = float(self.evaluate(x + h))
        f0 = float(self.evaluate(x))
        fm = float(self.evaluate(x - h))
        return abs(fp - 2 * f0 + fm) / (h * h + 1e-16)

    def detect_ood(self, x: Union[float, np.ndarray]) -> OODReport:
        x_val = float(np.asarray(x).ravel()[0]) if np.asarray(x).size > 0 else 0.0
        if not self.domain.contains(x_val):
            return OODReport(
                is_ood=True,
                reason=f"x={x_val:.4g} fuera de dominio {self.domain}",
                closest_valid=float(self.domain.project(x_val)),
                recommended_action="Consultar catálogo para este dominio o expandir con más datos.",
            )
        ub = self.uncertainty(x_val)
        if not ub.is_reliable:
            return OODReport(
                is_ood=True,
                reason=f"Alta incertidumbre local ε(x)={ub.epsilon:.2e} >> ε={self.epsilon:.2e}",
                recommended_action="Solicitar más observaciones en esta región.",
            )
        return OODReport(is_ood=False)

    # ── Auto-verificación ──────────────────────────────────────────

    def self_verify(self) -> VerificationRecord:
        """
        La neurona se verifica a sí misma: evalúa en puntos de prueba
        y verifica integridad numérica, dominio y energía.
        """
        checks: List[Check] = []

        # Check 1: Consistencia — evaluar en puntos de prueba usando self.evaluate()
        test_points = self.domain.sample(min(100, max(10, self.energy * 3)))
        all_finite = True
        for x in test_points:
            y = float(self.evaluate(x))
            if math.isnan(y) or math.isinf(y) or abs(y) > 1e15:
                all_finite = False
                break
        checks.append(Check("numerical_consistency", all_finite,
                            "Todos los valores finitos" if all_finite else "Valor no finito detectado"))

        # Check 2: Integridad de dominio — sin NaN/Inf en fronteras
        boundary_ok = True
        for x in self.domain.boundary_sample(10):
            y = float(self.evaluate(x))
            if math.isnan(y) or math.isinf(y):
                boundary_ok = False
                break
        checks.append(Check("domain_integrity", boundary_ok,
                            "Fronteras limpias" if boundary_ok else "NaN/Inf en frontera"))

        # Check 3: Conservación de energía
        energy_ok = len(self.fma_chain) == self.energy
        checks.append(Check("energy_conservation", energy_ok,
                            f"Chain={len(self.fma_chain)}, E(f)={self.energy}"))

        # Check 4: Suavidad
        mid_points = np.linspace(self.domain.lo, self.domain.hi, 50)
        values = np.array([float(self.evaluate(x)) for x in mid_points])
        max_jump = float(np.max(np.abs(np.diff(values))))
        smooth_ok = max_jump < 1000.0
        checks.append(Check("smoothness", smooth_ok,
                            f"Salto máximo={max_jump:.2f}" if not smooth_ok else "Suave"))

        record = VerificationRecord(
            timestamp=time.time(), checks=checks,
            all_passed=all(c.passed for c in checks),
            neuron_state_hash=hash_fma_chain(self.fma_chain),
        )
        self.self_verification_log.append(record)
        return record

    def _effective_degree(self) -> int:
        for i in range(len(self.coefficients) - 1, -1, -1):
            if abs(self.coefficients[i]) > 1e-15:
                return i
        return 0

    # ── Conocimiento compartido ────────────────────────────────────

    def publish_to_catalog(self) -> FunctionEntry:
        entry = FunctionEntry(
            name=self.name, neuron_id=self.neuron_id,
            fma_chain=copy.deepcopy(self.fma_chain),
            fingerprint=FunctionFingerprint(
                fma_chain_hash=hash_fma_chain(self.fma_chain),
                alpha_A=self.alpha_A, epsilon=self.epsilon,
                domain=self.domain, energy=self.energy,
                functional_form=self.functional_form, nc_class=self.nc_class,
            ),
            verified=(self.self_verification_log[-1].all_passed if self.self_verification_log else False),
        )
        self.catalog.publish(entry)
        return entry

    def find_similar_in_catalog(self, tolerance: float = 0.15) -> List[FunctionEntry]:
        fp = FunctionFingerprint(
            fma_chain_hash=hash_fma_chain(self.fma_chain),
            alpha_A=self.alpha_A, epsilon=self.epsilon,
            domain=self.domain, energy=self.energy,
            functional_form=self.functional_form, nc_class=self.nc_class,
        )
        return self.catalog.find_similar(fp, tolerance)

    # ── Estado ─────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        return {
            "neuron_id": self.neuron_id, "name": self.name,
            "functional_form": self.functional_form.value,
            "domain": (self.domain.lo, self.domain.hi),
            "energy": self.energy, "epsilon": self.epsilon,
            "alpha_A": self.alpha_A, "nc_class": self.nc_class.value,
            "total_observations": self.total_observations,
            "health": self.health.value,
            "expression": self._function_expression(),
        }

    def has_changed(self) -> bool:
        if not self.evolution_history:
            return False
        return self.evolution_history[-1].error > self.epsilon

    def __repr__(self) -> str:
        return (f"ACFNeuron('{self.name}', form={self.functional_form.value}, "
                f"E(f)={self.energy}, ε={self.epsilon:.2e}, α_A={self.alpha_A:.3f}, "
                f"domain={self.domain}, health={self.health.value})")
