"""
GideonTheoremSeeds — Base para Descubrimiento de Teoremas Matemáticos.

Este módulo establece la INFRAESTRUCTURA BASE del motor de descubrimiento
de teoremas de Poema/Gideon. Por ahora implementa:

  - TheoremCandidate: representación de un candidato a teorema
  - InvariantProbe: sonda numérica que detecta posibles invariantes
  - PatternMatcher: detecta patrones algebraicos en secuencias FMA
  - GideonTheoremSeeds: orquestador de sondas y candidatos

Filosofía:
  Gideon observa los invariantes numéricos que emergen de los programas
  compilados y los convierte en candidatos a teoremas verificables en Lean 4.
  El ciclo:
    1. Ejecutar programa en muchos puntos
    2. Detectar invariantes estadísticos (α, Lipschitz, URT bound)
    3. Formular conjeturas en lenguaje matemático formal
    4. Crear esqueletos Lean 4 para verificación

NOTA: El prover autónomo completo (Genesis) es un componente separado.
Este módulo es la capa de semillas que lo alimenta.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# TheoremCandidate
# ─────────────────────────────────────────────────────────────────────────────

from enum import Enum

class TheoremStatus(Enum):
    CONJECTURE  = "conjecture"   # No verificado
    NUMERICAL   = "numerical"    # Verificado numéricamente
    CERTIFIED   = "certified"    # Certificado formalmente (Lean 4)
    REFUTED     = "refuted"      # Contraejemplo encontrado


@dataclass
class TheoremCandidate:
    """
    Candidato a teorema matemático generado por Gideon.

    Representa una propiedad conjeturada sobre un programa Gideon
    (o una función matemática general) que podría ser verificable formalmente.
    """
    name: str
    statement: str                              # Descripción en lenguaje natural
    formal_statement: str = ""                  # Template Lean 4 / matemático
    status: TheoremStatus = TheoremStatus.CONJECTURE
    evidence: List[float] = field(default_factory=list)    # Valores numéricos de soporte
    counterexample: Optional[float] = None
    confidence: float = 0.0                    # [0, 1] nivel de confianza
    lean_skeleton: str = ""                    # Esqueleto Lean 4 generado
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "statement": self.statement,
            "formal_statement": self.formal_statement,
            "status": self.status.value,
            "confidence": self.confidence,
            "lean_skeleton": self.lean_skeleton,
            "tags": self.tags,
        }

    def __repr__(self) -> str:
        return (
            f"TheoremCandidate({self.name!r}, "
            f"status={self.status.value}, conf={self.confidence:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# InvariantProbe — sonda numérica
# ─────────────────────────────────────────────────────────────────────────────

class InvariantProbe:
    """
    Sonda que evalúa una función en muchos puntos para detectar invariantes.

    Invariantes buscados:
      - Lipschitz constant (|f(x)-f(y)| / |x-y|)
      - URT bound (||f - f_approx||_∞)
      - Monotonicidad
      - Simetría (par/impar)
      - Periodicidad
      - α-complejidad ACF
    """

    def __init__(self, fn: Callable, domain: Tuple[float, float], n_points: int = 1000) -> None:
        self.fn = fn
        self.domain = domain
        self.n_points = n_points
        self._xs: Optional[np.ndarray] = None
        self._ys: Optional[np.ndarray] = None

    def _evaluate(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._xs is not None:
            return self._xs, self._ys  # type: ignore
        lo, hi = self.domain
        self._xs = np.linspace(lo, hi, self.n_points)
        try:
            import torch
            x_t = torch.tensor(self._xs, dtype=torch.float64)
            y_t = self.fn(x_t)
            self._ys = y_t.detach().numpy() if hasattr(y_t, "detach") else np.asarray(y_t)
        except Exception:
            self._ys = np.vectorize(self.fn)(self._xs)
        return self._xs, self._ys  # type: ignore

    def lipschitz_estimate(self) -> float:
        """Estima la constante de Lipschitz empíricamente."""
        xs, ys = self._evaluate()
        diffs_x = np.diff(xs)
        diffs_y = np.diff(ys.flatten())
        ratios = np.abs(diffs_y) / (np.abs(diffs_x) + 1e-300)
        return float(np.max(ratios))

    def urt_bound_estimate(self, reference_fn: Callable) -> float:
        """Estima la cota URT comparando con función de referencia."""
        xs, ys = self._evaluate()
        try:
            ys_ref = np.vectorize(reference_fn)(xs)
        except Exception:
            return float("nan")
        return float(np.max(np.abs(ys.flatten() - ys_ref)))

    def is_monotone(self, tol: float = 1e-10) -> bool:
        _, ys = self._evaluate()
        diffs = np.diff(ys.flatten())
        return bool(np.all(diffs >= -tol) or np.all(diffs <= tol))

    def symmetry_type(self, tol: float = 1e-8) -> str:
        """Detecta si la función es par, impar, o ninguna."""
        xs, ys = self._evaluate()
        ys_f = ys.flatten()
        mid = len(xs) // 2
        # Par: f(-x) ≈ f(x); Impar: f(-x) ≈ -f(x)
        # Usamos solo el rango simétrico
        lo, hi = self.domain
        if abs(lo + hi) > 1e-6:
            return "none"  # Dominio no simétrico
        xs_pos = xs[mid:]
        ys_pos = ys_f[mid:]
        ys_neg_flipped = ys_f[:mid][::-1]
        if xs_pos.shape != ys_neg_flipped.shape:
            return "none"
        if np.allclose(ys_pos, ys_neg_flipped, atol=tol):
            return "even"
        if np.allclose(ys_pos, -ys_neg_flipped, atol=tol):
            return "odd"
        return "none"

    def alpha_complexity(self) -> float:
        """
        Estima el índice α ACF a partir del perfil espectral.
        α ≈ -log(||coefs Fourier||_2) / log(grado)
        """
        _, ys = self._evaluate()
        ys_f = ys.flatten()
        # Análisis espectral via FFT
        spectrum = np.abs(np.fft.rfft(ys_f))
        if len(spectrum) < 2:
            return 1.0
        # Tasa de decaimiento espectral
        spectrum_norm = spectrum / (spectrum[0] + 1e-300)
        decay = -np.polyfit(np.log(np.arange(1, min(20, len(spectrum)))),
                            np.log(spectrum_norm[1:min(20, len(spectrum))] + 1e-300), 1)[0]
        return max(0.0, float(decay))


# ─────────────────────────────────────────────────────────────────────────────
# PatternMatcher — detector de patrones algebraicos
# ─────────────────────────────────────────────────────────────────────────────

class PatternMatcher:
    """
    Detecta patrones algebraicos en secuencias FMA que sugieren teoremas.

    Patrones buscados:
      - Composición idempotente: f ∘ f ≈ f
      - Proyección: f ∘ f = f (exacto)
      - Contractividad: ||f(x) - f(y)|| < k||x-y||, k < 1
      - Conservación de energía: ||f(x)||² ≈ ||x||²
    """

    @staticmethod
    def is_contraction(fn: Callable, domain: Tuple[float, float], n: int = 100) -> Tuple[bool, float]:
        """Detecta si f es una contracción y estima la constante k."""
        probe = InvariantProbe(fn, domain, n_points=n)
        L = probe.lipschitz_estimate()
        return L < 1.0, L

    @staticmethod
    def is_energy_preserving(fn: Callable, domain: Tuple[float, float], tol: float = 1e-4) -> bool:
        """Detecta si f preserva la norma L2."""
        xs = np.linspace(domain[0], domain[1], 200)
        try:
            import torch
            x_t = torch.tensor(xs, dtype=torch.float64)
            y_t = fn(x_t).detach().numpy()
        except Exception:
            y_t = np.vectorize(fn)(xs)
        norm_x = float(np.linalg.norm(xs))
        norm_y = float(np.linalg.norm(y_t))
        return abs(norm_x - norm_y) / (norm_x + 1e-300) < tol

    @staticmethod
    def detect_idempotent(fn: Callable, domain: Tuple[float, float], tol: float = 1e-6) -> bool:
        """Detecta si f ∘ f ≈ f en el dominio."""
        xs = np.linspace(domain[0], domain[1], 200)
        try:
            import torch
            x_t = torch.tensor(xs, dtype=torch.float64)
            y1 = fn(x_t)
            y2 = fn(y1)
            err = float((y2 - y1).abs().max())
        except Exception:
            y1 = np.vectorize(fn)(xs)
            y2 = np.vectorize(fn)(y1)
            err = float(np.max(np.abs(y2 - y1)))
        return err < tol


# ─────────────────────────────────────────────────────────────────────────────
# GideonTheoremSeeds — orquestador
# ─────────────────────────────────────────────────────────────────────────────

class GideonTheoremSeeds:
    """
    Motor de generación de candidatos a teoremas.

    Dado un callable (función compilada por Gideon), analiza sus propiedades
    y genera TheoremCandidates con esqueletos Lean 4.
    """

    def __init__(self) -> None:
        self._candidates: List[TheoremCandidate] = []

    @property
    def candidates(self) -> List[TheoremCandidate]:
        return list(self._candidates)

    # ── Análisis principal ────────────────────────────────────────────────────

    def analyse(
        self,
        fn: Callable,
        domain: Tuple[float, float] = (-1.0, 1.0),
        fn_name: str = "f",
        n_points: int = 500,
    ) -> List[TheoremCandidate]:
        """
        Analiza fn y genera candidatos a teoremas.
        Devuelve la lista de nuevos candidatos generados.
        """
        probe = InvariantProbe(fn, domain, n_points=n_points)
        new_cands: List[TheoremCandidate] = []

        # 1. Lipschitz
        try:
            L = probe.lipschitz_estimate()
            if math.isfinite(L):
                cand = TheoremCandidate(
                    name=f"lipschitz_{fn_name}",
                    statement=f"La función {fn_name} es Lipschitz con constante ≤ {L:.6f} en {domain}",
                    formal_statement=(
                        f"theorem lipschitz_{fn_name} : ∀ x y ∈ [{domain[0]}, {domain[1]}],\n"
                        f"  |{fn_name}(x) - {fn_name}(y)| ≤ {L:.6f} * |x - y| := by\n"
                        f"  sorry -- Gideon seed: verificar con análisis de gradiente"
                    ),
                    status=TheoremStatus.NUMERICAL,
                    evidence=[L],
                    confidence=min(0.9, 1.0 / (1.0 + abs(L - 1.0))),
                    tags=["lipschitz", "continuity"],
                )
                cand.lean_skeleton = cand.formal_statement
                new_cands.append(cand)
        except Exception:
            pass

        # 2. Monotonicidad
        try:
            mono = probe.is_monotone()
            if mono:
                cand = TheoremCandidate(
                    name=f"monotone_{fn_name}",
                    statement=f"La función {fn_name} es monótona en {domain}",
                    formal_statement=(
                        f"theorem monotone_{fn_name} : Monotone {fn_name} := by\n"
                        f"  sorry -- Gideon seed: verificar con análisis de derivada"
                    ),
                    status=TheoremStatus.NUMERICAL,
                    confidence=0.85,
                    tags=["monotone", "order"],
                )
                cand.lean_skeleton = cand.formal_statement
                new_cands.append(cand)
        except Exception:
            pass

        # 3. Simetría (par / impar)
        try:
            sym = probe.symmetry_type()
            if sym in ("even", "odd"):
                parity_stmt = f"{fn_name}(-x) = {fn_name}(x)" if sym == "even" else f"{fn_name}(-x) = -{fn_name}(x)"
                cand = TheoremCandidate(
                    name=f"{sym}_function_{fn_name}",
                    statement=f"La función {fn_name} es {'par' if sym == 'even' else 'impar'}: {parity_stmt}",
                    formal_statement=(
                        f"theorem {sym}_{fn_name} : ∀ x : ℝ, {parity_stmt} := by\n"
                        f"  sorry -- Gideon seed: verificar algebraicamente"
                    ),
                    status=TheoremStatus.NUMERICAL,
                    confidence=0.9,
                    tags=["symmetry", sym],
                )
                cand.lean_skeleton = cand.formal_statement
                new_cands.append(cand)
        except Exception:
            pass

        # 4. α-complejidad (invariante ACF)
        try:
            alpha = probe.alpha_complexity()
            if math.isfinite(alpha) and alpha > 0:
                cand = TheoremCandidate(
                    name=f"acf_alpha_{fn_name}",
                    statement=(
                        f"El índice α-ACF de {fn_name} en {domain} es ≈ {alpha:.4f}"
                    ),
                    formal_statement=(
                        f"-- ACF Alpha invariant for {fn_name}\n"
                        f"-- α ≈ {alpha:.4f}: tasa de decaimiento espectral\n"
                        f"noncomputable def acf_alpha_{fn_name} : ℝ := {alpha:.6f}"
                    ),
                    status=TheoremStatus.NUMERICAL,
                    evidence=[alpha],
                    confidence=0.75,
                    tags=["acf", "alpha_complexity", "spectral"],
                )
                cand.lean_skeleton = cand.formal_statement
                new_cands.append(cand)
        except Exception:
            pass

        # 5. Contractividad
        try:
            is_contr, L_contr = PatternMatcher.is_contraction(fn, domain, n=200)
            if is_contr:
                cand = TheoremCandidate(
                    name=f"contraction_{fn_name}",
                    statement=f"La función {fn_name} es una contracción con k ≈ {L_contr:.4f} < 1",
                    formal_statement=(
                        f"theorem contraction_{fn_name} : ∃ k : ℝ, k < 1 ∧\n"
                        f"  ∀ x y ∈ [{domain[0]}, {domain[1]}],\n"
                        f"  |{fn_name}(x) - {fn_name}(y)| ≤ k * |x - y| := by\n"
                        f"  use {L_contr:.6f}\n"
                        f"  sorry -- Gideon seed: verificar vía Banach"
                    ),
                    status=TheoremStatus.NUMERICAL,
                    evidence=[L_contr],
                    confidence=0.8,
                    tags=["contraction", "banach", "fixed_point"],
                )
                cand.lean_skeleton = cand.formal_statement
                new_cands.append(cand)
        except Exception:
            pass

        self._candidates.extend(new_cands)
        return new_cands

    # ── Exportación ───────────────────────────────────────────────────────────

    def export_lean_file(self, path: str, module_name: str = "GideonTheorems") -> None:
        """Genera un archivo Lean 4 con todos los esqueletos de candidatos."""
        lines = [
            f"-- {module_name}.lean",
            f"-- Generado automáticamente por Gideon TheoremSeeds",
            f"-- Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"-- NOTA: Los 'sorry' deben ser reemplazados por pruebas formales.",
            "",
            "import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic",
            "import Mathlib.Topology.MetricSpace.Lipschitz",
            "",
            f"namespace {module_name}",
            "",
        ]
        for cand in self._candidates:
            if cand.lean_skeleton:
                lines.append(f"-- [{cand.status.value.upper()}] {cand.name}")
                lines.append(f"-- Confidence: {cand.confidence:.3f}")
                lines.append(cand.lean_skeleton)
                lines.append("")
        lines.append(f"end {module_name}")
        content = "\n".join(lines)
        with open(path, "w") as f:
            f.write(content)

    def summary(self) -> str:
        by_status: Dict[str, int] = {}
        for c in self._candidates:
            k = c.status.value
            by_status[k] = by_status.get(k, 0) + 1
        lines = [
            f"GideonTheoremSeeds: {len(self._candidates)} candidatos",
        ]
        for k, v in by_status.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
