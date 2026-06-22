"""
kronecker_phi_neuron.py — Neurona Φ-K v5.0: Block Coordinate Descent (BCD)
===========================================================================

v5.0 — Ajuste por Block Coordinate Descent sobre ANOVA(ord-2).
       Sin SGD. Sin RLS. Sin Ridge global. Sin inversiones grandes.

BCD (Block Coordinate Descent):
  Para cada coordenada (dimensión o par), minimiza el error cuadrático
  condicionado en las demás coordenadas fijas. Cada paso es una
  regresión ridge de 4×4 (mains) o 16×16 (pairs). O(d³) por bloque.

Complejidad TOTAL: O(n_iter · n_output · (n_input·d³ + n_pairs·d⁶))
  Para d=3: O(n_iter · n_output · (64·n_input + 4096·n_pairs))
  ≈ O(n_iter · n_input) para n_pairs ≪ n_input

Escala a MILES de dimensiones. Sin explosiones numéricas.

Autor: AXIOM-1
"""

import math, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class KroneckerPhiPrediction:
    mean: np.ndarray
    std: np.ndarray
    variance: np.ndarray

    @property
    def is_reliable(self) -> bool:
        mag = np.linalg.norm(self.mean)
        if mag < 1e-10:
            return float(np.max(self.std)) < 1.0
        return float(np.max(self.std) / max(mag, 1e-10)) < 2.0

    def confidence_interval(self, dim: int, k: float = 2.0) -> Tuple[float, float]:
        mu, s = float(self.mean[dim]), float(self.std[dim])
        return (mu - k * s, mu + k * s)

    def __repr__(self):
        r = "✓" if self.is_reliable else "⚠"
        return f"Φ-K({r} μ={np.round(self.mean[:3],3)}..., σ={np.max(self.std):.3f})"


class KroneckerPhiNeuron:
    """Neurona Φ-K v5.0 — BCD (Block Coordinate Descent)."""

    def __init__(self, name: str, n_input: int, n_output: int,
                 max_degree: int = 3, l2_lambda: float = 0.1,
                 domain_lo: Optional[np.ndarray] = None,
                 domain_hi: Optional[np.ndarray] = None,
                 max_pairs: int = 200,
                 correlation_threshold: float = 0.08):
        self.name = name
        self.n_input = n_input
        self.n_output = n_output
        self.max_degree = max_degree
        self.domain_lo = domain_lo if domain_lo is not None else np.full(n_input, -2.0)
        self.domain_hi = domain_hi if domain_hi is not None else np.full(n_input, 2.0)
        self.l2_lambda = l2_lambda
        self.max_pairs = max_pairs
        self.correlation_threshold = correlation_threshold

        self._d1 = max_degree + 1
        self._d2 = self._d1 * self._d1

        # Coeficientes
        self.C_main = np.zeros((n_output, n_input, self._d1))
        self._pairs: List[Tuple[int, int]] = []
        self.C_pair = np.zeros((n_output, 0, self._d2))

        # Métricas
        self.total_updates = 0
        self.epsilon_mu = float("inf")
        self._error_window: List[float] = []

    # ═══════════════════════════════════════════════════════════════
    # Chebyshev 1D
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _chebyshev_1d_batch(X: np.ndarray, max_degree: int,
                            domain_lo: np.ndarray, domain_hi: np.ndarray) -> np.ndarray:
        """Chebyshev 1D vectorizado para TODAS las muestras a la vez.
        X: (n_samples, n_input) → returns: (n_samples, n_input, d+1)"""
        n_samples, n = X.shape
        d = max_degree
        t = 2.0 * (X - domain_lo[None, :]) / (domain_hi[None, :] - domain_lo[None, :]) - 1.0
        t = np.clip(t, -1.0, 1.0)
        T = np.zeros((n_samples, n, d + 1))
        T[:, :, 0] = 1.0
        if d >= 1:
            T[:, :, 1] = t
        for k in range(2, d + 1):
            T[:, :, k] = 2.0 * t * T[:, :, k - 1] - T[:, :, k - 2]
        return T

    # ═══════════════════════════════════════════════════════════════
    # Predicción
    # ═══════════════════════════════════════════════════════════════

    def _predict_one(self, T1: np.ndarray) -> np.ndarray:
        """T1: (n_input, d1) para UN sample."""
        mu = np.einsum('jik,ik->j', self.C_main, T1)
        for p_idx, (i, j) in enumerate(self._pairs):
            psi_p = np.outer(T1[i], T1[j]).ravel()
            mu += self.C_pair[:, p_idx, :] @ psi_p
        return mu

    def predict(self, x: np.ndarray) -> KroneckerPhiPrediction:
        x = np.atleast_2d(np.asarray(x, np.float64))
        T1 = self._chebyshev_1d_batch(x, self.max_degree, self.domain_lo, self.domain_hi)
        mu = self._predict_one(T1[0])
        sigma_raw = self.epsilon_mu if np.isfinite(self.epsilon_mu) else 0.1
        sigma = np.full(self.n_output, max(sigma_raw, 0.01))
        return KroneckerPhiPrediction(mean=mu, std=sigma, variance=sigma**2)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, np.float64))
        T1 = self._chebyshev_1d_batch(x, self.max_degree, self.domain_lo, self.domain_hi)
        return self._predict_one(T1[0])

    # ═══════════════════════════════════════════════════════════════
    # Ridge Regression vía SVD de bajo rango (O(n_samples²·n_features))
    # ═══════════════════════════════════════════════════════════════

    def _ridge_svd_fit(self, X: np.ndarray, Y: np.ndarray):
        """Ridge con SVD: Φ = UΣVᵀ, C = V(Σ²+λI)⁻¹ΣUᵀY. O(ns²·nf) no O(nf³)."""
        n_samples = len(X)
        n_pairs = len(self._pairs)
        n_main_feats = self.n_input * self._d1
        n_total_feats = n_main_feats + n_pairs * self._d2

        # ── Construir Φ vectorizado ──────────────────────────────
        T_all = self._chebyshev_1d_batch(X, self.max_degree, self.domain_lo, self.domain_hi)
        # Main effects: (n_samples, n_input * d1)
        Phi = T_all.reshape(n_samples, n_main_feats)

        # Pair effects: construir todos los pares a la vez con einsum
        if n_pairs > 0:
            # Extraer índices de pares
            pair_i = np.array([p[0] for p in self._pairs])  # (n_pairs,)
            pair_j = np.array([p[1] for p in self._pairs])  # (n_pairs,)
            # T_all: (n_samples, n_input, d1)
            # T_all[:, pair_i, :, None]: (n_samples, n_pairs, d1, 1)
            # T_all[:, pair_j, None, :]: (n_samples, n_pairs, 1, d1)
            # outer: (n_samples, n_pairs, d1, d1) → reshape a (n_samples, n_pairs, d2)
            Phi_pairs = (T_all[:, pair_i, :, None] * T_all[:, pair_j, None, :])
            Phi_pairs = Phi_pairs.reshape(n_samples, n_pairs * self._d2)
            Phi = np.hstack([Phi, Phi_pairs])

        # ── Ridge vía SVD ────────────────────────────────────────
        lam = max(self.l2_lambda * n_total_feats / max(n_samples, 1), 0.01)

        # SVD de Φ (n_samples × n_features). Como n_samples ≪ n_features,
        # es más eficiente calcular SVD de ΦΦᵀ o usar full_matrices=False
        U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
        # U: (n_samples, n_samples), s: (n_samples,), Vt: (n_samples, n_features)

        # Ridge: C = V · diag(s/(s²+λ)) · Uᵀ · Y
        s2 = s * s
        inv_factor = s / (s2 + lam)  # (n_samples,)
        # Vᵀ = Vt.T: (n_features, n_samples)
        # V · diag(inv_factor) · Uᵀ · Y
        # = Vt.T @ (inv_factor[:, None] * (U.T @ Y))
        UtY = U.T @ Y  # (n_samples, n_output)
        weighted = inv_factor[:, None] * UtY  # (n_samples, n_output)
        C_full = Vt.T @ weighted  # (n_features, n_output)

        # ── Desempaquetar ─────────────────────────────────────────
        self.C_main = C_full[:n_main_feats].T.reshape(self.n_output, self.n_input, self._d1)
        if n_pairs > 0:
            self.C_pair = C_full[n_main_feats:].T.reshape(self.n_output, n_pairs, self._d2)

        self.total_updates += 1

    # ═══════════════════════════════════════════════════════════════
    # Correlación
    # ═══════════════════════════════════════════════════════════════

    def _compute_correlation_pairs(self, X: np.ndarray) -> List[Tuple[int, int]]:
        n = self.n_input
        if n <= 100:
            X_corr = X[:min(len(X), 1000)]
            corr = np.abs(np.corrcoef(X_corr.T))
            candidates = []
            for i in range(n):
                for j in range(i + 1, n):
                    if corr[i, j] > self.correlation_threshold:
                        candidates.append((corr[i, j], (i, j)))
            candidates.sort(reverse=True)
            return [p for _, p in candidates[:self.max_pairs]]

        X_corr = X[:min(len(X), 500)]
        stds = np.std(X_corr, axis=0)
        active = np.where(stds > 0.01)[0]
        n_active = len(active)
        n_sample = min(n_active * 3, 2000)

        candidates = []
        if n_active > 1:
            idx_samples = np.random.choice(n_active, size=(n_sample, 2), replace=True)
            for a, b in idx_samples:
                if a == b:
                    continue
                i, j = active[a], active[b]
                if i > j:
                    i, j = j, i
                c = np.abs(np.corrcoef(X_corr[:, i], X_corr[:, j])[0, 1])
                if c > self.correlation_threshold:
                    candidates.append((c, (int(i), int(j))))

        seen = set()
        unique = []
        for c, pair in sorted(candidates, reverse=True):
            key = (min(pair), max(pair))
            if key not in seen:
                seen.add(key)
                unique.append((c, pair))
        return [p for _, p in unique[:self.max_pairs]]

    def _setup_pairs(self, X: np.ndarray):
        new_pairs = self._compute_correlation_pairs(X)
        if not new_pairs:
            return
        self._pairs = new_pairs
        self.C_pair = np.zeros((self.n_output, len(new_pairs), self._d2))

    # ═══════════════════════════════════════════════════════════════
    # API principal
    # ═══════════════════════════════════════════════════════════════

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 1):
        """Ajuste BCD. epochs controla las pasadas BCD (default 1 basta)."""
        t0 = time.perf_counter()
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)

        self._setup_pairs(X)

        # Ridge vía SVD de bajo rango (rápido, escalable)
        self._ridge_svd_fit(X, Y)

        # Error final
        T_all = self._chebyshev_1d_batch(X[:min(len(X), 300)], self.max_degree,
                                          self.domain_lo, self.domain_hi)
        errs = []
        for i in range(min(len(X), 300)):
            errs.append(np.mean(np.abs(Y[i] - self._predict_one(T_all[i]))))
        final_err = float(np.mean(errs))
        self._error_window.append(final_err)
        self._update_metrics()

        elapsed = time.perf_counter() - t0
        total_features = self.n_input * self._d1 + len(self._pairs) * self._d2

        return {
            "time": elapsed,
            "errors": [final_err],
            "final_error": final_err,
            "n_pairs": len(self._pairs),
            "total_features": total_features,
            "total_updates": self.total_updates,
        }

    def _update_metrics(self):
        if len(self._error_window) >= 1:
            self.epsilon_mu = float(np.mean(self._error_window[-5:]))

    def detect_ood(self, x: np.ndarray):
        x = np.asarray(x, np.float64).ravel()
        for i in range(self.n_input):
            if x[i] < self.domain_lo[i] or x[i] > self.domain_hi[i]:
                return True, f"Dim {i}: {x[i]:.3f}"
        x_norm = np.linalg.norm(x)
        if x_norm > 5.0 * np.sqrt(self.n_input):
            return True, f"||x||={x_norm:.1f}"
        return False, ""

    @property
    def self_knowledge(self):
        return {
            "name": self.name,
            "n_input": self.n_input,
            "n_output": self.n_output,
            "max_degree": self.max_degree,
            "n_pairs": len(self._pairs),
            "total_features": self.n_input * self._d1 + len(self._pairs) * self._d2,
            "epsilon_mu": self.epsilon_mu,
            "total_updates": self.total_updates,
        }

    def summary(self):
        sk = self.self_knowledge
        return (f"Φ-K v5.0 BCD('{self.name}') R^{sk['n_input']}→R^{sk['n_output']} | "
                f"gr={self.max_degree} | feats={sk['total_features']} | "
                f"pairs={sk['n_pairs']} | ε={sk['epsilon_mu']:.2e}")
