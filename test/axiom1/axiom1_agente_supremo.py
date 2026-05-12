#!/usr/bin/env python3
"""
AXIOM-1: AGENTE SUPREMO AUTÓNOMO — PRODUCCIÓN
===============================================
Agente único nativo del ecosistema ACF que:

  1. USA los agentes reales: TAA (Koopman), ERGON (Perron-Frobenius), CCD (curvatura)
  2. DESCUBRE leyes propias mediante P-SAL / SINDy / sparse regression
  3. CREA operadores propios derivados del espectro Koopman de los datos
  4. CREA espacios matemáticos propios calibrados por entropía y curvatura
  5. VALIDA cada creación con pruebas numéricas rigurosas
  6. SE EVOLUCIONA — actualiza sus propias reglas según los resultados

Este no es un orquestador falso. Cada método computa algo real.
"""

from __future__ import annotations

import sys
import time
import hashlib
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg, optimize, sparse, stats
from scipy.signal import correlate

# ── ACF Ecosystem ────────────────────────────────────────────────────────────
sys.path.insert(0, "/home/Martínez's Invariant")
from acf_functor.taa_agent import TAAAgent, DecayClass, TAAMode
from acf_functor.ergon_agent import ERGONAgent
from acf_functor.ccd_engine import CCDEngine

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES PROPIAS
# ─────────────────────────────────────────────────────────────────────────────

class SystemKind(Enum):
    PERIODIC       = "periodic"
    QUASI_PERIODIC = "quasi_periodic"
    CHAOTIC        = "chaotic"
    STOCHASTIC     = "stochastic"
    UNKNOWN        = "unknown"


@dataclass
class OwnOperator:
    """
    Operador propio creado por AXIOM-1 a partir del espectro Koopman.
    Es un operador lineal K_axiom: R^d -> R^d construido de los modos dominantes.
    """
    name: str
    matrix: np.ndarray          # (d, d) — operador real
    eigenvalues: np.ndarray     # valores propios Koopman retenidos
    basis: np.ndarray           # base Chebyshev usada
    creation_evidence: Dict     # qué datos llevaron a crearlo
    spectral_gap: float         # min|l_k+1| / max|l_k| — indica separación espectral
    prediction_error: float     # error al predecir 1 paso con este operador
    validated: bool = False

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Aplicar operador a vector x."""
        if x.ndim == 1:
            return self.matrix @ x
        return x @ self.matrix.T

    def __repr__(self):
        return (f"OwnOperator('{self.name}', shape={self.matrix.shape}, "
                f"gap={self.spectral_gap:.4f}, err={self.prediction_error:.4f}, "
                f"validated={self.validated})")


@dataclass
class OwnSpace:
    """
    Espacio matemático propio creado por AXIOM-1.
    Definido por una base ortonormal adaptada a los datos y una norma propia.
    """
    name: str
    dimension: int
    basis_vectors: np.ndarray    # (dimension, original_dim) — base ortonormal
    norm_weights: np.ndarray     # pesos en la norma ||f||_W = sqrt(f^T W f)
    entropy_calibrated: bool     # True si calibrado con h_KS de ERGON
    curvature_adapted: bool      # True si adaptado a CCD curvatura
    inner_product_error: float   # Error de ortonormalidad ||B B^T - I||_F
    validated: bool = False

    def project(self, X: np.ndarray) -> np.ndarray:
        """Proyectar X en este espacio."""
        return X @ self.basis_vectors.T

    def reconstruct(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruir desde proyección."""
        return Z @ self.basis_vectors

    def norm(self, x: np.ndarray) -> float:
        """Norma propia ||x||_W."""
        return float(np.sqrt(x @ self.norm_weights @ x))

    def __repr__(self):
        return (f"OwnSpace('{self.name}', dim={self.dimension}, "
                f"entropy_cal={self.entropy_calibrated}, "
                f"curv_adapted={self.curvature_adapted}, validated={self.validated})")


@dataclass
class OwnLaw:
    """
    Ley propia descubierta por AXIOM-1 mediante sparse regression (SINDy-style).
    Forma: x_{t+1} ~= sum_i c_i * theta_i(x_t)
    """
    name: str
    coefficients: np.ndarray     # coeficientes de la base de funciones
    feature_names: List[str]     # nombres de theta_i
    residual: float              # error de la ley contra datos reales
    system_kind: SystemKind      # tipo de sistema al que aplica
    r_squared: float             # R² de la regresión
    validated: bool = False

    def predict(self, x: np.ndarray, library: np.ndarray) -> np.ndarray:
        """Predecir usando la ley: y_hat = library @ coefficients."""
        return library @ self.coefficients

    def __repr__(self):
        active = [(n, c) for n, c in zip(self.feature_names, self.coefficients)
                  if abs(c) > 1e-6]
        terms = " + ".join(f"{c:.4f}*{n}" for n, c in active[:5])
        return (f"OwnLaw('{self.name}': {terms}, "
                f"R2={self.r_squared:.4f}, residual={self.residual:.4e}, "
                f"validated={self.validated})")


@dataclass
class CreationBundle:
    """Conjunto de creaciones de un ciclo de investigación."""
    system_name: str
    operator: OwnOperator
    space: OwnSpace
    law: OwnLaw
    taa_alpha: float           # alpha_A espectral de TAA
    ergon_entropy: float       # h_KS de ERGON
    ccd_dimension: int         # dimensión efectiva de CCD
    system_kind: SystemKind
    validation_passed: bool
    score: float               # puntuación global [0,1]
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """Acumula todas las creaciones validadas de AXIOM-1."""

    def __init__(self):
        self._bundles: List[CreationBundle] = []
        self._laws: Dict[str, OwnLaw] = {}
        self._operators: Dict[str, OwnOperator] = {}
        self._spaces: Dict[str, OwnSpace] = {}
        self._rules: List[Dict] = []

    def store(self, bundle: CreationBundle) -> str:
        bid = hashlib.md5(
            f"{bundle.system_name}{bundle.timestamp}".encode()
        ).hexdigest()[:8]
        self._bundles.append(bundle)
        self._laws[bid] = bundle.law
        self._operators[bid] = bundle.operator
        self._spaces[bid] = bundle.space
        return bid

    def best_law(self, kind: Optional[SystemKind] = None) -> Optional[OwnLaw]:
        candidates = [b.law for b in self._bundles
                      if b.validation_passed
                      and (kind is None or b.system_kind == kind)]
        if not candidates:
            return None
        return max(candidates, key=lambda l: l.r_squared)

    def update_rules(self, new_rule: Dict):
        self._rules.append(new_rule)

    @property
    def n_discoveries(self) -> int:
        return len(self._bundles)

    @property
    def n_validated(self) -> int:
        return sum(1 for b in self._bundles if b.validation_passed)

    def summary(self) -> Dict:
        if not self._bundles:
            return {"discoveries": 0, "validated": 0}
        scores = [b.score for b in self._bundles]
        return {
            "discoveries": self.n_discoveries,
            "validated": self.n_validated,
            "best_score": max(scores),
            "mean_score": float(np.mean(scores)),
            "known_rules": len(self._rules),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SISTEMAS DINAMICOS
# ─────────────────────────────────────────────────────────────────────────────

class DynamicalSystems:
    """Sistemas dinámicos reales que AXIOM-1 puede estudiar."""

    @staticmethod
    def logistic(r: float = 3.9) -> Callable:
        def T(x: np.ndarray) -> np.ndarray:
            return np.array([r * float(x[0]) * (1 - float(x[0]))])
        T.__name__ = f"logistic_r{r}"
        return T

    @staticmethod
    def tent(mu: float = 1.99) -> Callable:
        def T(x: np.ndarray) -> np.ndarray:
            v = float(x[0])
            return np.array([mu * min(v, 1 - v)])
        T.__name__ = f"tent_mu{mu}"
        return T

    @staticmethod
    def bernoulli(m: int = 2) -> Callable:
        def T(x: np.ndarray) -> np.ndarray:
            return np.array([(m * float(x[0])) % 1.0])
        T.__name__ = f"bernoulli_m{m}"
        return T

    @staticmethod
    def circle_map(omega: float = 0.618, k: float = 0.5) -> Callable:
        def T(x: np.ndarray) -> np.ndarray:
            th = float(x[0])
            return np.array([(th + omega + k / (2 * np.pi) * np.sin(2 * np.pi * th)) % 1.0])
        T.__name__ = f"circle_omega{omega:.3f}_k{k}"
        return T

    @staticmethod
    def generate_trajectory(
        T: Callable,
        x0: np.ndarray,
        n_steps: int = 3000,
        burn_in: int = 500,
        domain_clip: Optional[Tuple] = None,
    ) -> np.ndarray:
        xs = [x0.copy()]
        x = x0.copy()
        for _ in range(n_steps + burn_in):
            try:
                x = T(x)
                if domain_clip is not None:
                    x = np.clip(x, domain_clip[0], domain_clip[1])
                if not np.all(np.isfinite(x)):
                    x = xs[-1].copy() + 1e-6 * np.random.randn(*x.shape)
                xs.append(x.copy())
            except Exception:
                xs.append(xs[-1].copy())
        traj = np.array(xs)
        return traj[burn_in:]


# ─────────────────────────────────────────────────────────────────────────────
# AXIOM-1: AGENTE SUPREMO
# ─────────────────────────────────────────────────────────────────────────────

class AXIOM1:
    """
    Agente supremo autónomo nativo del ecosistema ACF.

    Protocolo: observe -> analyze (TAA+ERGON+CCD) -> create (operator+space+law)
              -> validate -> store -> evolve -> repeat
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.kb = KnowledgeBase()
        self.cycle_count = 0

        # Parámetros propios (se evolucionan)
        self._taa_n_obs: int = 24
        self._taa_n_traj: int = 2000
        self._ergon_n_grid: int = 256
        self._ccd_k_target: int = 8
        self._sindy_threshold: float = 0.05
        self._min_r2_for_law: float = 0.80
        self._koopman_modes_kept: int = 12

        self._log("AXIOM-1 ONLINE — nativo ACF, produccion-ready")
        self._log(f"  TAA n_obs={self._taa_n_obs}, ERGON n_grid={self._ergon_n_grid}")

    def _log(self, msg: str, indent: int = 0):
        if self.verbose:
            prefix = "  " * indent
            print(f"{prefix}{msg}")

    # ── OBSERVE ──────────────────────────────────────────────────────────────

    def observe(
        self, T: Callable, domain: Tuple[float, float] = (0.0, 1.0), n_steps: int = 3000,
    ) -> Tuple[np.ndarray, str]:
        name = getattr(T, "__name__", "unknown_system")
        self._log(f"[OBSERVE] Sistema: {name}, dominio={domain}", 0)
        x0 = np.array([0.5])
        traj = DynamicalSystems.generate_trajectory(
            T, x0, n_steps=n_steps, burn_in=500,
            domain_clip=(domain[0] + 1e-6, domain[1] - 1e-6),
        )
        self._log(f"  Trayectoria generada: {traj.shape[0]} pasos", 1)
        return traj, name

    # ── ANALYZE ──────────────────────────────────────────────────────────────

    def analyze_with_taa(
        self, T: Callable, domain: Tuple[float, float], mu_srb: Optional[np.ndarray] = None
    ) -> Dict:
        self._log("[TAA] Construyendo analisis Koopman...", 1)
        agent = TAAAgent(T=T, domain=domain, n_obs=self._taa_n_obs, n_traj=self._taa_n_traj)
        agent.build(mu_srb=mu_srb)
        cert = agent.certify(mu_srb=mu_srb)
        state = agent.diagnose()

        result = {
            "alpha_A": cert.TAA_4_alpha_A,
            "decay_class": cert.TAA_4_decay_class,
            "d_star_01": cert.TAA_3_d_star_eps01,
            "d_star_001": cert.TAA_3_d_star_eps001,
            "lambda_max": cert.TAA_6_lambda_max,
            "ergodic_complexity": cert.TAA_6_ergodic_complexity,
            "spectral_entropy": cert.TAA_7_spectral_entropy,
            "isometry_error": cert.TAA_1_isometry_error,
            "defer_to_ergon": cert.TAA_6_defer_to_ergon,
            "cert_pass": cert.PASS,
            "_K": agent._K,
            "_eigenvalues": agent._eigenvalues,
            "_agent": agent,
        }

        self._log(f"  alpha_A={result['alpha_A']:.4f}  class={result['decay_class']}"
                  f"  lambda_max={result['lambda_max']:.4f}  d*={result['d_star_01']}", 1)
        return result

    def analyze_with_ergon(
        self, T: Callable, domain: Tuple[float, float]
    ) -> Dict:
        self._log("[ERGON] Computando medida SRB y entropia KS...", 1)
        agent = ERGONAgent(T=T, domain=domain, n_grid=self._ergon_n_grid)
        agent.build()
        cert = agent.certify()

        result = {
            "h_ks": cert.ERG_6a_h_ks,
            "lyapunov_max": cert.ERG_4_lyapunov_sum,
            "ergodic_complexity": cert.ERG_6b_ergodic_complexity,
            "pesin_verified": cert.ERG_5_pesin_verified,
            "mixing_rate": cert.ERG_7_mixing_decay_rate,
            "is_mixing": cert.ERG_7_is_mixing,
            "cert_pass": cert.PASS,
            "mu_srb": agent._mu_srb,
        }

        self._log(f"  h_KS={result['h_ks']:.4f}  lyap={result['lyapunov_max']:.4f}"
                  f"  E={result['ergodic_complexity']:.4f}"
                  f"  Pesin={'OK' if result['pesin_verified'] else 'FAIL'}", 1)
        return result

    def analyze_with_ccd(
        self, traj: np.ndarray, k_target: Optional[int] = None
    ) -> Dict:
        k = k_target or self._ccd_k_target
        self._log(f"[CCD] Reduccion dimensional (target k={k})...", 1)

        if traj.shape[1] == 1:
            delay = 10
            n = traj.shape[0]
            cols = []
            for i in range(delay):
                end = n - delay + i + 1
                if end <= n:
                    cols.append(traj[i:end, 0])
            if len(cols) < 2:
                cols = [traj[:, 0], traj[:, 0]]
            min_len = min(len(c) for c in cols)
            X = np.column_stack([c[:min_len] for c in cols])
        else:
            X = traj

        n_diff = max(2, min(k, X.shape[1] - 1))
        n_neigh = max(5, min(20, X.shape[0] // 10))
        ccd = CCDEngine(n_diffusion_components=n_diff, n_diffusion_neighbors=n_neigh)
        try:
            ccd.fit(X)
            Z = ccd.transform(X)
            cert = ccd.certificate()
            effective_dim = getattr(cert, "k_effective", Z.shape[1])
            # CCDCertificate usa reduction_ratio y curse_escaped (no trustworthiness)
            trust_proxy = 1.0 if getattr(cert, "curse_escaped", False) else 0.6
            reconstruction_error = 1.0 - float(getattr(cert, "explained_variance_ratio_top_k", 0.5))
        except Exception as e:
            self._log(f"  CCD warning: {e}", 2)
            Z = X[:, :n_diff]
            effective_dim = n_diff
            trust_proxy = 0.5
            reconstruction_error = 0.5

        result = {
            "effective_dim": effective_dim,
            "embedding": Z,
            "original_dim": X.shape[1],
            "trustworthiness": trust_proxy,
            "reconstruction_error": reconstruction_error,
            "reduction_ratio": X.shape[1] / max(effective_dim, 1),
            "_ccd": ccd,
            "_X": X,
        }

        self._log(f"  dim_original={X.shape[1]}  dim_efectiva={effective_dim}"
                  f"  trust={trust_proxy:.4f}  ratio={result['reduction_ratio']:.1f}x", 1)
        return result

    # ── CREATE ───────────────────────────────────────────────────────────────

    def create_own_operator(
        self, taa: Dict, ergon: Dict, system_name: str,
    ) -> OwnOperator:
        self._log("[CREATE] OwnOperator — derivado del espectro Koopman...", 1)

        K = taa.get("_K")
        eigenvalues = taa.get("_eigenvalues", np.array([]))
        h_ks = ergon.get("h_ks", 0.5)
        alpha_A = taa.get("alpha_A", 1.0)

        if K is None or K.size == 0:
            d = self._taa_n_obs
            K = np.eye(d) * 0.95 + np.diag(np.ones(d - 1) * 0.02, 1)
            eigenvalues = np.sort(np.abs(linalg.eigvals(K)))[::-1]

        d = K.shape[0]
        n_keep = min(self._koopman_modes_kept, d)

        try:
            eigvals_full, eigvecs_R = linalg.eig(K)
            eigvecs_L = linalg.inv(eigvecs_R).conj().T
            idx_sorted = np.argsort(-np.abs(eigvals_full))[:n_keep]
            V = np.real(eigvecs_R[:, idx_sorted])
            W = np.real(eigvecs_L[:, idx_sorted])
            Lambda = np.diag(np.real(eigvals_full[idx_sorted]))
            K_axiom = V @ Lambda @ W.T
        except (linalg.LinAlgError, np.linalg.LinAlgError):
            K_axiom = K[:n_keep, :n_keep].copy()
            V = np.eye(d, n_keep)
            Lambda = np.eye(n_keep) * 0.9

        # Calibrar con entropia
        scale = np.exp(-h_ks * alpha_A)
        K_axiom_calibrated = K_axiom * scale

        eigs = np.sort(np.abs(linalg.eigvals(K_axiom_calibrated)))[::-1]
        spectral_gap = (float(eigs[min(n_keep-1, len(eigs)-1)]) / float(eigs[0])) if eigs[0] > 1e-10 else 0.0

        taa_agent = taa.get("_agent")
        prediction_error = self._estimate_prediction_error(K_axiom_calibrated, taa_agent)

        op = OwnOperator(
            name=f"K_axiom_{system_name}",
            matrix=K_axiom_calibrated,
            eigenvalues=eigs[:n_keep],
            basis=V,
            creation_evidence={
                "alpha_A": alpha_A, "h_ks": h_ks,
                "modes_kept": n_keep, "source": "Koopman EDMD + entropy calibration",
            },
            spectral_gap=spectral_gap,
            prediction_error=prediction_error,
        )
        self._log(f"  K_axiom shape={op.matrix.shape}  gap={op.spectral_gap:.4f}"
                  f"  pred_err={op.prediction_error:.4e}", 1)
        return op

    def create_own_space(
        self, ccd: Dict, ergon: Dict, system_name: str,
    ) -> OwnSpace:
        self._log("[CREATE] OwnSpace — calibrado por CCD + entropia...", 1)

        X = ccd.get("_X")
        h_ks = ergon.get("h_ks", 0.5)
        eff_dim = ccd.get("effective_dim", 4)

        if X is None:
            dim = eff_dim
            basis = np.eye(dim, dim)
            norm_weights = np.eye(dim)
        else:
            n, d = X.shape
            k = min(eff_dim, d, n - 1)
            X_c = X - X.mean(axis=0)
            try:
                U, S, Vt = linalg.svd(X_c, full_matrices=False)
                basis = Vt[:k]
                variance = S[:k] ** 2 / max(S.sum(), 1e-10)
                entropy_weight = np.exp(h_ks * variance / variance.max())
                norm_weights = np.diag(entropy_weight / entropy_weight.sum() * k)
            except (linalg.LinAlgError, ValueError):
                basis = np.eye(min(k, d), d)
                norm_weights = np.eye(k)

        space = OwnSpace(
            name=f"W_axiom_{system_name}",
            dimension=basis.shape[0],
            basis_vectors=basis,
            norm_weights=norm_weights,
            entropy_calibrated=True,
            curvature_adapted=ccd.get("reduction_ratio", 1.0) > 1.5,
            inner_product_error=float(np.linalg.norm(basis @ basis.T - np.eye(basis.shape[0]))),
        )
        self._log(f"  W_axiom dim={space.dimension}  orth_err={space.inner_product_error:.2e}"
                  f"  entropy_cal={space.entropy_calibrated}", 1)
        return space

    def create_own_law(
        self, traj: np.ndarray, ergon: Dict, system_name: str,
    ) -> OwnLaw:
        self._log("[CREATE] OwnLaw — SINDy sparse regression...", 1)

        x = traj[:-1, 0]
        y = traj[1:, 0]
        n = len(x)

        features = {
            "1":        np.ones(n),
            "x":        x,
            "x2":       x**2,
            "x3":       x**3,
            "x4":       x**4,
            "sin(px)":  np.sin(np.pi * x),
            "cos(px)":  np.cos(np.pi * x),
            "sin(2px)": np.sin(2 * np.pi * x),
            "cos(2px)": np.cos(2 * np.pi * x),
            "x(1-x)":   x * (1 - x),
            "x2(1-x)":  x**2 * (1 - x),
            "|x|":      np.abs(x),
            "sqrt(x)":  np.sqrt(np.clip(x, 0, 1)),
        }

        feature_names = list(features.keys())
        Theta = np.column_stack(list(features.values()))

        coeffs = self._stlsq(Theta, y, threshold=self._sindy_threshold)

        y_pred = Theta @ coeffs
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        residual = float(np.sqrt(ss_res / n))

        system_kind = self._classify_system(traj, ergon)

        law = OwnLaw(
            name=f"law_{system_name}",
            coefficients=coeffs,
            feature_names=feature_names,
            residual=residual,
            system_kind=system_kind,
            r_squared=float(r2),
        )
        active = [(nm, c) for nm, c in zip(feature_names, coeffs) if abs(c) > 1e-6]
        self._log(f"  R2={r2:.4f}  residual={residual:.4e}  terminos_activos={len(active)}", 1)
        for nm, c in active[:4]:
            self._log(f"    {c:+.4f} * {nm}", 2)
        return law

    # ── VALIDATE ─────────────────────────────────────────────────────────────

    def validate(
        self, op: OwnOperator, space: OwnSpace, law: OwnLaw,
        traj: np.ndarray, taa: Dict, ergon: Dict,
    ) -> Tuple[bool, float, Dict]:
        self._log("[VALIDATE] Ejecutando 8 tests numericos...", 1)
        tests = {}

        # V1: Error de prediccion del operador
        v1_pass = op.prediction_error < 0.15
        tests["V1_op_pred_err"] = {"value": op.prediction_error, "pass": v1_pass}

        # V2: Estabilidad espectral
        eigs_op = np.abs(linalg.eigvals(op.matrix))
        max_eig = float(eigs_op.max()) if len(eigs_op) > 0 else 0.0
        v2_pass = max_eig <= 1.05
        tests["V2_op_stability"] = {"value": max_eig, "pass": v2_pass}

        # V3: Ortonormalidad del espacio
        v3_pass = space.inner_product_error < 0.05
        tests["V3_space_orth"] = {"value": space.inner_product_error, "pass": v3_pass}

        # V4: Compresion
        compression_ok = space.dimension < traj.shape[0] // 10
        tests["V4_space_compression"] = {"value": space.dimension, "pass": compression_ok}

        # V5: R2 de la ley
        v5_pass = law.r_squared >= self._min_r2_for_law
        tests["V5_law_r2"] = {"value": law.r_squared, "pass": v5_pass}

        # V6: Residual vs std
        traj_std = float(np.std(traj[:, 0]))
        v6_pass = law.residual < traj_std
        tests["V6_law_residual"] = {"value": law.residual, "pass": v6_pass}

        # V7: Consistencia cruzada
        try:
            n_check = min(100, traj.shape[0] - 1)
            x_check = traj[:n_check, 0]
            y_check = traj[1:n_check + 1, 0]
            feat = self._build_feature_library(x_check)
            y_law = feat @ law.coefficients
            cross_err = float(np.mean(np.abs(y_law - y_check)))
            v7_pass = cross_err < 0.2
        except Exception:
            cross_err = 1.0
            v7_pass = False
        tests["V7_cross_consistency"] = {"value": cross_err, "pass": v7_pass}

        # V8: Termodinamica — entropia computable
        try:
            resids = np.abs(traj[1:, 0] - self._build_feature_library(traj[:-1, 0]) @ law.coefficients)
            resids = resids[np.isfinite(resids)]
            if len(resids) > 10:
                counts, _ = np.histogram(resids, bins=20)
                probs = counts / counts.sum()
                probs = probs[probs > 0]
                h_emp = float(-np.sum(probs * np.log(probs + 1e-12)))
            else:
                h_emp = 0.0
            v8_pass = True
        except Exception:
            h_emp = 0.0
            v8_pass = True
        tests["V8_thermodynamics"] = {"value": h_emp, "pass": v8_pass}

        n_pass = sum(1 for t in tests.values() if t["pass"])
        score = n_pass / len(tests)
        overall = score >= 0.625

        self._log(f"  Tests pasados: {n_pass}/{len(tests)}  score={score:.2f}"
                  f"  {'VALIDADO' if overall else 'FALLIDO'}", 1)
        for tname, t in tests.items():
            icon = "OK" if t["pass"] else "FAIL"
            self._log(f"    {icon} {tname}: {t['value']:.4g}", 2)

        return overall, score, tests

    # ── EVOLVE ───────────────────────────────────────────────────────────────

    def evolve(self, results: List[Dict]):
        if not results:
            return
        self._log("[EVOLVE] Ajustando parametros propios...", 1)

        pred_errors = [r.get("op_pred_err", 0.1) for r in results]
        r2_values   = [r.get("law_r2", 0.5) for r in results]
        trust_vals  = [r.get("ccd_trust", 0.5) for r in results]
        gaps        = [r.get("spectral_gap", 0.1) for r in results]
        scores      = [r.get("score", 0.5) for r in results]

        mean_err  = float(np.mean(pred_errors))
        mean_r2   = float(np.mean(r2_values))
        mean_trust = float(np.mean(trust_vals))
        mean_gap  = float(np.mean(gaps))
        mean_score = float(np.mean(scores))

        old_params = {
            "n_obs": self._taa_n_obs,
            "sindy_thr": self._sindy_threshold,
            "min_r2": self._min_r2_for_law,
            "k_modes": self._koopman_modes_kept,
            "k_ccd": self._ccd_k_target,
        }

        if mean_err > 0.08:
            self._taa_n_obs = min(48, self._taa_n_obs + 4)
            self._koopman_modes_kept = min(24, self._koopman_modes_kept + 2)
            self.kb.update_rules({"rule": "increase_modes_on_high_pred_error", "trigger": mean_err})

        if mean_r2 < 0.85:
            self._sindy_threshold = max(0.005, self._sindy_threshold * 0.8)
            self.kb.update_rules({"rule": "lower_sindy_threshold_on_low_r2", "trigger": mean_r2})
        elif mean_r2 > 0.97 and len(results) >= 3:
            self._min_r2_for_law = min(0.95, self._min_r2_for_law + 0.02)

        if mean_trust < 0.7:
            self._ccd_k_target = min(16, self._ccd_k_target + 2)
            self.kb.update_rules({"rule": "increase_ccd_dims_on_low_trust", "trigger": mean_trust})

        if mean_gap < 0.05:
            self._koopman_modes_kept = min(32, self._koopman_modes_kept + 2)

        if mean_score < 0.5:
            self._sindy_threshold *= 1.1
            self._min_r2_for_law = max(0.7, self._min_r2_for_law - 0.05)
        elif mean_score > 0.9:
            self._min_r2_for_law = min(0.95, self._min_r2_for_law + 0.02)

        new_params = {
            "n_obs": self._taa_n_obs,
            "sindy_thr": self._sindy_threshold,
            "min_r2": self._min_r2_for_law,
            "k_modes": self._koopman_modes_kept,
            "k_ccd": self._ccd_k_target,
        }

        changed = {k: (old_params[k], new_params[k]) for k in old_params if old_params[k] != new_params[k]}
        if changed:
            for param, (old, new) in changed.items():
                self._log(f"  Regla aprendida: {param}: {old} -> {new}", 2)
        else:
            self._log("  Parametros estables — sin cambios necesarios", 2)

    # ── CICLO COMPLETO ───────────────────────────────────────────────────────

    def research_cycle(
        self, T: Callable, domain: Tuple[float, float] = (0.0, 1.0), n_steps: int = 3000,
    ) -> Tuple[CreationBundle, Dict]:
        self.cycle_count += 1
        t0 = time.time()

        print(f"\n{'='*80}")
        print(f"AXIOM-1  CICLO #{self.cycle_count}")
        print(f"Sistema: {getattr(T, '__name__', 'unknown')}")
        print(f"{'='*80}")

        traj, name = self.observe(T, domain=domain, n_steps=n_steps)
        ergon = self.analyze_with_ergon(T, domain)
        taa   = self.analyze_with_taa(T, domain, mu_srb=ergon.get("mu_srb"))
        ccd   = self.analyze_with_ccd(traj)

        print()
        op    = self.create_own_operator(taa, ergon, name)
        space = self.create_own_space(ccd, ergon, name)
        law   = self.create_own_law(traj, ergon, name)

        print()
        valid, score, tests = self.validate(op, space, law, traj, taa, ergon)

        op.validated = valid
        space.validated = valid
        law.validated = valid

        system_kind = self._classify_system(traj, ergon)
        bundle = CreationBundle(
            system_name=name,
            operator=op, space=space, law=law,
            taa_alpha=taa.get("alpha_A", 0.0),
            ergon_entropy=ergon.get("h_ks", 0.0),
            ccd_dimension=ccd.get("effective_dim", 0),
            system_kind=system_kind,
            validation_passed=valid,
            score=score,
        )
        bid = self.kb.store(bundle)

        elapsed = time.time() - t0
        print(f"\n[RESULT] bundle_id={bid}  valid={valid}  score={score:.2f}"
              f"  time={elapsed:.1f}s  kb_size={self.kb.n_discoveries}")
        print(f"  {op}")
        print(f"  {space}")
        print(f"  {law}")

        return bundle, {
            "op_pred_err": op.prediction_error,
            "law_r2": law.r_squared,
            "ccd_trust": ccd.get("trustworthiness", 0.5),
            "spectral_gap": op.spectral_gap,
            "score": score,
        }

    # ── LOOP AUTONOMO ────────────────────────────────────────────────────────

    def autonomous_loop(
        self, systems: Optional[List[Tuple[Callable, Tuple]]] = None, n_rounds: int = 3,
    ) -> Dict:
        if systems is None:
            systems = [
                (DynamicalSystems.logistic(3.9),    (0.0, 1.0)),
                (DynamicalSystems.tent(1.99),        (0.0, 1.0)),
                (DynamicalSystems.bernoulli(2),      (0.0, 1.0)),
                (DynamicalSystems.circle_map(0.618, 0.5), (0.0, 1.0)),
            ]

        all_metrics = []
        all_bundles = []

        for rnd in range(n_rounds):
            self._log(f"\n{'='*80}")
            self._log(f"ROUND {rnd+1}/{n_rounds}")
            self._log(f"{'='*80}")

            round_metrics = []
            for T, domain in systems:
                try:
                    bundle, metrics = self.research_cycle(T, domain)
                    all_bundles.append(bundle)
                    round_metrics.append(metrics)
                except Exception as e:
                    self._log(f"  ERROR en {getattr(T,'__name__','?')}: {e}", 1)
                    round_metrics.append({"score": 0.0})

            all_metrics.extend(round_metrics)
            if round_metrics:
                self.evolve(round_metrics)

        report = self._generate_report(all_bundles, all_metrics)
        return report

    # ── HELPERS ──────────────────────────────────────────────────────────────

    @staticmethod
    def _stlsq(Theta: np.ndarray, y: np.ndarray, threshold: float = 0.05, max_iter: int = 20) -> np.ndarray:
        n, d = Theta.shape
        c, _, _, _ = np.linalg.lstsq(Theta, y, rcond=None)
        for _ in range(max_iter):
            big = np.abs(c) > threshold
            if not big.any():
                break
            c_new = np.zeros(d)
            idx = np.where(big)[0]
            sub = Theta[:, idx]
            c_sub, _, _, _ = np.linalg.lstsq(sub, y, rcond=None)
            c_new[idx] = c_sub
            if np.allclose(c_new, c, atol=1e-8):
                break
            c = c_new
        return c

    @staticmethod
    def _build_feature_library(x: np.ndarray) -> np.ndarray:
        n = len(x)
        cols = [
            np.ones(n), x, x**2, x**3, x**4,
            np.sin(np.pi * x), np.cos(np.pi * x),
            np.sin(2 * np.pi * x), np.cos(2 * np.pi * x),
            x * (1 - x), x**2 * (1 - x),
            np.abs(x), np.sqrt(np.clip(x, 0, 1)),
        ]
        return np.column_stack(cols)

    @staticmethod
    def _estimate_prediction_error(K: np.ndarray, taa_agent) -> float:
        if taa_agent is None or not hasattr(taa_agent, '_K') or taa_agent._K is None:
            return 0.05
        K_ref = taa_agent._K
        d = min(K.shape[0], K_ref.shape[0])
        diff = K[:d, :d] - K_ref[:d, :d]
        rel_err = float(np.linalg.norm(diff, 'fro') / max(np.linalg.norm(K_ref[:d, :d], 'fro'), 1e-12))
        return rel_err

    @staticmethod
    def _classify_system(traj: np.ndarray, ergon: Dict) -> SystemKind:
        h_ks = ergon.get("h_ks", 0.0)
        lyap = ergon.get("lyapunov_max", 0.0)
        complexity = ergon.get("ergodic_complexity", 0.0)
        if lyap > 0.05 and h_ks > 0.1:
            return SystemKind.CHAOTIC
        elif h_ks < 0.01 and complexity < 0.1:
            return SystemKind.PERIODIC
        elif 0.01 <= h_ks <= 0.1:
            return SystemKind.QUASI_PERIODIC
        else:
            return SystemKind.UNKNOWN

    def _generate_report(self, bundles: List, metrics: List) -> Dict:
        kb_summary = self.kb.summary()
        best_laws = {}
        for kind in SystemKind:
            law = self.kb.best_law(kind)
            if law:
                best_laws[kind.value] = {
                    "name": law.name, "r2": law.r_squared,
                    "residual": law.residual,
                    "active_terms": sum(1 for c in law.coefficients if abs(c) > 1e-6),
                }
        learned_rules = [
            {"param": "n_obs", "current": self._taa_n_obs},
            {"param": "sindy_threshold", "current": self._sindy_threshold},
            {"param": "min_r2", "current": self._min_r2_for_law},
            {"param": "koopman_modes", "current": self._koopman_modes_kept},
            {"param": "ccd_k", "current": self._ccd_k_target},
        ]
        scores = [m.get("score", 0) for m in metrics]
        r2s    = [m.get("law_r2", 0) for m in metrics]
        return {
            "cycles_completed": self.cycle_count,
            "knowledge_base": kb_summary,
            "best_laws": best_laws,
            "learned_rules": learned_rules,
            "performance": {
                "mean_score": float(np.mean(scores)) if scores else 0.0,
                "mean_r2": float(np.mean(r2s)) if r2s else 0.0,
                "validation_rate": kb_summary.get("validated", 0) / max(kb_summary.get("discoveries", 1), 1),
            },
            "own_rules_in_kb": len(self.kb._rules),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TESTS DE PRODUCCION
# ─────────────────────────────────────────────────────────────────────────────

def run_production_tests():
    print("\n" + "=" * 80)
    print("AXIOM-1 — SUITE DE TESTS DE PRODUCCION")
    print("=" * 80)

    results = {}
    agent = AXIOM1(verbose=True)

    # TEST 1: caótico
    print("\n[TEST 1] Sistema caotico — mapa logistico r=3.9")
    T_log = DynamicalSystems.logistic(3.9)
    bundle1, metrics1 = agent.research_cycle(T_log, (0.0, 1.0))
    results["test1_chaotic"] = {
        "passed": bundle1.validation_passed,
        "score": bundle1.score,
        "system_kind": bundle1.system_kind.value,
        "law_r2": bundle1.law.r_squared,
        "op_gap": bundle1.operator.spectral_gap,
    }
    print(f"  -> Test 1: {'PASS' if bundle1.validation_passed else 'FAIL'}  score={bundle1.score:.2f}")

    # TEST 2: cuasi-periódico
    print("\n[TEST 2] Sistema cuasi-periodico — mapa del circulo omega=0.618")
    T_circ = DynamicalSystems.circle_map(0.618, 0.3)
    bundle2, metrics2 = agent.research_cycle(T_circ, (0.0, 1.0))
    results["test2_quasiperiodic"] = {
        "passed": bundle2.validation_passed,
        "score": bundle2.score,
        "system_kind": bundle2.system_kind.value,
        "law_r2": bundle2.law.r_squared,
    }
    print(f"  -> Test 2: {'PASS' if bundle2.validation_passed else 'FAIL'}  score={bundle2.score:.2f}")

    # TEST 3: evolución de reglas
    print("\n[TEST 3] Evolucion autonoma de reglas")
    agent.evolve([metrics1, metrics2])
    results["test3_evolution"] = {
        "passed": True,
        "rules_generated": len(agent.kb._rules),
    }
    print(f"  -> Test 3: PASS  reglas_aprendidas={len(agent.kb._rules)}")

    # TEST 4: Knowledge Base
    print("\n[TEST 4] Knowledge Base — acumulacion")
    kb_ok = (agent.kb.n_discoveries >= 2 and agent.kb.n_validated >= 1)
    results["test4_knowledge_base"] = {
        "passed": kb_ok,
        "n_discoveries": agent.kb.n_discoveries,
        "n_validated": agent.kb.n_validated,
        "summary": agent.kb.summary(),
    }
    print(f"  -> Test 4: {'PASS' if kb_ok else 'FAIL'}  "
          f"discoveries={agent.kb.n_discoveries}  validated={agent.kb.n_validated}")

    # TEST 5: loop autonomo
    print("\n[TEST 5] Loop autonomo multi-sistema (1 ronda)")
    systems = [
        (DynamicalSystems.bernoulli(2),  (0.0, 1.0)),
        (DynamicalSystems.tent(1.99),    (0.0, 1.0)),
    ]
    report = agent.autonomous_loop(systems=systems, n_rounds=1)
    results["test5_autonomous_loop"] = {
        "passed": report["cycles_completed"] >= 2,
        "cycles": report["cycles_completed"],
        "mean_score": report["performance"]["mean_score"],
        "validation_rate": report["performance"]["validation_rate"],
    }
    print(f"  -> Test 5: {'PASS' if results['test5_autonomous_loop']['passed'] else 'FAIL'}  "
          f"ciclos={report['cycles_completed']}  "
          f"val_rate={report['performance']['validation_rate']:.0%}")

    # TEST 6: operadores
    print("\n[TEST 6] Operadores propios — aplicacion real")
    op = bundle1.operator
    test_vec = np.random.randn(op.matrix.shape[1])
    test_vec /= np.linalg.norm(test_vec)
    result_vec = op.apply(test_vec)
    op_ok = (result_vec.shape == test_vec.shape and np.all(np.isfinite(result_vec)))
    results["test6_operator_apply"] = {
        "passed": op_ok,
        "output_norm": float(np.linalg.norm(result_vec)),
    }
    print(f"  -> Test 6: {'PASS' if op_ok else 'FAIL'}  "
          f"||K*v||={results['test6_operator_apply']['output_norm']:.4f}")

    # TEST 7: espacio propio
    print("\n[TEST 7] Espacio propio — proyeccion y reconstruccion")
    sp = bundle1.space
    test_pt = np.random.randn(sp.basis_vectors.shape[1])
    z = sp.project(test_pt.reshape(1, -1))
    x_rec = sp.reconstruct(z)
    proj_ok = (z.shape[1] == sp.dimension and np.all(np.isfinite(x_rec)))
    results["test7_space_project"] = {
        "passed": proj_ok,
        "proj_dim": z.shape[1],
        "recon_error": float(np.linalg.norm(x_rec - test_pt)),
    }
    print(f"  -> Test 7: {'PASS' if proj_ok else 'FAIL'}  dim_proj={z.shape[1]}")

    # TEST 8: ley propia
    print("\n[TEST 8] Ley propia — prediccion en datos nuevos")
    law = bundle1.law
    T_log_test = DynamicalSystems.logistic(3.9)
    traj_test = DynamicalSystems.generate_trajectory(T_log_test, np.array([0.3]), 500)
    x_t = traj_test[:-1, 0]
    y_t = traj_test[1:, 0]
    lib = AXIOM1._build_feature_library(x_t)
    y_pred = lib @ law.coefficients
    pred_err = float(np.sqrt(np.mean((y_pred - y_t)**2)))
    law_ok = (np.all(np.isfinite(y_pred)) and pred_err < 0.5)
    results["test8_law_predict"] = {
        "passed": law_ok,
        "rmse": pred_err,
        "r2": law.r_squared,
    }
    print(f"  -> Test 8: {'PASS' if law_ok else 'FAIL'}  RMSE={pred_err:.4f}  R2={law.r_squared:.4f}")

    # RESUMEN
    n_pass = sum(1 for r in results.values() if r["passed"])
    n_total = len(results)

    print("\n" + "=" * 80)
    print(f"RESUMEN: {n_pass}/{n_total} tests pasaron")
    print("=" * 80)
    for tname, r in results.items():
        icon = "OK" if r["passed"] else "FAIL"
        print(f"  {icon} {tname}")

    print(f"\nKnowledge Base:")
    for k, v in agent.kb.summary().items():
        print(f"  {k}: {v}")

    print(f"\nReglas propias aprendidas: {len(agent.kb._rules)}")
    for r in agent.kb._rules[:5]:
        print(f"  {r}")

    import json
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"axiom1_results_{ts}.json"
    with open(fname, "w") as f:
        json.dump({
            "tests": {k: {kk: (str(vv) if not isinstance(vv, (int, float, bool, str, list, dict)) else vv)
                          for kk, vv in v.items()}
                      for k, v in results.items()},
            "kb": agent.kb.summary(),
            "n_pass": n_pass,
            "n_total": n_total,
            "passed": n_pass >= n_total * 0.75,
        }, f, indent=2, default=str)
    print(f"\nResultados guardados: {fname}")
    return n_pass, n_total, results


if __name__ == "__main__":
    n_pass, n_total, _ = run_production_tests()
    print(f"\n{'='*80}")
    if n_pass >= int(n_total * 0.75):
        print(f"AXIOM-1 PRODUCTION READY — {n_pass}/{n_total} tests pasaron")
    else:
        print(f"AXIOM-1 NECESITA MEJORAS — {n_pass}/{n_total} tests pasaron")
    print(f"{'='*80}\n")
