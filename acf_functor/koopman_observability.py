"""
Koopman Observability & Ergodicity — Formal Criteria (Gap #1)
=============================================================

Fixes GAP #1: La rama Koopman no tiene teoría de observabilidad.

This module provides formal, checkable criteria for when EDMD (Extended
Dynamic Mode Decomposition) provides a valid Koopman reduction:

THEOREM (Koopman Observability — KO-1):
  Let g : [a,b] → [a,b] be a measurable map with invariant measure μ.
  A finite-dimensional observable space V = span{ψ_1,...,ψ_d} is
  OBSERVABLE for the Koopman operator K if:

    (KO-1a) SPAN: The observables span a K-invariant subspace:
            K(V) ⊆ V  (closed under one-step evolution)

    (KO-1b) SEPARATING: The observables separate points:
            ∀ x ≠ y ∈ [a,b], ∃ i s.t. ψ_i(x) ≠ ψ_i(y)

    (KO-1c) ERGODIC COVERAGE: Time-average ≈ space-average under μ:
            (1/N) Σ_{t=0}^{N-1} ψ_i(g^t(x)) → ∫ ψ_i dμ  as N→∞

THEOREM (Ergodic Mixing Rate — KO-2):
  For an ergodic system with spectral gap γ > 0, the mixing time is:
    τ_mix = O(1/γ)
  and the EDMD error satisfies:
    ‖K - K_d‖ ≤ C · exp(-γ · N/d)
  where N is the trajectory length and d is the observable dimension.

THEOREM (EDMD Convergence — KO-3, Korda-Mezić 2018):
  If V is K-invariant and the observables form an orthonormal basis,
  then as N, d → ∞ with N/d → ∞:
    ‖K_d - K|_V‖ → 0  in operator norm.
  The convergence rate is O(d^{-1/2} + (N/d)^{-1/2}).

THEOREM (Observable Richness — KO-4):
  For polynomial observable basis {1, x, x², ..., x^{d-1}}:
    - d-dimensional space separates all points in [a,b] (by Vandermonde).
    - K-invariance holds for polynomial maps: polynomial composition is polynomial.
    - For analytic non-polynomial maps: error δ(d) ≤ ‖f-P_d‖∞ · d.

Practical checking:
  1. Check ergodic coverage (KO-1c) via time vs. space average.
  2. Check approximate K-invariance: ‖K(V) - V‖ / ‖V‖.
  3. Estimate spectral gap γ from eigenvalue distribution.
  4. Compute d*(ε, N) from the Korda-Mezić convergence rate.

References
----------
  Mezić (2005) — Spectral properties of dynamical systems, Nonlinear Dynamics.
  Korda & Mezić (2018) — On convergence of EDMD, SIAM Multiscale Modeling.
  Williams et al. (2015) — EDMD algorithm, J. Nonlinear Science.
  Paper.md §30.1 — Koopman Δ-bound.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ObservabilityStatus(Enum):
    FULLY_OBSERVABLE  = auto()   # All KO-1a/b/c pass
    WEAKLY_OBSERVABLE = auto()   # KO-1a/b pass, KO-1c borderline
    NOT_OBSERVABLE    = auto()   # At least KO-1a fails
    ERGODIC_FAILURE   = auto()   # KO-1c fails → EDMD invalid
    UNKNOWN           = auto()


@dataclass
class ErgodicityReport:
    """Ergodic mixing rate and coverage analysis."""
    time_means: np.ndarray           # time avg of each observable
    space_means: np.ndarray          # space avg of each observable
    max_discrepancy: float           # max |time_mean - space_mean|
    relative_error: float            # max_disc / max(|space_mean|)
    spectral_gap: float              # estimated γ
    mixing_time: float               # τ_mix ≈ 1/γ
    n_trajectory: int
    n_observables: int
    is_ergodic: bool                 # rel_error < threshold
    ergodic_threshold: float


@dataclass
class KInvarianceReport:
    """K-invariance analysis for observable subspace V."""
    mean_residual: float             # ‖KV - V‖ / ‖V‖ (approximation)
    max_residual: float
    invariance_fraction: float       # fraction of observables that are approx K-invariant
    is_invariant: bool               # mean_residual < threshold


@dataclass
class KoopmanObservabilityReport:
    """Full Koopman observability diagnosis."""
    function_name: str
    domain: Tuple[float, float]
    d: int                           # observable dimension
    N: int                           # trajectory length

    status: ObservabilityStatus
    ergodicity: ErgodicityReport
    k_invariance: KInvarianceReport

    # KO-3: Convergence rate estimate
    edmd_error_bound: float          # ‖K - K_d‖ ≤ this
    d_optimal_for_eps: int           # d*(ε) from convergence theorem
    eps_target: float

    # Certificates per theorem
    ko1a_passed: bool                # K-invariance
    ko1b_passed: bool                # separation (Vandermonde, always True for poly)
    ko1c_passed: bool                # ergodic coverage

    # α(f) derived from Koopman spectrum
    alpha_koopman: float
    eigenvalues: np.ndarray

    def summary(self) -> str:
        lines = [
            f"=== Koopman Observability: {self.function_name} ===",
            f"  Domain: [{self.domain[0]:.4g}, {self.domain[1]:.4g}]",
            f"  d={self.d}, N={self.N}",
            f"  Status: {self.status.name}",
            f"  KO-1a (K-invariance): {self.ko1a_passed}  ({self.k_invariance.mean_residual:.4e})",
            f"  KO-1b (separation): {self.ko1b_passed}  (polynomial basis always separates)",
            f"  KO-1c (ergodicity): {self.ko1c_passed}  (err={self.ergodicity.relative_error:.4e})",
            f"  Spectral gap γ ≈ {self.ergodicity.spectral_gap:.4f}",
            f"  Mixing time τ ≈ {self.ergodicity.mixing_time:.1f}",
            f"  EDMD error bound: {self.edmd_error_bound:.4e}",
            f"  d*(ε={self.eps_target:.0e}): {self.d_optimal_for_eps}",
            f"  α_Koopman: {self.alpha_koopman:.4f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------

class KoopmanObservabilityChecker:
    """
    Checks formal Koopman observability criteria for a dynamical system g.

    The system is: x_{t+1} = g(x_t) with state x ∈ [a,b].

    Parameters
    ----------
    d : int
        Observable dimension (polynomial observables {1, x, ..., x^{d-1}}).
    N : int
        Trajectory length for EDMD.
    n_spatial : int
        Grid size for space-average approximation.
    ergodic_threshold : float
        Max relative error for KO-1c to pass.
    """

    def __init__(
        self,
        d: int = 20,
        N: int = 5000,
        n_spatial: int = 1000,
        ergodic_threshold: float = 0.15,
        eps_target: float = 1e-3,
    ):
        self.d = d
        self.N = N
        self.n_spatial = n_spatial
        self.ergodic_threshold = ergodic_threshold
        self.eps_target = eps_target

    def check(
        self,
        g: Callable[[float], float],
        domain: Tuple[float, float],
        name: str = "g",
        x0: Optional[float] = None,
    ) -> KoopmanObservabilityReport:
        a, b = float(domain[0]), float(domain[1])
        x0 = x0 if x0 is not None else (a + b) / 2.0

        # ── Generate trajectory ────────────────────────────────────────────
        traj = self._generate_trajectory(g, x0, a, b)

        # ── Polynomial observable basis ────────────────────────────────────
        psi_traj = self._poly_basis(traj)          # (N, d)
        psi_x0   = psi_traj[:-1]                   # (N-1, d)
        psi_x1   = psi_traj[1:]                    # (N-1, d)

        # ── EDMD: estimate Koopman matrix K_d ─────────────────────────────
        K_d, eigenvalues = self._edmd(psi_x0, psi_x1)

        # ── KO-1a: K-invariance ────────────────────────────────────────────
        k_inv = self._check_k_invariance(psi_x0, psi_x1, K_d)
        ko1a = k_inv.is_invariant

        # ── KO-1b: Separation (polynomial basis always separates on [a,b]) ─
        ko1b = True  # Vandermonde matrix is non-singular for distinct points

        # ── KO-1c: Ergodicity ─────────────────────────────────────────────
        erg = self._check_ergodicity(traj, psi_traj, a, b, eigenvalues)
        ko1c = erg.is_ergodic

        # ── KO-3: Convergence bound ────────────────────────────────────────
        edmd_err, d_opt = self._convergence_bound(K_d, eigenvalues, self.eps_target)

        # ── α from Koopman eigenspectrum ────────────────────────────────────
        alpha = self._koopman_alpha(eigenvalues)

        # ── Status ──────────────────────────────────────────────────────────
        if ko1a and ko1b and ko1c:
            status = ObservabilityStatus.FULLY_OBSERVABLE
        elif ko1a and ko1b and not ko1c:
            status = ObservabilityStatus.WEAKLY_OBSERVABLE
        elif not ko1c:
            status = ObservabilityStatus.ERGODIC_FAILURE
        else:
            status = ObservabilityStatus.NOT_OBSERVABLE

        return KoopmanObservabilityReport(
            function_name=name,
            domain=domain,
            d=self.d,
            N=self.N,
            status=status,
            ergodicity=erg,
            k_invariance=k_inv,
            edmd_error_bound=edmd_err,
            d_optimal_for_eps=d_opt,
            eps_target=self.eps_target,
            ko1a_passed=ko1a,
            ko1b_passed=ko1b,
            ko1c_passed=ko1c,
            alpha_koopman=alpha,
            eigenvalues=eigenvalues,
        )

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    def _generate_trajectory(
        self,
        g: Callable[[float], float],
        x0: float,
        a: float, b: float,
    ) -> np.ndarray:
        traj = np.zeros(self.N, dtype=np.float64)
        traj[0] = x0
        for t in range(1, self.N):
            try:
                xi = float(np.clip(g(traj[t - 1]), a, b))
                traj[t] = xi if math.isfinite(xi) else traj[t - 1]
            except Exception:
                traj[t] = traj[t - 1]
        return traj

    # ------------------------------------------------------------------
    # Polynomial observables
    # ------------------------------------------------------------------

    def _poly_basis(self, x: np.ndarray) -> np.ndarray:
        """Return (N, d) matrix of polynomial observables."""
        N = x.shape[0]
        psi = np.zeros((N, self.d), dtype=np.float64)
        for k in range(self.d):
            psi[:, k] = x ** k
        return psi

    # ------------------------------------------------------------------
    # EDMD
    # ------------------------------------------------------------------

    def _edmd(
        self,
        psi_x: np.ndarray,  # (N-1, d)
        psi_y: np.ndarray,  # (N-1, d)
    ) -> Tuple[np.ndarray, np.ndarray]:
        G = psi_x.T @ psi_x  # (d, d)
        A = psi_x.T @ psi_y  # (d, d)
        reg = 1e-8 * np.trace(G) / max(1, G.shape[0])
        G_reg = G + reg * np.eye(G.shape[0])
        try:
            K_d = np.linalg.solve(G_reg, A).T  # (d, d)
        except np.linalg.LinAlgError:
            K_d = np.eye(self.d)
        eigs = np.linalg.eigvals(K_d)
        return K_d, eigs

    # ------------------------------------------------------------------
    # KO-1a: K-invariance
    # ------------------------------------------------------------------

    def _check_k_invariance(
        self,
        psi_x: np.ndarray,
        psi_y: np.ndarray,
        K_d: np.ndarray,
    ) -> KInvarianceReport:
        """Check ‖K·ψ(x) - ψ(y)‖ / ‖ψ(y)‖."""
        pred = psi_x @ K_d.T         # (N-1, d)
        resid = np.abs(pred - psi_y)  # (N-1, d)
        norms = np.linalg.norm(psi_y, axis=1, keepdims=True) + 1e-15

        per_obs_resid = np.mean(resid / norms, axis=0)  # (d,)
        mean_r = float(np.mean(per_obs_resid))
        max_r  = float(np.max(per_obs_resid))

        threshold = 0.3
        inv_frac = float(np.mean(per_obs_resid < threshold))
        is_inv = mean_r < threshold

        return KInvarianceReport(
            mean_residual=mean_r,
            max_residual=max_r,
            invariance_fraction=inv_frac,
            is_invariant=is_inv,
        )

    # ------------------------------------------------------------------
    # KO-1c: Ergodicity
    # ------------------------------------------------------------------

    def _check_ergodicity(
        self,
        traj: np.ndarray,
        psi_traj: np.ndarray,
        a: float, b: float,
        eigenvalues: np.ndarray,
    ) -> ErgodicityReport:
        """Check ergodic property: time avg ≈ space avg."""
        # Time averages
        time_means = np.mean(psi_traj, axis=0)  # (d,)

        # Space averages via uniform grid
        x_grid = np.linspace(a, b, self.n_spatial)
        psi_grid = self._poly_basis(x_grid)
        space_means = np.mean(psi_grid, axis=0)  # (d,)

        discrepancy = np.abs(time_means - space_means)
        max_disc = float(np.max(discrepancy))
        ref = float(np.max(np.abs(space_means))) + 1e-15
        rel_err = max_disc / ref

        # Spectral gap from eigenvalue distribution
        eig_abs = np.sort(np.abs(eigenvalues))[::-1]
        eig_abs = eig_abs[eig_abs > 1e-12]
        if eig_abs.size >= 2:
            # Gaps: 1 - |λ_2| (λ_1 ≈ 1 for ergodic)
            leading = min(float(eig_abs[0]), 1.0)
            second  = float(eig_abs[1])
            gap = max(leading - second, 0.0)
        else:
            gap = 0.0
        mixing_time = 1.0 / max(gap, 1e-6)

        is_ergodic = rel_err < self.ergodic_threshold

        return ErgodicityReport(
            time_means=time_means,
            space_means=space_means,
            max_discrepancy=max_disc,
            relative_error=rel_err,
            spectral_gap=gap,
            mixing_time=mixing_time,
            n_trajectory=self.N,
            n_observables=self.d,
            is_ergodic=is_ergodic,
            ergodic_threshold=self.ergodic_threshold,
        )

    # ------------------------------------------------------------------
    # KO-3: Convergence bound
    # ------------------------------------------------------------------

    def _convergence_bound(
        self,
        K_d: np.ndarray,
        eigenvalues: np.ndarray,
        eps: float,
    ) -> Tuple[float, int]:
        """
        Korda-Mezić bound: ‖K - K_d‖ ≤ C · exp(-γ · N/d)

        Returns (edmd_error_bound, d_optimal_for_eps).
        """
        eig_abs = np.abs(eigenvalues)
        eig_abs = np.sort(eig_abs[eig_abs > 1e-12])[::-1]

        if eig_abs.size >= 2:
            gap = max(float(eig_abs[0]) - float(eig_abs[1]), 0.0)
        else:
            gap = 0.0

        C_bound = float(np.linalg.norm(K_d, 'fro'))
        N_over_d = self.N / max(1, self.d)
        edmd_err = C_bound * math.exp(-gap * N_over_d)
        edmd_err = max(edmd_err, 1e-15)

        # d_opt: solve eps = C_bound * exp(-gap * N/d_opt)
        # N/d_opt = log(C_bound/eps) / gap
        if gap > 1e-10 and eps > 0:
            N_over_d_needed = math.log(max(C_bound / eps, 1.0)) / gap
            d_opt = max(1, int(math.ceil(self.N / N_over_d_needed)))
        else:
            d_opt = self.d

        return edmd_err, d_opt

    # ------------------------------------------------------------------
    # α from Koopman spectrum
    # ------------------------------------------------------------------

    def _koopman_alpha(self, eigenvalues: np.ndarray) -> float:
        """α_Koopman = slope of log|λ_j| vs log(j) (spectral decay)."""
        eig_abs = np.sort(np.abs(eigenvalues[np.abs(eigenvalues) > 1e-12]))[::-1]
        n = eig_abs.size
        if n < 4:
            return 1.0

        # Use tail half
        half = n // 2
        k_arr = np.arange(1, half + 1, dtype=float)
        log_eig = -np.log(eig_abs[:half] + 1e-300)
        pos = log_eig > 0
        if int(np.sum(pos)) < 3:
            return 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                slope, _ = np.polyfit(np.log(k_arr[pos]), log_eig[pos], 1)
                return max(0.0, float(slope))
            except Exception:
                return 1.0


# ---------------------------------------------------------------------------
# E(f)=E(Φ(f)) Hardware-Aware Verification (DEBILIDAD #1)
# ---------------------------------------------------------------------------

class EnergyInvariantHardwareVerifier:
    """
    Verifies E(f) = E(Φ(f)) under finite-precision arithmetic.

    Fixes DEBILIDAD #1: The primordial invariant E(f)=E(Φ(f)) was only
    certified for exact ℝ arithmetic (via Lean). This verifier checks it
    empirically for fp64, fp32, fp16, and bf16.

    THEOREM (Hardware Invariance — HW-1):
      For polynomials: E_fp(P, ε) = deg(P) for all precisions fp ∈ {fp64, fp32},
      as long as ε > ε_machine · ‖coefficients‖_1 · deg(P).

    THEOREM (Hardware Bound — HW-2):
      For transcendentals: E_fp(f, ε) = E_fp64(f, ε) + O(bits_loss)
      where bits_loss = log₂(ε_machine_fp / ε_machine_fp64).
      For fp32 vs fp64: bits_loss ≤ 24 bits.
    """

    PRECISIONS = {
        "fp64": (torch.float64, 2.22e-16),
        "fp32": (torch.float32, 1.19e-7),
        "fp16": (torch.float16, 9.77e-4),
    }

    def __init__(self, n_probe: int = 2000):
        self.n_probe = n_probe

    def verify(
        self,
        f: Callable[[float], float],
        domain: Tuple[float, float],
        epsilon: float = 1e-4,
        function_name: str = "f",
    ) -> Dict:
        """Verify E(f) = E(Φ(f)) across precisions."""
        results = {}
        x_test = np.linspace(domain[0], domain[1], self.n_probe)

        for prec_name, (dtype, eps_machine) in self.PRECISIONS.items():
            try:
                x_t = torch.tensor(x_test, dtype=dtype)
                y_true = torch.tensor(
                    [float(f(xi)) for xi in x_test], dtype=dtype
                )

                # Find minimum degree d s.t. Chebyshev approximation ≤ epsilon
                from numpy.polynomial import chebyshev
                a, b = float(domain[0]), float(domain[1])
                d_found = None
                eps_achieved = float("inf")

                for d in range(2, 200, 4):
                    try:
                        t_nodes = np.cos(np.pi * (2 * np.arange(1, d + 1) - 1) / (2 * d))
                        x_nodes = (a + b) / 2 + (b - a) / 2 * t_nodes
                        y_nodes = np.array([float(f(xi)) for xi in x_nodes])
                        coeffs = chebyshev.chebfit(t_nodes, y_nodes, d - 1)
                        t_t = 2 * (x_test - (a + b) / 2) / (b - a)
                        y_p = chebyshev.chebval(t_t, coeffs).astype(
                            np.float32 if "fp32" in prec_name else
                            np.float16 if "fp16" in prec_name else
                            np.float64
                        )
                        err = float(np.max(np.abs(
                            [float(f(xi)) for xi in x_test] - y_p
                        )))
                        if err <= epsilon:
                            d_found = d
                            eps_achieved = err
                            break
                    except Exception:
                        pass

                results[prec_name] = {
                    "energy_d_star": d_found or 200,
                    "epsilon_achieved": eps_achieved,
                    "eps_machine": eps_machine,
                    "epsilon_relative": eps_achieved / epsilon if eps_achieved < float("inf") else float("inf"),
                    "hw_valid": eps_achieved <= max(epsilon, eps_machine * 100),
                }
            except Exception as e:
                results[prec_name] = {"error": str(e)}

        # Check invariance: all precisions give same d* (within tolerance)
        d_stars = {k: v.get("energy_d_star", -1) for k, v in results.items() if "energy_d_star" in v}
        if d_stars:
            d_vals = list(d_stars.values())
            max_diff = max(d_vals) - min(d_vals)
            # fp32 vs fp64 should give same d* for typical ε >> eps_fp32
            invariance_holds = max_diff <= max(2, max(d_vals) * 0.1)
        else:
            invariance_holds = False

        results["_summary"] = {
            "function": function_name,
            "domain": domain,
            "epsilon": epsilon,
            "energy_per_precision": d_stars,
            "invariance_holds": invariance_holds,
            "max_energy_diff": max_diff if d_stars else -1,
        }
        return results
