"""Thermodynamic formulation of the ACF truncation dimension selection.

This module implements Section 14.2 of Paper.md (previously "mentioned but
not formalized"):

  "The Functor exhibits pure thermodynamic phase transitions between sparse
   and dense dimensional configurations. By defining its entropy S_Φ(f,d)
   and E as the average computational error, we establish a strict free energy
   framework: F = E - S/β."

Free Energy Framework
---------------------
For a fixed target function f and truncation dimension d, define:

  E(f, d) = δ(d)       — reconstruction error at dimension d
  S(f, d) = log N(d)   — log-count of valid d-dimensional representations
                         ≈ d·log(m/d) when choosing d from m eigenmodes
  F(f, d, β) = E(f,d) - β⁻¹·S(f,d)  — Helmholtz free energy

Optimal dimension:
  d*(f, β) = argmin_d F(f, d, β)

Phase transitions:
  β → ∞ (zero temperature): d* → argmin_d E(d)   [minimize error, expensive]
  β → 0 (high temperature): d* → argmin_d -S(d)  [minimize entropy, compact]
  β = β_c               : phase transition (d* jumps abruptly)

Connection to MDL principle:
  The MDL (Minimum Description Length) principle is a special case at β=1:
  MDL cost = E(d) + S(d) = code length for data + model complexity.
  The thermodynamic formulation generalizes this with a temperature parameter
  that controls the tradeoff between accuracy and compression.

Connection to α(f):
  The optimal free energy F*(β) as a function of β reveals α(f):
  dF*/dβ ~ α(f) · β^{α(f)-1} at large β.

Modules
-------
  FreeEnergy          — compute F(f, d, β) and its derivatives
  CriticalityDetector — find β_c and d* phase transitions
  ThermodynamicACF    — unified API
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
import warnings

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FreeEnergyProfile:
    """Free energy F(d, β) for D = d_min, ..., d_max at fixed β."""
    beta: float
    dimensions: List[int]
    errors: List[float]           # E(d) = δ(d) values
    entropies: List[float]        # S(d) = log N(d) values
    free_energies: List[float]    # F(d, β) = E(d) - S(d)/β
    d_star: int                   # argmin_d F(d)
    f_star: float                 # F(d*, β)
    e_at_dstar: float             # E(d*)
    s_at_dstar: float             # S(d*)


@dataclass
class PhaseTransition:
    """Detected phase transition at critical β_c."""
    beta_c: float               # critical inverse temperature
    d_low: int                  # d* just below β_c (low-accuracy dim)
    d_high: int                 # d* just above β_c (high-accuracy dim)
    delta_d: int                # |d_high - d_low|  (jump size)
    energy_gap: float           # E(d_high) - E(d_low)
    entropy_gap: float          # S(d_high) - S(d_low)
    latent_heat: float          # Δ(free energy) at transition


@dataclass
class ThermodynamicReport:
    """Full thermodynamic analysis of an ACF reduction."""
    alpha: float                      # Índice Afín α(f) from spectral decay
    eigenvalues: torch.Tensor
    profiles: List[FreeEnergyProfile] # profiles at various β
    phase_transition: Optional[PhaseTransition]
    d_star_zero_temp: int             # d* at β→∞ (pure accuracy)
    d_star_high_temp: int             # d* at β→0  (pure compression)
    optimal_beta: float               # β that minimizes total cost
    mdl_dimension: int                # d* at β=1 (MDL principle)
    alpha_from_thermodynamics: float  # α estimated from dF*/dβ curve


# ---------------------------------------------------------------------------
# Entropy model
# ---------------------------------------------------------------------------

class EntropyModel:
    """
    Model for S(f, d) = log N(d) = log |{valid d-dim representations}|.

    The combinatorial count N(d) of choosing d modes from m eigenvalues is C(m, d).
    For large m,d: log C(m, d) ≈ d·log(m/d) + d  (Stirling approximation).

    We also include a spectral weight: mode j has "weight" log(1/|λ_j|),
    so the entropy of choosing the first d modes is:
    S_spectral(d) = -Σ_{j=1}^{d} log|λ_j|  (sum of log-inverse eigenvalues)
    """

    def __init__(self, m_total: int, eigenvalues: torch.Tensor):
        self.m = m_total
        self.eig = eigenvalues.double()

    def combinatorial(self, d: int) -> float:
        """S_comb(d) = log C(m, d) via Stirling."""
        if d <= 0 or d >= self.m:
            return 0.0
        # log C(m, d) = log m! - log d! - log(m-d)!
        # Stirling: log n! ≈ n log n - n
        def slog(n: int) -> float:
            return n * math.log(max(n, 1)) - n if n > 0 else 0.0
        return slog(self.m) - slog(d) - slog(self.m - d)

    def spectral(self, d: int) -> float:
        """S_spec(d) = -Σ_{j=1}^{d} log|λ_j| (information content of first d modes)."""
        if d <= 0:
            return 0.0
        d = min(d, self.eig.numel())
        return float(-torch.sum(torch.log(self.eig[:d] + 1e-300)).item())

    def mixed(self, d: int, weight_comb: float = 0.5) -> float:
        """Weighted combination of combinatorial and spectral entropy."""
        return weight_comb * self.combinatorial(d) + (1 - weight_comb) * self.spectral(d)


# ---------------------------------------------------------------------------
# Free energy computation
# ---------------------------------------------------------------------------

class FreeEnergyComputer:
    """
    Compute the Helmholtz free energy F(d, β) = E(d) - S(d)/β.
    """

    def __init__(
        self,
        eigenvalues: torch.Tensor,
        observable_norm: float = 1.0,
        entropy_mode: str = "spectral",  # "combinatorial" | "spectral" | "mixed"
    ):
        """
        Parameters
        ----------
        eigenvalues  : sorted (desc) Koopman eigenvalue magnitudes.
        observable_norm : ‖ψ‖, default 1.0.
        entropy_mode : how to compute S(d).
        """
        eig = eigenvalues.detach().double()
        eig = torch.abs(eig)
        eig, _ = torch.sort(eig, descending=True)
        self.eig = eig
        self.m = eig.numel()
        self.norm = float(observable_norm)
        self.entropy_model = EntropyModel(self.m, eig)
        self.entropy_mode = entropy_mode

    def error(self, d: int) -> float:
        """E(d) = δ(d) ≤ |λ_{d+1}| · ‖ψ‖."""
        if d >= self.m:
            return 0.0
        return float(self.eig[d].item()) * self.norm

    def entropy(self, d: int) -> float:
        """S(d) according to entropy_mode."""
        if self.entropy_mode == "combinatorial":
            return self.entropy_model.combinatorial(d)
        elif self.entropy_mode == "spectral":
            return self.entropy_model.spectral(d)
        else:
            return self.entropy_model.mixed(d)

    def free_energy(self, d: int, beta: float) -> float:
        """F(d, β) = E(d) - S(d)/β."""
        if beta <= 0:
            return -float("inf")
        return self.error(d) - self.entropy(d) / beta

    def profile(
        self,
        beta: float,
        d_min: int = 1,
        d_max: Optional[int] = None,
    ) -> FreeEnergyProfile:
        """Compute F(d, β) for all d in [d_min, d_max]."""
        d_max = d_max or self.m
        dims = list(range(d_min, min(d_max + 1, self.m + 1)))

        errors = [self.error(d) for d in dims]
        entropies = [self.entropy(d) for d in dims]
        free_es = [self.free_energy(d, beta) for d in dims]

        d_star_idx = int(np.argmin(free_es))
        d_star = dims[d_star_idx]

        return FreeEnergyProfile(
            beta=beta,
            dimensions=dims,
            errors=errors,
            entropies=entropies,
            free_energies=free_es,
            d_star=d_star,
            f_star=free_es[d_star_idx],
            e_at_dstar=errors[d_star_idx],
            s_at_dstar=entropies[d_star_idx],
        )

    def d_star(self, beta: float) -> int:
        """Quick computation of optimal d at given β."""
        return self.profile(beta).d_star


# ---------------------------------------------------------------------------
# Phase transition detector
# ---------------------------------------------------------------------------

class CriticalityDetector:
    """
    Find the critical inverse temperature β_c where d*(β) jumps.

    Method: sweep β from β_min to β_max and detect discontinuities in d*(β).
    """

    def __init__(
        self,
        computer: FreeEnergyComputer,
        beta_min: float = 0.01,
        beta_max: float = 100.0,
        n_beta: int = 200,
    ):
        self.computer = computer
        self.betas = np.geomspace(beta_min, beta_max, n_beta)

    def sweep(self) -> Tuple[List[float], List[int]]:
        """Return (betas, d_star_sequence)."""
        betas = self.betas.tolist()
        d_stars = [self.computer.d_star(b) for b in betas]
        return betas, d_stars

    def find_phase_transitions(self) -> List[PhaseTransition]:
        betas, d_stars = self.sweep()
        transitions = []

        for i in range(1, len(betas)):
            if d_stars[i] != d_stars[i - 1]:
                beta_c = (betas[i - 1] + betas[i]) / 2.0
                d_lo = d_stars[i - 1]
                d_hi = d_stars[i]
                e_lo = self.computer.error(d_lo)
                e_hi = self.computer.error(d_hi)
                s_lo = self.computer.entropy(d_lo)
                s_hi = self.computer.entropy(d_hi)
                F_lo = self.computer.free_energy(d_lo, beta_c)
                F_hi = self.computer.free_energy(d_hi, beta_c)
                transitions.append(PhaseTransition(
                    beta_c=beta_c,
                    d_low=d_lo,
                    d_high=d_hi,
                    delta_d=abs(d_hi - d_lo),
                    energy_gap=abs(e_hi - e_lo),
                    entropy_gap=abs(s_hi - s_lo),
                    latent_heat=abs(F_hi - F_lo),
                ))

        return transitions

    def alpha_from_free_energy(self, betas: List[float], d_star_seq: List[int]) -> float:
        """
        Estimate α(f) from the scaling dF*/dβ ~ α·β^{α-1}.

        We fit F*(β) = A·β^α for large β and extract α.
        """
        # F*(β) = F(d*(β), β)
        F_stars = [self.computer.free_energy(d, b)
                   for d, b in zip(d_star_seq, betas)]

        # Filter large β regime (upper 30%)
        n = len(betas)
        idx_start = int(0.7 * n)
        b_arr = np.array(betas[idx_start:])
        F_arr = np.array(F_stars[idx_start:])

        valid = (F_arr < 0) & (b_arr > 1.0)
        if valid.sum() < 3:
            return 1.0

        b_v = b_arr[valid]
        F_v = np.abs(F_arr[valid])

        # log|F*| ≈ α·log(β) + log(A)
        log_b = np.log(b_v)
        log_F = np.log(F_v + 1e-30)
        if log_b.std() < 1e-10:
            return 1.0
        slope = float(np.polyfit(log_b, log_F, 1)[0])
        return max(0.0, slope)


# ---------------------------------------------------------------------------
# Unified thermodynamic API
# ---------------------------------------------------------------------------

class ThermodynamicACF:
    """
    Full thermodynamic analysis of an ACF Koopman reduction.

    Usage:
        thermo = ThermodynamicACF(eigenvalues)
        report = thermo.analyze()
        print(thermo.summary(report))
        d_opt = report.d_star_zero_temp    # accuracy-first
        d_mdl = report.mdl_dimension       # MDL (balanced)
    """

    def __init__(
        self,
        eigenvalues: torch.Tensor,
        observable_norm: float = 1.0,
        entropy_mode: str = "spectral",
        beta_min: float = 0.01,
        beta_max: float = 200.0,
        n_beta: int = 300,
    ):
        self.eig = eigenvalues
        self.computer = FreeEnergyComputer(eigenvalues, observable_norm, entropy_mode)
        self.detector = CriticalityDetector(
            self.computer, beta_min, beta_max, n_beta
        )

    def analyze(
        self,
        beta_samples: Optional[List[float]] = None,
    ) -> ThermodynamicReport:
        """
        Run full thermodynamic analysis.

        Parameters
        ----------
        beta_samples : list of β values for profile computation.
                       Defaults to [0.1, 1.0, 5.0, 10.0, 50.0].
        """
        beta_list = beta_samples or [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
        profiles = [self.computer.profile(b) for b in beta_list]

        transitions = self.detector.find_phase_transitions()

        # Extremes
        d_hot  = self.computer.d_star(0.05)   # high-temp: most compressed
        d_cold = self.computer.d_star(1000.0)  # low-temp: most accurate

        # MDL (β=1) and optimal β
        d_mdl = self.computer.d_star(1.0)

        # Optimal β: minimize total cost E + 1/β · H(β)
        betas_sweep, d_stars_sweep = self.detector.sweep()
        F_totals = [
            self.computer.error(d) + self.computer.entropy(d) / max(b, 1e-6)
            for d, b in zip(d_stars_sweep, betas_sweep)
        ]
        opt_idx = int(np.argmin(F_totals))
        beta_opt = float(betas_sweep[opt_idx])

        # α from thermodynamics
        alpha_thermo = self.detector.alpha_from_free_energy(betas_sweep, d_stars_sweep)

        # α from direct spectral
        eig_d = self.eig.detach().double()
        eig_d = torch.abs(eig_d)
        eig_d, _ = torch.sort(eig_d, descending=True)
        if eig_d.numel() >= 2:
            j = torch.arange(1, eig_d.numel() + 1, dtype=torch.float64)
            alpha_spec = float(torch.max(-torch.log(eig_d + 1e-300) / torch.log(j + 1.0)).item())
        else:
            alpha_spec = 1.0

        first_transition = transitions[0] if transitions else None

        return ThermodynamicReport(
            alpha=alpha_spec,
            eigenvalues=eig_d,
            profiles=profiles,
            phase_transition=first_transition,
            d_star_zero_temp=d_cold,
            d_star_high_temp=d_hot,
            optimal_beta=beta_opt,
            mdl_dimension=d_mdl,
            alpha_from_thermodynamics=alpha_thermo,
        )

    def optimal_dimension(
        self,
        target_epsilon: float,
        beta: Optional[float] = None,
    ) -> Dict:
        """
        Find d*(ε, β) — the thermodynamically optimal truncation dimension.

        Unlike KoopmanDeltaBounds.optimal_dimension (which only minimizes error),
        this incorporates the entropy trade-off.

        Returns a dict with d*, achieved error, entropy at d*, and F(d*, β).
        """
        if beta is None:
            # Use the β where the error first drops below target
            # and free energy is minimized
            betas_sweep, d_stars = self.detector.sweep()
            for b, d in zip(betas_sweep, d_stars):
                e = self.computer.error(d)
                if e <= target_epsilon:
                    beta = b
                    break
            if beta is None:
                beta = float(betas_sweep[-1])

        prof = self.computer.profile(beta)
        # Find first d where error < target_epsilon
        d_opt = prof.d_star
        for d, e in zip(prof.dimensions, prof.errors):
            if e <= target_epsilon:
                d_opt = d
                break

        return {
            "d_star": d_opt,
            "beta_used": beta,
            "achieved_error": self.computer.error(d_opt),
            "entropy": self.computer.entropy(d_opt),
            "free_energy": self.computer.free_energy(d_opt, beta),
            "target_epsilon": target_epsilon,
            "is_feasible": self.computer.error(d_opt) <= target_epsilon * 1.01,
        }

    def summary(self, report: ThermodynamicReport) -> str:
        pt = report.phase_transition
        pt_str = (
            f"Phase transition: β_c = {pt.beta_c:.3f}, "
            f"d jumps {pt.d_low} → {pt.d_high} (Δd = {pt.delta_d})"
            if pt else "No phase transition detected in sweep range."
        )
        return (
            f"Thermodynamic ACF Report\n"
            f"{'='*50}\n"
            f"Índice Afín α(f) [spectral]  = {report.alpha:.4f}\n"
            f"Índice Afín α(f) [thermo]    = {report.alpha_from_thermodynamics:.4f}\n"
            f"d* at β→∞  (zero-temp, max accuracy)  = {report.d_star_zero_temp}\n"
            f"d* at β→0  (high-temp, max compression) = {report.d_star_high_temp}\n"
            f"d* at β=1  (MDL principle)             = {report.mdl_dimension}\n"
            f"Optimal β                              = {report.optimal_beta:.3f}\n"
            f"{pt_str}\n"
            f"\nFree energy profiles:\n"
            + "\n".join(
                f"  β={p.beta:6.2f}: d*={p.d_star:3d}, E(d*)={p.e_at_dstar:.3e}, "
                f"S(d*)={p.s_at_dstar:.2f}, F(d*)={p.f_star:.4f}"
                for p in report.profiles
            )
        )
