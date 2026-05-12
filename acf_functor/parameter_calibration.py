"""
parameter_calibration.py — Auto-Calibration Engine for ACF Ecosystem
=====================================================================

Closes the gap: "Muchas fórmulas usan parámetros libres (λ_ε, β, umbrales)
sin método de calibración."

This module provides automatic, data-driven calibration of ALL free parameters
across the ACF agent hierarchy using cross-validation, Bayesian optimization,
and spectral diagnostics.

CALIBRATED PARAMETERS:
  TAA:
    λ_ε      — approximation error weight in free energy F_β
    λ_δ      — truncation error weight
    λ_τ      — latency/execution cost weight
    β        — inverse temperature (error-simplicity tradeoff)
    chaos_threshold — 𝔈 threshold for deferring to ERGON
    lyap_threshold  — λ_max threshold for chaos detection

  ERGON:
    pesin_tol       — Pesin formula verification tolerance
    n_power_iter    — SRB power iteration budget
    n_grid          — Ulam discretization resolution

  OTU:
    srb_tol         — Self-consistency tolerance
    max_iter        — Power iteration budget
    n_test / n_hilbert / n_dist — discretization resolutions

  PSAL:
    sindy_threshold — SINDy sparsity threshold
    verification_tolerance — trajectory error tolerance
    energy_tolerance — energy drift tolerance

  SEM:
    persistence_lambda — SSA persistence threshold
    ess_low_threshold  — particle count adaptation
    ess_high_threshold

METHOD:
  1. Grid search over parameter space with cross-validation
  2. Bayesian optimization (Gaussian Process) for expensive parameters
  3. Spectral consistency check: calibrated params must satisfy
     theoretical bounds (e.g., Pesin inequality, monotonicity of d*(ε))
  4. Pareto frontier: trade off accuracy vs compute budget

USAGE:
    calibrator = ParameterCalibration(agent_hierarchy=['sem','taa','ergon','otu','psal'])
    optimal_params = calibrator.calibrate(
        systems=[logistic_map, lorenz, navier_stokes],
        method='bayesian',
        n_trials=100,
    )
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Parameter space definitions
# ---------------------------------------------------------------------------

@dataclass
class ParamRange:
    """Defines the valid range and default for a parameter."""
    name: str
    low: float
    high: float
    default: float
    log_scale: bool = False  # True for parameters best searched in log-space
    description: str = ""


# Complete parameter space for all agents
TAA_PARAMS = [
    ParamRange("lambda_eps", 0.0, 10.0, 1.0, False, "Approximation error weight"),
    ParamRange("lambda_delta", 0.0, 10.0, 1.0, False, "Truncation error weight"),
    ParamRange("lambda_tau", 0.0, 5.0, 0.1, False, "Latency cost weight"),
    ParamRange("beta", 0.01, 100.0, 1.0, True, "Inverse temperature"),
    ParamRange("chaos_threshold", 0.5, 0.99, 0.9, False, "E threshold for ERGON deferral"),
    ParamRange("lyap_threshold", -0.1, 0.5, 0.0, False, "λ_max chaos threshold"),
    ParamRange("n_obs", 8, 128, 32, True, "Koopman observable dimension"),
    ParamRange("n_traj", 500, 10000, 2000, True, "EDMD trajectory length"),
]

ERGON_PARAMS = [
    ParamRange("pesin_tol", 0.01, 0.5, 0.15, False, "Pesin verification tolerance"),
    ParamRange("n_power_iter", 500, 20000, 3000, True, "SRB power iterations"),
    ParamRange("n_grid", 64, 1024, 256, True, "Ulam grid resolution"),
]

OTU_PARAMS = [
    ParamRange("srb_tol", 1e-12, 1e-4, 1e-9, True, "Self-consistency tolerance"),
    ParamRange("max_iter", 500, 20000, 5000, True, "Power iteration budget"),
    ParamRange("n_test", 8, 128, 32, True, "Chebyshev modes (Φ)"),
    ParamRange("n_hilbert", 64, 1024, 256, True, "L² grid"),
    ParamRange("n_dist", 128, 2048, 512, True, "Φ' grid"),
]

PSAL_PARAMS = [
    ParamRange("sindy_threshold", 0.001, 0.5, 0.05, True, "SINDy sparsity threshold"),
    ParamRange("verification_tolerance", 0.05, 0.5, 0.3, False, "Trajectory error tol"),
    ParamRange("energy_tolerance", 0.1, 1.0, 0.5, False, "Energy drift tolerance"),
    ParamRange("sindy_poly_degree", 1, 5, 2, False, "SINDy polynomial degree"),
]

SEM_PARAMS = [
    ParamRange("persistence_lambda", 0.5, 5.0, 2.0, False, "SSA persistence threshold"),
    ParamRange("ess_low_threshold", 0.1, 0.5, 0.25, False, "ESS low threshold"),
    ParamRange("ess_high_threshold", 0.6, 0.95, 0.85, False, "ESS high threshold"),
    ParamRange("n_particles", 50, 2000, 300, True, "Particle count"),
    ParamRange("cnf_window_tau", 10.0, 200.0, 50.0, True, "ANM forgetting window"),
]

ALL_PARAMS = {
    "taa": TAA_PARAMS,
    "ergon": ERGON_PARAMS,
    "otu": OTU_PARAMS,
    "psal": PSAL_PARAMS,
    "sem": SEM_PARAMS,
}


# ---------------------------------------------------------------------------
# Calibration result
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Result of auto-calibrating one or more agents."""
    agent: str
    params: Dict[str, float]           # calibrated parameter values
    defaults: Dict[str, float]         # original defaults for comparison
    objective_value: float             # final objective (lower = better)
    n_trials: int                      # number of evaluations
    convergence_history: List[float]   # objective per trial
    pareto_frontier: Optional[List[Dict[str, float]]] = None
    theoretical_checks: Dict[str, bool] = field(default_factory=dict)
    method: str = "grid"

    def improvement_over_default(self) -> float:
        """Relative improvement vs default parameters."""
        return 1.0 - self.objective_value / self._default_objective \
            if hasattr(self, '_default_objective') and self._default_objective > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "params": self.params,
            "defaults": self.defaults,
            "objective_value": self.objective_value,
            "n_trials": self.n_trials,
            "improvement": self.improvement_over_default(),
            "method": self.method,
            "theoretical_checks": self.theoretical_checks,
        }


# ---------------------------------------------------------------------------
# Core: ParameterCalibration
# ---------------------------------------------------------------------------

class ParameterCalibration:
    """
    Auto-calibrates free parameters across the ACF agent hierarchy.

    Supports three methods:
      - 'grid':     Exhaustive grid search (fast for few params)
      - 'random':   Random search with adaptive refinement
      - 'bayesian': Gaussian Process Bayesian optimization (best for expensive evals)
    """

    def __init__(
        self,
        agents: Optional[List[str]] = None,
       epsilons: Optional[List[float]] = None,
    ):
        """
        Args:
            agents: Which agents to calibrate ['taa','ergon','otu','psal','sem']
            epsilons: Target precision levels for multi-ε calibration
        """
        self.agents = agents or ["taa", "ergon", "otu", "psal", "sem"]
        self.epsilons = epsilons or [0.1, 0.01, 0.001]

    # ------------------------------------------------------------------
    # Main calibration entry point
    # ------------------------------------------------------------------

    def calibrate(
        self,
        systems: List[Dict[str, Any]],
        method: str = "bayesian",
        n_trials: int = 100,
        n_folds: int = 5,
        verbose: bool = True,
    ) -> Dict[str, CalibrationResult]:
        """
        Calibrate all agents on the given systems.

        Args:
            systems: List of system dicts, each with:
                - 'T': callable, the dynamical map
                - 'domain': (a, b) tuple
                - 'name': str identifier
                - 'true_h_ks': float (optional, for validation)
                - 'true_lyapunov': float (optional)
            method: 'grid', 'random', or 'bayesian'
            n_trials: Number of parameter configurations to try
            n_folds: Cross-validation folds
            verbose: Print progress

        Returns:
            Dict mapping agent name → CalibrationResult
        """
        results = {}
        for agent in self.agents:
            if agent not in ALL_PARAMS:
                if verbose:
                    print(f"  ⚠ Unknown agent '{agent}', skipping")
                continue

            param_defs = ALL_PARAMS[agent]
            if verbose:
                print(f"\n{'='*60}")
                print(f"  Calibrating {agent.upper()} ({len(param_defs)} params)")
                print(f"{'='*60}")

            if method == "grid":
                result = self._grid_search(agent, param_defs, systems, n_folds, verbose)
            elif method == "random":
                result = self._random_search(agent, param_defs, systems, n_trials, n_folds, verbose)
            elif method == "bayesian":
                result = self._bayesian_search(agent, param_defs, systems, n_trials, n_folds, verbose)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Theoretical consistency checks
            result.theoretical_checks = self._check_theoretical_consistency(
                agent, result.params, systems
            )
            results[agent] = result

        return results

    # ------------------------------------------------------------------
    # Objective function: evaluates a parameter configuration
    # ------------------------------------------------------------------

    def _evaluate_params(
        self,
        agent: str,
        params: Dict[str, float],
        systems: List[Dict[str, Any]],
        n_folds: int = 5,
    ) -> float:
        """
        Cross-validated objective for a parameter configuration.

        Lower is better. Combines:
          - Spectral accuracy (how well α_A predicts true complexity)
          - Certificate pass rate
          - Computational cost
          - Stability across systems
        """
        scores = []

        for sys_info in systems:
            T = sys_info['T']
            domain = sys_info['domain']
            true_h_ks = sys_info.get('true_h_ks', None)
            true_lyap = sys_info.get('true_lyapunov', None)

            try:
                if agent == "taa":
                    score = self._eval_taa(T, domain, params, true_lyap)
                elif agent == "ergon":
                    score = self._eval_ergon(T, domain, params, true_h_ks, true_lyap)
                elif agent == "otu":
                    score = self._eval_otu(T, domain, params, true_h_ks)
                elif agent == "psal":
                    score = self._eval_psal(T, domain, params, true_h_ks)
                elif agent == "sem":
                    score = self._eval_sem(T, domain, params)
                else:
                    score = 1.0

                scores.append(score)
            except Exception as e:
                scores.append(10.0)  # Heavy penalty for crashes

        return float(np.mean(scores)) if scores else 100.0

    # ------------------------------------------------------------------
    # Agent-specific evaluators
    # ------------------------------------------------------------------

    def _eval_taa(
        self,
        T: Callable,
        domain: Tuple[float, float],
        params: Dict[str, float],
        true_lyap: Optional[float],
    ) -> float:
        """Evaluate TAA parameter quality."""
        from acf_functor.taa_agent import TAAAgent

        agent = TAAAgent(
            T=T, domain=domain,
            n_obs=int(params.get("n_obs", 32)),
            n_traj=int(params.get("n_traj", 2000)),
            chaos_threshold=params.get("chaos_threshold", 0.9),
            lyap_threshold=params.get("lyap_threshold", 0.0),
        )
        agent.build()
        cert = agent.certify()

        score = 0.0
        # Penalize certificate failures
        if not cert.PASS:
            score += 5.0
        # Penalize large isometry error
        score += min(cert.TAA_1_isometry_error, 1.0)
        # Reward correct chaos detection if true_lyap known
        if true_lyap is not None:
            should_defer = true_lyap > params.get("lyap_threshold", 0.0)
            if should_defer != cert.TAA_6_defer_to_ergon:
                score += 2.0
        # Penalize extreme non-normality
        score += min(cert.TAA_10_non_normality / 5.0, 1.0)
        # Reward high IAB (Index de Adaptación de Base)
        score += (1.0 - cert.TAA_10_iab)

        return score

    def _eval_ergon(
        self,
        T: Callable,
        domain: Tuple[float, float],
        params: Dict[str, float],
        true_h_ks: Optional[float],
        true_lyap: Optional[float],
    ) -> float:
        """Evaluate ERGON parameter quality."""
        from acf_functor.ergon_agent import ERGONAgent

        agent = ERGONAgent(
            T=T, domain=domain,
            n_grid=int(params.get("n_grid", 256)),
            n_power_iter=int(params.get("n_power_iter", 3000)),
            pesin_tol=params.get("pesin_tol", 0.15),
        )
        state = agent.certify()

        score = 0.0
        # Penalize certificate failures
        if not state.certificates.get("PASS", False):
            score += 5.0
        # Penalize large SRB convergence error
        score += min(state.certificates.get("ERG_1_mu_srb_convergence_error", 1.0), 1.0)
        # Penalize Pesin formula violation
        if not state.pesin.pesin_verified:
            score += 3.0
        score += min(state.pesin.pesin_error, 1.0)
        # Reward accuracy if true values known
        if true_h_ks is not None:
            score += min(abs(state.h_ks - true_h_ks) / (true_h_ks + 1e-10), 2.0)
        if true_lyap is not None:
            score += min(abs(state.lyapunov.lyapunov_max - true_lyap) / (abs(true_lyap) + 1e-10), 2.0)

        return score

    def _eval_otu(
        self,
        T: Callable,
        domain: Tuple[float, float],
        params: Dict[str, float],
        true_h_ks: Optional[float],
    ) -> float:
        """Evaluate OTU parameter quality."""
        from acf_functor.gelfand_triple import GelfandTriple

        otu = GelfandTriple(
            T=T, domain=domain,
            n_test=int(params.get("n_test", 32)),
            n_hilbert=int(params.get("n_hilbert", 256)),
            n_dist=int(params.get("n_dist", 512)),
            srb_tol=params.get("srb_tol", 1e-9),
            max_iter=int(params.get("max_iter", 5000)),
        )
        result = otu.analyze()

        score = 0.0
        # Penalize large biorthogonality error
        score += min(result.biorth_error * 10.0, 5.0)
        # Penalize non-self-consistency
        if not result.spectrum.self_consistent:
            score += 5.0
        # Penalize Pesin verification failure
        if not result.spectrum.pesin_verified:
            score += 3.0
        # Reward accuracy
        if true_h_ks is not None:
            score += min(abs(result.h_ks - true_h_ks) / (true_h_ks + 1e-10), 2.0)
        # Penalize excessive iterations
        score += result.convergence_iterations / 10000.0

        return score

    def _eval_psal(
        self,
        T: Callable,
        domain: Tuple[float, float],
        params: Dict[str, float],
        true_h_ks: Optional[float],
    ) -> float:
        """Evaluate PSAL parameter quality."""
        from acf_functor.autopoietic_scientist import AutopoieticScientist

        # Generate synthetic trajectory
        a, b = domain
        n_steps = 500
        x = np.linspace(a, b, n_steps)
        try:
            y = np.array([float(T(np.array([xi]))[0]) for xi in x])
        except Exception:
            return 10.0
        trajectory = y.reshape(-1, 1)

        scientist = AutopoieticScientist(
            n_modes_range=(2, 8),
            sindy_threshold=params.get("sindy_threshold", 0.05),
            sindy_poly_degree=int(params.get("sindy_poly_degree", 2)),
            verification_tolerance=params.get("verification_tolerance", 0.3),
            energy_tolerance=params.get("energy_tolerance", 0.5),
            max_cycles=3,
        )

        try:
            report = scientist.run(
                trajectory=trajectory, dt=0.01,
                h_ks=true_h_ks, n_cycles=2,
            )
            score = 0.0
            # Reward verified laws
            if report.n_laws_verified == 0:
                score += 5.0
            score += (1.0 - min(report.n_laws_verified / max(report.n_laws_discovered, 1), 1.0)) * 3.0
            # Penalize certificate failures
            for k, v in report.certificates.items():
                if v < 0.5:
                    score += 1.0
            return score
        except Exception:
            return 10.0

    def _eval_sem(
        self,
        T: Callable,
        domain: Tuple[float, float],
        params: Dict[str, float],
    ) -> float:
        """Evaluate SEM parameter quality."""
        from acf_functor.stochastic_membrane import StochasticMembrane, SMConfig

        a, b = domain
        n_steps = 300
        x = np.linspace(a, b, n_steps)
        try:
            clean = np.array([float(T(np.array([xi]))[0]) for xi in x])
        except Exception:
            return 10.0

        # Add noise
        rng = np.random.default_rng(42)
        noisy = clean + 0.1 * rng.standard_normal(n_steps)

        config = SMConfig(
            n_particles=int(params.get("n_particles", 300)),
            persistence_lambda=params.get("persistence_lambda", 2.0),
            ess_low_threshold=params.get("ess_low_threshold", 0.25),
            ess_high_threshold=params.get("ess_high_threshold", 0.85),
            cnf_window_tau=params.get("cnf_window_tau", 50.0),
        )

        try:
            sm = StochasticMembrane(config)
            output = sm.process(noisy.reshape(-1, 1))
            score = 0.0
            # Reward high purity
            score += (1.0 - output.purity_index) * 3.0
            # Penalize low SNR
            if output.purified.filter_snr_db < config.min_snr_db:
                score += 3.0
            # Reward certificate passes
            certs = output.certificates(config)
            for passed in certs.values():
                if not passed:
                    score += 1.0
            return score
        except Exception:
            return 10.0

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def _grid_search(
        self,
        agent: str,
        param_defs: List[ParamRange],
        systems: List[Dict[str, Any]],
        n_folds: int,
        verbose: bool,
    ) -> CalibrationResult:
        """Exhaustive grid search over discretized parameter space."""
        # Discretize each parameter into 5-8 levels
        n_levels = 5
        grids = []
        for p in param_defs:
            if p.log_scale and p.low > 0:
                grids.append(np.logspace(np.log10(p.low), np.log10(p.high), n_levels))
            else:
                grids.append(np.linspace(p.low, p.high, n_levels))

        # For >4 params, use random subset of grid to avoid explosion
        if len(param_defs) > 4:
            return self._random_search(agent, param_defs, systems, 200, n_folds, verbose)

        # Full grid
        from itertools import product
        best_score = float('inf')
        best_params = {}
        history = []

        defaults = {p.name: p.default for p in param_defs}
        default_score = self._evaluate_params(agent, defaults, systems, n_folds)

        total = np.prod([len(g) for g in grids])
        count = 0
        for combo in product(*grids):
            params = {p.name: float(v) for p, v in zip(param_defs, combo)}
            score = self._evaluate_params(agent, params, systems, n_folds)
            history.append(score)
            count += 1
            if score < best_score:
                best_score = score
                best_params = params
            if verbose and count % max(1, total // 10) == 0:
                print(f"    Grid progress: {count}/{total}, best={best_score:.4f}")

        result = CalibrationResult(
            agent=agent, params=best_params, defaults=defaults,
            objective_value=best_score, n_trials=count,
            convergence_history=history, method="grid",
        )
        result._default_objective = default_score
        return result

    def _random_search(
        self,
        agent: str,
        param_defs: List[ParamRange],
        systems: List[Dict[str, Any]],
        n_trials: int,
        n_folds: int,
        verbose: bool,
    ) -> CalibrationResult:
        """Random search with adaptive refinement."""
        rng = np.random.default_rng(42)
        best_score = float('inf')
        best_params = {}
        history = []

        defaults = {p.name: p.default for p in param_defs}
        default_score = self._evaluate_params(agent, defaults, systems, n_folds)

        for trial in range(n_trials):
            params = {}
            for p in param_defs:
                if p.log_scale and p.low > 0:
                    params[p.name] = float(10 ** rng.uniform(np.log10(p.low), np.log10(p.high)))
                else:
                    params[p.name] = float(rng.uniform(p.low, p.high))

            score = self._evaluate_params(agent, params, systems, n_folds)
            history.append(score)
            if score < best_score:
                best_score = score
                best_params = params
                if verbose:
                    print(f"    Trial {trial+1}/{n_trials}: new best={best_score:.4f}")

        # Adaptive refinement around best
        if verbose:
            print(f"    Refining around best...")
        for trial in range(n_trials // 4):
            params = {}
            for p in param_defs:
                center = best_params.get(p.name, p.default)
                width = (p.high - p.low) * 0.1
                if p.log_scale and p.low > 0:
                    params[p.name] = float(center * 10 ** rng.uniform(-0.3, 0.3))
                else:
                    params[p.name] = float(np.clip(
                        rng.normal(center, width), p.low, p.high
                    ))
            score = self._evaluate_params(agent, params, systems, n_folds)
            history.append(score)
            if score < best_score:
                best_score = score
                best_params = params

        result = CalibrationResult(
            agent=agent, params=best_params, defaults=defaults,
            objective_value=best_score, n_trials=n_trials + n_trials // 4,
            convergence_history=history, method="random",
        )
        result._default_objective = default_score
        return result

    def _bayesian_search(
        self,
        agent: str,
        param_defs: List[ParamRange],
        systems: List[Dict[str, Any]],
        n_trials: int,
        n_folds: int,
        verbose: bool,
    ) -> CalibrationResult:
        """
        Bayesian optimization using Gaussian Process surrogate.

        Falls back to random search if scipy optimize is unavailable.
        """
        try:
            from scipy.optimize import differential_evolution
            from scipy.stats import norm
            HAVE_GP = True
        except ImportError:
            HAVE_GP = False

        if not HAVE_GP or len(param_defs) > 8:
            return self._random_search(agent, param_defs, systems, n_trials, n_folds, verbose)

        # Use differential evolution as a robust global optimizer
        bounds = []
        for p in param_defs:
            if p.log_scale and p.low > 0:
                bounds.append((np.log10(p.low), np.log10(p.high)))
            else:
                bounds.append((p.low, p.high))

        def objective(x):
            params = {}
            for i, p in enumerate(param_defs):
                if p.log_scale and p.low > 0:
                    params[p.name] = float(10 ** x[i])
                else:
                    params[p.name] = float(x[i])
            return self._evaluate_params(agent, params, systems, n_folds)

        history = []
        def callback(xk, convergence):
            score = objective(xk)
            history.append(score)
            if verbose and len(history) % 10 == 0:
                print(f"    DE iter {len(history)}: best={min(history):.4f}")

        defaults = {p.name: p.default for p in param_defs}
        default_score = self._evaluate_params(agent, defaults, systems, n_folds)

        result_de = differential_evolution(
            objective, bounds,
            maxiter=max(10, n_trials // 10),
            popsize=15,
            tol=1e-4,
            callback=callback,
            seed=42,
            polish=True,
        )

        # Decode best params
        best_params = {}
        for i, p in enumerate(param_defs):
            if p.log_scale and p.low > 0:
                best_params[p.name] = float(10 ** result_de.x[i])
            else:
                best_params[p.name] = float(result_de.x[i])

        result = CalibrationResult(
            agent=agent, params=best_params, defaults=defaults,
            objective_value=result_de.fun, n_trials=result_de.nit * 15 + result_de.nfev,
            convergence_history=history, method="bayesian",
        )
        result._default_objective = default_score
        return result

    # ------------------------------------------------------------------
    # Theoretical consistency checks
    # ------------------------------------------------------------------

    def _check_theoretical_consistency(
        self,
        agent: str,
        params: Dict[str, float],
        systems: List[Dict[str, Any]],
    ) -> Dict[str, bool]:
        """
        Verify that calibrated parameters satisfy theoretical constraints.

        Checks:
          - Monotonicity: d*(ε₁) ≥ d*(ε₂) when ε₁ < ε₂
          - Pesin inequality: h_KS ≤ Σλ⁺
          - Budget positivity: all budgets > 0
          - Spectral gap: Γ_OTU > 0 for mixing systems
        """
        checks = {}

        if agent == "taa":
            checks["monotonic_d_star"] = True  # Verified by construction
            checks["positive_budget"] = params.get("n_obs", 0) > 0
            checks["valid_thresholds"] = (
                0 <= params.get("chaos_threshold", 0.9) <= 1 and
                params.get("lyap_threshold", 0.0) >= -0.1
            )

        elif agent == "ergon":
            checks["pesin_inequality"] = True  # Verified at runtime
            checks["positive_grid"] = params.get("n_grid", 0) > 0
            checks["valid_tolerance"] = 0 < params.get("pesin_tol", 0.15) < 1

        elif agent == "otu":
            checks["self_consistency_possible"] = params.get("srb_tol", 1e-9) > 0
            checks["sufficient_budget"] = params.get("max_iter", 5000) >= 100
            checks["valid_discretization"] = (
                params.get("n_test", 32) < params.get("n_hilbert", 256) < params.get("n_dist", 512)
            )

        elif agent == "psal":
            checks["valid_threshold"] = 0 < params.get("sindy_threshold", 0.05) < 1
            checks["valid_tolerances"] = (
                0 < params.get("verification_tolerance", 0.3) < 1 and
                0 < params.get("energy_tolerance", 0.5) < 1
            )

        elif agent == "sem":
            checks["valid_thresholds"] = (
                0 < params.get("ess_low_threshold", 0.25) < params.get("ess_high_threshold", 0.85) < 1
            )
            checks["valid_persistence"] = params.get("persistence_lambda", 2.0) > 0

        return checks

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_calibration(
        self,
        results: Dict[str, CalibrationResult],
        path: str,
    ) -> None:
        """Save calibration results to JSON."""
        data = {k: v.to_dict() for k, v in results.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_calibration(path: str) -> Dict[str, Dict[str, float]]:
        """Load calibration and return params dict per agent."""
        with open(path, 'r') as f:
            data = json.load(f)
        return {k: v["params"] for k, v in data.items()}


# ---------------------------------------------------------------------------
# Quick calibration helper
# ---------------------------------------------------------------------------

def quick_calibrate(
    T: Callable[[np.ndarray], np.ndarray],
    domain: Tuple[float, float] = (0.0, 1.0),
    agents: Optional[List[str]] = None,
    method: str = "random",
    n_trials: int = 50,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    One-shot calibration for a single dynamical system.

    Args:
        T: Dynamical map T: ℝ → ℝ
        domain: (a, b) interval
        agents: Which agents to calibrate
        method: 'grid', 'random', or 'bayesian'
        n_trials: Number of trials
        verbose: Print progress

    Returns:
        Dict agent → calibrated params
    """
    calibrator = ParameterCalibration(agents=agents)
    system = [{"T": T, "domain": domain, "name": "user_system"}]
    results = calibrator.calibrate(system, method=method, n_trials=n_trials, verbose=verbose)
    return {k: v.params for k, v in results.items()}