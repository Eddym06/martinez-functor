"""
GideonAgentRouter — Agent Routing for the ACF Ecosystem
========================================================

Implements the Diagnostic Frontier described in ERGON_AGENT.md and Gideon.md §8.

Routes incoming dynamical systems to the correct agent:
  𝔈(T) ≈ 0, λ_max ≤ 0  →  TAA alone  (FMA exact reduction)
  0 < 𝔈(T) < 1           →  TAA + ERGON (joint, μ_SRB handoff)
  𝔈(T) ≈ 1, λ_max > 0   →  ERGON alone (Pesin certified, no FMA structure)

This is the MISSING PIECE identified in Gideon.md — the router that decides
which agent acts, rather than running both sequentially.

USAGE:
    router = GideonAgentRouter()
    decision = router.route(T=my_map, x_data=trajectory, domain=(0,1))

    if decision.use_taa:
        taa_result = TAAAgent(T, domain=domain).build(mu_srb=decision.mu_srb)
    if decision.use_ergon:
        ergon_result = ERGONAgent(T, domain=domain).certify()
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class AgentDecision(Enum):
    TAA_ONLY = "taa_only"         # 𝔈 ≈ 0, exact FMA reduction
    ERGON_ONLY = "ergon_only"     # 𝔈 ≈ 1, Pesin certified chaos
    TAA_ERGON_JOINT = "joint"     # 0 < 𝔈 < 1, both needed
    INSUFFICIENT_DATA = "insufficient"


@dataclass
class RouteResult:
    """Complete routing decision with diagnostics."""
    decision: AgentDecision
    use_taa: bool
    use_ergon: bool

    # Diagnostics
    lambda_max: float              # Leading Lyapunov exponent
    alpha_A: float                 # Spectral decay index (from TAA if computed)
    h_ks: float                    # KS entropy (from ERGON if computed)
    ergodic_complexity: float      # 𝔈(T) ∈ [0,1]

    # Measure handoff
    mu_srb: Optional[np.ndarray] = None   # SRB measure (from ERGON → TAA)
    mu_srb_source: str = "none"           # "empirical", "srb", or "none"

    # Certificates
    taa_cert: Optional[dict] = None
    ergon_cert: Optional[dict] = None
    pesin_verified: bool = False

    # Timing
    diagnosis_time_ms: float = 0.0
    routing_reason: str = ""

    def summary(self) -> str:
        lines = [
            f"GideonAgentRouter: {self.decision.value.upper()}",
            f"  λ_max={self.lambda_max:.4f}  α_A={self.alpha_A:.3f}  h_KS={self.h_ks:.4f}  𝔈={self.ergodic_complexity:.3f}",
            f"  μ_SRB source: {self.mu_srb_source}  Pesin: {'✅' if self.pesin_verified else '❌'}",
            f"  Reason: {self.routing_reason}",
        ]
        return "\n".join(lines)


@dataclass
class AgentRouterConfig:
    """Configuration for the agent router."""
    # Thresholds
    ergon_complexity_threshold: float = 0.1   # 𝔈 < this → TAA only
    chaos_lambda_threshold: float = 0.05      # λ_max > this → activate ERGON
    joint_complexity_low: float = 0.3         # 𝔈 between low/high → joint
    joint_complexity_high: float = 0.85       # 𝔈 > this → ERGON only

    # ERGON settings
    ergon_iterations: int = 3_000             # SRB power iterations
    ergon_grid: int = 256                     # Ulam discretization
    pesin_tolerance: float = 0.40             # Pesin verification tolerance

    # TAA settings
    taa_observables: int = 32                 # Koopman observable dimension
    taa_trajectory: int = 2_000               # EDMD trajectory length

    # Fast mode: skip expensive ERGON when λ_max ≤ 0
    fast_mode: bool = True


# ---------------------------------------------------------------------------
# Core: GideonAgentRouter
# ---------------------------------------------------------------------------

class GideonAgentRouter:
    """
    Routes dynamical systems to TAA, ERGON, or both based on structural diagnostics.

    This is the implementation of the "Diagnostic Frontier" (Frontera de
    Diagnóstico Estructural) from ERGON_AGENT.md and the missing router
    described in Gideon.md §8.

    Algorithm:
      1. Quick Lyapunov estimate (λ_max proxy)
      2. TAA spectral classification (α_A, decay class)
      3. If chaotic (λ_max > threshold):
         a. ERGON computes μ_SRB and h_KS
         b. Compute 𝔈(T) = h_KS / log(1 + Σλ⁺)
         c. Route based on 𝔈(T)
      4. If non-chaotic: TAA alone
      5. Handoff μ_SRB from ERGON → TAA when in joint mode
    """

    def __init__(self, config: Optional[AgentRouterConfig] = None):
        self.config = config or AgentRouterConfig()
        self._last_taa = None
        self._last_ergon = None

    # ------------------------------------------------------------------
    # Main routing entry point
    # ------------------------------------------------------------------

    def route(
        self,
        T: Callable[[np.ndarray], np.ndarray],
        domain: Tuple[float, float],
        x_data: Optional[np.ndarray] = None,
        force_ergon: bool = False,
        force_taa: bool = False,
    ) -> RouteResult:
        """
        Diagnose system and route to the correct agent(s).

        Args:
            T: Dynamical map T: ℝ → ℝ
            domain: (a, b) interval
            x_data: Optional trajectory data for improved diagnosis
            force_ergon: Force ERGON analysis regardless of diagnosis
            force_taa: Force TAA analysis regardless of diagnosis

        Returns:
            RouteResult with decision and diagnostics
        """
        t0 = time.perf_counter()

        if force_ergon and force_taa:
            return self._route_joint(T, domain, x_data, t0, reason="forced both")
        if force_ergon:
            return self._route_ergon_only(T, domain, x_data, t0, reason="forced ERGON")
        if force_taa:
            return self._route_taa_only(T, domain, x_data, t0, reason="forced TAA")

        # ── Step 1: Quick Lyapunov estimate ────────────────────────
        lambda_max = self._quick_lyapunov(T, domain)

        # ── Step 2: TAA spectral diagnosis ─────────────────────────
        taa_result = self._run_taa_diagnosis(T, domain, x_data)

        # ── Step 3: ERGON if chaotic ───────────────────────────────
        if lambda_max > self.config.chaos_lambda_threshold:
            return self._diagnose_and_route(
                T, domain, x_data, lambda_max, taa_result, t0
            )

        # ── Non-chaotic: TAA alone ─────────────────────────────────
        return self._route_taa_only(
            T, domain, x_data, t0,
            lambda_max=lambda_max,
            taa_result=taa_result,
            reason=f"λ_max={lambda_max:.4f} ≤ threshold={self.config.chaos_lambda_threshold}",
        )

    # ------------------------------------------------------------------
    # Quick Lyapunov estimator (fast, no ERGON needed)
    # ------------------------------------------------------------------

    def _quick_lyapunov(
        self,
        T: Callable[[np.ndarray], np.ndarray],
        domain: Tuple[float, float],
        n_orbit: int = 5_000,
        n_warmup: int = 200,
    ) -> float:
        """Fast λ_max estimate via finite-difference orbit."""
        eps = 1e-7
        a, b = domain
        rng = np.random.default_rng(42)
        x = float(a + (b - a) * (0.3 + 0.4 * rng.random()))

        for _ in range(n_warmup):
            try:
                xn = float(T(np.array([x]))[0])
                if a + 1e-10 < xn < b - 1e-10 and abs(xn - x) > 1e-10:
                    x = xn
                else:
                    x = float(a + (b - a) * rng.random())
            except Exception:
                x = float(a + (b - a) * rng.random())

        log_sum = 0.0
        count = 0
        for _ in range(n_orbit):
            try:
                xp = float(T(np.array([x + eps]))[0])
                xm = float(T(np.array([x - eps]))[0])
                d = abs(xp - xm) / (2.0 * eps)
                if d > 1e-12:
                    log_sum += np.log(d)
                    count += 1
                xn = float(T(np.array([x]))[0])
                if a + 1e-10 < xn < b - 1e-10:
                    x = xn
                else:
                    x = float(a + (b - a) * rng.random())
            except Exception:
                x = float(a + (b - a) * rng.random())

        return log_sum / count if count > 0 else 0.0

    # ------------------------------------------------------------------
    # TAA spectral diagnosis
    # ------------------------------------------------------------------

    def _run_taa_diagnosis(
        self,
        T: Callable[[np.ndarray], np.ndarray],
        domain: Tuple[float, float],
        x_data: Optional[np.ndarray],
    ) -> Optional[dict]:
        """Run lightweight TAA classification. Returns dict or None on failure."""
        try:
            from acf_functor.taa_agent import TAAAgent

            agent = TAAAgent(
                T=T, domain=domain,
                n_obs=self.config.taa_observables,
                n_traj=self.config.taa_trajectory,
            )
            agent.build()
            cert = agent.certify()
            self._last_taa = agent

            return {
                "decay_class": cert.TAA_4_decay_class,
                "alpha_A": cert.TAA_4_alpha_A,
                "d_star_eps01": cert.TAA_3_d_star_eps01,
                "isometry_error": cert.TAA_1_isometry_error,
                "non_normality": cert.TAA_10_non_normality,
                "pass": cert.PASS,
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # ERGON analysis (expensive — only run if chaotic)
    # ------------------------------------------------------------------

    def _run_ergon_analysis(
        self,
        T: Callable[[np.ndarray], np.ndarray],
        domain: Tuple[float, float],
    ) -> Optional[dict]:
        """Run full ERGON analysis. Returns dict or None on failure."""
        try:
            from acf_functor.ergon_agent import ERGONAgent

            agent = ERGONAgent(
                T=T, domain=domain,
                n_grid=self.config.ergon_grid,
                n_power_iter=self.config.ergon_iterations,
                pesin_tol=self.config.pesin_tolerance,
            )
            cert = agent.certify()
            self._last_ergon = agent

            return {
                "h_ks": cert.ERG_6a_h_ks,
                "lyapunov_max": cert.ERG_6a_h_lyapunov,
                "lyapunov_sum": cert.ERG_4_lyapunov_sum,
                "ergodic_complexity": cert.ERG_6b_ergodic_complexity,
                "pesin_verified": cert.ERG_5_pesin_verified,
                "pesin_error": cert.ERG_5_pesin_error,
                "mu_srb": agent._mu_srb,
                "duality_error": cert.ERG_2_duality_error,
                "birkhoff_error": cert.ERG_3_birkhoff_error,
                "pass": cert.PASS,
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def _diagnose_and_route(
        self,
        T: Callable,
        domain: Tuple[float, float],
        x_data: Optional[np.ndarray],
        lambda_max: float,
        taa_result: Optional[dict],
        t0: float,
    ) -> RouteResult:
        """Full diagnosis: run ERGON and route based on 𝔈(T)."""
        ergon_result = self._run_ergon_analysis(T, domain)

        if ergon_result is None:
            # ERGON failed → TAA only with warning
            return self._route_taa_only(
                T, domain, x_data, t0,
                lambda_max=lambda_max, taa_result=taa_result,
                reason="ERGON analysis failed, falling back to TAA",
            )

        ec = ergon_result["ergodic_complexity"]
        h_ks = ergon_result["h_ks"]
        mu_srb = ergon_result["mu_srb"]
        pesin_ok = ergon_result["pesin_verified"]

        # Route based on ergodic complexity
        if ec < self.config.ergon_complexity_threshold:
            # Low complexity → TAA can reduce, but use μ_SRB for accuracy
            reason = f"𝔈={ec:.3f} < {self.config.ergon_complexity_threshold} → TAA with SRB measure"
            taa_with_mu = self._route_taa_only(
                T, domain, x_data, t0,
                lambda_max=lambda_max, taa_result=taa_result,
                mu_srb=mu_srb, reason=reason,
            )
            return taa_with_mu

        elif ec < self.config.joint_complexity_low:
            # Very low — TAA alone suffices
            return self._route_taa_only(
                T, domain, x_data, t0,
                lambda_max=lambda_max, taa_result=taa_result,
                mu_srb=mu_srb,
                reason=f"𝔈={ec:.3f} < joint_low={self.config.joint_complexity_low}",
            )

        elif ec < self.config.joint_complexity_high:
            # Mixed regime — both needed
            return self._route_joint(
                T, domain, x_data, t0,
                lambda_max=lambda_max, taa_result=taa_result,
                ergon_result=ergon_result,
                reason=f"𝔈={ec:.3f} in [{self.config.joint_complexity_low}, {self.config.joint_complexity_high}] → joint",
            )

        else:
            # Fully chaotic — ERGON only
            return self._route_ergon_only(
                T, domain, x_data, t0,
                lambda_max=lambda_max, ergon_result=ergon_result,
                reason=f"𝔈={ec:.3f} > {self.config.joint_complexity_high} → ERGON only",
            )

    # ------------------------------------------------------------------
    # Route helpers
    # ------------------------------------------------------------------

    def _route_taa_only(
        self, T, domain, x_data, t0,
        lambda_max=0.0, taa_result=None, mu_srb=None,
        reason="",
    ) -> RouteResult:
        dt_ms = (time.perf_counter() - t0) * 1000
        result = RouteResult(
            decision=AgentDecision.TAA_ONLY,
            use_taa=True, use_ergon=False,
            lambda_max=lambda_max,
            alpha_A=taa_result["alpha_A"] if taa_result else 0.0,
            h_ks=0.0,
            ergodic_complexity=0.0,
            mu_srb=mu_srb,
            mu_srb_source="srb" if mu_srb is not None else "none",
            taa_cert=taa_result,
            diagnosis_time_ms=dt_ms,
            routing_reason=reason,
        )
        return result

    def _route_ergon_only(
        self, T, domain, x_data, t0,
        lambda_max=0.0, ergon_result=None,
        reason="",
    ) -> RouteResult:
        dt_ms = (time.perf_counter() - t0) * 1000
        result = RouteResult(
            decision=AgentDecision.ERGON_ONLY,
            use_taa=False, use_ergon=True,
            lambda_max=lambda_max if ergon_result is None else ergon_result.get("lyapunov_max", lambda_max),
            alpha_A=0.0,
            h_ks=ergon_result["h_ks"] if ergon_result else 0.0,
            ergodic_complexity=ergon_result["ergodic_complexity"] if ergon_result else 0.0,
            mu_srb=ergon_result["mu_srb"] if ergon_result else None,
            mu_srb_source="srb" if ergon_result else "none",
            ergon_cert=ergon_result,
            pesin_verified=ergon_result["pesin_verified"] if ergon_result else False,
            diagnosis_time_ms=dt_ms,
            routing_reason=reason,
        )
        return result

    def _route_joint(
        self, T, domain, x_data, t0,
        lambda_max=0.0, taa_result=None, ergon_result=None,
        reason="",
    ) -> RouteResult:
        dt_ms = (time.perf_counter() - t0) * 1000
        result = RouteResult(
            decision=AgentDecision.TAA_ERGON_JOINT,
            use_taa=True, use_ergon=True,
            lambda_max=lambda_max if ergon_result is None else ergon_result.get("lyapunov_max", lambda_max),
            alpha_A=taa_result["alpha_A"] if taa_result else 0.0,
            h_ks=ergon_result["h_ks"] if ergon_result else 0.0,
            ergodic_complexity=ergon_result["ergodic_complexity"] if ergon_result else 0.0,
            mu_srb=ergon_result["mu_srb"] if ergon_result else None,
            mu_srb_source="srb" if ergon_result else "none",
            taa_cert=taa_result,
            ergon_cert=ergon_result,
            pesin_verified=ergon_result["pesin_verified"] if ergon_result else False,
            diagnosis_time_ms=dt_ms,
            routing_reason=reason,
        )
        return result

    # ------------------------------------------------------------------
    # Convenience: route and execute
    # ------------------------------------------------------------------

    def route_and_execute(
        self,
        T: Callable[[np.ndarray], np.ndarray],
        domain: Tuple[float, float],
        x_data: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Route to correct agent(s) AND execute them.

        Returns dict with:
          - 'route': RouteResult
          - 'taa_result': TAACertificate or None
          - 'ergon_result': ERGONCertificate or None
        """
        route = self.route(T, domain, x_data)

        output = {"route": route, "taa_result": None, "ergon_result": None}

        if route.use_taa:
            taa = self._last_taa
            if taa is None:
                from acf_functor.taa_agent import TAAAgent
                taa = TAAAgent(T=T, domain=domain, n_obs=self.config.taa_observables)
                taa.build(mu_srb=route.mu_srb)
            elif route.mu_srb is not None and route.mu_srb_source == "srb":
                taa.build(mu_srb=route.mu_srb)
            output["taa_result"] = taa.certify() if taa._is_built else None

        if route.use_ergon:
            ergon = self._last_ergon
            if ergon is None:
                from acf_functor.ergon_agent import ERGONAgent
                ergon = ERGONAgent(T=T, domain=domain)
            output["ergon_result"] = ergon.certify()

        return output
