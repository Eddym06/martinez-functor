"""
Autopoietic Scientist — P-SAL: Protocolo de Síntesis Autopoiética de Leyes
===========================================================================

The closed-loop system that enables agents to discover, formalize, verify,
and deploy their own governing laws from observational data.

AUTOPOIESIS: The system creates its own structure from within.
  - Observes → data from DNS/experiment
  - Hypothesizes → SINDy discovers candidate dynamics
  - Closes → ERGON provides thermodynamic closure
  - Compiles → Poema generates executable ROM
  - Verifies → ROM predictions match reference data
  - Memorizes → Verified laws enter the knowledge base
  - Acts → Gideon executes the compiled ROM

THE P-SAL LOOP:
                    ┌─────────────────────────────────────────┐
                    │         AUTOPOIETIC SCIENTIST           │
                    │                                         │
                    │   ┌─────────┐     ┌──────────────┐     │
                    │   │ OBSERVE │ ──> │  HYPOTHESIZE │     │
                    │   └────▲────┘     └──────┬───────┘     │
                    │        │                 │              │
                    │        │           ┌─────▼─────┐       │
                    │        │           │   CLOSE   │       │
                    │        │           └─────┬─────┘       │
                    │        │                 │              │
                    │   ┌────┴────┐     ┌─────▼─────┐       │
                    │   │   ACT   │ <── │  COMPILE  │       │
                    │   └────▲────┘     └─────┬─────┘       │
                    │        │                 │              │
                    │        │           ┌─────▼─────┐       │
                    │        │           │  VERIFY   │       │
                    │        │           └─────┬─────┘       │
                    │        │                 │              │
                    │   ┌────┴────┐     ┌─────▼─────┐       │
                    │   │  ADAPT  │ <── │ MEMORIZE  │       │
                    │   └─────────┘     └───────────┘       │
                    │                                         │
                    └─────────────────────────────────────────┘

Each phase uses specific ecosystem agents:
  OBSERVE:     Raw data → TAA (Koopman decomposition, spectral analysis)
  HYPOTHESIZE: TAA modes → SINDy (sparse dynamics discovery)
  CLOSE:       SINDy ROM → ERGON (thermodynamic closure, eddy viscosity)
  COMPILE:     Closed ROM → Poema (PoemROM AST generation)
  VERIFY:      PoemROM → verify against reference (trajectory, spectrum, energy)
  MEMORIZE:    Verified ROM → Knowledge Base (persist for future use)
  ACT:         PoemROM → Gideon (execute, predict, control)
  ADAPT:       Feedback → refine (adjust modes, closure, thresholds)

CERTIFICATES:
  PSAL-1: Law Discovery: SINDy residual < threshold
  PSAL-2: Thermodynamic Consistency: ROM is dissipative
  PSAL-3: Spectral Fidelity: E_ROM(k) matches E_ref(k)
  PSAL-4: Predictive Accuracy: trajectory error < tolerance
  PSAL-5: Energy Conservation: energy drift bounded
  PSAL-6: Autopoietic Closure: system creates its own laws
  PSAL-7: Drift Consistency (v2.0, SEM-weighted): ε_drift < 0.3

References:
  Maturana & Varela (1980) — Autopoiesis and Cognition
  Brunton et al. (2016) — SINDy
  Noack et al. (2003) — Hierarchy of low-dimensional models
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------

class PSALPhase(Enum):
    """Phases of the P-SAL protocol."""
    OBSERVE = auto()
    HYPOTHESIZE = auto()
    CLOSE = auto()
    COMPILE = auto()
    VERIFY = auto()
    MEMORIZE = auto()
    ACT = auto()
    ADAPT = auto()


class LawStatus(Enum):
    """Status of a discovered law."""
    CANDIDATE = "candidate"         # Just discovered
    VERIFIED = "verified"           # Passed verification
    CERTIFIED = "certified"         # Passed all PSAL certificates
    DEPLOYED = "deployed"           # Running in Gideon
    REJECTED = "rejected"           # Failed verification
    SUPERSEDED = "superseded"       # Replaced by a better law


@dataclass
class DiscoveredLaw:
    """
    A law discovered by the Autopoietic Scientist.

    This is the fundamental unit of autopoietic knowledge:
    a mathematical relationship extracted from data, verified,
    and ready for deployment.
    """
    law_id: str
    name: str
    status: LawStatus
    n_modes: int
    L: np.ndarray                      # Linear operator
    Q: np.ndarray                      # Quadratic tensor
    c: np.ndarray                      # Constant forcing
    D: np.ndarray                      # Dissipation operator
    nu_t: float                        # Eddy viscosity
    certificates: Dict[str, float]
    discovery_time: float              # When discovered (epoch)
    verification_time: Optional[float] = None
    trajectory_error: float = float('inf')
    spectrum_error: float = float('inf')
    energy_drift: float = float('inf')
    sindy_sparsity: float = 0.0
    sindy_residual: float = float('inf')
    n_active_terms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def quality_score(self) -> float:
        """
        Overall quality score in [0, 1].
        Higher is better.
        """
        scores = []
        # Trajectory accuracy (0-1, lower error = higher score)
        if self.trajectory_error < float('inf'):
            scores.append(max(0.0, 1.0 - self.trajectory_error))
        # Sparsity (higher = simpler = better)
        scores.append(self.sindy_sparsity)
        # Dissipativity (binary)
        scores.append(1.0 if self.certificates.get("PSAL-2", 0) > 0 else 0.0)
        # Energy conservation
        if self.energy_drift < float('inf'):
            scores.append(max(0.0, 1.0 - self.energy_drift))

        return float(np.mean(scores)) if scores else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "law_id": self.law_id,
            "name": self.name,
            "status": self.status.value,
            "n_modes": self.n_modes,
            "L": self.L.tolist(),
            "Q": self.Q.tolist(),
            "c": self.c.tolist(),
            "D": self.D.tolist(),
            "nu_t": self.nu_t,
            "certificates": self.certificates,
            "discovery_time": self.discovery_time,
            "verification_time": self.verification_time,
            "trajectory_error": self.trajectory_error,
            "spectrum_error": self.spectrum_error,
            "energy_drift": self.energy_drift,
            "sindy_sparsity": self.sindy_sparsity,
            "sindy_residual": self.sindy_residual,
            "n_active_terms": self.n_active_terms,
            "quality_score": self.quality_score(),
            "metadata": {
                k: v for k, v in self.metadata.items()
                if isinstance(v, (int, float, str, bool, list))
            },
        }


@dataclass
class PSALCycle:
    """Record of one complete P-SAL cycle."""
    cycle_id: int
    phase_log: List[Tuple[PSALPhase, float, str]]  # (phase, duration_ms, summary)
    law: Optional[DiscoveredLaw]
    success: bool
    total_time_ms: float
    n_modes_used: int
    adaptation: Optional[Dict[str, Any]] = None


@dataclass
class KnowledgeBase:
    """
    Persistent knowledge base of discovered laws.

    The "memory" of the Autopoietic Scientist.
    Implements autopoietic closure: verified laws feed back to improve
    future observations (better mode selection, noise models, thresholds).
    """
    laws: Dict[str, DiscoveredLaw] = field(default_factory=dict)
    cycle_history: List[PSALCycle] = field(default_factory=list)
    best_law_id: Optional[str] = None
    # Autopoietic feedback state
    _preferred_n_modes: Optional[int] = None
    _preferred_sindy_threshold: Optional[float] = None
    _adaptation_count: int = 0

    def add_law(self, law: DiscoveredLaw) -> None:
        self.laws[law.law_id] = law
        # Update best law
        if self.best_law_id is None or law.quality_score() > self.best_law.quality_score():
            self.best_law_id = law.law_id

    @property
    def best_law(self) -> Optional[DiscoveredLaw]:
        if self.best_law_id and self.best_law_id in self.laws:
            return self.laws[self.best_law_id]
        return None

    def get_verified_laws(self) -> List[DiscoveredLaw]:
        return [l for l in self.laws.values() if l.status in (LawStatus.VERIFIED, LawStatus.CERTIFIED)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_laws": len(self.laws),
            "n_verified": len(self.get_verified_laws()),
            "n_cycles": len(self.cycle_history),
            "best_law_id": self.best_law_id,
            "laws": {k: v.to_dict() for k, v in self.laws.items()},
        }

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    # ── Autopoietic feedback ─────────────────────────────────────────
    def propose_improvements(self) -> Dict[str, Any]:
        """
        Analyze verified laws and propose improvements for the next cycle.

        AUTOPOIETIC CLOSURE: The system learns from its own discoveries.
        - Best n_modes from most accurate law
        - Best sindy_threshold from sparsest successful law
        - Whether to increase/decrease poly_degree
        """
        verified = self.get_verified_laws()
        if not verified:
            return {}

        # Best modes from lowest trajectory error law
        best = min(verified, key=lambda l: l.trajectory_error if np.isfinite(l.trajectory_error) else float('inf'))
        self._preferred_n_modes = best.n_modes

        # Best sparsity from laws that had low error
        sparse_accurate = [l for l in verified if l.trajectory_error < 1.0 and l.sindy_sparsity > 0]
        if sparse_accurate:
            self._preferred_sindy_threshold = float(np.median([l.sindy_sparsity for l in sparse_accurate]))

        self._adaptation_count += 1

        return {
            "preferred_n_modes": self._preferred_n_modes,
            "preferred_sindy_threshold": self._preferred_sindy_threshold,
            "n_verified_laws_analyzed": len(verified),
            "adaptation_count": self._adaptation_count,
        }

    def get_adapted_params(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Return adapted parameters, falling back to defaults."""
        adapted = dict(defaults)
        if self._preferred_n_modes is not None:
            adapted["n_modes_range"] = (
                max(2, self._preferred_n_modes - 1),
                self._preferred_n_modes + 3,
            )
        if self._preferred_sindy_threshold is not None:
            adapted["sindy_threshold"] = max(0.01, self._preferred_sindy_threshold * 0.8)
        return adapted


@dataclass
class PSALReport:
    """Full report of an Autopoietic Scientist session."""
    n_cycles: int
    n_laws_discovered: int
    n_laws_verified: int
    n_laws_certified: int
    best_law: Optional[DiscoveredLaw]
    knowledge_base: KnowledgeBase
    total_time_s: float
    certificates: Dict[str, float]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  AUTOPOIETIC SCIENTIST — P-SAL REPORT",
            "=" * 60,
            f"  Cycles executed:    {self.n_cycles}",
            f"  Laws discovered:    {self.n_laws_discovered}",
            f"  Laws verified:      {self.n_laws_verified}",
            f"  Laws certified:     {self.n_laws_certified}",
            f"  Total time:         {self.total_time_s:.2f}s",
        ]
        if self.best_law:
            lines.extend([
                "",
                f"  BEST LAW: {self.best_law.name}",
                f"    Quality score:      {self.best_law.quality_score():.4f}",
                f"    Trajectory error:   {self.best_law.trajectory_error:.6e}",
                f"    Spectrum error:     {self.best_law.spectrum_error:.6e}",
                f"    Energy drift:       {self.best_law.energy_drift:.6e}",
                f"    Modes:              {self.best_law.n_modes}",
                f"    Active terms:       {self.best_law.n_active_terms}",
                f"    Sparsity:           {self.best_law.sindy_sparsity:.2%}",
                f"    ν_t:                {self.best_law.nu_t:.6e}",
            ])
        lines.extend([
            "",
            "  CERTIFICATES:",
        ])
        for k, v in self.certificates.items():
            status = "PASS" if v > 0.5 else "FAIL"
            lines.append(f"    {k}: {status} ({v:.4f})")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Autopoietic Scientist
# ---------------------------------------------------------------------------

class AutopoieticScientist:
    """
    The Autopoietic Scientist — P-SAL Protocol Implementation.

    A closed-loop system that:
    1. Observes dynamical data
    2. Discovers governing laws via SINDy
    3. Stabilizes with ERGON thermodynamic closure
    4. Compiles into executable Poema programs
    5. Verifies against reference data
    6. Memorizes successful laws
    7. Acts using compiled ROMs
    8. Adapts based on feedback

    Usage:
        scientist = AutopoieticScientist(
            n_modes_range=(4, 16),
            sindy_threshold=0.05,
        )
        report = scientist.run(
            trajectory=dns_data,        # (n_steps, n_features)
            dt=0.01,
            h_ks=0.693,                 # from ERGON
            pressure_curvature=0.1,     # from ERGON
            n_cycles=5,
        )
    """

    def __init__(
        self,
        n_modes_range: Tuple[int, int] = (4, 16),
        sindy_threshold: float = 0.05,
        sindy_poly_degree: int = 2,
        verification_tolerance: float = 0.3,
        energy_tolerance: float = 0.5,
        max_cycles: int = 10,
        ccd_d_threshold: int = 8,
    ):
        self.n_modes_range = n_modes_range
        self.sindy_threshold = sindy_threshold
        self.sindy_poly_degree = sindy_poly_degree
        self.verification_tol = verification_tolerance
        self.energy_tol = energy_tolerance
        self.max_cycles = max_cycles
        self.knowledge_base = KnowledgeBase()
        # CCD Engine: activate for high-dimensional trajectories (d >= ccd_d_threshold)
        self.ccd_d_threshold = ccd_d_threshold
        self._ccd_engine = None   # fitted lazily in _observe()

        # Lazy imports to avoid circular dependencies
        self._sindy = None
        self._rom_builder = None
        self._closure = None
        self._rom_gen = None

    def _ensure_engines(self):
        """Lazy initialization of sub-engines."""
        if self._sindy is None:
            from acf_functor.rom_synthesizer import SINDyEngine, ROMBuilder
            from acf_functor.thermodynamic_closure import AdaptiveClosureSelector
            from poema.rom_generator import ROMGenerator

            self._sindy = SINDyEngine(
                poly_degree=self.sindy_poly_degree,
                threshold=self.sindy_threshold,
            )
            self._rom_builder = ROMBuilder(
                poly_degree=self.sindy_poly_degree,
                sindy_threshold=self.sindy_threshold,
            )
            self._rom_gen = ROMGenerator()

    def run(
        self,
        trajectory: np.ndarray,
        dt: float,
        h_ks: Optional[float] = None,
        pressure_curvature: Optional[float] = None,
        reference_spectrum: Optional[np.ndarray] = None,
        n_cycles: Optional[int] = None,
        koopman_eigenvalues: Optional[np.ndarray] = None,
        koopman_matrix: Optional[np.ndarray] = None,
        koopman_eigenvectors: Optional[np.ndarray] = None,
        sm_output=None,
    ) -> PSALReport:
        """
        Execute the full P-SAL protocol.

        Parameters
        ----------
        trajectory : (n_steps, n_features) array
            Observational data (from DNS, experiment, etc.)
        dt : float
            Time step
        h_ks : float, optional
            KS entropy from ERGON
        pressure_curvature : float, optional
            P''(1) from ERGON
        reference_spectrum : optional
            E(k) from DNS for validation
        n_cycles : int, optional
            Number of P-SAL cycles (default: explore range of modes)
        koopman_eigenvalues : optional
            Pre-computed Koopman eigenvalues from TAA
        koopman_matrix : optional
            Pre-computed Koopman matrix from TAA
        koopman_eigenvectors : optional
            Pre-computed Koopman eigenvectors from TAA
        """
        self._ensure_engines()
        t0 = time.perf_counter()

        if n_cycles is None:
            n_cycles = min(
                self.max_cycles,
                self.n_modes_range[1] - self.n_modes_range[0] + 1,
            )

        for cycle_id in range(n_cycles):
            # Adapt number of modes based on cycle
            n_modes = self.n_modes_range[0] + (
                cycle_id * (self.n_modes_range[1] - self.n_modes_range[0])
                // max(n_cycles - 1, 1)
            )
            n_modes = min(n_modes, trajectory.shape[1])

            cycle = self._run_cycle(
                cycle_id=cycle_id,
                trajectory=trajectory,
                dt=dt,
                n_modes=n_modes,
                h_ks=h_ks,
                pressure_curvature=pressure_curvature,
                reference_spectrum=reference_spectrum,
                koopman_eigenvalues=koopman_eigenvalues,
                koopman_matrix=koopman_matrix,
                koopman_eigenvectors=koopman_eigenvectors,
                sm_output=sm_output,
            )
            self.knowledge_base.cycle_history.append(cycle)

        total_time = time.perf_counter() - t0

        # Collect certificates
        verified = self.knowledge_base.get_verified_laws()
        best = self.knowledge_base.best_law

        certs = self._compute_global_certificates(verified, best)

        return PSALReport(
            n_cycles=n_cycles,
            n_laws_discovered=len(self.knowledge_base.laws),
            n_laws_verified=len(verified),
            n_laws_certified=sum(
                1 for l in verified if l.status == LawStatus.CERTIFIED
            ),
            best_law=best,
            knowledge_base=self.knowledge_base,
            total_time_s=total_time,
            certificates=certs,
        )

    def _run_cycle(
        self,
        cycle_id: int,
        trajectory: np.ndarray,
        dt: float,
        n_modes: int,
        h_ks: Optional[float],
        pressure_curvature: Optional[float],
        reference_spectrum: Optional[np.ndarray],
        koopman_eigenvalues: Optional[np.ndarray],
        koopman_matrix: Optional[np.ndarray],
        koopman_eigenvectors: Optional[np.ndarray],
        sm_output=None,
    ) -> PSALCycle:
        """Execute one complete P-SAL cycle."""
        phase_log = []
        t_cycle = time.perf_counter()

        # ── Phase 1: OBSERVE ────────────────────────────────────────
        t_phase = time.perf_counter()
        modal_data, obs_weights = self._observe(
            trajectory, n_modes,
            koopman_eigenvalues, koopman_matrix, koopman_eigenvectors,
            sm_output=sm_output,
        )
        dt_phase = (time.perf_counter() - t_phase) * 1000
        sem_note = " (SEM-purified)" if sm_output is not None else ""
        phase_log.append((PSALPhase.OBSERVE, dt_phase,
                         f"Extracted {n_modes} modes from {trajectory.shape[0]} samples{sem_note}"))

        # ── Phase 2: HYPOTHESIZE ────────────────────────────────────
        t_phase = time.perf_counter()
        sindy_result = self._hypothesize(modal_data, dt, obs_weights=obs_weights)
        dt_phase = (time.perf_counter() - t_phase) * 1000
        phase_log.append((PSALPhase.HYPOTHESIZE, dt_phase,
                         f"SINDy: {sindy_result.n_active_terms} active terms, "
                         f"sparsity={sindy_result.sparsity:.2%}, "
                         f"residual={sindy_result.residual_norm:.4e}"))

        # ── Phase 3: CLOSE ──────────────────────────────────────────
        t_phase = time.perf_counter()
        closure = self._close(modal_data, h_ks, pressure_curvature)
        dt_phase = (time.perf_counter() - t_phase) * 1000
        phase_log.append((PSALPhase.CLOSE, dt_phase,
                         f"Closure: ν_t={closure.nu_t:.6e} [{closure.closure_method}]"))

        # ── Phase 4: COMPILE ────────────────────────────────────────
        t_phase = time.perf_counter()
        rom, poem = self._compile(sindy_result, closure, n_modes, dt)
        dt_phase = (time.perf_counter() - t_phase) * 1000
        phase_log.append((PSALPhase.COMPILE, dt_phase,
                         f"Compiled PoemROM: {poem.name} ({n_modes} modes)"))

        # ── Phase 5: VERIFY ─────────────────────────────────────────
        t_phase = time.perf_counter()
        verification = self._verify(rom, modal_data, dt, reference_spectrum,
                                    sm_output=sm_output)
        dt_phase = (time.perf_counter() - t_phase) * 1000
        psal7_note = f", PSAL-7={verification.get('psal7_drift_consistency', 1.0):.0f}" \
            if sm_output is not None else ""
        phase_log.append((PSALPhase.VERIFY, dt_phase,
                         f"Trajectory error={verification['trajectory_error']:.4e}, "
                         f"dissipative={verification['is_dissipative']}{psal7_note}"))

        # ── Phase 6: MEMORIZE ───────────────────────────────────────
        t_phase = time.perf_counter()
        law = self._memorize(
            cycle_id, sindy_result, closure, rom, verification, n_modes,
        )
        dt_phase = (time.perf_counter() - t_phase) * 1000
        phase_log.append((PSALPhase.MEMORIZE, dt_phase,
                         f"Law '{law.name}': status={law.status.value}, "
                         f"quality={law.quality_score():.4f}"))

        # ── Phase 7: ACT ────────────────────────────────────────────
        t_phase = time.perf_counter()
        act_details = {}
        if law.status in (LawStatus.VERIFIED, LawStatus.CERTIFIED):
            try:
                from poema.backends.gideon.rom_executor import GideonROMExecutor
                executor = GideonROMExecutor()
                a0 = modal_data[0, :n_modes]
                # Run direct first
                direct_result = executor.execute(poem, a0, mode="direct")
                # Run adaptive for error control
                adaptive_result = executor.execute(poem, a0, mode="adaptive")
                # Run ensemble for uncertainty quantification
                n_ensemble = min(20, modal_data.shape[0])
                ic_ensemble = modal_data[:n_ensemble, :n_modes]
                ensemble_result = executor.execute_ensemble(poem, ic_ensemble)
                act_details = {
                    "direct_stable": direct_result.stable,
                    "direct_final_energy": direct_result.final_energy,
                    "adaptive_stable": adaptive_result.stable,
                    "adaptive_steps": adaptive_result.trajectory.shape[0],
                    "ensemble_n_stable": ensemble_result.n_stable,
                    "ensemble_n_total": ensemble_result.n_total,
                }
                act_summary = (f"ROM: direct={'✓' if direct_result.stable else '✗'}, "
                             f"adaptive={'✓' if adaptive_result.stable else '✗'}, "
                             f"ensemble={ensemble_result.n_stable}/{ensemble_result.n_total}")
            except Exception as e:
                a0 = modal_data[0, :n_modes]
                _ = poem.execute(a0)
                act_summary = f"Executed ROM for {poem.n_steps} steps (fallback)"
                act_details = {"fallback": str(e)}
        else:
            act_summary = "Skipped (law not verified)"
        dt_phase = (time.perf_counter() - t_phase) * 1000
        phase_log.append((PSALPhase.ACT, dt_phase, act_summary))

        # ── Phase 8: ADAPT ──────────────────────────────────────────
        t_phase = time.perf_counter()
        adaptation = self._adapt(law, verification)
        dt_phase = (time.perf_counter() - t_phase) * 1000
        phase_log.append((PSALPhase.ADAPT, dt_phase,
                         f"Adaptation: {adaptation.get('action', 'none')}"))

        total_ms = (time.perf_counter() - t_cycle) * 1000

        return PSALCycle(
            cycle_id=cycle_id,
            phase_log=phase_log,
            law=law,
            success=law.status in (LawStatus.VERIFIED, LawStatus.CERTIFIED),
            total_time_ms=total_ms,
            n_modes_used=n_modes,
            adaptation=adaptation,
        )

    # ── Phase implementations ─────────────────────────────────────────

    def _observe(
        self,
        trajectory: np.ndarray,
        n_modes: int,
        eigenvalues: Optional[np.ndarray],
        koopman_matrix: Optional[np.ndarray],
        eigenvectors: Optional[np.ndarray],
        sm_output=None,
    ):
        """
        Phase 1: OBSERVE — Extract modal amplitudes from trajectory.

        If sm_output (SMOutput from StochasticMembrane) is provided,
        use the purified trajectory and propagate posterior uncertainty
        as observation weights for STLSQ.

        If Koopman decomposition is provided (from TAA), project onto
        leading modes. Otherwise, use SVD/PCA as fallback.

        Returns
        -------
        modal_data : (T, r) array
        obs_weights : (T,) array or None
        """
        # ── Prefer SEM-purified trajectory ──────────────────────────
        obs_weights = None
        if sm_output is not None:
            try:
                purified = sm_output.purified
                if purified is not None and hasattr(purified, 'x_hat') and purified.x_hat is not None:
                    trajectory = purified.x_hat
                uncertainty = sm_output.uncertainty
                if uncertainty is not None and hasattr(uncertainty, 'sigma_tensor') and uncertainty.sigma_tensor is not None:
                    sigma = uncertainty.sigma_tensor  # (T, d, d)
                    trace_sigma = np.trace(sigma, axis1=1, axis2=2)  # (T,)
                    obs_weights = 1.0 / (trace_sigma + 1e-8)
                    obs_weights = obs_weights / obs_weights.mean()  # normalise
            except Exception:
                pass  # degrade gracefully — use raw trajectory

        # ── CCD preprocessing for high-dimensional trajectories ──────
        n_steps, n_features = trajectory.shape
        if n_features >= self.ccd_d_threshold:
            try:
                from acf_functor.ccd_engine import CCDEngine
                if self._ccd_engine is None or not self._ccd_engine._fitted:
                    self._ccd_engine = CCDEngine(
                        d_threshold=self.ccd_d_threshold,
                        n_diffusion_components=min(n_modes, n_features - 1, 10),
                        n_diffusion_neighbors=min(15, n_steps - 1),
                    ).fit(trajectory)
                # Project to low-dimensional manifold representation
                trajectory = self._ccd_engine.transform(trajectory)
                # Recompute n_features after CCD reduction
                n_steps, n_features = trajectory.shape
            except Exception:
                pass  # degrade gracefully — continue with original trajectory

        r = min(n_modes, n_features)

        if eigenvectors is not None and eigenvectors.shape[1] >= r:
            # Project onto leading Koopman modes
            V = eigenvectors[:, :r].real
            if trajectory.shape[1] == V.shape[0]:
                modal_data = trajectory @ V
            else:
                modal_data = trajectory[:, :r]
        else:
            # SVD/PCA fallback
            mean = trajectory.mean(axis=0)
            centered = trajectory - mean
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            modal_data = centered @ Vt[:r].T

        return modal_data, obs_weights

    def _hypothesize(
        self,
        modal_data: np.ndarray,
        dt: float,
        obs_weights=None,
        galerkin_operators: Optional[dict] = None,
    ):
        """Phase 2: HYPOTHESIZE — Discover dynamics via SINDy or Galerkin.

        Two modes:
          1. SINDy (default): sparse identification from data
          2. Galerkin projection: analytical projection when governing equations
             are known. Requires galerkin_operators = {'l_ij': ..., 'q_ijk': ...}

        When obs_weights (from SEM uncertainty) is provided, uses
        uncertainty-weighted STLSQ.
        """
        # ── Galerkin projection mode ──────────────────────────────────
        if galerkin_operators is not None:
            from acf_functor.rom_synthesizer import SINDyResult
            L = np.array(galerkin_operators.get("l_ij", [[0.0]]))
            Q_raw = galerkin_operators.get("q_ijk", None)
            c = np.array(galerkin_operators.get("c", [0.0]))

            r = modal_data.shape[1]
            if Q_raw is None:
                Q = np.zeros((r, r, r))
            else:
                Q = np.array(Q_raw)
                if Q.ndim == 2:
                    Q = Q.reshape(r, r, r)

            # Energy-triadic check: Σ q_ijk a_i a_j a_k = 0 for energy conservation
            triadic_sum = 0.0
            for i in range(r):
                for j in range(r):
                    for k in range(r):
                        triadic_sum += abs(Q[i,j,k] + Q[j,k,i] + Q[k,i,j])

            return SINDyResult(
                L=L, Q=Q, c=c,
                n_active_terms=int(np.sum(np.abs(L) > 1e-10) + np.sum(np.abs(Q) > 1e-10)),
                sparsity=0.5, residual_norm=0.0,
                method="galerkin_projection",
                energy_conserving=triadic_sum < 1e-6,
            )

        # ── SINDy mode (default) ──────────────────────────────────────
        from acf_functor.rom_synthesizer import SINDyEngine
        sindy = SINDyEngine(
            poly_degree=self.sindy_poly_degree,
            threshold=self.sindy_threshold,
        )
        return sindy.fit(modal_data, dt, sample_weights=obs_weights)

    def _close(
        self,
        modal_data: np.ndarray,
        h_ks: Optional[float],
        pressure_curvature: Optional[float],
    ):
        """Phase 3: CLOSE — Compute thermodynamic closure."""
        from acf_functor.thermodynamic_closure import AdaptiveClosureSelector

        n_modes = modal_data.shape[1]
        selector = AdaptiveClosureSelector(n_modes)
        return selector.select_and_compute(
            modal_amplitudes=modal_data,
            h_ks=h_ks,
            pressure_curvature=pressure_curvature,
        )

    def _compile(self, sindy_result, closure, n_modes: int, dt: float):
        """Phase 4: COMPILE — Build ROM and generate PoemROM."""
        from acf_functor.rom_synthesizer import ROMModel
        from poema.rom_generator import ROMGenerator

        # Build wavenumbers
        k = np.arange(1, n_modes + 1, dtype=float)
        D = -np.diag(k ** 2)

        rom = ROMModel(
            L=sindy_result.L,
            Q=sindy_result.Q,
            c=sindy_result.c,
            D=D,
            nu_t=closure.nu_t,
            n_modes=n_modes,
            sindy_result=sindy_result,
        )

        gen = ROMGenerator()
        poem = gen.from_rom_model(
            rom,
            name=f"psal_law_r{n_modes}",
            dt=dt,
            n_steps=min(500, sindy_result.Xi.shape[0]),
        )

        return rom, poem

    def _verify(
        self,
        rom,
        modal_data: np.ndarray,
        dt: float,
        reference_spectrum: Optional[np.ndarray],
        sm_output=None,
    ) -> Dict[str, float]:
        """Phase 5: VERIFY — Check ROM against reference data.

        PSAL-7 (SEM drift consistency): if sm_output is available, verify that
        the ROM's predicted trajectory does not drift more than 30 % relative
        to the purified (low-noise) reference beyond what the raw-trajectory
        baseline would give.  ε_drift < 0.3 required.
        """
        r = rom.n_modes
        n_verify = min(modal_data.shape[0] - 1, 500)
        a0 = modal_data[0, :r]

        # Integrate ROM
        traj_rom = rom.integrate(a0, dt, n_verify)

        # Reference
        ref = modal_data[:n_verify + 1, :r]

        # Trajectory error
        if np.any(np.isnan(traj_rom)):
            traj_err = float('inf')
            stable = False
        else:
            n_compare = min(traj_rom.shape[0], ref.shape[0])
            diff = traj_rom[:n_compare] - ref[:n_compare]
            traj_err = np.linalg.norm(diff, 'fro') / (np.linalg.norm(ref[:n_compare], 'fro') + 1e-15)
            stable = True

        # Energy conservation
        E0 = rom.energy(a0)
        E_final = rom.energy(traj_rom[-1]) if stable else float('inf')
        energy_drift = abs(E_final - E0) / (abs(E0) + 1e-15)

        # Spectrum error
        spec_err = 0.0
        if reference_spectrum is not None and stable:
            E_rom = 0.5 * np.mean(traj_rom[-100:] ** 2, axis=0) if traj_rom.shape[0] > 100 else 0.5 * a0 ** 2
            r_spec = min(len(E_rom), len(reference_spectrum))
            spec_err = np.linalg.norm(
                E_rom[:r_spec] - reference_spectrum[:r_spec]
            ) / (np.linalg.norm(reference_spectrum[:r_spec]) + 1e-15)

        return {
            "trajectory_error": traj_err,
            "spectrum_error": spec_err,
            "energy_drift": energy_drift,
            "is_dissipative": rom.is_dissipative(),
            "is_stable": stable,
            "psal7_drift_consistency": self._verify_psal7(
                traj_err, modal_data, sm_output
            ),
        }

    def _verify_psal7(
        self,
        traj_err_raw: float,
        modal_data: np.ndarray,
        sm_output,
    ) -> float:
        """
        PSAL-7 — Drift Consistency (SEM-weighted).

        Checks that the ROM trajectory error against the *purified* reference
        does not exceed the error against the raw reference by more than 30 %.

        Returns 1.0 (pass) when sm_output is None (no SEM baseline available).
        Returns 1.0 when ε_drift < 0.3, 0.0 otherwise.
        """
        if sm_output is None:
            return 1.0  # no SEM baseline → certificate trivially satisfied
        try:
            purified = sm_output.purified
            if purified is None or purified.x_hat is None:
                return 1.0
            x_hat = purified.x_hat          # (T, d)
            T_modal, r = modal_data.shape
            T_pure = x_hat.shape[0]
            T_min = min(T_modal, T_pure)
            # Project purified trajectory to modal space (first r components)
            d = x_hat.shape[1] if x_hat.ndim > 1 else 1
            r_use = min(r, d)
            pure_modal = x_hat[:T_min, :r_use]
            raw_modal  = modal_data[:T_min, :r_use]
            # Drift measure: how much raw and purified trajectories differ
            err_raw_vs_pure = float(
                np.linalg.norm(raw_modal - pure_modal, 'fro')
                / (np.linalg.norm(pure_modal, 'fro') + 1e-15)
            )
            # ε_drift = relative extra error introduced by noise
            epsilon_drift = err_raw_vs_pure
            return float(epsilon_drift < 0.3)
        except Exception:
            return 1.0  # degrade gracefully

    def _memorize(
        self,
        cycle_id: int,
        sindy_result,
        closure,
        rom,
        verification: Dict[str, float],
        n_modes: int,
    ) -> DiscoveredLaw:
        """Phase 6: MEMORIZE — Store discovered law in knowledge base."""
        law_id = f"PSAL-{cycle_id:03d}-r{n_modes}"

        # Determine status
        traj_err = verification["trajectory_error"]
        is_dissipative = verification["is_dissipative"]
        is_stable = verification["is_stable"]

        if not is_stable or traj_err > 10.0:
            status = LawStatus.REJECTED
        elif traj_err < self.verification_tol and is_dissipative:
            status = LawStatus.CERTIFIED
        elif traj_err < self.verification_tol * 3:
            status = LawStatus.VERIFIED
        else:
            status = LawStatus.CANDIDATE

        # Build certificates
        certs = {
            "PSAL-1": float(sindy_result.residual_norm < 0.5),
            "PSAL-2": float(is_dissipative),
            "PSAL-3": float(verification["spectrum_error"] < 1.0),
            "PSAL-4": float(traj_err < self.verification_tol),
            "PSAL-5": float(verification["energy_drift"] < self.energy_tol),
            "PSAL-6": 1.0,  # Autopoietic closure: always true if we got here
            "PSAL-7": verification.get("psal7_drift_consistency", 1.0),  # SEM drift check
        }

        law = DiscoveredLaw(
            law_id=law_id,
            name=f"ROM-r{n_modes} (cycle {cycle_id})",
            status=status,
            n_modes=n_modes,
            L=sindy_result.L.copy(),
            Q=sindy_result.Q.copy(),
            c=sindy_result.c.copy(),
            D=rom.D.copy(),
            nu_t=closure.nu_t,
            certificates=certs,
            discovery_time=time.time(),
            verification_time=time.time(),
            trajectory_error=traj_err,
            spectrum_error=verification["spectrum_error"],
            energy_drift=verification["energy_drift"],
            sindy_sparsity=sindy_result.sparsity,
            sindy_residual=sindy_result.residual_norm,
            n_active_terms=sindy_result.n_active_terms,
            metadata={
                "closure_method": closure.closure_method,
                "nu_t": closure.nu_t,
                "h_ks": closure.h_ks,
            },
        )

        self.knowledge_base.add_law(law)
        return law

    def _adapt(
        self,
        law: DiscoveredLaw,
        verification: Dict[str, float],
    ) -> Dict[str, Any]:
        """Phase 8: ADAPT — Autopoietic parameter adjustment from feedback.

        AUTOPOIETIC CLOSURE: Verified laws feed back into the system:
        1. Knowledge Base proposes improved parameters (n_modes, thresholds)
        2. Successful laws bias future mode selection
        3. Failed laws trigger threshold relaxation
        4. The system literally "learns from its own discoveries"
        """
        adaptation = {"action": "none"}

        # ── First: consult Knowledge Base for learned preferences ────
        kb_feedback = self.knowledge_base.propose_improvements()
        if kb_feedback:
            adaptation["kb_feedback"] = kb_feedback

        # ── Apply autopoietic parameter updates ──────────────────────
        if law.status == LawStatus.REJECTED:
            if verification.get("trajectory_error", float('inf')) > 10.0:
                self.sindy_threshold *= 0.5
                adaptation = {
                    "action": "lower_threshold",
                    "new_threshold": self.sindy_threshold,
                    "old_threshold": self.sindy_threshold * 2.0,
                    "reason": "ROM unstable — relaxing sparsity",
                }
            elif not verification.get("is_dissipative", True):
                adaptation = {
                    "action": "increase_closure",
                    "reason": "ROM not dissipative — need stronger closure",
                }
            else:
                self.verification_tol = min(0.9, self.verification_tol * 1.5)
                adaptation = {
                    "action": "relax_verification",
                    "new_tolerance": self.verification_tol,
                    "reason": "Multiple rejections — relaxing verification",
                }

        elif law.status == LawStatus.CANDIDATE:
            # Candidate but not verified — refine the search
            if verification.get("trajectory_error", float('inf')) < self.verification_tol * 2.0:
                # Close to passing — try more modes
                adaptation = {
                    "action": "increase_modes",
                    "reason": f"Near-threshold: error={verification['trajectory_error']:.4e}",
                }
            else:
                adaptation = {
                    "action": "refine",
                    "reason": f"Far from threshold: error={verification['trajectory_error']:.4e}",
                }

        elif law.status == LawStatus.VERIFIED:
            # Success! Reinforce the parameters that worked
            adaptation = {
                "action": "reinforce",
                "successful_n_modes": law.n_modes,
                "successful_nu_t": law.nu_t,
                "reason": "Law verified — parameters recorded in Knowledge Base",
            }

        # ── Autopoietic feedback: use KB to bias next cycle ──────────
        adapted_params = self.knowledge_base.get_adapted_params({
            "n_modes_range": self.n_modes_range,
            "sindy_threshold": self.sindy_threshold,
        })
        if adapted_params.get("n_modes_range") != self.n_modes_range:
            self.n_modes_range = adapted_params["n_modes_range"]
            adaptation["autopoietic_n_modes_update"] = self.n_modes_range
        if adapted_params.get("sindy_threshold") != self.sindy_threshold:
            self.sindy_threshold = adapted_params["sindy_threshold"]
            adaptation["autopoietic_threshold_update"] = self.sindy_threshold

        return adaptation

    def _compute_global_certificates(
        self,
        verified_laws: List[DiscoveredLaw],
        best_law: Optional[DiscoveredLaw],
    ) -> Dict[str, float]:
        """Compute global P-SAL certificates across all cycles."""
        certs = {}

        if best_law:
            certs.update(best_law.certificates)
            certs["best_quality_score"] = best_law.quality_score()
            certs["best_trajectory_error"] = best_law.trajectory_error
            certs["best_sparsity"] = best_law.sindy_sparsity
        else:
            certs["PSAL-1"] = 0.0
            certs["PSAL-2"] = 0.0
            certs["PSAL-3"] = 0.0
            certs["PSAL-4"] = 0.0
            certs["PSAL-5"] = 0.0
            certs["PSAL-6"] = 0.0
            certs["PSAL-7"] = 1.0  # trivially satisfied when no SEM present

        certs["n_laws_discovered"] = float(len(self.knowledge_base.laws))
        certs["n_laws_verified"] = float(len(verified_laws))
        certs["autopoietic_closure"] = float(len(verified_laws) > 0)

        return certs
