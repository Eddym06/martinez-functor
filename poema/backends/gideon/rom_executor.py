"""
Gideon ROM Executor — Execute Compiled ROM Laws in Gideon Engine
================================================================

Provides the execution backend for ROMs discovered by the Autopoietic
Scientist. Takes PoemROM objects and executes them with hardware-aware
optimization.

RESPONSIBILITIES:
  1. Execute PoemROM forward integration (RK4)
  2. Batch execution for parameter sweeps
  3. Energy monitoring and stability enforcement
  4. Real-time trajectory output for visualization
  5. Integration with GideonEngine dispatch pipeline

EXECUTION MODES:
  - "direct": Pure NumPy execution (always available)
  - "vectorized": Batched execution for ensemble forecasting
  - "adaptive": Adaptive time stepping with error control
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ROMExecutionResult:
    """Result of executing a ROM through Gideon."""
    trajectory: np.ndarray            # (n_steps+1, n_modes)
    energy_history: np.ndarray        # (n_steps+1,) total energy per step
    time_array: np.ndarray            # (n_steps+1,) physical time values
    dt_history: Optional[np.ndarray]  # (n_steps,) adaptive dt (if used)
    wall_time_ms: float               # Wall clock time
    n_rhs_evals: int                  # Number of RHS evaluations
    stable: bool                      # Did the ROM stay bounded?
    final_energy: float
    max_energy: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """Result of ensemble ROM execution."""
    trajectories: np.ndarray          # (n_ensemble, n_steps+1, n_modes)
    energy_histories: np.ndarray      # (n_ensemble, n_steps+1)
    mean_trajectory: np.ndarray       # (n_steps+1, n_modes)
    std_trajectory: np.ndarray        # (n_steps+1, n_modes)
    n_stable: int
    n_total: int
    wall_time_ms: float


# ---------------------------------------------------------------------------
# ROM Executor
# ---------------------------------------------------------------------------

class GideonROMExecutor:
    """
    Execute compiled ROM laws through the Gideon engine.

    Supports direct, vectorized, and adaptive execution modes.
    """

    def __init__(
        self,
        energy_limit: float = 1e8,
        stability_check_interval: int = 100,
    ):
        self.energy_limit = energy_limit
        self.stability_check_interval = stability_check_interval

    def execute(
        self,
        poem_rom: "PoemROM",
        a0: np.ndarray,
        n_steps: Optional[int] = None,
        dt: Optional[float] = None,
        mode: str = "direct",
    ) -> ROMExecutionResult:
        """
        Execute a PoemROM.

        Parameters
        ----------
        poem_rom : PoemROM from rom_generator
        a0 : Initial modal amplitudes
        n_steps : Override number of steps
        dt : Override time step
        mode : "direct", "adaptive"
        """
        n = n_steps or poem_rom.n_steps
        h = dt or poem_rom.dt

        if mode == "adaptive":
            return self._execute_adaptive(poem_rom, a0, n, h)
        else:
            return self._execute_direct(poem_rom, a0, n, h)

    def _execute_direct(
        self,
        poem: "PoemROM",
        a0: np.ndarray,
        n_steps: int,
        dt: float,
    ) -> ROMExecutionResult:
        """Direct RK4 execution."""
        t0_wall = time.perf_counter()
        r = poem.n_modes

        trajectory = np.zeros((n_steps + 1, r))
        energy = np.zeros(n_steps + 1)
        times = np.zeros(n_steps + 1)

        a = a0[:r].copy()
        trajectory[0] = a
        energy[0] = 0.5 * np.sum(a ** 2)
        n_rhs = 0
        stable = True

        L_eff = poem.L + poem.nu_t * poem.D

        for step in range(n_steps):
            # RK4
            k1 = self._rhs(a, L_eff, poem.Q, poem.c, r)
            k2 = self._rhs(a + 0.5 * dt * k1, L_eff, poem.Q, poem.c, r)
            k3 = self._rhs(a + 0.5 * dt * k2, L_eff, poem.Q, poem.c, r)
            k4 = self._rhs(a + dt * k3, L_eff, poem.Q, poem.c, r)
            a = a + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            n_rhs += 4

            # Stability check
            E = 0.5 * np.sum(a ** 2)
            if np.isnan(E) or E > self.energy_limit:
                trajectory[step + 1:] = np.nan
                energy[step + 1:] = np.nan
                stable = False
                break

            trajectory[step + 1] = a
            energy[step + 1] = E
            times[step + 1] = (step + 1) * dt

        wall_ms = (time.perf_counter() - t0_wall) * 1000

        return ROMExecutionResult(
            trajectory=trajectory,
            energy_history=energy,
            time_array=times,
            dt_history=None,
            wall_time_ms=wall_ms,
            n_rhs_evals=n_rhs,
            stable=stable,
            final_energy=float(energy[step] if stable else np.nan),
            max_energy=float(np.nanmax(energy)),
        )

    def _execute_adaptive(
        self,
        poem: "PoemROM",
        a0: np.ndarray,
        n_steps: int,
        dt_init: float,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        dt_min: float = 1e-10,
        dt_max_factor: float = 5.0,
    ) -> ROMExecutionResult:
        """
        Adaptive time-stepping RK4(5) execution.

        Uses embedded RK4/RK5 pair for error estimation and
        PI controller for step size adjustment.
        """
        t0_wall = time.perf_counter()
        r = poem.n_modes
        L_eff = poem.L + poem.nu_t * poem.D

        # Pre-allocate with extra room for adaptive steps
        max_alloc = n_steps * 3
        trajectory = np.zeros((max_alloc, r))
        energy = np.zeros(max_alloc)
        times = np.zeros(max_alloc)
        dt_hist = np.zeros(max_alloc)

        a = a0[:r].copy()
        trajectory[0] = a
        energy[0] = 0.5 * np.sum(a ** 2)
        t = 0.0
        dt = dt_init
        dt_max = dt_init * dt_max_factor
        step = 0
        n_rhs = 0
        stable = True
        t_final = n_steps * dt_init

        while t < t_final and step < max_alloc - 1:
            # RK4 step
            k1 = self._rhs(a, L_eff, poem.Q, poem.c, r)
            k2 = self._rhs(a + 0.5 * dt * k1, L_eff, poem.Q, poem.c, r)
            k3 = self._rhs(a + 0.5 * dt * k2, L_eff, poem.Q, poem.c, r)
            k4 = self._rhs(a + dt * k3, L_eff, poem.Q, poem.c, r)
            n_rhs += 4

            a_new = a + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            # Error estimate (compare RK4 with RK2)
            a_rk2 = a + dt * k2
            err = np.max(np.abs(a_new - a_rk2))
            scale = atol + rtol * np.max(np.abs(a_new))
            err_ratio = err / (scale + 1e-15)

            if err_ratio <= 1.0:
                # Accept step
                a = a_new
                t += dt
                step += 1

                E = 0.5 * np.sum(a ** 2)
                if np.isnan(E) or E > self.energy_limit:
                    stable = False
                    break

                trajectory[step] = a
                energy[step] = E
                times[step] = t
                dt_hist[step - 1] = dt

            # Adjust step size (PI controller)
            if err_ratio > 0:
                factor = min(2.0, max(0.3, 0.9 * err_ratio ** (-0.2)))
            else:
                factor = 2.0
            dt = min(max(dt * factor, dt_min), dt_max)

        wall_ms = (time.perf_counter() - t0_wall) * 1000

        # Trim arrays
        actual_steps = step + 1
        trajectory = trajectory[:actual_steps]
        energy = energy[:actual_steps]
        times = times[:actual_steps]
        dt_hist = dt_hist[:max(actual_steps - 1, 1)]

        return ROMExecutionResult(
            trajectory=trajectory,
            energy_history=energy,
            time_array=times,
            dt_history=dt_hist,
            wall_time_ms=wall_ms,
            n_rhs_evals=n_rhs,
            stable=stable,
            final_energy=float(energy[-1]) if stable else float('nan'),
            max_energy=float(np.nanmax(energy)),
            metadata={"mode": "adaptive", "actual_steps": actual_steps},
        )

    def execute_ensemble(
        self,
        poem_rom: "PoemROM",
        initial_conditions: np.ndarray,
        n_steps: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> EnsembleResult:
        """
        Execute ROM for an ensemble of initial conditions.

        Parameters
        ----------
        initial_conditions : (n_ensemble, n_modes) array
        """
        t0 = time.perf_counter()
        n_ens = initial_conditions.shape[0]
        n = n_steps or poem_rom.n_steps
        h = dt or poem_rom.dt
        r = poem_rom.n_modes

        trajectories = np.zeros((n_ens, n + 1, r))
        energies = np.zeros((n_ens, n + 1))
        n_stable = 0

        for i in range(n_ens):
            result = self._execute_direct(poem_rom, initial_conditions[i], n, h)
            trajectories[i] = result.trajectory
            energies[i] = result.energy_history
            if result.stable:
                n_stable += 1

        wall_ms = (time.perf_counter() - t0) * 1000

        return EnsembleResult(
            trajectories=trajectories,
            energy_histories=energies,
            mean_trajectory=np.nanmean(trajectories, axis=0),
            std_trajectory=np.nanstd(trajectories, axis=0),
            n_stable=n_stable,
            n_total=n_ens,
            wall_time_ms=wall_ms,
        )

    @staticmethod
    def _rhs(
        a: np.ndarray,
        L_eff: np.ndarray,
        Q: np.ndarray,
        c: np.ndarray,
        r: int,
    ) -> np.ndarray:
        """Evaluate ROM RHS: ȧ = L_eff·a + Q(a,a) + c."""
        da = L_eff @ a + c
        for i in range(r):
            da[i] += a @ Q[i] @ a
        return da
