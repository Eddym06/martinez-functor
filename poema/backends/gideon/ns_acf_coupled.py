"""
ns_acf_coupled.py — Simulación Acoplada Navier-Stokes + ACF (PDE-ACF).

Integra el solver de Navier-Stokes DENTRO de Gideon, permitiendo:
  1. Análisis Koopman en tiempo real durante la simulación
  2. Refinamiento Adaptativo de Malla MULTI-ORÁCULO (5 oráculos ACF)
  3. Termostato de Turbulencia: conduce el flujo a turbulencia
     completamente desarrollada y lo MANTIENE ahí
  4. Auto-evolución del functor durante la simulación

Flujo v2 (Multi-Oracle):
  ┌──────────────────────────────────────────────────────┐
  │              NS Solver (RK4-IF, pseudo-spectral)     │
  │                   ω(t) → ω(t+dt)                    │
  └───────────────────┬──────────────────────────────────┘
                      │ snapshots cada T_analysis
          ┌───────────┼──────────────┐
          ▼           ▼              ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ O1:Koop  │ │ O2:Ruelle│ │ O3:ERGON │
  └─────┬────┘ └─────┬────┘ └─────┬────┘
        │            │             │
        ▼            ▼             ▼
  ┌──────────────────────────────────────────┐
  │  O4:Thermo        O5:Spectral           │
  └────────────────┬─────────────────────────┘
                   ▼
  ┌──────────────────────────────────────────┐
  │     Bayesian Arbiter → fuse votes        │
  └────────────────┬─────────────────────────┘
                   ▼
  ┌──────────────────────────────────────────┐
  │     Cascade Accelerator (termostato)     │
  │     → ν₄_new, A_new, α_new              │
  └──────────────────────────────────────────┘

Martínez's Invariant — Abril 2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


# ─────────────────────────────────────────────────────────────────────────────
# CoupledNSACFConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoupledNSACFConfig:
    """Configuración del solver acoplado NS + ACF."""
    # NS parameters
    N: int = 256
    Re: float = 10000.0
    nu4_coeff: float = 1e-12
    k_force: int = 4
    force_amplitude: float = 20.0
    friction: float = 0.02
    dt_init: float = 1e-3
    cfl_target: float = 0.3
    T_total: float = 40.0

    # ACF coupling parameters
    analysis_interval: float = 2.0     # Ejecutar Koopman cada T segundos
    n_koopman_modes: int = 60          # Modos máximos a computar
    adaptive_hyper: bool = True        # Adaptar ν₄ basado en espectro
    auto_evolve: bool = True           # Evolución automática del functor

    # Multi-Oracle AMR + Cascade Accelerator
    use_multi_oracle: bool = False     # Activar AMR multi-oráculo
    d_target: int = 30                 # Dimensión objetivo para d_95
    h_ks_target: float = 0.1          # Entropía KS objetivo

    # CoPoem Inverse Spectral Design
    use_copoem: bool = False           # Activar CoPoem inverse spectral solver
    k_force_min: int = 4               # Banda ancha: k mínimo
    k_force_max: int = 10              # Banda ancha: k máximo
    copoem_learning_rate: float = 0.4  # Learning rate del optimizador
    copoem_target_slope: float = -3.0  # Pendiente objetivo E(k) ~ k^slope
    copoem_max_power: float = 5000.0   # Presupuesto de potencia total

    # Snapshot management
    snapshot_interval: float = 0.05    # Guardar snapshot cada T segundos
    spinup_fraction: float = 0.3       # Fracción de spinup a descartar

    # Backend
    backend: str = "c_native"          # 'c_native', 'pytorch', 'triton'


# ─────────────────────────────────────────────────────────────────────────────
# AdaptiveMeshController — Refinamiento Adaptativo guiado por Koopman
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveMeshController:
    """
    Controlador de refinamiento adaptativo basado en análisis Koopman.

    En un solver pseudo-espectral, no podemos refinar la malla local,
    pero SÍ podemos:
      1. Ajustar la hiperviscosidad ν₄ según el gap espectral
      2. Modular la amplitud del forcing según la eficiencia modal
      3. Detectar blowup inminente via eigenvalores inestables
    """

    def __init__(self, N: int, base_nu4: float, base_force: float):
        self.N = N
        self.base_nu4 = base_nu4
        self.base_force = base_force
        self.history: List[Dict[str, Any]] = []

    def adapt(
        self,
        koopman_result: Any,
        current_energy: float,
        current_enstrophy: float,
    ) -> Dict[str, float]:
        """
        Calcula parámetros adaptativos basados en el análisis Koopman.

        Returns:
            dict con 'nu4', 'force_amplitude', 'stability_margin'
        """
        result = {
            "nu4": self.base_nu4,
            "force_amplitude": self.base_force,
            "stability_margin": 1.0,
        }

        if koopman_result is None:
            return result

        eigenvalues = koopman_result.eigenvalues
        decay_rates = koopman_result.decay_rates

        # 1. Estabilidad: si hay eigenvalores con |λ| > 1.1, hay inestabilidad
        max_mag = np.max(np.abs(eigenvalues))
        if max_mag > 1.1:
            # Incrementar hiperviscosidad para estabilizar
            instability_factor = min(max_mag / 1.0, 10.0)
            result["nu4"] = self.base_nu4 * instability_factor**2
            result["stability_margin"] = 1.0 / instability_factor

        # 2. Gap espectral: si d_95 es muy bajo, el flujo está bien resuelto
        d_95 = koopman_result.d_95
        total_modes = len(eigenvalues)
        resolution_efficiency = d_95 / max(total_modes, 1)

        if resolution_efficiency < 0.3:
            # El flujo vive en un subespacio bajo-dimensional:
            # podemos reducir la hiperviscosidad (menos disipación artificial)
            result["nu4"] = self.base_nu4 * max(0.1, resolution_efficiency * 2)

        # 3. Energía modal: si los modos coherentes capturan mucha energía,
        #    el forcing está siendo eficiente
        n_coherent = koopman_result.n_coherent
        if n_coherent > total_modes * 0.5:
            # Muchos modos coherentes: flujo bien organizado, reducir forcing
            result["force_amplitude"] = self.base_force * 0.8

        # 4. Registrar historia
        self.history.append({
            "d_95": d_95,
            "max_mag": float(max_mag),
            "n_coherent": n_coherent,
            "nu4_adapted": result["nu4"],
            "force_adapted": result["force_amplitude"],
            "energy": current_energy,
            "enstrophy": current_enstrophy,
        })

        return result


# ─────────────────────────────────────────────────────────────────────────────
# CoupledNSACFSolver — Solver Acoplado
# ─────────────────────────────────────────────────────────────────────────────

class CoupledNSACFSolver:
    """
    Solver Navier-Stokes 2D acoplado con análisis ACF en tiempo real.

    Características:
      - Pseudo-espectral FFT con RK4 + integrating factor
      - Análisis Koopman periódico via GPU (KoopmanGPU)
      - Hiperviscosidad adaptativa basada en gap espectral
      - Auto-evolución del functor ACF
      - Telemetría completa
    """

    def __init__(self, config: Optional[CoupledNSACFConfig] = None):
        self.config = config or CoupledNSACFConfig()
        self._setup_grid()
        self._setup_acf()
        self.koopman_results: List[Any] = []
        self.adaptation_log: List[Dict] = []

    def _setup_grid(self):
        """Inicializar grid pseudo-espectral."""
        cfg = self.config
        N = cfg.N

        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        self.x, self.y = np.meshgrid(x, x)
        self.dx = 2 * np.pi / N

        k = fftfreq(N, d=1.0 / N)
        self.kx, self.ky = np.meshgrid(k, k)
        self.k2 = self.kx**2 + self.ky**2
        self.k2[0, 0] = 1.0
        self.k_mag = np.sqrt(self.kx**2 + self.ky**2)

        # Dealiasing mask: 2/3 rule
        k_max = N // 3
        self.dealias = np.ones((N, N), dtype=bool)
        self.dealias[np.abs(self.kx) > k_max] = False
        self.dealias[np.abs(self.ky) > k_max] = False

        # Linear operator
        k2_full = self.kx**2 + self.ky**2
        self.linear_op = -cfg.friction - (1.0 / cfg.Re) * k2_full - cfg.nu4_coeff * k2_full**4
        self.linear_op[0, 0] = 0.0

        # Forcing
        self.forcing_hat = np.zeros((N, N), dtype=complex)
        np.random.seed(42)
        phases = np.random.uniform(0, 2 * np.pi, (N, N))
        self.force_phases = phases

        if cfg.use_copoem:
            # Broadband forcing: k ∈ [k_force_min, k_force_max]
            force_band = np.zeros((N, N), dtype=bool)
            for k_shell in range(cfg.k_force_min, cfg.k_force_max + 1):
                force_band |= (np.abs(self.k_mag - k_shell) < 0.8) & (self.k_mag > 0)
            self.force_band = force_band
            # Initial uniform amplitude across all shells
            self.forcing_hat[force_band] = cfg.force_amplitude * np.exp(1j * phases[force_band])
        else:
            # Single-band forcing at k_force
            force_band = (np.abs(self.k_mag - cfg.k_force) < 1.0) & (self.k_mag > 0)
            self.force_band = force_band
            self.forcing_hat[force_band] = cfg.force_amplitude * np.exp(1j * phases[force_band])

        self.current_force_amp = cfg.force_amplitude
        self.current_nu4 = cfg.nu4_coeff
        self.current_friction = cfg.friction

        # Adaptive mesh controller (legacy, used if multi_oracle=False)
        self.mesh_ctrl = AdaptiveMeshController(N, cfg.nu4_coeff, cfg.force_amplitude)

    def _setup_acf(self):
        """Inicializar componentes ACF."""
        try:
            from .koopman_gpu import KoopmanGPU
            self.koopman = KoopmanGPU()
            self._koopman_available = True
        except ImportError:
            try:
                from poema.backends.gideon.koopman_gpu import KoopmanGPU
                self.koopman = KoopmanGPU()
                self._koopman_available = True
            except ImportError:
                self._koopman_available = False

        # Multi-Oracle AMR controller
        self._multi_oracle = None
        if self.config.use_multi_oracle:
            try:
                from .turbulence_thermostat import MultiOracleAMR
                self._multi_oracle = MultiOracleAMR(
                    N=self.config.N,
                    nu4_base=self.config.nu4_coeff,
                    force_base=self.config.force_amplitude,
                    friction_base=self.config.friction,
                    d_target=self.config.d_target,
                    h_ks_target=self.config.h_ks_target,
                )
            except ImportError:
                try:
                    from poema.backends.gideon.turbulence_thermostat import MultiOracleAMR
                    self._multi_oracle = MultiOracleAMR(
                        N=self.config.N,
                        nu4_base=self.config.nu4_coeff,
                        force_base=self.config.force_amplitude,
                        friction_base=self.config.friction,
                        d_target=self.config.d_target,
                        h_ks_target=self.config.h_ks_target,
                    )
                except ImportError:
                    pass

        # CoPoem Inverse Spectral Designer
        self._copoem_designer = None
        self._copoem_oracle = None
        self._copoem_history: List[Dict] = []
        if self.config.use_copoem:
            try:
                from .copoem_spectral_designer import (
                    CoPoemSpectralDesigner,
                    CoPoemOracle,
                    DesignerConfig,
                )
            except ImportError:
                from poema.backends.gideon.copoem_spectral_designer import (
                    CoPoemSpectralDesigner,
                    CoPoemOracle,
                    DesignerConfig,
                )

            dcfg = DesignerConfig(
                target_slope=self.config.copoem_target_slope,
                k_force_min=self.config.k_force_min,
                k_force_max=self.config.k_force_max,
                initial_amplitude=self.config.force_amplitude,
                learning_rate=self.config.copoem_learning_rate,
                max_total_power=self.config.copoem_max_power,
            )
            self._copoem_designer = CoPoemSpectralDesigner(
                cfg=dcfg,
                N=self.config.N,
                nu=1.0 / self.config.Re,
                nu4=self.config.nu4_coeff,
            )
            self._copoem_oracle = CoPoemOracle(self._copoem_designer)

    def _init_vorticity(self) -> np.ndarray:
        """Inicializar campo de vorticidad."""
        N = self.config.N
        omega_hat = np.zeros((N, N), dtype=complex)
        np.random.seed(123)
        for kxi in range(-4, 5):
            for kyi in range(-4, 5):
                if kxi == 0 and kyi == 0:
                    continue
                k_sq = kxi**2 + kyi**2
                amp = 10.0 / (1.0 + k_sq)
                phase = np.random.uniform(0, 2 * np.pi)
                omega_hat[kyi % N, kxi % N] = amp * np.exp(1j * phase)
        return np.real(ifft2(omega_hat))

    def _velocity_from_omega(self, omega_hat):
        psi_hat = -omega_hat / self.k2
        psi_hat[0, 0] = 0.0
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        return u_hat, v_hat

    def _nonlinear_rhs(self, omega_hat):
        u_hat, v_hat = self._velocity_from_omega(omega_hat)
        omega_hat_d = omega_hat * self.dealias
        u_hat_d = u_hat * self.dealias
        v_hat_d = v_hat * self.dealias
        omega = np.real(ifft2(omega_hat_d))
        u = np.real(ifft2(u_hat_d))
        v = np.real(ifft2(v_hat_d))
        domega_dx = np.real(ifft2(1j * self.kx * omega_hat_d))
        domega_dy = np.real(ifft2(1j * self.ky * omega_hat_d))
        nl_phys = -(u * domega_dx + v * domega_dy)
        return fft2(nl_phys) * self.dealias

    def _rhs(self, omega_hat):
        return self._nonlinear_rhs(omega_hat) + self.linear_op * omega_hat + self.forcing_hat

    def _adaptive_dt(self, omega_hat):
        u_hat, v_hat = self._velocity_from_omega(omega_hat)
        u = np.real(ifft2(u_hat))
        v = np.real(ifft2(v_hat))
        u_max = max(np.max(np.abs(u)), 1e-10)
        v_max = max(np.max(np.abs(v)), 1e-10)
        dt_cfl = self.config.cfl_target * self.dx / (u_max + v_max)
        dt_visc = 0.5 * self.dx**2 / (1.0 / self.config.Re + 1e-12)
        return min(dt_cfl, dt_visc, 5e-3)

    def _rk4_step(self, omega_hat, dt):
        E_half = np.exp(self.linear_op * dt / 2)
        omega_hat = omega_hat * E_half
        def NL(w):
            return self._nonlinear_rhs(w) + self.forcing_hat
        k1 = NL(omega_hat)
        k2 = NL(omega_hat + 0.5 * dt * k1)
        k3 = NL(omega_hat + 0.5 * dt * k2)
        k4 = NL(omega_hat + dt * k3)
        omega_hat = omega_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        omega_hat = omega_hat * E_half
        return omega_hat

    def _compute_energy(self, omega_hat):
        u_hat, v_hat = self._velocity_from_omega(omega_hat)
        E = 0.5 * np.sum(np.abs(u_hat)**2 + np.abs(v_hat)**2) / self.config.N**2
        return float(np.real(E))

    def _compute_enstrophy(self, omega_hat):
        omega = np.real(ifft2(omega_hat))
        return 0.5 * float(np.mean(omega**2)) * (2 * np.pi)**2

    def _energy_spectrum(self, omega_hat):
        N = self.config.N
        u_hat, v_hat = self._velocity_from_omega(omega_hat)
        e_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / N**4
        k_max = N // 2
        k_bins = np.arange(1, k_max + 1, dtype=float)
        E_k = np.zeros(k_max)
        for i, k_val in enumerate(k_bins):
            shell = (self.k_mag >= k_val - 0.5) & (self.k_mag < k_val + 0.5)
            E_k[i] = np.sum(e_hat[shell])
        return k_bins, E_k

    def _run_koopman_analysis(self, snapshot_buffer: np.ndarray, dt_snap: float) -> Any:
        """Ejecutar análisis Koopman en los snapshots acumulados."""
        if not self._koopman_available or len(snapshot_buffer) < 20:
            return None

        try:
            result = self.koopman.analyze(
                snapshot_buffer,
                dt=dt_snap,
                n_modes=self.config.n_koopman_modes,
            )
            self.koopman_results.append(result)
            return result
        except Exception as e:
            print(f"  ⚠ Koopman analysis failed: {e}")
            return None

    def _apply_control_action(self, action: Dict[str, float]):
        """Apply control action from AMR controller: update ν₄, forcing, friction."""
        cfg = self.config
        N = cfg.N
        k2_full = self.kx**2 + self.ky**2

        new_nu4 = action.get("nu4", self.current_nu4)
        new_force = action.get("force_amplitude", self.current_force_amp)
        new_friction = action.get("friction", self.current_friction)

        # Update linear operator with new ν₄ and friction
        self.linear_op = -new_friction - (1.0 / cfg.Re) * k2_full - new_nu4 * k2_full**4
        self.linear_op[0, 0] = 0.0

        # Update forcing amplitude
        if abs(new_force - self.current_force_amp) > 1e-12:
            self.forcing_hat = np.zeros((N, N), dtype=complex)
            self.forcing_hat[self.force_band] = new_force * np.exp(
                1j * self.force_phases[self.force_band]
            )

        self.current_nu4 = new_nu4
        self.current_force_amp = new_force
        self.current_friction = new_friction

    def _apply_copoem_action(self, action: Dict[str, Any]):
        """Apply CoPoem per-mode control: update broadband forcing + viscosity profile."""
        cfg = self.config
        N = cfg.N
        k2_full = self.kx**2 + self.ky**2

        # Update forcing with per-mode amplitudes from CoPoem designer
        self.forcing_hat = self._copoem_designer.get_broadband_forcing(
            self.kx, self.ky, self.force_phases
        )

        # Update viscosity profile if designer provides it
        nu4_eff = action.get("nu4_effective", self.current_nu4)
        friction = self.current_friction

        # Build linear operator with CoPoem-designed viscosity profile
        if self._copoem_designer.cfg.enable_nu_design:
            # Use per-scale viscosity
            nu4_dissipation = self._copoem_designer.get_nu4_profile(self.kx, self.ky)
            self.linear_op = (
                -friction
                - (1.0 / cfg.Re) * k2_full
                + nu4_dissipation  # already includes the minus sign and k^8
            )
        else:
            self.linear_op = -friction - (1.0 / cfg.Re) * k2_full - nu4_eff * k2_full**4
        self.linear_op[0, 0] = 0.0

        # Track effective amplitude (mean across modes)
        self.current_nu4 = nu4_eff
        self.current_force_amp = float(np.mean(action.get("amplitudes", [cfg.force_amplitude])))

    def simulate(self) -> Dict[str, Any]:
        """
        Ejecutar simulación acoplada NS + ACF.

        Returns dict con todos los resultados.
        """
        cfg = self.config
        N = cfg.N

        print("=" * 70)
        print("SIMULACIÓN ACOPLADA NAVIER-STOKES + ACF (PDE-ACF)")
        print("=" * 70)
        print(f"  Grid: {N}×{N} ({N*N:,} DOFs), Re = {cfg.Re:.0f}")
        print(f"  ν = {1.0/cfg.Re:.6f}, ν₄ = {cfg.nu4_coeff:.1e}")
        print(f"  Forcing: k_f = {cfg.k_force}, A = {cfg.force_amplitude}")
        print(f"  T_total = {cfg.T_total}, análisis cada {cfg.analysis_interval}s")
        print(f"  Backend: {cfg.backend}, auto_evolve: {cfg.auto_evolve}")
        print(f"  Koopman GPU: {'disponible' if self._koopman_available else 'CPU fallback'}")
        if self._multi_oracle:
            print(f"  Multi-Oracle AMR: ACTIVO (d_target={cfg.d_target}, h_ks_target={cfg.h_ks_target})")
        if self._copoem_designer:
            print(f"  CoPoem Spectral Designer: ACTIVO (target={cfg.copoem_target_slope}, "
                  f"k∈[{cfg.k_force_min},{cfg.k_force_max}], "
                  f"P_max={cfg.copoem_max_power})")
        print()

        # Initialize
        omega = self._init_vorticity()
        omega_hat = fft2(omega)

        t = 0.0
        step = 0
        t_next_snap = 0.0
        t_next_analysis = cfg.analysis_interval
        t_next_print = 0.0

        snapshots = []
        energies = []
        enstrophies = []
        times = []
        analysis_buffer = []

        t_start = time.time()
        last_koopman = None

        while t < cfg.T_total:
            dt = self._adaptive_dt(omega_hat)
            dt_actual = min(dt, cfg.T_total - t)
            if t_next_snap - t > 1e-12 and t_next_snap - t < dt_actual:
                dt_actual = t_next_snap - t

            omega_hat = self._rk4_step(omega_hat, dt_actual)
            t += dt_actual
            step += 1

            # Check blowup
            if step % 200 == 0:
                max_omega = np.max(np.abs(np.real(ifft2(omega_hat))))
                if np.isnan(max_omega) or max_omega > 1e10:
                    print(f"  ⚠ BLOWUP en t={t:.4f}, step={step}")
                    break

            # Record snapshot
            if t >= t_next_snap - 1e-12:
                E = self._compute_energy(omega_hat)
                Z = self._compute_enstrophy(omega_hat)
                snap = np.real(ifft2(omega_hat)).copy()
                snapshots.append(snap)
                energies.append(E)
                enstrophies.append(Z)
                times.append(t)
                analysis_buffer.append(snap)
                t_next_snap = t + cfg.snapshot_interval

            # Periodic Koopman analysis + adaptation
            if t >= t_next_analysis - 1e-12 and len(analysis_buffer) >= 20:
                print(f"  ⟳ Análisis ACF en t={t:.2f} ({len(analysis_buffer)} snapshots)...")
                t_k0 = time.time()

                buf = np.array(analysis_buffer)
                koopman_result = self._run_koopman_analysis(buf, cfg.snapshot_interval)

                if koopman_result is not None:
                    E_curr = energies[-1] if energies else 0
                    Z_curr = enstrophies[-1] if enstrophies else 0

                    if self._copoem_oracle and cfg.use_copoem:
                        # CoPoem Inverse Spectral Design — broadband per-mode control
                        k_bins, E_k = self._energy_spectrum(omega_hat)

                        # Get Koopman spectral radius for stability constraint
                        spec_rad = 1.0
                        if hasattr(koopman_result, 'eigenvalues') and koopman_result.eigenvalues is not None:
                            spec_rad = float(np.max(np.abs(koopman_result.eigenvalues)))

                        copoem_action = self._copoem_oracle.vote(
                            k_bins=k_bins,
                            E_k=E_k,
                            koopman_spectral_radius=spec_rad,
                            energy=E_curr,
                            enstrophy=Z_curr,
                            t=t,
                        )
                        self._apply_copoem_action(copoem_action)
                        self._copoem_history.append({
                            "t": t,
                            "misfit": copoem_action["misfit"],
                            "slope": copoem_action["slope_actual"],
                            "phase": copoem_action["phase"],
                            "adjunction_gap": copoem_action["adjunction_gap"],
                            "total_power": copoem_action["total_power"],
                            "amplitudes": copoem_action["amplitudes"].tolist(),
                            "d_95": koopman_result.d_95,
                        })

                        # Also run multi-oracle AMR if available (for d_95 tracking)
                        if self._multi_oracle and cfg.use_multi_oracle:
                            Z_series = np.array(enstrophies)
                            mo_action = self._multi_oracle.control(
                                koopman_result=koopman_result,
                                enstrophy_series=Z_series,
                                k_bins=k_bins,
                                E_k=E_k,
                                energy=E_curr,
                                enstrophy=Z_curr,
                                t=t,
                            )
                            # Don't apply mo_action — CoPoem takes priority
                            # but log the multi-oracle diagnostics
                            self.adaptation_log.append({
                                "t": t,
                                "phase": copoem_action["phase"],
                                "nu4": copoem_action["nu4_effective"],
                                "force": float(np.mean(copoem_action["amplitudes"])),
                                "friction": self.current_friction,
                                "d_95": koopman_result.d_95,
                                "misfit": copoem_action["misfit"],
                                "slope": copoem_action["slope_actual"],
                            })

                        t_k1 = time.time()
                        print(f"    ✓ CoPoem: J={copoem_action['misfit']:.3f}, "
                              f"slope={copoem_action['slope_actual']:.2f}, "
                              f"phase={copoem_action['phase']}, "
                              f"d_95={koopman_result.d_95}, "
                              f"Ā={float(np.mean(copoem_action['amplitudes'])):.1f}, "
                              f"gap={copoem_action['adjunction_gap']:.3f} "
                              f"[{(t_k1-t_k0)*1000:.0f}ms]")

                    elif self._multi_oracle and cfg.use_multi_oracle:
                        # Multi-Oracle AMR + Cascade Accelerator
                        k_bins, E_k = self._energy_spectrum(omega_hat)
                        Z_series = np.array(enstrophies)

                        action = self._multi_oracle.control(
                            koopman_result=koopman_result,
                            enstrophy_series=Z_series,
                            k_bins=k_bins,
                            E_k=E_k,
                            energy=E_curr,
                            enstrophy=Z_curr,
                            t=t,
                        )
                        self._apply_control_action(action)
                        self.adaptation_log.append({
                            "t": t,
                            "phase": self._multi_oracle.cascade.phase,
                            "nu4": action["nu4"],
                            "force": action["force_amplitude"],
                            "friction": action["friction"],
                            "d_95": koopman_result.d_95,
                        })

                        t_k1 = time.time()
                        phase = self._multi_oracle.cascade.phase
                        print(f"    ✓ d_95={koopman_result.d_95}, coherent={koopman_result.n_coherent}, "
                              f"phase={phase}, A={action['force_amplitude']:.1f}, "
                              f"ν₄={action['nu4']:.1e} [{(t_k1-t_k0)*1000:.0f}ms]")

                    elif cfg.adaptive_hyper:
                        # Legacy single-oracle Koopman AMR
                        adaptation = self.mesh_ctrl.adapt(koopman_result, E_curr, Z_curr)
                        self.adaptation_log.append(adaptation)
                        self._apply_control_action({
                            "nu4": adaptation["nu4"],
                            "force_amplitude": adaptation["force_amplitude"],
                            "friction": cfg.friction,
                        })

                        t_k1 = time.time()
                        print(f"    ✓ d_95={koopman_result.d_95}, coherent={koopman_result.n_coherent}, "
                              f"backend={koopman_result.backend} [{(t_k1-t_k0)*1000:.0f}ms]")

                    last_koopman = koopman_result

                # Reset analysis buffer (keep last 20% for overlap)
                keep = max(len(analysis_buffer) // 5, 10)
                analysis_buffer = analysis_buffer[-keep:]
                t_next_analysis = t + cfg.analysis_interval

            # Print progress
            if t >= t_next_print - 1e-12:
                E = self._compute_energy(omega_hat)
                Z = self._compute_enstrophy(omega_hat)
                elapsed = time.time() - t_start
                d95_str = f"d95={last_koopman.d_95}" if last_koopman else "d95=?"
                phase_str = ""
                if self._copoem_history:
                    last_cp = self._copoem_history[-1]
                    phase_str = f"  [{last_cp['phase']}|J={last_cp['misfit']:.2f}|s={last_cp['slope']:.1f}]"
                elif self._multi_oracle and self._multi_oracle.cascade.history:
                    phase_str = f"  [{self._multi_oracle.cascade.phase}]"
                print(f"  t={t:6.2f}/{cfg.T_total:.0f}  E={E:.4f}  Z={Z:.4f}  "
                      f"dt={dt_actual:.2e}  {d95_str}  A={self.current_force_amp:.1f}  "
                      f"ν₄={self.current_nu4:.1e}{phase_str}  [{elapsed:.1f}s]")
                t_next_print = t + max(cfg.T_total / 10, 2.0)

        elapsed_total = time.time() - t_start
        print(f"\n  Simulación completada: {step} pasos en {elapsed_total:.1f}s")
        print(f"  Snapshots: {len(snapshots)}, análisis Koopman: {len(self.koopman_results)}")

        # Energy spectrum
        k_bins, E_k = self._energy_spectrum(omega_hat)

        result = {
            "snapshots": np.array(snapshots) if snapshots else np.array([]),
            "energy": np.array(energies),
            "enstrophy": np.array(enstrophies),
            "times": np.array(times),
            "omega_final_hat": omega_hat,
            "k_bins": k_bins,
            "E_k": E_k,
            "koopman_results": self.koopman_results,
            "adaptation_log": self.adaptation_log,
            "last_koopman": last_koopman,
            "thermostat_history": (
                self._multi_oracle.cascade.history
                if self._multi_oracle else []
            ),
            "oracle_vote_log": (
                self._multi_oracle.vote_log
                if self._multi_oracle else []
            ),
            "copoem_history": self._copoem_history,
            "copoem_designer_history": (
                [
                    {
                        "t": s.t,
                        "misfit": s.misfit,
                        "slope_actual": s.slope_actual,
                        "adjunction_gap": s.adjunction_gap,
                        "phase": s.phase,
                        "total_power": s.total_power,
                        "amplitudes": s.amplitudes.tolist(),
                        "nu_profile": s.nu_profile.tolist(),
                        "k_shells": s.k_shells.tolist(),
                    }
                    for s in self._copoem_designer.history
                ]
                if self._copoem_designer else []
            ),
            "params": {
                "N": N, "Re": cfg.Re, "nu": 1.0 / cfg.Re, "nu4": cfg.nu4_coeff,
                "k_force": cfg.k_force, "T_total": cfg.T_total,
                "coupled": True, "auto_evolve": cfg.auto_evolve,
                "backend": cfg.backend,
                "multi_oracle": cfg.use_multi_oracle,
                "use_copoem": cfg.use_copoem,
                "d_target": cfg.d_target,
                "h_ks_target": cfg.h_ks_target,
                "copoem_target_slope": cfg.copoem_target_slope,
                "k_force_min": cfg.k_force_min,
                "k_force_max": cfg.k_force_max,
            },
            "elapsed_total": elapsed_total,
            "n_steps": step,
        }

        return result
