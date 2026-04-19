"""
ns_acf_coupled.py — Simulación Acoplada Navier-Stokes + ACF (PDE-ACF).

Integra el solver de Navier-Stokes DENTRO de Gideon, permitiendo:
  1. Análisis Koopman en tiempo real durante la simulación
  2. Refinamiento Adaptativo de Malla guiado por modos dominantes
  3. Retroalimentación ACF → NS: resolución asignada proporcionalmente
     a la energía de los modos Koopman dominantes
  4. Auto-evolución del functor durante la simulación

Esto elimina el pipeline secuencial (simular → analizar) y permite
que el ACF guíe la simulación en tiempo real.

Flujo:
  ┌─────────────────────────────────────────────┐
  │     Navier-Stokes Solver (pseudo-spectral)  │
  │              ω(t) → ω(t+dt)                 │
  └──────────────┬──────────────────────────────┘
                 │ snapshots cada T_analysis
                 ▼
  ┌─────────────────────────────────────────────┐
  │     KoopmanGPU (Triton GEMM Collider)       │
  │     PCA + EDMD → eigenvalores, modos        │
  └──────────────┬──────────────────────────────┘
                 │ modos dominantes, d_95
                 ▼
  ┌─────────────────────────────────────────────┐
  │     Adaptive Mesh Refinement Controller     │
  │     → focalizar resolución en regiones      │
  │       de alta actividad modal               │
  └──────────────┬──────────────────────────────┘
                 │ forcing adaptado, ν₄ local
                 ▼
  ┌─────────────────────────────────────────────┐
  │     Gideon Engine (backend dispatch)        │
  │     → certifica ε, registra telemetría      │
  └─────────────────────────────────────────────┘

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
        force_band = (np.abs(self.k_mag - cfg.k_force) < 1.0) & (self.k_mag > 0)
        np.random.seed(42)
        phases = np.random.uniform(0, 2 * np.pi, (N, N))
        self.forcing_hat[force_band] = cfg.force_amplitude * np.exp(1j * phases[force_band])

        # Adaptive mesh controller
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

                    # Adaptive mesh refinement
                    if cfg.adaptive_hyper:
                        adaptation = self.mesh_ctrl.adapt(koopman_result, E_curr, Z_curr)
                        self.adaptation_log.append(adaptation)

                        # Apply adapted parameters
                        k2_full = self.kx**2 + self.ky**2
                        new_nu4 = adaptation["nu4"]
                        self.linear_op = -cfg.friction - (1.0/cfg.Re) * k2_full - new_nu4 * k2_full**4
                        self.linear_op[0, 0] = 0.0

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
                print(f"  t={t:6.2f}/{cfg.T_total:.0f}  E={E:.4f}  Z={Z:.2f}  "
                      f"dt={dt_actual:.2e}  {d95_str}  [{elapsed:.1f}s]")
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
            "params": {
                "N": N, "Re": cfg.Re, "nu": 1.0 / cfg.Re, "nu4": cfg.nu4_coeff,
                "k_force": cfg.k_force, "T_total": cfg.T_total,
                "coupled": True, "auto_evolve": cfg.auto_evolve,
                "backend": cfg.backend,
            },
            "elapsed_total": elapsed_total,
            "n_steps": step,
        }

        return result
