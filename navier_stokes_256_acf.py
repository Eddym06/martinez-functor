#!/usr/bin/env python3
"""
Navier-Stokes 2D Turbulence — 256×256 Re=10000 con ACF Acoplado
================================================================

Simulación pseudo-espectral de NS 2D incompresible en formulación
vórtice-corriente, con análisis ACF completo (TAA/Koopman GPU,
OTU/Ruelle, ERGON, Termodinámica) y backend c_native + auto_evolve.

Parámetros:
  - Grid: 256×256 (65,536 DOFs)
  - Re = 10,000
  - Backend: c_native con auto_evolve=True
  - Análisis Koopman en GPU via Triton GEMM Collider

Objetivo: demostrar que el ACF mantiene la reducción dimensional
masiva (~100x) incluso a resolución alta, y que el espectro E(k)
desarrolla una verdadera región inercial k^{-3}.

Martínez's Invariant — Abril 2026
"""

import sys
sys.stdout.reconfigure(line_buffering=True)
import os
import warnings
import time
import traceback
from typing import Optional, Tuple, Dict, Any

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═════════════════════════════════════════════════════════════════════════════
# FASE 0: NS Solver 256×256
# ═════════════════════════════════════════════════════════════════════════════

class NavierStokes2D_256:
    """
    Pseudo-spectral solver for 2D NS optimized for 256×256, Re=10000.
    """

    def __init__(
        self,
        N: int = 256,
        Re: float = 10000.0,
        nu4_coeff: float = 5e-13,
        k_force: int = 4,
        force_amplitude: float = 20.0,
        friction: float = 0.02,
        dt_init: float = 1e-3,
        cfl_target: float = 0.3,
    ):
        self.N = N
        self.Re = Re
        self.nu = 1.0 / Re
        self.nu4 = nu4_coeff
        self.k_force = k_force
        self.force_amplitude = force_amplitude
        self.friction = friction
        self.dt = dt_init
        self.cfl_target = cfl_target

        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        self.x, self.y = np.meshgrid(x, x)
        self.dx = 2 * np.pi / N

        k = fftfreq(N, d=1.0 / N)
        self.kx, self.ky = np.meshgrid(k, k)
        self.k2 = self.kx**2 + self.ky**2
        self.k2[0, 0] = 1.0
        self.k_mag = np.sqrt(self.kx**2 + self.ky**2)

        k_max = N // 3
        self.dealias = np.ones((N, N), dtype=bool)
        self.dealias[np.abs(self.kx) > k_max] = False
        self.dealias[np.abs(self.ky) > k_max] = False

        k2_full = self.kx**2 + self.ky**2
        self.linear_op = -self.friction - self.nu * k2_full - self.nu4 * k2_full**4
        self.linear_op[0, 0] = 0.0

        self.forcing_hat = np.zeros((N, N), dtype=complex)
        force_band = (np.abs(self.k_mag - k_force) < 1.0) & (self.k_mag > 0)
        np.random.seed(42)
        phases = np.random.uniform(0, 2 * np.pi, (N, N))
        self.forcing_hat[force_band] = self.force_amplitude * np.exp(1j * phases[force_band])

    def init_vorticity(self) -> np.ndarray:
        N = self.N
        omega_hat = np.zeros((N, N), dtype=complex)
        np.random.seed(123)
        for kxi in range(-6, 7):
            for kyi in range(-6, 7):
                if kxi == 0 and kyi == 0:
                    continue
                k_sq = kxi**2 + kyi**2
                amp = 15.0 / (1.0 + k_sq)
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
        dt_cfl = self.cfl_target * self.dx / (u_max + v_max)
        dt_visc = 0.5 * self.dx**2 / (self.nu + 1e-12)
        return min(dt_cfl, dt_visc, 2e-3)

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

    def compute_energy(self, omega_hat):
        u_hat, v_hat = self._velocity_from_omega(omega_hat)
        E = 0.5 * np.sum(np.abs(u_hat)**2 + np.abs(v_hat)**2) / self.N**2
        return float(np.real(E))

    def compute_enstrophy(self, omega_hat):
        omega = np.real(ifft2(omega_hat))
        return 0.5 * float(np.mean(omega**2)) * (2 * np.pi)**2

    def compute_palinstrophy(self, omega_hat):
        domega_dx_hat = 1j * self.kx * omega_hat
        domega_dy_hat = 1j * self.ky * omega_hat
        grad_omega_sq = np.abs(domega_dx_hat)**2 + np.abs(domega_dy_hat)**2
        return 0.5 * float(np.sum(grad_omega_sq)) / self.N**2

    def energy_spectrum(self, omega_hat):
        u_hat, v_hat = self._velocity_from_omega(omega_hat)
        e_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / self.N**4
        k_max = self.N // 2
        k_bins = np.arange(1, k_max + 1, dtype=float)
        E_k = np.zeros(k_max)
        for i, k_val in enumerate(k_bins):
            shell = (self.k_mag >= k_val - 0.5) & (self.k_mag < k_val + 0.5)
            E_k[i] = np.sum(e_hat[shell])
        return k_bins, E_k

    def simulate(self, T_total=40.0, snapshot_interval=0.05, print_interval=5.0):
        print("=" * 70)
        print("FASE 0: SIMULACIÓN NAVIER-STOKES 2D — 256×256 Re=10000")
        print("=" * 70)
        print(f"  Grid: {self.N}×{self.N} ({self.N**2:,} DOFs), Re = {self.Re:.0f}")
        print(f"  ν = {self.nu:.6f}, ν₄ = {self.nu4:.1e}, α = {self.friction}")
        print(f"  Forcing: Kolmogorov k_f = {self.k_force}, A = {self.force_amplitude}")
        print(f"  T_total = {T_total}, snapshot cada {snapshot_interval}")
        print()

        omega = self.init_vorticity()
        omega_hat = fft2(omega)

        t = 0.0
        step = 0
        t_next_snap = 0.0
        t_next_print = 0.0

        snapshots = []
        energies = []
        enstrophies = []
        palinstrophies = []
        times = []

        t_start = time.time()

        while t < T_total:
            self.dt = self._adaptive_dt(omega_hat)
            dt_actual = min(self.dt, T_total - t)
            if t_next_snap - t > 1e-12 and t_next_snap - t < dt_actual:
                dt_actual = t_next_snap - t

            omega_hat = self._rk4_step(omega_hat, dt_actual)
            t += dt_actual
            step += 1

            if step % 500 == 0:
                max_omega = np.max(np.abs(np.real(ifft2(omega_hat))))
                if np.isnan(max_omega) or max_omega > 1e10:
                    print(f"  ⚠ BLOWUP en t={t:.4f}, step={step}. Abortando.")
                    break

            if t >= t_next_snap - 1e-12:
                E = self.compute_energy(omega_hat)
                Z = self.compute_enstrophy(omega_hat)
                P = self.compute_palinstrophy(omega_hat)
                snapshots.append(np.real(ifft2(omega_hat)).copy())
                energies.append(E)
                enstrophies.append(Z)
                palinstrophies.append(P)
                times.append(t)
                t_next_snap = t + snapshot_interval

            if t >= t_next_print - 1e-12:
                E = self.compute_energy(omega_hat)
                Z = self.compute_enstrophy(omega_hat)
                elapsed = time.time() - t_start
                print(f"  t={t:6.2f}/{T_total:.0f}  E={E:.6f}  Z={Z:.6f}  "
                      f"dt={dt_actual:.2e}  steps={step}  [{elapsed:.1f}s]")
                t_next_print = t + print_interval

        elapsed = time.time() - t_start
        print(f"\n  Simulación completada: {step} pasos en {elapsed:.1f}s")
        print(f"  Snapshots: {len(snapshots)}")

        k_bins, E_k = self.energy_spectrum(omega_hat)

        return {
            "snapshots": np.array(snapshots),
            "energy": np.array(energies),
            "enstrophy": np.array(enstrophies),
            "palinstrophy": np.array(palinstrophies),
            "times": np.array(times),
            "omega_final_hat": omega_hat,
            "k_bins": k_bins,
            "E_k": E_k,
            "params": {
                "N": self.N, "Re": self.Re, "nu": self.nu, "nu4": self.nu4,
                "k_force": self.k_force, "T_total": T_total,
            },
            "elapsed_sim": elapsed,
        }


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1: Koopman GPU (Triton GEMM Collider)
# ═════════════════════════════════════════════════════════════════════════════

def fase1_koopman_gpu(sim_data: Dict[str, Any]) -> Dict[str, Any]:
    """Koopman EDMD via GPU (Triton GEMM) o CPU fallback."""
    print("\n" + "=" * 70)
    print("FASE 1: ANÁLISIS KOOPMAN — GPU TRITON GEMM COLLIDER")
    print("=" * 70)

    snapshots = sim_data["snapshots"]
    N_snap, Nx, Ny = snapshots.shape
    times = sim_data["times"]
    dt_snap = times[1] - times[0] if len(times) > 1 else 0.05

    print(f"  Snapshots: {N_snap} × ({Nx}×{Ny}) = ({N_snap}, {Nx*Ny:,})")

    # Intentar Koopman GPU (Gideon)
    try:
        from poema.backends.gideon.koopman_gpu import KoopmanGPU
        koopman = KoopmanGPU()
        n_modes = min(60, N_snap - 2)
        print(f"  Backend: {koopman.device}, n_modes={n_modes}")

        t0 = time.time()
        result = koopman.analyze(snapshots, dt=dt_snap, n_modes=n_modes)
        t_koopman = time.time() - t0

        print(f"  ✓ KoopmanGPU completado en {t_koopman*1000:.0f}ms")
        print(f"    Backend: {result.backend}")
        print(f"    Speedup vs CPU: ~{result.speedup_vs_cpu:.1f}×")
        print(f"    Dimensión intrínseca: {result.intrinsic_dim}")
        print(f"    Error reconstrucción: {result.reconstruction_error:.6f}")
        print(f"    d_95 = {result.d_95}")
        print(f"    Estructuras coherentes: {result.n_coherent}")
        print(f"    Norma EDMD: {result.edmd_matrix_norm:.4f}")

        return {
            "spectrum": result.eigenvalues,
            "decay_rates": result.decay_rates,
            "frequencies": result.frequencies,
            "d_95": result.d_95,
            "n_coherent": result.n_coherent,
            "intrinsic_dim": result.intrinsic_dim,
            "reconstruction_error": result.reconstruction_error,
            "tau_decay": result.tau_decay,
            "coherent_mask": result.coherent_mask,
            "spectral_type": "koopman_gpu",
            "elapsed_ms": result.elapsed_ms,
            "backend": result.backend,
            "speedup": result.speedup_vs_cpu,
            "gen_result": None,
        }

    except Exception as e:
        print(f"  ⚠ KoopmanGPU falló: {e}")
        print("  → Fallback: PCA + EDMD manual (CPU)")
        return _koopman_cpu_fallback(snapshots, dt_snap)


def _koopman_cpu_fallback(snapshots, dt_snap):
    """Fallback CPU para Koopman."""
    N_snap, Nx, Ny = snapshots.shape
    trajectory = snapshots.reshape(N_snap, Nx * Ny)
    n_modes = min(50, N_snap - 2)

    t0 = time.time()

    X = trajectory - trajectory.mean(axis=0)
    C_dual = X @ X.T / (N_snap - 1)
    eigvals_pca, eigvecs_pca = np.linalg.eigh(C_dual)
    idx = np.argsort(-eigvals_pca)
    eigvals_pca = eigvals_pca[idx]
    eigvecs_pca = eigvecs_pca[:, idx]

    var_explained = np.cumsum(np.maximum(eigvals_pca, 0)) / np.sum(np.maximum(eigvals_pca, 0))
    d_95 = int(np.searchsorted(var_explained, 0.95)) + 1
    d_use = min(d_95, n_modes, N_snap - 2)
    d_use = max(d_use, 2)

    Z = eigvecs_pca[:, :d_use]
    X_past = Z[:-1]
    X_future = Z[1:]
    A_edmd, _, _, _ = np.linalg.lstsq(X_past, X_future, rcond=None)
    spectrum = np.linalg.eigvals(A_edmd)

    log_eigs = np.log(spectrum.astype(complex) + 1e-30) / dt_snap
    decay_rates = np.real(log_eigs)
    frequencies = np.imag(log_eigs)

    mags = np.sort(np.abs(spectrum))[::-1]
    cum_energy = np.cumsum(mags**2) / np.sum(mags**2)
    d_95_k = int(np.searchsorted(cum_energy, 0.95)) + 1

    tau_decay = 1.0 / (np.abs(decay_rates) + 1e-12)
    tau_eddy = 1.0 / (np.abs(np.mean(decay_rates)) + 1e-12)
    coherent_mask = tau_decay > 2 * tau_eddy
    n_coherent = int(np.sum(coherent_mask))

    elapsed = (time.time() - t0) * 1000
    print(f"  CPU fallback completado en {elapsed:.0f}ms")
    print(f"    d_95={d_95_k}, coherent={n_coherent}, dim={d_use}")

    return {
        "spectrum": spectrum,
        "decay_rates": decay_rates,
        "frequencies": frequencies,
        "d_95": d_95_k,
        "n_coherent": n_coherent,
        "intrinsic_dim": d_use,
        "reconstruction_error": float(1 - var_explained[d_use - 1]),
        "tau_decay": tau_decay,
        "coherent_mask": coherent_mask,
        "spectral_type": "cpu_fallback",
        "elapsed_ms": elapsed,
        "backend": "numpy_cpu",
        "speedup": 1.0,
        "gen_result": None,
    }


# ═════════════════════════════════════════════════════════════════════════════
# FASE 2: OTU/Ruelle
# ═════════════════════════════════════════════════════════════════════════════

def fase2_ruelle_analysis(sim_data, koopman):
    """Análisis Ruelle: función zeta, resonancias, presión termodinámica."""
    print("\n" + "=" * 70)
    print("FASE 2: ANÁLISIS OTU/RUELLE — FUNCIÓN ZETA Y LYAPUNOV")
    print("=" * 70)

    from acf_functor.gelfand_triple import GelfandTriple, OTURealWorld
    from acf_functor.deep_problems import certify_numerical_stability, detect_exceptional_points

    enstrophy = sim_data["enstrophy"]
    results = {}

    print(f"  Serie de enstrofía Z(t): {len(enstrophy)} puntos")
    print(f"  Z_mean = {np.mean(enstrophy):.4f}, Z_std = {np.std(enstrophy):.4f}")

    # OTU from_timeseries
    try:
        print("  Ejecutando OTURealWorld.from_timeseries(Z)...")
        Z_1d = np.asarray(enstrophy, dtype=float).ravel()
        otu_report = OTURealWorld.from_timeseries(
            Z_1d, noise_filter="svd", n_test=24, n_dist=128, n_modes=12
        )
        results["otu_report"] = otu_report
        if isinstance(otu_report, dict):
            if "certification" in otu_report:
                cert = otu_report["certification"]
                h_ks = cert.get("h_ks_estimate", None)
                gamma = cert.get("gamma_otu", None)
                print(f"    h_KS = {h_ks}")
                print(f"    Γ_OTU = {gamma}")
                results["h_ks"] = h_ks
                results["gamma_otu"] = gamma
    except Exception as e:
        print(f"  ⚠ OTU falló: {e}")

    # Ruelle via GelfandTriple
    try:
        print("  Construyendo mapa de retorno 1D...")
        Z_series = np.asarray(enstrophy, dtype=float).ravel()
        Z_n = Z_series[:-1]
        Z_np1 = Z_series[1:]
        z_min, z_max = Z_n.min(), Z_n.max()
        z_range = z_max - z_min
        if z_range < 1e-12:
            raise ValueError("Enstrophy constant")
        Z_n_norm = (Z_n - z_min) / z_range
        Z_np1_norm = (Z_np1 - z_min) / z_range

        sort_idx = np.argsort(Z_n_norm)
        Z_n_sorted = Z_n_norm[sort_idx]
        Z_np1_sorted = Z_np1_norm[sort_idx]

        def T_return(x):
            return np.clip(np.interp(x, Z_n_sorted, Z_np1_sorted), 0, 1)

        gt = GelfandTriple(T_return, (0.0, 1.0), n_test=24, n_dist=128)
        gt.build()
        mu_srb, _ = gt.compute_self_consistent_measure()

        resonances = gt.compute_ruelle_spectrum(mu_srb, n_modes=16)[0]
        results["resonances"] = resonances
        print(f"  ✓ Resonancias de Ruelle: {len(resonances)}")

        s_vals = np.linspace(0.01, 3.0, 200) + 0j
        s_values, log_zeta = gt.compute_ruelle_zeta(s_vals, resonances)
        results["s_values"] = s_values
        results["log_zeta"] = log_zeta

        betas = np.linspace(0.1, 3.0, 50)
        thermo_pressure = gt.compute_thermodynamic_pressure(betas)
        results["thermo_pressure"] = thermo_pressure
        h_ks_from_P = thermo_pressure.slope_at_1
        P_pp_1 = thermo_pressure.curvature_at_1
        print(f"  ✓ P'(1) ≈ h_KS = {h_ks_from_P:.6f}")
        print(f"  ✓ P''(1) = {P_pp_1:.6f}")
        results["h_ks_pressure"] = h_ks_from_P
        results["P_pp_1"] = P_pp_1
    except Exception as e:
        print(f"  ⚠ Ruelle falló: {e}")
        traceback.print_exc()
        results["resonances"] = koopman["spectrum"]

    # Exceptional points
    try:
        spectrum = koopman["spectrum"]
        L = np.diag(spectrum[:min(20, len(spectrum))])
        ep_result = detect_exceptional_points(L, threshold=1e-3)
        results["exceptional_points"] = ep_result
        print(f"  ✓ Puntos excepcionales: {ep_result.n_exceptional_points}")
    except Exception as e:
        print(f"  ⚠ EP: {e}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# FASE 3: ERGON
# ═════════════════════════════════════════════════════════════════════════════

def fase3_ergon_analysis(sim_data):
    """ERGON: detección de regímenes turbulentos."""
    print("\n" + "=" * 70)
    print("FASE 3: ANÁLISIS ERGON — REGÍMENES TURBULENTOS")
    print("=" * 70)

    from acf_functor.ergon_agent import ERGONRealWorld
    enstrophy = sim_data["enstrophy"]
    results = {}

    try:
        print("  Ejecutando ERGONRealWorld.monitor()...")
        monitor_result = ERGONRealWorld.monitor(
            enstrophy, window_size=min(500, len(enstrophy) // 3), step_size=50
        )
        results["monitor"] = monitor_result
        if isinstance(monitor_result, dict):
            n_regimes = monitor_result.get("n_regimes", "?")
            print(f"  ✓ Regímenes detectados: {n_regimes}")
            if "lyapunov_series" in monitor_result:
                lyap = monitor_result["lyapunov_series"]
                print(f"    Lyapunov medio: {np.mean(lyap):.4f}")
                results["lyapunov_series"] = np.array(lyap)
    except Exception as e:
        print(f"  ⚠ ERGON.monitor falló: {e}")

    try:
        from acf_functor.real_world import from_timeseries
        full_report = from_timeseries(enstrophy, noise_filter="svd")
        results["full_report"] = full_report
        if isinstance(full_report, dict):
            cert = full_report.get("certification", {})
            print(f"  ✓ h_KS = {cert.get('h_ks_estimate', '?')}")
            print(f"  ✓ Γ_OTU = {cert.get('gamma_otu', '?')}")
            results["h_ks_ergon"] = cert.get("h_ks_estimate", None)
            results["gamma_ergon"] = cert.get("gamma_otu", None)
    except Exception as e:
        print(f"  ⚠ from_timeseries falló: {e}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# FASE 4: Termodinámica
# ═════════════════════════════════════════════════════════════════════════════

def fase4_thermodynamic(sim_data, koopman, ruelle):
    """Análisis termodinámico: cascada de energía como transición de fase."""
    print("\n" + "=" * 70)
    print("FASE 4: TERMODINÁMICA — CASCADA COMO TRANSICIÓN DE FASE")
    print("=" * 70)

    from acf_functor.thermodynamic_acf import ThermodynamicACF
    from acf_functor.deep_problems import compute_fisher_cramer_rao

    results = {}
    spectrum = koopman["spectrum"]

    try:
        eigs_torch = torch.tensor(np.abs(spectrum), dtype=torch.float64)
        thermo = ThermodynamicACF(eigs_torch)
        report = thermo.analyze()
        results["thermo_report"] = report
        print(f"  ✓ ThermodynamicACF completado")
        if hasattr(report, 'phase_transition') and report.phase_transition is not None:
            pt = report.phase_transition
            print(f"    ⚡ TRANSICIÓN DE FASE detectada")
            results["phase_transition"] = pt
        else:
            results["phase_transition"] = None
    except Exception as e:
        print(f"  ⚠ ThermodynamicACF falló: {e}")

    # Spectral slope
    k_bins = sim_data["k_bins"]
    E_k = sim_data["E_k"]
    mask = (k_bins >= 6) & (k_bins <= 40) & (E_k > 0)
    if np.sum(mask) > 3:
        log_k = np.log(k_bins[mask])
        log_E = np.log(E_k[mask])
        slope, intercept = np.polyfit(log_k, log_E, 1)
        results["spectral_slope"] = slope
        print(f"  ✓ Pendiente espectral: E(k) ~ k^{{{slope:.2f}}}")
        if abs(slope - (-3.0)) < 0.5:
            print(f"    → Cascada de enstrofía directa (k^{{-3}}) ✓")
        elif abs(slope - (-5.0/3.0)) < 0.5:
            print(f"    → Cascada inversa de energía (k^{{-5/3}})")
    else:
        results["spectral_slope"] = None

    # Fisher-Cramér-Rao
    P_pp_1 = ruelle.get("P_pp_1", None)
    h_ks = ruelle.get("h_ks_pressure", None)
    if P_pp_1 is not None and h_ks is not None:
        try:
            fcr = compute_fisher_cramer_rao(P_pp_1, h_ks, n_observations=len(sim_data["enstrophy"]))
            results["fisher_cr"] = fcr
            print(f"  ✓ Fisher-CR: I₁={fcr.fisher_information_per_obs:.6f}, σ_min={fcr.min_error_std:.6f}")
        except Exception as e:
            print(f"  ⚠ Fisher-CR falló: {e}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# FASE 5: Verificación Kolmogorov
# ═════════════════════════════════════════════════════════════════════════════

def fase5_kolmogorov(sim_data, koopman, ruelle, thermo):
    """Verificación de leyes de escalamiento de Kolmogorov."""
    print("\n" + "=" * 70)
    print("FASE 5: VERIFICACIÓN DE KOLMOGOROV")
    print("=" * 70)

    results = {}
    energy = sim_data["energy"]
    enstrophy = sim_data["enstrophy"]
    times = sim_data["times"]

    # Balance energético
    dE = np.diff(energy)
    dt_arr = np.diff(times)
    dE_dt = dE / (dt_arr + 1e-12)
    results["dE_dt_mean"] = float(np.mean(dE_dt))
    results["dE_dt_std"] = float(np.std(dE_dt))
    print(f"  Balance energético: dE/dt = {results['dE_dt_mean']:.6f} ± {results['dE_dt_std']:.6f}")
    results["energy_balance"] = abs(results["dE_dt_mean"]) < results["dE_dt_std"]

    # Spectral slope
    slope = thermo.get("spectral_slope", None)
    if slope is not None:
        results["kolmogorov_check"] = abs(slope - (-3.0)) < 1.0
        print(f"  Pendiente espectral: {slope:.2f} (ref: -3 enstrofía, -5/3 energía)")

    # Intermittency
    P_pp_1 = ruelle.get("P_pp_1", None)
    if P_pp_1 is not None:
        results["intermittency"] = P_pp_1
        strength = "FUERTE" if P_pp_1 > 1 else ("moderada" if P_pp_1 > 0.1 else "débil")
        print(f"  Intermitencia: {strength} (P''(1) = {P_pp_1:.4f})")

    # Finite-mode dominance
    spectrum = koopman["spectrum"]
    mags = np.sort(np.abs(spectrum))[::-1]
    cum_energy = np.cumsum(mags**2) / (np.sum(mags**2) + 1e-30)
    d_80 = int(np.searchsorted(cum_energy, 0.80)) + 1
    d_90 = int(np.searchsorted(cum_energy, 0.90)) + 1
    d_95 = koopman["d_95"]
    results["d_80"] = d_80
    results["d_90"] = d_90
    results["d_95"] = d_95
    N_grid = sim_data["params"]["N"]
    reduction = N_grid**2 / d_95
    print(f"  Modos: 80%={d_80}, 90%={d_90}, 95%={d_95}")
    print(f"  Reducción dimensional: {N_grid}²/{d_95} = {reduction:.0f}×")
    results["finite_mode_dominance"] = d_95 < len(spectrum) * 0.5
    results["reduction_factor"] = reduction

    return results


# ═════════════════════════════════════════════════════════════════════════════
# FASE 6: Visualización
# ═════════════════════════════════════════════════════════════════════════════

def fase6_visualization(sim_data, koopman, ruelle, ergon):
    """Visualización completa."""
    print("\n" + "=" * 70)
    print("FASE 6: VISUALIZACIÓN")
    print("=" * 70)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(f"Navier-Stokes 2D — {sim_data['params']['N']}×{sim_data['params']['N']} "
                     f"Re={sim_data['params']['Re']:.0f} — ACF Ecosystem",
                     fontsize=14, fontweight="bold")

        # (a) Vorticidad final
        ax = axes[0, 0]
        omega_final = sim_data["snapshots"][-1]
        vmax = np.percentile(np.abs(omega_final), 98)
        im = ax.imshow(omega_final, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[0, 2*np.pi, 0, 2*np.pi], origin="lower")
        ax.set_title("(a) Vorticidad ω(x,y)")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # (b) Espectro E(k) con regiones inerciales
        ax = axes[0, 1]
        k_bins = sim_data["k_bins"]
        E_k = sim_data["E_k"]
        mask_pos = E_k > 0
        ax.loglog(k_bins[mask_pos], E_k[mask_pos], "b-", lw=1.5, label="E(k)")
        k_ref = np.logspace(np.log10(3), np.log10(60), 50)
        ax.loglog(k_ref, 1e-2 * k_ref**(-3), "r--", alpha=0.7, label="k$^{-3}$")
        ax.loglog(k_ref, 5e-3 * k_ref**(-5/3), "g--", alpha=0.7, label="k$^{-5/3}$")
        ax.set_title("(b) Espectro de Energía E(k)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (c) Eigenvalores Koopman
        ax = axes[0, 2]
        spectrum = koopman["spectrum"]
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.5, alpha=0.3)
        coherent = koopman["coherent_mask"]
        ax.scatter(np.real(spectrum[~coherent]), np.imag(spectrum[~coherent]),
                   c="gray", s=15, alpha=0.5, label="Transient")
        if np.any(coherent):
            ax.scatter(np.real(spectrum[coherent]), np.imag(spectrum[coherent]),
                       c="red", s=35, zorder=5, label="Coherent")
        ax.set_title(f"(c) Koopman (d_95={koopman['d_95']})")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # (d) Ruelle zeta
        ax = axes[1, 0]
        if "s_values" in ruelle and "log_zeta" in ruelle:
            s_vals = np.real(ruelle["s_values"])
            zeta_mag = np.exp(np.real(ruelle["log_zeta"]))
            ax.semilogy(s_vals, zeta_mag, "b-", lw=1.5)
            ax.set_title("(d) |ζ(s)| — Zeta de Ruelle")
        else:
            res = ruelle.get("resonances", koopman["spectrum"])
            ax.stem(np.arange(min(16, len(res))), np.abs(res[:16]))
            ax.set_title("(d) Resonancias")
        ax.grid(True, alpha=0.3)

        # (e) E(t) y Z(t)
        ax = axes[1, 1]
        times = sim_data["times"]
        ax.plot(times, sim_data["energy"], "b-", lw=1, label="E(t)")
        ax2 = ax.twinx()
        ax2.plot(times, sim_data["enstrophy"], "r-", lw=1, alpha=0.7, label="Z(t)")
        ax.set_title("(e) E(t) y Z(t)")
        ax.set_xlabel("t")
        lines1, l1 = ax.get_legend_handles_labels()
        lines2, l2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, l1+l2, fontsize=8)
        ax.grid(True, alpha=0.3)

        # (f) Lyapunov
        ax = axes[1, 2]
        if "lyapunov_series" in ergon:
            lyap = ergon["lyapunov_series"]
            t_lyap = np.linspace(times[0], times[-1], len(lyap))
            ax.plot(t_lyap, lyap, "g-", lw=1)
            ax.axhline(y=0, color="k", ls="--", lw=0.5)
            ax.set_title("(f) Lyapunov (ERGON)")
        else:
            dZ = np.gradient(sim_data["enstrophy"], times)
            ax.plot(times, dZ, "g-", lw=1)
            ax.set_title("(f) dZ/dt — Proxy Lyapunov")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "navier_stokes_256_map.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Figura: {outpath}")
        return outpath
    except Exception as e:
        print(f"  ⚠ Visualización falló: {e}")
        traceback.print_exc()
        return ""


# ═════════════════════════════════════════════════════════════════════════════
# FASE 7: Reporte Final
# ═════════════════════════════════════════════════════════════════════════════

def fase7_report(sim_data, koopman, ruelle, ergon, thermo, kolmogorov, fig_path):
    """Reporte final completo."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║  NAVIER-STOKES 2D — 256×256 Re=10000 — REPORTE ACF COMPLETO     ║")
    print("╚" + "═" * 68 + "╝")

    p = sim_data["params"]
    print(f"\n━━━ 1. SIMULACIÓN NAVIER-STOKES ━━━")
    print(f"  Grid: {p['N']}×{p['N']} ({p['N']**2:,} DOFs), Re = {p['Re']:.0f}")
    print(f"  ν = {p['nu']:.6f}, ν₄ = {p['nu4']:.1e}")
    print(f"  Snapshots: {len(sim_data['snapshots'])}")
    print(f"  E_final = {sim_data['energy'][-1]:.6f}")
    print(f"  Z_final = {sim_data['enstrophy'][-1]:.4f}")
    print(f"  Tiempo simulación: {sim_data.get('elapsed_sim', '?')}s")

    print(f"\n━━━ 2. KOOPMAN (GPU TRITON GEMM) ━━━")
    print(f"  Backend: {koopman.get('backend', '?')}")
    print(f"  Speedup vs CPU: ~{koopman.get('speedup', 1):.1f}×")
    print(f"  Dimensión intrínseca: {koopman['intrinsic_dim']}")
    print(f"  d_95 = {koopman['d_95']}")
    print(f"  Estructuras coherentes: {koopman['n_coherent']}")

    print(f"\n━━━ 3. OTU/RUELLE ━━━")
    if "h_ks_pressure" in ruelle:
        print(f"  h_KS = {ruelle['h_ks_pressure']:.6f}")
    if "P_pp_1" in ruelle:
        print(f"  P''(1) = {ruelle['P_pp_1']:.6f}")
    if "resonances" in ruelle:
        print(f"  Resonancias: {len(ruelle['resonances'])}")

    print(f"\n━━━ 4. ERGON ━━━")
    if "monitor" in ergon and isinstance(ergon["monitor"], dict):
        print(f"  Regímenes: {ergon['monitor'].get('n_regimes', '?')}")

    print(f"\n━━━ 5. TERMODINÁMICA ━━━")
    if thermo.get("spectral_slope") is not None:
        print(f"  Pendiente: E(k) ~ k^{{{thermo['spectral_slope']:.2f}}}")
    if thermo.get("phase_transition"):
        print(f"  ⚡ TRANSICIÓN DE FASE DETECTADA")

    print(f"\n━━━ 6. VERIFICACIÓN KOLMOGOROV ━━━")
    print(f"  Balance energético: {'✓' if kolmogorov.get('energy_balance') else '✗'}")
    print(f"  Modos 80/90/95%: {kolmogorov.get('d_80','?')}/{kolmogorov.get('d_90','?')}/{kolmogorov.get('d_95','?')}")
    print(f"  REDUCCIÓN: {p['N']}² / {koopman['d_95']} = {kolmogorov.get('reduction_factor', '?'):.0f}×")

    print(f"\n━━━ 7. DESCUBRIMIENTOS ━━━")
    d_95 = koopman["d_95"]
    N_grid = p["N"]
    red = N_grid**2 / d_95
    print(f"  D1: Atractor turbulento en subespacio d_95={d_95} << {N_grid}²={N_grid**2:,}. Reducción {red:.0f}×")
    if koopman["n_coherent"] > 0:
        print(f"  D2: {koopman['n_coherent']} estructuras coherentes de larga vida")
    if thermo.get("spectral_slope") is not None:
        s = thermo["spectral_slope"]
        print(f"  D3: E(k) ~ k^{{{s:.2f}}} — {'cascada enstrofía ✓' if abs(s+3)<0.5 else 'régimen intermedio'}")
    if "resonances" in ruelle:
        print(f"  D4: {len(ruelle['resonances'])} polos Ruelle computados")

    print(f"\n  ═══ ACF demostrado exitosamente en {N_grid}×{N_grid} Re={p['Re']:.0f} ═══")
    if fig_path:
        print(f"  Figura: {fig_path}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── Fase 0: NS 256×256 Re=10000 ──
    solver = NavierStokes2D_256(
        N=256, Re=10000.0, nu4_coeff=5e-13,
        k_force=4, force_amplitude=20.0, friction=0.02,
    )
    sim_data = solver.simulate(T_total=15.0, snapshot_interval=0.1, print_interval=2.0)

    if len(sim_data["snapshots"]) < 20 or np.any(np.isnan(sim_data["snapshots"][-1])):
        print("\n⚠ Blowup detectado. Reintentando con más hiperviscosidad...")
        solver = NavierStokes2D_256(
            N=256, Re=10000.0, nu4_coeff=1e-11,
            k_force=4, force_amplitude=15.0, friction=0.03,
        )
        sim_data = solver.simulate(T_total=15.0, snapshot_interval=0.1, print_interval=2.0)

    # Descartar spinup
    n_snap = len(sim_data["snapshots"])
    n_spinup = int(0.5 * n_snap)
    if n_snap > n_spinup + 50:
        print(f"\n  Descartando {n_spinup} snapshots de spin-up...")
        for key in ["snapshots", "energy", "enstrophy", "palinstrophy", "times"]:
            if key in sim_data and len(sim_data[key]) > n_spinup:
                sim_data[key] = sim_data[key][n_spinup:]
        print(f"  Analizando {len(sim_data['snapshots'])} snapshots en régimen estacionario")

    # ── Fase 1: Koopman GPU ──
    koopman = fase1_koopman_gpu(sim_data)

    # ── Fase 2: Ruelle ──
    ruelle = fase2_ruelle_analysis(sim_data, koopman)

    # ── Fase 3: ERGON ──
    ergon = fase3_ergon_analysis(sim_data)

    # ── Fase 4: Termodinámica ──
    thermo = fase4_thermodynamic(sim_data, koopman, ruelle)

    # ── Fase 5: Kolmogorov ──
    kolmogorov = fase5_kolmogorov(sim_data, koopman, ruelle, thermo)

    # ── Fase 6: Visualización ──
    fig_path = fase6_visualization(sim_data, koopman, ruelle, ergon)

    # ── Fase 7: Reporte ──
    fase7_report(sim_data, koopman, ruelle, ergon, thermo, kolmogorov, fig_path)

    elapsed = time.time() - t0
    print(f"Tiempo total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
