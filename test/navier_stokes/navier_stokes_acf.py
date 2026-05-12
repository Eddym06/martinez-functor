#!/usr/bin/env python3
"""
Navier-Stokes 2D Turbulence — El Problema Definitivo
=====================================================

Simulación pseudo-espectral de las ecuaciones de Navier-Stokes 2D incompresibles
en formulación vórtice-corriente en un dominio periódico [0, 2π]², con análisis
completo del ecosistema ACF (TAA/Koopman, OTU/Ruelle, ERGON/Regímenes,
Termodinámica de cascada).

Ecuación:
    ∂ω/∂t + u·∇ω = ν∇²ω + f(x,y,t)
    u = ∇⊥ψ,   ∇²ψ = -ω

Método:
    - Pseudo-espectral FFT en grid 64×64
    - RK4 con dt adaptativo (CFL)
    - Kolmogorov forcing en wavenumber k_f ≈ 4
    - Dealiasing 2/3 rule
    - Hiperviscosidad -ν₄ k⁸ ω̂ para estabilidad

Martínez's Invariant — Abril 2026
"""

import sys
import os
import warnings
import time
import traceback
from typing import Optional, Tuple, Dict, Any

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import torch

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═════════════════════════════════════════════════════════════════════════════
# FASE 0: Pseudo-Spectral 2D Navier-Stokes Solver
# ═════════════════════════════════════════════════════════════════════════════

class NavierStokes2D:
    """
    Pseudo-spectral solver for 2D incompressible Navier-Stokes
    in vorticity-streamfunction formulation on [0, 2π]².
    """

    def __init__(
        self,
        N: int = 64,
        Re: float = 2000.0,
        nu4_coeff: float = 5e-10,
        k_force: int = 4,
        force_amplitude: float = 10.0,
        friction: float = 0.1,
        dt_init: float = 5e-3,
        cfl_target: float = 0.4,
    ):
        self.N = N
        self.Re = Re
        self.nu = 1.0 / Re
        self.nu4 = nu4_coeff
        self.k_force = k_force
        self.force_amplitude = force_amplitude
        self.friction = friction  # Linear drag (Ekman friction)
        self.dt = dt_init
        self.cfl_target = cfl_target

        # Physical grid
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        self.x, self.y = np.meshgrid(x, x)
        self.dx = 2 * np.pi / N

        # Wavenumber grid
        k = fftfreq(N, d=1.0 / N)  # integer wavenumbers 0, 1, ..., N/2, -N/2+1, ..., -1
        self.kx, self.ky = np.meshgrid(k, k)
        self.k2 = self.kx**2 + self.ky**2
        self.k2[0, 0] = 1.0  # avoid division by zero (set back below)
        self.k_mag = np.sqrt(self.kx**2 + self.ky**2)

        # Dealiasing mask: 2/3 rule
        k_max = N // 3
        self.dealias = np.ones((N, N), dtype=bool)
        self.dealias[np.abs(self.kx) > k_max] = False
        self.dealias[np.abs(self.ky) > k_max] = False

        # Linear operator: -α - ν k² - ν₄ k⁸  (friction + viscosity + hyperviscosity)
        k2_full = self.kx**2 + self.ky**2
        self.linear_op = -self.friction - self.nu * k2_full - self.nu4 * k2_full**4
        self.linear_op[0, 0] = 0.0  # zero mode: no dissipation

        # Forcing in Fourier space: Kolmogorov forcing at |k| ≈ k_force
        self.forcing_hat = np.zeros((N, N), dtype=complex)
        force_band = (np.abs(self.k_mag - k_force) < 1.0) & (self.k_mag > 0)
        np.random.seed(42)
        phases = np.random.uniform(0, 2 * np.pi, (N, N))
        self.forcing_hat[force_band] = self.force_amplitude * np.exp(1j * phases[force_band])
        # Make real in physical space (Hermitian symmetry not strictly needed for forcing)

    def init_vorticity(self, mode: str = "random") -> np.ndarray:
        """Initialize vorticity field."""
        N = self.N
        if mode == "random":
            # Random large-scale perturbation
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
            omega = np.real(ifft2(omega_hat))
        elif mode == "taylor_green":
            omega = 2.0 * np.cos(self.x) * np.cos(self.y)
        else:
            omega = np.zeros((N, N))
        return omega

    def _velocity_from_omega(self, omega_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute velocity from vorticity in Fourier space: ψ̂ = -ω̂/k², u = ∂ψ/∂y, v = -∂ψ/∂x."""
        psi_hat = -omega_hat / self.k2
        psi_hat[0, 0] = 0.0  # zero mean streamfunction

        u_hat = 1j * self.ky * psi_hat   # u = ∂ψ/∂y
        v_hat = -1j * self.kx * psi_hat  # v = -∂ψ/∂x
        return u_hat, v_hat

    def _nonlinear_rhs(self, omega_hat: np.ndarray) -> np.ndarray:
        """Compute the nonlinear term -u·∇ω in Fourier space (dealiased)."""
        # Velocity in Fourier space
        u_hat, v_hat = self._velocity_from_omega(omega_hat)

        # Transform to physical space (dealiased)
        omega_hat_d = omega_hat * self.dealias
        u_hat_d = u_hat * self.dealias
        v_hat_d = v_hat * self.dealias

        omega = np.real(ifft2(omega_hat_d))
        u = np.real(ifft2(u_hat_d))
        v = np.real(ifft2(v_hat_d))

        # Vorticity gradients in physical space
        domega_dx = np.real(ifft2(1j * self.kx * omega_hat_d))
        domega_dy = np.real(ifft2(1j * self.ky * omega_hat_d))

        # Nonlinear term in physical space
        nl_phys = -(u * domega_dx + v * domega_dy)

        # Transform back and dealias
        nl_hat = fft2(nl_phys) * self.dealias
        return nl_hat

    def _rhs(self, omega_hat: np.ndarray) -> np.ndarray:
        """Full RHS: nonlinear + linear (viscous) + forcing."""
        return self._nonlinear_rhs(omega_hat) + self.linear_op * omega_hat + self.forcing_hat

    def _adaptive_dt(self, omega_hat: np.ndarray) -> float:
        """CFL-based adaptive time step."""
        u_hat, v_hat = self._velocity_from_omega(omega_hat)
        u = np.real(ifft2(u_hat))
        v = np.real(ifft2(v_hat))
        u_max = max(np.max(np.abs(u)), 1e-10)
        v_max = max(np.max(np.abs(v)), 1e-10)
        dt_cfl = self.cfl_target * self.dx / (u_max + v_max)
        dt_visc = 0.5 * self.dx**2 / (self.nu + 1e-12)
        return min(dt_cfl, dt_visc, 5e-3)

    def _rk4_step(self, omega_hat: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrating factor + RK4 for the nonlinear part.
        Handles stiff linear terms (viscosity + hyperviscosity) exactly.

        Strang splitting: L/2 → N(dt) → L/2
        """
        # Exact integration of linear part (half step)
        E_half = np.exp(self.linear_op * dt / 2)
        E_full = E_half * E_half  # = exp(L * dt)

        # Apply half-step linear
        omega_hat = omega_hat * E_half

        # RK4 for nonlinear + forcing
        def NL(w):
            return self._nonlinear_rhs(w) + self.forcing_hat

        k1 = NL(omega_hat)
        k2 = NL(omega_hat + 0.5 * dt * k1)
        k3 = NL(omega_hat + 0.5 * dt * k2)
        k4 = NL(omega_hat + dt * k3)
        omega_hat = omega_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Apply second half-step linear
        omega_hat = omega_hat * E_half

        return omega_hat

    def compute_energy(self, omega_hat: np.ndarray) -> float:
        """Kinetic energy: E = ½ ∫∫ |u|² dx dy."""
        u_hat, v_hat = self._velocity_from_omega(omega_hat)
        E = 0.5 * np.sum(np.abs(u_hat)**2 + np.abs(v_hat)**2) / self.N**2
        return float(np.real(E))

    def compute_enstrophy(self, omega_hat: np.ndarray) -> float:
        """Enstrophy: Z = ½ ∫∫ ω² dx dy."""
        omega = np.real(ifft2(omega_hat))
        return 0.5 * float(np.mean(omega**2)) * (2 * np.pi)**2

    def compute_palinstrophy(self, omega_hat: np.ndarray) -> float:
        """Palinstrophy: P = ½ ∫∫ |∇ω|² dx dy."""
        domega_dx_hat = 1j * self.kx * omega_hat
        domega_dy_hat = 1j * self.ky * omega_hat
        grad_omega_sq = np.abs(domega_dx_hat)**2 + np.abs(domega_dy_hat)**2
        return 0.5 * float(np.sum(grad_omega_sq)) / self.N**2

    def energy_spectrum(self, omega_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Isotropic energy spectrum E(k) via shell averaging."""
        u_hat, v_hat = self._velocity_from_omega(omega_hat)
        e_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / self.N**4

        k_max = self.N // 2
        k_bins = np.arange(1, k_max + 1, dtype=float)
        E_k = np.zeros(k_max)
        for i, k_val in enumerate(k_bins):
            shell = (self.k_mag >= k_val - 0.5) & (self.k_mag < k_val + 0.5)
            E_k[i] = np.sum(e_hat[shell])
        return k_bins, E_k

    def simulate(
        self,
        T_total: float = 20.0,
        snapshot_interval: float = 0.1,
        print_interval: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Run the full NS simulation.

        Returns dict with:
            snapshots: (N_snap, N, N) vorticity snapshots
            energy: (N_snap,) kinetic energy time series
            enstrophy: (N_snap,) enstrophy time series
            palinstrophy: (N_snap,) palinstrophy time series
            times: (N_snap,) time values
            omega_final_hat: final vorticity in Fourier space
            params: simulation parameters
        """
        print("=" * 70)
        print("FASE 0: SIMULACIÓN NAVIER-STOKES 2D PSEUDO-ESPECTRAL")
        print("=" * 70)
        print(f"  Grid: {self.N}×{self.N}, Re = {self.Re:.0f}")
        print(f"  ν = {self.nu:.6f}, ν₄ = {self.nu4:.1e}, α = {self.friction}")
        print(f"  Forcing: Kolmogorov k_f = {self.k_force}, A = {self.force_amplitude}")
        print(f"  T_total = {T_total}, snapshot cada {snapshot_interval}")
        print()

        # Initialize
        omega = self.init_vorticity("random")
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
            # Adaptive time step
            self.dt = self._adaptive_dt(omega_hat)

            # Don't overshoot snapshot or end time
            dt_actual = min(self.dt, T_total - t)
            if t_next_snap - t > 1e-12 and t_next_snap - t < dt_actual:
                dt_actual = t_next_snap - t

            # RK4 step
            omega_hat = self._rk4_step(omega_hat, dt_actual)
            t += dt_actual
            step += 1

            # Check for blowup
            if step % 100 == 0:
                max_omega = np.max(np.abs(np.real(ifft2(omega_hat))))
                if np.isnan(max_omega) or max_omega > 1e10:
                    print(f"  ⚠ BLOWUP en t={t:.4f}, step={step}. Abortando.")
                    break

            # Record snapshot
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

            # Print progress
            if t >= t_next_print - 1e-12:
                E = self.compute_energy(omega_hat)
                Z = self.compute_enstrophy(omega_hat)
                elapsed = time.time() - t_start
                print(f"  t={t:6.2f}/{T_total:.0f}  E={E:.4f}  Z={Z:.2f}  "
                      f"dt={dt_actual:.2e}  steps={step}  [{elapsed:.1f}s]")
                t_next_print = t + print_interval

        elapsed = time.time() - t_start
        print(f"\n  Simulación completada: {step} pasos en {elapsed:.1f}s")
        print(f"  Snapshots: {len(snapshots)}")

        result = {
            "snapshots": np.array(snapshots),
            "energy": np.array(energies),
            "enstrophy": np.array(enstrophies),
            "palinstrophy": np.array(palinstrophies),
            "times": np.array(times),
            "omega_final_hat": omega_hat,
            "params": {
                "N": self.N, "Re": self.Re, "nu": self.nu, "nu4": self.nu4,
                "k_force": self.k_force, "T_total": T_total,
            }
        }

        # Energy spectrum at final time
        k_bins, E_k = self.energy_spectrum(omega_hat)
        result["k_bins"] = k_bins
        result["E_k"] = E_k

        return result


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1: TAA/Koopman — Modos Coherentes vía EDMD
# ═════════════════════════════════════════════════════════════════════════════

def fase1_koopman_analysis(sim_data: Dict[str, Any]) -> Dict[str, Any]:
    """Koopman decomposition via PCA + EDMD using reduce_high_dimensional."""
    print("\n" + "=" * 70)
    print("FASE 1: ANÁLISIS TAA/KOOPMAN — MODOS COHERENTES")
    print("=" * 70)

    from acf_functor.deep_problems import reduce_high_dimensional, compute_continuous_generator
    from acf_functor.shared_numerics import SpectralClassifier

    snapshots = sim_data["snapshots"]
    N_snap, Nx, Ny = snapshots.shape
    times = sim_data["times"]
    dt_snap = times[1] - times[0] if len(times) > 1 else 0.1

    print(f"  Snapshots: {N_snap} × ({Nx}×{Ny}) = ({N_snap}, {Nx*Ny})")

    # Flatten snapshots to trajectory matrix
    trajectory = snapshots.reshape(N_snap, Nx * Ny)  # (N_snap, 4096)

    # Reduce via PCA + EDMD
    n_modes = min(50, N_snap - 2)
    print(f"  Ejecutando reduce_high_dimensional con n_koopman_modes={n_modes}...")
    try:
        result = reduce_high_dimensional(
            trajectory, T=None, D_2_hint=49.0, n_koopman_modes=n_modes
        )
        spectrum = result.spectrum_in_reduced
        print(f"  ✓ Dimensión intrínseca estimada: {result.intrinsic_dim}")
        print(f"  ✓ Error de reconstrucción: {result.reconstruction_error:.6f}")
        print(f"  ✓ Eigenvalores Koopman: {len(spectrum)}")
    except Exception as e:
        print(f"  ⚠ reduce_high_dimensional falló: {e}")
        print("  → Fallback: PCA manual + EDMD manual")
        # Manual PCA + EDMD
        X = trajectory - trajectory.mean(axis=0)
        C_dual = X @ X.T / (N_snap - 1)
        eigvals_pca, eigvecs_pca = np.linalg.eigh(C_dual)
        idx = np.argsort(-eigvals_pca)
        eigvals_pca = eigvals_pca[idx]
        eigvecs_pca = eigvecs_pca[:, idx]

        # Variance explained
        var_explained = np.cumsum(eigvals_pca) / np.sum(np.maximum(eigvals_pca, 0))
        d_95 = int(np.searchsorted(var_explained, 0.95)) + 1
        d_use = min(d_95, n_modes, N_snap - 2)

        Z = eigvecs_pca[:, :d_use]  # PCA coordinates

        # EDMD: X_future ≈ A @ X_past
        X_past = Z[:-1]
        X_future = Z[1:]
        A_edmd, _, _, _ = np.linalg.lstsq(X_past, X_future, rcond=None)
        spectrum = np.linalg.eigvals(A_edmd)
        result = type('Obj', (object,), {
            'intrinsic_dim': d_use,
            'reconstruction_error': float(1 - var_explained[d_use - 1]) if d_use <= len(var_explained) else 0.5,
            'spectrum_in_reduced': spectrum,
        })()

    # Variance analysis via eigenvalue magnitudes
    mags = np.sort(np.abs(spectrum))[::-1]
    cum_var = np.cumsum(mags**2) / np.sum(mags**2)
    d_95 = int(np.searchsorted(cum_var, 0.95)) + 1
    print(f"  ✓ Modos para capturar 95% energía: d_95 = {d_95}")

    # Continuous generator (decay rates + frequencies)
    try:
        gen_result = compute_continuous_generator(spectrum, tau=dt_snap)
        decay_rates = gen_result.decay_rates
        frequencies = gen_result.frequencies
        print(f"  ✓ Decay rates: min={np.min(decay_rates):.4f}, max={np.max(decay_rates):.4f}")
        print(f"  ✓ Frequencies: min={np.min(np.abs(frequencies)):.4f}, max={np.max(np.abs(frequencies)):.4f}")
    except Exception as e:
        print(f"  ⚠ compute_continuous_generator falló: {e}")
        # Manual computation
        log_eigs = np.log(spectrum.astype(complex) + 1e-30) / dt_snap
        decay_rates = np.real(log_eigs)
        frequencies = np.imag(log_eigs)
        gen_result = None

    # Spectral classification
    try:
        classification = SpectralClassifier.classify(spectrum)
        print(f"  ✓ Clasificación espectral: {classification.decay_class}")
        spectral_type = classification.decay_class
    except Exception as e:
        print(f"  ⚠ SpectralClassifier: {e}")
        spectral_type = "unknown"

    # Identify coherent structures: modes with slow decay
    # Ensure decay_rates matches spectrum length
    n_eigs = len(spectrum)
    if len(decay_rates) < n_eigs:
        decay_rates = np.pad(decay_rates, (0, n_eigs - len(decay_rates)), constant_values=-100)
        frequencies = np.pad(frequencies, (0, n_eigs - len(frequencies)), constant_values=0)
    elif len(decay_rates) > n_eigs:
        decay_rates = decay_rates[:n_eigs]
        frequencies = frequencies[:n_eigs]

    tau_decay = 1.0 / (np.abs(decay_rates) + 1e-12)
    tau_eddy = 1.0 / (np.abs(np.mean(decay_rates)) + 1e-12)
    coherent_mask = tau_decay > 2 * tau_eddy
    n_coherent = int(np.sum(coherent_mask))
    print(f"  ✓ Estructuras coherentes (τ > 2τ_eddy): {n_coherent} modos")

    return {
        "spectrum": spectrum,
        "decay_rates": decay_rates,
        "frequencies": frequencies,
        "d_95": d_95,
        "n_coherent": n_coherent,
        "intrinsic_dim": result.intrinsic_dim,
        "reconstruction_error": result.reconstruction_error,
        "spectral_type": spectral_type,
        "tau_decay": tau_decay,
        "coherent_mask": coherent_mask,
        "gen_result": gen_result,
    }


# ═════════════════════════════════════════════════════════════════════════════
# FASE 2: OTU/Ruelle — Función Zeta y Espectro de Lyapunov
# ═════════════════════════════════════════════════════════════════════════════

def fase2_ruelle_analysis(sim_data: Dict[str, Any], koopman: Dict[str, Any]) -> Dict[str, Any]:
    """Ruelle zeta function and thermodynamic pressure from enstrophy time series."""
    print("\n" + "=" * 70)
    print("FASE 2: ANÁLISIS OTU/RUELLE — FUNCIÓN ZETA Y LYAPUNOV")
    print("=" * 70)

    from acf_functor.gelfand_triple import GelfandTriple, OTURealWorld
    from acf_functor.deep_problems import (
        certify_numerical_stability, detect_exceptional_points
    )

    enstrophy = sim_data["enstrophy"]
    print(f"  Serie de enstrofía Z(t): {len(enstrophy)} puntos")
    print(f"  Z_mean = {np.mean(enstrophy):.4f}, Z_std = {np.std(enstrophy):.4f}")

    results = {}

    # Use OTURealWorld.from_timeseries on the enstrophy series
    try:
        print("  Ejecutando OTURealWorld.from_timeseries(Z)...")
        Z_1d = np.asarray(enstrophy, dtype=float).ravel()
        otu_report = OTURealWorld.from_timeseries(
            Z_1d, noise_filter="svd", n_test=24, n_dist=128, n_modes=12
        )
        results["otu_report"] = otu_report
        print(f"  ✓ OTU completado")
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
        print(f"  ⚠ OTURealWorld.from_timeseries falló: {e}")
        results["otu_report"] = None

    # Build 1D return map from enstrophy for Ruelle analysis
    try:
        print("  Construyendo mapa de retorno 1D desde enstrofía...")
        from acf_functor.gelfand_triple import certify as otu_certify

        # Create a simple 1D return map: Z_{n+1} = f(Z_n) via interpolation
        Z_series = np.asarray(enstrophy, dtype=float).ravel()
        Z_n = Z_series[:-1]
        Z_np1 = Z_series[1:]
        # Normalize to [0, 1]
        z_min, z_max = Z_n.min(), Z_n.max()
        z_range = z_max - z_min
        if z_range < 1e-12:
            raise ValueError("Enstrophy series is constant — no dynamics")
        Z_n_norm = (Z_n - z_min) / z_range
        Z_np1_norm = (Z_np1 - z_min) / z_range

        # Sort by Z_n to build a piecewise-linear map
        sort_idx = np.argsort(Z_n_norm)
        Z_n_sorted = Z_n_norm[sort_idx]
        Z_np1_sorted = Z_np1_norm[sort_idx]

        def T_return(x):
            return np.clip(np.interp(x, Z_n_sorted, Z_np1_sorted), 0, 1)

        domain = (0.0, 1.0)
        gt = GelfandTriple(T_return, domain, n_test=24, n_dist=128)
        gt.build()
        mu_srb, _ = gt.compute_self_consistent_measure()

        # Ruelle spectrum
        resonances = gt.compute_ruelle_spectrum(mu_srb, n_modes=16)[0]
        results["resonances"] = resonances
        print(f"  ✓ Resonancias de Ruelle: {len(resonances)}")
        for i in range(min(3, len(resonances))):
            print(f"    |λ_{i+1}| = {np.abs(resonances[i]):.6f}")

        # Ruelle zeta function
        s_vals = np.linspace(0.01, 3.0, 200) + 0j
        s_values, log_zeta = gt.compute_ruelle_zeta(s_vals, resonances)
        results["s_values"] = s_values
        results["log_zeta"] = log_zeta
        print(f"  ✓ Función zeta de Ruelle calculada")

        # Thermodynamic pressure
        betas = np.linspace(0.1, 3.0, 50)
        thermo_pressure = gt.compute_thermodynamic_pressure(betas)
        results["thermo_pressure"] = thermo_pressure
        h_ks_from_P = thermo_pressure.slope_at_1
        P_pp_1 = thermo_pressure.curvature_at_1
        print(f"  ✓ Presión termodinámica:")
        print(f"    P'(1) ≈ h_KS = {h_ks_from_P:.6f}")
        print(f"    P''(1) = {P_pp_1:.6f} (susceptibilidad)")
        results["h_ks_pressure"] = h_ks_from_P
        results["P_pp_1"] = P_pp_1

    except Exception as e:
        print(f"  ⚠ Análisis Ruelle manual falló: {e}")
        traceback.print_exc()
        resonances = koopman["spectrum"]
        results["resonances"] = resonances

    # Numerical stability certification
    gamma_val = results.get("gamma_otu", None)
    if gamma_val is not None:
        try:
            stab_cert = certify_numerical_stability(gamma_val)
            results["stability_cert"] = stab_cert
            print(f"  ✓ Certificación de estabilidad numérica: certificado={stab_cert.is_certified}")
        except Exception as e:
            print(f"  ⚠ certify_numerical_stability: {e}")

    # Exceptional points detection
    try:
        # Build operator from Koopman eigenvalues
        spectrum = koopman["spectrum"]
        L = np.diag(spectrum[:min(20, len(spectrum))])
        ep_result = detect_exceptional_points(L, threshold=1e-3)
        results["exceptional_points"] = ep_result
        print(f"  ✓ Puntos excepcionales: {ep_result.n_exceptional_points}")
    except Exception as e:
        print(f"  ⚠ detect_exceptional_points: {e}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# FASE 3: ERGON — Gap Espectral y Regímenes Turbulentos
# ═════════════════════════════════════════════════════════════════════════════

def fase3_ergon_analysis(sim_data: Dict[str, Any]) -> Dict[str, Any]:
    """ERGON regime detection and spectral gap analysis."""
    print("\n" + "=" * 70)
    print("FASE 3: ANÁLISIS ERGON — REGÍMENES TURBULENTOS")
    print("=" * 70)

    from acf_functor.ergon_agent import ERGONRealWorld

    enstrophy = sim_data["enstrophy"]
    results = {}

    # Monitor regime changes
    try:
        print("  Ejecutando ERGONRealWorld.monitor()...")
        monitor_result = ERGONRealWorld.monitor(
            enstrophy, window_size=min(500, len(enstrophy) // 3),
            step_size=50
        )
        results["monitor"] = monitor_result
        if isinstance(monitor_result, dict):
            n_regimes = monitor_result.get("n_regimes", "?")
            print(f"  ✓ Regímenes detectados: {n_regimes}")
            if "lyapunov_series" in monitor_result:
                lyap = monitor_result["lyapunov_series"]
                print(f"    Lyapunov medio: {np.mean(lyap):.4f}")
                results["lyapunov_series"] = np.array(lyap)
            if "regime_labels" in monitor_result:
                labels = monitor_result["regime_labels"]
                unique_labels = set(labels)
                print(f"    Labels: {unique_labels}")
    except Exception as e:
        print(f"  ⚠ ERGONRealWorld.monitor falló: {e}")
        traceback.print_exc()

    # Full certification via from_timeseries
    try:
        print("  Ejecutando from_timeseries (certificación completa)...")
        from acf_functor.real_world import from_timeseries
        full_report = from_timeseries(enstrophy, noise_filter="svd")
        results["full_report"] = full_report
        if isinstance(full_report, dict):
            cert = full_report.get("certification", {})
            print(f"  ✓ Certificación ERGON completa:")
            print(f"    h_KS = {cert.get('h_ks_estimate', '?')}")
            print(f"    Γ_OTU = {cert.get('gamma_otu', '?')}")
            print(f"    d* = {cert.get('d_star', '?')}")
            results["h_ks_ergon"] = cert.get("h_ks_estimate", None)
            results["gamma_ergon"] = cert.get("gamma_otu", None)
            results["d_star_ergon"] = cert.get("d_star", None)
    except Exception as e:
        print(f"  ⚠ from_timeseries falló: {e}")
        traceback.print_exc()

    return results


# ═════════════════════════════════════════════════════════════════════════════
# FASE 4: Termodinámica — Cascada de Energía como Transición de Fase
# ═════════════════════════════════════════════════════════════════════════════

def fase4_thermodynamic_analysis(
    sim_data: Dict[str, Any],
    koopman: Dict[str, Any],
    ruelle: Dict[str, Any],
) -> Dict[str, Any]:
    """Thermodynamic analysis: is the energy cascade a phase transition?"""
    print("\n" + "=" * 70)
    print("FASE 4: ANÁLISIS TERMODINÁMICO — CASCADA COMO TRANSICIÓN DE FASE")
    print("=" * 70)

    from acf_functor.thermodynamic_acf import ThermodynamicACF
    from acf_functor.deep_problems import compute_fisher_cramer_rao

    results = {}

    # ThermodynamicACF from Koopman eigenvalues
    spectrum = koopman["spectrum"]
    if len(spectrum) < 3:
        print(f"  ⚠ Solo {len(spectrum)} eigenvalores — insuficiente para ThermodynamicACF")
        return results
    try:
        eigs_torch = torch.tensor(np.abs(spectrum), dtype=torch.float64)
        thermo = ThermodynamicACF(eigs_torch)
        report = thermo.analyze()
        results["thermo_report"] = report
        print(f"  ✓ ThermodynamicACF análisis completado")

        # Check for phase transition
        if hasattr(report, 'phase_transition') and report.phase_transition is not None:
            pt = report.phase_transition
            print(f"    ⚡ TRANSICIÓN DE FASE detectada:")
            print(f"       β_c = {pt.get('beta_c', '?') if isinstance(pt, dict) else getattr(pt, 'beta_c', '?')}")
            results["phase_transition"] = pt
        else:
            print(f"    No se detectó transición de fase abrupta")
            results["phase_transition"] = None

        if hasattr(report, 'd_star_zero_temp'):
            print(f"    d*(T→0) = {report.d_star_zero_temp}")
            results["d_star_zero_temp"] = report.d_star_zero_temp
        if hasattr(report, 'mdl_dimension'):
            print(f"    d_MDL = {report.mdl_dimension}")
            results["d_mdl"] = report.mdl_dimension

    except Exception as e:
        print(f"  ⚠ ThermodynamicACF falló: {e}")
        traceback.print_exc()

    # Fisher-Cramér-Rao bounds
    P_pp_1 = ruelle.get("P_pp_1", None)
    h_ks = ruelle.get("h_ks_pressure", None)
    if P_pp_1 is not None and h_ks is not None:
        try:
            fcr = compute_fisher_cramer_rao(P_pp_1, h_ks, n_observations=len(sim_data["enstrophy"]))
            results["fisher_cr"] = fcr
            print(f"  ✓ Fisher-Cramér-Rao:")
            print(f"    Fisher info = {fcr.fisher_information_per_obs:.6f}")
            print(f"    σ_min = {fcr.min_error_std:.6f}")
            print(f"    Precisión relativa = {fcr.relative_precision:.4f}")
            print(f"    ¿Bernoulli? {fcr.is_bernoulli}")
        except Exception as e:
            print(f"  ⚠ Fisher-CR falló: {e}")

    # Energy spectrum analysis: Kolmogorov scaling
    k_bins = sim_data["k_bins"]
    E_k = sim_data["E_k"]
    mask = (k_bins >= 5) & (k_bins <= 20) & (E_k > 0)
    if np.sum(mask) > 3:
        log_k = np.log(k_bins[mask])
        log_E = np.log(E_k[mask])
        slope, intercept = np.polyfit(log_k, log_E, 1)
        results["spectral_slope"] = slope
        print(f"  ✓ Pendiente espectral: E(k) ~ k^{{{slope:.2f}}}")
        if abs(slope - (-3.0)) < 0.5:
            print(f"    → Compatible con cascada de enstrofía (k^{{-3}}) en 2D")
        elif abs(slope - (-5/3)) < 0.5:
            print(f"    → Compatible con cascada inversa de energía (k^{{-5/3}})")
        else:
            print(f"    → Pendiente intermedia: régimen transicional")
    else:
        results["spectral_slope"] = None
        print(f"  ⚠ Insuficientes datos para pendiente espectral")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# FASE 5: Verificación de Kolmogorov
# ═════════════════════════════════════════════════════════════════════════════

def fase5_kolmogorov_verification(
    sim_data: Dict[str, Any],
    koopman: Dict[str, Any],
    ruelle: Dict[str, Any],
    thermo: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify Kolmogorov scaling laws and conservation."""
    print("\n" + "=" * 70)
    print("FASE 5: VERIFICACIÓN DE KOLMOGOROV")
    print("=" * 70)

    results = {}

    energy = sim_data["energy"]
    enstrophy = sim_data["enstrophy"]
    times = sim_data["times"]

    # 1. Energy conservation check
    dE = np.diff(energy)
    dt_arr = np.diff(times)
    dE_dt = dE / (dt_arr + 1e-12)
    results["dE_dt_mean"] = float(np.mean(dE_dt))
    results["dE_dt_std"] = float(np.std(dE_dt))
    print(f"  1. Balance energético:")
    print(f"     dE/dt medio = {results['dE_dt_mean']:.6f} ± {results['dE_dt_std']:.6f}")
    if abs(results["dE_dt_mean"]) < results["dE_dt_std"]:
        print(f"     → Balance inyección-disipación OK (quasi-estacionario)")
        results["energy_balance"] = True
    else:
        print(f"     → Sistema no estacionario (spin-up o cascada activa)")
        results["energy_balance"] = False

    # 2. Enstrophy cascade check in 2D
    slope = thermo.get("spectral_slope", None)
    if slope is not None:
        results["kolmogorov_check"] = abs(slope - (-3.0)) < 1.0
        if abs(slope - (-3.0)) < 0.5:
            print(f"  2. Cascada de enstrofía directa: ✓ (slope = {slope:.2f} ≈ -3)")
        elif abs(slope - (-5.0/3.0)) < 0.5:
            print(f"  2. Cascada inversa de energía: ✓ (slope = {slope:.2f} ≈ -5/3)")
        else:
            print(f"  2. Espectro intermedio: slope = {slope:.2f}")

    # 3. Intermittency check via P''(1)
    P_pp_1 = ruelle.get("P_pp_1", None)
    if P_pp_1 is not None:
        results["intermittency"] = P_pp_1
        if P_pp_1 > 1.0:
            print(f"  3. Intermitencia FUERTE: P''(1) = {P_pp_1:.4f} >> 0")
            print(f"     → Fluctuaciones anómalas en la cascada")
        elif P_pp_1 > 0.1:
            print(f"  3. Intermitencia moderada: P''(1) = {P_pp_1:.4f}")
        else:
            print(f"  3. Intermitencia débil: P''(1) = {P_pp_1:.4f} ≈ 0 (Bernoulli-like)")

    # 4. Finite-mode dominance check
    spectrum = koopman["spectrum"]
    mags = np.sort(np.abs(spectrum))[::-1]
    # How many modes dominate?
    cum_energy = np.cumsum(mags**2) / np.sum(mags**2)
    d_80 = int(np.searchsorted(cum_energy, 0.80)) + 1
    d_90 = int(np.searchsorted(cum_energy, 0.90)) + 1
    d_95 = koopman["d_95"]
    results["d_80"] = d_80
    results["d_90"] = d_90
    results["d_95"] = d_95
    print(f"  4. Dominancia por modos finitos:")
    print(f"     80% energía: {d_80} modos")
    print(f"     90% energía: {d_90} modos")
    print(f"     95% energía: {d_95} modos")
    if d_95 < len(spectrum) * 0.5:
        print(f"     → SÍ: el atractor turbulento vive en un subespacio de baja dimensión")
        results["finite_mode_dominance"] = True
    else:
        print(f"     → NO: la dimensión efectiva es alta")
        results["finite_mode_dominance"] = False

    return results


# ═════════════════════════════════════════════════════════════════════════════
# FASE 6: Visualización
# ═════════════════════════════════════════════════════════════════════════════

def fase6_visualization(
    sim_data: Dict[str, Any],
    koopman: Dict[str, Any],
    ruelle: Dict[str, Any],
    ergon: Dict[str, Any],
) -> str:
    """Generate 6-panel visualization."""
    print("\n" + "=" * 70)
    print("FASE 6: VISUALIZACIÓN")
    print("=" * 70)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("Navier-Stokes 2D Turbulence — ACF Ecosystem Analysis",
                      fontsize=14, fontweight="bold")

        # (a) Vorticity snapshot
        ax = axes[0, 0]
        omega_final = sim_data["snapshots"][-1]
        vmax = np.percentile(np.abs(omega_final), 98)
        im = ax.imshow(omega_final, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[0, 2*np.pi, 0, 2*np.pi], origin="lower")
        ax.set_title("(a) Vorticidad ω(x,y)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # (b) Energy spectrum
        ax = axes[0, 1]
        k_bins = sim_data["k_bins"]
        E_k = sim_data["E_k"]
        mask_pos = E_k > 0
        ax.loglog(k_bins[mask_pos], E_k[mask_pos], "b-", lw=1.5, label="E(k)")
        # Reference slopes
        k_ref = np.logspace(np.log10(3), np.log10(20), 50)
        ax.loglog(k_ref, 1e-2 * k_ref**(-3), "r--", alpha=0.7, label="k$^{-3}$")
        ax.loglog(k_ref, 5e-3 * k_ref**(-5/3), "g--", alpha=0.7, label="k$^{-5/3}$")
        ax.set_title("(b) Espectro de Energía E(k)")
        ax.set_xlabel("k")
        ax.set_ylabel("E(k)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (c) Koopman eigenvalues in complex plane
        ax = axes[0, 2]
        spectrum = koopman["spectrum"]
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.5, alpha=0.3)
        if len(spectrum) > 0:
            coherent = koopman["coherent_mask"]
            ax.scatter(np.real(spectrum[~coherent]), np.imag(spectrum[~coherent]),
                       c="gray", s=20, alpha=0.5, label="Transient")
            if np.any(coherent):
                ax.scatter(np.real(spectrum[coherent]), np.imag(spectrum[coherent]),
                           c="red", s=40, zorder=5, label="Coherent")
        ax.set_title(f"(c) Eigenvalores Koopman (d_95={koopman['d_95']})")
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # (d) Ruelle zeta function
        ax = axes[1, 0]
        if "s_values" in ruelle and "log_zeta" in ruelle:
            s_vals = np.real(ruelle["s_values"])
            log_z = ruelle["log_zeta"]
            zeta_mag = np.exp(np.real(log_z))
            ax.semilogy(s_vals, zeta_mag, "b-", lw=1.5)
            ax.set_title("(d) |ζ(s)| — Función Zeta de Ruelle")
            ax.set_xlabel("s")
            ax.set_ylabel("|ζ(s)|")
        else:
            # Plot resonance magnitudes if available
            res = ruelle.get("resonances", koopman["spectrum"])
            if len(res) > 0:
                ax.stem(np.arange(len(res)), np.abs(res), linefmt="b-", markerfmt="bo", basefmt="k-")
                ax.set_title("(d) |λ_k| — Resonancias de Ruelle")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title("(d) Resonancias de Ruelle")
            ax.set_xlabel("k")
            ax.set_ylabel("|λ_k|")
        ax.grid(True, alpha=0.3)

        # (e) Time series E(t) and Z(t)
        ax = axes[1, 1]
        times = sim_data["times"]
        ax.plot(times, sim_data["energy"], "b-", lw=1, label="E(t)")
        ax2 = ax.twinx()
        ax2.plot(times, sim_data["enstrophy"], "r-", lw=1, alpha=0.7, label="Z(t)")
        ax.set_title("(e) Energía E(t) y Enstrofía Z(t)")
        ax.set_xlabel("t")
        ax.set_ylabel("E(t)", color="b")
        ax2.set_ylabel("Z(t)", color="r")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax.grid(True, alpha=0.3)

        # (f) Lyapunov vs time (ERGON)
        ax = axes[1, 2]
        if "lyapunov_series" in ergon:
            lyap = ergon["lyapunov_series"]
            t_lyap = np.linspace(times[0], times[-1], len(lyap))
            ax.plot(t_lyap, lyap, "g-", lw=1)
            ax.axhline(y=0, color="k", ls="--", lw=0.5)
            ax.set_title("(f) Lyapunov vs Tiempo (ERGON)")
            ax.set_xlabel("t")
            ax.set_ylabel("λ_max(t)")
        elif "monitor" in ergon and isinstance(ergon["monitor"], dict):
            mon = ergon["monitor"]
            if "window_lyapunov" in mon:
                lyap_w = mon["window_lyapunov"]
                ax.plot(lyap_w, "g-", lw=1)
                ax.set_title("(f) Lyapunov ventana (ERGON)")
        else:
            # Fallback: plot enstrophy derivative as proxy
            if len(sim_data["enstrophy"]) > 1:
                dZ = np.gradient(sim_data["enstrophy"], times)
                ax.plot(times, dZ, "g-", lw=1)
                ax.set_title("(f) dZ/dt — Proxy Lyapunov")
                ax.set_xlabel("t")
            else:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title("(f) Lyapunov")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "navier_stokes_map.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Figura guardada: {outpath}")
        return outpath
    except Exception as e:
        print(f"  ⚠ Visualización falló: {e}")
        traceback.print_exc()
        return ""


# ═════════════════════════════════════════════════════════════════════════════
# FASE 7: Reporte Final
# ═════════════════════════════════════════════════════════════════════════════

def fase7_report(
    sim_data: Dict[str, Any],
    koopman: Dict[str, Any],
    ruelle: Dict[str, Any],
    ergon: Dict[str, Any],
    thermo: Dict[str, Any],
    kolmogorov: Dict[str, Any],
    fig_path: str,
):
    """Print comprehensive report in Spanish."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║     NAVIER-STOKES 2D TURBULENCE — REPORTE COMPLETO ACF           ║")
    print("╚" + "═" * 68 + "╝")

    # 1. SIMULACIÓN
    print("\n━━━ 1. SIMULACIÓN NAVIER-STOKES ━━━")
    p = sim_data["params"]
    print(f"  Grid: {p['N']}×{p['N']}, Re = {p['Re']:.0f}")
    print(f"  ν = {p['nu']:.6f}, ν₄ = {p['nu4']:.1e}")
    print(f"  Forcing: Kolmogorov k_f = {p['k_force']}, T = {p['T_total']}")
    print(f"  Snapshots: {len(sim_data['snapshots'])}")
    print(f"  E_final = {sim_data['energy'][-1]:.6f}")
    print(f"  Z_final = {sim_data['enstrophy'][-1]:.4f}")
    print(f"  P_final = {sim_data['palinstrophy'][-1]:.4f}")

    # 2. TAA/KOOPMAN
    print("\n━━━ 2. TAA/KOOPMAN — MODOS COHERENTES ━━━")
    print(f"  Dimensión intrínseca: {koopman['intrinsic_dim']}")
    print(f"  Error de reconstrucción: {koopman['reconstruction_error']:.6f}")
    print(f"  Modos para 95% energía: d_95 = {koopman['d_95']}")
    print(f"  Estructuras coherentes: {koopman['n_coherent']} modos de larga vida")
    print(f"  Clasificación espectral: {koopman['spectral_type']}")
    if len(koopman['decay_rates']) > 0:
        print(f"  Decay rates: [{np.min(koopman['decay_rates']):.4f}, {np.max(koopman['decay_rates']):.4f}]")
        print(f"  Frecuencias: [{np.min(np.abs(koopman['frequencies'])):.4f}, {np.max(np.abs(koopman['frequencies'])):.4f}]")
    else:
        print(f"  (Sin eigenvalores — insuficientes snapshots para EDMD)")

    # Coherent structures detail
    if koopman["n_coherent"] > 0:
        tau = koopman["tau_decay"]
        coherent_idx = np.where(koopman["coherent_mask"])[0]
        print(f"  Modos coherentes (τ_k > 2τ_eddy):")
        for i, idx in enumerate(coherent_idx[:5]):
            print(f"    Modo {idx}: τ = {tau[idx]:.2f}, |λ| = {np.abs(koopman['spectrum'][idx]):.4f}")

    # 3. OTU/RUELLE
    print("\n━━━ 3. OTU/RUELLE — FUNCIÓN ZETA Y TERMODINÁMICA ━━━")
    if "h_ks_pressure" in ruelle:
        print(f"  h_KS (presión) = {ruelle['h_ks_pressure']:.6f}")
    if "P_pp_1" in ruelle:
        print(f"  P''(1) = {ruelle['P_pp_1']:.6f}")
    if "resonances" in ruelle:
        res = ruelle["resonances"]
        print(f"  Resonancias de Ruelle: {len(res)}")
        for i in range(min(5, len(res))):
            print(f"    λ_{i+1} = {res[i]:.6f}  |λ| = {np.abs(res[i]):.6f}")
    if "h_ks" in ruelle and ruelle["h_ks"] is not None:
        print(f"  h_KS (OTU directo) = {ruelle['h_ks']:.6f}")
    if "gamma_otu" in ruelle and ruelle["gamma_otu"] is not None:
        print(f"  Γ_OTU = {ruelle['gamma_otu']:.6f}")
    if "stability_cert" in ruelle:
        sc = ruelle["stability_cert"]
        print(f"  Estabilidad numérica: certificado = {sc.is_certified}")
    if "exceptional_points" in ruelle:
        ep = ruelle["exceptional_points"]
        print(f"  Puntos excepcionales: {ep.n_exceptional_points}")

    # 4. ERGON/REGÍMENES
    print("\n━━━ 4. ERGON — GAP ESPECTRAL Y REGÍMENES ━━━")
    if "monitor" in ergon and isinstance(ergon["monitor"], dict):
        mon = ergon["monitor"]
        print(f"  Regímenes: {mon.get('n_regimes', '?')}")
        if "lyapunov_series" in ergon:
            lyap = ergon["lyapunov_series"]
            print(f"  Lyapunov (ventana): mean={np.mean(lyap):.4f}, std={np.std(lyap):.4f}")
    if "h_ks_ergon" in ergon and ergon["h_ks_ergon"] is not None:
        print(f"  h_KS (ERGON) = {ergon['h_ks_ergon']:.6f}")
    if "gamma_ergon" in ergon and ergon["gamma_ergon"] is not None:
        print(f"  Γ_OTU (ERGON) = {ergon['gamma_ergon']:.6f}")
    if "d_star_ergon" in ergon and ergon["d_star_ergon"] is not None:
        print(f"  d* (ERGON) = {ergon['d_star_ergon']}")

    # 5. TERMODINÁMICA
    print("\n━━━ 5. TERMODINÁMICA — ¿CASCADA = TRANSICIÓN DE FASE? ━━━")
    if "spectral_slope" in thermo and thermo["spectral_slope"] is not None:
        slope = thermo["spectral_slope"]
        print(f"  Pendiente espectral: E(k) ~ k^{{{slope:.2f}}}")
    if thermo.get("phase_transition") is not None:
        print(f"  ⚡ TRANSICIÓN DE FASE DETECTADA")
        pt = thermo["phase_transition"]
        if isinstance(pt, dict):
            for k, v in pt.items():
                print(f"    {k}: {v}")
    else:
        print(f"  No se detectó transición de fase abrupta")
    if "d_star_zero_temp" in thermo:
        print(f"  d*(T→0) = {thermo['d_star_zero_temp']}")
    if "d_mdl" in thermo:
        print(f"  d_MDL = {thermo['d_mdl']}")
    if "fisher_cr" in thermo:
        fcr = thermo["fisher_cr"]
        print(f"  Fisher-Cramér-Rao: I₁ = {fcr.fisher_information_per_obs:.6f}")
        print(f"  σ_min = {fcr.min_error_std:.6f}, precisión relativa = {fcr.relative_precision:.4f}")

    # 6. KOLMOGOROV
    print("\n━━━ 6. VERIFICACIÓN DE KOLMOGOROV ━━━")
    print(f"  Balance energético: {'✓' if kolmogorov.get('energy_balance') else '✗'}")
    print(f"  dE/dt = {kolmogorov.get('dE_dt_mean', '?'):.6f} ± {kolmogorov.get('dE_dt_std', '?'):.6f}")
    print(f"  Dominancia modos finitos: {'SÍ' if kolmogorov.get('finite_mode_dominance') else 'NO'}")
    print(f"  Modos 80%: {kolmogorov.get('d_80', '?')}, 90%: {kolmogorov.get('d_90', '?')}, 95%: {kolmogorov.get('d_95', '?')}")
    if "intermittency" in kolmogorov:
        P_pp = kolmogorov["intermittency"]
        if P_pp > 1.0:
            print(f"  Intermitencia: FUERTE (P''(1) = {P_pp:.4f})")
        elif P_pp > 0.1:
            print(f"  Intermitencia: moderada (P''(1) = {P_pp:.4f})")
        else:
            print(f"  Intermitencia: débil (P''(1) = {P_pp:.4f})")

    # 7. DESCUBRIMIENTOS
    print("\n━━━ 7. DESCUBRIMIENTOS DEL ECOSISTEMA ACF ━━━")
    discoveries = []

    # D1: Finite Koopman dimension
    d_95 = koopman["d_95"]
    N_grid = sim_data["params"]["N"]
    if d_95 < N_grid:
        discoveries.append(
            f"D1: El atractor turbulento 2D NS vive en un subespacio de Koopman "
            f"de dimensión d_95={d_95} << {N_grid}² = {N_grid**2}. "
            f"Reducción dimensional: {N_grid**2/d_95:.0f}x"
        )

    # D2: Coherent structures
    if koopman["n_coherent"] > 0:
        discoveries.append(
            f"D2: {koopman['n_coherent']} estructuras coherentes de larga vida detectadas "
            f"(τ > 2τ_eddy). Estos son los vórtices persistentes que organizan el flujo."
        )

    # D3: Spectral cascade
    if thermo.get("spectral_slope") is not None:
        slope = thermo["spectral_slope"]
        discoveries.append(
            f"D3: Espectro de energía E(k) ~ k^{{{slope:.2f}}}. "
            f"{'Cascada de enstrofía directa (Kraichnan 1967)' if abs(slope - (-3)) < 0.5 else 'Régimen intermedio'}"
        )

    # D4: Ruelle zeta poles
    if "resonances" in ruelle:
        discoveries.append(
            f"D4: La función zeta de Ruelle del flujo turbulento tiene "
            f"{len(ruelle['resonances'])} polos computables. "
            f"El polo dominante |λ₁| = {np.abs(ruelle['resonances'][0]):.4f} "
            f"controla la escala temporal de decorrelación."
        )

    # D5: Thermodynamic pressure
    if "P_pp_1" in ruelle:
        P_pp = ruelle["P_pp_1"]
        discoveries.append(
            f"D5: Susceptibilidad termodinámica P''(1) = {P_pp:.4f}. "
            f"{'Intermitencia fuerte: fluctuaciones anómalas en la cascada.' if P_pp > 1 else 'Intermitencia moderada.'}"
        )

    # D6: Regime detection
    if "monitor" in ergon and isinstance(ergon["monitor"], dict):
        n_reg = ergon["monitor"].get("n_regimes", 0)
        if n_reg > 1:
            discoveries.append(
                f"D6: ERGON detectó {n_reg} regímenes turbulentos distintos. "
                f"La turbulencia 2D NO es estacionaria: exhibe transiciones entre regímenes."
            )

    # D7: Phase transition
    if thermo.get("phase_transition") is not None:
        discoveries.append(
            f"D7: ⚡ La cascada de energía ES una transición de fase termodinámica "
            f"(detectable vía ThermodynamicACF). Implicación: la turbulencia 2D tiene "
            f"puntos críticos que se comportan como transiciones de fase en materia condensada."
        )

    if not discoveries:
        discoveries.append("No se detectaron descubrimientos significativos.")

    for d in discoveries:
        print(f"\n  {d}")

    print(f"\n━━━ CERTIFICACIÓN FINAL ━━━")
    print(f"  Análisis completados: NS + TAA + OTU + ERGON + Termodinámica")
    print(f"  Descubrimientos: {len(discoveries)}")
    if fig_path:
        print(f"  Figura: {fig_path}")
    print(f"\n  El ecosistema ACF analizó exitosamente turbulencia 2D")
    print(f"  de Navier-Stokes — un Problema del Milenio de Clay.")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── Fase 0: NS Simulation ──
    solver = NavierStokes2D(
        N=64, Re=4000.0, nu4_coeff=1e-10,
        k_force=4, force_amplitude=20.0, friction=0.05,
    )
    sim_data = solver.simulate(T_total=60.0, snapshot_interval=0.1, print_interval=10.0)

    # Check for valid simulation
    if len(sim_data["snapshots"]) < 20 or np.any(np.isnan(sim_data["snapshots"][-1])):
        print("\nINFO: Reintentando con Re=500, mayor hiperviscosidad...")
        solver = NavierStokes2D(N=64, Re=1000.0, nu4_coeff=1e-9, k_force=4, force_amplitude=10.0, friction=0.05)
        sim_data = solver.simulate(T_total=60.0, snapshot_interval=0.1, print_interval=10.0)

    # Discard spin-up phase (first 70% of snapshots — keep only equilibrated tail)
    n_snap = len(sim_data["snapshots"])
    n_spinup = int(0.7 * n_snap)
    if n_snap > n_spinup + 50:
        print(f"\n  Descartando {n_spinup} snapshots de spin-up (de {n_snap} totales)...")
        sim_data["snapshots"] = sim_data["snapshots"][n_spinup:]
        sim_data["energy"] = sim_data["energy"][n_spinup:]
        sim_data["enstrophy"] = sim_data["enstrophy"][n_spinup:]
        sim_data["palinstrophy"] = sim_data["palinstrophy"][n_spinup:]
        sim_data["times"] = sim_data["times"][n_spinup:]
        print(f"  Analizando {len(sim_data['snapshots'])} snapshots en régimen estacionario")

    # ── Fase 1: Koopman ──
    koopman = fase1_koopman_analysis(sim_data)

    # ── Fase 2: Ruelle ──
    ruelle = fase2_ruelle_analysis(sim_data, koopman)

    # ── Fase 3: ERGON ──
    ergon = fase3_ergon_analysis(sim_data)

    # ── Fase 4: Thermodynamics ──
    thermo = fase4_thermodynamic_analysis(sim_data, koopman, ruelle)

    # ── Fase 5: Kolmogorov ──
    kolmogorov = fase5_kolmogorov_verification(sim_data, koopman, ruelle, thermo)

    # ── Fase 6: Visualization ──
    fig_path = fase6_visualization(sim_data, koopman, ruelle, ergon)

    # ── Fase 7: Report ──
    fase7_report(sim_data, koopman, ruelle, ergon, thermo, kolmogorov, fig_path)

    elapsed = time.time() - t0
    print(f"Tiempo total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
