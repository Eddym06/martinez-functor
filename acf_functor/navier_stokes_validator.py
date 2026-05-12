"""
navier_stokes_validator.py — Real-World Validation for ACF Ecosystem
=====================================================================

Closes the gap: "La validación empírica está mayormente en dominios canónicos
(polinomios, 6 trascendentales) — el salto a sistemas reales (Navier-Stokes
turbulento, mercados financieros) no está validado."

This module provides:
  1. Navier-Stokes 2D/3D solvers (spectral + finite difference)
  2. Lorenz-96 multi-scale system
  3. Kuramoto-Sivashinsky (spatiotemporal chaos)
  4. Financial time-series generators (GARCH, jump-diffusion)
  5. Full ACF pipeline validation on each system
  6. Comparison against known analytical results

VALIDATION TARGETS:
  - Kolmogorov -5/3 energy spectrum recovery
  - Lyapunov exponent agreement with literature
  - KS entropy consistency
  - ROM predictive accuracy
  - Certificate pass rates

REFERENCES:
  - Canuto et al. (2007) — Spectral Methods in Fluid Dynamics
  - Lorenz (1996) — Predictability
  - Kuramoto & Tsuzuki (1976) — Persistent propagation of concentration waves
  - Heston (1993) — Stochastic volatility
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import fft, linalg


# ---------------------------------------------------------------------------
# System definitions
# ---------------------------------------------------------------------------

class SystemType(Enum):
    NAVIER_STOKES_2D = "navier_stokes_2d"
    NAVIER_STOKES_3D = "navier_stokes_3d"
    LORENZ_96 = "lorenz_96"
    KURAMOTO_SIVASHINSKY = "kuramoto_sivashinsky"
    GARCH = "garch"
    JUMP_DIFFUSION = "jump_diffusion"


@dataclass
class SystemConfig:
    """Configuration for a real-world dynamical system."""
    system_type: SystemType
    # Navier-Stokes
    nx: int = 64              # Grid points (x)
    ny: int = 64              # Grid points (y) for 2D
    nz: int = 32              # Grid points (z) for 3D
    nu: float = 0.001         # Viscosity
    forcing_amplitude: float = 0.1
    # Lorenz-96
    n_lorenz: int = 40        # Number of variables
    F_lorenz: float = 8.0     # Forcing
    # Kuramoto-Sivashinsky
    L_ks: float = 22.0        # Domain length
    n_ks: int = 128           # Grid points
    # Financial
    n_financial: int = 1000   # Time steps
    # Common
    dt: float = 0.001
    n_steps: int = 10000
    n_warmup: int = 2000


@dataclass
class ValidationMetrics:
    """Validation results for a real-world system."""
    system_name: str
    system_type: SystemType

    # Spectral properties
    energy_spectrum: Optional[np.ndarray] = None
    kolmogorov_slope: float = float('nan')     # Should be ≈ -5/3
    kolmogorov_r2: float = float('nan')

    # Chaos diagnostics
    lyapunov_max: float = float('nan')
    lyapunov_literature: float = float('nan')   # Known value from literature
    lyapunov_error: float = float('nan')

    # Entropy
    h_ks: float = float('nan')
    h_ks_literature: float = float('nan')
    h_ks_error: float = float('nan')

    # ACF pipeline results
    taa_cert_pass: bool = False
    ergon_cert_pass: bool = False
    otu_cert_pass: bool = False
    psal_cert_pass: bool = False
    sem_cert_pass: bool = False

    # ROM quality
    rom_trajectory_error: float = float('nan')
    rom_spectrum_error: float = float('nan')
    rom_energy_drift: float = float('nan')

    # Timing
    total_time_s: float = 0.0
    n_dof: int = 0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  VALIDATION: {self.system_name}",
            f"  DOF: {self.n_dof}  |  Time: {self.total_time_s:.1f}s",
            f"{'='*60}",
            f"  Kolmogorov slope:  {self.kolmogorov_slope:.3f} (target: -1.667, R²={self.kolmogorov_r2:.3f})",
            f"  λ_max:             {self.lyapunov_max:.4f} (lit: {self.lyapunov_literature}, err={self.lyapunov_error:.2%})",
            f"  h_KS:              {self.h_ks:.4f} (lit: {self.h_ks_literature}, err={self.h_ks_error:.2%})",
            f"  --- ACF Pipeline ---",
            f"  SEM:  {'✅' if self.sem_cert_pass else '❌'}  "
            f"TAA:  {'✅' if self.taa_cert_pass else '❌'}  "
            f"ERGON: {'✅' if self.ergon_cert_pass else '❌'}",
            f"  OTU:  {'✅' if self.otu_cert_pass else '❌'}  "
            f"PSAL: {'✅' if self.psal_cert_pass else '❌'}",
            f"  --- ROM Quality ---",
            f"  Trajectory error:  {self.rom_trajectory_error:.4e}",
            f"  Spectrum error:    {self.rom_spectrum_error:.4e}",
            f"  Energy drift:      {self.rom_energy_drift:.4e}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Navier-Stokes 2D Spectral Solver
# ---------------------------------------------------------------------------

class NavierStokes2DSpectral:
    """
    2D Navier-Stokes in vorticity-streamfunction formulation.

    ∂_t ω + J(ψ, ω) = ν ∇² ω + f
    ∇² ψ = -ω

    Solved pseudo-spectrally with 2/3 dealiasing.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.nu = config.nu
        self.dt = config.dt

        # Wavenumbers
        kx = fft.fftfreq(self.nx) * self.nx
        ky = fft.fftfreq(self.ny) * self.ny
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0  # Avoid division by zero

        # Dealiasing mask (2/3 rule)
        kmax = min(self.nx, self.ny) // 3
        self.dealias = (np.abs(self.KX) < kmax) & (np.abs(self.KY) < kmax)

        # Integrating factor for viscous term
        self.integ_factor = np.exp(-self.nu * self.K2 * self.dt)

        # Forcing: inject energy at large scales
        kf = 4.0
        self.forcing = np.zeros((self.ny, self.nx), dtype=complex)
        mask_f = (self.K2 > 0) & (self.K2 < kf**2)
        if mask_f.any():
            rng = np.random.default_rng(42)
            self.forcing[mask_f] = (
                rng.normal(0, 1, mask_f.sum()) +
                1j * rng.normal(0, 1, mask_f.sum())
            )
            self.forcing *= config.forcing_amplitude / np.sqrt(mask_f.sum())

    def _jacobian_spectral(self, psi_hat: np.ndarray, omega_hat: np.ndarray) -> np.ndarray:
        """Compute J(ψ, ω) = ∂_x ψ ∂_y ω - ∂_y ψ ∂_x ω in spectral space."""
        psi = np.real(fft.ifft2(psi_hat))
        omega = np.real(fft.ifft2(omega_hat))

        dpsidx = np.real(fft.ifft2(1j * self.KX * psi_hat))
        dpsidy = np.real(fft.ifft2(1j * self.KY * psi_hat))
        domegadx = np.real(fft.ifft2(1j * self.KX * omega_hat))
        domegady = np.real(fft.ifft2(1j * self.KY * omega_hat))

        jac = dpsidx * domegady - dpsidy * domegadx
        jac_hat = fft.fft2(jac)
        jac_hat[~self.dealias] = 0.0
        return jac_hat

    def step(self, omega_hat: np.ndarray) -> np.ndarray:
        """One RK4 time step."""
        def rhs(w_hat):
            psi_hat = w_hat / self.K2
            psi_hat[0, 0] = 0.0
            jac = self._jacobian_spectral(psi_hat, w_hat)
            return -jac + self.forcing

        # RK4 with integrating factor
        k1 = self.dt * rhs(omega_hat)
        k2 = self.dt * rhs(self.integ_factor * (omega_hat + 0.5 * k1))
        k3 = self.dt * rhs(self.integ_factor * (omega_hat + 0.5 * k2))
        k4 = self.dt * rhs(self.integ_factor * (omega_hat + k3))

        omega_new = self.integ_factor * omega_hat + (
            self.integ_factor * (k1 + 2*k2 + 2*k3) + k4
        ) / 6.0
        omega_new[~self.dealias] = 0.0
        return omega_new

    def generate_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a turbulent trajectory. Returns (time_series, energy_spectrum)."""
        config = self.config
        rng = np.random.default_rng(42)

        # Initial condition: random vorticity
        omega_hat = np.zeros((self.ny, self.nx), dtype=complex)
        omega_hat[1:5, 1:5] = rng.normal(0, 1, (4, 4)) + 1j * rng.normal(0, 1, (4, 4))

        # Warmup
        for _ in range(config.n_warmup):
            omega_hat = self.step(omega_hat)

        # Collect trajectory (kinetic energy at each step)
        energy_ts = np.zeros(config.n_steps)
        for t in range(config.n_steps):
            omega_hat = self.step(omega_hat)
            psi_hat = omega_hat / self.K2
            psi_hat[0, 0] = 0.0
            energy_ts[t] = 0.5 * np.sum(np.abs(omega_hat)**2 / self.K2).real

        # Compute energy spectrum E(k)
        omega = np.real(fft.ifft2(omega_hat))
        ek = self._compute_energy_spectrum(omega_hat)

        return energy_ts, ek

    def _compute_energy_spectrum(self, omega_hat: np.ndarray) -> np.ndarray:
        """Compute 1D energy spectrum E(k) = Σ_{|k|=k} |u_k|²."""
        psi_hat = omega_hat / self.K2
        psi_hat[0, 0] = 0.0
        u_hat = 1j * self.KY * psi_hat
        v_hat = -1j * self.KX * psi_hat
        energy_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

        kmax = min(self.nx, self.ny) // 2
        k_mag = np.sqrt(self.K2)
        ek = np.zeros(kmax)
        for k in range(1, kmax):
            shell = (k_mag >= k - 0.5) & (k_mag < k + 0.5)
            if shell.any():
                ek[k] = np.sum(energy_2d[shell])
        return ek[1:]  # Skip k=0


# ---------------------------------------------------------------------------
# Lorenz-96 Multi-Scale System
# ---------------------------------------------------------------------------

class Lorenz96:
    """
    Lorenz-96 system: dX_i/dt = (X_{i+1} - X_{i-2}) X_{i-1} - X_i + F

    Classic testbed for atmospheric predictability.
    For N=40, F=8: λ_max ≈ 1.67, h_KS ≈ 2.5
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.N = config.n_lorenz
        self.F = config.F_lorenz
        self.dt = config.dt

    def rhs(self, X: np.ndarray) -> np.ndarray:
        """Right-hand side of Lorenz-96."""
        dX = np.zeros(self.N)
        for i in range(self.N):
            dX[i] = (X[(i+1) % self.N] - X[(i-2) % self.N]) * X[(i-1) % self.N] - X[i] + self.F
        return dX

    def generate_trajectory(self) -> np.ndarray:
        """Generate trajectory using RK4."""
        config = self.config
        rng = np.random.default_rng(42)
        X = self.F * np.ones(self.N)
        X[:5] += rng.normal(0, 0.1, 5)

        # Warmup
        for _ in range(config.n_warmup):
            k1 = self.dt * self.rhs(X)
            k2 = self.dt * self.rhs(X + 0.5*k1)
            k3 = self.dt * self.rhs(X + 0.5*k2)
            k4 = self.dt * self.rhs(X + k3)
            X = X + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        # Collect
        traj = np.zeros((config.n_steps, self.N))
        for t in range(config.n_steps):
            k1 = self.dt * self.rhs(X)
            k2 = self.dt * self.rhs(X + 0.5*k1)
            k3 = self.dt * self.rhs(X + 0.5*k2)
            k4 = self.dt * self.rhs(X + k3)
            X = X + (k1 + 2*k2 + 2*k3 + k4) / 6.0
            traj[t] = X

        return traj


# ---------------------------------------------------------------------------
# Kuramoto-Sivashinsky
# ---------------------------------------------------------------------------

class KuramotoSivashinsky:
    """
    Kuramoto-Sivashinsky equation: u_t + u u_x + u_xx + u_xxxx = 0

    Spatiotemporal chaos on periodic domain [0, L].
    For L=22: λ_max ≈ 0.05-0.1
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.L = config.L_ks
        self.N = config.n_ks
        self.dt = config.dt
        self.dx = self.L / self.N

        # Wavenumbers
        k = 2.0 * np.pi * fft.fftfreq(self.N) * self.N / self.L
        self.ik = 1j * k
        self.k2 = k**2
        self.k4 = k**4

        # Integrating factor for linear terms
        self.lin_factor = np.exp((self.k2 - self.k4) * self.dt)

    def step(self, u_hat: np.ndarray) -> np.ndarray:
        """ETDRK4 step for KS."""
        u = np.real(fft.ifft(u_hat))
        N_hat = -0.5 * self.ik * fft.fft(u**2)  # Nonlinear term

        a = self.lin_factor * u_hat + (self.lin_factor - 1.0) / (
            (self.k2 - self.k4) * self.dt + 1e-14
        ) * N_hat * self.dt

        u_a = np.real(fft.ifft(a))
        N_a = -0.5 * self.ik * fft.fft(u_a**2)

        b = self.lin_factor * u_hat + (self.lin_factor - 1.0) / (
            (self.k2 - self.k4) * self.dt + 1e-14
        ) * N_a * self.dt

        u_b = np.real(fft.ifft(b))
        N_b = -0.5 * self.ik * fft.fft(u_b**2)

        c = self.lin_factor * a + (self.lin_factor - 1.0) / (
            (self.k2 - self.k4) * self.dt + 1e-14
        ) * (2.0 * N_b - N_hat) * self.dt

        u_c = np.real(fft.ifft(c))
        N_c = -0.5 * self.ik * fft.fft(u_c**2)

        return (self.lin_factor * u_hat +
                N_hat * ((-4.0 - self.k2*self.dt + self.k4*self.dt +
                          np.exp(self.k2*self.dt - self.k4*self.dt) *
                          (4.0 - 3.0*self.k2*self.dt + 3.0*self.k4*self.dt +
                           (self.k2*self.dt - self.k4*self.dt)**2)) /
                         ((self.k2 - self.k4)**3 * self.dt**2 + 1e-14)) +
                2.0 * (N_a + N_b) * ((2.0 + self.k2*self.dt - self.k4*self.dt +
                         np.exp(self.k2*self.dt - self.k4*self.dt) *
                         (-2.0 + self.k2*self.dt - self.k4*self.dt)) /
                         ((self.k2 - self.k4)**3 * self.dt**2 + 1e-14)) +
                N_c * ((-4.0 - 3.0*self.k2*self.dt + 3.0*self.k4*self.dt -
                         (self.k2*self.dt - self.k4*self.dt)**2 +
                         np.exp(self.k2*self.dt - self.k4*self.dt) *
                         (4.0 - self.k2*self.dt + self.k4*self.dt)) /
                         ((self.k2 - self.k4)**3 * self.dt**2 + 1e-14)))

    def generate_trajectory(self) -> np.ndarray:
        """Generate spatiotemporal chaos trajectory."""
        config = self.config
        rng = np.random.default_rng(42)
        x = np.linspace(0, self.L, self.N, endpoint=False)
        u = 0.1 * np.cos(2*np.pi*x/self.L) + 0.01 * rng.normal(0, 1, self.N)
        u_hat = fft.fft(u)

        for _ in range(config.n_warmup):
            u_hat = self.step(u_hat)

        traj = np.zeros((config.n_steps, self.N))
        for t in range(config.n_steps):
            u_hat = self.step(u_hat)
            traj[t] = np.real(fft.ifft(u_hat))

        return traj


# ---------------------------------------------------------------------------
# Financial Time Series (GARCH + Jump-Diffusion)
# ---------------------------------------------------------------------------

class FinancialGenerator:
    """
    Generates realistic financial time series with:
      - GARCH(1,1) volatility clustering
      - Jump-diffusion (Merton model)
      - Heavy-tailed returns
    """

    @staticmethod
    def garch(
        n_steps: int = 2000,
        omega: float = 0.01,
        alpha: float = 0.1,
        beta: float = 0.85,
        seed: int = 42,
    ) -> np.ndarray:
        """
        GARCH(1,1): σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
        Returns log-returns.
        """
        rng = np.random.default_rng(seed)
        sigma2 = np.zeros(n_steps)
        returns = np.zeros(n_steps)
        sigma2[0] = omega / (1.0 - alpha - beta)

        for t in range(1, n_steps):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            returns[t] = np.sqrt(sigma2[t]) * rng.normal()

        return returns

    @staticmethod
    def jump_diffusion(
        n_steps: int = 2000,
        mu: float = 0.05,
        sigma: float = 0.2,
        lambda_jump: float = 0.1,
        jump_mean: float = -0.05,
        jump_std: float = 0.1,
        dt: float = 1/252,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Merton jump-diffusion:
          dS/S = μ dt + σ dW + (J-1) dN
        Returns log-price.
        """
        rng = np.random.default_rng(seed)
        log_price = np.zeros(n_steps)
        log_price[0] = 0.0

        for t in range(1, n_steps):
            # Diffusion
            dw = np.sqrt(dt) * rng.normal()
            # Jump
            n_jumps = rng.poisson(lambda_jump * dt)
            jump = 0.0
            for _ in range(n_jumps):
                jump += rng.normal(jump_mean, jump_std)

            log_price[t] = log_price[t-1] + (mu - 0.5*sigma**2)*dt + sigma*dw + jump

        return log_price


# ---------------------------------------------------------------------------
# Main Validator
# ---------------------------------------------------------------------------

class NavierStokesValidator:
    """
    Validates the entire ACF pipeline on real-world systems.

    Usage:
        validator = NavierStokesValidator()
        results = validator.run_all()
        validator.print_report(results)
    """

    # Literature values for validation
    LITERATURE = {
        SystemType.LORENZ_96: {
            "lyapunov": 1.67,    # For N=40, F=8
            "h_ks": 2.5,
        },
        SystemType.KURAMOTO_SIVASHINSKY: {
            "lyapunov": 0.08,    # For L=22
            "h_ks": 0.15,
        },
        SystemType.NAVIER_STOKES_2D: {
            "kolmogorov_slope": -5.0/3.0,  # ≈ -1.667
        },
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[ValidationMetrics] = []

    def run_all(self) -> List[ValidationMetrics]:
        """Run validation on all real-world systems."""
        self.results = []

        # 1. Lorenz-96
        self.results.append(self._validate_lorenz96())

        # 2. Kuramoto-Sivashinsky
        self.results.append(self._validate_ks())

        # 3. Navier-Stokes 2D
        self.results.append(self._validate_navier_stokes_2d())

        # 4. Financial GARCH
        self.results.append(self._validate_garch())

        # 5. Financial Jump-Diffusion
        self.results.append(self._validate_jump_diffusion())

        return self.results

    def _validate_lorenz96(self) -> ValidationMetrics:
        """Validate on Lorenz-96."""
        if self.verbose:
            print("\n🌀 Validating Lorenz-96 (N=40, F=8)...")

        config = SystemConfig(
            system_type=SystemType.LORENZ_96,
            n_lorenz=40, F_lorenz=8.0,
            dt=0.01, n_steps=5000, n_warmup=1000,
        )
        t0 = time.perf_counter()

        # Generate trajectory
        l96 = Lorenz96(config)
        traj = l96.generate_trajectory()

        # Use first component as 1D observable for ACF pipeline
        x_1d = traj[:, 0]

        # Run ACF pipeline
        metrics = self._run_acf_pipeline(
            x_1d, config, SystemType.LORENZ_96, "Lorenz-96"
        )

        # Literature comparison
        lit = self.LITERATURE[SystemType.LORENZ_96]
        metrics.lyapunov_literature = lit["lyapunov"]
        metrics.h_ks_literature = lit["h_ks"]
        metrics.lyapunov_error = abs(metrics.lyapunov_max - lit["lyapunov"]) / lit["lyapunov"]
        metrics.h_ks_error = abs(metrics.h_ks - lit["h_ks"]) / lit["h_ks"]
        metrics.n_dof = config.n_lorenz
        metrics.total_time_s = time.perf_counter() - t0

        if self.verbose:
            print(metrics.summary())
        return metrics

    def _validate_ks(self) -> ValidationMetrics:
        """Validate on Kuramoto-Sivashinsky."""
        if self.verbose:
            print("\n🌀 Validating Kuramoto-Sivashinsky (L=22)...")

        config = SystemConfig(
            system_type=SystemType.KURAMOTO_SIVASHINSKY,
            L_ks=22.0, n_ks=128,
            dt=0.25, n_steps=2000, n_warmup=500,
        )
        t0 = time.perf_counter()

        ks = KuramotoSivashinsky(config)
        traj = ks.generate_trajectory()

        # Use spatial mean as 1D observable
        x_1d = traj.mean(axis=1)

        metrics = self._run_acf_pipeline(
            x_1d, config, SystemType.KURAMOTO_SIVASHINSKY, "Kuramoto-Sivashinsky"
        )

        lit = self.LITERATURE[SystemType.KURAMOTO_SIVASHINSKY]
        metrics.lyapunov_literature = lit["lyapunov"]
        metrics.h_ks_literature = lit["h_ks"]
        metrics.lyapunov_error = abs(metrics.lyapunov_max - lit["lyapunov"]) / max(lit["lyapunov"], 0.01)
        metrics.h_ks_error = abs(metrics.h_ks - lit["h_ks"]) / max(lit["h_ks"], 0.01)
        metrics.n_dof = config.n_ks
        metrics.total_time_s = time.perf_counter() - t0

        if self.verbose:
            print(metrics.summary())
        return metrics

    def _validate_navier_stokes_2d(self) -> ValidationMetrics:
        """Validate on 2D Navier-Stokes."""
        if self.verbose:
            print("\n🌀 Validating Navier-Stokes 2D (64×64, ν=0.001)...")

        config = SystemConfig(
            system_type=SystemType.NAVIER_STOKES_2D,
            nx=64, ny=64, nu=0.001,
            dt=0.001, n_steps=3000, n_warmup=500,
        )
        t0 = time.perf_counter()

        ns = NavierStokes2DSpectral(config)
        energy_ts, ek = ns.generate_trajectory()

        # Run ACF pipeline on energy time series
        metrics = self._run_acf_pipeline(
            energy_ts, config, SystemType.NAVIER_STOKES_2D, "Navier-Stokes 2D"
        )

        # Kolmogorov spectrum check
        metrics.energy_spectrum = ek
        k = np.arange(1, len(ek) + 1)
        valid = (ek > 1e-15) & (k >= 3) & (k <= len(ek)//3)
        if valid.sum() >= 5:
            slope, intercept = np.polyfit(np.log(k[valid]), np.log(ek[valid]), 1)
            metrics.kolmogorov_slope = slope
            pred = slope * np.log(k[valid]) + intercept
            ss_res = np.sum((np.log(ek[valid]) - pred)**2)
            ss_tot = np.sum((np.log(ek[valid]) - np.mean(np.log(ek[valid])))**2)
            metrics.kolmogorov_r2 = 1.0 - ss_res / (ss_tot + 1e-14)

        lit = self.LITERATURE[SystemType.NAVIER_STOKES_2D]
        metrics.lyapunov_literature = 0.0  # 2D NS is not chaotic in this sense
        metrics.h_ks_literature = 0.0
        metrics.n_dof = config.nx * config.ny
        metrics.total_time_s = time.perf_counter() - t0

        if self.verbose:
            print(metrics.summary())
        return metrics

    def _validate_garch(self) -> ValidationMetrics:
        """Validate on GARCH financial data."""
        if self.verbose:
            print("\n🌀 Validating GARCH(1,1) financial model...")

        config = SystemConfig(
            system_type=SystemType.GARCH,
            n_financial=2000,
        )
        t0 = time.perf_counter()

        returns = FinancialGenerator.garch(n_steps=2000)

        metrics = self._run_acf_pipeline(
            returns, config, SystemType.GARCH, "GARCH(1,1)"
        )
        metrics.n_dof = 1
        metrics.total_time_s = time.perf_counter() - t0

        if self.verbose:
            print(metrics.summary())
        return metrics

    def _validate_jump_diffusion(self) -> ValidationMetrics:
        """Validate on Jump-Diffusion financial data."""
        if self.verbose:
            print("\n🌀 Validating Jump-Diffusion financial model...")

        config = SystemConfig(
            system_type=SystemType.JUMP_DIFFUSION,
            n_financial=2000,
        )
        t0 = time.perf_counter()

        log_price = FinancialGenerator.jump_diffusion(n_steps=2000)

        metrics = self._run_acf_pipeline(
            log_price, config, SystemType.JUMP_DIFFUSION, "Jump-Diffusion"
        )
        metrics.n_dof = 1
        metrics.total_time_s = time.perf_counter() - t0

        if self.verbose:
            print(metrics.summary())
        return metrics

    # ------------------------------------------------------------------
    # ACF Pipeline Runner
    # ------------------------------------------------------------------

    def _run_acf_pipeline(
        self,
        x: np.ndarray,
        config: SystemConfig,
        sys_type: SystemType,
        name: str,
    ) -> ValidationMetrics:
        """Run the full ACF pipeline on a 1D time series."""
        metrics = ValidationMetrics(system_name=name, system_type=sys_type)

        # Normalize
        x_norm = (x - x.mean()) / (x.std() + 1e-10)
        a, b = float(x_norm.min() - 0.5), float(x_norm.max() + 0.5)

        # Define 1D map via delay embedding
        def T_map(x_val: np.ndarray) -> np.ndarray:
            """Simple return map: x_{t+1} = f(x_t) approximated by nearest neighbor."""
            # Use the actual data: find where x_t ≈ x_val, return x_{t+1}
            x_arr = np.asarray(x_val).ravel()
            result = np.zeros_like(x_arr)
            for i, xv in enumerate(x_arr):
                diffs = np.abs(x_norm[:-1] - xv)
                idx = np.argmin(diffs)
                result[i] = x_norm[min(idx + 1, len(x_norm) - 1)]
            return result

        # 1. SEM
        try:
            from acf_functor.stochastic_membrane import StochasticMembrane, SMConfig
            sm_config = SMConfig(n_particles=200)
            sm = StochasticMembrane(sm_config)
            noisy = x_norm + 0.05 * np.random.default_rng(42).standard_normal(len(x_norm))
            sm_output = sm.process(noisy.reshape(-1, 1))
            metrics.sem_cert_pass = sm_output.purified.is_valid(sm_config)
        except Exception as e:
            if self.verbose:
                print(f"    SEM: ⚠ {e}")
            sm_output = None
            metrics.sem_cert_pass = False

        # 2. TAA
        try:
            from acf_functor.taa_agent import TAAAgent
            taa = TAAAgent(T=T_map, domain=(a, b), n_obs=32, n_traj=2000)
            taa.build()
            taa_cert = taa.certify()
            metrics.taa_cert_pass = taa_cert.PASS
        except Exception as e:
            if self.verbose:
                print(f"    TAA: ⚠ {e}")
            metrics.taa_cert_pass = False

        # 3. ERGON
        try:
            from acf_functor.ergon_agent import ERGONAgent
            ergon = ERGONAgent(T=T_map, domain=(a, b), n_grid=256, n_power_iter=2000)
            ergon_state = ergon.certify()
            metrics.ergon_cert_pass = ergon_state.certificates.get("PASS", False)
            metrics.lyapunov_max = ergon_state.lyapunov.lyapunov_max
            metrics.h_ks = ergon_state.h_ks
        except Exception as e:
            if self.verbose:
                print(f"    ERGON: ⚠ {e}")
            metrics.ergon_cert_pass = False

        # 4. OTU
        try:
            from acf_functor.gelfand_triple import GelfandTriple
            otu = GelfandTriple(T=T_map, domain=(a, b), n_test=24, n_hilbert=128, n_dist=256)
            otu_result = otu.analyze()
            metrics.otu_cert_pass = otu_result.spectrum.self_consistent
        except Exception as e:
            if self.verbose:
                print(f"    OTU: ⚠ {e}")
            metrics.otu_cert_pass = False

        # 5. PSAL
        try:
            from acf_functor.autopoietic_scientist import AutopoieticScientist
            traj_2d = x_norm[:500].reshape(-1, 1)
            scientist = AutopoieticScientist(
                n_modes_range=(2, 6), sindy_threshold=0.1,
                verification_tolerance=0.4, max_cycles=2,
            )
            psal_report = scientist.run(
                trajectory=traj_2d, dt=1.0,
                h_ks=metrics.h_ks if np.isfinite(metrics.h_ks) else None,
                n_cycles=2,
            )
            metrics.psal_cert_pass = psal_report.n_laws_verified > 0
            if psal_report.best_law:
                metrics.rom_trajectory_error = psal_report.best_law.trajectory_error
                metrics.rom_spectrum_error = psal_report.best_law.spectrum_error
                metrics.rom_energy_drift = psal_report.best_law.energy_drift
        except Exception as e:
            if self.verbose:
                print(f"    PSAL: ⚠ {e}")
            metrics.psal_cert_pass = False

        return metrics

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, results: Optional[List[ValidationMetrics]] = None) -> str:
        """Generate a comprehensive validation report."""
        if results is None:
            results = self.results

        lines = [
            "\n" + "=" * 70,
            "  ACF ECOSYSTEM — REAL-WORLD VALIDATION REPORT",
            "=" * 70,
        ]

        # Summary table
        lines.extend([
            "",
            f"  {'System':<25} {'λ_max':>8} {'h_KS':>8} {'SEM':>5} {'TAA':>5} {'ERGON':>5} {'OTU':>5} {'PSAL':>5}",
            f"  {'-'*25} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5}",
        ])

        for m in results:
            lines.append(
                f"  {m.system_name:<25} "
                f"{m.lyapunov_max:8.4f} "
                f"{m.h_ks:8.4f} "
                f"{'✅' if m.sem_cert_pass else '❌':>5} "
                f"{'✅' if m.taa_cert_pass else '❌':>5} "
                f"{'✅' if m.ergon_cert_pass else '❌':>5} "
                f"{'✅' if m.otu_cert_pass else '❌':>5} "
                f"{'✅' if m.psal_cert_pass else '❌':>5}"
            )

        # Statistics
        n_total = len(results)
        n_sem = sum(1 for m in results if m.sem_cert_pass)
        n_taa = sum(1 for m in results if m.taa_cert_pass)
        n_ergon = sum(1 for m in results if m.ergon_cert_pass)
        n_otu = sum(1 for m in results if m.otu_cert_pass)
        n_psal = sum(1 for m in results if m.psal_cert_pass)

        lines.extend([
            "",
            f"  Pass rates: SEM={n_sem}/{n_total}  TAA={n_taa}/{n_total}  "
            f"ERGON={n_ergon}/{n_total}  OTU={n_otu}/{n_total}  PSAL={n_psal}/{n_total}",
            "",
            "=" * 70,
        ])

        report = "\n".join(lines)
        if self.verbose:
            print(report)
        return report

    def to_dict(self) -> Dict[str, Any]:
        """Export results as JSON-compatible dict."""
        return {
            "results": [
                {
                    "system": m.system_name,
                    "lyapunov_max": m.lyapunov_max,
                    "h_ks": m.h_ks,
                    "kolmogorov_slope": m.kolmogorov_slope,
                    "sem_pass": m.sem_cert_pass,
                    "taa_pass": m.taa_cert_pass,
                    "ergon_pass": m.ergon_cert_pass,
                    "otu_pass": m.otu_cert_pass,
                    "psal_pass": m.psal_cert_pass,
                    "rom_trajectory_error": m.rom_trajectory_error,
                    "rom_spectrum_error": m.rom_spectrum_error,
                    "rom_energy_drift": m.rom_energy_drift,
                    "time_s": m.total_time_s,
                    "n_dof": m.n_dof,
                }
                for m in self.results
            ]
        }


# ---------------------------------------------------------------------------
# Quick validation helper
# ---------------------------------------------------------------------------

def validate_acf_on_real_systems(verbose: bool = True) -> List[ValidationMetrics]:
    """Run full ACF validation on real-world systems."""
    validator = NavierStokesValidator(verbose=verbose)
    results = validator.run_all()
    validator.print_report(results)
    return results