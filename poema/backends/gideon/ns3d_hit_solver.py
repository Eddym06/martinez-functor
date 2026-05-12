"""
ns3d_hit_solver.py — 3D Homogeneous Isotropic Turbulence Solver with CoPoem.

Pseudo-spectral solver for the 3D incompressible Navier-Stokes equations
in a triply-periodic box [0, 2π]³, integrated with CoPoem Inverse Spectral
Design to achieve the Kolmogorov k^{-5/3} energy spectrum.

Physics:
  ∂u/∂t + (u·∇)u = -∇p + ν∇²u - ν₄∇⁸u + F
  ∇·u = 0  (incompressibility via Leray projection)

Method:
  - 3D FFT pseudo-spectral with 2/3 dealiasing
  - RK4 time-stepping with integrating factor for diffusion
  - Leray-Helmholtz projector P = I - k⊗k/|k|² in Fourier space
  - Shell-averaged energy spectrum E(k) for CoPoem feedback

CoPoem Integration:
  - Broadband forcing F(k) across k ∈ [k_min, k_max]
  - Per-shell amplitudes A(k) optimized by CoPoemSpectralDesigner
  - Target spectrum: E(k) ~ k^{-5/3} (Kolmogorov direct cascade)

Martínez's Invariant — Abril 2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HIT3DConfig:
    """Configuration for 3D HIT solver with CoPoem."""
    # Grid
    N: int = 64                        # 64³ grid → 262,144 DOFs
    Re: float = 1600.0                 # Reynolds number
    nu4_coeff: float = 1e-14           # Hyperviscosity coefficient
    
    # Time
    T_total: float = 20.0
    dt_init: float = 1e-3
    cfl_target: float = 0.4
    
    # Forcing
    k_force: int = 3                   # Central forcing wavenumber (for non-CoPoem)
    force_amplitude: float = 100.0
    
    # CoPoem Inverse Spectral Design
    use_copoem: bool = True
    k_force_min: int = 2               # Broadband forcing low end
    k_force_max: int = 6               # Broadband forcing high end
    copoem_target_slope: float = -5.0 / 3.0  # Kolmogorov -5/3
    copoem_learning_rate: float = 0.3
    copoem_max_power: float = 100000.0
    
    # Analysis
    analysis_interval: float = 1.0     # Koopman + CoPoem every T seconds
    snapshot_interval: float = 0.1     # Energy/enstrophy sampling
    
    # Koopman
    n_koopman_modes: int = 40


# ═══════════════════════════════════════════════════════════════════════════
# 3D HIT Solver
# ═══════════════════════════════════════════════════════════════════════════

class HIT3DSolver:
    """
    3D pseudo-spectral Navier-Stokes solver for Homogeneous Isotropic Turbulence.
    
    Works in velocity formulation with Leray projection for incompressibility.
    Coupled with CoPoem for inverse spectral design targeting k^{-5/3}.
    """
    
    def __init__(self, config: Optional[HIT3DConfig] = None):
        self.config = config or HIT3DConfig()
        self._setup_grid()
        self._setup_copoem()
        self.koopman_results: List[Any] = []
        self._copoem_history: List[Dict] = []
    
    def _setup_grid(self):
        """Initialize 3D spectral grid."""
        cfg = self.config
        N = cfg.N
        
        # Physical grid
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        self.x, self.y, self.z = np.meshgrid(x, x, x, indexing='ij')
        self.dx = 2 * np.pi / N
        
        # Wavenumber grid
        k = fftfreq(N, d=1.0 / N)
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_safe = self.k2.copy()
        self.k2_safe[0, 0, 0] = 1.0  # Avoid division by zero
        self.k_mag = np.sqrt(self.k2)
        
        # Dealiasing: 2/3 rule
        k_max = N // 3
        self.dealias = (
            (np.abs(self.kx) <= k_max) & 
            (np.abs(self.ky) <= k_max) & 
            (np.abs(self.kz) <= k_max)
        )
        
        # Linear operator: viscous + hyperviscous dissipation
        nu = 1.0 / cfg.Re
        self.linear_op = -nu * self.k2 - cfg.nu4_coeff * self.k2**4
        self.linear_op[0, 0, 0] = 0.0
        
        # Leray projector components: P_ij = δ_ij - k_i k_j / |k|²
        # We store the k_i k_j / |k|² tensor for projection
        self._kk_over_k2 = np.zeros((3, 3, N, N, N))
        kvecs = [self.kx, self.ky, self.kz]
        for i in range(3):
            for j in range(3):
                self._kk_over_k2[i, j] = kvecs[i] * kvecs[j] / self.k2_safe
        self._kk_over_k2[:, :, 0, 0, 0] = 0.0
        
        # Forcing setup
        self._setup_forcing()
    
    def _setup_forcing(self):
        """Initialize forcing field."""
        cfg = self.config
        N = cfg.N
        
        np.random.seed(42)
        # Random phases for each component
        self.force_phases = [
            np.random.uniform(0, 2 * np.pi, (N, N, N)) for _ in range(3)
        ]
        
        if cfg.use_copoem:
            # Broadband forcing: will be updated per CoPoem cycle
            self.force_band = {}
            for k_shell in range(cfg.k_force_min, cfg.k_force_max + 1):
                band = (np.abs(self.k_mag - k_shell) < 0.6) & (self.k_mag > 0)
                self.force_band[k_shell] = band
            self._build_forcing(np.full(
                cfg.k_force_max - cfg.k_force_min + 1, cfg.force_amplitude
            ))
        else:
            band = (np.abs(self.k_mag - cfg.k_force) < 1.0) & (self.k_mag > 0)
            self.force_band = {cfg.k_force: band}
            self._build_forcing(np.array([cfg.force_amplitude]))
    
    def _build_forcing(self, amplitudes: np.ndarray):
        """Build solenoidal forcing field from per-shell amplitudes."""
        cfg = self.config
        N = cfg.N
        
        # Build raw forcing in Fourier space (3 components)
        F_hat = [np.zeros((N, N, N), dtype=complex) for _ in range(3)]
        
        shell_keys = sorted(self.force_band.keys())
        for idx, k_shell in enumerate(shell_keys):
            if idx >= len(amplitudes):
                break
            band = self.force_band[k_shell]
            if not np.any(band):
                continue
            A = amplitudes[idx]
            for comp in range(3):
                F_hat[comp][band] = A * np.exp(1j * self.force_phases[comp][band])
        
        # Project forcing to be solenoidal: F_sol = P · F
        self.forcing_hat = self._leray_project(F_hat)
        self.current_force_amp = float(np.mean(amplitudes))
    
    def _leray_project(self, u_hat: List[np.ndarray]) -> List[np.ndarray]:
        """Apply Leray-Helmholtz projector: P_ij = δ_ij - k_i k_j / |k|²."""
        projected = []
        for i in range(3):
            comp = u_hat[i].copy()
            for j in range(3):
                comp -= self._kk_over_k2[i, j] * u_hat[j]
            projected.append(comp)
        return projected
    
    def _setup_copoem(self):
        """Initialize CoPoem spectral designer for 3D."""
        self._copoem_designer = None
        self._copoem_oracle = None
        
        if self.config.use_copoem:
            try:
                from .copoem_spectral_designer import (
                    CoPoemSpectralDesigner, CoPoemOracle, DesignerConfig,
                )
            except ImportError:
                from poema.backends.gideon.copoem_spectral_designer import (
                    CoPoemSpectralDesigner, CoPoemOracle, DesignerConfig,
                )
            
            dcfg = DesignerConfig(
                target_slope=self.config.copoem_target_slope,
                k_force_min=self.config.k_force_min,
                k_force_max=self.config.k_force_max,
                initial_amplitude=self.config.force_amplitude,
                learning_rate=self.config.copoem_learning_rate,
                max_total_power=self.config.copoem_max_power,
                # Inertial range for 3D: measure slope in [k_force_max+2, N/3]
                target_slope_range=(self.config.k_force_max + 2, self.config.N // 3),
                misfit_threshold=0.15,    # Tighter for 3D (should converge better)
                slope_tolerance=0.3,
            )
            self._copoem_designer = CoPoemSpectralDesigner(
                cfg=dcfg,
                N=self.config.N,
                nu=1.0 / self.config.Re,
                nu4=self.config.nu4_coeff,
            )
            self._copoem_oracle = CoPoemOracle(self._copoem_designer)
        
        # Koopman
        self._koopman_available = False
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
                pass
    
    # ─────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────
    
    def _init_velocity(self) -> List[np.ndarray]:
        """Initialize solenoidal velocity field with energy in low modes."""
        N = self.config.N
        np.random.seed(123)
        
        # Random velocity in Fourier space
        u_hat = []
        for comp in range(3):
            raw = np.zeros((N, N, N), dtype=complex)
            # Energy concentrated at low k
            mask = (self.k_mag > 0) & (self.k_mag < 6)
            n_modes = np.sum(mask)
            if n_modes > 0:
                amps = 2.0 / (1.0 + self.k2[mask])
                phases = np.random.uniform(0, 2 * np.pi, n_modes)
                raw[mask] = amps * np.exp(1j * phases)
            u_hat.append(raw)
        
        # Project to solenoidal
        u_hat = self._leray_project(u_hat)
        return u_hat
    
    # ─────────────────────────────────────────────────────────────────────
    # Nonlinear term
    # ─────────────────────────────────────────────────────────────────────
    
    def _nonlinear_rhs(self, u_hat: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute P[-(u·∇)u] via the rotational (Lamb vector) form:
          (u·∇)u = ω×u + ∇(|u|²/2)
        After Leray projection P, the gradient vanishes:
          P[-(u·∇)u] = P[-(ω×u)]
        
        This uses 6 IFFTs + 3 FFTs = 9 total (vs 15 for convective form).
        """
        dealias = self.dealias
        
        # Velocity in physical space: 3 IFFTs
        ux = np.real(ifftn(u_hat[0] * dealias))
        uy = np.real(ifftn(u_hat[1] * dealias))
        uz = np.real(ifftn(u_hat[2] * dealias))
        
        # Vorticity ω = ∇×u in physical space: 3 IFFTs
        # ωx = ∂uz/∂y - ∂uy/∂z
        # ωy = ∂ux/∂z - ∂uz/∂x
        # ωz = ∂uy/∂x - ∂ux/∂y
        wx = np.real(ifftn((1j * self.ky * u_hat[2] - 1j * self.kz * u_hat[1]) * dealias))
        wy = np.real(ifftn((1j * self.kz * u_hat[0] - 1j * self.kx * u_hat[2]) * dealias))
        wz = np.real(ifftn((1j * self.kx * u_hat[1] - 1j * self.ky * u_hat[0]) * dealias))
        
        # Lamb vector L = ω×u (in physical space, then FFT): 3 FFTs
        Lx = fftn(wy * uz - wz * uy) * dealias
        Ly = fftn(wz * ux - wx * uz) * dealias
        Lz = fftn(wx * uy - wy * ux) * dealias
        
        # P[-(ω×u)]: Leray projection removes gradient part
        nl = self._leray_project([-Lx, -Ly, -Lz])
        return nl
    
    # ─────────────────────────────────────────────────────────────────────
    # Time stepping
    # ─────────────────────────────────────────────────────────────────────
    
    def _rk4_step(self, u_hat: List[np.ndarray], dt: float) -> List[np.ndarray]:
        """RK4 with integrating factor for the linear (viscous) term."""
        E_half = np.exp(self.linear_op * dt / 2)
        E_full = E_half * E_half  # Not used directly but conceptually
        
        # Apply half-step integrating factor
        u_hat = [u * E_half for u in u_hat]
        
        def NL(w):
            nl = self._nonlinear_rhs(w)
            # Add forcing
            return [nl[i] + self.forcing_hat[i] for i in range(3)]
        
        k1 = NL(u_hat)
        w2 = [u_hat[i] + 0.5 * dt * k1[i] for i in range(3)]
        k2 = NL(w2)
        w3 = [u_hat[i] + 0.5 * dt * k2[i] for i in range(3)]
        k3 = NL(w3)
        w4 = [u_hat[i] + dt * k3[i] for i in range(3)]
        k4 = NL(w4)
        
        result = []
        for i in range(3):
            u_new = u_hat[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            u_new = u_new * E_half  # second half of integrating factor
            result.append(u_new)
        
        return result
    
    def _adaptive_dt(self, u_hat: List[np.ndarray]) -> float:
        """CFL-based adaptive timestep."""
        u_max = 0.0
        for comp in range(3):
            u_phys = np.real(ifftn(u_hat[comp]))
            u_max = max(u_max, np.max(np.abs(u_phys)))
        u_max = max(u_max, 1e-10)
        
        dt_cfl = self.config.cfl_target * self.dx / (3 * u_max)  # factor 3 for 3D
        dt_visc = 0.25 * self.dx**2 * self.config.Re
        return min(dt_cfl, dt_visc, 5e-3)
    
    # ─────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────
    
    def _compute_energy(self, u_hat: List[np.ndarray]) -> float:
        """Total kinetic energy: E = ½ <|u|²>."""
        N = self.config.N
        E = 0.0
        for comp in range(3):
            E += 0.5 * np.sum(np.abs(u_hat[comp])**2) / N**3
        return float(np.real(E)) / N**3  # normalize by grid volume
    
    def _compute_enstrophy(self, u_hat: List[np.ndarray]) -> float:
        """Enstrophy: Z = ½ <|ω|²> where ω = ∇×u."""
        # ω = (∂u_z/∂y - ∂u_y/∂z, ∂u_x/∂z - ∂u_z/∂x, ∂u_y/∂x - ∂u_x/∂y)
        N = self.config.N
        omega_hat = [
            1j * self.ky * u_hat[2] - 1j * self.kz * u_hat[1],
            1j * self.kz * u_hat[0] - 1j * self.kx * u_hat[2],
            1j * self.kx * u_hat[1] - 1j * self.ky * u_hat[0],
        ]
        Z = 0.0
        for comp in range(3):
            Z += 0.5 * np.sum(np.abs(omega_hat[comp])**2) / N**3
        return float(np.real(Z)) / N**3
    
    def _energy_spectrum(self, u_hat: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Shell-averaged energy spectrum E(k)."""
        N = self.config.N
        k_max = N // 2
        
        # Energy density in Fourier space
        e_hat = np.zeros((N, N, N))
        for comp in range(3):
            e_hat += 0.5 * np.abs(u_hat[comp])**2 / N**6
        
        k_bins = np.arange(1, k_max + 1, dtype=float)
        E_k = np.zeros(k_max)
        for i, k_val in enumerate(k_bins):
            shell = (self.k_mag >= k_val - 0.5) & (self.k_mag < k_val + 0.5)
            E_k[i] = np.sum(e_hat[shell])
        
        return k_bins, E_k
    
    def _compute_taylor_reynolds(self, u_hat: List[np.ndarray]) -> float:
        """Taylor-scale Reynolds number Re_λ."""
        E = self._compute_energy(u_hat)
        Z = self._compute_enstrophy(u_hat)
        if Z < 1e-30:
            return 0.0
        nu = 1.0 / self.config.Re
        # u_rms = sqrt(2E/3), λ = sqrt(15 ν u_rms² / ε), ε ≈ 2νZ
        u_rms2 = 2 * E / 3
        epsilon = 2 * nu * Z
        if epsilon < 1e-30:
            return 0.0
        Re_lambda = np.sqrt(15 * u_rms2 / (nu * epsilon)) * np.sqrt(u_rms2)
        return float(Re_lambda)
    
    # ─────────────────────────────────────────────────────────────────────
    # CoPoem action
    # ─────────────────────────────────────────────────────────────────────
    
    def _apply_copoem_action(self, action: Dict[str, Any]):
        """Apply CoPoem per-mode control: rebuild solenoidal forcing."""
        amplitudes = action.get("amplitudes", None)
        if amplitudes is not None:
            self._build_forcing(np.array(amplitudes))
        
        # Update viscosity profile if designed
        nu4_eff = action.get("nu4_effective", self.config.nu4_coeff)
        if self._copoem_designer and self._copoem_designer.cfg.enable_nu_design:
            nu = 1.0 / self.config.Re
            # Per-scale viscosity: use nu_multipliers on k shells
            # For simplicity, update the linear_op directly
            nu4_field = np.full_like(self.k2, self.config.nu4_coeff)
            for i, k_shell in enumerate(self._copoem_designer.k_shells):
                band = (np.abs(self.k_mag - k_shell) < 0.6)
                if np.any(band):
                    nu4_field[band] = self.config.nu4_coeff * self._copoem_designer.nu_multipliers[i]
            self.linear_op = -nu * self.k2 - nu4_field * self.k2**4
            self.linear_op[0, 0, 0] = 0.0
    
    # ─────────────────────────────────────────────────────────────────────
    # Koopman analysis (flattened 3D → 2D for KoopmanGPU)
    # ─────────────────────────────────────────────────────────────────────
    
    def _run_koopman_analysis(self, snapshot_buffer: List[np.ndarray], dt_snap: float):
        """Run Koopman on energy-containing z-slice snapshots."""
        if not self._koopman_available or len(snapshot_buffer) < 20:
            return None
        try:
            # Use mid-z slice of velocity magnitude as 2D proxy
            buf = np.array(snapshot_buffer)  # (T, Nx, Ny)
            result = self.koopman.analyze(buf, dt=dt_snap, n_modes=self.config.n_koopman_modes)
            self.koopman_results.append(result)
            return result
        except Exception as e:
            print(f"  ⚠ Koopman 3D analysis failed: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════
    # Main simulation loop
    # ═══════════════════════════════════════════════════════════════════════
    
    def simulate(self) -> Dict[str, Any]:
        """Run 3D HIT simulation with CoPoem spectral design."""
        cfg = self.config
        N = cfg.N
        
        print("=" * 70)
        print(f"3D HOMOGENEOUS ISOTROPIC TURBULENCE — N={N}³ ({N**3:,} DOFs)")
        print("=" * 70)
        print(f"  Re = {cfg.Re:.0f}, ν = {1/cfg.Re:.6f}, ν₄ = {cfg.nu4_coeff:.1e}")
        print(f"  T_total = {cfg.T_total}, análisis cada {cfg.analysis_interval}s")
        if self._copoem_designer:
            print(f"  CoPoem: target slope = {cfg.copoem_target_slope:.4f} (Kolmogorov -5/3)")
            print(f"  Broadband forcing: k ∈ [{cfg.k_force_min}, {cfg.k_force_max}]")
            print(f"  Power budget: {cfg.copoem_max_power}")
        print(f"  Koopman GPU: {'disponible' if self._koopman_available else 'CPU fallback'}")
        print()
        
        # Initialize
        u_hat = self._init_velocity()
        
        t = 0.0
        step = 0
        t_next_snap = 0.0
        t_next_analysis = cfg.analysis_interval
        t_next_print = 0.0
        
        energies = []
        enstrophies = []
        times = []
        analysis_buffer = []  # z-slices for Koopman
        
        t_start = time.time()
        last_koopman = None
        
        while t < cfg.T_total:
            dt = self._adaptive_dt(u_hat)
            dt_actual = min(dt, cfg.T_total - t)
            if t_next_snap - t > 1e-12 and t_next_snap - t < dt_actual:
                dt_actual = t_next_snap - t
            
            u_hat = self._rk4_step(u_hat, dt_actual)
            t += dt_actual
            step += 1
            
            # Blowup check
            if step % 100 == 0:
                max_u = max(
                    np.max(np.abs(np.real(ifftn(u_hat[c])))) for c in range(3)
                )
                if np.isnan(max_u) or max_u > 1e8:
                    print(f"  ⚠ BLOWUP at t={t:.4f}, step={step}")
                    break
            
            # Snapshot
            if t >= t_next_snap - 1e-12:
                E = self._compute_energy(u_hat)
                Z = self._compute_enstrophy(u_hat)
                energies.append(E)
                enstrophies.append(Z)
                times.append(t)
                
                # Mid-z slice of |u| for Koopman
                mid = N // 2
                u_mag_slice = np.zeros((N, N))
                for c in range(3):
                    u_c = np.real(ifftn(u_hat[c]))
                    u_mag_slice += u_c[:, :, mid]**2
                u_mag_slice = np.sqrt(u_mag_slice)
                analysis_buffer.append(u_mag_slice)
                
                t_next_snap = t + cfg.snapshot_interval
            
            # Analysis + CoPoem control
            if t >= t_next_analysis - 1e-12 and len(analysis_buffer) >= 15:
                print(f"  ⟳ Análisis 3D en t={t:.2f} ({len(analysis_buffer)} slices)...", flush=True)
                t_k0 = time.time()
                
                koopman_result = self._run_koopman_analysis(analysis_buffer, cfg.snapshot_interval)
                
                spec_rad = 1.0
                d_95 = 0
                if koopman_result is not None:
                    if hasattr(koopman_result, 'eigenvalues') and koopman_result.eigenvalues is not None:
                        spec_rad = float(np.max(np.abs(koopman_result.eigenvalues)))
                    d_95 = koopman_result.d_95 if hasattr(koopman_result, 'd_95') else 0
                
                if self._copoem_oracle and cfg.use_copoem:
                    k_bins, E_k = self._energy_spectrum(u_hat)
                    E_curr = energies[-1] if energies else 0
                    Z_curr = enstrophies[-1] if enstrophies else 0
                    
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
                        "d_95": d_95,
                    })
                    
                    t_k1 = time.time()
                    Re_lambda = self._compute_taylor_reynolds(u_hat)
                    print(f"    ✓ CoPoem: J={copoem_action['misfit']:.3f}, "
                          f"slope={copoem_action['slope_actual']:.2f}, "
                          f"phase={copoem_action['phase']}, "
                          f"d_95={d_95}, Reλ={Re_lambda:.0f}, "
                          f"Ā={float(np.mean(copoem_action['amplitudes'])):.1f}, "
                          f"gap={copoem_action['adjunction_gap']:.3f} "
                          f"[{(t_k1-t_k0)*1000:.0f}ms]")
                
                last_koopman = koopman_result
                keep = max(len(analysis_buffer) // 5, 5)
                analysis_buffer = analysis_buffer[-keep:]
                t_next_analysis = t + cfg.analysis_interval
            
            # Progress
            if t >= t_next_print - 1e-12:
                E = self._compute_energy(u_hat)
                Z = self._compute_enstrophy(u_hat)
                elapsed = time.time() - t_start
                d95_str = f"d95={last_koopman.d_95}" if (last_koopman and hasattr(last_koopman, 'd_95')) else "d95=?"
                phase_str = ""
                if self._copoem_history:
                    lc = self._copoem_history[-1]
                    phase_str = f"  [{lc['phase']}|J={lc['misfit']:.2f}|s={lc['slope']:.2f}]"
                print(f"  t={t:6.2f}/{cfg.T_total:.0f}  E={E:.6f}  Z={Z:.6f}  "
                      f"dt={dt_actual:.2e}  {d95_str}  A={self.current_force_amp:.1f}  "
                      f"ν₄={cfg.nu4_coeff:.1e}{phase_str}  [{elapsed:.1f}s]", flush=True)
                t_next_print = t + max(cfg.T_total / 20, 1.0)
        
        elapsed_total = time.time() - t_start
        print(f"\n  Simulación 3D completada: {step} pasos en {elapsed_total:.1f}s")
        print(f"  Análisis Koopman: {len(self.koopman_results)}")
        
        # Final spectrum
        k_bins, E_k = self._energy_spectrum(u_hat)
        Re_lambda = self._compute_taylor_reynolds(u_hat)
        
        return {
            "energy": np.array(energies),
            "enstrophy": np.array(enstrophies),
            "times": np.array(times),
            "k_bins": k_bins,
            "E_k": E_k,
            "Re_lambda": Re_lambda,
            "koopman_results": self.koopman_results,
            "copoem_history": self._copoem_history,
            "copoem_designer_history": (
                [
                    {
                        "t": s.t, "misfit": s.misfit, "slope_actual": s.slope_actual,
                        "adjunction_gap": s.adjunction_gap, "phase": s.phase,
                        "total_power": s.total_power,
                        "amplitudes": s.amplitudes.tolist(),
                        "nu_profile": s.nu_profile.tolist(),
                        "k_shells": s.k_shells.tolist(),
                    }
                    for s in self._copoem_designer.history
                ]
                if self._copoem_designer else []
            ),
            "last_koopman": last_koopman,
            "params": {
                "N": N, "Re": cfg.Re, "nu": 1.0 / cfg.Re,
                "nu4": cfg.nu4_coeff, "T_total": cfg.T_total,
                "use_copoem": cfg.use_copoem,
                "target_slope": cfg.copoem_target_slope,
                "k_force_min": cfg.k_force_min,
                "k_force_max": cfg.k_force_max,
            },
            "elapsed_total": elapsed_total,
            "n_steps": step,
        }
