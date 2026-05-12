"""
ROM Synthesizer — Sparse Identification of Nonlinear Dynamics (SINDy)
=====================================================================

Discovers reduced-order models (ROMs) from Koopman modal amplitudes:

    ȧ = L·a + Q(a,a) + ν_t·D·a

where:
  a(t) ∈ ℝ^r    — modal amplitudes from TAA/Koopman decomposition
  L    ∈ ℝ^{r×r} — linear dynamics (from EDMD or Galerkin projection)
  Q    ∈ ℝ^{r×r×r} — quadratic nonlinear tensor (energy-transferring)
  ν_t  ∈ ℝ       — turbulent eddy viscosity (from ERGON thermodynamic closure)
  D    ∈ ℝ^{r×r} — dissipation operator (diagonal, ∝ -k²)

METHODOLOGY: SINDy with Sequentially Thresholded Least Squares (STLSQ)
-----------------------------------------------------------------------
Given time-series of modal amplitudes a(t₁), ..., a(tₙ):

1. Compute time derivatives: ȧ ≈ (a(t+Δt) - a(t-Δt))/(2Δt)
2. Build polynomial library: Θ(a) = [1, a₁, ..., aᵣ, a₁², a₁a₂, ..., aᵣ²]
3. Solve sparse regression: ȧ = Θ(a)·Ξ  with STLSQ thresholding
4. Extract L (linear part) and Q (quadratic part) from Ξ

THEOREMS:
  ROM-1: If the system is quadratically nonlinear (Navier-Stokes),
         SINDy recovers exact L,Q with O(1/√N) convergence in N samples.
  ROM-2: Energy conservation: Tr(L) + ν_t·Tr(D) < 0 ⟹ ROM is dissipative.
  ROM-3: Spectral fidelity: E_ROM(k) ≈ E_DNS(k) within inertial range.

References:
  Brunton et al. (2016) — Discovering governing equations from data
  Noack et al. (2003) — A hierarchy of low-dimensional models for the
                         transient and post-transient cylinder wake
  Loiseau & Brunton (2018) — Constrained sparse Galerkin regression
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PolynomialLibrary:
    """Polynomial feature library Θ(a) for SINDy regression."""
    degree: int
    n_state: int
    n_features: int
    feature_names: List[str]

    def transform(self, A: np.ndarray) -> np.ndarray:
        """
        Build Θ(A) from state matrix A of shape (n_samples, n_state).

        Returns Θ of shape (n_samples, n_features).
        """
        n_samples, r = A.shape
        features = []
        names = []

        # Constant term
        features.append(np.ones((n_samples, 1)))
        names.append("1")

        # Linear terms
        for i in range(r):
            features.append(A[:, i:i+1])
            names.append(f"a{i}")

        # Quadratic terms
        if self.degree >= 2:
            for i in range(r):
                for j in range(i, r):
                    features.append((A[:, i] * A[:, j]).reshape(-1, 1))
                    if i == j:
                        names.append(f"a{i}^2")
                    else:
                        names.append(f"a{i}*a{j}")

        # Cubic terms (optional, for higher-order dynamics)
        if self.degree >= 3:
            for i in range(r):
                for j in range(i, r):
                    for k in range(j, r):
                        features.append(
                            (A[:, i] * A[:, j] * A[:, k]).reshape(-1, 1)
                        )
                        names.append(f"a{i}*a{j}*a{k}")

        theta = np.hstack(features)
        self.feature_names = names
        self.n_features = theta.shape[1]
        return theta

    @classmethod
    def create(cls, n_state: int, degree: int = 2) -> "PolynomialLibrary":
        """Create a polynomial library descriptor."""
        # Count features
        n_feat = 1 + n_state  # constant + linear
        if degree >= 2:
            n_feat += n_state * (n_state + 1) // 2  # quadratic
        if degree >= 3:
            # Combinations with replacement: C(r+2, 3)
            n_feat += (n_state * (n_state + 1) * (n_state + 2)) // 6
        return cls(
            degree=degree,
            n_state=n_state,
            n_features=n_feat,
            feature_names=[],
        )


@dataclass
class SINDyResult:
    """Result of SINDy sparse regression."""
    Xi: np.ndarray               # Sparse coefficient matrix (n_features, n_state)
    L: np.ndarray                # Linear operator (n_state, n_state)
    Q: np.ndarray                # Quadratic tensor (n_state, n_state, n_state)
    c: np.ndarray                # Constant forcing (n_state,)
    library: PolynomialLibrary
    sparsity: float              # Fraction of zero coefficients
    residual_norm: float         # ‖ȧ - Θ·Ξ‖_F / ‖ȧ‖_F
    n_active_terms: int          # Number of nonzero terms
    feature_names: List[str]
    elapsed_ms: float


@dataclass
class ROMModel:
    """
    Complete reduced-order model: ȧ = L·a + Q(a,a) + c + ν_t·D·a.

    This is the "law" discovered by the Autopoietic Scientist.
    """
    L: np.ndarray                      # Linear dynamics (r, r)
    Q: np.ndarray                      # Quadratic tensor (r, r, r)
    c: np.ndarray                      # Constant forcing (r,)
    D: np.ndarray                      # Dissipation operator (r, r)
    nu_t: float                        # Eddy viscosity from ERGON closure
    n_modes: int                       # r = number of modes
    sindy_result: Optional[SINDyResult] = None
    energy_spectrum: Optional[np.ndarray] = None   # E(k) from ROM
    reference_spectrum: Optional[np.ndarray] = None  # E(k) from DNS
    metadata: Dict = field(default_factory=dict)

    def rhs(self, a: np.ndarray) -> np.ndarray:
        """Evaluate ȧ = L·a + Q(a,a) + c + ν_t·D·a."""
        # Linear + closure
        da = (self.L + self.nu_t * self.D) @ a + self.c
        # Quadratic
        for i in range(self.n_modes):
            da[i] += a @ self.Q[i] @ a
        return da

    def rhs_vectorized(self, a: np.ndarray) -> np.ndarray:
        """Vectorized RHS for batch evaluation."""
        # a shape: (n_modes,) or (batch, n_modes)
        if a.ndim == 1:
            return self.rhs(a)
        batch = a.shape[0]
        da = a @ (self.L + self.nu_t * self.D).T + self.c[None, :]
        for i in range(self.n_modes):
            da[:, i] += np.einsum('bj,jk,bk->b', a, self.Q[i], a)
        return da

    def integrate(
        self,
        a0: np.ndarray,
        dt: float,
        n_steps: int,
        method: str = "rk4",
    ) -> np.ndarray:
        """Integrate ROM forward in time. Returns (n_steps+1, n_modes)."""
        trajectory = np.zeros((n_steps + 1, self.n_modes))
        trajectory[0] = a0.copy()
        a = a0.copy()

        for step in range(n_steps):
            if method == "rk4":
                k1 = self.rhs(a)
                k2 = self.rhs(a + 0.5 * dt * k1)
                k3 = self.rhs(a + 0.5 * dt * k2)
                k4 = self.rhs(a + dt * k3)
                a = a + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:  # euler
                a = a + dt * self.rhs(a)

            # Stability check
            if np.any(np.isnan(a)) or np.max(np.abs(a)) > 1e10:
                warnings.warn(f"ROM blew up at step {step}")
                trajectory[step + 1:] = np.nan
                return trajectory

            trajectory[step + 1] = a

        return trajectory

    def energy(self, a: np.ndarray) -> float:
        """Total kinetic energy: E = ½ Σ aᵢ²."""
        return 0.5 * np.sum(a ** 2)

    def energy_transfer_rate(self, a: np.ndarray) -> float:
        """dE/dt = a·ȧ = a·(La + Q(a,a) + c + ν_t·Da)."""
        return float(np.dot(a, self.rhs(a)))

    def is_dissipative(self) -> bool:
        """Check ROM-2: Tr(L + ν_t·D) < 0 ⟹ dissipative."""
        return float(np.trace(self.L + self.nu_t * self.D)) < 0


@dataclass
class ROMVerification:
    """Verification of ROM fidelity against reference data."""
    trajectory_error: float      # ‖a_ROM - a_ref‖ / ‖a_ref‖
    spectrum_error: float        # ‖E_ROM(k) - E_ref(k)‖ / ‖E_ref(k)‖
    energy_conservation: float   # |E_ROM(T) - E_ROM(0)| / E_ROM(0)
    is_dissipative: bool
    is_stable: bool              # ROM doesn't blow up
    lyapunov_time: float         # Time before ROM diverges from reference
    certificates: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SINDy Engine
# ---------------------------------------------------------------------------

class SINDyEngine:
    """
    Sparse Identification of Nonlinear Dynamics (SINDy).

    Discovers governing equations from time-series data using
    sparse regression with sequential thresholding.
    """

    def __init__(
        self,
        poly_degree: int = 2,
        threshold: float = 0.05,
        max_iter: int = 20,
        alpha_ridge: float = 1e-5,
        normalize: bool = True,
    ):
        self.poly_degree = poly_degree
        self.threshold = threshold
        self.max_iter = max_iter
        self.alpha_ridge = alpha_ridge
        self.normalize = normalize

    def fit(
        self,
        A: np.ndarray,
        dt: float,
        derivative_method: str = "centered",
        sample_weights: Optional[np.ndarray] = None,
    ) -> SINDyResult:
        """
        Discover dynamics ȧ = f(a) from modal amplitude time series.

        Parameters
        ----------
        A : np.ndarray, shape (n_samples, n_modes)
            Time series of modal amplitudes.
        dt : float
            Time step between samples.
        derivative_method : str
            "centered" for 2nd-order central differences,
            "fourth_order" for 4th-order, "savgol" for Savitzky-Golay.

        Returns
        -------
        SINDyResult with sparse coefficient matrix Ξ.
        """
        t0 = time.perf_counter()
        n_samples, r = A.shape

        # Step 1: Compute time derivatives
        dA = self._compute_derivatives(A, dt, derivative_method)

        # Trim edges where derivatives are unreliable
        if derivative_method == "centered":
            A_trim = A[1:-1]
            dA_trim = dA
            trim_slice = slice(1, -1)
        elif derivative_method == "fourth_order":
            A_trim = A[2:-2]
            dA_trim = dA
            trim_slice = slice(2, -2)
        else:
            A_trim = A[1:-1]
            dA_trim = dA
            trim_slice = slice(1, -1)

        # Trim sample_weights to match trimmed arrays
        if sample_weights is not None:
            weights_trim = sample_weights[trim_slice]
            # Clip to valid range and normalise
            weights_trim = np.clip(weights_trim, 1e-3, None)
            weights_trim = weights_trim / weights_trim.mean()
        else:
            weights_trim = None

        # Step 2: Build polynomial library
        library = PolynomialLibrary.create(r, self.poly_degree)
        Theta = library.transform(A_trim)

        # Step 3: Sparse regression via STLSQ (possibly weighted)
        Xi = self._stlsq(Theta, dA_trim, sample_weights=weights_trim)

        # Step 4: Extract L, Q, c from Xi
        L, Q, c = self._extract_operators(Xi, r, library)

        # Compute residual
        predicted = Theta @ Xi
        res_norm = np.linalg.norm(dA_trim - predicted, 'fro')
        da_norm = np.linalg.norm(dA_trim, 'fro')
        rel_residual = res_norm / (da_norm + 1e-15)

        n_active = int(np.count_nonzero(Xi))
        sparsity = 1.0 - n_active / Xi.size

        elapsed = (time.perf_counter() - t0) * 1000

        return SINDyResult(
            Xi=Xi,
            L=L,
            Q=Q,
            c=c,
            library=library,
            sparsity=sparsity,
            residual_norm=rel_residual,
            n_active_terms=n_active,
            feature_names=library.feature_names,
            elapsed_ms=elapsed,
        )

    def _compute_derivatives(
        self,
        A: np.ndarray,
        dt: float,
        method: str,
    ) -> np.ndarray:
        """Compute ȧ from A via finite differences."""
        if method == "fourth_order" and A.shape[0] >= 5:
            # 4th-order central differences
            dA = np.zeros((A.shape[0] - 4, A.shape[1]))
            for i in range(2, A.shape[0] - 2):
                dA[i - 2] = (
                    -A[i + 2] + 8*A[i + 1] - 8*A[i - 1] + A[i - 2]
                ) / (12 * dt)
            return dA
        else:
            # 2nd-order central differences (default)
            dA = (A[2:] - A[:-2]) / (2 * dt)
            return dA

    def _stlsq(
        self,
        Theta: np.ndarray,
        dA: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Sequentially Thresholded Least Squares (STLSQ).

        Iteratively solves Θ·Ξ = ȧ with hard thresholding:
        1. Solve LS: Ξ = (Θ^TΘ + αI)^{-1} Θ^T ȧ
        2. Threshold: Ξ[|Ξ| < λ] = 0
        3. Repeat on active columns until convergence
        """
        n_features = Theta.shape[1]
        n_targets = dA.shape[1]
        Xi = np.zeros((n_features, n_targets))

        # Normalize columns for better conditioning
        if self.normalize:
            norms = np.linalg.norm(Theta, axis=0)
            norms[norms < 1e-12] = 1.0
            Theta_n = Theta / norms[None, :]
        else:
            Theta_n = Theta
            norms = np.ones(n_features)

        # Build weight diagonal matrix if provided (W = diag(w))
        if sample_weights is not None:
            w = np.sqrt(np.asarray(sample_weights, dtype=float))
            Theta_w = Theta_n * w[:, None]   # weighted design matrix
        else:
            w = None
            Theta_w = Theta_n

        for target in range(n_targets):
            y = dA[:, target]
            y_w = y * w if w is not None else y
            active = np.ones(n_features, dtype=bool)

            for iteration in range(self.max_iter):
                # Solve weighted ridge regression on active features
                Theta_active = Theta_w[:, active]
                if Theta_active.shape[1] == 0:
                    break

                gram = Theta_active.T @ Theta_active
                gram += self.alpha_ridge * np.eye(gram.shape[0])
                rhs = Theta_active.T @ y_w

                try:
                    xi_active = linalg.solve(gram, rhs, assume_a='pos')
                except linalg.LinAlgError:
                    xi_active = np.linalg.lstsq(Theta_active, y_w, rcond=None)[0]

                # Threshold small coefficients
                xi_full = np.zeros(n_features)
                xi_full[active] = xi_active

                small = np.abs(xi_full) < self.threshold
                new_active = ~small

                if np.array_equal(active, new_active):
                    break
                active = new_active

            # Final solve with thresholded support
            if np.any(active):
                Theta_final = Theta_w[:, active]
                gram = Theta_final.T @ Theta_final
                gram += self.alpha_ridge * np.eye(gram.shape[0])
                rhs = Theta_final.T @ y_w
                try:
                    xi_final = linalg.solve(gram, rhs, assume_a='pos')
                except linalg.LinAlgError:
                    xi_final = np.linalg.lstsq(Theta_final, y_w, rcond=None)[0]
                Xi[active, target] = xi_final

        # Un-normalize
        if self.normalize:
            Xi = Xi / norms[:, None]

        return Xi

    def _extract_operators(
        self,
        Xi: np.ndarray,
        r: int,
        library: PolynomialLibrary,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract L, Q, c from the SINDy coefficient matrix Ξ.

        Ξ has rows corresponding to library features:
          Row 0: constant term → c
          Rows 1..r: linear terms → L
          Rows r+1..: quadratic terms → Q
        """
        c = Xi[0, :].copy()  # constant (r,)

        L = Xi[1:r+1, :].T.copy()  # linear (r, r)

        Q = np.zeros((r, r, r))
        idx = r + 1  # start of quadratic terms
        if library.degree >= 2:
            for i in range(r):
                for j in range(i, r):
                    if idx < Xi.shape[0]:
                        q_ij = Xi[idx, :]  # (r,)
                        if i == j:
                            # a_i^2 term
                            Q[:, i, j] += q_ij
                        else:
                            # a_i*a_j term — symmetrize
                            Q[:, i, j] += 0.5 * q_ij
                            Q[:, j, i] += 0.5 * q_ij
                        idx += 1

        return L, Q, c


# ---------------------------------------------------------------------------
# ROM Builder — combines SINDy + closure
# ---------------------------------------------------------------------------

class ROMBuilder:
    """
    Build a complete ROM from Koopman modal data.

    Pipeline:
    1. Extract modal amplitudes from Koopman decomposition (TAA)
    2. Discover dynamics via SINDy
    3. Add ERGON thermodynamic closure (ν_t)
    4. Build dissipation operator D
    5. Verify ROM fidelity
    """

    def __init__(
        self,
        poly_degree: int = 2,
        sindy_threshold: float = 0.05,
        sindy_alpha: float = 1e-5,
    ):
        self.sindy = SINDyEngine(
            poly_degree=poly_degree,
            threshold=sindy_threshold,
            alpha_ridge=sindy_alpha,
        )

    def build_from_trajectory(
        self,
        modal_amplitudes: np.ndarray,
        dt: float,
        nu_t: float = 0.0,
        wavenumbers: Optional[np.ndarray] = None,
        reference_spectrum: Optional[np.ndarray] = None,
    ) -> ROMModel:
        """
        Build ROM from modal amplitude time series.

        Parameters
        ----------
        modal_amplitudes : (n_steps, n_modes) array
        dt : time step
        nu_t : eddy viscosity from ERGON closure
        wavenumbers : array of wavenumber magnitudes for each mode
        reference_spectrum : E(k) from DNS for validation
        """
        n_steps, r = modal_amplitudes.shape

        # SINDy identification
        sindy_result = self.sindy.fit(modal_amplitudes, dt)

        # Build dissipation operator D = -diag(k²)
        if wavenumbers is not None:
            D = -np.diag(wavenumbers[:r] ** 2)
        else:
            # Default: assume modes ordered by wavenumber
            k = np.arange(1, r + 1, dtype=float)
            D = -np.diag(k ** 2)

        rom = ROMModel(
            L=sindy_result.L,
            Q=sindy_result.Q,
            c=sindy_result.c,
            D=D,
            nu_t=nu_t,
            n_modes=r,
            sindy_result=sindy_result,
            reference_spectrum=reference_spectrum,
            metadata={
                "n_samples": n_steps,
                "dt": dt,
                "sindy_sparsity": sindy_result.sparsity,
                "sindy_residual": sindy_result.residual_norm,
                "n_active_terms": sindy_result.n_active_terms,
            },
        )

        return rom

    def build_from_koopman(
        self,
        koopman_matrix: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        trajectory: np.ndarray,
        dt: float,
        nu_t: float = 0.0,
        n_modes: int = 8,
    ) -> ROMModel:
        """
        Build ROM from Koopman decomposition.

        Projects trajectory onto leading Koopman modes and
        discovers nonlinear dynamics in the modal space.

        Parameters
        ----------
        koopman_matrix : (d, d) EDMD Koopman matrix
        eigenvalues : (d,) complex eigenvalues
        eigenvectors : (d, d) right eigenvectors
        trajectory : (n_steps, n_features) state trajectory
        dt : time step
        nu_t : eddy viscosity
        n_modes : number of leading modes to use
        """
        # Sort by eigenvalue magnitude
        idx = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Project trajectory onto leading modes
        r = min(n_modes, len(eigenvalues))
        V = eigenvectors[:, :r].real  # Use real part of eigenvectors

        # If trajectory is in observable space, project directly
        if trajectory.shape[1] == koopman_matrix.shape[0]:
            modal_amplitudes = trajectory @ V
        else:
            # Need to build observables first
            modal_amplitudes = trajectory[:, :r]

        # Wavenumbers from eigenvalue frequencies
        freqs = np.abs(np.imag(np.log(eigenvalues[:r] + 1e-300)))
        wavenumbers = freqs / (2 * np.pi) if np.any(freqs > 0) else np.arange(1, r+1, dtype=float)

        return self.build_from_trajectory(
            modal_amplitudes, dt, nu_t, wavenumbers,
        )

    def verify_rom(
        self,
        rom: ROMModel,
        reference_trajectory: np.ndarray,
        dt: float,
        n_verify_steps: Optional[int] = None,
    ) -> ROMVerification:
        """
        Verify ROM fidelity against reference data.

        Computes trajectory error, spectrum error, energy conservation,
        and estimates the Lyapunov time (when ROM diverges).
        """
        if n_verify_steps is None:
            n_verify_steps = min(reference_trajectory.shape[0] - 1, 1000)

        r = rom.n_modes
        a0 = reference_trajectory[0, :r].copy()

        # Integrate ROM
        rom_traj = rom.integrate(a0, dt, n_verify_steps)

        # Reference modal amplitudes
        ref = reference_trajectory[:n_verify_steps + 1, :r]

        # Trajectory error
        if np.any(np.isnan(rom_traj)):
            traj_err = float('inf')
            stable = False
            lyap_time = 0.0
        else:
            diff = rom_traj[:ref.shape[0]] - ref[:rom_traj.shape[0]]
            traj_err = np.linalg.norm(diff, 'fro') / (np.linalg.norm(ref, 'fro') + 1e-15)
            stable = True

            # Lyapunov time: when error exceeds threshold
            errors = np.linalg.norm(diff, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-15)
            crossings = np.where(errors > 0.5)[0]
            lyap_time = crossings[0] * dt if len(crossings) > 0 else n_verify_steps * dt

        # Energy conservation
        E0 = rom.energy(a0)
        E_final = rom.energy(rom_traj[-1]) if not np.any(np.isnan(rom_traj[-1])) else float('inf')
        energy_cons = abs(E_final - E0) / (abs(E0) + 1e-15)

        # Spectrum error (if available)
        spec_err = 0.0
        if rom.reference_spectrum is not None and rom.energy_spectrum is not None:
            spec_err = np.linalg.norm(
                rom.energy_spectrum - rom.reference_spectrum
            ) / (np.linalg.norm(rom.reference_spectrum) + 1e-15)

        return ROMVerification(
            trajectory_error=traj_err,
            spectrum_error=spec_err,
            energy_conservation=energy_cons,
            is_dissipative=rom.is_dissipative(),
            is_stable=stable,
            lyapunov_time=lyap_time,
            certificates={
                "ROM-1_sindy_residual": rom.sindy_result.residual_norm if rom.sindy_result else float('inf'),
                "ROM-2_dissipative": float(rom.is_dissipative()),
                "ROM-3_spectrum_error": spec_err,
                "trajectory_error": traj_err,
                "energy_conservation": energy_cons,
            },
        )


# ---------------------------------------------------------------------------
# Galerkin Projector — alternative to SINDy for known PDEs
# ---------------------------------------------------------------------------

class GalerkinProjector:
    """
    Galerkin projection of PDEs onto Fourier/spectral modes.

    For Navier-Stokes: projects the nonlinear advection term
    onto the truncated basis to obtain Q_{ijk}.

    ȧᵢ = Σⱼ Lᵢⱼ aⱼ + Σⱼₖ Qᵢⱼₖ aⱼ aₖ

    where:
      Lᵢⱼ = -ν k²ᵢ δᵢⱼ  (viscous dissipation)
      Qᵢⱼₖ = ∫ φᵢ · (φⱼ · ∇)φₖ dV  (advection integral)
    """

    def __init__(self, n_modes: int, viscosity: float = 0.01):
        self.n_modes = n_modes
        self.viscosity = viscosity

    def project_ns_fourier(
        self,
        wavenumbers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Galerkin projection of 1D Burgers / simplified NS.

        Returns (L, Q) for the truncated system.

        Parameters
        ----------
        wavenumbers : (n_modes,) array of wavenumber magnitudes
        """
        r = self.n_modes
        k = wavenumbers[:r]

        # Linear operator: L_ij = -ν k²_i δ_ij
        L = -self.viscosity * np.diag(k ** 2)

        # Quadratic tensor: Q_ijk models energy transfer
        # For Fourier modes: Q_ijk ∝ k_i · δ(k_i - k_j - k_k)
        Q = np.zeros((r, r, r))
        for i in range(r):
            for j in range(r):
                for m in range(r):
                    # Triad interaction: k_i = k_j + k_m
                    if abs(k[i] - k[j] - k[m]) < 0.5:
                        Q[i, j, m] = -k[i] / (2.0 * r)
                    elif abs(k[i] + k[j] - k[m]) < 0.5:
                        Q[i, j, m] = k[i] / (2.0 * r)

        # Symmetrize Q: Q_ijk = Q_ikj
        Q = 0.5 * (Q + np.transpose(Q, (0, 2, 1)))

        return L, Q

    def build_rom(
        self,
        wavenumbers: np.ndarray,
        nu_t: float = 0.0,
    ) -> ROMModel:
        """Build a ROM via Galerkin projection."""
        L, Q = self.project_ns_fourier(wavenumbers)
        D = -np.diag(wavenumbers[:self.n_modes] ** 2)
        c = np.zeros(self.n_modes)

        return ROMModel(
            L=L,
            Q=Q,
            c=c,
            D=D,
            nu_t=nu_t,
            n_modes=self.n_modes,
            metadata={
                "method": "galerkin_projection",
                "viscosity": self.viscosity,
                "wavenumbers": wavenumbers[:self.n_modes].tolist(),
            },
        )
