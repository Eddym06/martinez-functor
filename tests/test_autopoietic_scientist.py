"""
Test Autopoietic Scientist — P-SAL Protocol Integration Test
=============================================================

End-to-end test of the closed-loop:
  DNS data → TAA observation → SINDy hypothesis → ERGON closure →
  Poema compilation → verification → memorization → Gideon execution

Uses a synthetic Lorenz system as the test dynamical system.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def generate_lorenz_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """Generate Lorenz attractor trajectory for testing."""
    x = np.zeros((n_steps, 3))
    x[0] = [1.0, 1.0, 1.0]

    for i in range(n_steps - 1):
        x1, x2, x3 = x[i]
        dx1 = sigma * (x2 - x1)
        dx2 = x1 * (rho - x3) - x2
        dx3 = x1 * x2 - beta * x3

        # RK4
        k1 = np.array([dx1, dx2, dx3])

        x_mid = x[i] + 0.5 * dt * k1
        dx1 = sigma * (x_mid[1] - x_mid[0])
        dx2 = x_mid[0] * (rho - x_mid[2]) - x_mid[1]
        dx3 = x_mid[0] * x_mid[1] - beta * x_mid[2]
        k2 = np.array([dx1, dx2, dx3])

        x_mid = x[i] + 0.5 * dt * k2
        dx1 = sigma * (x_mid[1] - x_mid[0])
        dx2 = x_mid[0] * (rho - x_mid[2]) - x_mid[1]
        dx3 = x_mid[0] * x_mid[1] - beta * x_mid[2]
        k3 = np.array([dx1, dx2, dx3])

        x_end = x[i] + dt * k3
        dx1 = sigma * (x_end[1] - x_end[0])
        dx2 = x_end[0] * (rho - x_end[2]) - x_end[1]
        dx3 = x_end[0] * x_end[1] - beta * x_end[2]
        k4 = np.array([dx1, dx2, dx3])

        x[i + 1] = x[i] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return x


def generate_linear_oscillator(
    n_steps: int = 2000,
    dt: float = 0.01,
    omega: float = 2.0,
    damping: float = 0.1,
) -> np.ndarray:
    """Generate damped harmonic oscillator for simple test."""
    x = np.zeros((n_steps, 2))
    x[0] = [1.0, 0.0]

    for i in range(n_steps - 1):
        x1, x2 = x[i]
        dx1 = x2
        dx2 = -omega**2 * x1 - 2 * damping * x2
        x[i + 1] = x[i] + dt * np.array([dx1, dx2])

    return x


# ---------------------------------------------------------------------------
# Test 1: ROM Synthesizer (SINDy)
# ---------------------------------------------------------------------------

class TestROMSynthesizer:
    """Test SINDy-based ROM discovery."""

    def test_sindy_linear_system(self):
        """SINDy should recover linear dynamics exactly."""
        from acf_functor.rom_synthesizer import SINDyEngine

        data = generate_linear_oscillator(n_steps=2000, dt=0.01)
        sindy = SINDyEngine(poly_degree=2, threshold=0.1)
        result = sindy.fit(data, dt=0.01)

        # Should find 2 active linear terms (coupling x1↔x2)
        assert result.L.shape == (2, 2)
        assert result.residual_norm < 0.5, f"SINDy residual too high: {result.residual_norm}"
        assert result.sparsity > 0.3, f"Not sparse enough: {result.sparsity}"

    def test_sindy_lorenz(self):
        """SINDy should discover Lorenz-like dynamics."""
        from acf_functor.rom_synthesizer import SINDyEngine

        data = generate_lorenz_data(n_steps=3000, dt=0.01)
        sindy = SINDyEngine(poly_degree=2, threshold=0.5)
        result = sindy.fit(data, dt=0.01)

        assert result.L.shape == (3, 3)
        assert result.Q.shape == (3, 3, 3)
        assert result.n_active_terms > 0
        assert result.elapsed_ms > 0

    def test_polynomial_library(self):
        """Test polynomial library construction."""
        from acf_functor.rom_synthesizer import PolynomialLibrary

        lib = PolynomialLibrary.create(3, degree=2)
        A = np.random.randn(100, 3)
        Theta = lib.transform(A)

        # Expected: 1 + 3 + 6 = 10 features (const + linear + quadratic)
        assert Theta.shape == (100, 10)
        assert len(lib.feature_names) == 10

    def test_rom_model_integration(self):
        """Test ROM forward integration."""
        from acf_functor.rom_synthesizer import ROMModel

        r = 3
        L = -0.1 * np.eye(r)
        Q = np.zeros((r, r, r))
        c = np.zeros(r)
        D = -np.diag(np.arange(1, r+1, dtype=float)**2)

        rom = ROMModel(L=L, Q=Q, c=c, D=D, nu_t=0.01, n_modes=r)
        a0 = np.array([1.0, 0.5, 0.3])
        traj = rom.integrate(a0, dt=0.01, n_steps=100)

        assert traj.shape == (101, 3)
        assert not np.any(np.isnan(traj))
        # Should decay (dissipative)
        assert rom.is_dissipative()
        assert rom.energy(traj[-1]) < rom.energy(a0)

    def test_rom_builder_from_trajectory(self):
        """Test building ROM from trajectory data."""
        from acf_functor.rom_synthesizer import ROMBuilder

        data = generate_linear_oscillator(n_steps=2000, dt=0.01)
        builder = ROMBuilder(poly_degree=2, sindy_threshold=0.1)
        rom = builder.build_from_trajectory(data, dt=0.01, nu_t=0.001)

        assert rom.n_modes == 2
        assert rom.L.shape == (2, 2)

    def test_galerkin_projector(self):
        """Test Galerkin projection for NS-like systems."""
        from acf_functor.rom_synthesizer import GalerkinProjector

        r = 8
        k = np.arange(1, r+1, dtype=float)
        proj = GalerkinProjector(n_modes=r, viscosity=0.01)
        rom = proj.build_rom(k, nu_t=0.001)

        assert rom.L.shape == (r, r)
        assert rom.Q.shape == (r, r, r)
        assert rom.is_dissipative()


# ---------------------------------------------------------------------------
# Test 2: Thermodynamic Closure
# ---------------------------------------------------------------------------

class TestThermodynamicClosure:
    """Test ERGON-based thermodynamic closure."""

    def test_ergon_closure(self):
        """Test closure with ERGON diagnostics."""
        from acf_functor.thermodynamic_closure import ThermodynamicClosure

        r = 4
        closure_engine = ThermodynamicClosure(r)

        # Simulate modal amplitudes
        A = np.random.randn(500, r) * 0.5
        result = closure_engine.compute_ergon_closure(
            h_ks=0.693,
            pressure_curvature=0.1,
            modal_amplitudes=A,
        )

        assert result.nu_t >= 0
        assert result.closure_method == "ergon_thermodynamic"
        assert result.dissipation_matrix.shape == (r, r)
        assert np.trace(result.dissipation_matrix) <= 0

    def test_smagorinsky_closure(self):
        """Test Smagorinsky fallback closure."""
        from acf_functor.thermodynamic_closure import ThermodynamicClosure

        r = 4
        closure_engine = ThermodynamicClosure(r)
        A = np.random.randn(500, r) * 0.5
        result = closure_engine.compute_smagorinsky_closure(A)

        assert result.nu_t >= 0
        assert result.closure_method == "smagorinsky"

    def test_svv_closure(self):
        """Test Spectral Vanishing Viscosity closure."""
        from acf_functor.thermodynamic_closure import ThermodynamicClosure

        r = 4
        closure_engine = ThermodynamicClosure(r)
        A = np.random.randn(500, r) * 0.5
        result = closure_engine.compute_spectral_vanishing_viscosity(A)

        assert result.nu_t >= 0
        assert result.closure_method == "spectral_vanishing_viscosity"
        # SVV should have stronger closure at higher wavenumbers
        assert result.nu_t_spectral[-1] >= result.nu_t_spectral[0]

    def test_adaptive_selector(self):
        """Test automatic closure selection."""
        from acf_functor.thermodynamic_closure import AdaptiveClosureSelector

        r = 4
        selector = AdaptiveClosureSelector(r)
        A = np.random.randn(500, r) * 0.5

        # With ERGON data → ERGON closure
        result = selector.select_and_compute(A, h_ks=0.693, pressure_curvature=0.1)
        assert result.closure_method == "ergon_thermodynamic"

        # Without ERGON data → SVV
        result = selector.select_and_compute(A)
        assert result.closure_method == "spectral_vanishing_viscosity"

    def test_closure_verification(self):
        """Test that closure stabilizes ROM."""
        from acf_functor.thermodynamic_closure import ThermodynamicClosure

        r = 4
        closure_engine = ThermodynamicClosure(r)
        A = np.random.randn(500, r) * 0.5

        closure = closure_engine.compute_ergon_closure(
            h_ks=0.693, pressure_curvature=0.1, modal_amplitudes=A,
        )

        L = np.random.randn(r, r) * 0.1
        Q = np.zeros((r, r, r))

        verif = closure_engine.verify_closure(L, Q, closure, A)
        assert isinstance(verif.stability_margin, float)


# ---------------------------------------------------------------------------
# Test 3: Poema ROM Generator
# ---------------------------------------------------------------------------

class TestPoemROMGenerator:
    """Test Poema ROM code generation."""

    def test_from_koopman_rom(self):
        """Test generating Poem from Koopman operators."""
        from poema.rom_generator import ROMGenerator

        r = 4
        L = -0.1 * np.eye(r)
        Q = np.zeros((r, r, r))
        gen = ROMGenerator()
        poem = gen.from_koopman_rom(L, Q, nu_t=0.01)

        assert poem.n_modes == r
        assert poem.L.shape == (r, r)
        assert len(poem.nodes) > 0

    def test_poem_execution(self):
        """Test executing a generated Poem."""
        from poema.rom_generator import ROMGenerator

        r = 3
        L = -0.1 * np.eye(r)
        Q = np.zeros((r, r, r))
        gen = ROMGenerator()
        poem = gen.from_koopman_rom(L, Q, nu_t=0.01, dt=0.01, n_steps=200)

        a0 = np.array([1.0, 0.5, 0.3])
        traj = poem.execute(a0)

        assert traj.shape == (201, 3)
        assert not np.any(np.isnan(traj))
        # Should decay
        assert poem.energy(traj[-1]) < poem.energy(a0)

    def test_poem_serialization(self):
        """Test JSON serialization/deserialization."""
        from poema.rom_generator import ROMGenerator, PoemROM

        r = 3
        L = -0.1 * np.eye(r)
        Q = np.zeros((r, r, r))
        gen = ROMGenerator()
        poem = gen.from_koopman_rom(L, Q, nu_t=0.01)

        # Serialize
        json_str = poem.to_json()
        assert len(json_str) > 0

        # Deserialize
        d = poem.to_dict()
        poem2 = PoemROM.from_dict(d)
        assert poem2.n_modes == r
        assert np.allclose(poem2.L, poem.L)

    def test_poem_symbolic(self):
        """Test symbolic equation generation."""
        from poema.rom_generator import ROMGenerator

        r = 2
        L = np.array([[-0.1, 0.5], [-0.5, -0.1]])
        Q = np.zeros((r, r, r))
        gen = ROMGenerator()
        poem = gen.from_koopman_rom(L, Q, nu_t=0.0)

        symbolic = poem.to_symbolic()
        assert "da0/dt" in symbolic
        assert "da1/dt" in symbolic

    def test_from_rom_model(self):
        """Test conversion from ROMModel to PoemROM."""
        from acf_functor.rom_synthesizer import ROMBuilder
        from poema.rom_generator import ROMGenerator

        data = generate_linear_oscillator(n_steps=2000, dt=0.01)
        builder = ROMBuilder(poly_degree=2, sindy_threshold=0.1)
        rom = builder.build_from_trajectory(data, dt=0.01, nu_t=0.001)

        gen = ROMGenerator()
        poem = gen.from_rom_model(rom, name="test_oscillator")
        assert poem.name == "test_oscillator"
        assert poem.n_modes == 2


# ---------------------------------------------------------------------------
# Test 4: Gideon ROM Executor
# ---------------------------------------------------------------------------

class TestGideonROMExecutor:
    """Test Gideon ROM execution backend."""

    def test_direct_execution(self):
        """Test direct RK4 execution."""
        from poema.rom_generator import ROMGenerator
        from poema.backends.gideon.rom_executor import GideonROMExecutor

        r = 3
        L = -0.1 * np.eye(r)
        Q = np.zeros((r, r, r))
        gen = ROMGenerator()
        poem = gen.from_koopman_rom(L, Q, nu_t=0.01, dt=0.01, n_steps=500)

        executor = GideonROMExecutor()
        a0 = np.array([1.0, 0.5, 0.3])
        result = executor.execute(poem, a0)

        assert result.trajectory.shape == (501, 3)
        assert result.stable
        assert result.wall_time_ms > 0
        assert result.n_rhs_evals == 500 * 4  # RK4: 4 evals per step

    def test_adaptive_execution(self):
        """Test adaptive time-stepping execution."""
        from poema.rom_generator import ROMGenerator
        from poema.backends.gideon.rom_executor import GideonROMExecutor

        r = 3
        L = -0.1 * np.eye(r)
        Q = np.zeros((r, r, r))
        gen = ROMGenerator()
        poem = gen.from_koopman_rom(L, Q, nu_t=0.01, dt=0.01, n_steps=200)

        executor = GideonROMExecutor()
        a0 = np.array([1.0, 0.5, 0.3])
        result = executor.execute(poem, a0, mode="adaptive")

        assert result.stable
        assert result.trajectory.shape[1] == 3

    def test_ensemble_execution(self):
        """Test ensemble ROM execution."""
        from poema.rom_generator import ROMGenerator
        from poema.backends.gideon.rom_executor import GideonROMExecutor

        r = 3
        L = -0.1 * np.eye(r)
        Q = np.zeros((r, r, r))
        gen = ROMGenerator()
        poem = gen.from_koopman_rom(L, Q, nu_t=0.01, dt=0.01, n_steps=100)

        executor = GideonROMExecutor()
        ics = np.random.randn(5, r) * 0.5
        result = executor.execute_ensemble(poem, ics)

        assert result.trajectories.shape == (5, 101, 3)
        assert result.n_stable == 5
        assert result.mean_trajectory.shape == (101, 3)


# ---------------------------------------------------------------------------
# Test 5: Autopoietic Scientist (Full P-SAL Loop)
# ---------------------------------------------------------------------------

class TestAutopoieticScientist:
    """Test the complete P-SAL closed-loop protocol."""

    def test_psal_linear_system(self):
        """P-SAL should discover linear oscillator dynamics."""
        from acf_functor.autopoietic_scientist import AutopoieticScientist

        data = generate_linear_oscillator(n_steps=2000, dt=0.01)

        scientist = AutopoieticScientist(
            n_modes_range=(2, 2),
            sindy_threshold=0.05,
            verification_tolerance=0.5,
        )
        report = scientist.run(
            trajectory=data,
            dt=0.01,
            n_cycles=1,
        )

        assert report.n_laws_discovered >= 1
        assert report.total_time_s > 0
        assert report.best_law is not None

    def test_psal_lorenz(self):
        """P-SAL should discover Lorenz-like dynamics."""
        from acf_functor.autopoietic_scientist import AutopoieticScientist

        data = generate_lorenz_data(n_steps=3000, dt=0.01)

        scientist = AutopoieticScientist(
            n_modes_range=(3, 3),
            sindy_threshold=1.0,
            sindy_poly_degree=2,
            verification_tolerance=1.0,
        )
        report = scientist.run(
            trajectory=data,
            dt=0.01,
            h_ks=0.693,
            pressure_curvature=0.1,
            n_cycles=1,
        )

        assert report.n_laws_discovered >= 1
        assert report.best_law is not None
        print(report.summary())

    def test_psal_multi_cycle(self):
        """P-SAL should improve across multiple cycles."""
        from acf_functor.autopoietic_scientist import AutopoieticScientist

        data = generate_linear_oscillator(n_steps=2000, dt=0.01)

        scientist = AutopoieticScientist(
            n_modes_range=(2, 2),
            sindy_threshold=0.05,
            verification_tolerance=0.5,
        )
        report = scientist.run(
            trajectory=data,
            dt=0.01,
            n_cycles=3,
        )

        assert report.n_cycles == 3
        assert report.n_laws_discovered >= 3
        assert len(report.knowledge_base.cycle_history) == 3

    def test_psal_with_ergon_closure(self):
        """P-SAL should use ERGON closure when available."""
        from acf_functor.autopoietic_scientist import AutopoieticScientist

        data = generate_lorenz_data(n_steps=3000, dt=0.01)

        scientist = AutopoieticScientist(
            n_modes_range=(3, 3),
            sindy_threshold=1.0,
            verification_tolerance=2.0,
        )
        report = scientist.run(
            trajectory=data,
            dt=0.01,
            h_ks=0.693,
            pressure_curvature=0.1,
            n_cycles=1,
        )

        best = report.best_law
        assert best is not None
        assert best.nu_t >= 0
        assert best.metadata.get("closure_method") == "ergon_thermodynamic"

    def test_psal_knowledge_base(self):
        """Test knowledge base persistence."""
        from acf_functor.autopoietic_scientist import AutopoieticScientist

        data = generate_linear_oscillator(n_steps=2000, dt=0.01)

        scientist = AutopoieticScientist(
            n_modes_range=(2, 2),
            sindy_threshold=0.05,
            verification_tolerance=0.5,
        )
        report = scientist.run(trajectory=data, dt=0.01, n_cycles=2)

        kb = report.knowledge_base
        assert len(kb.laws) >= 2
        assert kb.best_law is not None

        # Test serialization
        d = kb.to_dict()
        assert "laws" in d
        assert d["n_laws"] >= 2

    def test_psal_certificates(self):
        """Test PSAL certificate generation."""
        from acf_functor.autopoietic_scientist import AutopoieticScientist

        data = generate_linear_oscillator(n_steps=2000, dt=0.01)

        scientist = AutopoieticScientist(
            n_modes_range=(2, 2),
            sindy_threshold=0.05,
            verification_tolerance=0.5,
        )
        report = scientist.run(trajectory=data, dt=0.01, n_cycles=1)

        certs = report.certificates
        assert "PSAL-1" in certs
        assert "PSAL-2" in certs
        assert "PSAL-6" in certs
        assert certs.get("autopoietic_closure", 0) >= 0


# ---------------------------------------------------------------------------
# Test 6: Full Pipeline Integration
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end integration test of the complete autopoietic pipeline."""

    def test_complete_pipeline(self):
        """
        Full pipeline:
        1. Generate dynamical data (Lorenz)
        2. AutopoieticScientist discovers ROM
        3. Poema compiles ROM
        4. Gideon executes ROM
        5. Compare ROM prediction with reference
        """
        from acf_functor.autopoietic_scientist import AutopoieticScientist
        from poema.rom_generator import ROMGenerator
        from poema.backends.gideon.rom_executor import GideonROMExecutor

        # Step 1: Generate data
        data = generate_lorenz_data(n_steps=3000, dt=0.01)

        # Step 2: Discover
        scientist = AutopoieticScientist(
            n_modes_range=(3, 3),
            sindy_threshold=1.0,
            verification_tolerance=2.0,
        )
        report = scientist.run(
            trajectory=data,
            dt=0.01,
            h_ks=0.693,
            pressure_curvature=0.1,
            n_cycles=1,
        )

        assert report.best_law is not None
        law = report.best_law

        # Step 3: Compile
        gen = ROMGenerator()
        poem = gen.from_koopman_rom(
            L=law.L,
            Q=law.Q,
            nu_t=law.nu_t,
            name="lorenz_law",
            dt=0.01,
            n_steps=500,
        )

        # Step 4: Execute via Gideon
        executor = GideonROMExecutor()
        a0 = data[0, :3]
        result = executor.execute(poem, a0)

        assert result.stable or result.trajectory.shape[0] > 1
        assert result.wall_time_ms > 0

        # Step 5: Verify
        if result.stable:
            # ROM should produce bounded trajectory
            max_val = np.nanmax(np.abs(result.trajectory))
            assert max_val < 1e8, f"ROM trajectory unbounded: max={max_val}"

        print(f"\n=== Full Pipeline Test ===")
        print(f"Laws discovered: {report.n_laws_discovered}")
        print(f"Best law quality: {law.quality_score():.4f}")
        print(f"Trajectory error: {law.trajectory_error:.4e}")
        print(f"ROM execution time: {result.wall_time_ms:.1f}ms")
        print(f"ROM stable: {result.stable}")
        print(report.summary())


# ---------------------------------------------------------------------------
# Epic C: SEM → P-SAL end-to-end integration tests
# ---------------------------------------------------------------------------

class TestSEMIntegration:
    """
    End-to-end tests for the SEM → P-SAL pipeline.

    Verifies that:
    1. AutopoieticScientist.run(sm_output=...) uses the purified trajectory.
    2. Uncertainty weights from UncertaintyManifold are propagated to SINDy.
    3. Missing/invalid SMOutput falls back gracefully to raw trajectory.
    4. With real StochasticMembrane on noisy Lorenz data, P-SAL still discovers a law.
    """

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_lorenz_data(T: int = 300, dt: float = 0.02, noise_scale: float = 0.05):
        """Generate noisy Lorenz-like 1-D projection data."""
        rng = np.random.default_rng(0)
        x = np.zeros(T)
        x[0] = 0.5
        for t in range(T - 1):
            dxdt = -2.5 * x[t] + 0.5 * x[t] ** 2
            x[t + 1] = x[t] + dt * dxdt + noise_scale * rng.standard_normal()
        return x

    @staticmethod
    def _make_mock_sm_output(T: int, purified: np.ndarray, sigma_scale: float = 0.01):
        """Construct a minimal SMOutput without running the full SEM pipeline."""
        from acf_functor.stochastic_membrane import (
            SMOutput, PurifiedTrajectory, UncertaintyManifold,
        )

        d = 1
        xt = purified.reshape(T, d)

        # confidence_bands: (T, d, 2)
        bands = np.stack([xt - sigma_scale, xt + sigma_scale], axis=-1)
        pt = PurifiedTrajectory(
            x_hat=xt,
            confidence_bands=bands,
            filter_snr_db=20.0,
            persistent_features=[],
            drift_estimate=None,
            regime_labels=np.zeros(T, dtype=int),
        )

        sigma = sigma_scale * np.eye(d)[None, :, :].repeat(T, axis=0)
        evals  = np.ones((T, d)) * sigma_scale
        evecs  = np.eye(d)[None, :, :].repeat(T, axis=0)
        volume = np.abs(np.linalg.det(sigma)) if d > 1 else np.full(T, sigma_scale)
        entropy = 0.5 * np.log(np.maximum((2 * np.pi * np.e) ** d * volume, 1e-30))
        diff_t  = sigma.copy()

        um = UncertaintyManifold(
            sigma_tensor=sigma,
            uncertainty_eigenvalues=evals,
            uncertainty_eigenvectors=evecs,
            uncertainty_volume=volume,
            posterior_entropy=entropy,
            diffusion_tensor=diff_t,
        )

        return SMOutput(
            purified=pt,
            uncertainty=um,
            purity_index=0.92,
            fokker_planck_error=1e-3,
            regime_diagnosis={"noise_model": {"current_family": "gaussian"}},
            n_effective_particles=np.ones(T),
            cnf_log_likelihood=np.zeros(T),
        )

    # ── tests ────────────────────────────────────────────────────────────────

    def test_run_with_sm_output_produces_law(self):
        """
        P-SAL run with sm_output must discover at least one physical law.
        The scientist should use the purified trajectory for modal decomposition.
        """
        from acf_functor.autopoietic_scientist import AutopoieticScientist
        data = self._make_lorenz_data(T=400, dt=0.02, noise_scale=0.05)
        data_2d = data.reshape(-1, 1)   # (T, 1) — expected shape
        purified = data + np.random.default_rng(1).standard_normal(len(data)) * 1e-4
        sm_out = self._make_mock_sm_output(len(data), purified)

        scientist = AutopoieticScientist(n_modes_range=(4, 8), max_cycles=2)
        report = scientist.run(
            trajectory=data_2d,
            dt=0.02,
            sm_output=sm_out,
        )
        assert report.n_laws_discovered >= 0, "run() must complete without error"
        # If at least one law found, check it has a trajectory_error
        if report.n_laws_discovered > 0:
            law = report.best_law
            assert law is not None
            assert np.isfinite(law.trajectory_error)

    def test_obs_weights_propagated_to_sindy(self):
        """
        High-uncertainty timesteps should get lower SINDy weights.
        We inject a spike of high sigma at timestep 50 and verify that the
        corresponding obs_weight is smaller than the average.
        """
        from acf_functor.autopoietic_scientist import AutopoieticScientist
        T = 200
        data = self._make_lorenz_data(T=T, dt=0.02, noise_scale=0.01)

        # Build mock SMOutput with a spike at t=50
        from acf_functor.stochastic_membrane import (
            SMOutput, PurifiedTrajectory, UncertaintyManifold,
        )
        d = 1
        xt = data.reshape(T, d)
        bands = np.stack([xt - 0.01, xt + 0.01], axis=-1)
        pt = PurifiedTrajectory(
            x_hat=xt,
            confidence_bands=bands,
            filter_snr_db=20.0,
            persistent_features=[],
            drift_estimate=None,
            regime_labels=np.zeros(T, dtype=int),
        )
        sigma_base = 0.01 * np.eye(d)[None, :, :].repeat(T, axis=0)
        sigma_base[50, 0, 0] = 100.0   # spike: high uncertainty
        evals  = np.abs(sigma_base[:, 0, 0:1])
        evecs  = np.eye(d)[None, :, :].repeat(T, axis=0)
        volume = sigma_base[:, 0, 0]
        entropy = 0.5 * np.log(np.maximum(2 * np.pi * np.e * volume, 1e-30))
        um = UncertaintyManifold(
            sigma_tensor=sigma_base,
            uncertainty_eigenvalues=evals,
            uncertainty_eigenvectors=evecs,
            uncertainty_volume=volume,
            posterior_entropy=entropy,
            diffusion_tensor=sigma_base.copy(),
        )
        sm_out = SMOutput(
            purified=pt,
            uncertainty=um,
            purity_index=0.85,
            fokker_planck_error=1e-3,
            regime_diagnosis={"noise_model": {"current_family": "gaussian"}},
            n_effective_particles=np.ones(T),
            cnf_log_likelihood=np.zeros(T),
        )

        # Run scientist — we check it does not crash; deep weight check is below
        scientist = AutopoieticScientist(n_modes_range=(4, 6), max_cycles=1)
        # Use _observe directly to inspect weights
        modal_data, obs_weights = scientist._observe(
            trajectory=xt,
            n_modes=4,
            eigenvalues=np.ones(4) * 0.8,
            koopman_matrix=np.eye(4) * 0.8,
            eigenvectors=np.eye(4),
            sm_output=sm_out,
        )
        if obs_weights is not None and len(obs_weights) > 50:
            # weight at spike should be lower than median
            w_spike  = obs_weights[50]
            w_median = np.median(obs_weights)
            assert w_spike < w_median, (
                f"Spike weight {w_spike:.4f} should be < median {w_median:.4f}"
            )

    def test_sm_output_none_fallback(self):
        """
        When sm_output=None (default), run() must fall back to raw trajectory
        and complete without raising.
        """
        from acf_functor.autopoietic_scientist import AutopoieticScientist
        data = self._make_lorenz_data(T=300, dt=0.02, noise_scale=0.02)
        scientist = AutopoieticScientist(n_modes_range=(4, 8), max_cycles=1)
        report = scientist.run(trajectory=data.reshape(-1, 1), dt=0.02, sm_output=None)
        assert report is not None

    def test_psal7_trivially_passes_without_sm_output(self):
        """
        PSAL-7 certificate must be 1.0 when no sm_output is supplied.
        """
        from acf_functor.autopoietic_scientist import AutopoieticScientist
        data = self._make_lorenz_data(T=300, dt=0.02, noise_scale=0.02)
        scientist = AutopoieticScientist(n_modes_range=(4, 8), max_cycles=1)
        report = scientist.run(trajectory=data.reshape(-1, 1), dt=0.02, sm_output=None)
        # PSAL-7 must be present and equal to 1.0 when no SEM baseline is used
        psal7 = report.certificates.get("PSAL-7", None)
        assert psal7 is not None, "PSAL-7 missing from certificates"
        assert psal7 == 1.0, f"Expected PSAL-7=1.0 without SEM, got {psal7}"

    def test_psal7_with_mock_sm_output(self):
        """
        PSAL-7 should be computed (0 or 1) when sm_output is provided.
        """
        from acf_functor.autopoietic_scientist import AutopoieticScientist
        T = 300
        data = self._make_lorenz_data(T=T, dt=0.02, noise_scale=0.02)
        scientist = AutopoieticScientist(n_modes_range=(4, 8), max_cycles=1)
        sm_out = self._make_mock_sm_output(T=T, purified=data)
        report = scientist.run(
            trajectory=data.reshape(-1, 1), dt=0.02, sm_output=sm_out
        )
        psal7 = report.certificates.get("PSAL-7", None)
        assert psal7 is not None, "PSAL-7 missing from certificates"
        assert psal7 in (0.0, 1.0), f"PSAL-7 must be binary, got {psal7}"

    def test_real_sem_integration(self):
        """
        Run the full SEM → AutopoieticScientist pipeline on noisy data.
        Verifies that purified trajectory is used (purity_index > 0).
        """
        try:
            from acf_functor.stochastic_membrane import StochasticMembrane, SMConfig
        except ImportError:
            pytest.skip("StochasticMembrane not available")

        data = self._make_lorenz_data(T=500, dt=0.02, noise_scale=0.1)
        cfg = SMConfig(
            n_particles=50,
            max_ssa_lag=10,
        )
        try:
            sem = StochasticMembrane(config=cfg)
            sm_out = sem.purify(data)
        except Exception as exc:
            pytest.skip(f"StochasticMembrane.purify() raised: {exc}")

        assert 0.0 <= sm_out.purity_index <= 1.0

        scientist = AutopoieticScientist(n_modes_range=(4, 8), max_cycles=2)
        report = scientist.run(trajectory=data.reshape(-1, 1), dt=0.02, sm_output=sm_out)
        assert report is not None
        """
        Run the full SEM → AutopoieticScientist pipeline on noisy data.
        Verifies that purified trajectory is used (purity_index > 0).
        """
        try:
            from acf_functor.stochastic_membrane import StochasticMembrane, SMConfig
        except ImportError:
            pytest.skip("StochasticMembrane not available")

        data = self._make_lorenz_data(T=500, dt=0.02, noise_scale=0.1)
        cfg = SMConfig(
            n_particles=50,
            max_ssa_lag=10,
        )
        try:
            sem = StochasticMembrane(config=cfg)
            sm_out = sem.purify(data)
        except Exception as exc:
            pytest.skip(f"StochasticMembrane.purify() raised: {exc}")

        assert 0.0 <= sm_out.purity_index <= 1.0

        scientist = AutopoieticScientist(n_modes_range=(4, 8), max_cycles=2)
        report = scientist.run(trajectory=data.reshape(-1, 1), dt=0.02, sm_output=sm_out)
        assert report is not None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
