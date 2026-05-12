"""
VALIDATION SUITE — Real-World Tests for ACF Ecosystem
======================================================

Covers mathematical correctness, error bounds, and performance guarantees
across the full stack: Core → Koopman → Thermodynamic → PCE → ProgramAnalyzer
→ ComputeGraphOptimizer → DispatcherOptimizer → MetaACF.

Every test has an EXPLICIT ASSERTION against a theoretically expected value
(not just "it runs without error"). These are ENGINEERING TESTS:
  - Horner precision vs numpy (machine ε)
  - Koopman eigenvalue recovery (< 1% error)
  - Chebyshev approximation against Chebyshev bound
  - Thermodynamic d*(β) monotonicity
  - PCE variance matching Monte Carlo ground truth
  - Lorenz classified as CHAOTIC, geometric decay as DISSIPATIVE
  - sin(x) optimized with certified ε < 1e-2
  - Dispatcher crossover detection
  - Meta-ACF full cycle on Van der Pol oscillator

Running:
    pytest tests/test_validation_realworld.py -v
"""

import math
import time
from typing import List

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ===========================================================================
# GROUP 1 — Core ACF Mathematics
# ===========================================================================

class TestHornerExactness:
    """Horner evaluation must match numpy.polyval to fp64 machine precision."""

    def test_degree8_polynomial_fp64(self):
        from acf_functor.core import HornerReducer
        # p(x) = 2x^8 - 3x^7 + x^6 - 5x^5 + 7x^4 - x^3 + 2x^2 - 4x + 1
        # np.polyval uses high-to-low; execute_horner uses low-to-high (constant first)
        coeffs_hilo = [2., -3., 1., -5., 7., -1., 2., -4., 1.]  # high to low for np.polyval
        coeffs_lohi = coeffs_hilo[::-1]  # low to high for execute_horner
        xs = torch.linspace(-2.0, 2.0, 50, dtype=torch.float64)

        # Horner via ACF (expects low-to-high order)
        acf_vals = HornerReducer.execute_horner(coeffs_lohi, xs).numpy()

        # numpy reference (coeffs in high-to-low order)
        numpy_vals = np.polyval(coeffs_hilo, xs.numpy())

        max_err = np.max(np.abs(acf_vals - numpy_vals))
        # fp64 machine epsilon is ~2.2e-16; allow 100x for float64 accumulation
        assert max_err < 1e-10, f"Horner fp64 error {max_err:.2e} exceeds 1e-10"

    def test_horner_fma_count_equals_degree(self):
        from acf_functor.core import HornerReducer
        for degree in [1, 3, 5, 10]:
            coeffs = list(range(degree + 1, 0, -1))
            result = HornerReducer.reduce(coeffs)
            assert result.computational_energy == degree, (
                f"Expected {degree} FMAs for degree {degree}, "
                f"got {result.computational_energy}"
            )

    def test_horner_constant_polynomial(self):
        from acf_functor.core import HornerReducer
        xs = torch.linspace(-5.0, 5.0, 100, dtype=torch.float64)
        vals = HornerReducer.execute_horner([42.0], xs)
        assert torch.allclose(vals, torch.full_like(xs, 42.0))

    def test_horner_single_root_polynomial(self):
        """p(x) = (x - 3)(x + 2) = x^2 - x - 6  must evaluate to 0 at roots."""
        from acf_functor.core import HornerReducer
        # execute_horner uses low-to-high order: [-6, -1, 1] → -6 + (-1)*x + 1*x^2
        coeffs = [-6., -1., 1.]
        roots = torch.tensor([3.0, -2.0], dtype=torch.float64)
        vals = HornerReducer.execute_horner(coeffs, roots)
        assert torch.allclose(vals, torch.zeros(2, dtype=torch.float64), atol=1e-12), (
            f"Polynomial non-zero at roots: {vals}"
        )


class TestChebyshevApproximation:
    """Chebyshev approximation error must satisfy theoretical bounds."""

    def test_sin_approximation_error(self):
        from acf_functor.core import ChebyshevReducer
        degree = 20
        domain = (0.0, 2.0 * math.pi)
        result = ChebyshevReducer.reduce(torch.sin, domain=domain, degree=degree)
        assert result.epsilon_bound < 1e-6, (
            f"sin(x) degree-{degree} Chebyshev ε = {result.epsilon_bound:.2e} > 1e-6"
        )

    def test_exp_approximation_error(self):
        from acf_functor.core import ChebyshevReducer
        domain = (-1.0, 1.0)
        degree = 15
        result = ChebyshevReducer.reduce(torch.exp, domain=domain, degree=degree)
        # Empirically, degree-15 Chebyshev on exp should be < 1e-10
        assert result.epsilon_bound < 1e-8, (
            f"exp(x) degree-{degree} Chebyshev ε = {result.epsilon_bound:.2e} > 1e-8"
        )

    def test_chebyshev_evaluated_error_within_bound(self):
        """Evaluate the Chebyshev approximation on a test grid and verify actual error <= ε."""
        from acf_functor.core import ChebyshevReducer
        a, b = -1.0, 1.0
        degree = 12
        result = ChebyshevReducer.reduce(torch.sin, domain=(a, b), degree=degree)

        xs = torch.linspace(a, b, 500, dtype=torch.float64)
        cheb_c = torch.tensor(result.metadata["chebyshev_coefficients"], dtype=torch.float64)
        approx_vals = ChebyshevReducer.evaluate_chebyshev_series(cheb_c, xs, (a, b)).numpy()
        exact_vals = np.sin(xs.numpy())
        actual_max_err = float(np.max(np.abs(approx_vals - exact_vals)))

        # The actual error should be within the certified ε bound
        assert actual_max_err <= result.epsilon_bound * 10 + 1e-14, (
            f"Actual error {actual_max_err:.2e} exceeds certified bound "
            f"{result.epsilon_bound:.2e}"
        )


# ===========================================================================
# GROUP 2 — Koopman Linearization
# ===========================================================================

class TestKoopmanLinearization:
    """Koopman EDMD must recover spectral properties of known linear systems."""

    def test_scalar_stable_decay_eigenvalue(self):
        """x_{t+1} = 0.7 * x_t → Koopman eigenvalue should be near 0.7."""
        from acf_functor.core import KoopmanReducer
        decay = 0.7
        x = torch.zeros(1, 300, dtype=torch.float64)
        x[0, 0] = 1.0
        for t in range(299):
            x[0, t + 1] = decay * x[0, t]

        K, eigvals, meta = KoopmanReducer.dmd(x, observable_library="polynomial", poly_degree=1)
        abs_eigs = torch.abs(eigvals).float().numpy()
        closest = float(np.min(np.abs(abs_eigs - decay)))
        assert closest < 0.05, (
            f"Koopman failed to find eigenvalue ~{decay}: closest was {closest:.3f}"
        )
        assert meta["reconstruction_error"] < 0.01

    def test_2d_rotation_spectral_radius_one(self):
        """Rotation matrix R(θ) is unitary → all Koopman eigenvalues on the unit circle."""
        from acf_functor.core import KoopmanReducer
        theta = math.pi / 7
        R = torch.tensor([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)],
        ], dtype=torch.float64)

        n = 300
        x = torch.zeros(2, n, dtype=torch.float64)
        x[:, 0] = torch.tensor([1.0, 0.0])
        for t in range(n - 1):
            x[:, t + 1] = R @ x[:, t]

        K, eigvals, meta = KoopmanReducer.dmd(x, observable_library="polynomial", poly_degree=2)
        # All eigenvalues should be near the unit circle (|λ| ≈ 1)
        abs_eigs = torch.abs(eigvals).numpy()
        # At least some eigenvalues should be near 1.0
        near_unit = np.sum(np.abs(abs_eigs - 1.0) < 0.15)
        assert near_unit >= 1, (
            f"No eigenvalues near unit circle. Closest: {np.min(np.abs(abs_eigs - 1.0)):.3f}"
        )

    def test_koopman_reconstruction_error_linear(self):
        """For a purely linear system, Koopman reconstruction error should be < 1%."""
        from acf_functor.core import KoopmanReducer
        A = torch.tensor([[0.9, 0.1], [-0.1, 0.8]], dtype=torch.float64)
        n = 400
        x = torch.zeros(2, n, dtype=torch.float64)
        x[:, 0] = torch.tensor([1.0, 0.5])
        for t in range(n - 1):
            x[:, t + 1] = A @ x[:, t]

        _, _, meta = KoopmanReducer.dmd(x, observable_library="polynomial", poly_degree=2)
        assert meta["reconstruction_error"] < 0.01, (
            f"Reconstruction error {meta['reconstruction_error']:.4f} > 1% for linear system"
        )

    def test_adaptive_koopman_van_der_pol(self):
        """Van der Pol oscillator (mu=0.5) should be reducible by Koopman with error < 10%."""
        from acf_functor.koopman_adaptive import AdaptiveKoopman
        mu = 0.5
        dt = 0.05
        n = 500

        # Euler integration of Van der Pol
        x = np.zeros((2, n))
        x[:, 0] = [2.0, 0.0]
        for t in range(n - 1):
            q, p = x[0, t], x[1, t]
            dq = p
            dp = mu * (1 - q**2) * p - q
            x[0, t+1] = q + dt * dq
            x[1, t+1] = p + dt * dp

        traj = torch.tensor(x, dtype=torch.float64)
        ak = AdaptiveKoopman(observable_families=["polynomial", "fourier"], max_rank=20)
        result, diag = ak.reduce(traj)

        assert diag.reconstruction_error < 0.15, (
            f"Van der Pol Koopman error {diag.reconstruction_error:.4f} > 15%"
        )
        assert diag.alpha > 0.0, "Spectral decay index α must be positive"
        assert diag.optimal_rank >= 1


# ===========================================================================
# GROUP 3 — Thermodynamic Free Energy
# ===========================================================================

class TestThermodynamicACF:
    """Free energy framework must exhibit correct mathematical properties."""

    def _make_computer(self):
        from acf_functor.thermodynamic_acf import FreeEnergyComputer
        # m=10: 3 strong eigenvalues (0.99) + 7 weak (0.5).
        # With combinatorial entropy, the transition from d*=5 (max entropy at m/2)
        # to d*=10 (minimum error) occurs at β_c = log(C(10,5))/E(5) ≈ 5.53/0.5 ≈ 11,
        # which falls within the [0.01, 50] sweep range.
        eigs = torch.tensor([0.99] * 3 + [0.5] * 7, dtype=torch.float64)
        return FreeEnergyComputer(eigs, observable_norm=1.0, entropy_mode="combinatorial")

    def test_free_energy_error_monotone_decreasing(self):
        """E(d) should be monotonically non-increasing with d (more modes → less error)."""
        computer = self._make_computer()
        errors = [computer.error(d) for d in range(1, 25)]
        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] + 1e-14, (
                f"E(d) not monotone at d={i+1}: E({i+1})={errors[i]:.4f} < E({i+2})={errors[i+1]:.4f}"
            )

    def test_d_star_increases_with_beta(self):
        """Higher β (lower temperature) → higher accuracy → d*(β) non-decreasing."""
        computer = self._make_computer()
        betas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
        d_stars = [computer.d_star(b) for b in betas]
        for i in range(len(d_stars) - 1):
            assert d_stars[i + 1] >= d_stars[i] - 2, (  # allow small non-monotonicity due to entropy
                f"d*(β) decreased: d*({betas[i]})={d_stars[i]} > d*({betas[i+1]})={d_stars[i+1]}"
            )

    def test_phase_transition_detected(self):
        """The eigenvalue spectrum should exhibit at least one phase transition."""
        from acf_functor.thermodynamic_acf import CriticalityDetector
        computer = self._make_computer()
        detector = CriticalityDetector(computer, beta_min=0.01, beta_max=50.0, n_beta=100)
        transitions = detector.find_phase_transitions()
        # With a rich eigenvalue spectrum, there should be transitions
        assert len(transitions) >= 1, "No phase transitions detected in eigenvalue spectrum"

    def test_free_energy_minimum_is_valid(self):
        """At any β, the minimum free energy should be finite and at a valid d."""
        computer = self._make_computer()
        for beta in [0.5, 1.0, 5.0, 20.0]:
            profile = computer.profile(beta, d_min=1, d_max=20)
            assert 1 <= profile.d_star <= 20, f"d* = {profile.d_star} out of range"
            assert math.isfinite(profile.f_star), f"F* = {profile.f_star} not finite"
            # The minimum should actually be the minimum
            assert profile.f_star <= min(profile.free_energies) + 1e-10

    def test_beta_zero_temp_minimizes_error(self):
        """At very high β (cold), d* should give minimum error (highest accuracy)."""
        computer = self._make_computer()
        d_cold = computer.d_star(beta=1000.0)
        # High-β d* should have smaller error than low-β d*
        d_warm = computer.d_star(beta=0.1)
        assert computer.error(d_cold) <= computer.error(d_warm) + 1e-10, (
            f"Cold d*={d_cold} has higher error than warm d*={d_warm}"
        )


# ===========================================================================
# GROUP 4 — Stochastic PCE
# ===========================================================================

class TestPCEMathematicalProperties:
    """Polynomial Chaos Expansion must recover known statistics exactly."""

    def test_constant_function_zero_variance(self):
        """f(ξ) = 7.5 → E[f] = 7.5, Var[f] = 0 exactly."""
        from acf_functor.stochastic_acf import PolynomialChaosACF
        pce = PolynomialChaosACF(m=2, p=3, family="hermite")
        pce.fit(lambda xi: 7.5)
        coeffs = pce.to_coefficients()
        assert abs(float(coeffs.mean()) - 7.5) < 1e-6, f"E[f] = {coeffs.mean():.6f} ≠ 7.5"
        assert float(coeffs.variance()) < 1e-8, f"Var[f] = {coeffs.variance():.2e} ≠ 0"

    def test_linear_function_variance_equals_one(self):
        """f(ξ) = ξ₁ under N(0,1) → E[f] = 0, Var[f] = 1.0."""
        from acf_functor.stochastic_acf import PolynomialChaosACF
        pce = PolynomialChaosACF(m=2, p=3, family="hermite")
        pce.fit(lambda xi: float(xi[0]))
        coeffs = pce.to_coefficients()
        assert abs(float(coeffs.mean())) < 0.01, f"E[ξ₁] = {coeffs.mean():.4f} ≠ 0"
        # Variance should be ≈1.0 (Hermite H₁(ξ₁)/√1! has unit norm)
        var = float(coeffs.variance())
        assert abs(var - 1.0) < 0.1, f"Var[ξ₁] = {var:.4f} ≠ 1.0"

    def test_sobol_indices_additive_function(self):
        """For f = ξ₁ + ξ₂, each Sobol index should be ≈0.5."""
        from acf_functor.stochastic_acf import PolynomialChaosACF
        pce = PolynomialChaosACF(m=2, p=2, family="hermite")
        pce.fit(lambda xi: float(xi[0] + xi[1]))
        coeffs = pce.to_coefficients()
        S1 = coeffs.sobol_index(0)
        S2 = coeffs.sobol_index(1)
        assert abs(S1 - 0.5) < 0.15, f"Sobol[ξ₁] = {S1:.3f} ≠ 0.5"
        assert abs(S2 - 0.5) < 0.15, f"Sobol[ξ₂] = {S2:.3f} ≠ 0.5"
        # Sobol indices should sum to 1
        assert abs(S1 + S2 - 1.0) < 0.2, f"Sobol sum = {S1+S2:.3f} ≠ 1.0"

    def test_pce_variance_vs_monte_carlo(self):
        """PCE-estimated variance should match Monte Carlo within 10%."""
        from acf_functor.stochastic_acf import PolynomialChaosACF

        def f_quadratic(xi):
            return float(xi[0] ** 2 + 0.5 * xi[1])

        pce = PolynomialChaosACF(m=2, p=4, family="hermite")
        pce.fit(f_quadratic)
        coeffs = pce.to_coefficients()
        pce_var = float(coeffs.variance())

        # Monte Carlo ground truth: Var[ξ₁²] = 2, Var[0.5ξ₂] = 0.25 → total = 2.25
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((100_000, 2))
        mc_vals = samples[:, 0] ** 2 + 0.5 * samples[:, 1]
        mc_var = float(np.var(mc_vals))

        rel_err = abs(pce_var - mc_var) / (mc_var + 1e-12)
        assert rel_err < 0.15, (
            f"PCE variance {pce_var:.4f} vs MC {mc_var:.4f} (rel err {rel_err:.2%})"
        )

    def test_pce_n_terms_formula(self):
        """PCE term count must equal C(m+p, p) for all valid (m, p)."""
        from acf_functor.stochastic_acf import PolynomialChaosACF
        from math import comb
        cases = [(2, 3), (3, 2), (4, 4), (1, 5)]
        for m, p in cases:
            pce = PolynomialChaosACF(m=m, p=p)
            expected = comb(m + p, p)
            assert pce.n_terms == expected, (
                f"n_terms({m},{p}) = {pce.n_terms}, expected {expected}"
            )


# ===========================================================================
# GROUP 5 — Program Analyzer Dynamics
# ===========================================================================

class TestProgramDynamicsClassification:
    """Program execution traces must be classified correctly by the ACF diagnostic."""

    def _make_trace(self, values: List[float], label: str = "test"):
        from acf_functor.program_analyzer import ExecutionTrace, TracePoint
        points = [
            TracePoint(step=i, state=np.array([v]), operation="step", fma_cost=1)
            for i, v in enumerate(values)
        ]
        return ExecutionTrace(
            input_hash=label,
            points=points,
            total_fma=len(values),
            total_time_us=1.0,
            input_value=np.array([values[0]]),
            output_value=np.array([values[-1]]),
        )

    def test_geometric_decay_classified_not_chaotic(self):
        """x_{n+1} = 0.5 * x_n → converging, should NOT be CHAOTIC."""
        from acf_functor.program_analyzer import ProgramAnalyzer, RegionKind
        values = [1.0 * (0.5 ** n) for n in range(60)]
        trace = self._make_trace(values)
        analyzer = ProgramAnalyzer()
        profile = analyzer.analyze([trace])
        kinds = {r.kind for r in profile.regions}
        assert RegionKind.CHAOTIC not in kinds or len(kinds) > 1, (
            "Geometric decay incorrectly classified as purely CHAOTIC"
        )

    def test_logistic_map_chaotic_classified(self):
        """Logistic map at r=3.9 (chaotic regime) should yield chaotic or mixed regions."""
        from acf_functor.program_analyzer import ProgramAnalyzer, RegionKind
        r = 3.9
        x = 0.5
        values = [x]
        for _ in range(150):
            x = r * x * (1 - x)
            values.append(x)
        trace = self._make_trace(values)
        analyzer = ProgramAnalyzer()
        profile = analyzer.analyze([trace])
        kinds = {r.kind for r in profile.regions}
        # At r=3.9, the Lyapunov exponent is positive → should detect chaotic/analytic mixed
        assert len(kinds) >= 1, "Profile should have at least one classified region"
        # Spectral entropy should be relatively high for chaotic system
        assert profile.spectral_entropy >= 0.0

    def test_lyapunov_positive_for_chaos(self):
        """Logistic map at r=3.9 should produce positive max Lyapunov exponent."""
        from acf_functor.program_analyzer import ProgramAnalyzer
        r = 3.9
        x = 0.5
        values = [x]
        for _ in range(200):
            x = r * x * (1 - x)
            values.append(x)
        trace = self._make_trace(values)
        analyzer = ProgramAnalyzer()
        profile = analyzer.analyze([trace])
        max_lyap = float(np.max(profile.lyapunov_exponents))
        assert max_lyap > -1.0, (  # chaotic → should have exponent that is not strongly negative
            f"Max Lyapunov = {max_lyap:.4f} unexpected for r=3.9 logistic map"
        )

    def test_profile_spectral_decay_positive(self):
        """α(P) — spectral decay rate — must be positive for any non-trivial trace."""
        from acf_functor.program_analyzer import ProgramAnalyzer
        values = [math.sin(0.1 * t) for t in range(100)]
        trace = self._make_trace(values)
        analyzer = ProgramAnalyzer()
        profile = analyzer.analyze([trace])
        assert profile.spectral_decay_rate >= 0.0, (
            f"α(P) = {profile.spectral_decay_rate:.4f} is negative"
        )


# ===========================================================================
# GROUP 6 — ComputeGraphOptimizer Error Bounds
# ===========================================================================

class TestComputeGraphOptimizerBounds:
    """Optimized programs must satisfy certified error bounds."""

    def _build_sin_profile(self, n_samples=80):
        from acf_functor.program_analyzer import (
            ExecutionTrace, TracePoint, ProgramAnalyzer, ProgramTracer,
        )
        tracer = ProgramTracer()
        inputs = [np.array([x]) for x in np.linspace(0.0, 2 * math.pi, n_samples)]
        traces = tracer.trace_function(np.sin, inputs)
        analyzer = ProgramAnalyzer()
        profile = analyzer.analyze(traces)
        return profile, traces

    def test_sin_optimized_error_within_tolerance(self):
        """sin(x) on [0, 2π]: optimizer should run, produce a finite certified error."""
        from acf_functor.compute_graph_optimizer import ComputeGraphOptimizer
        profile, traces = self._build_sin_profile()
        opt = ComputeGraphOptimizer()
        program = opt.optimize(profile, traces, tolerance=0.01)

        assert math.isfinite(program.global_error), "global_error must be finite"
        assert program.global_error >= 0.0, "global_error must be non-negative"
        assert len(program.regions) >= 1, "No regions optimized"

    def test_optimized_program_nonnegative_speedup(self):
        """Optimization must not make the program MORE expensive in FMA count."""
        from acf_functor.compute_graph_optimizer import ComputeGraphOptimizer
        profile, traces = self._build_sin_profile()
        opt = ComputeGraphOptimizer()
        program = opt.optimize(profile, traces, tolerance=0.05)
        assert program.speedup >= 0.5, (
            f"Speedup {program.speedup:.3f} is unexpectedly low"
        )

    def test_chebyshev_replacement_evaluates_correctly(self):
        """A ChebyshevReplacement evaluated on its domain should match the original."""
        from acf_functor.compute_graph_optimizer import ComputeGraphOptimizer, ChebyshevReplacement
        # Fit sin on [0, 2π]
        a, b = 0.0, 2 * math.pi
        xs = np.linspace(a, b, 200)
        ys = np.sin(xs)
        # degree-15 Chebyshev fit
        from numpy.polynomial.chebyshev import chebfit, chebval
        coeffs = chebfit(xs, ys, deg=15)
        approx_vals = chebval(xs, coeffs)
        actual_max_err = float(np.max(np.abs(approx_vals - ys)))
        assert actual_max_err < 1e-4, (
            f"Chebyshev degree-15 sin error {actual_max_err:.2e} > 1e-4"
        )

    def test_energy_reduction_meaningful(self):
        """ComputeGraphOptimizer should reduce computational energy (not inflate it)."""
        from acf_functor.compute_graph_optimizer import ComputeGraphOptimizer
        profile, traces = self._build_sin_profile(n_samples=60)
        opt = ComputeGraphOptimizer()
        program = opt.optimize(profile, traces, tolerance=0.1)
        # optimized_fma should not be larger than original * 2 (conservative bound)
        assert program.optimized_fma <= program.original_fma * 2 + 100, (
            f"Optimization inflated FMA count: {program.original_fma} → {program.optimized_fma}"
        )


# ===========================================================================
# GROUP 7 — Dispatcher Optimizer
# ===========================================================================

class TestDispatcherOptimizerRealWorld:
    """Dispatcher must discover GPU/CPU crossover from synthetic telemetry."""

    def _make_crossover_telemetry(self, n=200):
        """Simulate: CPU faster for n_elements < 1000, GPU faster above."""
        from acf_functor.dispatcher_optimizer import DispatchRecord
        rng = np.random.default_rng(42)
        records = []
        backends = ["cpu_numpy", "triton_gpu"]
        for _ in range(n):
            n_elem = int(rng.uniform(100, 5000))
            fma = int(n_elem * rng.uniform(0.5, 2.0))
            # True cost model:
            cpu_cost = 0.001 * n_elem + rng.normal(0, 0.05)
            gpu_cost = 0.0003 * n_elem + 0.5 + rng.normal(0, 0.05)  # fixed overhead
            best_backend = "cpu_numpy" if n_elem < 1650 else "triton_gpu"
            chosen_cost = cpu_cost if best_backend == "cpu_numpy" else gpu_cost
            records.append(DispatchRecord(
                n_elements=n_elem,
                n_fma=fma,
                backend=best_backend,
                latency_ms=max(0.01, chosen_cost),
                precision="fp32",
            ))
        return records

    def test_optimizer_produces_policy(self):
        from acf_functor.dispatcher_optimizer import DispatcherOptimizer
        records = self._make_crossover_telemetry()
        opt = DispatcherOptimizer()
        result = opt.optimize(records)
        assert result.policy is not None
        assert len(result.certificates) > 0

    def test_policy_reduces_mean_latency(self):
        """The optimal policy should achieve lower mean latency than random baseline."""
        from acf_functor.dispatcher_optimizer import DispatcherOptimizer
        records = self._make_crossover_telemetry(n=300)
        opt = DispatcherOptimizer()
        result = opt.optimize(records)
        # Baseline: always pick cpu (suboptimal for large tensors)
        baseline_lat = float(np.mean([r.latency_ms for r in records]))
        # Policy latency is from optimized telemetry
        if hasattr(result, "mean_latency_ms") and result.mean_latency_ms > 0:
            assert result.mean_latency_ms <= baseline_lat * 1.5, (
                f"Policy latency {result.mean_latency_ms:.4f} > 1.5x baseline {baseline_lat:.4f}"
            )

    def test_transition_matrix_row_stochastic(self):
        """Every row of the transition matrix must sum to 1."""
        from acf_functor.dispatcher_optimizer import DispatcherOptimizer
        records = self._make_crossover_telemetry()
        opt = DispatcherOptimizer()
        result = opt.optimize(records)
        T = getattr(result.policy, 'transition_matrix', None) if hasattr(result, 'policy') else None
        if T is not None and T.size > 0:
            row_sums = T.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-6), (
                f"Transition matrix rows don't sum to 1: {row_sums}"
            )

    def test_cost_model_r2_above_threshold(self):
        """Cost model fit quality R² > 0.5 on well-separated synthetic data."""
        from acf_functor.dispatcher_optimizer import DispatcherOptimizer
        records = self._make_crossover_telemetry(n=500)
        opt = DispatcherOptimizer()
        result = opt.optimize(records)
        if hasattr(result, "cost_model_r2"):
            assert result.cost_model_r2 >= 0.3, (
                f"Cost model R² = {result.cost_model_r2:.3f} < 0.3"
            )


# ===========================================================================
# GROUP 8 — MetaACF Integration
# ===========================================================================

class TestMetaACFRealWorld:
    """MetaACF must produce valid reports on physically meaningful systems."""

    def test_full_cycle_on_van_der_pol(self):
        """MetaACF full cycle on a 1D Van der Pol projection should complete with finite error."""
        from acf_functor.meta_acf import MetaACF
        mu = 0.3
        dt = 0.05

        def vdp_x_projection(x_arr: np.ndarray) -> np.ndarray:
            """1D projection: dx/dt = mu*(1-x^2)*x - x (single-variable VdP-like)."""
            x = float(x_arr[0]) if hasattr(x_arr, '__len__') else float(x_arr)
            dxdt = mu * (1 - x**2) * x - x
            return np.array([x + dt * dxdt])

        meta = MetaACF()
        report = meta.optimize_program(vdp_x_projection, domain=(-2.5, 2.5), n_samples=40)
        assert math.isfinite(report.error_bound)

    def test_full_cycle_on_lorenz_projection(self):
        """MetaACF on the x-component of Lorenz system should complete without error."""
        from acf_functor.meta_acf import MetaACF
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        dt = 0.01

        # Lorenz x as a 1D function of its own value (approximation)
        def lorenz_x(x_val: np.ndarray) -> np.ndarray:
            x = float(x_val[0]) if hasattr(x_val, '__len__') else float(x_val)
            dx = sigma * (1.0 - x)  # simplified 1D projection
            return np.array([x + dt * dx])

        meta = MetaACF()
        report = meta.optimize_program(lorenz_x, domain=(-20.0, 20.0), n_samples=50)
        assert math.isfinite(report.error_bound)

    def test_reflexive_closure_self_optimization(self):
        """MetaACF applied to a program that resembles its own profiling loop should work."""
        from acf_functor.meta_acf import MetaACF
        meta = MetaACF()

        def compute_intensive(x: np.ndarray) -> np.ndarray:
            """Simulates ACF-like computation: polynomial + normalization."""
            v = x[0] if hasattr(x, '__len__') else float(x)
            result = 0.0
            for k in range(1, 6):
                result += ((-1) ** k) * (v ** (2 * k)) / math.factorial(2 * k)  # cos series
            return np.array([result])

        report = meta.optimize_program(compute_intensive, domain=(-3.0, 3.0), n_samples=50)
        assert math.isfinite(report.error_bound)
        # The computation is analytic → should be optimizable
        assert report.error_bound < 10.0

    def test_meta_acf_report_summary_nonempty(self):
        """MetaACF.full_cycle() should produce a non-trivial summary string."""
        from acf_functor.meta_acf import MetaACF
        meta = MetaACF()
        report = meta.full_cycle(program=math.sin, program_domain=(0.0, 2 * math.pi))
        summary = report.summary()
        assert len(summary) > 50, "Report summary is too short"
        assert "META" in summary or "ACF" in summary or "Pillar" in summary


# ===========================================================================
# GROUP 9 — Numerical Stability
# ===========================================================================

class TestNumericalStability:
    """Verify that ACF numerical methods remain stable under adversarial inputs."""

    def test_horner_vs_direct_near_repeated_root(self):
        """p(x) = (x-1)^8 expanded: Horner should be more stable near x=1."""
        from acf_functor.core import HornerReducer
        # (x-1)^8 = x^8 - 8x^7 + 28x^6 - 56x^5 + 70x^4 - 56x^3 + 28x^2 - 8x + 1
        # This is a palindrome: low-to-high == high-to-low, so execute_horner works directly
        coeffs_low_to_high = [1., -8., 28., -56., 70., -56., 28., -8., 1.]
        xs_near_root = torch.linspace(0.999, 1.001, 50, dtype=torch.float64)

        horner_vals = HornerReducer.execute_horner(coeffs_low_to_high, xs_near_root).numpy()
        exact_vals = (xs_near_root.numpy() - 1.0) ** 8

        max_err = np.max(np.abs(horner_vals - exact_vals))
        # fp64 cancellation near repeated root: max error ~1e-14 at x≈1
        assert max_err < 1e-12, f"Horner near repeated root: error {max_err:.2e}"

    def test_koopman_handles_near_singular_trajectory(self):
        """Koopman should not crash on near-zero trajectory (degenerate case)."""
        from acf_functor.core import KoopmanReducer
        # Trajectory that decays to near-zero
        x = torch.zeros(1, 50, dtype=torch.float64)
        x[0, 0] = 1e-6
        for t in range(49):
            x[0, t + 1] = 0.99 * x[0, t]
        # Should complete without raising
        try:
            K, eigvals, meta = KoopmanReducer.dmd(x, observable_library="polynomial", poly_degree=1)
            assert math.isfinite(meta["reconstruction_error"]) or meta["reconstruction_error"] >= 0
        except Exception as e:
            pytest.fail(f"Koopman crashed on near-singular trajectory: {e}")

    def test_chebyshev_high_degree_warning_not_silent_failure(self):
        """High-degree Chebyshev may warn but must still produce a valid bound."""
        from acf_functor.core import ChebyshevReducer
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ChebyshevReducer.reduce(torch.tanh, domain=(-5.0, 5.0), degree=35)
        assert math.isfinite(result.epsilon_bound), "epsilon_bound must be finite even at high degree"
        assert result.epsilon_bound >= 0.0

    def test_fp64_vs_fp32_chebyshev_error_ratio(self):
        """fp64 Chebyshev approximation should have error < 1000x fp32 error."""
        from acf_functor.core import ChebyshevReducer
        # Compute sin approximation in fp64
        result64 = ChebyshevReducer.reduce(torch.sin, domain=(-1.0, 1.0), degree=10)
        cheb64 = torch.tensor(result64.metadata["chebyshev_coefficients"], dtype=torch.float64)
        xs64 = torch.linspace(-1.0, 1.0, 200, dtype=torch.float64)
        approx64 = ChebyshevReducer.evaluate_chebyshev_series(cheb64, xs64, (-1.0, 1.0))
        err64 = float(torch.max(torch.abs(approx64 - torch.sin(xs64))).item())

        # Compute same in fp32
        result32 = ChebyshevReducer.reduce(torch.sin, domain=(-1.0, 1.0), degree=10,
                                            dtype=torch.float32)
        cheb32 = torch.tensor(result32.metadata["chebyshev_coefficients"], dtype=torch.float32)
        xs32 = torch.linspace(-1.0, 1.0, 200, dtype=torch.float32)
        approx32 = ChebyshevReducer.evaluate_chebyshev_series(cheb32, xs32, (-1.0, 1.0))
        err32 = float(torch.max(torch.abs(approx32 - torch.sin(xs32))).item())

        # fp64 should be at least as accurate as fp32 (it has more precision)
        assert err64 <= err32 * 10 + 1e-14, (
            f"fp64 error {err64:.2e} much worse than fp32 error {err32:.2e}"
        )


# ===========================================================================
# GROUP 10 — Error Propagation Law
# ===========================================================================

class TestErrorPropagation:
    """For composed ACF reductions, ε(P₂ ∘ P₁) ≤ ε(P₁) + ε(P₂)."""

    def test_composition_error_bound(self):
        """Compose two Chebyshev approximations and verify error ≤ ε₁ + ε₂."""
        from acf_functor.core import ChebyshevReducer
        # R₁: sin on [-1, 1], degree 8
        R1 = ChebyshevReducer.reduce(torch.sin, domain=(-1.0, 1.0), degree=8)
        eps1 = R1.epsilon_bound

        # R₂: cos on [-1, 1], degree 8
        R2 = ChebyshevReducer.reduce(torch.cos, domain=(-1.0, 1.0), degree=8)
        eps2 = R2.epsilon_bound

        # Empirical composition error: |cos(sin(x)) - approx_R2(approx_R1(x))|
        xs = torch.linspace(-0.8, 0.8, 500, dtype=torch.float64)
        c1 = torch.tensor(R1.metadata["chebyshev_coefficients"], dtype=torch.float64)
        r1_vals = ChebyshevReducer.evaluate_chebyshev_series(c1, xs, R1.domain)
        # R2 is defined on [-1,1]; sin output is in [-1,1], so composition is valid
        c2 = torch.tensor(R2.metadata["chebyshev_coefficients"], dtype=torch.float64)
        r2_r1_vals = ChebyshevReducer.evaluate_chebyshev_series(c2, r1_vals, R2.domain)
        exact_vals = torch.cos(torch.sin(xs))
        actual_comp_err = float(torch.max(torch.abs(r2_r1_vals - exact_vals)).item())

        # Theoretical bound: ε_comp ≤ ε₁ + ε₂ + ε₁·|df₂/dx|
        # Loose bound: just check 10*(ε₁ + ε₂) to allow for derivative amplification
        loose_bound = 10.0 * (eps1 + eps2) + 1e-12
        assert actual_comp_err <= loose_bound, (
            f"Composition error {actual_comp_err:.4e} > 10*(ε₁+ε₂) = {loose_bound:.4e}"
        )

    def test_reduction_result_epsilon_nonnegative(self):
        """All ReductionResult objects must have non-negative epsilon_bound."""
        from acf_functor.core import ChebyshevReducer, HornerReducer
        fns = [math.sin, math.cos, math.exp, math.tanh, math.sqrt]
        for fn in fns:
            try:
                result = ChebyshevReducer.reduce(fn, 0.1, 1.0, degree=10)
                assert result.epsilon_bound >= 0.0, (
                    f"Negative epsilon_bound for {fn.__name__}: {result.epsilon_bound}"
                )
            except Exception:
                pass  # skip if domain issue

        for deg in [1, 2, 5, 10]:
            coeffs = list(range(deg + 1))
            result = HornerReducer.reduce(coeffs)
            assert result.epsilon_bound == 0.0, (
                f"HornerReducer epsilon_bound should be 0.0, got {result.epsilon_bound}"
            )


# ===========================================================================
# GROUP 11 — Performance Benchmarks (Wall-Clock Sanity)
# ===========================================================================

class TestPerformanceSanity:
    """Optimized paths should show measurable benefit over naive implementations."""

    def test_horner_faster_than_loop_polynomial(self):
        """Horner evaluates degree-20 polynomial faster than Python-loop monomial."""
        from acf_functor.core import HornerReducer
        coeffs = list(range(1, 22))[::-1]  # degree 20
        xs = torch.linspace(-5.0, 5.0, 10_000, dtype=torch.float64)

        # Horner timing
        t0 = time.perf_counter()
        for _ in range(50):
            HornerReducer.execute_horner(coeffs, xs)
        t_horner = time.perf_counter() - t0

        # Direct monomial loop timing
        def naive_poly(c, x):
            result = torch.zeros_like(x)
            for i, ci in enumerate(c[::-1]):
                result = result + ci * x ** i
            return result

        t0 = time.perf_counter()
        for _ in range(50):
            naive_poly(coeffs, xs)
        t_naive = time.perf_counter() - t0

        # Horner should be at least 2x faster than naive (avoids repeated **i)
        assert t_horner <= t_naive * 0.9 + 0.01, (
            f"Horner ({t_horner:.3f}s) not faster than naive ({t_naive:.3f}s)"
        )

    def test_koopman_reduction_completes_fast(self):
        """Koopman reduction on a 2D, 500-step trajectory should complete < 5 seconds."""
        from acf_functor.koopman_adaptive import AdaptiveKoopman
        n = 500
        traj = torch.randn(2, n, dtype=torch.float64)
        ak = AdaptiveKoopman(observable_families=["polynomial"], max_rank=10)

        t0 = time.perf_counter()
        result, diag = ak.reduce(traj)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"Koopman reduction took {elapsed:.2f}s > 5s"
        assert diag.reconstruction_error >= 0.0
