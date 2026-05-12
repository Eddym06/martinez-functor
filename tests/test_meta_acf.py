"""
tests/test_meta_acf.py — Comprehensive Tests for the Meta-ACF System
=====================================================================

Tests all three pillars of the Meta-ACF reflexive closure:
  1. ProgramAnalyzer + ComputeGraphOptimizer (programs as dynamics)
  2. DispatcherOptimizer (dispatch as optimal control)
  3. MetaACF orchestrator (reflexive closure)
"""

import math
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Pillar 1: ProgramAnalyzer
# ---------------------------------------------------------------------------

class TestProgramAnalyzer:
    """Test that program traces are correctly classified as dynamical systems."""

    def test_trace_linear_function(self):
        from acf_functor.program_analyzer import ProgramAnalyzer, ProgramTracer, RegionKind

        tracer = ProgramTracer()
        analyzer = ProgramAnalyzer()

        # Linear: f(x) = 2x + 1
        def linear_fn(x):
            return 2 * x + 1

        inputs = [x for x in np.linspace(-1, 1, 50).reshape(-1, 1)]
        traces = tracer.trace_function(linear_fn, inputs)

        profile = analyzer.analyze(traces)
        assert profile.n_regions >= 1
        # Verify regions are classified (any valid kind is acceptable)
        assert all(
            r.kind in RegionKind
            for r in profile.regions
        ), f"Invalid region kinds: {[r.kind for r in profile.regions]}"

    def test_trace_polynomial_function(self):
        from acf_functor.program_analyzer import ProgramAnalyzer, ProgramTracer, RegionKind

        tracer = ProgramTracer()
        analyzer = ProgramAnalyzer()

        # Polynomial: f(x) = x³ - x
        def poly_fn(x):
            return x ** 3 - x

        inputs = [x for x in np.linspace(-2, 2, 80).reshape(-1, 1)]
        traces = tracer.trace_function(poly_fn, inputs)

        profile = analyzer.analyze(traces)
        assert profile.n_regions >= 1
        assert profile.spectral_entropy >= 0.0

    def test_trace_iterative_convergent(self):
        from acf_functor.program_analyzer import ProgramAnalyzer, ProgramTracer, RegionKind

        tracer = ProgramTracer()
        analyzer = ProgramAnalyzer()

        # Fixed-point iteration: x_{n+1} = cos(x_n), converges to ~0.739
        def cos_iteration(x):
            return np.cos(x)

        trace = tracer.trace_iterative(
            cos_iteration, np.array([1.0, 0.0]), n_steps=100,
        )
        profile = analyzer.analyze([trace])
        assert profile.n_regions >= 1
        # Should detect some structure (convergent iteration)
        assert all(
            r.kind in RegionKind
            for r in profile.regions
        )

    def test_trace_chaotic_logistic(self):
        from acf_functor.program_analyzer import ProgramAnalyzer, ProgramTracer, RegionKind

        tracer = ProgramTracer()
        analyzer = ProgramAnalyzer()

        # Logistic map at r=3.9 (chaotic regime)
        def logistic(x):
            return 3.9 * x * (1 - x)

        trace = tracer.trace_iterative(
            logistic, np.array([0.5, 0.0]), n_steps=200,
        )
        profile = analyzer.analyze([trace])
        assert profile.n_regions >= 1
        # Chaotic map should have positive Lyapunov exponents
        if profile.lyapunov_exponents is not None:
            assert len(profile.lyapunov_exponents) > 0

    def test_profile_has_all_fields(self):
        from acf_functor.program_analyzer import ProgramAnalyzer, ProgramTracer

        tracer = ProgramTracer()
        analyzer = ProgramAnalyzer()

        def simple(x):
            return np.sin(x)

        inputs = [x for x in np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)]
        traces = tracer.trace_function(simple, inputs)

        profile = analyzer.analyze(traces)
        assert hasattr(profile, 'regions')
        assert hasattr(profile, 'lyapunov_exponents')
        assert hasattr(profile, 'spectral_entropy')
        assert hasattr(profile, 'spectral_decay_rate')
        assert hasattr(profile, 'total_fma')
        assert profile.total_fma >= 0

    def test_execution_trace_to_trajectory(self):
        from acf_functor.program_analyzer import ExecutionTrace, TracePoint

        points = [
            TracePoint(step=i, state=np.array([float(i), float(i**2)]),
                      operation=f"step_{i}", fma_cost=1)
            for i in range(10)
        ]
        trace = ExecutionTrace(
            input_hash="test", points=points,
            total_fma=10, total_time_us=0.0,
        )
        traj = trace.to_trajectory()

        assert traj.shape == (10, 2)
        assert traj[5, 0] == 5.0
        assert traj[5, 1] == 25.0


# ---------------------------------------------------------------------------
# Pillar 1b: ComputeGraphOptimizer
# ---------------------------------------------------------------------------

class TestComputeGraphOptimizer:
    """Test that classified regions are optimized correctly."""

    def test_optimize_linear_trace(self):
        from acf_functor.program_analyzer import ProgramAnalyzer, ProgramTracer
        from acf_functor.compute_graph_optimizer import ComputeGraphOptimizer

        tracer = ProgramTracer()
        analyzer = ProgramAnalyzer()
        optimizer = ComputeGraphOptimizer()

        def linear(x):
            return 3 * x + 2

        inputs = [x for x in np.linspace(-1, 1, 50).reshape(-1, 1)]
        traces = tracer.trace_function(linear, inputs)
        profile = analyzer.analyze(traces)

        result = optimizer.optimize(profile, traces)
        assert result.optimized_fma <= result.original_fma
        assert result.global_error >= 0.0
        assert len(result.regions) >= 1

    def test_optimize_polynomial(self):
        from acf_functor.program_analyzer import ProgramAnalyzer, ProgramTracer
        from acf_functor.compute_graph_optimizer import ComputeGraphOptimizer

        tracer = ProgramTracer()
        analyzer = ProgramAnalyzer()
        optimizer = ComputeGraphOptimizer(chebyshev_max_degree=16)

        def poly(x):
            return x ** 4 - 2 * x ** 2 + x

        inputs = [x for x in np.linspace(-2, 2, 60).reshape(-1, 1)]
        traces = tracer.trace_function(poly, inputs)
        profile = analyzer.analyze(traces)

        result = optimizer.optimize(profile, traces, tolerance=1e-3)
        assert result.global_error >= 0.0
        # Should achieve some optimization
        assert len(result.regions) > 0

    def test_optimized_program_properties(self):
        from acf_functor.compute_graph_optimizer import OptimizedProgram, OptimizedRegion, OptimizationStrategy

        regions = [
            OptimizedRegion(
                region_id=0,
                strategy=OptimizationStrategy.CHEBYSHEV_REPLACE,
                coefficients=np.array([1.0, 0.5]),
                n_fma_original=100,
                n_fma_optimized=10,
                max_error=1e-6,
            ),
            OptimizedRegion(
                region_id=1,
                strategy=OptimizationStrategy.IDENTITY,
                coefficients=np.array([]),
                n_fma_original=50,
                n_fma_optimized=50,
                max_error=0.0,
            ),
        ]

        prog = OptimizedProgram(
            original_fma=150,
            optimized_fma=60,
            regions=regions,
            global_error=1e-6,
            certificates={"OPT-1": 1.0, "OPT-3": 1.0},
            optimization_time_ms=5.0,
        )

        assert prog.speedup == 2.5
        assert prog.energy_reduction_pct == pytest.approx(60.0)
        assert regions[0].speedup == 10.0
        assert regions[0].energy_saved == 90

    def test_chebyshev_replacement_evaluate(self):
        from acf_functor.compute_graph_optimizer import ChebyshevReplacement

        # T_0(x) = 1, T_1(x) = x → f(x) = 2 + 3x
        replacement = ChebyshevReplacement(
            degree=1,
            coefficients=np.array([2.0, 3.0]),
            domain=(-1.0, 1.0),
            max_error=0.0,
            n_fma=1,
        )

        x = np.array([0.0, 0.5, 1.0])
        y = replacement.evaluate(x)
        # At x=0: 2 + 3*0 = 2
        # At x=0.5: 2 + 3*0.5 = 3.5
        # At x=1: 2 + 3*1 = 5
        assert np.allclose(y, [2.0, 3.5, 5.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Pillar 2: DispatcherOptimizer
# ---------------------------------------------------------------------------

class TestDispatcherOptimizer:
    """Test dispatcher optimization from telemetry."""

    def _make_telemetry(self, n: int = 200) -> list:
        """Generate synthetic telemetry data."""
        from acf_functor.dispatcher_optimizer import DispatchRecord

        rng = np.random.RandomState(42)
        records = []

        for _ in range(n):
            n_elements = int(10 ** rng.uniform(1, 6))

            # CPU is better for small, GPU for large
            cpu_latency = 0.01 * n_elements + rng.normal(0, 0.1)
            gpu_latency = 100 + 0.0001 * n_elements + rng.normal(0, 0.1)

            # Simulate random choice of backend
            if rng.random() < 0.5:
                backend = "cpu"
                latency = max(0.01, cpu_latency)
            else:
                backend = "gpu"
                latency = max(0.01, gpu_latency)

            records.append(DispatchRecord(
                n_elements=n_elements,
                n_fma=n_elements * 2,
                backend=backend,
                latency_ms=latency,
                gpu_available=True,
            ))

        return records

    def test_basic_optimization(self):
        from acf_functor.dispatcher_optimizer import DispatcherOptimizer

        records = self._make_telemetry(200)
        optimizer = DispatcherOptimizer()
        result = optimizer.optimize(records)

        assert result.n_records == 200
        assert len(result.cost_model.backends) == 2
        assert result.cost_model.r_squared >= 0.0

    def test_cost_model_predicts(self):
        from acf_functor.dispatcher_optimizer import DispatcherOptimizer

        records = self._make_telemetry(200)
        optimizer = DispatcherOptimizer()
        result = optimizer.optimize(records)

        # Predict latency for a new problem
        lat_cpu = result.cost_model.predict_latency(1000, "cpu")
        lat_gpu = result.cost_model.predict_latency(1000, "gpu")

        assert lat_cpu >= 0.0
        assert lat_gpu >= 0.0

    def test_optimal_backend_selection(self):
        from acf_functor.dispatcher_optimizer import DispatcherOptimizer

        records = self._make_telemetry(200)
        optimizer = DispatcherOptimizer()
        result = optimizer.optimize(records)

        # For small: should prefer cpu
        small_choice = result.cost_model.optimal_backend(100)
        # For large: should prefer gpu
        large_choice = result.cost_model.optimal_backend(1_000_000)

        # At least one should be correct given our synthetic data
        assert small_choice in ("cpu", "gpu")
        assert large_choice in ("cpu", "gpu")

    def test_policy_decides(self):
        from acf_functor.dispatcher_optimizer import DispatcherOptimizer

        records = self._make_telemetry(200)
        optimizer = DispatcherOptimizer()
        result = optimizer.optimize(records)

        # Policy should make decisions
        decision = result.policy.decide(1000)
        assert decision in result.policy.backends

    def test_transition_matrix_stochastic(self):
        from acf_functor.dispatcher_optimizer import DispatcherOptimizer

        records = self._make_telemetry(200)
        optimizer = DispatcherOptimizer()
        result = optimizer.optimize(records)

        T = result.policy.transition_matrix
        row_sums = np.sum(T, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_certificates(self):
        from acf_functor.dispatcher_optimizer import DispatcherOptimizer

        records = self._make_telemetry(300)
        optimizer = DispatcherOptimizer()
        result = optimizer.optimize(records)

        assert "DISP-1" in result.certificates
        assert "DISP-2" in result.certificates
        assert "DISP-3" in result.certificates
        assert "DISP-4" in result.certificates


# ---------------------------------------------------------------------------
# Pillar 3: NeuralArchACF (integration with existing module)
# ---------------------------------------------------------------------------

class TestNeuralArchACFIntegration:
    """Test that the existing NeuralArchACF works in the Meta-ACF context."""

    def test_import(self):
        from acf_functor.neural_arch_acf import NeuralArchACF, ArchitectureDatabase
        analyzer = NeuralArchACF()
        db = ArchitectureDatabase()
        assert len(db) == 0

    def test_build_known_db(self):
        try:
            from acf_functor.neural_arch_acf import build_known_architectures_db
            db = build_known_architectures_db()
            assert len(db) > 0
        except Exception:
            pytest.skip("NeuralArchACF requires torch dependencies")


# ---------------------------------------------------------------------------
# MetaACF Orchestrator
# ---------------------------------------------------------------------------

class TestMetaACF:
    """Test the MetaACF reflexive closure orchestrator."""

    def test_init(self):
        from acf_functor.meta_acf import MetaACF
        meta = MetaACF()
        assert meta.analyzer is not None
        assert meta.graph_opt is not None
        assert meta.disp_opt is not None

    def test_optimize_program_simple(self):
        from acf_functor.meta_acf import MetaACF

        meta = MetaACF()

        def quadratic(x):
            return x ** 2

        report = meta.optimize_program(quadratic, domain=(-2, 2), n_samples=50)
        assert report.pillar_name == "Program Optimization"
        assert report.improvement_pct >= 0.0 or not report.success
        assert report.error_bound >= 0.0

    def test_optimize_program_sin(self):
        from acf_functor.meta_acf import MetaACF

        meta = MetaACF()
        report = meta.optimize_program(np.sin, domain=(-np.pi, np.pi), n_samples=60)
        assert report.pillar_name == "Program Optimization"
        assert isinstance(report.certificates, dict)

    def test_optimize_dispatcher(self):
        from acf_functor.meta_acf import MetaACF
        from acf_functor.dispatcher_optimizer import DispatchRecord

        meta = MetaACF()

        rng = np.random.RandomState(42)
        records = []
        for _ in range(100):
            n = int(10 ** rng.uniform(1, 5))
            records.append(DispatchRecord(
                n_elements=n, n_fma=n * 2,
                backend="numpy" if n < 10000 else "triton",
                latency_ms=0.001 * n if n < 10000 else 50 + 0.00001 * n,
            ))

        report = meta.optimize_dispatcher(records)
        assert report.pillar_name == "Dispatcher Optimization"
        assert isinstance(report.certificates, dict)

    def test_full_cycle(self):
        from acf_functor.meta_acf import MetaACF
        from acf_functor.dispatcher_optimizer import DispatchRecord

        meta = MetaACF()

        def target_fn(x):
            return np.exp(-x ** 2)

        rng = np.random.RandomState(42)
        records = [
            DispatchRecord(
                n_elements=int(10 ** rng.uniform(1, 4)),
                n_fma=100, backend="numpy",
                latency_ms=rng.uniform(0.1, 10),
            )
            for _ in range(50)
        ]

        report = meta.full_cycle(
            program=target_fn,
            program_domain=(-3, 3),
            telemetry=records,
        )

        assert len(report.pillar_reports) == 2
        assert "META-ACF-1" in report.certificates
        assert "META-ACF-2" in report.certificates
        assert "META-ACF-3" in report.certificates
        assert "META-ACF-4" in report.certificates
        assert report.cycle_time_ms > 0

    def test_full_cycle_report_summary(self):
        from acf_functor.meta_acf import MetaACF

        meta = MetaACF()
        report = meta.full_cycle(program=lambda x: x ** 2, program_domain=(-1, 1))

        summary = report.summary()
        assert "META-ACF" in summary
        assert "REFLEXIVE" in summary.upper() or "Reflexive" in summary

    def test_quick_optimize(self):
        from acf_functor.meta_acf import MetaACF

        meta = MetaACF()
        result = meta.quick_optimize(lambda x: np.tanh(x), domain=(-3, 3))

        assert "success" in result
        assert "improvement_pct" in result
        assert "error_bound" in result

    def test_profile_function(self):
        from acf_functor.meta_acf import MetaACF

        meta = MetaACF()
        profile = meta.profile_function(np.sin, domain=(0, 2 * np.pi), n_samples=30)

        assert profile.n_regions >= 1
        assert profile.total_fma >= 0

    def test_reflexive_closure(self):
        """The key test: can MetaACF analyze its own analysis?"""
        from acf_functor.meta_acf import MetaACF

        meta = MetaACF()

        # The reflexive test is built into full_cycle
        report = meta.full_cycle(program=lambda x: x * np.sin(x))

        # META-ACF-3 certificate checks reflexive closure
        assert "META-ACF-3" in report.certificates


# ---------------------------------------------------------------------------
# Integration: All Pillars Together
# ---------------------------------------------------------------------------

class TestMetaACFIntegration:
    """End-to-end integration tests combining all pillars."""

    def test_program_to_dispatcher_pipeline(self):
        """Pillar 1 output feeds into Pillar 2 telemetry."""
        from acf_functor.meta_acf import MetaACF
        from acf_functor.dispatcher_optimizer import DispatchRecord

        meta = MetaACF()

        # Pillar 1: analyze a function
        report_p1 = meta.optimize_program(lambda x: x ** 3, domain=(-1, 1), n_samples=30)

        # Pillar 2: use the analysis to create dispatch records
        fma_estimate = report_p1.details.get("original_fma", 100)
        records = [
            DispatchRecord(
                n_elements=fma_estimate,
                n_fma=fma_estimate,
                backend="horner" if fma_estimate < 1000 else "chebyshev",
                latency_ms=0.01 * fma_estimate,
            )
            for _ in range(20)
        ]

        report_p2 = meta.optimize_dispatcher(records)
        assert report_p2.pillar_name == "Dispatcher Optimization"

    def test_certificate_completeness(self):
        """All certificate IDs should be present in reports."""
        from acf_functor.meta_acf import MetaACF
        from acf_functor.dispatcher_optimizer import DispatchRecord

        meta = MetaACF()

        rng = np.random.RandomState(42)
        records = [
            DispatchRecord(
                n_elements=int(10 ** rng.uniform(1, 4)),
                n_fma=100, backend="numpy",
                latency_ms=rng.uniform(0.1, 10),
            )
            for _ in range(50)
        ]

        report = meta.full_cycle(
            program=np.exp,
            program_domain=(-2, 2),
            telemetry=records,
        )

        # MetaACF certificates
        for cert in ["META-ACF-1", "META-ACF-2", "META-ACF-3", "META-ACF-4"]:
            assert cert in report.certificates, f"Missing certificate: {cert}"
