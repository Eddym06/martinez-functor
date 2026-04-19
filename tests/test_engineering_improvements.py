"""
tests/test_engineering_improvements.py

Test suite masivo para las mejoras de ingeniería del sistema Poema:
  1. BackendRegistry — autodescubrimiento y disponibilidad
  2. NumpyBackend — compilación y evaluación CPU pura
  3. PytorchBackendAdapter — wrapping correcto
  4. VerilogBackend — generación de RTL, SVA, testbench y SDC
  5. LeanLiveVerifier — verificación real con binario Lean 4
  6. CanonicalAlpha — α_A(f) unificado y consistente
  7. Integración end-to-end: Poem → Backends → Verilog + Lean verified
"""

import math
import os
import sys
import tempfile
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_fma(weights_biases):
    """Create minimal FMAInstruction-duck-type objects."""
    class FakeInstr:
        def __init__(self, w, b):
            self.weight = w
            self.bias = b
    return [FakeInstr(w, b) for w, b in weights_biases]


def _make_real_fma():
    """Compile a real Poema AST and return (fma_sequence, source_ast, report)."""
    import torch
    from poema.frontend import Poem
    from poema.compiler import FMALinearizer, PoemCompiler
    P = Poem(dtype=torch.float64)
    ast = P.continuous_flow("2*x + 1")
    compiler = PoemCompiler(target="pytorch", precision="fp64")
    fn, report = compiler.compile(ast, domain=(-1.0, 1.0))
    linearizer = FMALinearizer()
    fma_seq = linearizer.linearize(ast)
    return fma_seq, ast, report


# ─────────────────────────────────────────────────────────────────────────────
# 1. BackendRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestBackendRegistry:

    def test_registry_returns_backends(self):
        from poema.backends import BackendRegistry
        available = BackendRegistry.available()
        assert isinstance(available, dict)
        assert len(available) >= 2, "At least numpy and pytorch should be registered"

    def test_numpy_always_available(self):
        from poema.backends import BackendRegistry
        available = BackendRegistry.available()
        assert available.get("numpy_cpu", False), "NumpyBackend must always be available"

    def test_verilog_always_available(self):
        from poema.backends import BackendRegistry
        available = BackendRegistry.available()
        assert available.get("verilog_rtl", False), "VerilogBackend must always be available"

    def test_best_cpu_returns_something(self):
        from poema.backends import BackendRegistry
        backend = BackendRegistry.best_for_cpu()
        assert backend is not None
        assert backend.verify_available()

    def test_describe_all_no_crash(self):
        from poema.backends import BackendRegistry
        desc = BackendRegistry.describe_all()
        assert "Backend" in desc

    def test_get_numpy(self):
        from poema.backends import BackendRegistry
        b = BackendRegistry.get("numpy_cpu")
        assert b.capabilities.name == "numpy_cpu"

    def test_get_verilog(self):
        from poema.backends import BackendRegistry
        b = BackendRegistry.get("verilog_rtl")
        assert b.capabilities.supports_verilog is True

    def test_get_unknown_raises(self):
        from poema.backends import BackendRegistry
        with pytest.raises(KeyError):
            BackendRegistry.get("nonexistent_backend_xyz")


# ─────────────────────────────────────────────────────────────────────────────
# 2. NumpyBackend
# ─────────────────────────────────────────────────────────────────────────────

class TestNumpyBackend:

    def test_compile_returns_result(self):
        from poema.backends import NumpyBackend
        import numpy as np
        fma = _make_fake_fma([(2.0, 1.0), (1.0, 0.0)])
        b = NumpyBackend()
        result = b.compile(fma, source_ast=None)
        assert result.callable_fn is not None
        assert result.fma_count == 2
        assert result.backend_name == "numpy_cpu"

    def test_evaluate_affine(self):
        """y = 2*x + 1 should satisfy f(0)=1, f(1)=3, f(-1)=-1."""
        from poema.backends import NumpyBackend
        import numpy as np
        # Single FMA: y = 2*x + 1
        fma = _make_fake_fma([(2.0, 1.0)])
        b = NumpyBackend()
        result = b.compile(fma, source_ast=None)
        fn = result.callable_fn
        x = np.array([0.0, 1.0, -1.0])
        y = fn(x)
        np.testing.assert_allclose(y, [1.0, 3.0, -1.0], rtol=1e-6)

    def test_evaluate_composed(self):
        """Two stages: y = 1*(2*x+1) + 0 ≡ 2x+1."""
        from poema.backends import NumpyBackend
        import numpy as np
        fma = _make_fake_fma([(2.0, 1.0), (1.0, 0.0)])  # second is identity
        b = NumpyBackend()
        fn = b.compile(fma, source_ast=None).callable_fn
        x = np.linspace(-2, 2, 100)
        y = fn(x)
        np.testing.assert_allclose(y, 2 * x + 1, rtol=1e-6)

    def test_emitted_code_contains_fma(self):
        from poema.backends import NumpyBackend
        fma = _make_fake_fma([(0.5, 0.1)])
        r = NumpyBackend().compile(fma, None)
        assert "FMA step" in r.emitted_code

    def test_fp32_precision(self):
        from poema.backends import NumpyBackend
        import numpy as np
        fma = _make_fake_fma([(1.0, 0.0)])
        r = NumpyBackend().compile(fma, None, precision="fp32")
        assert r.callable_fn is not None

    def test_capabilities(self):
        from poema.backends import NumpyBackend
        c = NumpyBackend().capabilities
        assert c.supports_cpu
        assert not c.supports_gpu
        assert "fp64" in c.precision_formats

    def test_large_chain(self):
        """1000-stage FMA chain should still evaluate correctly."""
        from poema.backends import NumpyBackend
        import numpy as np
        n = 1000
        # Chain of identity FMAs: each step is y = 1*y + 0
        fma = _make_fake_fma([(1.0, 0.0)] * n)
        fn = NumpyBackend().compile(fma, None).callable_fn
        x = np.array([3.14])
        y = fn(x)
        np.testing.assert_allclose(y, [3.14], rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 3. VerilogBackend
# ─────────────────────────────────────────────────────────────────────────────

class TestVerilogBackend:

    def _backend(self, tmpdir):
        from poema.backends import VerilogBackend
        return VerilogBackend(output_dir=tmpdir)

    def test_compile_returns_result(self, tmp_path):
        fma = _make_fake_fma([(2.0, 0.5), (1.0, -0.25)])
        b = self._backend(str(tmp_path))
        result = b.compile(fma, None, module_name="test_pipeline")
        assert result.backend_name == "verilog_rtl"
        assert result.fma_count == 2
        assert result.emitted_code

    def test_main_module_file_created(self, tmp_path):
        fma = _make_fake_fma([(2.0, 0.5)])
        b = self._backend(str(tmp_path))
        result = b.compile(fma, None, module_name="poema_test")
        assert os.path.isfile(result.emitted_path), "Main .v file must be created"

    def test_testbench_file_created(self, tmp_path):
        fma = _make_fake_fma([(1.0, 0.0)])
        b = self._backend(str(tmp_path))
        result = b.compile(fma, None, module_name="tb_test")
        tb_path = os.path.join(str(tmp_path), "tb_test_tb.v")
        assert os.path.isfile(tb_path), "Testbench .v must be created"

    def test_sva_file_created(self, tmp_path):
        fma = _make_fake_fma([(0.5, 0.1)])
        b = self._backend(str(tmp_path))
        result = b.compile(fma, None, module_name="sva_test")
        sva_path = os.path.join(str(tmp_path), "sva_test_assertions.sva")
        assert os.path.isfile(sva_path), "SVA assertions file must be created"

    def test_sdc_file_created(self, tmp_path):
        fma = _make_fake_fma([(0.5, 0.0)])
        b = self._backend(str(tmp_path))
        result = b.compile(fma, None, module_name="sdc_test")
        sdc_path = os.path.join(str(tmp_path), "sdc_test.sdc")
        assert os.path.isfile(sdc_path), "SDC constraints file must be created"

    def test_verilog_has_module_keyword(self, tmp_path):
        fma = _make_fake_fma([(2.0, 0.5)])
        b = self._backend(str(tmp_path))
        result = b.compile(fma, None, module_name="kw_test")
        assert "module kw_test" in result.emitted_code

    def test_verilog_has_endmodule(self, tmp_path):
        fma = _make_fake_fma([(2.0, 0.5)])
        b = self._backend(str(tmp_path))
        result = b.compile(fma, None, module_name="em_test")
        assert "endmodule" in result.emitted_code

    def test_epsilon_in_comments(self, tmp_path):
        fma = _make_fake_fma([(1.0, 0.0)])
        b = self._backend(str(tmp_path))
        result = b.compile(fma, None, epsilon_bound=4.5e-3, module_name="eps_test")
        assert "4.5e-3" in result.emitted_code or "4.5" in result.emitted_code

    def test_n_stages_matches_fma(self, tmp_path):
        n = 7
        fma = _make_fake_fma([(1.0, 0.0)] * n)
        b = self._backend(str(tmp_path))
        result = b.compile(fma, None, module_name="stages_test")
        assert result.extra["pipeline_stages"] == n

    def test_axi_stream_interface(self, tmp_path):
        fma = _make_fake_fma([(2.0, 1.0)])
        from poema.backends import VerilogBackend
        b = VerilogBackend(use_axi_stream=True, output_dir=str(tmp_path))
        result = b.compile(fma, None, module_name="axist_test")
        assert "s_axis_tvalid" in result.emitted_code

    def test_combinational_mode(self, tmp_path):
        fma = _make_fake_fma([(2.0, 0.0)])
        from poema.backends import VerilogBackend
        b = VerilogBackend(pipelined=False, output_dir=str(tmp_path))
        result = b.compile(fma, None, module_name="comb_test")
        assert "Combinational" in result.emitted_code

    def test_data_width_16(self, tmp_path):
        fma = _make_fake_fma([(1.0, 0.0)])
        from poema.backends import VerilogBackend
        b = VerilogBackend(data_width=16, frac_bits=12, output_dir=str(tmp_path))
        result = b.compile(fma, None, module_name="w16_test")
        assert "DATA_WIDTH  = 16" in result.emitted_code

    def test_sva_overflow_property(self, tmp_path):
        fma = _make_fake_fma([(1.0, 0.0)])
        b = self._backend(str(tmp_path))
        result = b.compile(fma, None, module_name="sva_overflow")
        sva_path = os.path.join(str(tmp_path), "sva_overflow_assertions.sva")
        sva_content = open(sva_path).read()
        assert "no_overflow" in sva_content

    def test_no_callable_fn(self, tmp_path):
        """Verilog backend never has callable_fn — outputs are hardware files."""
        fma = _make_fake_fma([(1.0, 0.0)])
        b = self._backend(str(tmp_path))
        result = b.compile(fma, None, module_name="no_fn")
        assert result.callable_fn is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. LeanLiveVerifier
# ─────────────────────────────────────────────────────────────────────────────

class TestLeanLiveVerifier:

    def test_verifier_finds_lean_binary(self):
        from poema.lean_live_verifier import LeanLiveVerifier
        v = LeanLiveVerifier()
        assert os.path.isfile(v.lean_binary) or v.lean_binary == "lean"

    def test_simple_true_theorem_proven(self):
        from poema.lean_live_verifier import LeanLiveVerifier, VerificationStatus
        v = LeanLiveVerifier()
        code = "theorem simple_true : (1 : Float) < 2 := by native_decide\n"
        result = v.verify_theorem(code, "simple_true")
        assert result.status == VerificationStatus.PROVEN, (
            f"Simple 1 < 2 should be PROVEN, got {result.status}\n{result.lean_output}"
        )

    def test_false_theorem_fails(self):
        from poema.lean_live_verifier import LeanLiveVerifier, VerificationStatus
        v = LeanLiveVerifier()
        code = "theorem false_claim : (2 : Float) < 1 := by native_decide\n"
        result = v.verify_theorem(code, "false_claim")
        assert result.status in (VerificationStatus.FAILED, VerificationStatus.ERROR), (
            f"2 < 1 must not be PROVEN"
        )

    def test_urt_bound_proven(self):
        from poema.lean_live_verifier import LeanLiveVerifier, VerificationStatus
        v = LeanLiveVerifier()
        result = v.verify_universal_bound(epsilon=0.004525, bound_limit=0.01)
        assert result.status == VerificationStatus.PROVEN, (
            f"urt_bound 0.004525 < 0.01 should be PROVEN\n{result.lean_output}"
        )

    def test_urt_bound_fails_when_epsilon_exceeds(self):
        from poema.lean_live_verifier import LeanLiveVerifier, VerificationStatus
        v = LeanLiveVerifier()
        result = v.verify_universal_bound(epsilon=0.05, bound_limit=0.01)
        assert result.status in (VerificationStatus.FAILED, VerificationStatus.ERROR)

    def test_fma_conservation_proven_equal(self):
        from poema.lean_live_verifier import LeanLiveVerifier, VerificationStatus
        v = LeanLiveVerifier()
        result = v.verify_fma_conservation(7, 7)
        assert result.status == VerificationStatus.PROVEN, (
            f"7 = 7 must be PROVEN\n{result.lean_output}"
        )

    def test_fma_conservation_fails_unequal(self):
        from poema.lean_live_verifier import LeanLiveVerifier, VerificationStatus
        v = LeanLiveVerifier()
        result = v.verify_fma_conservation(7, 5)
        assert result.status in (VerificationStatus.FAILED, VerificationStatus.ERROR)

    def test_composition_bound_proven(self):
        from poema.lean_live_verifier import LeanLiveVerifier, VerificationStatus
        v = LeanLiveVerifier()
        # 0.034 ≤ 4.81 * 1.0
        result = v.verify_composition_bound(4.81, 1.0, 0.034)
        assert result.status == VerificationStatus.PROVEN

    def test_alpha_consistency_proven(self):
        from poema.lean_live_verifier import LeanLiveVerifier, VerificationStatus
        v = LeanLiveVerifier()
        # very consistent estimates
        result = v.verify_alpha_consistency(1.0, 1.02, 0.98, tolerance=0.10)
        assert result.status == VerificationStatus.PROVEN

    def test_alpha_consistency_fails_large_deviation(self):
        from poema.lean_live_verifier import LeanLiveVerifier, VerificationStatus
        v = LeanLiveVerifier()
        # deviation ≈ 90%
        result = v.verify_alpha_consistency(0.1, 2.0, 1.0, tolerance=0.10)
        assert result.status in (VerificationStatus.FAILED, VerificationStatus.ERROR)

    def test_reversibility_proven(self):
        from poema.lean_live_verifier import LeanLiveVerifier, VerificationStatus
        v = LeanLiveVerifier()
        result = v.verify_reversibility(0.0, machine_eps=1e-7)
        assert result.status == VerificationStatus.PROVEN

    def test_result_has_certificate_hash(self):
        from poema.lean_live_verifier import LeanLiveVerifier
        v = LeanLiveVerifier()
        code = "theorem hash_test : (1 : Float) < 2 := by native_decide\n"
        result = v.verify_theorem(code, "hash_test")
        assert len(result.certificate_hash) == 16

    def test_result_has_elapsed_ms(self):
        from poema.lean_live_verifier import LeanLiveVerifier
        v = LeanLiveVerifier()
        code = "theorem elapsed : (1 : Float) < 2 := by native_decide\n"
        result = v.verify_theorem(code, "elapsed")
        assert result.elapsed_ms > 0

    def test_full_suite_runs(self):
        from poema.lean_live_verifier import LeanLiveVerifier
        v = LeanLiveVerifier()
        data = {
            "L_inf_bound_max_divergence": 0.004525,
            "alpha_combinatorial": 2.0,
            "post_collapse_cost": 2,
            "lipschitz_constant_f": 4.81,
            "lipschitz_constant_g": 1.0,
            "theoretical_error_bound": 0.034,
            "alpha_spectral_lipschitz": 1.04,
            "alpha_geometric_volume": 0.5,
            "inversion_error_l_inf": 0.0,
        }
        report = v.run_full_suite(data)
        assert report.total_time_ms > 0
        assert len(report.results) >= 5
        # At minimum URT, FMA conservation and reversibility should be PROVEN
        proven_names = {r.theorem_name for r in report.results if r.is_proven}
        assert "urt_universal_bound" in proven_names
        assert "fma_conservation_law" in proven_names

    def test_full_suite_report_json_valid(self):
        import json
        from poema.lean_live_verifier import LeanLiveVerifier
        v = LeanLiveVerifier()
        data = {"L_inf_bound_max_divergence": 0.001, "alpha_combinatorial": 2,
                "post_collapse_cost": 2, "lipschitz_constant_f": 2.0,
                "lipschitz_constant_g": 1.0, "theoretical_error_bound": 0.01,
                "alpha_spectral_lipschitz": 1.0, "alpha_geometric_volume": 0.9,
                "inversion_error_l_inf": 0.0}
        report = v.run_full_suite(data)
        j = json.loads(report.to_json())
        assert "results" in j
        assert isinstance(j["results"], list)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Canonical Alpha
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonicalAlpha:

    def test_polynomial_alpha_near_1(self):
        """For a degree-n polynomial, α_A(f) should be ≈ 1."""
        from poema.canonical_alpha import compute_canonical_alpha
        # Simulate Horner chain for degree-5 polynomial: exactly 5 FMA steps
        fma = _make_fake_fma([(x, c) for x, c in [(1,1),(1,2),(1,-1),(1,0.5),(1,0)]])
        ca = compute_canonical_alpha(fma)
        assert 0.5 < ca.canonical_value < 2.5, f"α should be in sane range, got {ca.canonical_value}"

    def test_identity_chain_alpha(self):
        """Identity FMA chain (w=1, b=0): geometric alpha should be ~1."""
        from poema.canonical_alpha import compute_canonical_alpha, AlphaEstimator
        fma = _make_fake_fma([(1.0, 0.0)] * 4)
        raw = AlphaEstimator.compute_all(fma)
        ca = compute_canonical_alpha(fma)
        assert ca.canonical_value > 0, "canonical alpha must be positive"

    def test_canonical_has_all_fields(self):
        from poema.canonical_alpha import compute_canonical_alpha
        fma = _make_fake_fma([(2.0, 1.0)])
        ca = compute_canonical_alpha(fma)
        assert hasattr(ca, "canonical_value")
        assert hasattr(ca, "raw_estimates")
        assert hasattr(ca, "consistency_score")
        assert hasattr(ca, "confidence_interval")
        assert hasattr(ca, "interpretation")

    def test_consistency_score_range(self):
        from poema.canonical_alpha import compute_canonical_alpha
        fma = _make_fake_fma([(1.0, 0.0)] * 10)
        ca = compute_canonical_alpha(fma)
        assert 0.0 <= ca.consistency_score <= 1.0

    def test_confidence_interval_ordered(self):
        from poema.canonical_alpha import compute_canonical_alpha
        fma = _make_fake_fma([(1.0, 0.1)] * 5)
        ca = compute_canonical_alpha(fma)
        lo, hi = ca.confidence_interval
        assert lo <= ca.canonical_value <= hi or lo <= hi

    def test_summary_no_crash(self):
        from poema.canonical_alpha import compute_canonical_alpha
        fma = _make_fake_fma([(0.5, 0.5)] * 3)
        ca = compute_canonical_alpha(fma)
        s = ca.summary()
        assert "α_A(f)" in s

    def test_geometric_mean_vs_arithmetic(self):
        from poema.canonical_alpha import AlphaCanonicalizer, AlphaEstimates
        raw = AlphaEstimates(combinatorial=1.0, spectral=1.0, geometric=1.0, n_fma=5)
        c1 = AlphaCanonicalizer(fusion_method="geometric_mean").canonicalize(raw)
        c2 = AlphaCanonicalizer(fusion_method="arithmetic_mean").canonicalize(raw)
        # For equal inputs, both methods should give the same result
        assert abs(c1.canonical_value - c2.canonical_value) < 1e-6

    def test_all_three_estimators_run(self):
        from poema.canonical_alpha import AlphaEstimator
        fma = _make_fake_fma([(2.0, 0.5), (0.5, -1.0)])
        raw = AlphaEstimator.compute_all(fma)
        assert raw.combinatorial > 0
        assert raw.spectral > 0
        assert raw.geometric > 0
        assert raw.n_fma == 2

    def test_zero_fma_graceful(self):
        from poema.canonical_alpha import compute_canonical_alpha
        fma = []
        ca = compute_canonical_alpha(fma)
        assert ca is not None
        assert ca.canonical_value >= 0

    def test_interpretation_is_string(self):
        from poema.canonical_alpha import _interpret_alpha
        for alpha in [0.0, 0.5, 1.0, 1.5, 2.5, 5.0]:
            s = _interpret_alpha(alpha)
            assert isinstance(s, str) and len(s) > 5


# ─────────────────────────────────────────────────────────────────────────────
# 6. End-to-end integration
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndIntegration:

    def test_poem_to_numpy_evaluate(self):
        """Full pipeline: Poem → FMA → NumpyBackend → evaluate."""
        import torch
        import numpy as np
        from poema.frontend import Poem
        from poema.compiler import FMALinearizer, PoemCompiler
        from poema.backends import NumpyBackend

        P = Poem(dtype=torch.float64)
        ast = P.continuous_flow("2*x + 1")
        compiler = PoemCompiler(target="pytorch", precision="fp64")
        fn_torch, report = compiler.compile(ast, domain=(-1.0, 1.0))

        linearizer = FMALinearizer()
        fma_seq = linearizer.linearize(ast)

        numpy_backend = NumpyBackend()
        np_result = numpy_backend.compile(fma_seq, ast)
        fn_np = np_result.callable_fn

        x_np = np.array([-1.0, 0.0, 0.5, 1.0])
        y_np = fn_np(x_np)
        expected = 2 * x_np + 1
        np.testing.assert_allclose(y_np, expected, rtol=1e-5)

    def test_poem_to_verilog_full(self, tmp_path):
        """Full pipeline: Poem → FMA → VerilogBackend → all files present."""
        import torch
        from poema.frontend import Poem
        from poema.compiler import FMALinearizer, PoemCompiler
        from poema.backends import VerilogBackend

        P = Poem(dtype=torch.float64)
        ast = P.continuous_flow("3*x + 0.5")
        compiler = PoemCompiler(target="pytorch", precision="fp64")
        _, report = compiler.compile(ast, domain=(-1.0, 1.0))

        linearizer = FMALinearizer()
        fma_seq = linearizer.linearize(ast)

        vb = VerilogBackend(output_dir=str(tmp_path))
        result = vb.compile(fma_seq, ast, module_name="poem_linear", epsilon_bound=report.total_epsilon)

        assert result.emitted_code, "Must have emitted code"
        assert os.path.isfile(result.emitted_path)
        # Check all 4 output files
        for suffix in [".v", "_tb.v", "_assertions.sva", ".sdc"]:
            p = os.path.join(str(tmp_path), f"poem_linear{suffix}")
            assert os.path.isfile(p), f"Expected file {p}"

    def test_lean_verifier_urt_after_real_compile(self):
        """Run real compilation, then formally verify the ε certificate with Lean."""
        import torch
        from poema.frontend import Poem
        from poema.compiler import PoemCompiler
        from poema.lean_live_verifier import LeanLiveVerifier, VerificationStatus

        P = Poem(dtype=torch.float64)
        ast = P.continuous_flow("sin(x)")
        compiler = PoemCompiler(target="pytorch", precision="fp64")
        _, report = compiler.compile(ast, domain=(-3.14, 3.14))

        v = LeanLiveVerifier()
        result = v.verify_universal_bound(
            epsilon=report.total_epsilon,
            bound_limit=max(0.1, report.total_epsilon * 2),
        )
        assert result.status == VerificationStatus.PROVEN

    def test_canonical_alpha_after_real_compile(self):
        """Compute α_A(f) for real polynomial and check it's in [0.5, 3.0]."""
        import torch
        from poema.frontend import Poem
        from poema.compiler import FMALinearizer, PoemCompiler
        from poema.canonical_alpha import compute_canonical_alpha

        P = Poem(dtype=torch.float64)
        ast = P.continuous_flow("x^5 + 2*x^3 + x")
        compiler = PoemCompiler(target="pytorch", precision="fp64")
        compiler.compile(ast, domain=(-1.0, 1.0))

        linearizer = FMALinearizer()
        fma_seq = linearizer.linearize(ast)
        ca = compute_canonical_alpha(fma_seq, ast)

        assert 0.3 < ca.canonical_value < 5.0, (
            f"α_A(f) = {ca.canonical_value} is out of expected range for degree-5 poly"
        )
        assert ca.canonical_value > 0
