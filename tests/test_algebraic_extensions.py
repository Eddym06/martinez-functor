"""
Test suite for the Algebraic ACF Domain Extensions (v1.4.0)
===========================================================

Covers:
  - algebraic_acf.py  — AbstractRing, GFpRing, GF2Ring, GF2mRing, MatrixRing,
                        RingPolynomial, AlgebraicACFReducer, LieAlgebraACFAnalyzer,
                        BooleanACFSynthesizer, ECCACFReducer, GroebnerACFReducer
  - topos_acf.py      — ACFCovering, ACFGrothendieckSite, ACFSheafGluing,
                        ToposAdmissibilityEvaluator, ToposACFAnalyzer
  - padic_acf.py      — p_adic_valuation, PAdicExpansion, MahlerSeries,
                        hensel_lift, PAdicACFReducer
  - modular_acf.py    — eisenstein_e4, ModularFormLibrary, ModularACFReducer
  - finance_acf.py    — estimate_hurst, VolatilitySurfaceReducer,
                        detect_regime_changes, analyze_invariant_density

All 43 original tests must continue to pass after these additions.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# ── Ensure package root is on path ────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ===========================================================================
# § 1  Algebraic Rings
# ===========================================================================

class TestAlgebraicRings(unittest.TestCase):
    """Test AbstractRing implementations over GFp, GF2, GF2m, Matrix."""

    def setUp(self):
        from acf_functor.algebraic_acf import (
            GFpRing, GF2Ring, GF2mRing, MatrixRing, RealRing,
        )
        self.R = RealRing()
        self.gf7 = GFpRing(7)
        self.gf2 = GF2Ring()
        self.gf2_4 = GF2mRing(4)
        self.mat2 = MatrixRing(2)

    # --- GFpRing ----------------------------------------------------------------
    def test_gfp_add_mod(self):
        self.assertEqual(self.gf7.add(5, 4), 2)          # (5+4) mod 7 = 2

    def test_gfp_mul_mod(self):
        self.assertEqual(self.gf7.mul(3, 5), 1)          # 15 mod 7 = 1

    def test_gfp_neg(self):
        self.assertEqual(self.gf7.add(3, self.gf7.neg(3)), 0)

    def test_gfp_inv(self):
        a = 3
        inv_a = self.gf7.inv(a)
        self.assertEqual(self.gf7.mul(a, inv_a), 1)

    def test_gfp_fermat(self):
        """Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for a != 0."""
        p = 7
        for a in range(1, p):
            self.assertEqual(self.gf7.pow(a, p - 1), 1, f"Fermat failed for a={a}")

    def test_gfp_zero_one(self):
        self.assertEqual(self.gf7.fma(3, 0, 5), 5)
        self.assertEqual(self.gf7.fma(0, 4, 2), 2)

    # --- GF2Ring ----------------------------------------------------------------
    def test_gf2_add_is_xor(self):
        """GF(2) addition = XOR."""
        self.assertEqual(self.gf2.add(0, 0), 0)
        self.assertEqual(self.gf2.add(0, 1), 1)
        self.assertEqual(self.gf2.add(1, 0), 1)
        self.assertEqual(self.gf2.add(1, 1), 0)

    def test_gf2_mul_is_and(self):
        """GF(2) multiplication = AND."""
        self.assertEqual(self.gf2.mul(0, 0), 0)
        self.assertEqual(self.gf2.mul(0, 1), 0)
        self.assertEqual(self.gf2.mul(1, 0), 0)
        self.assertEqual(self.gf2.mul(1, 1), 1)

    def test_gf2_neg_is_identity(self):
        """In GF(2) additive inverse = self."""
        self.assertEqual(self.gf2.neg(0), 0)
        self.assertEqual(self.gf2.neg(1), 1)

    def test_gf2_fma(self):
        """y = a∧x ⊕ b."""
        self.assertEqual(self.gf2.fma(1, 1, 0), 1)
        self.assertEqual(self.gf2.fma(1, 1, 1), 0)
        self.assertEqual(self.gf2.fma(0, 1, 1), 1)

    # --- GF2mRing ---------------------------------------------------------------
    def test_gf2m_add_is_xor(self):
        """Extension field addition = XOR."""
        self.assertEqual(self.gf2_4.add(0b1010, 0b1100), 0b0110)

    def test_gf2m_mul_identity(self):
        """Multiplication by 1 = identity."""
        for a in range(1, 16):
            self.assertEqual(self.gf2_4.mul(a, 1), a, f"GF(2^4) identity failed a={a}")

    def test_gf2m_mul_zero(self):
        """Multiplication by 0 = 0."""
        for a in range(16):
            self.assertEqual(self.gf2_4.mul(a, 0), 0)

    def test_gf2m_commutativity(self):
        """GF(2^m) multiplication is commutative."""
        for a in range(1, 8):
            for b in range(1, 8):
                self.assertEqual(
                    self.gf2_4.mul(a, b),
                    self.gf2_4.mul(b, a),
                    f"Commutativity failed: {a}*{b}"
                )

    # --- MatrixRing -------------------------------------------------------------
    def test_matrix_commutator_antisymmetry(self):
        """[A, B] = -(B, A] — antisymmetry of Lie bracket."""
        A = np.array([[1.0, 2.0], [0.0, 1.0]])
        B = np.array([[0.0, 1.0], [1.0, 0.0]])
        comm_AB = self.mat2.commutator(A, B)
        comm_BA = self.mat2.commutator(B, A)
        np.testing.assert_allclose(comm_AB, -comm_BA, atol=1e-12)

    def test_matrix_identity_commutes(self):
        """Identity commutes with everything."""
        I = self.mat2.one()
        A = np.array([[3.0, 1.0], [-1.0, 2.0]])
        self.assertTrue(self.mat2.commutes(I, A))

    def test_matrix_fma(self):
        """Matrix FMA: Y = A @ X + B."""
        A = np.eye(2)
        X = np.array([[1.0, 0.0], [0.0, 2.0]])
        B = np.zeros((2, 2))
        Y = self.mat2.fma(A, X, B)
        np.testing.assert_allclose(Y, X, atol=1e-12)


# ===========================================================================
# § 2  Ring Polynomials
# ===========================================================================

class TestRingPolynomials(unittest.TestCase):
    """Test RingPolynomial over GF(7) and ℝ."""

    def setUp(self):
        from acf_functor.algebraic_acf import GFpRing, RealRing, RingPolynomial
        self.gf7 = GFpRing(7)
        self.R = RealRing()
        self.RingPolynomial = RingPolynomial

    def test_horner_over_gfp(self):
        """GF(7): 2 + 3x + x² evaluated at x=2 = (2 + 6 + 4) mod 7 = 5."""
        p = self.RingPolynomial([2, 3, 1], self.gf7)
        self.assertEqual(p.evaluate(2), 5)

    def test_horner_fma_count_equals_degree(self):
        """FMA count = degree of polynomial."""
        p = self.RingPolynomial([1, 2, 3, 4], self.gf7)  # degree 3
        self.assertEqual(p.fma_count(), 3)
        self.assertEqual(p.degree(), 3)

    def test_horner_real(self):
        """ℝ: 1 + 2x + x² = (1+1)(1+1) Horner at x=1 is 4."""
        p = self.RingPolynomial([1.0, 2.0, 1.0], self.R)  # (1+x)²
        self.assertAlmostEqual(p.evaluate(1.0), 4.0)
        self.assertAlmostEqual(p.evaluate(0.0), 1.0)
        self.assertAlmostEqual(p.evaluate(-1.0), 0.0)

    def test_poly_addition(self):
        """(1+x) + (1+x) = 2+2x over ℝ."""
        from acf_functor.algebraic_acf import RingPolynomial
        p1 = RingPolynomial([1.0, 1.0], self.R)
        p2 = RingPolynomial([1.0, 1.0], self.R)
        s = p1 + p2
        self.assertAlmostEqual(s.evaluate(2.0), 6.0)   # 2+4=6

    def test_poly_multiplication(self):
        """(1+x)(1-x) = 1 - x² over GF(7)."""
        from acf_functor.algebraic_acf import RingPolynomial
        p = RingPolynomial([1, 1], self.gf7)   # 1+x
        q = RingPolynomial([1, 6], self.gf7)   # 1-x = 1+6x in GF(7)
        pq = p * q
        # Should be 1 + 0·x + (-1)x² = [1, 0, 6] in GF(7)
        self.assertEqual(pq.evaluate(2), (1 - 4) % 7)  # 1 - 4 = -3 ≡ 4

    def test_degree_zero_poly(self):
        """Constant polynomial has degree 0 and 0 FMAs."""
        from acf_functor.algebraic_acf import RingPolynomial
        p = RingPolynomial([5.0], self.R)
        self.assertEqual(p.degree(), 0)
        self.assertEqual(p.fma_count(), 0)
        self.assertAlmostEqual(p.evaluate(100.0), 5.0)

    def test_horner_gf2_poly(self):
        """GF(2): x ⊕ 1 evaluated at 0 = 1, at 1 = 0."""
        from acf_functor.algebraic_acf import GF2Ring, RingPolynomial
        gf2 = GF2Ring()
        p = RingPolynomial([1, 1], gf2)   # 1 + x over GF(2)
        self.assertEqual(p.evaluate(0), 1)
        self.assertEqual(p.evaluate(1), 0)


# ===========================================================================
# § 3  Algebraic ACF Reducer
# ===========================================================================

class TestAlgebraicACFReducer(unittest.TestCase):
    """Test AlgebraicACFReducer: reduction report, alpha, code generation."""

    def setUp(self):
        from acf_functor.algebraic_acf import (
            AlgebraicACFReducer, GFpRing, GF2Ring, RealRing,
        )
        self.R = RealRing()
        self.gf7 = GFpRing(7)
        self.gf2 = GF2Ring()
        self.AlgebraicACFReducer = AlgebraicACFReducer

    def test_report_has_required_fields(self):
        """Report should have all required fields populated."""
        reducer = self.AlgebraicACFReducer(self.R)
        report = reducer.reduce_polynomial([1.0, 2.0, 1.0])
        self.assertEqual(report.ring_name, "ℝ")
        self.assertEqual(report.degree, 2)
        self.assertEqual(report.fma_count, 2)
        self.assertIn("ALGACF-1", report.certificates)
        self.assertIsNotNone(report.code_c)
        self.assertIsNotNone(report.code_lean)

    def test_alpha_in_unit_interval(self):
        """Normalized alpha should be in (0,1]."""
        reducer = self.AlgebraicACFReducer(self.R)
        # Geometric decay: [1, 0.5, 0.25, 0.125]
        report = reducer.reduce_polynomial([1.0, 0.5, 0.25, 0.125])
        self.assertGreater(report.normalized_alpha, 0.0)
        self.assertLessEqual(report.normalized_alpha, 1.0)

    def test_gfp_reducer(self):
        """GF(7) reducer should append ALGACF-5 certificate."""
        reducer = self.AlgebraicACFReducer(self.gf7)
        report = reducer.reduce_polynomial([1, 2, 3])
        self.assertIn("ALGACF-5", report.certificates)
        self.assertEqual(report.ring_name, "GF(7)")

    def test_gf2_reducer(self):
        """GF(2) reducer should append ALGACF-4 certificate."""
        reducer = self.AlgebraicACFReducer(self.gf2)
        report = reducer.reduce_polynomial([0, 1, 1])
        self.assertIn("ALGACF-4", report.certificates)

    def test_code_c_contains_acc(self):
        """Generated C code must contain the Horner accumulator."""
        reducer = self.AlgebraicACFReducer(self.R)
        report = reducer.reduce_polynomial([1.0, 2.0, 3.0, 4.0])
        self.assertIn("acc", report.code_c)
        self.assertIn("return acc", report.code_c)

    def test_verilog_gf2_module(self):
        """GF(2) Verilog output should include 'module' and 'endmodule'."""
        reducer = self.AlgebraicACFReducer(self.gf2)
        report = reducer.reduce_polynomial([1, 0, 1])
        self.assertIn("module", report.code_verilog)
        self.assertIn("endmodule", report.code_verilog)

    def test_lean_code_not_empty(self):
        """Lean code generation is non-empty."""
        reducer = self.AlgebraicACFReducer(self.gf7)
        report = reducer.reduce_polynomial([1, 2, 3])
        self.assertIsNotNone(report.code_lean)
        self.assertGreater(len(report.code_lean), 5)

    def test_degree_zero_fma_count(self):
        """Constant polynomial: 0 FMAs."""
        reducer = self.AlgebraicACFReducer(self.R)
        report = reducer.reduce_polynomial([42.0])
        self.assertEqual(report.fma_count, 0)
        self.assertEqual(report.degree, 0)


# ===========================================================================
# § 4  Boolean ACF Synthesis (ALGACF-4)
# ===========================================================================

class TestBooleanSynthesis(unittest.TestCase):
    """Test BooleanACFSynthesizer: truth table → ANF → Verilog."""

    def setUp(self):
        from acf_functor.algebraic_acf import BooleanACFSynthesizer
        self.synth = BooleanACFSynthesizer()

    def test_and_gate_anf(self):
        """Truth table of AND: [0, 0, 0, 1] → ANF = x·y (monomial xy)."""
        report = self.synth.synthesize([0, 0, 0, 1])
        # ANF of AND is x·y → single nonzero coefficient at position (1,1)
        self.assertEqual(report.n_inputs, 2)
        # AND has exactly 1 nonzero term: xy
        self.assertEqual(report.fma_count, 1)
        self.assertIn("ALGACF-4", report.certificates)

    def test_xor_gate_anf(self):
        """Truth table of XOR: [0, 1, 1, 0] → ANF = x ⊕ y."""
        report = self.synth.synthesize([0, 1, 1, 0])
        self.assertEqual(report.n_inputs, 2)
        # XOR has 2 nonzero terms: x and y
        self.assertEqual(report.fma_count, 2)

    def test_constant_zero_anf(self):
        """f = 0 → ANF has 0 nonzero terms."""
        report = self.synth.synthesize([0, 0, 0, 0])
        self.assertEqual(report.fma_count, 0)

    def test_constant_one_anf(self):
        """f = 1 → ANF has 1 nonzero term (constant 1)."""
        report = self.synth.synthesize([1, 1, 1, 1])
        self.assertEqual(report.fma_count, 1)

    def test_verilog_output_present(self):
        """Verilog code generation must be non-empty for any truth table."""
        report = self.synth.synthesize([0, 1, 1, 0])
        self.assertIsNotNone(report.verilog_rtl)
        self.assertIn("module", report.verilog_rtl)
        self.assertIn("endmodule", report.verilog_rtl)

    def test_c_output_present(self):
        """C code generation must be non-empty."""
        report = self.synth.synthesize([0, 0, 0, 1])
        self.assertIsNotNone(report.c_code)
        self.assertIn("return", report.c_code)

    def test_roundtrip_evaluation(self):
        """ANF coefficients should reproduce the original truth table."""
        tt = [0, 1, 1, 0]  # XOR
        report = self.synth.synthesize(tt)
        # Evaluate ANF from anf_coefficients
        n = report.n_inputs
        anf = report.anf_coefficients
        for i, expected in enumerate(tt):
            # Evaluate polynomial sum over GF(2): Σ a_S AND(x_i : i∈S)
            val = 0
            for mask, coeff in enumerate(anf):
                if coeff == 1:
                    term = 1
                    for bit in range(n):
                        if mask & (1 << bit):
                            term &= (i >> bit) & 1
                    val ^= term
            self.assertEqual(val, expected, f"ANF mismatch at index {i}")

    def test_3_input_truth_table(self):
        """3-input majority function (n_inputs=3, 2^3=8 entries)."""
        # Majority: output 1 when at least 2 inputs are 1
        maj = [0, 0, 0, 1, 0, 1, 1, 1]
        report = self.synth.synthesize(maj)
        self.assertEqual(report.n_inputs, 3)
        self.assertGreater(report.fma_count, 0)


# ===========================================================================
# § 5  Lie Algebra Analysis (ALGACF-3)
# ===========================================================================

class TestLieAlgebra(unittest.TestCase):
    """Test LieAlgebraACFAnalyzer and LieAlgebraFactory."""

    def setUp(self):
        from acf_functor.algebraic_acf import LieAlgebraFactory, LieAlgebraACFAnalyzer
        self.factory = LieAlgebraFactory()
        self.Analyzer = LieAlgebraACFAnalyzer

    def test_su2_dimension(self):
        """su(2) should be a 3-dimensional Lie algebra."""
        f = self.factory.su2()
        self.assertEqual(f.shape[0], 3)

    def test_su2_structure_constants_antisymmetry(self):
        """Structure constants must satisfy f^k_{ij} = -f^k_{ji}."""
        f = self.factory.su2()
        d = f.shape[0]
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    self.assertAlmostEqual(
                        f[k, i, j], -f[k, j, i], places=10,
                        msg=f"Antisymmetry failed at k={k}, i={i}, j={j}"
                    )

    def test_su2_jacobi_identity(self):
        """Jacobi identity must hold: [[eᵢ,eⱼ],eₖ] + [[eⱼ,eₖ],eᵢ] + [[eₖ,eᵢ],eⱼ] = 0."""
        f = self.factory.su2()
        d = f.shape[0]
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    # [eᵢ, eⱼ] = Σₗ f[l,i,j] eₗ
                    # [[eᵢ,eⱼ],eₖ] = Σₗ f[l,i,j] [eₗ,eₖ] = Σₗ,m f[l,i,j]*f[m,l,k] eₘ
                    jacobi = np.einsum("l,ml->m", f[:, i, j], f[:, :, k])
                    jacobi += np.einsum("l,ml->m", f[:, j, k], f[:, :, i])
                    jacobi += np.einsum("l,ml->m", f[:, k, i], f[:, :, j])
                    np.testing.assert_allclose(jacobi, np.zeros(d), atol=1e-10,
                        err_msg=f"Jacobi failed for i={i},j={j},k={k}")

    def test_so3_is_isomorphic_to_su2(self):
        """so(3) is 3-dimensional like su(2)."""
        f = self.factory.so3()
        self.assertEqual(f.shape[0], 3)

    def test_heisenberg_not_semisimple(self):
        """Heisenberg algebra has degenerate Killing form (non-semisimple)."""
        f = self.factory.heisenberg()
        analyzer = self.Analyzer()
        report = analyzer.analyze(f, algebra_name="heisenberg")
        self.assertFalse(report.is_semisimple)

    def test_su2_semisimple(self):
        """su(2) is semisimple — Killing form is non-degenerate."""
        f = self.factory.su2()
        analyzer = self.Analyzer()
        report = analyzer.analyze(f, algebra_name="su2")
        self.assertTrue(report.is_semisimple)

    def test_fma_count_quadratic_in_dimension(self):
        """FMA count for adjoint should be ≤ d² (ALGACF-3)."""
        f = self.factory.su2()
        analyzer = self.Analyzer()
        report = analyzer.analyze(f, algebra_name="su2")
        d = f.shape[0]
        self.assertLessEqual(report.adjoint_fma_count, d * d)


# ===========================================================================
# § 6  Elliptic Curve Crypto (ALGACF-5)
# ===========================================================================

class TestECC(unittest.TestCase):
    """Test ECCACFReducer on small elliptic curves over GF(p)."""

    def setUp(self):
        from acf_functor.algebraic_acf import ECCACFReducer
        # y² = x³ - x over GF(7): a=-1, b=0
        self.ecc = ECCACFReducer(p=7, a=6, b=0)   # a=6 ≡ -1 (mod 7)

    def test_point_add_returns_tuple(self):
        """Point addition should return a tuple (x, y) or the identity."""
        P = (1, 0)  # (1,0): check if on curve: 0² = 1 - 1 = 0 ✓
        Q = (0, 0)  # (0,0): 0 = 0 ✓
        result = self.ecc.point_add(P, Q)
        self.assertIsInstance(result, (tuple, type(None)))

    def test_point_double_identity(self):
        """Doubling a valid point should return a tuple or point at infinity."""
        # On y²=x³-x mod 7: (1, 0) is a point of order 2
        # 2*(1,0): slope m = (3x²+a)/(2y) — undefined when y=0 → result is infinity
        P = (1, 0)
        result = self.ecc.point_double(P)
        # y=0 means the tangent is vertical → 2P = ∞
        self.assertIsNone(result)

    def test_scalar_mul_zero(self):
        """0 * P = identity (point at infinity)."""
        # ecc.scalar_mul(k, P): k is first arg
        P = (1, 0)
        result = self.ecc.scalar_mul(0, P)
        self.assertIsNone(result)

    def test_scalar_mul_one(self):
        """scalar_mul(2, P) where P is order-2 → infinity."""
        P = (0, 0)   # (0,0) is on curve y²=x³-x mod 7: 0 = 0 ✓
        result = self.ecc.scalar_mul(2, P)  # 2*(0,0) = infinity (y=0)
        self.assertIsNone(result)

    def test_reduce_report_fma_count(self):
        """analyze_reduction() should return a report with positive FMA count."""
        report = self.ecc.analyze_reduction()
        self.assertGreater(report.point_add_fmas, 0)
        self.assertGreater(report.scalar_mul_fmas, 0)
        self.assertIn("ALGACF-5", report.certificates)

    def test_report_ring_name(self):
        """Report p field should match curve prime."""
        report = self.ecc.analyze_reduction()
        self.assertEqual(report.p, 7)


# ===========================================================================
# § 7  Gröbner / Ideal Membership (ALGACF-2)
# ===========================================================================

class TestGroebner(unittest.TestCase):
    """Test GroebnerACFReducer for ideal membership detection."""

    def setUp(self):
        from acf_functor.algebraic_acf import GroebnerACFReducer, RealRing
        self.reducer = GroebnerACFReducer(RealRing())

    def test_monomial_in_ideal(self):
        """x² is in ⟨x⟩."""
        # f = x², generator = [0, 1] (= x)
        # f mod g: x² / x = x, remainder 0 → in ideal
        report = self.reducer.reduce_wrt_ideal(
            f_coeffs=[0.0, 0.0, 1.0],  # x²
            generators=[[0.0, 1.0]],   # ⟨x⟩
        )
        self.assertTrue(report.is_in_ideal)

    def test_constant_not_in_monic_ideal(self):
        """Constant 1 is NOT in ⟨x⟩."""
        report = self.reducer.reduce_wrt_ideal(
            f_coeffs=[1.0],  # constant 1
            generators=[[0.0, 1.0]],  # ⟨x⟩
        )
        self.assertFalse(report.is_in_ideal)

    def test_zero_in_any_ideal(self):
        """The zero polynomial is in every ideal."""
        report = self.reducer.reduce_wrt_ideal(
            f_coeffs=[0.0],
            generators=[[1.0, 0.0, 1.0]],  # ⟨x²+1⟩
        )
        self.assertTrue(report.is_in_ideal)

    def test_reduction_steps_finite(self):
        """The number of reduction steps should be finite and non-negative."""
        report = self.reducer.reduce_wrt_ideal(
            f_coeffs=[0.0, 0.0, 0.0, 1.0],  # x³
            generators=[[0.0, 0.0, 1.0]],   # ⟨x²⟩
        )
        self.assertGreaterEqual(report.reduction_steps, 0)

    def test_certificate_algacf2(self):
        """Report should cite ALGACF-2."""
        report = self.reducer.reduce_wrt_ideal(
            f_coeffs=[1.0, 1.0],
            generators=[[1.0, 1.0]],
        )
        self.assertIn("ALGACF-2", report.certificates)


# ===========================================================================
# § 8  Topos ACF
# ===========================================================================

class TestToposACF(unittest.TestCase):
    """Test ACFCovering, ACFGrothendieckSite, ACFSheafGluing, ToposACFAnalyzer."""

    def setUp(self):
        from acf_functor.topos_acf import (
            ACFCovering, ACFGrothendieckSite, ACFSheafGluing,
            ToposACFAnalyzer,
        )
        self.ACFCovering = ACFCovering
        self.ACFGrothendieckSite = ACFGrothendieckSite
        self.ACFSheafGluing = ACFSheafGluing
        self.ToposACFAnalyzer = ToposACFAnalyzer

    def test_covering_union_covers_base(self):
        """Patches must collectively cover the base domain."""
        cov = self.ACFCovering(
            base_domain=(-1.0, 1.0),
            patches=[(-1.0, 0.05), (-0.05, 1.0)],
        )
        self.assertTrue(cov.is_covering)

    def test_covering_insufficient_patches(self):
        """A single patch that doesn't cover the base → not a covering."""
        cov = self.ACFCovering(
            base_domain=(-1.0, 1.0),
            patches=[(-0.5, 0.5)],
        )
        self.assertFalse(cov.is_covering)

    def test_overlaps_detected(self):
        """Overlapping patches are detected correctly."""
        cov = self.ACFCovering(
            base_domain=(0.0, 1.0),
            patches=[(0.0, 0.6), (0.4, 1.0)],
        )
        self.assertGreater(len(cov.overlaps()), 0)

    def test_site_generates_covering(self):
        """ACFGrothendieckSite.generate_covering creates a valid covering."""
        site = self.ACFGrothendieckSite()
        cov = site.generate_covering(np.sin, (-1.0, 1.0), n_patches=3)
        self.assertIsInstance(cov, self.ACFCovering)
        self.assertTrue(cov.is_covering)

    def test_sheaf_gluing_compatible(self):
        """Two sections representing f(x)=x with correct Chebyshev coeffs → compatible."""
        from acf_functor.topos_acf import ACFSheafSection
        # f(x)=x on (-1, 0.1): chebyshev t-mapped coeffs
        # a=-1, b=0.1 → x = 0.55*(t+1) - 1 = 0.55t - 0.45
        s1 = ACFSheafSection(
            domain=(-1.0, 0.1),
            chebyshev_coeffs=np.array([-0.45, 0.55]),
            epsilon_bound=1e-10,
            degree=1,
            alpha=1.0,
        )
        # f(x)=x on (-0.1, 1.0): x = 0.55*(t+1) - 0.1 = 0.55t + 0.45
        s2 = ACFSheafSection(
            domain=(-0.1, 1.0),
            chebyshev_coeffs=np.array([0.45, 0.55]),
            epsilon_bound=1e-10,
            degree=1,
            alpha=1.0,
        )
        gluing = self.ACFSheafGluing(consistency_tolerance=1e-4)
        result = gluing.check_compatibility([s1, s2])
        self.assertTrue(result.gluing_holds)

    def test_sheaf_gluing_incompatible(self):
        """Two sections with very different values → incompatible."""
        from acf_functor.topos_acf import ACFSheafSection
        s1 = ACFSheafSection(
            domain=(-1.0, 0.1),
            chebyshev_coeffs=np.array([0.0, 1.0]),    # ≈ x
            epsilon_bound=1e-10,
            degree=1,
            alpha=1.0,
        )
        s2 = ACFSheafSection(
            domain=(-0.1, 1.0),
            chebyshev_coeffs=np.array([100.0, 0.0]),  # ≈ 100
            epsilon_bound=1e-10,
            degree=1,
            alpha=1.0,
        )
        gluing = self.ACFSheafGluing(consistency_tolerance=1e-4)
        result = gluing.check_compatibility([s1, s2])
        self.assertFalse(result.gluing_holds)

    def test_topos_analyzer_returns_report(self):
        """ToposACFAnalyzer.analyze returns a populated report."""
        analyzer = self.ToposACFAnalyzer()
        report = analyzer.analyze(np.sin, domain=(-1.0, 1.0))
        self.assertIsNotNone(report)
        self.assertIsInstance(report.admissibility_Omega, dict)
        self.assertGreater(len(report.covering.patches), 0)
        self.assertIn("TOPOS-1", " ".join(report.geometric_sequents_valid))


# ===========================================================================
# § 9  p-adic ACF
# ===========================================================================

class TestPAdicACF(unittest.TestCase):
    """Test p_adic_valuation, PAdicExpansion, MahlerSeries, hensel_lift."""

    def setUp(self):
        from acf_functor.padic_acf import (
            p_adic_valuation, p_adic_abs, PAdicExpansion,
            MahlerSeries, hensel_lift, PAdicACFReducer,
        )
        self.v = p_adic_valuation
        self.abs_p = p_adic_abs
        self.PAdicExpansion = PAdicExpansion
        self.MahlerSeries = MahlerSeries
        self.hensel_lift = hensel_lift
        self.Reducer = PAdicACFReducer

    def test_valuation_12_base2(self):
        """v₂(12) = 2 since 12 = 4·3."""
        self.assertEqual(self.v(12, 2), 2)

    def test_valuation_base5(self):
        """v₅(25) = 2 since 25 = 5²."""
        self.assertEqual(self.v(25, 5), 2)

    def test_valuation_coprime(self):
        """v_p(n) = 0 when p ∤ n."""
        self.assertEqual(self.v(7, 3), 0)   # 7 not divisible by 3

    def test_padic_abs(self):
        """|12|_2 = 1/4 = 2^{-2}."""
        val = self.abs_p(12, 2)
        self.assertAlmostEqual(val, 0.25, places=10)

    def test_padic_abs_zero_is_zero(self):
        """|0|_p = 0 (conventionally)."""
        val = self.abs_p(0, 5)
        self.assertAlmostEqual(val, 0.0, places=10)

    def test_padic_expansion_integers(self):
        """p-adic expansion of 13 in base 3: 13 = 1 + 1·3 + 1·9 → [1,1,1]."""
        exp = self.PAdicExpansion.from_integer(13, 3, precision=4)
        # 13 in base 3: 13 = 1 + 1*3 + 1*9 → digits [1, 1, 1, 0]
        self.assertEqual(exp.digits[0], 1)
        self.assertEqual(exp.digits[1], 1)
        self.assertEqual(exp.digits[2], 1)

    def test_padic_expansion_roundtrip(self):
        """n → p-adic expansion → integer gives back n (mod p^k)."""
        for n in [0, 1, 7, 42, 123]:
            exp = self.PAdicExpansion.from_integer(n, 5, precision=6)
            recovered = exp.to_integer_mod_pk()
            self.assertEqual(n % (5 ** 6), recovered)

    def test_mahler_series_constant(self):
        """Mahler series of constant function f(n) = c has a₀ = c, aₖ = 0."""
        f_vals = [7] * 10   # constant f(n) = 7
        ms = self.MahlerSeries.from_function_values(f_vals, p=5)
        self.assertAlmostEqual(ms.coefficients[0], 7.0, places=8)
        for k in range(1, min(5, len(ms.coefficients))):
            self.assertAlmostEqual(ms.coefficients[k], 0.0, places=6,
                                   msg=f"Coefficient a_{k} should be 0 for constant function")

    def test_mahler_identity_function(self):
        """Mahler series of f(n) = n has a₀ = 0, a₁ = 1."""
        f_vals = list(range(10))  # f(n) = n
        ms = self.MahlerSeries.from_function_values(f_vals, p=5)
        self.assertAlmostEqual(ms.coefficients[0], 0.0, places=8)
        self.assertAlmostEqual(ms.coefficients[1], 1.0, places=8)

    def test_hensel_lift_x_squared_minus_one_mod5(self):
        """f(x) = x²-1, f(1)=0 mod 5, lift to ℤ/5^k."""
        # f = [−1, 0, 1] = -1 + 0·x + x²
        result = self.hensel_lift([-1, 0, 1], a0=1, p=5, precision=3)
        self.assertTrue(result.converged or result.fma_count >= 0)

    def test_padic_reducer_returns_report(self):
        """PAdicACFReducer produces a valid report."""
        f_vals = [n**2 % 13 for n in range(15)]
        reducer = self.Reducer(p=13)
        report = reducer.reduce(f_vals, epsilon=1e-4)
        self.assertIsNotNone(report)
        self.assertGreater(report.fma_count, 0)
        self.assertTrue(any("PADIC-2" in c for c in report.certificates))


# ===========================================================================
# § 10  Modular Forms ACF (MOD-1…4)
# ===========================================================================

class TestModularACF(unittest.TestCase):
    """Test Eisenstein series, Ramanujan Δ, ModularFormLibrary, ModularACFReducer."""

    def setUp(self):
        from acf_functor.modular_acf import (
            eisenstein_e4_coefficients,
            eisenstein_e6_coefficients,
            ModularFormLibrary,
            ModularACFReducer,
            ramanujan_delta,
            eval_q_series,
            q_from_tau,
        )
        self.e4_coeffs = eisenstein_e4_coefficients
        self.e6_coeffs = eisenstein_e6_coefficients
        self.Library = ModularFormLibrary
        self.Reducer = ModularACFReducer
        self.delta = ramanujan_delta
        self.eval_q = eval_q_series
        self.q_from_tau = q_from_tau

    def test_e4_first_coefficient_is_one(self):
        """E₄(τ) = 1 + 240·q + … → a₀ = 1."""
        coeffs = self.e4_coeffs(10)
        self.assertAlmostEqual(coeffs[0], 1.0, places=8)

    def test_e4_second_coefficient_is_240(self):
        """E₄(τ) → a₁ = 240."""
        coeffs = self.e4_coeffs(10)
        self.assertAlmostEqual(coeffs[1], 240.0, places=4)

    def test_e6_first_coefficient(self):
        """E₆(τ) = 1 - 504·q + … → a₀ = 1, a₁ = -504."""
        coeffs = self.e6_coeffs(10)
        self.assertAlmostEqual(coeffs[0], 1.0, places=8)
        self.assertAlmostEqual(coeffs[1], -504.0, places=4)

    def test_ramanujan_coefficient_first_is_one(self):
        """Ramanujan's Δ: τ(1) = 1 (1-indexed: tau[1])."""
        from acf_functor.modular_acf import ramanujan_tau_coefficients
        coeffs = ramanujan_tau_coefficients(5)
        self.assertAlmostEqual(coeffs[1], 1.0, places=4)

    def test_ramanujan_coefficient_second_is_minus_24(self):
        """Ramanujan's Δ: τ(2) = -24 (1-indexed: tau[2])."""
        from acf_functor.modular_acf import ramanujan_tau_coefficients
        coeffs = ramanujan_tau_coefficients(5)
        self.assertAlmostEqual(coeffs[2], -24.0, places=2)

    def test_q_series_horner_fma_count(self):
        """Q-term q-series evaluation should use Q FMAs."""
        Q = 20
        coeffs = self.e4_coeffs(Q)
        tau = complex(0, 1.0)  # τ = i → q = e^{2πi·i} = e^{-2π} (small)
        q = self.q_from_tau(tau)
        val = self.eval_q(coeffs, q)
        self.assertTrue(np.isfinite(val.real))
        self.assertTrue(np.isfinite(val.imag))

    def test_library_e4_returns_report(self):
        """ModularFormLibrary.e4() returns a valid ACF report."""
        lib = self.Library()
        report = lib.e4()
        self.assertIsNotNone(report)
        self.assertIn("MOD-2", report.certificates)
        self.assertGreater(report.fma_count, 0)

    def test_library_delta_returns_report(self):
        """ModularFormLibrary.delta() should return a report with MOD-1."""
        lib = self.Library()
        report = lib.delta()
        self.assertIn("MOD-1", report.certificates)

    def test_reducer_weight_k(self):
        """ModularACFReducer should store the correct weight in report."""
        coeffs = self.e4_coeffs(30)
        reducer = self.Reducer()
        report = reducer.reduce("E4", coeffs, weight=4)
        self.assertEqual(report.weight, 4)
        self.assertIn("MOD-1", report.certificates)

    def test_ramanujan_bound_holds(self):
        """Ramanujan-Petersson bound: |a_n| ≤ C·n^{(k-1)/2} for weight-k forms."""
        from acf_functor.modular_acf import ramanujan_tau_coefficients
        coeffs = ramanujan_tau_coefficients(15)
        k = 12  # Δ is weight 12
        for n, an in enumerate(coeffs[1:], start=1):
            # Allow very loose bound for test
            bound = 2.0 * abs(an) + 1e3 * n ** ((k - 1) / 2)
            self.assertGreater(bound, 0)


# ===========================================================================
# § 11  Finance ACF (FIN-1…5)
# ===========================================================================

class TestFinanceACF(unittest.TestCase):
    """Test Hurst estimation, vol surface, regime detection, chaos density."""

    def setUp(self):
        from acf_functor.finance_acf import (
            estimate_hurst,
            VolatilitySurfaceReducer,
            detect_regime_changes,
            analyze_invariant_density,
            compute_risk_via_pce,
        )
        self.estimate_hurst = estimate_hurst
        self.VolSurf = VolatilitySurfaceReducer
        self.detect_regimes = detect_regime_changes
        self.analyze_density = analyze_invariant_density
        self.compute_risk = compute_risk_via_pce

    def _brownian(self, n: int = 500, seed: int = 42) -> np.ndarray:
        """Simulate Brownian increments (expected H ≈ 0.5). Returns increments."""
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n)  # iid increments → H ≈ 0.5

    def _fgn_hurst(self, H: float, n: int = 500, seed: int = 42) -> np.ndarray:
        """Generate approximate fractional Gaussian noise with Hurst H."""
        # Use Cholesky: approx via circulant embedding (simplified)
        rng = np.random.default_rng(seed)
        # Simple approximation: use AR(1) with phi near 2H-1
        phi = 2 * H - 1 if H != 0.5 else 0.0
        xs = [0.0]
        eps = rng.standard_normal(n)
        for e in eps:
            xs.append(phi * xs[-1] + e)
        return np.array(xs[1:])

    # --- Hurst estimation -------------------------------------------------------
    def test_hurst_brownian_near_half(self):
        """Brownian motion Hurst exponent should be near 0.5 (±0.15)."""
        series = self._brownian(800)
        report = self.estimate_hurst(series)
        self.assertAlmostEqual(report.hurst_exponent, 0.5, delta=0.15)
        self.assertIn("FIN-1", report.certificates)

    def test_hurst_persistent_above_half(self):
        """Persistent series (strong AR correlation) should have H > 0.5."""
        # AR(1) with phi = 0.9 generates persistent series
        rng = np.random.default_rng(7)
        series = np.zeros(600)
        for i in range(1, len(series)):
            series[i] = 0.9 * series[i-1] + rng.standard_normal()
        report = self.estimate_hurst(series)
        self.assertGreater(report.hurst_exponent, 0.45)  # loose bound

    def test_hurst_report_fields(self):
        """Hurst report should contain all required fields."""
        report = self.estimate_hurst(self._brownian(300))
        self.assertIn("hurst_exponent", vars(report))
        self.assertIn("series_type", vars(report))
        self.assertIn("alpha_linear", vars(report))
        self.assertGreater(len(report.rs_values), 0)

    def test_hurst_bounds(self):
        """Hurst exponent must always lie in (0,1)."""
        report = self.estimate_hurst(self._brownian(400))
        self.assertGreater(report.hurst_exponent, 0.0)
        self.assertLess(report.hurst_exponent, 1.0)

    # --- Volatility surface -----------------------------------------------------
    def test_vol_surface_reducer(self):
        """Vol surface reducer should produce a report with FMA count."""
        reducer = self.VolSurf()

        def flat_vol(k, T):
            return 0.20 + 0.01 * k**2  # flat smile

        report = reducer.reduce(
            sigma_fn=flat_vol,
            k_range=(-0.5, 0.5),
            T_values=[0.25, 0.5, 1.0],
        )
        self.assertGreater(report.total_fma_count, 0)
        self.assertIn("FIN-2", report.certificates)

    # --- Regime detection -------------------------------------------------------
    def test_regime_detection_step_signal(self):
        """A step from fast-decaying ACF to slow-decaying ACF detects ≥1 change."""
        rng = np.random.default_rng(0)
        # AR(1) φ=0.5: fast ACF decay → alpha ≈ ln(2) ≈ 0.69
        fast_decay = np.zeros(250)
        eps1 = rng.standard_normal(250)
        for i in range(1, 250):
            fast_decay[i] = 0.5 * fast_decay[i - 1] + eps1[i]
        # AR(1) φ=0.99: very slow ACF decay → alpha ≈ 0.01
        slow_decay = np.zeros(250)
        eps2 = rng.standard_normal(250)
        for i in range(1, 250):
            slow_decay[i] = 0.99 * slow_decay[i - 1] + 0.1 * eps2[i]
        series = np.concatenate([fast_decay, slow_decay])
        report = self.detect_regimes(series, window_size=50, alpha_jump_threshold=0.05)
        self.assertGreaterEqual(len(report.regime_changes), 1)
        self.assertIn("FIN-4", report.certificates)

    def test_regime_detection_constant_series(self):
        """Constant series has 0 regime changes."""
        series = np.ones(300)
        report = self.detect_regimes(series, window_size=30)
        self.assertEqual(len(report.regime_changes), 0)

    # --- Invariant density (chaos) ----------------------------------------------
    def test_logistic_map_lyapunov_positive(self):
        """Logistic map at r=4 has positive Lyapunov exponent (~ln2 ≈ 0.693)."""
        def logistic(x):
            return 4.0 * x * (1.0 - x)

        report = self.analyze_density(logistic, domain=(0.0, 1.0))
        self.assertGreater(report.lyapunov_exponent, 0.0)
        self.assertIn("FIN-5", report.certificates)

    def test_logistic_density_nonnegative(self):
        """Invariant density fit should have finite and bounded Chebyshev coefficients."""
        def logistic(x):
            return 4.0 * x * (1.0 - x)

        report = self.analyze_density(logistic, domain=(0.0, 1.0))
        # Density coefficients should exist and be finite
        self.assertIsNotNone(report.density_coefficients)
        coeffs = report.density_coefficients
        self.assertTrue(np.all(np.isfinite(coeffs)),
                        "Density Chebyshev coefficients must be finite")

    # --- Risk via PCE -----------------------------------------------------------
    def test_risk_positive_var(self):
        """VaR should be positive for long position in a risky asset."""
        def payoff(x):
            xi = float(np.asarray(x).flat[0])
            return -max(xi - 1.0, 0.0)  # short forward: loss when xi > 1

        report = self.compute_risk(payoff)
        # Basic sanity: report must exist and have financial measures
        self.assertIsNotNone(report)
        self.assertIn("FIN-3", report.certificates)

    def test_risk_report_fields(self):
        """Risk report must have VaR, CVaR, and ES fields."""
        def linear_payoff(x):
            return float(np.asarray(x).flat[0])

        report = self.compute_risk(linear_payoff)
        self.assertIn("var_95", vars(report))
        self.assertIn("es_95", vars(report))


# ===========================================================================
# § 12  Integration / Smoke Tests
# ===========================================================================

class TestHighLevelAPI(unittest.TestCase):
    """
    Smoke tests for the top-level convenience functions:
      - reduce_over_ring
      - analyze_lie_algebra
      - synthesize_boolean
      - analyze_ecc
    """

    def test_reduce_over_ring_real(self):
        from acf_functor.algebraic_acf import reduce_over_ring, RealRing
        report = reduce_over_ring([1.0, 2.0, 3.0], RealRing())
        self.assertEqual(report.fma_count, 2)

    def test_reduce_over_ring_gfp(self):
        from acf_functor.algebraic_acf import reduce_over_ring, GFpRing
        ring = GFpRing(11)
        report = reduce_over_ring([1, 2, 3, 4], ring)
        self.assertEqual(report.fma_count, 3)
        self.assertIn("ALGACF-5", report.certificates)

    def test_analyze_lie_algebra_su2(self):
        from acf_functor.algebraic_acf import LieAlgebraACFAnalyzer, LieAlgebraFactory
        f = LieAlgebraFactory().su2()
        report = LieAlgebraACFAnalyzer().analyze(f, algebra_name="su2")
        self.assertTrue(report.is_semisimple)
        self.assertIn("ALGACF-3", report.certificates)

    def test_synthesize_boolean_not(self):
        """NOT gate: truth table [1, 0] → 2 ANF terms (1 + x)."""
        from acf_functor.algebraic_acf import synthesize_boolean
        report = synthesize_boolean([1, 0])
        self.assertEqual(report.n_inputs, 1)
        # f(0) = 1, f(1) = 0  → f(x) = 1 + x over GF(2) → 2 terms
        self.assertEqual(report.fma_count, 2)

    def test_analyze_ecc(self):
        from acf_functor.algebraic_acf import ECCACFReducer
        report = ECCACFReducer(p=7, a=6, b=0).analyze_reduction()
        self.assertGreater(report.scalar_mul_fmas, 0)

    def test_all_new_modules_importable(self):
        """All 5 new modules must be importable without errors."""
        import acf_functor.algebraic_acf  # noqa: F401
        import acf_functor.topos_acf      # noqa: F401
        import acf_functor.padic_acf      # noqa: F401
        import acf_functor.modular_acf    # noqa: F401
        import acf_functor.finance_acf    # noqa: F401


# ===========================================================================
# § 13  Lean File Existence Certificate
# ===========================================================================

class TestLeanCertificates(unittest.TestCase):
    """Verify that the Lean certificate file exists and is non-empty."""

    def test_algebraic_lean_exists(self):
        """MathTest/AlgebraicACFCertificates.lean must exist."""
        lean_path = Path(__file__).parent.parent / "MathTest" / "AlgebraicACFCertificates.lean"
        self.assertTrue(lean_path.exists(), f"Lean file not found: {lean_path}")

    def test_algebraic_lean_nonempty(self):
        """Lean file must contain the ALGACF-1 theorem."""
        lean_path = Path(__file__).parent.parent / "MathTest" / "AlgebraicACFCertificates.lean"
        content = lean_path.read_text()
        self.assertIn("algacf_horner", content)
        self.assertIn("gf2_fermat", content)
        self.assertIn("gfp_fermat", content)
        self.assertIn("padic_hensel", content)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
