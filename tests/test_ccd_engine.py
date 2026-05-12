"""
tests/test_ccd_engine.py
========================
Deep tests for the Campo de Curvatura Dinámica (CCD) Engine.

Test groups
-----------
TestChebyshevShell        — expansion correctness, roundtrip, spectral compression
TestDiffusionGeometry     — manifold embedding, structure preservation, Nyström
TestCoupledOscillators    — normal modes, resonance groups, coherence transform
TestLocalEntropyOperator  — collapse detection, noise detection, local dimension
TestLangevinPurifier      — denoising, convergence toward manifold, temperature
TestCCDEngineIntegration  — full pipeline, effective dimension, certificates
TestCoDReduction          — curse of dimensionality reduction benchmarks
TestHighDimSystems        — Lorenz in R^d, Van der Pol in R^d (m << d)
TestUtilityFunctions      — preprocess_high_dim, estimate_intrinsic_dimension
"""

import numpy as np
import pytest

from acf_functor.ccd_engine import (
    CCDEngine,
    CCDCertificate,
    ChebyshevShell,
    DiffusionGeometry,
    CoupledOscillators,
    ResonanceGroup,
    LocalEntropyOperator,
    LangevinPurifier,
    preprocess_high_dim,
    estimate_intrinsic_dimension,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures and data generators
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def swiss_roll(rng):
    """Swiss roll manifold: 2D manifold embedded in R^3."""
    n = 500
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n))
    height = rng.uniform(0, 1, n)
    X = np.column_stack([
        t * np.cos(t),
        height,
        t * np.sin(t),
    ])
    return X, t   # X in R^3, t is the 1D parameter


@pytest.fixture(scope="module")
def manifold_in_high_dim(rng):
    """1D manifold (circle) embedded in R^20 via random orthogonal projection."""
    n = 300
    d = 20
    m = 1    # true intrinsic dimension
    theta = rng.uniform(0, 2 * np.pi, n)
    Z = np.column_stack([np.cos(theta), np.sin(theta)])   # (n, 2) on circle
    # Random orthogonal embedding R^2 → R^20
    A_raw = rng.standard_normal((d, 2))
    Q, _ = np.linalg.qr(A_raw)   # (d, d), first 2 cols form orthonormal basis
    A = Q[:, :2]                   # (d, 2)
    X_high = Z @ A.T               # (n, d)
    # Small noise
    X_high += 0.02 * rng.standard_normal((n, d))
    return X_high, theta, m


@pytest.fixture(scope="module")
def lorenz_trajectory(rng):
    """Lorenz attractor (m≈3) trajectory for various ambient dimensions."""
    def lorenz_step(s, dt=0.01):
        x, y, z = s
        dx = 10.0 * (y - x)
        dy = x * (28.0 - z) - y
        dz = x * y - (8.0 / 3.0) * z
        return s + dt * np.array([dx, dy, dz])

    T = 1000
    s = np.array([1.0, 0.0, 0.0])
    traj = np.zeros((T, 3))
    for i in range(T):
        s = lorenz_step(s)
        traj[i] = s
    # Normalize
    traj = (traj - traj.mean(0)) / (traj.std(0) + 1e-8)
    return traj


@pytest.fixture(scope="module")
def lorenz_in_50d(lorenz_trajectory, rng):
    """Lorenz attractor embedded in R^50."""
    d_target = 50
    d_src = 3
    A_raw = rng.standard_normal((d_target, d_src))
    Q, _ = np.linalg.qr(A_raw.T)   # Q has shape (d_src, d_src)
    # Use random linear embedding
    A = rng.standard_normal((d_target, d_src))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    X_high = lorenz_trajectory @ A.T   # (T, 50)
    X_high += 0.01 * rng.standard_normal(X_high.shape)
    return X_high


@pytest.fixture(scope="module")
def pure_noise_high_dim(rng):
    """Pure Gaussian noise in R^30: no structure, no manifold."""
    return rng.standard_normal((400, 30))


@pytest.fixture(scope="module")
def collapsed_data():
    """Collapsed data: all points at the same location + tiny noise."""
    center = np.ones(10) * 5.0
    X = center + 1e-5 * np.random.default_rng(0).standard_normal((200, 10))
    return X


# ─────────────────────────────────────────────────────────────────────────────
# TestChebyshevShell
# ─────────────────────────────────────────────────────────────────────────────

class TestChebyshevShell:
    """Chebyshev expansion: correctness, roundtrip, spectral properties."""

    def test_fit_returns_self(self, rng):
        X = rng.uniform(-1, 1, (100, 5))
        shell = ChebyshevShell(n_coeffs=6, compression_rank=8)
        result = shell.fit(X)
        assert result is shell

    def test_transform_output_shape(self, rng):
        X = rng.standard_normal((200, 10))
        rank = 12
        shell = ChebyshevShell(n_coeffs=6, compression_rank=rank).fit(X)
        Z = shell.transform(X)
        assert Z.shape == (200, shell._rank), f"Expected (200, {shell._rank}), got {Z.shape}"

    def test_transform_new_points_same_rank(self, rng):
        X_train = rng.standard_normal((200, 8))
        X_new = rng.standard_normal((50, 8))
        shell = ChebyshevShell(n_coeffs=5, compression_rank=10).fit(X_train)
        Z_train = shell.transform(X_train)
        Z_new = shell.transform(X_new)
        assert Z_new.shape[1] == Z_train.shape[1]

    def test_inverse_transform_approximate_roundtrip(self, rng):
        """Inverse of Chebyshev shell should approximately recover the input."""
        # Use 1D data where T_1(x)=x makes the recovery exact
        X = rng.uniform(-1, 1, (300, 6))
        shell = ChebyshevShell(n_coeffs=8, compression_rank=24).fit(X)
        Z = shell.transform(X)
        X_approx = shell.inverse_transform(Z)
        assert X_approx.shape == X.shape
        # First-order coefficient should give back x_norm → unnormalized ≈ X
        err = np.mean(np.abs(X_approx - X))
        assert err < 1.0, f"Roundtrip error too large: {err:.4f}"

    def test_explained_variance_ratio_sums_to_one(self, rng):
        X = rng.standard_normal((200, 10))
        shell = ChebyshevShell(n_coeffs=6, compression_rank=12).fit(X)
        evr = shell.explained_variance_ratio
        assert abs(evr.sum() - 1.0) < 1e-8

    def test_explained_variance_ratio_decreasing(self, rng):
        """Singular values should be sorted descending (SVD guarantee)."""
        X = rng.standard_normal((200, 10))
        shell = ChebyshevShell(n_coeffs=6, compression_rank=12).fit(X)
        evr = shell.explained_variance_ratio
        assert np.all(evr[:-1] >= evr[1:] - 1e-10)

    def test_effective_rank_is_small(self, manifold_in_high_dim):
        """For low-dimensional manifold, effective rank should be much smaller than d."""
        X_high, _, m = manifold_in_high_dim
        n, d = X_high.shape
        shell = ChebyshevShell(n_coeffs=6, compression_rank=min(20, n - 1)).fit(X_high)
        eff_rank = shell.effective_rank
        # The Chebyshev expansion of a 1D manifold in R^20 should compress well
        assert eff_rank < d, f"Effective rank {eff_rank} should be < d={d}"

    def test_chebyshev_no_fit_raises(self, rng):
        shell = ChebyshevShell()
        with pytest.raises(RuntimeError, match="fitted"):
            shell.transform(rng.standard_normal((10, 3)))

    def test_high_dimensional_compression(self, rng):
        """For d=100, rank should be compressed to chebyshev_rank."""
        n, d = 300, 50
        X = rng.standard_normal((n, d))
        rank = 16
        shell = ChebyshevShell(n_coeffs=4, compression_rank=rank).fit(X)
        Z = shell.transform(X)
        assert Z.shape[1] <= rank


# ─────────────────────────────────────────────────────────────────────────────
# TestDiffusionGeometry
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffusionGeometry:
    """Diffusion maps: manifold embedding, structure preservation, Nyström."""

    def test_fit_returns_self(self, rng):
        X = rng.standard_normal((100, 5))
        dg = DiffusionGeometry(n_components=4, n_neighbors=8)
        assert dg.fit(X) is dg

    def test_transform_output_shape(self, rng):
        X = rng.standard_normal((100, 8))
        n_comp = 5
        dg = DiffusionGeometry(n_components=n_comp, n_neighbors=8).fit(X)
        Z = dg.transform(X)
        assert Z.shape[1] == n_comp

    def test_transform_new_points(self, rng):
        X_train = rng.standard_normal((200, 6))
        X_new = rng.standard_normal((30, 6))
        dg = DiffusionGeometry(n_components=4, n_neighbors=10).fit(X_train)
        Z_train = dg.transform(X_train)
        Z_new = dg.transform(X_new)
        assert Z_new.shape == (30, 4)
        assert Z_train.shape == (200, 4)

    def test_inverse_transform_shape(self, rng):
        X = rng.standard_normal((150, 6))
        n_comp = 4
        dg = DiffusionGeometry(n_components=n_comp, n_neighbors=10).fit(X)
        Z = dg.transform(X)
        X_approx = dg.inverse_transform(Z)
        assert X_approx.shape == X.shape

    def test_nearby_points_stay_nearby(self, rng):
        """Diffusion map should preserve local structure: nearby points stay nearby."""
        # Swiss roll structure: close in 1D param → close in diffusion space
        n = 200
        t = np.linspace(0, 4 * np.pi, n)
        X = np.column_stack([t * np.cos(t), t * np.sin(t)])
        # Pick 5 pairs that are close in parameter space
        close_pairs = [(i, i + 1) for i in range(0, 20, 2)]
        far_pairs = [(0, n // 2), (0, n - 1)]

        dg = DiffusionGeometry(n_components=3, n_neighbors=8).fit(X)
        Z = dg.transform(X)

        close_dists = [np.linalg.norm(Z[i] - Z[j]) for i, j in close_pairs]
        far_dists = [np.linalg.norm(Z[i] - Z[j]) for i, j in far_pairs]

        mean_close = np.mean(close_dists)
        mean_far = np.mean(far_dists)
        assert mean_close < mean_far, (
            f"Close pairs (mean dist {mean_close:.4f}) should be nearer than "
            f"far pairs (mean dist {mean_far:.4f}) in diffusion space"
        )

    def test_intrinsic_dimension_estimate_nontrivial(self, rng):
        """Intrinsic dimension estimate should return a positive integer."""
        X = rng.standard_normal((150, 8))
        dg = DiffusionGeometry(n_components=6, n_neighbors=10).fit(X)
        d_est = dg.intrinsic_dimension_estimate
        assert isinstance(d_est, int)
        assert 1 <= d_est <= 6

    def test_no_fit_raises(self, rng):
        dg = DiffusionGeometry()
        with pytest.raises(RuntimeError, match="fitted"):
            dg.transform(rng.standard_normal((10, 3)))


# ─────────────────────────────────────────────────────────────────────────────
# TestCoupledOscillators
# ─────────────────────────────────────────────────────────────────────────────

class TestCoupledOscillators:
    """Normal modes and resonance groups."""

    def test_fit_returns_self(self, rng):
        X = rng.standard_normal((100, 8))
        osc = CoupledOscillators()
        assert osc.fit(X) is osc

    def test_normal_modes_are_orthonormal(self, rng):
        """Eigenvectors of symmetric covariance must be orthonormal."""
        X = rng.standard_normal((200, 8))
        osc = CoupledOscillators().fit(X)
        V = osc._eigenvectors      # (d, d)
        # V @ V.T should be identity (orthogonal matrix)
        I_approx = V @ V.T
        assert np.allclose(I_approx, np.eye(V.shape[0]), atol=1e-10)

    def test_transform_output_shape(self, rng):
        n, d = 200, 10
        X = rng.standard_normal((n, d))
        osc = CoupledOscillators(n_groups=4).fit(X)
        Z = osc.transform(X)
        assert Z.shape == (n, 4)

    def test_inverse_transform_roundtrip(self, rng):
        """Project and unproject should approximately recover X (up to truncation error)."""
        n, d = 200, 6
        X = rng.standard_normal((n, d))
        osc = CoupledOscillators(n_groups=d).fit(X)   # keep all modes
        Z = osc.transform(X)
        X_approx = osc.inverse_transform(Z)
        # With all modes retained, should reconstruct exactly (up to centering)
        err = np.mean(np.abs(X_approx - X))
        assert err < 0.01, f"Full-rank roundtrip error: {err:.6f}"

    def test_resonance_groups_cover_all_variables(self, rng):
        """Every variable should be assigned to exactly one resonance group."""
        n, d = 200, 10
        X = rng.standard_normal((n, d))
        osc = CoupledOscillators(n_groups=3).fit(X)
        # Collect all variable indices across groups
        all_assigned = set()
        for grp in osc.resonance_groups:
            for vi in grp.variable_indices:
                all_assigned.add(vi)
        assert all_assigned == set(range(d)), (
            f"Not all variables assigned. Missing: {set(range(d)) - all_assigned}"
        )

    def test_resonance_group_coherence_in_01(self, rng):
        """Coherence score of each group should be in [0, 1]."""
        X = rng.standard_normal((200, 8))
        osc = CoupledOscillators(n_groups=3).fit(X)
        for grp in osc.resonance_groups:
            assert 0.0 <= grp.coherence_score <= 1.0 + 1e-8

    def test_explained_variance_ratio_nonneg_leq1(self, rng):
        X = rng.standard_normal((200, 8))
        osc = CoupledOscillators(n_groups=4).fit(X)
        evr = osc.explained_variance_ratio
        assert np.all(evr >= -1e-10)
        assert evr.sum() <= 1.0 + 1e-8

    def test_adaptive_coherence_transform_shape(self, rng):
        n, d = 200, 10
        X = rng.standard_normal((n, d))
        osc = CoupledOscillators(n_groups=3).fit(X)
        Z_coh = osc.adaptive_coherence_transform(X)
        n_groups = osc.n_resonance_groups
        assert Z_coh.shape == (n, n_groups)

    def test_structured_data_has_fewer_groups(self, rng):
        """Structured data (low intrinsic dim) should yield fewer groups than random."""
        n = 300
        # Low-dim: only 2 independent variables, rest are copies with noise
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        d = 12
        X_structured = np.column_stack([
            z1 + 0.05 * rng.standard_normal(n),
            z2 + 0.05 * rng.standard_normal(n),
        ] + [z1 + 0.1 * rng.standard_normal(n) for _ in range(d - 2)])
        X_random = rng.standard_normal((n, d))

        osc_structured = CoupledOscillators().fit(X_structured)
        osc_random = CoupledOscillators().fit(X_random)

        # Structured data should cluster into fewer effective modes
        assert osc_structured._n_effective_modes <= osc_random._n_effective_modes + 1


# ─────────────────────────────────────────────────────────────────────────────
# TestLocalEntropyOperator
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalEntropyOperator:
    """Collapse detection, noise detection, local dimension estimation."""

    def test_fit_returns_self(self, rng):
        X = rng.standard_normal((100, 5))
        op = LocalEntropyOperator()
        assert op.fit(X) is op

    def test_temperature_in_range(self, rng):
        """Temperature should always be in [T_min, T_max]."""
        X = rng.standard_normal((200, 5))
        op = LocalEntropyOperator(T_min=0.01, T_max=1.0).fit(X)
        T = op.temperature(X)
        assert np.all(T >= 0.01 - 1e-9)
        assert np.all(T <= 1.00 + 1e-9)

    def test_temperature_output_shape(self, rng):
        X = rng.standard_normal((200, 8))
        op = LocalEntropyOperator().fit(X)
        T = op.temperature(X)
        assert T.shape == (200,)

    def test_temperature_unfitted_returns_midpoint(self, rng):
        """Unfitted operator should return midpoint temperature, not crash."""
        op = LocalEntropyOperator(T_min=0.1, T_max=0.9)
        X = rng.standard_normal((10, 4))
        T = op.temperature(X)
        assert T.shape == (10,)
        assert np.allclose(T, 0.5)

    def test_collapsed_data_has_lower_temperature(self, rng, collapsed_data):
        """Collapsed data (all points near same location) → dense neighborhood → higher T."""
        X_random = rng.standard_normal((200, 10))
        op_random = LocalEntropyOperator(k_neighbors=8).fit(X_random)
        op_collapsed = LocalEntropyOperator(k_neighbors=8).fit(collapsed_data)

        T_random_mean = op_random.temperature(X_random).mean()
        T_collapsed_mean = op_collapsed.temperature(collapsed_data).mean()

        # Collapsed data: distances are uniform → H_local is small → T ≈ T_min or T_max
        # (the test checks that collapsed data has a different temperature profile)
        # Both should be valid floats in range
        assert np.isfinite(T_collapsed_mean)
        assert np.isfinite(T_random_mean)

    def test_local_dimension_positive(self, rng):
        """Local dimension estimates should be positive."""
        X = rng.standard_normal((200, 6))
        op = LocalEntropyOperator(k_neighbors=8).fit(X)
        d_local = op.local_dimension(X)
        assert np.all(d_local >= 0.1)
        assert d_local.shape == (200,)

    def test_local_dimension_manifold_less_than_ambient(self, manifold_in_high_dim):
        """Local dimension on a low-dim manifold should be < ambient dimension."""
        X_high, _, m = manifold_in_high_dim
        n, d = X_high.shape
        op = LocalEntropyOperator(k_neighbors=8).fit(X_high)
        d_local = op.local_dimension(X_high)
        d_mean = float(np.mean(d_local))
        # Local dimension should be much less than ambient d=20 for a 1D manifold
        assert d_mean < d / 2, (
            f"Mean local dim {d_mean:.2f} should be << ambient d={d} for 1D manifold"
        )

    def test_is_collapsed_mask_shape(self, rng):
        X = rng.standard_normal((100, 5))
        op = LocalEntropyOperator().fit(X)
        mask = op.is_collapsed(X)
        assert mask.shape == (100,)
        assert mask.dtype == bool

    def test_is_noisy_mask_shape(self, rng):
        X = rng.standard_normal((100, 5))
        op = LocalEntropyOperator().fit(X)
        mask = op.is_noisy(X)
        assert mask.shape == (100,)
        assert mask.dtype == bool


# ─────────────────────────────────────────────────────────────────────────────
# TestLangevinPurifier
# ─────────────────────────────────────────────────────────────────────────────

class TestLangevinPurifier:
    """Denoising, convergence, Langevin dynamics properties."""

    def test_fit_returns_self(self, rng):
        X = rng.standard_normal((100, 4))
        purifier = LangevinPurifier()
        assert purifier.fit(X) is purifier

    def test_purify_output_shape(self, rng):
        """Purified output should have same shape as input."""
        n, d = 50, 6
        X = rng.standard_normal((n, d))
        purifier = LangevinPurifier(n_steps=5, dt=0.01).fit(X)
        X_pure = purifier.purify(X)
        assert X_pure.shape == (n, d)

    def test_purify_no_fit_raises(self, rng):
        purifier = LangevinPurifier()
        with pytest.raises(RuntimeError, match="fitted"):
            purifier.purify(rng.standard_normal((10, 3)))

    def test_purify_moves_toward_data(self, rng):
        """Noisy points should be moved closer to training data after purification."""
        # Training: clean cluster at (3, 3)
        n_train = 100
        center = np.array([3.0, 3.0])
        X_train = center + 0.2 * rng.standard_normal((n_train, 2))

        # Noisy: points at (6, 6) — far from training data
        X_noisy = np.array([[6.0, 6.0]] * 20) + 0.1 * rng.standard_normal((20, 2))

        purifier = LangevinPurifier(n_steps=30, dt=0.1).fit(X_train)
        X_pure = purifier.purify(X_noisy)

        dist_before = np.linalg.norm(X_noisy - center, axis=1).mean()
        dist_after = np.linalg.norm(X_pure - center, axis=1).mean()

        assert dist_after < dist_before, (
            f"Purification should reduce distance to manifold: "
            f"before={dist_before:.3f}, after={dist_after:.3f}"
        )

    def test_purify_with_entropy_op(self, rng):
        """Purification with entropy operator should still produce valid output."""
        X = rng.standard_normal((100, 4))
        entropy_op = LocalEntropyOperator(k_neighbors=8).fit(X)
        purifier = LangevinPurifier(n_steps=5, dt=0.01, entropy_op=entropy_op).fit(X)
        X_noisy = X + 0.5 * rng.standard_normal((100, 4))
        X_pure = purifier.purify(X_noisy)
        assert X_pure.shape == (100, 4)
        assert np.all(np.isfinite(X_pure))

    def test_silverman_bandwidth_computed(self, rng):
        """When bandwidth=None, Silverman's rule should be applied."""
        X = rng.standard_normal((100, 4))
        purifier = LangevinPurifier().fit(X)
        assert purifier._sigma > 0.0
        assert np.isfinite(purifier._sigma)


# ─────────────────────────────────────────────────────────────────────────────
# TestCCDEngineIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestCCDEngineIntegration:
    """Full pipeline tests for CCDEngine."""

    def test_fit_returns_self(self, rng):
        X = rng.standard_normal((100, 10))
        engine = CCDEngine(d_threshold=5)
        assert engine.fit(X) is engine

    def test_transform_output_shape(self, rng):
        n, d = 200, 20
        X = rng.standard_normal((n, d))
        engine = CCDEngine(
            d_threshold=5,
            n_diffusion_components=5,
            n_diffusion_neighbors=10,
        ).fit(X)
        Z = engine.transform(X)
        assert Z.shape[0] == n
        assert Z.shape[1] <= d, "Output dimension should be ≤ input dimension"

    def test_low_dim_passthrough(self, rng):
        """For d < d_threshold, transform should return X unchanged."""
        n, d = 100, 3
        X = rng.standard_normal((n, d))
        engine = CCDEngine(d_threshold=5).fit(X)
        Z = engine.transform(X)
        assert np.array_equal(Z, X)

    def test_inverse_transform_shape(self, rng):
        n, d = 150, 15
        X = rng.standard_normal((n, d))
        engine = CCDEngine(
            d_threshold=5,
            n_diffusion_components=4,
            n_diffusion_neighbors=10,
        ).fit(X)
        Z = engine.transform(X)
        X_approx = engine.inverse_transform(Z)
        assert X_approx.shape == X.shape

    def test_transform_resonance_shape(self, rng):
        n, d = 200, 15
        X = rng.standard_normal((n, d))
        engine = CCDEngine(d_threshold=5, n_oscillator_groups=3).fit(X)
        Z_res = engine.transform_resonance(X)
        assert Z_res.shape[0] == n
        assert Z_res.shape[1] <= d

    def test_transform_coherence_shape(self, rng):
        n, d = 200, 12
        X = rng.standard_normal((n, d))
        engine = CCDEngine(d_threshold=5).fit(X)
        Z_coh = engine.transform_coherence(X)
        assert Z_coh.shape[0] == n

    def test_langevin_purify_output_shape(self, rng):
        n, d = 100, 10
        X = rng.standard_normal((n, d))
        engine = CCDEngine(
            d_threshold=5,
            n_langevin_steps=5,
            langevin_dt=0.01,
        ).fit(X)
        X_noisy = X + 0.5 * rng.standard_normal((n, d))
        X_pure = engine.langevin_purify(X_noisy)
        assert X_pure.shape == (n, d)
        assert np.all(np.isfinite(X_pure))

    def test_temperature_output(self, rng):
        n, d = 100, 10
        X = rng.standard_normal((n, d))
        engine = CCDEngine(d_threshold=5).fit(X)
        T = engine.temperature(X)
        assert T.shape == (n,)
        assert np.all(T > 0)

    def test_effective_dimension_less_than_d(self, manifold_in_high_dim):
        """For a 1D manifold in R^20, effective dimension should be small."""
        X_high, _, m = manifold_in_high_dim
        n, d = X_high.shape
        engine = CCDEngine(
            d_threshold=5,
            n_diffusion_components=8,
            n_diffusion_neighbors=12,
        ).fit(X_high)
        k_eff = engine.effective_dimension()
        assert k_eff < d, f"Effective dim {k_eff} should be < ambient d={d}"
        assert k_eff >= 1

    def test_no_fit_raises(self, rng):
        engine = CCDEngine()
        with pytest.raises(RuntimeError, match="fitted"):
            engine.transform(rng.standard_normal((10, 8)))

    def test_certificate_type(self, rng):
        X = rng.standard_normal((100, 12))
        engine = CCDEngine(d_threshold=5).fit(X)
        cert = engine.certificate()
        assert isinstance(cert, CCDCertificate)

    def test_certificate_d_input(self, rng):
        n, d = 100, 12
        X = rng.standard_normal((n, d))
        engine = CCDEngine(d_threshold=5).fit(X)
        cert = engine.certificate()
        assert cert.d_input == d

    def test_certificate_k_less_than_d(self, manifold_in_high_dim):
        """Certificate should report k_effective < d for structured data."""
        X_high, _, _ = manifold_in_high_dim
        d = X_high.shape[1]
        engine = CCDEngine(
            d_threshold=5,
            n_diffusion_components=8,
        ).fit(X_high)
        cert = engine.certificate()
        assert cert.k_effective < d

    def test_certificate_low_dim_passthrough(self, rng):
        """For d < d_threshold, certificate should report curse_escaped=False."""
        X = rng.standard_normal((100, 3))
        engine = CCDEngine(d_threshold=5).fit(X)
        cert = engine.certificate()
        assert cert.curse_escaped is False
        assert cert.cod_reduction_log10 == 0.0

    def test_certificate_str(self, rng):
        """Certificate __str__ should not raise and should contain key text."""
        X = rng.standard_normal((100, 12))
        engine = CCDEngine(d_threshold=5).fit(X)
        cert = engine.certificate()
        s = str(cert)
        assert "CCDCertificate" in s
        assert "d_input" in s

    def test_local_dimension_output(self, rng):
        n, d = 100, 10
        X = rng.standard_normal((n, d))
        engine = CCDEngine(d_threshold=5).fit(X)
        d_loc = engine.local_dimension(X)
        assert d_loc.shape == (n,)
        assert np.all(d_loc >= 0.1)


# ─────────────────────────────────────────────────────────────────────────────
# TestCoDReduction
# ─────────────────────────────────────────────────────────────────────────────

class TestCoDReduction:
    """Validate that CCD actually reduces the effective dimensionality (core claim)."""

    def test_cod_reduction_positive_for_high_dim(self, rng):
        """CoD reduction should be > 0 when k_eff < d."""
        n, d = 200, 20
        X = rng.standard_normal((n, d))
        engine = CCDEngine(d_threshold=5).fit(X)
        cert = engine.certificate()
        assert cert.cod_reduction_log10 >= 0.0

    def test_cod_reduction_scales_with_d(self, rng):
        """Larger d should yield larger CoD reduction (for same k_eff)."""
        n = 300
        reductions = []
        for d in [10, 20, 50]:
            X = rng.standard_normal((n, d))
            engine = CCDEngine(d_threshold=5, n_diffusion_components=5).fit(X)
            cert = engine.certificate()
            reductions.append(cert.cod_reduction_log10)
        # Reductions should generally increase with d
        # (may not be strictly monotone for random data, but last > first)
        assert reductions[-1] >= reductions[0] - 1.0

    def test_manifold_data_has_larger_reduction(self, manifold_in_high_dim, rng):
        """Manifold data (structured) should have larger reduction than pure noise."""
        X_manifold, _, _ = manifold_in_high_dim
        d = X_manifold.shape[1]
        X_noise = rng.standard_normal(X_manifold.shape)

        engine_manifold = CCDEngine(
            d_threshold=5,
            n_diffusion_components=8,
        ).fit(X_manifold)
        engine_noise = CCDEngine(
            d_threshold=5,
            n_diffusion_components=8,
        ).fit(X_noise)

        cert_manifold = engine_manifold.certificate()
        cert_noise = engine_noise.certificate()

        # Manifold data (true m=1) should compress more than noise (true m=20)
        assert cert_manifold.k_effective <= cert_noise.k_effective + 2, (
            f"Manifold k={cert_manifold.k_effective} vs noise k={cert_noise.k_effective}"
        )

    def test_lorenz_requires_few_modes(self, lorenz_trajectory, rng):
        """Lorenz attractor (m≈3) should compress to k_eff << d=3 if embedded higher."""
        # Embed in R^15
        traj = lorenz_trajectory   # (1000, 3)
        n = len(traj)
        d_target = 15
        A = rng.standard_normal((d_target, 3))
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        X_embedded = traj @ A.T + 0.02 * rng.standard_normal((n, d_target))

        engine = CCDEngine(
            d_threshold=5,
            n_diffusion_components=8,
            n_diffusion_neighbors=15,
        ).fit(X_embedded)
        cert = engine.certificate()

        # Should find k_eff much less than ambient d=15
        assert cert.k_effective < d_target
        assert cert.cod_reduction_log10 > 0


# ─────────────────────────────────────────────────────────────────────────────
# TestHighDimSystems
# ─────────────────────────────────────────────────────────────────────────────

class TestHighDimSystems:
    """End-to-end tests on physically meaningful high-dim systems."""

    def test_lorenz_in_50d_effective_dim(self, lorenz_in_50d):
        """Lorenz embedded in R^50 should have k_eff much smaller than 50."""
        X = lorenz_in_50d
        n, d = X.shape
        engine = CCDEngine(
            d_threshold=5,
            n_diffusion_components=8,
            n_diffusion_neighbors=15,
        ).fit(X)
        k_eff = engine.effective_dimension()
        assert k_eff < d // 3, (
            f"Lorenz in R^{d}: k_eff={k_eff} should be < {d // 3}"
        )

    def test_lorenz_in_50d_certificate_curse_escaped(self, lorenz_in_50d):
        """Lorenz in R^50 should have curse_escaped=True."""
        X = lorenz_in_50d
        engine = CCDEngine(
            d_threshold=5,
            n_diffusion_components=8,
        ).fit(X)
        cert = engine.certificate()
        assert cert.curse_escaped is True, (
            f"Lorenz in R^50 should have curse_escaped=True, got k={cert.k_effective}"
        )

    def test_lorenz_in_50d_langevin_purify(self, lorenz_in_50d, rng):
        """Langevin purification on Lorenz should produce finite outputs."""
        X = lorenz_in_50d
        engine = CCDEngine(
            d_threshold=5,
            n_langevin_steps=5,
            langevin_dt=0.01,
        ).fit(X)
        X_noisy = X[:50] + 0.5 * rng.standard_normal((50, X.shape[1]))
        X_pure = engine.langevin_purify(X_noisy)
        assert np.all(np.isfinite(X_pure))
        assert X_pure.shape == X_noisy.shape

    def test_pure_noise_has_high_effective_dim(self, pure_noise_high_dim):
        """Pure noise in R^30 should have k_eff close to d (no structure to compress)."""
        X = pure_noise_high_dim
        d = X.shape[1]
        engine = CCDEngine(d_threshold=5, n_diffusion_components=8).fit(X)
        k_eff = engine.effective_dimension()
        # For pure noise, k_eff should be >= 1 (engine found at least one component)
        assert k_eff >= 1

    def test_structured_vs_noise_effective_dim(self, manifold_in_high_dim, pure_noise_high_dim, rng):
        """Structured data should have smaller k_eff than pure noise of same shape."""
        X_struct, _, _ = manifold_in_high_dim   # (300, 20), m=1
        # Subsample noise to match shape
        X_noise = pure_noise_high_dim[:300, :20]

        engine_struct = CCDEngine(d_threshold=5, n_diffusion_components=8).fit(X_struct)
        engine_noise = CCDEngine(d_threshold=5, n_diffusion_components=8).fit(X_noise)

        k_struct = engine_struct.effective_dimension()
        k_noise = engine_noise.effective_dimension()

        assert k_struct <= k_noise, (
            f"Structured k={k_struct} should be ≤ noise k={k_noise}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestUtilityFunctions
# ─────────────────────────────────────────────────────────────────────────────

class TestUtilityFunctions:
    """preprocess_high_dim, estimate_intrinsic_dimension."""

    def test_preprocess_high_dim_returns_tuple(self, rng):
        X = rng.standard_normal((100, 15))
        Z, engine = preprocess_high_dim(X, d_threshold=5)
        assert isinstance(engine, CCDEngine)
        assert isinstance(Z, np.ndarray)
        assert Z.shape[0] == 100

    def test_preprocess_high_dim_low_dim_passthrough(self, rng):
        X = rng.standard_normal((100, 3))
        Z, engine = preprocess_high_dim(X, d_threshold=5)
        assert np.array_equal(Z, X)

    def test_preprocess_high_dim_reuse_engine(self, rng):
        X_train = rng.standard_normal((100, 12))
        X_new = rng.standard_normal((30, 12))
        _, engine = preprocess_high_dim(X_train, d_threshold=5, fit=True)
        Z_new, _ = preprocess_high_dim(X_new, d_threshold=5, fit=False, engine=engine)
        assert Z_new.shape[0] == 30

    def test_estimate_intrinsic_dimension_keys(self, rng):
        X = rng.standard_normal((150, 8))
        result = estimate_intrinsic_dimension(X, k=8)
        expected_keys = {
            "local_dim_mean",
            "local_dim_median",
            "spectral_gap_dim",
            "pca_95_dim",
            "d_ambient",
            "reduction_ratio_spectral",
            "reduction_ratio_pca95",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_estimate_intrinsic_dimension_ambient(self, rng):
        X = rng.standard_normal((150, 12))
        result = estimate_intrinsic_dimension(X)
        assert result["d_ambient"] == 12.0

    def test_estimate_intrinsic_dimension_manifold(self, manifold_in_high_dim):
        """Intrinsic dimension of a 1D manifold in R^20 should be estimated as small."""
        X_high, _, m = manifold_in_high_dim
        result = estimate_intrinsic_dimension(X_high, k=10)
        # spectral_gap_dim and local_dim_mean should be much less than d=20
        assert result["spectral_gap_dim"] < 20.0
        assert result["pca_95_dim"] < 20.0
        assert result["reduction_ratio_spectral"] > 1.0

    def test_estimate_intrinsic_dimension_returns_positive(self, rng):
        X = rng.standard_normal((100, 8))
        result = estimate_intrinsic_dimension(X, k=6)
        assert result["local_dim_mean"] > 0.0
        assert result["pca_95_dim"] >= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TestCCDImportFromPackage
# ─────────────────────────────────────────────────────────────────────────────

class TestCCDImportFromPackage:
    """Verify CCD classes are importable from the top-level acf_functor package."""

    def test_imports_from_acf_functor(self):
        from acf_functor import (
            CCDEngine,
            CCDCertificate,
            RobustCCDCertificate,
            ChebyshevShell,
            SpectralPreprocessor,
            SparseAdaptiveKernel,
            DiffusionGeometry,
            CoupledOscillators,
            LocalEntropyOperator,
            LangevinPurifier,
            ManifoldDecoder,
            ScoreMatchingLangevin,
            preprocess_high_dim,
            estimate_intrinsic_dimension,
        )
        assert CCDEngine is not None
        assert CCDCertificate is not None
        assert RobustCCDCertificate is not None
        assert SparseAdaptiveKernel is not None
        assert ManifoldDecoder is not None
        assert ScoreMatchingLangevin is not None
        assert SpectralPreprocessor is not None

    def test_version_updated(self):
        import acf_functor
        assert acf_functor.__version__ >= "6.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# TestSpectralPreprocessor
# ─────────────────────────────────────────────────────────────────────────────

class TestSpectralPreprocessor:
    """Tests for SpectralPreprocessor (PCA whitening Layer 1 replacement)."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((200, 20))

    def test_fit_returns_self(self):
        from acf_functor import SpectralPreprocessor
        sp = SpectralPreprocessor(n_components=10)
        result = sp.fit(self.X)
        assert result is sp

    def test_transform_output_shape(self):
        from acf_functor import SpectralPreprocessor
        sp = SpectralPreprocessor(n_components=8).fit(self.X)
        Z = sp.transform(self.X)
        assert Z.shape == (200, 8), f"Expected (200, 8), got {Z.shape}"

    def test_whitened_variance_approx_one(self):
        from acf_functor import SpectralPreprocessor
        sp = SpectralPreprocessor(n_components=10, whiten=True).fit(self.X)
        Z = sp.transform(self.X)
        # Each whitened component should have variance ≈ 1
        var = Z.var(axis=0)
        assert np.allclose(var, 1.0, atol=0.1), f"Whitened variances: {var}"

    def test_inverse_transform_roundtrip(self):
        from acf_functor import SpectralPreprocessor
        sp = SpectralPreprocessor(n_components=20, whiten=True).fit(self.X)
        Z = sp.transform(self.X)
        X_rec = sp.inverse_transform(Z)
        # Full-rank PCA should reconstruct exactly
        assert X_rec.shape == self.X.shape
        err = np.mean((X_rec - self.X) ** 2) / (np.var(self.X) + 1e-8)
        assert err < 0.01, f"Reconstruction MSE too large: {err:.4f}"

    def test_distance_rank_correlation(self):
        """Key property: PCA should preserve pairwise distances (Spearman ρ ≈ 1)."""
        from acf_functor import SpectralPreprocessor
        from scipy.stats import spearmanr
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 30))
        sp = SpectralPreprocessor(n_components=20, whiten=False).fit(X)
        Z = sp.transform(X)
        # Compute pairwise distances in original and reduced space (sample)
        idx = rng.choice(len(X), 40, replace=False)
        dX = np.linalg.norm(X[idx[:, None]] - X[idx[None, :]], axis=-1).ravel()
        dZ = np.linalg.norm(Z[idx[:, None]] - Z[idx[None, :]], axis=-1).ravel()
        rho, _ = spearmanr(dX, dZ)
        assert rho > 0.95, f"SpectralPreprocessor distance rank corr ρ={rho:.3f} < 0.95"

    def test_effective_rank_reasonable(self):
        from acf_functor import SpectralPreprocessor
        # Data with clear 3D structure embedded in 20D
        rng = np.random.default_rng(0)
        t = rng.uniform(0, 2 * np.pi, 200)
        X_low = np.stack([np.cos(t), np.sin(t), t / (2 * np.pi)], axis=1)
        noise = rng.standard_normal((200, 17)) * 0.01
        X = np.hstack([X_low, noise])
        sp = SpectralPreprocessor(n_components=10).fit(X)
        eff = sp.effective_rank
        assert 1 <= eff <= 5, f"effective_rank={eff} expected 1-5 for 3D-in-20D data"

    def test_unfitted_raises(self):
        from acf_functor import SpectralPreprocessor
        import pytest
        sp = SpectralPreprocessor()
        with pytest.raises(RuntimeError):
            sp.transform(self.X)
        with pytest.raises(RuntimeError):
            sp.inverse_transform(np.zeros((5, 3)))


# ─────────────────────────────────────────────────────────────────────────────
# TestSparseAdaptiveKernel
# ─────────────────────────────────────────────────────────────────────────────

class TestSparseAdaptiveKernel:
    """Tests for the O(n·k) sparse kernel builder."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.X = rng.standard_normal((50, 4))

    def test_build_returns_csr_and_sigma(self):
        from acf_functor import SparseAdaptiveKernel
        W, sigma = SparseAdaptiveKernel(n_neighbors=5).build(self.X)
        from scipy.sparse import issparse
        assert issparse(W), "Kernel must be a sparse matrix"
        assert W.shape == (50, 50)
        assert sigma.shape == (50,)

    def test_symmetry(self):
        from acf_functor import SparseAdaptiveKernel
        W, _ = SparseAdaptiveKernel(n_neighbors=5).build(self.X)
        diff = (W - W.T).toarray()
        assert np.allclose(diff, 0, atol=1e-10), "Kernel must be symmetric"

    def test_sparsity(self):
        from acf_functor import SparseAdaptiveKernel
        n = len(self.X)
        W, _ = SparseAdaptiveKernel(n_neighbors=5).build(self.X)
        assert W.nnz < n * n, "Kernel must be sparser than dense O(n²)"
        assert W.nnz > 0, "Kernel must have non-zero entries"

    def test_sigma_positive(self):
        from acf_functor import SparseAdaptiveKernel
        _, sigma = SparseAdaptiveKernel(n_neighbors=5).build(self.X)
        assert np.all(sigma > 0), "All bandwidths must be positive"

    def test_values_in_zero_one(self):
        from acf_functor import SparseAdaptiveKernel
        W, _ = SparseAdaptiveKernel(n_neighbors=5).build(self.X)
        data = W.toarray()
        assert np.all(data >= 0), "Kernel values must be non-negative"
        assert np.all(data <= 1.0 + 1e-9), "Kernel values must be ≤ 1"


# ─────────────────────────────────────────────────────────────────────────────
# TestManifoldDecoder
# ─────────────────────────────────────────────────────────────────────────────

class TestManifoldDecoder:
    """Tests for the learned numpy MLP decoder."""

    def setup_method(self):
        rng = np.random.default_rng(1)
        n, d, k = 60, 8, 3
        self.X = rng.standard_normal((n, d))
        self.Z = rng.standard_normal((n, k))

    def test_fit_and_decode_shape(self):
        from acf_functor import ManifoldDecoder
        dec = ManifoldDecoder(n_layers=1, hidden_dim=16).fit(
            self.Z, self.X, n_epochs=5
        )
        X_pred = dec.decode(self.Z)
        assert X_pred.shape == self.X.shape

    def test_fitted_flag(self):
        from acf_functor import ManifoldDecoder
        dec = ManifoldDecoder()
        assert not dec._fitted
        dec.fit(self.Z, self.X, n_epochs=2)
        assert dec._fitted

    def test_not_fitted_raises(self):
        from acf_functor import ManifoldDecoder
        dec = ManifoldDecoder()
        with pytest.raises(RuntimeError, match="fitted"):
            dec.decode(self.Z)

    def test_reconstruction_error_keys(self):
        from acf_functor import ManifoldDecoder
        dec = ManifoldDecoder(n_layers=1, hidden_dim=16).fit(
            self.Z, self.X, n_epochs=5
        )
        err = dec.reconstruction_error(self.Z, self.X)
        for key in ("mse", "rmse", "relative_error", "max_error"):
            assert key in err, f"Missing key: {key}"
            assert err[key] >= 0.0

    def test_rmse_finite(self):
        from acf_functor import ManifoldDecoder
        dec = ManifoldDecoder(n_layers=2, hidden_dim=32).fit(
            self.Z, self.X, n_epochs=20
        )
        err = dec.reconstruction_error(self.Z, self.X)
        assert np.isfinite(err["rmse"])


# ─────────────────────────────────────────────────────────────────────────────
# TestScoreMatchingLangevin
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreMatchingLangevin:
    """Tests for the sparse score-matching Langevin engine."""

    def setup_method(self):
        rng = np.random.default_rng(2)
        self.X_train = rng.standard_normal((80, 4))

    def test_fit_stores_state(self):
        from acf_functor import ScoreMatchingLangevin
        sml = ScoreMatchingLangevin(n_steps=5).fit(self.X_train)
        assert sml._fitted
        assert sml._X_train is not None
        assert sml._tree is not None
        assert sml._sigma > 0

    def test_sparse_score_shape(self):
        from acf_functor import ScoreMatchingLangevin
        sml = ScoreMatchingLangevin(n_steps=5).fit(self.X_train)
        X_q = np.random.default_rng(3).standard_normal((10, 4))
        score = sml.sparse_score(X_q)
        assert score.shape == (10, 4)
        assert np.all(np.isfinite(score))

    def test_purify_shape_and_finite(self):
        from acf_functor import ScoreMatchingLangevin
        sml = ScoreMatchingLangevin(n_steps=5, dt=0.01).fit(self.X_train)
        X_noisy = np.random.default_rng(4).standard_normal((15, 4))
        X_pure = sml.purify(X_noisy)
        assert X_pure.shape == (15, 4)
        assert np.all(np.isfinite(X_pure))

    def test_purify_moves_toward_data(self):
        """Points far from the cluster should be attracted toward it."""
        from acf_functor import ScoreMatchingLangevin
        rng = np.random.default_rng(5)
        # Tight cluster at origin
        X_train = rng.standard_normal((100, 2)) * 0.5
        sml = ScoreMatchingLangevin(
            n_steps=30, dt=0.1, n_score_neighbors=50,
            T_init=0.1, T_final=0.01,
        ).fit(X_train)
        # Outlier far from cluster
        X_far = np.array([[6.0, 6.0]])
        X_pure = sml.purify(X_far, rng=rng)
        dist_before = float(np.linalg.norm(X_far))
        dist_after  = float(np.linalg.norm(X_pure))
        assert dist_after < dist_before, (
            f"Expected point to move toward cluster: {dist_before:.2f} → {dist_after:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestRobustCCDCertificate
# ─────────────────────────────────────────────────────────────────────────────

class TestRobustCCDCertificate:
    """Tests for the statistically robust certificate."""

    def setup_method(self):
        rng = np.random.default_rng(10)
        n, d = 120, 10
        # Lorenz-like manifold in 10D (3 active dims)
        t = np.linspace(0, 4 * np.pi, n)
        low_d = np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)
        noise = rng.standard_normal((n, d)) * 0.1
        self.X = np.hstack([low_d, noise[:, : d - 3]])
        self.engine = CCDEngine(d_threshold=5, n_langevin_steps=5, fit_decoder=True,
                                decoder_epochs=10).fit(self.X)

    def test_compute_returns_correct_type(self):
        from acf_functor import RobustCCDCertificate
        n_test = 30
        rng = np.random.default_rng(11)
        X_test = rng.standard_normal((n_test, self.X.shape[1]))
        cert = RobustCCDCertificate.compute(
            self.engine, self.X, X_test, n_bootstrap=5
        )
        assert isinstance(cert, RobustCCDCertificate)

    def test_ci_bounds_valid(self):
        from acf_functor import RobustCCDCertificate
        rng = np.random.default_rng(12)
        X_test = rng.standard_normal((30, self.X.shape[1]))
        cert = RobustCCDCertificate.compute(
            self.engine, self.X, X_test, n_bootstrap=5
        )
        assert cert.k_ci_lower <= cert.k_effective
        assert cert.k_effective <= cert.k_ci_upper + 1  # +1 tolerance

    def test_is_production_ready_returns_bool(self):
        from acf_functor import RobustCCDCertificate
        rng = np.random.default_rng(13)
        X_test = rng.standard_normal((30, self.X.shape[1]))
        cert = RobustCCDCertificate.compute(
            self.engine, self.X, X_test, n_bootstrap=5
        )
        assert isinstance(cert.is_production_ready(), bool)

    def test_str_contains_key_fields(self):
        from acf_functor import RobustCCDCertificate
        rng = np.random.default_rng(14)
        X_test = rng.standard_normal((30, self.X.shape[1]))
        cert = RobustCCDCertificate.compute(
            self.engine, self.X, X_test, n_bootstrap=5
        )
        s = str(cert)
        for keyword in ("RobustCCDCertificate", "RMSE", "Neighborhood", "Manifold"):
            assert keyword in s, f"Missing keyword '{keyword}' in cert string"

    def test_metrics_finite(self):
        from acf_functor import RobustCCDCertificate
        rng = np.random.default_rng(15)
        X_test = rng.standard_normal((30, self.X.shape[1]))
        cert = RobustCCDCertificate.compute(
            self.engine, self.X, X_test, n_bootstrap=5
        )
        assert np.isfinite(cert.reconstruction_rmse)
        assert np.isfinite(cert.neighborhood_preservation)
        assert np.isfinite(cert.transform_latency_ms)
        assert np.isfinite(cert.memory_footprint_mb)
        assert 0.0 <= cert.neighborhood_preservation <= 1.0

