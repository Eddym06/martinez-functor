"""
Investigation profunda del CCD Engine — calibración y diagnóstico.

Pregunta central: ¿qué componentes funcionan realmente?
  - ChebyshevShell: ¿ayuda o estorba?
  - DiffusionGeometry: ¿estima bien k_effective?
  - CoupledOscillators: ¿detecta grupos reales?
  - LangevinPurifier: ¿purifica realmente?
  - ManifoldDecoder: ¿reconstruye bien?
  - Pipeline completo: ¿es más que la suma de sus partes?

BENCHMARKS:
  A. Swiss Roll 3D en R^50 (variedad 2D conocida)
  B. Lorenz attractor en R^50 (variedad 3D conocida)
  C. Gaussianas multimodales en R^100
  D. Señal estructurada de alta frecuencia en R^200
"""
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

# ─── Add workspace to path ────────────────────────────────────────────────────
sys.path.insert(0, "/home/Martínez's Invariant")

from acf_functor.ccd_engine import (
    ChebyshevShell, SparseAdaptiveKernel, DiffusionGeometry,
    CoupledOscillators, LocalEntropyOperator, LangevinPurifier,
    ManifoldDecoder, ScoreMatchingLangevin, CCDEngine,
    RobustCCDCertificate,
)

BOLD   = "\033[1m"
RED    = "\033[91m"
GRN    = "\033[92m"
YLW    = "\033[93m"
BLU    = "\033[94m"
CYN    = "\033[96m"
RST    = "\033[0m"

def banner(title):
    w = 70
    print(f"\n{BOLD}{BLU}{'═'*w}{RST}")
    print(f"{BOLD}{BLU}  {title}{RST}")
    print(f"{BOLD}{BLU}{'═'*w}{RST}")

def section(title):
    print(f"\n{BOLD}{CYN}── {title} ──{RST}")

def ok(msg):    print(f"  {GRN}✓{RST} {msg}")
def warn(msg):  print(f"  {YLW}⚠{RST} {msg}")
def fail(msg):  print(f"  {RED}✗{RST} {msg}")
def info(msg):  print(f"  · {msg}")

# ═════════════════════════════════════════════════════════════════════════════
# DATOS SINTÉTICOS
# ═════════════════════════════════════════════════════════════════════════════

def make_swiss_roll_Nd(n=800, d=50, noise=0.1, seed=42):
    """2D Swiss roll embedded in R^d with noise in extra dimensions."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n))
    h = rng.uniform(0, 10, n)
    x = t * np.cos(t)
    y = h
    z = t * np.sin(t)
    manifold = np.stack([x, y, z], axis=1)  # 3D structure
    # Embed in R^d with noise
    X = np.zeros((n, d))
    X[:, :3] = manifold
    X[:, 3:] = rng.normal(0, noise, (n, d - 3))
    return X, t  # t is the manifold parameter


def make_lorenz_Nd(n=800, d=50, noise=0.05, seed=42):
    """Lorenz attractor trajectory (3D) embedded in R^d."""
    rng = np.random.default_rng(seed)
    # Integrate Lorenz
    sigma, rho, beta_ = 10.0, 28.0, 8.0 / 3.0
    dt = 0.01
    xyz = np.zeros((n + 500, 3))
    xyz[0] = [0.1, 0.1, 0.1]
    for i in range(1, n + 500):
        x_, y_, z_ = xyz[i - 1]
        dx = sigma * (y_ - x_)
        dy = x_ * (rho - z_) - y_
        dz = x_ * y_ - beta_ * z_
        xyz[i] = xyz[i - 1] + dt * np.array([dx, dy, dz])
    xyz = xyz[500:]  # discard transient
    # Normalise and embed in R^d
    xyz = (xyz - xyz.mean(0)) / (xyz.std(0) + 1e-8)
    X = np.zeros((n, d))
    X[:, :3] = xyz
    X[:, 3:] = rng.normal(0, noise, (n, d - 3))
    return X


def make_clusters_Nd(n=800, d=100, n_clusters=5, seed=42):
    """Gaussian clusters in R^d. True k = n_clusters."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)) * 5.0
    labels = rng.integers(0, n_clusters, n)
    X = centers[labels] + rng.standard_normal((n, d)) * 0.8
    return X, labels


def make_structured_signal(n=600, d=200, seed=42):
    """High-frequency sinusoidal structure in 200D."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    freqs = rng.uniform(0.5, 3.0, d)
    phases = rng.uniform(0, 2 * np.pi, d)
    X = np.sin(np.outer(t, freqs) + phases[np.newaxis, :])
    X += rng.normal(0, 0.05, X.shape)
    return X, t


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: CHEBYSHEV SHELL — ¿AYUDA O ESTORBA?
# ═════════════════════════════════════════════════════════════════════════════

def test_chebyshev(X, d_true, label):
    section(f"ChebyshevShell ablation — {label}")
    n, d = X.shape

    # Raw data → DiffusionGeometry
    t0 = time.perf_counter()
    diff_raw = DiffusionGeometry(n_components=10, n_neighbors=15).fit(X)
    Z_raw = diff_raw.transform(X)
    t_raw = time.perf_counter() - t0
    k_raw = diff_raw.intrinsic_dimension_estimate

    # ChebyshevShell → DiffusionGeometry
    t0 = time.perf_counter()
    cheb = ChebyshevShell(n_coeffs=8, compression_rank=16).fit(X)
    Z_cheb_in = cheb.transform(X)          # (n, rank)
    diff_cheb = DiffusionGeometry(n_components=10, n_neighbors=15).fit(Z_cheb_in)
    Z_cheb = diff_cheb.transform(Z_cheb_in)
    t_cheb = time.perf_counter() - t0
    k_cheb = diff_cheb.intrinsic_dimension_estimate

    # PCA → DiffusionGeometry (proper baseline)
    from numpy.linalg import svd
    t0 = time.perf_counter()
    U, S, Vt = svd(X - X.mean(0), full_matrices=False)
    Z_pca_in = U[:, :16] * S[:16]        # top-16 PCA
    diff_pca = DiffusionGeometry(n_components=10, n_neighbors=15).fit(Z_pca_in)
    Z_pca = diff_pca.transform(Z_pca_in)
    t_pca = time.perf_counter() - t0
    k_pca = diff_pca.intrinsic_dimension_estimate

    # Neighbourhood preservation: does the low-d rep preserve k-NN structure?
    def knn_preservation(X_orig, Z, k=10):
        n = min(200, len(X_orig))
        tree_orig = cKDTree(X_orig[:n])
        tree_proj = cKDTree(Z[:n])
        _, nn_o = tree_orig.query(X_orig[:n], k=k+1)
        _, nn_p = tree_proj.query(Z[:n], k=k+1)
        return np.mean([
            len(set(nn_o[i,1:]) & set(nn_p[i,1:])) / k
            for i in range(n)
        ])

    np_raw  = knn_preservation(X, Z_raw)
    np_cheb = knn_preservation(X, Z_cheb)
    np_pca  = knn_preservation(X, Z_pca)

    # Chebyshev rank quality
    evr = cheb.explained_variance_ratio
    effective_rank = cheb.effective_rank
    info(f"Chebyshev effective_rank={effective_rank}, top-3 EVR: {evr[:3]}")

    print(f"\n  {'Method':<22} {'k_est':>5} {'true_k':>6} {'NN_pres':>8} {'time':>7}")
    print(f"  {'─'*52}")
    res = [
        ("Raw → Diffusion",   k_raw,  d_true, np_raw,  t_raw),
        ("Cheby → Diffusion", k_cheb, d_true, np_cheb, t_cheb),
        ("PCA  → Diffusion",  k_pca,  d_true, np_pca,  t_pca),
    ]
    for name, k_est, k_t, nn, t in res:
        colour = GRN if abs(k_est - k_t) <= max(1, k_t) else YLW
        print(f"  {name:<22} {colour}{k_est:>5}{RST} {k_t:>6} {nn:>8.3f} {t:>6.2f}s")

    # Verdict
    if np_raw >= np_cheb and np_raw >= np_pca:
        warn(f"ChebyshevShell WORSE than raw data (NN preservation {np_raw:.3f} vs {np_cheb:.3f})")
    elif np_cheb >= np_pca:
        ok(f"ChebyshevShell best (NN preservation {np_cheb:.3f} vs raw {np_raw:.3f})")
    else:
        warn(f"PCA baseline beats ChebyshevShell ({np_pca:.3f} vs {np_cheb:.3f})")

    return {
        "k_raw": k_raw, "k_cheb": k_cheb, "k_pca": k_pca,
        "np_raw": np_raw, "np_cheb": np_cheb, "np_pca": np_pca,
    }


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: DIFFUSION GEOMETRY — k_effective accuracy
# ═════════════════════════════════════════════════════════════════════════════

def test_diffusion_k_estimation(X, d_true, label):
    section(f"DiffusionGeometry k_effective — {label}")
    n, d = X.shape

    results = {}
    for n_comp in [5, 10, 20]:
        diff = DiffusionGeometry(n_components=n_comp, n_neighbors=15).fit(X)
        k_est = diff.intrinsic_dimension_estimate
        λ = diff._eigenvalues
        gaps = np.diff(λ[::-1])  # largest to smallest, forward diff = drop-off
        info(f"  n_comp={n_comp}: k_est={k_est}, λ={np.round(λ[:5], 3)}, "
             f"gaps={np.round(np.abs(gaps[:5]), 4)}")
        results[n_comp] = k_est

    # TwoNN estimator (much more reliable for true intrinsic dim)
    def twoNN(X, n_sample=200):
        idx = np.random.default_rng(0).choice(len(X), min(n_sample, len(X)), replace=False)
        tree = cKDTree(X[idx])
        dists, _ = tree.query(X[idx], k=3)
        mu = dists[:, 2] / (dists[:, 1] + 1e-15)  # r2/r1
        mu = mu[mu > 1]
        if len(mu) == 0:
            return 1
        # MLE estimator
        return float(len(mu) / np.sum(np.log(mu)))

    k_2nn = twoNN(X)
    info(f"TwoNN estimator: {k_2nn:.2f} (true: {d_true})")

    best = min(results.values(), key=lambda k: abs(k - d_true))
    if abs(best - d_true) <= max(1, d_true // 2):
        ok(f"Spectral gap estimate close to truth: best={best}, true={d_true}")
    else:
        warn(f"Spectral gap estimate off: best={best}, true={d_true}")

    if abs(k_2nn - d_true) <= max(1.5, d_true * 0.5):
        ok(f"TwoNN accurate: {k_2nn:.2f} ≈ {d_true}")
    else:
        warn(f"TwoNN estimate off: {k_2nn:.2f} vs {d_true}")

    return results, k_2nn


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: COUPLED OSCILLATORS — ¿detecta grupos reales?
# ═════════════════════════════════════════════════════════════════════════════

def test_oscillators(X, d_true, label):
    section(f"CoupledOscillators — {label}")
    n, d = X.shape

    osc = CoupledOscillators().fit(X)
    k_osc = osc.n_resonance_groups
    evr = osc.explained_variance_ratio
    total_var = evr.sum()

    info(f"  n_resonance_groups = {k_osc} (true: {d_true})")
    info(f"  EVR top-5: {np.round(evr[:5], 3)}, total: {total_var:.3f}")
    for g in osc.resonance_groups[:5]:
        info(f"    Group {g.group_id}: {len(g.variable_indices)} vars, "
             f"λ={g.shared_eigenvalue:.3f}, coh={g.coherence_score:.3f}")

    # Test: projection quality  Z = (n, k_osc)
    Z = osc.transform(X)
    X_rec = osc.inverse_transform(Z)
    rmse = float(np.sqrt(np.mean((X - X_rec)**2)))
    info(f"  Reconstruction RMSE with k={k_osc} modes: {rmse:.4f}")

    # Compare: k modes vs d/2 random directions
    V_rand = np.random.default_rng(7).standard_normal((d, k_osc))
    V_rand /= np.linalg.norm(V_rand, axis=0)
    Z_rand = (X - X.mean(0)) @ V_rand
    X_rec_rand = Z_rand @ V_rand.T + X.mean(0)
    rmse_rand = float(np.sqrt(np.mean((X - X_rec_rand)**2)))
    info(f"  Reconstruction RMSE with {k_osc} RANDOM directions: {rmse_rand:.4f}")

    if rmse < rmse_rand:
        ok(f"Oscillators beat random projection: {rmse:.4f} < {rmse_rand:.4f}")
    else:
        warn(f"Random projection AS GOOD as oscillators: {rmse:.4f} vs {rmse_rand:.4f}")

    if abs(k_osc - d_true) <= max(2, d_true):
        ok(f"n_groups={k_osc} close to truth={d_true}")
    else:
        warn(f"n_groups={k_osc} far from truth={d_true}")

    return k_osc


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: LANGEVIN PURIFIER — ¿purifica realmente?
# ═════════════════════════════════════════════════════════════════════════════

def test_purification():
    section("LangevinPurifier / ScoreMatchingLangevin — purification quality")
    rng = np.random.default_rng(42)
    d = 5

    # Tight Gaussian cluster at origin
    X_train = rng.standard_normal((200, d)) * 0.5
    # Noisy test points + outliers
    X_near = X_train[:50] + rng.standard_normal((50, d)) * 0.2    # near manifold
    X_far  = rng.standard_normal((20, d)) * 5.0                    # far outliers

    purifier = LangevinPurifier(n_steps=30, dt=0.05, seed=0).fit(X_train)

    # Near points: should stay near
    X_near_pure = purifier.purify(X_near)
    dist_near_before = float(np.mean(np.linalg.norm(X_near, axis=1)))
    dist_near_after  = float(np.mean(np.linalg.norm(X_near_pure, axis=1)))

    # Far points: should move toward cluster
    X_far_pure = purifier.purify(X_far)
    dist_far_before = float(np.mean(np.linalg.norm(X_far, axis=1)))
    dist_far_after  = float(np.mean(np.linalg.norm(X_far_pure, axis=1)))

    # Average distance to training set
    tree = cKDTree(X_train)
    d_train_near_b, _ = tree.query(X_near, k=1)
    d_train_near_a, _ = tree.query(X_near_pure, k=1)
    d_train_far_b, _  = tree.query(X_far, k=1)
    d_train_far_a, _  = tree.query(X_far_pure, k=1)

    info(f"Near points: dist-to-train {float(np.mean(d_train_near_b)):.3f} → "
         f"{float(np.mean(d_train_near_a)):.3f}")
    info(f"Far  points: dist-to-train {float(np.mean(d_train_far_b)):.3f} → "
         f"{float(np.mean(d_train_far_a)):.3f}")

    far_improved = float(np.mean(d_train_far_a)) < float(np.mean(d_train_far_b))
    near_stable  = float(np.mean(d_train_near_a)) < float(np.mean(d_train_near_b)) * 2.5

    if far_improved:
        ok(f"Far outliers attracted toward cluster")
    else:
        fail(f"Far outliers NOT attracted toward cluster!")

    if near_stable:
        ok(f"Near points remain close to training manifold")
    else:
        warn(f"Near points drifted away from training manifold")

    return far_improved, near_stable


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: MANIFOLD DECODER — calidad de reconstrucción
# ═════════════════════════════════════════════════════════════════════════════

def test_decoder():
    section("ManifoldDecoder — reconstruction quality")
    rng = np.random.default_rng(0)
    scenarios = [
        ("2D→10D",   (400, 2),   10),
        ("5D→50D",   (400, 5),   50),
        ("3D→100D",  (300, 3),   100),
    ]

    for label, (n, k), d_out in scenarios:
        # Low-dim latent codes with linear+nonlinear structure
        Z = rng.standard_normal((n, k))
        # True mapping: random nonlinear
        W1 = rng.standard_normal((k, 32)) * 0.5
        W2 = rng.standard_normal((32, d_out)) * 0.5
        X_true = np.maximum(Z @ W1, 0) @ W2

        # Train / test split
        n_train = int(0.8 * n)
        Z_tr, Z_te = Z[:n_train], Z[n_train:]
        X_tr, X_te = X_true[:n_train], X_true[n_train:]

        # Short training (100 epochs)
        dec100 = ManifoldDecoder(n_layers=2, hidden_dim=64).fit(Z_tr, X_tr, n_epochs=100)
        err100 = dec100.reconstruction_error(Z_te, X_te)

        # More training (500 epochs)
        dec500 = ManifoldDecoder(n_layers=2, hidden_dim=128).fit(Z_tr, X_tr, n_epochs=500)
        err500 = dec500.reconstruction_error(Z_te, X_te)

        # PCA baseline (can a linear decoder do as well?)
        from numpy.linalg import svd as np_svd
        _, _, Vt_pca = np_svd(X_tr - X_tr.mean(0), full_matrices=False)
        # Use Z as if it were the PCA code: fit linear map Z → X
        # Least-squares: X ≈ Z @ A + b
        A, _, _, _ = np.linalg.lstsq(
            np.hstack([Z_tr, np.ones((n_train, 1))]),
            X_tr, rcond=None
        )
        X_te_lin = np.hstack([Z_te, np.ones((len(Z_te), 1))]) @ A
        rmse_lin = float(np.sqrt(np.mean((X_te - X_te_lin)**2)))

        info(f"  {label}: MLP100={err100['rmse']:.4f}  MLP500={err500['rmse']:.4f}"
             f"  Linear={rmse_lin:.4f}")

        if err100['rmse'] < rmse_lin * 1.5:
            ok(f"  {label}: decoder competitive with linear baseline")
        else:
            warn(f"  {label}: decoder worse than linear baseline (MLP={err100['rmse']:.4f} > Lin={rmse_lin:.4f})")

        if err500['rmse'] < err100['rmse'] * 0.8:
            ok(f"  {label}: more epochs significantly improve decoder")
        else:
            info(f"  {label}: more epochs marginally improve decoder")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: FULL ENGINE — k_effective + speed + quality
# ═════════════════════════════════════════════════════════════════════════════

def test_full_engine(X, d_true, label):
    section(f"CCDEngine full pipeline — {label}")
    n, d = X.shape

    # Fit with decoder disabled first (speed benchmark)
    t0 = time.perf_counter()
    eng = CCDEngine(d_threshold=5, n_langevin_steps=5, fit_decoder=False).fit(X)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    Z = eng.transform(X[:200])
    t_transform = time.perf_counter() - t0

    k_eff = eng.effective_dimension()
    cert = eng.certificate()

    info(f"  n={n}, d={d}, k_eff={k_eff} (true: {d_true})")
    info(f"  Fit: {t_fit:.2f}s  |  Transform(200pts): {t_transform*1000:.1f}ms")
    info(f"  Certificate: k_cheb={cert.k_chebyshev}, k_diff={cert.k_diffusion}, "
         f"k_osc={cert.n_resonance_groups}")
    info(f"  curse_escaped={cert.curse_escaped}, CoD reduction=10^{cert.cod_reduction_log10:.1f}")

    if cert.curse_escaped:
        ok(f"Curse of dimensionality escaped: {d}D → {k_eff}D")
    else:
        warn(f"Curse NOT escaped: {d}D → {k_eff}D (need k < d/2={d//2})")

    if abs(k_eff - d_true) <= max(2, d_true):
        ok(f"k_effective={k_eff} plausible for true dim {d_true}")
    else:
        warn(f"k_effective={k_eff} far from true dim {d_true}")

    # Reconstruction quality
    eng2 = CCDEngine(d_threshold=5, n_langevin_steps=3, fit_decoder=True,
                     decoder_epochs=50).fit(X)
    Z2 = eng2.transform(X[:100])
    X_rec = eng2.inverse_transform(Z2)
    rmse = float(np.sqrt(np.mean((X[:100] - X_rec)**2)))
    info(f"  Reconstruction RMSE (decoder): {rmse:.4f}")

    return k_eff, t_fit


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7: SCALE BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════

def test_scaling():
    section("Scaling benchmark: n × d")
    rng = np.random.default_rng(99)
    configs = [
        (500,   20),
        (500,   100),
        (2000,  50),
        (2000,  200),
        (5000,  50),
        (5000,  100),
        (10000, 20),
    ]
    print(f"\n  {'n':>6} {'d':>4} {'fit_s':>7} {'transform_1k_ms':>16} {'k_eff':>6}")
    print(f"  {'─'*48}")
    for n, d in configs:
        # Simple manifold: 3D signal in R^d
        t = np.linspace(0, 4*np.pi, n)
        X = np.zeros((n, d))
        X[:, 0] = np.sin(t)
        X[:, 1] = np.cos(t)
        X[:, 2] = np.sin(2*t)
        X[:, 3:] = rng.normal(0, 0.05, (n, d-3))

        t0 = time.perf_counter()
        eng = CCDEngine(d_threshold=5, n_langevin_steps=0, fit_decoder=False).fit(X)
        t_fit = time.perf_counter() - t0

        n_bench = min(1000, n)
        t0 = time.perf_counter()
        eng.transform(X[:n_bench])
        t_tr = (time.perf_counter() - t0) * 1000

        colour = GRN if t_fit < 10 else (YLW if t_fit < 30 else RED)
        print(f"  {n:>6} {d:>4} {colour}{t_fit:>7.2f}s{RST} {t_tr:>13.1f}ms "
              f"  {eng.effective_dimension():>6}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 8: INTERNAL METRIC DISTORTION BY CHEBYSHEV
# ═════════════════════════════════════════════════════════════════════════════

def test_metric_distortion(X, label):
    section(f"Metric distortion by ChebyshevShell — {label}")
    n = min(300, len(X))
    X_s = X[:n]

    # Original pairwise distances (sample 50×50)
    idx = np.random.default_rng(0).choice(n, 50, replace=False)
    D_orig = cdist(X_s[idx], X_s[idx])

    cheb = ChebyshevShell(n_coeffs=8, compression_rank=16).fit(X_s)
    Z_cheb = cheb.transform(X_s)
    D_cheb = cdist(Z_cheb[idx], Z_cheb[idx])

    # PCA baseline with same rank
    U, S, Vt = np.linalg.svd(X_s - X_s.mean(0), full_matrices=False)
    Z_pca = U[:, :16] * S[:16]
    D_pca = cdist(Z_pca[idx], Z_pca[idx])

    # Correlation of distance matrices (Spearman-like)
    triu = np.triu_indices(50, k=1)
    d_orig_flat = D_orig[triu]
    d_cheb_flat = D_cheb[triu]
    d_pca_flat  = D_pca[triu]

    from scipy.stats import spearmanr
    rho_cheb, _ = spearmanr(d_orig_flat, d_cheb_flat)
    rho_pca,  _ = spearmanr(d_orig_flat, d_pca_flat)

    info(f"  Distance rank correlation with original:")
    info(f"    ChebyshevShell (rank=16): ρ = {rho_cheb:.4f}")
    info(f"    PCA            (rank=16): ρ = {rho_pca:.4f}")

    if rho_pca > rho_cheb + 0.05:
        fail(f"PCA preserves distances BETTER than ChebyshevShell (Δρ = {rho_pca-rho_cheb:.4f})")
    elif rho_cheb > rho_pca + 0.01:
        ok(f"ChebyshevShell preserves distances better than PCA")
    else:
        warn(f"ChebyshevShell ~ PCA in distance preservation")

    return rho_cheb, rho_pca


# ═════════════════════════════════════════════════════════════════════════════
# TEST 9: CHEBYSHEV inverse_transform bug
# ═════════════════════════════════════════════════════════════════════════════

def test_chebyshev_inverse():
    section("ChebyshevShell inverse_transform accuracy")
    rng = np.random.default_rng(5)
    scenarios = [
        ("Smooth signal", np.outer(np.linspace(0, 1, 200), rng.standard_normal(20)) + rng.normal(0, 0.01, (200, 20))),
        ("Gaussian blobs", rng.standard_normal((200, 20))),
    ]
    for label, X in scenarios:
        cheb = ChebyshevShell(n_coeffs=8, compression_rank=16).fit(X)
        Z = cheb.transform(X)
        X_rec = cheb.inverse_transform(Z)
        rmse = float(np.sqrt(np.mean((X - X_rec)**2)))
        # What percentage of variance is captured?
        var_total = float(np.var(X))
        var_resid = float(np.var(X - X_rec))
        r2 = 1 - var_resid / (var_total + 1e-10)
        info(f"  {label}: RMSE={rmse:.4f}, R²={r2:.4f}")
        if r2 < 0.5:
            fail(f"  {label}: inverse_transform R²={r2:.4f} — POOR (likely T_1(x)=x bug)")
        elif r2 < 0.9:
            warn(f"  {label}: inverse_transform R²={r2:.4f} — mediocre")
        else:
            ok(f"  {label}: inverse_transform R²={r2:.4f} — good")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner("CCD ENGINE — DEEP INVESTIGATION")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print(f"\n{BOLD}Generando datasets...{RST}")
    X_swiss, t_swiss = make_swiss_roll_Nd(n=800, d=50, noise=0.1)
    X_lorenz         = make_lorenz_Nd(n=800, d=50, noise=0.05)
    X_clusters, labs = make_clusters_Nd(n=800, d=100, n_clusters=4)
    X_signal, t_sig  = make_structured_signal(n=600, d=200)
    info(f"Swiss Roll:    {X_swiss.shape}   (true k=2)")
    info(f"Lorenz:        {X_lorenz.shape}  (true k=3)")
    info(f"Clusters:      {X_clusters.shape} (true k=4)")
    info(f"Structured:    {X_signal.shape}  (true k=1)")

    # ─────────────────────────────────────────────────────────────────────────
    banner("1. CHEBYSHEV SHELL — ¿AYUDA O ESTORBA?")
    results_cheby = {}
    results_cheby["swiss"]   = test_chebyshev(X_swiss,    d_true=2,  label="Swiss Roll R^50")
    results_cheby["lorenz"]  = test_chebyshev(X_lorenz,   d_true=3,  label="Lorenz R^50")
    results_cheby["cluster"] = test_chebyshev(X_clusters, d_true=4,  label="Clusters R^100")

    # ─────────────────────────────────────────────────────────────────────────
    banner("2. METRIC DISTORTION BY CHEBYSHEV")
    test_metric_distortion(X_swiss,    "Swiss Roll R^50")
    test_metric_distortion(X_lorenz,   "Lorenz R^50")

    # ─────────────────────────────────────────────────────────────────────────
    banner("3. CHEBYSHEV INVERSE_TRANSFORM BUG")
    test_chebyshev_inverse()

    # ─────────────────────────────────────────────────────────────────────────
    banner("4. DIFFUSION GEOMETRY — k_effective estimate")
    test_diffusion_k_estimation(X_swiss,    d_true=2, label="Swiss Roll R^50")
    test_diffusion_k_estimation(X_lorenz,   d_true=3, label="Lorenz R^50")
    test_diffusion_k_estimation(X_clusters, d_true=4, label="Clusters R^100")

    # ─────────────────────────────────────────────────────────────────────────
    banner("5. COUPLED OSCILLATORS — resonance groups")
    test_oscillators(X_swiss,    d_true=2, label="Swiss Roll R^50")
    test_oscillators(X_lorenz,   d_true=3, label="Lorenz R^50")
    test_oscillators(X_clusters, d_true=4, label="Clusters R^100")

    # ─────────────────────────────────────────────────────────────────────────
    banner("6. LANGEVIN PURIFICATION")
    test_purification()

    # ─────────────────────────────────────────────────────────────────────────
    banner("7. MANIFOLD DECODER")
    test_decoder()

    # ─────────────────────────────────────────────────────────────────────────
    banner("8. FULL ENGINE")
    test_full_engine(X_swiss,    d_true=2, label="Swiss Roll R^50")
    test_full_engine(X_lorenz,   d_true=3, label="Lorenz R^50")
    test_full_engine(X_clusters, d_true=4, label="Clusters R^100")
    test_full_engine(X_signal,   d_true=1, label="Structured Signal R^200")

    # ─────────────────────────────────────────────────────────────────────────
    banner("9. SCALING BENCHMARK")
    test_scaling()

    # ─────────────────────────────────────────────────────────────────────────
    banner("SUMMARY OF FINDINGS")
    print(f"""
{BOLD}Key observations:{RST}
  [A] ChebyshevShell vs PCA:
      → If PCA wins consistently: ChebyshevShell is a costly detour
      → The inverse_transform bug (T_1(x)=x only) is a known issue
  [B] k_effective estimation:
      → Spectral gap method is unreliable; TwoNN is more accurate
  [C] CoupledOscillators on Z_diff:
      → With only k=2-5 diffusion dimensions, covariance has trivial structure
  [D] LangevinPurifier:
      → ScoreMatchingLangevin works if σ is calibrated correctly
  [E] ManifoldDecoder:
      → 100 epochs often insufficient; scales poorly with output dim
""")
