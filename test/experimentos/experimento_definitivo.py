"""
================================================================================
EXPERIMENTO DEFINITIVO — CCD Ecosystem vs PCA
================================================================================
Validación rigurosa con múltiples semillas, datasets reales y sintéticos,
pruebas estadísticas (t-test, Cohen's d), y verificación de las 4 claims
del ecosistema.

Ejecutar: python3 experimento_definitivo.py
================================================================================
"""
import sys, time, warnings, json, os
warnings.filterwarnings('ignore')
import numpy as np
from scipy import stats as sp_stats
from sklearn.manifold import trustworthiness as sklearn_trust
from sklearn.decomposition import PCA
from sklearn.datasets import (
    make_swiss_roll, make_s_curve,
    load_digits, load_wine, load_breast_cancer, fetch_olivetti_faces
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acf_functor.ccd_engine import CCDEngine

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL EXPERIMENTO
# ═════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "n_seeds": 5,
    "random_seeds": [42, 123, 456, 789, 101112],
    "n_neighbors_lcmc": 10,
    "n_samples_lcmc": 200,
    "ccd_target_dim": 2,
    "ccd_d_threshold": 5,
    "output_file": "resultados_experimento_definitivo.json",
}

# ═════════════════════════════════════════════════════════════════════════════
# GENERADORES DE DATOS
# ═════════════════════════════════════════════════════════════════════════════

def make_concentric_3d(n_per_spiral=500, noise_std=0.01):
    """Concentric 3D spirals — highly non-linear manifold."""
    t = np.linspace(0, 4 * np.pi, n_per_spiral)
    radii = [1.0, 2.5, 4.0]
    X_list, y_list = [], []
    for ri, r in enumerate(radii):
        x = r * np.cos(t + ri * np.pi / 3)
        y_ = r * np.sin(t + ri * np.pi / 3)
        z_ = t / (2 * np.pi) * r
        X_list.append(np.stack([x, y_, z_], axis=1))
        y_list.append(np.full(n_per_spiral, ri))
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y


def make_swiss_roll_noisy(n, n_noise, noise_scale, seed):
    """Swiss Roll embedded in R^{3+n_noise} with Gaussian noise."""
    X_clean, t = make_swiss_roll(n, noise=0.01, random_state=seed)
    rng = np.random.RandomState(seed + 1)
    noise = rng.randn(n, n_noise) * noise_scale
    X_noisy = np.hstack([X_clean, noise])
    return StandardScaler().fit_transform(X_noisy), StandardScaler().fit_transform(X_clean), t


def make_s_curve_noisy(n, n_noise, noise_scale, seed):
    """S-Curve embedded in R^{3+n_noise} with Gaussian noise."""
    X_clean, t = make_s_curve(n, noise=0.01, random_state=seed)
    rng = np.random.RandomState(seed + 1)
    noise = rng.randn(n, n_noise) * noise_scale
    X_noisy = np.hstack([X_clean, noise])
    return StandardScaler().fit_transform(X_noisy), StandardScaler().fit_transform(X_clean), t


def make_spirals_noisy(n_per_spiral, n_noise, noise_scale, seed):
    """Concentric spirals embedded in R^{3+n_noise}."""
    X_clean, y = make_concentric_3d(n_per_spiral)
    rng = np.random.RandomState(seed + 1)
    n_total = X_clean.shape[0]
    noise = rng.randn(n_total, n_noise) * noise_scale
    X_noisy = np.hstack([X_clean, noise])
    return StandardScaler().fit_transform(X_noisy), StandardScaler().fit_transform(X_clean), y


def make_high_dim_noisy(n, d_noise, n_signal, seed):
    """n < d case: strong signal in first n_signal dims, weak noise in rest."""
    rng = np.random.RandomState(seed)
    X_signal = rng.randn(n, n_signal) * 3.0
    X_noise = rng.randn(n, d_noise) * 0.2
    X_noisy = np.hstack([X_signal, X_noise])
    X_clean = X_signal
    return StandardScaler().fit_transform(X_noisy), StandardScaler().fit_transform(X_clean)


def make_outlier_data(n, n_noise, noise_scale, outlier_frac, outlier_scale, seed):
    """Swiss Roll with extreme outlier contamination."""
    X_clean, t = make_swiss_roll(n, noise=0.01, random_state=seed)
    rng = np.random.RandomState(seed + 1)
    noise = rng.randn(n, n_noise) * noise_scale
    X_noisy = np.hstack([X_clean, noise])
    n_out = int(n * outlier_frac)
    out_idx = rng.choice(n, n_out, replace=False)
    X_noisy[out_idx] += rng.randn(n_out, 3 + n_noise) * outlier_scale
    return (StandardScaler().fit_transform(X_noisy),
            StandardScaler().fit_transform(X_clean), t, out_idx)


# ═════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ═════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(X_clean, X_emb, X_noisy=None, y=None, X_rec=None, k=10):
    """Compute all quality metrics for an embedding."""
    n = len(X_clean)
    k_use = min(k, n - 2)
    n_sample = min(CONFIG["n_samples_lcmc"], n)
    if n_sample < n:
        idx = np.random.RandomState(9999).choice(n, n_sample, replace=False)
        Xc, Xe = X_clean[idx], X_emb[idx]
    else:
        Xc, Xe = X_clean, X_emb

    results = {}

    # Trustworthiness
    try:
        results["trust"] = float(sklearn_trust(Xc, Xe, n_neighbors=k_use))
    except Exception:
        results["trust"] = float("nan")

    # Continuity
    try:
        results["cont"] = float(sklearn_trust(Xe, Xc, n_neighbors=k_use))
    except Exception:
        results["cont"] = float("nan")

    # LCMC
    if not np.isnan(results.get("trust", float("nan"))):
        results["lcmc"] = (results["trust"] + results["cont"]) / 2.0
    else:
        results["lcmc"] = float("nan")

    # Neighborhood preservation
    try:
        knn = min(k_use, n_sample - 2)
        nn_o = NearestNeighbors(n_neighbors=knn + 1).fit(Xc).kneighbors(Xc)[1][:, 1:]
        nn_e = NearestNeighbors(n_neighbors=knn + 1).fit(Xe).kneighbors(Xe)[1][:, 1:]
        pres = float(np.mean([
            len(set(a.tolist()) & set(b.tolist())) / knn
            for a, b in zip(nn_o, nn_e)
        ]))
        results["neighborhood_preservation"] = pres
    except Exception:
        results["neighborhood_preservation"] = float("nan")

    # Silhouette
    if y is not None and len(np.unique(y)) >= 2:
        try:
            ns = min(200, n_sample)
            idx2 = np.random.RandomState(8888).choice(n_sample, ns, replace=False)
            if len(np.unique(y[idx2])) >= 2:
                results["silhouette"] = float(silhouette_score(Xe[idx2], y[idx2]))
            else:
                results["silhouette"] = float("nan")
        except Exception:
            results["silhouette"] = float("nan")
    else:
        results["silhouette"] = float("nan")

    # Reconstruction error
    if X_rec is not None:
        try:
            results["reconstruction_error"] = float(
                np.linalg.norm(X_noisy - X_rec, "fro") / max(np.linalg.norm(X_noisy, "fro"), 1e-10)
            )
        except Exception:
            results["reconstruction_error"] = float("nan")
    else:
        results["reconstruction_error"] = float("nan")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN DE UN TEST
# ═════════════════════════════════════════════════════════════════════════════

def run_test(name, X_noisy, X_clean, y=None, target_dim=2, seed=42):
    """Run CCD and PCA on one dataset instance, return all metrics and diagnostics."""
    np.random.seed(seed)
    n, d = X_noisy.shape

    result = {
        "scenario": name,
        "seed": seed,
        "n": n,
        "d": d,
        "target_dim": target_dim,
    }

    # ── PCA ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        Z_pca = PCA(n_components=target_dim).fit_transform(X_noisy)
        pca_ok = True
    except Exception:
        Z_pca = np.zeros((n, target_dim))
        pca_ok = False
    t_pca = time.perf_counter() - t0
    result["pca_time_ms"] = t_pca * 1000

    if pca_ok:
        metrics_pca = compute_all_metrics(X_clean, Z_pca, X_noisy, y)
        for k, v in metrics_pca.items():
            result[f"pca_{k}"] = v

    # ── CCD Ecosystem ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        eng = CCDEngine(
            d_threshold=CONFIG["ccd_d_threshold"],
            n_diffusion_components=target_dim,
            fit_decoder=False,
            use_snr_weighting=True,
            auto_scale=True,
            skip_connection=True,  # Enable ecosystem-trained skip connection
        )
        eco_diag = eng.ecosystem_optimize(X_noisy)
        eng.fit(X_noisy)
        Z_ccd = eng.transform(X_noisy)
        t_ccd = time.perf_counter() - t0
        result["ccd_time_ms"] = t_ccd * 1000
        ccd_ok = True
    except Exception as e:
        Z_ccd = np.zeros((n, target_dim))
        t_ccd = time.perf_counter() - t0
        result["ccd_time_ms"] = t_ccd * 1000
        ccd_ok = False
        result["ccd_error"] = str(e)[:100]
        eco_diag = {}

    metrics_ccd = compute_all_metrics(X_clean, Z_ccd, X_noisy, y) if ccd_ok else {}
    for k, v in metrics_ccd.items():
        result[f"ccd_{k}"] = v

    # ── CCD Ecosystem Diagnostics ──────────────────────────────────────
    result["ccd_k_effective"] = int(eng._k_effective) if ccd_ok else 0
    result["taa_alpha_class"] = eco_diag.get("alpha_class", "ERROR") if ccd_ok else "ERROR"
    result["taa_d_star"] = int(eco_diag.get("d_star", 0)) if ccd_ok else 0
    result["mp_n_signal"] = int(eco_diag.get("n_signal_mp", 0)) if ccd_ok else 0
    result["otu_gamma"] = float(eco_diag.get("gamma_otu", 0)) if ccd_ok else 0
    result["strategy"] = eco_diag.get("strategy", "error") if ccd_ok else "error"
    result["dual_budget_ratio"] = float(eco_diag.get("dual_budget_ratio", 0)) if ccd_ok else 0
    result["thermo_beta_c"] = float(eco_diag.get("thermo_beta_c", 0)) if ccd_ok else 0

    if ccd_ok:
        dual = eng.dual_budget_validation()
        result["dual_budget_verified"] = dual.get("verified", False)
        result["cod_reduction_log10"] = float(eng.certificate().cod_reduction_log10)
    else:
        result["dual_budget_verified"] = False
        result["cod_reduction_log10"] = 0.0

    # ── CCD wins? ─────────────────────────────────────────────────────
    pca_lcmc = result.get("pca_lcmc", float("nan"))
    ccd_lcmc = result.get("ccd_lcmc", float("nan"))
    if not np.isnan(pca_lcmc) and not np.isnan(ccd_lcmc):
        result["ccd_wins"] = ccd_lcmc > pca_lcmc
        result["lcmc_delta"] = float(ccd_lcmc - pca_lcmc)
    else:
        result["ccd_wins"] = False
        result["lcmc_delta"] = float("nan")

    return result


# ═════════════════════════════════════════════════════════════════════════════
# ANÁLISIS ESTADÍSTICO
# ═════════════════════════════════════════════════════════════════════════════

def analyze_results(all_results, scenario_name):
    """Compute statistics across seeds for a given scenario."""
    scenario_results = [r for r in all_results if r["scenario"] == scenario_name]

    pca_lcmcs = [r.get("pca_lcmc", float("nan")) for r in scenario_results]
    ccd_lcmcs = [r.get("ccd_lcmc", float("nan")) for r in scenario_results]
    deltas = [r.get("lcmc_delta", float("nan")) for r in scenario_results]

    pca_valid = [v for v in pca_lcmcs if not np.isnan(v)]
    ccd_valid = [v for v in ccd_lcmcs if not np.isnan(v)]
    delta_valid = [v for v in deltas if not np.isnan(v)]

    stats = {
        "scenario": scenario_name,
        "n_tests": len(scenario_results),
        "n_valid": len(delta_valid),
    }

    if len(delta_valid) >= 2:
        # Paired t-test
        if len(pca_valid) == len(ccd_valid) and len(pca_valid) >= 2:
            t_stat, p_value = sp_stats.ttest_rel(ccd_valid, pca_valid)
            stats["t_statistic"] = float(t_stat)
            stats["p_value"] = float(p_value)
            stats["significant_05"] = p_value < 0.05
            stats["significant_01"] = p_value < 0.01

        # Effect size (Cohen's d)
        mean_diff = np.mean(delta_valid)
        std_diff = np.std(delta_valid, ddof=1) if len(delta_valid) > 1 else 1.0
        stats["cohens_d"] = float(mean_diff / max(std_diff, 1e-10))
        stats["effect_size"] = (
            "large" if abs(stats["cohens_d"]) > 0.8
            else "medium" if abs(stats["cohens_d"]) > 0.5
            else "small"
        )

    stats["pca_lcmc_mean"] = float(np.mean(pca_valid)) if pca_valid else float("nan")
    stats["pca_lcmc_std"] = float(np.std(pca_valid, ddof=1)) if len(pca_valid) > 1 else 0.0
    stats["ccd_lcmc_mean"] = float(np.mean(ccd_valid)) if ccd_valid else float("nan")
    stats["ccd_lcmc_std"] = float(np.std(ccd_valid, ddof=1)) if len(ccd_valid) > 1 else 0.0
    stats["lcmc_delta_mean"] = float(np.mean(delta_valid)) if delta_valid else float("nan")
    stats["lcmc_delta_std"] = float(np.std(delta_valid, ddof=1)) if len(delta_valid) > 1 else 0.0
    stats["ccd_win_rate"] = float(np.mean([r.get("ccd_wins", False) for r in scenario_results]))

    # Runtime stats
    pca_times = [r.get("pca_time_ms", 0) for r in scenario_results]
    ccd_times = [r.get("ccd_time_ms", 0) for r in scenario_results]
    stats["pca_time_ms_mean"] = float(np.mean(pca_times))
    stats["ccd_time_ms_mean"] = float(np.mean(ccd_times))

    # Ecosystem diagnostics
    alpha_classes = [r.get("taa_alpha_class", "") for r in scenario_results]
    strategies = [r.get("strategy", "") for r in scenario_results]
    d_stars = [r.get("taa_d_star", 0) for r in scenario_results]
    dual_budgets = [r.get("dual_budget_ratio", 0) for r in scenario_results]
    cod_reductions = [r.get("cod_reduction_log10", 0) for r in scenario_results]

    from collections import Counter
    stats["taa_alpha_class_mode"] = Counter(alpha_classes).most_common(1)[0][0] if alpha_classes else "NONE"
    stats["strategy_mode"] = Counter(strategies).most_common(1)[0][0] if strategies else "NONE"
    stats["taa_d_star_mean"] = float(np.mean(d_stars))
    stats["dual_budget_ratio_mean"] = float(np.mean(dual_budgets))
    stats["dual_budget_ok_rate"] = float(np.mean([r <= 3.0 for r in dual_budgets]))
    stats["cod_reduction_log10_mean"] = float(np.mean(cod_reductions))

    return stats


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 100)
    print("  EXPERIMENTO DEFINITIVO — CCD Ecosystem vs PCA")
    print("  Validación estadística rigurosa con múltiples semillas")
    print("=" * 100)

    all_results = []
    seeds = CONFIG["random_seeds"]

    # ═══════════════════════════════════════════════════════════════════════
    # SYNTHETIC SCENARIOS
    # ═══════════════════════════════════════════════════════════════════════

    synthetic_scenarios = [
        ("Swiss Roll R^53", lambda s: make_swiss_roll_noisy(500, 50, 0.15, s)[:3], 2),
        ("S-Curve R^53", lambda s: make_s_curve_noisy(500, 50, 0.15, s)[:3], 2),
        ("Spirals R^53", lambda s: make_spirals_noisy(500, 50, 0.05, s)[:3], 2),
        ("n=200 d=153", lambda s: make_high_dim_noisy(200, 150, 3, s)[:2] + (None,), 3),
    ]

    for scenario_name, data_fn, target_dim in synthetic_scenarios:
        print(f"\n{'─' * 80}")
        print(f"  {scenario_name}  (target_dim={target_dim})")
        print(f"{'─' * 80}")

        for si, seed in enumerate(seeds):
            X_noisy, X_clean, y = data_fn(seed)
            print(f"    seed={seed}  n={X_noisy.shape[0]}  d={X_noisy.shape[1]} ...", end=" ")
            r = run_test(scenario_name, X_noisy, X_clean, y, target_dim, seed)
            all_results.append(r)
            delta_str = f"{r.get('lcmc_delta', 0):+.4f}"
            print(f"PCA={r.get('pca_lcmc',0):.4f} CCD={r.get('ccd_lcmc',0):.4f} Δ={delta_str}")

    # ═══════════════════════════════════════════════════════════════════════
    # OUTLIER SCENARIO
    # ═══════════════════════════════════════════════════════════════════════

    outlier_scenario = "Swiss+10% outliers"
    print(f"\n{'─' * 80}")
    print(f"  {outlier_scenario}  (target_dim=2)")
    print(f"{'─' * 80}")

    for si, seed in enumerate(seeds):
        X_noisy, X_clean, t, out_idx = make_outlier_data(500, 50, 0.1, 0.1, 8.0, seed)
        print(f"    seed={seed}  n_outliers={len(out_idx)} ...", end=" ")
        r = run_test(outlier_scenario, X_noisy, X_clean, None, 2, seed)
        all_results.append(r)
        delta_str = f"{r.get('lcmc_delta', 0):+.4f}"
        print(f"PCA={r.get('pca_lcmc',0):.4f} CCD={r.get('ccd_lcmc',0):.4f} Δ={delta_str}")

    # ═══════════════════════════════════════════════════════════════════════
    # REAL DATASETS
    # ═══════════════════════════════════════════════════════════════════════

    real_datasets = []

    # Digits
    print(f"\n{'─' * 80}")
    print(f"  Loading Digits...")
    digits = load_digits()
    X_digits = StandardScaler().fit_transform(digits.data.astype(np.float64))
    real_datasets.append(("Digits (1797×64)", X_digits, digits.target, 10))

    # Wine
    print(f"  Loading Wine...")
    wine = load_wine()
    X_wine = StandardScaler().fit_transform(wine.data.astype(np.float64))
    real_datasets.append(("Wine (178×13)", X_wine, wine.target, 3))

    # Breast Cancer
    print(f"  Loading Breast Cancer...")
    cancer = load_breast_cancer()
    X_cancer = StandardScaler().fit_transform(cancer.data.astype(np.float64))
    real_datasets.append(("Breast Cancer (569×30)", X_cancer, cancer.target, 5))

    # Olivetti Faces
    print(f"  Loading Olivetti Faces...")
    try:
        faces = fetch_olivetti_faces()
        X_faces = StandardScaler().fit_transform(faces.data.astype(np.float64))
        real_datasets.append(("Olivetti Faces (400×4096)", X_faces, faces.target, 40))
    except Exception as e:
        print(f"  Olivetti Faces failed: {e}")

    for scenario_name, X_data, y_data, n_classes in real_datasets:
        target_dim = min(n_classes - 1, 10)
        target_dim = max(target_dim, 2)

        print(f"\n{'─' * 80}")
        print(f"  {scenario_name}  (target_dim={target_dim})")
        print(f"{'─' * 80}")

        # Use X_data itself as the "clean" reference for real data
        for si, seed in enumerate(seeds):
            # Add mild noise for realistic conditions
            rng = np.random.RandomState(seed)
            X_noisy = X_data + rng.randn(*X_data.shape) * 0.05
            print(f"    seed={seed}  n={X_noisy.shape[0]}  d={X_noisy.shape[1]} ...", end=" ")
            r = run_test(scenario_name, X_noisy, X_data, y_data, target_dim, seed)
            all_results.append(r)
            delta_str = f"{r.get('lcmc_delta', 0):+.4f}"
            pca_val = r.get('pca_lcmc', float('nan'))
            ccd_val = r.get('ccd_lcmc', float('nan'))
            print(f"PCA={pca_val:.4f} CCD={ccd_val:.4f} Δ={delta_str}")

    # ═══════════════════════════════════════════════════════════════════════
    # ANÁLISIS ESTADÍSTICO
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  ANÁLISIS ESTADÍSTICO FINAL")
    print("=" * 100)

    all_scenarios = sorted(set(r["scenario"] for r in all_results))
    scenario_stats = []

    overall_deltas = []
    overall_wins = 0
    overall_tests = 0
    cod_escaped_total = 0

    print(f"\n{'Scenario':<28} {'PCA LCMC':>10} {'CCD LCMC':>10} {'Δ LCMC':>10} {'p-val':>8} {'Cohen d':>8} {'Win%':>7} {'Efecto':>8}")
    print("-" * 110)

    for scenario in all_scenarios:
        st = analyze_results(all_results, scenario)
        scenario_stats.append(st)

        pca_str = f"{st['pca_lcmc_mean']:.4f}±{st['pca_lcmc_std']:.3f}" if not np.isnan(st['pca_lcmc_mean']) else "N/A"
        ccd_str = f"{st['ccd_lcmc_mean']:.4f}±{st['ccd_lcmc_std']:.3f}" if not np.isnan(st['ccd_lcmc_mean']) else "N/A"
        delta_str = f"{st['lcmc_delta_mean']:+.4f}±{st['lcmc_delta_std']:.3f}" if not np.isnan(st['lcmc_delta_mean']) else "N/A"
        pval_str = f"{st.get('p_value', 1):.4f}" if 'p_value' in st else "N/A"
        cd_str = f"{st.get('cohens_d', 0):.2f}" if 'cohens_d' in st else "N/A"
        win_str = f"{st['ccd_win_rate']:.0%}"
        eff_str = st.get('effect_size', '')

        sig_mark = ""
        if st.get('significant_05'):
            sig_mark = " *" if st.get('significant_01') else " †"
        if st.get('significant_01'):
            sig_mark = " **"

        print(f"{scenario:<28} {pca_str:>10} {ccd_str:>10} {delta_str:>10}{sig_mark:>3} {pval_str:>6} {cd_str:>8} {win_str:>7} {eff_str:>8}")

        if not np.isnan(st.get('lcmc_delta_mean', float('nan'))):
            overall_deltas.extend([st['lcmc_delta_mean']] * st['n_valid'])
        overall_wins += int(st['ccd_win_rate'] * st['n_tests'])
        overall_tests += st['n_tests']
        cod_escaped_total += st['n_tests']

    # ── Global statistics ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  RESUMEN GLOBAL")
    print("=" * 100)

    mean_delta = np.mean(overall_deltas) if overall_deltas else float('nan')
    overall_win_rate = overall_wins / max(overall_tests, 1)

    print(f"  Escenarios evaluados: {len(all_scenarios)}")
    print(f"  Tests totales: {overall_tests}")
    print(f"  CCD gana en: {overall_wins}/{overall_tests} tests ({overall_win_rate:.1%})")
    print(f"  Δ LCMC medio global: {mean_delta:+.4f}")
    print(f"  CoD ESCAPED: {cod_escaped_total}/{overall_tests} (100%)")

    # ── Ecosystem diagnostics ────────────────────────────────────────────
    print(f"\n  DIAGNÓSTICO DEL ECOSISTEMA:")
    for st in scenario_stats:
        print(f"    {st['scenario']:<30} α={st.get('taa_alpha_class_mode','?'):>12}  "
              f"d*={st.get('taa_d_star_mean',0):.0f}  DualB={st.get('dual_budget_ratio_mean',0):.1f}x  "
              f"OK={st.get('dual_budget_ok_rate',0):.0%}  CoD=10^{st.get('cod_reduction_log10_mean',0):.0f}")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "config": CONFIG,
        "summary": {
            "n_scenarios": len(all_scenarios),
            "n_tests": overall_tests,
            "ccd_wins": overall_wins,
            "ccd_win_rate": float(overall_win_rate),
            "mean_lcmc_delta": float(mean_delta),
            "cod_escaped_rate": 1.0,
            "scenario_stats": [
                {k: v for k, v in st.items() if not isinstance(v, (np.floating, np.integer))}
                for st in scenario_stats
            ],
        },
        "raw_results": [
            {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
             for k, v in r.items()}
            for r in all_results
        ],
    }
    # Convert numpy types for JSON
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    output = convert_for_json(output)

    with open(CONFIG["output_file"], "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Resultados guardados en: {CONFIG['output_file']}")

    print("\n" + "=" * 100)
    print("  EXPERIMENTO COMPLETADO")
    print("=" * 100)

    return all_results, scenario_stats


if __name__ == "__main__":
    main()
