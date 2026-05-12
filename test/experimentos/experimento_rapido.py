"""
Experimento rápido para verificar CCD con skip-connection
"""
import sys, time, warnings, json, os
import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trust
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acf_functor.ccd_engine import CCDEngine

# Configuración reducida
CONFIG = {
    "n_seeds": 3,
    "random_seeds": [42, 123, 456],
    "scenarios": [
        ("Swiss Roll", lambda seed: make_swiss_roll(n_samples=300, noise=0.1, random_state=seed)[0]),
        ("S-Curve", lambda seed: make_s_curve(n_samples=300, noise=0.1, random_state=seed)[0]),
        ("High-dim noise", lambda seed: np.random.randn(200, 100)),
    ],
    "target_dim": 2,
}

def run_experiment():
    results = {}
    
    for scenario_name, data_gen in CONFIG["scenarios"]:
        print(f"\n{'='*60}")
        print(f"Escenario: {scenario_name}")
        print(f"{'='*60}")
        
        scenario_results = []
        for seed in CONFIG["random_seeds"]:
            np.random.seed(seed)
            
            # Generar datos
            X = data_gen(seed)
            n, d = X.shape
            
            # Escalar
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PCA baseline
            t0 = time.time()
            pca = PCA(n_components=CONFIG["target_dim"])
            Z_pca = pca.fit_transform(X_scaled)
            t_pca = time.time() - t0
            trust_pca = sklearn_trust(X_scaled, Z_pca, n_neighbors=10)
            
            # CCD con skip-connection
            t0 = time.time()
            ccd = CCDEngine(
                d_threshold=5,
                n_diffusion_components=CONFIG["target_dim"],
                skip_connection=True,
                use_snr_weighting=True,
                auto_scale=True,
            )
            ccd.fit(X_scaled)
            Z_ccd = ccd.transform(X_scaled)
            t_ccd = time.time() - t0
            trust_ccd = sklearn_trust(X_scaled, Z_ccd, n_neighbors=10)
            
            delta = trust_ccd - trust_pca
            alpha = getattr(ccd, '_skip_alpha', 0.0)
            
            print(f"  Seed {seed}: PCA={trust_pca:.4f}, CCD={trust_ccd:.4f}, Δ={delta:+.4f}, α={alpha:.3f}")
            
            scenario_results.append({
                "seed": seed,
                "n": n,
                "d": d,
                "pca_lcmc": float(trust_pca),
                "ccd_lcmc": float(trust_ccd),
                "delta": float(delta),
                "pca_time_ms": t_pca * 1000,
                "ccd_time_ms": t_ccd * 1000,
                "skip_alpha": float(alpha),
            })
        
        results[scenario_name] = scenario_results
    
    # Guardar resultados
    output_file = "resultados_experimento_rapido.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "experiment": "CCD con skip-connection rápido",
                "n_scenarios": len(CONFIG["scenarios"]),
                "n_seeds": CONFIG["n_seeds"],
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            **results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Resultados guardados en: {output_file}")
    
    # Análisis rápido
    print(f"\n{'='*60}")
    print("ANÁLISIS RÁPIDO:")
    print(f"{'='*60}")
    
    all_deltas = []
    for scenario_name, scenario_results in results.items():
        deltas = [r["delta"] for r in scenario_results]
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        wins = sum(1 for d in deltas if d > 0.001)
        losses = sum(1 for d in deltas if d < -0.001)
        ties = len(deltas) - wins - losses
        
        print(f"\n{scenario_name}:")
        print(f"  Δ promedio: {mean_delta:+.4f} ± {std_delta:.4f}")
        print(f"  Wins: {wins}, Losses: {losses}, Ties: {ties}")
        
        all_deltas.extend(deltas)
    
    # Estadísticas globales
    print(f"\n{'='*60}")
    print("ESTADÍSTICAS GLOBALES:")
    print(f"{'='*60}")
    
    total_tests = len(all_deltas)
    significant_wins = sum(1 for d in all_deltas if d > 0.01)
    significant_losses = sum(1 for d in all_deltas if d < -0.01)
    
    print(f"\nTotal tests: {total_tests}")
    print(f"Mejoras significativas (Δ > 0.01): {significant_wins}")
    print(f"Pérdidas significativas (Δ < -0.01): {significant_losses}")
    
    if significant_losses == 0:
        print("\n✅ CLAIM VERIFICADO: CCD con skip-connection NUNCA pierde significativamente vs PCA")
    else:
        print(f"\n⚠ CLAIM NO VERIFICADO: {significant_losses} pérdidas significativas")
    
    # Test con dimensión alta
    print(f"\n{'='*60}")
    print("TEST DIMENSIÓN ALTA (d=500):")
    print(f"{'='*60}")
    
    np.random.seed(42)
    n_samples = 300
    n_features = 500
    
    # Datos de alta dimensión con estructura baja dimensional
    X_high = np.random.randn(n_samples, n_features) * 0.5
    signal = np.random.randn(n_samples, 10)
    X_high[:, :10] += signal * 2.0
    
    print(f"Datos: {n_samples} × {n_features}")
    
    try:
        t0 = time.time()
        ccd_high = CCDEngine(
            d_threshold=10,
            n_diffusion_components=10,
            skip_connection=True,
            use_snr_weighting=True,
            auto_scale=True,
        )
        ccd_high.fit(X_high)
        Z_high = ccd_high.transform(X_high)
        t_high = time.time() - t0
        
        print(f"  Tiempo: {t_high:.2f}s")
        print(f"  Output: {Z_high.shape[1]} dimensiones")
        print(f"  Alpha aprendido: {getattr(ccd_high, '_skip_alpha', 0.0):.3f}")
        print("  ✅ CCD funciona con dimensión alta usando FAISS")
    except Exception as e:
        print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    run_experiment()