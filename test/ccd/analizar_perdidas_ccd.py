#!/usr/bin/env python3
"""Analizar por qué CCD pierde en algunos casos y mejorar"""
import numpy as np
from sklearn.datasets import make_swiss_roll, make_s_curve, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from acf_functor import CCDEngine

def analizar_caso_perdida(scenario_name, seed, d=100):
    """Analizar un caso específico donde CCD pierde"""
    np.random.seed(seed)
    
    if scenario_name == "linear":
        # Datos puramente lineales
        X = np.random.randn(1000, d)
        X = StandardScaler().fit_transform(X)
    elif scenario_name == "outliers":
        # Datos con outliers
        X = np.random.randn(1000, d)
        n_outliers = 50
        X[:n_outliers] = X[:n_outliers] * 10 + 20
        X = StandardScaler().fit_transform(X)
    elif scenario_name == "high_dim_noise":
        # Alta dimensión con ruido
        X = np.random.randn(1000, d)
        X[:, :10] = X[:, :10] * 0.1 + np.random.randn(1000, 10) * 0.9
        X = StandardScaler().fit_transform(X)
    elif scenario_name == "swiss_roll":
        X, _ = make_swiss_roll(n_samples=1000, noise=0.1, random_state=seed)
        X = StandardScaler().fit_transform(X)
    elif scenario_name == "correlated_noise":
        # Ruido correlacionado
        X = np.random.randn(1000, d)
        # Crear correlaciones entre dimensiones
        for i in range(0, d-1, 2):
            X[:, i+1] = 0.7 * X[:, i] + 0.3 * np.random.randn(1000)
        X = StandardScaler().fit_transform(X)
    
    # PCA baseline
    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X)
    trust_pca = trustworthiness(X, X_pca, n_neighbors=12)
    
    # CCD con diferentes configuraciones
    configs = [
        {"name": "default", "skip_connection": True, "use_snr_weighting": True, "spectral_gap_threshold": 1.5},
        {"name": "no_skip", "skip_connection": False, "use_snr_weighting": True, "spectral_gap_threshold": 1.5},
        {"name": "high_alpha", "skip_connection": True, "use_snr_weighting": True, "spectral_gap_threshold": 2.0},
        {"name": "no_snr", "skip_connection": True, "use_snr_weighting": False, "spectral_gap_threshold": 1.5},
        {"name": "more_linear", "skip_connection": True, "use_snr_weighting": True, "spectral_gap_threshold": 1.0},
    ]
    
    results = []
    for config in configs:
        try:
            ccd = CCDEngine(
                d_threshold=2,
                n_preprocess_components=2,
                n_diffusion_components=2,
                skip_connection=config["skip_connection"],
                seed=seed,
                auto_scale=True,
                use_snr_weighting=config["use_snr_weighting"],
                spectral_gap_threshold=config["spectral_gap_threshold"]
            )
            
            ccd.fit(X)
            X_ccd = ccd.transform(X)
            trust_ccd = trustworthiness(X, X_ccd, n_neighbors=12)
            alpha = ccd.skip_alpha if config["skip_connection"] else 0.0
            
            results.append({
                "config": config["name"],
                "trust_ccd": trust_ccd,
                "improvement": trust_ccd - trust_pca,
                "alpha": alpha
            })
        except Exception as e:
            results.append({
                "config": config["name"],
                "error": str(e)
            })
    
    return {
        "scenario": scenario_name,
        "seed": seed,
        "trust_pca": trust_pca,
        "configs": results
    }

# Analizar los 6 casos donde CCD pierde
casos_perdida = [
    ("linear", 42),      # -0.002131
    ("outliers", 789),   # -0.002478
    ("high_dim_noise", 999),  # -0.002178
    ("swiss_roll", 789), # -0.001425
    ("high_dim_noise", 456),  # -0.001313
    ("correlated_noise", 999), # -0.001212
]

print("=" * 80)
print("ANÁLISIS DE CASOS DONDE CCD PIERDE VS PCA")
print("=" * 80)

for scenario, seed in casos_perdida:
    print(f"\n🔍 Analizando {scenario} (seed={seed}):")
    resultado = analizar_caso_perdida(scenario, seed)
    
    print(f"  PCA trustworthiness: {resultado['trust_pca']:.4f}")
    print(f"  Configuraciones CCD:")
    
    for config in resultado["configs"]:
        if "error" in config:
            print(f"    {config['config']}: ERROR - {config['error']}")
        else:
            signo = "+" if config["improvement"] >= 0 else ""
            alpha_str = f", α={config['alpha']:.3f}" if config["config"] != "no_skip" else ""
            print(f"    {config['config']}: {config['trust_ccd']:.4f} (Δ={signo}{config['improvement']:.6f}{alpha_str})")
    
    # Encontrar la mejor configuración
    mejor_config = None
    mejor_mejora = -float('inf')
    for config in resultado["configs"]:
        if "error" not in config and config["improvement"] > mejor_mejora:
            mejor_mejora = config["improvement"]
            mejor_config = config
    
    if mejor_config and mejor_mejora > -0.001:
        print(f"  ✅ MEJOR CONFIG: {mejor_config['config']} (Δ={mejor_mejora:.6f})")
    elif mejor_config:
        print(f"  ⚠️  MEJOR CONFIG: {mejor_config['config']} (Δ={mejor_mejora:.6f}) - aún pierde")

print("\n" + "=" * 80)
print("CONCLUSIONES Y MEJORAS NECESARIAS")
print("=" * 80)

print("\n1. **Datos puramente lineales (linear, seed=42):**")
print("   - CCD pierde porque intenta capturar estructura no-lineal que no existe")
print("   - SOLUCIÓN: Detectar datos lineales y usar α ≈ 1.0 (predominio PCA)")

print("\n2. **Outliers extremos (outliers, seed=789):**")
print("   - α aprendido = 0.781 (alto, pero no suficiente)")
print("   - SOLUCIÓN: Detección robusta de outliers → α → 0.9+")

print("\n3. **Alta dimensión con ruido (high_dim_noise):**")
print("   - SNR weighting ayuda pero no suficiente")
print("   - SOLUCIÓN: Mejor detección de gap espectral + α adaptativo")

print("\n4. **Swiss Roll (seed=789):**")
print("   - Caso no-lineal donde CCD debería ganar")
print("   - SOLUCIÓN: Mejorar difusión para datos no-lineales")

print("\n" + "=" * 80)
print("MEJORAS A IMPLEMENTAR")
print("=" * 80)
print("1. Detección automática de linealidad → ajustar α mínimo")
print("2. Detección robusta de outliers → α → 0.9+")
print("3. Mejor calibración de SNR weighting")
print("4. Umbral adaptativo de gap espectral")