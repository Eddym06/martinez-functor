#!/usr/bin/env python3
"""
Experimento completo de 45 tests (9 escenarios × 5 semillas)
para validar estadísticamente que CCD+skip nunca pierde vs PCA
"""
import numpy as np
import json
import time
from datetime import datetime
from sklearn.datasets import make_swiss_roll, make_s_curve, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from acf_functor import CCDEngine

def generate_scenario(scenario_name, n_samples=1000, d=100, seed=42):
    """Genera datos para cada escenario"""
    np.random.seed(seed)
    
    if scenario_name == "swiss_roll":
        X, _ = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=seed)
        X = StandardScaler().fit_transform(X)
        
    elif scenario_name == "s_curve":
        X, _ = make_s_curve(n_samples=n_samples, noise=0.1, random_state=seed)
        X = StandardScaler().fit_transform(X)
        
    elif scenario_name == "high_dim_noise":
        # Alta dimensión con ruido estructurado
        X = np.random.randn(n_samples, d)
        # Añadir estructura débil
        X[:, :10] = X[:, :10] * 0.1 + np.random.randn(n_samples, 10) * 0.9
        X = StandardScaler().fit_transform(X)
        
    elif scenario_name == "outliers":
        # Datos con outliers extremos
        X = np.random.randn(n_samples, d)
        # Añadir 5% outliers
        n_outliers = int(0.05 * n_samples)
        # Crear outliers en todas las dimensiones
        outlier_multiplier = 10.0
        outlier_shift = 20.0
        X[:n_outliers] = X[:n_outliers] * outlier_multiplier + outlier_shift
        X = StandardScaler().fit_transform(X)
        
    elif scenario_name == "mnist_like":
        # Datos tipo MNIST (baja dimensión intrínseca)
        intrinsic_dim = 10
        basis = np.random.randn(d, intrinsic_dim)
        coefs = np.random.randn(n_samples, intrinsic_dim)
        X = coefs @ basis.T
        X = X + np.random.randn(n_samples, d) * 0.1
        X = StandardScaler().fit_transform(X)
        
    elif scenario_name == "linear":
        # Datos puramente lineales
        X = np.random.randn(n_samples, d)
        X = StandardScaler().fit_transform(X)
        
    elif scenario_name == "nonlinear_mixture":
        # Mezcla de componentes lineales y no lineales
        X_linear = np.random.randn(n_samples, d//2)
        t = np.linspace(0, 4*np.pi, n_samples)
        X_nonlinear = np.column_stack([np.sin(t), np.cos(t), np.sin(2*t), np.cos(2*t)])
        X_nonlinear = np.tile(X_nonlinear, (1, d//8))[:, :d//2]
        X = np.column_stack([X_linear, X_nonlinear])
        X = StandardScaler().fit_transform(X)
        
    elif scenario_name == "sparse_high_dim":
        # Datos sparse en alta dimensión
        X = np.zeros((n_samples, d))
        for i in range(n_samples):
            active = np.random.choice(d, size=20, replace=False)
            X[i, active] = np.random.randn(20)
        X = StandardScaler().fit_transform(X)
        
    elif scenario_name == "correlated_noise":
        # Ruido correlacionado
        cov = np.random.randn(d, d)
        cov = cov @ cov.T + np.eye(d) * 0.1
        X = np.random.multivariate_normal(np.zeros(d), cov, size=n_samples)
        X = StandardScaler().fit_transform(X)
        
    else:
        raise ValueError(f"Escenario desconocido: {scenario_name}")
    
    return X

def run_experiment(scenario_name, seed, output_dim=2):
    """Ejecuta un experimento individual"""
    print(f"  Ejecutando {scenario_name} (seed={seed})...")
    
    # Generar datos
    X = generate_scenario(scenario_name, seed=seed)
    
    # PCA baseline
    pca = PCA(n_components=output_dim, random_state=seed)
    X_pca = pca.fit_transform(X)
    trust_pca = trustworthiness(X, X_pca, n_neighbors=12)
    
    # CCD con skip-connection y ecosistema
    ccd = CCDEngine(
        d_threshold=2,  # Siempre activar CCD
        n_preprocess_components=output_dim,
        n_diffusion_components=output_dim,
        skip_connection=True,
        seed=seed,
        auto_scale=True,
        use_snr_weighting=True,
        spectral_gap_threshold=1.5
    )
    
    start_time = time.time()
    ccd.fit(X)
    X_ccd = ccd.transform(X)
    ccd_time = time.time() - start_time
    
    trust_ccd = trustworthiness(X, X_ccd, n_neighbors=12)
    
    # Obtener alpha aprendido
    alpha = getattr(ccd, 'skip_alpha', 0.5)
    
    # Métricas adicionales del ecosistema
    ecosystem_info = {}
    # Intentar obtener información del ecosistema si está disponible
    try:
        if hasattr(ccd, 'taa_report'):
            ecosystem_info['taa_class'] = ccd.taa_report.alpha_class.value
    except:
        pass
    try:
        if hasattr(ccd, 'otu_result'):
            ecosystem_info['spectral_gap'] = ccd.otu_result.spectral_gap
            ecosystem_info['correlation_dim'] = ccd.otu_result.correlation_dimension
    except:
        pass
    
    return {
        'scenario': scenario_name,
        'seed': seed,
        'trust_pca': float(trust_pca),
        'trust_ccd': float(trust_ccd),
        'improvement': float(trust_ccd - trust_pca),
        'alpha_learned': float(alpha),
        'ccd_time': float(ccd_time),
        'ecosystem_info': ecosystem_info,
        'timestamp': datetime.now().isoformat()
    }

def main():
    """Ejecuta el experimento completo"""
    scenarios = [
        "swiss_roll",
        "s_curve", 
        "high_dim_noise",
        "outliers",
        "mnist_like",
        "linear",
        "nonlinear_mixture",
        "sparse_high_dim",
        "correlated_noise"
    ]
    
    seeds = [42, 123, 456, 789, 999]
    
    print("=" * 80)
    print("EXPERIMENTO COMPLETO: 45 tests (9 escenarios × 5 semillas)")
    print("=" * 80)
    
    results = []
    total_tests = len(scenarios) * len(seeds)
    completed = 0
    
    for scenario in scenarios:
        print(f"\nEscenario: {scenario}")
        for seed in seeds:
            try:
                result = run_experiment(scenario, seed)
                results.append(result)
                completed += 1
                
                # Mostrar progreso
                print(f"    ✓ Seed {seed}: PCA={result['trust_pca']:.4f}, "
                      f"CCD={result['trust_ccd']:.4f}, Δ={result['improvement']:.4f}, "
                      f"α={result['alpha_learned']:.3f}, time={result['ccd_time']:.1f}s")
                
            except Exception as e:
                print(f"    ✗ Seed {seed}: ERROR - {str(e)}")
                results.append({
                    'scenario': scenario,
                    'seed': seed,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
    
    # Análisis estadístico
    print("\n" + "=" * 80)
    print("ANÁLISIS ESTADÍSTICO")
    print("=" * 80)
    
    # Filtrar resultados exitosos
    successful = [r for r in results if 'trust_pca' in r]
    
    if successful:
        improvements = [r['improvement'] for r in successful]
        alphas = [r['alpha_learned'] for r in successful]
        
        print(f"Tests exitosos: {len(successful)}/{total_tests}")
        print(f"Mejora promedio: {np.mean(improvements):.6f}")
        print(f"Mejora std: {np.std(improvements):.6f}")
        print(f"Mejora mínima: {np.min(improvements):.6f}")
        print(f"Mejora máxima: {np.max(improvements):.6f}")
        print(f"Alpha promedio: {np.mean(alphas):.3f}")
        
        # Contar victorias/empates/derrotas
        wins = sum(1 for r in successful if r['improvement'] > 0.001)
        ties = sum(1 for r in successful if abs(r['improvement']) <= 0.001)
        losses = sum(1 for r in successful if r['improvement'] < -0.001)
        
        print(f"\nResultados:")
        print(f"  Victorias CCD (Δ > 0.001): {wins}")
        print(f"  Empates (|Δ| ≤ 0.001): {ties}")
        print(f"  Derrotas CCD (Δ < -0.001): {losses}")
        
        # Tests estadísticos
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(improvements, 0)
        print(f"\nTest t de una muestra:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significativo (p < 0.05): {'SÍ' if p_value < 0.05 else 'NO'}")
        
        # Test de signos
        pos = sum(1 for imp in improvements if imp > 0)
        neg = sum(1 for imp in improvements if imp < 0)
        zero = sum(1 for imp in improvements if imp == 0)
        print(f"\nTest de signos:")
        print(f"  Positivos: {pos}, Negativos: {neg}, Ceros: {zero}")
        
    else:
        wins = ties = losses = 0
        t_stat = p_value = None
        
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"resultados_experimento_completo_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'total_tests': total_tests,
                'successful_tests': len(successful) if successful else 0,
                'timestamp': datetime.now().isoformat(),
                'scenarios': scenarios,
                'seeds': seeds
            },
            'results': results,
            'statistical_summary': {
                'mean_improvement': float(np.mean(improvements)) if successful else None,
                'std_improvement': float(np.std(improvements)) if successful else None,
                'min_improvement': float(np.min(improvements)) if successful else None,
                'max_improvement': float(np.max(improvements)) if successful else None,
                'wins': wins,
                'ties': ties,
                'losses': losses,
                't_statistic': float(t_stat) if t_stat is not None else None,
                'p_value': float(p_value) if p_value is not None else None
            }
        }, f, indent=2)
    
    print(f"\nResultados guardados en: {output_file}")
    
    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    
    if successful:
        if losses == 0:
            print("✅ CCD CON SKIP-CONNECTION NUNCA PIERDE SIGNIFICATIVAMENTE VS PCA")
            print(f"   ({wins} mejoras, {ties} empates, {losses} pérdidas)")
        else:
            print(f"⚠️  CCD tiene {losses} pérdidas significativas vs PCA")
    else:
        print("❌ No se completó ningún test exitosamente")
    
    return results

if __name__ == "__main__":
    main()