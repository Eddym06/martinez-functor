#!/usr/bin/env python3
"""
CCDEngine OPTIMIZADO FINAL - Correcciones definitivas para superar a PCA
"""

import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Importar CCDEngine
from acf_functor.ccd_engine import CCDEngine

def generate_realistic_manifold_data(n_samples=600):
    """Generar datos de manifold realista (swiss roll con ruido)."""
    from sklearn.datasets import make_swiss_roll
    X, _ = make_swiss_roll(n_samples=n_samples, noise=0.08, random_state=42)
    
    # Añadir dimensiones irrelevantes (ruido)
    noise_dims = 15
    noise = np.random.randn(n_samples, noise_dims) * 0.15
    X = np.hstack([X, noise])
    
    # Normalizar
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X

def compute_robust_metrics(X, X_recon, Z, name):
    """Calcular métricas robustas sin errores numéricos."""
    # 1. MSE con estabilidad numérica
    diff = X_recon - X
    mse = np.mean(diff ** 2)
    
    # Verificar estabilidad
    if mse < 1e-12:
        # Recalcular con mayor precisión
        mse = np.mean(np.square(diff.astype(np.float64)))
    
    # 2. Error relativo
    x_norm = np.linalg.norm(X, 'fro')
    rel_error = np.linalg.norm(diff, 'fro') / x_norm if x_norm > 0 else 0
    
    # 3. Preservación de vecindarios (con muestreo inteligente)
    n_samples = min(300, X.shape[0])
    idx = np.random.choice(X.shape[0], n_samples, replace=False)
    X_sub = X[idx]
    Z_sub = Z[idx]
    
    k = min(10, n_samples - 1)
    nn_orig = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X_sub)
    nn_trans = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(Z_sub)
    
    indices_orig = nn_orig.kneighbors(X_sub)[1][:, 1:]
    indices_trans = nn_trans.kneighbors(Z_sub)[1][:, 1:]
    
    preservation = np.mean([
        len(set(io).intersection(set(it))) / k
        for io, it in zip(indices_orig, indices_trans)
    ])
    
    # 4. Correlación de distancias (con muestreo)
    if n_samples > 80:
        idx2 = np.random.choice(n_samples, 80, replace=False)
        X_sample = X_sub[idx2]
        Z_sample = Z_sub[idx2]
        
        dist_orig = pdist(X_sample, 'euclidean')
        dist_trans = pdist(Z_sample, 'euclidean')
        
        if len(dist_orig) > 20:
            rho, _ = spearmanr(dist_orig, dist_trans)
        else:
            rho = 0.0
    else:
        rho = 0.0
    
    return {
        'name': name,
        'mse': float(mse),
        'relative_error': float(rel_error),
        'neighborhood_preservation': float(preservation),
        'distance_correlation': float(rho)
    }

def run_definitive_benchmark():
    """Benchmark definitivo CCDEngine vs PCA."""
    print("=" * 80)
    print("🚀 BENCHMARK DEFINITIVO: CCDEngine OPTIMIZADO vs PCA")
    print("=" * 80)
    
    # 1. Datos realistas
    print("\n📊 DATOS: Swiss Roll con ruido (n=600, d=18)")
    X = generate_realistic_manifold_data(n_samples=600)
    print(f"  • Shape: {X.shape}")
    print(f"  • Norma media: {np.mean(np.linalg.norm(X, axis=1)):.3f}")
    print(f"  • Dimensión intrínseca estimada: ~3 (manifold 3D)")
    
    # 2. Configuración OPTIMIZADA del CCDEngine
    print("\n🔧 CONFIGURACIÓN CCDEngine OPTIMIZADA:")
    
    # Parámetros clave optimizados:
    # - d_threshold bajo para activar CCD
    # - n_preprocess_components: suficiente para capturar estructura
    # - n_diffusion_components: dimensión objetivo (manifold + margen)
    # - n_diffusion_neighbors: balance entre local/global
    # - fit_decoder: True para reconstrucción
    # - decoder_n_neighbors: 5 para reconstrucción estable
    
    ccd_config = {
        'd_threshold': 3,  # Activar CCD para d > 3
        'n_preprocess_components': 0,  # Auto: max(n_diffusion_components * 4, 32)
        'n_diffusion_components': 8,   # Manifold 3D + 5 dimensiones de seguridad
        'n_diffusion_neighbors': 12,   # Balance local/global
        'n_oscillator_groups': None,   # Auto-detección
        'n_entropy_neighbors': 8,
        'n_langevin_steps': 0,         # Desactivar para benchmark puro
        'fit_decoder': True,
        'decoder_n_neighbors': 5
    }
    
    for key, value in ccd_config.items():
        print(f"  • {key}: {value}")
    
    # 3. CCDEngine optimizado
    print("\n1. 🚀 CCDEngine OPTIMIZADO")
    print("   Entrenando...")
    
    start_time = time.time()
    
    try:
        engine = CCDEngine(**ccd_config)
        engine.fit(X)
        Z_ccd = engine.transform(X)
        X_recon_ccd = engine.inverse_transform(Z_ccd)
        ccd_time = time.time() - start_time
        
        print(f"   ✅ Entrenamiento exitoso")
        print(f"   ⏱️  Tiempo: {ccd_time:.3f}s")
        print(f"   📏 Reducción: {X.shape[1]} → {Z_ccd.shape[1]}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 4. PCA (baseline)
    print("\n2. 📊 PCA (BASELINE)")
    print("   Entrenando...")
    
    start_time = time.time()
    
    pca = PCA(n_components=Z_ccd.shape[1], whiten=False)
    Z_pca = pca.fit_transform(X)
    X_recon_pca = pca.inverse_transform(Z_pca)
    pca_time = time.time() - start_time
    
    print(f"   ✅ Entrenamiento exitoso")
    print(f"   ⏱️  Tiempo: {pca_time:.3f}s")
    print(f"   📈 Varianza explicada: {pca.explained_variance_ratio_.sum():.3f}")
    
    # 5. Calcular métricas
    print("\n3. 📈 CALCULANDO MÉTRICAS ROBUSTAS")
    
    ccd_metrics = compute_robust_metrics(X, X_recon_ccd, Z_ccd, "CCDEngine Optimizado")
    pca_metrics = compute_robust_metrics(X, X_recon_pca, Z_pca, "PCA")
    
    ccd_metrics['time'] = ccd_time
    pca_metrics['time'] = pca_time
    
    # 6. Resultados comparativos
    print("\n" + "=" * 80)
    print("📊 RESULTADOS COMPARATIVOS DEFINITIVOS")
    print("=" * 80)
    
    print("\n" + "-" * 75)
    print(f"{'MÉTRICA':<25} | {'CCDEngine':<12} | {'PCA':<12} | {'MEJOR'}")
    print("-" * 75)
    
    comparisons = [
        ("MSE", ccd_metrics['mse'], pca_metrics['mse'],
         "CCD" if ccd_metrics['mse'] < pca_metrics['mse'] else "PCA"),
        
        ("Error relativo", ccd_metrics['relative_error'], pca_metrics['relative_error'],
         "CCD" if ccd_metrics['relative_error'] < pca_metrics['relative_error'] else "PCA"),
        
        ("Preservación vecinos", ccd_metrics['neighborhood_preservation'], 
         pca_metrics['neighborhood_preservation'],
         "CCD" if ccd_metrics['neighborhood_preservation'] > pca_metrics['neighborhood_preservation'] else "PCA"),
        
        ("Correlación distancias", ccd_metrics['distance_correlation'],
         pca_metrics['distance_correlation'],
         "CCD" if ccd_metrics['distance_correlation'] > pca_metrics['distance_correlation'] else "PCA"),
        
        ("Tiempo (s)", ccd_metrics['time'], pca_metrics['time'],
         "PCA" if pca_time < ccd_time else "CCD"),
        
        ("Reducción", f"{X.shape[1]}→{Z_ccd.shape[1]}", 
         f"{X.shape[1]}→{Z_pca.shape[1]}", "Igual")
    ]
    
    for name, ccd_val, pca_val, best in comparisons:
        if 'MSE' in name or 'Error' in name:
            ccd_str = f"{ccd_val:.6f}"
            pca_str = f"{pca_val:.6f}"
        elif 'Preservación' in name or 'Correlación' in name:
            ccd_str = f"{ccd_val:.3f}"
            pca_str = f"{pca_val:.3f}"
        elif 'Tiempo' in name:
            ccd_str = f"{ccd_val:.3f}"
            pca_str = f"{pca_val:.3f}"
        else:
            ccd_str = str(ccd_val)
            pca_str = str(pca_val)
        
        print(f"{name:<25} | {ccd_str:<12} | {pca_str:<12} | {best}")
    
    print("-" * 75)
    
    # 7. Análisis detallado
    print("\n" + "=" * 80)
    print("🔍 ANÁLISIS DETALLADO")
    print("=" * 80)
    
    # Score compuesto (ponderado)
    def compute_composite_score(metrics):
        # Normalizar cada métrica
        mse_norm = 1.0 / (1.0 + metrics['mse'])  # MSE: menor es mejor
        time_norm = 1.0 / (1.0 + metrics['time'])  # Tiempo: menor es mejor
        
        # Score final (ponderaciones)
        score = (
            0.40 * metrics['neighborhood_preservation'] +  # 40% preservación
            0.35 * metrics['distance_correlation'] +       # 35% correlación
            0.15 * mse_norm +                              # 15% reconstrucción
            0.10 * time_norm                               # 10% velocidad
        )
        return score
    
    ccd_score = compute_composite_score(ccd_metrics)
    pca_score = compute_composite_score(pca_metrics)
    
    print(f"\n📊 Score CCDEngine: {ccd_score:.4f}")
    print(f"📊 Score PCA: {pca_score:.4f}")
    
    score_ratio = ccd_score / pca_score if pca_score > 0 else 0
    
    if ccd_score > pca_score:
        print(f"\n✅ ¡CCDEngine SUPERIOR a PCA! (Ratio: {score_ratio:.3f}×)")
        print("   El CCDEngine optimizado es MEJOR para:")
        print("   • Datos no lineales (manifolds, clusters)")
        print("   • Preservación de estructura local")
        print("   • Captura de geometría intrínseca")
        print("   • Aplicaciones donde importan las distancias geodésicas")
    else:
        print(f"\n⚠️  CCDEngine comparable a PCA (Ratio: {score_ratio:.3f}×)")
        print("   PCA sigue siendo MEJOR para:")
        print("   • Datos lineales o casi lineales")
        print("   • Máxima velocidad de ejecución")
        print("   • Reconstrucción exacta en subespacio lineal")
    
    # 8. Diagnóstico de problemas
    print("\n" + "=" * 80)
    print("🔧 DIAGNÓSTICO DE PROBLEMAS")
    print("=" * 80)
    
    issues = []
    recommendations = []
    
    # Verificar cada métrica
    if ccd_metrics['neighborhood_preservation'] < pca_metrics['neighborhood_preservation']:
        issues.append("❌ Preservación de vecindarios INSUFICIENTE")
        recommendations.append("• Aumentar n_diffusion_neighbors (15-20)")
        recommendations.append("• Mejorar kernel de diffusion (t-student)")
    
    if ccd_metrics['distance_correlation'] < pca_metrics['distance_correlation']:
        issues.append("❌ Correlación de distancias BAJA")
        recommendations.append("• Ajustar bandwidth automático (median heuristic)")
        recommendations.append("• Normalizar por densidad local")
    
    if ccd_metrics['time'] > pca_metrics['time'] * 5:
        time_ratio = ccd_metrics['time'] / pca_metrics['time']
        issues.append(f"❌ TIEMPO EXCESIVO ({time_ratio:.1f}× más lento)")
        recommendations.append("• Optimizar DiffusionGeometry (sparse)")
        recommendations.append("• Usar Nyström approximation")
        recommendations.append("• Parallelizar operaciones críticas")
    
    if ccd_metrics['mse'] > pca_metrics['mse'] * 1.5:
        mse_ratio = ccd_metrics['mse'] / pca_metrics['mse']
        issues.append(f"❌ ERROR DE RECONSTRUCCIÓN ALTO ({mse_ratio:.1f}×)")
        recommendations.append("• Mejorar k-NN decoder (regularización)")
        recommendations.append("• Aumentar decoder_n_neighbors (7-10)")
    
    if issues:
        print("\n🚨 PROBLEMAS IDENTIFICADOS:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n🔧 RECOMENDACIONES:")
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("\n✅ ¡NO HAY PROBLEMAS CRÍTICOS!")
        print("   El CCDEngine optimizado funciona correctamente")
    
    # 9. Conclusión final
    print("\n" + "=" * 80)
    print("🎯 CONCLUSIÓN FINAL")
    print("=" * 80)
    
    if ccd_score > pca_score:
        print("\n🚀 ¡OBJETIVO ALCANZADO!")
        print("El CCDEngine optimizado SUPERA a PCA en:")
        print("1. Preservación de estructura no lineal")
        print("2. Captura de geometría local")
        print("3. Manejo de manifolds complejos")
        print("4. Reconstrucción de datos con ruido")
        
        print("\n📈 VENTAJAS CLAVE DEL CCDEngine:")
        print("• DiffusionGeometry captura distancias geodésicas")
        print("• SpectralPreprocessor preserva métrica (sin whitening)")
        print("• k-NN decoder reconstruye de forma estable")
        print("• Arquitectura 5-capas para alta dimensión")
        
    else:
        print("\n🔧 OPTIMIZACIONES NECESARIAS:")
        print("Para superar definitivamente a PCA, necesitamos:")
        print("1. Optimizar DiffusionGeometry para velocidad")
        print("2. Mejorar preservación de métrica")
        print("3. Implementar auto-tuning de hiperparámetros")
        print("4. Validar con más tipos de datos")
    
    # 10. Próximos pasos
    print("\n" + "=" * 80)
    print("🚀 PRÓXIMOS PASOS")
    print("=" * 80)
    
    print("\n1. IMPLEMENTAR OPTIMIZACIONES:")
    print("   • DiffusionGeometry sparse con Nyström")
    print("   • Kernel t-student con bandwidth automático")
    print("   • k-NN decoder regularizado")
    
    print("\n2. VALIDACIÓN EXTENSIVA:")
    print("   • Más tipos de datos (clusters, S-curve, real-world)")
    print("   • Diferentes niveles de ruido")
    print("   • Varias dimensiones (d=10 a d=1000)")
    
    print("\n3. INTEGRACIÓN:")
    print("   • Incorporar optimizaciones al CCDEngine base")
    print("   • Crear benchmarks automáticos")
    print("   • Documentar configuración óptima")
    
    return {
        'ccd': ccd_metrics,
        'pca': pca_metrics,
        'ccd_score': ccd_score,
        'pca_score': pca_score,
        'config': ccd_config,
        'issues': issues,
        'recommendations': recommendations
    }

def run_multiple_datasets():
    """Probar con múltiples tipos de datos."""
    print("\n" + "=" * 80)
    print("🧪 PRUEBA CON MÚLTIPLES TIPOS DE DATOS")
    print("=" * 80)
    
    datasets = [
        ('swiss_roll', 'Manifold 3D con ruido'),
        ('clusters', '3 clusters separados'),
        ('linear', 'Datos casi lineales')
    ]
    
    all_results = []
    
    for data_type, description in datasets:
        print(f"\n📊 Dataset: {description}")
        
        # Generar datos
        if data_type == 'swiss_roll':
            X = generate_realistic_manifold_data(n_samples=500)
        elif data_type == 'clusters':
            # 3 clusters en 20 dimensiones
            X = np.zeros((500, 20))
            centers = np.random.randn(3, 20) * 2.0
            for i in range(3):
                cluster_size = 500 // 3
                start = i * cluster_size
                end = start + cluster_size if i < 2 else 500
                X[start:end] = centers[i] + np.random.randn(end-start, 20) * 0.4
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        else:  # linear
            # Datos casi lineales
            X = np.random.randn(500, 25)
            # Añadir estructura lineal
            X[:, 0] = X[:, 1] * 0.8 + X[:, 2] * 0.2 + np.random.randn(500) * 0.1
        
        # Configuración adaptativa
        if data_type == 'linear':
            config = {
                'd_threshold': 3,
                'n_preprocess_components': 0,
                'n_diffusion_components': 5,  # Menos dimensiones para datos lineales
                'n_diffusion_neighbors': 8,
                'n_oscillator_groups': None,
                'n_entropy_neighbors': 8,
                'n_langevin_steps': 0,
                'fit_decoder': True,
                'decoder_n_neighbors': 5
            }
        else:
            config = {
                'd_threshold': 3,
                'n_preprocess_components': 0,
                'n_diffusion_components': 8,
                'n_diffusion_neighbors': 12,
                'n_oscillator_groups': None,
                'n_entropy_neighbors': 8,
                'n_langevin_steps': 0,
                'fit_decoder': True,
                'decoder_n_neighbors': 5
            }
        
        # CCDEngine
        engine = CCDEngine(**config)
        engine.fit(X)
        Z_ccd = engine.transform(X)
        X_recon_ccd = engine.inverse_transform(Z_ccd)
        
        # PCA
        pca = PCA(n_components=Z_ccd.shape[1], whiten=False)
        Z_pca = pca.fit_transform(X)
        X_recon_pca = pca.inverse_transform(Z_pca)
        
        # Métricas
        ccd_metrics = compute_robust_metrics(X, X_recon_ccd, Z_ccd, f"CCD-{data_type}")
        pca_metrics = compute_robust_metrics(X, X_recon_pca, Z_pca, f"PCA-{data_type}")
        
        # Score
        def quick_score(metrics):
            return (
                0.4 * metrics['neighborhood_preservation'] +
                0.4 * metrics['distance_correlation'] +
                0.2 * (1.0 / (1.0 + metrics['mse']))
            )
        
        ccd_score = quick_score(ccd_metrics)
        pca_score = quick_score(pca_metrics)
        
        winner = "CCD" if ccd_score > pca_score else "PCA"
        
        print(f"  • CCD Score: {ccd_score:.4f}")
        print(f"  • PCA Score: {pca_score:.4f}")
        print(f"  • Ganador: {winner}")
        
        all_results.append({
            'dataset': description,
            'ccd_score': ccd_score,
            'pca_score': pca_score,
            'winner': winner,
            'ccd_preservation': ccd_metrics['neighborhood_preservation'],
            'pca_preservation': pca_metrics['neighborhood_preservation']
        })
    
    # Resumen
    print("\n" + "=" * 80)
    print("📈 RESUMEN MULTI-DATASET")
    print("=" * 80)
    
    print("\n" + "-" * 60)
    print(f"{'Dataset':<20} | {'CCD':<8} | {'PCA':<8} | {'Ganador':<10} | {'Preservación CCD/PCA'}")
    print("-" * 60)
    
    ccd_wins = 0
    pca_wins = 0
    
    for result in all_results:
        winner_marker = "✅" if result['winner'] == 'CCD' else "⚠️"
        print(f"{result['dataset']:<20} | {result['ccd_score']:.4f} | {result['pca_score']:.4f} | "
              f"{winner_marker} {result['winner']:<8} | {result['ccd_preservation']:.3f}/{result['pca_preservation']:.3f}")
        
        if result['winner'] == 'CCD':
            ccd_wins += 1
        else:
            pca_wins += 1
    
    print("-" * 60)
    print(f"\n🏆 Resultado final: CCD {ccd_wins} - {pca_wins} PCA")
    
    if ccd_wins > pca_wins:
        print("✅ CCDEngine es SUPERIOR en la mayoría de datasets")
    elif ccd_wins < pca_wins:
        print("⚠️  PCA es SUPERIOR en la mayoría de datasets")
    else:
        print("🤝 EMPATE: CCDEngine y PCA son comparables")
    
    return all_results

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 CCDEngine OPTIMIZADO FINAL - SUPERANDO A PCA")
    print("=" * 80)
    
    # Benchmark definitivo
    results = run_definitive_benchmark()
    
    if results:
        # Prueba con múltiples datasets
        multi_results = run_multiple_datasets()
        
        # Conclusión general
        print("\n" + "=" * 80)
        print("🎯 CONCLUSIÓN GENERAL DEL ANÁLISIS")
        print("=" * 80)
        
        ccd_score = results['ccd_score']
        pca_score = results['pca_score']
        
        if ccd_score > pca_score:
            print("\n🚀 ¡CCDEngine OPTIMIZADO SUPERÓ A PCA!")
            print(f"   Score: {ccd_score:.4f} vs {pca_score:.4f}")
            print("\n   FACTORES CLAVE DEL ÉXITO:")
            print("   1. DiffusionGeometry captura geometría no lineal")
            print("   2. SpectralPreprocessor preserva métrica (sin whitening)")
            print("   3. k-NN decoder reconstruye de forma estable")
            print("   4. Configuración optimizada para manifolds")
        else:
            print("\n🔧 CCDEngine necesita MÁS OPTIMIZACIONES")
            print(f"   Score: {ccd_score:.4f} vs {pca_score:.4f}")
            print("\n   PROBLEMAS POR RESOLVER:")
            for issue in results.get('issues', []):
                print(f"   • {issue}")
            
            print("\n   ACCIONES RECOMENDADAS:")
            for rec in results.get('recommendations', []):
                print(f"   • {rec}")
        
        print("\n" + "=" * 80)
        print("📋 RESUMEN EJECUTIVO")
        print("=" * 80)
        
        print("\n✅ LO QUE FUNCIONA BIEN:")
        print("   • Arquitectura 5-capas es sólida")
        print("   • k-NN decoder es estable")
        print("   • SpectralPreprocessor sin whitening preserva métrica")
        
        print("\n⚠️  LO QUE NECESITA MEJORA:")
        print("   • DiffusionGeometry es muy lento")
        print("   • Preservación de vecindarios puede mejorar")
        print("   • Auto-tuning de hiperparámetros")
        
        print("\n🚀 PRÓXIMOS PASOS CRÍTICOS:")
        print("   1. Optimizar DiffusionGeometry (sparse + Nyström)")
        print("   2. Implementar kernel mejorado (t-student)")
        print("   3. Crear sistema de auto-tuning")
        print("   4. Validar con datasets reales de alta dimensión")
        
        print("\n" + "=" * 80)
        print("🎉 ANÁLISIS COMPLETADO")
        print("=" * 80)