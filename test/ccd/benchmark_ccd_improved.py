"""
benchmark_ccd_improved.py
==========================
Benchmark para validar las correcciones del CCDEngine mejorado.

Compara:
1. CCDEngine original (con ChebyshevShell y MLP decoder)
2. CCDEngine mejorado (con SpectralPreprocessor y k-NN decoder)
3. PCA baseline
4. UMAP baseline

Métricas:
- Error de reconstrucción (MSE)
- Preservación de vecindarios
- Tiempo de ejecución
- Reducción de dimensionalidad efectiva
"""

import numpy as np
import time
import json
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from acf_functor.ccd_engine import CCDEngine


def load_mnist_subset(n_samples=1000, seed=42):
    """Cargar subset de MNIST para benchmark."""
    try:
        from sklearn.datasets import fetch_openml
        print("Cargando MNIST desde OpenML...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32) / 255.0
        
        # Tomar subset aleatorio
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X), n_samples, replace=False)
        return X[indices]
    except Exception as e:
        print(f"Error cargando MNIST: {e}")
        print("Generando datos sintéticos similares a MNIST...")
        return generate_synthetic_mnist(n_samples, 784, seed)


def generate_synthetic_mnist(n_samples, n_features, seed=42):
    """Generar datos sintéticos similares a MNIST."""
    rng = np.random.default_rng(seed)
    
    # Generar dígitos sintéticos (manifolds gaussianos)
    X = np.zeros((n_samples, n_features))
    
    # Crear 10 clusters (como dígitos)
    for i in range(10):
        cluster_size = n_samples // 10
        start_idx = i * cluster_size
        end_idx = start_idx + cluster_size if i < 9 else n_samples
        
        # Centro del cluster
        center = rng.normal(0, 1, n_features)
        
        # Puntos alrededor del centro con estructura de manifold
        for j in range(start_idx, end_idx):
            # Añadir estructura no-lineal
            t = j / cluster_size * 2 * np.pi
            manifold_component = np.sin(t) * center + np.cos(t) * rng.normal(0, 0.5, n_features)
            X[j] = center + 0.3 * manifold_component + 0.1 * rng.normal(0, 1, n_features)
    
    # Normalizar
    X = (X - X.mean()) / (X.std() + 1e-8)
    return np.clip(X, 0, 1)


def compute_metrics(X, X_reconstructed, Z, method_name):
    """Calcular métricas de calidad."""
    # Error de reconstrucción
    mse = np.mean((X_reconstructed - X) ** 2)
    relative_error = np.linalg.norm(X_reconstructed - X, 'fro') / np.linalg.norm(X, 'fro')
    
    # Preservación de vecindarios
    nn_original = NearestNeighbors(n_neighbors=10).fit(X)
    indices_original = nn_original.kneighbors(X)[1][:, 1:]
    
    nn_transformed = NearestNeighbors(n_neighbors=10).fit(Z)
    indices_transformed = nn_transformed.kneighbors(Z)[1][:, 1:]
    
    preservation_scores = []
    for i in range(X.shape[0]):
        intersection = len(set(indices_original[i]).intersection(set(indices_transformed[i])))
        preservation_scores.append(intersection / 9)
    
    avg_preservation = np.mean(preservation_scores)
    
    # Preservación de métrica (correlación de distancias)
    if X.shape[0] > 500:  # Para no sobrecargar memoria
        sample_idx = np.random.choice(X.shape[0], 500, replace=False)
        X_sample = X[sample_idx]
        Z_sample = Z[sample_idx]
    else:
        X_sample = X
        Z_sample = Z
    
    dist_original = pdist(X_sample, 'euclidean')
    dist_transformed = pdist(Z_sample, 'euclidean')
    rho, p_value = spearmanr(dist_original, dist_transformed)
    
    return {
        'method': method_name,
        'mse': float(mse),
        'relative_error': float(relative_error),
        'neighborhood_preservation': float(avg_preservation),
        'distance_correlation': float(rho),
        'distance_correlation_pvalue': float(p_value),
        'input_dimension': X.shape[1],
        'output_dimension': Z.shape[1],
        'reduction_ratio': X.shape[1] / Z.shape[1]
    }


def benchmark_pca(X, target_dim=50):
    """Benchmark de PCA."""
    print("  Ejecutando PCA...")
    start_time = time.time()
    
    pca = PCA(n_components=target_dim)
    Z = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(Z)
    
    fit_time = time.time() - start_time
    
    metrics = compute_metrics(X, X_reconstructed, Z, 'PCA')
    metrics['fit_time'] = fit_time
    metrics['explained_variance'] = float(pca.explained_variance_ratio_.sum())
    
    return metrics


def benchmark_ccd_improved(X, target_dim=50):
    """Benchmark del CCDEngine mejorado."""
    print("  Ejecutando CCDEngine mejorado...")
    start_time = time.time()
    
    engine = CCDEngine(
        d_threshold=10,
        n_preprocess_components=min(100, X.shape[1]),
        n_diffusion_components=target_dim,
        n_diffusion_neighbors=15,
        n_oscillator_groups=None,  # auto
        n_entropy_neighbors=10,
        n_langevin_steps=0,  # Desactivar purificación para benchmark
        fit_decoder=True,
        decoder_n_neighbors=5
    )
    
    engine.fit(X)
    Z = engine.transform(X)
    X_reconstructed = engine.inverse_transform(Z)
    
    fit_time = time.time() - start_time
    
    metrics = compute_metrics(X, X_reconstructed, Z, 'CCDEngine Mejorado')
    metrics['fit_time'] = fit_time
    metrics['effective_dimension'] = engine._k_effective
    
    # Obtener certificado
    cert = engine.certificate()
    metrics['certificate'] = {
        'curse_escaped': cert.curse_escaped,
        'reduction_ratio': cert.reduction_ratio,
        'explained_variance_top_k': cert.explained_variance_ratio_top_k,
        'cod_reduction_log10': cert.cod_reduction_log10
    }
    
    return metrics


def run_benchmark():
    """Ejecutar benchmark completo."""
    print("=" * 70)
    print("BENCHMARK CCDEngine MEJORADO")
    print("=" * 70)
    
    # Configuración
    n_samples = 1000
    target_dim = 50
    
    print(f"\nConfiguración:")
    print(f"  Muestras: {n_samples}")
    print(f"  Dimensión objetivo: {target_dim}")
    
    # Cargar datos
    print(f"\nCargando datos...")
    X = load_mnist_subset(n_samples)
    print(f"  Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
    
    # Ejecutar benchmarks
    results = []
    
    # 1. PCA
    print(f"\n1. Benchmark PCA:")
    results.append(benchmark_pca(X, target_dim))
    
    # 2. CCDEngine mejorado
    print(f"\n2. Benchmark CCDEngine mejorado:")
    results.append(benchmark_ccd_improved(X, target_dim))
    
    # Análisis comparativo
    print(f"\n" + "=" * 70)
    print("RESULTADOS COMPARATIVOS")
    print("=" * 70)
    
    for metrics in results:
        print(f"\n{metrics['method']}:")
        print(f"  Tiempo: {metrics['fit_time']:.3f}s")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  Error relativo: {metrics['relative_error']:.6f}")
        print(f"  Preservación vecindarios: {metrics['neighborhood_preservation']:.4f}")
        print(f"  Correlación distancias: {metrics['distance_correlation']:.4f}")
        print(f"  Reducción: {metrics['input_dimension']} → {metrics['output_dimension']} ({metrics['reduction_ratio']:.1f}×)")
        
        if 'effective_dimension' in metrics:
            print(f"  Dimensión efectiva: {metrics['effective_dimension']}")
        if 'explained_variance' in metrics:
            print(f"  Varianza explicada: {metrics['explained_variance']:.4f}")
    
    # Comparación directa
    print(f"\n" + "=" * 70)
    print("COMPARACIÓN CCDEngine vs PCA")
    print("=" * 70)
    
    pca_metrics = results[0]
    ccd_metrics = results[1]
    
    print(f"\nMejora CCDEngine vs PCA:")
    print(f"  MSE: {pca_metrics['mse'] / ccd_metrics['mse']:.2f}× mejor")
    print(f"  Tiempo: {pca_metrics['fit_time'] / ccd_metrics['fit_time']:.2f}× más lento")
    print(f"  Preservación vecindarios: {ccd_metrics['neighborhood_preservation'] / pca_metrics['neighborhood_preservation']:.2f}×")
    print(f"  Correlación distancias: {ccd_metrics['distance_correlation'] / pca_metrics['distance_correlation']:.2f}×")
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_ccd_improved_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'n_samples': n_samples,
                'target_dim': target_dim,
                'input_dimension': X.shape[1]
            },
            'results': results,
            'comparison': {
                'mse_improvement': pca_metrics['mse'] / ccd_metrics['mse'],
                'time_ratio': ccd_metrics['fit_time'] / pca_metrics['fit_time'],
                'preservation_ratio': ccd_metrics['neighborhood_preservation'] / pca_metrics['neighborhood_preservation'],
                'correlation_ratio': ccd_metrics['distance_correlation'] / pca_metrics['distance_correlation']
            }
        }, f, indent=2)
    
    print(f"\nResultados guardados en: {output_file}")
    
    # Conclusión
    print(f"\n" + "=" * 70)
    print("CONCLUSIÓN")
    print("=" * 70)
    
    if ccd_metrics['mse'] < pca_metrics['mse']:
        print("✓ CCDEngine mejorado tiene MEJOR reconstrucción que PCA")
    else:
        print("✗ CCDEngine mejorado tiene PEOR reconstrucción que PCA")
    
    if ccd_metrics['neighborhood_preservation'] > pca_metrics['neighborhood_preservation']:
        print("✓ CCDEngine mejorado preserva MEJOR los vecindarios")
    else:
        print("✗ CCDEngine mejorado preserva PEOR los vecindarios")
    
    if ccd_metrics['certificate']['curse_escaped']:
        print("✓ CCDEngine mejorado ESCAPA de la maldición de dimensionalidad")
    else:
        print("✗ CCDEngine mejorado NO escapa de la maldición de dimensionalidad")
    
    return results


if __name__ == "__main__":
    try:
        results = run_benchmark()
        print(f"\n¡Benchmark completado exitosamente!")
    except Exception as e:
        print(f"\nError en benchmark: {e}")
        import traceback
        traceback.print_exc()
        exit(1)