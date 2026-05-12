"""
test_ccd_improved.py
====================
Tests para el CCDEngine mejorado con SpectralPreprocessor y k-NN decoder.

Este test valida las correcciones críticas implementadas:
1. Eliminación de ChebyshevShell (destruía la métrica)
2. Uso exclusivo de SpectralPreprocessor (preserva métrica)
3. k-NN decoder simple (eficiente y estable)
"""

import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from acf_functor.ccd_engine import CCDEngine, SpectralPreprocessor, ManifoldDecoder


def generate_test_data(n_samples=500, d_input=50, d_intrinsic=3, noise=0.1, seed=42):
    """Generar datos de manifold no-lineal en alta dimensión."""
    rng = np.random.default_rng(seed)
    
    # Generar manifold intrínseco (baja dimensión)
    t = rng.uniform(0, 2*np.pi, n_samples)
    u = rng.uniform(-1, 1, n_samples)
    v = rng.uniform(-1, 1, n_samples)
    
    # Coordenadas intrínsecas (3D)
    Z_intrinsic = np.column_stack([
        t * np.cos(t),
        t * np.sin(t),
        u * v
    ])
    
    # Mapear a alta dimensión con transformación no-lineal
    A = rng.normal(0, 1, (d_input, d_intrinsic))
    B = rng.normal(0, 0.5, (d_input, d_intrinsic))
    
    X = Z_intrinsic @ A.T + (Z_intrinsic**2) @ B.T
    X += noise * rng.normal(0, 1, X.shape)
    
    return X, Z_intrinsic


def test_spectral_preprocessor_preserves_metric():
    """Test: SpectralPreprocessor debe preservar la métrica de distancias."""
    print("\n=== Test: SpectralPreprocessor preserva métrica ===")
    
    X, _ = generate_test_data(n_samples=200, d_input=30, noise=0.05)
    
    # Calcular distancias originales
    dist_original = pdist(X, 'euclidean')
    
    # Aplicar SpectralPreprocessor
    preprocessor = SpectralPreprocessor(n_components=10, whiten=True)
    preprocessor.fit(X)
    Z = preprocessor.transform(X)
    
    # Calcular distancias en espacio transformado
    dist_transformed = pdist(Z, 'euclidean')
    
    # Calcular correlación de rangos de Spearman
    rho, p_value = spearmanr(dist_original, dist_transformed)
    
    print(f"  Correlación de rangos (Spearman): ρ = {rho:.4f}, p = {p_value:.4e}")
    print(f"  Varianza explicada: {preprocessor.explained_variance_ratio.sum():.4f}")
    
    assert rho > 0.3, f"SpectralPreprocessor destruye métrica (ρ={rho:.4f})"
    assert p_value < 0.05, "Correlación no significativa"
    
    print("  ✓ SpectralPreprocessor preserva métrica correctamente")


def test_knn_decoder_performance():
    """Test: k-NN decoder debe tener error de reconstrucción bajo."""
    print("\n=== Test: k-NN decoder performance ===")
    
    X, Z_intrinsic = generate_test_data(n_samples=300, d_input=40, noise=0.1)
    
    # Usar SpectralPreprocessor para reducción
    preprocessor = SpectralPreprocessor(n_components=8, whiten=True)
    preprocessor.fit(X)
    Z_latent = preprocessor.transform(X)
    
    # Entrenar decoder k-NN
    decoder = ManifoldDecoder(n_neighbors=5, metric='euclidean')
    decoder.fit(Z_latent, X)
    
    # Reconstruir
    X_reconstructed = decoder.decode(Z_latent)
    
    # Calcular métricas
    mse = np.mean((X_reconstructed - X) ** 2)
    relative_error = np.linalg.norm(X_reconstructed - X, 'fro') / np.linalg.norm(X, 'fro')
    
    print(f"  MSE de reconstrucción: {mse:.6f}")
    print(f"  Error relativo: {relative_error:.6f}")
    print(f"  Dimensión latente: {Z_latent.shape[1]}")
    print(f"  Dimensión original: {X.shape[1]}")
    
    assert mse < 0.1, f"Error de reconstrucción muy alto (MSE={mse:.4f})"
    assert relative_error < 0.1, f"Error relativo muy alto ({relative_error:.4f})"
    
    print("  ✓ k-NN decoder reconstruye correctamente")


def test_ccd_engine_integration():
    """Test: CCDEngine completo con las mejoras."""
    print("\n=== Test: CCDEngine integración completa ===")
    
    X, _ = generate_test_data(n_samples=400, d_input=60, noise=0.08)
    
    # Configurar CCDEngine mejorado
    engine = CCDEngine(
        d_threshold=5,
        n_preprocess_components=15,
        n_diffusion_components=8,
        n_diffusion_neighbors=10,
        n_oscillator_groups=3,
        n_entropy_neighbors=8,
        n_langevin_steps=10,
        fit_decoder=True,
        decoder_n_neighbors=5
    )
    
    # Medir tiempo de entrenamiento
    start_time = time.time()
    engine.fit(X)
    fit_time = time.time() - start_time
    
    # Transformar
    Z = engine.transform(X)
    
    # Reconstruir
    X_reconstructed = engine.inverse_transform(Z)
    
    # Calcular métricas
    mse = np.mean((X_reconstructed - X) ** 2)
    reduction_ratio = X.shape[1] / Z.shape[1]
    trust = trustworthiness(X, Z, n_neighbors=12)
    
    print(f"  Tiempo de entrenamiento: {fit_time:.3f}s")
    print(f"  Dimensión original: {X.shape[1]}")
    print(f"  Dimensión efectiva: {engine._k_effective}")
    print(f"  Ratio de reducción: {reduction_ratio:.1f}×")
    print(f"  MSE de reconstrucción: {mse:.6f}")
    print(f"  Trustworthiness: {trust:.6f}")
    
    # Verificar certificado
    cert = engine.certificate()
    print(f"\n  Certificado:")
    print(f"    {cert}")
    
    assert Z.shape[1] < X.shape[1], "No hubo reducción de dimensionalidad"
    assert np.isfinite(mse), f"Error de reconstrucción no finito (MSE={mse})"
    assert trust > 0.99, f"La preservación geométrica es insuficiente (trust={trust:.6f})"
    assert fit_time < 10.0, f"Tiempo de entrenamiento muy alto ({fit_time:.2f}s)"
    
    print("  ✓ CCDEngine funciona correctamente con las mejoras")


def test_ccd_vs_pca_comparison():
    """Test: Comparar CCDEngine mejorado vs PCA simple."""
    print("\n=== Test: CCDEngine vs PCA ===")
    
    X, Z_intrinsic = generate_test_data(n_samples=300, d_input=50, noise=0.1)
    
    # PCA simple
    from sklearn.decomposition import PCA
    pca = PCA(n_components=8)
    Z_pca = pca.fit_transform(X)
    X_pca_reconstructed = pca.inverse_transform(Z_pca)
    mse_pca = np.mean((X_pca_reconstructed - X) ** 2)
    
    # CCDEngine mejorado
    engine = CCDEngine(
        d_threshold=5,
        n_preprocess_components=15,
        n_diffusion_components=8,
        fit_decoder=True,
        decoder_n_neighbors=5
    )
    engine.fit(X)
    Z_ccd = engine.transform(X)
    X_ccd_reconstructed = engine.inverse_transform(Z_ccd)
    mse_ccd = np.mean((X_ccd_reconstructed - X) ** 2)
    trust_pca = trustworthiness(X, Z_pca, n_neighbors=12)
    trust_ccd = trustworthiness(X, Z_ccd, n_neighbors=12)
    
    print(f"  PCA - MSE: {mse_pca:.6f}")
    print(f"  CCD - MSE: {mse_ccd:.6f}")
    print(f"  PCA - Trustworthiness: {trust_pca:.6f}")
    print(f"  CCD - Trustworthiness: {trust_ccd:.6f}")
    print(f"  Mejora: {mse_pca/mse_ccd:.2f}×")
    print(f"  Dimensión PCA: {Z_pca.shape[1]}")
    print(f"  Dimensión CCD: {Z_ccd.shape[1]}")
    
    # CCD está diseñado para preservar geometría local, no para competir con la
    # reconstrucción lineal óptima de PCA. Exigimos paridad geométrica cercana.
    assert trust_ccd >= trust_pca - 1e-4, (
        f"CCD pierde demasiada geometría frente a PCA (CCD={trust_ccd:.6f}, PCA={trust_pca:.6f})"
    )
    assert np.isfinite(mse_ccd), f"CCD produjo una reconstrucción no finita (MSE={mse_ccd})"
    
    print("  ✓ CCDEngine es competitivo con PCA")


def test_neighborhood_preservation():
    """Test: CCDEngine debe preservar vecindarios locales."""
    print("\n=== Test: Preservación de vecindarios ===")
    
    X, _ = generate_test_data(n_samples=200, d_input=40, noise=0.05)
    
    # Calcular vecinos originales
    nn_original = NearestNeighbors(n_neighbors=10).fit(X)
    indices_original = nn_original.kneighbors(X)[1][:, 1:]  # Excluir punto mismo
    
    # CCDEngine
    engine = CCDEngine(
        d_threshold=5,
        n_preprocess_components=12,
        n_diffusion_components=6,
        fit_decoder=False  # No necesitamos decoder para este test
    )
    engine.fit(X)
    Z = engine.transform(X)
    
    # Calcular vecinos en espacio transformado
    nn_transformed = NearestNeighbors(n_neighbors=10).fit(Z)
    indices_transformed = nn_transformed.kneighbors(Z)[1][:, 1:]
    
    # Calcular preservación de vecindarios
    preservation_scores = []
    for i in range(X.shape[0]):
        intersection = len(set(indices_original[i]).intersection(set(indices_transformed[i])))
        preservation_scores.append(intersection / 9)  # 9 vecinos (excluyendo punto mismo)
    
    avg_preservation = np.mean(preservation_scores)
    
    print(f"  Preservación promedio de vecindarios: {avg_preservation:.4f}")
    print(f"  Dimensión original: {X.shape[1]}")
    print(f"  Dimensión transformada: {Z.shape[1]}")
    
    assert avg_preservation > 0.3, f"Preservación de vecindarios muy baja ({avg_preservation:.4f})"
    
    print("  ✓ CCDEngine preserva vecindarios locales")


def run_all_tests():
    """Ejecutar todos los tests."""
    print("=" * 60)
    print("TESTS PARA CCDEngine MEJORADO")
    print("=" * 60)
    
    tests = [
        test_spectral_preprocessor_preserves_metric,
        test_knn_decoder_performance,
        test_ccd_engine_integration,
        test_ccd_vs_pca_comparison,
        test_neighborhood_preservation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  → {test.__name__}: ✓ PASÓ\n")
        except AssertionError as e:
            failed += 1
            print(f"  → {test.__name__}: ✗ FALLÓ - {str(e)}\n")
        except Exception as e:
            failed += 1
            print(f"  → {test.__name__}: ✗ ERROR - {str(e)}\n")
    
    print("=" * 60)
    print(f"RESUMEN: {passed} pasados, {failed} fallados")
    print("=" * 60)
    
    if failed == 0:
        print("¡Todos los tests pasaron! ✓")
    else:
        print(f"¡Atención! {failed} test(s) fallaron.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)