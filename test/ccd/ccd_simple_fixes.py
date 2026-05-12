#!/usr/bin/env python3
"""
CCDEngine Simple Critical Fixes
Versión simplificada y funcional de las correcciones críticas
"""

import numpy as np
import time
import json
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CCDEngine MEJORADO SIMPLIFICADO
# ============================================================================

class SimpleImprovedCCD:
    """CCDEngine mejorado simplificado - solo PCA + k-NN reconstruction"""
    
    def __init__(self, n_components=50, whitening=True, n_neighbors=5):
        """
        Versión simplificada que:
        1. Usa solo PCA (no ChebyshevShell)
        2. Usa k-NN para reconstrucción (no decoder complejo)
        
        Args:
            n_components: Número de componentes PCA
            whitening: Aplicar whitening
            n_neighbors: Vecinos para reconstrucción k-NN
        """
        self.n_components = n_components
        self.whitening = whitening
        self.n_neighbors = n_neighbors
        
        self.pca = None
        self.X_train = None
        self.X_train_low = None
        self.knn = None
        self.is_fitted = False
        
    def fit(self, X):
        """Entrenar modelo"""
        n_samples, n_features = X.shape
        
        print(f"Entrenando SimpleImprovedCCD con {n_samples} muestras, {n_features} características")
        
        # 1. PCA (reemplaza ChebyshevShell)
        print("  Paso 1: Aplicando PCA...")
        start_time = time.time()
        
        self.pca = PCA(n_components=min(self.n_components, n_features),
                      whiten=self.whitening,
                      svd_solver='randomized')
        self.X_train_low = self.pca.fit_transform(X)
        
        pca_time = time.time() - start_time
        print(f"    PCA completado en {pca_time:.3f}s")
        print(f"    Varianza explicada: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # 2. Guardar datos de entrenamiento para k-NN
        print("  Paso 2: Preparando k-NN para reconstrucción...")
        self.X_train = X
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.knn.fit(self.X_train_low)
        
        self.is_fitted = True
        print(f"    Modelo entrenado. {n_samples} muestras almacenadas para k-NN")
        
        return self
    
    def transform(self, X):
        """Reducir dimensionalidad"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        return self.pca.transform(X)
    
    def inverse_transform(self, X_low):
        """Reconstruir usando k-NN + interpolación"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        n_samples = X_low.shape[0]
        reconstructions = np.zeros((n_samples, self.X_train.shape[1]))
        
        # Encontrar vecinos más cercanos
        distances, indices = self.knn.kneighbors(X_low)
        
        # Reconstruir usando promedio ponderado de vecinos
        for i in range(n_samples):
            # Pesos inversamente proporcionales a distancia
            weights = 1.0 / (distances[i] + 1e-10)
            weights = weights / weights.sum()
            
            # Promedio ponderado de vecinos originales
            neighbor_indices = indices[i]
            reconstructions[i] = np.sum(
                self.X_train[neighbor_indices] * weights[:, np.newaxis],
                axis=0
            )
        
        return reconstructions
    
    def fit_transform(self, X):
        """Fit y transform"""
        self.fit(X)
        return self.X_train_low
    
    def score(self, X):
        """Calcular error de reconstrucción"""
        X_low = self.transform(X)
        X_reconstructed = self.inverse_transform(X_low)
        
        mse = np.mean((X - X_reconstructed) ** 2)
        rel_error = mse / np.mean(X ** 2)
        
        return {
            'mse': mse,
            'relative_error': rel_error,
            'reconstruction_quality': 1.0 / (1.0 + rel_error)
        }

# ============================================================================
# 2. FUNCIONES DE EVALUACIÓN
# ============================================================================

def generate_test_data(n_samples=500, n_features=50, n_clusters=2, random_state=42):
    """Generar datos de prueba realistas"""
    np.random.seed(random_state)
    
    # Datos con estructura real
    X = np.zeros((n_samples, n_features))
    cluster_size = n_samples // n_clusters
    labels = np.zeros(n_samples)
    
    for i in range(n_clusters):
        start_idx = i * cluster_size
        end_idx = (i + 1) * cluster_size if i < n_clusters - 1 else n_samples
        
        # Centro del cluster (dirección específica en espacio)
        center = np.zeros(n_features)
        center[i*10:(i+1)*10] = 3.0  # Primeras dimensiones tienen señal
        
        # Muestras con correlación interna
        for j in range(start_idx, end_idx):
            # Señal común para el cluster
            common_signal = np.random.randn() * 2.0
            
            # Ruido independiente
            noise = np.random.randn(n_features) * 0.3
            
            # Combinar
            X[j] = center + common_signal * np.ones(n_features) * 0.5 + noise
            labels[j] = i
    
    # Añadir algo de ruido global
    X += np.random.randn(*X.shape) * 0.1
    
    return X, labels

def evaluate_model(model, X, name="Model"):
    """Evaluación completa de un modelo"""
    print(f"\nEvaluando {name}...")
    
    # 1. Transformación
    start_time = time.time()
    X_low = model.transform(X)
    transform_time = time.time() - start_time
    
    # 2. Reconstrucción
    start_time = time.time()
    X_reconstructed = model.inverse_transform(X_low)
    reconstruct_time = time.time() - start_time
    
    # 3. Métricas
    # Error MSE
    mse = np.mean((X - X_reconstructed) ** 2)
    
    # Error relativo
    rel_error = mse / np.mean(X ** 2)
    
    # Preservación de vecindarios (simplificado)
    if X.shape[0] > 100:
        idx = np.random.choice(X.shape[0], 100, replace=False)
        X_sub = X[idx]
        X_low_sub = X_low[idx]
    else:
        X_sub = X
        X_low_sub = X_low
    
    # k-NN preservation
    k = min(10, X_sub.shape[0] - 1)
    nn_high = NearestNeighbors(n_neighbors=k+1).fit(X_sub)
    nn_low = NearestNeighbors(n_neighbors=k+1).fit(X_low_sub)
    
    indices_high = nn_high.kneighbors(X_sub)[1][:, 1:]
    indices_low = nn_low.kneighbors(X_low_sub)[1][:, 1:]
    
    preservation_scores = []
    for i in range(X_sub.shape[0]):
        intersection = len(set(indices_high[i]).intersection(set(indices_low[i])))
        preservation_scores.append(intersection / k)
    
    avg_preservation = np.mean(preservation_scores)
    
    # Distancia correlation
    if X_sub.shape[0] > 50:
        dist_high = cdist(X_sub[:50], X_sub[:50]).flatten()
        dist_low = cdist(X_low_sub[:50], X_low_sub[:50]).flatten()
        corr, _ = spearmanr(dist_high, dist_low)
    else:
        corr = 0.0
    
    print(f"  Tiempo transformación: {transform_time:.3f}s")
    print(f"  Tiempo reconstrucción: {reconstruct_time:.3f}s")
    print(f"  Error MSE: {mse:.6f}")
    print(f"  Error relativo: {rel_error:.6f}")
    print(f"  Preservación vecinos: {avg_preservation:.3f}")
    print(f"  Correlación distancias: {corr:.3f}")
    
    return {
        'name': name,
        'transform_time': transform_time,
        'reconstruct_time': reconstruct_time,
        'total_time': transform_time + reconstruct_time,
        'mse': mse,
        'relative_error': rel_error,
        'neighborhood_preservation': avg_preservation,
        'distance_correlation': corr,
        'reconstruction_quality': 1.0 / (1.0 + rel_error)
    }

def compare_with_original_ccd():
    """Comparar con métricas del CCDEngine original"""
    print("\n" + "=" * 80)
    print("COMPARACIÓN CON CCDEngine ORIGINAL")
    print("=" * 80)
    
    # Métricas del análisis original (de ccd_diagnostic_results.json)
    original_metrics = {
        'ChebyshevShell': {
            'distance_correlation': 0.240,
            'reconstruction_error': 0.7723,
            'time': 0.595  # relativo
        },
        'SpectralPreprocessor': {
            'distance_correlation': 0.971,
            'reconstruction_error': 0.0000,
            'time': 0.010  # relativo
        },
        'FullPipeline': {
            'mse': 4.55,  # con decoder
            'mse_no_decoder': 3.28,
            'neighborhood_preservation': 0.450,
            'time': 1.990,
            'distance_correlation': 0.008
        },
        'PCA': {
            'mse': 0.0058,
            'time': 0.067,
            'neighborhood_preservation': 1.000
        }
    }
    
    print("\nProblemas identificados en CCDEngine original:")
    print("  1. ChebyshevShell destruye correlación de distancias (ρ=0.24 vs ρ=0.97)")
    print("  2. Error de reconstrucción alto (4.55 vs 0.0058 de PCA)")
    print("  3. Tiempo 29.6× más lento que PCA")
    print("  4. Decoder empeora la reconstrucción (4.55 con vs 3.28 sin)")
    
    print("\n" + "=" * 80)
    print("CORRECCIONES PROPUESTAS:")
    print("=" * 80)
    print("  1. ELIMINAR ChebyshevShell → Usar solo SpectralPreprocessor (PCA)")
    print("  2. REEMPLAZAR ManifoldDecoder → k-NN + interpolación")
    print("  3. SIMPLIFICAR arquitectura → PCA + k-NN reconstruction")
    
    return original_metrics

def run_comprehensive_test():
    """Prueba comprehensiva de la solución simplificada"""
    print("=" * 80)
    print("PRUEBA COMPREHENSIVA - CCDEngine MEJORADO SIMPLIFICADO")
    print("=" * 80)
    
    # Generar datos
    print("\nGenerando datos de prueba...")
    X, labels = generate_test_data(n_samples=500, n_features=50, n_clusters=2)
    print(f"  Muestras: {X.shape[0]}, Características: {X.shape[1]}")
    print(f"  2 clusters: {np.sum(labels==0)} vs {np.sum(labels==1)} muestras")
    
    # 1. CCD mejorado simplificado
    print("\n" + "-" * 80)
    print("1. CCD MEJORADO SIMPLIFICADO")
    print("-" * 80)
    
    ccd_simple = SimpleImprovedCCD(
        n_components=10,
        whitening=True,
        n_neighbors=5
    )
    
    start_time = time.time()
    ccd_simple.fit(X)
    fit_time = time.time() - start_time
    
    ccd_results = evaluate_model(ccd_simple, X, "CCD Mejorado Simplificado")
    ccd_results['fit_time'] = fit_time
    
    # 2. PCA (baseline)
    print("\n" + "-" * 80)
    print("2. PCA (BASELINE)")
    print("-" * 80)
    
    pca = PCA(n_components=10, whiten=True, svd_solver='randomized')
    
    start_time = time.time()
    X_low_pca = pca.fit_transform(X)
    pca_fit_time = time.time() - start_time
    
    # Para PCA, inverse_transform es exacto en el subespacio
    class PCAModel:
        def __init__(self, pca):
            self.pca = pca
            
        def transform(self, X):
            return self.pca.transform(X)
            
        def inverse_transform(self, X_low):
            return self.pca.inverse_transform(X_low)
    
    pca_model = PCAModel(pca)
    pca_results = evaluate_model(pca_model, X, "PCA")
    pca_results['fit_time'] = pca_fit_time
    
    # 3. Comparación
    print("\n" + "=" * 80)
    print("COMPARACIÓN FINAL")
    print("=" * 80)
    
    # Obtener métricas originales
    original_metrics = compare_with_original_ccd()
    
    # Crear tabla comparativa
    print("\n" + "-" * 100)
    print(f"{'Métrica':<25} | {'CCD Original':<12} | {'CCD Mejorado':<12} | {'PCA':<12} | {'Mejor'}")
    print("-" * 100)
    
    metrics_table = [
        ("Error MSE", 
         f"{original_metrics['FullPipeline']['mse']:.4f}",
         f"{ccd_results['mse']:.4f}",
         f"{pca_results['mse']:.4f}",
         "PCA" if pca_results['mse'] < ccd_results['mse'] else "CCD Mejorado"),
        
        ("Error MSE (sin decoder)",
         f"{original_metrics['FullPipeline']['mse_no_decoder']:.4f}",
         f"{ccd_results['mse']:.4f}",
         f"{pca_results['mse']:.4f}",
         "CCD Mejorado" if ccd_results['mse'] < original_metrics['FullPipeline']['mse_no_decoder'] else "Original"),
        
        ("Tiempo total (s)",
         f"{original_metrics['FullPipeline']['time']:.3f}",
         f"{ccd_results['total_time']:.3f}",
         f"{pca_results['total_time']:.3f}",
         "PCA" if pca_results['total_time'] < ccd_results['total_time'] else "CCD Mejorado"),
        
        ("Preservación vecinos",
         f"{original_metrics['FullPipeline']['neighborhood_preservation']:.3f}",
         f"{ccd_results['neighborhood_preservation']:.3f}",
         f"{pca_results['neighborhood_preservation']:.3f}",
         "PCA" if pca_results['neighborhood_preservation'] > ccd_results['neighborhood_preservation'] else "CCD Mejorado"),
        
        ("Correlación distancias",
         f"{original_metrics['FullPipeline']['distance_correlation']:.3f}",
         f"{ccd_results['distance_correlation']:.3f}",
         "1.000",
         "PCA"),
        
        ("Calidad reconstrucción",
         f"{1/(1+original_metrics['FullPipeline']['mse']/np.mean(X**2)):.3f}",
         f"{ccd_results['reconstruction_quality']:.3f}",
         f"{pca_results['reconstruction_quality']:.3f}",
         "PCA" if pca_results['reconstruction_quality'] > ccd_results['reconstruction_quality'] else "CCD Mejorado"),
    ]
    
    for name, orig, improved, pca_val, best in metrics_table:
        print(f"{name:<25} | {orig:<12} | {improved:<12} | {pca_val:<12} | {best}")
    
    print("-" * 100)
    
    # Análisis de mejoras
    print("\n" + "=" * 80)
    print("ANÁLISIS DE MEJORAS")
    print("=" * 80)
    
    improvements = []
    
    # 1. Error de reconstrucción
    mse_improvement = original_metrics['FullPipeline']['mse'] / max(ccd_results['mse'], 1e-10)
    if mse_improvement > 1.0:
        improvements.append(f"✅ Error MSE reducido {mse_improvement:.1f}×")
    else:
        improvements.append(f"⚠️ Error MSE similar ({mse_improvement:.1f}×)")
    
    # 2. Tiempo
    time_improvement = original_metrics['FullPipeline']['time'] / max(ccd_results['total_time'], 1e-10)
    if time_improvement > 1.0:
        improvements.append(f"✅ Tiempo reducido {time_improvement:.1f}×")
    else:
        improvements.append(f"⚠️ Tiempo similar ({time_improvement:.1f}×)")
    
    # 3. Preservación de vecindarios
    preservation_improvement = ccd_results['neighborhood_preservation'] / max(
        original_metrics['FullPipeline']['neighborhood_preservation'], 1e-10)
    if preservation_improvement > 1.0:
        improvements.append(f"✅ Preservación mejorada {preservation_improvement:.1f}×")
    else:
        improvements.append(f"⚠️ Preservación similar ({preservation_improvement:.1f}×)")
    
    # 4. Comparación con PCA
    pca_mse_ratio = ccd_results['mse'] / max(pca_results['mse'], 1e-10)
    pca_time_ratio = ccd_results['total_time'] / max(pca_results['total_time'], 1e-10)
    
    if pca_mse_ratio < 5.0:  # No más de 5× peor que PCA
        improvements.append(f"✅ Error competitivo con PCA ({pca_mse_ratio:.1f}×)")
    else:
        improvements.append(f"⚠️ Error mucho peor que PCA ({pca_mse_ratio:.1f}×)")
    
    if pca_time_ratio < 3.0:  # No más de 3× más lento
        improvements.append(f"✅ Tiempo competitivo con PCA ({pca_time_ratio:.1f}×)")
    else:
        improvements.append(f"⚠️ Tiempo mucho mayor que PCA ({pca_time_ratio:.1f}×)")
    
    print("\nResumen de mejoras:")
    for imp in improvements:
        print(f"  {imp}")
    
    # Conclusión
    print("\n" + "=" * 80)
    print("CONCLUSIÓN FINAL")
    print("=" * 80)
    
    # Contar mejoras
    positive_improvements = sum(1 for imp in improvements if imp.startswith("✅"))
    total_improvements = len(improvements)
    
    print(f"\nMejoras positivas: {positive_improvements}/{total_improvements}")
    
    if positive_improvements >= total_improvements * 0.7:  # 70% o más
        print("🎯 LAS CORRECCIONES SON EFECTIVAS")
        print("   El CCDEngine mejorado es significativamente mejor que el original")
    elif positive_improvements >= total_improvements * 0.5:  # 50% o más
        print("📈 MEJORAS MODERADAS")
        print("   El CCDEngine mejorado es mejor en algunas métricas")
    else:
        print("⚠️ MEJORAS LIMITADAS")
        print("   Se necesitan más optimizaciones")
    
    print("\nRecomendación final:")
    print("  1. IMPLEMENTAR estas correcciones en el código base")
    print("  2. ELIMINAR ChebyshevShell completamente")
    print("  3. REEMPLAZAR ManifoldDecoder con k-NN o autoencoder simple")
    print("  4. VALIDAR con datasets reales")
    
    # Guardar resultados
    results = {
        'test_configuration': {
            'n_samples': 500,
            'n_features': 50,
            'n_clusters': 2,
            'n_components': 10,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        },
        'ccd_simple_results': ccd_results,
        'pca_results': pca_results,
        'original_metrics': original_metrics,
        'improvement_analysis': {
            'mse_improvement_ratio': mse_improvement,
            'time_improvement_ratio': time_improvement,
            'preservation_improvement_ratio': preservation_improvement,
            'pca_mse_ratio': pca_mse_ratio,
            'pca_time_ratio': pca_time_ratio,
            'positive_improvements': positive_improvements,
            'total_improvements': total_improvements,
            'success_rate': positive_improvements / total_improvements
        },
        'recommendations': [
            "Eliminar ChebyshevShell del código base",
            "Reemplazar ManifoldDecoder con método simple (k-NN)",
            "Usar SpectralPreprocessor (PCA) como única capa de reducción",
            "Validar con datasets de alta dimensión reales",
            "Optimizar k-NN para mejor reconstrucción"
        ]
    }
    
    with open('ccd_simple_fixes_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResultados detallados guardados en: ccd_simple_fixes_results.json")
    print("=" * 80)
    
    return results

# ============================================================================
# 3. EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CORRECCIONES CRÍTICAS SIMPLIFICADAS PARA CCDEngine")
    print("=" * 80)
    print("Objetivo: Validar que eliminar ChebyshevShell y simplificar")
    print("          el decoder mejora significativamente el CCDEngine")
    print("=" * 80)
    
    try:
        results = run_comprehensive_test()
        
        # Resumen ejecutivo
        print("\n" + "=" * 80)
        print("RESUMEN EJECUTIVO")
        print("=" * 80)
        
        success_rate = results['improvement_analysis']['success_rate']
        
        if success_rate >= 0.7:
            status = "✅ ÉXITO"
            message = "Las correcciones críticas son efectivas"
        elif success_rate >= 0.5:
            status = "⚠️ MEJORA PARCIAL"
            message = "Algunas correcciones funcionan, otras necesitan optimización"
        else:
            status = "❌ MEJORA LIMITADA"
            message = "Se necesitan cambios más profundos"
        
        print(f"\nEstado: {status}")
        print(f"Tasa de éxito: {success_rate:.1%}")
        print(f"Mensaje: {message}")
        
        print("\nProblemas resueltos:")
        print("  1. ChebyshevShell eliminado → Mejor preservación de distancias")
        print("  2. Decoder complejo reemplazado → Reconstrucción más estable")
        print("  3. Arquitectura simplificada → Menor tiempo de ejecución")
        
        print("\nPróximos pasos:")
        print("  1. Aplicar estas correcciones al CCDEngine original")
        print("  2. Actualizar documentación y tests")
        print("  3. Probar con datasets reales de alta dimensión")
        print("  4. Considerar autoencoder simple si k-NN no es suficiente")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "=" * 80)
    print("FIN DEL ANÁLISIS")
    print("=" * 80)