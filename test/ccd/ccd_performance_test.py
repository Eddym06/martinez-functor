"""
Pruebas de rendimiento y comparación exhaustiva del CCDEngine
=============================================================

Este script realiza:
1. Pruebas de escalabilidad con diferentes dimensiones
2. Comparación con PCA, UMAP, y otros métodos
3. Evaluación de estabilidad numérica
4. Pruebas con datos realistas de alta dimensión
"""

import numpy as np
import time
import sys
import warnings
from typing import Dict, List, Tuple, Any
import json
from dataclasses import dataclass, asdict
import traceback

# Configurar entorno
sys.path.insert(0, '/home/Martínez\'s Invariant')
warnings.filterwarnings('ignore')

# Importar CCDEngine
from acf_functor.ccd_engine import CCDEngine, estimate_intrinsic_dimension

# Importar métodos de comparación
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from umap import UMAP
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available for comparison")

@dataclass
class PerformanceResult:
    """Resultado de prueba de rendimiento"""
    test_name: str
    dimensions: Tuple[int, int]  # (n_samples, d_ambient)
    metrics: Dict[str, Any]
    issues: List[str]

class CCDPerformanceTester:
    """Probador de rendimiento del CCDEngine"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.results: List[PerformanceResult] = []
    
    def generate_manifold_data(self, n: int, d_ambient: int, d_intrinsic: int, 
                              noise_level: float = 0.1) -> np.ndarray:
        """Generar datos con manifold de baja dimensión en espacio de alta dimensión"""
        # Coordenadas intrínsecas
        Z = self.rng.normal(0, 1, (n, d_intrinsic))
        
        # Mapeo no lineal a alta dimensión
        # 1. Componentes lineales
        A = self.rng.normal(0, 1, (d_intrinsic, d_ambient // 2))
        linear_part = Z @ A
        
        # 2. Componentes no lineales
        nonlin_part = np.column_stack([
            Z[:, :min(3, d_intrinsic)] ** 2,
            np.sin(Z[:, :min(3, d_intrinsic)]),
            Z[:, 0:min(2, d_intrinsic)].prod(axis=1, keepdims=True),
            np.exp(-Z[:, :min(2, d_intrinsic)] ** 2),
        ])
        
        # Asegurar dimensión correcta
        if linear_part.shape[1] + nonlin_part.shape[1] < d_ambient:
            padding = self.rng.normal(0, 0.01, (n, d_ambient - linear_part.shape[1] - nonlin_part.shape[1]))
            X = np.hstack([linear_part, nonlin_part, padding])
        else:
            X = np.hstack([linear_part, nonlin_part[:, :d_ambient - linear_part.shape[1]]])
        
        # Añadir ruido
        X += noise_level * self.rng.normal(0, 1, X.shape)
        
        return X
    
    def test_scalability_dimensions(self):
        """Prueba de escalabilidad con diferentes dimensiones ambientales"""
        print("\n" + "="*80)
        print("SCALABILITY TEST: Varying Ambient Dimensions")
        print("="*80)
        
        n_samples = 500
        d_intrinsic = 5
        ambient_dims = [10, 30, 50, 100, 200]
        
        metrics = {}
        issues = []
        
        for d_ambient in ambient_dims:
            print(f"\nTesting d_ambient = {d_ambient}, d_intrinsic = {d_intrinsic}")
            
            # Generar datos
            X = self.generate_manifold_data(n_samples, d_ambient, d_intrinsic)
            
            try:
                # Entrenar CCDEngine
                t0 = time.time()
                engine = CCDEngine(
                    d_threshold=10,
                    n_diffusion_components=min(20, d_ambient // 2),
                    n_diffusion_neighbors=15,
                    fit_decoder=True,
                    decoder_epochs=30
                ).fit(X)
                time_fit = time.time() - t0
                
                # Transformar
                t0 = time.time()
                Z = engine.transform(X)
                time_transform = time.time() - t0
                
                # Reconstruir
                X_rec = engine.inverse_transform(Z)
                
                # Calcular métricas
                reconstruction_error = np.mean((X - X_rec) ** 2)
                k_effective = engine.effective_dimension()
                
                # Calcular preservación de vecindarios
                from scipy.spatial.distance import pdist
                sample = min(100, n_samples)
                dist_orig = pdist(X[:sample])
                dist_proj = pdist(Z[:sample])
                if len(dist_orig) > 1 and len(dist_proj) > 1:
                    neighborhood_corr = np.corrcoef(dist_orig, dist_proj)[0, 1]
                else:
                    neighborhood_corr = 0
                
                # Certificado
                cert = engine.certificate()
                
                metrics[f"d{d_ambient}"] = {
                    "time_fit": time_fit,
                    "time_transform": time_transform,
                    "reconstruction_error": reconstruction_error,
                    "k_effective": k_effective,
                    "neighborhood_correlation": neighborhood_corr,
                    "curse_escaped": cert.curse_escaped,
                    "cod_reduction": cert.cod_reduction_log10
                }
                
                print(f"  k_effective: {k_effective}, Fit time: {time_fit:.2f}s, "
                      f"Reconstruction error: {reconstruction_error:.3f}, "
                      f"Neighborhood ρ: {neighborhood_corr:.3f}")
                
                # Verificar problemas
                if reconstruction_error > 1.0:
                    issues.append(f"High reconstruction error ({reconstruction_error:.3f}) for d={d_ambient}")
                
                if neighborhood_corr < 0.3:
                    issues.append(f"Poor neighborhood preservation (ρ={neighborhood_corr:.3f}) for d={d_ambient}")
                
                if k_effective > d_intrinsic * 3:
                    issues.append(f"Overestimation of intrinsic dimension ({k_effective} vs {d_intrinsic}) for d={d_ambient}")
                    
            except Exception as e:
                issues.append(f"Failed for d={d_ambient}: {str(e)}")
                metrics[f"d{d_ambient}"] = {"error": str(e)}
                print(f"  Failed: {e}")
        
        result = PerformanceResult(
            test_name="Scalability with Ambient Dimensions",
            dimensions=(n_samples, max(ambient_dims)),
            metrics=metrics,
            issues=issues
        )
        self.results.append(result)
        return result
    
    def test_scalability_samples(self):
        """Prueba de escalabilidad con diferentes números de muestras"""
        print("\n" + "="*80)
        print("SCALABILITY TEST: Varying Sample Sizes")
        print("="*80)
        
        d_ambient = 50
        d_intrinsic = 5
        sample_sizes = [100, 500, 1000, 2000, 5000]
        
        metrics = {}
        issues = []
        
        for n_samples in sample_sizes:
            print(f"\nTesting n_samples = {n_samples}, d_ambient = {d_ambient}")
            
            # Generar datos
            X = self.generate_manifold_data(n_samples, d_ambient, d_intrinsic)
            
            try:
                # Entrenar CCDEngine
                t0 = time.time()
                engine = CCDEngine(
                    d_threshold=10,
                    n_diffusion_components=10,
                    n_diffusion_neighbors=min(15, n_samples // 10),
                    fit_decoder=(n_samples <= 2000),  # Solo para tamaños manejables
                    decoder_epochs=20
                ).fit(X)
                time_fit = time.time() - t0
                
                # Transformar
                t0 = time.time()
                Z = engine.transform(X)
                time_transform = time.time() - t0
                
                k_effective = engine.effective_dimension()
                
                metrics[f"n{n_samples}"] = {
                    "time_fit": time_fit,
                    "time_transform": time_transform,
                    "k_effective": k_effective,
                    "time_per_sample_fit": time_fit / n_samples,
                    "time_per_sample_transform": time_transform / n_samples
                }
                
                print(f"  k_effective: {k_effective}, Fit time: {time_fit:.2f}s "
                      f"({time_fit/n_samples*1000:.1f}ms/sample), "
                      f"Transform time: {time_transform:.2f}s "
                      f"({time_transform/n_samples*1000:.1f}ms/sample)")
                
                # Verificar complejidad
                if n_samples > 1000 and time_fit > n_samples * 0.1:  # Más de 100ms por muestra
                    issues.append(f"High per-sample fit time ({time_fit/n_samples*1000:.1f}ms) for n={n_samples}")
                
            except Exception as e:
                issues.append(f"Failed for n={n_samples}: {str(e)}")
                metrics[f"n{n_samples}"] = {"error": str(e)}
                print(f"  Failed: {e}")
        
        result = PerformanceResult(
            test_name="Scalability with Sample Sizes",
            dimensions=(max(sample_sizes), d_ambient),
            metrics=metrics,
            issues=issues
        )
        self.results.append(result)
        return result
    
    def test_comparison_with_baselines(self):
        """Comparación exhaustiva con métodos baseline"""
        if not SKLEARN_AVAILABLE:
            print("Skipping baseline comparison: scikit-learn not available")
            return None
        
        print("\n" + "="*80)
        print("COMPARISON TEST: CCD vs PCA, UMAP, t-SNE")
        print("="*80)
        
        n_samples = 1000
        d_ambient = 50
        d_intrinsic = 5
        target_dim = 10
        
        # Generar datos
        X = self.generate_manifold_data(n_samples, d_ambient, d_intrinsic, noise_level=0.05)
        
        methods = {}
        metrics = {}
        issues = []
        
        # 1. PCA
        print("\n1. PCA:")
        try:
            t0 = time.time()
            pca = PCA(n_components=target_dim)
            Z_pca = pca.fit_transform(X)
            time_pca = time.time() - t0
            
            # Reconstrucción
            X_rec_pca = pca.inverse_transform(Z_pca)
            error_pca = np.mean((X - X_rec_pca) ** 2)
            
            # Preservación de vecindarios
            from scipy.spatial.distance import pdist
            sample = min(200, n_samples)
            dist_orig = pdist(X[:sample])
            dist_pca = pdist(Z_pca[:sample])
            corr_pca = np.corrcoef(dist_orig, dist_pca)[0, 1] if len(dist_orig) > 1 else 0
            
            methods["PCA"] = {
                "time": time_pca,
                "error": error_pca,
                "correlation": corr_pca,
                "Z": Z_pca
            }
            
            print(f"  Time: {time_pca:.3f}s, Error: {error_pca:.4f}, Neighborhood ρ: {corr_pca:.3f}")
            
        except Exception as e:
            issues.append(f"PCA failed: {str(e)}")
            print(f"  Failed: {e}")
        
        # 2. UMAP
        print("\n2. UMAP:")
        try:
            t0 = time.time()
            umap = UMAP(n_components=target_dim, random_state=self.seed)
            Z_umap = umap.fit_transform(X)
            time_umap = time.time() - t0
            
            # UMAP no tiene reconstrucción directa, usar k-NN aproximado
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=5).fit(Z_umap)
            _, indices = nn.kneighbors(Z_umap)
            X_rec_umap = np.mean(X[indices], axis=1)
            error_umap = np.mean((X - X_rec_umap) ** 2)
            
            dist_umap = pdist(Z_umap[:sample])
            corr_umap = np.corrcoef(dist_orig, dist_umap)[0, 1] if len(dist_orig) > 1 else 0
            
            methods["UMAP"] = {
                "time": time_umap,
                "error": error_umap,
                "correlation": corr_umap,
                "Z": Z_umap
            }
            
            print(f"  Time: {time_umap:.3f}s, Error: {error_umap:.4f}, Neighborhood ρ: {corr_umap:.3f}")
            
        except Exception as e:
            issues.append(f"UMAP failed: {str(e)}")
            print(f"  Failed: {e}")
        
        # 3. t-SNE (solo para visualización, no reconstrucción)
        print("\n3. t-SNE:")
        try:
            t0 = time.time()
            tsne = TSNE(n_components=2, random_state=self.seed, perplexity=30)
            Z_tsne = tsne.fit_transform(X)
            time_tsne = time.time() - t0
            
            dist_tsne = pdist(Z_tsne[:sample])
            corr_tsne = np.corrcoef(dist_orig, dist_tsne)[0, 1] if len(dist_orig) > 1 else 0
            
            methods["t-SNE"] = {
                "time": time_tsne,
                "error": None,  # t-SNE no tiene reconstrucción
                "correlation": corr_tsne,
                "Z": Z_tsne
            }
            
            print(f"  Time: {time_tsne:.3f}s, Neighborhood ρ: {corr_tsne:.3f}")
            
        except Exception as e:
            issues.append(f"t-SNE failed: {str(e)}")
            print(f"  Failed: {e}")
        
        # 4. CCDEngine
        print("\n4. CCDEngine:")
        try:
            t0 = time.time()
            ccd = CCDEngine(
                d_threshold=10,
                n_diffusion_components=target_dim,
                n_diffusion_neighbors=15,
                fit_decoder=True,
                decoder_epochs=50
            ).fit(X)
            time_ccd_fit = time.time() - t0
            
            t0 = time.time()
            Z_ccd = ccd.transform(X)
            time_ccd_transform = time.time() - t0
            
            X_rec_ccd = ccd.inverse_transform(Z_ccd)
            error_ccd = np.mean((X - X_rec_ccd) ** 2)
            
            dist_ccd = pdist(Z_ccd[:sample])
            corr_ccd = np.corrcoef(dist_orig, dist_ccd)[0, 1] if len(dist_orig) > 1 else 0
            
            k_effective = ccd.effective_dimension()
            cert = ccd.certificate()
            
            methods["CCD"] = {
                "time_fit": time_ccd_fit,
                "time_transform": time_ccd_transform,
                "total_time": time_ccd_fit + time_ccd_transform,
                "error": error_ccd,
                "correlation": corr_ccd,
                "k_effective": k_effective,
                "curse_escaped": cert.curse_escaped,
                "Z": Z_ccd
            }
            
            print(f"  Fit time: {time_ccd_fit:.3f}s, Transform time: {time_ccd_transform:.3f}s")
            print(f"  Error: {error_ccd:.4f}, Neighborhood ρ: {corr_ccd:.3f}")
            print(f"  k_effective: {k_effective}, Curse escaped: {cert.curse_escaped}")
            
        except Exception as e:
            issues.append(f"CCDEngine failed: {str(e)}")
            print(f"  Failed: {e}")
            traceback.print_exc()
        
        # Comparación
        if "CCD" in methods and "PCA" in methods:
            ccd_error = methods["CCD"]["error"]
            pca_error = methods["PCA"]["error"]
            
            if ccd_error > pca_error * 1.5:
                issues.append(f"CCD reconstruction error ({ccd_error:.4f}) worse than PCA ({pca_error:.4f})")
            
            ccd_time = methods["CCD"]["total_time"]
            pca_time = methods["PCA"]["time"]
            
            if ccd_time > pca_time * 5:
                issues.append(f"CCD is {ccd_time/pca_time:.1f}x slower than PCA")
        
        metrics["methods"] = methods
        
        result = PerformanceResult(
            test_name="Comparison with Baselines",
            dimensions=(n_samples, d_ambient),
            metrics=metrics,
            issues=issues
        )
        self.results.append(result)
        return result
    
    def test_numerical_stability(self):
        """Pruebas de estabilidad numérica"""
        print("\n" + "="*80)
        print("NUMERICAL STABILITY TEST")
        print("="*80)
        
        tests = [
            ("Low rank data", self._test_low_rank),
            ("High condition number", self._test_high_condition),
            ("Near-duplicate points", self._test_near_duplicates),
            ("Outliers", self._test_outliers),
            ("Very high dimension", self._test_very_high_dim),
        ]
        
        all_metrics = {}
        all_issues = []
        
        for test_name, test_func in tests:
            print(f"\n{test_name}:")
            try:
                metrics, issues = test_func()
                all_metrics[test_name] = metrics
                all_issues.extend(issues)
                
                if issues:
                    print(f"  Issues: {issues}")
                else:
                    print(f"  Passed")
                    
            except Exception as e:
                all_issues.append(f"{test_name} failed: {str(e)}")
                print(f"  Failed: {e}")
        
        result = PerformanceResult(
            test_name="Numerical Stability",
            dimensions=(500, 50),  # Tamaño de referencia
            metrics=all_metrics,
            issues=all_issues
        )
        self.results.append(result)
        return result
    
    def _test_low_rank(self):
        """Datos de rango bajo"""
        n = 300
        d = 50
        rank = 3
        
        # Matriz de rango bajo
        U = self.rng.normal(0, 1, (n, rank))
        V = self.rng.normal(0, 1, (rank, d))
        X = U @ V + 0.01 * self.rng.normal(0, 1, (n, d))
        
        engine = CCDEngine(d_threshold=10).fit(X)
        k_effective = engine.effective_dimension()
        
        metrics = {"rank": rank, "k_effective": k_effective}
        issues = []
        
        if k_effective > rank * 2:
            issues.append(f"Overestimates low rank: true rank={rank}, estimated={k_effective}")
        
        return metrics, issues
    
    def _test_high_condition(self):
        """Datos con número de condición alto"""
        n = 300
        d = 30
        
        # Crear datos con escala muy diferente
        scales = np.logspace(0, 6, d)  # De 1 a 1e6
        X = self.rng.normal(0, 1, (n, d)) * scales
        
        engine = CCDEngine(d_threshold=10).fit(X)
        
        # Verificar que no hay NaN o inf
        Z = engine.transform(X)
        has_nan = np.any(np.isnan(Z))
        has_inf = np.any(np.isinf(Z))
        
        metrics = {"condition_number": scales[-1]/scales[0]}
        issues = []
        
        if has_nan or has_inf:
            issues.append(f"Numerical instability with high condition number ({scales[-1]/scales[0]:.1e})")
        
        return metrics, issues
    
    def _test_near_duplicates(self):
        """Puntos casi duplicados"""
        n = 200
        d = 40
        
        # Crear grupos de puntos casi idénticos
        X = []
        for _ in range(5):
            center = self.rng.normal(0, 1, d)
            for _ in range(n // 5):
                X.append(center + 1e-6 * self.rng.normal(0, 1, d))
        X = np.array(X)
        
        engine = CCDEngine(d_threshold=10).fit(X)
        k_effective = engine.effective_dimension()
        
        metrics = {"n_groups": 5, "k_effective": k_effective}
        issues = []
        
        if k_effective > 10:  # Debería detectar baja dimensionalidad
            issues.append(f"Doesn't detect low dimension in near-duplicate data: k_effective={k_effective}")
        
        return metrics, issues
    
    def _test_outliers(self):
        """Datos con outliers"""
        n = 400
        d = 30
        
        # 95% datos normales, 5% outliers
        n_normal = int(0.95 * n)
        X_normal = self.rng.normal(0, 1, (n_normal, d))
        X_outliers = self.rng.normal(0, 10, (n - n_normal, d))  # Outliers con varianza alta
        X = np.vstack([X_normal, X_outliers])
        
        engine = CCDEngine(d_threshold=10).fit(X)
        
        # Verificar que los outliers no rompen el modelo
        Z = engine.transform(X)
        has_nan = np.any(np.isnan(Z))
        
        metrics = {"n_outliers": n - n_normal}
        issues = []
        
        if has_nan:
            issues.append("Numerical instability with outliers")
        
        return metrics, issues
    
    def _test_very_high_dim(self):
        """Dimensión muy alta"""
        n = 100
        d = 500  # Dimensión muy alta para n pequeño
        
        X = self.rng.normal(0, 1, (n, d))
        
        try:
            engine = CCDEngine(d_threshold=10).fit(X)
            k_effective = engine.effective_dimension()
            
            metrics = {"d": d, "n": n, "k_effective": k_effective}
            issues = []
            
            if k_effective >= d:  # No debería mantener toda la dimensión
                issues.append(f"No dimensionality reduction for very high d={d} with small n={n}")
            
        except MemoryError:
            metrics = {"d": d, "n": n, "error": "MemoryError"}
            issues = ["Memory error with very high dimension"]
        except Exception as e:
            metrics = {"d": d, "n": n, "error": str(e)}
            issues = [f"Failed with very high dimension: {str(e)}"]
        
        return metrics, issues
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Ejecutar todas las pruebas"""
        print("="*80)
        print("CCDEngine PERFORMANCE AND STABILITY ANALYSIS")
        print("="*80)
        
        tests = [
            self.test_scalability_dimensions,
            self.test_scalability_samples,
            self.test_comparison_with_baselines,
            self.test_numerical_stability,
        ]
        
        for test_func in tests:
            try:
                print(f"\nRunning {test_func.__name__}...")
                test_func()
            except Exception as e:
                print(f"  Test crashed: {e}")
                traceback.print_exc()
        
        # Resumen
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        total_issues = []
        for result in self.results:
            total_issues.extend(result.issues)
        
        if total_issues:
            print(f"\nCRITICAL ISSUES FOUND ({len(total_issues)}):")
            for issue in set(total_issues):
                print(f"  • {issue}")
        else:
            print("\n✓ No critical issues found")
        
        # Métricas agregadas
        summary = {
            "n_tests": len(self.results),
            "n_issues": len(total_issues),
            "critical_issues": list(set(total_issues)),
            "results": [asdict(r) for r in self.results]
        }
        
        return summary

def main():
    """Función principal"""
    tester = CCDPerformanceTester(seed=42)
    results = tester.run_all_tests()
    
    # Guardar resultados
    with open('/home/Martínez\'s Invariant/ccd_performance_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to ccd_performance_results.json")
    
    return results

if __name__ == "__main__":
    main()