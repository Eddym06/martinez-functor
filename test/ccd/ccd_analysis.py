"""
Análisis profundo del CCDEngine - Evaluación completa
=====================================================

Este script realiza pruebas exhaustivas del CCDEngine para:
1. Evaluar cada capa individualmente
2. Medir rendimiento con datos sintéticos
3. Probar estabilidad numérica
4. Comparar con métodos alternativos
5. Identificar áreas de mejora
"""

import numpy as np
import time
import sys
import warnings
from typing import Dict, List, Tuple, Any
import json
from dataclasses import dataclass, asdict

# Importar CCDEngine
sys.path.insert(0, '/home/Martínez\'s Invariant')
from acf_functor.ccd_engine import (
    CCDEngine, ChebyshevShell, SpectralPreprocessor, 
    DiffusionGeometry, CoupledOscillators, LocalEntropyOperator,
    LangevinPurifier, estimate_intrinsic_dimension
)

warnings.filterwarnings('ignore')

@dataclass
class TestResult:
    """Resultado de una prueba individual"""
    test_name: str
    passed: bool
    metrics: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]

class CCDAnalyzer:
    """Analizador completo del CCDEngine"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.results: List[TestResult] = []
    
    def add_result(self, result: TestResult):
        """Agregar resultado de prueba"""
        self.results.append(result)
        print(f"\n{'✓' if result.passed else '✗'} {result.test_name}")
        if result.issues:
            print(f"  Issues: {result.issues}")
        if result.recommendations:
            print(f"  Recommendations: {result.recommendations}")
    
    def test_1_chebyshev_shell(self) -> TestResult:
        """Prueba de ChebyshevShell vs SpectralPreprocessor"""
        print("\n" + "="*80)
        print("TEST 1: ChebyshevShell vs SpectralPreprocessor")
        print("="*80)
        
        # Generar datos sintéticos
        n = 1000
        d = 50
        X = self.rng.normal(0, 1, (n, d))
        
        issues = []
        recommendations = []
        
        # Test ChebyshevShell
        try:
            t0 = time.time()
            cheb = ChebyshevShell(n_coeffs=8, compression_rank=16).fit(X)
            Z_cheb = cheb.transform(X)
            X_rec_cheb = cheb.inverse_transform(Z_cheb)
            time_cheb = time.time() - t0
            
            # Calcular métricas
            error_cheb = np.mean((X - X_rec_cheb) ** 2)
            rank_cheb = cheb.effective_rank
            var_explained_cheb = cheb.explained_variance_ratio.sum()
            
            # Calcular correlación de distancias
            from scipy.spatial.distance import pdist, squareform
            dist_orig = pdist(X[:100])  # Muestra para velocidad
            dist_cheb = pdist(Z_cheb[:100])
            corr_cheb = np.corrcoef(dist_orig, dist_cheb)[0, 1]
            
        except Exception as e:
            issues.append(f"ChebyshevShell failed: {e}")
            error_cheb = float('inf')
            time_cheb = float('inf')
            corr_cheb = 0
            rank_cheb = 0
            var_explained_cheb = 0
        
        # Test SpectralPreprocessor
        try:
            t0 = time.time()
            spec = SpectralPreprocessor(n_components=32, whiten=True).fit(X)
            Z_spec = spec.transform(X)
            X_rec_spec = spec.inverse_transform(Z_spec)
            time_spec = time.time() - t0
            
            error_spec = np.mean((X - X_rec_spec) ** 2)
            rank_spec = spec.effective_rank
            var_explained_spec = spec.explained_variance_ratio.sum()
            
            dist_spec = pdist(Z_spec[:100])
            corr_spec = np.corrcoef(dist_orig, dist_spec)[0, 1]
            
        except Exception as e:
            issues.append(f"SpectralPreprocessor failed: {e}")
            error_spec = float('inf')
            time_spec = float('inf')
            corr_spec = 0
            rank_spec = 0
            var_explained_spec = 0
        
        # Comparación
        if corr_cheb < 0.5:
            issues.append(f"ChebyshevShell destroys distance correlation: ρ={corr_cheb:.3f}")
            recommendations.append("Replace ChebyshevShell with SpectralPreprocessor as Layer 1")
        
        if time_cheb > 2 * time_spec:
            recommendations.append(f"ChebyshevShell is {time_cheb/time_spec:.1f}x slower than SpectralPreprocessor")
        
        passed = (corr_spec > 0.9 and error_spec < 0.1)
        
        return TestResult(
            test_name="ChebyshevShell vs SpectralPreprocessor",
            passed=passed,
            metrics={
                "chebyshev_time": time_cheb,
                "chebyshev_error": error_cheb,
                "chebyshev_correlation": corr_cheb,
                "chebyshev_rank": rank_cheb,
                "chebyshev_var_explained": var_explained_cheb,
                "spectral_time": time_spec,
                "spectral_error": error_spec,
                "spectral_correlation": corr_spec,
                "spectral_rank": rank_spec,
                "spectral_var_explained": var_explained_spec,
            },
            issues=issues,
            recommendations=recommendations
        )
    
    def test_2_diffusion_geometry_scalability(self) -> TestResult:
        """Prueba de escalabilidad de DiffusionGeometry"""
        print("\n" + "="*80)
        print("TEST 2: DiffusionGeometry Scalability")
        print("="*80)
        
        issues = []
        recommendations = []
        metrics = {}
        
        # Probar diferentes tamaños
        sizes = [100, 500, 1000, 2000]
        times = []
        memory_usage = []
        
        for n in sizes:
            d = 20
            X = self.rng.normal(0, 1, (n, d))
            
            try:
                t0 = time.time()
                diff = DiffusionGeometry(
                    n_components=10,
                    n_neighbors=15,
                    diffusion_time=1.0,
                    alpha=0.5
                ).fit(X)
                Z = diff.transform(X)
                elapsed = time.time() - t0
                
                times.append(elapsed)
                
                # Estimación de memoria (aproximada)
                mem = 0
                if diff._Phi is not None:
                    mem += diff._Phi.nbytes
                if diff._X_train is not None:
                    mem += diff._X_train.nbytes
                memory_usage.append(mem / 1e6)  # MB
                
                metrics[f"time_n{n}"] = elapsed
                metrics[f"mem_n{n}"] = mem / 1e6
                
                # Verificar dimensionalidad intrínseca
                intrinsic_dim = diff.intrinsic_dimension_estimate
                if intrinsic_dim > d:
                    issues.append(f"Intrinsic dimension estimate ({intrinsic_dim}) > ambient dim ({d}) for n={n}")
                
            except Exception as e:
                issues.append(f"Failed for n={n}: {e}")
                times.append(float('inf'))
                memory_usage.append(float('inf'))
        
        # Verificar complejidad O(n·k)
        if len(times) >= 3:
            # Calcular ratio de crecimiento
            ratio = times[-1] / times[0]
            expected_ratio = sizes[-1] / sizes[0]  # O(n)
            if ratio > 5 * expected_ratio:
                issues.append(f"Time scaling worse than O(n): ratio={ratio:.1f}, expected={expected_ratio:.1f}")
        
        passed = (len(issues) == 0)
        
        return TestResult(
            test_name="DiffusionGeometry Scalability",
            passed=passed,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )
    
    def test_3_coupled_oscillators_resonance(self) -> TestResult:
        """Prueba de detección de resonancia en CoupledOscillators"""
        print("\n" + "="*80)
        print("TEST 3: CoupledOscillators Resonance Detection")
        print("="*80)
        
        # Crear datos con estructura de resonancia conocida
        n = 500
        d = 30
        
        # Grupo 1: 5 variables correlacionadas
        group1 = self.rng.normal(0, 1, (n, 1))
        X_group1 = group1 @ self.rng.normal(0, 1, (1, 5))
        
        # Grupo 2: 3 variables correlacionadas
        group2 = self.rng.normal(0, 1, (n, 1))
        X_group2 = group2 @ self.rng.normal(0, 1, (1, 3))
        
        # Ruido independiente
        noise = self.rng.normal(0, 0.1, (n, d - 8))
        
        X = np.hstack([X_group1, X_group2, noise])
        
        issues = []
        recommendations = []
        
        try:
            osc = CoupledOscillators(
                n_groups=None,  # Auto-detect
                coherence_threshold=0.3
            ).fit(X)
            
            n_groups = osc.n_resonance_groups
            groups = osc.resonance_groups
            
            # Verificar que detectó aproximadamente 2 grupos
            if n_groups < 2 or n_groups > 4:
                issues.append(f"Expected ~2 resonance groups, found {n_groups}")
            
            # Verificar tamaño de grupos
            group_sizes = [len(g.variable_indices) for g in groups]
            print(f"Group sizes: {group_sizes}")
            
            # Verificar que los grupos grandes corresponden a las variables correlacionadas
            large_groups = [i for i, size in enumerate(group_sizes) if size >= 3]
            if len(large_groups) < 2:
                issues.append(f"Expected 2 large groups (size ≥3), found {len(large_groups)}")
            
            # Probar transformaciones
            Z_transform = osc.transform(X)
            Z_coherence = osc.adaptive_coherence_transform(X)
            
            metrics = {
                "n_groups": n_groups,
                "group_sizes": group_sizes,
                "transform_shape": Z_transform.shape,
                "coherence_shape": Z_coherence.shape,
                "explained_variance": osc.explained_variance_ratio.sum()
            }
            
            passed = (n_groups >= 2 and n_groups <= 4)
            
        except Exception as e:
            issues.append(f"CoupledOscillators failed: {e}")
            metrics = {}
            passed = False
        
        return TestResult(
            test_name="CoupledOscillators Resonance Detection",
            passed=passed,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )
    
    def test_4_langevin_purification(self) -> TestResult:
        """Prueba de purificación con Langevin"""
        print("\n" + "="*80)
        print("TEST 4: Langevin Purification")
        print("="*80)
        
        # Crear datos con manifold conocido (esfera 2D en R^10)
        n = 300
        d_ambient = 10
        d_intrinsic = 2
        
        # Manifold: esfera 2D
        theta = self.rng.uniform(0, 2*np.pi, n)
        phi = self.rng.uniform(0, np.pi, n)
        manifold = np.column_stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
            np.zeros((n, d_ambient - 3))
        ])
        
        # Añadir ruido
        noise = self.rng.normal(0, 0.3, (n, d_ambient))
        X_noisy = manifold + noise
        
        issues = []
        recommendations = []
        
        try:
            # Entrenar purificador
            purifier = LangevinPurifier(
                n_steps=20,
                dt=0.05,
                n_score_neighbors=30,
                seed=self.seed
            ).fit(manifold)
            
            # Purificar
            t0 = time.time()
            X_pure = purifier.purify(X_noisy)
            time_purify = time.time() - t0
            
            # Calcular métricas
            error_before = np.mean(np.linalg.norm(X_noisy - manifold, axis=1))
            error_after = np.mean(np.linalg.norm(X_pure - manifold, axis=1))
            improvement = (error_before - error_after) / error_before
            
            metrics = {
                "error_before": error_before,
                "error_after": error_after,
                "improvement": improvement,
                "time_purify": time_purify
            }
            
            if improvement < 0.1:
                issues.append(f"Poor purification improvement: {improvement:.1%}")
                recommendations.append("Increase n_steps or adjust dt")
            
            passed = improvement > 0.1
            
        except Exception as e:
            issues.append(f"Langevin purification failed: {e}")
            metrics = {}
            passed = False
        
        return TestResult(
            test_name="Langevin Purification",
            passed=passed,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )
    
    def test_5_full_ccd_pipeline(self) -> TestResult:
        """Prueba completa del pipeline CCD"""
        print("\n" + "="*80)
        print("TEST 5: Full CCD Pipeline")
        print("="*80)
        
        # Datos de alta dimensión con estructura de baja dimensión
        n_train = 1000
        n_test = 200
        d_ambient = 50
        d_intrinsic = 5
        
        # Generar manifold de baja dimensión
        Z_low = self.rng.normal(0, 1, (n_train + n_test, d_intrinsic))
        # Mapeo no lineal a alta dimensión
        A = self.rng.normal(0, 1, (d_intrinsic, d_ambient))
        X = Z_low @ A + 0.1 * self.rng.normal(0, 1, (n_train + n_test, d_ambient))
        
        X_train = X[:n_train]
        X_test = X[n_train:]
        
        issues = []
        recommendations = []
        
        try:
            # Entrenar CCDEngine completo
            t0 = time.time()
            engine = CCDEngine(
                d_threshold=10,
                n_diffusion_components=10,
                n_diffusion_neighbors=15,
                fit_decoder=True,
                decoder_epochs=50
            ).fit(X_train)
            time_fit = time.time() - t0
            
            # Transformar
            t0 = time.time()
            Z_test = engine.transform(X_test)
            time_transform = time.time() - t0
            
            # Reconstruir
            X_rec = engine.inverse_transform(Z_test)
            
            # Calcular métricas
            reconstruction_error = np.mean((X_test - X_rec) ** 2)
            k_effective = engine.effective_dimension()
            
            # Certificado
            cert = engine.certificate()
            
            # Robust certificate (simplificado)
            from scipy.spatial.distance import pdist, squareform
            sample_size = min(100, len(X_test))
            dist_orig = pdist(X_test[:sample_size])
            dist_proj = pdist(Z_test[:sample_size])
            if len(dist_orig) > 1 and len(dist_proj) > 1:
                neighborhood_corr = np.corrcoef(dist_orig, dist_proj)[0, 1]
            else:
                neighborhood_corr = 0
            
            metrics = {
                "time_fit": time_fit,
                "time_transform": time_transform,
                "reconstruction_error": reconstruction_error,
                "k_effective": k_effective,
                "d_ambient": d_ambient,
                "reduction_ratio": d_ambient / max(k_effective, 1),
                "neighborhood_correlation": neighborhood_corr,
                "curse_escaped": cert.curse_escaped,
                "cod_reduction": cert.cod_reduction_log10
            }
            
            # Evaluar
            if k_effective > d_intrinsic * 3:
                issues.append(f"Effective dimension ({k_effective}) much larger than true intrinsic dim ({d_intrinsic})")
            
            if reconstruction_error > 0.5:
                issues.append(f"High reconstruction error: {reconstruction_error:.3f}")
                recommendations.append("Increase decoder capacity or training epochs")
            
            if neighborhood_corr < 0.7:
                issues.append(f"Poor neighborhood preservation: ρ={neighborhood_corr:.3f}")
                recommendations.append("Adjust diffusion parameters or use different kernel")
            
            passed = (reconstruction_error < 1.0 and neighborhood_corr > 0.5)
            
        except Exception as e:
            issues.append(f"Full pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            metrics = {}
            passed = False
        
        return TestResult(
            test_name="Full CCD Pipeline",
            passed=passed,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )
    
    def test_6_comparison_with_baselines(self) -> TestResult:
        """Comparación con métodos baseline (PCA, etc.)"""
        print("\n" + "="*80)
        print("TEST 6: Comparison with Baselines")
        print("="*80)
        
        n = 500
        d = 40
        d_intrinsic = 8
        
        # Datos con estructura no lineal
        Z_low = self.rng.normal(0, 1, (n, d_intrinsic))
        # Mapeo no lineal
        X = np.column_stack([
            Z_low,
            Z_low[:, :3] ** 2,
            np.sin(Z_low[:, 3:6]),
            Z_low[:, 0:2].prod(axis=1, keepdims=True),
            self.rng.normal(0, 0.1, (n, d - d_intrinsic - 3 - 3 - 1))
        ])
        
        issues = []
        recommendations = []
        metrics = {}
        
        try:
            # 1. PCA
            from sklearn.decomposition import PCA
            t0 = time.time()
            pca = PCA(n_components=d_intrinsic)
            Z_pca = pca.fit_transform(X)
            X_rec_pca = pca.inverse_transform(Z_pca)
            time_pca = time.time() - t0
            error_pca = np.mean((X - X_rec_pca) ** 2)
            
            # 2. CCDEngine
            t0 = time.time()
            ccd = CCDEngine(
                d_threshold=10,
                n_diffusion_components=d_intrinsic,
                fit_decoder=True,
                decoder_epochs=30
            ).fit(X)
            Z_ccd = ccd.transform(X)
            X_rec_ccd = ccd.inverse_transform(Z_ccd)
            time_ccd = time.time() - t0
            error_ccd = np.mean((X - X_rec_ccd) ** 2)
            
            # 3. Calcular preservación de vecindarios
            from scipy.spatial.distance import pdist
            sample = min(100, n)
            dist_orig = pdist(X[:sample])
            dist_pca = pdist(Z_pca[:sample])
            dist_ccd = pdist(Z_ccd[:sample])
            
            corr_pca = np.corrcoef(dist_orig, dist_pca)[0, 1] if len(dist_orig) > 1 else 0
            corr_ccd = np.corrcoef(dist_orig, dist_ccd)[0, 1] if len(dist_orig) > 1 else 0
            
            metrics = {
                "pca_time": time_pca,
                "pca_error": error_pca,
                "pca_correlation": corr_pca,
                "ccd_time": time_ccd,
                "ccd_error": error_ccd,
                "ccd_correlation": corr_ccd,
                "ccd_vs_pca_error_ratio": error_ccd / max(error_pca, 1e-10),
                "ccd_vs_pca_time_ratio": time_ccd / max(time_pca, 1e-10)
            }
            
            # Evaluar
            if error_ccd > error_pca * 1.5:
                issues.append(f"CCD error ({error_ccd:.3f}) worse than PCA ({error_pca:.3f})")
            
            if time_ccd > time_pca * 10:
                issues.append(f"CCD is {time_ccd/time_pca:.1f}x slower than PCA")
                recommendations.append("Optimize computational bottlenecks")
            
            if corr_ccd < corr_pca:
                issues.append(f"CCD neighborhood preservation ({corr_ccd:.3f}) worse than PCA ({corr_pca:.3f})")
            
            passed = (error_ccd < error_pca * 1.2 or corr_ccd > corr_pca)
            
        except Exception as e:
            issues.append(f"Comparison test failed: {e}")
            import traceback
            traceback.print_exc()
            passed = False
        
        return TestResult(
            test_name="Comparison with Baselines",
            passed=passed,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Ejecutar todas las pruebas"""
        print("="*80)
        print("COMPREHENSIVE CCDEngine ANALYSIS")
        print("="*80)
        
        tests = [
            self.test_1_chebyshev_shell,
            self.test_2_diffusion_geometry_scalability,
            self.test_3_coupled_oscillators_resonance,
            self.test_4_langevin_purification,
            self.test_5_full_ccd_pipeline,
            self.test_6_comparison_with_baselines,
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                self.add_result(result)
            except Exception as e:
                print(f"\n✗ {test_func.__name__} crashed: {e}")
                import traceback
                traceback.print_exc()
        
        # Resumen
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
        
        # Agregar issues y recomendaciones globales
        all_issues = []
        all_recommendations = []
        
        for result in self.results:
            all_issues.extend(result.issues)
            all_recommendations.extend(result.recommendations)
        
        if all_issues:
            print("\nCRITICAL ISSUES:")
            for issue in set(all_issues):
                print(f"  • {issue}")
        
        if all_recommendations:
            print("\nRECOMMENDATIONS:")
            for rec in set(all_recommendations):
                print(f"  • {rec}")
        
        return {
            "summary": {
                "passed": passed,
                "total": total,
                "success_rate": passed/total*100
            },
            "results": [asdict(r) for r in self.results],
            "critical_issues": list(set(all_issues)),
            "recommendations": list(set(all_recommendations))
        }

def main():
    """Función principal"""
    analyzer = CCDAnalyzer(seed=42)
    results = analyzer.run_all_tests()
    
    # Guardar resultados
    with open('/home/Martínez\'s Invariant/ccd_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to ccd_analysis_results.json")
    
    return results

if __name__ == "__main__":
    main()