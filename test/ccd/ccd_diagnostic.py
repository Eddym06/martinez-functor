"""
Diagnóstico profundo de los problemas del CCDEngine
===================================================

Este script analiza los problemas identificados y propone soluciones específicas.
"""

import numpy as np
import sys
import warnings
from typing import Dict, List, Tuple, Any
import json

sys.path.insert(0, '/home/Martínez\'s Invariant')
warnings.filterwarnings('ignore')

from acf_functor.ccd_engine import (
    CCDEngine, ChebyshevShell, SpectralPreprocessor, 
    DiffusionGeometry, CoupledOscillators, LocalEntropyOperator,
    LangevinPurifier, ManifoldDecoder
)

class CCDDiagnostic:
    """Diagnóstico detallado del CCDEngine"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.findings = []
    
    def add_finding(self, category: str, issue: str, severity: str, 
                   evidence: str, recommendation: str):
        """Agregar hallazgo"""
        self.findings.append({
            "category": category,
            "issue": issue,
            "severity": severity,  # "critical", "high", "medium", "low"
            "evidence": evidence,
            "recommendation": recommendation
        })
    
    def diagnose_chebyshev_problem(self):
        """Diagnosticar problema con ChebyshevShell"""
        print("\n" + "="*80)
        print("DIAGNOSIS: ChebyshevShell Problem")
        print("="*80)
        
        n = 500
        d = 30
        X = self.rng.normal(0, 1, (n, d))
        
        # Probar ChebyshevShell
        cheb = ChebyshevShell(n_coeffs=8, compression_rank=16).fit(X)
        Z_cheb = cheb.transform(X)
        X_rec_cheb = cheb.inverse_transform(Z_cheb)
        
        # Probar SpectralPreprocessor
        spec = SpectralPreprocessor(n_components=32, whiten=True).fit(X)
        Z_spec = spec.transform(X)
        X_rec_spec = spec.inverse_transform(Z_spec)
        
        # Calcular métricas
        error_cheb = np.mean((X - X_rec_cheb) ** 2)
        error_spec = np.mean((X - X_rec_spec) ** 2)
        
        # Calcular correlación de distancias
        from scipy.spatial.distance import pdist
        dist_orig = pdist(X[:100])
        dist_cheb = pdist(Z_cheb[:100])
        dist_spec = pdist(Z_spec[:100])
        
        corr_cheb = np.corrcoef(dist_orig, dist_cheb)[0, 1] if len(dist_orig) > 1 else 0
        corr_spec = np.corrcoef(dist_orig, dist_spec)[0, 1] if len(dist_orig) > 1 else 0
        
        print(f"ChebyshevShell:")
        print(f"  Reconstruction error: {error_cheb:.4f}")
        print(f"  Distance correlation: {corr_cheb:.3f}")
        print(f"  Effective rank: {cheb.effective_rank}")
        
        print(f"\nSpectralPreprocessor:")
        print(f"  Reconstruction error: {error_spec:.4f}")
        print(f"  Distance correlation: {corr_spec:.3f}")
        print(f"  Effective rank: {spec.effective_rank}")
        
        # Análisis
        if corr_cheb < 0.5:
            self.add_finding(
                category="ChebyshevShell",
                issue="Destroys distance correlation",
                severity="critical",
                evidence=f"Distance correlation ρ={corr_cheb:.3f} (should be close to 1.0)",
                recommendation="Replace ChebyshevShell with SpectralPreprocessor as Layer 1"
            )
        
        if error_cheb > error_spec * 2:
            self.add_finding(
                category="ChebyshevShell",
                issue="Poor reconstruction quality",
                severity="high",
                evidence=f"Reconstruction error {error_cheb:.4f} vs {error_spec:.4f} for SpectralPreprocessor",
                recommendation="Use learned decoder instead of T_1(x)=x approximation"
            )
    
    def diagnose_reconstruction_problem(self):
        """Diagnosticar problema de reconstrucción"""
        print("\n" + "="*80)
        print("DIAGNOSIS: Reconstruction Problem")
        print("="*80)
        
        n = 500
        d = 50
        d_intrinsic = 5
        
        # Generar datos con manifold conocido
        Z_low = self.rng.normal(0, 1, (n, d_intrinsic))
        A = self.rng.normal(0, 1, (d_intrinsic, d))
        X = Z_low @ A + 0.1 * self.rng.normal(0, 1, (n, d))
        
        # Entrenar CCDEngine con diferentes configuraciones
        configs = [
            ("Default", {"fit_decoder": True, "decoder_epochs": 50}),
            ("More epochs", {"fit_decoder": True, "decoder_epochs": 200}),
            ("Larger decoder", {"fit_decoder": True, "decoder_epochs": 100, "decoder_hidden_dim": 128}),
            ("No decoder", {"fit_decoder": False}),
        ]
        
        results = {}
        
        for config_name, params in configs:
            print(f"\nTesting {config_name}:")
            try:
                engine = CCDEngine(
                    d_threshold=10,
                    n_diffusion_components=10,
                    n_diffusion_neighbors=15,
                    **params
                ).fit(X)
                
                Z = engine.transform(X)
                X_rec = engine.inverse_transform(Z)
                
                error = np.mean((X - X_rec) ** 2)
                k_effective = engine.effective_dimension()
                
                results[config_name] = {
                    "error": error,
                    "k_effective": k_effective,
                    "has_decoder": params.get("fit_decoder", False)
                }
                
                print(f"  Reconstruction error: {error:.4f}")
                print(f"  k_effective: {k_effective}")
                
            except Exception as e:
                print(f"  Failed: {e}")
                results[config_name] = {"error": float('inf'), "failed": str(e)}
        
        # Análisis
        if "Default" in results:
            default_error = results["Default"]["error"]
            
            if default_error > 1.0:
                self.add_finding(
                    category="Reconstruction",
                    issue="High reconstruction error",
                    severity="critical",
                    evidence=f"Default configuration error: {default_error:.4f}",
                    recommendation="Increase decoder capacity and training epochs"
                )
            
            # Comparar con/sin decoder
            if "No decoder" in results:
                no_decoder_error = results["No decoder"]["error"]
                if default_error > no_decoder_error * 1.5:
                    self.add_finding(
                        category="Reconstruction",
                        issue="Decoder makes reconstruction worse",
                        severity="high",
                        evidence=f"Decoder error {default_error:.4f} vs no decoder {no_decoder_error:.4f}",
                        recommendation="Improve decoder training or architecture"
                    )
    
    def diagnose_diffusion_problem(self):
        """Diagnosticar problema con DiffusionGeometry"""
        print("\n" + "="*80)
        print("DIAGNOSIS: DiffusionGeometry Problem")
        print("="*80)
        
        n = 500
        d = 30
        
        # Datos con estructura clara
        t = np.linspace(0, 4*np.pi, n)
        X = np.column_stack([
            np.sin(t),
            np.cos(t),
            t/10,
            self.rng.normal(0, 0.1, n),
            self.rng.normal(0, 0.1, n),
            self.rng.normal(0, 0.1, (n, d-5))
        ])
        
        # Probar diferentes configuraciones
        configs = [
            ("Default", {"n_neighbors": 15, "alpha": 0.5}),
            ("More neighbors", {"n_neighbors": 30, "alpha": 0.5}),
            ("Different alpha", {"n_neighbors": 15, "alpha": 1.0}),
            ("Few components", {"n_neighbors": 15, "alpha": 0.5, "n_components": 3}),
        ]
        
        for config_name, params in configs:
            print(f"\nTesting {config_name}:")
            try:
                diff = DiffusionGeometry(
                    n_components=params.get("n_components", 10),
                    n_neighbors=params["n_neighbors"],
                    diffusion_time=1.0,
                    alpha=params["alpha"]
                ).fit(X)
                
                Z = diff.transform(X)
                intrinsic_dim = diff.intrinsic_dimension_estimate
                
                # Calcular preservación de estructura
                from scipy.spatial.distance import pdist
                dist_orig = pdist(X[:100])
                dist_proj = pdist(Z[:100])
                if len(dist_orig) > 1 and len(dist_proj) > 1:
                    corr = np.corrcoef(dist_orig, dist_proj)[0, 1]
                else:
                    corr = 0
                
                print(f"  Intrinsic dimension estimate: {intrinsic_dim}")
                print(f"  Neighborhood correlation: {corr:.3f}")
                
                if corr < 0.5:
                    self.add_finding(
                        category="DiffusionGeometry",
                        issue="Poor neighborhood preservation",
                        severity="high",
                        evidence=f"Config {config_name}: ρ={corr:.3f}",
                        recommendation="Adjust kernel parameters or use different kernel"
                    )
                
                if intrinsic_dim < 2 or intrinsic_dim > 10:  # Debería detectar ~3 dimensiones
                    self.add_finding(
                        category="DiffusionGeometry",
                        issue="Inaccurate intrinsic dimension estimation",
                        severity="medium",
                        evidence=f"Config {config_name}: estimated {intrinsic_dim} (expected ~3)",
                        recommendation="Improve spectral gap detection algorithm"
                    )
                    
            except Exception as e:
                print(f"  Failed: {e}")
    
    def diagnose_oscillator_problem(self):
        """Diagnosticar problema con CoupledOscillators"""
        print("\n" + "="*80)
        print("DIAGNOSIS: CoupledOscillators Problem")
        print("="*80)
        
        n = 500
        d = 40
        
        # Crear datos con grupos de resonancia claros
        # Grupo 1: 6 variables correlacionadas
        signal1 = self.rng.normal(0, 1, (n, 1))
        group1 = signal1 @ self.rng.normal(0, 1, (1, 6))
        
        # Grupo 2: 4 variables correlacionadas
        signal2 = self.rng.normal(0, 1, (n, 1))
        group2 = signal2 @ self.rng.normal(0, 1, (1, 4))
        
        # Ruido independiente
        noise = self.rng.normal(0, 0.2, (n, d - 10))
        
        X = np.hstack([group1, group2, noise])
        
        # Probar detección de resonancia
        osc = CoupledOscillators(
            n_groups=None,  # Auto-detect
            coherence_threshold=0.3
        ).fit(X)
        
        n_groups = osc.n_resonance_groups
        groups = osc.resonance_groups
        
        print(f"Detected {n_groups} resonance groups")
        print(f"Group sizes: {[len(g.variable_indices) for g in groups]}")
        
        # Verificar que detectó los grupos correctos
        large_groups = [g for g in groups if len(g.variable_indices) >= 3]
        
        if len(large_groups) < 2:
            self.add_finding(
                category="CoupledOscillators",
                issue="Misses clear resonance groups",
                severity="medium",
                evidence=f"Expected 2 large groups, found {len(large_groups)}",
                recommendation="Adjust coherence threshold or improve group detection"
            )
        
        # Verificar coherencia
        for i, group in enumerate(groups[:3]):
            print(f"Group {i}: size={len(group.variable_indices)}, coherence={group.coherence_score:.3f}")
            
            if group.coherence_score < 0.5 and len(group.variable_indices) > 2:
                self.add_finding(
                    category="CoupledOscillators",
                    issue="Low coherence in detected groups",
                    severity="low",
                    evidence=f"Group {i} coherence: {group.coherence_score:.3f}",
                    recommendation="Improve resonance detection algorithm"
                )
    
    def diagnose_performance_problem(self):
        """Diagnosticar problemas de rendimiento"""
        print("\n" + "="*80)
        print("DIAGNOSIS: Performance Problem")
        print("="*80)
        
        # Analizar complejidad computacional
        bottlenecks = []
        
        # 1. ChebyshevShell es O(n·d·m) donde m = n_coeffs
        bottlenecks.append({
            "component": "ChebyshevShell",
            "complexity": "O(n·d·m) where m = n_coeffs (typically 8)",
            "issue": "Expands dimension before compression",
            "impact": "High"
        })
        
        # 2. DiffusionGeometry con kernel denso es O(n²·d)
        bottlenecks.append({
            "component": "DiffusionGeometry (dense)",
            "complexity": "O(n²·d)",
            "issue": "Quadratic in n",
            "impact": "Critical for n > 5000"
        })
        
        # 3. Sparse kernel es O(n·k·d) donde k = n_neighbors
        bottlenecks.append({
            "component": "DiffusionGeometry (sparse)",
            "complexity": "O(n·k·d) where k = n_neighbors",
            "issue": "Much better but still linear in n·d",
            "impact": "Medium"
        })
        
        # 4. Decoder training es O(epochs·n·hidden_dim²)
        bottlenecks.append({
            "component": "ManifoldDecoder",
            "complexity": "O(epochs·n·hidden_dim²)",
            "issue": "Training time scales with epochs and hidden_dim²",
            "impact": "High for large hidden_dim"
        })
        
        print("Computational bottlenecks:")
        for b in bottlenecks:
            print(f"  {b['component']}: {b['complexity']} ({b['impact']} impact)")
            if b['impact'] in ["Critical", "High"]:
                self.add_finding(
                    category="Performance",
                    issue=f"Computational bottleneck in {b['component']}",
                    severity=b['impact'].lower(),
                    evidence=f"Complexity: {b['complexity']}",
                    recommendation=f"Optimize {b['component']} algorithm"
                )
    
    def run_all_diagnostics(self) -> Dict[str, Any]:
        """Ejecutar todos los diagnósticos"""
        print("="*80)
        print("CCDEngine COMPREHENSIVE DIAGNOSIS")
        print("="*80)
        
        diagnostics = [
            self.diagnose_chebyshev_problem,
            self.diagnose_reconstruction_problem,
            self.diagnose_diffusion_problem,
            self.diagnose_oscillator_problem,
            self.diagnose_performance_problem,
        ]
        
        for diagnostic in diagnostics:
            try:
                diagnostic()
            except Exception as e:
                print(f"\nDiagnostic {diagnostic.__name__} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Resumen por categoría
        print("\n" + "="*80)
        print("DIAGNOSIS SUMMARY")
        print("="*80)
        
        categories = {}
        for finding in self.findings:
            cat = finding["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(finding)
        
        for category, findings in categories.items():
            print(f"\n{category}:")
            critical = [f for f in findings if f["severity"] == "critical"]
            high = [f for f in findings if f["severity"] == "high"]
            medium = [f for f in findings if f["severity"] == "medium"]
            low = [f for f in findings if f["severity"] == "low"]
            
            print(f"  Critical: {len(critical)}, High: {len(high)}, "
                  f"Medium: {len(medium)}, Low: {len(low)}")
            
            if critical:
                print(f"  Critical issues:")
                for f in critical:
                    print(f"    • {f['issue']}")
        
        # Recomendaciones prioritarias
        print("\n" + "="*80)
        print("PRIORITY RECOMMENDATIONS")
        print("="*80)
        
        critical_recs = []
        high_recs = []
        
        for finding in self.findings:
            if finding["severity"] == "critical":
                critical_recs.append(finding["recommendation"])
            elif finding["severity"] == "high":
                high_recs.append(finding["recommendation"])
        
        if critical_recs:
            print("\nCRITICAL (must fix):")
            for rec in set(critical_recs):
                print(f"  • {rec}")
        
        if high_recs:
            print("\nHIGH (should fix):")
            for rec in set(high_recs):
                print(f"  • {rec}")
        
        return {
            "total_findings": len(self.findings),
            "findings_by_category": categories,
            "findings": self.findings
        }

def main():
    """Función principal"""
    diagnostic = CCDDiagnostic(seed=42)
    results = diagnostic.run_all_diagnostics()
    
    # Guardar resultados
    with open('/home/Martínez\'s Invariant/ccd_diagnostic_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDiagnostic results saved to ccd_diagnostic_results.json")
    
    return results

if __name__ == "__main__":
    main()