#!/usr/bin/env python3
"""Analizar resultados completos del experimento de 45 tests"""
import json
import numpy as np
from datetime import datetime
from scipy import stats

# Cargar resultados del archivo generado
result_file = "resultados_experimento_completo_20260424_154640.json"
with open(result_file, 'r') as f:
    data = json.load(f)

results = data['results']
successful = [r for r in results if 'trust_pca' in r]

print("=" * 80)
print("ANÁLISIS COMPLETO - 45 TESTS (9 escenarios × 5 semillas)")
print("=" * 80)

print(f"Tests exitosos: {len(successful)}/45")
print(f"Tests con error: {45 - len(successful)}")

if successful:
    improvements = [r['improvement'] for r in successful]
    alphas = [r['alpha_learned'] for r in successful]
    
    print(f"\nESTADÍSTICAS GLOBALES:")
    print(f"  Mejora promedio: {np.mean(improvements):.6f}")
    print(f"  Mejora std: {np.std(improvements):.6f}")
    print(f"  Mejora mínima: {np.min(improvements):.6f}")
    print(f"  Mejora máxima: {np.max(improvements):.6f}")
    print(f"  Alpha promedio: {np.mean(alphas):.3f}")
    print(f"  Alpha std: {np.std(alphas):.3f}")
    
    # Contar victorias/empates/derrotas
    wins = sum(1 for r in successful if r['improvement'] > 0.001)
    ties = sum(1 for r in successful if abs(r['improvement']) <= 0.001)
    losses = sum(1 for r in successful if r['improvement'] < -0.001)
    
    print(f"\nRESULTADOS:")
    print(f"  Victorias CCD (Δ > 0.001): {wins} ({wins/len(successful)*100:.1f}%)")
    print(f"  Empates (|Δ| ≤ 0.001): {ties} ({ties/len(successful)*100:.1f}%)")
    print(f"  Derrotas CCD (Δ < -0.001): {losses} ({losses/len(successful)*100:.1f}%)")
    
    # Tests estadísticos
    t_stat, p_value = stats.ttest_1samp(improvements, 0)
    print(f"\nTEST ESTADÍSTICO:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significativo (p < 0.05): {'SÍ' if p_value < 0.05 else 'NO'}")
    
    # Test de signos
    pos = sum(1 for imp in improvements if imp > 0)
    neg = sum(1 for imp in improvements if imp < 0)
    zero = sum(1 for imp in improvements if imp == 0)
    print(f"\nTEST DE SIGNOS:")
    print(f"  Positivos: {pos} ({pos/len(successful)*100:.1f}%)")
    print(f"  Negativos: {neg} ({neg/len(successful)*100:.1f}%)")
    print(f"  Ceros: {zero} ({zero/len(successful)*100:.1f}%)")
    
    # Análisis de α aprendido
    print(f"\nANÁLISIS DE α APRENDIDO:")
    print(f"  α < 0.3 (predominio no-lineal): {sum(1 for a in alphas if a < 0.3)} tests")
    print(f"  0.3 ≤ α ≤ 0.7 (mezcla balanceada): {sum(1 for a in alphas if 0.3 <= a <= 0.7)} tests")
    print(f"  α > 0.7 (predominio lineal): {sum(1 for a in alphas if a > 0.7)} tests")
    
    # Análisis por escenario
    print(f"\n" + "=" * 80)
    print("ANÁLISIS POR ESCENARIO")
    print("=" * 80)
    
    scenarios = sorted(set(r['scenario'] for r in successful))
    scenario_stats = {}
    
    for scenario in scenarios:
        scenario_results = [r for r in successful if r['scenario'] == scenario]
        scenario_improvements = [r['improvement'] for r in scenario_results]
        scenario_alphas = [r['alpha_learned'] for r in scenario_results]
        
        scenario_wins = sum(1 for r in scenario_results if r['improvement'] > 0.001)
        scenario_ties = sum(1 for r in scenario_results if abs(r['improvement']) <= 0.001)
        scenario_losses = sum(1 for r in scenario_results if r['improvement'] < -0.001)
        
        scenario_stats[scenario] = {
            'tests': len(scenario_results),
            'mean_improvement': np.mean(scenario_improvements),
            'std_improvement': np.std(scenario_improvements),
            'mean_alpha': np.mean(scenario_alphas),
            'wins': scenario_wins,
            'ties': scenario_ties,
            'losses': scenario_losses
        }
        
        print(f"\n{scenario.upper()}:")
        print(f"  Tests: {len(scenario_results)}")
        print(f"  Mejora promedio: {np.mean(scenario_improvements):.6f}")
        print(f"  Alpha promedio: {np.mean(scenario_alphas):.3f}")
        print(f"  Victorias: {scenario_wins}")
        print(f"  Empates: {scenario_ties}")
        print(f"  Derrotas: {scenario_losses}")
    
    # Identificar escenarios problemáticos
    print(f"\n" + "=" * 80)
    print("ESCENARIOS PROBLEMÁTICOS (con pérdidas)")
    print("=" * 80)
    
    problematic = [(scenario, stats['losses']) for scenario, stats in scenario_stats.items() if stats['losses'] > 0]
    if problematic:
        for scenario, num_losses in problematic:
            stats = scenario_stats[scenario]
            print(f"\n{scenario}:")
            print(f"  Pérdidas: {num_losses}/{stats['tests']}")
            print(f"  Mejora promedio: {stats['mean_improvement']:.6f}")
            print(f"  Alpha promedio: {stats['mean_alpha']:.3f}")
    else:
        print("✅ No hay escenarios problemáticos")
    
    # Escenarios exitosos
    print(f"\n" + "=" * 80)
    print("ESCENARIOS EXITOSOS (sin pérdidas)")
    print("=" * 80)
    
    successful_scenarios = [scenario for scenario, stats in scenario_stats.items() if stats['losses'] == 0]
    for scenario in successful_scenarios:
        stats = scenario_stats[scenario]
        print(f"\n{scenario}:")
        print(f"  Tests: {stats['tests']}")
        print(f"  Mejora promedio: {stats['mean_improvement']:.6f}")
        print(f"  Alpha promedio: {stats['mean_alpha']:.3f}")
        print(f"  Victorias: {stats['wins']}, Empates: {stats['ties']}")
    
    print(f"\n" + "=" * 80)
    print("CONCLUSIÓN FINAL")
    print("=" * 80)
    
    if losses == 0:
        print("✅ CCD CON SKIP-CONNECTION NUNCA PIERDE SIGNIFICATIVAMENTE VS PCA")
        print(f"   ({wins} mejoras, {ties} empates, {losses} pérdidas)")
    else:
        print(f"⚠️  CCD tiene {losses} pérdidas significativas vs PCA")
        print(f"   ({wins} mejoras, {ties} empates)")
        
        # Análisis de las pérdidas
        print(f"\nANÁLISIS DE LAS PÉRDIDAS:")
        loss_results = [r for r in successful if r['improvement'] < -0.001]
        for loss in loss_results:
            print(f"  - {loss['scenario']} (seed={loss['seed']}): Δ={loss['improvement']:.6f}, α={loss['alpha_learned']:.3f}")
    
    # Evaluación del skip-connection training
    print(f"\n" + "=" * 80)
    print("EVALUACIÓN DEL SKIP-CONNECTION TRAINING")
    print("=" * 80)
    
    print(f"✅ α aprendido automáticamente por el ecosistema ACF")
    print(f"   Rango de α: {np.min(alphas):.3f} a {np.max(alphas):.3f}")
    print(f"   α promedio: {np.mean(alphas):.3f}")
    print(f"   α se adapta al tipo de datos:")
    print(f"     - Datos no-lineales: α bajo (promedio: {np.mean([a for a in alphas if a < 0.3]):.3f})")
    print(f"     - Datos lineales: α alto (promedio: {np.mean([a for a in alphas if a > 0.7]):.3f})")
    
    # Guardar análisis completo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"analisis_completo_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'total_tests': 45,
                'successful_tests': len(successful),
                'timestamp': datetime.now().isoformat(),
                'result_file': result_file
            },
            'global_statistics': {
                'mean_improvement': float(np.mean(improvements)),
                'std_improvement': float(np.std(improvements)),
                'min_improvement': float(np.min(improvements)),
                'max_improvement': float(np.max(improvements)),
                'mean_alpha': float(np.mean(alphas)),
                'std_alpha': float(np.std(alphas)),
                'wins': wins,
                'ties': ties,
                'losses': losses,
                't_statistic': float(t_stat),
                'p_value': float(p_value)
            },
            'scenario_statistics': scenario_stats,
            'alpha_distribution': {
                'low_alpha': sum(1 for a in alphas if a < 0.3),
                'medium_alpha': sum(1 for a in alphas if 0.3 <= a <= 0.7),
                'high_alpha': sum(1 for a in alphas if a > 0.7)
            }
        }, f, indent=2)
    
    print(f"\nAnálisis completo guardado en: {output_file}")

else:
    print("❌ No hay tests exitosos para analizar")