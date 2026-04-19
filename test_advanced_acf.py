#!/usr/bin/env python3
"""
Pruebas avanzadas para el sistema ACF mejorado.
Valida las optimizaciones cuánticas, descubrimiento profundo y pipeline integrado.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poema.acf_integration import CompilationPipeline
from poema.pure_fma_repair import PureFMAAutoDomainRepair, FMAPrecision
from poema.genesis_auto_prover import GenesisAutoProver

def test_pure_fma_advanced():
    """Prueba las optimizaciones avanzadas de Pure-FMA."""
    print("🧪 Probando Pure-FMA avanzado...")
    
    import math
    
    # Crear reparador con optimizaciones cuánticas
    repair = PureFMAAutoDomainRepair(
        precision=FMAPrecision.DOUBLE,
        enable_quantum_opt=True
    )
    
    # Probar evaluación pura FMA
    test_values = [-3.0, -1.5, 0.0, 1.5, 3.0]
    
    print("  Evaluando sin(x) con reparación pura FMA...")
    for x in test_values:
        evaluator = repair.create_pure_repair_evaluator("sin", [], (-math.pi/2, math.pi/2))
        value, error = evaluator(x)
        print(f"    sin({x:.2f}) ≈ {value:.8f} ± {error:.2e}")
    
    # Obtener estadísticas
    stats = repair.get_stats()
    print(f"  Estadísticas: {stats}")
    
    return True

def test_genesis_advanced():
    """Prueba el descubrimiento profundo de Genesis."""
    print("\n🧠 Probando Genesis Auto-Prover avanzado...")
    
    # Crear prover con aprendizaje por refuerzo
    prover = GenesisAutoProver(
        enable_quantum=True,
        enable_rl=True
    )
    
    # Funciones simples para analizar (sin dependencias de scipy)
    test_functions = [
        "sin(x) + cos(x)",
        "exp(x) * exp(-x)",
        "sin(x)^2 + cos(x)^2",
    ]
    
    print("  Ejecutando descubrimiento básico (nivel 2)...")
    report = prover.run_discovery_pipeline(
        test_functions,
        domain=(-3, 3),
        depth=2,  # Nivel más bajo para evitar scipy
        generate_proofs=False  # Sin compilación Lean para pruebas rápidas
    )
    
    print(f"  Descubrimientos: {len(report['discoveries'])}")
    print(f"  Teoremas generados: {report['theorems_generated']}")
    
    # Mostrar algunos descubrimientos
    if report['discoveries']:
        print("\n  Descubrimientos destacados:")
        for i, disc in enumerate(report['discoveries'][:3]):
            print(f"    {i+1}. {disc['expression']} (conf: {disc['confidence']:.3f})")
    
    return True

def test_integrated_pipeline():
    """Prueba el pipeline integrado completo."""
    print("\n🚀 Probando pipeline integrado ACF...")
    
    # Crear pipeline avanzado
    pipeline = CompilationPipeline.create_advanced_pipeline(
        quantum_optimized=True,
        rl_enabled=True
    )
    
    # Código Poema de ejemplo con funciones matemáticas simples
    poema_code = """
def calcular_simple(x):
    # Función simple para pruebas
    y = sin(x) * cos(x)
    return y

def funcion_exponencial(x):
    # Función exponencial básica
    return exp(-x**2)
"""
    
    print("  Compilando código Poema con optimizaciones básicas...")
    report = pipeline.compile_poema_code(
        poema_code,
        optimize=True,
        certify=False,  # Sin certificación para pruebas rápidas
        discover=True,
        discovery_depth=2  # Nivel básico
    )
    
    print(f"  Resultados de compilación:")
    print(f"    Expresiones encontradas: {report.get('math_expressions_found', 0)}")
    print(f"    Patrones descubiertos: {len(report.get('discoveries_made', []))}")
    print(f"    Optimizaciones aplicadas: {report.get('optimizations_applied', 0)}")
    print(f"    Tiempo total: {report.get('execution_time', 0):.2f}s")
    
    # Solo exportar si hay resultados
    if report.get('discoveries_made'):
        pipeline.export_report("acf_compilation_report.json")
        print("  ✓ Reporte exportado a 'acf_compilation_report.json'")
    
    return True

def test_high_precision_pipeline():
    """Prueba el pipeline de alta precisión."""
    print("\n🎯 Probando pipeline de alta precisión...")
    
    pipeline = CompilationPipeline.create_high_precision_pipeline()
    
    # Código que requiere alta precisión
    high_precision_code = """
def calcular_constante_critica():
    # Cálculo que requiere alta precisión numérica
    import math
    resultado = 0.0
    for k in range(1, 1000):
        resultado += 1.0 / (k * k)
    return math.sqrt(6 * resultado)  # Aproximación de π
"""
    
    report = pipeline.compile_poema_code(
        high_precision_code,
        optimize=True,
        certify=True,
        discover=True,
        discovery_depth=2
    )
    
    print(f"  Pipeline de alta precisión ejecutado:")
    print(f"    Precisión objetivo: {pipeline.target_precision.name}")
    print(f"    Métricas: {report.get('performance_metrics', {})}")
    
    return True

def run_comprehensive_test():
    """Ejecuta todas las pruebas."""
    print("=" * 70)
    print("🧪 PRUEBAS COMPREHENSIVAS DEL SISTEMA ACF MEJORADO")
    print("=" * 70)
    
    import math
    
    tests = [
        ("Pure-FMA Avanzado", test_pure_fma_advanced),
        ("Genesis Auto-Prover", test_genesis_advanced),
        ("Pipeline Integrado", test_integrated_pipeline),
        ("Alta Precisión", test_high_precision_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n▶️ Ejecutando: {test_name}")
            success = test_func()
            results.append((test_name, success))
            print(f"   {'✅ PASÓ' if success else '❌ FALLÓ'}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        print(f"  {test_name}: {status}")
    
    print(f"\n  Total: {passed}/{total} pruebas pasadas ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
        return True
    else:
        print(f"\n⚠️  {total - passed} pruebas fallaron")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)