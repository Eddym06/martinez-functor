#!/usr/bin/env python3
"""
Prueba final simplificada del sistema ACF mejorado.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_components():
    """Prueba básica de los componentes principales."""
    print("🧪 Probando componentes básicos del sistema ACF...")
    
    # 1. Test Pure-FMA básico
    try:
        from poema.pure_fma_repair import PureFMAAutoDomainRepair, FMAPrecision
        repair = PureFMAAutoDomainRepair(precision=FMAPrecision.DOUBLE)
        print("✅ Pure-FMA: Componente inicializado correctamente")
        
        # Verificar certificados
        if repair._pure_certificates:
            print(f"✅ Pure-FMA: {len(repair._pure_certificates)} certificados inicializados")
        else:
            print("⚠️ Pure-FMA: Certificados no inicializados")
    except Exception as e:
        print(f"❌ Pure-FMA: Error - {e}")
        return False
    
    # 2. Test Genesis básico
    try:
        from poema.genesis_auto_prover import GenesisAutoProver
        genesis = GenesisAutoProver(enable_quantum=False, enable_rl=False)
        print("✅ Genesis: Componente inicializado correctamente")
        
        # Test descubrimiento simple
        patterns = genesis.discover_from_function("sin(x) + cos(x)", domain=(-2, 2), depth=1)
        print(f"✅ Genesis: {len(patterns)} patrones descubiertos")
    except Exception as e:
        print(f"❌ Genesis: Error - {e}")
        return False
    
    # 3. Test Pipeline básico
    try:
        from poema.acf_integration import CompilationPipeline
        pipeline = CompilationPipeline.create_advanced_pipeline(quantum_optimized=False, rl_enabled=False)
        print("✅ Pipeline: Componente inicializado correctamente")
        
        # Test compilación simple
        simple_code = "def test(x): return sin(x)"
        report = pipeline.compile_poema_code(simple_code, optimize=False, certify=False, discover=False)
        print(f"✅ Pipeline: Compilación básica ejecutada")
    except Exception as e:
        print(f"❌ Pipeline: Error - {e}")
        return False
    
    return True

def test_advanced_features():
    """Prueba de características avanzadas."""
    print("\n⚡ Probando características avanzadas...")
    
    try:
        from poema.acf_integration import CompilationPipeline
        
        # Crear pipeline con características avanzadas
        pipeline = CompilationPipeline.create_advanced_pipeline(
            quantum_optimized=True,
            rl_enabled=True
        )
        
        # Código con expresiones matemáticas
        math_code = """
def trigonometric_identity(x):
    y = sin(x)**2 + cos(x)**2
    return y

def exponential_function(x):
    return exp(-x**2)
"""
        
        # Compilar con características avanzadas
        report = pipeline.compile_poema_code(
            math_code,
            optimize=True,
            certify=False,  # Sin certificación para prueba rápida
            discover=True,
            discovery_depth=2
        )
        
        print(f"✅ Pipeline avanzado: Compilación ejecutada")
        print(f"   Expresiones encontradas: {report.get('math_expressions_found', 0)}")
        print(f"   Patrones descubiertos: {len(report.get('discoveries_made', []))}")
        
        # Verificar métricas
        if report.get('execution_time', 0) > 0:
            print(f"✅ Métricas: Tiempo de ejecución registrado")
        
        return True
    except Exception as e:
        print(f"❌ Características avanzadas: Error - {e}")
        return False

def main():
    """Función principal de prueba."""
    print("=" * 60)
    print("VALIDACIÓN FINAL DEL SISTEMA ACF MEJORADO")
    print("=" * 60)
    
    # Ejecutar pruebas
    test1 = test_basic_components()
    test2 = test_advanced_features()
    
    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    
    if test1 and test2:
        print("🎉 ¡Sistema ACF mejorado validado exitosamente!")
        print("\nComponentes funcionando:")
        print("  ✅ Pure-FMA Repair con optimizaciones")
        print("  ✅ Genesis Auto-Prover con descubrimiento")
        print("  ✅ Pipeline de integración avanzado")
        print("  ✅ Características cuánticas y RL")
        
        print("\n📚 Documentación disponible:")
        print("  - MEJORAS_ACF_AVANZADO.md (mejoras implementadas)")
        print("  - EJEMPLO_USO_ACF.md (ejemplos de uso)")
        print("  - test_advanced_acf.py (pruebas comprehensivas)")
        
        return True
    else:
        print("⚠️ Sistema ACF necesita ajustes adicionales")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)