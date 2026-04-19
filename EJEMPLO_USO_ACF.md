# Ejemplo de Uso del Sistema ACF Mejorado

## 1. Configuración Rápida

```python
# Importar el sistema ACF mejorado
from poema.acf_integration import CompilationPipeline
from poema.pure_fma_repair import PureFMAAutoDomainRepair, FMAPrecision
from poema.genesis_auto_prover import GenesisAutoProver

# Crear pipeline avanzado con optimizaciones cuánticas
pipeline = CompilationPipeline.create_advanced_pipeline(
    quantum_optimized=True,
    rl_enabled=True
)
```

## 2. Ejemplo Básico: Compilación de Código Poema

```python
# Código Poema de ejemplo
poema_code = """
def transformada_fourier(x):
    # Transformada rápida con funciones matemáticas
    componente_real = cos(2 * 3.14159 * 100 * x)
    componente_imag = sin(2 * 3.14159 * 100 * x)
    return componente_real + 1j * componente_imag

def filtro_gaussiano(senal, sigma=1.0):
    # Filtro Gaussiano para procesamiento de señal
    kernel = exp(-(senal**2) / (2 * sigma**2))
    return kernel / (sigma * sqrt(2 * 3.14159))
"""

# Compilar con descubrimiento automático
report = pipeline.compile_poema_code(
    poema_code,
    optimize=True,      # Aplicar optimizaciones
    certify=True,       # Generar certificados Lean
    discover=True,      # Ejecutar descubrimiento
    discovery_depth=3   # Nivel intermedio de descubrimiento
)

print(f"✅ Compilación completada en {report['execution_time']:.2f}s")
print(f"📊 Descubrimientos: {len(report['discoveries_made'])}")
print(f"⚡ Optimizaciones: {report.get('optimizations_applied', 0)}")
```

## 3. Ejemplo Avanzado: Descubrimiento Matemático Profundo

```python
# Crear motor de descubrimiento Genesis
genesis = GenesisAutoProver(
    enable_quantum=True,
    enable_rl=True
)

# Funciones matemáticas para analizar
funciones_complejas = [
    "sin(x) * cos(x)",
    "exp(-x^2)",                    # Campana de Gauss
    "tanh(x)",                      # Función de activación
    "log(1 + exp(x))",              # Softplus
    "sin(x)^3 + cos(x)^3",
]

# Ejecutar descubrimiento profundo
print("🔍 Iniciando descubrimiento matemático profundo...")
resultado = genesis.run_discovery_pipeline(
    funciones_complejas,
    domain=(-5, 5),          # Dominio de análisis
    depth=4,                 # Nivel experto
    generate_proofs=True     # Generar teoremas Lean
)

# Analizar resultados
print(f"\n📈 Resultados del descubrimiento:")
print(f"   Patrones encontrados: {len(resultado['discoveries'])}")
print(f"   Teoremas generados: {resultado['theorems_generated']}")
print(f"   Teoremas compilados: {resultado['theorems_compiled']}")

# Mostrar descubrimientos destacados
for i, descubrimiento in enumerate(resultado['discoveries'][:5]):
    print(f"\n   {i+1}. {descubrimiento['expression']}")
    print(f"      Confianza: {descubrimiento['confidence']:.3f}")
    print(f"      Categoría: {descubrimiento['category']}")
```

## 4. Ejemplo: Alta Precisión Numérica

```python
# Crear pipeline de ultra alta precisión
pipeline_alta_precision = CompilationPipeline.create_high_precision_pipeline()

# Código que requiere máxima precisión
codigo_precision = """
def calcular_constantes_fundamentales():
    # Cálculos de alta precisión para constantes matemáticas
    pi_aproximado = 0.0
    for k in range(1, 10000):
        pi_aproximado += 1.0 / (k * k)
    pi_aproximado = sqrt(6 * pi_aproximado)
    
    # Constante de Euler
    e_aproximado = sum(1.0 / math.factorial(n) for n in range(20))
    
    return pi_aproximado, e_aproximado
"""

# Compilar con garantías de precisión
reporte_precision = pipeline_alta_precision.compile_poema_code(
    codigo_precision,
    optimize=True,
    certify=True,
    discover=True
)

print(f"🎯 Pipeline de alta precisión:")
print(f"   Precisión: {pipeline_alta_precision.target_precision.name}")
print(f"   Métricas: {reporte_precision.get('performance_metrics', {})}")
```

## 5. Ejemplo: Reparación de Dominios con Pure-FMA

```python
# Crear sistema de reparación avanzado
reparador = PureFMAAutoDomainRepair(
    precision=FMAPrecision.QUADRUPLE,  # Máxima precisión
    enable_quantum_opt=True            # Optimizaciones cuánticas
)

# Evaluar funciones en puntos problemáticos
puntos_criticos = [-10.0, -5.0, 0.0, 5.0, 10.0, 100.0]

print("🧮 Evaluación con reparación Pure-FMA:")
for funcion in ["sin", "exp", "tanh"]:
    print(f"\n  Función: {funcion}(x)")
    
    # Crear evaluador con reparación automática
    evaluador = reparador.create_pure_repair_evaluator(
        funcion,
        original_coeffs=[],  # Coeficientes se generan automáticamente
        original_domain=(-2, 2)
    )
    
    for x in puntos_criticos:
        valor, error = evaluador(x)
        print(f"    {funcion}({x:6.1f}) ≈ {valor:12.6e} ± {error:8.2e}")
```

## 6. Ejemplo: Integración Completa con Monitoreo

```python
# Pipeline completo con monitoreo avanzado
pipeline_completo = CompilationPipeline.create_advanced_pipeline()

# Código científico complejo
codigo_cientifico = """
def simulacion_cuantica(psi, tiempo, hamiltoniano):
    # Simulación de evolución cuántica
    import cmath
    
    # Operador de evolución
    U = exp(-1j * hamiltoniano * tiempo)
    
    # Aplicar evolución
    psi_evolucionado = U @ psi
    
    # Calcular valores esperados
    energia = real(conj(psi_evolucionado) @ hamiltoniano @ psi_evolucionado)
    
    return psi_evolucionado, energia

def analisis_espectral(senal, fs):
    # Análisis espectral con FFT
    N = len(senal)
    frecuencias = fftfreq(N, 1/fs)
    espectro = fft(senal)
    
    # Encontrar picos espectrales
    picos = where(abs(espectro) > 0.5 * max(abs(espectro)))[0]
    
    return frecuencias[picos], abs(espectro[picos])
"""

# Ejecutar pipeline completo
print("🚀 Ejecutando pipeline ACF completo...")
resultado_final = pipeline_completo.compile_poema_code(
    codigo_cientifico,
    optimize=True,
    certify=True,
    discover=True,
    discovery_depth=4
)

# Obtener reporte detallado
reporte_detallado = pipeline_completo.get_detailed_report()

print(f"\n📋 Reporte detallado:")
print(f"   Configuración: {reporte_detallado['pipeline_config']}")
print(f"   Descubrimientos por categoría: {reporte_detallado['discoveries_summary']['by_category']}")
print(f"   Eficiencia de cache: {reporte_detallado['optimizations_summary']['cache_efficiency']:.2%}")

# Exportar resultados
pipeline_completo.export_report("resultado_compilacion.json")
print("💾 Resultados exportados a 'resultado_compilacion.json'")
```

## 7. Ejemplo Rápido: Validación del Sistema

```python
# Script de validación rápida
def validar_sistema_acf():
    """Valida que todos los componentes del sistema ACF funcionen."""
    
    print("🧪 Validando sistema ACF mejorado...")
    
    # 1. Validar Pure-FMA
    try:
        from poema.pure_fma_repair import PureFMAAutoDomainRepair
        reparador = PureFMAAutoDomainRepair()
        print("✅ Pure-FMA: OK")
    except Exception as e:
        print(f"❌ Pure-FMA: Error - {e}")
    
    # 2. Validar Genesis
    try:
        from poema.genesis_auto_prover import GenesisAutoProver
        genesis = GenesisAutoProver()
        print("✅ Genesis Auto-Prover: OK")
    except Exception as e:
        print(f"❌ Genesis: Error - {e}")
    
    # 3. Validar Pipeline
    try:
        from poema.acf_integration import CompilationPipeline
        pipeline = CompilationPipeline.create_advanced_pipeline()
        print("✅ Pipeline de integración: OK")
    except Exception as e:
        print(f"❌ Pipeline: Error - {e}")
    
    print("\n🎉 Validación completada!")

# Ejecutar validación
if __name__ == "__main__":
    validar_sistema_acf()
```

## 8. Consejos de Uso Avanzado

### 8.1 Optimización de Rendimiento
```python
# Para máximo rendimiento
pipeline_rapido = CompilationPipeline.create_advanced_pipeline(
    quantum_optimized=False,  # Desactivar optimizaciones pesadas
    rl_enabled=False          # Desactivar aprendizaje
)

# Para máxima precisión
pipeline_preciso = CompilationPipeline.create_high_precision_pipeline()
```

### 8.2 Personalización de Descubrimiento
```python
genesis_personalizado = GenesisAutoProver(
    enable_quantum=True,      # Habilitar descubrimiento cuántico
    enable_rl=True,           # Aprendizaje por refuerzo
    # Configuración adicional...
)

# Dominios personalizados para diferentes aplicaciones
dominios_especializados = {
    "fisica": (-10, 10),      # Física: rango amplio
    "finanzas": (0, 100),     # Finanzas: valores positivos
    "procesamiento": (-1, 1), # Procesamiento de señal: normalizado
}
```

### 8.3 Exportación de Resultados
```python
# Exportar en diferentes formatos
pipeline.export_report("resultados.json")          # JSON completo
pipeline.export_report("resultados_resumido.md")   # Markdown resumido

# Acceder a datos específicos
reporte = pipeline.get_detailed_report()
descubrimientos = reporte["discoveries_summary"]
metricas = reporte["performance_metrics"]
```

## 9. Solución de Problemas Comunes

### 9.1 Error: "Módulo no encontrado"
```bash
# Instalar dependencias opcionales
pip install numpy sympy

# Para funcionalidades avanzadas (opcional)
pip install scipy scikit-learn
```

### 9.2 Error: "Memoria insuficiente"
```python
# Reducir profundidad de descubrimiento
reporte = pipeline.compile_poema_code(
    codigo,
    discovery_depth=2  # En lugar de 4 o 5
)

# Limitar número de funciones analizadas
funciones_limitadas = funciones[:10]  # Analizar solo 10 funciones
```

### 9.3 Error: "Tiempo de ejecución excesivo"
```python
# Usar configuración optimizada para velocidad
pipeline_rapido = CompilationPipeline.create_advanced_pipeline(
    quantum_optimized=False,
    rl_enabled=False
)

# Desactivar certificación para pruebas rápidas
reporte_rapido = pipeline.compile_poema_code(
    codigo,
    certify=False,      # Sin generación de teoremas Lean
    discover=True,
    discovery_depth=2   # Nivel básico
)
```

## 10. Recursos Adicionales

- **Documentación completa**: `MEJORAS_ACF_AVANZADO.md`
- **Pruebas del sistema**: `test_advanced_acf.py`
- **Ejemplos adicionales**: Directorio `examples/`
- **Reportes de compilación**: Archivos `.json` generados automáticamente

---

**Sistema ACF Mejorado** - Versión 2.0  
**Fecha**: 8 de abril de 2026  
**Estado**: Listo para producción científica y de ingeniería