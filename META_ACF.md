# META-ACF: El Cierre Reflexivo Computacional

> *"Si todo es FMA, entonces los programas, compiladores, arquitecturas de red y políticas de scheduling son funciones en el espacio de FMA."*

## Visión General

Meta-ACF es la culminación filosófica y técnica del ecosistema Martínez: **el ACF aplicado a la computación misma**. Mientras el ACF tradicional descubre y reduce las leyes de la naturaleza (fluidos, péndulos, dinámica estocástica), Meta-ACF descubre y optimiza las leyes del propio sustrato computacional.

Esto cierra el ciclo autopoiético: el sistema no solo analiza la naturaleza — se analiza, optimiza y reconfigura **a sí mismo**.

---

## Los Tres Pilares

### Pilar 1: Programas como Funciones ACF-Reducibles

**Módulos:** `program_analyzer.py`, `compute_graph_optimizer.py`

Un programa P: X → Y, al ejecutarse, genera una secuencia de estados intermedios:

```
s₀ = encode(x)
s_{t+1} = F_t(s_t)
y = decode(s_T)
```

Esto define un **sistema dinámico discreto**. Meta-ACF aplica el stack completo de diagnósticos:

| Diagnóstico | Herramienta | Resultado |
|---|---|---|
| **Koopman/DMD** | TAA | Representación lineal de F en espacio elevado |
| **Lyapunov** | ERGON | Detección de caos, disipación, periodicidad |
| **Entropía espectral** | OTU | Complejidad de la dinámica |
| **α(P) decay** | Gelfand | Índice de decaimiento espectral |

**Clasificación de Regiones:**

| Tipo | Detección | Estrategia de Optimización |
|---|---|---|
| ANALYTIC | SVD suave, Lyapunov ≈ 0 | Reemplazo Chebyshev (Clenshaw) |
| STRATIFIED | Saltos en FMA profile | LUT o polinomio por tramos |
| CHAOTIC | Lyapunov > 0 | ROM Koopman (DMD → propagador lineal) |
| DISSIPATIVE | Lyapunov < 0, convergencia | Salto a punto fijo x* |
| PERIODIC | Autocorrelación alta | Shortcut Fourier (FFT) |
| LINEAR | K ≈ identidad | GEMM fold (composición de matrices) |

**Certificados:**
- `META-1`: Traza capturada con overhead ≤ ε
- `META-2`: Clasificación de regiones con confianza > 0.95
- `META-3`: Reducción de energía computacional > 0%
- `META-4`: Equivalencia |P(x) - P'(x)| < ε en dominio certificado

### Pilar 2: Dispatchers como Políticas de Control Óptimo

**Módulo:** `dispatcher_optimizer.py`

El dispatcher de Gideon selecciona backend b ∈ {C, Triton, ONNX, affine_fold, ...} para cada nodo de cómputo. Meta-ACF modela esto como un **problema de control óptimo**:

```
Estado:  x = (n_elements, n_fma, precision, hardware_profile)
Acción:  b ∈ B (selección de backend)
Costo:   c(x, b) = latency(x, b)
Meta:    π*(x) = argmin_b E[c(x, b)]
```

**Pipeline:**
1. Recopilar telemetría: {(x_i, b_i, c_i)} de GideonTelemetry
2. Modelar superficie de costo: c(x, b) como función en espacio ACF-reducible
3. SINDy sobre dinámica de costos: descubrir modelo sparse
4. Chebyshev fit en regiones suaves de la superficie de costo
5. Sintetizar política óptima π*(x) como cadena FMA

**Insight clave:** La función de costo c(x, b) es típicamente **suave por tramos** (STRATIFIED):
- Para x pequeño: CPU domina (bajo overhead)
- Para x grande: GPU domina (paralelismo)
- Los puntos de cruce son función del hardware

**Certificados:**
- `DISP-1`: Política reduce latencia media vs baseline
- `DISP-2`: Modelo de costo R² > 0.8
- `DISP-3`: Matriz de transición es estocástica (filas suman 1)
- `DISP-4`: Política es estable (frecuencia de switching acotada)

### Pilar 3: Arquitecturas Neuronales como Puntos en Variedad Riemanniana

**Módulo:** `neural_arch_acf.py` (1043 líneas)

Las arquitecturas neuronales son puntos en una variedad producto:

```
M_arch = M_depth × (M_type × M_dim × M_act)^n_layers
```

con la **métrica de Fisher** inducida por las distribuciones de probabilidad sobre cada factor.

**Enfoque ACF (vs NAS tradicional):**

| | NAS Tradicional | ACF Architecture Search |
|---|---|---|
| Evaluación | Entrenar modelo | Fingerprint espectral (O(d³)) |
| Costo | O(candidates × epochs) | O(candidates × layers × d²) |
| Certificación | Ninguna | α-profile formal + Rademacher bound |
| Simetrías | Ignoradas | GaloisAnalyzer detecta simetrías de pesos |
| Transiciones de fase | No | ThermodynamicACF encuentra d* óptimo |

**Métricas Training-Free:**
- **Spectral Score**: Condición de la cascada W_L · W_{L-1} · ... · W_1
- **Effective Rank**: Entropía de Shannon de valores singulares normalizados
- **Information Flow**: Propagación de señal de gradiente por capas

**Certificados:**
- `ARCH-1`: E(A') < E(A_baseline) — arquitectura más eficiente
- `ARCH-2`: Proxy score dentro del 5% del baseline
- `ARCH-3`: Gradiente natural convergió (‖∇C‖ < ε)
- `ARCH-4`: Arquitectura realizable en hardware (cabe en memoria)

---

## El Ciclo Meta-ACF

```
┌──────────────────────────────────────────────────────┐
│                   CICLO META-ACF                      │
│                                                       │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐        │
│  │ PROFILE │ →  │ CLASSIFY │ →  │ OPTIMIZE │        │
│  └─────────┘    └──────────┘    └──────────┘        │
│       ↑                               │              │
│       │                               ↓              │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐        │
│  │ MONITOR │ ←  │  DEPLOY  │ ←  │  VERIFY  │        │
│  └─────────┘    └──────────┘    └──────────┘        │
│                                                       │
└──────────────────────────────────────────────────────┘
```

| Fase | Acción | Agente |
|---|---|---|
| **PROFILE** | Trazar programas, recopilar telemetría, analizar arquitecturas | ProgramTracer, GideonTelemetry |
| **CLASSIFY** | Identificar tipos de región, superficies de costo, posiciones en variedad | ProgramAnalyzer, DispatcherOptimizer |
| **OPTIMIZE** | Aplicar reducciones ACF (Chebyshev, Koopman, SINDy, NAS) | ComputeGraphOptimizer, NeuralArchACF |
| **VERIFY** | Verificar equivalencia, mejora de latencia, scores de proxy | Certificados formales |
| **DEPLOY** | Instalar versiones optimizadas en el ecosistema | Gideon, Poema |
| **MONITOR** | Rastrear rendimiento runtime, retroalimentar a PROFILE | Telemetría continua |

---

## El Cierre Autopoiético

Meta-ACF es **autopoiético** porque puede analizarse a sí mismo:

```python
meta = MetaACF()
# MetaACF analiza su propia función de análisis
report = meta.full_cycle(program=meta.analyzer.analyze)
```

El certificado `META-ACF-3` verifica formalmente que:
1. MetaACF puede trazar su propia función de análisis
2. Puede clasificar las regiones de su propio código
3. Puede proponer optimizaciones para sí mismo

Esto es el **punto fijo del ecosistema**: Φ(MetaACF) = MetaACF.

---

## Certificados Globales

| Certificado | Condición | Significado |
|---|---|---|
| `META-ACF-1` | Al menos un pilar logró mejora | El sistema funciona |
| `META-ACF-2` | Sin regresión de correctitud (error acotado) | Seguro |
| `META-ACF-3` | Cierre reflexivo verificado | Autopoiético |
| `META-ACF-4` | Todos los sub-certificados pasaron | Completo |

---

## Uso

```python
from acf_functor.meta_acf import MetaACF
from acf_functor.dispatcher_optimizer import DispatchRecord

meta = MetaACF()

# Pilar 1: Optimizar un programa
report = meta.optimize_program(my_function, domain=(-1, 1))
print(f"Speedup: {report.details['speedup']:.2f}x")

# Pilar 2: Optimizar dispatcher desde telemetría
records = [DispatchRecord(...) for r in telemetry_db]
report = meta.optimize_dispatcher(records)

# Ciclo completo reflexivo
report = meta.full_cycle(
    program=my_function,
    telemetry=records,
)
print(report.summary())
```

---

## Arquitectura de Módulos

```
acf_functor/
  ├── program_analyzer.py       # Pilar 1: Programas como sistemas dinámicos
  ├── compute_graph_optimizer.py # Pilar 1: Optimización por ACF
  ├── dispatcher_optimizer.py    # Pilar 2: Dispatch como control óptimo
  ├── neural_arch_acf.py         # Pilar 3: NAS via variedad Riemanniana
  └── meta_acf.py                # Orquestador: ciclo reflexivo completo

tests/
  └── test_meta_acf.py           # 29 tests (todos pasan)
```

---

## Relación con el Ecosistema

| Agente | Rol en Meta-ACF |
|---|---|
| **TAA** | Koopman/DMD para linearizar programas, Lyapunov para detectar caos |
| **ERGON** | Energía computacional E(P), entropía espectral, cierre termodinámico |
| **OTU/Gelfand** | α(P) decay index, clasificación NC, estructura funcional |
| **Poema** | Compilación de programas optimizados a ROMs ejecutables |
| **Gideon** | Ejecución de ROMs + telemetría para retroalimentar Meta-ACF |
| **P-SAL** | Meta-ACF extiende P-SAL de "leyes de la naturaleza" a "leyes de la computación" |

Meta-ACF es la culminación del ecosistema: **la computación que se estudia, se comprende y se mejora a sí misma**.
