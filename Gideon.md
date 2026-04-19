# Gideon — Motor Unificado de Poema

> **"Gideon no es una librería. Es el corazón que late bajo cada programa que Poema ejecuta."**

---

## Índice

1. [¿Qué es Gideon?](#qué-es-gideon)
2. [Arquitectura General](#arquitectura-general)
3. [Los Cinco Subsistemas](#los-cinco-subsistemas)
4. [GideonIR — La Representación Intermedia](#gideonir--la-representación-intermedia)
5. [GideonGraph — El Grafo de Cómputo](#gideongraph--el-grafo-de-cómputo)
6. [GideonDispatcher — El Cerebro de Despacho](#gideondispatcher--el-cerebro-de-despacho)
7. [GideonEngine — El Motor Principal](#gideonengine--el-motor-principal)
8. [GideonAgentRouter — El Enrutador de Agentes ACF](#gideonagentrouter--el-enrutador-de-agentes-acf)
9. [Bases Futuras: IA y Descubrimiento de Teoremas](#bases-futuras-ia-y-descubrimiento-de-teoremas)
10. [Integración Nativa con Poema](#integración-nativa-con-poema)
11. [Rendimiento Medido (Hardware Real)](#rendimiento-medido-hardware-real)
12. [Diagrama de Flujo Completo](#diagrama-de-flujo-completo)
13. [Propiedades Matemáticas Garantizadas](#propiedades-matemáticas-garantizadas)
14. [Estado y Roadmap](#estado-y-roadmap)

---

## ¿Qué es Gideon?

**Gideon** es el motor de ejecución unificado de Poema. Su función es actuar como el sistema nervioso central que conecta el compilador frontend de Poema con los backends de ejecución de hardware (C/AVX2, GPU, ONNX, WebAssembly, FPGA/Verilog).

Gideon es **nativo para Poema**: el compilador `PoemCompiler` lo integra como target de primera clase (`target="gideon"`), y toda la arquitectura de backends existente —`CNativeEngine`, `PytorchBackend`, `ONNXBackend`, `WasmBackend`, `VerilogBackend`— pasa a ser orquestada por Gideon en lugar de ser invocada directamente.

### ¿Por qué se necesita un motor unificado?

Antes de Gideon, Poema tenía backends poderosos pero **desconectados**: compilabas con PyTorch o con C nativo, pero no había un sistema central que:

- Eligiera el backend óptimo según el hardware y carga de trabajo.
- Propagara formalmente las cotas de error a través del grafo.
- Detectara patrones fusables para optimizar el número real de instrucciones.
- Sirviera de puente hacia futuros módulos de IA y descubrimiento matemático.

Gideon resuelve exactamente eso.

---

## Arquitectura General

```
┌───────────────────────────────────────────────────────────┐
│                    Frontend de Poema                      │
│   Poem / CoPoem / BiPoem   →   AST de Poema               │
└───────────────────────┬───────────────────────────────────┘
                        │ ASTNode / FMASequence
                        ▼
┌───────────────────────────────────────────────────────────┐
│                    PoemCompiler                           │
│   target="gideon"   →   phases 1-5 (simplify, check,     │
│   domain_guard, linearize, error_bound)                   │
│                        │ FMA sequence + ε certified       │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
╔═══════════════════════════════════════════════════════════╗
║                    G I D E O N                           ║
║  ┌─────────────┐  ┌───────────────┐  ┌────────────────┐  ║
║  │  GideonIR   │→ │  GideonGraph  │→ │GideonDispatcher│  ║
║  │  (lowering) │  │  (topología)  │  │  (despacho)    │  ║
║  └─────────────┘  └───────────────┘  └────────┬───────┘  ║
║                                               │           ║
║  ┌─────────────────────────────────────────────▼────────┐  ║
║  │                  GideonEngine                        │  ║
║  │  compile → execute → telemetry → theorem_seeds      │  ║
║  └─────────────────────────────────────────────────────┘  ║
║                                                           ║
║  ┌─────────────────────────────────────────────────────┐  ║
║  │              GideonAgentRouter  [nuevo]              │  ║
║  │  diagnose(T, x_data) → route → TAA / ERGON / joint  │  ║
║  │  μ_SRB interface: ERGON → TAA (eliminates δ_μ)      │  ║
║  └─────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════╝
                        │
          ┌─────────────┼──────────────────┐
          ▼             ▼                  ▼
    ┌───────────┐ ┌───────────┐    ┌───────────┐
    │ c_native  │ │ pytorch   │    │  onnx     │
    │ AVX2/512  │ │ CUDA/ROCm │    │  wasm     │
    │ OpenMP    │ │ TensorRT  │    │  verilog  │
    └───────────┘ └───────────┘    └───────────┘
```

---

## Los Cinco Subsistemas

| Subsistema | Archivo | Responsabilidad |
|---|---|---|
| `GideonIR` | `ir.py` | Baja AST/FMA a representación intermedia tipada |
| `GideonGraph` | `graph.py` | Construye grafo, analiza topología, detecta fusiones |
| `GideonDispatcher` | `dispatcher.py` | Elige backend óptimo por hardware y carga |
| `GideonEngine` | `engine.py` | Orquesta el pipeline completo: IR → Graph → Dispatch → Exec |
| `GideonNeuralHints` + `GideonTheoremSeeds` | `neural_hints.py` / `theorem_seeds.py` | Base para IA y descubrimiento matemático (futuro) |
| **`GideonAgentRouter`** | **`agent_router.py`** | **Diagnóstico TAA/ERGON: decide qué agente actúa** |

---

## GideonAgentRouter — El Enrutador de Agentes ACF

### ¿Por qué un router de agentes?

Con TAA y ERGON como agentes del ecosistema, Gideon necesita un subsistema que:

1. **Diagnostique** la naturaleza del sistema dinámico (¿ordenado o caótico?)
2. **Enrute** la señal al agente correcto (TAA solo, ERGON solo, o ambos)
3. **Coordine** la interfaz $\mu_{SRB}$: cuando ERGON encuentra la medida SRB, la pasa a TAA
4. **Propague** los certificados formales al pipeline de compilación

El enrutador implementa la **Frontera de Diagnóstico Estructural** descrita en `ERGON_AGENT.md`.

### Diagrama de flujo del router

```
SEÑAL ENTRANTE (sistema T, datos {x_k})
               │
               ▼
    ┌──────────────────────────────┐
    │   GideonAgentRouter          │
    │                              │
    │  Diagnóstico rápido:         │
    │  • λ_max (Lyapunov proxy)    │
    │  • α_A (spectral index TAA)  │
    │  • h_KS / Σλ⁺ = 𝔈(T)       │
    └──────────┬───────────────────┘
               │
    ┌──────────┼──────────────────┐
    │          │                  │
 𝔈 ≈ 0    0 < 𝔈 < 1         𝔈 ≈ 1
 λ_max≤0   región mixta      λ_max>0
    │          │                  │
    ▼          ▼                  ▼
 TAA solo  TAA+ERGON          ERGON solo
 FMA exact coordinados        Pesin cert.
    │          │                  │
    └──────────┴──────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │  GideonDispatcher            │
    │  (backend: pytorch/c_native) │
    └──────────────────────────────┘
```

### API del GideonAgentRouter

```python
from poema.backends.gideon import GideonAgentRouter, AgentRouterConfig

config = AgentRouterConfig(
    ergon_complexity_threshold=0.1,    # 𝔈 < 0.1 → TAA solo
    chaos_lambda_threshold=0.05,       # λ_max > 0.05 → activa ERGON
    ergon_iterations=50_000,           # Iteraciones Birkhoff para μ_SRB
    epsilon=1e-4,                      # Tolerancia objetivo
)

router = GideonAgentRouter(config)

# Análisis automático: decide TAA vs ERGON vs joint
result = router.route(T=my_map, x0=initial_condition, x_data=trajectory)

print(result.agent_used)        # 'taa', 'ergon', 'joint'
print(result.taa_report)        # TAAReport (siempre disponible)
print(result.ergon_report)      # ERGONReport (None si 𝔈 ≈ 0)
print(result.measure_source)    # 'empirical' o 'srb'
print(result.pesin_verified)    # True si ERGON verificó Pesin
```

---

## GideonIR — La Representación Intermedia

### ¿Por qué un IR?

Análogamente a LLVM-IR o MLIR, GideonIR es la capa que desacopla el *qué* (semántica del programa Poema) del *cómo* (instrucciones nativas de cada backend). Sin IR, cada backend implementa su propia versión del lowering, duplicando lógica y erros.

### Tipos de nodos — IRNodeKind (28 tipos)

```
Primitivos:     CONST, INPUT, FMA, IDENTITY
Afines:         SCALE, SHIFT, AFFINE
Composición:    COMPOSE, PARALLEL, BRANCH
Polinomios:     POLY_HORNER, POLY_CHEB
Trascendentales: SIN, COS, EXP, LOG, TANH, SIGMOID
Álgebra lineal: MATMUL, GEMM, CONV, NORM, ATTENTION
Control:        LOOP, RECURSIVE
Futuro (base):  ARCH_PROBE, THEOREM_SEED
```

### IRNode — nodo tipado

Cada `IRNode` porta:
- **`kind`**: tipo de la operación
- **`params`**: parámetros (`weight`, `bias`, `coefficients`, etc.)
- **`inputs`**: IDs de nodos predecesores (DAG)
- **`meta`**: metadatos de seguridad numérica con:

$$\varepsilon_i \approx |w_i| \cdot \varepsilon_{i-1} + \varepsilon_{\text{machine}}$$

$$I_{\text{out}} = [w \cdot I_{\text{in,lo}} + b,\ w \cdot I_{\text{in,hi}} + b] \quad (w \geq 0)$$

donde $I_{\text{in}}$ es el intervalo de valores posibles propagado desde el nodo anterior.

### GideonProgram — programa compilado

```python
prog = GideonIR().from_fma_sequence(fma_chain, domain=(-1, 1))
print(prog.summary())
# GideonProgram('fma_program')
#   Nodos:    1001
#   FMA:      1000
#   ε global: 2.22e-13
#   Outputs:  ['fma_1000']
```

### Serialización JSON completa

```python
# Exportar
json_str = GideonIR.to_json(prog)

# Restaurar exactamente
prog2 = GideonIR.from_json(json_str)
assert prog2.total_fma == prog.total_fma
```

---

## GideonGraph — El Grafo de Cómputo

### Construcción automática

`GideonGraph` toma un `GideonProgram` y construye:
- Una tabla de adyacencia de sucesores `adj[id] → [ids]`
- Una tabla de predecesores `pred[id] → [ids]`
- Lista de aristas `GraphEdge` tipadas con cotas de intervalo

### Análisis topológico (Kahn's BFS)

El grafo se ordena en *niveles* (BFS topológico):

```
Nivel 0: [inp]               ← nodos sin predecesores
Nivel 1: [fma_001]           ← nodos con todos sus preds en nivel 0
Nivel 2: [fma_002]           ← ...
...
Nivel N: [output]
```

Nodos en el **mismo nivel** no tienen dependencias entre sí y pueden ejecutarse **en paralelo**.

```python
g = GideonGraph(prog)
plan = g.analyse()
print(plan.summary())
# ExecutionPlan
#   Fases:           1001
#   Nodos:            1001
#   Paralelizables:   0.0%  ← cadena lineal
#   Camino crítico:   1001 fases
```

### Detección de cadenas fusables

Gideon detecta tramos de nodos `FMA` consecutivos (1 entrada, 1 salida) que pueden compilarse como un único kernel C/AVX en lugar de N llamadas individuales:

```
[fma_001] → [fma_002] → ... → [fma_200]
     └──────── fusable_chain(200) ──────┘
              ↓ compile as 1 C kernel
        AVX2: 200 FMAs / (4 doubles × cycle)
```

```python
chains = g.find_fusable_chains()
# [['fma_001', 'fma_002', ..., 'fma_200']]  → 1 kernel
```

### Estadísticas del grafo

```python
stats = g.stats()
# {
#   "n_nodes": 1001,  "n_edges": 1000,
#   "n_phases": 1001, "total_fma": 1000,
#   "fusable_chains": 1,
#   "global_epsilon": 2.22e-13,
#   "ai_layers": {}                    # vacío en cadena FMA pura
# }
```

---

## GideonDispatcher — El Cerebro de Despacho

### Detección automática de hardware

`HardwareProfile.detect()` detecta en tiempo de ejecución:

| Feature | Método de detección |
|---|---|
| AVX2 / AVX-512 | `/proc/cpuinfo` (Linux) / `sysctl` (macOS) |
| CUDA (NVIDIA) | `torch.cuda.is_available()` |
| ROCm (AMD) | `torch.version.hip` |
| cffi (C compile) | `import cffi` |
| ONNX Runtime | `import onnx` |

### Algoritmo de ranking de backends

```
score[backend] = BASE_PRIORITY[backend]
              + hardware_bonuses          // AVX512: +20, AVX2: +10
              + ai_workload_bonus         // MATMUL/ATTENTION detectado: +50
              + heavy_fma_bonus           // n_fma > 1000: +30
              + user_hints               // prioridad configurada por usuario
              - latency_penalty           // histórico de latencias >= 10ms
              * availability_gate         // 0 si backend no disponible
```

Prioridades base:

| Backend | Prioridad base |
|---|---|
| `c_native` | 100 |
| `pytorch` / `rocm` | 80 |
| `onnx` | 60 |
| `numpy_cpu` | 40 |
| `wasm` | 20 |
| `verilog` | 5 |

### Ejecución heterogénea por nodo

Para grafos mixtos (FMA + IA), el dispatcher asigna backends por nodo:

```python
dec = dispatcher.decide(prog)
# nodo matmul → "pytorch"  (CUDA para GEMM)
# nodo fma    → "c_native" (AVX2 para FMA puro)
dec.node_backend_map
# {"mm1": "pytorch", "fma_001": "c_native", ...}
```

### Feedback loop de latencias

```python
dispatcher.record_latency("c_native", 2.5)  # ms
# → re-ranking futuro usa promedio de las últimas 20 mediciones
```

### Garantía O(1) de despacho y condición de pre-compilación

> **Advertencia de ingeniería:** La afirmación de latencia de despacho $\tau = O(1)$ es válida **si y solo si** los kernels Triton están ya compilados y residentes en VRAM antes del inicio del ciclo temporal de inferencia. Si el grafo requiere recompilar un kernel Triton durante el ciclo — ya sea por cambio de umbrales D-PES, nueva forma de tensor, o JIT tardío — el overhead del compilador JIT de Python destruye la garantía O(1).

**Protocolo obligatorio de pre-compilación (warm-up estático):**

```python
from gideon_core.engine import GideonEngine

engine = GideonEngine()

# ── FASE INIT (antes del loop temporal) ──────────────────────────────────
# Compilar y cachear todos los kernels sobre las formas esperadas.
# Esto ejecuta el JIT de Triton y ancla los kernels en VRAM.
engine.warmup(
    fma_chain=my_fma_chain,
    input_shapes=[(batch_size, state_dim)],  # formas concretas, no simbólicas
    backends=["triton", "c_native"],
    freeze=True,  # congela el grafo: rechaza recompilaciones en caliente
)
# Después de warmup(), engine.is_frozen == True

# ── CICLO TEMPORAL (inferencia O(1)) ──────────────────────────────────────
for t in range(n_steps):
    result = engine.run_fma(fma_chain, x_t)  # O(1): kernel ya en VRAM
    # Si freeze=True y se solicita una nueva forma → RuntimeError explícito,
    # no recompilación silenciosa que rompe la garantía de latencia.
```

**¿Qué hace `freeze=True`?**
- Congela el mapa `{(backend, tensor_shape) → compiled_kernel}` en memoria
- Convierte la decisión de despacho en un lookup de tabla hash: $O(1)$ real
- Cualquier solicitud de forma/backend fuera del mapa levanta `FrozenGraphError`
  en lugar de recompilar silenciosamente en runtime

**Colapsado estático del GideonGraph:**

El `GideonGraph` debe colapsar todas las fusiones de cadenas FMA y la asignación D-PES **antes** de `warmup()`. Los umbrales D-PES se fijan en la fase INIT y no se modifican durante el ciclo temporal. Si el sistema requiere adaptar los umbrales D-PES en tiempo real, debe hacerlo en una fase de recalibración explícita separada del ciclo de inferencia, con un nuevo calling a `warmup()` sobre las formas actualizadas.

---

## GideonEngine — El Motor Principal

### Pipeline completo

```python
engine = GideonEngine()
result = engine.run_fma(fma_chain, x)
# output: ndarray / tensor con resultado
# result.backend_used: "c_native"
# result.total_fma: 100
# result.global_epsilon: 2.22e-12
# result.elapsed_ms: 0.6
# result.graph_stats: {"n_nodes": 101, "fusable_chains": 1, ...}
# result.dispatch_decision: DispatchDecision(...)
```

### Pasos internos del pipeline `run_fma`

```
1. GideonIR.from_fma_sequence()        → GideonProgram
2. GideonGraph(prog)                   → grafo tipado
3. GideonDispatcher.decide(prog)       → DispatchDecision
4. BackendRegistry.get(backend_name)   → BackendProtocol
5.   backend.compile(fma_seq)          → callable_fn
6. callable_fn(x)                      → output numérico
7. dispatcher.record_latency(backend, ms)
8. (opcional) GideonTheoremSeeds.analyse(fn)  → [TheoremCandidate]
```

### Modo benchmark

```python
cfg = GideonEngineConfig(benchmark_mode=True, benchmark_repeats=1000)
engine = GideonEngine(cfg)
result = engine.run_fma(chain, x_1M)
# Ejecuta 1000 veces y reporta tiempo promedio
```

### Compilación con target "gideon" en PoemCompiler

```python
from poema.compiler import PoemCompiler
from poema.frontend import Poem

poem = Poem()
ast  = poem.continuous_flow("sin(x^2) + 3*x - 1")

compiler = PoemCompiler(target="gideon", precision="fp64")
fn, report = compiler.compile(ast, domain=(-3, 3))

x = torch.linspace(-3, 3, 10000, dtype=torch.float64)
y = fn(x)   # ejecuta vía Gideon → c_native/pytorch/numpy
```

El acceso al engine interno:

```python
engine = compiler.gideon   # GideonEngine (lazy, instanciado 1 sola vez)
print(engine.info())
```

---

## Bases Futuras: IA y Descubrimiento de Teoremas

### GideonNeuralHints — Infraestructura para NAS

`GideonNeuralHints` establece la base para que Gideon, en el futuro, **busque y descubra arquitecturas de IA**. Por ahora implementa:

#### ArchitectureBlueprint

Cada blueprint es un grafo de `LayerSpec` con métricas de complejidad intrínseca:

$$\alpha_{\text{blueprint}} = \frac{\log(\text{total\_flops})}{\log(\text{total\_params} + 1)}$$

El índice $\alpha$ mide la *densidad computacional* de la arquitectura: cuánto cómputo por parámetro. Mayor $\alpha$ → arquitectura más computacionalmente densa (típico de transformers vs MLPs con mismos params).

```python
bp = GideonNeuralHints.transformer(d_model=1024, n_heads=16, n_layers=12)
print(bp.summary())
# Blueprint('transformer', kind=transformer)
#   Capas:       48
#   Parámetros:  402,653,184
#   FLOPs:       3,423,379,456
#   α-complejidad: 1.0338
#   FMA-equiv:   1,711,689,728
```

Blueprints soportados: `mlp(layer_dims)`, `transformer(d_model, n_heads, n_layers)`, `cnn_resnet_block(channels, n_blocks)`.

**Roadmap**: integrar búsqueda con RL/evolución sobre espacio de blueprints.

### GideonTheoremSeeds — Infraestructura para Descubrimiento Matemático

`GideonTheoremSeeds` cierra el ciclo entre **cómputo numérico** y **verificación formal**:

```
Gideon ejecuta fn en muchos puntos
            ↓
InvariantProbe detecta propiedades estadísticas:
  - Constante de Lipschitz: L = max|f(x)-f(y)|/|x-y|
  - Monotonicidad: f'(x) ≥ 0 ∀x
  - Simetría (par/impar)
  - Contractividad: L < 1
  - α-complejidad ACF del espectro
            ↓
PatternMatcher confirma patrones algebraicos
            ↓
TheoremCandidate con lean_skeleton generado
            ↓
GideonEngine.export_lean_theorems(path)  → .lean file
            ↓
Genesis Auto-Prover verifica formalmente en Lean 4
```

#### Ejemplo de candidato generado

```lean
-- Lipschitz bound for identity_fn
-- L ≈ 1.0000: computed empirically on 1000 points
noncomputable def lipschitz_identity_fn : ℝ := 1.0000
-- Claim: |f(x) - f(y)| ≤ lipschitz_identity_fn * |x - y|
```

---

## Integración Nativa con Poema

### `PoemCompiler` — nuevo target `"gideon"`

```python
# Antes (targets existentes)
PoemCompiler(target="pytorch")   # ✓
PoemCompiler(target="triton")    # ✓

# Ahora (nuevo target nativo)
PoemCompiler(target="gideon")    # ✓  ← Gideon orquesta todo
```

### Acceso directo al engine desde el compilador

```python
compiler = PoemCompiler(target="gideon")
# La primera vez que se accede, se instancia el engine
engine = compiler.gideon   # → GideonEngine
print(engine.info())       # Hardware, backends disponibles, versión
```

### Workflow integrado completo

```python
from poema.compiler import PoemCompiler
from poema.frontend import Poem
import torch

# 1. Definir programa
poem = Poem(dtype=torch.float64)
ast  = poem.continuous_flow("x^3 - 2*x + 1")

# 2. Compilar con Gideon
compiler = PoemCompiler(target="gideon", precision="fp64")
fn, report = compiler.compile(ast, domain=(-2.0, 2.0))

# 3. Ejecutar
x = torch.linspace(-2, 2, 1_000_000, dtype=torch.float64)
y = fn(x)

# 4. Inspeccionar
print(report.total_fma_ops)     # Operaciones FMA
print(report.total_epsilon)     # Cota de error certificada
print(report.compilation_time_ms)
for w in report.warnings:
    print(w)   # "gideon:backend_used=c_native"
```

### Registro de backends — Gideon visible globalmente

```python
from poema.backends import BackendRegistry, GideonEngine

# Gideon exportado desde el paquete backends
engine = GideonEngine()
print(engine.info())

# BackendRegistry sigue funcionando para backends individuales
avail = BackendRegistry.available()
```

---

## Rendimiento Medido (Hardware Real)

> **Hardware**: Linux, CPU con AVX2, CUDA GPU (RTX 4050), Python 3.12, NumPy 1.26

### Benchmark: Gideon (c_native) vs NumPy baseline

| Configuración | NumPy | Gideon (c_native) | Speedup |
|---|---|---|---|
| 10 stages × 10M elementos | 210.18 ms | **14.69 ms** | **14.3×** |
| 50 stages × 1M elementos | 29.01 ms | **2.55 ms** | **11.4×** |
| 100 stages × 100K elementos | 2.83 ms | **0.60 ms** | **4.8×** |
| 1000 stages × 10K elementos | 3.83 ms | **1.02 ms** | **3.8×** |

### Tests de estrés superados

| Test | Resultado |
|---|---|
| 5000 stages, cadena contráctiva | ✓ ε propagado = 1.66e-13 |
| 5M elementos, 20 stages | ✓ 519 ms (c_native) |
| 10M elementos, 10 stages | ✓ 578 ms (c_native) |
| JSON roundtrip, 1000 nodos | ✓ Exacto bit a bit |
| Ejecución idempotente ×5 | ✓ Salidas estrictamente iguales |

### Precisión numérica verificada

Para cadenas de N stages con pesos $|w_i| \in [0.8, 1.2]$:

$$\| y_{\text{Gideon}} - y_{\text{NumPy ref}} \|_\infty < \varepsilon_{\text{machine}} \cdot N \cdot \prod_{i=1}^{N} |w_i|$$

Verificado para N = 1, 10, 100, 200 stages con tolerancias `rtol=1e-9, atol=1e-12`.

---

## Diagrama de Flujo Completo

```
Usuario escribe:
  poem.continuous_flow("sin(x^2) + 3*x")
         │
         ▼
  AST de Poema (Compose/Poly/Transcendental nodes)
         │
         ▼ PoemCompiler.compile() fases 1–5
  ┌────────────────────────────────────────────┐
  │  1. Simplify   (scale(1) → identity, etc.) │
  │  2. TypeCheck  (dimensiones, Lie brackets) │
  │  3. DomainGuard (propagación de intervalos)│
  │  4. FMALinearize → [FMA(w1,b1), ...]       │
  │  5. ErrorBound → ε certified               │
  └──────────────────────┬─────────────────────┘
                         │ FMA sequence + ε
                         ▼
         ┌───────────────────────────────┐
         │      G I D E O N              │
         │                               │
         │  IR  →  Graph  →  Dispatch    │
         │                     │         │
         │             ┌───────┘         │
         │             ▼                 │
         │     best_backend(hw, load)    │
         │       c_native / pytorch      │
         │       rocm / onnx / numpy     │
         │             │                 │
         │             ▼                 │
         │      backend.compile()        │
         │      → callable_fn            │
         │             │                 │
         │    (opcional) TheoremSeeds    │
         └─────────────┬─────────────────┘
                       │
                       ▼
              fn(x: Tensor) → y: Tensor
              + CompilationReport
              + GideonExecutionResult
```

---

## Propiedades Matemáticas Garantizadas

### 1. Propagación de Error Acotada

Para una cadena de $N$ instrucciones FMA con pesos $w_1, \ldots, w_N$:

$$\varepsilon_N = \sum_{k=1}^{N} \varepsilon_{\text{machine}} \cdot \prod_{j=k+1}^{N} |w_j|$$

GideonIR computa este bound nodo a nodo en tiempo $O(N)$ durante el lowering.

### 2. Propagación de Intervalos (Aritmética de Intervalos)

Para cada nodo FMA $y = wx + b$ con $x \in [a, b]$:

$$[y_{\min}, y_{\max}] = \begin{cases} [wa + b,\ wb + b] & w \geq 0 \\ [wb + b,\ wa + b] & w < 0 \end{cases}$$

Esto garantiza que GideonIR conoce los límites de valores posibles en cada nodo, lo que permite al dispatcher detectar overflows potenciales antes de ejecutar.

### 3. Idempotencia de Ejecución

$$\forall x,\ \text{run}(P, x)_1 = \text{run}(P, x)_2 = \ldots = \text{run}(P, x)_N$$

La misma cadena FMA compilada con el mismo backend produce salidas bit-a-bit idénticas en ejecuciones repetidas (verificado en `TestGideonStress::test_repeated_executions_idempotent`).

### 4. Correctitud del Grafo Topológico

El algoritmo de Kahn garantiza que el orden de ejecución respeta todas las dependencias:

$$\forall \text{arista} (u \to v): \text{phase}(u) < \text{phase}(v)$$

### 5. Invariant del Dispatcher

El dispatcher garantiza que el backend seleccionado siempre está disponible:

```python
backend = registry.get(decision.primary_backend)
assert backend.verify_available() == True
# Si no: falla → fallback → numpy_cpu (siempre disponible)
```

### 6. Clasificación de Familias (Formal — Teoremas FAM-1/2/3/4)

Las clases `fast`, `algebraic`, `slow` usadas en el dispatcher NO son heurísticas: son clases de complejidad rigurosas probadas en `MathTest/FormalEmpiricalTheorems.lean`:

- **FAM-1**: `IsFastFamily c ↔ ∃ C ρ > 1, ∀ k, |c k| ≤ C·ρ⁻ᵏ` — decaimiento exponencial
- **FAM-2**: `IsAlgebraicFamily c ↔ ∃ C s > 0, ∀ k > 0, |c k| ≤ C·k⁻ˢ` — decaimiento polinomial  
- **FAM-3**: fast ⊂ algebraic con C' = C·ρ/(ρ-1) (inclusión estricta, probada)
- **FAM-4**: NC0/NC1/NC2/NC3 es una partición completa y disjunta de [0,∞)

### 7. Grado de Filtro Óptimo (Formal — Teoremas FIEDLER-1/2/3)

Para grafos con valor de Fiedler λ₂, el grado óptimo satisface:

$$d^*(\varepsilon, \lambda_2, \lambda_{\max}) = \left\lceil \frac{\log(2/\varepsilon)}{\log(1 + \lambda_2/\lambda_{\max})} \right\rceil$$

- **FIEDLER-1**: d* > 0 para todo ε ∈ (0,2) (probado con `div_pos` + `log_pos`)
- **FIEDLER-2**: d* es monótonamente decreciente en λ₂ (mayor conectividad → menor grado)
- **FIEDLER-3**: λ₂ = 0.5 requiere exactamente log(3/2)/log(5/4) ≈ **1.817×** más de grado que λ₂ = 1.0

### 8. Consistencia Alpha (Formal — Teoremas ALPHA-1/2/3/4)

El índice α es formalmente exacto:

| Teorema | Resultado |
|---------|-----------|
| ALPHA-1 | α̂_spec = α + \|log C\|/log k (expresión exacta, no estimación) |
| ALPHA-2 | \|α̂ − α\| = \|log C\|/log k (tasa de convergencia explícita) |
| ALPHA-3 | k ≥ exp(10·\|log C\|/α) ⟹ \|α̂ − α\| ≤ 0.1 (umbral formal para 10% tolerancia) |
| ALPHA-4 | log(ε₁/ε₂)/log(d₂/d₁) = α (tres estimadores coinciden exactamente) |

### 9. Convergencia del Ciclo Adjunto (Formal — Teoremas ADJ-1/2)

- **ADJ-1**: Si el ciclo Φ ⇌ Φ* es L-Lipschitz con L < 1, converge a punto fijo único (Banach, probado con `LipschitzWith.toContraction`)
- **ADJ-2**: Sin condición Lipschitz no hay punto fijo — contraejemplo f(x)=x+1 (probado constructivamente)

### 10. Isomorfismo AIC/BIC (Formal — Teoremas AIC-1/2/3/4)

La función de coste de gramática es formalmente equivalente a AIC/BIC (no una analogía):

$$\mathcal{C}(G, f, \beta = n/2) = \varepsilon - \frac{2S}{n} \quad \text{(= AIC normalizado, Teorema AIC-1)}$$

**Archivo fuente**: `MathTest/FormalEmpiricalTheorems.lean` (17 teoremas, 0 sorry)  
**Tests Python**: `tests/test_formal_empirical_bounds.py` (35 tests, todos pasan)

---

## Estado y Roadmap

### Estado actual (Abril 2026) ✓ — v1.4.0

| Componente | Estado | Versión |
|---|---|---|
| `GideonIR` — lowering FMA | ✅ Completo, 58 tests pasando | v1.0 |
| `GideonIR` — lowering AST | ✅ Completo | v1.0 |
| `GideonGraph` — análisis topológico | ✅ Completo | v1.0 |
| `GideonGraph` — fusión de cadenas | ✅ Completo | v1.0 |
| `GideonDispatcher` — detección hardware | ✅ Completo | v1.0 |
| `GideonDispatcher` — ranking dinámico | ✅ Completo | v1.0 |
| `GideonEngine` — pipeline completo | ✅ Completo | v1.0 |
| `GideonEngine` — benchmark mode | ✅ Completo | v1.0 |
| `PoemCompiler(target="gideon")` | ✅ Nativo | v1.0 |
| `GideonNeuralHints` — blueprints IA | ✅ Base implementada | v1.1 |
| `GideonTheoremSeeds` — invariantes | ✅ Base implementada | v1.1 |
| `GideonEngine` — métodos ACF de grafos | ✅ v1.1.0 | v1.1 |
| `GideonEngine` — métodos Neural-ACF | ✅ v1.1.0 | v1.1 |
| `GideonEngine` — meta-compilador | ✅ v1.1.0 | v1.1 |
| `GideonAutotune` — perfil hardware dinámico v2.0 | ✅ Completo | v1.4 |
| `gideon_autotune.py` — autotune adaptativo (JSON profile) | ✅ Activo y adaptativo | v1.4 |
| `triton_kernels.py` — FMA chain en GPU (Triton JIT) | ✅ fp64 garantizado | v1.4 |
| `fma_avx512.c` — prefetch + unroll AVX-512 | ✅ Completo | v1.4 |
| `gemm_kernel.c` — micro-kernel 8×4 AVX-512 | ✅ Completo | v1.4 |
| `engine.rs` — buffers alineados 64 bytes | ✅ Completo | v1.4 |
| `build.rs` — flags dinámicos desde perfil JSON | ✅ Completo | v1.4 |
| `engine.py` — integración Triton como primer backend GPU | ✅ Completo | v1.4 |
| Tests suite de regresión | ✅ 1427/1427 pasando | v1.4 |
| `test_gideon_baremetal.py` — 18 tests bare-metal | ✅ 18/18 pasando | v1.4 |

### Roadmap Q2-Q4 2026

| Fase | Objetivo | Estado |
|---|---|---|
| **Q2 2026** | ~~Paralelización real de fases independientes del GideonGraph~~ | ✅ Triton kernels paralelizan automáticamente |
| **Q2 2026** | Fusión de kernels cross-backend (C chain + GPU GEMM en mismo pass) | 🔄 Parcial: Triton + C independientes |
| **Q3 2026** | GideonIR → código MLIR nativo (puente a LLVM) | ⏳ Pendiente |
| **Q3 2026** | NAS básico: búsqueda de arquitecturas MLP sobre GideonGraph | ⏳ Pendiente |
| **Q4 2026** | Genesis Auto-Prover integrado vía TheoremSeeds → Lean 4 | ⏳ Pendiente |
| **Q1 2027** | Motor Gideon v2.0: Rust core con bindings Python | ⏳ Pendiente |

---

## Análisis de Grafos con Gideon

### ¿Qué es el ACF espectral de grafos?

`GideonEngine` v1.1.0 incluye dos nuevos métodos para trabajar con señales de grafos usando el functor ACF extendido al dominio espectral. El principio es sencillo: todo grafo tiene un Laplaciano L = D − A con descomposición espectral L = UΛUᵀ, y toda señal **s** sobre ese grafo puede filtrarse mediante un polinomio H(λ) que opera sobre los valores propios. Gideon automatiza el diseño de ese polinomio usando el functor ACF.

### Método `reduce_graph_signal`

```python
engine.reduce_graph_signal(
    signal,          # np.ndarray shape (n,) — señal sobre los nodos
    adjacency,       # np.ndarray shape (n,n) — matriz de adyacencia
    normalization="unnormalized",   # "unnormalized" | "symmetric"
    degree=8,        # grado del polinomio filtro
    domain=None,     # si None, se detecta automáticamente [λ_min, λ_max]
) -> GraphReductionResult
```

**Qué hace:**
1. Construye el Laplaciano L a partir de `adjacency`.
2. Descompone espectralmente L = UΛUᵀ.
3. Aplica el functor ACF sobre la función identidad H(λ) = λ para obtener el filtro óptimo.
4. Filtra la señal: **s**_filtered = U · H(Λ) · Uᵀ **s**.
5. Retorna el resultado con la señal filtrada, el polinomio H, el ε de aproximación y los invariantes espectrales.

```python
import numpy as np
from poema.backends.gideon.engine import GideonEngine

engine = GideonEngine()

# Grafo camino P₅
A = np.array([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[0,0,0,1,0]], dtype=float)
signal = np.array([1.0, 0.5, 0.2, 0.5, 1.0])

result = engine.reduce_graph_signal(signal, A, degree=6)
print(result.filtered_signal)      # señal suavizada
print(f"ε = {result.epsilon:.2e}") # cota de error del filtro
```

### Método `analyse_graph`

```python
engine.analyse_graph(
    adjacency,       # np.ndarray shape (n,n)
    normalization="unnormalized",
) -> GraphACFInvariants
```

Retorna el análisis completo del grafo: α, δ, clase NC, valor de Fiedler λ₂, entropía espectral y grado óptimo de filtro. Útil para caracterizar grafos antes de procesarlos.

```python
inv = engine.analyse_graph(A)
print(f"α = {inv.alpha:.4f}")
print(f"Fiedler = {inv.fiedler_value:.4f}")
print(f"Grado óptimo = {inv.optimal_filter_degree}")
```

---

## Análisis de Redes Neuronales con Gideon

### ¿Por qué analizar una red con ACF?

Una red neuronal es un modelo de cómputo, y `GideonEngine` v1.1.0 puede analizar ese modelo con el functor ACF: qué tan compleja es capa a capa (vía SVD y α), qué dinámica sigue durante el entrenamiento (vía Koopman), y qué tan comprimible es la función que implementa (vía auto-evolución). Estos tres análisis son completamente nuevos en la versión 1.1.0.

### Método `analyse_network`

```python
engine.analyse_network(
    network,         # torch.nn.Module (Sequential, Linear, Conv1d, etc.)
    degree=6,        # grado de reducción por capa
    domain=(-1.0, 1.0),
) -> NetworkACFReport
```

Recorre todas las capas del módulo PyTorch, aplica `NeuralLayerReducer` a cada `nn.Linear` y `nn.Conv1d`, y calcula el `NetworkACFReport`:

| Campo del resultado | Contenido |
|--------------------|-----------|
| `layer_reductions` | Reducción ACF por cada capa |
| `layer_invariants` | α por capa (via SVD de W) |
| `global_alpha` | α ponderado de toda la red |
| `global_nc_class` | Clase de complejidad NC global |
| `total_fma_count` | Operaciones FMA totales en representación polinomial |

```python
import torch.nn as nn
from poema.backends.gideon.engine import GideonEngine

engine = GideonEngine()
net = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 8), nn.Tanh(), nn.Linear(8, 1))

report = engine.analyse_network(net)
print(f"α global = {report.global_alpha:.4f}")
print(f"clase NC = {report.global_nc_class}")
for i, inv in enumerate(report.layer_invariants):
    print(f"  Capa {i}: α = {inv.alpha:.4f}")
```

### Método `analyse_training_trajectory`

```python
engine.analyse_training_trajectory(
    trajectory,      # np.ndarray shape (T,) — serie temporal de pérdidas
) -> KoopmanNetworkResult
```

Aplica el operador de Koopman a la trayectoria de entrenamiento. Los valores propios del operador linealizado revelan los modos del proceso de entrenamiento:

```python
import numpy as np

# Simular trayectoria de entrenamiento
losses = np.array([2.3, 1.8, 1.4, 1.1, 0.9, 0.75, 0.65, 0.58, 0.54, 0.51])
k_result = engine.analyse_training_trajectory(losses)
print(k_result.koopman_eigenvalues)    # modos espectrales
print(k_result.spectral_diagnostics)  # diagnóstico de convergencia
```

### Método `evolve_network_function`

```python
engine.evolve_network_function(
    network,         # torch.nn.Module
    domain=(-1.0, 1.0),
    input_dim=1,     # dimensión de entrada para construir f_net
) -> NeuralEvolutionResult
```

Construye f_net(x) = salida media de `network` con entrada x·**1** y aplica `ACFAutoEvolver`. Determina qué tan comprimible es la función que implementa la red.

```python
evo = engine.evolve_network_function(net, domain=(-1.0, 1.0), input_dim=16)
print(f"ε₀ → ε_f: {evo.initial_epsilon:.2e} → {evo.final_epsilon:.2e}")
print(f"Mejora: {evo.improvement_ratio:.1f}×")
```

---

## El Meta-Compilador en Gideon

### ¿Qué es seleccionar la base óptima?

El functor ACF estándar usa siempre polinomios de Chebyshev. Para funciones periodicas, con discontinuidades, o de origen dinámico, otras familias de bases pueden ser mucho mejores. `engine.meta_compile()` automatiza esta selección: evalúa múltiples familias (Chebyshev, Fourier, RBF, Koopman-poly, …) con distintos grados y elige la *gramática* G* que minimiza la energía libre:

$$\mathcal{C}(G, f, \beta) = \varepsilon(G, f) - \frac{S(G)}{\beta}$$

### Método `meta_compile`

```python
engine.meta_compile(
    f,                          # callable — función a compilar
    domain=(-1.0, 1.0),
    strategy="greedy",          # "grid" | "random" | "greedy"
    beta=1.0,                   # parámetro de temperatura
    target_epsilon=1e-6,        # ε objetivo
    enable_auto_evolution=False,# combinar con ACFAutoEvolver
    config=None,                # MetaCompilerConfig completo (opcional)
) -> MetaCompilerResult
```

**Estrategias:**

| Estrategia | Descripción | Cuándo usar |
|-----------|-------------|-------------|
| `"grid"` | Exhaustiva — evalúa todo el espacio | Presupuesto ilimitado, máxima calidad |
| `"random"` | Muestreo aleatorio | Espacio grande, presupuesto acotado |
| `"greedy"` | Búsqueda voraz con reinicios | Uso general, rápido y bueno |

**Ejemplo completo:**

```python
import numpy as np
from poema.backends.gideon.engine import GideonEngine

engine = GideonEngine()

# Función con kink — difícil para Chebyshev
f = lambda x: np.abs(x - 0.3)

result = engine.meta_compile(f, domain=(-1.0, 1.0), strategy="greedy", beta=1.0)
print(result.best_grammar)                         # Grammar(basis=RBF, degree=12, ...)
print(f"Mejora vs Chebyshev: {result.improvement_ratio:.1f}×")
print(f"ε inicial: {result.initial_epsilon:.2e}")
print(f"ε final:   {result.final_epsilon:.2e}")
```

**`MetaCompilerResult` — campos:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `best_grammar` | `Grammar` | Gramática optima encontrada |
| `best_reduction` | `ReductionResult` | Reducción ACF con esa gramática |
| `initial_epsilon` | `float` | ε con Chebyshev baseline (grado 8) |
| `final_epsilon` | `float` | ε con la gramática óptima |
| `improvement_ratio` | `float` | initial_epsilon / final_epsilon |
| `trace` | `MetaCompilerTrace` | Historial de todas las gramáticas evaluadas |

### Las nueve familias de BasisFamily

```python
from acf_functor import BasisFamily

# Bases clásicas
BasisFamily.CHEBYSHEV      # Funciones analíticas
BasisFamily.LEGENDRE       # Integración numérica L²
BasisFamily.HORNER         # Evaluación eficiente
BasisFamily.FOURIER        # Señales periódicas
BasisFamily.RBF            # Funciones con discontinuidades

# Bases Koopman (para sistemas dinámicos)
BasisFamily.KOOPMAN_POLY     # Observables polinomiales, EDMD estándar
BasisFamily.KOOPMAN_FOURIER  # Observables trigonométricos
BasisFamily.KOOPMAN_RBF      # Observables locales
BasisFamily.KOOPMAN_MIXED    # Combinación polinomial + RBF
```

---

## Tensor ACF — Dominio multivariable en Gideon

Gideon v1.2.0 incorpora soporte nativo para funciones $f: \mathbb{R}^d \to \mathbb{R}$ mediante **Tensor ACF**. El módulo `tensor_acf` descompone funciones multivariable en formato Tensor Train (TT) sobre bases Chebyshev, permitiendo evaluación eficiente en $O(d \cdot n \cdot r^2)$ FMAs.

### Capacidades

- **Descomposición TT-SVD**: Factorización automática del tensor de coeficientes Chebyshev en cadena de matrices 3D. Rangos TT determinados adaptativamente por truncamiento SVD.
- **Descomposición Tucker (HOSVD)**: Para dimensiones $d \leq 6$, decomposición alternativa con tensor core y factores ortogonales.
- **Invariantes tensoriales**: $\alpha$ por modo, $\alpha$ global, dimensión efectiva, entropía espectral, clase NC.
- **Evaluación zipper**: Contracción FMA de izquierda a derecha — composable con el pipeline ACF estándar.
- **Certificación**: Error $L_\infty$ certificado contra evaluación directa en puntos aleatorios.

### Funciones de test estándar

`StandardTensorFunctions` incluye: `separable_product`, `rosenbrock`, `gaussian_2d`, `multivariate_polynomial`, `friedman1` (5D), `wave_3d`.

---

## Matrix ACF — Funciones de matrices en Gideon

El **Matrix ACF** (`matrix_acf`) extiende Gideon al álgebra lineal numérica. Cualquier función escalar $f$ se aplica a una matriz simétrica $A$ mediante polinomios de Chebyshev matriciales con recurrencia de Clenshaw.

### Funciones matriciales

| Clase | Función | Costo |
|-------|---------|-------|
| `MatrixExponential` | $e^{tA}$ | $d$ multiplicaciones $n \times n$ |
| `MatrixSquareRoot` | $A^{1/2}$ | $d$ multiplicaciones $n \times n$ |
| `MatrixLogarithm` | $\log A$ | $d$ multiplicaciones $n \times n$ |
| `MatrixResolvent` | $(A+\sigma I)^{-1}$ | $d$ multiplicaciones $n \times n$ |
| `MatrixSign` | $\text{sign}(A)$ | $d$ multiplicaciones $n \times n$ |

### Invariantes

`MatrixACFAnalyzer.analyse(A, func)` devuelve: $\alpha$ matricial, rango espectral, número de condición, gap espectral, grado efectivo, entropía Chebyshev, clase NC.

### Formalización

Certificados Lean 4 en `MathTest/TensorACFCertificates.lean`: cotas de error TT-SVD, convergencia Chebyshev matricial, decaimiento exponencial.

---

*Gideon es propiedad intelectual del proyecto Poema / Martínez's Invariant.*  
*Versión de motor: 1.4.0 — Fecha: Abril 2026*

---

## Nuevas Capacidades — Motor 1.4.0 (Abril 2026): Bare-Metal Avanzado

Esta versión implementa los cuatro frentes de optimización a nivel de silicio que llevan el rendimiento
del motor a su límite físico en la plataforma Intel Ultra 9 185H + RTX 4050.

### Frente 1 — Kernels Triton GPU (fp64 nativo)

**Archivo**: `poema/backends/gideon/triton_kernels.py`

`GideonTritonBackend` ejecuta cadenas FMA completas en un **único kernel launch** sobre la GPU,
eliminando los N round-trips Python→CUDA de la implementación anterior.

| Kernel | Descripción | Precisión |
|--------|-------------|-----------|
| `_fma_chain_kernel_f64` | Cadena de N FMAs, `tl.static_range` compile-time unrolled | fp64 |
| `_fma_chain_kernel_f32` | Variante fp32 para throughput máximo | fp32 |
| `_fma_chain_dyn_kernel` | Cadenas dinámicas N>64 sin unrolling estático | fp64/fp32 |
| `_pointwise_fma_kernel` | Un solo FMA y=w·x+b, punteros para garantizar fp64 | fp64 |

**Garantía de precisión fp64**: los escalares W,B se pasan como tensores de 1 elemento (no como
argumentos escalares Python), evitando el downcast a fp32 interno de Triton.

```python
from poema.backends.gideon.triton_kernels import GideonTritonBackend

tb = GideonTritonBackend(hw_caps=engine._hw_caps)

# Cadena de 32 FMAs sobre 1M elementos
fn = tb.get_fma_chain_fn(weights, biases)   # JIT compilado en primer uso (FASE INIT)
# ⚠ Este JIT debe ocurrir en warmup(), no en el ciclo temporal.
# Usar engine.warmup(freeze=True) antes del loop para garantizar O(1) de despacho.
y  = fn(x_numpy)                             # np.ndarray → np.ndarray (GPU interno)

# Benchmark integrado
bm = tb.benchmark(n=1_000_000, n_fma=32, dtype="fp64", repeats=200)
# {"triton_ms": 0.8, "pytorch_ms": 3.2, "speedup": 4.0, "throughput_gbs": 8.1}
```

`engine.py` usa Triton como **backend primario** en la ruta GPU, con PyTorch como fallback:

```python
# backend_used = "triton_folded"  →  cadena reducida a un FMA total
# backend_used = "triton_chain_N" →  cadena de N FMAs
```

### Frente 2 — AVX-512 C: Prefetch Software + Unrolling

**Archivo**: `gideon_core/c_backends/fma_avx512.c`

Todos los bucles AVX-512 y AVX2 incorporan ahora:

- **Prefetch software** a distancia configurable (`GIDEON_PREFETCH_DIST`, default 8):
  `_mm_prefetch(x + i + DIST*8, _MM_HINT_T0)` una vez por iteración
- **Unrolling ×2** activado en tiempo de compilación (`GIDEON_FMA_UNROLL >= 2`):
  procesa 16 doubles/iter (2 zmm chunks) en lugar de 8

Ambas constantes son sobreescribibles desde `build.rs` según el perfil de autotune.

### Frente 3 — Micro-kernel GEMM 8×4 AVX-512

**Archivo**: `gideon_core/c_backends/gemm_kernel.c`

Nuevo micro-kernel `microkernel_8x4_avx512` con arquitectura de acumuladores:

- 4 acumuladores ZMM (`acc0..acc3`) — 8 filas × 4 columnas
- Patrón broadcast-FMA: `_mm512_set1_pd(B[k,j])` → `_mm512_fmadd_pd(aVec, bBroadcast, acc)`
- Tres rutas en `gideon_gemm()`: `USE_AVX512_GEMM` → `USE_AVX2_GEMM` → scalar

Los parámetros de tiling (MC, KC, NC, MR, NR) usan patrón `#ifndef` para que `build.rs`
los inyecte en compilación sin modificar el fuente.

### Frente 4 — Rust: Buffers de Salida Alineados 64 Bytes

**Archivo**: `gideon_core/src/engine.rs`

`AlignedF64Buffer` es un buffer Rust con alineación de cache-line garantizada:

```rust
struct AlignedF64Buffer { ptr: *mut f64, len: usize, layout: Layout }

impl AlignedF64Buffer {
    fn new(n: usize) -> Self {
        // Layout::from_size_align(n * 8, 64).unwrap()
        // alloc_zeroed(layout) — sin overhead de inicialización Vec
    }
    fn as_mut_slice(&mut self) -> &mut [f64]
    fn to_vec(&self) -> Vec<f64>
}
// Drop implementado correctamente con dealloc(ptr, same_layout)
```

`run_fma()` usa `AlignedF64Buffer::new(n)` en lugar de `vec![0.0f64; n]`, garantizando que
la salida esté siempre alineada para instrucciones AVX-512 subsiguientes.

### Autotune Dinámico v2.0 — build.rs integrado

**Archivo**: `gideon_core/build.rs`

`build.rs` reescrito como sistema de compilación adaptativo:

1. Lee perfil JSON de hardware (`$GIDEON_PROFILE_PATH` env var) generado por `gideon_autotune.py`
2. Detecta soporte AVX-512 via `/proc/cpuinfo` sin dependencias externas
3. Inyecta flags derivados como `-D` al compilador C:
   - `GIDEON_PREFETCH_DIST` — distancia óptima según L2/L3 cache
   - `GIDEON_FMA_UNROLL` — factor de unroll según ancho de vector AVX
   - `GIDEON_MC, KC, NC, MR, NR` — tiles GEMM según capacidad L1/L2
4. Activa `-mavx512f -mavx512dq` automáticamente si el hardware lo soporta

```json
// Perfil generado por gideon_autotune.py v2.0 (ejemplo):
{
  "cpu_avx_level": 512,
  "l1d_kb": 48,
  "l2_kb": 2048,
  "l3_kb": 24576,
  "triton_block": 1024,
  "triton_num_warps": 4,
  "triton_num_stages": 3,
  "prefetch_dist": 8,
  "fma_unroll": 2,
  "gemm_mc": 256, "gemm_kc": 256, "gemm_nc": 3072
}
```

### Test Suite Bare-Metal — 18/18

**Archivo**: `tests/test_gideon_baremetal.py`

| Clase de tests | Cobertura |
|----------------|-----------|
| `TestTritonFMAChain` | Disponibilidad, corrección fp64, cadenas N=8/32, benchmark, speedup vs PyTorch |
| `TestGideonEngineCPUBareMetal` | Corrección c_native, throughput AVX-512, affine fold |
| `TestRustAlignedBuffers` | Corrección del path Rust, alineación de salida |
| `TestHeavyPDEComputation` | Laplace 2D GPU (256×256, 500 iter), Laplace 2D CPU NumPy, forward pass neural (10 capas × 1M) |
| `TestGideonEndToEndIntegration` | Perfil autotune tiene campos derivados, engine selecciona Triton en GPU, consistencia cross-backend |

**Resultado hardware real (RTX 4050 Laptop + Intel Ultra 9 185H):**

```
18 passed in 6.03s
```

| Métrica | Valor |
|---------|-------|
| Precisión fp64 Triton (max_err) | < 1e-13 |
| Throughput CPU AVX-512 | > 2 GB/s (1M doubles, depth 10) |
| Laplace 2D GPU 256×256 (500 iter) | ~0.4 s |
| Consisten. cross-backend (Triton vs NumPy) | rtol=1e-10 |

### Tabla de archivos modificados en v1.4.0

| Archivo | Tipo | Cambio |
|---------|------|--------|
| `poema/backends/gideon/triton_kernels.py` | **CREADO** | 300 líneas, `GideonTritonBackend` completo |
| `poema/backends/gideon/engine.py` | Modificado | Import Triton + `_triton_backend` + `_compile_fma_gpu` reescrito |
| `gideon_core/c_backends/fma_avx512.c` | Modificado | Prefetch software + unroll ×2 para AVX-512/AVX2 |
| `gideon_core/c_backends/gemm_kernel.c` | Modificado | Micro-kernel 8×4 AVX-512, flags `#ifndef`, ruta AVX-512 |
| `gideon_core/src/engine.rs` | Modificado | `AlignedF64Buffer` + `run_fma` usa aligned alloc |
| `gideon_core/build.rs` | **REESCRITO** | Autotune JSON → flags C dinámicos, detección AVX-512 |
| `tests/test_gideon_baremetal.py` | **CREADO** | 18 tests, 5 clases, PDE pesada, benchmarks reales |

---

### Verificación de Dominio Antes de Compilar

Gideon v1.3.0 verifica automáticamente la admisibilidad del dominio antes de cualquier compilación. Internamente llama a `DomainAdmissibilityChecker` y sólo procede si las seis condiciones AD-1…AD-6 pasan:

```python
# Sin cambio de API — Gideon verifica automáticamente
from acf_functor.domain_admissibility import DomainAdmissibilityChecker, AdaptiveFunctorRouter
import math

# Diagnóstico explícito
cert = DomainAdmissibilityChecker().check(math.sin, (-1.0, 1.0))
print(f"Admisible: {cert.admissible}")       # True
print(f"ρ_f = {cert.estimated_bernstein_rho:.4f}")  # ≈ 2.72
print(f"α(f) = {cert.estimated_alpha:.4f}")         # ≈ 0.23

# Función problemática — detectada antes de compilar
cert_bad = DomainAdmissibilityChecker().check(math.tan, (0.0, 1.58))
print(f"Admisible: {cert_bad.admissible}")   # False — singularidad en π/2
```

### Enrutamiento Adaptativo

El `AdaptiveFunctorRouter` elige la rama óptima según el certificado:

```python
branch, report = AdaptiveFunctorRouter().route(math.exp, (-2.0, 2.0))
# FunctorBranch.CHEBYSHEV para funciones analíticas generales
# FunctorBranch.HORNER para polinomios de bajo grado
# FunctorBranch.KOOPMAN para dinámicas caóticas
print(f"Rama: {branch.value}")
```

### Índice α Normalizado

`AlphaEstimate` ahora incluye `normalized_alpha = 1/(1+alpha) ∈ (0,1]`:

```python
from acf_functor.invariant_unified import ACFInvariantUnified
result = ACFInvariantUnified().compute(math.sin, (-1.0, 1.0), skip_geometric=True)
print(f"α crudo: {result.best_estimate:.4f}")          # ∈ [0, ∞)
print(f"α normalizado: {result.normalized_alpha:.4f}") # ∈ (0, 1]
```

### Teorema Nyquist-ACF

Consulta el grado mínimo y la clase de complejidad de cualquier función:

```python
from acf_functor.nyquist_acf import NyquistACFTheorem, NyquistComplexityClass
import math

result = NyquistACFTheorem().apply(math.sin, (-math.pi, math.pi), epsilon=1e-8)
print(f"d* = {result.d_star_empirical}")         # 18
print(f"Clase: {result.complexity_class}")       # EASY
print(f"Bits: {result.information_bits:.1f}")    # información mínima necesaria
```

### Observabilidad Koopman y Hardware

```python
from acf_functor.koopman_observability import KoopmanObservabilityChecker, EnergyInvariantHardwareVerifier

# Verificar KO-1a/b/c + KO-3
report = KoopmanObservabilityChecker(d=20, N=2000).check(math.tanh, (-1.5, 1.5))
print(f"Observable: {report.status}")

# Verificar E(f) = E(Φ(f)) en hardware real
hw = EnergyInvariantHardwareVerifier().verify(lambda x: x**3 - x, (-1.0, 1.0))
print(f"fp64 d*={hw['fp64']['energy_d_star']}, fp32 d*={hw['fp32']['energy_d_star']}")
```

### ACF Diferenciable (para redes neuronales)

Gideon v1.3.0 expone el compilador ACF como capa PyTorch diferenciable:

```python
import torch
from acf_functor.differentiable_acf import DifferentiableACFLayer

layer = DifferentiableACFLayer(degree=16, domain=(-1.0, 1.0))
x = torch.linspace(-1, 1, 100)
y = layer(x)
# y es tensorialmente diferenciable — úsala en nn.Sequential o como activación
```

### Solver PDE-ACF

Reduce ecuaciones diferenciales a secuencias FMA:

```python
import numpy as np
from acf_functor.pde_acf import PDEACFSolver, PDEConfig, PDEType

cfg = PDEConfig(pde_type=PDEType.HEAT, n_modes=8, t_end=0.2, dt=1e-5, nu=0.01)
solver = PDEACFSolver(cfg)
u0 = np.sin(np.pi * solver.x_grid)
report = solver.solve(u0)
print(f"α espectral(T): {report.alpha_spectral:.4f}")
print(f"FMA totales: {report.total_fma_count:,}")
```

### Meta-Compilador Riemanniano

Búsqueda de gramática óptima via gradiente natural de Fisher:

```python
from acf_functor.riemannian_meta_compiler import RiemannianMetaCompiler
import math

rmc = RiemannianMetaCompiler(target_epsilon=1e-4, max_iter=20)
result = rmc.compile(math.sin, domain=(-1.0, 1.0))
print(f"Base óptima: {result.best_basis}")
print(f"Grado: {result.best_degree}, ε: {result.best_epsilon:.2e}")
```

### Verificación de Adjunción

```python
import torch, math
from acf_functor.adjunction import AdjunctionVerifier

f = lambda x: torch.tensor([math.sin(xi.item()) for xi in x])
result = AdjunctionVerifier(tolerance=1e-3).verify_triangle_identities(
    f, domain=(-1.0, 1.0), degree=25
)
print(f"Identidad izquierda error: {result.left_triangle_error:.2e}")
print(f"Adjunción se sostiene: {result.adjunction_holds}")
```

### Certificación Genesis-Lean (anti-tautología)

```python
from acf_functor.genesis_lean_bridge import GenesisLeanBridge, is_tautological

# La función guardián rechaza pruebas vacías
assert is_tautological("exact h_bound")        # True
assert not is_tautological("linarith [h1]")   # False

bridge = GenesisLeanBridge(catalog_path="/tmp/acf_catalog.json")
conj = bridge.conjecture_from_evidence({
    "func": "sin", "degree": 20, "epsilon": 1e-8, "bernstein_rho": 2.718
})
```

---

## Tabla de Módulos v1.3.0

| Módulo | Clase principal | Descripción breve |
|--------|----------------|-------------------|
| `domain_admissibility.py` | `DomainAdmissibilityChecker` | 6 condiciones AD-1…AD-6 |
| `nyquist_acf.py` | `NyquistACFTheorem` | d*(ε,α), clases EASY/MEDIUM/HARD/EXTREME |
| `koopman_observability.py` | `KoopmanObservabilityChecker` | KO-1a/b/c, KO-3, EDMD |
| `koopman_observability.py` | `EnergyInvariantHardwareVerifier` | E(f)=E(Φ(f)) en fp64/fp32/fp16 |
| `differentiable_acf.py` | `DifferentiableACFLayer` | Compilador diferenciable (PyTorch) |
| `pde_acf.py` | `PDEACFSolver` | Galerkin espectral + RK4 para PDEs |
| `genesis_lean_bridge.py` | `GenesisLeanBridge` | Ciclo conjetura-verificación-catálogo |
| `riemannian_meta_compiler.py` | `RiemannianMetaCompiler` | Gradiente natural Fisher en gramáticas |
| `adjunction.py` | `AdjunctionVerifier` | Identidades triangulares Φ*⊣Φ |
| `invariant_unified.py` | `ACFInvariantUnified` | α crudo + α normalizado ∈ (0,1] — 5 estimadores |
| `neural_arch_acf.py` | `NeuralArchACF` | Fingerprinting de arquitecturas + reemplazo de NAS |
| `stochastic_acf.py` | `HighEntropyAnalyzer` | Hurst, Lévy-α, entropía espectral, K-S entropy |
| `stochastic_acf.py` | `FinancialACF` | VaR certificado Chebyshev, regímenes, Sharpe UQ |
| `stochastic_acf.py` | `BayesianNNAnalyzer` | Distribución de pesos PCE, UQ de predicciones, Sobol |
| `meta_compiler.py` | `BayesianSearch` | Surrogate GP + adquisición UCB para gramáticas |
| `meta_compiler.py` | `AdaptiveGrammarSpace` | Espacio de gramáticas adaptativo vía α estimado |
| `benchmarks/benchmark_comprehensive.py` | `run_full_benchmark` | 22 casos benchmark |

---

## Dominios nuevos — versión 1.4.0

### Búsqueda de Arquitecturas Neurales (NeuralArchACF)

Reemplaza a NAS clásico: O(d³) vs O(épocas × batches).

```python
from acf_functor.neural_arch_acf import NeuralArchACF, NASReplacementSearch
import torch.nn as nn

# Fingerprint de una arquitectura
analyzer = NeuralArchACF()
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
fp = analyzer.fingerprint(model)
print(f"α global = {fp.global_alpha:.3f}, clase NC = {fp.global_nc_class}")
print(f"Capas cuello de botella: {fp.bottleneck_layers}")

# Búsqueda de arquitectura óptima (sin entrenamiento)
nas = NASReplacementSearch()
spec = {"in_dim": 32, "out_dim": 10, "max_layers": 4, "hidden_dims": [64, 128]}
candidates = nas.search(spec, n_candidates=20)
pareto = nas.pareto_front(candidates)
best = pareto[0]
print(f"Mejor candidata: α={best.global_alpha:.3f}, params={best.total_params}")
```

### Análisis de Alta Entropía (FinancialACF)

Diseñado para bolsa de valores y datos con alta entropía.

```python
import numpy as np
from acf_functor.stochastic_acf import FinancialACF, HighEntropyAnalyzer

# Retornos simulados (o reales)
returns = np.random.normal(0.001, 0.02, size=500)

# VaR certificado por Chebyshev
financial = FinancialACF(pce_degree=4)
report = financial.analyze(returns)
print(f"VaR 95%  = {report.var_certified.var_95:.4f}")
print(f"VaR 99%  = {report.var_certified.var_99:.4f}")
print(f"CVaR 95% = {report.var_certified.cvar_95:.4f}")
print(f"Volatilidad = {report.volatility:.4f}")
print(f"Sharpe LB = {report.sharpe_lower_bound:.4f}")

# Exponente de Hurst y persistencia
analyzer = HighEntropyAnalyzer(pce_degree=4)
price_series = np.cumsum(returns)
hurst = analyzer.hurst_exponent(price_series)
print(f"H = {hurst.H:.3f}  →  {hurst.interpretation}")
levy = analyzer.levy_alpha_stable(returns)
print(f"Lévy α = {levy.alpha_stable:.3f}, colas gordas = {levy.fat_tails}")
```

### Redes Bayesianas (BayesianNNAnalyzer)

Cuantifica incertidumbre en pesos y predicciones vía PCE.

```python
import torch, torch.nn as nn
import numpy as np
from acf_functor.stochastic_acf import BayesianNNAnalyzer

ensemble = [nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
            for _ in range(10)]
x = np.array([1.0, 0.5, -0.3, 0.8])

bnn = BayesianNNAnalyzer(pce_degree=4, n_ensemble=10)
report = bnn.analyze(ensemble, x)
print(f"Entropía de pesos = {report.weight_entropy:.4f}")
print(f"Banda de confianza predicción = {report.prediction_confidence_band:.4f}")
print(f"Índices Sobol = {[f'{s:.3f}' for s in report.sobol_indices[:4]]}")
print(f"Params efectivos ≤ totales: {report.effective_params} ≤ {report.total_params}")
```

### Meta-Compilador Adaptativo (BayesianSearch + AdaptiveGrammarSpace)

El compilador selecciona automáticamente la gramática más eficiente usando un surrogate GP.

```python
from acf_functor.meta_compiler import ACFMetaCompiler, MetaCompilerConfig
import numpy as np

f = lambda x: np.sin(3 * x) * np.exp(-x**2)
cfg = MetaCompilerConfig(
    strategy="adaptive",      # selección automática Bayesiana vs greedy
    adaptive_grammar=True,    # espacio de gramáticas adaptado al α estimado
    bayesian_budget=40,       # iteraciones máximas del surrogate GP
    max_degree=35,
    tolerance=1e-5,
)
compiler = ACFMetaCompiler(cfg)
result = compiler.compile(f, domain=(-2.0, 2.0))
print(f"Familia: {result.best_grammar.family}  grado: {result.best_grammar.degree}")
print(f"Error: {result.best_error:.2e}  α estimado: {result.alpha_estimate:.3f}")
```

### Estimador Unificado con 5 Métodos (ACFInvariantUnified)

El invariante α ahora es la mediana ponderada de 5 estimadores independientes.

```python
from acf_functor.invariant_unified import ACFInvariantUnified
import numpy as np

f = lambda x: np.sin(5 * x)
estimator = ACFInvariantUnified(use_stochastic=True, use_ode=True)
result = estimator.compute(f, domain=(-1.0, 1.0))
print(f"α combinado  = {result.alpha_combined:.4f}")
print(f"α espectral  = {result.alpha_spectral:.4f}")
print(f"α geométrico = {result.alpha_geometric:.4f}")
print(f"α estocástico= {result.alpha_stochastic:.4f}")
print(f"α ODE        = {result.alpha_ode:.4f}")
```

---

*Gideon v1.4.0 — Martínez's Invariant, Abril 2026*

---

## v1.4.0 — Dominios Algebraicos Extendidos (Mayo 2026)

Gideon ahora soporta cinco nuevas categorías de dominio algebraico.

### Anillos, Cuerpos y Polinomios (ALGACF-1..5)

```python
from acf_functor.algebraic_acf import (
    RealRing, GFpRing, GF2Ring, MatrixRing,
    AlgebraicACFReducer, reduce_over_ring
)

# Reducción sobre ℝ
report = reduce_over_ring([1.0, -3.0, 2.0, 1.0], RealRing())
print(f"α = {report.alpha:.4f}, FMAs = {report.fma_count}")

# Reducción sobre 𝔽₁₁
ring = GFpRing(11)
report = reduce_over_ring([4, 7, 2, 1], ring)
# Certificados: ALGACF-5
```

### Álgebras de Lie (ALGACF-3)

```python
from acf_functor.algebraic_acf import LieAlgebraFactory, LieAlgebraACFAnalyzer

# su(2) — constantes de estructura
f = LieAlgebraFactory.su2()   # ndarray shape (3,3,3)
report = LieAlgebraACFAnalyzer().analyze(f, algebra_name="su2")
print(f"Semisimple: {report.is_semisimple}, α = {report.alpha:.4f}")

# so(3) y Heisenberg también disponibles
f_heis = LieAlgebraFactory.heisenberg()
```

### Síntesis Booleana sobre GF(2) (ALGACF-4)

```python
from acf_functor.algebraic_acf import BooleanACFSynthesizer

# XOR de 2 bits: tabla de verdad [0,1,1,0]
synth = BooleanACFSynthesizer()
report = synth.synthesize([0, 1, 1, 0])
print(report.verilog_rtl)   # RTL Verilog generado
print(report.c_code)        # Código C generado
print(f"Grado ANF = {report.anf_degree}, Gate depth = {report.gate_depth}")
```

### Curvas Elípticas sobre 𝔽ₚ (ALGACF-5)

```python
from acf_functor.algebraic_acf import ECCACFReducer

ecc = ECCACFReducer(p=17, a=2, b=2)
P = (5, 1)
Q = (6, 3)
R = ecc.point_add(P, Q)
S = ecc.scalar_mul(5, P)     # k es el PRIMER argumento
report = ecc.analyze_reduction()
print(f"field_alpha = {report.field_alpha:.4f}")
```

### Topos ACF — Haces de Grothendieck (TOPOS-1..4)

```python
from acf_functor.topos_acf import ToposACFAnalyzer
import numpy as np

report = ToposACFAnalyzer().analyze(
    lambda x: np.sin(x),
    domain=(-np.pi, np.pi)
)
print(f"Gluing holds: {report.gluing_result.gluing_holds}")
print(f"Admissibility: {report.admissibility_Omega}")
print(f"Sequents: {report.geometric_sequents_valid}")
```

### p-ádico ACF — Mahler y Hensel (PADIC-1..3)

```python
from acf_functor.padic_acf import (
    PAdicACFReducer, hensel_lift,
    MahlerSeries, ramanujan_tau_coefficients
)

# Reducción p-ádica
reducer = PAdicACFReducer(p=5)
report = reducer.reduce([1, 2, 3, 4, 5], epsilon=1e-3)
print(any("PADIC-2" in c for c in report.certificates))  # True

# Levantamiento de Hensel: x² - 2 ≡ 0 mod 7
result = hensel_lift([2, 0, 1], a0=3, p=7, precision=5)  # 3² = 9 ≡ 2 mod 7
print(f"Converge: {result.converged}")

# Coeficientes de Ramanujan (1-indexados: tau[1]=1, tau[2]=-24)
tau = ramanujan_tau_coefficients(10)
```

### Formas Modulares ACF (MOD-1..4)

```python
from acf_functor.modular_acf import ModularFormLibrary, ModularACFReducer

lib = ModularFormLibrary()
e4  = lib.e4()      # Eisenstein peso 4
e6  = lib.e6()      # Eisenstein peso 6
delta = lib.delta() # Función Δ de Ramanujan, weight=12

reducer = ModularACFReducer()
report = reducer.reduce("E4", [1, 240, 2160], weight=4)
print(f"α = {report.alpha:.4f}, certificados = {report.certificates}")
```

### Finanzas y Caos ACF (FIN-1..5)

```python
from acf_functor.finance_acf import (
    estimate_hurst, VolatilitySurfaceReducer,
    compute_risk_via_pce, detect_regime_changes,
    analyze_invariant_density
)
import numpy as np

# FIN-1: Hurst — PASAR INCREMENTOS, no precios
increments = np.random.standard_normal(500)
hurst = estimate_hurst(increments)
print(f"H = {hurst.hurst_exponent:.3f}")  # ≈ 0.5 para BM

# FIN-2: Superficie de volatilidad
reducer = VolatilitySurfaceReducer()
vol_report = reducer.reduce(
    sigma_fn=lambda k, T: 0.20 + 0.01 * k**2,
    k_range=(-0.5, 0.5),
    T_values=[0.25, 0.5, 1.0]
)

# FIN-3: VaR/CVaR vía PCE
# payoff_fn recibe np.array([xi]) — usar float(np.asarray(x).flat[0])
def call_payoff(x):
    xi = float(np.asarray(x).flat[0])
    return max(xi - 1.0, 0.0)
risk = compute_risk_via_pce(call_payoff)
print(f"VaR 95% = {risk.var_95:.4f}, ES 95% = {risk.es_95:.4f}")

# FIN-4: Detección de cambios de régimen
series = np.concatenate([np.random.standard_normal(200),
                          0.5 * np.random.standard_normal(200)])
reg = detect_regime_changes(series, window_size=50, alpha_jump_threshold=0.05)
print(f"Cambios de régimen: {len(reg.regime_changes)}")

# FIN-5: Densidad invariante de la logística
density = analyze_invariant_density(lambda x: 4*x*(1-x), domain=(0,1))
print(f"Lyapunov = {density.lyapunov_exponent:.4f}")  # ≈ ln(2)
```

---

*Gideon v1.4.0 — 8 dominios algebraicos, 105 tests, 17 certificados Lean 4*
