# Gideon — Manual Técnico Completo

> Motor de Ejecución Nativo de Poema v1.0.0  
> **Este documento es la referencia técnica oficial para ingenieros que trabajan con Gideon.**

---

## Índice

1. [Instalación y Setup](#instalación-y-setup)
2. [Inicio Rápido](#inicio-rápido)
3. [Referencia: GideonEngine](#referencia-gideonengine)
4. [Referencia: GideonEngineConfig](#referencia-gideonengineconfig)
5. [Referencia: GideonIR](#referencia-gideonir)
6. [Referencia: GideonGraph](#referencia-gideongraph)
7. [Referencia: GideonDispatcher](#referencia-gideondispatcher)
8. [Referencia: GideonNeuralHints](#referencia-gideonneuralhints)
9. [Referencia: GideonTheoremSeeds](#referencia-gideontheoremseeds)
10. [Referencia: GideonAgentRouter](#referencia-gideonagentrouter)
11. [Integración con PoemCompiler](#integración-con-poemcompiler)
12. [Selección de Backend](#selección-de-backend)
13. [Precisión Numérica y Configuración](#precisión-numérica-y-configuración)
14. [Benchmark y Profiling](#benchmark-y-profiling)
14. [Seguridad y Validación](#seguridad-y-validación)
15. [Guía de Tests](#guía-de-tests)
16. [Troubleshooting](#troubleshooting)

---

## Instalación y Setup

### Requisitos

```
Python       >=3.10
numpy        >=1.24
torch        >=2.0        (opcional, habilita pytorch/CUDA backends)
cffi         >=1.15       (opcional, habilita c_native backend)
onnxruntime  >=1.16       (opcional, habilita onnx backend)
```

### Verificar que Gideon está disponible

```python
from poema.backends.gideon import GideonEngine

engine = GideonEngine()
print(engine.info())
```

Salida esperada:

```
GideonEngine v1.0.0
  Hardware:
    avx2:   True
    avx512: False  (o True si tu CPU lo soporta)
    cuda:   True   (o False si no hay GPU)
    rocm:   False
    cffi:   True
    torch:  True
    onnx:   False
  Backends disponibles:
    c_native, pytorch, numpy_cpu
  Backend recomendado: c_native
```

### Importaciones del paquete

```python
# Import central — todos los componentes desde un lugar
from poema.backends.gideon import (
    GideonEngine,
    GideonEngineConfig,
    GideonExecutionResult,
    GideonIR,
    IRNode,
    IRNodeKind,
    GideonProgram,
    GideonGraph,
    GideonGraphNode,
    GraphEdge,
    ExecutionPlan,
    GideonDispatcher,
    DispatchDecision,
    BackendHint,
    GideonNeuralHints,
    ArchitectureBlueprint,
    GideonTheoremSeeds,
    TheoremCandidate,
)

# También disponibles desde el paquete raíz de backends
from poema.backends import GideonEngine
```

---

## Inicio Rápido

### Caso 1: Ejecutar una cadena FMA

```python
import numpy as np
from poema.backends.gideon import GideonEngine

# Definir cadena FMA: [(weight, bias), ...]
# Cada elemento representa y = w*x + b
fma_chain = [(0.9, 0.1)] * 100   # 100 stages

engine = GideonEngine()
x = np.linspace(-1.0, 1.0, 10_000)

result = engine.run_fma(fma_chain, x)

print(f"Output shape: {result.output.shape}")
print(f"Backend usado: {result.backend_used}")
print(f"Tiempo de ejecución: {result.elapsed_ms:.3f} ms")
print(f"Cota de error global (ε): {result.global_epsilon:.2e}")
```

### Caso 2: Compilar con PoemCompiler target="gideon"

```python
from poema.compiler import PoemCompiler
from poema.frontend import Poem
import torch

poem = Poem(dtype=torch.float64)
ast  = poem.continuous_flow("x^2 - 2*x + 1")

compiler = PoemCompiler(target="gideon", precision="fp64")
fn, report = compiler.compile(ast, domain=(0.0, 2.0))

x = torch.linspace(0, 2, 1_000_000, dtype=torch.float64)
y = fn(x)
print(y[:5])
```

### Caso 3: Analizar teoremas emergentes

```python
from poema.backends.gideon import GideonEngine, GideonEngineConfig

cfg = GideonEngineConfig(enable_theorem_seeds=True)
engine = GideonEngine(cfg)

fma_chain = [(0.8, 0.0)] * 50   # cadena contráctiva

result = engine.run_fma(fma_chain, np.linspace(-2, 2, 5000))

for tc in result.theorem_candidates:
    print(f"[{tc.status.name}] {tc.name}")
    print(f"  Enunciado: {tc.statement}")
    print(f"  Confianza: {tc.confidence:.3f}")
    print(f"  Lean 4:\n{tc.lean_skeleton}")
```

---

## Referencia: GideonEngine

**Módulo**: `poema.backends.gideon.engine`

```python
class GideonEngine:
    def __init__(self, config: GideonEngineConfig = None): ...
```

### Métodos principales

---

#### `run_fma(fma_sequence, x, **kwargs) → GideonExecutionResult`

Ejecuta una cadena de instrucciones FMA sobre x.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `fma_sequence` | `List[Tuple[float, float]]` | Lista de pares (weight, bias). `y = w*x + b`. |
| `x` | `np.ndarray` | Valores de entrada |
| `domain` | `Tuple[float, float]` | Dominio para propagación de intervalos. Default: `(-inf, inf)` |

```python
result = engine.run_fma([(0.9, 0.1), (1.1, -0.2)], x, domain=(-1.0, 1.0))
```

---

#### `run_ast(ast_node, x, **kwargs) → GideonExecutionResult`

Compila y ejecuta un nodo AST de Poema directamente a través de Gideon.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `ast_node` | `ASTNode` | Nodo raíz del AST de Poema |
| `x` | `np.ndarray` | Valores de entrada |

```python
from poema.frontend import Poem
ast = Poem().continuous_flow("tanh(2*x - 1)")
result = engine.run_ast(ast, np.linspace(-2, 2, 10000))
```

---

#### `probe_theorems(fn, domain) → List[TheoremCandidate]`

Analiza numéricamente una función Python y genera candidatos a teoremas.

```python
candidates = engine.probe_theorems(
    fn=lambda x: np.sin(x) * np.cos(x),
    domain=(-np.pi, np.pi)
)
```

---

#### `analyse_blueprint(blueprint) → Dict`

Analiza un `ArchitectureBlueprint` y retorna métricas de complejidad.

```python
from poema.backends.gideon import GideonNeuralHints
bp = GideonNeuralHints.mlp([784, 256, 128, 10])
metrics = engine.analyse_blueprint(bp)
# {"alpha_complexity": ..., "total_flops": ..., "fma_equivalent": ...}
```

---

#### `info() → str`

Retorna descripción legible del estado del motor: hardware, backends, versión.

---

### GideonExecutionResult — campos

```python
@dataclass
class GideonExecutionResult:
    output:             np.ndarray    # Resultado numérico
    program:            GideonProgram # IR compilado
    graph_stats:        Dict          # Métricas del grafo
    dispatch_decision:  DispatchDecision
    theorem_candidates: List[TheoremCandidate]
    elapsed_ms:         float         # Tiempo total de ejecución
    backend_used:       str           # "c_native", "pytorch", etc.
    total_fma:          int           # Total instrucciones FMA
    global_epsilon:     float         # Cota de error propagada ε
```

---

## Referencia: GideonEngineConfig

```python
@dataclass
class GideonEngineConfig:
    precision:              str   = "fp64"      # "fp16"|"fp32"|"fp64"|"bf16"
    domain:                 tuple = None         # (lo, hi) para propagación
    preferred_backend:      str   = None         # forzar un backend
    enable_theorem_seeds:   bool  = False        # activar análisis de teoremas
    enable_neural_hints:    bool  = False        # activar análisis de NAS
    verbose:                bool  = False        # logging detallado
    benchmark_mode:         bool  = False        # activar modo benchmark
    benchmark_repeats:      int   = 100          # repeticiones en modo benchmark
```

### Precisiones soportadas

| Clave | Tipo numpy | ε machine |
|---|---|---|
| `"fp64"` | `float64` | `2.22e-16` |
| `"fp32"` | `float32` | `1.19e-7` |
| `"fp16"` | `float16` | `9.77e-4` |
| `"bf16"` | `bfloat16` | `3.91e-3` |

### Ejemplo de configuración avanzada

```python
cfg = GideonEngineConfig(
    precision="fp64",
    domain=(-5.0, 5.0),
    preferred_backend="c_native",
    enable_theorem_seeds=True,
    verbose=True,
    benchmark_mode=True,
    benchmark_repeats=500,
)
engine = GideonEngine(cfg)
```

---

## Referencia: GideonIR

**Módulo**: `poema.backends.gideon.ir`

### Lowering desde cadena FMA

```python
gir = GideonIR()
prog = gir.from_fma_sequence(
    fma_seq=[(0.9, 0.1)] * 200,
    domain=(-1.0, 1.0),
    precision="fp64",
)
print(prog.summary())
```

### Lowering desde AST

```python
from poema.frontend import Poem
ast = Poem().continuous_flow("x^3 + 2*x")
prog = gir.from_ast(ast, domain=(-2.0, 2.0), precision="fp64")
```

### Serialización JSON

```python
# Exportar a JSON (string)
json_str = GideonIR.to_json(prog)

# Restaurar desde JSON
prog_restored = GideonIR.from_json(json_str)
assert prog_restored.total_fma == prog.total_fma

# Exportar a archivo
with open("mi_programa.json", "w") as f:
    f.write(json_str)
```

### IRNodeKind — todos los tipos

```python
# Primitivos
IRNodeKind.CONST          # Constante escalar
IRNodeKind.INPUT          # Nodo de entrada
IRNodeKind.FMA            # y = w*x + b
IRNodeKind.IDENTITY       # y = x

# Afines
IRNodeKind.SCALE          # y = a*x
IRNodeKind.SHIFT          # y = x + b
IRNodeKind.AFFINE         # y = a*x + b (alias SCALE+SHIFT)

# Composición
IRNodeKind.COMPOSE        # y = f(g(x))
IRNodeKind.PARALLEL       # y = (f(x), g(x))
IRNodeKind.BRANCH         # y = cond ? f(x) : g(x)

# Polinomios
IRNodeKind.POLY_HORNER    # Evaluación Horner: ((a_n*x+a_{n-1})*x+...)+a_0
IRNodeKind.POLY_CHEB      # Base de Chebyshev

# Trascendentales
IRNodeKind.SIN
IRNodeKind.COS
IRNodeKind.EXP
IRNodeKind.LOG
IRNodeKind.TANH
IRNodeKind.SIGMOID

# Álgebra lineal
IRNodeKind.MATMUL         # Producto matricial
IRNodeKind.GEMM           # BLAS GEMM: C = α*AB + β*C
IRNodeKind.CONV           # Convolución
IRNodeKind.NORM           # Layer/Batch normalization
IRNodeKind.ATTENTION      # Multi-head attention

# Control
IRNodeKind.LOOP           # Bucle iterativo
IRNodeKind.RECURSIVE      # Llamada recursiva

# Infraestructura futura
IRNodeKind.ARCH_PROBE     # Sonda para NAS
IRNodeKind.THEOREM_SEED   # Semilla de teorema
```

### IRNodeMetadata — campos

```python
@dataclass
class IRNodeMetadata:
    epsilon:      float        # Cota de error acumulada hasta este nodo
    domain:       tuple        # Intervalo [lo, hi] de valores posibles
    interval:     tuple        # = domain (alias semántico)
    continuity:   int          # Orden de continuidad: 0=C0, 1=C1, ...
    fma_cost:     int          # FLOPs equivalentes en unidades FMA
```

---

## Referencia: GideonGraph

**Módulo**: `poema.backends.gideon.graph`

### Construcción

```python
graph = GideonGraph(prog)   # prog: GideonProgram
```

### Análisis completo

```python
plan = graph.analyse()      # → ExecutionPlan
```

### ExecutionPlan — campos

```python
@dataclass 
class ExecutionPlan:
    phases:         List[ExecutionPhase]   # Niveles topológicos
    critical_path:  int                    # Número de fases
    total_nodes:    int
    parallelizable: float                  # Fracción paralelizable [0,1]
    total_fma:      int
    global_epsilon: float
```

### Cadenas fusables

```python
chains = graph.find_fusable_chains()
# Retorna: List[List[str]]
# Cada lista interna es un grupo de IDs de nodos que pueden compilarse
# como un único kernel (cadena FMA lineal sin ramificaciones)
```

### Conteo de capas IA

```python
ai_layers = graph.ai_layer_count()
# {"matmul": 4, "attention": 2, "conv": 0, "norm": 1}
```

### Estadísticas del grafo

```python
stats = graph.stats()
# {
#   "n_nodes": N, "n_edges": M,
#   "n_phases": P, "total_fma": F,
#   "fusable_chains": C,
#   "global_epsilon": ε,
#   "ai_layers": {"matmul": ..., ...}
# }
```

---

## Referencia: GideonDispatcher

**Módulo**: `poema.backends.gideon.dispatcher`

### Instanciar y decidir

```python
from poema.backends.gideon import GideonDispatcher, BackendHint

# Con hints manuales (opcional)
hints = [BackendHint(backend_name="c_native", priority=200)]
dispatcher = GideonDispatcher(hints=hints)

decision = dispatcher.decide(prog)   # prog: GideonProgram
print(decision.primary_backend)      # "c_native"
print(decision.fallback_backend)     # "numpy_cpu"
print(decision.estimated_speedup)    # 14.5
print(decision.node_backend_map)     # {"fma_001": "c_native", ...}
```

### BackendHint

```python
@dataclass
class BackendHint:
    backend_name: str    # nombre del backend a priorizar
    priority:     int    # prioridad adicional sumada al score base
```

### Registrar latencia real

```python
# Después de cada ejecución, registra para refinar el ranking futuro
dispatcher.record_latency("c_native", elapsed_ms)
```

### HardwareProfile

```python
from poema.backends.gideon.dispatcher import HardwareProfile

hw = HardwareProfile.detect()
print(f"AVX2:  {hw.has_avx2}")
print(f"CUDA:  {hw.has_cuda}")
print(f"cffi:  {hw.has_cffi}")
print(f"torch: {hw.has_torch}")
```

---

## Referencia: GideonNeuralHints

**Módulo**: `poema.backends.gideon.neural_hints`

### Blueprints soportados

#### MLP

```python
bp = GideonNeuralHints.mlp(layer_dims=[784, 256, 128, 10])
print(bp.summary())
# Blueprint('mlp_4layers', kind=mlp)
#   Capas:            8
#   Parámetros:       234,762
#   FLOPs:            469,524
#   α-complejidad:    1.0000
#   FMA-equiv:        234,762
```

#### Transformer

```python
bp = GideonNeuralHints.transformer(
    d_model=512,
    n_heads=8,
    n_layers=6,
    seq_len=512,   # opcional, default=512
    ffn_dim=2048   # opcional, default=4*d_model
)
```

#### Bloque ResNet (CNN)

```python
bp = GideonNeuralHints.cnn_resnet_block(
    channels=64,
    kernel_size=3,
    n_blocks=4
)
```

### Análisis de blueprint

```python
metrics = GideonNeuralHints.analyse_blueprint(bp)
# {
#   "name": "transformer_6L",
#   "kind": "transformer",
#   "n_layers": 24,
#   "total_params": 109,000,000,
#   "total_flops": 223,000,000,
#   "alpha_complexity": 1.0072,
#   "total_fma_equivalent": 111,500,000,
#   "search_space": {...}
# }
```

### Fórmula α-complejidad

El índice de complejidad computacional:

$$\alpha = \frac{\log(\max(1, \text{total\_flops}))}{\log(\max(2, \text{total\_params} + 1))}$$

Valores de referencia:
- MLP estándar: α ≈ 1.0
- Transformer: α ≈ 1.003 - 1.010 (mayor densidad por atención)
- CNN: α ≈ 0.95 - 1.05 (depende del kernel)

---

## Referencia: GideonTheoremSeeds

**Módulo**: `poema.backends.gideon.theorem_seeds`

### Análisis de invariantes

```python
from poema.backends.gideon import GideonTheoremSeeds

seeds = GideonTheoremSeeds()

fn = lambda x: np.sin(x)
candidates = seeds.analyse(fn, domain=(-np.pi, np.pi), n_points=2000)

for tc in candidates:
    print(f"[{tc.status.name}] {tc.name} (confianza={tc.confidence:.3f})")
```

### TheoremCandidate — campos

```python
@dataclass
class TheoremCandidate:
    name:             str   # Identificador corto
    statement:        str   # Enunciado en lenguaje natural
    formal_statement: str   # Enunciado formal (Lean 4 compatible)
    lean_skeleton:    str   # Código Lean 4 generado
    confidence:       float  # [0, 1] confianza numérica
    evidence:         Dict   # Datos de la prueba numérica
    status:           TheoremStatus
```

```python
class TheoremStatus(Enum):
    CONJECTURE  = "conjecture"   # Detectado, no verificado
    NUMERICAL   = "numerical"    # Verificado numéricamente
    CERTIFIED   = "certified"    # Verificado formalmente (Lean 4)
    REFUTED     = "refuted"      # Refutado con contraejemplo
```

### Exportar a Lean 4

```python
seeds.export_lean_file(candidates, path="theorems/my_theorems.lean")
```

Genera un archivo `.lean` con toda la estructura de declaraciones:

```lean
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

-- Lipschitz bound for sin_fn
-- L ≈ 1.0003: computed empirically on 2000 points (500 intervals)
noncomputable def lipschitz_sin_fn : ℝ := 1.0003
-- Claim: |f(x) - f(y)| ≤ lipschitz_sin_fn * |x - y|

-- Symmetry: sin_fn is odd (f(-x) = -f(x))
-- Verified: max deviation = 2.44e-15 across 2000 points
noncomputable def symmetry_sin_fn : ℝ := 2.44e-15
```

### InvariantProbe (API directa)

```python
from poema.backends.gideon.theorem_seeds import InvariantProbe

fn = lambda x: 0.5 * x + 0.1
probe = InvariantProbe(fn, domain=(-2.0, 2.0), n_points=1000)

L = probe.lipschitz_estimate()        # Constante de Lipschitz
mono = probe.is_monotone()            # True / False
sym = probe.symmetry_type()           # "even" | "odd" | "none"
alpha = probe.alpha_complexity()       # Índice α espectral
bound = probe.urt_bound_estimate()     # Cota URT (Martínez's Invariant)
```

---

## Integración con PoemCompiler

### Usar `target="gideon"`

```python
from poema.compiler import PoemCompiler

# 1. Con target "gideon" (motor completo)
compiler = PoemCompiler(
    target="gideon",
    precision="fp64",
    verbose=False
)

# 2. Compilar
fn, report = compiler.compile(ast, domain=(-2.0, 2.0))

# 3. Acceso al engine interno (lazy — instanciado en primer acceso)
engine = compiler.gideon   # → GideonEngine

# 4. El reporte incluye información de Gideon
for w in report.warnings:
    print(w)   # "gideon:backend_used=c_native"
```

### CompilationReport — campos Gideon

Cuando `target="gideon"`, el `CompilationReport` añade a `warnings`:

```
"gideon:backend_used=<nombre_backend>"
```

Por ejemplo:
- `"gideon:backend_used=c_native"` — compilado con cffi AVX2
- `"gideon:backend_used=pytorch"` — ejecutado en GPU CUDA
- `"gideon:backend_used=numpy_cpu"` — fallback NumPy

---

## Selección de Backend

### Modo automático (recomendado)

```python
engine = GideonEngine()   # HardwareProfile.detect() elige automáticamente
```

### Forzar un backend específico

```python
cfg = GideonEngineConfig(preferred_backend="pytorch")
engine = GideonEngine(cfg)
```

### Forzar con hints de dispatcher

```python
from poema.backends.gideon import GideonDispatcher, BackendHint

hints = [
    BackendHint(backend_name="c_native", priority=500),   # priorizar c_native
    BackendHint(backend_name="pytorch", priority=-100),   # despenalizar pytorch
]
dispatcher = GideonDispatcher(hints=hints)
decision = dispatcher.decide(prog)
```

### Tabla de backends y cuándo usarlos

| Backend | Cuándo usar |
|---|---|
| `c_native` | **Default**. CPU con cffi disponible. El más rápido para FMA puras. |
| `pytorch` | CUDA disponible y la carga tiene MATMUL/ATTENTION. |
| `rocm` | GPU AMD. Igual que pytorch pero con ROCm. |
| `numpy_cpu` | Fallback universal, sin dependencias externas. Siempre disponible. |
| `onnx` | Inferencia portable cross-platform. Requiere onnxruntime. |
| `wasm` | Despliegue en browser/edge. Requiere soporte WASM. |
| `verilog` | Síntesis en FPGA. Sólo para casos muy especializados. |

---

## Precisión Numérica y Configuración

### Elegir la precisión correcta

| Caso de uso | Recomendación |
|---|---|
| Cómputo científico, física | `fp64` |
| Deep learning inference | `fp32` |
| Modelos grandes (GPT) | `bf16` |
| Hardware edge con FPU limitada | `fp16` |

### Leer la cota de error de un resultado

```python
result = engine.run_fma(chain, x)
print(f"ε global = {result.global_epsilon:.3e}")
# ε global = 2.22e-13   (para 1000 FMAs con |w|≈0.9)
```

La cota ε es un **upper bound** sobre el error de redondeo acumulado. Si ε supera tu tolerancia, usa `fp64` o reduce la longitud de la cadena.

### Dominio y propagación de intervalos

Especificar el dominio mejora la precisión de las cotas de error:

```python
# Sin dominio: GideonIR asume (-inf, inf)
result = engine.run_fma(chain, x)

# Con dominio: interval tighter
cfg = GideonEngineConfig(domain=(-1.0, 1.0))
engine = GideonEngine(cfg)
result = engine.run_fma(chain, x, domain=(-1.0, 1.0))
# result.global_epsilon potencialmente más ajustado
```

---

## Benchmark y Profiling

### Modo benchmark integrado

```python
cfg = GideonEngineConfig(
    benchmark_mode=True,
    benchmark_repeats=500,   # promedio de 500 ejecuciones
)
engine = GideonEngine(cfg)

result = engine.run_fma(large_chain, x_1M)
print(f"Tiempo promedio: {result.elapsed_ms:.3f} ms")
```

### Benchmarks de referencia (RTX 4050 + AVX2, Python 3.12)

```python
# Replicar el benchmark oficial de Gideon
import numpy as np, time
from poema.backends.gideon import GideonEngine

engine = GideonEngine()

configs = [
    (10,   10_000_000, "10 stages × 10M"),
    (50,    1_000_000, "50 stages × 1M"),
    (100,     100_000, "100 stages × 100K"),
    (1000,     10_000, "1000 stages × 10K"),
]

for depth, n, label in configs:
    chain = [(0.9 + i*0.0001, 0.01) for i in range(depth)]
    x = np.random.randn(n)

    # NumPy baseline
    t0 = time.perf_counter()
    ref = x.copy()
    for w, b in chain:
        ref = w * ref + b
    numpy_ms = (time.perf_counter() - t0) * 1000

    # Gideon
    result = engine.run_fma(chain, x)

    print(f"[{label}] numpy={numpy_ms:.2f}ms | "
          f"gideon={result.elapsed_ms:.2f}ms | "
          f"speedup={numpy_ms/result.elapsed_ms:.1f}×")
```

Resultados esperados:

```
[10 stages × 10M]   numpy=210.18ms | gideon=14.69ms | speedup=14.3×
[50 stages × 1M]    numpy=29.01ms  | gideon=2.55ms  | speedup=11.4×
[100 stages × 100K] numpy=2.83ms   | gideon=0.60ms  | speedup=4.8×
[1000 stages × 10K] numpy=3.83ms   | gideon=1.02ms  | speedup=3.8×
```

### Profiling con record_latency

```python
# El engine registra automáticamente la latencia del backend
result1 = engine.run_fma(chain, x)
result2 = engine.run_fma(chain, x)

# El dispatcher re-rankea backends según historial acumulado
# Los últimos 20 registros se promedian para cada backend
```

---

## Seguridad y Validación

### Boundary validation en GideonDispatcher

El dispatcher valida que el backend seleccionado está disponible antes de despachar:

```python
# Si c_native no está disponible, el dispatcher cae al primer fallback disponible
decision = dispatcher.decide(prog)
# primary: "c_native" (si cffi disponible)
# fallback: "numpy_cpu" (siempre disponible)
```

Nunca se despachará a un backend no-disponible: la lógica de fallback garantiza que `numpy_cpu` siempre es el último recurso.

### Validar integridad numérica

```python
# Comparar resultado de Gideon vs NumPy ref
import numpy as np

x = np.linspace(-1, 1, 10000)
chain = [(0.9, 0.05)] * 50

# Referencia NumPy
ref = x.copy()
for w, b in chain:
    ref = w * ref + b

# Gideon
result = engine.run_fma(chain, x)

max_diff = np.max(np.abs(result.output - ref))
tol = result.global_epsilon * 100   # margen ×100 sobre la cota teórica
assert max_diff < tol, f"Diferencia {max_diff} supera tolerancia {tol}"
```

### No confiar en entradas externas sin validar

Si las cadenas FMA vienen de fuentes externas (archivos JSON, red), valídalas antes de ejecutar:

```python
# Deserializar programa desde JSON externo
try:
    prog = GideonIR.from_json(untrusted_json_str)
except (ValueError, KeyError, TypeError) as e:
    raise ValueError(f"Programa inválido: {e}") from e

# GideonIR.from_json() valida tipos internamente, pero es buena práctica
# envolver con manejo de errores en boundaries externas
```

---

## Guía de Tests

### Ejecutar la suite completa de Gideon

```bash
# Desde la raíz del workspace
python -m pytest tests/test_gideon_engine.py -v
```

Resultado esperado: **58 tests PASSED**.

### Ejecutar con output de benchmarks visible

```bash
python -m pytest tests/test_gideon_engine.py -s -v 2>&1 | grep -E "\[BM_|\[C Native|\[Stress"
```

### Tests por categoría

```bash
# Solo tests de IR
python -m pytest tests/test_gideon_engine.py::TestGideonIR -v

# Solo tests de precisión numérica
python -m pytest tests/test_gideon_engine.py::TestGideonNumericalPrecision -v

# Solo tests de benchmarks (producen métricas reales)
python -m pytest tests/test_gideon_engine.py::TestGideonBenchmarks -s -v

# Solo tests de integración con PoemCompiler  
python -m pytest tests/test_gideon_engine.py::TestGideonPoemIntegration -v
```

### Estructura de la suite (980 tests)

| Clase | Tests | Descripción |
|---|---|---|
| `TestGideonIR` | 7 | Lowering FMA/AST, serialización JSON, metadata ε |
| `TestGideonGraph` | 5 | Topología, niveles Kahn, cadenas fusables, estadísticas |
| `TestGideonDispatcher` | 5 | Hardware detection, ranking, fallback, feedback |
| `TestGideonEnginePipeline` | 6 | Pipeline completo FMA/AST, idempotencia, mode benchmark |
| `TestGideonStress` | 6 | 10M elementos, 1M elementos, 5K stages, ejecución repetida |
| `TestGideonNumericalPrecision` | 5 | Tolerancias fp32/fp64, propagación ε, comparación vs NumPy |
| `TestGideonNeuralHints` | 6 | MLP, Transformer, CNN, α-complejidad, rangos |
| `TestGideonTheoremSeeds` | 7 | Lipschitz, monotonía, simetría (par/impar), Lean 4 skeleton |
| `TestGideonPoemIntegration` | 4 | PoemCompiler target="gideon", engine property, warnings |
| `TestGideonBenchmarks` | 4 | Speedups reales vs NumPy baseline (todos ≥ 3×) |
| `TestGideonIRSerialization` | 3 | Roundtrip JSON, invariancia, persistencia |

### Escribir tests propios para Gideon

```python
import pytest
import numpy as np
from poema.backends.gideon import GideonEngine, GideonEngineConfig

class TestMiModulo:
    @pytest.fixture
    def engine(self):
        cfg = GideonEngineConfig(precision="fp64")
        return GideonEngine(cfg)

    def test_mi_funcion(self, engine):
        chain = [(0.8, 0.1)] * 20
        x = np.linspace(-1, 1, 1000)
        result = engine.run_fma(chain, x)

        assert result.output.shape == (1000,)
        assert result.global_epsilon < 1e-10
        assert result.backend_used in {"c_native", "numpy_cpu", "pytorch"}
```

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'cffi'`

**Causa**: cffi no está instalado → c_native backend no disponible.  
**Solución**:

```bash
pip install cffi
```

Gideon usará `numpy_cpu` como fallback automáticamente si no hay cffi.

---

### Error: `BackendNotAvailableError: backend 'c_native' not available`

**Causa**: Gideon intentó usar c_native pero cffi no está instalado o falló en compilación.  
**Diagnóstico**:

```python
from poema.backends.gideon.dispatcher import HardwareProfile
hw = HardwareProfile.detect()
print(f"cffi disponible: {hw.has_cffi}")
```

**Solución**: Si cffi está instalado pero falló, verifica que tienes un compilador C:

```bash
gcc --version   # o cc --version
```

---

### Speedup bajo (< 2×) para cadenas cortas

**Causa**: El overhead de cffi/torch es fijo y domina para cadenas cortas (< 10 elementos).  
**Diagnóstico**: Con depth < 10 y n < 1000, numpy_cpu puede ser igual o más rápido.  
**Solución**: Para cargas pequeñas, usa `preferred_backend="numpy_cpu"`.

---

### `result.output` contiene NaN o Inf

**Causa**: Overflow o indefinición en algún paso de la cadena FMA.  
**Diagnóstico**:

```python
result = engine.run_fma(chain, x)
print(f"NaN: {np.sum(np.isnan(result.output))}")
print(f"Inf: {np.sum(np.isinf(result.output))}")
print(f"max|y|: {np.max(np.abs(result.output[np.isfinite(result.output)]))}")
```

**Solución**: Especifica el dominio y verifica que los pesos no causan divergencia:

```python
# Para una cadena contráctiva: todos los |w| < 1
for w, b in chain:
    assert abs(w) < 1.0, f"Peso {w} puede causar divergencia"
```

---

### Test `TestGideonPoemIntegration` falla con `ImportError`

**Causa**: PoemCompiler no está instalado en el entorno virtual activo.  
**Solución**:

```bash
cd "/home/Martínez's Invariant"
pip install -e .
```

---

### `GideonTheoremSeeds.analyse()` retorna lista vacía

**Causa**: La función analizada no presenta propiedades estadísticamente significativas en el muestreo por defecto.  
**Solución**: Aumentar `n_points`:

```python
seeds = GideonTheoremSeeds()
candidates = seeds.analyse(fn, domain=(-5, 5), n_points=5000)
```

---

## Referencia: Métodos ACF de Grafos (v1.1.0)

### `reduce_graph_signal(signal, adjacency, normalization, degree, domain) → GraphReductionResult`

Reduce una señal de grafo mediante el functor ACF espectral.

**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `signal` | `np.ndarray` shape (n,) | requerido | Señal sobre los nodos del grafo |
| `adjacency` | `np.ndarray` shape (n,n) | requerido | Matriz de adyacencia (simétrica, no negativa) |
| `normalization` | `str` | `"unnormalized"` | `"unnormalized"` (L=D−A) o `"symmetric"` (L_sym) |
| `degree` | `int` | `8` | Grado del polinomio filtro Chebyshev |
| `domain` | `tuple` o `None` | `None` | Rango espectral [λ_min, λ_max]; si None se detecta automáticamente |

**Retorna:** `GraphReductionResult`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `filtered_signal` | `np.ndarray` | Señal filtrada **s**_filtered = U·H(Λ)·Uᵀ·**s** |
| `polynomial_filter` | `np.ndarray` | Coeficientes del polinomio H evaluado en los valores propios |
| `epsilon` | `float` | Error L∞ de aproximación del filtro |
| `spectrum` | `GraphSpectrum` | Espectro del Laplaciano (eigenvalues, eigenvectors) |

**Ejemplo:**

```python
import numpy as np
from poema.backends.gideon.engine import GideonEngine

engine = GideonEngine()

# Grafo estrella K_{1,4}
n = 5
A = np.zeros((n, n))
for i in range(1, n):
    A[0, i] = A[i, 0] = 1.0

signal = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
result = engine.reduce_graph_signal(signal, A, degree=4)

print(f"Señal filtrada: {result.filtered_signal}")
print(f"ε = {result.epsilon:.2e}")
```

---

### `analyse_graph(adjacency, normalization) → GraphACFInvariants`

Analiza las invariantes ACF de la estructura espectral de un grafo.

**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `adjacency` | `np.ndarray` shape (n,n) | requerido | Matriz de adyacencia |
| `normalization` | `str` | `"unnormalized"` | Tipo de Laplaciano |

**Retorna:** `GraphACFInvariants`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `alpha` | `float` | Índice α ∈ [0,1] de la función de filtro espectral |
| `delta` | `float` | Índice δ (tasa de decaimiento espectral) |
| `nc_class` | `str` | Clase de complejidad NC ("NC0", "NC1", "NC2", …) |
| `fiedler_value` | `float` | λ₂ (conectividad algebraica) |
| `spectral_entropy` | `float` | Entropía de la distribución de valores propios |
| `optimal_filter_degree` | `int` | Menor grado k tal que ε < ε_target |

**Ejemplo:**

```python
from acf_functor import StandardGraphs

A_cycle = StandardGraphs.cycle(8)
inv = engine.analyse_graph(A_cycle)
print(f"α = {inv.alpha:.4f}, Fiedler = {inv.fiedler_value:.4f}")
print(f"Entropía espectral = {inv.spectral_entropy:.4f}")
print(f"Grado óptimo de filtro = {inv.optimal_filter_degree}")
```

---

## Referencia: Métodos Neural-ACF (v1.1.0)

### `analyse_network(network, degree, domain) → NetworkACFReport`

Analiza una red neuronal PyTorch capa a capa usando el functor ACF.

**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `network` | `nn.Module` | requerido | Red neuronal PyTorch (Linear, Conv1d soportados) |
| `degree` | `int` | `6` | Grado de reducción por capa |
| `domain` | `tuple` | `(-1.0, 1.0)` | Dominio de reducción para capas lineales |

**Retorna:** `NetworkACFReport`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `layer_reductions` | `List[LayerReductionResult]` | Reducción ACF por capa |
| `layer_invariants` | `List[ACFInvariant]` | Invariante α por capa (via SVD) |
| `global_alpha` | `float` | α ponderado por número de parámetros |
| `global_nc_class` | `str` | Clase NC del α global |
| `total_fma_count` | `int` | FMAs totales en representación polinomial |

**Ejemplo:**

```python
import torch.nn as nn
from poema.backends.gideon.engine import GideonEngine

engine = GideonEngine()
net = nn.Sequential(
    nn.Linear(32, 64), nn.Tanh(),
    nn.Linear(64, 32), nn.Tanh(),
    nn.Linear(32, 1),
)

report = engine.analyse_network(net, degree=6)
print(f"α global = {report.global_alpha:.4f}  ({report.global_nc_class})")
print(f"FMA totales = {report.total_fma_count}")
for i, inv in enumerate(report.layer_invariants):
    if inv is not None:
        print(f"  Capa {i}: α = {inv.alpha:.4f}")
```

---

### `analyse_training_trajectory(trajectory) → KoopmanNetworkResult`

Aplica el análisis de Koopman a una trayectoria temporal de pérdidas de entrenamiento.

**Parámetros:**

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `trajectory` | `np.ndarray` shape (T,) | Serie temporal de valores de pérdida |

**Retorna:** `KoopmanNetworkResult`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `koopman_eigenvalues` | `np.ndarray` | Valores propios del operador de Koopman |
| `spectral_diagnostics` | `dict` | Modo dominante, radio espectral, diagnóstico de convergencia |
| `trajectory_length` | `int` | Longitud T de la trayectoria analizada |

**Interpretación de eigenvalues:**

| Magnitud | Significado |
|----------|-------------|
| ≈ 1 | Modo quasi-periódico (oscilación de la pérdida) |
| < 1 | Modo convergente (el entrenamiento estabiliza) |
| > 1 | Modo divergente (inestabilidad localizada) |

**Ejemplo:**

```python
import numpy as np

losses = np.exp(-0.3 * np.arange(20)) + 0.02 * np.random.randn(20)
k_result = engine.analyse_training_trajectory(losses)
print("Eigenvalues de Koopman:", k_result.koopman_eigenvalues[:5])
print("Diagnóstico:", k_result.spectral_diagnostics)
```

---

### `evolve_network_function(network, domain, input_dim) → NeuralEvolutionResult`

Auto-evoluciona la función escalar representativa que implementa una red neuronal.

**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `network` | `nn.Module` | requerido | Red neuronal PyTorch |
| `domain` | `tuple` | `(-1.0, 1.0)` | Dominio de la función escalar |
| `input_dim` | `int` | `1` | Dimensión de entrada para construir **f**_net(x) |

**Retorna:** `NeuralEvolutionResult`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `initial_epsilon` | `float` | ε₀ con grado inicial |
| `final_epsilon` | `float` | ε_f tras auto-evolución |
| `improvement_ratio` | `float` | ε₀ / ε_f |
| `best_reduction` | `ReductionResult` | Mejor reducción encontrada |

**Ejemplo:**

```python
evo = engine.evolve_network_function(net, domain=(-2.0, 2.0), input_dim=32)
print(f"Compresibilidad: {evo.improvement_ratio:.2f}×")
print(f"ε final = {evo.final_epsilon:.2e}")
```

---

## Referencia: Meta-Compilador (v1.1.0)

### `meta_compile(f, domain, strategy, beta, target_epsilon, enable_auto_evolution, config) → MetaCompilerResult`

Busca la gramática ACF óptima para aproximar `f` en `domain`.

**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `f` | `callable` | requerido | Función a compilar f: float → float |
| `domain` | `tuple` | `(-1.0, 1.0)` | Intervalo [a, b] de evaluación |
| `strategy` | `str` | `"greedy"` | `"grid"` / `"random"` / `"greedy"` |
| `beta` | `float` | `1.0` | Temperatura del criterio de energía libre |
| `target_epsilon` | `float` | `1e-6` | ε objetivo que se busca alcanzar |
| `enable_auto_evolution` | `bool` | `False` | Aplicar ACFAutoEvolver sobre G* |
| `config` | `MetaCompilerConfig` | `None` | Configuración completa (anula parámetros individuales) |

**Retorna:** `MetaCompilerResult`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `best_grammar` | `Grammar` | `Grammar(basis, degree, n_observables, method)` |
| `best_reduction` | `ReductionResult` | Reducción ACF con la gramática óptima |
| `initial_epsilon` | `float` | ε baseline (Chebyshev grado 8) |
| `final_epsilon` | `float` | ε con G* (y auto-evolución si activada) |
| `improvement_ratio` | `float` | initial_epsilon / final_epsilon |
| `trace` | `MetaCompilerTrace` | `all_grammars`, `best`, `n_evaluated`, `n_failed` |

**Fórmula de energía libre:**

$$\mathcal{C}(G, f, \beta) = \varepsilon(G, f) - \frac{\log(1+d) + \log(1+k)}{\beta}$$

**Ejemplo completo:**

```python
import numpy as np
from poema.backends.gideon.engine import GideonEngine

engine = GideonEngine()

# Función periódica — Fourier debería ganar
f_periodic = lambda x: np.sin(3 * x) + 0.5 * np.cos(5 * x)

result = engine.meta_compile(
    f_periodic,
    domain=(-np.pi, np.pi),
    strategy="greedy",
    beta=1.0,
    target_epsilon=1e-8,
)
print(result.best_grammar)           # Grammar(basis=FOURIER, degree=...)
print(f"Mejora: {result.improvement_ratio:.1f}×")
print(f"Grammars evaluadas: {result.trace.n_evaluated}")
```

**Configuración manual del espacio de búsqueda:**

```python
from acf_functor import (
    ACFMetaCompiler, MetaCompilerConfig, GrammarSpace, BasisFamily,
    GridSearch, RandomSearch, GreedySearch,
)

space = GrammarSpace(
    families=[BasisFamily.CHEBYSHEV, BasisFamily.FOURIER, BasisFamily.RBF],
    degree_range=(4, 20),
    degree_step=2,
    n_observables_options=[8, 16],
)
config = MetaCompilerConfig(
    grammar_space=space,
    strategy=GreedySearch(n_restarts=5),
    beta=2.0,
    target_epsilon=1e-7,
    enable_auto_evolution=True,
    n_probe=200,
)
mc = ACFMetaCompiler(config)
result = mc.compile(f_periodic, domain=(-np.pi, np.pi))
```

**Analizar el espacio de gramáticas sin compilar:**

```python
trace = mc.analyse_grammar_space(f_periodic, domain=(-np.pi, np.pi))
print(f"Total evaluadas: {trace.n_evaluated}, fallidas: {trace.n_failed}")
best = trace.best
print(f"Mejor gramática: {best.grammar}, ε = {best.epsilon:.2e}, C = {best.free_energy:.4f}")
```
*Manual técnico Gideon v1.1.0 — Martínez's Invariant, Abril 2026*

---

## Tensor ACF — Guía de uso

### Reducción de funciones multivariable

```python
import math
from acf_functor import TensorACFReducer, StandardTensorFunctions

# Función de 3 variables
def my_func(x, y, z):
    return math.sin(x + y) * math.cos(z) + x * z

reducer = TensorACFReducer(
    default_degree=10,   # grado Chebyshev por dimensión
    max_rank=12,         # rango TT máximo
    target_epsilon=1e-6, # precisión objetivo
    method="tt",         # "tt" o "tucker"
)

result = reducer.reduce(
    my_func,
    domains=[(-3, 3), (-3, 3), (-2, 2)],
)

# Inspeccionar resultados
print(f"ε certificado: {result.epsilon:.2e}")
print(f"Rangos TT: {result.invariants.tt_ranks}")
print(f"α por modo: {result.invariants.alpha_per_mode}")
print(f"Dim efectiva: {result.invariants.effective_dimension:.2f}")
print(f"Compresión: {result.tt.compression_ratio:.1f}×")
```

### Evaluación batch

```python
import torch
# Evaluar en 1000 puntos
batch = torch.rand(1000, 3, dtype=torch.float64) * 6 - 3
values = result.evaluate(batch)
print(f"Shape: {values.shape}")  # (1000,)
```

### Tucker para dimensión baja

```python
reducer_tucker = TensorACFReducer(method="tucker", default_degree=12)
result_t = reducer_tucker.reduce(
    StandardTensorFunctions.gaussian_2d,
    domains=[(-3, 3), (-3, 3)],
)
```

### Funciones de test de 5 dimensiones

```python
result_5d = reducer.reduce(
    StandardTensorFunctions.friedman1,
    domains=[(0, 1)] * 5,
    degrees=[6, 6, 6, 6, 6],
)
print(f"5D: ε = {result_5d.epsilon:.2e}, FMAs = {result_5d.invariants.total_fma_count}")
```

---

## Matrix ACF — Guía de uso

### Funciones matriciales básicas

```python
import torch
from acf_functor import (
    MatrixExponential, MatrixSquareRoot, MatrixLogarithm,
    MatrixResolvent, MatrixSign, MatrixACFAnalyzer,
    ChebyshevMatrixReducer,
)

# Generar matriz SPD
A = torch.randn(20, 20, dtype=torch.float64)
A = A @ A.T + 0.5 * torch.eye(20, dtype=torch.float64)

# Exponencial
exp_r = MatrixExponential.reduce(A, t=0.5, degree=30, target_epsilon=1e-8)
print(f"exp(0.5·A): ε = {exp_r.epsilon:.2e}, grado = {exp_r.degree}")

# Raíz cuadrada
sqrt_r = MatrixSquareRoot.reduce(A, degree=30)
print(f"A^(1/2): ε = {sqrt_r.epsilon:.2e}")

# Logaritmo
log_r = MatrixLogarithm.reduce(A, degree=30)
print(f"log(A): ε = {log_r.epsilon:.2e}")

# Resolvente
inv_r = MatrixResolvent.reduce(A, sigma=1.0, degree=30)
print(f"(A+I)⁻¹: ε = {inv_r.epsilon:.2e}")
```

### Función matricial personalizada

```python
import math
result = ChebyshevMatrixReducer.reduce(
    lambda x: math.tanh(x),  # cualquier función escalar
    A,
    degree=25,
    target_epsilon=1e-6,
)
print(f"tanh(A): ε = {result.epsilon:.2e}")
```

### Análisis de invariantes

```python
inv = MatrixACFAnalyzer.analyse(A, func="exp", degree=40)
print(f"α matricial: {inv.matrix_alpha:.4f}")
print(f"Clase NC: {inv.nc_class}")
print(f"Grado efectivo: {inv.effective_degree}")
print(f"[λ_min, λ_max] = [{inv.spectral_range[0]:.4f}, {inv.spectral_range[1]:.4f}]")
print(f"Entropía Chebyshev: {inv.chebyshev_entropy:.4f}")
```

---

## Garantías Formales — Referencia Rápida (v1.2.0)

> Todos los teoremas en `MathTest/FormalEmpiricalTheorems.lean` — 17 teoremas, **0 sorry** (Lean 4.29.0-rc6 + Mathlib)  
> Tests en `tests/test_formal_empirical_bounds.py` — **35 tests pasando**

### Tabla maestra: afirmación empírica → teorema formal

| # | Afirmación anterior | Teorema Lean | Archivo |
|---|--------------------|----|---------|
| 1 | "λ₂ > 0.5 → admite filtros de menor grado (empírico)" | **FIEDLER-2** `fiedler_monotone_degree_reduction` | `FormalEmpiricalTheorems.lean` |
| 2 | "ratio λ₂=0.5 vs λ₂=1: ≈ 1.71 (observado)" | **FIEDLER-3**: ratio = log(3/2)/log(5/4) ≈ **1.817** (exacto) | idem |
| 3 | "fast/algebraic/slow son perfiles heurísticos" | **FAM-1/2/3/4**: clases de complejidad rigurosas con inclusión fast⊂algebraic | idem |
| 4 | "isomórfico al criterio AIC (analogía)" | **AIC-1**: C(G,f,β=n/2) = ε−2S/n (identidad exacta) | idem |
| 5 | "BIC penaliza más (comparativo empírico)" | **AIC-3**: β_BIC > β_AIC para n ≥ e² ≈ 7.4 (probado formalmente) | idem |
| 6 | "Consistencia Alpha: 10% tolerancia (empírico)" | **ALPHA-3**: k ≥ exp(10·\|log C\|/α) ⟹ error ≤ 0.1 (umbral explícito) | idem |
| 7 | "α empírico = α espectral = α operacional (aprox.)" | **ALPHA-4**: log(ε₁/ε₂)/log(d₂/d₁) = α (identidad exacta, tres estimadores) | idem |
| 8 | "ciclo Φ⇌Φ* converge (sin hipótesis)" | **ADJ-1**: convergencia ↔ L-Lipschitz con L<1 (Banach) + **ADJ-2**: contraejemplo | idem |

### Cotas numéricas clave (probadas, no estimadas)

| Cantidad | Valor formal | Teorema |
|----------|-------------|---------|
| Ratio de grado λ₂=0.5/1.0 | log(3/2)/log(5/4) = **1.8171...** | FIEDLER-3 |
| Umbral k para 10% error en α | k_min = exp(10·\|log C\|/α) | ALPHA-3 |
| Constante fast→algebraic | C' = C·ρ/(ρ−1) | FAM-3 |
| β_AIC vs β_BIC crossover | n ≥ e² ≈ 7.39 | AIC-3 |
| Tasa convergencia α̂_spec | \|α̂−α\| = \|log C\|/log k | ALPHA-2 |

### Cómo consultar una garantía desde Python

```python
# Verificar cota Fiedler
import math
def fiedler_degree_bound(eps: float, lam2: float, lam_max: float) -> int:
    """Teorema FIEDLER-1/2 — d* = ceil(log(2/eps)/log(1+lam2/lam_max))"""
    return math.ceil(math.log(2.0 / eps) / math.log(1.0 + lam2 / lam_max))

# Verificar umbral alpha
def alpha_threshold_k(C: float, alpha: float) -> float:
    """Teorema ALPHA-3 — para k >= k_min, |alpha_hat - alpha| <= 0.1"""
    return math.exp(10.0 * abs(math.log(C)) / alpha)

# Verificar isomorfismo AIC
def aic_from_cost(epsilon: float, S: float, n: int) -> float:
    """Teorema AIC-1 — C(G,f,beta=n/2) = epsilon - 2S/n"""
    return epsilon - 2.0 * S / n
```

*Manual técnico Gideon v1.2.0 — Martínez's Invariant, Abril 2026*

---

## Nuevas Garantías Formales v1.3.0 — Expansiones E3–E6 y Correcciones DEBILIDAD #1–#6

### Módulos y teoremas nuevos

| Módulo | Teoremas | Descripción |
|--------|----------|-------------|
| `domain_admissibility.py` | AD-1…AD-6 | 6 condiciones constructivas de admisibilidad C^ω |
| `nyquist_acf.py` | NYC-1…NYC-5 | Teorema Nyquist-ACF: d*(ε,α) = ⌈(C_f/ε)^{1/α}⌉ |
| `koopman_observability.py` | KO-1a/b/c, KO-3, KO-4 | Observabilidad EDMD + convergencia Korda-Mezić |
| `koopman_observability.py` | HW-1, HW-2 | Invariante E(f)=E(Φ(f)) en fp64/fp32/fp16 |
| `differentiable_acf.py` | DA-1, DA-2, DA-3 | Cadena de gradientes a través del compilador |
| `pde_acf.py` | PDE-1, PDE-2, PDE-3, PDE-4 | Reducción Galerkin-ACF de PDEs |
| `genesis_lean_bridge.py` | anti-tautología | Guardián is_tautological() + KD-2/THERMO-1/INFGEO-1 |
| `riemannian_meta_compiler.py` | RMC-1, RMC-2, RMC-3 | Meta-compilador con gradiente natural Fisher |
| `adjunction.py` | ADJ-triangle-L, ADJ-triangle-R | Identidades triangulares Φ*⊣Φ operacionalizadas |

### Cotas nuevas (probadas, no estimadas)

| Cantidad | Fórmula | Módulo |
|----------|---------|--------|
| Grado mínimo Chebyshev | $d^*(\varepsilon,\alpha) = \lceil(C_f/\varepsilon)^{1/\alpha}\rceil$ | `NyquistACFTheorem` |
| Error Galerkin espectral | $\|\Pi_d u - u\|_{L^2} \leq C\rho^{-d}\|u\|_{H^{d+1}}$ | `PDEACFSolver` |
| FMAs por paso PDE | $O(d^2)$ para coeficiente constante, $O(d^3)$ variable | `PDEACFSolver` |
| Convergencia EDMD | $\|K-K_d\|_F \leq C_K/\sqrt{N}$ (Korda-Mezić) | `KoopmanObservabilityChecker` |
| Composición delta | $\delta(f\circ g) \leq \delta_f + L_f\cdot\delta_g$ | `KoopmanObservabilityChecker` |
| Error triangular adjunto | $\|\Phi(\Phi^*(\Phi(f))) - \Phi(f)\|_\infty < 10^{-5}$ | `AdjunctionVerifier` |
| Pérdida precision fp32 | bits\_loss $\leq \log_2(\varepsilon_\text{fp32}/\varepsilon_\text{fp64}) \leq 24$ | `EnergyInvariantHardwareVerifier` |

### Cómo consultar una garantía desde Python

```python
import math
import numpy as np

# ── Dominio constructivo ──────────────────────────────────────
from acf_functor.domain_admissibility import DomainAdmissibilityChecker
cert = DomainAdmissibilityChecker().check(math.sin, (-1.0, 1.0))
print(f"Admisible: {cert.admissible}, ρ_f={cert.estimated_bernstein_rho:.4f}")

# ── Grado mínimo Nyquist-ACF ──────────────────────────────────
from acf_functor.nyquist_acf import NyquistACFTheorem
result = NyquistACFTheorem().apply(math.sin, (-math.pi, math.pi), epsilon=1e-8)
print(f"d* = {result.d_star_empirical}, clase: {result.complexity_class}")

# ── Observabilidad Koopman ────────────────────────────────────
from acf_functor.koopman_observability import KoopmanObservabilityChecker
obs = KoopmanObservabilityChecker(d=20, N=2000).check(math.tanh, (-1.5, 1.5))
print(f"KO-1a={obs.ko1a_passed}, ergódico={obs.ko1c_passed}")

# ── Hardware energy invariant ────────────────────────────────
from acf_functor.koopman_observability import EnergyInvariantHardwareVerifier
hw = EnergyInvariantHardwareVerifier().verify(lambda x: x**3 - x, (-1.0, 1.0))
print(f"fp64 d*={hw['fp64']['energy_d_star']}, fp32 d*={hw['fp32']['energy_d_star']}")

# ── ACF diferenciable (PyTorch) ──────────────────────────────
import torch
from acf_functor.differentiable_acf import DifferentiableACFLayer
layer = DifferentiableACFLayer(degree=16, domain=(-1.0, 1.0))
x = torch.linspace(-1, 1, 100, requires_grad=True)
(layer(x) - torch.sin(x)).pow(2).mean().backward()

# ── PDE solver ───────────────────────────────────────────────
from acf_functor.pde_acf import PDEACFSolver, PDEConfig, PDEType
cfg = PDEConfig(pde_type=PDEType.HEAT, n_modes=8, t_end=0.2, dt=1e-5, nu=0.01)
solver = PDEACFSolver(cfg)
rep = solver.solve(np.sin(np.pi * solver.x_grid))
print(f"α(T)={rep.alpha_spectral:.4f}, FMAs={rep.total_fma_count:,}")

# ── Anti-tautología Lean ─────────────────────────────────────
from acf_functor.genesis_lean_bridge import is_tautological
print(is_tautological("exact h_bound"))        # True  — rechazado
print(is_tautological("linarith [h1, h2]"))   # False — aceptado

# ── Meta-compilador Riemanniano ──────────────────────────────
from acf_functor.riemannian_meta_compiler import RiemannianMetaCompiler
rmc = RiemannianMetaCompiler(target_epsilon=1e-4, max_iter=15)
r = rmc.compile(math.sin, domain=(-1.0, 1.0))
print(f"Base: {r.best_basis}, d: {r.best_degree}, ε: {r.best_epsilon:.2e}")

# ── Identidades triangulares ─────────────────────────────────
from acf_functor.adjunction import AdjunctionVerifier
f = lambda x: torch.tensor([math.sin(xi.item()) for xi in x])
tri = AdjunctionVerifier(tolerance=1e-3).verify_triangle_identities(
    f, domain=(-1.0, 1.0), degree=25
)
print(f"Error izquierdo: {tri.left_triangle_error:.2e}")
print(f"Adjunción: {tri.adjunction_holds}")

# ── α normalizado ────────────────────────────────────────────
from acf_functor.invariant_unified import ACFInvariantUnified
a = ACFInvariantUnified().compute(math.sin, (-1.0,1.0), skip_geometric=True)
print(f"α={a.best_estimate:.3f}, ᾱ={a.normalized_alpha:.3f}")

# ── Benchmarks comprensivos ──────────────────────────────────
from benchmarks.benchmark_comprehensive import run_full_benchmark
report = run_full_benchmark(verbose=False)
print(f"Pass rate: {report.pass_rate*100:.1f}%")  # ≥ 70%
```

### Estado global de correcciones

| # | Debilidad 2025 | Resuelta | Módulo | Test |
|---|----------------|---------|--------|------|
| 1 | E(f)=E(Φ(f)) solo en ℝ exacto | ✅ | `koopman_observability.py` | `test_hardware_invariant_polynomial` |
| 2 | Pruebas Lean tautológicas | ✅ | `genesis_lean_bridge.py` + Lean | `test_tautology_rejection` |
| 3 | Dominio C^ω no constructivo | ✅ | `domain_admissibility.py` | `test_sin_is_admissible` |
| 4 | Solo 4 benchmarks, con errores | ✅ | `benchmark_comprehensive.py` | `test_benchmark_suite_runs` |
| 5 | Adjunción no demostrada | ✅ | `adjunction.py` | `test_left_triangle_sin` |
| 6 | α fuera de [0,1] sin aviso | ✅ | `invariant_unified.py` | `test_alpha_normalization` |

**Suite de tests:** `tests/test_all_expansions.py` — **43/43 passing** (Abril 2026)

*Manual técnico Gideon v1.3.0 — Martínez's Invariant, Abril 2026*

---

## Nuevos Módulos v1.4.0 — Dominios Algebraicos (Mayo 2026)

### Introducción

La versión 1.4.0 añade cinco nuevos módulos algebraicos que extienden el invariante de Martínez a dominios de álgebra abstracta, geometría p-ádica, teoría de categorías, teoría de números y finanzas cuantitativas.

### Guía Rápida de los 5 Módulos

#### 1. `algebraic_acf.py` — Anillos, Cuerpos, Lie, ECC, Boole

**Casos de uso principales:**
- Evaluar complejidad FMA de polinomios sobre anillos arbitrarios
- Sintetizar circuitos booleanos mínimos en forma ANF
- Analizar álgebras de Lie (su2, so3, Heisenberg)
- Operaciones en curvas elípticas $E/\mathbb{F}_p$

**Gotchas y errores frecuentes:**

| Error | Causa | Fix |
|-------|-------|-----|
| `LieAlgebraFactory.su2()` devuelve ndarray | Devuelve constantes de estructura shape `(3,3,3)` | Pasar a `LieAlgebraACFAnalyzer().analyze(f, algebra_name="su2")` |
| `scalar_mul(P, k)` no funciona | El orden correcto es `scalar_mul(k, P)` — k **primero** | `ecc.scalar_mul(5, P)` |
| `generator_list` undefined | El kwarg correcto es `generators` | `reduce_wrt_ideal(f, generators=[...])` |

#### 2. `topos_acf.py` — Sitio de Grothendieck / Haces

**Casos de uso:**
- Verificar la condición de pegamiento (sheaf gluing) sobre particiones del dominio
- Diagnosticar admisibilidad de cubrimientos

**Gotchas:**
- `generate_covering(f, domain)` — la función `f` es el **primer argumento** requerido
- `report.admissibility_Omega` es un `dict`, no un `bool`
- `report.geometric_sequents_valid` es una lista de strings, no `.certificates`
- `cov.overlaps()` es un **método** — llamarlo con `()`, no como propiedad

#### 3. `padic_acf.py` — Números p-ádicos

**Casos de uso:**
- Expansiones p-ádicas de enteros
- Levantamiento de Hensel para raíces modulares
- Series de Mahler sobre $\mathbb{Z}_p$

**Gotchas:**
- `ramanujan_tau_coefficients(n)` devuelve lista 1-indexada: `tau[0]=0`, `tau[1]=τ(1)=1`, `tau[2]=τ(2)=-24`
- Los certificados son `["PADIC-2 (Mahler)"]`, no bare `"PADIC-2"` → usar `any("PADIC-2" in c for c in report.certificates)`

#### 4. `modular_acf.py` — Formas Modulares

**Casos de uso:**
- Coeficientes de Eisenstein $E_4$, $E_6$ y función $\Delta$ de Ramanujan
- Reducción Horner de series-$q$

**Gotchas:**
- `eisenstein_e4_coefficients(n)` devuelve `[1, 240, 2160, ...]` (a₀=1)
- `ModularACFReducer().reduce(name, coeffs, weight)` — tres argumentos posicionales

#### 5. `finance_acf.py` — Finanzas y Caos

**Casos de uso:**
- Estimar exponente de Hurst de series temporales
- Reducir superficies de volatilidad implícita a series de Chebyshev
- VaR/CVaR por expansión en Caos Polinomial (Gauss-Hermite)
- Detección de cambios de régimen (rolling α)
- Densidad invariante de mapas caóticos

**Gotchas críticos:**

| Error | Causa | Fix |
|-------|-------|-----|
| Hurst H incorrecto | Pasar `cumsum(noise)` (precio) en vez de `noise` (retornos) | Pasar incrementos/retornos directamente |
| `sigma_function=` undefined | El kwarg es `sigma_fn=` | `reducer.reduce(sigma_fn=f, ...)` |
| `TypeError: max() arg` | `compute_risk_via_pce` pasa `np.array([xi])` al payoff | Usar `xi = float(np.asarray(x).flat[0])` en payoff |
| `VolatilitySurfaceReport` sin `fma_count` | El campo correcto es `total_fma_count` | `report.total_fma_count` |
| Régimen no detectado con threshold=0.3 | Umbral alto + suavizado por ventana | Usar `alpha_jump_threshold=0.05` o señales con ACF contrastante |

#### Análisis de ruido fp32 en `detect_regime_changes`

El umbral `alpha_jump_threshold=0.05` es fino. La pregunta legítima es: ¿puede el ruido de precisión de punto flotante acumulado en las operaciones FMA de la ventana rolling disparar falsos positivos?

**Análisis cuantitativo:**

El campo `alpha` de cada ventana es la pendiente de `polyfit(k, log|ρ(k)|, 1)` sobre `k = 1..max_lag`. Con `max_lag = 20` y ventana de `window_size = 50`:

- Cada `ρ(k)` involucra ~50 productos + 50 sumas → $\varepsilon_{\mathrm{fp32}}^{\mathrm{acc}} \approx 50 \times 3.05 \times 10^{-5} \approx 1.5 \times 10^{-3}$ sobre `ρ(k)`.
- El `log|ρ(k)|` amplifica el error cerca de cero: si `|ρ(k)| ≈ ε` el log diverge.
- La pendiente del `polyfit` sobre 20 puntos tiene error de escala $O(\varepsilon / \Delta k) = O(0.0015)$ para diferencias de índice unitarias.

**Conclusión numérica:** el ruido de precisión fp32 en la alpha rolling es $\approx 0.001$–$0.003$, un orden de magnitud por debajo del umbral 0.05. **No hay falsos positivos sistemáticos por fp32 con este umbral** en condiciones normales.

**Sin embargo**, hay dos casos patológicos a vigilar:

| Escenario | Síntoma | Mitigación |
|-----------|---------|-----------|
| Serie con `|ρ(k)| < 1e-6` para varios `k` (ruido puro de alta frecuencia) | `log(1e-12 + noise)` → alpha volátil | Usar `fp64` (`detect_regime_changes` opera en float64 internamente si `np.float64`) |
| Ventana con casi toda la señal constante (varianza ≈ 0) | División por `var + 1e-15` → alpha ≈ 0 artificialmente → salto al siguiente cambio | Filtrar series con `np.std(sub) < 1e-8` antes de pasar a `detect_regime_changes` |

**Tabla de thresholds seguros por precisión:**

| Precisión interna | Ruido estimado en α | Threshold mínimo seguro | Recomendado |
|-------------------|-------------------|------------------------|-------------|
| `fp32` (numpy default) | ~0.003 | 0.015 | **0.05** |
| `fp64` (numpy float64) | ~1e-11 | 1e-9 | **0.05** (sin cambio) |

El umbral `0.05` tiene margen 16× sobre el ruido fp32. Es seguro. Si se necesita detectar cambios más sutiles (`threshold < 0.015`), forzar `fp64` explícitamente:

```python
import numpy as np
from acf_functor.finance_acf import detect_regime_changes

# Forzar fp64 para máxima sensibilidad (threshold muy fino)
series_f64 = series.astype(np.float64)
report = detect_regime_changes(
    series_f64,
    window_size=50,
    alpha_jump_threshold=0.01,   # seguro solo en fp64
)
```

### Tests asociados

```bash
# Solo los nuevos tests algebraicos (105 tests)
python3 -m pytest tests/test_algebraic_extensions.py -v

# Suite completa
python3 -m pytest tests/ -q
# Resultado esperado: 1617+ passed, 5 failed (pre-existentes)
```

### Certificados Lean 4

El archivo `MathTest/AlgebraicACFCertificates.lean` (~380 líneas) contiene las formalizaciones:
- `algacf_horner_fma_count_le_degree`
- `gf2_fermat`, `gfp_fermat`, `gfp_is_field`
- `jacobi_identity`, `lie_bracket_antisymmetry`
- `padic_hensel_correction_zero_mod_p`
- `mahler_finite_difference_formula`
- `topos_sheaf_gluing_error`
- `fin5_logistic_maps_unit_interval`, `fin1_hurst_positive`

---

## Referencia: GideonAgentRouter

El `GideonAgentRouter` es el subsistema de Gideon que orquesta los agentes TAA y ERGON del ecosistema ACF.

### Importación

```python
from poema.backends.gideon import GideonAgentRouter, AgentRouterConfig, AgentRouteResult
# O directamente:
from poema.taa_agent import TAAAgent, TAAReport
from poema.ergon import ERGONAgent, ERGONReport
```

### AgentRouterConfig

```python
@dataclass
class AgentRouterConfig:
    ergon_complexity_threshold: float = 0.1
    # 𝔈(T) < threshold → sistema integrable → TAA solo
    # 𝔈(T) ≥ threshold → caos detectado → activar ERGON

    chaos_lambda_threshold: float = 0.05
    # λ_max > threshold → caos genuino → TAA necesita μ_SRB de ERGON

    ergon_iterations: int = 50_000
    # Iteraciones de Birkhoff para convergencia de Ψ_ER a μ_SRB

    taa_edmd_delay: int = 1
    # Retardo temporal para EDMD en TAA

    epsilon: float = 1e-4
    # Tolerancia objetivo compartida por TAA y ERGON
```

### AgentRouteResult

```python
@dataclass
class AgentRouteResult:
    agent_used: str          # 'taa', 'ergon', 'joint'
    taa_report: TAAReport    # Siempre presente
    ergon_report: Optional[ERGONReport]  # None si 𝔈(T) ≈ 0
    measure_source: str      # 'empirical' o 'srb'
    pesin_verified: bool     # True si ERGON verificó Pesin
    ergodic_complexity: float  # 𝔈(T) ∈ [0, 1]
    lambda_max: float        # Exponente de Lyapunov máximo
    routing_reason: str      # Explicación textual de la decisión
```

### GideonAgentRouter.route()

```python
def route(
    self,
    T: Callable,           # Mapa dinámico T: ℝ^d → ℝ^d
    x0: np.ndarray,        # Condición inicial para ERGON
    x_data: np.ndarray,    # Datos de trayectoria para TAA (shape: n_steps × dim)
) -> AgentRouteResult
```

**Decisión de enrutamiento:**

```
𝔈(T) < 0.1 (integrable)
    → agent_used = 'taa'
    → taa_report con measure_used = 'empirical'
    → ergon_report = None

0.1 ≤ 𝔈(T) < 0.9 (mixto)
    → agent_used = 'joint'
    → ERGON corre primero → μ_SRB → TAA usa L²(𝒳, μ_SRB)
    → taa_report con measure_used = 'srb'
    → ergon_report con μ_SRB, h_KS, Pesin

𝔈(T) ≥ 0.9 (caos puro)
    → agent_used = 'joint' (con bandera ergon_dominant=True)
    → ERGON es el agente primario
    → TAA da d* recomendado pero ERGON da el certificado principal
```

### Ejemplo completo: sistema de Lorenz

```python
import numpy as np
from poema.taa_agent import TAAAgent
from poema.ergon import ERGONAgent

# Sistema dinámico: Lorenz
def lorenz_map(x, sigma=10.0, rho=28.0, beta=8/3, dt=0.005):
    dx = np.array([
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2]
    ])
    return x + dt * dx

x0 = np.array([1.0, 1.0, 1.0])

# Generar datos de trayectoria
n_steps = 5000
x_data = np.zeros((n_steps, 3))
x = x0.copy()
for k in range(n_steps):
    x_data[k] = x
    x = lorenz_map(x)

# ── Pipeline independiente TAA ─────────────────────────────────────────────
taa = TAAAgent()
taa_report = taa.analyze(lorenz_map, x_data, epsilon=1e-3)
print(f"[TAA] α_A: {taa_report.alpha_class.name}")
print(f"[TAA] d*:  {taa_report.d_star}")
print(f"[TAA] δ:   {taa_report.delta_d:.3e}  (puede estar inflado)")
print(f"[TAA] ERGON needed: {taa_report.ergon_required}")

# ── Pipeline independiente ERGON ───────────────────────────────────────────
ergon = ERGONAgent(n_iterations=50_000)
ergon_report = ergon.analyze(lorenz_map, x0, epsilon=1e-3)
print(f"[ERGON] h_KS:     {ergon_report.h_ks:.4f}")
print(f"[ERGON] λ_max:    {ergon_report.lambda_max:.4f}")
print(f"[ERGON] 𝔈(T):    {ergon_report.ergodic_complexity:.4f}")
print(f"[ERGON] Pesin ok: {ergon_report.pesin_verified}")
print(f"[ERGON] d* para TAA: {ergon_report.recommended_d_star}")

# ── Pipeline conjunto: ERGON → TAA (μ_SRB → δ mínimo) ─────────────────────
ergon_report2, taa_report2 = ergon.joint_analyze(
    T=lorenz_map, x0=x0, x_data=x_data, epsilon=1e-3
)
print(f"[JOINT] medida TAA: {taa_report2.measure_used}")  # 'srb'
print(f"[JOINT] δ óptimo:   {taa_report2.delta_d:.3e}")   # < TAA solo
print(f"[JOINT] inflación:  {taa_report2.measure_inflation}")  # 0.0
```

### Ejemplo: sistema integrable (TAA solo)

```python
import numpy as np
from poema.taa_agent import TAAAgent
from poema.ergon import ERGONAgent

# Sistema estable: x_{t+1} = 0.9 * x_t  (eigenvalor 0.9 < 1)
T_stable = lambda x: 0.9 * x

x0 = np.array([1.0])
x_data = np.zeros((500, 1))
x = x0.copy()
for k in range(500):
    x_data[k] = x
    x = T_stable(x)

ergon = ERGONAgent(n_iterations=10_000)
report = ergon.analyze(T_stable, x0)

print(f"𝔈(T) = {report.ergodic_complexity:.4f}")   # ≈ 0.0
print(f"handoff_to_taa = {report.handoff_to_taa}") # True
# → TAA actúa solo, no se necesita ERGON
```

### Certificados Lean 4 del router

El router implementa directamente:
- **ERG-9** (`ergon_valid_without_taa`, `taa_valid_without_ergon`): cada agente es válido independientemente
- **TAA-5b** (`taa_ergon_interface_correct`): cuando ERGON provee μ_SRB, δ_μ = 0
- **ERG-7b** (`taa_ergon_interface_correct`): la interfaz es formalmente correcta
- **ERG-6b** (`ergodic_complexity_bounded`): la decisión de routing usa 𝔈(T) ∈ [0, 1]

```lean
-- La decisión de routing está certificada por:
-- ERGONCertificates.lean ERG-9b: taa_valid_without_ergon
-- ERGONCertificates.lean ERG-7b: taa_ergon_interface_correct
-- TAAAgentCertificates.lean TAA-5b: taa_ergon_interface_eliminates_inflation
```

*Gideon-guide v1.4.0 — Martínez's Invariant, Mayo 2026*
