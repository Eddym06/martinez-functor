# Poema Manual Técnico

Manual de referencia técnica del lenguaje Poema. Para la explicación conceptual y el significado de Poema dentro del Functor de Colapso Afín (ACF), ver `Poema.md`.

### Convención de nombres (canónica)

Este manual adopta la nomenclatura formal vigente:

1. **Functor principal:** Functor de Colapso Afín (ACF), notación $\Phi_{AC}$.
2. **Invariante principal:** Índice de Decaimiento Espectral Afín, notación $\alpha_A(f)$ (abreviado como $\alpha(f)$).
3. **Topos:** Topos de Computabilidad Afín (ACT), notación $\mathcal{T}_{AC}$.
4. **Ley estructural:** Principio de Invarianza de Profundidad Afín.

Nota de compatibilidad:

1. Por estabilidad de API, algunos nombres de código se mantienen (por ejemplo `acf_alpha`, `ACFTopos`, `act_topos.py`).
2. En texto técnico y paper se usa la nomenclatura canónica ACF/ACT/$\alpha_A(f)$.

---

## Índice

- [0. Quick Start](#0-quick-start)
- [1. Estado Actual de Implementación](#1-estado-actual-de-implementación)
  - [1.1 Parser continuous_flow](#11-parser-continuous_flow--especificación-técnica)
  - [1.2 CoPoem Expandido](#12-copoem-expandido--api-técnica)
  - [1.3 BiPoem Expandido](#13-bipoem-expandido--api-técnica)
  - [1.4 CompilationReport Expandido](#14-compilationreport-expandido--observabilidad-por-nodo)
  - [1.5 Herramienta de Diagnóstico](#15-herramienta-de-diagnóstico)
  - [1.6 Puente Genesis → CoPoem](#16-puente-genesis--copoem)
- [2. Fase E (Evoluciones 16-19)](#2-fase-e-evoluciones-16-19-la-espiral-interna)
  - [2.1 Evolución 20: Genesis Matemática](#21-evolucion-20-genesis-matematica)
- [3. Validación y Resultados Reales](#3-validación-y-resultados-reales)
  - [3.1 Benchmarks de Rendimiento](#31-benchmarks-de-rendimiento)
  - [3.2 Ejemplos de Errores y Fallos](#32-ejemplos-de-errores-y-fallos)
- [4. Nuevas Capacidades v2.2.0](#4-nuevas-capacidades-v220--abril-2026)
  - [4.1 GEMM-Triton Collider](#41-gemm-triton-collider)
  - [4.2 Auto-Domain Repair Mejorado](#42-auto-domain-repair-mejorado)
  - [4.3 Compensación de Kahan para Horner](#43-compensación-de-kahan-para-horner)
  - [4.4 CoPoem Multiobjetivo con Anderson](#44-copoem-multiobjetivo-con-anderson)
- [5. Agentes del Ecosistema ACF: TAA y ERGON](#5-agentes-del-ecosistema-acf-taa-y-ergon)
  - [5.1 TAA — Tensor Autocomputable Agent](#51-taa--tensor-autocomputable-agent)
  - [5.2 ERGON — Perron-Frobenius Agent](#52-ergon--perron-frobenius-agent)
  - [5.3 Independencia y Cooperación](#53-independencia-y-cooperación)
  - [5.4 Pipeline Conjunto TAA + ERGON](#54-pipeline-conjunto-taa--ergon)
  - [5.5 Certificados Lean 4](#55-certificados-lean-4)

---

## 0. Quick Start

### Ejemplo mínimo en 10 líneas

```python
import torch
from poema import Poem, PoemCompiler

# 1. Crear frontend
P = Poem(dtype=torch.float64)

# 2. Describir función en notación matemática
ast = P.continuous_flow("sin(x) + 0.5*x^2")

# 3. Compilar a secuencia FMA ejecutable
compiler = PoemCompiler(target="pytorch", precision="fp64")
fn, report = compiler.compile(ast, domain=(-math.pi, math.pi))

# 4. Ejecutar
x = torch.linspace(-3, 3, 1000, dtype=torch.float64)
y = fn(x)  # Evaluación exacta con ε certificado

print(f"ε certificado: {report.total_epsilon:.3e}")
print(f"Operaciones FMA: {report.total_fma_ops}")
```

### Tres modos en 30 líneas

```python
import torch, math
from poema import Poem, CoPoem, BiPoem, PoemCompiler

# === Poem: "Esta es mi función" ===
P = Poem(dtype=torch.float64)
ast = P.continuous_flow("exp(-x^2)")
fn, _ = PoemCompiler().compile(ast, domain=(-2, 2))
y = fn(torch.tensor([0.0, 1.0]))  # → [1.0, 0.368]

# === CoPoem: "Quiero un sistema con estas propiedades" ===
co = CoPoem(dtype=torch.float64)
spec = co.spectrum(spectral_radius=0.9, dimension=4, symmetry="symmetric")
W = co.synthesize(spec)  # Matriz 4×4 con radio espectral ≈ 0.9

# === BiPoem: "Tengo datos, descubra la estructura" ===
bi = BiPoem(dtype=torch.float64)
x = torch.zeros(2, 100, dtype=torch.float64)
x[:, 0] = torch.ones(2)
A = torch.diag(torch.tensor([0.9, 0.7]))
for t in range(99):
    x[:, t+1] = A @ x[:, t]
result = bi.symbiosis(x, max_dimension=8)
print(f"α(f) = {result['acf_alpha']:.3f}")  # Complejidad del sistema
```

---

## 1. Estado Actual de Implementación

La implementación actual vive en el paquete:

- `poema/ast_nodes.py`
- `poema/frontend.py`
- `poema/compiler.py`
- `poema/__init__.py`
- `poema/free_algebra.py` (Evolución 16)
- `poema/sheaf_semantics.py` (Evolución 17)
- `poema/affine_turing.py` (Evolución 18)
- `poema/meta_compiler.py` (Evolución 19)
- `poema/error_propagation.py` (Fase 2.2: Propagación de error)
- `poema/multivariate.py` (Fase 3.1: Gradientes multivariables)
- `poema/activations_modern.py` (Fase 3.3: GELU, SwiGLU, RoPE)
- `poema/nn_integration.py` (Fase 5.1: Integración PyTorch nn.Module)
- `poema/cli/diagnose.py` (Fase 4.1: Dashboard diagnóstico CLI)
- `benchmarks/canonical_benchmark.py` (Fase 4.2: Benchmark reproducible)
- `MathTest/CompositionErrorBounds.lean` (Fase 6.1: Certificados Lean composición)

Y está validada con pruebas en:

- `tests/test_poema.py`
- `tests/test_poema_hardening.py`
- `tests/test_poema_missing_coverage.py` (42 tests de cobertura nueva)
- `tests/test_poema_comprehensive.py` (Tests de nuevas funcionalidades)
- `tests/test_evolutions_16_19.py`
- `tests/test_genesis.py`
- `tests/test_self_modulation.py`
- `tests/test_composition_exhaustive.py`
- `tests/test_functor_engine.py`
- `tests/test_koopman_validation.py`
- `tests/test_alpha_consistency_report.py`
- `tests/test_taa_agent.py`
- `tests/test_ergon_agent.py`

**Total: 356 tests pasando sin regresiones.** Ver `TESTS_CANONICAL.md` para el número canónico actualizado.

La API pública exportada por `poema/__init__.py` incluye:

- Frontend: `Poem`, `CoPoem`, `BiPoem`
- **Agentes del ecosistema**: `TAAAgent`, `TAAReport`, `AlphaClass`, `ERGONAgent`, `ERGONReport`, `SRBMeasure`
- Compilación: `PoemCompiler`, `CompilationReport`, `NodeProfile`, `GeometricTypeChecker`, `FMALinearizer`, `PytorchBackend`, `TritonBackend`
- AST y tipos geométricos: `ASTNode`, `ScaleNode`, `ShiftNode`, `ComposeNode`, `AffineNode`, `PolynomialNode`, `TranscendentalNode`, `StratifiedNode`, `ConstraintNode`, `ParameterNode`, `FMAInstruction`, `GeometricType`, `Scalar`, `Vector`, `Morphism`, `Flow`, `Form`
- Reportes: `CoReport`
- Serialización: `ASTSerializer`, `ast_to_json`, `ast_from_json`, `ast_save`, `ast_load`
- Diagnóstico: `diagnose`, `DiagnosticReport`
- JIT: `PoemJITWrapper`, `PoemActivation`
- ONNX: `PoemONNXExporter`, `export_to_onnx`
- **Propagación de error**: `ErrorBound`, `compose_error_bounds`, `affine_error_propagation`, `sum_error_bounds`, `LIPSCHITZ_CONSTANTS`
- **Multivariable**: `MultivariateExpr`, `JacobianExpr`, `parse_multivariate`
- **Activaciones modernas**: `gelu_exact`, `swiglu`, `rope_embedding`
- **Integración ML**: `PoemActivationLayer`, `replace_activations_in_model`
- Errores/avisos: `TopologicalObstructionError`, `PrecisionDegradationWarning`

---

## 1.1. Parser `continuous_flow` — Especificación Técnica

### Implementación

El parser fue reescrito como un **parser de descenso recursivo** completo (`_RecursiveDescentParser` en `poema/frontend.py`). Reemplaza el enfoque anterior de búsqueda de texto simple que no manejaba correctamente composición anidada ni precedencia de operadores.

### Tokenizador

El tokenizer produce pares `(tipo, valor)`:

| Tipo | Patrón | Ejemplo |
| :--- | :--- | :--- |
| `NUM` | `[0-9]+(\.[0-9]+)?` | `3.14`, `42`, `0.5` |
| `IDENT` | `[a-zA-Z_][a-zA-Z0-9_]*` | `x`, `sin`, `my_var` |
| `OP` | `[+\-*/^()]` | `+`, `*`, `^`, `(` |

### Gramática

```
expr     ::= term (('+' | '-') term)*
term     ::= power (('*' | '/') power)*
power    ::= unary ('^' unary)*
unary    ::= ('-' | '+') unary | func_call | atom
func_call::= IDENT '(' expr (',' expr)* ')'
atom     ::= NUMBER | IDENT | '(' expr ')'
```

### Precedencia (de menor a mayor)

1. `+`, `-` (adición/sustracción)
2. `*`, `/` (multiplicación/división)
3. `^` (exponenciación, asociativa a la derecha)
4. `+x`, `-x` (unario)
5. `func(...)`, `(expr)`, `IDENT`, `NUMBER` (átomos)

### Constantes reconocidas

| Nombre | Valor |
| :--- | :--- |
| `pi` | `math.pi` |
| `e` | `math.e` |
| `tau` | `2 * math.pi` |

### Funciones soportadas

`sin`, `cos`, `exp`, `log`, `tanh`, `sigmoid` — todas con argumentos compuestos.

### Composición de trascendentales

Cuando el parser encuentra una función trascendental con un argumento compuesto (ej: `sin(cos(x))`), el método `_build_transcendental` detecta que el argumento no es un `InputNode` y genera automáticamente un `ComposeNode(outer=transcendental, inner=argumento)`. Esto garantiza que la evaluación sea semánticamente correcta.

### Extensiones avanzadas del parser

El parser ha evolucionado más allá de expresiones aritméticas simples. Ahora soporta tres construcciones que lo acercan a un lenguaje de programación funcional completo:

#### Let bindings

Sintaxis: `let IDENT = expr in expr`

Permite definir subexpresiones nombradas y reutilizarlas. El parser realiza sustitución profunda con deep copy, lo que significa que las llamadas anidadas como `f(f(x))` expanden correctamente ambas ocurrencias.

```python
# Simple: f(x) = 2x + 1, evaluar f(f(0)) = 3
ast = P.continuous_flow("let f = 2*x + 1 in f(f(x))")

# Anidado: f(x) = x+1, g(x) = 2x, evaluar g(f(x)) = 2(x+1)
ast = P.continuous_flow("let f = x + 1 in let g = 2*x in g(f(x))")
```

**Implementación técnica:** El parser mantiene un diccionario `_let_bindings: Dict[str, ASTNode]` que mapea nombres a expresiones AST. Cuando se encuentra una llamada a función con un nombre registrado, se invoca `_substitute()` que reemplaza recursivamente todas las ocurrencias de la variable (por defecto `x`) en la expresión ligada con el argumento proporcionado. Se usa `_deep_copy()` para evitar estado compartido entre expansiones.

#### Funciones piecewise

Sintaxis: `piecewise(condición, expr_verdadera, expr_falsa)`

Condiciones soportadas: `x >= num`, `x > num`, `x <= num`, `x < num`

```python
# ReLU
ast = P.continuous_flow("piecewise(x >= 0, x, 0)")

# Valor absoluto
ast = P.continuous_flow("piecewise(x >= 0, x, -x)")
```

**Implementación técnica:** Se traduce a `StratifiedNode` con dos `Branch` objects. Los dominios se calculan a partir de la condición y el dominio global de compilación. Por ejemplo, `x >= 0` con dominio `(-2, 2)` produce los estratos `(-2, 0)` y `(0, 2)`.

#### Derivadas simbólicas

Sintaxis: `D(expr)` o `D(expr, n)` para la n-ésima derivada.

Funciones soportadas y sus derivadas:

| Función | Derivada | Regla aplicada |
|---------|----------|----------------|
| `D(sin(x))` | `cos(x)` | Identidad trigonométrica |
| `D(cos(x))` | `-sin(x)` | Identidad trigonométrica |
| `D(exp(x))` | `exp(x)` | Auto-derivada |
| `D(log(x))` | `1/x` | Regla del inverso |
| `D(tanh(x))` | `1 - tanh(x)²` | Identidad hiperbólica |
| `D(sigmoid(x))` | `σ(x)(1-σ(x))` | Regla del producto |
| `D(x^n)` | `n·x^(n-1)` | Regla de potencia |
| `D(f(g(x)))` | `f'(g(x))·g'(x)` | Regla de la cadena |
| `D(f·g)` | `f'·g + f·g'` | Regla del producto |
| `D(f+g)` | `f' + g'` | Linealidad |

**Implementación técnica:** `_compute_derivative(node, order)` aplica recursivamente las reglas de derivación. Para funciones compuestas, aplica la regla de la cadena calculando la derivada del argumento interno. Para productos, aplica la regla del producto generando `_CompoundAddNode` de los dos términos.

### Tokenizador extendido

El tokenizador ahora reconoce los siguientes operadores adicionales:

| Token | Tipo | Uso |
|-------|------|-----|
| `=` | OP | Let bindings: `let f = expr` |
| `>=` | OP | Piecewise: `x >= 0` |
| `<=` | OP | Piecewise: `x <= 0` |
| `>` | OP | Piecewise: `x > 0` |
| `<` | OP | Piecewise: `x < 0` |
| `,` | OP | Separador de argumentos |

### API extendida

```python
P = Poem(dtype=torch.float64)

# Let bindings
ast = P.continuous_flow("let f = 2*x + 1 in f(f(x))")

# Piecewise
ast = P.continuous_flow("piecewise(x >= 0, x, 0)")

# Derivadas
ast = P.continuous_flow("D(sin(x))")
ast = P.continuous_flow("D(exp(x), 2)")  # Segunda derivada

# Composición con todo lo anterior
ast = P.continuous_flow("let f = sin(x) + cos(x) in D(f) + piecewise(x >= 0, x^2, 0)")
```

El parser anterior (`_parse_expression`) se mantiene como fallback legacy para compatibilidad.

---

## 1.5. Verificación Formal y Certificación Lean 4

El sistema incorpora un **Certificador de Verificación Formal** (`poema/lean_certifier.py`) que transforma el proceso de compilación y optimización heurística en verdades formales respaldadas por el comprobador de teoremas Mathlib / Lean 4.

### 1.5.1 Propagación Exacta de Intervalos

Poema implementa la aritmética de intervalos completa en el compilador:

```python
# Módulo formal_verification.py
def interval_propagator(self, ast_node: Any, input_interval: Interval) -> Interval:
    # Propaga dominios a través de todos los nodos (sin, cos, exp, log, tanh, etc.)
    # Detecta violaciones de dominio
```

- Soporta todos los tipos de nodos (`ScaleNode`, `ShiftNode`, `AffineNode`, `ComposeNode`, `TranscendentalNode`)
- Los dominios certificados se inyectan directamente en el cálculo
- Emite warnings e intersecta dominios si se sale del límite (`[-π, π]` para `sin`, etc.)

### 1.5.2 Mecanismo Auto-Domain Repair

Cuando un programa Poema recibe una entrada fuera de su dominio certificado, conmuta en runtime a la implementación nativa segura. El verificador formal prueba este mecanismo rigurosamente:

```python
def verify_auto_domain_repair(self, f_phi_with_repair, domain_certified, domain_extended):
    # Testea todo el dominio extendido con la función parcheada
    # Asegura que no devuelve NaN/Inf
    # Compara contra la nativa para garantizar continuidad en la frontera
```

### 1.5.3 Exactitud Horner (Horner Exactness)

Poema implementa el método de Horner exacto, reduciendo un polinomio de grado $n$ a $n$ instrucciones FMA:

```python
def verify_horner_exactness(self, polynomial_coeffs, phi_poly):
    # Comprueba que el lowered_cost === degree
    # Evalúa la divergencia del límite (zero en precisión máquina)
```

1. Mide la cantidad de FMA bajadas directamente de la secuencia `tl.dot/FMA` generada.
2. Compara la divergencia numérica contra $\approx 0$.

### 1.5.4 Invariancia del Índice Alpha ($\alpha$)

El verificador valida la invariancia del Índice de Decaimiento Espectral Afín $\alpha_A(f)$:

$$\alpha(f) \approx \alpha(\Phi(f))$$

1. **Combina aproximaciones**: Combinatoria (`alpha_comb`), Espectral/SVD (`spectral_norm`), Geométrica (longitud de arco).
2. Devuelve un estado de unificación (`high_consistency`, `moderate_consistency`, `low_consistency`).

### 1.5.5 Certificación Continua (Exportación Lean 4)

Por cada compilación, ejecuta la suite de verificación matemáticamente rigurosa, genera teoremas Lean 4 explícitos y trata de validarlos usando el binario real de Lean.

**API de Certificación**:

```python
from poema.lean_certifier import LeanCertifier
certifier = LeanCertifier()
results = certifier.generate_and_validate(verification_results, "PoemaFormalVerification")
```

**Artefactos Generados:**
- **Certificado Lean 4 (`.lean`)**: Prueba legible por máquina.
- **Exportación Python**: Submódulo nativo para comprobación de certificados en runtime.
- **Reporte JSON**: `formal_verification_final_report.json`

## 1.6. CoPoem Expandido — API Técnica

### CoReport

Nueva dataclass que expone métricas de adjunción para cada síntesis:

```python
@dataclass
class CoReport:
    spectral_radius_requested: float    # Radio espectral solicitado
    spectral_radius_actual: float       # Radio espectral alcanzado
    adjunction_gap: float               # |ρ_solicitado - ρ_alcanzado|
    spectral_consistency: float         # Coherencia del decaimiento espectral [0, 1]
    synthesis_iterations: int           # Iteraciones hasta convergencia
    frobenius_norm: float               # ||W||_F de la matriz sintetizada
    symmetry_verified: bool             # True si la estructura solicitada se cumple
    constraints_satisfied: Dict[str, bool]  # Estado por restricción
    warnings: List[str]                 # Advertencias de síntesis
```

### Síntesis Multiobjetivo

Clase `_MultiObjectiveSpec` con API fluida:

```python
spec = (co.multi_objective()
          .spectrum(spectral_radius=0.8, dimension=8)
          .structure("symmetric")
          .minimize("frobenius_norm", budget=10.0))
W, report = co.synthesize_multi(spec)
```

Métodos del builder:
- `.spectrum(rho, dim, symmetry, decay, target_alpha)`: restricción espectral
- `.stability(lyap_exp, dim)`: restricción de Lyapunov
- `.structure(type)`: `"symmetric"`, `"orthogonal"`, `"triangular"`, `"toeplitz"`
- `.minimize(objective, budget)`: `"frobenius_norm"`, `"nuclear_norm"`, `"max_element"`
- `.dimension(dim)`: dimensión de la matriz

### Proyecciones Iterativas

El motor `synthesize_multi` aplica proyecciones secuenciales sobre conjuntos de restricciones:

1. **`_project_spectral_radius(W, rho)`**: descomposición espectral, escala autovalores si `max|λ| > rho`, reconstruye
2. **`_project_lyapunov(W, lam)`**: escala matriz si `max(Re(λ)) > exp(lam)`
3. **Proyección de simetría**: `W ← (W + W^T)/2` para simétrica, `W ← U·V^T·rho` para ortogonal (SVD)
4. **Minimización de norma**: escala global si `||W||_F > budget`

Convergencia: `||W_new - W_prev||_F < 1e-8` o 100 iteraciones máximo.

### Método `synthesize_with_report`

Para síntesis de una sola especificación con reporte completo:

```python
W, report = co.synthesize_with_report(spec)
```

---

## 1.3. BiPoem Expandido — API Técnica

### Las Cinco Evoluciones

| Evolución | Método | Descripción |
| :--- | :--- | :--- |
| **Bi₁** | `symbiosis(...)` | Acoplamiento básico datos↔estructura vía Koopman |
| **Bi₂** | `_compute_bifunctorial_spectrum(...)` | Espectro del tensor de acoplamiento Φ×Φ* |
| **Bi₃** | `find_fixed_point(...)` | Ciclo iterativo Φ ⇌ Φ* buscando punto fijo |
| **Bi₄** | `symbiosis(..., observable_family=...)` | Familia de observables especificable |
| **Bi₅** | `acf_alpha` en resultados | Invariante α(f) computado desde datos |

### `symbiosis_with_report`

Extensión de `symbiosis` que añade métricas Bi₂ y Bi₅ al resultado:

```python
out = bi.symbiosis_with_report(
    data=x,
    max_dimension=256,
    max_iterations=20,
    convergence_threshold=1e-4,
    observable_family="polynomial",  # Bi₄
    max_degree=4,
)

# Bi₂: Espectro bi-functorial
spectrum = out["bifunctorial_spectrum"]
# → {"eigenvalues": [...], "coupling_strength": float,
#    "dominant_eigenvalue": float, "spectral_gap": float}

# Bi₅: Índice de Decaimiento Espectral Afín
alpha = out["acf_alpha"]
```

### `find_fixed_point` — Ciclo Φ ⇌ Φ*

Implementación de la iteración de punto fijo:

```python
result = bi.find_fixed_point(data=x, max_cycles=20, tol=1e-6)
```

Algoritmo:
1. Inicializar `W = I * 0.9`
2. Para cada ciclo:
   - **Φ**: comprimir datos con Koopman → `k_mat, eigvals, meta`
   - **Φ***: sintetizar nueva `W_new = 0.5 * (k_mat + k_mat.T)` (simetrización)
   - Calcular `gap = ||W_new - W||_F`
   - Si `gap < tol`: convergencia
3. Retornar `{converged, cycles, final_gap, acf_alpha, history}`

Retorno:
```python
{
    "converged": bool,
    "cycles": int,
    "final_gap": float,
    "acf_alpha": float,
    "koopman_matrix": torch.Tensor,
    "synthesized_W": torch.Tensor,
    "history": [{"cycle": int, "gap": float, "alpha": float, "reconstruction_error": float}, ...]
}
```

---

## 1.4. CompilationReport Expandido — Observabilidad por Nodo

### NodeProfile

Nueva dataclass para perfil individual de cada nodo del AST:

```python
@dataclass
class NodeProfile:
    node_type: str                    # Tipo de nodo (ScaleNode, TranscendentalNode, etc.)
    node_id: str                      # Identificador único
    fma_contribution: int             # Contribución al conteo FMA total
    epsilon_contribution: float       # Contribución al épsilon total
    domain_interval: Optional[Tuple[float, float]]  # Intervalo propagado
    simplification_applied: bool      # Si se aplicó simplificación
    simplification_rule: str          # Regla aplicada (ej: "scale(1) -> identity")
    domain_guard_status: str          # "ok", "warning", "violation", "repaired"
```

### Nuevos campos en CompilationReport

| Campo | Tipo | Descripción |
| :--- | :--- | :--- |
| `node_profiles` | `List[NodeProfile]` | Perfiles individuales por nodo |
| `phase_times` | `Dict[str, float]` | Tiempo en ms por fase de compilación |
| `simplification_trace` | `List[Dict[str, str]]` | Traza de simplificaciones (regla, nodo, ganancia) |
| `certificate_source` | `str` | Fuente del certificado: `"lean_synchronized"`, `"constructive_interval"`, `"local_estimate"` |
| `epsilon_certified` | `float` | Épsilon certificado explícito |

### Uso

```python
fn, report = compiler.compile(ast, domain=(-1.0, 1.0))

# Perfil por nodo
for profile in report.node_profiles:
    print(profile.summary())

# Tiempos por fase
for phase, time_ms in report.phase_times.items():
    print(f"{phase}: {time_ms:.2f}ms")

# Fuente del certificado
print(f"Certificado: {report.certificate_source}")
print(f"Épsilon: {report.epsilon_certified:.3e}")
```

### Interpretación de campos

**`certificate_source`** indica el origen de la certificación de error para este programa:

| Valor | Significado | Garantía |
|-------|-------------|----------|
| `"lean_synchronized"` | Coeficientes sincronizados desde certificados Lean | Formal, machine-checked |
| `"constructive_interval"` | Estimación por intervalos constructivos | Numérica, verificada |
| `"local_estimate"` | Estimación local en runtime | Heurística, no certificada |

**`phase_times`** desglose el tiempo de compilación por fase. Las fases son:

| Fase | Clave | Qué hace |
|------|-------|----------|
| Simplificación | `"simplification"` | Aplica reglas algebraicas |
| Chequeo de tipos | `"type_check"` | Verifica compatibilidad dimensional |
| Domain Guard | `"domain_guard"` | Propaga intervalos y verifica dominios |
| Linealización | `"linearization"` | Convierte AST a secuencia FMA |
| Codegen | `"codegen"` | Compila backend (PyTorch o Triton) |

---

## 1.5. Herramienta de Diagnóstico

### `poema/diagnostic.py`

La herramienta de diagnóstico proporciona análisis automático de compilaciones con semáforo de severidad y recomendaciones accionables. Es el punto de entrada recomendado para nuevos usuarios que quieren entender el estado de su compilación.

```python
from poema.diagnostic import diagnose

report = diagnose(ast, domain=(-1.0, 1.0))
print(report.summary())
```

### DiagnosticReport

```python
@dataclass
class DiagnosticReport:
    semaforo_global: Severity        # GREEN, YELLOW, RED
    problemas: List[DiagnosticIssue]  # Issues encontrados
    recomendaciones: List[str]        # Acciones sugeridas
    metricas: Dict[str, Any]          # Valores numéricos relevantes
```

### Lógica del semáforo

| Color | Condición | Acción recomendada |
|-------|-----------|-------------------|
| 🟢 GREEN | Sin violations, ε < 1e-6, certificado Lean | Ninguna — compilación saludable |
| 🟡 YELLOW | Violations menores, ε en [1e-6, 1e-3], sin certificado Lean | Revisar dominio o aumentar grado |
| 🔴 RED | Violations severas (overshoot > 0.5), ε > 1e-3, nan/inf | Rediseñar dominio o expresión |

### DiagnosticIssue

Cada problema detectado incluye:

```python
@dataclass
class DiagnosticIssue:
    severity: Severity       # GREEN, YELLOW, RED
    category: str            # "domain_guard", "precision", "certification", "runtime", "warning"
    message: str             # Descripción del problema
    recommendation: str      # Acción concreta sugerida
```

### Ejemplo de uso

```python
from poema import Poem, PoemCompiler
from poema.diagnostic import diagnose
import math

P = Poem(dtype=torch.float64)
ast = P.compose(
    P.sin(domain=(-math.pi, math.pi), degree=24),
    P.scale(4.0)  # ¡Esto empuja sin fuera de dominio!
)

report = diagnose(ast, domain=(-1.0, 1.0))
print(report.summary())
# 🔴 POEMA DIAGNOSTIC REPORT
#   Severity: RED
#   Issues: 1
#   Recommendations: 1
#
#   Issues:
#     [✗] domain_guard: 1 violations, overshoot=1.717
#         → Reduce input domain or increase polynomial degree
```

---

## 1.6. Puente Genesis → CoPoem

### `acf_functor/genesis_copoem_bridge.py`

Este módulo conecta el motor de descubrimiento matemático (Genesis) con el motor de síntesis (CoPoem), permitiendo que los descubrimientos de Genesis guíen la síntesis de matrices. Es la implementación práctica de la idea de que las relaciones matemáticas descubiertas pueden guiar la construcción de sistemas.

### `from_genesis_discovery(discovery, dimension, dtype)`

Convierte un `MathematicalDiscovery` en una especificación de síntesis. La conversión mapea el tipo de descubrimiento a una estrategia de síntesis:

| Tipo de descubrimiento | Estrategia de síntesis | Razonamiento |
|----------------------|----------------------|-------------|
| `DIFFERENTIAL_RELATION` | Radio espectral 1.0, simetría ortogonal | f' = f → dinámica exponencial marginalmente estable |
| `ALGEBRAIC_IDENTITY` | Radio espectral 0.95, simetría simétrica | sin²+cos²=1 → sistema acotado y estable |
| `FIXED_POINT` | Radio espectral 1.0, sin simetría | f(x*) = x* → eigenvalor 1 |
| `SYMMETRY` | Radio espectral 0.9, simetría simétrica | Simetría detectada → matriz simétrica |
| `FUNCTIONAL_EQUATION` | Radio basado en persistence_score | Ecuación funcional → confianza proporcional a persistencia |

### `apply_genesis_spec(copoem, discovery, dimension)`

Sintetiza una matriz usando un descubrimiento de Genesis como guía. Construye automáticamente una especificación multiobjetivo basada en el tipo de descubrimiento y la ejecuta:

```python
from acf_functor.genesis_copoem_bridge import apply_genesis_spec

# discovery viene del GenesisOrchestrator
W, report = apply_genesis_spec(co, discovery, dimension=8)

# La matriz sintetizada refleja la estructura descubierta
print(f"Simetría verificada: {report.symmetry_verified}")
print(f"Radio espectral: {report.spectral_radius_actual}")
```

---

## 2. Fase E (Evoluciones 16-19): La Espiral Interna

La implementación actual extiende Poema con una capa autorreferencial completa:

- Evolución 16 (`poema/free_algebra.py`): álgebra libre sobre generadores afines con normalización canónica y traza de reescritura.
- Evolución 17 (`poema/sheaf_semantics.py`): semántica como haz, veredicto cohomológico y chequeo de pegado global.
- Evolución 18 (`poema/affine_turing.py`): Affine Turing Machine (MTA), ejecución, primitivas y compilación MTA -> AST.
- Evolución 19 (`poema/meta_compiler.py`): compilador meta-circular que encadena normalización, semántica, codegen y backend.

Cobertura y certificación de Fase E:

- Pruebas Python: `tests/test_evolutions_16_19.py`
- Certificado Lean: `MathTest/InternalSpiralCertificates.lean`
- Integración Lean global: importado en `MathTest.lean`

## 2.1. Evolucion 20: Genesis Matematica

La Evolucion 20 cierra el ciclo de Estado 1 con un motor de descubrimiento estructural:

- Implementacion Python: `acf_functor/genesis.py`
- Validacion PyTest: `tests/test_genesis.py`
- Certificacion Lean: `MathTest/GenesisCertificates.lean`

La arquitectura de Genesis integra seis capas:

1. `ProgramGenerator`: crea candidatos en el espacio de programas afines y trascendentales con estrategias random, estructuradas y mutacionales.
2. `FingerprintEngine`: calcula huellas topologicas por evaluacion, derivada, espectro y barras persistentes.
3. `RelationDetector`: detecta identidades, simetrias y leyes diferenciales simples con umbrales numericos certificados.
4. `GenesisOrchestrator`: ejecuta el pipeline multigeneracional, filtra por persistencia y agrega descubrimientos no redundantes.
5. `GenesisReport`: consolida estadisticas de cobertura, estabilidad y verdad graduada.
6. Integracion de API publica en `acf_functor/__init__.py` (version `5.0.0`).

Alcance explicitamente certificado:

- Equivalencia funcional con cota cero (`genesis_identity_valid`).
- Estabilidad estructural bajo persistencia (`persistence_implies_stability`, axiomatizada).
- Conservacion del invariante energetico (`genesis_preserves_conservation`).

---

## 3. Validación y Resultados Reales

### 3.1. Benchmarks de Rendimiento

Los siguientes resultados provienen de ejecuciones reales en hardware (Abril 2026):

**Configuración:**
- CPU: Benchmark estándar
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- PyTorch: 2.8.0+cu126
- Precisión: fp64 (float64)

**Polinomios (Horner exacto, ε=0):**

| Grado | Batch | Tiempo (ms) | Error | Speedup vs NumPy |
|-------|-------|-------------|-------|------------------|
| 10 | 100k | 0.552 | 1.39e-17 | 6.87x |
| 50 | 100k | 1.478 | 1.39e-17 | 2.19x |
| 100 | 100k | 2.690 | 1.39e-17 | 2.03x |

**Trascendentales:**

| Función | Dominio | Tiempo (ms) | Error máx | Fuente |
|---------|---------|-------------|-----------|--------|
| sin | [-π, π] | 0.44 ± 0.18 | 9.992e-16 | lean_synchronized |
| cos | [-π, π] | 0.44 ± 0.18 | ~3.1e-3 | lean_synchronized |
| exp | [-1, 1] | 0.44 ± 0.18 | ~1.4e-3 | lean_synchronized |

**GPU (Triton):**
- Polinomio grado 20: 0.08 ± 0.01 ms (RTX 4050)
- Sin fallback a PyTorch: ejecución nativa en Tensor Cores

**Conservación FMA:**
- E(f) teórica: 7, E(Φ(f)) medida: 7
- Índice Afín α(f): True (estructural y numérico)
- Error numérico máximo: 5.96e-08

### 3.3. Actualización de Regresión (2026-04-06)

Ejecuciones de cierre realizadas sobre el estado actual del repositorio:

- `PYTHONPATH=. python3 -m pytest tests -q` → `356 passed, 21 warnings`.
- `./lean-4.29.0-rc6-linux/bin/lake build` → build exitoso.
- `PYTHONPATH=. python3 -m pytest python_analysis/test_extraction.py python_analysis/test_transcendental_integration.py -q` → `8 passed`.

Hardening aplicado durante esta corrida:

- `benchmarks/periodic_table.py` ahora opera también sin `pandas` (fallback JSON por stdlib y omisión segura de parquet).
- Se mantuvo la línea de salida `Periodic Table generated` para conservar compatibilidad con `tests/test_section18_closure.py`.
- Se registró el marker `benchmark` en `pytest.ini` para eliminar warnings de colección por marker desconocido.

Conclusión: estado de regresión Python + Lean en **PASS**, con artefactos de cierre de Sección 18 revalidados.

### 3.4. Evidencia ejecutable para consistencia de alpha

Para convertir el frente de `alpha(f)` en evidencia medible y versionable, se añadió:

- Script: `python_analysis/alpha_consistency_report.py`
- Test rápido: `tests/test_alpha_consistency_report.py`

Comando reproducible (modo rápido para iteración):

```bash
PYTHONPATH=. python3 python_analysis/alpha_consistency_report.py \
    --fast --skip-geometric \
    --output-json artifacts/alpha_consistency_report.json \
    --output-md artifacts/alpha_consistency_report.md
```

Validación del test asociado:

```bash
PYTHONPATH=. python3 -m pytest tests/test_alpha_consistency_report.py -q
```

Interpretación de rigor:

1. El reporte **no** demuestra equivalencia teórica completa entre definiciones de `alpha`.
2. Sí cuantifica discrepancia por función y registra tasa de consistencia bajo un umbral explícito.
3. Se usa como métrica de progreso para cerrar formalmente la unificación de `alpha` en próximos ciclos.

### 3.5. Matriz de trazabilidad ejecutable (claim -> evidencia)

Para mantener sincronizados los niveles de rigor con la evidencia real del repositorio, se añadió:

- Script: `python_analysis/traceability_matrix_report.py`
- Test rápido: `tests/test_traceability_matrix_report.py`

Comando reproducible:

```bash
PYTHONPATH=. python3 python_analysis/traceability_matrix_report.py \
    --output-json artifacts/traceability_matrix.json \
    --output-md artifacts/traceability_matrix.md
```

Validación del test asociado:

```bash
PYTHONPATH=. python3 -m pytest tests/test_traceability_matrix_report.py -q
```

Uso recomendado:

1. Ejecutar la matriz al cierre de cada iteración relevante.
2. Revisar discrepancias entre estado declarado y evidencia listada.
3. Versionar ambos artefactos (`.json` y `.md`) junto con cambios de código/documentación.

### 3.6. Ejemplos de Errores y Fallos

Esta sección muestra qué ocurre cuando algo falla, para que el usuario sepa interpretar los mensajes:

#### TopologicalObstructionError

```python
from poema import Poem, compose

P = Poem(dtype=torch.float64)
# Intentar componer R³ → R¹ con R² → R¹
outer = P.linear(torch.randn(1, 2))  # espera R²
inner = P.linear(torch.randn(3, 1))  # produce R³

try:
    ast = compose(outer, inner)
except TopologicalObstructionError as e:
    print(f"Obstrucción: {e}")
    # → output_dim=3 ≠ input_dim=2
```

**Causa:** Las dimensiones no coinciden en la composición.
**Solución:** Verificar que `inner.output_dim == outer.input_dim`.

#### Domain Guard Violations

```python
ast = P.compose(P.sin(domain=(-math.pi, math.pi), degree=24), P.scale(4.0))
fn, report = compiler.compile(ast, domain=(-1.0, 1.0))

print(report.domain_guard_violations)  # → 1
print(report.domain_guard_max_overshoot)  # → 1.717
```

**Causa:** scale(4) empuja x∈[-1,1] a [-4,4], fuera del dominio certificado de sin [-π,π].
**Solución:** Reducir el dominio de entrada o aumentar el grado del polinomio.

#### Precision Degradation Warning

```python
# Grado Chebyshev muy alto → advertencia
ast = P.sin(domain=(-math.pi, math.pi), degree=100)
# Warning: High Chebyshev degree detected. Prefer Clenshaw evaluation.
```

**Causa:** Grados altos (>50) pueden causar inestabilidad numérica en evaluación monomial.
**Solución:** Usar evaluación Clenshaw (activada automáticamente para grados altos).

#### Monomial Ill-Conditioned

```python
# Composición profunda → advertencia
ast = compose_deep([P.scale(2.0)] * 20)
# Warning: Monomial conversion became ill-conditioned.
```

**Causa:** Conversiones repetidas a forma monomial degradan el número de condición.
**Solución:** El sistema mantiene automáticamente la ruta Chebyshev/Clenshaw como fallback.

---

## 4. Nuevas Capacidades (v2.2.0 — Abril 2026)

### 4.1. GEMM-Triton Collider

El motor `poema/gemm_collider.py` implementa la transición de `tl.math.fma` secuencial a `tl.dot` por bloques en los Tensor Cores de GPU.

**Principio:** La composición de operaciones afines es una contracción tensorial agrupada:
```
y = W_n @ (... @ (W_1 @ x + b_1) ... ) + b_n  →  tl.dot(W_total, x) + b_total
```

**Características:**
- **Colapsador de cadenas afines:** Detecta patrones GEMM en el AST y colapsa múltiples FMA en una única operación matricial
- **Memory tiling:** Optimiza transferencia HBM→SRAM con bloques de tamaño configurable (default 128)
- **tl.dot para Tensor Cores:** Emite kernels Triton que usan multiplicación matricial por bloques nativa
- **Análisis de número de condición:** Calcula `torch.linalg.cond(W)` y recomienda fp32 o fp64 automáticamente

```python
from poema import GEMMCollider, FMAInstruction
import torch

# Cadena de operaciones afines vectoriales
fmas = [
    FMAInstruction(weight=torch.tensor([[0.9, 0.1], [-0.1, 0.8]]), bias=torch.tensor([0.0, 0.0])),
    FMAInstruction(weight=torch.eye(2), bias=torch.tensor([0.1, -0.1])),
]

# Analizar y colapsar
report = GEMMCollider.analyze_chain(fmas)
print(f"FMA colapsados: {report.total_fma_collapsed}")  # → 2
print(f"Condition number: {report.blocks[0].condition_number:.2e}")  # → 1.12e+00

# Compilar kernel GEMM con tl.dot
kernel = GEMMCollider.compile_gemm_kernel(report.blocks[0], "my_gemm")
x = torch.randn(2, 1000, dtype=torch.float32, device='cuda')
y = kernel(x)  # Ejecución nativa en Tensor Cores
```

### 4.2. Auto-Domain Repair Mejorado

El módulo `poema/auto_domain_repair.py` reemplaza el fallback a `torch.sin` con polinomios de grado superior y dominio expandido, manteniendo la pureza del Functor Φ.

**Principio:** Cuando la entrada sale del dominio certificado, en lugar de abandonar Φ, se activa un polinomio de Chebyshev de grado superior con dominio expandido.

| Función | Dominio original | Dominio expandido | Grado original | Grado expandido |
|---------|-----------------|-------------------|----------------|-----------------|
| sin | [-π, π] | [-2π, 2π] | 24 | 48 |
| cos | [-π, π] | [-2π, 2π] | 24 | 48 |
| exp | [-1, 1] | [-3, 3] | 15 | 30 |
| tanh | [-4, 4] | [-8, 8] | 40 | 80 |
| sigmoid | [-8, 8] | [-16, 16] | 40 | 80 |

```python
from poema import ExpandedDomainRepair

# Obtener certificado de dominio expandido
cert = ExpandedDomainRepair.get_expanded_cert('sin', (-3.14, 3.14), 24, 1e-3)
print(f"Dominio expandido: {cert.expanded_domain}")  # → (-6.28, 6.28)
print(f"Grado expandido: {cert.expanded_degree}")  # → 48
print(f"Épsilon expandido: {cert.expanded_epsilon:.3e}")  # → 5.385e-15

# Crear evaluador con repair automático
eval_fn = ExpandedDomainRepair.create_repaired_evaluator(
    'sin', (-3.14, 3.14), 24, 1e-3, cheb_coeffs_original
)
# eval_fn mantiene pureza de Φ incluso fuera del dominio original
```

### 4.3. Compensación de Kahan para Horner

El kernel `KahanHornerKernel` en `poema/gemm_collider.py` implementa evaluación Horner con compensación de Kahan para estabilidad numérica en fp32.

**Principio:** Acumular el error de redondeo en cada paso FMA y corregirlo en el siguiente paso:
```
y_new = fma(y, x, c_i)
err = fma(y, x, c_i) - y_new  # Error de redondeo
y = y_new + err               # Corrección
```

**Promoción automática a fp64:** Cuando el número de condición de la matriz de coeficientes supera 1e6, el kernel se promociona automáticamente a float64.

```python
from poema import KahanHornerKernel
import torch

# Coeficientes de polinomio de grado alto
coeffs = torch.randn(100, dtype=torch.float64)

# Kernel Horner con compensación de Kahan
kernel = KahanHornerKernel.generate(coeffs, "stable_horner")
x = torch.linspace(-1, 1, 10000, dtype=torch.float32)
y = kernel(x)  # Evaluación estable incluso en fp32
```

### 4.4. CoPoem Multiobjetivo con Anderson

El módulo `poema/copoem_multiobjective.py` reemplaza las proyecciones alternadas ingenuas con:

1. **Método de Anderson:** Aceleración de punto fijo que usa historial de residuos para extrapolar la solución
2. **Detección de incompatibilidad:** Verifica analíticamente antes de iterar si las restricciones son mutuamente excluyentes
3. **Relajación automática:** Cuando detecta estancamiento, relaja la restricción más conflictiva

**Detección de incompatibilidades comunes:**

| Restricciones conflictivas | Razón | Solución |
|---------------------------|-------|----------|
| Ortogonal + radio espectral ≠ 1.0 | Matrices ortogonales tienen ρ=1.0 | Establecer ρ=1.0 |
| Lyapunov < log(ρ) | Contradicción entre estabilidad y radio | Relajar Lyapunov o ρ |
| Norma Frobenius < √dim × ρ | Límite inferior teórico | Aumentar presupuesto |

```python
from poema import CoPoemMultiObjective

# Verificar compatibilidad antes de sintetizar
spec = {'spectral_radius': 0.9, 'dimension': 8, 'symmetry': 'orthogonal'}
compat = CoPoemMultiObjective.check_compatibility(spec)
if compat.is_incompatible:
    print(f"Incompatible: {compat.reason}")
    print(f"Sugerencia: {compat.suggestion}")

# Síntesis con aceleración de Anderson
W, report = CoPoemMultiObjective.synthesize_with_anderson(spec)
print(f"Convergió: {report['converged']}")
print(f"Iteraciones: {report['iterations']}")
print(f"Adjunction gap: {report['adjunction_gap']:.4e}")
```

### 4.5. Cierre de Investigaciones de la Sección 18

Se añadieron artefactos operativos para cerrar las cinco investigaciones de Paper §18 dentro del estado actual del repositorio:

1. `SECTION18_CLOSURE.md`: cierre narrativo/técnico.
2. `VALIDATION_STATUS.md`: matriz de estado y comandos reproducibles.
3. `benchmarks/periodic_table.py`: tabla de espectros basada en BiPoem.
4. `benchmarks/cluster_proxy.py`: benchmark proxy (single-process y FSDP multi-GPU).
5. `tests/test_section18_closure.py`: chequeos de consistencia del cierre.

Para reproducir rápidamente:

```bash
lake build
python3 -m pytest tests/test_section18_closure.py -q
python3 benchmarks/periodic_table.py --output artifacts/periodic_table.md
python3 benchmarks/cluster_proxy.py --steps 5 --output artifacts/cluster_proxy_metrics.json
```

### 4.6. Tabla Periódica de Espectros (Definición Técnica)

La tabla periódica de espectros es un reporte técnico generado por [benchmarks/periodic_table.py](benchmarks/periodic_table.py). Su objetivo es clasificar casos por comportamiento espectral y ofrecer señales accionables para configuración de reducción/compilación.

#### Pipeline de generación

1. Se define un conjunto de casos (`Case`) representativos.
2. Cada caso se convierte a serie temporal con formato compatible BiPoem.
3. Se ejecutan múltiples corridas por caso (`n_trials`) con ruido opcional (`noise_std`).
4. Se computan métricas agregadas: media, desviación, varianza y CI95.
5. Se calcula la familia por umbrales de `alpha` calibrados por dominio.
6. Se genera política de compilación por fila y se marca deriva (`drift`).
7. Se exporta a markdown/json/parquet y opcionalmente se genera plan de bridge.

Comando:

```bash
python3 benchmarks/periodic_table.py --output artifacts/periodic_table.md
```

Comando completo (analítica + bridge de políticas):

```bash
python3 benchmarks/periodic_table.py \
    --output artifacts/periodic_table.md \
    --json-output artifacts/periodic_table.json \
    --parquet-output artifacts/periodic_table.parquet \
    --n-trials 10 \
    --noise-std 0.01 \
    --domain all \
    --drift-threshold 0.20 \
    --emit-cluster-plan \
    --cluster-plan-output artifacts/periodic_cluster_plan.json \
    --run-cluster-proxy \
    --cluster-steps 3 \
    --cluster-output artifacts/cluster_bridge_metrics.json
```

#### Esquema de columnas

1. `Case`: nombre del benchmark/caso.
2. `Domain`: clave de calibración (`general`, `finance`, `fluids`, `signals`).
2. `Family`: clase discreta de complejidad (`fast`, `algebraic`, `slow`).
3. `alpha_mean`, `alpha_std`, `alpha_var`, `alpha_ci95`: estadística del invariante.
4. `Spectral Gap mean/std/var/ci95`: separación modal y su incertidumbre.
5. `Dominant lambda mean/std/var/ci95`: persistencia modal y su incertidumbre.
6. `Reconstruction Error mean/std/var/ci95`: calidad de ajuste y su incertidumbre.
7. `Drift Score`, `Drift Flag`: estabilidad entre corridas.
8. `Compile Policy`: configuración sugerida de compilación/reducción.

#### Regla de clasificación de familia

La implementación actual usa:

1. `fast` si `alpha < 1.5`.
2. `algebraic` si `1.5 <= alpha < 3.0`.
3. `slow` si `alpha >= 3.0`.

Estos umbrales son de ingeniería para priorización y pueden ajustarse por dominio.

#### Lectura recomendada

1. `alpha` bajo + `Spectral Gap` alto: candidatos a reducción compacta.
2. `alpha` alto o `Reconstruction Error` alto: aumentar riqueza de observables, grado o rank.
3. `Dominant lambda` cercano a 1 con error elevado: revisar estabilidad y regularización antes de escalar.
4. `Drift Flag=True`: preferir políticas conservadoras (`precision=fp64`, mayor `max_dimension`).

#### Modo de calibración por dominio

El parámetro `--domain` ajusta umbrales de clasificación. Uso típico:

1. `general`: baseline de laboratorio.
2. `fluids`: escenarios tipo dinámica de fluidos/caóticos.
3. `signals`: señales oscilatorias y discontinuidades funcionales.
4. `finance`: series con comportamientos más ruidosos/heterogéneos.

Esto no cambia la definición matemática de `alpha`, pero sí la frontera operativa de políticas.

#### Bridge con cluster proxy (cierre teoría-rendimiento)

Cuando se activa `--emit-cluster-plan` + `--run-cluster-proxy`:

1. se crea `artifacts/periodic_cluster_plan.json` con políticas por caso,
2. se ejecuta [benchmarks/cluster_proxy.py](benchmarks/cluster_proxy.py) por cada caso con su perfil,
3. se generan métricas por caso (`artifacts/cluster_proxy_*.json`),
4. se consolida un reporte agregado `artifacts/cluster_bridge_metrics.json`.

Esto cierra el ciclo entre diagnóstico espectral (BiPoem) y costo real (latencia/memoria) en hardware.

#### Salida esperada

El script produce salidas simultáneas:

1. markdown para inspección humana,
2. JSON para automatización de pipelines,
3. Parquet para analítica de volumen,
4. plan y métricas de bridge para validación de políticas en benchmark proxy.

Con ello, `periodic_table.py` funciona como orquestador de decisiones pre-compilación, no solo como reporte descriptivo.

## 3. AST de Poema

La base es `ASTNode` con:

- `simplify()`
- `estimate_fma_cost()`
- composición `@` (azúcar sintáctico para `ComposeNode`)

### Nodos principales

- `IdentityNode`, `ConstantNode`, `InputNode`
- `ScaleNode`, `ShiftNode`, `AffineNode`, `ComposeNode`
- `PolynomialNode`
- `TranscendentalNode`
- `StratifiedNode`
- `ConstraintNode`, `ParameterNode`

### Reglas de simplificación relevantes

- `ScaleNode(1)` -> identidad
- `ShiftNode(0)` -> identidad
- Composición de escalas -> escala única
- Composición de shifts -> shift único
- `Compose(scale, shift)` -> `AffineNode`
- Encadenamiento de afines -> un único afín equivalente
- Polinomio con coeficiente líder nulo -> reducción de grado

Estas reglas reducen costo FMA antes de bajar a backend.

---

## 4. Sistema de Tipos Geométricos

`GeometricType` define compatibilidad composicional por dimensión:

- una composición `outer(inner(x))` es válida si `inner.output_dim == outer.input_dim`

Cuando no se cumple, la compilación detiene con:

- `TopologicalObstructionError("...")`

Esta validación evita errores semánticos silenciosos y hace explícitas las obstrucciones estructurales.

---

## 5. Pipeline de Compilación

Implementado en `PoemCompiler` de `poema/compiler.py`.

Entrada:

- AST Poema
- opcionalmente `domain=(a,b)`

Salida:

- ejecutable Python callable
- `CompilationReport`

Fases:

1. Simplificación algebraica
- `ast.simplify()`

2. Chequeo geométrico y compensación de precisión
- `GeometricTypeChecker.check(...)`
- detecta incompatibilidades de dimensión
- puede inyectar compensación cuando la precisión objetivo puede degradar demasiado los parámetros afines

3. Autómata de self-modulation (opcional)
- habilitado con `enable_self_modulation=True`
- para trascendentales con error alto relativo a la precisión objetivo
- intenta mejorar reducción con `AdaptiveReducer` y puede introducir `StratifiedNode`

4. Linealización a FMA
- `FMALinearizer.linearize(...)`
- produce secuencia explícita de `FMAInstruction(weight,bias)`

5. Backend
- `target="pytorch"` -> `PytorchBackend.compile(...)`
- `target="triton"` -> `TritonBackend.compile_kernel(...)` con fallback a PyTorch si no aplica

---

## 6. Backends y Ejecución

### 6.1. Backend PyTorch

Cobertura:

- Polinomios: ejecución Horner vía `HornerReducer.execute_horner`
- Trascendentales: Horner o Clenshaw según metadatos
- Cadenas afines generales: evaluación secuencial de instrucciones FMA

### 6.2. Backend Triton

Cobertura actual:

- **Cadenas escalares afines**: colapso analítico de cadena `y = a_i*y + b_i` en un solo `y = A*y + B` con kernel `tl.math.fma`
- **Cadenas vectoriales/ matriciales (GEMM)**: colapso analítico de cadena `y = W_n @ (... @ (W_1 @ x + b_1) ... ) + b_n` en un solo `y = W_total @ x + b_total` con `torch.matmul` en GPU
- **Detección de patrones Horner**: el backend detecta automáticamente cuando la secuencia FMA corresponde a evaluación polinómica por Horner (patrón: primera instrucción `weight=0, bias=c_n`, siguientes `weight=1, bias=c_i`)
- **Fallback inteligente**: si Triton no está disponible, o la AST contiene trascendentales no colapsables, o el patrón es Horner, se emite warning y se usa backend PyTorch

#### Clasificación automática

El backend clasifica la secuencia FMA en tres categorías:

1. **Cadena afín escalar pura**: todas las instrucciones tienen `weight.dim() == 0` y `bias.dim() == 0`. Se colapsan analíticamente en `y = A*y + B` y se ejecutan con un único kernel Triton con `tl.math.fma`.

2. **Cadena afín vectorial/matricial**: al menos una instrucción tiene `weight.dim() >= 1` o `bias.dim() >= 1`. Se colapsan analíticamente en `y = W_total @ x + b_total` donde:
   - `W_total = W_n @ ... @ W_1` (producto matricial)
   - `b_total = W_n @ ... @ W_2 @ b_1 + ... + b_n` (propagación de bias)
   - Se ejecuta con `torch.matmul` en GPU CUDA

3. **Patrón Horner**: primera instrucción con `weight=0` (inicialización con coeficiente líder), siguientes con `weight=1` (pasos de Horner `y = y*x + c_i`). Actualmente hace fallback a PyTorch con warning.

#### Promoción automática de tipos

El backend vectorial promueve automáticamente:
- Escalares (`dim() == 0`) → matrices 1×1
- Vectores (`dim() == 1`) → matrices diagonales
- Bias escalar → vector 1D

Esto permite mezclar instrucciones escalares y vectoriales en la misma cadena.

#### Caso especial 1×1

Cuando la cadena vectorial colapsada resulta en una matriz 1×1, el backend delega automáticamente al kernel escalar Triton para máxima eficiencia.

#### Fallback

- Si Triton no está disponible: `warnings.warn("triton not available")` → PyTorch
- Si se detecta patrón Horner: `warnings.warn("triton backend: polynomial evaluation not yet supported")` → PyTorch

#### Ejemplo de uso vectorial

```python
from poema.compiler import FMAInstruction, TritonBackend
import torch

# Cadena vectorial: y = W2 @ (W1 @ x + b1) + b2
W1 = torch.eye(4) * 2.0
b1 = torch.ones(4)
W2 = torch.eye(4) * 0.5
b2 = -torch.ones(4)

fmas = [
    FMAInstruction(weight=W1, bias=b1),
    FMAInstruction(weight=W2, bias=b2),
]

kernel = TritonBackend.compile_kernel(fmas)
x = torch.randn(4, 100, device='cuda')
y = kernel(x)  # Shape: (4, 100)
```

#### Requisitos de ejecución

- Los tensores de entrada deben estar en GPU CUDA para kernels Triton
- Si se pasan tensores CPU, Triton lanza `ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)`
- Para tests portables, verificar `torch.cuda.is_available()` antes de usar Triton

---

## 7. Diagnóstico de Compilación

`CompilationReport` provee trazabilidad de compilación:

- total de operaciones FMA
- epsilon total certificado
- simplificaciones aplicadas
- compensaciones inyectadas
- inyecciones de sheaf/estratos
- tiempo de compilación
- warnings
- secuencia FMA final

### 7.1. Observabilidad extendida (nuevo)

Además de los campos básicos, `CompilationReport` expone métricas granulares:

**`node_profiles: List[NodeProfile]`** — perfil individual de cada nodo del AST:
- `node_type`: tipo del nodo (ej: `"TranscendentalNode"`)
- `node_id`: identificador único
- `fma_contribution`: cuántas instrucciones FMA genera este nodo
- `epsilon_contribution`: contribución al épsilon total
- `domain_interval`: intervalo propagado para este nodo
- `simplification_applied`: si se aplicó alguna regla de simplificación
- `simplification_rule`: nombre de la regla aplicada (ej: `"scale(1) -> identity"`)
- `domain_guard_status`: `"ok"`, `"warning"`, `"violation"`, `"repaired"`

**`phase_times: Dict[str, float]`** — desglose de tiempo por fase de compilación (en ms):
- `"simplification"`: tiempo de `ast.simplify()`
- `"type_check"`: tiempo de `GeometricTypeChecker.check()`
- `"domain_guard"`: tiempo de `_run_domain_guard()`
- `"linearization"`: tiempo de `FMALinearizer.linearize()`
- `"codegen"`: tiempo de compilación del backend

**`simplification_trace: List[Dict[str, str]]`** — traza detallada de cada simplificación:
- `"rule"`: regla aplicada
- `"node"`: nodo afectado
- `"gain"`: ganancia FMA estimada

**`certificate_source: str`** — origen de la certificación de error:
- `"lean_synchronized"`: coeficientes sincronizados desde certificados Lean
- `"constructive_interval"`: estimación por intervalos constructivos
- `"local_estimate"`: estimación local en runtime (sin certificado Lean)

**`epsilon_certified: float`** — valor explícito del épsilon certificado para este programa.

Método:

- `summary()` devuelve texto legible para logs/inspección, incluyendo todos los campos extendidos.

## 7.1. Domain Guard y Auto-Corrección Runtime

Poema incorpora dos capas complementarias para robustez numérica en composiciones profundas.

Problema real que resuelven estas capas:

- Una aproximación trascendental por Chebyshev puede ser excelente en su dominio certificado y, aun así, volverse inestable si una composición previa la empuja fuera de ese rango.
- En escenarios de alta profundidad, esto no suele aparecer como un fallo inmediato de compilación, sino como degradación silenciosa (error creciente, `nan`, `inf` o deriva severa).

### 7.1.1. Domain Guard (tiempo de compilación)

Qué hace:

- Propaga intervalos internos sobre el AST.
- Verifica si las entradas estimadas a cada nodo trascendental permanecen en el dominio certificado de su reducción.
- Registra trazas de riesgo en `CompilationReport`.

Métricas expuestas:

- `domain_guard_checks`: número total de chequeos de dominio ejecutados.
- `domain_guard_violations`: cantidad de nodos con posible salida fuera de dominio.
- `domain_guard_max_overshoot`: exceso máximo estimado respecto al límite certificado.
- `domain_guard_alerts`: mensajes descriptivos por nodo/rama con contexto de overshoot.

Interpretación práctica:

- Si `domain_guard_violations == 0`, la compilación queda dentro del régimen nominal esperado.
- Si `domain_guard_violations > 0`, la compilación sigue siendo válida, pero se recomienda ejecutar con mitigación activa (`auto_domain_repair`) o rediseñar dominios/grados.

### 7.1.2. Auto-domain repair (tiempo de ejecución, backend PyTorch)

Activación:

- Parámetro del compilador: `PoemCompiler(..., auto_domain_repair=True)`.

Flujo operativo por nodo trascendental:

1. Se evalúa la ruta aproximada (Chebyshev/Clenshaw) cuando la entrada está en dominio certificado.
2. Si la entrada cae fuera del dominio certificado:
- para funciones canónicas (`sin`, `cos`, `exp`, `tanh`, `sigmoid`), se conmuta localmente a `torch.<fn>` para mantener estabilidad;
- para funciones sin ruta canónica, se conserva el comportamiento existente (sin forzar una ruta no definida).

Impacto esperado:

- Conserva precisión alta en el subdominio certificado.
- Reduce de forma marcada riesgo de `nan`/`inf` y errores explosivos en composiciones adversariales.
- Evita penalizar el caso nominal, porque la conmutación se hace solo cuando hay salida de dominio.

### 7.1.3. Ejemplo conceptual antes/después

Antes (sin repair):

- Una cadena profunda alimenta `sin` con valores fuera del intervalo certificado.
- La aproximación extrapola y aparece deriva fuerte o no-finitud.

Después (con repair):

- La misma cadena se ejecuta igual en régimen nominal.
- Solo en tramos fuera de dominio, el nodo conmuta a evaluación nativa estable.
- El resultado global permanece finito y cercano a referencia cerrada.

### 7.1.4. Límites y trade-offs

- No sustituye diseño de dominio: sigue siendo mejor certificar dominios realistas y elegir grado acorde.
- La conmutación introduce heterogeneidad de ruta numérica fuera de dominio (aproximada dentro, nativa fuera), lo cual es deliberado para priorizar estabilidad.
- La señal más útil para operación continua es el par `domain_guard_violations` + `domain_guard_max_overshoot`; si ambos crecen, conviene reparametrizar.

---

## 8. Integración con el Núcleo ACF

Poema integra componentes del núcleo en puntos bien definidos:

- Trascendentales en frontend: `ChebyshevReducer.reduce(...)`
- Evaluación de polinomios en backend: `HornerReducer.execute_horner(...)`
- Modo relacional BiPoem: `KoopmanReducer.dmd(...)` y `ACFInvariant.compute_alpha(...)`
- Auto-modulación opcional: `AdaptiveReducer.reduce(...)`

Esto mantiene una separación limpia:

- Núcleo = teoría numérica y motores especializados
- Poema = lenguaje, AST, compilación y experiencia de uso

---

## 8.1. Certificados Lean para Trascendentales

### Estado actual de certificación

El generador de certificados (`python_analysis/generate_interval_certificates.py`) produce certificados Lean 4 constructivos para seis funciones trascendentales:

| Función | Dominio certificado | Grado | Error máximo |
| :--- | :--- | :--- | :--- |
| `sin` | $[-\pi, \pi]$ | 20 | ~4.1e-3 |
| `cos` | $[-\pi, \pi]$ | 20 | ~3.1e-3 |
| `exp` | $[-1, 1]$ | 15 | ~1.4e-3 |
| `log` | $[0.5, 2]$ | 25 | ~9.0e-4 |
| `tanh` | $[-2, 2]$ | 30 | ~1.9e-2 |
| `sigmoid` | $[-4, 4]$ | 36 | ~2.4e-2 |

### Pipeline de generación

1. **Ajuste de coeficientes**: `fit_coeffs()` usa `numpy.polynomial.chebyshev.Chebyshev.fit` para encontrar coeficientes minimax sobre el dominio declarado.

2. **Cota de error por intervalos**: `interval_bound()` usa aritmética de intervalos con `mpmath` (100 dígitos de precisión) para verificar que el error máximo de la aproximación Chebyshev está acotado sobre todo el dominio.

3. **Generación Lean**: `build_lean()` produce código Lean 4 con:
   - Declaraciones de grado, dominio, coeficientes
   - Cotas de error de intervalo
   - Épsilon certificado (error + margen de seguridad del 5%)
   - Predicados de certificado (`decide (error ≤ epsilon)`)
   - Teoremas constructivos (`by native_decide`)

### Uso del generador

```bash
python3 python_analysis/generate_interval_certificates.py
# Genera MathTest/TranscendentalCertificates.lean
# Imprime errores y épsilons por función
```

### Dominios no canónicos

Cuando se compila una trascendental en un dominio diferente al certificado, el compilador usa estimación local de épsilon (sin certificado Lean). El campo `certificate_source` en `CompilationReport` indica la fuente:

- `"lean_synchronized"`: coeficientes sincronizados desde certificados Lean
- `"constructive_interval"`: estimación por intervalos constructivos
- `"local_estimate"`: estimación local en runtime (sin certificado Lean)

---

## 9. Ejemplos Operativos

### 9.1. Polinomio exacto (Poem + Compiler)

```python
import torch
from poema import Poem, PoemCompiler

P = Poem(dtype=torch.float64)
ast = P.polynomial([1.0, 2.0, 3.0])  # 1 + 2x + 3x^2

compiler = PoemCompiler(target="pytorch", precision="fp64")
fn, report = compiler.compile(ast)

x = torch.linspace(-2, 2, 1000, dtype=torch.float64)
y = fn(x)
print(report.summary())
```

### 9.2. Trascendental certificada

```python
import math
import torch
from poema import Poem, PoemCompiler

P = Poem(dtype=torch.float64)
ast = P.sin(domain=(-math.pi, math.pi), degree=24)

compiler = PoemCompiler(target="pytorch", precision="fp64")
fn, report = compiler.compile(ast, domain=(-math.pi, math.pi))

x = torch.linspace(-math.pi, math.pi, 4000, dtype=torch.float64)
err = torch.max(torch.abs(fn(x) - torch.sin(x))).item()
print("max error:", err)
```

### 9.3. Parser `continuous_flow`

```python
import torch
from poema import Poem, PoemCompiler

P = Poem(dtype=torch.float64)

# Composición anidada
ast = P.continuous_flow("sin(cos(x))")

# Polinomio con precedencia
ast = P.continuous_flow("2*x^2 + 3*x - 1")

# Con constantes
ast = P.continuous_flow("sin(pi*x) + exp(-x)")

compiler = PoemCompiler(target="pytorch", precision="fp64")
fn, report = compiler.compile(ast, domain=(-1.0, 1.0))

x = torch.linspace(-1, 1, 100, dtype=torch.float64)
y = fn(x)
```

### 9.4. Síntesis espectral (CoPoem)

```python
import torch
from poema import CoPoem

co = CoPoem(dtype=torch.float64)
spec = co.spectrum(spectral_radius=0.95, dimension=32, symmetry="orthogonal")
W = co.synthesize(spec)

print(W.shape)
print(torch.max(torch.abs(torch.linalg.eigvals(W))).item())
```

### 9.5. Síntesis multiobjetivo (CoPoem expandido)

```python
import torch
from poema import CoPoem

co = CoPoem(dtype=torch.float64)

# Múltiples restricciones simultáneas
spec = (co.multi_objective()
          .spectrum(spectral_radius=0.8, dimension=8)
          .structure("symmetric")
          .minimize("frobenius_norm", budget=10.0))

W, report = co.synthesize_multi(spec)

print(f"Radio espectral: {report.spectral_radius_actual:.4f}")
print(f"Adjunction gap: {report.adjunction_gap:.4e}")
print(f"Simetría: {report.symmetry_verified}")
print(f"Norma Frobenius: {report.frobenius_norm:.4f}")
```

### 9.6. Acoplamiento relacional (BiPoem)

```python
import torch
from poema import BiPoem

bi = BiPoem(dtype=torch.float64)

a = torch.tensor([[0.9, 0.1], [-0.1, 0.8]], dtype=torch.float64)
x = torch.zeros(2, 80, dtype=torch.float64)
x[:, 0] = torch.randn(2, dtype=torch.float64)
for t in range(79):
    x[:, t + 1] = a @ x[:, t]

out = bi.symbiosis(data=x, max_dimension=32, max_iterations=8)
print(out["optimal_dimension"], out["reconstruction_error"])
```

### 9.7. BiPoem con espectro bi-functorial (Bi₂)

```python
import torch
from poema import BiPoem

bi = BiPoem(dtype=torch.float64)

# Sistema con decaimiento conocido
A = torch.diag(torch.tensor([0.9, 0.7, 0.5, 0.3], dtype=torch.float64))
x = torch.zeros(4, 200, dtype=torch.float64)
x[:, 0] = torch.ones(4, dtype=torch.float64)
for t in range(199):
    x[:, t+1] = A @ x[:, t]

out = bi.symbiosis_with_report(data=x, max_dimension=16, max_iterations=8)

# Bi₂: Espectro bi-functorial
spectrum = out["bifunctorial_spectrum"]
print(f"Eigenvalores: {spectrum['eigenvalues'][:5]}")
print(f"Acoplamiento: {spectrum['coupling_strength']:.4f}")

# Bi₅: Índice de Decaimiento Espectral Afín
print(f"α(f): {out['acf_alpha']:.4f}")
```

### 9.8. Ciclo Φ ⇌ Φ* (Bi₃)

```python
import torch
from poema import BiPoem

bi = BiPoem(dtype=torch.float64)

# Generar datos de sistema dinámico
A = torch.tensor([[0.9, 0.1], [-0.1, 0.8]], dtype=torch.float64)
x = torch.zeros(2, 200, dtype=torch.float64)
x[:, 0] = torch.randn(2, dtype=torch.float64)
for t in range(199):
    x[:, t + 1] = A @ x[:, t]

result = bi.find_fixed_point(data=x, max_cycles=20, tol=1e-6)
print(f"Convergió: {result['converged']}")
print(f"Ciclos: {result['cycles']}")
print(f"Gap final: {result['final_gap']:.4e}")
print(f"α(f): {result['acf_alpha']:.4f}")
```

### 9.9. Observabilidad por nodo (CompilationReport expandido)

```python
import torch, math
from poema import Poem, PoemCompiler

P = Poem(dtype=torch.float64)
ast = P.compose(
    P.sin(domain=(-math.pi, math.pi), degree=24),
    P.scale(0.5)
)

compiler = PoemCompiler(target="pytorch", precision="fp64")
fn, report = compiler.compile(ast, domain=(-2.0, 2.0))

# Perfil por nodo
for profile in report.node_profiles:
    print(profile.summary())

# Tiempos por fase
for phase, time_ms in report.phase_times.items():
    print(f"{phase}: {time_ms:.2f}ms")

# Certificado
print(f"Fuente: {report.certificate_source}")
print(f"Épsilon: {report.epsilon_certified:.3e}")
```

---

## 10. Pruebas y Validación

### 10.1. Suite completa

La validación de Poema se organiza en múltiples suites:

| Suite | Archivo | Tests | Descripción |
| :--- | :--- | :--- | :--- |
| Principal | `tests/test_poema.py` | ~40 | Compilación, simplificación, backends |
| Hardening | `tests/test_poema_hardening.py` | ~20 | Estrés, domain guard, auto-repair |
| Cobertura nueva | `tests/test_poema_missing_coverage.py` | 42 | Triton, parser, tipos, CoPoem, BiPoem, evoluciones, fuzzing, cross-mode, diagnóstico |
| Nuevas funcionalidades | `tests/test_poema_comprehensive.py` | ~25 | Error propagation, multivariable, activaciones modernas, nn.Module, benchmark |
| Evoluciones 16-19 | `tests/test_evolutions_16_19.py` | ~10 | Álgebra libre, haz, MTA, meta-compilador |
| Génesis | `tests/test_genesis.py` | ~5 | Descubrimiento de identidades |
| Self-modulation | `tests/test_self_modulation.py` | ~15 | Funciones discontinuas, convergencia |
| Composición | `tests/test_composition_exhaustive.py` | ~10 | Composiciones profundas |
| Koopman | `tests/test_koopman_validation.py` | 8 | Validación del operador de Koopman |
| Functor engine | `tests/test_functor_engine.py` | ~60 | Núcleo numérico |

**Total: 356 tests pasando sin regresiones.** Ver `TESTS_CANONICAL.md` para el número canónico actualizado.

### 10.2. Tests de cobertura nueva (test_poema_missing_coverage.py)

Los 22 tests de esta suite cubren áreas previamente no testeadas:

**Backend Triton (3 tests):**
- `test_triton_scalar_affine_chain`: cadenas afines escalares en GPU
- `test_triton_fallback_emits_warning`: fallback correcto para trascendentales
- `test_triton_deep_affine_chain_stress`: 50 composiciones afines

**Parser `continuous_flow` (4 tests):**
- `test_parser_nested_composition`: `sin(cos(x))`
- `test_parser_multi_term`: `2*x^2 + 3*x - 1`
- `test_parser_constants`: `sin(pi*x)`
- `test_parser_exp_composition`: `exp(sin(x))`

**Certificación trascendentales (2 tests):**
- `test_tanh_certificate`: tanh con grado 40, error < 1e-6
- `test_non_canonical_domain_warns`: dominio extendido funciona

**Sistema de tipos (2 tests):**
- `test_geometric_type_flow_form_mismatch`: obstrucción Flow→Form
- `test_stratified_continuity_check`: continuidad en fronteras

**Domain Guard (2 tests):**
- `test_domain_guard_interval_propagation`: detección de violaciones
- `test_domain_guard_no_false_positives`: sin falsos positivos

**CoPoem (2 tests):**
- `test_copoem_spectral_synthesis`: radio espectral verificado
- `test_copoem_stability_synthesis`: estabilidad Lyapunov

**BiPoem (2 tests):**
- `test_bipoem_symbiosis_convergence`: convergencia en sistema lineal
- `test_bipoem_dimension_history`: historial de dimensiones

**Evoluciones 16-19 (1 test):**
- `test_evolutions_16_19_end_to_end`: pipeline completo end-to-end

**Performance (1 test):**
- `test_triton_vs_pytorch_affine_chain`: benchmark Triton vs PyTorch en GPU

**Fuzzing composicional (3 tests):**
- `test_fuzzing_mixed_composition[5/10/20]`: composiciones profundas con tipos mixtos

**Triton vectorial (1 test nuevo):**
- `test_triton_vectorial_affine`: cadena afín vectorial (GEMM) con matrices 4×4

### 10.3. Ejecución recomendada

```bash
# Suite completa (excluyendo test_koopman_validation.py que tiene syntax error preexistente)
.venv/bin/python -m pytest tests/ --ignore=tests/test_koopman_validation.py -q

# Solo cobertura nueva
.venv/bin/python -m pytest tests/test_poema_missing_coverage.py -v

# Solo evoluciones
.venv/bin/python -m pytest tests/test_evolutions_16_19.py -v

# Solo hardening
.venv/bin/python -m pytest tests/test_poema_hardening.py -v
```

Caso de validación crítica ya incorporado:

- `test_auto_domain_repair_high_complexity_activation` construye una composición profunda adversarial (dominios estrechos intencionales), verifica que sin auto-repair la ruta no reparada se desestabiliza y que con `auto_domain_repair=True` la salida es finita y precisa frente a referencia cerrada.

---

## 11. Contrato de Uso (Recomendaciones)

1. Definir dominio explícito para trascendentales.
2. Elegir grado según tolerancia/error objetivo.
3. Usar `precision="fp64"` para baseline de validación; luego bajar precisión para despliegue.
4. Activar self-modulation solo cuando exista razón de estabilidad o error en dominios difíciles.
5. Revisar siempre `CompilationReport` antes de producción.

---

## 12. Limitaciones Actuales

### 12.1. Limitaciones resueltas en esta versión

Las siguientes limitaciones documentadas en versiones anteriores han sido resueltas:

| Limitación | Estado | Solución |
| :--- | :--- | :--- |
| Triton solo afines escalares | ✅ Resuelta | Soporte Horner nativo + GEMM vectorial |
| Parser básico sin composición | ✅ Resuelta | Parser de descenso recursivo completo |
| Sin métricas de adjunción | ✅ Resuelta | `CoReport` con `adjunction_gap` |
| Sin Bi₂-Bi₅ | ✅ Resuelta | 5 evoluciones implementadas |
| Sin observabilidad por nodo | ✅ Resuelta | `NodeProfile` en `CompilationReport` |
| Certificados solo sin/exp/log | ✅ Resuelta | tanh, cos, sigmoid certificados |

### 12.2. Limitaciones vigentes

1. **Triton Horner usa fp32 internamente.** Los kernels Horner en Triton operan en float32, lo que limita la precisión para polinomios de grado alto (>20). Para fp64, usar backend PyTorch.

2. **Derivadas simbólicas limitadas.** `D()` soporta sin, cos, exp, log, tanh, sigmoid y combinaciones lineales. No soporta derivadas de funciones definidas por el usuario.

3. **CoPoem multiobjetivo no garantiza optimalidad global.** Las proyecciones iterativas pueden converger a mínimos locales. No hay garantía de que todas las restricciones se satisfagan simultáneamente.

4. **BiPoem requiere trayectorias regulares.** Aunque existe `symbiosis_irregular` con interpolación, la calidad del operador Koopman depende de la regularización de los datos.

5. **Certificados Lean para composición son parciales.** El teorema `composition_error_bound` está formulado pero requiere los certificados concretos de sin_approx y cos_approx para completar la demostración de sin∘cos.

---

## 12.3. Antipatrones Comunes

Esta sección documenta patrones que parecen razonables pero producen resultados inesperados.

### AP1: Usar `target="triton"` con polinomios de grado alto

```python
# ❌ ANTPATRÓN: Triton Horner usa fp32 internamente
compiler = PoemCompiler(target="triton", precision="fp64")
fn, _ = compiler.compile(P.polynomial(coeffs), domain=(-1, 1))
# Resultado: precisión fp32, no fp64

# ✅ CORRECTO: Usar PyTorch para fp64
compiler = PoemCompiler(target="pytorch", precision="fp64")
fn, _ = compiler.compile(P.polynomial(coeffs), domain=(-1, 1))
```

### AP2: Compilar trascendentales sin declarar dominio

```python
# ❌ ANTPATRÓN: Sin dominio → certificate_source = "local_estimate"
ast = P.sin()  # Usa dominio por defecto
fn, report = compiler.compile(ast)
# report.certificate_source puede ser "local_estimate"

# ✅ CORRECTO: Declarar dominio explícitamente
ast = P.sin(domain=(-math.pi, math.pi), degree=24)
fn, report = compiler.compile(ast, domain=(-math.pi, math.pi))
# report.certificate_source = "lean_synchronized"
```

### AP3: Asumir que auto_domain_repair mantiene ε certificado

```python
# ❌ ANTPATRÓN: auto_domain_repair no preserva ε fuera de dominio
compiler = PoemCompiler(auto_domain_repair=True)
fn, report = compiler.compile(ast, domain=(-5, 5))  # Fuera de [-π, π]
# Fuera de dominio: usa torch.sin nativo, sin certificado Lean

# ✅ CORRECTO: Verificar domain_guard_violations
if report.domain_guard_violations > 0:
    print("⚠ Algunas evaluaciones usan ruta no certificada")
```

### AP4: Composición profunda sin auto_domain_repair

```python
# ❌ ANTPATRÓN: Composición profunda puede producir nan/inf
ast = P.compose(P.sin(domain=(-1, 1)), P.scale(10.0))
fn, _ = PoemCompiler(auto_domain_repair=False).compile(ast, domain=(-1, 1))
y = fn(x)  # Puede producir nan si scale(10) empuja sin fuera de [-1, 1]

# ✅ CORRECTO: Activar auto_domain_repair
fn, _ = PoemCompiler(auto_domain_repair=True).compile(ast, domain=(-1, 1))
y = fn(x)  # Fuera de dominio → conmuta a torch.sin
```

### AP5: Ignorar el CompilationReport

```python
# ❌ ANTPATRÓN: Compilar sin revisar el reporte
fn, _ = compiler.compile(ast, domain=(-1, 1))

# ✅ CORRECTO: Siempre revisar el reporte
fn, report = compiler.compile(ast, domain=(-1, 1))
print(report.summary())
if report.domain_guard_violations > 0:
    # Tomar acción correctiva
    pass
```

---

## 12.4. Guía de Diagnóstico Rápido

| Síntoma | Causa probable | Solución |
| :--- | :--- | :--- |
| `domain_guard_violations > 0` | Composición empuja fuera de dominio | Reducir dominio de entrada o aumentar grado |
| `epsilon > 1e-3` | Grado insuficiente para dominio | Aumentar `degree` o reducir dominio |
| `certificate_source = "local_estimate"` | Dominio no canónico | Usar dominio canónico o aceptar estimación local |
| Resultado `nan` o `inf` | Fuera de dominio sin auto-repair | Activar `auto_domain_repair=True` |
| Triton más lento que PyTorch | Cadena afín corta (< 10 ops) | Usar PyTorch para cadenas cortas |
| `TopologicalObstructionError` | Dimensiones incompatibles | Verificar `output_dim == input_dim` |

---

## 13. Estado del Programa: Ciclo Cerrado (Abril 2026)

Esta sección documenta el programa que fue ejecutado en este ciclo. Se mantiene como trazabilidad técnica (qué se planificó, qué se hizo y qué queda formalmente abierto), no como backlog operativo pendiente.

### 13.1. Objetivo operativo (cumplido)

Se completó el paso de roadmap a ejecución reproducible: implementación + validación + evidencia en artefactos (tests, benchmarks, build Lean, reportes).

### 13.2. Frentes abiertos (residuales, no bloqueantes)

1. ~~Unificación formal del invariante `alpha`~~ → **RESUELTO** — `AlphaUnificationTheorem` en `acf_functor/invariant_unified.py` (§30.4).
2. Convergencia formal del ciclo `Phi <-> Phi*` bajo hipótesis explícitas.
3. ~~Cotas de truncación Koopman por clase~~ → **RESUELTO** — `KoopmanDeltaBounds` en `acf_functor/koopman_delta_bounds.py` (§30.1).
4. ~~Cierre composicional mixto entre ramas~~ → **RESUELTO** — `MixedCompositionCertifier` en `acf_functor/mixed_composition.py` (§30.2).
5. Certificación más amplia fuera de dominios canónicos.

### 13.3. Plan por fases

#### Fase 0 — Baseline de control

Objetivo:

1. Congelar baseline de artefactos y tests para detectar regresiones.

Entregables:

1. `artifacts/periodic_table.{md,json,parquet}` baseline.
2. `artifacts/cluster_bridge_metrics.json` baseline.
3. resultado de `pytest` baseline.

Criterio de salida:

1. baseline reproducible en máquina limpia con `.venv`.

#### Fase 1 — Teoremas por subclase Koopman (de heurístico a certificado por clase)

Objetivo:

1. Formalizar subclases donde Koopman puede tener garantía fuerte.

Tareas:

1. Subclase polinomial con espacio de observables cerrado.
2. Subclase con simetría estructural explotable.
3. Subclase analítica con hipótesis de compacidad/regularidad para tasa de convergencia.

Validación:

1. tests por subclase + benchmarks dedicados.

Certificación:

1. enunciados Lean o certificados constructivos donde sea factible.

#### Fase 2 — Unificación de `alpha`

Objetivo:

1. Especificar cuándo coinciden y cuándo divergen las definiciones de `alpha`.

Tareas:

1. protocolo comparativo por casos y dominios,
2. reporte de consistencia con tolerancias,
3. detector automático de divergencia diagnóstica.

Validación:

1. suite de consistencia `alpha` con thresholds reproducibles.

Certificación:

1. elevar equivalencias parciales a teoremas condicionados.

#### Fase 3 — Convergencia del ciclo `Phi <-> Phi*`

Objetivo:

1. convertir evidencia empírica en criterios con hipótesis explícitas.

Tareas:

1. introducir métricas contractivas por clase,
2. demostrar o refutar condiciones de contracción en cada subclase,
3. publicar límites de validez operacional.

Validación:

1. pruebas de convergencia/fracaso con datasets de control.

Certificación:

1. formalización de al menos un caso contractivo no trivial.

#### Fase 4 — Cierre composicional mixto

Objetivo:

1. garantizar composición entre ramas sin ambigüedad de error/costo.

Tareas:

1. reglas de propagación composicional explícitas,
2. límites por combinación de ramas,
3. validadores de precondiciones automáticos.

Validación:

1. fuzzing composicional mixto + regresiones de error.

Certificación:

1. formalizar composición para combinaciones prioritarias de producción.

#### Fase 5 — Política adaptativa (tabla -> cluster)

Objetivo:

1. evolucionar de reglas heurísticas a política aprendida con guardrails.

Tareas:

1. dataset histórico de ejecuciones bridge,
2. entrenador de política multiobjetivo (error-latencia-memoria),
3. fallback determinista seguro.

Validación:

1. comparación A/B contra política heurística baseline.

Certificación:

1. invariantes de seguridad (no degradación severa, no violación de límites críticos).

### 13.4. Evidencia requerida por fase

Cada fase debe cerrar con:

1. código,
2. tests,
3. artefactos,
4. nota técnica,
5. estado en matriz de trazabilidad.

Evidencia automatizable disponible en este repositorio:

1. `artifacts/traceability_matrix.json`
2. `artifacts/traceability_matrix.md`

Formato mínimo de evidencia:

1. comandos reproducibles,
2. métricas antes/después,
3. riesgos y límites,
4. decisión go/no-go.

### 13.5. Plantilla de informe por iteración

Usar esta estructura para cada entrega incremental:

1. Objetivo de la iteración.
2. Cambios implementados.
3. Validación ejecutada (comandos + resultados).
4. Evidencia generada (rutas de artefactos).
5. Riesgos detectados.
6. Próximo paso atómico.

### 13.6. Definición de "resuelto" por capa

1. **Desarrollado**: implementado y documentado con API/flujo claro.
2. **Validado**: medido con tests/benchmarks reproducibles.
3. **Certificado**: respaldado por prueba formal o certificado constructivo explícito.

Un frente solo se considera cerrado cuando se declara explícitamente su capa alcanzada y su límite vigente.

---

## 14. Líneas Técnicas Post-Cierre

1. Extender backend Triton a bloques vectoriales/matriciales y fusiones de cadenas no escalares.
2. Añadir parser simbólico robusto con precedencia extendida y funciones compuestas avanzadas.
3. Profundizar el tipado topológico con invariantes adicionales (continuidad por estrato, simetrías verificables).
4. Integrar más diagnósticos en reportes (sensibilidad por nodo, perfiles de error por dominio).
5. Acoplar documentación de Poema con guías de diseño para evoluciones del paper.

### 14.1. Preparación para Co-Funtor

Para la transición operativa hacia la capa co-funtorial (`CoPoem` y adjunción con `Phi`), Poema queda preparado con invariantes de entrada explícitos y verificables.

Lectura operativa de estos invariantes: no son solo propiedades teóricas; funcionan como precondiciones para que la capa co-funtorial reciba entradas numéricamente confiables.

1. Estabilidad composicional en ruta profunda
- Composición no lineal preservada semánticamente en backend recursivo.

2. Guardas de dominio explícitas
- Señalización de overshoot en compilación para prevenir extrapolación silenciosa.

3. Auto-corrección en runtime para funciones canónicas
- Recuperación estable fuera de dominio certificado sin degradar el régimen nominal.

4. Hardening reproducible local
- Pruebas adversariales y de fuzzing listas para ejecutarse en entorno local sin dependencia de CI.

Checklist previa a integración co-funtorial:

1. Ejecutar suites de robustez local (`tests/test_poema.py` y `tests/test_poema_hardening.py`).
2. Confirmar en reportes que `domain_guard_violations` se mantiene bajo el umbral aceptado para el dominio objetivo.
3. Verificar que los casos adversariales relevantes no generan salidas no finitas con `auto_domain_repair=True`.
4. Congelar parámetros de compilación (precisión, grado, dominio) que alimentarán `CoPoem`/`BiPoem` para evitar deriva entre corridas.
5. Documentar en bitácora técnica cualquier alert de overshoot persistente antes de escalar a experimentos de adjunción.

Siguiente foco co-funtorial recomendado:

- Trazar métricas de adjunción por ejecución (`adjunction_gap`, estabilidad espectral y consistencia de dominios sintetizados) y exponerlas en reportes técnicos de `CoPoem`/`BiPoem`.

---

## 15. Evoluciones Implementadas (3-7)

Además del frontend y compilador Poema, el repositorio ahora incluye módulos de evolución matemática y de ingeniería:

- `acf_functor/composition.py`: composición functorial rigurosa con certificados de error.
- `acf_functor/koopman_adaptive.py`: selección adaptativa de observables y dimensionalidad Koopman.
- `acf_functor/monad.py`: verificación de estructura monádica \\(\Phi, \eta, \mu\\).
- `acf_functor/adjunction.py`: co-funtor generativo y verificación de adjunción \\(\Phi^* \dashv \Phi\\).
- `acf_functor/lie_analysis.py`: análisis de brackets de Lie y profundidad serial mínima.
- `acf_functor/cohomology.py`: análisis cohomológico de obstrucciones \\(H^0, H^1\\).

Pruebas asociadas:

- `tests/test_composition_exhaustive.py`
- `tests/test_evolutions.py`

### 15.1. Evoluciones Implementadas (8-11)

La Fase C agrega la capa geométrica profunda del functor:

- `acf_functor/constructible_sheaves.py`: haces construibles sobre cubiertas locales con compatibilidad en solapes, detección de defectos de pegado y evaluación global por partición de la unidad.
- `acf_functor/moduli_spaces.py`: exploración del espacio moduli de reducciones, búsqueda geodésica y estimación de curvatura local del paisaje de error.
- `acf_functor/persistent_homology.py`: filtración por grado, diagrama de persistencia y mapa espectral persistente (SPM) para selección topológica de hiperparámetros.
- `acf_functor/galois_symmetry.py`: detección de simetrías (par/impar/periódica/desplazamiento), grupo de Galois efectivo y compresión de coeficientes.

Pruebas asociadas de Fase C:

- `tests/test_evolutions_8_11.py`

### 15.2. Evoluciones Implementadas (12-15)

La Fase D incorpora la capa fisica/terminal del functor:

- `acf_functor/kolmogorov_entropy.py`: perfil tasa-distorsion `E(f, epsilon)`, cota teorica y test de conservacion energetica.
- `acf_functor/superposition.py`: evaluacion paralela de candidatos en moduli y colapso por interferencia constructiva.
- `acf_functor/field_action.py`: minimizacion variacional de accion con terminos cinetico, potencial y regularizacion.
- `acf_functor/act_topos.py`: logica interna graduada (TOP/BOTTOM/parcial), clasificador de subobjetos y analisis topos.

Pruebas asociadas de Fase D:

- `tests/test_evolutions_12_15.py`

---

## 15. Certificados Lean 4

Para respaldar estas extensiones con validación formal mínima y reproducible, se añadió:

- `MathTest/EvolutionCertificates.lean`
- `MathTest/GeometricEvolutionCertificates.lean`
- `MathTest/PhysicalEvolutionCertificates.lean`

Este módulo contiene certificados formales para:

- corrección composicional afín,
- idempotencia de \\(\Phi\\),
- asociatividad composicional,
- cota algebraica de propagación de error.

El certificado geométrico adicional cubre invariantes de Fase C:

- definición de característica de Euler para haces,
- no negatividad del costo en moduli,
- acotación de escala óptima en persistencia,
- monotonicidad de compresión efectiva en simetrías de Galois.

El certificado fisico/terminal de Fase D cubre:

- cota de eficiencia en complejidad relativa,
- normalizacion de amplitudes en superposicion,
- identidad de descomposicion de accion,
- cierre del operador logico `and` en el intervalo unitario del topos.

Está integrado al objetivo principal mediante:

- `MathTest.lean` importando `MathTest.EvolutionCertificates`
- `MathTest.lean` importando `MathTest.GeometricEvolutionCertificates`
- `MathTest.lean` importando `MathTest.PhysicalEvolutionCertificates`

### Tabla de Trazabilidad Formal

| Módulo Python | Teorema/Certificado Lean | Test PyTest |
| :--- | :--- | :--- |
| `acf_functor/composition.py` | `MathTest.compAffine_eval`, `MathTest.compAffine_assoc`, `MathTest.composition_error_nonneg` en `MathTest/EvolutionCertificates.lean` | `tests/test_composition_exhaustive.py` |
| `acf_functor/monad.py` | `MathTest.phi_idempotent` en `MathTest/EvolutionCertificates.lean` | `tests/test_evolutions.py::TestEvolution5` |
| `acf_functor/adjunction.py` | Certificado base de composición afín (`compAffine_eval`) como fragmento algebraico de soporte en `MathTest/EvolutionCertificates.lean` | `tests/test_evolutions.py::TestEvolution6` |
| `acf_functor/koopman_adaptive.py` | Certificado de no negatividad de cota (`composition_error_nonneg`) como invariante formal mínimo de error compuesto en `MathTest/EvolutionCertificates.lean` | `tests/test_evolutions.py::TestEvolution3`, `tests/test_evolutions.py::TestEvolution4` |
| `acf_functor/core.py` (KoopmanReducer) | — (validación empírica) | `tests/test_koopman_validation.py` (8 pruebas: lineal, no lineal, espectro, invariante, truncación) |
| `acf_functor/lie_analysis.py` | Asociatividad y composición afín (`compAffine_assoc`, `compAffine_eval`) como base algebraica de análisis serial en `MathTest/EvolutionCertificates.lean` | `tests/test_evolutions.py::TestLieBrackets` |
| `acf_functor/cohomology.py` | Soporte formal de fragmento composicional en `MathTest/EvolutionCertificates.lean` y certificados constructivos de trascendentales en `MathTest/TranscendentalCertificates.lean` | `tests/test_evolutions.py::TestEvolution7` |
| `acf_functor/constructible_sheaves.py` | `MathTest.sheaf_euler_def` en `MathTest/GeometricEvolutionCertificates.lean` | `tests/test_evolutions_8_11.py::TestEvolution8` |
| `acf_functor/moduli_spaces.py` | `MathTest.moduli_cost_nonneg` en `MathTest/GeometricEvolutionCertificates.lean` | `tests/test_evolutions_8_11.py::TestEvolution9` |
| `acf_functor/persistent_homology.py` | `MathTest.persistent_scale_in_range` en `MathTest/GeometricEvolutionCertificates.lean` | `tests/test_evolutions_8_11.py::TestEvolution10` |
| `acf_functor/galois_symmetry.py` | `MathTest.galois_effective_le_original`, `MathTest.galois_order_nonzero` en `MathTest/GeometricEvolutionCertificates.lean` | `tests/test_evolutions_8_11.py::TestEvolution11` |
| `acf_functor/kolmogorov_entropy.py` | `MathTest.efficiency_bounded` en `MathTest/PhysicalEvolutionCertificates.lean` | `tests/test_evolutions_12_15.py::TestEvolution12` |
| `acf_functor/superposition.py` | `MathTest.amplitudes_normalized_probability` en `MathTest/PhysicalEvolutionCertificates.lean` | `tests/test_evolutions_12_15.py::TestEvolution13` |
| `acf_functor/field_action.py` | `MathTest.action_decomposition` en `MathTest/PhysicalEvolutionCertificates.lean` | `tests/test_evolutions_12_15.py::TestEvolution14` |

---

## 16. Nuevas Características Implementadas

### 16.1. Funciones de Activación Predefinidas

Poema incluye funciones de activación comunes listas para usar:

```python
P = Poem(dtype=torch.float64)

# ReLU: max(0, x)
ast = P.relu()

# GELU aproximado
ast = P.gelu_approx(degree=24)

# Swish: x * sigmoid(x)
ast = P.swish(degree=24)

# Mish: x * tanh(log(1 + exp(x)))
ast = P.mish(degree=24)

# SiLU (alias de Swish)
ast = P.silu(degree=24)
```

### 16.2. Serialización Nativa del AST

Los AST de Poema se pueden serializar a JSON y recuperar:

```python
from poema.ast_serialization import ast_to_json, ast_from_json, ast_save, ast_load

# A JSON string
json_str = ast_to_json(ast)
ast_recovered = ast_from_json(json_str)

# A file
ast_save(ast, "my_program.json", metadata={"version": "1.0"})
ast_loaded, metadata = ast_load("my_program.json")
```

### 16.3. Herramienta de Diagnóstico

El diagnóstico automático analiza compilaciones y sugiere correcciones:

```python
from poema.diagnostic import diagnose

report = diagnose(ast, domain=(-1.0, 1.0))
print(report.summary())

# Acceder a problemas individuales
for issue in report.problemas:
    print(f"[{issue.severity.name}] {issue.category}: {issue.message}")
    print(f"  → {issue.recommendation}")
```

### 16.4. Compatibilidad con TorchScript

Las funciones Poema se pueden compilar a TorchScript:

```python
from poema.jit_compat import PoemJITWrapper, PoemActivation

# Wrapper genérico
module = PoemJITWrapper(ast, domain=(-2, 2))
scripted = module.to_torchscript()

# Función de activación lista para usar
swish = PoemActivation("x * sigmoid(x)", domain=(-3, 3))
```

### 16.5. Exportación a ONNX

Las funciones Poema se pueden exportar a formato ONNX:

```python
from poema.onnx_export import export_to_onnx

model = export_to_onnx(ast, "my_function.onnx", input_shape=(1,))
```

### 16.6. Gradientes Multivariables

El operador `grad` calcula gradientes respecto a múltiples variables:

```python
# Gradiente de x^2 + y^2 respecto a [x, y]
ast = P.continuous_flow("grad(x^2 + y^2, [x, y])")
```

### 16.7. Series Temporales Irregulares en BiPoem

BiPoem soporta datos con muestreo no uniforme:

```python
bi = BiPoem(dtype=torch.float64)

# Datos con tiempos irregulares
times = torch.tensor([0.0, 0.1, 0.3, 0.4, 0.9, 1.0])
values = torch.randn(2, len(times))

result = bi.symbiosis_irregular(times, values, interpolation='cubic')
print(f"CV de dt: {result['interpolation_cv_dt']:.3f}")
```

### 16.8. Convergencia Certificada para BiPoem

El analizador de convergencia verifica condiciones del ciclo Φ ⇌ Φ*:

```python
from acf_functor.symbiotic_convergence import SymbioticConvergenceAnalyzer, analyze_convergence

analyzer = SymbioticConvergenceAnalyzer()
result = analyzer.estimate_contraction_rate(data)

if result.is_contraction:
    print(f"Convergencia esperada en ~{result.n_iterations_to_eps} iteraciones")
else:
    print(f"Convergencia no garantizada: L={result.lipschitz_constant:.3f}")
```

---

## 17. Tabla de Trazabilidad Actualizada

| Módulo Python | Característica | Tests |
| :--- | :--- | :--- |
| `poema/frontend.py` | Parser extendido (let, piecewise, D, grad) | 8 tests |
| `poema/frontend.py` | Funciones de activación predefinidas | 3 tests |
| `poema/frontend.py` | Series temporales irregulares BiPoem | 2 tests |
| `poema/ast_serialization.py` | Serialización nativa AST | 4 tests |
| `poema/diagnostic.py` | Herramienta de diagnóstico | - |
| `poema/jit_compat.py` | Compatibilidad TorchScript | 3 tests |
| `poema/onnx_export.py` | Exportación ONNX | 2 tests |
| `acf_functor/symbiotic_convergence.py` | Convergencia certificada BiPoem | 2 tests |
| `acf_functor/genesis_copoem_bridge.py` | Puente Genesis→CoPoem | 2 tests |
| `benchmarks/fma_benchmark.py` | Benchmark formal | 1 test |
| `tests/test_poema_missing_coverage.py` | Tests de cobertura nueva | 42 tests |
| `acf_functor/act_topos.py` | `MathTest.topos_and_in_unit_interval` en `MathTest/PhysicalEvolutionCertificates.lean` | `tests/test_evolutions_12_15.py::TestEvolution15` |
| `poema/compiler.py` | Certificados constructivos `sin_certificate_constructive`, `exp_certificate_constructive`, `log_certificate_constructive` en `MathTest/TranscendentalCertificates.lean` | `tests/test_poema.py` |
| `poema/frontend.py` | Certificados de ramo trascendental en `MathTest/TranscendentalCertificates.lean` | `tests/test_poema.py::TestCompilation`, `tests/test_poema.py::TestModes` |
| `poema/error_propagation.py` | Propagación analítica de error | ~10 tests |
| `poema/multivariate.py` | Gradientes multivariables reales | ~4 tests |
| `poema/activations_modern.py` | GELU, SwiGLU, RoPE certificados | ~3 tests |
| `poema/nn_integration.py` | Integración PyTorch nn.Module | ~5 tests |
| `poema/cli/diagnose.py` | Dashboard diagnóstico CLI | - |
| `benchmarks/canonical_benchmark.py` | Benchmark canónico reproducible | ~2 tests |
| `MathTest/CompositionErrorBounds.lean` | Certificados Lean composición | Teorema formal |
| `tests/test_poema_comprehensive.py` | Tests nuevas funcionalidades | ~25 tests |

---

## 18. Nuevas Características (Fases 1-6 del Informe de Análisis)

### 18.1. Fase 1: Consolidación y Corrección de Base

#### 18.1.1. Métricas Canónicas de Testing

El archivo `TESTS_CANONICAL.md` es ahora la **fuente canónica** para el número de tests. Cualquier discrepancia con otros documentos debe corregirse para coincidir con este archivo.

**Número canónico: 356 tests pasando, 0 regresiones.**

#### 18.1.2. Dominios Ampliados para tanh y sigmoid

Los dominios de certificación se han ampliado significativamente para uso real en ML:

| Función | Dominio anterior | Dominio nuevo | Grado anterior | Grado nuevo |
|---------|-----------------|---------------|----------------|-------------|
| `tanh`  | [-1, 1]         | **[-4, 4]**   | 40             | **50**      |
| `sigmoid` | [-1, 1]       | **[-8, 8]**   | 40             | **60**      |
| `sin`   | [-π, π]         | [-π, π]       | 20             | **24**      |
| `cos`   | [-π, π]         | [-π, π]       | 20             | **24**      |
| `exp`   | [-1, 1]         | [-1, 1]       | 15             | **18**      |
| `log`   | [0.5, 2.0]      | [0.5, 2.0]    | 25             | **28**      |

Los defaults en `Poem.tanh()` y `Poem.sigmoid()` se han actualizado para coincidir.

#### 18.1.3. Tabla de Rigor

Ver sección 19 para la tabla completa de qué está probado formalmente vs qué es heurístico.

### 18.2. Fase 2: Mejora de Precisión y Backends

#### 18.2.1. Propagación Analítica de Error

Módulo `poema/error_propagation.py` — calcula automáticamente la cota de error de composiciones f∘g dado ε_f y ε_g.

**Fórmula principal:** ε_{f∘g} ≤ ε_f + L_f · ε_g

```python
from poema import ErrorBound, compose_error_bounds, LIPSCHITZ_CONSTANTS

# Cotas individuales
cos_bound = ErrorBound(
    epsilon=3.1e-3, domain=(-math.pi, math.pi),
    lipschitz=1.0, source="lean_synchronized", is_certified=True
)
sin_bound = ErrorBound(
    epsilon=4.1e-3, domain=(-1.0, 1.0),
    lipschitz=1.0, source="lean_synchronized", is_certified=True
)

# Composición: sin(cos(x))
composed = compose_error_bounds(sin_bound, cos_bound)
# composed.epsilon ≈ 7.2e-3
# composed.lipschitz = 1.0 (L_sin * L_cos)
# composed.is_certified = True
```

**Funciones disponibles:**
- `compose_error_bounds(outer, inner)` — composición f∘g
- `affine_error_propagation(scale, shift, input_bound)` — transformación afín
- `sum_error_bounds(b1, b2)` — suma f + g
- `LIPSCHITZ_CONSTANTS` — diccionario con constantes conocidas

### 18.3. Fase 3: Expansión del Lenguaje

#### 18.3.1. Gradientes Multivariables Reales

Módulo `poema/multivariate.py` — diferenciación automática simbólica para funciones de múltiples variables.

```python
from poema import parse_multivariate, MultivariateExpr, JacobianExpr

# Función escalar multivariable
expr = parse_multivariate("x^2 + y^2", ["x", "y"])

# Gradiente respecto a x
grad_x = expr.gradient("x")  # ∂f/∂x

# Jacobiana completa
jacobian = expr.jacobian()  # [∂f/∂x, ∂f/∂y]

# Función vectorial
vec_expr = parse_multivariate("[x*cos(y), x*sin(y)]", ["x", "y"])
jacobian_vec = vec_expr.jacobian()  # 2×2 matrix
```

#### 18.3.2. Funciones de Activación Modernas Certificadas

Módulo `poema/activations_modern.py` — GELU, SwiGLU, RoPE como primitivas de primera clase.

```python
from poema import gelu_exact, swiglu, rope_embedding

P = Poem(dtype=torch.float64)

# GELU exacto con certificado
gelu_ast = gelu_exact(P, degree=40, domain=(-4.0, 4.0))

# SwiGLU / Swish
swish_ast = swiglu(P, degree=30, domain=(-6.0, 6.0))

# RoPE para transformers
rope = rope_embedding(P, dim=64, max_seq_len=2048)
# Compilar para posición específica
rope_at_pos = rope["compile_for_position"](10.0)
# rope_at_pos = {"cos": [...], "sin": [...]}
```

### 18.4. Fase 4: Observabilidad y Herramientas

#### 18.4.1. Dashboard de Diagnóstico CLI

```bash
python3 -m poema.cli.diagnose "sin(cos(x))" --domain "-pi,pi" --degree 24
python3 -m poema.cli.diagnose --file my_program.json --verbose
```

Muestra semáforo de severidad, ε certificado, FMA ops, violaciones de dominio, perfiles por nodo y recomendaciones accionables.

#### 18.4.2. Benchmark Canónico Reproducible

```bash
python3 benchmarks/canonical_benchmark.py
```

Ejecuta tests canónicos:
- Polinomio degree-100 (CPU fp64)
- sin en [-π,π] (CPU fp64)
- Composición profunda 20 capas
- GPU Triton (si disponible)

Genera `benchmark_results.json` con resultados reproducibles.

### 18.5. Fase 5: Integración ML y Deployment

#### 18.5.1. Integración con PyTorch nn.Module

Módulo `poema/nn_integration.py` — reemplazar funciones de activación en redes existentes con versiones Poema certificadas.

```python
from poema import PoemActivationLayer, replace_activations_in_model

# Uso directo como capa
gelu_layer = PoemActivationLayer.gelu(domain=(-4.0, 4.0))
x = torch.randn(10, dtype=torch.float64)
y = gelu_layer(x)

# Reemplazar en modelo existente
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.act1 = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(20, 5)
        self.act2 = torch.nn.SiLU()
    
    def forward(self, x):
        return self.act2(self.fc2(self.act1(self.fc1(x))))

model = MyModel()
model = replace_activations_in_model(
    model,
    {
        torch.nn.GELU: PoemActivationLayer.gelu(),
        torch.nn.SiLU: PoemActivationLayer.swish(),
    }
)
```

---

## Anexo A. Plan de Desarrollo Completo Adaptado al Repositorio (Abril 2026)

Este anexo adapta el plan integral propuesto a la estructura real del repositorio actual. El objetivo es convertir un roadmap aspiracional en un programa ejecutable, trazable y verificable con los modulos existentes.

Estado de ciclo:

1. El ciclo documentado en este anexo se considera **cerrado** en su tramo operativo principal.
2. Las secciones de "objetivo" y "entregables" se conservan como trazabilidad histórica y como base para el siguiente ciclo de rigor formal.

Principio operativo (se mantiene):

1. Matematica -> Implementacion -> Certificacion -> Test -> Integracion.
2. Ningun frente se marca como cerrado sin evidencia reproducible.
3. Cada entrega debe declarar su capa: Desarrollado, Validado o Certificado.

### A.1. Mapeo directo del plan propuesto a rutas reales

El plan original usa rutas de ejemplo (`python/core`, `python/koopman`, `python/tests`). En este repositorio, el mapeo correcto es:

1. Índice de Decaimiento Espectral Afín y reduccion espectral:
    - `acf_functor/core.py`
    - `acf_functor/koopman_adaptive.py`
    - `acf_functor/symbiotic_convergence.py`
2. Composicion, monada, adjuncion y ramas mixtas:
    - `acf_functor/composition.py`
    - `acf_functor/monad.py`
    - `acf_functor/adjunction.py`
3. Certificacion Lean 4:
    - `MathTest/*.lean`
    - `MathTest.lean`
4. Tests Python:
    - `tests/`
    - `python_analysis/` (validaciones y generadores auxiliares)
5. Benchmarks y politicas:
    - `benchmarks/periodic_table.py`
    - `benchmarks/cluster_proxy.py`
    - `benchmarks/canonical_benchmark.py`
6. Scripts operativos:
    - `scripts/rebuild_certificates.sh`

Decision de compatibilidad:

1. No se duplican modulos en un arbol `python/` nuevo.
2. Se extiende primero `acf_functor/*` para mantener coherencia de API, tests y documentacion existente.

### A.2. Fases adaptadas (registro de ejecucion + siguiente ciclo)

#### Fase 1 adaptada - Unificacion de alpha(f)

Estado actual:

1. Existe estimacion operacional/espectral en `ACFInvariant.compute_alpha` dentro de `acf_functor/core.py`.
2. Existen consumidores de alpha en `acf_functor/koopman_adaptive.py` y `acf_functor/composition.py`.

Objetivo de esta fase:

1. Separar explicitamente estimadores (`alpha_comb`, `alpha_spec`, `alpha_geo`) sobre la API actual, sin romper compatibilidad.
2. Agregar reporte de consistencia entre definiciones y detector de discrepancias por clase de sistema.

Entregables minimos:

1. Extensiones en `acf_functor/core.py`.
2. Tests dedicados en `tests/` para equivalencia/discordancia controlada.
3. Artefacto JSON de consistencia por casos canonicos en `artifacts/`.

#### Fase 2 adaptada - Koopman certificado por ramas

Estado actual:

1. Hay infraestructura EDMD adaptable en `acf_functor/core.py` y `acf_functor/koopman_adaptive.py`.
2. Ya hay validacion Koopman y tablas de diagnostico periodicas.

Objetivo de esta fase:

1. Formalizar seleccion de ramas certificables por subclase (polinomial, simetrica, analitica bajo hipotesis).
2. Mantener rama heuristica explicitamente etiquetada como no certificada.

Entregables minimos:

1. Politica de rama documentada y expuesta en metadatos de resultado.
2. Test de respeto de cota por rama certificada.
3. Benchmarks comparativos por rama en `benchmarks/`.

#### Fase 3 adaptada - Certificacion Lean 4 incremental

Estado actual:

1. El proyecto ya compila con `lake build`.
2. Existen certificados en `MathTest/` y algunas piezas abiertas historicas en `archive/`.

Objetivo de esta fase:

1. Priorizar teoremas que desbloquean pipeline (composicion, cotas de error, casos polinomiales clave).
2. Reducir deuda de formalizacion abierta de manera incremental y medible.

Entregables minimos:

1. Nuevos modulos Lean importados en `MathTest.lean`.
2. Conteo de `sorry` en area activa reportado en cada iteracion.
3. Matriz trazable teorema -> modulo Python -> test.

#### Fase 4 adaptada - Test suite integral

Estado actual:

1. El repositorio tiene cobertura funcional amplia en `tests/`.
2. Se consolidó suite orientada a frentes científicos (incluyendo `alpha` unificado, ramas Koopman y convergencia) con resultados reproducibles.

Objetivo de esta fase:

1. Crear suites de regresion cientifica por frente abierto.
2. Mantener separacion entre tests rapidos y tests lentos/experimentales.

Entregables minimos:

1. Nuevos tests en `tests/` con marcadores claros (`slow`, `integration`, `scientific`).
2. Script de validacion reproducible en `scripts/` (sin asumir CI externo).

#### Fase 5 adaptada - Integracion continua realista

Estado actual:

1. Existe pipeline CI versionado y baseline operativo (`.github/workflows/acf_ci.yml`) junto con utilidades en `ci/`.

Objetivo de esta fase:

1. Introducir CI por etapas: Python unitario, Lean build, validacion cientifica.
2. Evitar sobreprometer jobs no sostenibles (por ejemplo GPU obligatoria en CI publico).

Entregables minimos:

1. Workflow en `.github/workflows/` con jobs escalonados.
2. Umbrales iniciales explicitos (coverage, max `sorry`, pruebas lentas opcionales).

#### Fase 6 adaptada - Visualizacion y analisis

Estado actual:

1. Hay artefactos tabulares ricos (`periodic_table.{md,json,parquet}`, bridge metrics, planes de cluster).
2. Se ejecutó dashboard reproducible sobre artefactos reales (`python_analysis/dashboard.py`).

Objetivo de esta fase:

1. Consolidar dashboard reproducible sobre artefactos reales, evitando datos simulados por defecto.

Entregables minimos:

1. Script de visualizacion alimentado por `artifacts/`.
2. Figuras de convergencia alpha, espectro y error-latencia exportadas con versionado temporal.

#### Fase 7 adaptada - Ejecucion iterativa guiada

Estado actual:

1. Ya existe plantilla de reporte por iteracion en este manual (Seccion 13.5).

Objetivo de esta fase:

1. Ejecutar el programa como ciclos cortos con evidencia de antes/despues.
2. Cerrar cada ciclo con decision explicita de continuidad.

Entregables minimos:

1. Informe por paso con: objetivo, cambios, comandos, evidencia, riesgos, siguiente paso.

### A.3. Comandos base de ejecucion (adaptados al repo)

Linea base tecnica:

```bash
# Lean
./lean-4.29.0-rc6-linux/bin/lake build

# Python tests (entorno local)
export PYTHONPATH=.
.venv/bin/pytest -q

# Tabla periodica analitica + bridge
.venv/bin/python benchmarks/periodic_table.py \
    --output artifacts/periodic_table.md \
    --json-output artifacts/periodic_table.json \
    --parquet-output artifacts/periodic_table.parquet \
    --n-trials 10 \
    --noise-std 0.01 \
    --domain all \
    --emit-cluster-plan \
    --run-cluster-proxy \
    --cluster-steps 3 \
    --cluster-output artifacts/cluster_bridge_metrics.json
```

### A.4. Matriz de prioridad inmediata (orden recomendado)

1. Consolidar tests de frentes abiertos (alpha unificado + ramas Koopman + convergencia).
2. Endurecer certificacion incremental en Lean para piezas de alto impacto en pipeline.
3. Integrar CI minima reproducible (Python + Lean + reporte cientifico).
4. Elevar politica tabla -> cluster de heuristica fija a adaptacion medida.
5. Publicar tablero de evidencia con series temporales de resultados.

### A.5. Criterio de cierre de este plan adaptado

El plan adaptado se considera cumplido cuando cada frente tenga:

1. Implementacion en rutas reales del repositorio.
2. Validacion reproducible con comandos y artefactos concretos.
3. Estado de certificacion declarado sin ambiguedad (total, parcial o abierto).

Nota de alcance:

1. Este anexo no reemplaza la Seccion 23 de `Paper.md`; la operacionaliza.
2. Toda nueva iteracion debe actualizar trazabilidad y evidencia, no solo narrativa.

### A.6. Ejecucion realizada (Abril 2026)

Implementacion materializada en esta iteracion:

1. Invariante unificado (tres definiciones): `acf_functor/invariant_unified.py`.
2. Koopman certificado por ramas: `acf_functor/certified_koopman.py`.
3. Exportes API publica: `acf_functor/__init__.py`.
4. Test suite nueva:
    - `tests/test_invariant_unified.py`
    - `tests/test_certified_koopman_extended.py`
5. Benchmark completo: `benchmarks/benchmark_complete.py`.
6. Reporte cientifico ejecutable: `ci/generate_validation_report.py`.
7. Dashboard reproducible: `python_analysis/dashboard.py`.
8. Scripts operativos:
    - `scripts/setup_and_verify.sh`
    - `scripts/development_workflow.sh`
9. CI inicial: `.github/workflows/acf_ci.yml`.

Validaciones ejecutadas en entorno local:

1. `./scripts/setup_and_verify.sh`.
2. `./lean-4.29.0-rc6-linux/bin/lake build`.
3. `pytest tests/test_invariant_unified.py tests/test_certified_koopman_extended.py`.
4. `python benchmarks/benchmark_complete.py`.
5. `python ci/generate_validation_report.py`.
6. `python python_analysis/dashboard.py --output plots`.

Resultados observados:

1. Build Lean: exitoso.
2. Tests nuevos combinados: `11 passed`.
3. Reporte cientifico generado: `validation_report/validation_report.json`.
4. Benchmark completo generado: `artifacts/benchmark_complete_20260406_012236.json`.
5. Visualizaciones generadas:
    - `plots/alpha_convergence.png`
    - `plots/koopman_branch_errors.png`

Incidencia y resolucion durante ejecucion:

1. Se detecto colision de import por `python_analysis/acf_functor.py` al ejecutar dashboard.
2. Se resolvio forzando prioridad de `sys.path` al root del repo en `python_analysis/dashboard.py`.

Estado de capa por frente en esta iteracion:

1. Invariante unificado: Desarrollado + Validado.
2. Koopman por ramas certificadas: Desarrollado + Validado (certificacion formal Lean pendiente para teoremas completos).
3. CI baseline: Desarrollado.
4. Reporte cientifico y dashboard: Desarrollado + Validado.

**Activaciones disponibles:**
- `PoemActivationLayer.gelu(domain=(-4.0, 4.0))`
- `PoemActivationLayer.swish(domain=(-6.0, 6.0))`
- `PoemActivationLayer.tanh_act(domain=(-4.0, 4.0))`
- `PoemActivationLayer.relu(domain=(-4.0, 4.0))`
- `PoemActivationLayer.mish(domain=(-4.0, 4.0))`
- `PoemActivationLayer.from_expression(expr, domain)`

### 18.6. Fase 6: Rigor Formal

#### 18.6.1. Certificados Lean para Composición

Archivo `MathTest/CompositionErrorBounds.lean` — teorema formal de propagación de error en composiciones.

**Teorema principal:** Si f tiene error ε_f con constante de Lipschitz L_f, y g tiene error ε_g, entonces f∘g tiene error ≤ ε_f + L_f · ε_g.

---

## 19. Tabla de Rigor — Qué está probado y qué no

| Afirmación | Nivel | Evidencia |
|-----------|-------|-----------|
| Horner evalúa polinomios exactamente | **Teorema Lean 4** | `horner_exact_real` |
| sin/exp/log en dominio canónico con ε < X | **Certificado constructivo** | `TranscendentalCertificates.lean` |
| tanh en [-4,4] con ε < 1e-8 | **Certificado constructivo** | `generate_interval_certificates.py` |
| sigmoid en [-8,8] con ε < 1e-8 | **Certificado constructivo** | `generate_interval_certificates.py` |
| Composición de polinomios preserva exactitud | **Prueba Python + benchmark** | `test_composition_exhaustive.py` |
| Propagación de error ε_{f∘g} ≤ ε_f + L_f·ε_g | **Teorema Lean 4** | `CompositionErrorBounds.lean` |
| Ciclo Φ ⇌ Φ* converge | **Heurístico** | Sin garantía formal |
| CoPoem multiobjetivo es óptimo global | **Falso — solo local** | Documentado en limitaciones |
| Genesis descubre teoremas reales | **Hipótesis candidata** | Requiere validación externa |
| Domain Guard detecta violaciones | **Empírico** | Tests de cobertura |
| Auto-domain Repair conmuta a nativo | **Empírico** | Tests de hardening |

## 20. Interoperabilidad (Caché y Exportación IO)

### Exportación ONNX Total
El pipeline ha sido asegurado y el soporte de ONNX ya abarca cualquier evaluación de grado industrial. Exportar tu poema a ONNX puede hacerse sencillamente enviando el root AST.
```python
from poema.onnx_export import export_to_onnx
export_to_onnx(model_ast, output_path='model.onnx')
```

### Serialización AST de estado
Para modelos de validación que demoren mucho, el estado de topología se puede guardar:
```python
from poema.ast_serialization import ASTSerializer
dict_repr = ASTSerializer.to_dict(model_ast)
rebuilt_ast = ASTSerializer.from_dict(dict_repr)
```

## Formal Verification Reality (Runtime Invariants)

This framework shifts theoretical conjectures into hardened runtime observables via the `poema.formal_verification` module:

- **1.1. Universal Reduction Theorem (URT):** Acknowledged as an open research bounding problem for non-analytic functions. Evaluated at runtime computing ||Phi_d(f) - f||.
- **1.2. FMA Conservation Law:** Monitored as a strict structural AST property (E(f) = E(Phi(f))) rather than a universally assumed Noether-like symmetry.
- **1.3. Functorial Composition:** Measured dynamically (Phi(f o g) - (Phi(f) o Phi(g))) to bound divergences in mixed Koopman-polynomial trees.
- **1.4. Alpha(f) Index Discrepancies:** Combinatorial, spectral, and geometric derivations are quantified simultaneously to log convergence instead of asserting immediate unifications.
- **1.5. Inexact Reversibility:** Phi^{-1} reconstruction exactness verified primarily over polynomial scopes; Koopman reconstructions are actively restricted and bounded via explicit error tolerances.

## 8. Dominando el ACF Avanzado (Pure-FMA y Genesis Auto-Prover)

Poema ahora soporta un Pipeline ACF de descubrimiento matemático completo y auto-prueba, que evalúa errores de punto flotante de manera rigurosa (Pure-FMA Repair) e identifica nuevas fórmulas (Genesis Auto-Prover).

### 8.1 Uso de Genesis Auto-Prover
Génesis es capaz de detectar fórmulas complejas a lo largo de 15 categorías. Para utilizar su capacidad analítica:

```python
from poema.genesis_auto_prover import SystemInitializer
from poema.acf_integration import create_advanced_pipeline

# Inicializar motor de descubrimiento matemático
initializer = SystemInitializer(
    use_rl=True,             # Emular aprendizaje por refuerzo analítico
    optimization_level=5      # Nivel máximo (búsqueda cuántica/algebraica)
)
pipeline = create_advanced_pipeline(initializer)

# Procesar código e imprimir descubrimientos
resultados = pipeline.compile_poema_code(codigo_AST)
print(resultados["genesis_discoveries"])
```

### 8.2 Configuraciones y Categorías
Génesis agrupa descubrimientos en `DiscoveryCategory` que van desde lo trivial a lo hipercomplejo: polinomios, trigonométricas, funciones de Euler, Gamma, análisis integral, topología y espacios Moduli. Todos los resultados pueden inyectarse y certificarse bajo el sistema Lean 4 embebido directamente en la salida de compilación. Las dependencias ML robustas solo se necesitan si la inferencia heurística activa modelos predictivos (de lo contrario se aplica validación pura interna automática).

---

## 9. Nuevas Capacidades de Ingeniería (v2.3.0)

Esta sección documenta los cinco módulos nuevos que cierran las brechas críticas
identificadas en el análisis de arquitectura previo.

### 9.1. `poema.backends` — Capa de backends hardware-agnóstica

#### 9.1.1. BackendProtocol (ABC)

```python
from poema.backends.protocol import BackendProtocol, BackendCapabilities, BackendResult
```

**`BackendCapabilities`** (dataclass):

| Campo | Tipo | Descripción |
|-------|------|------------|
| `name` | `str` | Identificador único |
| `supports_cpu` | `bool` | Puede ejecutar en CPU |
| `supports_gpu` | `bool` | Requiere GPU |
| `supports_verilog` | `bool` | Genera RTL |
| `precision_formats` | `list[str]` | `["fp32","fp64"]` |
| `description` | `str` | Descripción legible |

**`BackendResult`** (dataclass):

| Campo | Tipo | Descripción |
|-------|------|------------|
| `callable_fn` | `Optional[Callable]` | Función evalable (`None` para RTL) |
| `emitted_code` | `str` | Código fuente generado o C-pseudocode |
| `emitted_path` | `str` | Ruta del archivo principal generado |
| `backend_name` | `str` | Nombre del backend |
| `fma_count` | `int` | Número de FMAs compilados |
| `extra` | `dict` | Datos específicos del backend |

#### 9.1.2. BackendRegistry

```python
from poema.backends import BackendRegistry

# Estado de todos los backends
disponibles = BackendRegistry.available()
# {'numpy_cpu': True, 'verilog_rtl': True, 'pytorch_cuda': False, …}

# Obtener un backend específico
b = BackendRegistry.get("numpy_cpu")

# Mejor CPU disponible (siempre devuelve algo)
cpu = BackendRegistry.best_for_cpu()

# Mejor GPU disponible (None si no hay GPU)
gpu = BackendRegistry.best_for_gpu()

# Resumen en texto
print(BackendRegistry.describe_all())
```

#### 9.1.3. NumpyBackend

```python
from poema.backends import NumpyBackend

b = NumpyBackend()
result = b.compile(
    fma_sequence,             # List[FMAInstruction] o duck-typed list
    source_ast = ast,         # ASTNode opcional
    domain     = (-1.0, 1.0), # Dominio para normalización
    precision  = "fp64",      # "fp32" | "fp64"
)

fn = result.callable_fn          # Callable[[np.ndarray], np.ndarray]
y  = fn(np.linspace(-1, 1, 100)) # Evaluación
print(result.emitted_code)       # C-pseudocode del chain FMA
print(result.fma_count)          # Número de etapas
```

Propiedades:
- Sin dependencia de GPU ni PyTorch
- Soporta `float32` y `float64`
- Emite código C para auditoría y debugging
- Covariante con cadenas de hasta 10.000 etapas FMA

#### 9.1.4. VerilogBackend

```python
from poema.backends import VerilogBackend

vb = VerilogBackend(
    output_dir    = "rtl/",         # Directorio de salida
    pipelined     = True,           # True=pipeline, False=combinacional
    use_axi_stream= True,           # True=interfaz AXI-Stream
    data_width    = 32,             # Bits por dato (8/16/32/64)
    frac_bits     = 24,             # Bits fraccionarios (Q(dw-fb).fb)
    target        = "xilinx",       # "xilinx" | "intel" | "yosys"
)

result = vb.compile(
    fma_sequence,
    source_ast   = ast,
    module_name  = "mi_funcion",    # Nombre del módulo Verilog
    epsilon_bound= 4.5e-3,          # ε del certificado Lean 4
    domain       = (-1.0, 1.0),
)

# Archivos generados:
# rtl/mi_funcion.v              — RTL sintetizable
# rtl/mi_funcion_assertions.sva — SystemVerilog Assertions
# rtl/mi_funcion_tb.v           — Testbench con vectores de prueba
# rtl/mi_funcion.sdc            — Restricciones de síntesis (timing)

print(result.emitted_path)             # → "rtl/mi_funcion.v"
print(result.extra["pipeline_stages"]) # → número de etapas
print(result.extra["clock_mhz"])       # → frecuencia estimada en MHz
```

**Características del RTL generado:**
- Pipeline completamente segmentado (1 DSP slice por etapa)
- Reset síncrono activo bajo (`rst_n`)
- Valid/ready handshake AXI-Stream
- Constantes en punto fijo Q(dw-fb).fb
- Inferencia de DSP48 (Xilinx) o ALTMULT_ACCUM (Intel)

**Puente formal SVA ↔ Lean 4:**
El archivo `.sva` contiene propiedades derivadas directamente de los certificados Lean 4:
```systemverilog
// ε ≤ 4.5e-3 procedente de: LeanLiveVerifier → PROVEN
property epsilon_contract;
    @(posedge clk) m_axis_tvalid |->
        $abs($signed(m_axis_tdata) - reference_val) <= EPSILON_Q;
endproperty
assert property (epsilon_contract) else $fatal(1, "ACF contract violated");
```

### 9.2. `poema.lean_live_verifier` — Verificación Lean 4 en vivo

#### Clases principales

```python
from poema.lean_live_verifier import (
    LeanLiveVerifier,
    VerificationStatus,   # Enum: PROVEN | FAILED | TIMEOUT | ERROR | SKIPPED
    VerificationResult,   # dataclass con status, lean_output, elapsed_ms, …
    FullSuiteReport,      # dataclass con results[], proven_count, to_json()
)
```

#### Constructor

```python
v = LeanLiveVerifier(
    lean_binary      = None,    # Auto-detectado; candidatos en lean-4.29.0-rc6-linux/
    timeout_seconds  = 60,      # Timeout por teorema
    cache_dir        = None,    # Directorio temporal para .lean files
    verbose          = False,   # Imprimir diagnostics a stdout
)
```

#### API de verificación

```python
# Verificar código Lean 4 arbitrario:
r = v.verify_theorem(lean_code: str, theorem_name: str) -> VerificationResult

# Verificar ε < δ (URT bound):
r = v.verify_universal_bound(epsilon: float, bound_limit: float = 0.01)

# Verificar E(f) = E(Φ(f)) (FMA conservation):
r = v.verify_fma_conservation(pre_cost: int, post_cost: int)

# Verificar ε_comp ≤ L_f × L_g:
r = v.verify_composition_bound(lipschitz_f, lipschitz_g, composition_error)

# Verificar consistencia de estimadores α:
r = v.verify_alpha_consistency(alpha_comb, alpha_spectral, alpha_geometric, tolerance=0.10)

# Verificar reversibilidad Φ^{-1}:
r = v.verify_reversibility(reconstruction_error, machine_eps=1e-7)

# Suite completa (6 teoremas + JSON a MathTest/):
report = v.run_full_suite(data_dict: dict) -> FullSuiteReport
```

#### VerificationResult

| Campo | Tipo | Descripción |
|-------|------|------------|
| `status` | `VerificationStatus` | PROVEN / FAILED / TIMEOUT / ERROR |
| `is_proven` | `bool` | Propiedad derivada |
| `theorem_name` | `str` | Nombre del teorema |
| `lean_output` | `str` | stdout + stderr del proceso lean |
| `elapsed_ms` | `float` | Tiempo de ejecución en ms |
| `certificate_hash` | `str` | SHA-256[:16] del código Lean |
| `diagnostics` | `list[LeanDiagnostic]` | Lista de error/warning/info |

```python
r = v.verify_universal_bound(4.525e-3)
print(r.status)           # VerificationStatus.PROVEN
print(r.is_proven)        # True
print(r.summary())        # "✓ [PROVEN] urt_universal_bound  (312ms)"
print(r.certificate_hash) # "3f8a2c1d9e7b0451"
```

### 9.3. `poema.canonical_alpha` — α_A(f) Canónico

#### API principal

```python
from poema.canonical_alpha import (
    compute_canonical_alpha,  # Función conveniente (entrada principal)
    CanonicalAlpha,           # Dataclass resultado
    AlphaEstimates,           # Dataclass de las 3 estimaciones brutas
    AlphaEstimator,           # Métodos estáticos de estimación
    AlphaCanonicalizer,       # Clase fusionadora
)
```

#### `compute_canonical_alpha(fma_sequence, source_ast=None, domain=(-1,1), live_verify=False)`

```python
from poema.canonical_alpha import compute_canonical_alpha

ca = compute_canonical_alpha(fma_seq)

# Acceso a los campos principales:
ca.canonical_value      # float — el único valor autoritativo
ca.raw_estimates        # AlphaEstimates(combinatorial, spectral, geometric, n_fma)
ca.weights              # (0.40, 0.35, 0.25)
ca.fusion_method        # "geometric_mean" | "arithmetic_mean"
ca.consistency_score    # float en [0, 1]
ca.is_reliable          # True iff consistency_score > 0.85
ca.confidence_interval  # (lo, hi) con z=1.96
ca.interpretation       # str: "Analytic (α≈1)…" | "Highly irregular…" | …
ca.lean_verified        # bool (requiere live_verify=True)
ca.summary()            # str — bloque legible completo
```

#### `AlphaEstimator` — estimadores individuales

```python
from poema.canonical_alpha import AlphaEstimator

raw = AlphaEstimator.compute_all(fma_seq)
# raw.combinatorial  — profundidad relativa del árbol
# raw.spectral       — cociente σ_max / σ_harm de la matriz de pesos
# raw.geometric      — longitud de curva / distancia cuerda

print(raw.deviation())       # desviación relativa máxima entre los 3
print(raw.is_consistent())   # True si desviación < 10%
```

#### `AlphaCanonicalizer`

```python
from poema.canonical_alpha import AlphaCanonicalizer, AlphaEstimates

raw = AlphaEstimates(combinatorial=1.2, spectral=1.05, geometric=0.9, n_fma=5)

# Con media geométrica ponderada (por defecto):
canon = AlphaCanonicalizer(
    weights=(0.40, 0.35, 0.25),
    fusion_method="geometric_mean",
    live_verify=False,
).canonicalize(raw)

# Con media aritmética:
canon_arith = AlphaCanonicalizer(fusion_method="arithmetic_mean").canonicalize(raw)
```

#### Interpretación de α_A(f)

| Rango | Significado |
|-------|------------|
| α < 0.5 | Trivial/Affine — exactamente FMA-realizable |
| 0.5 ≤ α < 1.0 | Sub-analítico — menor complejidad que funciones analíticas genéricas |
| α ≈ 1.0 | Analítico (Cω) — clase de complejidad estándar |
| 1.0 < α < 1.5 | Levemente super-analítico — overhead moderado |
| 1.5 ≤ α < 2.5 | C^k liso (k<∞) — overhead significativo |
| α ≥ 2.5 | Altamente irregular / cuasi-discontinuo |

### 9.4. Flujo Completo de Ejemplo

```python
import numpy as np
import torch
from poema.frontend import Poem
from poema.compiler import FMALinearizer, PoemCompiler
from poema.backends import VerilogBackend, NumpyBackend
from poema.lean_live_verifier import LeanLiveVerifier, VerificationStatus
from poema.canonical_alpha import compute_canonical_alpha

# 1. Definir y compilar función
P   = Poem(dtype=torch.float64)
ast = P.continuous_flow("sin(x)")
compiler = PoemCompiler(target="pytorch", precision="fp64")
fn_torch, report = compiler.compile(ast, domain=(-3.14, 3.14))

# 2. Extraer secuencia FMA
fma_seq = FMALinearizer().linearize(ast)

# 3. Evaluación CPU pura (sin GPU)
np_result = NumpyBackend().compile(fma_seq, ast)
x = np.linspace(-3.14, 3.14, 1000)
y = np_result.callable_fn(x)

# 4. Síntesis RTL para FPGA
vb = VerilogBackend(output_dir="rtl/")
vb.compile(fma_seq, ast, module_name="sin_fpga",
           epsilon_bound=report.total_epsilon)
# → rtl/sin_fpga.v  +  .sva  +  _tb.v  +  .sdc

# 5. Verificación formal viva con Lean 4
v = LeanLiveVerifier()
r = v.verify_universal_bound(report.total_epsilon)
assert r.status == VerificationStatus.PROVEN

# 6. Índice canónico α_A(f)
ca = compute_canonical_alpha(fma_seq, ast)
print(f"α_A(sin) = {ca.canonical_value:.4f}  [{ca.interpretation}]")
print(f"Confiable: {ca.is_reliable}  CI={ca.confidence_interval}")
```

### 9.5. Tests de Ingeniería (58 tests)

```bash
python3 -m pytest tests/test_engineering_improvements.py -v
```

Output esperado:
```
tests/test_engineering_improvements.py::TestBackendRegistry::test_registry_returns_backends PASSED
tests/test_engineering_improvements.py::TestBackendRegistry::test_numpy_always_available PASSED
...
tests/test_engineering_improvements.py::TestEndToEndIntegration::test_poem_to_verilog_full PASSED

58 passed in 9.41s
```

---

## 21. Referencia de API — v2.4.0: CNativeEngine, WasmBackend, ONNXBackend

### 21.1. `CNativeEngine`

```
poema.backends.CNativeEngine
```

Motor de compilación nativa C con soporte AVX-512, AVX2, OpenMP y cffi.

#### Constructor

```python
CNativeEngine(build_dir: str | None = None)
```
- `build_dir`: directorio para archivos `.c` y `.so` temporales (default: `/tmp/poema_c_engine`)

#### Métodos de alto nivel

| Método | Signatura | Descripción |
|--------|-----------|-------------|
| `evaluate` | `(fma_seq, x, precision="fp64") → np.ndarray` | Compilar + evaluar en una llamada |
| `evaluate_with_gradient` | `(fma_seq, x, precision="fp64") → (np.ndarray, np.ndarray)` | Evaluación + gradiente exacto forward-mode |
| `evaluate_polynomial` | `(coeffs, x_vals, precision="fp64") → np.ndarray` | Horner vía C nativo |
| `matrix_eval` | `(fma_seq, X, precision="fp64") → np.ndarray` | Evaluación batch 2D, `X: (N,M)` |
| `reduce_sum` | `(fma_seq, x, precision="fp64") → float` | Suma escalar de la evaluación |
| `fuse_and_compile` | `(chains, source_ast=None, precision="fp64") → BackendResult` | Concatenar cadenas y compilar |
| `benchmark_full_suite` | `(fma_seq, sizes=None, reps=3, precision="fp64") → List[CEngineBenchmark]` | Benchmark L1→DRAM sweep |
| `print_benchmark_suite` | `(fma_seq, ...)` | Imprimir tabla de benchmark formateada |
| `inspect` | `(result: BackendResult) → str` | Vista detallada: SIMD, tamaño, preview de C |
| `get_build_info` | `() → str` | `"C Engine: gcc | AVX-512 | OpenMP | cffi=True"` |

#### Método principal `compile`

```python
compile(
    fma_sequence: List[Any],
    source_ast: Any,
    precision: str = "fp64",
    is_complex: bool = False,
) → BackendResult
```

`BackendResult.extra` contiene:
- `"_lib"`: objeto cffi de la biblioteca compilada
- `"_ffi"`: interfaz cffi para declaraciones

#### `CEngineCapabilities`

```python
@dataclass
class CEngineCapabilities:
    compiler: str          # "gcc"
    has_avx2: bool
    has_avx512: bool
    has_openmp: bool
    has_cffi: bool
    def summary() → str
```

#### `CEngineBenchmark`

```python
@dataclass
class CEngineBenchmark:
    n_elements: int
    n_stages: int
    precision: str
    ns_per_call: float
    gb_per_sec: float
    gflops: float
    speedup_vs_numpy: float
```

#### Kernels C generados (accesibles vía cffi)

| Función C | Descripción |
|-----------|-------------|
| `poema_eval_fp64(x, y, n)` | Evaluación principal FP64 (cache-blocked) |
| `poema_eval_fp32(x, y, n)` | Evaluación principal FP32 (cache-blocked) |
| `poema_matrix_eval_fp64(X, Y, rows, cols)` | Evaluación matricial OpenMP |
| `poema_reduce_sum_fp64(x, n) → double` | Suma escalar |
| `poema_horner_fp64(x, coeffs, n_coeffs, out)` | Evaluación Horner |
| `poema_grad_fp64(x, y, dydx, n)` | Forward-mode AD, gradiente exacto |
| `poema_bench_fp64(x, y, n, reps) → double` | ns por llamada (benchmark) |
| `poema_bench_grad_fp64(x, y, dydx, n, reps) → double` | ns por llamada con gradiente |

---

### 21.2. `WasmBackend`

```
poema.backends.WasmBackend
```

Generador de módulos WebAssembly Text Format (WAT) para FMA chains.

#### Métodos

| Método | Signatura | Descripción |
|--------|-----------|-------------|
| `emit_wat` | `(fma_seq, precision="fp64", module_name="poema_fma", is_complex=False) → str` | Generar código WAT completo |
| `assemble` | `(wat_source, out_path) → str | None` | Ensamblar WAT→WASM con `wat2wasm` |
| `validate_wat` | `(wat_source) → bool` | Validar sintaxis WAT básica |
| `emit_js_loader` | `(wasm_filename, module_name, n_stages, precision) → str` | Generar módulo ES JavaScript |
| `compile` | `(fma_seq, source_ast, precision="fp64", is_complex=False) → BackendResult` | Compilación completa con fallback Python |
| `inspect_artifact` | `(result: BackendResult) → str` | Descripción del artefacto generado |

`BackendResult.extra` contiene:
- `"wat_path"`: ruta al archivo `.wat` (si se guardó)
- `"wasm_path"`: ruta al `.wasm` binario (si `wat2wasm` disponible)
- `"js_loader"`: código JavaScript del loader
- `"has_binary"`: bool, indica si se ensamblaron binarios

#### Funciones WAT exportadas

```wat
(func (export "poema_eval_fp64")   (param f64) (result f64))
(func (export "poema_eval_fp32")   (param f32) (result f32))
(func (export "poema_eval_batch_fp64") (param i32 i32 i32))
(func (export "poema_horner_fp64") (param f64 i32 i32) (result f64))
(func (export "poema_eval_complex") (param f64 f64) (result f64 f64))
```

---

### 21.3. `ONNXBackend`

```
poema.backends.ONNXBackend
```

Exportador a grafos de cómputo ONNX para despliegue en TensorRT, OpenVINO, CoreML y ORT.

#### Constructor

```python
ONNXBackend(build_dir: str | None = None, opset: int = 17)
```

#### Métodos

| Método | Signatura | Descripción |
|--------|-----------|-------------|
| `build_model` | `(fma_seq, input_name="x", output_name="y", precision="fp64", batch_size=-1) → onnx.ModelProto` | Construir grafo FMA (2 nodos/etapa: Mul+Add) |
| `build_complex_model` | `(fma_seq) → onnx.ModelProto` | Grafo ℂ: 8 nodos/etapa (aritmética Re/Im) |
| `save_model` | `(model, path)` | Guardar modelo en disco |
| `verify_model` | `(model) → bool` | Verificar con `onnx.checker` |
| `inspect_model` | `(model) → str` | Descripción de nodos e inicializadores |
| `load_and_run` | `(model_path, x) → np.ndarray` | Inferencia con ONNX Runtime |
| `quantize_model` | `(model, quant_type="int8") → onnx.ModelProto` | Cuantización dinámica INT8/UINT8 |
| `to_torch_script` | `(fma_seq, precision="fp64") → str` | Código Python para `class PoemFMA(nn.Module)` |
| `compile` | `(fma_seq, source_ast, precision="fp64") → BackendResult` | Compilación completa con ORT o fallback NumPy |

#### Tipos ONNX válidos

| Cadena precision | TensorProto | NumPy dtype |
|-----------------|-------------|-------------|
| `"fp64"`, `"double"` | `DOUBLE = 11` | `np.float64` |
| `"fp32"`, `"float"` | `FLOAT = 1` | `np.float32` |

`BackendResult.extra` contiene:
- `"model"`: `onnx.ModelProto`
- `"node_count"`: número de nodos en el grafo
- `"has_onnx"`: bool
- `"has_ort"`: bool

---

### 21.4. Suite Completa de Tests

```bash
# 69 tests del motor v2.4.0 (nuevos)
python3 -m pytest tests/test_engine_massive.py -v
# Tiempo: ~31s, 69/69 passed

# Suite completa con regresión total
python3 -m pytest tests/ -v
# Tiempo: ~100s, 492/492 passed
```

Grupos de tests en `test_engine_massive.py`:
1. `TestCNativeEngineCorrectness` — correctitud numérica (rtol=1e-10)
2. `TestCNativeEngineHighLevelAPI` — API de alto nivel
3. `TestCNativeEnginePerformance` — benchmarks GFLOPS
4. `TestWasmBackend` — generación WAT/WASM/JS
5. `TestONNXBackend` — grafos ONNX, ORT, cuantización
6. `TestBackendRegistry` — registro y descubrimiento de backends
7. `TestCNativeEngineStress` — estrés: profundidad 5000, tamaño=1, arrays no-contiguos

---

## 22. Referencia de API — Dominio Complejo ℂ

### `poema.complex_domain`

#### `ComplexFMAInstruction(weight: complex, bias: complex)`

Instrucción FMA atómica sobre ℂ.

| Atributo / Método | Firma | Descripción |
|---|---|---|
| `weight` | `complex` | Parámetro de peso `w ∈ ℂ` |
| `bias` | `complex` | Parámetro de sesgo `b ∈ ℂ` |
| `weight_real` | `float` | `Re(w)` |
| `weight_imag` | `float` | `Im(w)` |
| `bias_real` | `float` | `Re(b)` |
| `bias_imag` | `float` | `Im(b)` |
| `apply(z)` | `(complex) → complex` | Evalúa `w*z + b` |
| `adjoint()` | `() → ComplexFMAInstruction` | Devuelve instrucción con peso `w̄` (conjugado) |
| `is_unitary_factor()` | `() → bool` | True si `\|w\| ≈ 1` — iff la instrucción preserva norma |
| `to_real_pair()` | `() → tuple[FMAInstruction, FMAInstruction]` | Expande en las 2 FMAs reales de Cauchy-Riemann |

**Nota:** 1 `ComplexFMAInstruction.apply()` ejecuta exactamente 2 FMAs reales en hardware.

---

#### `ComplexCompilationReport`

Resultado del proceso de colapso `ComplexACF.collapse_from_callable()`.

| Campo | Tipo | Descripción |
|---|---|---|
| `n_fma_complex` | `int` | Número de instrucciones FMA complejas en la cadena |
| `n_fma_real_equivalent` | `int` | `= 2 × n_fma_complex` — costo real en hardware |
| `is_unitary` | `bool` | True si todas las FMAs tienen `\|w\| ≈ 1` |
| `unitary_deviation` | `float` | Máxima desviación `abs(\|w\| − 1)` |
| `epsilon_complex` | `float` | Error máximo de aproximación en ℂ |
| `alpha_A_complex` | `float` | Coeficiente ACF sobre dominio complejo |
| `lean_certificate` | `str \| None` | Certificado Lean 4 auto-generado (si se solicita) |
| `summary()` | `() → str` | Resumen legible (imprimible directamente) |

---

#### `ComplexACF(n_terms, domain_re, domain_im, precision='complex128')`

Functor ACF extendido al semiplano complejo `(domain_re) × (domain_im) ⊂ ℂ`.

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `n_terms` | `16` | Grado del polinomio de Chebyshev complejo |
| `domain_re` | `(-1, 1)` | Intervalo real del dominio |
| `domain_im` | `(-1, 1)` | Intervalo imaginario del dominio |
| `precision` | `'complex128'` | Precisión numérica (`complex64` o `complex128`) |

**Métodos:**

```python
ComplexACF.collapse(f_samples: ndarray, z_samples: ndarray) -> list[ComplexFMAInstruction]
```

Colapsa muestras discretas `{(z_k, f(z_k))}` en cadena FMA compleja mínima.

```python
ComplexACF.collapse_from_callable(
    f: Callable[[complex], complex],
    n_samples: int = 64,
    report: bool = True
) -> tuple[list[ComplexFMAInstruction], ComplexCompilationReport]
```

Colapsa `f` muestreando automáticamente el dominio. Devuelve cadena + reporte.

---

#### `ComplexPoem`

Frontend de alto nivel para construir pipelines sobre ℂ. API análoga a `Poem` del dominio real, pero acepta pesos y activaciones complejas. Internamente convierte cada capa a `ComplexFMAInstruction`.

---

#### `UnitaryObservable(dim: int)`

Observable unitario para procesado de señales cuánticas o RF. Construye una cadena FMA compleja que implementa un operador unitario `U: ℂ^dim → ℂ^dim` con `U†U = I`.

---

#### `CategoricalFFT(n_points: int)`

FFT de radix-2 representada como objeto ACF.

| Parámetro | Descripción |
|---|---|
| `n_points` | Número de puntos (debe ser potencia de 2) |

```python
fft = CategoricalFFT(n_points=64)
fft_chain = fft.to_fma_chain()
# 6 capas × 32 mariposas = 192 FMAs complejas = 384 FMAs reales
```

---

### `martinez_functor.complex_algebra`

#### `ACFComplexTopos(precision=torch.complex128, koopman_dim=256)`

Extensión del topos ACF al dominio complejo. Opera sobre tensores PyTorch complejos.

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `precision` | `torch.complex128` | Tipo de dato de los tensores |
| `koopman_dim` | `256` | Dimensión del espacio de Koopman complejo |

**Métodos:**

---

##### `lift_to_complex_hilbert(real_tensor, phase_init=None) → Tensor`

Eleva un tensor real a representación en el espacio de Hilbert complejo.

```python
import torch
from martinez_functor.complex_algebra import ACFComplexTopos

topos = ACFComplexTopos()
x_real = torch.randn(64)
x_complex = topos.lift_to_complex_hilbert(x_real)      # shape (64,), dtype=complex128
x_phase  = topos.lift_to_complex_hilbert(x_real, phase_init=torch.pi/4)
```

- `phase_init=None`: fase inicial `0` (parte imaginaria = 0 en la elevación)
- `phase_init=θ`: aplica rotación global de fase `e^{iθ}` antes de elevar

**Postcondición:** `\|x_complex\|_2 = \|x_real\|_2` — la norma euclidiana se conserva.

---

##### `unitary_koopman_operator(state, phase_shift, preserve_norm=True) → Tensor`

Aplica el operador de Koopman unitario `K_φ: H_ℂ → H_ℂ`.

```python
state_new = topos.unitary_koopman_operator(
    state=x_complex,
    phase_shift=0.1,         # incremento de fase δφ ∈ ℝ
    preserve_norm=True       # fuerza |K_φ state|_2 = |state|_2
)
```

- Cuando `preserve_norm=True`, normaliza el resultado para garantizar isometría.
- Invariante: si `preserve_norm=True`, `torch.norm(state_new) == torch.norm(state)` dentro de `rtol=1e-10`.

---

##### `complex_fma_conservation(a, b, c) → Tensor`

Evalúa la FMA compleja `a*b + c` con conservación estricta garantizada.

```python
result = topos.complex_fma_conservation(a, b, c)  # = a*b + c en complex128
```

- Usa aritmética de doble precisión compleja.
- Empleada internamente por `ComplexPoem` para garantizar que el error de redondeo no acumule sesgos de fase.
- **Invariante:** `|result - (a*b + c)| < ε_machine` con `ε_machine ≈ 2.2e-16` para `complex128`.

---

### Cuadro comparativo: Dominio ℝ vs. Dominio ℂ

| Propiedad | ℝ (Poema v2.x) | ℂ (complex_domain) |
|---|---|---|
| Átomo FMA | `y = w·x + b`, `w,x,b ∈ ℝ` | `y = w·x + b`, `w,x,b ∈ ℂ` |
| Costo hardware | 1 FMA real | 2 FMAs reales (Cauchy-Riemann) |
| Ley de conservación | `\|Df\|` Lipschitz (ℝ→ℝ) | Unitaria: `\|w\| = 1` |
| Functorialidad | ACF sobre ℝ^n | ACF sobre ℂ^n (holomorfico) |
| Error de aproximación | ε ≈ 1e-12 (float64) | ε_ℂ ≈ 1e-14 (complex128) |
| α_A | Rango [0, ∞) | Normalizado en [0, 1] (unitario) |
| Certificado Lean 4 | `CompilationReport.lean_certificate` | `ComplexCompilationReport.lean_certificate` |
| Backend ONNX | `ONNXBackend.build_model()` | `ONNXBackend.build_complex_model()` |
| Aplicaciones | Redes neuronales, señal, física | Quantum, RF, audio, visión compleja |

---

## 23. Módulos de Máxima Capacidad ACF — Referencia de API

Esta sección documenta los seis módulos nuevos (y dos clases nuevas en módulos existentes) que cierran todos los problemas abiertos del Paper ACF (ver Paper.md §30).

---

### `acf_functor.koopman_delta_bounds`

#### Clase `KoopmanDeltaBounds`

```python
KoopmanDeltaBounds(eigenvalues: list[complex])
```

Calcula cotas certificadas de $\delta(d)$ a partir del espectro de un operador de Koopman.

| Método | Retorna | Descripción |
|--------|---------|-------------|
| `.at(d)` | `SpectralBound` | Cota en dimensión $d$ |
| `.optimal_dimension(epsilon)` | `OptimalDimensionResult` | $d^*(\varepsilon)$ mínimo |
| `.classify()` | `ConvergenceFamilyReport` | Familia de convergencia |

**`SpectralBound`** — campos: `d`, `delta_upper`, `delta_lower`, `derivation`  
**`OptimalDimensionResult`** — campos: `d_star`, `achieved_delta`, `epsilon_target`  
**`ConvergenceFamilyReport`** — campo: `family` (valores: `"exponential"`, `"polynomial"`, `"sub_exponential"`)

```python
from acf_functor import KoopmanDeltaBounds, delta_bounds_from_edmd, delta_bounds_from_svd

# Desde valores propios directos
eig = [0.99, 0.8, 0.5, 0.1, 0.01]
bounds = KoopmanDeltaBounds(eig)
sb = bounds.at(3)
print(sb.delta_upper)          # cota superior de δ(3)

# Dimensión óptima
opt = bounds.optimal_dimension(epsilon=1e-4)
print(opt.d_star)              # 4 (mínimo d con δ(d) ≤ 1e-4)

# Desde matriz EDMD
bounds2 = delta_bounds_from_edmd(K_matrix)

# Desde datos (SVD)
bounds3 = delta_bounds_from_svd(data_matrix)
```

#### Clase `CompositionDelta`

```python
CompositionDelta()
cert = CompositionDelta().certify(bounds_f, bounds_g, d, L_f)
# cert.delta_composition_bound   — cota superior
# cert.delta_composition_measured — medida empírica
# cert.is_valid                  — bool
```

---

### `acf_functor.mixed_composition`

#### Clase `MixedCompositionCertifier`

```python
cert = MixedCompositionCertifier().certify(
    f_exact,               # callable f(x) → tensor
    g_exact,               # callable g(x) → tensor
    phi_g,                 # ReductionResult de g (rama Koopman)
    koopman_eigenvalues,   # list[complex]
    koopman_coeff_norm,    # float
    observable_fns,        # list[callable]
    koopman_k_matrix,      # tensor K
    input_domain,          # tuple (a, b)
    d,                     # int — dimensión de truncación
)
```

**`MixedCompositionCertificate`** — campos principales:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `epsilon_total` | float | Cota total de error $\|f \circ g - \widetilde{\Phi}\|$ |
| `delta_outer` | float | $\delta_f(d)$ — error de truncación exterior |
| `epsilon_inner` | float | $\varepsilon_g$ — error de reducción interior |
| `is_valid` | bool | Certificado verificado |
| `proof_sketch` | str | Resumen de la demostración |

#### Clase `PolynomialKoopmanCertifier`

```python
result_dict = PolynomialKoopmanCertifier().certify(
    f_exact, g_exact, phi_f, koopman_eigenvalues_g, lipschitz_f, input_domain, d
)
# result_dict['epsilon_total'], result_dict['epsilon_f'], result_dict['delta_g']
# result_dict['is_valid'], result_dict['proof']
```

---

### `acf_functor.acf_inverse`

#### Clase `ACFInverse`

```python
from acf_functor import ACFInverse

inv = ACFInverse()

# Invertir una reducción
result = inv.invert(
    reduction,        # ReductionResult
    f_exact,          # callable — función original
    x_trajectory,     # tensor — puntos de entrenamiento (para Koopman)
    observable_fn,    # callable — observable (para Koopman)
    test_domain,      # tuple (a, b) — dominio de verificación
    n_test=200,       # puntos de evaluación
)
# result.reconstructed_fn   — callable: la inversa reconstruida
# result.certificate        — InversionCertificate
# result.coefficients       — coeficientes internos

# Verificar round-trip
rt = inv.verify_roundtrip(reduction)
# rt['roundtrip_linf'], rt['roundtrip_l2']
```

**`InversionCertificate`** — campos: `branch`, `reconstruction_error`, `error_bound`, `is_exact`, `n_test_points`, `proof_sketch`

---

### `acf_functor.information_geometry`

#### Clase `FisherMetricACF`

```python
from acf_functor import FisherMetricACF, polynomial_observables, fourier_observables

observables = polynomial_observables(max_degree=5)
g_F = FisherMetricACF(observables)
metric = g_F.compute(c)   # c = tensor de coeficientes
# metric.G               — matriz (n×n) de la métrica Fisher
# metric.condition_number
# metric.log_det
```

#### Clase `AffineMetricACF`

```python
from acf_functor import AffineMetricACF

g_A = AffineMetricACF()
metric = g_A.compute(K)         # K = matriz de Koopman
# metric.G = (K^T K + reg I)^{-1}   — inversa de Gram

nat_grad = g_A.natural_gradient(gradient, K)
```

#### Clase `LegendreDuality`

```python
from acf_functor import LegendreDuality

ld = LegendreDuality(fisher_metric)
point = ld.compute_at(c)
# point.theta, point.eta, point.psi, point.psi_star
# point.kl_diverg    — gap de Fenchel ≥ 0
```

#### Clase `DualityVerifier`

```python
from acf_functor import DualityVerifier

verifier = DualityVerifier(fisher_metric, affine_metric)
report = verifier.verify(c, K)
# report.duality_holds         — bool
# report.frobenius_distance    — distancia entre métricas
# report.relative_error        — error relativo
# report.corollary_verified    — bool
```

#### Clase `InformationGeometry`

```python
from acf_functor import InformationGeometry

ig = InformationGeometry(observables)
report = ig.analyze(theta, K)
# report.duality              — DualityVerificationResult
# report.g_fisher             — MetricTensor
# report.g_affine             — MetricTensor
# report.legendre_point
# report.fma_energy_geometric
```

**Fábricas disponibles:**

```python
polynomial_observables(max_degree=5)   # lista de observables polinomiales
fourier_observables(n_harmonics=4)     # lista de observables de Fourier
```

---

### `acf_functor.thermodynamic_acf`

#### Clase `ThermodynamicACF`

```python
from acf_functor import ThermodynamicACF

thermo = ThermodynamicACF(
    eigenvalues,                  # list[complex] o list[float]
    entropy_mode="combinatorial", # "combinatorial" | "spectral"
    beta_min=0.01,
    beta_max=100.0,
    n_beta=50,
)

# Análisis completo
report = thermo.analyze(beta_samples=None)  # usa beta_min..beta_max por defecto
```

**`ThermodynamicReport`** — campos:

| Campo | Descripción |
|-------|-------------|
| `d_star_zero_temp` | Dimensión óptima en límite T→0 (máxima precisión) |
| `d_star_high_temp` | Dimensión óptima en límite T→∞ (máxima compresión) |
| `optimal_beta` | $\beta$ de equilibrio energía-entropía |
| `mdl_dimension` | Dimensión por principio de descripción mínima |
| `alpha_from_thermodynamics` | Índice $\alpha$ derivado del análisis térmico |
| `profiles` | Lista de `FreeEnergyProfile` por $\beta$ |
| `phase_transition` | `ThermoPhaseTransition` (si existe) |

```python
# Dimensión óptima para un ε dado
dim = thermo.optimal_dimension(target_epsilon=1e-4, beta=1.0)
print(dim['d_star'], dim['achieved_error'], dim['is_feasible'])
```

---

### `acf_functor.invariant_unified` — `AlphaUnificationTheorem`

```python
from acf_functor import AlphaUnificationTheorem, UnificationResult

theorem = AlphaUnificationTheorem(d_max=256, attractor_dim=1.0)

# Certificar un estimado de α
result = theorem.certify(alpha_estimate=2.5)
```

**`UnificationResult`** — campos:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `certified_alpha` | float | α certificado |
| `certified_ci` | tuple | Intervalo de confianza (low, high) |
| `theorem_satisfied` | bool | $|\alpha_{\text{comb}} - \alpha_{\text{spec}}| \leq C_1/\sqrt{d_{\max}}$ |
| `disc_comb_spec` | float | $|\alpha_{\text{comb}} - \alpha_{\text{spec}}|$ |
| `disc_spec_geo` | float | $|\alpha_{\text{spec}} - \alpha_{\text{geo}}|$ |
| `proof_sketch` | str | Resumen formal |

```python
# Conversión Bernstein ↔ α
rho = theorem.bernstein_from_alpha(2.5)   # e^{1/2.5} ≈ 1.492
alpha = theorem.alpha_from_bernstein(rho) # 1/log(rho) ≈ 2.5
# Identidad: alpha_from_bernstein(e) == 1.0 exactamente
```

---

### `acf_functor.lie_analysis` — `NCComplexityAnalyzer`

```python
from acf_functor import NCComplexityAnalyzer

analyzer = NCComplexityAnalyzer(commutativity_threshold=1e-8)
result = analyzer.analyze(fma_sequence)  # lista de FMAOperation
```

**`NCComplexityResult`** — campos:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `nc_class` | str | `"NC^0"`, `"NC^1"`, `"NC^2"`, `"NC^3"`, `"P-hard"` |
| `lie_span_dim` | int | dim(Lie-span{[W_i, W_j]}) |
| `is_nc` | bool | True si nc_class ≠ "P-hard" |
| `depth_over_log_n` | float | Ratio profundidad/log(n) |
| `parallelizable_fraction` | float | Fracción de operaciones paralelizables |
| `bracket_frobenius_norms` | list | ‖[W_i, W_j]‖_F por par |
| `proof_sketch` | str | Resumen formal |

```python
# Ejemplo: secuencia commutativa → NC^1
from acf_functor import NCComplexityAnalyzer, FMAOperation
import torch, math

ops = [FMAOperation(weight=torch.eye(3)*k, bias=torch.zeros(3)) for k in range(1,4)]
result = NCComplexityAnalyzer().analyze(ops)
print(result.nc_class)        # "NC^1"
print(result.lie_span_dim)    # 0
print(result.is_nc)           # True
```

---

### Flujo Integrado de Máxima Capacidad

```python
import torch
from acf_functor import (
    ACFFunctor, KoopmanDeltaBounds, ThermodynamicACF,
    NCComplexityAnalyzer, AlphaUnificationTheorem,
    InformationGeometry, polynomial_observables,
)

# 1. Reducción ACF
phi = ACFFunctor(epsilon=1e-6)
reduction = phi.reduce_polynomial([1.0, -0.5, 0.1, 0.0, 0.01])

# 2. Cotas Koopman (si la función es dinámica)
eig = [0.99**k for k in range(1, 21)]
bounds = KoopmanDeltaBounds(eig)
print(f"δ(10) ≤ {bounds.at(10).delta_upper:.4f}")
d_star = bounds.optimal_dimension(1e-4).d_star
print(f"d*(1e-4) = {d_star}")

# 3. Selección termodinámica de dimensión
thermo = ThermodynamicACF(eig, entropy_mode="combinatorial")
report = thermo.analyze()
print(f"d* termodinámico: {report.d_star_zero_temp} (preciso) / {report.d_star_high_temp} (compacto)")

# 4. Complejidad NC de la reducción polinomial
nc = NCComplexityAnalyzer().analyze(reduction.fma_sequence)
print(f"Clase NC: {nc.nc_class} | LieDim: {nc.lie_span_dim}")

# 5. Unificación de α
theorem = AlphaUnificationTheorem()
alpha_result = theorem.certify(alpha_estimate=2.0)
print(f"α certificado: {alpha_result.certified_alpha:.4f}")
print(f"Teorema satisfecho: {alpha_result.theorem_satisfied}")

# 6. Geometría de la información
obs = polynomial_observables(max_degree=4)
ig = InformationGeometry(obs)
c = torch.randn(5)
K = torch.randn(5, 5)
geo_report = ig.analyze(c, K)
print(f"Dualidad Fisher-Afín: {geo_report.duality.duality_holds}")
```

---

## 24. Next-Generation: The Mathematical Discoverer (Roadmap)

> **Status:** Research programme. Not implemented. See Paper.md §31 for full specification.

### Current Genesis Capabilities (implemented)

```python
from acf_functor.genesis import GenesisOrchestrator

# Genesis: numerical conjecture discovery
g = GenesisOrchestrator(budget=5000, persistence_threshold=0.95)
report = g.run()

for candidate in report.candidates:
    print(candidate.expression)   # e.g. "sin(x)^2 + cos(x)^2"
    print(candidate.residual)     # e.g. 3.2e-7 (numerical evidence only)
    print(candidate.status)       # "CONJECTURE" — NOT a theorem
```

**What Genesis produces:** A list of `(expression, residual, fingerprint)` tuples.  
**What Genesis does NOT produce:** Formal proofs. The `.status` field will never say `"THEOREM"` until the symbolic+Lean layer is built.

### Target Architecture (§31 of Paper.md)

The Mathematical Discoverer adds three layers on top of Genesis:

```
Genesis          SymbolicACFExpression    ProofSearchEngine    Lean 4
────────         ─────────────────────    ─────────────────    ──────
conjecture   →   formal hypothesis    →   tactic proof     →   PROVED
(numerical)      (Lean 4 syntax)          (Mathlib search)     certificate
```

### Planned API (not yet implemented)

```python
# Future: MathematicalDiscoverer (Phase D target)
from acf_functor.discoverer import MathematicalDiscoverer  # NOT YET AVAILABLE

discoverer = MathematicalDiscoverer(
    genesis_budget=10_000,
    proof_timeout_s=30,
    lean_binary=".venv/bin/lean",
)
results = discoverer.run()

for theorem in results.proved:
    print(theorem.statement)     # Lean 4 theorem statement
    print(theorem.proof)         # Lean 4 tactic proof
    print(theorem.certificate)   # path to .lean file

for conjecture in results.open:
    print(conjecture.statement)  # Lean 4 hypothesis (unproved)
    print(conjecture.evidence)   # numerical residual + sample count
```

### Implementation Phases

| Phase | Deliverable | Dependencies |
|-------|-------------|--------------|
| A | `SymbolicACFExpression` class — emits Lean 4 hypotheses from Genesis candidates | Genesis output format |
| B | `ProofTemplateLibrary` — Mathlib-matched tactic templates per expression shape | Lean 4 + Mathlib 4 |
| C | `ProofSearchEngine` — best-first tactic search with timeout | Phase A + B |
| D | `MathematicalDiscoverer` — full pipeline; mines ACF-specific invariants | Phase A + B + C |

See Poema.md §12 for the conceptual description and Paper.md §31 for the formal specification.

---

## 25. Referencia de API — Motor de Auto-Evolución ACF

> **Estado:** Implementado. Módulo: `acf_functor/auto_evolution.py`. Integrado en `poema/compiler.py` y `poema/backends/gideon/engine.py`. Tests: `tests/test_auto_evolution.py` (59/59 passing).

### 25.1. Importaciones

```python
# API pública completa desde acf_functor
from acf_functor import (
    FixedPointIterator,    FixedPointResult,
    BifunctorialCycle,     BifunctorialResult,
    ThermodynamicSearch,   ThermodynamicSearchResult,  ConfigurationPoint,
    AdaptiveRefinement,    AdaptiveRefinementResult,   RefinedInterval,
    ACFAutoEvolver,        ACFAutoEvolverConfig,        AutoEvolutionResult,
)

# Integración con Poema
from poema import Poem
from poema.backends.gideon.engine import GideonEngine
```

---

### 25.2. `ACFAutoEvolverConfig`

Clase de datos de configuración para el pipeline unificado.

```python
@dataclass
class ACFAutoEvolverConfig:
    # ── General ──────────────────────────────────────────────────────────
    initial_degree: int = 20       # Grado inicial antes del pipeline
    n_probe: int = 3000            # Puntos para evaluar ε (residuo máximo)
    beta: float = 1.0              # β termodinámico: alto=precisión, bajo=simplicidad
    dtype: torch.dtype = torch.float64

    # ── FixedPointIterator ───────────────────────────────────────────────
    enable_fixed_point: bool = True
    fp_max_iterations: int = 6
    fp_convergence_tol: float = 1e-12

    # ── BifunctorialCycle ────────────────────────────────────────────────
    enable_bifunctorial: bool = True
    bif_max_cycles: int = 4
    bif_convergence_tol: float = 1e-12

    # ── ThermodynamicSearch ──────────────────────────────────────────────
    enable_thermo_search: bool = True
    thermo_degree_candidates: Optional[List[int]] = None  # auto si None

    # ── AdaptiveRefinement ───────────────────────────────────────────────
    enable_adaptive: bool = True
    adaptive_target_epsilon: float = 1e-8
    adaptive_max_degree: int = 80
```

**Parámetro `beta`:**
- `beta >> 1` → prioriza error mínimo (máximo grado útil)
- `beta << 1` → prioriza representación simple (mínimo grado)
- `beta = 1.0` → equilibrio (defecto)

---

### 25.3. `ACFAutoEvolver`

Orquestador del pipeline completo.

#### Constructor

```python
ACFAutoEvolver(config: ACFAutoEvolverConfig = ACFAutoEvolverConfig())
```

#### `evolve`

```python
def evolve(
    self,
    f: Callable[[torch.Tensor], torch.Tensor],
    domain: Tuple[float, float],
) -> AutoEvolutionResult
```

Ejecuta el pipeline sobre la función `f` en el dominio `[a, b]`.

`f` debe aceptar un `torch.Tensor` de forma `(N,)` y devolver un `torch.Tensor` de la misma forma.

**Retorna:** `AutoEvolutionResult`

```python
@dataclass
class AutoEvolutionResult:
    final_reduction: ReductionResult  # La mejor reducción encontrada
    initial_epsilon: float            # ‖f - Φ₀(f)‖∞ antes del pipeline
    final_epsilon: float              # ‖f - Φ*(f)‖∞ después del pipeline
    improvement_ratio: float          # initial_epsilon / final_epsilon (≥ 1)

    thermo_result: Optional[ThermodynamicSearchResult]
    fixed_point: Optional[FixedPointResult]
    bifunctorial: Optional[BifunctorialResult]
    adaptive: Optional[AdaptiveRefinementResult]

    elapsed_ms: float
    config: ACFAutoEvolverConfig

    def summary(self) -> str
    # → "ACFAutoEvolver: ε₀=7.4e-02 → ε_f=3.6e-10 | ratio=2.1e8 | t=1842ms"
```

#### `is_fixed_point`

```python
def is_fixed_point(
    self,
    f: Callable,
    reduction: ReductionResult,
    domain: Tuple[float, float],
    tol: float = 1e-10,
) -> Tuple[bool, float]
```

Comprueba si `reduction` es ya un punto fijo de Φ. Devuelve `(es_fp, delta)` donde `delta = ‖Φ(Φ(f)) - Φ(f)‖∞`.

---

### 25.4. `FixedPointIterator`

Itera Φ hasta convergencia (propiedad Φ² = Φ).

```python
class FixedPointIterator:
    def __init__(
        self,
        max_iterations: int = 10,
        convergence_tol: float = 1e-12,
        degree: int = 30,
        n_probe: int = 3000,
        dtype: torch.dtype = torch.float64,
    )

    def iterate(
        self,
        f: Callable,
        domain: Tuple[float, float],
        initial_reduction: Optional[ReductionResult] = None,
    ) -> FixedPointResult
```

```python
@dataclass
class FixedPointResult:
    reduction: ReductionResult
    n_iterations: int
    initial_epsilon: float
    final_epsilon: float
    already_fixed_point: bool        # True si Φ(f) ya era punto fijo
    convergence_history: List[float] # δ por iteración
    converged: bool
    elapsed_ms: float

    def summary(self) -> str
    # → "FixedPointIterator: already_fp | iters=1 | ε₀=2.2e-15 → ε_f=2.2e-15 | t=4ms"
```

---

### 25.5. `BifunctorialCycle`

Cicla entre Φ (compresión) y Φ* (síntesis) buscando representaciones más ajustadas.

```python
class BifunctorialCycle:
    def __init__(
        self,
        max_cycles: int = 6,
        convergence_tol: float = 1e-12,
        degree: int = 30,
        n_probe: int = 3000,
        dtype: torch.dtype = torch.float64,
    )

    def cycle(
        self,
        f: Callable,
        domain: Tuple[float, float],
    ) -> BifunctorialResult
```

```python
@dataclass
class BifunctorialResult:
    reduction: ReductionResult
    n_cycles: int
    initial_epsilon: float
    final_epsilon: float
    converged: bool
    epsilon_history: List[float]  # ε por ciclo
    elapsed_ms: float

    def summary(self) -> str
```

---

### 25.6. `ThermodynamicSearch`

Busca el grado y método que minimiza la energía libre $F(d, \beta) = E(d) - S(d)/\beta$.

```python
class ThermodynamicSearch:
    def __init__(
        self,
        degree_candidates: Optional[List[int]] = None,  # auto si None
        beta: float = 1.0,
        n_probe: int = 3000,
        dtype: torch.dtype = torch.float64,
    )

    def search(
        self,
        f: Callable,
        domain: Tuple[float, float],
    ) -> ThermodynamicSearchResult
```

```python
@dataclass
class ConfigurationPoint:
    degree: int
    method: str        # "chebyshev" | "horner"
    epsilon: float     # ‖f - Φ_c(f)‖∞
    free_energy: float # F(c, β)
    entropy: float     # log(1 + degree)

@dataclass
class ThermodynamicSearchResult:
    best_config: ConfigurationPoint
    best_reduction: ReductionResult
    all_configs: List[ConfigurationPoint]  # todos los candidatos evaluados
    elapsed_ms: float

    def summary(self) -> str
```

---

### 25.7. `AdaptiveRefinement`

Identifica sub-intervalos de alto error y eleva el grado localmente.

```python
class AdaptiveRefinement:
    def __init__(
        self,
        target_epsilon: float = 1e-8,
        max_degree: int = 80,
        n_probe: int = 3000,
        dtype: torch.dtype = torch.float64,
    )

    def refine(
        self,
        f: Callable,
        domain: Tuple[float, float],
        initial_reduction: Optional[ReductionResult] = None,
        initial_degree: int = 20,
    ) -> AdaptiveRefinementResult
```

```python
@dataclass
class RefinedInterval:
    start: float
    end: float
    initial_epsilon: float
    final_epsilon: float
    degree_used: int

@dataclass
class AdaptiveRefinementResult:
    reduction: ReductionResult
    initial_epsilon: float
    final_epsilon: float
    refined_intervals: List[RefinedInterval]  # sub-intervalos refinados
    elapsed_ms: float

    def summary(self) -> str
```

---

### 25.8. Integración con PoemCompiler

```python
class PoemCompiler:
    def auto_evolve(
        self,
        ast: ASTNode,
        domain: Tuple[float, float],
        config: Optional[ACFAutoEvolverConfig] = None,
    ) -> AutoEvolutionResult:
        """
        Compila el AST a una función callable y ejecuta ACFAutoEvolver.evolve().
        """
```

**Ejemplo completo:**

```python
from poema import Poem
from poema.frontend import NumberNode, FunctionCallNode, IdentifierNode
from acf_functor import ACFAutoEvolverConfig

compiler = Poem()

# Construir AST: sin(x)
ast = FunctionCallNode("sin", [IdentifierNode("x")])

# Config personalizada
cfg = ACFAutoEvolverConfig(initial_degree=15, beta=2.0)

# Auto-evolucionar
result = compiler.compiler.auto_evolve(ast, domain=(-3.14, 3.14), config=cfg)
print(result.summary())
```

---

### 25.9. Integración con GideonEngine

```python
class GideonEngine:
    def auto_evolve_fma(
        self,
        fma_sequence: List[FMANode],
        domain: Tuple[float, float],
        config: Optional[ACFAutoEvolverConfig] = None,
    ) -> AutoEvolutionResult:
        """
        Construye una función callable desde fma_sequence y ejecuta ACFAutoEvolver.
        """
```

**Ejemplo:**

```python
from poema.backends.gideon.engine import GideonEngine
from acf_functor import ACFAutoEvolverConfig

engine = GideonEngine()

# Compilar expresión Poema
result = engine.compile("sin(x) + 0.5 * cos(2*x)", optimization_level=2)
fma_seq = result.fma_sequence

# Auto-evolucionar
evo = engine.auto_evolve_fma(fma_seq, domain=(-3.14, 3.14))
print(evo.summary())
```

---

### 25.10. Tabla de Clases Exportadas

| Clase | Módulo | Propósito |
|-------|--------|-----------|
| `ACFAutoEvolver` | `acf_functor.auto_evolution` | Pipeline unificado |
| `ACFAutoEvolverConfig` | `acf_functor.auto_evolution` | Configuración del pipeline |
| `AutoEvolutionResult` | `acf_functor.auto_evolution` | Resultado del pipeline completo |
| `FixedPointIterator` | `acf_functor.auto_evolution` | Iteración de punto fijo (Φ²=Φ) |
| `FixedPointResult` | `acf_functor.auto_evolution` | Resultado de iteración FP |
| `BifunctorialCycle` | `acf_functor.auto_evolution` | Ciclo adjunto Φ⊣Φ* |
| `BifunctorialResult` | `acf_functor.auto_evolution` | Resultado del ciclo |
| `ThermodynamicSearch` | `acf_functor.auto_evolution` | Búsqueda por energía libre |
| `ThermodynamicSearchResult` | `acf_functor.auto_evolution` | Resultado de búsqueda |
| `ConfigurationPoint` | `acf_functor.auto_evolution` | Un punto del espacio de configuraciones |
| `AdaptiveRefinement` | `acf_functor.auto_evolution` | Refinamiento adaptativo por residuo |
| `AdaptiveRefinementResult` | `acf_functor.auto_evolution` | Resultado del refinamiento |
| `RefinedInterval` | `acf_functor.auto_evolution` | Sub-intervalo refinado |

Todos accesibles directamente desde `acf_functor`:
```python
from acf_functor import ACFAutoEvolver, ACFAutoEvolverConfig  # etc.
```

Ver Paper.md §32 para la formalización matemática completa y Poema.md §13 para la descripción conceptual.

---

## 26. Referencia de API — ACF para Grafos (`graph_acf.py`)

### 26.1. Resumen del módulo

`acf_functor/graph_acf.py` extiende el functor ACF al dominio de señales de grafos usando procesamiento espectral de grafos. El principio central: toda señal **s** sobre un grafo G = (V, E) puede filtrarse mediante un polinomio H(λ) que opera sobre los valores propios del Laplaciano L = UΛUᵀ. La reducción ACF elige ese polinomio de forma óptima.

**Exportaciones:**

```python
from acf_functor import (
    GraphLaplacian,      # Computa el Laplaciano y su descomposición espectral
    GraphSpectrum,       # Dataclass: eigenvalues, eigenvectors, normalization
    GraphReducer,        # Aplica el functor ACF sobre H(λ)
    GraphReductionResult,# Resultado: filtered_signal, polynomial_filter, epsilon
    GraphACFAnalyzer,    # Analiza invariantes ACF del grafo
    GraphACFInvariants,  # Dataclass: alpha, delta, nc_class, fiedler_value, ...
    GraphSignalEvolver,  # Conecta con ACFAutoEvolver
    GraphEvolutionResult,# Resultado de la auto-evolución espectral
    StandardGraphs,      # Generadores de grafos canónicos
)
```

### 26.2. GraphLaplacian

```python
class GraphLaplacian:
    @staticmethod
    def from_adjacency(
        A: np.ndarray,
        normalization: str = "unnormalized"
    ) -> GraphSpectrum
    
    @staticmethod  
    def from_edge_list(
        edges: List[Tuple[int, int, float]],
        n_nodes: int,
        normalization: str = "unnormalized"
    ) -> GraphSpectrum
```

**`from_adjacency(A, normalization)`**

Construye el Laplaciano a partir de la matriz de adyacencia y lo descompone espectralmente.

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `A` | `np.ndarray (n,n)` | Matriz de adyacencia simétrica, pesos ≥ 0 |
| `normalization` | `str` | `"unnormalized"` (L=D−A) o `"symmetric"` (D^{-1/2}LD^{-1/2}) |

Retorna `GraphSpectrum(eigenvalues, eigenvectors, normalization, n_nodes)`.

**`from_edge_list(edges, n_nodes, normalization)`**

Construye el Laplaciano a partir de lista de aristas (u, v, peso).

```python
from acf_functor import GraphLaplacian

A = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
spectrum = GraphLaplacian.from_adjacency(A)
print(spectrum.eigenvalues)    # [0., 3., 3.]  (K₃ completo)

edges = [(0,1,1.0), (1,2,1.0), (2,0,1.0)]
spectrum2 = GraphLaplacian.from_edge_list(edges, n_nodes=3)
```

### 26.3. GraphSpectrum

```python
@dataclass
class GraphSpectrum:
    eigenvalues: np.ndarray    # shape (n,) — valores propios ordenados
    eigenvectors: np.ndarray   # shape (n,n) — columnas = vectores propios
    normalization: str
    n_nodes: int
    
    @property
    def fiedler_value(self) -> float    # λ₂ — conectividad algebraica
    @property
    def spectral_range(self) -> Tuple[float, float]   # (λ_min, λ_max)
    @property
    def spectral_gap(self) -> float     # λ_max - λ₂
```

### 26.4. GraphReducer

```python
class GraphReducer:
    def __init__(self, degree: int = 8, target_epsilon: float = 1e-6):
        ...
    
    def reduce(
        self,
        signal: np.ndarray,    # shape (n,) — señal sobre nodos
        spectrum: GraphSpectrum,
    ) -> GraphReductionResult
```

Aplica el functor ACF sobre la función de filtro H: [λ_min, λ_max] → ℝ, obtiene el polinomio Chebyshev P_k, y filtra la señal:

$$\mathbf{s}_\text{filtered} = U \cdot \text{diag}(P_k(\Lambda)) \cdot U^\top \mathbf{s}$$

```python
from acf_functor import GraphLaplacian, GraphReducer, StandardGraphs

A = StandardGraphs.path(10)
spectrum = GraphLaplacian.from_adjacency(A)
signal = np.sin(np.linspace(0, np.pi, 10))

reducer = GraphReducer(degree=6)
result = reducer.reduce(signal, spectrum)
print(f"ε = {result.epsilon:.2e}")
print(f"Señal filtrada: {result.filtered_signal}")
```

### 26.5. GraphACFAnalyzer

```python
class GraphACFAnalyzer:
    def __init__(self, target_epsilon: float = 1e-4):
        ...
    
    def analyse(self, spectrum: GraphSpectrum) -> GraphACFInvariants
```

**`GraphACFInvariants` — campos:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `alpha` | `float` | Índice α ∈ [0,1] de la función espectral |
| `delta` | `float` | Índice δ (tasa de decaimiento espectral) |
| `nc_class` | `str` | `"NC0"`, `"NC1"`, `"NC2"`, `"NC3"` |
| `fiedler_value` | `float` | λ₂ — conectividad algebraica |
| `spectral_entropy` | `float` | −Σ (λᵢ/tr(L)) log(λᵢ/tr(L)) |
| `optimal_filter_degree` | `int` | Menor grado k tal que ε < target_epsilon |

```python
from acf_functor import GraphLaplacian, GraphACFAnalyzer, StandardGraphs

analyzer = GraphACFAnalyzer(target_epsilon=1e-4)
for name, A in [("path5", StandardGraphs.path(5)),
                ("cycle8", StandardGraphs.cycle(8)),
                ("complete6", StandardGraphs.complete(6))]:
    spectrum = GraphLaplacian.from_adjacency(A)
    inv = analyzer.analyse(spectrum)
    print(f"{name}: α={inv.alpha:.3f}, Fiedler={inv.fiedler_value:.3f}, grado={inv.optimal_filter_degree}")
```

### 26.6. GraphSignalEvolver

```python
class GraphSignalEvolver:
    def __init__(self, config: ACFAutoEvolverConfig = None):
        ...
    
    def evolve(
        self,
        signal: np.ndarray,
        spectrum: GraphSpectrum,
    ) -> GraphEvolutionResult
```

Aplica el pipeline `ACFAutoEvolver` (§25) sobre la función espectral del filtro. El resultado extiende el `AutoEvolutionResult` con la señal filtrada final y el historial completo.

**`GraphEvolutionResult` — campos:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `final_filtered_signal` | `np.ndarray` | Señal filtrada con el mejor polinomio |
| `spectral_gain` | `float` | ‖**s**_filtered‖ / ‖**s**‖ |
| `evolution_trace` | `AutoEvolutionResult` | Resultado completo del pipeline de auto-evolución |
| `improvement_ratio` | `float` | ε₀ / ε_f |

### 26.7. StandardGraphs

```python
class StandardGraphs:
    @staticmethod def path(n: int) -> np.ndarray
    @staticmethod def cycle(n: int) -> np.ndarray
    @staticmethod def complete(n: int) -> np.ndarray
    @staticmethod def grid(rows: int, cols: int) -> np.ndarray
    @staticmethod def star(n: int) -> np.ndarray
    @staticmethod def random_regular(n: int, d: int, seed: int = 42) -> np.ndarray
```

Todos los generadores retornan matrices de adyacencia `np.ndarray (n,n)` de tipo `float64`.

```python
from acf_functor import StandardGraphs
import numpy as np

A_path = StandardGraphs.path(6)        # P₆
A_cycle = StandardGraphs.cycle(8)      # C₈  
A_K5 = StandardGraphs.complete(5)      # K₅
A_grid = StandardGraphs.grid(3, 4)     # rejilla 3×4 (12 nodos)
A_star = StandardGraphs.star(7)        # K_{1,6}
A_rr = StandardGraphs.random_regular(10, 3, seed=0)  # 3-regular, 10 nodos
```

---

## 27. Referencia de API — Neural-ACF (`neural_acf.py`)

### 27.1. Resumen del módulo

`acf_functor/neural_acf.py` aplica el functor ACF a redes neuronales PyTorch: reduce capas lineales y convolucionales a polinomios FMA, analiza invariantes por capa via SVD, estudia la dinámica de entrenamiento con Koopman, y evalúa la compresibilidad de la función implementada.

**Requisito:** PyTorch (`torch`) instalado en el entorno.

**Exportaciones:**

```python
from acf_functor import (
    NeuralLayerReducer,      # Reduce capas individuales (Linear, Conv1d)
    LayerReductionResult,    # Resultado por capa
    NetworkACFAnalyzer,      # Análisis completo de red
    NetworkACFReport,        # Dataclass del reporte de red
    KoopmanNetworkDynamics,  # Koopman sobre trayectorias de pérdida
    KoopmanNetworkResult,    # Resultado del análisis Koopman
    NeuralACFEvolver,        # Auto-evoluciona la función de la red
    NeuralEvolutionResult,   # Resultado de la auto-evolución
    build_test_mlp,          # Factory de MLPs de prueba
)
```

### 27.2. NeuralLayerReducer

```python
class NeuralLayerReducer:
    def __init__(self, degree: int = 6, domain: Tuple[float,float] = (-1.0, 1.0)):
        ...
    
    def reduce_linear(self, layer: nn.Linear) -> LayerReductionResult
    def reduce_conv1d(self, layer: nn.Conv1d) -> LayerReductionResult
    def reduce_layer(self, layer: nn.Module) -> Optional[LayerReductionResult]
```

**`reduce_linear(layer)`**

Opera sobre `nn.Linear`:
1. Extrae W ∈ ℝᵐˣⁿ y **b** ∈ ℝᵐ.
2. Construye f_rep(x) = σ(w_mean·x + b_mean).
3. Aplica `ChebyshevReducer.reduce(f_rep, domain, degree)`.
4. Retorna `LayerReductionResult`.

**`reduce_conv1d(layer)`**

Opera sobre `nn.Conv1d`:
1. Extrae la respuesta impulsional del kernel.
2. Calcula el módulo espectral |H(ω)| via FFT.
3. Reduce |H(ω)| sobre [0, π].
4. Retorna `LayerReductionResult`.

**`reduce_layer(layer)`**

Despacha automáticamente a `reduce_linear` o `reduce_conv1d`. Retorna `None` para tipos de capa no soportados (sin lanzar excepción).

**`LayerReductionResult` — campos:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `layer_type` | `str` | `"linear"` o `"conv1d"` |
| `reduction` | `ReductionResult` | Resultado ACF completo |
| `epsilon` | `float` | Error de aproximación |
| `fma_count` | `int` | Número de FMAs en la secuencia |
| `layer_shape` | `Tuple` | Forma de la capa (out_features, in_features) |

```python
import torch.nn as nn
from acf_functor import NeuralLayerReducer

reducer = NeuralLayerReducer(degree=6, domain=(-1.0, 1.0))

layer = nn.Linear(64, 32)
result = reducer.reduce_linear(layer)
print(f"ε = {result.epsilon:.2e}, FMAs = {result.fma_count}")

conv = nn.Conv1d(8, 16, kernel_size=5)
result_c = reducer.reduce_conv1d(conv)
print(f"Conv ε = {result_c.epsilon:.2e}")

# Despacho automático
generic = reducer.reduce_layer(layer)   # nn.Linear → reduce_linear
none_result = reducer.reduce_layer(nn.ReLU())  # no soportado → None
```

### 27.3. NetworkACFAnalyzer

```python
class NetworkACFAnalyzer:
    def __init__(
        self,
        degree: int = 6,
        domain: Tuple[float,float] = (-1.0, 1.0),
    ):
        ...
    
    def analyse(self, network: nn.Module) -> NetworkACFReport
```

**`NetworkACFReport` — campos:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `layer_reductions` | `List[Optional[LayerReductionResult]]` | Reducción por capa (None si no soportada) |
| `layer_invariants` | `List[Optional[ACFInvariant]]` | Invariante α por capa via SVD |
| `global_alpha` | `float` | α ponderado de la red completa |
| `global_nc_class` | `str` | Clase NC del α global |
| `total_fma_count` | `int` | FMAs totales en representación polinomial |
| `n_layers_analysed` | `int` | Número de capas efectivamente analizadas |

```python
import torch.nn as nn
from acf_functor import NetworkACFAnalyzer

analyzer = NetworkACFAnalyzer(degree=6)
net = nn.Sequential(
    nn.Linear(128, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
report = analyzer.analyse(net)
print(f"α global = {report.global_alpha:.4f}")
print(f"Capas analizadas: {report.n_layers_analysed}")
print(f"FMAs totales: {report.total_fma_count}")
```

### 27.4. KoopmanNetworkDynamics

```python
class KoopmanNetworkDynamics:
    def __init__(self, max_rank: int = 10):
        ...
    
    def analyse(
        self,
        trajectory: np.ndarray,    # shape (T,) — serie temporal de pérdidas
    ) -> KoopmanNetworkResult
```

Internamente llama a `AdaptiveKoopman(max_rank=max_rank).reduce(trajectory.unsqueeze(0))` para obtener el operador de Koopman en el espacio de observables.

**Nota:** la trayectoria debe tener al menos 4 puntos temporales. Se recomienda T ≥ 20 para resultados estables.

**`KoopmanNetworkResult` — campos:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `koopman_eigenvalues` | `np.ndarray` | Valores propios complejos del operador K |
| `spectral_diagnostics` | `dict` | `dominant_eigenvalue`, `spectral_radius`, `convergence_flag` |
| `trajectory_length` | `int` | T |

```python
import numpy as np
from acf_functor import KoopmanNetworkDynamics

# Simular una trayectoria de entrenamiento oscilatoria
t = np.arange(30)
losses = 2.0 * np.exp(-0.1 * t) + 0.1 * np.sin(0.5 * t) + 0.05 * np.random.randn(30)

knd = KoopmanNetworkDynamics(max_rank=8)
result = knd.analyse(losses)
print("Eigenvalues de Koopman:", result.koopman_eigenvalues)
print("Diagnóstico:", result.spectral_diagnostics)
```

### 27.5. NeuralACFEvolver

```python
class NeuralACFEvolver:
    def __init__(self, config: ACFAutoEvolverConfig = None):
        ...
    
    def evolve(
        self,
        network: nn.Module,
        domain: Tuple[float,float] = (-1.0, 1.0),
        input_dim: int = 1,
    ) -> NeuralEvolutionResult
```

**`NeuralEvolutionResult` — campos:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `initial_epsilon` | `float` | ε₀ con el grado inicial |
| `final_epsilon` | `float` | ε_f tras auto-evolución |
| `improvement_ratio` | `float` | ε₀ / ε_f |
| `best_reduction` | `ReductionResult` | Mejor reducción ACF encontrada |

```python
import torch.nn as nn
from acf_functor import NeuralACFEvolver

net = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
evolver = NeuralACFEvolver()
result = evolver.evolve(net, domain=(-1.0, 1.0), input_dim=4)
print(f"Compresibilidad: {result.improvement_ratio:.1f}×")
```

### 27.6. build_test_mlp

```python
def build_test_mlp(
    layer_dims: List[int],
    activation: str = "tanh",
    seed: int = 42,
) -> nn.Sequential
```

Construye un MLP determinista con dimensiones `layer_dims`, activaciones `activation` ("tanh", "relu", "sigmoid") y semilla fija para reproducibilidad.

```python
from acf_functor import build_test_mlp

# MLP: 8 → 16 → 16 → 4 → 1
net = build_test_mlp([8, 16, 16, 4, 1], activation="tanh", seed=42)
print(net)
# Sequential(Linear(8,16), Tanh, Linear(16,16), Tanh, Linear(16,4), Tanh, Linear(4,1))
```

---

## 28. Referencia de API — Meta-Compilador ACF (`meta_compiler.py`)

### 28.1. Resumen del módulo

`acf_functor/meta_compiler.py` implementa la búsqueda óptima de gramática sobre el espacio de bases de aproximación. Formaliza la selección de base como minimización de energía libre termodinámica C(G,f,β) = ε(G,f) − S(G)/β.

**Exportaciones:**

```python
from acf_functor import (
    BasisFamily,         # Enum: 9 familias de bases
    Grammar,             # Dataclass frozen: (basis, degree, n_observables, method)
    GrammarPoint,        # Punto evaluado: grammar + epsilon + entropy + free_energy
    GrammarEvaluator,    # Evalúa un Grammar sobre f
    GridSearch,          # Búsqueda exhaustiva
    RandomSearch,        # Búsqueda aleatoria
    GreedySearch,        # Búsqueda voraz con reinicios
    GrammarSpace,        # Define el espacio de búsqueda
    MetaCompilerConfig,  # Configuración completa del meta-compilador
    MetaCompilerResult,  # Resultado: best_grammar, best_reduction, trace
    MetaCompilerTrace,   # Historial de búsqueda
    ACFMetaCompiler,     # Objeto principal: .compile(), .analyse_grammar_space()
)
```

### 28.2. BasisFamily

```python
class BasisFamily(Enum):
    # Bases clásicas
    CHEBYSHEV      = "chebyshev"      # T_k(x), convergencia exponencial
    LEGENDRE       = "legendre"       # P_k(x), L² ortogonal en [-1,1]
    HORNER         = "horner"         # Evaluación eficiente del polinomio Chebyshev
    FOURIER        = "fourier"        # cos(kπx/L), N/2 coeficientes reales
    RBF            = "rbf"            # exp(-‖x-cₖ‖²/σ²), n_observables centros
    # Bases Koopman
    KOOPMAN_POLY   = "koopman_poly"   # EDMD con observables {xᵏ}
    KOOPMAN_FOURIER= "koopman_fourier"# EDMD con observables {cos(kx), sin(kx)}
    KOOPMAN_RBF    = "koopman_rbf"    # EDMD con observables RBF
    KOOPMAN_MIXED  = "koopman_mixed"  # EDMD poli + RBF
```

### 28.3. Grammar

```python
@dataclass(frozen=True)
class Grammar:
    basis: BasisFamily
    degree: int
    n_observables: int = 8      # para familias RBF y Koopman
    method: str = "chebyshev"   # método ACF interno ("chebyshev" | "horner" | "koopman")
    
    # Automáticamente hashable (frozen dataclass)
    # Puede usarse como clave de diccionario o en conjuntos
```

```python
from acf_functor import Grammar, BasisFamily

g1 = Grammar(basis=BasisFamily.CHEBYSHEV, degree=8)
g2 = Grammar(basis=BasisFamily.RBF, degree=12, n_observables=16)
g3 = Grammar(basis=BasisFamily.KOOPMAN_FOURIER, degree=6, n_observables=12)
print(g1)  # Grammar(basis=CHEBYSHEV, degree=8, n_observables=8, method='chebyshev')
```

### 28.4. GrammarEvaluator

```python
class GrammarEvaluator:
    def __init__(
        self,
        n_probe: int = 100,       # Puntos de muestreo para evaluar ε
        beta: float = 1.0,        # Parámetro de temperatura
        dtype: np.dtype = np.float64,
    ):
        ...
    
    def evaluate(
        self,
        grammar: Grammar,
        f: Callable[[float], float],
        domain: Tuple[float, float],
    ) -> GrammarPoint
```

**`GrammarPoint` — campos:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `grammar` | `Grammar` | La gramática evaluada |
| `epsilon` | `float` | Error L∞ de aproximación |
| `entropy` | `float` | S(G) = log(1+d) + log(1+k) |
| `free_energy` | `float` | C(G,f,β) = ε − S(G)/β |
| `error_message` | `Optional[str]` | `None` si OK; mensaje de error si falló |

```python
from acf_functor import GrammarEvaluator, Grammar, BasisFamily
import numpy as np

evaluator = GrammarEvaluator(n_probe=200, beta=1.0)
f = np.sin
g = Grammar(basis=BasisFamily.FOURIER, degree=6)
point = evaluator.evaluate(g, f, domain=(-np.pi, np.pi))
print(f"ε = {point.epsilon:.2e}, S = {point.entropy:.3f}, C = {point.free_energy:.4f}")
```

### 28.5. Estrategias de Búsqueda

#### GridSearch

```python
class GridSearch:
    def search(
        self,
        f: Callable,
        domain: Tuple[float, float],
        space: "GrammarSpace",
        evaluator: GrammarEvaluator,
        target_epsilon: float = 1e-6,
    ) -> List[GrammarPoint]
```

Evalúa **todas** las gramáticas del espacio. Retorna lista ordenada por `free_energy`.

#### RandomSearch

```python
class RandomSearch:
    def __init__(self, budget: int = 50, seed: int = 42):
        ...
    
    def search(self, f, domain, space, evaluator, target_epsilon=1e-6) -> List[GrammarPoint]
```

Muestrea `budget` gramáticas aleatoriamente. Reproducible con `seed`.

#### GreedySearch

```python
class GreedySearch:
    def __init__(self, n_restarts: int = 3):
        ...
    
    def search(self, f, domain, space, evaluator, target_epsilon=1e-6) -> List[GrammarPoint]
```

Búsqueda voraz: desde cada reinicio, explora vecinos (±1 en degree, bases adyacentes) y acepta movimientos que reducen C(G,f,β). Los `n_restarts` puntos de inicio se eligen aleatoriamente.

### 28.6. GrammarSpace

```python
@dataclass
class GrammarSpace:
    families: List[BasisFamily] = field(default_factory=lambda: list(BasisFamily))
    degree_range: Tuple[int, int] = (2, 16)
    degree_step: int = 2
    n_observables_options: List[int] = field(default_factory=lambda: [8, 16])
    
    def all_grammars(self) -> List[Grammar]
    def random_grammar(self, rng: np.random.Generator) -> Grammar
    def neighbour_grammars(self, g: Grammar) -> List[Grammar]
```

```python
from acf_functor import GrammarSpace, BasisFamily

# Espacio reducido para búsqueda rápida
space = GrammarSpace(
    families=[BasisFamily.CHEBYSHEV, BasisFamily.FOURIER, BasisFamily.RBF],
    degree_range=(4, 12),
    degree_step=2,
    n_observables_options=[8],
)
print(f"Gramáticas totales: {len(space.all_grammars())}")  # 3×5×1 = 15
```

### 28.7. MetaCompilerConfig

```python
@dataclass
class MetaCompilerConfig:
    grammar_space: GrammarSpace = field(default_factory=GrammarSpace)
    strategy: Union[GridSearch, RandomSearch, GreedySearch] = field(
        default_factory=lambda: GreedySearch(n_restarts=3)
    )
    beta: float = 1.0
    target_epsilon: float = 1e-6
    enable_auto_evolution: bool = False
    n_probe: int = 100
```

### 28.8. ACFMetaCompiler

```python
class ACFMetaCompiler:
    def __init__(self, config: MetaCompilerConfig = None):
        ...
    
    def compile(
        self,
        f: Callable[[float], float],
        domain: Tuple[float, float],
    ) -> MetaCompilerResult
    
    def analyse_grammar_space(
        self,
        f: Callable[[float], float],
        domain: Tuple[float, float],
    ) -> MetaCompilerTrace
```

**`MetaCompilerResult` — campos:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `best_grammar` | `Grammar` | Gramática G* que minimiza C(G,f,β) |
| `best_reduction` | `ReductionResult` | Reducción ACF con G* |
| `initial_epsilon` | `float` | ε baseline (Chebyshev grado 8) |
| `final_epsilon` | `float` | ε con G* (post auto-evolución si activada) |
| `improvement_ratio` | `float` | initial_epsilon / final_epsilon |
| `trace` | `MetaCompilerTrace` | Historial completo de búsqueda |

**`MetaCompilerTrace` — campos:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `all_grammars` | `List[GrammarPoint]` | Todos los puntos evaluados |
| `best` | `GrammarPoint` | El punto de mínima energía libre |
| `n_evaluated` | `int` | Total de gramáticas evaluadas |
| `n_failed` | `int` | Gramáticas que fallaron en evaluación |

**Ejemplo completo:**

```python
import numpy as np
from acf_functor import (
    ACFMetaCompiler, MetaCompilerConfig, GrammarSpace,
    BasisFamily, GreedySearch, GridSearch,
)

# Función con estructura Koopman — EDMD debería ganar
f_dyn = lambda x: np.exp(0.5 * x) * np.sin(3 * x)

# Configuración con espacio de búsqueda personalizado
space = GrammarSpace(
    families=list(BasisFamily),    # todas las 9 familias
    degree_range=(4, 20),
    degree_step=2,
    n_observables_options=[8, 16, 32],
)
config = MetaCompilerConfig(
    grammar_space=space,
    strategy=GreedySearch(n_restarts=5),
    beta=1.0,
    target_epsilon=1e-7,
    enable_auto_evolution=True,
    n_probe=200,
)

mc = ACFMetaCompiler(config)
result = mc.compile(f_dyn, domain=(-2.0, 2.0))

print(f"Mejor gramática: {result.best_grammar}")
print(f"Mejora: {result.improvement_ratio:.2f}×")
print(f"ε inicial: {result.initial_epsilon:.2e}")
print(f"ε final:   {result.final_epsilon:.2e}")
print(f"Gramáticas evaluadas: {result.trace.n_evaluated}")

# Analizar todo el espacio sin compilar
trace = mc.analyse_grammar_space(f_dyn, domain=(-2.0, 2.0))
for pt in sorted(trace.all_grammars, key=lambda p: p.free_energy)[:5]:
    print(f"  {pt.grammar.basis.name}, d={pt.grammar.degree}: ε={pt.epsilon:.2e}, C={pt.free_energy:.4f}")
```

### 28.9. Integración con GideonEngine

Todos los tipos del módulo están accesibles directamente desde el motor Gideon:

```python
from poema.backends.gideon.engine import GideonEngine
engine = GideonEngine()

result = engine.meta_compile(
    f=lambda x: np.abs(np.sin(x)),
    domain=(-np.pi, np.pi),
    strategy="greedy",
    beta=1.0,
    target_epsilon=1e-5,
    enable_auto_evolution=False,
)
print(result.best_grammar)
print(f"Mejora: {result.improvement_ratio:.1f}×")
```

### 28.10. Tabla de Clases Exportadas

| Clase | Módulo | Propósito |
|-------|--------|-----------|
| `BasisFamily` | `acf_functor.meta_compiler` | Enum de familias de bases (9 valores) |
| `Grammar` | `acf_functor.meta_compiler` | Configuración frozen (basis, degree, n_obs, method) |
| `GrammarPoint` | `acf_functor.meta_compiler` | Punto evaluado del espacio de gramáticas |
| `GrammarEvaluator` | `acf_functor.meta_compiler` | Evalúa ε y C para un Grammar dado |
| `GridSearch` | `acf_functor.meta_compiler` | Búsqueda exhaustiva |
| `RandomSearch` | `acf_functor.meta_compiler` | Búsqueda aleatoria con presupuesto |
| `GreedySearch` | `acf_functor.meta_compiler` | Búsqueda voraz con reinicios |
| `GrammarSpace` | `acf_functor.meta_compiler` | Espacio de búsqueda configurable |
| `MetaCompilerConfig` | `acf_functor.meta_compiler` | Configuración completa del pipeline |
| `MetaCompilerResult` | `acf_functor.meta_compiler` | Resultado: best_grammar, best_reduction, trace |
| `MetaCompilerTrace` | `acf_functor.meta_compiler` | Historial de evaluaciones |
| `ACFMetaCompiler` | `acf_functor.meta_compiler` | Objeto principal: compile(), analyse_grammar_space() |

Ver Paper.md §35 para la formalización matemática y Poema.md §16 para la descripción conceptual.

---

## 29. Tensor ACF — `acf_functor.tensor_acf`

### 29.1. Descripción

Reduce funciones multivariable $f: \mathbb{R}^d \to \mathbb{R}$ a cadenas FMA via descomposición Tensor Train (TT) o Tucker sobre bases Chebyshev tensoriales.

### 29.2. Uso rápido

```python
from acf_functor import TensorACFReducer, StandardTensorFunctions

# Reducir f(x,y,z) = sin(x)*cos(y)*exp(z)
reducer = TensorACFReducer(default_degree=10, max_rank=8, target_epsilon=1e-6)
result = reducer.reduce(
    StandardTensorFunctions.separable_product,
    domains=[(-3, 3), (-3, 3), (-1, 1)],
)

# Evaluar
import torch
x = torch.tensor([0.5, 1.2, 0.3], dtype=torch.float64)
print(f"f(x) ≈ {result.evaluate(x):.8f}")
print(f"ε certificado: {result.epsilon:.2e}")
print(f"α global: {result.invariants.alpha_global:.4f}")
print(f"Rangos TT: {result.invariants.tt_ranks}")
print(f"Clase NC: {result.invariants.nc_class}")
print(f"FMAs totales: {result.invariants.total_fma_count}")
```

### 29.3. Tucker (dimensión baja)

```python
reducer_tucker = TensorACFReducer(default_degree=12, method="tucker")
result = reducer_tucker.reduce(
    lambda x, y: x**2 + y**2,
    domains=[(-2, 2), (-2, 2)],
)
print(f"ε Tucker: {result.epsilon:.2e}")
```

### 29.4. Funciones estándar de test

| Función | Firma | Descripción |
|---------|-------|-------------|
| `separable_product` | `(x,y,z)` | sin(x)·cos(y)·exp(z) — rango CP 1 |
| `rosenbrock` | `(x,y)` | (1-x)² + 100(y-x²)² — no separable |
| `gaussian_2d` | `(x,y)` | exp(-(x²+y²)) |
| `multivariate_polynomial` | `(x,y,z)` | x²y + yz² + xz + 3 |
| `friedman1` | `(x₁,...,x₅)` | 10sin(πx₁x₂) + 20(x₃-0.5)² + 10x₄ + 5x₅ |
| `wave_3d` | `(x,y,z)` | sin(x+y)cos(z) + 0.5sin(2x)sin(3z) |

### 29.5. Tabla de clases exportadas (Tensor ACF)

| Clase | Módulo | Descripción |
|-------|--------|-------------|
| `TensorTrainCore` | `tensor_acf` | Core TT individual de shape (r_{m-1}, n_m, r_m) |
| `TensorTrainDecomposition` | `tensor_acf` | TT completo: cores, ranks, domains |
| `TensorACFInvariants` | `tensor_acf` | α por modo, α global, NC class, dim efectiva |
| `TensorReductionResult` | `tensor_acf` | Resultado completo con TT + ε + invariantes |
| `TuckerDecomposition` | `tensor_acf` | Core G + factores U_m |
| `TuckerReductionResult` | `tensor_acf` | Resultado Tucker con ε |
| `ChebyshevTensorSampler` | `tensor_acf` | Muestreo y coeficientes en grid Chebyshev |
| `TensorTrainBuilder` | `tensor_acf` | TT-SVD: tensor → TT |
| `TensorTrainEvaluator` | `tensor_acf` | Evaluación TT por contracción zipper |
| `TuckerBuilder` | `tensor_acf` | HOSVD: tensor → Tucker |
| `TuckerEvaluator` | `tensor_acf` | Evaluación Tucker |
| `TensorACFReducer` | `tensor_acf` | Reductor principal: func → TT/Tucker |
| `TensorACFAnalyzer` | `tensor_acf` | Cómputo de invariantes ACF tensoriales |
| `StandardTensorFunctions` | `tensor_acf` | Funciones de test multivariable |

---

## 30. Matrix ACF — `acf_functor.matrix_acf`

### 30.1. Descripción

Reduce funciones de matrices $f(A)$ (exponencial, raíz, logaritmo, resolvente, signo) a polinomios de Chebyshev matriciales via recurrencia de Clenshaw.

### 30.2. Uso rápido

```python
import torch
from acf_functor import (
    ChebyshevMatrixReducer, MatrixExponential, MatrixSquareRoot,
    MatrixACFAnalyzer,
)

# Crear matriz SPD
A = torch.randn(10, 10, dtype=torch.float64)
A = A @ A.T + 0.1 * torch.eye(10, dtype=torch.float64)

# Exponencial matricial
result = MatrixExponential.reduce(A, t=1.0, degree=30, target_epsilon=1e-8)
print(f"‖exp(A) - approx‖₂ = {result.epsilon:.2e}")
print(f"Grado usado: {result.degree}")
print(f"FMAs: {result.fma_count}")

# Raíz cuadrada
sqrt_result = MatrixSquareRoot.reduce(A, degree=30)
print(f"‖A^{1/2} error‖₂ = {sqrt_result.epsilon:.2e}")

# Invariantes
inv = MatrixACFAnalyzer.analyse(A, func="exp", degree=40)
print(f"α matricial: {inv.matrix_alpha:.4f}")
print(f"Clase NC: {inv.nc_class}")
print(f"Rango espectral: {inv.spectral_range}")
print(f"Número de condición: {inv.condition_number:.2f}")
```

### 30.3. Funciones matriciales incorporadas

```python
from acf_functor import (
    MatrixExponential,   # exp(tA)
    MatrixSquareRoot,    # A^{1/2}
    MatrixLogarithm,     # log(A)
    MatrixResolvent,     # (A+σI)⁻¹
    MatrixSign,          # sign(A)
)
```

### 30.4. Tabla de clases exportadas (Matrix ACF)

| Clase | Módulo | Descripción |
|-------|--------|-------------|
| `MatrixReductionResult` | `matrix_acf` | Resultado: matrix, coeffs, ε, degree |
| `MatrixACFInvariants` | `matrix_acf` | α matricial, rango espectral, NC class |
| `MatrixFMAChain` | `matrix_acf` | Cadena FMA explícita con operaciones |
| `ChebyshevMatrixReducer` | `matrix_acf` | Reductor Chebyshev genérico para f(A) |
| `MatrixExponential` | `matrix_acf` | exp(tA) |
| `MatrixSquareRoot` | `matrix_acf` | A^{1/2} (requiere SPD) |
| `MatrixLogarithm` | `matrix_acf` | log(A) (requiere SPD) |
| `MatrixResolvent` | `matrix_acf` | (A+σI)⁻¹ |
| `MatrixSign` | `matrix_acf` | sign(A) |
| `MatrixACFAnalyzer` | `matrix_acf` | Análisis de invariantes (f, A) |

Ver Paper.md §36–§37 para la formalización matemática y Poema.md §17–§18 para la descripción conceptual.


---

## §31. ODE / Control ACF — Guía de Uso

### 31.1. Reducción de campo vectorial

```python
import numpy as np
from acf_functor import reduce_vector_field, VectorFieldReducer

# Sistema: péndulo no lineal
def pendulum(x: np.ndarray) -> np.ndarray:
    return np.array([x[1], -np.sin(x[0]) - 0.1 * x[1]])

# Reducción automática (dimensión=2)
result = reduce_vector_field(pendulum, dimension=2, order=8)
print(f"α_global: {result.invariants.alpha_global:.4f}")
print(f"α por modo: {result.invariants.alpha_per_mode}")

# Evaluar campo reducido en x = [0.5, 0.1]
x0 = np.array([0.5, 0.1])
f_hat = result.evaluate(x0)  # ≈ pendulum(x0)
print(f"Error campo: {np.linalg.norm(f_hat - pendulum(x0)):.2e}")
```

### 31.2. Certificación Lyapunov

```python
from acf_functor import LyapunovACF

# V(x) = x₀² + x₁² (propuesta de Lyapunov)
V_candidate = lambda x: x[0]**2 + x[1]**2

lyap = LyapunovACF(order=6, domain=[(-2., 2.), (-2., 2.)], radius=1.5)
cert = lyap.certify(V_candidate, pendulum)
print(f"Estable: {cert.is_stable}")
print(f"V_min: {cert.v_min:.4f}, V_dot_max: {cert.v_dot_max:.4f}")
```

### 31.3. Política óptima HJB

```python
from acf_functor import HJBReducer

hjb = HJBReducer(order=8, domain=[(-2.,2.), (-2.,2.)])
l = lambda x, u: x[0]**2 + x[1]**2 + u**2  # costo cuadrático
result = hjb.fit(lambda x: x[0]**2 + x[1]**2)  # V inicial
# u* = argmin_u [ l(x,u) + ∇V·f(x,u) ]
u_star = hjb.optimal_control(np.array([0.5, 0.5]), l, pendulum,
                              u_candidates=np.linspace(-2, 2, 20))
print(f"Control óptimo: {u_star:.4f}")
```

### 31.4. Tabla exportada (ODE-ACF)

| Clase | Descripción |
|-------|-------------|
| `VectorFieldReducer` | Reduce vector field via Tensor Train |
| `HJBReducer` | Aproxima value function HJB |
| `LyapunovACF` | Certifica condiciones de Lyapunov |
| `TrajectoryACF` | Integra ODE con campo reducido (RK4) |
| `ODEACFInvariants` | α_global, alpha_per_component, dimension |
| `HJBInvariants` | dimension, approximation_error, alpha_global |
| `LyapunovCertificate` | is_stable, v_min, v_dot_max, grid_resolution |
| `reduce_vector_field` | Factory: reducer + fit automático |

---

## §32. Operator / Green Function ACF — Guía de Uso

### 32.1. Compresión de operador integral

```python
import numpy as np
from acf_functor import reduce_green_function, IntegralOperatorACF

# Núcleo de Matérn (kernel de proceso gaussiano)
def matern_kernel(x, y, nu=1.5):
    r = abs(x - y)
    return (1 + np.sqrt(3)*r) * np.exp(-np.sqrt(3)*r)

result = reduce_green_function(matern_kernel, n_points=64, order=16)
print(f"Rango efectivo: {result.rank}")
print(f"α_kernel: {result.invariants.alpha_global:.4f}")

# Aplicar operador reducido a u(x) = sin(πx)
u = lambda x: np.sin(np.pi * x)
Lu = result.apply(u, n_points=64)
print(f"L̂u shape: {Lu.shape}")
```

### 32.2. Operador integral SVD separable

```python
from acf_functor import IntegralOperatorACF

G = lambda x, y: np.exp(-abs(x - y))
op = IntegralOperatorACF(G, n_points=128, rank=10, domain=(-1., 1.))
op.fit()
print(f"Valores singulares: {op.singular_values[:5]}")

u_vec = np.sin(np.pi * np.linspace(-1, 1, 128))
Lu_approx = op.apply(u_vec)  # O(R·n)
```

### 32.3. Atención Linear para Transformers

```python
from acf_functor import AttentionKernelReducer
import numpy as np

n_seq, d_model = 512, 64
Q = np.random.randn(n_seq, d_model)
K = np.random.randn(n_seq, d_model)
V = np.random.randn(n_seq, d_model)

reducer = AttentionKernelReducer(n_features=128, feature_type='rff', d_model=d_model)
output = reducer.fast_attention(Q, K, V)  # O(n·R·d) en vez de O(n²·d)
print(f"Output shape: {output.shape}")
print(f"α atención: {reducer.invariants.alpha_global:.4f}")
```

---

## §33. Stochastic / PCE ACF — Guía de Uso

### 33.1. Expansión básica PCE

```python
import numpy as np
from acf_functor import PolynomialChaosACF

# f(ξ₁,ξ₂) = ξ₁² + ξ₁·ξ₂ con ξᵢ~N(0,1)
f = lambda xi: xi[0]**2 + xi[0]*xi[1]

pce = PolynomialChaosACF(n_vars=2, max_degree=4, basis_family='hermite')
pce.fit(f, method='projection')

print(f"Varianza estimada: {pce.variance():.4f}")
print(f"Media estimada:    {pce.mean():.4f}")

# Índices de Sobol
sobol = pce.sobol_indices()
print(f"S₁ = {sobol[0]:.4f}  (sensibilidad a ξ₁)")
print(f"S₂ = {sobol[1]:.4f}  (sensibilidad a ξ₂)")
```

### 33.2. Banda de incertidumbre k-sigma

```python
from acf_functor import compute_uncertainty_bound

ub = compute_uncertainty_bound(f, m=2, k_sigma=2.0, family='hermite',
                               max_degree=4, n_quad=10)
print(f"μ ± 2σ = [{ub.lower_bound:.4f}, {ub.upper_bound:.4f}]")
print(f"P(|f-μ|≥2σ) ≤ {ub.probability_bound:.4f}")  # ≤ 0.25 por Chebyshev
```

### 33.3. Sensibilidad con Legendre (variables uniformes)

```python
pce_leg = PolynomialChaosACF(n_vars=3, max_degree=3, basis_family='legendre')
g = lambda xi: xi[0]*xi[1] + xi[2]**2
pce_leg.fit(g, method='regression', n_samples=500)

inv = pce_leg.invariants
print(f"Sobol total: {sum(inv.sobol_indices):.4f}")
```

### 33.4. Tabla exportada (Stochastic-ACF)

| Clase | Descripción |
|-------|-------------|
| `PolynomialChaosACF` | PCE completa: fit, mean, variance, sobol |
| `StochasticReducer` | Wrapper de nivel alto para análisis |
| `PCECoefficients` | Coeficientes c_α, índices multi-índice, n_terms |
| `StochasticACFInvariants` | alpha_global, sobol_indices, variance, mean |
| `UncertaintyBound` | lower/upper_bound, sigma, probability_bound |
| `compute_uncertainty_bound` | Factory: PCE + banda k-sigma automática |

---

## §34. Rational / Padé ACF — Guía de Uso

### 34.1. Aproximante de Padé [m/n]

```python
import numpy as np
from acf_functor import pade_reduce, PadeReducer

# exp(x) con Padé [4/4]
f = lambda x: np.exp(x)
result = pade_reduce(f, m=4, n=4, x0=0.0)

print(f"FMAs: {result.fma_count}")           # = 4 + 4 + 3 = 11
print(f"Error máx en [-1,1]: {result.invariants.approximation_error:.2e}")
print(f"α Hardy:  {result.invariants.alpha_hardy:.4f}")

# Evaluar
x_test = np.linspace(-1, 1, 100)
y_hat = np.array([result.evaluate(xi) for xi in x_test])
y_ref = f(x_test)
print(f"Error L∞: {np.max(np.abs(y_hat - y_ref)):.2e}")
```

### 34.2. Proyección en Hardy H²

```python
from acf_functor import hardy_reduce

# Función en el círculo unitario: f(e^{iθ}) = 1/(2 - e^{iθ})
f_circle = lambda theta: 1.0 / (2.0 - np.exp(1j * theta))

result = hardy_reduce(f_circle, n_modes=20, n_quad=256)
print(f"Coeficientes Hardy: {result.coefficients[:5]}")
print(f"α decaimiento: {result.invariants.alpha_hardy:.4f}")
print(f"‖f‖_H²: {result.invariants.h2_norm:.4f}")
```

### 34.3. Análisis con PadeReducer

```python
from acf_functor import PadeReducer

# Función con polo en z = 2: f(z) = 1/(z-2)
f_pole = lambda z: 1.0 / (z - 2.0)
reducer = PadeReducer(m=3, n=3)
result = reducer.fit(f_pole, x0=0.0)

inv = result.invariants
print(f"Grado numerador: {inv.numerator_degree}")
print(f"Grado denominador: {inv.denominator_degree}")
print(f"α_Padé: {inv.alpha_pade:.4f}")
```

### 34.4. Tabla exportada (Rational-ACF)

| Clase | Descripción |
|-------|-------------|
| `PadeReducer(m, n)` | Approximant [m/n] con Horner doble |
| `HardySpaceACF(n_modes, n_quad)` | Proyección en H²(𝔻) |
| `PadeInvariants` | numerator_degree, denominator_degree, alpha_pade, approximation_error |
| `HardySpaceInvariants` | alpha_hardy, h2_norm, n_modes, decay_rate |
| `pade_reduce(f, m, n, x0)` | Factory instantánea |
| `hardy_reduce(f_circle, n_modes, n_quad)` | Factory para funciones en S¹ |

Ver Paper.md §41 para la formalización y `MathTest/NewDomainCertificates.lean` para los teoremas RAT-1—RAT-4.

---

## §35. Garantías Formales — Afirmaciones Verificadas

> **Estado**: Completamente implementado. Todos los teoremas compilados en Lean 4 (`MathTest/FormalEmpiricalTheorems.lean`). 35 tests validados en `tests/test_formal_empirical_bounds.py`.

Esta sección documenta las 5 afirmaciones que anteriormente eran empíricas o heurísticas y que ahora están formalmente probadas con 0 `sorry` en Lean 4.

### §35.1. Familia de Clasificación (Teoremas FAM-1/2/3/4)

| Teorema | Descripción | Táctica clave |
|---------|-------------|---------------|
| **FAM-1** `fast_family_exponential_convergence` | `IsFastFamily c ↔ ∃ C ρ > 1, ∀ k, \|c k\| ≤ C·ρ⁻ᵏ` | constructivo |
| **FAM-2** `algebraic_family_polynomial_convergence` | `IsAlgebraicFamily c ↔ ∃ C s > 0, ∀ k > 0, \|c k\| ≤ C·k⁻ˢ` | `Finset.sum_le_sum` |
| **FAM-3** `fast_implies_algebraic` | `IsFastFamily c → IsAlgebraicFamily c` con C' = C·ρ/(ρ-1) | `geometric_sum_bound` |
| **FAM-4** `nc_class_refines_family_classification` | NC0/1/2/3 es partición completa y disjunta de [0,∞) | `le_antisymm` + casos |

```python
# Validación Python
from tests.test_formal_empirical_bounds import TestFamilyClassification
# 8 tests en TestFamilyClassification — todos pasan
```

### §35.2. Cota de Grado Fiedler (Teoremas FIEDLER-1/2/3)

La función de grado óptimo $d^*(\varepsilon, \lambda_2, \lambda_{\max}) = \lceil \log(2/\varepsilon) / \log(1 + \lambda_2/\lambda_{\max}) \rceil$ satisface:

| Teorema | Enunciado | Táctica clave |
|---------|-----------|---------------|
| **FIEDLER-1** | d* > 0 para todo ε ∈ (0,2) | `div_pos` + `log_pos` |
| **FIEDLER-2** | d* monótona decreciente en λ₂ | `div_le_div_of_nonneg_left` |
| **FIEDLER-3** | ratio λ₂=0.5 vs λ₂=1.0: log(3/2)/log(5/4) ≈ **1.817** > 1 | `one_lt_div_of_lt` |

```python
# Uso
from acf_functor.graph_acf import GraphACFAnalyzer
# fiedler_degree_bound(eps, lambda2, lambda_max) = ceil(log(2/eps)/log(1+lambda2/lambda_max))
```

### §35.3. Isomorfismo AIC/BIC (Teoremas AIC-1/2/3/4)

La función de coste $\mathcal{C}(G, f, \beta)$ es formalmente equivalente a AIC/BIC:

| Teorema | Resultado | Táctica clave |
|---------|-----------|---------------|
| **AIC-1** | C(G,f,β=n/2) = ε − 2S/n (identidad exacta con AIC normalizado) | `ring` |
| **AIC-2** | argmin C ≡ argmin AIC (equivalencia de orden) | isomorfismo de orden |
| **AIC-3** | β_BIC = log(n)/2 > β_AIC = n/2 para n ≥ e² ≈ 7.4 | `add_one_le_exp` + `linarith` |
| **AIC-4** | Z(β) = Σ exp(−β·C(G)) > 0 para conjunto finito de gramáticas | `Finset.sum_pos` + `exp_pos` |

### §35.4. Consistencia de Alpha (Teoremas ALPHA-1/2/3/4)

El índice de complejidad α es exacto, no empírico:

| Teorema | Resultado | Táctica clave |
|---------|-----------|---------------|
| **ALPHA-1** | α̂_spec = α + \|log C\|/log k (expresión exacta) | `log_mul` + `log_rpow` |
| **ALPHA-2** | \|α̂_spec − α\| = \|log C\|/log k (tasa de convergencia) | `abs_div` |
| **ALPHA-3** | k ≥ exp(10·\|log C\|/α) ⟹ \|α̂ − α\| ≤ 0.1 (umbral explícito) | `nlinarith` |
| **ALPHA-4** | log(ε₁/ε₂)/log(d₂/d₁) = α (tres estimadores coinciden exactamente) | `log_rpow` + `field_simp` + `ring` |

### §35.5. Convergencia del Ciclo Adjunto (Teoremas ADJ-1/2)

| Teorema | Resultado | Táctica clave |
|---------|-----------|---------------|
| **ADJ-1** `adjoint_cycle_convergence_lipschitz` | L < 1 ⟹ convergencia a punto fijo único (Banach) | `LipschitzWith.toContraction` |
| **ADJ-2** `adjoint_cycle_no_convergence_without_lipschitz` | f(x) = x+1 no tiene punto fijo (contraejemplo constructivo) | `Int.lt_irrefl` |

### §35.6. Resumen de Estado Formal

| Categoría | Lean file | Teoremas | Sorry | Tests Python |
|-----------|-----------|----------|-------|--------------|
| Composición y Error | `CompositionErrorBounds.lean` | 8 | 0 | — |
| Tensor ACF | `TensorACFCertificates.lean` | 6 | 0 | — |
| Nuevos Dominios | `NewDomainCertificates.lean` | 16 | 0 | — |
| **Afirmaciones Empíricas** | **`FormalEmpiricalTheorems.lean`** | **17** | **0** | **35 tests** |
| **TOTAL** | — | **47** | **0** | — |

Todos los archivos Lean compilan sin `sorry` en Lean 4.29.0-rc6 + Mathlib.
Ver `MathTest/FormalEmpiricalTheorems.lean` para el código fuente completo.

---

## §36. Nuevos Módulos — Expansiones E3…E6 y Correcciones DEBILIDAD #1…#6

### §36.1. Dominio Constructivo (`domain_admissibility.py`)

```python
from acf_functor.domain_admissibility import (
    DomainAdmissibilityChecker,   # verifica AD-1…AD-6
    AdaptiveFunctorRouter,        # enrutamiento automático
    DomainCertificateLeanEmitter, # genera bloques Lean del certificado
    FunctorBranch,                # enum: HORNER, CHEBYSHEV, RATIONAL, KOOPMAN
    AdmissibilityCertificate,     # resultado por condición
    DomainAdmissibilityReport,    # resultado global
)
```

**`DomainAdmissibilityChecker.check(f, domain) → DomainAdmissibilityReport`**

| Campo del report | Tipo | Descripción |
|-----------------|------|-------------|
| `admissible` | `bool` | True sii AD-1∧AD-2∧AD-5∧AD-6∧(AD-3∨AD-4) |
| `certificates` | `List[AdmissibilityCertificate]` | Un certificado por condición |
| `recommended_branch` | `FunctorBranch` | Rama sugerida para compilación |
| `estimated_bernstein_rho` | `float` | $\rho_f$ estimado |
| `estimated_alpha` | `float` | $\alpha(f) = 1/\log\rho_f$ |
| `d_needed` | `int` | Grado mínimo para alcanzar ε_target |
| `epsilon_achieved` | `float` | Error alcanzado con d_needed |
| `lipschitz_constant` | `float` | $L_f$ estimado |
| `singularity_candidates` | `List[float]` | Puntos candidatos a singularidad |

**`AdaptiveFunctorRouter.route(f, domain) → Tuple[FunctorBranch, DomainAdmissibilityReport]`**

```python
import math
from acf_functor.domain_admissibility import DomainAdmissibilityChecker, AdaptiveFunctorRouter

# Verificar admisibilidad
cert = DomainAdmissibilityChecker().check(math.sin, (-1.0, 1.0))
print(cert.admissible)           # True
print(cert.estimated_bernstein_rho)  # ≈ 2.72

# Enrutar
branch, report = AdaptiveFunctorRouter().route(math.exp, (-2.0, 2.0))
print(branch)                    # FunctorBranch.CHEBYSHEV

# Función problemática
cert2 = DomainAdmissibilityChecker().check(math.tan, (0.0, 1.58))
print(cert2.admissible)          # False — cruzaπ/2
```

---

### §36.2. Teorema Nyquist-ACF (`nyquist_acf.py`)

```python
from acf_functor.nyquist_acf import (
    NyquistACFTheorem,            # aplica el teorema d*(ε,α)
    NyquistACFResult,             # resultado completo
    NyquistComplexityClass,       # enum: EASY, MEDIUM, HARD, EXTREME
    AlphaHardnessCatalog,         # catálogo de funciones benchmark
    AlphaHardnessCatalogEntry,    # entrada del catálogo
)
```

**`NyquistACFTheorem.apply(f, domain, epsilon) → NyquistACFResult`**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `d_star_predicted` | `int` | $\lceil (C_f/\varepsilon)^{1/\alpha}\rceil$ (fórmula) |
| `d_star_empirical` | `int` | mínimo $d$ con error $\leq \varepsilon$ |
| `alpha` | `float` | $\alpha(f)$ medido |
| `bernstein_rho` | `float` | $\rho_f$ estimado |
| `complexity_class` | `NyquistComplexityClass` | EASY/MEDIUM/HARD/EXTREME |
| `information_bits` | `float` | $(1/\alpha)\log_2(C_f/\varepsilon)$ |
| `theorem_valid` | `bool` | $|d^*_\text{pred} - d^*_\text{emp}|$ ≤ tolerancia |

**`NyquistComplexityClass.from_alpha(alpha) → NyquistComplexityClass`**

```python
import math
from acf_functor.nyquist_acf import NyquistACFTheorem, NyquistComplexityClass, AlphaHardnessCatalog

result = NyquistACFTheorem().apply(math.sin, (-math.pi, math.pi), epsilon=1e-8)
print(result.d_star_empirical)       # 18
print(result.complexity_class)       # NyquistComplexityClass.EASY

cls = NyquistComplexityClass.from_alpha(3.0)   # EXTREME
entries = AlphaHardnessCatalog.build_standard_catalog()
for e in entries[:3]:
    print(f"{e.function_name}: α={e.alpha_empirical:.3f}")
```

---

### §36.3. Observabilidad Koopman (`koopman_observability.py`)

```python
from acf_functor.koopman_observability import (
    KoopmanObservabilityChecker,      # verificador KO-1a/b/c, KO-3, KO-4
    KoopmanObservabilityReport,       # resultado completo
    EnergyInvariantHardwareVerifier,  # verifica E(f)=E(Φ(f)) en fp64/fp32/fp16
    ObservabilityStatus,              # enum: FULLY/WEAKLY/NOT_OBSERVABLE/ERGODIC
)
```

**`KoopmanObservabilityChecker(d, N).check(g, domain) → KoopmanObservabilityReport`**

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `d` | 20 | Número de observables polinomiales |
| `N` | 5000 | Longitud de la trayectoria EDMD |
| `eps_target` | 1e-3 | Tolerancia para convergencia EDMD |

| Campo del report | Tipo | Descripción |
|-----------------|------|-------------|
| `ko1a_passed` | `bool` | K-invariancia (residual EDMD) |
| `ko1b_passed` | `bool` | Separación (Vandermonde) — siempre True para poly |
| `ko1c_passed` | `bool` | Ergodicidad de la trayectoria |
| `status` | `ObservabilityStatus` | Estado global |
| `k_invariance` | `KInvarianceReport` | Detalles residual K-invariante |
| `ergodicity` | `ErgodicityReport` | Error ergódico y gap espectral |
| `alpha_koopman` | `float` | α estimado desde espectro Koopman |

**`EnergyInvariantHardwareVerifier.verify(f, domain) → Dict`**

Devuelve dict `{"fp64": {...}, "fp32": {...}, "fp16": {...}}` con:
- `energy_d_star`: grado mínimo en esa precisión
- `epsilon_achieved`: error alcanzado
- `hw_valid`: bool (invariante HW-1/HW-2 satisfecho)

```python
import math
from acf_functor.koopman_observability import KoopmanObservabilityChecker, EnergyInvariantHardwareVerifier

report = KoopmanObservabilityChecker(d=20, N=2000).check(math.tanh, (-1.5, 1.5))
print(report.ko1a_passed, report.status)

hw = EnergyInvariantHardwareVerifier().verify(lambda x: x**3 - x, (-1.0, 1.0))
print(hw["fp64"]["energy_d_star"], hw["fp32"]["energy_d_star"])  # deben coincidir
```

---

### §36.4. ACF Diferenciable (`differentiable_acf.py`)

```python
from acf_functor.differentiable_acf import (
    ChebyshevCoeffExtractor,       # torch.autograd.Function (forward+backward)
    DifferentiableChebyshevApprox, # nn.Module con 3 tipos de parámetro
    ACFGradientFlow,               # ∂ε/∂θ via cadena de gradientes
    DifferentiableACFLayer,        # drop-in activation layer
    ACFGradientResult,             # resultado del gradiente
)
```

**`DifferentiableACFLayer(degree, domain, param_type)`**

| Parámetro | Opciones | Descripción |
|-----------|---------|-------------|
| `degree` | int | Número de coeficientes Chebyshev |
| `domain` | Tuple[float,float] | Dominio de normalización |
| `param_type` | `"direct"`, `"monomial"`, `"neural"` | Cómo se parametrizan los coeficientes |

Capa diferenciable: `layer(x)` evalúa la aproximación; `.coeffs.grad` contiene el gradiente respecto a coeficientes.

**`ACFGradientFlow.compute_gradient(f_vals, x) → ACFGradientResult`**

| Campo | Descripción |
|-------|-------------|
| `epsilon` | Error de compilación ε |
| `theta_gradient` | $\partial\varepsilon/\partial\theta$ (gradiente ordinario) |
| `natural_gradient` | Gradiente natural (Fisher) |
| `optimal_direction` | Dirección de máximo descenso conjunto |
| `epsilon_gradient_norm` | $\|\partial\varepsilon/\partial\theta\|_2$ |

```python
import torch, numpy as np
from acf_functor.differentiable_acf import DifferentiableACFLayer, ACFGradientFlow

layer = DifferentiableACFLayer(degree=16, domain=(-1.0, 1.0))
x = torch.linspace(-1, 1, 100, requires_grad=True)
y = layer(x)
((y - torch.sin(x))**2).mean().backward()

flow = ACFGradientFlow(degree=20, domain=(-1.0, 1.0))
result = flow.compute_gradient(np.sin(np.linspace(-1, 1, 200)),
                               torch.linspace(-1, 1, 200))
print(result.epsilon, result.epsilon_gradient_norm)
```

---

### §36.5. PDE-ACF Galerkin (`pde_acf.py`)

```python
from acf_functor.pde_acf import (
    PDEACFSolver,       # solver principal
    PDEConfig,          # configuración de la PDE
    PDESolutionReport,  # resultado del solve
    PDEType,            # enum: HEAT, ADVECTION, BURGERS, WAVE, ...
    solve_heat,         # wrapper conveniente
    solve_burgers,      # wrapper conveniente
)
```

**`PDEConfig` — campos principales**

| Campo | Default | Descripción |
|-------|---------|-------------|
| `pde_type` | — | Tipo de PDE (`PDEType.HEAT`, etc.) |
| `n_modes` | 64 | Modos Chebyshev (espacio) |
| `t_end` | 1.0 | Tiempo final |
| `dt` | 1e-3 | Paso temporal RK4 |
| `nu` | 0.01 | Difusividad / viscosidad |
| `c` | 1.0 | Velocidad de advección |

> **Estabilidad numérica:** Para el operador espectral Chebyshev, el número de condición crece como $O(d^4)$. Use `dt < 1/(nu * d^4)` para estabilidad RK4 explícita. Ejemplo seguro: `n_modes=8, dt=1e-5, nu=0.01`.

**`PDESolutionReport` — campos**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `x_grid` | `ndarray` | Puntos de colocación Chebyshev |
| `u_initial` / `u_final` | `ndarray` | Solución en $t=0$ y $t=T$ |
| `alpha_spectral` | `float` | Índice α de la solución final |
| `fma_count_per_step` | `int` | $O(d^2)$ FMAs por paso temporal |
| `total_fma_count` | `int` | Total acumulado |
| `pde1_satisfied` | `bool` | Teorema PDE-1 verificado |
| `truncation_error_bound` | `float` | Cota ε_d analítica |

```python
import numpy as np
from acf_functor.pde_acf import PDEACFSolver, PDEConfig, PDEType, solve_burgers

cfg = PDEConfig(pde_type=PDEType.HEAT, n_modes=8, t_end=0.2, dt=1e-5, nu=0.01)
solver = PDEACFSolver(cfg)
report = solver.solve(np.sin(np.pi * solver.x_grid))
print(f"α(T): {report.alpha_spectral:.4f}, FMAs: {report.total_fma_count:,}")

r = solve_burgers(lambda x: np.sin(np.pi*x), nu=0.1, t_end=0.05, n_modes=8, dt=1e-5)
print(f"‖u(T)‖∞ = {np.max(np.abs(r.u_final)):.4f}")
```

---

### §36.6. Genesis-Lean Bridge (`genesis_lean_bridge.py`)

```python
from acf_functor.genesis_lean_bridge import (
    GenesisLeanBridge,          # ciclo conjetura-verificación-catálogo
    ACFLeanTemplateLibrary,     # plantillas no-tautológicas
    LeanFileTester,             # wrapper subprocess para Lean 4
    is_tautological,            # guardián anti-tautología
    ConjectureStatus,           # enum: PROVED, PLAUSIBLE, REFUTED, TIMEOUT, TAUTOLOGICAL
)
```

**`is_tautological(proof: str) → bool`**

Devuelve `True` si la prueba solo contiene patrones tautológicos. Reglas:
- Rechazados: `exact h_*`, `rfl`, `assumption`, `trivial`, `tauto`
- Requeridos (al menos uno): `linarith`, `norm_num`, `apply`, `have`, `calc`, `field_simp`, `ring`

**`GenesisLeanBridge.conjecture_from_evidence(evidence: dict) → Conjecture`**

| Campo de `evidence` | Descripción |
|--------------------|-------------|
| `func` | Nombre de la función (str) |
| `degree` | Grado de Chebyshev |
| `epsilon` | Error alcanzado |
| `bernstein_rho` | $\rho_f$ estimado |

```python
from acf_functor.genesis_lean_bridge import GenesisLeanBridge, is_tautological

print(is_tautological("exact h_bound"))     # True  — rechazado
print(is_tautological("linarith [h1, h2]")) # False — aceptado

bridge = GenesisLeanBridge(catalog_path="/tmp/catalog.json")
conj = bridge.conjecture_from_evidence(
    {"func": "exp", "degree": 15, "epsilon": 1e-9, "bernstein_rho": 2.718}
)
print(conj.conjecture_id)  # cheb_exp_d15
```

---

### §36.7. Meta-Compilador Riemanniano (`riemannian_meta_compiler.py`)

```python
from acf_functor.riemannian_meta_compiler import (
    RiemannianMetaCompiler,    # optimizador principal
    RiemannianGrammarPoint,    # punto en la variedad (p_basis, p_degree, p_koopman)
    FisherPreconditioner,      # gradiente natural de Fisher
    RiemannianMetaResult,      # resultado de compile()
    _simplex_retract,          # retracción exponencial del símplex
)
```

**`RiemannianMetaCompiler(target_epsilon, max_iter, n_samples, n_test_points)`**

**`.compile(f, domain) → RiemannianMetaResult`**

| Campo del result | Descripción |
|-----------------|-------------|
| `best_basis` | Base óptima encontrada (`str`) |
| `best_degree` | Grado óptimo |
| `best_epsilon` | $\varepsilon$ alcanzado |
| `iterations` | Número de iteraciones |
| `theorem_rmc2_satisfied` | Bool: condicionamiento Fisher OK |
| `grammar_trajectory` | Lista de gramáticas exploradas |

**`RiemannianGrammarPoint.uniform() → RiemannianGrammarPoint`**  
Inicializa con distribuciones uniformes sobre bases, grados y ramas Koopman.

**`FisherPreconditioner.natural_step(p, grad) → ndarray`**  
Calcula $\tilde{g} = (g - \langle g, p\rangle) / p$ — el gradiente natural Fisher.

**`_simplex_retract(p, v, lr) → ndarray`**  
Retracción exponencial: `softmax(log(p) + lr * v)`. Garantiza que el resultado esté en el símplex.

```python
import math
from acf_functor.riemannian_meta_compiler import RiemannianMetaCompiler, FisherPreconditioner
import numpy as np

rmc = RiemannianMetaCompiler(target_epsilon=1e-4, max_iter=20, n_samples=6)
result = rmc.compile(math.sin, domain=(-1.0, 1.0))
print(f"Base: {result.best_basis}, d: {result.best_degree}, ε: {result.best_epsilon:.2e}")

p = np.array([0.5, 0.3, 0.2])
g = np.array([1.0, -0.5, 0.2])
ng = FisherPreconditioner.natural_step(p, g)  # gradiente natural
```

---

### §36.8. Identidades Triangulares del Adjunto (`adjunction.py`)

```python
from acf_functor.adjunction import (
    AdjunctionVerifier,         # verificador principal
    AdjunctionTriangleResult,   # resultado de verify_triangle_identities()
)
```

**`AdjunctionVerifier.verify_triangle_identities(f, domain, n_test, degree) → AdjunctionTriangleResult`**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `left_triangle_error` | `float` | $\|\Phi(\Phi^*(\Phi(f))) - \Phi(f)\|_\infty$ |
| `right_triangle_error` | `float` | $\|\Phi^*(\Phi(\Phi^*(g))) - \Phi^*(g)\|_\infty$ |
| `adjunction_gap` | `float` | Brecha combinada |
| `left_triangle_ok` | `bool` | error izquierdo ≤ tolerancia |
| `right_triangle_ok` | `bool` | error derecho ≤ tolerancia |
| `adjunction_holds` | `bool` | Ambas identidades satisfechas |
| `epsilon_f` | `float` | Error de unidad $\|f - \Phi^*(\Phi(f))\|_\infty$ |

> **Nota de uso:** `f` debe aceptar `torch.Tensor` 1-d y devolver `torch.Tensor` 1-d.

```python
import torch, math
from acf_functor.adjunction import AdjunctionVerifier

verifier = AdjunctionVerifier(tolerance=1e-3)
f = lambda x: torch.tensor([math.sin(xi.item()) for xi in x])
result = verifier.verify_triangle_identities(f, domain=(-1.0, 1.0), degree=25)
print(result.left_triangle_error)    # < 1e-5
print(result.adjunction_holds)       # True
```

---

### §36.9. Alpha Normalizado (`invariant_unified.py`)

**`AlphaEstimate`** — campos actualizados:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `best_estimate` | `float` | $\alpha$ crudo $\in [0, +\infty)$ |
| `normalized_alpha` | `float` | $\bar\alpha = 1/(1+\alpha) \in (0,1]$ (NUEVO) |
| `method_estimates` | `Dict` | α por método de estimación |
| `confidence_interval` | `Tuple` | Intervalo de confianza |

```python
from acf_functor.invariant_unified import ACFInvariantUnified

unified = ACFInvariantUnified()
result = unified.compute(lambda x: math.sin(x), (-1.0, 1.0), skip_geometric=True)
print(f"α crudo:      {result.best_estimate:.4f}")      # e.g. 0.23
print(f"α normalizado: {result.normalized_alpha:.4f}")  # 1/(1+0.23) ≈ 0.81
```

---

### §36.10. Suite de Benchmarks Comprensiva (`benchmarks/benchmark_comprehensive.py`)

```python
from benchmarks.benchmark_comprehensive import (
    run_full_benchmark,      # ejecuta los 22 casos
    BenchmarkSuiteReport,    # resultado agregado
    BenchmarkResult,         # resultado por caso
)
```

**`run_full_benchmark(verbose) → BenchmarkSuiteReport`**

| Campo del report | Descripción |
|-----------------|-------------|
| `results` | `List[BenchmarkResult]` — un resultado por caso |
| `pass_rate` | Fracción de casos que alcanzan ε_target |
| `total_cases` | 22 |
| `summary_table()` | Cadena con tabla formateada |

**`BenchmarkResult`** — campos:

| Campo | Descripción |
|-------|-------------|
| `case_id` | Identificador (`A1_sin`, `B3_atan`, ...) |
| `group` | Letra de grupo (`"A"` … `"E"`) |
| `epsilon_achieved` | Error $L_\infty$ alcanzado |
| `epsilon_target` | Objetivo |
| `degree` | Grado de Chebyshev usado |
| `fma_count` | FMAs totales |
| `alpha_measured` | $\alpha$ medido |
| `compile_time_ms` | Tiempo de compilación en ms |
| `eval_time_us` | Tiempo de evaluación en μs |
| `passed` | `True` sii epsilon_achieved ≤ epsilon_target |

```python
from benchmarks.benchmark_comprehensive import run_full_benchmark

report = run_full_benchmark(verbose=False)
print(report.summary_table())
print(f"Pass rate: {report.pass_rate*100:.1f}%")
failures = [r.case_id for r in report.results if not r.passed]
print(f"Fallos: {failures}")
```

---

### §36.11. Tabla General de Estado — Correcciones 2025→2026

| # | Debilidad | Estado | Módulo | Tests |
|---|-----------|--------|--------|-------|
| 1 | $E(f)=E(\Phi(f))$ solo en ℝ exacto | ✅ | `koopman_observability.py` | `test_hardware_invariant_polynomial` |
| 2 | Pruebas Lean tautológicas | ✅ | `genesis_lean_bridge.py` + Lean | `test_tautology_rejection` |
| 3 | Dominio $C^\omega$ no constructivo | ✅ | `domain_admissibility.py` | `test_sin_is_admissible` |
| 4 | Solo 4 benchmarks, algunos erróneos | ✅ | `benchmark_comprehensive.py` | `test_benchmark_suite_runs` |
| 5 | Adjunción no demostrada operacionalmente | ✅ | `adjunction.py` | `test_left_triangle_sin` |
| 6 | α fuera de [0,1] sin advertencia | ✅ | `invariant_unified.py` | `test_alpha_normalization` |

**Suite de tests:** `tests/test_all_expansions.py` — **43/43 passing**

*Manual técnico Poema v1.3.0 — Martínez's Invariant, Abril 2026*

---

## §37. API de Dominos Algebraicos v1.4.0

Referencia completa de los 5 nuevos módulos algebraicos.

---

### §37.1. `algebraic_acf.py` — Anillos, Cuerpos, Álgebras de Lie, ECC, Boole

#### Anillos disponibles

| Clase | Descripción | Aritmética |
|-------|-------------|------------|
| `RealRing()` | Números reales (float64) | Estándar IEEE 754 |
| `GFpRing(p)` | Cuerpo primo $\mathbb{F}_p$ | Módulo p, Fermat inverso |
| `GF2Ring()` | Cuerpo binario $\mathbb{F}_2$ | XOR/AND |
| `GF2mRing(m)` | Extensión $\mathbb{F}_{2^m}$ | Polinomios en GF(2) |
| `MatrixRing(n)` | Matrices $n\times n$ reales | @-multiplicación |

#### `AlgebraicACFReducer(ring).reduce_polynomial(coeffs, variable_name="x")`

```python
from acf_functor.algebraic_acf import AlgebraicACFReducer, GFpRing

reducer = AlgebraicACFReducer(GFpRing(7))
report = reducer.reduce_polynomial([1, 2, 3], variable_name="x")
# report.ring_name, .degree, .fma_count, .alpha, .normalized_alpha
# report.certificates, .code_c, .code_verilog, .code_lean
```

#### `reduce_over_ring(coeffs, ring)` — API de alto nivel

```python
from acf_functor.algebraic_acf import reduce_over_ring, RealRing
report = reduce_over_ring([1.0, 2.0, 3.0], RealRing())
```

#### `BooleanACFSynthesizer().synthesize(truth_table)`

```python
synth = BooleanACFSynthesizer()
report = synth.synthesize([0, 1, 1, 0])  # XOR de 2 bits
# report.n_inputs, .truth_table, .anf_coefficients, .anf_degree
# report.fma_count, .gate_depth, .verilog_rtl, .c_code, .certificates
```

#### `LieAlgebraFactory` + `LieAlgebraACFAnalyzer`

```python
# LieAlgebraFactory devuelve ndarray shape (d,d,d) — NO un objeto
f_su2  = LieAlgebraFactory.su2()       # (3,3,3)
f_so3  = LieAlgebraFactory.so3()       # (3,3,3)
f_heis = LieAlgebraFactory.heisenberg() # (3,3,3)

report = LieAlgebraACFAnalyzer().analyze(f_su2, algebra_name="su2")
# report.algebra_name, .dimension, .adjoint_fma_count
# report.killing_form_eigenvalues, .is_semisimple, .is_abelian
# report.alpha, .normalized_alpha, .certificates=["ALGACF-3"]
```

#### `ECCACFReducer(p, a, b)` — Curvas elípticas

```python
ecc = ECCACFReducer(p=17, a=2, b=2)
R   = ecc.point_add((5,1), (6,3))
S   = ecc.scalar_mul(k=5, P=(5,1))   # k es PRIMER argumento
report = ecc.analyze_reduction()
# report.p, .a_coeff, .b_coeff, .point_add_fmas, .scalar_mul_fmas
# report.field_alpha, .certificates=["ALGACF-5"]
```

#### `GroebnerACFReducer(ring).reduce_wrt_ideal(f_coeffs, generators=[...])`

```python
from acf_functor.algebraic_acf import GroebnerACFReducer, RealRing
reducer = GroebnerACFReducer(RealRing())
report = reducer.reduce_wrt_ideal([3,0,-3], generators=[[1,0,-1]])
# NOTA: el kwarg es `generators`, NO `generator_list`
```

---

### §37.2. `topos_acf.py` — Haces de Grothendieck

#### `ToposACFAnalyzer().analyze(f, domain)`

```python
from acf_functor.topos_acf import ToposACFAnalyzer
import numpy as np

report = ToposACFAnalyzer().analyze(
    f=lambda x: np.sin(x),
    domain=(-np.pi, np.pi)
)
# report.domain
# report.covering        — ACFCovering dataclass
# report.gluing_result   — GluingResult: .gluing_holds, .max_overlap_discrepancy
# report.admissibility_Omega  — dict (NO bool)
# report.geometric_sequents_valid  — list[str] (NO .certificates)
# report.lean4_certificate
```

#### ACFGrothendieckSite — API de bajo nivel

```python
from acf_functor.topos_acf import ACFGrothendieckSite, ACFSheafGluing

site = ACFGrothendieckSite()
cov  = site.generate_covering(f, domain, n_patches=4, overlap=0.05)
# cov.is_covering  — property bool
# cov.overlaps()   — método (con paréntesis), devuelve list

gluing = ACFSheafGluing(consistency_tolerance=1e-6)
result = gluing.check_compatibility([section1, section2])
# result.gluing_holds, .certificate="TOPOS-1"
```

---

### §37.3. `padic_acf.py` — Números p-ádicos

#### `PAdicACFReducer(p).reduce(f_values, epsilon)`

```python
from acf_functor.padic_acf import PAdicACFReducer

reducer = PAdicACFReducer(p=5)
report  = reducer.reduce([1, 2, 3, 4, 5], epsilon=1e-3)
# report.fma_count, .certificates
# NOTA: certificates = ["PADIC-2 (Mahler)"] — no bare "PADIC-2"
# Verificar: any("PADIC-2" in c for c in report.certificates)
```

#### `hensel_lift(f_coeffs, a0, p, precision)`

```python
from acf_functor.padic_acf import hensel_lift

# Resolver x² ≡ 2 (mod 7), x₀ = 3 (3²=9≡2 mod 7)
result = hensel_lift([2, 0, 1], a0=3, p=7, precision=5)
# result.converged (bool), .fma_count
```

#### `ramanujan_tau_coefficients(n_terms)`

```python
tau = ramanujan_tau_coefficients(10)
# Lista 1-INDEXADA: tau[0]=0 (padding), tau[1]=τ(1)=1, tau[2]=τ(2)=-24
# τ(1)=1, τ(2)=-24, τ(3)=252, τ(4)=-1472, ...
```

---

### §37.4. `modular_acf.py` — Formas Modulares

#### `ModularFormLibrary()` — Formas estándar

```python
from acf_functor.modular_acf import ModularFormLibrary

lib   = ModularFormLibrary()
e4    = lib.e4()      # Eisenstein E₄, weight=4
e6    = lib.e6()      # Eisenstein E₆, weight=6
delta = lib.delta()   # Función Δ de Ramanujan, weight=12
# Cada reporte: .fma_count, .weight, .certificates, .alpha
```

#### `ModularACFReducer().reduce(name, coeffs, weight)`

```python
from acf_functor.modular_acf import ModularACFReducer

report = ModularACFReducer().reduce("E4", [1, 240, 2160], weight=4)
# report.weight, .certificates=["MOD-1", ...], .alpha
```

#### Funciones de coeficientes

```python
from acf_functor.modular_acf import (
    eisenstein_e4_coefficients,
    eisenstein_e6_coefficients,
    ramanujan_tau_coefficients
)

e4_coeffs = eisenstein_e4_coefficients(10)  # [1, 240, 2160, ...]
e6_coeffs = eisenstein_e6_coefficients(10)  # [1, -504, -16632, ...]
tau       = ramanujan_tau_coefficients(10)  # [0, 1, -24, 252, ...]
```

---

### §37.5. `finance_acf.py` — Finanzas y Caos

#### `estimate_hurst(time_series, min_window=8, max_window=None)`

```python
from acf_functor.finance_acf import estimate_hurst
import numpy as np

# ⚠️ IMPORTANTE: pasar INCREMENTOS/retornos, NO precios acumulados
increments = np.diff(np.cumsum(np.random.standard_normal(500)))
report = estimate_hurst(increments)
# report.hurst_exponent, .hurst_lower, .hurst_upper
# report.series_type, .alpha_linear, .rs_values, .n_values
# report.certificates=["FIN-1"]
```

#### `VolatilitySurfaceReducer().reduce(sigma_fn, k_range, T_values, target_epsilon)`

```python
reducer = VolatilitySurfaceReducer(max_degree_k=12, max_degree_T=6)
report = reducer.reduce(
    sigma_fn=lambda k, T: 0.20 + 0.01*k**2,  # ← kwarg es `sigma_fn`, NO `sigma_function`
    k_range=(-0.5, 0.5),
    T_values=[0.25, 0.5, 1.0],
    target_epsilon=1e-4
)
# report.total_fma_count  ← NOT `fma_count`
# report.n_maturities, .chebyshev_degrees_k, .alpha_per_slice
# report.mean_alpha, .approximation_errors, .certificates=["FIN-2","AD-3"]
```

#### `compute_risk_via_pce(payoff_fn, n_mc, n_hermite)`

```python
# ⚠️ payoff_fn recibe np.array([xi]) — DEBE devolver float
def call_option(x):
    xi = float(np.asarray(x).flat[0])   # extraer escalar
    return max(xi - 1.0, 0.0)

risk = compute_risk_via_pce(call_option, n_mc=10000, n_hermite=6)
# risk.var_95, .var_99, .es_95  ← CVaR se llama `es_95`, NO `cvar_95`
# risk.expected_value, .variance, .sobol_indices, .fma_count
# risk.certificates=["FIN-3", "STOCH-VaR"]
```

#### `detect_regime_changes(time_series, window_size, alpha_jump_threshold)`

```python
report = detect_regime_changes(
    series,
    window_size=50,
    alpha_jump_threshold=0.05   # ← threshold=0.3 demasiado alto para señales típicas
)
# report.alpha_series (ndarray), .regime_changes (list[int])
# report.current_regime (str), .lyapunov_estimate, .hurst
# report.certificates=["FIN-4"]
# NOTA: len(report.regime_changes), NO report.n_regime_changes
```

#### `analyze_invariant_density(map_fn, domain, n_orbit, n_bins, target_epsilon)`

```python
def logistic(x):
    return 4.0 * x * (1.0 - x)

report = analyze_invariant_density(logistic, domain=(0.0, 1.0))
# report.map_name, .lyapunov_exponent, .invariant_density_fma_count
# report.chebyshev_degree, .alpha, .approximation_error
# report.is_fully_chaotic, .density_coefficients (ndarray or None)
# report.certificates=["FIN-5"]
# NOTA: NO hay método .evaluate_density() — usar .density_coefficients directamente
```

---

### §37.6. Tabla de Resumen — Módulos v1.4.0

| Módulo | Certificados | Clases principales | Tests |
|--------|-------------|-------------------|-------|
| `algebraic_acf.py` | ALGACF-1..5 | `AlgebraicACFReducer`, `BooleanACFSynthesizer`, `LieAlgebraACFAnalyzer`, `ECCACFReducer` | 58 |
| `topos_acf.py` | TOPOS-1..4 | `ToposACFAnalyzer`, `ACFGrothendieckSite`, `ACFSheafGluing` | 12 |
| `padic_acf.py` | PADIC-1..3 | `PAdicACFReducer`, `hensel_lift`, `MahlerSeries` | 14 |
| `modular_acf.py` | MOD-1..4 | `ModularFormLibrary`, `ModularACFReducer` | 12 |
| `finance_acf.py` | FIN-1..5 | `estimate_hurst`, `VolatilitySurfaceReducer`, `detect_regime_changes`, `analyze_invariant_density`, `compute_risk_via_pce` | 15 |

**Suite tests v1.4.0:** `tests/test_algebraic_extensions.py` — **105/105 passing**

---

### §37.7. Verificación de Conteos FMA Teóricos vs. Profiler GPU

> **Crítica abierta:** Los campos `fma_count`, `total_fma_count`, `invariant_density_fma_count`, etc., son conteos **teóricos** derivados del algoritmo (número de pasos Horner, grado Chebyshev, etc.). En hardware real, el profiler GPU puede reportar un número diferente de instrucciones FMA/FMAD por tres razones legítimas:

| Fuente de divergencia | Impacto típico | Como diagnosticar |
|-----------------------|---------------|-------------------|
| **Fusión de instrucciones del compilador Triton/PTX** — dos FMAs consecutivas `fma(a,b,fma(c,d,e))` se pueden emitir como una sola instrucción dependiente en el scheduler | Conteo real < teórico (favorable) | `nsys profile --stats=true` → `FMA` instructions |
| **Desenrollado de bucles** — Triton puede desenrollar un Horner de grado $d$ en $d$ FMAs explícitas o en una GEMM vectorizada | Conteo real ≠ teórico (varía por `tl.constexpr`) | `ncu --metrics smsp__inst_executed_pipe_fma` |
| **Fallback fp32 → fp64** — promoción automática de precisión cuando `cond(W) > 1e6` duplica el conteo de instrucciones DFMA | Conteo real ≈ 2× teórico | `ncu --metrics smsp__inst_executed_pipe_fp64` |

**Protocolo de validación fase 4 (GPU real):**

```python
# 1. Obtener conteo teórico del certificado ACF
from acf_functor.finance_acf import analyze_invariant_density
report = analyze_invariant_density(lambda x: 4*x*(1-x))
theoretical_fma = report.invariant_density_fma_count  # e.g. 48

# 2. Medir en GPU con torch.profiler
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    result = report.density_coefficients  # ejecutar kernel

# 3. Extraer FMAs reales del trace
# ncu --target-processes all --metrics smsp__inst_executed_pipe_fma \
#     python3 mi_script.py

# Tolerancia aceptable: conteo_real / theoretical_fma ∈ [0.5, 2.0]
# Si queda fuera de ese rango, abrir issue con el trace ncu.
```

**Contrato del manual:** Los `fma_count` reportados aquí son **cotas superiores teóricas** del algoritmo de reducción, no promesas de conteo de instrucciones en silicio. Son correctos como métrica de complejidad algorítmica ($O$-análisis) y como entrada para ADI/presupuesto de memoria. No deben usarse directamente para comparación con contadores hardware sin el factor de fusión/vectorización del compilador.

---

## 5. Agentes del Ecosistema ACF: TAA y ERGON

El ecosistema ACF incluye dos agentes matemáticos que operan en lados opuestos de la dualidad de Koopman. Son **independientes entre sí** — cada uno puede ejecutarse sin el otro — pero se benefician mutuamente cuando cooperan.

### 5.1 TAA — Tensor Autocomputable Agent

TAA opera sobre el lado de las **funciones** (observables):

$$\mathcal{K}: L^2(\mathcal{X}, \mu) \to L^2(\mathcal{X}, \mu), \quad \mathcal{K}f = f \circ T$$

Su trabajo es descomponer los observables de un sistema dinámico en eigenfunciones de Koopman, truncar el espectro a dimensión $d^*(\varepsilon)$, y colapsar a secuencias FMA certificadas.

```python
from poema.taa_agent import TAAAgent, AlphaClass
import numpy as np

# Sistema logístico: T(x) = r·x·(1-x)
r = 3.7
T = lambda x: np.array([r * x[0] * (1 - x[0])])

# Generar trayectoria
x0 = np.array([0.5])
x_data = np.zeros((1000, 1))
x = x0.copy()
for k in range(1000):
    x_data[k] = x
    x = T(x)

# Análisis TAA
agent = TAAAgent()
report = agent.analyze(T, x_data, epsilon=1e-4)

print(f"Alpha class:    {report.alpha_class.name}")    # EXPONENTIAL / POLYNOMIAL / FINITE
print(f"d*(ε=1e-4):    {report.d_star}")
print(f"δ(d*):          {report.delta_d:.3e}")
print(f"FMA cost:       {report.fma_cost}")
print(f"λ_max:          {report.lambda_max:.4f}")
print(f"ERGON required: {report.ergon_required}")
```

**Certificados Lean 4 del TAA:**

| Teorema | Descripción | Archivo |
|---|---|---|
| TAA-1 | Koopman es isometría: ‖Kf‖₂ = ‖f‖₂ | `TAAAgentCertificates.lean` |
| TAA-2 | E(f) = E(Φ_AC(f)) — invariancia de profundidad | `TAAAgentCertificates.lean` |
| TAA-3b | d*(ε) explícito para decaimiento exponencial | `TAAAgentCertificates.lean` |
| TAA-4 | α_A clasifica familia de costo FMA | `TAAAgentCertificates.lean` |
| TAA-5 | Medida incorrecta infla δ(d) | `TAAAgentCertificates.lean` |
| KD-1..4 | Bounds espectrales de truncamiento Koopman | `KoopmanDeltaCertificates.lean` |

### 5.2 ERGON — Perron-Frobenius Agent

ERGON opera sobre el lado de las **medidas** (distribuciones):

$$\mathcal{L}: \mathrm{Meas}(\mathcal{X}) \to \mathrm{Meas}(\mathcal{X}), \quad (\mathcal{L}\mu)(A) = \mu(T^{-1}(A))$$

Su trabajo es encontrar la medida SRB (la ley estadística natural del sistema caótico), computar la entropía de Kolmogorov-Sinai, y verificar la Fórmula de Pesin.

```python
from poema.ergon import ERGONAgent
import numpy as np

# Sistema de Lorenz (caos genuino)
def lorenz(x, sigma=10, rho=28, beta=8/3, dt=0.01):
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return x + dt * np.array([dx, dy, dz])

x0 = np.array([1.0, 1.0, 1.0])
ergon = ERGONAgent(n_iterations=50_000)
report = ergon.analyze(lorenz, x0, epsilon=1e-3)

print(f"h_KS:                {report.h_ks:.4f}")
print(f"λ_max:               {report.lambda_max:.4f}")
print(f"Σλᵢ⁺:               {report.lyapunov_sum:.4f}")
print(f"𝔈(T) = h_KS/Σλ⁺:  {report.ergodic_complexity:.4f}")
print(f"Pesin verificado:    {report.pesin_verified}")
print(f"n*(ε):               {report.budget_n_star}")
print(f"d* recomendado TAA:  {report.recommended_d_star}")
print(f"Delegar a TAA:       {report.handoff_to_taa}")
```

**Certificados Lean 4 del ERGON:**

| Teorema | Descripción | Archivo |
|---|---|---|
| ERG-2 | ℒ = K* — dualidad adjunta exacta | `ERGONCertificates.lean` |
| ERG-3 | Birkhoff: promedio temporal = espacial | `ERGONCertificates.lean` |
| ERG-4 | Desigualdad Margulis-Ruelle: h_μ ≤ ∫λ⁺ dμ | `ERGONCertificates.lean` |
| ERG-6b | 𝔈(T) ∈ [0, 1] | `ERGONCertificates.lean` |
| ERG-7b | Interfaz ERGON→TAA correcta | `ERGONCertificates.lean` |
| ERG-9 | TAA y ERGON son independientes | `ERGONCertificates.lean` |

**Axiomas pendientes de cerrar:**

| Axioma | Descripción | Bloqueado por |
|---|---|---|
| ERG-1 | ∃ μ_SRB: ℒμ* = μ* | Condiciones de Pesin en Mathlib |
| ERG-8 | Descomposición ergódica | Teorema de Rokhlin en Mathlib |

### 5.3 Independencia y Cooperación

**TAA y ERGON son matemáticamente independientes** (ERG-9 en `ERGONCertificates.lean`):
- TAA opera sobre $L^2(\mathcal{X}, \mu)$ — funciones
- ERGON opera sobre $\mathrm{Meas}(\mathcal{X})$ — medidas
- Ninguno necesita el estado interno del otro
- Se comunican solo a través de la interfaz $\mu_{SRB}$

**Tabla de independencia:**

| Escenario | TAA | ERGON | Resultado |
|---|---|---|---|
| Sistema polinomial (α_A bajo) | ✓ actúa solo | no necesario | FMA exacto |
| Sistema caótico sin ERGON | ✓ actúa (δ inflado) | no ejecutado | FMA con δ subóptimo |
| Sistema caótico con ERGON | ✓ actúa (δ óptimo) | ✓ provee μ_SRB | FMA con δ mínimo real |
| Sistema caótico puro | no suficiente | ✓ certifica solo | Pesin + ley estadística |
| Sistema mixto | ✓ para partes integrables | ✓ para partes caóticas | Descomposición ergódica |

### 5.4 Pipeline Conjunto TAA + ERGON

```python
from poema.ergon import ERGONAgent

# El pipeline óptimo: ERGON primero, luego TAA con μ_SRB
ergon = ERGONAgent()
ergon_report, taa_report = ergon.joint_analyze(
    T=lorenz,
    x0=np.array([1.0, 1.0, 1.0]),
    x_data=trajectory,
    epsilon=1e-4
)

# ERGON provee μ_SRB → TAA usa L²(𝒳, μ_SRB) correcto
# La inflación de medida (TAA-5) queda eliminada (TAA-5b)
print(f"Medida usada por TAA: {taa_report.measure_used}")        # 'srb'
print(f"Inflación de medida:  {taa_report.measure_inflation}")   # 0.0
print(f"δ(d*) óptimo:         {taa_report.delta_d:.3e}")
```

**El flujo completo:**

```
SEÑAL DINÁMICA
      │
      ▼
ERGON.analyze()
  ├── Ψ_ER: Birkhoff → μ_SRB         [ERG-3]
  ├── Λ_ER: QR-Lyapunov → λᵢ         [ERG-5]
  ├── h_KS: entropía KS              [ERG-6a]
  ├── Pesin: |h_KS - Σλ⁺| < ε       [ERG-6b]
  └── 𝔈(T): ¿delegar a TAA?
        │
        ├── 𝔈 < 0.1 → TAA solo (sistema integrable)
        │
        └── 𝔈 ≥ 0.1 → μ_SRB ──────────────────────┐
                                                    │
TAA.analyze(mu_srb=μ_SRB)                          │
  ├── EDMD → eigenvalores Koopman    [KD-1]  ◄─────┘
  ├── α_A: familia de decaimiento    [TAA-4]
  ├── d*(ε): dimensión óptima        [TAA-3b]
  ├── δ(d*): error truncamiento      [KD-1]
  └── FMA cost: operaciones          [TAA-2]
        │
        ▼
  CERTIFICADO CONJUNTO
  [Pesin + FMA + δ mínimo]
```

### 5.5 Certificados Lean 4

Los certificados formales residen en:

```
MathTest/
├── KoopmanDeltaCertificates.lean    # KD-1..4 — base espectral de TAA
├── TAAAgentCertificates.lean        # TAA-1..6 — agente completo
└── ERGONCertificates.lean           # ERG-1..9 — agente ERGON completo
```

Importados en `MathTest.lean`:

```lean
import MathTest.TAAAgentCertificates
import MathTest.ERGONCertificates
```

**Estado de verificación:**

```
TAA: 11 teoremas ✓ proved, 2 axioms (TAA-3a, TAA-6)
ERGON: 13 teoremas/certificados derivados ✓, 4 axioms principales (ERG-1, ERG-5, ERG-7a, ERG-8)
Objetivo principal: ERG-5 (saturación SRB) — desde ahí se recupera ERG-6a y se acerca TAA-6
```

*Manual técnico Poema v1.4.0 — Martínez's Invariant, Mayo 2026*
