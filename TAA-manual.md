# TAA: Tensor Autocomputable Agent — Manual Técnico
### Agente Koopman del Ecosistema ACF

**Versión:** 2.0  
**Módulo:** `poema/taa_agent.py`  
**Certificados Lean 4:** `MathTest/TAAAgentCertificates.lean`  
**Certificados Koopman:** `MathTest/KoopmanDeltaCertificates.lean`  
**Agente dual:** ERGON (`poema/ergon.py`) — ver `ERGON-manual.md`

---

## Índice

1. [¿Qué es TAA?](#1-qué-es-taa)
2. [Fundamentos Matemáticos](#2-fundamentos-matemáticos)
3. [Pipeline Completo](#3-pipeline-completo)
4. [Los Cuatro Operadores de TAA](#4-los-cuatro-operadores-de-taa)
5. [El Índice Alpha-A](#5-el-índice-alpha-a)
6. [Dimensión Óptima d\*(ε)](#6-dimensión-óptima-dε)
7. [Costo FMA](#7-costo-fma)
8. [Diagnóstico de Medida](#8-diagnóstico-de-medida)
9. [API Reference](#9-api-reference)
10. [Certificados Formales (TAA-1 a TAA-6)](#10-certificados-formales-taa-1-a-taa-6)
11. [Interfaz TAA ↔ ERGON](#11-interfaz-taa--ergon)
12. [Integración con el Ecosistema](#12-integración-con-el-ecosistema)
13. [Ejemplos de Uso](#13-ejemplos-de-uso)
14. [Errores y Diagnóstico](#14-errores-y-diagnóstico)
15. [Problemas Abiertos](#15-problemas-abiertos)
16. [Nuevos Certificados (TAA-10 a TAA-12)](#16-nuevos-certificados-taa-10-a-taa-12)
17. [Puente al Mundo Real (TAAAgentRealWorld)](#17-puente-al-mundo-real-taaagentreal-world)
18. [Tabla de Archivos Actualizada](#18-tabla-de-archivos-actualizada)
19. [Shared Numerical Infrastructure (Epic 9)](#19-shared-numerical-infrastructure-epic-9)

---

## 1. ¿Qué es TAA?

**TAA** (Tensor Autocomputable Agent) es el agente de operador de Koopman del ecosistema ACF. Opera sobre el **lado de funciones** de la dualidad de Koopman:

$$\mathcal{K}: L^2(\mathcal{X}, \mu) \to L^2(\mathcal{X}, \mu), \quad \mathcal{K}f = f \circ T$$

Su misión es descomponer las observables de un sistema dinámico en **eigenfunciones de Koopman**, truncar el espectro a la dimensión óptima $d^*(\varepsilon)$, y colapsar la representación resultante a secuencias **FMA certificadas**.

### Posición en el Ecosistema ACF

```
SEÑAL / SISTEMA DINÁMICO T
           │
           ▼
    ┌─────────────────────────────────────┐
    │         FRONTERA TAA/ERGON          │
    │                                     │
    │  TAA: mundo de funciones            │
    │  K : L²(𝒳,μ) → L²(𝒳,μ)            │
    │  Busca: eigenfunciones Kφ = λφ      │
    │                                     │
    │  ERGON: mundo de medidas            │
    │  ℒ : Meas(𝒳) → Meas(𝒳)            │
    │  Busca: medida SRB ℒμ* = μ*        │
    └─────────────────────────────────────┘
           │              │
           ▼              ▼
    CERTIFICADO TAA    CERTIFICADO ERGON
    E(f) = E(Φ_AC(f))  h_KS = ∫λ⁺ dμ_SRB
    δ(d) < ε           Pesin verificado
```

**TAA es independiente de ERGON**: puede operar con medida empírica. Con μ_SRB de ERGON, TAA alcanza el mínimo absoluto de error de truncamiento δ(d).

---

## 2. Fundamentos Matemáticos

### 2.1 El Operador de Koopman

Sea $T: \mathcal{X} \to \mathcal{X}$ un mapa dinámico medible. El **operador de Koopman** $\mathcal{K}$ actúa sobre funciones observables:

$$(\mathcal{K}f)(x) = f(T(x)) \quad \forall f \in L^2(\mathcal{X}, \mu)$$

**Propiedades fundamentales:**
- $\mathcal{K}$ es **lineal** aunque $T$ sea no lineal
- Para $T$ que preserva medida: $\|\mathcal{K}f\|_2 = \|f\|_2$ — es una **isometría** (TAA-1)
- Espectro de $\mathcal{K}$ contenido en el disco unitario cerrado: $|\lambda_k| \leq 1$ para todo eigenvalor (TAA-1b)

### 2.2 Eigenfunciones de Koopman

Las eigenfunciones $\varphi_k$ satisfacen:

$$\mathcal{K}\varphi_k = \lambda_k \varphi_k, \quad |\lambda_k| \leq 1$$

- $|\lambda_k| = 1$: **modos estructurales** — TAA los conserva exactamente
- $|\lambda_k| < 1$: **modos transitorios/de mixing** — decaen; ERGON los gestiona
- $|\lambda_k| = 0$: **modos muertos** — contribución nula

La expansión espectral de cualquier observable $f$:

$$f = \sum_{k=1}^\infty c_k \varphi_k, \quad c_k = \langle f, \varphi_k \rangle_\mu$$

### 2.3 El Error de Truncamiento δ(d)

Al retener solo los $d$ eigenmodos dominantes, el error de truncamiento es (KD-1):

$$\delta(d) = |\lambda_{d+1}| \leq \epsilon \quad \Longleftrightarrow \quad d \geq d^*(\epsilon)$$

Esta es la cota espectral que TAA garantiza formalmente. Es el **contrato de error central** de TAA.

### 2.4 La Dualidad TAA ↔ ERGON

La ecuación que une TAA y ERGON (ERG-2):

$$\langle \mathcal{K}g, \mu \rangle_{L^2} = \langle g, \mathcal{L}\mu \rangle_{L^2}$$

$\mathcal{L}$ (Perron-Frobenius) es el adjunto exacto de $\mathcal{K}$ (Koopman). TAA opera en el primer espacio, ERGON en el segundo. No hay conflicto de estado — son mundos matemáticamente disjuntos que se comunican solo a través de $\mu_{SRB}$.

---

## 3. Pipeline Completo

```
Datos de trayectoria x_data (shape: n_steps × dim)
             │
             ▼
  ┌─────────────────────┐
  │  EDMD               │  ← Paso 1: Aproximación Koopman
  │  K = X'·X†          │
  │  eigenvalues, modes │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  Alpha-A Classifier │  ← Paso 2: Clasificación de decaimiento espectral
  │  EXPONENTIAL /      │
  │  POLYNOMIAL /       │
  │  FINITE / UNKNOWN   │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  d*(ε) Computation  │  ← Paso 3: Dimensión óptima de Koopman
  │  TAA-3b formula     │
  │  δ(d*) < ε          │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  FMA Cost Estimation│  ← Paso 4: Presupuesto computacional
  │  cost = d* × dim    │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  Measure Diagnostics│  ← Paso 5: ¿Se necesita μ_SRB de ERGON?
  │  'empirical' / 'srb'│
  │  TAA-5 / TAA-5b     │
  └──────────┬──────────┘
             │
             ▼
         TAAReport
```

---

## 4. Los Cuatro Operadores de TAA

### 4.1 EDMD — Extended Dynamic Mode Decomposition

TAA aproxima el operador de Koopman usando **EDMD**:

Dadas pares de snapshots $(x_k, x_{k+1})$:

$$K_{\text{approx}} = X' \cdot X^\dagger \quad \text{donde} \quad X = [x_0, \ldots, x_{n-1}], \quad X' = [x_1, \ldots, x_n]$$

Los eigenvalores $\{\lambda_k\}$ de $K_{\text{approx}}$ aproximan el espectro de Koopman.

**Implementación:**
```python
X = x_data[:-1].T    # snapshots anteriores (dim × n-1)
Xp = x_data[1:].T    # snapshots posteriores (dim × n-1)
K_matrix, _, _, _ = np.linalg.lstsq(X.T, Xp.T)  # mínimos cuadrados
eigenvalues, modes = np.linalg.eig(K_matrix.T)
```

**Garantías:**
- Los eigenvalores resultantes están acotados a $[0, 1]$ (TAA-1: isometría Koopman)
- Ordenados por $|\lambda|$ descendente → el truncamiento a $d$ retiene los modos más importantes

### 4.2 Alpha-A Classifier

Clasifica el **perfil de decaimiento espectral** de los eigenvalores:

| Clase | Condición | Costo d* | Uso típico |
|---|---|---|---|
| `EXPONENTIAL` | $|\lambda_k| \leq C \cdot \rho^{-k}$, $\rho > 1$ | $O(\log 1/\varepsilon)$ | Sistemas estables, atractores |
| `POLYNOMIAL` | $|\lambda_k| \leq C \cdot k^{-s}$, $s > 0$ | $O(\varepsilon^{-1/s})$ | Sistemas débilmente mezcladose |
| `FINITE` | Espectro tiene $\leq d$ eigenvalores no nulos | $d$ (exacto) | Sistemas lineales, polinomiales |
| `UNKNOWN` | Datos insuficientes para clasificar | Heurístico | Señales cortas |

**Implementación:** Ajuste por mínimos cuadrados en escala log-log y log-linear para comparar bondad de ajuste.

### 4.3 Cálculo de d*(ε)

La dimensión óptima depende de la clase Alpha-A. Operacionalmente TAA usa estas fórmulas y, en Lean, TAA-3b ya quedó cerrado como teorema real para el caso exponencial; la deuda formal remanente está en TAA-3a y en los lemas asintóticos de comparación más finos:

| Clase | Fórmula d* |
|---|---|
| EXPONENTIAL | $d^* = \lceil \log(C/\varepsilon) / \log(\rho) \rceil$ |
| POLYNOMIAL | $d^* = \lceil (C/\varepsilon)^{1/s} \rceil$ |
| FINITE | $d^* = d$ (exacto) |
| UNKNOWN | Búsqueda directa: min $d$ tal que $\lambda_{d+1} < \varepsilon$ |

### 4.4 Estimación de Costo FMA

$$\text{cost}_{\text{FMA}} = d^* \times \dim(\mathcal{X})$$

Cada modo de Koopman requiere `dim` multiplicaciones y adiciones (Horner, TAA-2c).

---

## 5. El Índice Alpha-A

El índice $\alpha_A$ (tipo `AlphaClass`) determina la **clase de costo computacional** del sistema:

```
EXPONENTIAL   → log(1/ε) FMAs    [más barato]
POLYNOMIAL    → (1/ε)^{1/s} FMAs [moderado]
FINITE        → d FMAs exactos   [exacto]
UNKNOWN       → heurístico
```

**Relación con el ACF:**
- El $\alpha_A$ de TAA es el análogo Koopman del índice afín $\alpha_A$ del ACF
- Determina si el sistema es "computable" en el sentido ACF a precisión $\varepsilon$ con costo finito
- Clase `EXPONENTIAL` ↔ función analítica en el ACF (Horner óptimo)
- Clase `POLYNOMIAL` ↔ función continuamente diferenciable (Chebyshev)
- Clase `FINITE` ↔ sistema lineal o polinomial (FMA exacto)

---

## 6. Dimensión Óptima d*(ε)

La existencia de $d^*(\varepsilon)$ es garantizada por TAA-3a (axioma). Para los casos con decaimiento conocido, TAA usa las fórmulas explícitas de TAA-3b tanto a nivel operativo como en la capa Lean, donde el caso exponencial ya quedó probado directamente.

### Conexión con KoopmanDeltaCertificates

Los certificados KD-1 a KD-4 proveen las cotas espectrales que sustentan TAA-3b:

| Certificado | Descripción | Usado por |
|---|---|---|
| KD-1 | $\delta(d) \leq \lambda_{d+1}$ | TAA: cota de error de truncamiento |
| KD-2 | Subaditividad de $\delta$ en composición | TAA: error compuesto |
| KD-3 | $\forall \varepsilon > 0, \exists d^*(\varepsilon)$ | TAA-3a: existencia de budget |
| KD-4 | Decaimiento explícito para $\alpha$-class | TAA-3b: fórmula explícita |

---

## 7. Costo FMA

El costo FMA total de TAA en un sistema de dimensión `dim` con precisión `ε`:

$$\text{FMA total} = d^*(\varepsilon) \times \dim(\mathcal{X})$$

**Ejemplo concreto:**
- Sistema 1D, EXPONENTIAL con $\rho = 2$, $C = 1$, $\varepsilon = 10^{-6}$:
  - $d^* = \lceil \log_2(10^6) \rceil = 20$
  - FMA total = $20 \times 1 = 20$ operaciones

- Sistema 3D, POLYNOMIAL con $s = 2$, $C = 1$, $\varepsilon = 10^{-3}$:
  - $d^* = \lceil 10^{3/2} \rceil = 32$
  - FMA total = $32 \times 3 = 96$ operaciones

---

## 8. Diagnóstico de Medida

TAA puede operar con dos tipos de medida de referencia:

### TAA-5: Medida Empírica (default)

Cuando no se provee $\mu_{SRB}$, TAA usa la medida empírica de la trayectoria. El error de truncamiento efectivo tiene **inflación de medida**:

$$\delta(d)_{\text{efectivo}} \geq \delta(d)_{\text{correcto}} + \|f\|_\infty \cdot \|\mu - \mu_{SRB}\|_{TV}$$

Diagnóstico: `measure_used = 'empirical'`, `measure_inflation > 0`.

### TAA-5b: Medida SRB de ERGON (óptimo)

Cuando ERGON provee $\mu_{SRB}$:

$$\delta(d)_{\text{efectivo}} = \delta(d)_{\text{correcto}} \quad (\text{inflación} = 0)$$

Diagnóstico: `measure_used = 'srb'`, `measure_inflation = 0.0`.

**Regla práctica:** Para sistemas con `lambda_max > 0.05` (caos detectado), siempre proveer $\mu_{SRB}$ de ERGON. Ver TAA-6.

---

## 9. API Reference

### Clase `TAAAgent`

```python
from poema.taa_agent import TAAAgent, TAAReport, AlphaClass

agent = TAAAgent(edmd_delay=1)
report = agent.analyze(T, x_data, epsilon=1e-6, mu_srb=None)
```

**Constructor:**

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `edmd_delay` | `int` | `1` | Delay de coordenadas para EDMD |

**Método `analyze()`:**

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `T` | `Callable` | — | El mapa dinámico $T: \mathcal{X} \to \mathcal{X}$ |
| `x_data` | `np.ndarray` | — | Trayectoria, shape `(n_steps, dim)` |
| `epsilon` | `float` | `1e-6` | Error objetivo de truncamiento |
| `mu_srb` | `np.ndarray \| None` | `None` | Medida SRB de ERGON (elimina inflación) |

**Retorna:** `TAAReport`

---

### Clase `TAAReport`

```python
@dataclass
class TAAReport:
    eigenvalues: np.ndarray      # Eigenvalores Koopman |λ₁| ≥ |λ₂| ≥ ...
    d_star: int                  # Dimensión Koopman óptima para ε
    delta_d: float               # Error de truncamiento δ(d*) — KD-1
    alpha_class: AlphaClass      # Familia de decaimiento espectral
    alpha_rate: float            # Tasa (ρ para exp, s para poly, d para finite)
    fma_cost: int                # FMAs para evaluar expansión Koopman
    measure_used: str            # 'empirical' o 'srb'
    measure_inflation: float     # Error adicional por medida incorrecta
    koopman_modes: np.ndarray    # Eigenvectores Koopman (columnas de K)
    koopman_freqs: np.ndarray    # Frecuencias Koopman (partes imaginarias)
    koopman_damping: np.ndarray  # Tasas de amortiguamiento (partes reales)
    ergon_required: bool         # True si λ_max > umbral y no hay μ_SRB
    lambda_max: float            # Exponente máximo de Lyapunov estimado
    epsilon_target: float        # El ε para el que se calculó d_star
```

---

### Clase `AlphaClass`

```python
class AlphaClass(Enum):
    EXPONENTIAL = auto()  # |λ_k| ≤ C·ρ^{-k} — coste O(log 1/ε)
    POLYNOMIAL  = auto()  # |λ_k| ≤ C·k^{-s} — coste O(ε^{-1/s})
    FINITE      = auto()  # d eigenvalores no nulos — coste exacto d
    UNKNOWN     = auto()  # insuficientes datos para clasificar
```

---

### Método `koopman_predict()`

Predicción de trayectoria usando expansión Koopman truncada:

```python
trajectory = agent.koopman_predict(report, x0, n_steps=100)
# shape: (n_steps, dim)
# Usa los d* modos dominantes: x_{t+n} ≈ Σ_{k=1}^{d*} c_k · λ_k^n · φ_k(x₀)
```

---

## 10. Certificados Formales (TAA-1 a TAA-6)

Módulo Lean 4: `MathTest/TAAAgentCertificates.lean`  
Namespace: `TAAAgent`

| ID | Nombre | Enunciado | Status |
|---|---|---|---|
| TAA-1 | `koopman_spectrum_bounded` | $\|\mathcal{K}f\|_2 = \|f\|_2$ (isometría) | ✓ probado |
| TAA-1b | `koopman_eigenvalue_bound` | $\|\lambda\| \leq 1$ para eigenvalores | ✓ probado |
| TAA-2 | `acf_energy_invariant` | $E(f) = E(\Phi_{AC}(f))$ fragmento afín | ✓ probado |
| TAA-2b | `composed_energy_subadditive` | Energía compuesta subditiva | ✓ probado |
| TAA-2c | `horner_energy_optimal` | Horner logra energía mínima $d$ FMAs | ✓ probado |
| TAA-3a | `taa_budget_exists` | $\forall \varepsilon > 0, \exists d^*(\varepsilon)$ | **axioma** |
| TAA-3b | `taa_budget_exponential_decay` | Fórmula explícita para decaimiento exp. | ✓ probado |
| TAA-4 | `alpha_classifies_budget` | $\alpha_A$ determina clase de costo FMA | ✓ probado |
| TAA-4b | `exponential_cheaper_than_polynomial` | Exp. $<$ poly asintóticamente | ✓ **probado** (2026-05-06) |
| TAA-5 | `taa_measure_error_inflation` | Medida incorrecta infla $\delta(d)$ | ✓ probado |
| TAA-5b | `taa_ergon_interface_eliminates_inflation` | ERGON elimina la inflación | ✓ probado |
| TAA-6 | `taa_defer_to_ergon` | $\lambda_{\max} > 0$ → necesita $\mu_{SRB}$ | **axioma** |
| TAA-6b | `taa_acts_independently_for_integrable` | Sin caos, TAA actúa solo | ✓ probado |

**Placeholders activos:** 0  
**Axiomas restantes en Lean:** 3 (TAA-3a, TAA-6, TAA-12)  
**Teoremas demostrados nuevos (2026-05-05):**
- TAA-9 → `taa_ergon_lyapunov_calibration_proved` (demostrado — calibración Lyapunov)
- TAA-3c → `taa_budget_polynomial_decay` (demostrado — presupuesto decaimiento polinomial)
- TAA-7a → `spectral_entropy_nonneg` (demostrado — H(K) ≥ 0)
- TAA-7b → `spectral_entropy_zero_iff_one_mode` (demostrado — H = 0 ↔ espectro one-hot)

**Teoremas demostrados nuevos (2026-05-06) — verificados por `lake build`:**
- TAA-4b → `exponential_cheaper_than_polynomial` (testigo $\varepsilon_0 = \rho^{-1}$; usa `rpow_lt_rpow`)
- TAA-7 → `spectral_entropy_bounded` (prueba KL completa: $\sum p_k\log(p_k d) \geq 0$ implica $H \leq \log d$)
- TAA-9 compat → `taa_ergon_lyapunov_calibration` (delega a `_proved` via `neg_mul`)
- TAA-11b → `taa_budget_is_logarithmic` (testigo $C = 1/\Gamma + 1/\log(1/\varepsilon)$)

**Axiomas abiertos (3):** TAA-3a requiere teoría espectral completa de operadores compactos; TAA-6 requiere Fórmula de Pesin; TAA-12 requiere teoría de perturbación de Kato para operadores no normales.

### Descripción de Axiomas

**TAA-3a** (`taa_budget_exists`): La existencia de $d^*(\varepsilon)$ requiere teoría espectral completa de operadores compactos en $L^2(\mu_{SRB})$. El existencial es constructivo via KD-3 una vez conocida $\mu_{SRB}$.

**TAA-6** (`taa_defer_to_ergon`): La deducción de que caos genuino ($\lambda_{\max} > 0$, $\mathfrak{E} \approx 1$) implica necesidad de $\mu_{SRB}$ requiere la Fórmula de Pesin (ERG-6a) y clasificación de medidas SRB.

---

## 11. Interfaz TAA ↔ ERGON

### El Canal de Medida

La interfaz es unidireccional: ERGON produce $\mu_{SRB}$, TAA la consume.

```
ERGON.analyze(T, x0)
    ├─ report.mu_srb_density    ──→  TAA.analyze(T, x_data, mu_srb=...)
    ├─ report.recommended_d_star ──→  referencia para epsilon selection
    └─ report.handoff_to_taa    ──→  routing decision
```

### Cuándo Usar la Interfaz

| Condición | Acción | Resultado |
|---|---|---|
| `ergon_report.handoff_to_taa = True` | TAA solo, `mu_srb=None` | Sistema integrable, FMA exacto |
| `ergon_report.ergodic_complexity < 0.1` | TAA solo | Caos negligible |
| `0.1 ≤ ergon_complexity < 0.9` | TAA con `mu_srb` de ERGON | Caos mixto, $\delta(d)$ minimizado |
| `ergon_complexity ≥ 0.9` | Joint: ERGON → TAA | Caos dominante, Pesin activo |

### Código Completo: Análisis Conjunto

```python
from poema.ergon import ERGONAgent
from poema.taa_agent import TAAAgent
import numpy as np

# Sistema Lorenz 3D
def lorenz_step(x, dt=0.01, sigma=10, rho=28, beta=8/3):
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return x + dt * np.array([dx, dy, dz])

x0 = np.array([0.1, 0.0, 0.0])

# 1. ERGON primero: encontrar μ_SRB y diagnosticar
ergon = ERGONAgent(n_iterations=50_000)
ergon_report = ergon.analyze(lorenz_step, x0)

print(f"𝔈(T) = {ergon_report.ergodic_complexity:.3f}")
print(f"λ_max = {ergon_report.lambda_max:.4f}")
print(f"h_KS = {ergon_report.h_ks:.4f}")
print(f"Pesin OK: {ergon_report.pesin_verified}")
print(f"Handoff a TAA: {ergon_report.handoff_to_taa}")

# 2. TAA con μ_SRB: espectro Koopman correcto
x_data = np.array([lorenz_step(x0 + 0.01*np.random.randn(3))
                   for _ in range(1000)])

taa = TAAAgent()
taa_report = taa.analyze(
    lorenz_step,
    x_data,
    epsilon=1e-4,
    mu_srb=ergon_report.mu_srb_density  # ← elimina inflación TAA-5
)

print(f"\nAlpha class: {taa_report.alpha_class}")
print(f"d* = {taa_report.d_star}")
print(f"δ(d*) = {taa_report.delta_d:.6f} < {1e-4:.6f}")
print(f"FMA cost: {taa_report.fma_cost}")
print(f"Measure: {taa_report.measure_used}")
print(f"ERGON required: {taa_report.ergon_required}")
```

---

## 12. Integración con el Ecosistema

### Con Poema

TAA se integra con el compilador de Poema a través del backend Koopman:

```python
from poema import Poem, PoemCompiler
from poema.taa_agent import TAAAgent

# TAA diagnostica el sistema
agent = TAAAgent()
report = agent.analyze(T, x_data)

# Poema usa el diagnóstico para seleccionar el backend óptimo
compiler = PoemCompiler()
# report.alpha_class → selección de backend
# report.d_star → dimensión del espacio de Koopman
# report.fma_cost → presupuesto de compilación
```

### Con Gideon (GideonAgentRouter)

Gideon usa TAA como uno de sus agentes de análisis:

```python
# GideonAgentRouter (ver Gideon-guide.md)
from poema.taa_agent import TAAAgent
from poema.ergon import ERGONAgent

class GideonAgentRouter:
    def route(self, T, x0, x_data, epsilon):
        ergon = ERGONAgent()
        taa = TAAAgent()
        ergon_report = ergon.analyze(T, x0, epsilon)

        if ergon_report.handoff_to_taa:
            return AgentRouteResult(
                agent_used='taa',
                taa_report=taa.analyze(T, x_data, epsilon)
            )
        else:
            taa_report = taa.analyze(T, x_data, epsilon,
                                     mu_srb=ergon_report.mu_srb_density)
            return AgentRouteResult(
                agent_used='joint',
                taa_report=taa_report,
                ergon_report=ergon_report
            )
```

### Con el ACF (Invariante Primordial)

TAA es la extensión del ACF a sistemas no lineales via el levantamiento de Koopman:

```
ACF clásico:    E(f) = E(Φ_AC(f))     — para funciones analíticas
TAA (Koopman):  δ(d) ≤ |λ_{d+1}|     — para sistemas dinámicos
```

La clase `FINITE` del Alpha-A es el caso exacto: el espectro colapsa a un número finito de modos, y TAA reproduce exactamente el comportamiento del ACF clásico.

---

## 13. Ejemplos de Uso

### Ejemplo 1: Sistema Estable (Decaimiento Exponencial)

```python
import numpy as np
from poema.taa_agent import TAAAgent, AlphaClass

# Sistema lineal amortiguado: x_{t+1} = 0.9 * x_t
def stable_1d(x): return 0.9 * x
x_data = np.array([[0.9**k] for k in range(200)])

agent = TAAAgent()
report = agent.analyze(stable_1d, x_data, epsilon=1e-4)

assert report.alpha_class == AlphaClass.EXPONENTIAL
print(f"d* = {report.d_star}")     # 1 (solo el modo dominante)
print(f"δ = {report.delta_d:.6f}")
print(f"ERGON needed: {report.ergon_required}")  # False
```

### Ejemplo 2: Mapa Logístico en Caos

```python
import numpy as np
from poema.taa_agent import TAAAgent
from poema.ergon import ERGONAgent

# Mapa logístico r=4: caos máximo
def logistic(x): return np.array([4 * x[0] * (1 - x[0])])
x0 = np.array([0.3])

# Diagnóstico ERGON primero
ergon = ERGONAgent(n_iterations=50_000)
erg_report = ergon.analyze(logistic, x0)
print(f"𝔈 = {erg_report.ergodic_complexity:.3f}")  # ≈ 1.0

# TAA con μ_SRB correcto
x_data = np.vstack([logistic(np.array([0.3 + 0.01*k/1000]))
                    for k in range(1000)])
taa = TAAAgent()
report = taa.analyze(logistic, x_data, epsilon=1e-3,
                     mu_srb=erg_report.mu_srb_density)
print(f"Measure: {report.measure_used}")         # 'srb'
print(f"Inflation: {report.measure_inflation}")   # 0.0
```

### Ejemplo 3: Predicción Koopman

```python
import numpy as np
from poema.taa_agent import TAAAgent

# Oscilador armónico: x_{t+1} = R(θ) x_t
theta = 0.1
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
def harmonic(x): return R @ x

x0 = np.array([1.0, 0.0])
x_data = np.array([np.linalg.matrix_power(R, k) @ x0
                   for k in range(100)])

agent = TAAAgent()
report = agent.analyze(harmonic, x_data, epsilon=1e-6)

# Predecir 50 pasos
pred = agent.koopman_predict(report, x0, n_steps=50)
# pred shape: (50, 2)
```

---

## 14. Errores y Diagnóstico

### `ergon_required = True`

**Síntoma:** `taa_report.ergon_required = True`  
**Causa:** `lambda_max > 0.05` y no se proveyó `mu_srb`  
**Solución:** Ejecutar ERGONAgent primero y pasar `mu_srb=ergon_report.mu_srb_density`

### `alpha_class = UNKNOWN`

**Síntoma:** TAA no puede clasificar el decaimiento espectral  
**Causa:** Datos insuficientes (menos de 3 eigenvalores estimables)  
**Solución:** Aumentar `n_steps` en `x_data` — mínimo recomendado: `max(50, 10 * dim)`

### `delta_d > epsilon`

**Síntoma:** El error de truncamiento supera el objetivo  
**Causa:** `d_star` calculado insuficiente para el sistema dado  
**Solución:** 
- Verificar que `x_data` cubre el atractor suficientemente
- Proveer `mu_srb` de ERGON para eliminar inflación de medida
- Reducir `epsilon` y reanalizar

### `measure_inflation` Alto

**Síntoma:** `measure_inflation >> 0`  
**Causa:** La medida empírica difiere significativamente de $\mu_{SRB}$  
**Solución:** Usar ERGON para obtener $\mu_{SRB}$ y pasar `mu_srb=...` a TAA

---

## 15. Problemas Abiertos

### TAA-3a: Existencia General de d*(ε)

El axioma TAA-3a afirma $\forall \varepsilon, \exists d^*(\varepsilon)$ para operadores compactos generales en $L^2(\mu_{SRB})$. Requiere teoría espectral completa de operadores de Koopman — un resultado abierto en matemáticas modernas para sistemas caóticos generales.

**Progreso:** TAA usa ya las fórmulas de TAA-3b en la capa operativa y en Lean para el caso exponencial; lo pendiente es cerrar TAA-3a en general y los frentes asintóticos de comparación fina.

### TAA-6: Caracterización del Caos que Requiere ERGON

El axioma TAA-6 afirma que $\lambda_{\max} > 0$ con $\mathfrak{E} \approx 1$ implica necesidad de $\mu_{SRB}$. Ahora ya no depende de mantener ERG-6a como axioma independiente; la frontera dura quedó concentrada en ERG-5 y en la clasificación geométrica de medidas SRB.

**Cierre:** Se cierra cuando ERGONCertificates.lean cierre ERG-5 y la clasificación geométrica asociada.

### Conjetura del Espectro Unificado

**Conjetura (trabajo):** Los eigenvalores Koopman $\lambda_k^{\mathcal{K}}$ satisfacen:

$$\sum_k \text{Re}(\log \lambda_k^{\mathcal{K}}) = -h_{KS}(T)$$

Si se prueba, conectaría $\delta(d)$ de KD-1 con $h_{KS}$ de ERG-6a en una sola ecuación, cerrando TAA-3a y TAA-6 simultáneamente.

---

## 16. Nuevos Certificados (TAA-10 a TAA-12)

Certificados agregados en abril 2026, extendiendo la cobertura formal y la capa de certificados explícitos de TAA sobre propiedades intrínsecas del diccionario EDMD, umbrales de complejidad crítica, y corrección de error por proyección biortogonal.

### TAA-10: Índice de Adaptación de Base (IAB)

Mide la adecuación del diccionario EDMD a la estructura de Koopman:

$$\text{IAB} = \frac{N(K)}{N(K_{\text{Gaussian}})}$$

donde $N(K)$ es la **no-normalidad** de la matriz EDMD:

$$N(K) = \frac{\|KK^* - K^*K\|_F}{\|K\|_F^2}$$

**Interpretación:**
- $\text{IAB} \approx 0$: la base está bien adaptada a la estructura de Koopman — los modos EDMD aproximan eigenfunciones verdaderas
- $\text{IAB} \approx 1$: base genérica — el diccionario no captura la geometría espectral

**Descubrimiento clave:** $N(K)$ es intrínseco al diccionario EDMD, independiente de $\mu_{SRB}$. Esto permite diagnosticar la calidad de la base *antes* de invocar ERGON.

### TAA-11: Umbral de Complejidad Crítica E*

Conecta el presupuesto de TAA con el gap espectral OTU y la entropía ERGON:

$$E^* = 1 - \frac{\Gamma_{\text{OTU}}}{h_{KS}}$$

**Régimen subumbral** ($\mathfrak{E}(T) < E^*$):

$$d^*(\varepsilon) = O\!\left(\log \frac{1}{\varepsilon}\right) \quad \text{— TAA opera con presupuesto logarítmico}$$

**Régimen superumbral** ($\mathfrak{E}(T) > E^*$):

$$d^*(\varepsilon) = O\!\left(\varepsilon^{-1/(1-\mathfrak{E})}\right) \quad \text{— TAA necesita presupuesto polinomial}$$

Este certificado conecta el presupuesto de TAA ($d^*$) con el gap espectral de OTU ($\Gamma_{\text{OTU}}$) y la entropía de Kolmogorov-Sinai ($h_{KS}$) de ERGON — unificando los tres agentes en una sola desigualdad de umbral.

### TAA-12: Error de Proyección Biortogonal

Corrige el error de truncamiento usando eigenvectores izquierdos y derechos:

$$\Pi_d = \sum_{k=1}^{d} \frac{|r_k\rangle\langle l_k|}{\langle l_k | r_k \rangle}$$

donde $|r_k\rangle$ son eigenvectores derechos y $\langle l_k|$ son eigenvectores izquierdos de la matriz Koopman.

**Garantía:** Para $K$ no normal:

$$\delta_{\text{biorth}} \leq \delta_{\text{naive}} \quad \text{(siempre)}$$

La cota biortogonal es estrictamente mejor o igual que la cota ingenua basada solo en eigenvalores. Los eigenvectores izquierdos provienen de OTU (`koopman_modes` = eigenvectores izquierdos de $\mathcal{L}$).

### Tabla de Certificados Actualizada (TAA-10 a TAA-12)

| ID | Nombre | Enunciado | Status |
|---|---|---|---|
| TAA-10 | `basis_adaptation_index` | IAB = N(K)/N(K_Gaussian) intrínseco al diccionario | ✓ probado |
| TAA-11 | `critical_complexity_threshold` | E* = 1 - Γ_OTU/h_KS separa régimen log/poly | mixto: igualdad base probada, cota logarítmica como certificado explícito |
| TAA-11b | `taa_budget_is_logarithmic` | Budget logarítmico en $1/\varepsilon$ | ✓ **probado** (2026-05-06) |
| TAA-12 | `biorthogonal_projection_error` | δ_biorth ≤ δ_naive para K no normal | **axioma** |

---

## 17. Puente al Mundo Real (TAAAgentRealWorld)

Nueva clase `TAAAgentRealWorld` en `acf_functor/taa_agent.py` que tiende un puente sobre el **Abismo de Datos** (Barrera 1): la brecha entre señales reales ruidosas y el operador de Koopman limpio que TAA requiere.

### 17.1 Constructor desde Series Temporales

```python
TAAAgentRealWorld.from_timeseries(y, noise_filter="auto", n_obs=None)
```

**Pipeline interno:**

```
Señal cruda y(t)
       │
       ▼
  ┌──────────────────┐
  │  Filtro de ruido  │  ← noise_filter: "svd", "kalman", "wavelet",
  │                    │                  "particle", "auto"
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Embedding Takens │  ← Reconstrucción de atractor desde 1D
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Modelo lineal    │  ← Modelo local para EDMD
  │  local            │
  └────────┬─────────┘
           │
           ▼
       TAAAgent        ← Agente Koopman estándar
```

**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `y` | `np.ndarray` | — | Serie temporal cruda (1D o multivariada) |
| `noise_filter` | `str` | `"auto"` | Filtro de ruido: `"svd"`, `"kalman"`, `"wavelet"`, `"particle"`, `"auto"` |
| `n_obs` | `int \| None` | `None` | Número de observables para el embedding de Takens |

**Retorna:** Instancia de `TAAAgentRealWorld` con:
- `.taa` — el `TAAAgent` interno listo para `.analyze()` y `.diagnose()`
- `.reconstruction_info` — metadatos del pipeline (filtro usado, dimensión de embedding, error de reconstrucción)

**Entrada multivariada:** Si `y` es multivariada, se proyecta a 1D para el análisis de Koopman (componente principal de mayor varianza).

### 17.2 Rastreo de Koopman en Ventana Deslizante

```python
TAAAgentRealWorld.track_koopman(y, window_size=1000, step=100)
```

Rastrea propiedades espectrales de Koopman sobre ventanas deslizantes para detectar transiciones de régimen dinámico.

**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `y` | `np.ndarray` | — | Serie temporal (puede ser streaming) |
| `window_size` | `int` | `1000` | Tamaño de cada ventana de análisis |
| `step` | `int` | `100` | Desplazamiento entre ventanas |

**Retorna:** Diccionario con:
- `eigenvalue_trajectories` — trayectorias de eigenvalores Koopman por ventana
- `spectral_entropies` — entropía espectral en cada ventana
- `d_star_evolution` — evolución de $d^*(\varepsilon)$ a lo largo del tiempo
- `transitions` — lista de transiciones detectadas (cambio de clase de decaimiento)

**Detección de transiciones:** El método detecta cambios de clase de decaimiento (e.g., exponencial → caótico) comparando la clasificación Alpha-A entre ventanas consecutivas.

### 17.3 Ejemplo Completo de Uso

```python
from acf_functor.taa_agent import TAAAgentRealWorld

# Desde datos de sensores crudos
agent = TAAAgentRealWorld.from_timeseries(vibration_data, noise_filter="auto")
state = agent.taa.diagnose()

print(f"Alpha class: {state.alpha_class}")
print(f"d* = {state.d_star}")
print(f"Filtro usado: {agent.reconstruction_info['filter']}")

# Rastreo de evolución espectral sobre datos en streaming
tracking = TAAAgentRealWorld.track_koopman(streaming_data, window_size=1000)
for t in tracking["transitions"]:
    print(f"Cambio de régimen en t={t['time']}: {t['from']} → {t['to']}")
```

---

## 18. Tabla de Archivos Actualizada

(Ver tabla completa en la sección [Archivos Relacionados](#archivos-relacionados) a continuación.)

---

## 19. Shared Numerical Infrastructure (Epic 9)

TAA's numerical routines are backed by module-level shared instances from
`acf_functor.shared_numerics`, eliminating redundant initialisation across the
agent ecosystem (TAA, ERGON, OTU).

### Import

```python
from acf_functor.shared_numerics import LyapunovEstimator, SpectralClassifier, ChebyshevBasis
```

### Lyapunov Delegation

`TAAAgent.estimate_lyapunov()` now delegates entirely to `LyapunovEstimator`:

```python
# Inside taa_agent.py — module-level shared instances
_lyapunov  = LyapunovEstimator()
_spectral  = SpectralClassifier()
_chebyshev = ChebyshevBasis()
```

`LyapunovEstimator` caches results by `(id(T), domain, n_orbit)`. When ERGON
or OTU have already computed the Lyapunov exponent for the same system, TAA
returns the cached value instantly with zero recomputation.

### Public API

The public interface of `TAAAgent` is **unchanged** — all existing call sites
work identically. The delegation is an internal implementation detail.

### Archivos

| Archivo | Propósito |
|---|---|
| `acf_functor/shared_numerics.py` | `LyapunovEstimator`, `SpectralClassifier`, `ChebyshevBasis` |

---

## Archivos Relacionados

| Archivo | Propósito |
|---|---|
| `acf_functor/taa_agent.py` | Implementación Python de TAA (actual) |
| `acf_functor/real_world.py` | Bridge mundo real → agentes |
| `acf_functor/shared_numerics.py` | Infraestructura compartida TAA/ERGON/OTU |
| `poema/taa_agent.py` | Implementación Python de TAA (legacy) |
| `poema/ergon.py` | Agente dual ERGON |
| `MathTest/TAAAgentCertificates.lean` | Certificados formales TAA-1..TAA-12 |
| `MathTest/KoopmanDeltaCertificates.lean` | Cotas espectrales KD-1..KD-4 |
| `MathTest/ERGONCertificates.lean` | Certificados ERGON (interfaz TAA-5b) |
| `ERGON-manual.md` | Manual del agente dual |
| `Gideon-guide.md` | GideonAgentRouter (routing TAA/ERGON) |
| `Poema-manual.md` | Ecosistema completo Poema |
| `Paper.md` | Teoría madre ACF |

---

*"El Koopman pregunta: ¿qué función del sistema es invariante bajo la dinámica? TAA responde: esta, con error δ(d) < ε, en d* FMAs exactos."*
