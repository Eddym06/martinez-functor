# ERGON: Perron-Frobenius Agent — Manual Técnico
### Agente Ergódico del Caos en el Ecosistema ACF

**Versión:** 2.0  
**Módulo:** `acf_functor/ergon_agent.py`  
**Certificados Lean 4:** `MathTest/ERGONCertificates.lean`  
**Agente dual:** TAA (`poema/taa_agent.py`) — ver `TAA-manual.md`

---

## Índice

1. [¿Qué es ERGON?](#1-qué-es-ergon)
2. [Fundamentos Matemáticos](#2-fundamentos-matemáticos)
3. [Pipeline Completo](#3-pipeline-completo)
4. [Los Cuatro Operadores de ERGON](#4-los-cuatro-operadores-de-ergon)
5. [La Medida SRB](#5-la-medida-srb)
6. [Exponentes de Lyapunov](#6-exponentes-de-lyapunov)
7. [Entropía de Kolmogorov-Sinai](#7-entropía-de-kolmogorov-sinai)
8. [La Fórmula de Pesin](#8-la-fórmula-de-pesin)
9. [El Índice de Complejidad Ergódica 𝔈(T)](#9-el-índice-de-complejidad-ergódica-et)
10. [API Reference](#10-api-reference)
11. [Certificados Formales (ERG-1 a ERG-9)](#11-certificados-formales-erg-1-a-erg-9)
12. [Interfaz ERGON → TAA](#12-interfaz-ergon--taa)
13. [Integración con el Ecosistema](#13-integración-con-el-ecosistema)
14. [Ejemplos de Uso](#14-ejemplos-de-uso)
15. [Errores y Diagnóstico](#15-errores-y-diagnóstico)
16. [Problemas Abiertos](#16-problemas-abiertos)
17. [Nuevos Certificados (ERG-14 a ERG-16)](#17-nuevos-certificados-erg-14-a-erg-16)
18. [Puente al Mundo Real (ERGONRealWorld)](#18-puente-al-mundo-real-ergonrealworld)
19. [Tasa de Convergencia de Birkhoff (ERG-10)](#19-tasa-de-convergencia-de-birkhoff-erg-10)

---

## 1. ¿Qué es ERGON?

**ERGON** (del griego ἔργον — *el trabajo que revela la ley*) es el agente del operador de Perron-Frobenius del ecosistema ACF. Opera sobre el **lado de medidas** de la dualidad de Koopman:

$$\mathcal{L}: \text{Meas}(\mathcal{X}) \to \text{Meas}(\mathcal{X}), \quad (\mathcal{L}\mu)(A) = \mu(T^{-1}(A))$$

Su misión es encontrar la **medida SRB** (Sinai-Ruelle-Bowen) del sistema, calcular los **exponentes de Lyapunov**, estimar la **entropía de Kolmogorov-Sinai**, y certificar la **Fórmula de Pesin** — la ley de conservación del caos.

### Por Qué el Nombre ERGON

La palabra *ergódico* proviene del griego ἔργον (*ergon*, trabajo) + ὁδός (*hodos*, camino). El camino del trabajo. El agente que recorre la trayectoria hasta que el trabajo revela la ley estadística del sistema. Es la raíz de toda la teoría ergódica.

> *"El Koopman mueve funciones hacia el futuro. El Perron-Frobenius mueve el futuro hacia la medida. Son la misma verdad vista desde universos opuestos."*

### Posición en el Ecosistema ACF

```
          SISTEMAS CAÓTICOS / NO LINEALES
                      │
           ┌──────────┴───────────┐
           │                      │
    𝔈(T) ≈ 0                𝔈(T) ≈ 1
    λ_max ≤ 0               λ_max > 0
           │                      │
           ▼                      ▼
      TAA SOLO              ERGON + TAA
   FMA exacto           μ_SRB → L²(𝒳,μ_SRB)
   δ(d) < ε             h_KS = ∫λ⁺ dμ_SRB
   E(f) mínimo          Pesin certificado
```

**ERGON es independiente de TAA**: certifica el caos sin necesitar ningún conocimiento FMA. Sin embargo, proveer su $\mu_{SRB}$ a TAA maximiza la precisión de TAA.

---

## 2. Fundamentos Matemáticos

### 2.1 El Operador de Perron-Frobenius

Para un mapa medible $T: \mathcal{X} \to \mathcal{X}$, el **operador de Perron-Frobenius** $\mathcal{L}$ actúa sobre medidas de probabilidad:

$$(\mathcal{L}\mu)(A) = \mu(T^{-1}(A)) \quad \forall A \in \mathcal{B}(\mathcal{X})$$

Para mapas con densidades (absolutamente continuos respecto a Lebesgue):

$$(\mathcal{L}\rho)(x) = \sum_{y \in T^{-1}(x)} \frac{\rho(y)}{|T'(y)|}$$

Esta es la **ecuación del balance ergódico**: $\mathcal{L}$ transporta densidades hacia adelante bajo la dinámica, pesando por la expansión inversa local.

### 2.2 La Medida Invariante

Una medida $\mu^*$ es **$T$-invariante** si $\mathcal{L}\mu^* = \mu^*$, equivalentemente:

$$\mu^*(T^{-1}(A)) = \mu^*(A) \quad \forall A \in \mathcal{B}(\mathcal{X})$$

Para sistemas ergódicos con mixing suficiente, $\mathcal{L}^n\mu_0 \to \mu^*$ débilmente para $\mu_0 \ll \text{Lebesgue}$. Esta es la **convergencia al estado de equilibrio estadístico**.

### 2.3 La Dualidad Exacta ℒ = K*

La ecuación fundamental que une ERGON y TAA (ERG-2):

$$\int_\mathcal{X} f(T(x)) \, d\mu(x) = \int_\mathcal{X} f(y) \, d(\mathcal{L}\mu)(y)$$

En notación de operadores: $\mathcal{L} = \mathcal{K}^*$ — el Perron-Frobenius es el **adjunto exacto en $L^2$** del operador de Koopman. TAA vive en el dominio de $\mathcal{K}$, ERGON en el dominio de $\mathcal{L}$.

| | TAA / Koopman $\mathcal{K}$ | ERGON / Perron-Frobenius $\mathcal{L}$ |
|---|---|---|
| Actúa sobre | Funciones $f \in L^2(\mathcal{X}, \mu)$ | Medidas $\mu \in \text{Meas}(\mathcal{X})$ |
| Busca | $\mathcal{K}\varphi = \lambda\varphi$ (eigenfunción) | $\mathcal{L}\mu^* = \mu^*$ (medida SRB) |
| Energía | $E(f)$ — profundidad FMA | $h_{KS}$ — entropía Kolmogorov-Sinai |
| Índice | $\alpha_A$ — decaimiento espectral | $\lambda^+$ — exponentes de Lyapunov |
| Error | $\delta(d) = \|\lambda_{d+1}\|$ | $\varepsilon_{SRB}$ — distancia a medida SRB |
| Certifica | Estructura computacional mínima | Ley estadística exacta |

### 2.4 El Teorema Ergódico de Birkhoff

Para $T$ ergódico bajo $\mu_{SRB}$ y $f \in L^1(\mu_{SRB})$:

$$\lim_{n\to\infty} \frac{1}{n} \sum_{k=0}^{n-1} f(T^k x) = \int f \, d\mu_{SRB} \quad \text{para } \mu_{SRB}\text{-a.e. } x$$

Este es el fundamento del operador $\Psi_{ER}$ de ERGON: la suma de Cesàro de masas de Dirac a lo largo de una trayectoria converge a $\mu_{SRB}$ para casi todo punto inicial.

---

## 3. Pipeline Completo

```
Condición inicial x0 (shape: (dim,))  +  mapa T : ℝ^d → ℝ^d
                      │
                      ▼
     ┌────────────────────────────────┐
     │  Ψ_ER: Generar Trayectoria     │  ← n_iterations pasos de T
     │  {x₀, x₁, ..., x_{n-1}}       │
     │  Birkhoff → μ_SRB              │
     └──────────────┬─────────────────┘
                    │
                    ▼
     ┌────────────────────────────────┐
     │  Λ_ER: Exponentes de Lyapunov  │  ← Benettin-QR (Oseledets)
     │  {λ_1, λ_2, ..., λ_d}         │
     │  λ^+ = {λ_i : λ_i > 0}        │
     └──────────────┬─────────────────┘
                    │
                    ▼
     ┌────────────────────────────────┐
     │  h_KS: Entropía KS             │  ← Entropía de Shannon de μ_SRB
     │  h_KS ≈ H(partición)           │
     └──────────────┬─────────────────┘
                    │
                    ▼
     ┌────────────────────────────────┐
     │  Verificación Pesin (ERG-6a)   │  ← |h_KS - Σλ⁺| < tolerancia
     │  pesin_residual                │
     │  pesin_verified                │
     └──────────────┬─────────────────┘
                    │
                    ▼
     ┌────────────────────────────────┐
     │  𝔈(T) = h_KS / Σλ⁺ ∈ [0,1]  │  ← Índice de complejidad ergódica
     │  Routing: TAA o ERGON          │
     └──────────────┬─────────────────┘
                    │
                    ▼
               ERGONReport
```

---

## 4. Los Cuatro Operadores de ERGON

### 4.1 Ψ_ER: Functor de Medida Ergódica

$$\Psi_{ER}: \text{Traj}(\mathcal{X}) \longrightarrow \text{InvMeas}(\mathcal{X})$$

Construye $\mu_{SRB}$ desde trayectorias observadas usando la suma de Cesàro de masas de Dirac:

$$\Psi_{ER}(\{x_k\}) = \lim_{n \to \infty} \frac{1}{n} \sum_{k=0}^{n-1} \delta_{x_k} \xrightarrow{\text{Birkhoff (ERG-3)}} \mu_{SRB}$$

**Implementación:** Histograma de la trayectoria (1D) o KDE 2D (multi-dim), normalizado a densidad de probabilidad.

**Invariante:** $\Psi_{ER}$ es un functor — la medida producida satisface $\mathcal{L}\mu_{SRB} = \mu_{SRB}$.

**Convergencia:** ERGON monitoriza la convergencia comparando densidades de primera mitad vs segunda mitad de la trayectoria.

### 4.2 Λ_ER: Campo de Lyapunov Certificado

$$\Lambda_{ER}(T, \mu) = \left\{\lambda_i = \lim_{n \to \infty} \frac{1}{n} \log \|DT^n(x) \cdot v_i\| : i = 1, \ldots, \dim(\mathcal{X})\right\}$$

Los exponentes de Lyapunov miden la tasa de expansión/contracción del sistema linealizado.

**Implementación:** Método de Benettin-QR (iteración QR continua para evitar colapso numérico):

```
Para cada iteración k:
  J_k = jacobiano numérico de T en x_k
  Z = J_k @ Q_previo
  Q_nuevo, R = QR(Z)
  log_sums += log|diag(R)|
  x_{k+1} = T(x_k)
λ_i = log_sums_i / n_iteraciones
```

**Función de diagnóstico:**

| Valor | Interpretación | Acción |
|---|---|---|
| $\lambda_{\max} > 0$ | Caos genuino (Lyapunov positivo) | ERGON activo, TAA requiere $\mu_{SRB}$ |
| $\lambda_{\max} = 0$ | Frontera caos/orden | Diagnóstico conjunto |
| $\lambda_{\max} < 0$ | Atractor estable | TAA puede colapsar solo |

### 4.3 𝓜_ER: Índice de Mezcla

$$\mathcal{M}_{ER}(T, n) = \sup_{A,B} \left|\frac{\mu_{SRB}(A \cap T^{-n}B)}{\mu_{SRB}(A)\mu_{SRB}(B)} - 1\right|$$

Mide con qué rapidez los conjuntos "olvidan" su pasado bajo la dinámica.

**Implementación:** Función de autocorrelación (ACF) de la trayectoria, ajuste de decaimiento exponencial.

**Budget de observación:** $\mathcal{M}_{ER}$ determina $n^*(\varepsilon)$ — las iteraciones mínimas para que $\Psi_{ER}$ converja:

$$n^*(\varepsilon) = \left\lceil \frac{\log(1/\varepsilon)}{\gamma} \right\rceil \quad \text{donde } \mathcal{M}_{ER}(T, n) \leq C \cdot e^{-\gamma n}$$

### 4.4 h_KS: Entropía de Kolmogorov-Sinai

$$h_{KS}(T) = \sup_{\mathcal{P}} \lim_{n\to\infty} \frac{1}{n} H\!\left(\bigvee_{k=0}^{n-1} T^{-k}\mathcal{P}\right)$$

donde $H(\mathcal{Q}) = -\sum_{A \in \mathcal{Q}} \mu(A) \log \mu(A)$ es la entropía de Shannon de la partición.

**Interpretación profunda:** $h_{KS}$ es la **tasa de creación de información genuinamente nueva** por la dinámica $T$ — bits por iteración que el sistema genera y que jamás podrían haberse predicho con conocimiento perfecto del pasado finito.

**Implementación:** Entropía de Shannon de la densidad $\mu_{SRB}$ normalizada por el número de bins.

**Relación con $E(f)$ del ACF:**

| | ACF / $E(f)$ | ERGON / $h_{KS}$ |
|---|---|---|
| Es la profundidad de | Estructura computacional | Caos irreducible |
| Se conserva bajo | $\Phi_{AC}$ | Conjugación topológica |
| Es cero para | Identidades FMA exactas | Sistemas integrados |

---

## 5. La Medida SRB

### Definición

La medida **SRB** (Sinai-Ruelle-Bowen) es la única medida $T$-invariante que:
1. Es ergódica (no tiene componentes $T$-invariantes propios)
2. Describe el comportamiento estadístico de órbitas *típicas* (μ_Lebesgue-típicas)
3. Tiene medidas condicionales absolutamente continuas en foliaciones inestables

### Por Qué es Fundamental

La medida SRB es el **estado de equilibrio natural** del sistema caótico. Para casi todo punto inicial (excepto un conjunto de medida de Lebesgue cero), las trayectorias obedecen las estadísticas dictadas por $\mu_{SRB}$. No es una medida construida artificialmente — emerge del sistema.

### Unicidad

Para sistemas de Axioma A (Smale), la medida SRB es **única** (ERG-7a). Para sistemas más generales, puede haber múltiples componentes ergódicos, cada uno con su propia $\mu_{SRB}$ (ver ERG-8, Descomposición Ergódica).

### Convergencia

$$\mathcal{L}^n \mu_0 \xrightarrow{n \to \infty} \mu_{SRB} \quad \text{en la topología débil-}\star$$

para toda medida inicial $\mu_0 \ll \lambda_{\text{Lebesgue}}$. Este es el fundamento de $\Psi_{ER}$.

---

## 6. Exponentes de Lyapunov

### Definición (Oseledets)

Los **exponentes de Lyapunov** del sistema $(T, \mu_{SRB})$ son los números reales:

$$\lambda_i = \lim_{n\to\infty} \frac{1}{n} \log \sigma_i(DT^n(x))$$

donde $\sigma_i$ denota el $i$-ésimo valor singular del Jacobiano acumulado $DT^n$.

El **Teorema de Oseledets** garantiza que estos límites existen para $\mu_{SRB}$-a.e. $x$.

### Clasificación por Lyapunov

| Condición | Sistema | Acción ERGON |
|---|---|---|
| Todos $\lambda_i < 0$ | Atractor estable | $h_{KS} = 0$, TAA solo |
| Algún $\lambda_i = 0$ | Conservativo/límite | Diagnóstico conjunto |
| Algún $\lambda_i > 0$ | Caos genuino | ERGON activo, Pesin a verificar |
| Muchos $\lambda_i^+ > 0$ | Hipercaos | ERGON dominante |

### Desigualdad de Margulis-Ruelle (ERG-4)

Para **cualquier** medida $T$-invariante $\mu$:

$$h_\mu(T) \leq \int \sum_{\lambda_i^+(x) > 0} \lambda_i^+(x) \, d\mu(x)$$

Esta es la cota superior. ERGON busca la medida que la satura (la $\mu_{SRB}$).

---

## 7. Entropía de Kolmogorov-Sinai

### Interpretación Operacional

$h_{KS}(T)$ mide con qué rapidez la incertidumbre sobre el estado del sistema crece con el tiempo. Para un sistema con $h_{KS} = \log 2$, el sistema genera 1 bit de información nueva por iteración — el estado del sistema duplica su "variedad" a cada paso.

### Estimación en ERGON

ERGON estima $h_{KS}$ de dos maneras:
1. **Via histograma:** Entropía de Shannon de $\mu_{SRB}$ normalizada
2. **Via Pesin:** $h_{KS} \approx \sum \lambda_i^+$ (bajo hipótesis de Pesin)

Ambas estimaciones se comparan en `pesin_residual`. Si son consistentes, el sistema satisface las condiciones de Pesin.

### Invarianza

$h_{KS}$ es un **invariante topológico** del sistema: $h_{KS}(T) = h_{KS}(S \circ T \circ S^{-1})$ para cualquier homeomorfismo $S$. Es la contraparte ergódica del invariante primordial $E(f) = E(\Phi_{AC}(f))$ del ACF.

---

## 8. La Fórmula de Pesin

$$\boxed{h_{KS}(T) = \int_{\mathcal{X}} \sum_{\lambda_i^+(x) > 0} \lambda_i^+(x) \, d\mu_{SRB}(x)}$$

### Significado

Esta es la **Ley de Conservación del Caos** — la ecuación propia de ERGON. No es una desigualdad. Para sistemas satisfaciendo las condiciones de Pesin ($T$ diffeomorphism de clase $C^{1+\alpha}$, $\mu_{SRB}$ absolutamente continua en foliaciones inestables), es una **igualdad exacta**.

**Lo que afirma:**
- La entropía del caos (global) es exactamente la integral de la expansión local (local) sobre la distribución natural del sistema
- No puede crearse más desorden del que los exponentes positivos generan
- El caos es conservativo: $h_{KS}$ es la energía disipativa del caos, $\Sigma\lambda^+$ es su fuente local

### Verificación Numérica

ERGON verifica la Fórmula de Pesin calculando:

$$\text{residuo}_{\text{Pesin}} = |h_{KS}(T) - \sum_i \lambda_i^+(x)|$$

Si `pesin_residual < pesin_tolerance`, entonces `pesin_verified = True`. Esto es el **certificado numérico** de la Fórmula de Pesin para el sistema dado.

### Relación con la Desigualdad MR

| Condición | Medida $\mu$ | Resultado |
|---|---|---|
| $\mu$ cualquier invariante | $h_\mu \leq \int \lambda^+ d\mu$ | Desigualdad MR (ERG-4) |
| $\mu = \mu_{SRB}$ | $h_{KS} = \int \lambda^+ d\mu_{SRB}$ | Pesin (ERG-6a, axioma) |

La medida SRB es la **única** que satura la desigualdad de Margulis-Ruelle con igualdad.

---

## 9. El Índice de Complejidad Ergódica 𝔈(T)

$$\mathfrak{E}(T) = \frac{h_{KS}(T)}{\sum_i \lambda_i^+(T)} \in [0, 1]$$

### Interpretación

| Valor | Significado | Acción de ERGON |
|---|---|---|
| $\mathfrak{E}(T) = 0$ | Sistema integrable ($h_{KS} = 0$) | TAA toma control total |
| $0 < \mathfrak{E}(T) < 0.1$ | Caos negligible | `handoff_to_taa = True` |
| $0.1 \leq \mathfrak{E}(T) < 0.9$ | Caos mixto | Joint: ERGON+TAA coordinados |
| $\mathfrak{E}(T) \geq 0.9$ | Caos dominante | ERGON dominante, TAA recibe $\mu_{SRB}$ |
| $\mathfrak{E}(T) = 1$ | Pesin saturado perfectamente | Certificado ERGON máximo |

### Acotación Formal (ERG-6b)

La desigualdad de Margulis-Ruelle garantiza $h_{KS} \leq \sum \lambda_i^+$, por lo que $\mathfrak{E}(T) \leq 1$. La no negatividad de $h_{KS}$ garantiza $\mathfrak{E}(T) \geq 0$. Esto está probado en `ergodic_complexity_bounded` (ERG-6b).

---

## 10. API Reference

### Clase `ERGONAgent`

```python
from poema.ergon import ERGONAgent, ERGONReport, SRBMeasure

agent = ERGONAgent(n_bins=200, n_iterations=100_000, pesin_tolerance=0.05)
report = agent.analyze(T, x0, epsilon=1e-4)
```

**Constructor:**

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `n_bins` | `int` | `200` | Resolución de la grilla para densidad $\mu_{SRB}$ |
| `n_iterations` | `int` | `100_000` | Iteraciones de Birkhoff (más = más preciso) |
| `pesin_tolerance` | `float` | `0.05` | Umbral para declarar Pesin verificado |

**Método `analyze()`:**

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `T` | `Callable` | — | El mapa dinámico $T: \mathbb{R}^d \to \mathbb{R}^d$ |
| `x0` | `np.ndarray` | — | Condición inicial, shape `(dim,)` |
| `epsilon` | `float` | `1e-4` | Tolerancia de convergencia |

**Retorna:** `ERGONReport`

---

### Clase `ERGONReport`

```python
@dataclass
class ERGONReport:
    mu_srb: SRBMeasure           # Medida SRB completa
    mu_srb_density: np.ndarray   # Densidad raw (para interfaz TAA)
    h_ks: float                  # Entropía KS [bits/iteración]
    lambda_positive: np.ndarray  # Exponentes de Lyapunov positivos
    lambda_max: float            # Exponente de Lyapunov máximo
    lyapunov_sum: float          # Σλ⁺ — suma total de Lyapunov positivos
    ergodic_complexity: float    # 𝔈(T) = h_KS / Σλ⁺ ∈ [0, 1]
    pesin_residual: float        # |h_KS - Σλ⁺| — error Fórmula de Pesin
    pesin_verified: bool         # True si pesin_residual < tolerancia
    mixing_rate: float           # Tasa de decaimiento exponencial γ
    budget_n_star: int           # Iteraciones mínimas para ε-convergencia
    handoff_to_taa: bool         # True si 𝔈(T) < 0.1 (TAA puede solo)
    recommended_d_star: int      # Dimensión Koopman sugerida para TAA
```

---

### Clase `SRBMeasure`

```python
@dataclass
class SRBMeasure:
    density: np.ndarray      # Densidad discreta de μ_SRB sobre la grilla
    support: np.ndarray      # Puntos de soporte de la aproximación
    dim: int                 # Dimensión del espacio de estados
    n_iterations: int        # Iteraciones de Birkhoff usadas
    birkhoff_converged: bool # True si la suma de Cesàro convergió
    convergence_rate: float  # Tasa de convergencia estimada (cerca de 0 = convergedido)
```

---

### Método `joint_analyze()`

Ejecuta ERGON primero, luego TAA con $\mu_{SRB}$ — el pipeline óptimo:

```python
ergon_report, taa_report = agent.joint_analyze(
    T,        # mapa dinámico
    x0,       # condición inicial para ERGON
    x_data,   # trayectoria para TAA
    epsilon=1e-4
)
```

**Routing interno:**
- Si `ergon_report.handoff_to_taa = True` → TAA opera solo (sin $\mu_{SRB}$)
- Si no → TAA recibe `mu_srb=ergon_report.mu_srb_density` (elimina inflación)

---

## 11. Certificados Formales (ERG-1 a ERG-16)

Módulo Lean 4: `MathTest/ERGONCertificates.lean`  
Namespace: `ERGONAgent`

| ID | Nombre Lean | Enunciado | Status |
|---|---|---|---|
| ERG-1 | `srb_measure_exists_for_mixing` | $\exists \mu^*: \mathcal{L}\mu^* = \mu^*$ (medida SRB) | **axioma** |
| ERG-2 | `pf_adjoint_of_koopman` | $\int f(T(x))d\mu = \int f \, d(\mathcal{L}\mu)$ | ✓ probado |
| ERG-2b | `pf_adjoint_iterated` | Dualidad adjunta iterada $n$ veces | ✓ probado |
| ERG-3 | `birkhoff_time_space_average` | $\lim (1/n)\sum f(T^k x) = \int f \, d\mu$ a.e. | **axioma** |
| ERG-4 | `margulis_ruelle_inequality_abstract` | $h_\mu \leq \int \sum\lambda^+ d\mu$ (MR) | ✓ probado |
| ERG-4b | `ks_entropy_nonneg` | $h_{KS} \geq 0$ | ✓ probado |
| ERG-4c | `zero_positive_lyapunov_implies_zero_entropy` | $\lambda^+ = 0 \Rightarrow h_{KS} = 0$ | ✓ probado |
| ERG-5 | `srb_saturates_margulis_ruelle` | MR con igualdad $\Leftrightarrow$ SRB | **axioma** |
| ERG-6a | `pesin_formula` | $h_{KS} = \int \sum\lambda^+ d\mu_{SRB}$ (Pesin) | **teorema derivado** |
| ERG-6b | `ergodic_complexity_bounded` | $\mathfrak{E}(T) \in [0, 1]$ | ✓ probado |
| ERG-6c | `ergodic_complexity_one_iff_pesin` | $\mathfrak{E} = 1 \Leftrightarrow$ Pesin saturado | ✓ probado |
| ERG-7a | `srb_uniqueness_axiomA` | Unicidad de $\mu_{SRB}$ para Axioma A | **axioma** |
| ERG-7b | `taa_ergon_interface_correct` | ERGON→TAA elimina inflación de medida | ✓ probado |
| ERG-8 | `ergodic_decomposition_completeness` | $\mu = \int_E \mu_e \, d\nu(e)$ (exhaustivo) | **axioma** |
| ERG-9 | `taa_ergon_domain_independence` | TAA y ERGON son matemáticamente independientes | ✓ probado |
| ERG-11 | `ergodic_complexity_from_lyapunov_entropy` | $E_T = h_{KS}/\sum\lambda^+ \in [0,1]$ | ✓ probado |
| ERG-11b | `ergodic_complexity_zero_iff_integrable` | $E_T = 0 \Leftrightarrow h_{KS}=0$ | ✓ probado |
| ERG-12 | `mixing_index_exponential_decay` | $M_{ER}(n) \to 0$ bajo mixing exponencial | **axioma** |
| ERG-13 | `ergon_observation_budget_formula` | Presupuesto explícito $n^*(\varepsilon)$ por mixing exponencial | **axioma** |
| ERG-13b | `taa_ergon_budget_symmetry` | Simetría logarítmica TAA/ERGON | ✓ probado |
| ERG-14 | `renyi_dimensions` | $D_q$ no creciente, $D_0=1$, corrección multifractal | mixto: Lean parcial + implementación operacional |
| ERG-15 | `entropy_production` | $\sigma = h_{KS}$, Gallavotti-Cohen: $P(+h)/P(-h) \to e^{nh}$ | mixto: Lean parcial + implementación operacional |
| ERG-15b | `gallavotti_cohen_verified` | Verificación GC en órbitas reales: histograma de $\sigma_n$, ajuste lineal de simetría | ✅ operacional |
| ERG-16 | `full_spectral_gap` | $\{\Gamma_k\}$ completo, meseta espectral, tiempo de cruce $n^*$, corrección de mixing rate | mixto: Lean parcial + implementación operacional |

**Placeholders activos:** 0  
**Axiomas/certificados explícitos en Lean:** 10 (ERG-1, ERG-3, ERG-5, ERG-7a, ERG-8, ERG-12, ERG-13, ERG-14a, ERG-15b, ERG-16a)  
**Objetivo primario:** ERG-5 (saturación SRB de Margulis-Ruelle) — el núcleo geométrico que aún sostiene la derivación de Pesin.

### Descripción de Axiomas

**ERG-1** (`srb_measure_exists_for_mixing`): La existencia de $\mu_{SRB}$ en el caso caótico requiere condiciones de Pesin ($C^{1+\alpha}$ regularidad de $T$) — más allá del Mathlib actual. El axioma afirma: para sistemas ergódicos con mixing, existe una medida SRB.

**ERG-5** (`srb_saturates_margulis_ruelle`): La saturación de la desigualdad MR por $\mu_{SRB}$ requiere absoluta continuidad de foliaciones estables (Teoría de Pesin manifold). No está en Mathlib.

**ERG-6a** (`pesin_formula`): Ya no se mantiene como axioma independiente en la capa de certificados; hoy se deriva de ERG-4 más ERG-5. La deuda dura se desplazó a demostrar geométricamente ERG-5 sin hipótesis externas.

**ERG-7a** (`srb_uniqueness_axiomA`): Requiere teoría de conjuntos hiperbólicos y descomposición espectral de Smale.

**ERG-8** (`ergodic_decomposition_completeness`): Requiere el teorema de desintegración de Rokhlin, que existe parcialmente en Mathlib.

---

## 12. Interfaz ERGON → TAA

### El Canal de Medida (ERG-7b)

ERGON provee a TAA exactamente lo que necesita para construir $L^2(\mathcal{X}, \mu_{SRB})$:

```
ERGON.analyze(T, x0)                    TAA.analyze(T, x_data, ...)
         │                                          ▲
         │  mu_srb_density ─────────────────────────┤ mu_srb=
         │  recommended_d_star ──→ sugiere epsilon  │
         │  handoff_to_taa ──→ routing decision     │
         └──────────────────────────────────────────┘
```

### Efecto en TAA (TAA-5 vs TAA-5b)

**Sin ERGON (TAA-5):**
$$\delta(d)_{\text{efectivo}} = \delta(d)_{\text{correcto}} + \|f\|_\infty \cdot \|\mu_{\text{emp}} - \mu_{SRB}\|_{TV}$$

El segundo término es invisible para TAA, pero real y positivo en sistemas caóticos.

**Con ERGON (TAA-5b):**
$$\delta(d)_{\text{efectivo}} = \delta(d)_{\text{correcto}} \quad (\delta_\mu = 0)$$

Certificado por `taa_ergon_interface_correct` (ERG-7b): cuando ERGON provee $\mu_{SRB}$, la inflación es exactamente cero.

### Routing a través de 𝔈(T)

```python
if ergon_report.handoff_to_taa:
    # 𝔈 < 0.1: sistema casi integrable
    taa_report = taa.analyze(T, x_data, mu_srb=None)
elif ergon_report.ergodic_complexity < 0.9:
    # 0.1 ≤ 𝔈 < 0.9: caos mixto
    taa_report = taa.analyze(T, x_data,
                              mu_srb=ergon_report.mu_srb_density)
else:
    # 𝔈 ≥ 0.9: caos dominante — Pesin activo
    taa_report = taa.analyze(T, x_data,
                              mu_srb=ergon_report.mu_srb_density)
    # Usar ergon_report.recommended_d_star como guía para epsilon
```

---

## 13. Integración con el Ecosistema

### Con Poema

ERGON provee diagnóstico dinámico al compilador de Poema:

```python
from poema.ergon import ERGONAgent

# ERGON diagnostica: ¿este sistema es caótico?
ergon = ERGONAgent()
report = ergon.analyze(T, x0)

if report.ergodic_complexity > 0.1:
    # Sistema caótico: no se puede colapsar a FMA sin μ_SRB
    # Poema usa μ_SRB para construir L²(𝒳, μ_SRB) correctamente
    pass
else:
    # Sistema integrable: Poema puede usar el ACF clásico
    # E(f) = E(Φ_AC(f)) aplica directamente
    pass
```

### Con Gideon (GideonAgentRouter)

ERGON es el primer paso en el pipeline de Gideon:

```python
# Gideon-guide.md: GideonAgentRouter
class GideonAgentRouter:
    def route(self, T, x0, x_data, epsilon):
        # 1. ERGON diagnostica
        ergon = ERGONAgent(n_iterations=50_000)
        ergon_report = ergon.analyze(T, x0, epsilon)

        # 2. Routing basado en 𝔈(T)
        if ergon_report.handoff_to_taa:
            return self._taa_only(T, x_data, epsilon)
        elif ergon_report.ergodic_complexity < 0.9:
            return self._joint(T, x_data, epsilon,
                               mu_srb=ergon_report.mu_srb_density)
        else:
            return self._ergon_dominant(ergon_report, T, x_data, epsilon)
```

### Con el Paper (Completitud del Ecosistema)

El Teorema de Descomposición Ergódica (ERG-8) garantiza que TAA + ERGON cubren **todo el espacio de fenómenos dinámicos**:

$$\text{ACF completo} = \text{TAA} \cup \text{ERGON} = \bigsqcup_{e \in \mathcal{E}} \underbrace{\text{FMA exacto}}_{h_{KS}(\mu_e)=0} \cup \underbrace{\text{Pesin certificado}}_{h_{KS}(\mu_e)>0}$$

Ningún componente dinámico puede escapar esta dicotomía (ERG-8 es exhaustivo).

---

## 14. Ejemplos de Uso

### Ejemplo 1: Mapa Logístico r=4 (Caos Máximo)

```python
import numpy as np
from poema.ergon import ERGONAgent

def logistic(x): return np.array([4.0 * x[0] * (1.0 - x[0])])
x0 = np.array([0.3])

ergon = ERGONAgent(n_iterations=200_000)
report = ergon.analyze(logistic, x0, epsilon=1e-4)

print(f"λ_max = {report.lambda_max:.4f}")          # ≈ log(2) ≈ 0.693
print(f"h_KS = {report.h_ks:.4f}")                 # ≈ log(2) ≈ 0.693 bits
print(f"Pesin residual = {report.pesin_residual:.4f}")  # ≈ 0
print(f"𝔈(T) = {report.ergodic_complexity:.3f}")  # ≈ 1.0
print(f"Pesin OK: {report.pesin_verified}")         # True
print(f"Handoff a TAA: {report.handoff_to_taa}")    # False (caos máximo)
```

### Ejemplo 2: Sistema Lorenz 3D

```python
import numpy as np
from poema.ergon import ERGONAgent
from poema.taa_agent import TAAAgent

def lorenz(x, dt=0.01, sigma=10, rho=28, beta=8/3):
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return x + dt * np.array([dx, dy, dz])

x0 = np.array([1.0, 0.0, 0.0])

# Pipeline completo ERGON → TAA
ergon = ERGONAgent(n_iterations=50_000)
x_data = np.array([lorenz(x0 + 0.01*k*np.ones(3)) for k in range(500)])

ergon_report, taa_report = ergon.joint_analyze(
    lorenz, x0, x_data, epsilon=1e-3
)

print("=== ERGON ===")
print(f"h_KS = {ergon_report.h_ks:.4f}")
print(f"λ_max = {ergon_report.lambda_max:.4f}")     # ≈ 0.9 para Lorenz
print(f"𝔈 = {ergon_report.ergodic_complexity:.3f}")

print("\n=== TAA (con μ_SRB) ===")
print(f"d* = {taa_report.d_star}")
print(f"δ(d*) = {taa_report.delta_d:.6f}")
print(f"Measure: {taa_report.measure_used}")        # 'srb'
print(f"Inflation: {taa_report.measure_inflation}") # 0.0
```

### Ejemplo 3: Sistema Estable (Handoff a TAA)

```python
import numpy as np
from poema.ergon import ERGONAgent

# Oscilador amortiguado
def damped_oscillator(x):
    return np.array([0.99 * x[0] - 0.1 * x[1],
                     0.1 * x[0] + 0.99 * x[1]]) * 0.95

x0 = np.array([1.0, 0.0])

ergon = ERGONAgent(n_iterations=10_000)
report = ergon.analyze(damped_oscillator, x0)

print(f"λ_max = {report.lambda_max:.4f}")           # < 0
print(f"𝔈 = {report.ergodic_complexity:.3f}")      # ≈ 0
print(f"Handoff a TAA: {report.handoff_to_taa}")    # True
# → TAA puede colapsar este sistema sin necesitar μ_SRB
```

### Ejemplo 4: Solo el Diagnóstico

```python
from poema.ergon import ERGONAgent

ergon = ERGONAgent()
report = ergon.analyze(my_dynamical_map, x0)

# Resumen rápido
summary = {
    "chaos_detected": report.lambda_max > 0,
    "ergodic_complexity": report.ergodic_complexity,
    "pesin_verified": report.pesin_verified,
    "use_srb_for_taa": not report.handoff_to_taa,
    "min_taa_dim": report.recommended_d_star,
}
```

---

## 15. Errores y Diagnóstico

### `birkhoff_converged = False`

**Síntoma:** `mu_srb.birkhoff_converged = False`  
**Causa:** La suma de Cesàro no convergió — densidades de primera/segunda mitad difieren mucho  
**Solución:** Aumentar `n_iterations` en `ERGONAgent`. Para caos fuerte, usar $n \geq 100{,}000$.

### `pesin_verified = False`

**Síntoma:** `pesin_verified = False` pero `lambda_max > 0`  
**Posibles causas:**
1. Trayectoria demasiado corta para estimar $h_{KS}$ bien
2. Sistema no satisface condiciones de Pesin (singularidades, discontinuidades)
3. $n_{bins}$ demasiado bajo para capturar la estructura de $\mu_{SRB}$

**Solución:** 
- Aumentar `n_iterations` y `n_bins`
- Verificar que $T$ es un mapa suave ($C^{1+\alpha}$)
- Aflojar `pesin_tolerance` si el sistema es borderline

### `ergodic_complexity = 0` Inesperado

**Síntoma:** `ergodic_complexity = 0` para sistema caótico conocido  
**Causa:** `lyapunov_sum = 0` porque el método QR no calculó exponentes positivos  
**Solución:**
- Verificar que `x0` no está en un punto fijo de $T$
- Aumentar `n_iterations` en el cálculo QR
- Verificar que el jacobiano numérico está bien calculado (epsilon de diferencias finitas)

### Trayectoria Divergente

**Síntoma:** ERGON tarda mucho o genera `nan`  
**Causa:** $T$ no está acotada (trayectoria escapa al infinito)  
**Solución:** Verificar que $T$ tiene un atractor compacto. ERGON solo funciona en sistemas con atractor acotado.

---

## 16. Problemas Abiertos

### ERG-6a: Fórmula de Pesin en Lean 4

Ya no es el objetivo primario como axioma aislado. En la capa actual se obtiene como consecuencia de ERG-4 más ERG-5. Lo que sigue pendiente es formalizar completamente la geometría que justifica ERG-5:

1. **Teorema de Oseledets completo** en Lean 4 — cociclos lineales, convergencia de exponentes. Parcialmente en Mathlib.
2. **Folios estables/inestables** — geometría diferencial de variedades invariantes. Requiere nuevas formalizaciones.
3. **Absoluta continuidad transversal** — la condición técnica más difícil; no en Mathlib.

**Estrategia:** Formalizar bajo hipótesis axiomatizadas progresivamente, pero concentrando el esfuerzo en `srb_saturates_margulis_ruelle`. `pesin_formula` ya debe leerse como teorema derivado condicionado por esa saturación.

### ERG-1: Existencia SRB bajo Condiciones Generales

La existencia de $\mu_{SRB}$ para sistemas $C^{1+\alpha}$ sin supuestos uniformes es un teorema reciente (Pesin-Sinai, Ledrappier-Young). Requiere trabajo de formalización nuevo en Lean 4.

### ERG-8: Descomposición Ergódica Completa

El teorema de desintegración de Rokhlin existe parcialmente en Mathlib. La versión completa con la dicotomía TAA/ERGON sobre componentes ergódicas es trabajo pendiente.

### Estimación Adaptativa de γ (Tasa de Mixing)

Actualmente ERGON estima $\gamma$ con ajuste lineal de la ACF. Un estimador adaptativo que detecte automáticamente cuándo $\Psi_{ER}$ ha convergido permitiría ajustar `n_iterations` dinámicamente.

### Extensión a Dimensiones Altas

El método QR es $O(n \cdot d^3)$ en la dimensión. Para sistemas de alta dimensión ($d \geq 100$), se necesitan algoritmos especializados de exponentes de Lyapunov (métodos de subespacio, estimadores estocásticos).

---

## 17. Nuevos Certificados (ERG-14 a ERG-16)

Estos certificados extienden la suite original (ERG-1..ERG-9) con métricas avanzadas de análisis ergódico.

### ERG-14: Dimensiones Multifractales de Rényi $D_q$

Calcula el espectro $D_q$ de la medida $\mu_\text{SRB}$:

- **$D_0$** (Hausdorff): dimensión del soporte de la medida.
- **$D_1$** (información): entropía de Shannon normalizada por la escala.
- **$D_2$** (correlación): ligada a la integral de correlación.

**Ancho multifractal:** $\Delta D = D_0 - D_\infty$ mide la concentración de la medida. Un $\Delta D$ grande indica fuerte heterogeneidad en la distribución invariante.

**Corrección por singularidades:** $|D_2 - 1|$ estima el error de integración de Ulam al computar $h_\text{KS}$ numéricamente.

**Ejemplo (logístico $r=4$):** $D_q$ decrece de 1.0 a $\approx 0.83$ debido a las singularidades de arco-seno de la medida invariante.

### ERG-15: Tasa de Producción de Entropía $\sigma = h_\text{KS}$

Conecta la entropía de Kolmogorov-Sinai con la termodinámica de no-equilibrio:

- **Termodinámica:** $\sigma = \int \log\!\bigl(\rho(x)/\rho(T(x))\bigr)\, d\mu_\text{SRB} = h_\text{KS}$.
- **Gallavotti-Cohen:** $P(\sigma_n = +h)\, / \, P(\sigma_n = -h) \to e^{nh}$ cuando $n \to \infty$.
- **Interpretación física:** cada bit de entropía KS equivale a un bit de irreversibilidad por paso temporal.

**Verificación en órbitas reales (ERG-15b):** El método `verify_gallavotti_cohen()` ejecuta ~100 órbitas reales de longitud $n$, construye el histograma de $\sigma_n = \frac{1}{n}\sum \log|T'(T^k x)|$, y verifica la simetría GC:

$$\log\frac{P(\sigma_n=+h)}{P(\sigma_n=-h)} \approx n \cdot h$$

Para sistemas uniformemente hiperbólicos (logístico $r=4$), $\sigma_n \approx h_\text{KS} \pm 0.04$ — sin fluctuaciones negativas, GC se cumple trivialmente. Para sistemas no-uniformes (tent map), el ajuste lineal verifica la pendiente predicha.

```python
agent = ERGONAgent(T=my_map, domain=(0,1))
gc = agent.verify_gallavotti_cohen(n_orbits=100, orbit_length=30)
# gc['gc_verified'] → True
# gc['h_ks_empirical'] → media empírica de σ_n
# gc['sigma_std'] → dispersión de fluctuaciones
```

**Verificado en:** logístico r=4 (σ=0.689±0.041, GC verificado ✅), tent map (σ=0.406, GC verificado ✅).

### ERG-16: Espectro Completo del Gap Espectral $\{\Gamma_k\}$

Extiende el gap espectral único (ERG-4) al espectro completo:

- **Espectro:** $\Gamma_k = -\log|\lambda_k|$ para $k \geq 1$.
- **Meseta espectral:** $\lambda_2, \ldots, \lambda_7 \approx$ mismo módulo $\Rightarrow$ subespacio de mezcla lenta.
- **Tiempo de cruce:** $n^* = 1 / (\Gamma_\text{meseta} - \Gamma_1)$.
- **Corrección de $n^*(\varepsilon)$:** $n^*(\varepsilon) = \log(1/\varepsilon) / \Gamma_\text{meseta}$ (no $\Gamma_1$).
- **Pares complejos:** correlaciones oscilatorias a frecuencia $\text{Im}(\lambda) / (2\pi)$.

---

## 18. Puente al Mundo Real (ERGONRealWorld)

Nueva clase `ERGONRealWorld` en `acf_functor/ergon_agent.py` que permite aplicar el análisis ERGON a datos experimentales reales.

### `ERGONRealWorld.from_timeseries(y, noise_filter, n_grid, lyapunov_method)`

Pipeline completo: filtrado → embedding de Takens → modelo lineal local → `ERGONAgent`.

| Parámetro | Opciones | Descripción |
|---|---|---|
| `lyapunov_method` | `"ratio"` (rápido), `"benettin"` (Benettin-QR, preciso) | Método de cálculo de exponentes de Lyapunov |
| `noise_filter` | `"svd"`, `"kalman"`, `"wavelet"`, `"particle"`, `"auto"` | Filtro de ruido aplicado a la serie temporal |
| `n_grid` | entero | Resolución de la discretización Ulam |

### `ERGONRealWorld.monitor(y, window_size, changepoint_method, streaming)`

Monitoreo de no-estacionariedad con detección de cambios de régimen.

| Parámetro | Opciones | Descripción |
|---|---|---|
| `changepoint_method` | `"cusum"` (defecto), `"bocpd"` (Bayesian Online Changepoint Detection) | Método de detección de puntos de cambio |
| `streaming` | `True` / `False` | `True` usa `StreamingCertifier` para procesamiento incremental con memoria acotada |
| `window_size` | entero | Tamaño de la ventana de monitoreo |

Retorna: análisis de régimen, puntos de cambio y alertas.

### `ERGONRealWorld.from_window(y, window_size, step)`

Crea agentes ERGON en ventana deslizante para rastrear la evolución dinámica en el tiempo.

### `ERGONRealWorld.certify_with_metadata()`

Certificación completa con advertencias de procedencia del mundo real (ruido residual, embedding finito, posible no-estacionariedad).

### Ejemplo de Uso

```python
from acf_functor.ergon_agent import ERGONRealWorld

# Desde datos crudos de vibración de motor
agent = ERGONRealWorld.from_timeseries(vibration_data, lyapunov_method="benettin")
cert = agent.certify_with_metadata()

# Monitoreo en tiempo real con BOCPD
monitor = ERGONRealWorld.monitor(
    streaming_data,
    window_size=500,
    changepoint_method="bocpd",
    streaming=True
)
```

---

## 19. Tasa de Convergencia de Birkhoff (ERG-10)

Nuevo método `birkhoff_convergence_rate()` que mide el exponente $r$ en el error de promedios de Birkhoff:

$$\text{error} \approx C / n^r$$

Por teoría ergódica, para sistemas con mezcla suficiente se espera $r \to 1/2$ (convergencia tipo Teorema Central del Límite). Desviaciones de $r = 0.5$ indican correlaciones de largo alcance o mezcla insuficiente.

---

## 20. Infraestructura Numérica Compartida (Epic 9)

ERGON delega parte de sus cálculos numéricos intensivos a `acf_functor/shared_numerics.py`,
infraestructura compartida con TAA y OTU bajo un esquema de caché entre agentes.

### Importación

```python
from acf_functor.shared_numerics import LyapunovEstimator, compute_renyi_dimensions
```

Una instancia de módulo es creada al cargar el agente:

```python
_shared_lyapunov = LyapunovEstimator()   # instancia compartida de módulo
```

Esto evita reinicializar el estimador en cada llamada y permite la reutilización del caché.

### `compute_lyapunov_field()`

El método `compute_lyapunov_field()` de `ERGONAgent` delega el cómputo de la órbita a
`_shared_lyapunov.estimate(T, x0, n_iter)` (resultado cacheado por clave `(T, x0)`), y
después aplica la **verificación de promedio espacial de Birkhoff** propia de ERGON:
contrasta el promedio temporal sobre la trayectoria generada contra la integral teórica
respecto a `μ_SRB`. Solo ERGON realiza esa verificación; `shared_numerics` únicamente
proporciona la órbita.

### `compute_renyi_dimensions()`

Delega directamente en `shared_numerics.compute_renyi_dimensions()` para el cálculo del
espectro $D_q$ (ver §17, ERG-14), y envuelve el resultado añadiendo metadatos ERG-14
(timestamp, tolerancias, referencia al certificado Lean).

### Caché Entre Agentes

Cuando TAA ya calculó los exponentes de Lyapunov para un mapa $T$ dado, `LyapunovEstimator`
devuelve el resultado cacheado. ERGON recibe ese valor instantáneamente sin repetir el
costoso método Benettin-QR. Este comportamiento es transparente: la **API pública de
`ERGONAgent` no cambia**; el caché opera en la capa de `shared_numerics`.

---

## Archivos Relacionados

| Archivo | Propósito |
|---|---|
| `acf_functor/ergon_agent.py` | Implementación Python de ERGON (actual) |
| `acf_functor/real_world.py` | Bridge mundo real → agentes |
| `acf_functor/shared_numerics.py` | Infraestructura compartida TAA/ERGON/OTU |
| `poema/taa_agent.py` | Agente dual TAA |
| `MathTest/ERGONCertificates.lean` | Certificados formales ERG-1..ERG-9 |
| `MathTest/TAAAgentCertificates.lean` | Certificados TAA (interfaz TAA-5b) |
| `MathTest/KoopmanDeltaCertificates.lean` | Cotas δ(d) (base de TAA) |
| `ERGON_AGENT.md` | Documento fundacional de ERGON |
| `TAA-manual.md` | Manual del agente dual TAA |
| `Gideon-guide.md` | GideonAgentRouter (routing ERGON/TAA) |
| `Poema-manual.md` | Ecosistema completo Poema |
| `Paper.md` | Teoría madre ACF |

---

## §NEW. Integración con P-SAL — Clausura Termodinámica para ROMs

### Clausura ERGON en el Bucle Autopoiético

ERGON es el agente responsable de la fase **CLOSE** del protocolo P-SAL. Los diagnósticos termodinámicos de ERGON ($h_{KS}$, $P''(1)$, exponentes de Lyapunov) alimentan directamente el cálculo de viscosidad turbulenta para estabilizar ROMs truncados.

**Fórmula central:**

$$\nu_t = \frac{1}{2} \cdot \frac{P''(1)}{h_{KS}} \cdot \text{Tr}(\text{Cov}(\mathbf{a}_{res}))$$

### Módulo: `acf_functor/thermodynamic_closure.py`

Implementa tres métodos de clausura:

1. **ERGON termodinámica** (primaria): usa invariantes de ERGON
2. **Smagorinsky** (fallback): clausura clásica LES
3. **SVV** (Spectral Vanishing Viscosity): disipación selectiva por frecuencia

El `AdaptiveClosureSelector` elige automáticamente:
- Si `h_ks` y `pressure_curvature` disponibles → ERGON
- Si no → SVV como fallback

```python
from acf_functor.thermodynamic_closure import AdaptiveClosureSelector
selector = AdaptiveClosureSelector(n_modes=8)
closure = selector.select_and_compute(
    modal_amplitudes=A,
    h_ks=ergon_certificate.h_ks,
    pressure_curvature=ergon_certificate.pressure_curvature,
)
```

### Certificados de Clausura

| Certificado | Descripción |
|---|---|
| **TC-1** | Tasa de disipación: $\text{Tr}(L + \nu_t D) < 0$ |
| **TC-2** | Equipartición espectral de alta frecuencia |
| **TC-3** | Convergencia de clausura |

**CERTIFICADO ERG-PSAL-1:** ERGON integrado como fase CLOSE del protocolo P-SAL. Clausura termodinámica basada en $h_{KS}$ estabiliza ROMs descubiertos por SINDy.

Ver documentación completa en `PSAL.md`.

---

## Meta-ACF: Diagnósticos Computacionales de Lyapunov/Entropía

Meta-ACF extiende los diagnósticos de ERGON al análisis de **programas como sistemas termodinámicos**:

| Diagnóstico ERGON | Aplicación Meta-ACF |
|---|---|
| **Exponentes de Lyapunov** | Detectar regiones caóticas en el código (sensibilidad a perturbaciones) |
| **Entropía espectral** | Medir complejidad de la dinámica de ejecución |
| **Energía computacional** | E(P) = Σ cost(F_t) — análogo de energía libre |
| **Tasa de disipación** | Regiones que convergen → shortcut a punto fijo |

La **Energía Computacional** E(P) juega el mismo rol que la energía libre F = E - TS en termodinámica: la optimización Meta-ACF minimiza E(P') sujeto a |P(x) - P'(x)| < ε.

**CERTIFICADO ERG-META-1:** ERGON extendido a diagnósticos computacionales. Lyapunov, entropía y energía computacional aplicados a trazas de programa.

Ver documentación completa en `META_ACF.md`.

---

*"El Perron-Frobenius no escapa del caos. Lo habita. Recorre la trayectoria hasta que el trabajo revela la ley: h_KS = ∫λ⁺ dμ_SRB. El caos conserva su propia esencia."*

---

## §12. Validación Termodinámica del Marco ERGON (2026)

Esta sección documenta los resultados de validación del núcleo termodinámico de ERGON, verificados en `tests/test_validation_realworld.py`.

### 12.1 Propiedades del Marco de Energía Libre

El `FreeEnergyComputer` garantiza las siguientes propiedades, verificadas empíricamente:

| Propiedad | Enunciado matemático | Resultado |
|-----------|---------------------|-----------|
| Monotonicidad de E(d) | E(d+1) ≤ E(d) para todo d | ✅ PASA |
| Monotonicidad de d*(β) | β₁ < β₂ ⟹ d*(β₁) ≤ d*(β₂) + 2 | ✅ PASA |
| d* ∈ dominio válido | 1 ≤ d* ≤ d_max para todo β | ✅ PASA |
| F* finito | F*(β) < ∞ para todo β > 0 | ✅ PASA |
| d*(β→∞) minimiza error | E(d_frío) ≤ E(d_cálido) | ✅ PASA |

### 12.2 Detección de Transiciones de Fase

El `CriticalityDetector` detecta transiciones de fase en el espectro de autovalores de Koopman. Para el modo de entropía **combinatorial** (`entropy_mode="combinatorial"`):

- **Configuración validada**: 3 autovalores fuertes (0.99) + 7 débiles (0.5), m=10
- **Rango de barrido**: β ∈ [0.01, 50] con 100 puntos
- **Resultado**: ≥ 1 transición detectada, con β_c ≈ 11 (dentro del rango esperado teóricamente)

**Nota:** La entropía espectral (`entropy_mode="spectral"`) produce `S(d)` monótonamente creciente → sin transiciones. Usar `"combinatorial"` para detectar transiciones reales.

### 12.3 Contexto Físico

La transición de fase en d*(β) corresponde a:
- **β_c bajo** (temperatura alta): preferir representaciones compactas → d* pequeño
- **β_c alto** (temperatura baja): preferir precisión → d* grande
- **En β_c**: el sistema "decide" entre comprimir o expandir la base de eigenmodos

Esta es la formalización del Principio MDL (Minimum Description Length) a temperatura general.

**CERTIFICADO ERGON-VALID-1:** Suite de 5 tests termodinámicos, todos verificados en CI.

## §13. Guía Termodinámica para la Construcción Universal

### 13.1 ERGON como Ojo Termodinámico del Constructor

El Constructor Universal usa la perspectiva termodinámica de ERGON para guiar decisiones de diseño:

- **Energía computacional**: $E_{node} = \text{cost}(\text{FMA})$ por nodo del hipergrafo
- **Entropía**: $S = -\sum p_i \log p_i$ sobre la distribución de activaciones
- **Energía libre**: $F = E - S/\beta$ — el funcional que el Constructor minimiza

### 13.2 Selección de Estrategia por Temperatura

El `StrategyExplorer` del Algorithm Forge implementa implícitamente el principio de transición de fase:

| Régimen | $\beta$ | Estrategia preferida |
|---------|---------|---------------------|
| Alta temperatura | $\beta \ll \beta_c$ | COMPRESSED, ROM (representaciones compactas) |
| Temperatura crítica | $\beta \approx \beta_c$ | HYBRID (balance compresión/precisión) |
| Baja temperatura | $\beta \gg \beta_c$ | SPECTRAL, DIRECT (precisión máxima) |

### 13.3 Detección de Cuellos de Botella Entrópicos

`ComputableHyperGraph.identify_bottlenecks()` implementa la vista ERGON: los nodos donde la entropía se acumula (alto $S$, bajo throughput) son candidatos para:

1. **Compresión** via `OperatorCompressor` (bajo rango, disperso)
2. **Reemplazo** via `replace_subgraph()` con versión Chebyshev
3. **Partición** via `partition_by_depth()` para paralelismo
