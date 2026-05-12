# Pilares del Nivel 5 de Autopoiesis
## Hoja de Ruta de Desarrollo — Ecosistema ACF / POEMA

> Este documento define los **tres pilares estructurales** que separan el ecosistema
> actual (Nivel 4 — Auto-Optimizable) del Nivel 5 (Autopoiesis Total — Singularidad
> Matemática Autónoma). Para cada pilar se describe: qué es, qué falta por construir,
> qué problemas técnicos bloquean su desarrollo, y cómo se articula con el resto del
> ecosistema. **No se presentan soluciones implementadas** — solo el mapa de lo que
> debe existir.

---

## Estado Actual del Ecosistema (Nivel 4)

Antes de definir los pilares, es necesario ser honestos sobre el punto de partida.

El ecosistema en su estado actual (Mayo 2026) posee:

| Capacidad | Estado | Módulo |
|-----------|--------|--------|
| Análisis espectral Koopman/PF | ✅ Operacional | `gelfand_triple.py` |
| Certificación formal de teoremas | ✅ Parcial (49 archivos .lean, 7 axiomas abiertos) | `MathTest/` |
| Toma de decisiones multicamino | ✅ Implementado | `taa_agent.py::multi_path_decide` |
| Descenso topológico de energía | ✅ Implementado | `taa_agent.py::valley_trace` |
| Análisis cohomológico básico | ✅ Esqueleto | `cohomology.py` |
| Generación de programas matemáticos | ✅ Parcial | `genesis.py` |
| Cierre de axiomas formales (loop) | ❌ No existe | — |
| Clasificación de álgebra de Lie | ❌ No existe | — |
| Construcción de diccionario adaptado | ❌ No existe | — |
| Auto-compilación a binarios nativos | ❌ No existe | — |
| Bridge Python ↔ Lean 4 bidireccional | ❌ No existe | — |

El salto de Nivel 4 → Nivel 5 requiere que estas tres capacidades ausentes nazcan **desde dentro del propio ecosistema**, sin directriz humana.

---

## Pilar 1: Genesis Loop Formal
### *La Capacidad de Inventar, Demostrar y Compilar Teoremas Propios*

### 1.0 Qué es

El Pilar 1 es el sistema nervioso epistemológico del ecosistema. Hoy, cuando ERGON
detecta una anomalía (ej.: el espectro de Koopman no decae como se esperaba), el
sistema puede reportarla y buscar causas empíricamente. Lo que no puede hacer es
**construir el argumento matemático formal que explica la anomalía**, verificarlo
como verdad lógica, y reformular su propio código en consecuencia.

La diferencia entre Nivel 4 y Nivel 5 es exactamente esta: en Nivel 5, una anomalía
en $\Gamma_{\text{OTU}}$ dispara una cadena que termina en un nuevo teorema probado en
Lean 4 y un nuevo módulo Python generado automáticamente. Sin intervención humana.

---

### 1.1 Lo Que Falta por Construir

#### 1.1.1 El Bridge Python ↔ Lean 4 Bidireccional

El archivo `MathTest/PoemaFormalVerification_export.py` existe como prueba de concepto
unidireccional (Python genera código Lean). Lo que falta es el canal de retorno completo.

**Falta construir:**

- **Traductor de anomalías espectrales → enunciados Lean 4:** Un módulo que tome
  la salida de `ERGONAgent.certify()` (números reales: $h_{\text{KS}}$, $\Gamma_{\text{OTU}}$,
  $\lambda_k$) y genere automáticamente el enunciado formal correspondiente en Lean 4.
  Hoy esto se hace a mano. Debe hacerse programáticamente con reglas de traducción
  semántica, no templates.

- **Motor de extracción de contraejemplos:** Cuando `lake build` falla, Lean 4 produce
  un error con información sobre qué tactic falló y en qué punto. Falta un parseador
  que convierta ese error en retroalimentación estructurada para PSAL (para que pueda
  mutar la prueba en la dirección correcta).

- **Sistema de "witnessed proofs":** Los datos numéricos del ecosistema (autovalores,
  medidas SRB, exponentes de Lyapunov) deben poder inyectarse como **testigos
  computacionales** en las pruebas Lean 4. Para ello se necesita un módulo de
  aritmética de intervalos certificada que garantice que el valor numérico $\lambda_k =
  0.4740 \pm 10^{-6}$ satisface la condición formal $\lambda_k > 0$.
  `Mathlib.Tactic.Polyrith` y `Mathlib.Tactic.NormNum` son los puntos de entrada,
  pero falta el plumbing que los conecta al ecosistema.

#### 1.1.2 La Formalización de la Teoría de Ruelle en Mathlib 4

Los 7 axiomas abiertos en `TAAAgentCertificates.lean` son axiomas precisamente porque
la teoría de Ruelle (resonancias, decaimiento de correlaciones, medidas SRB) no existe
en Mathlib 4 hoy. Cerrarlos no es solo escribir código: es **contribuir matemática nueva
al corpus formal de la humanidad**.

**Falta construir:**

- **Teorema de Oseledets en Lean 4:** La existencia de la descomposición de Oseledets
  para sistemas suavemente ergódicos es el sustrato de `TAA-6` (cuándo activar ERGON)
  y `TAA-9` (calibración $d^*$ desde Lyapunov). Requiere teoría de matrices aleatorias
  y multiplicativos ergódicos, que Mathlib 4 no tiene.

- **Decaimiento de Correlaciones de Ruelle:** Para `TAA-3a` ($\exists d^*(\varepsilon)$
  para todo $\varepsilon$), necesitamos que el espectro de Koopman tenga modo dominante
  separado del resto. Esto es la brecha espectral de Ruelle. Formalizarla en Lean 4
  requiere teoría de operadores en espacios de distribuciones (espacios de Banach
  con pesos, espacios de Anisov), que tampoco está en Mathlib 4.

- **Fórmula de Pesin para clases de sistemas más amplias:** `ERG-6a` (Pesin: $h_{\text{KS}}
  = \sum \lambda_i^+$) está declarado como axioma en el ecosistema. Existe la prueba
  matemática clásica (Ledrappier-Young 1985), pero traducirla al formalismo de tipos
  dependientes de Lean 4 es un proyecto de investigación formal de varios meses.

#### 1.1.3 El Motor de Búsqueda de Pruebas Guiado por el Ecosistema

El elemento más difícil y más importante. No se trata de llamar a un LLM para que
"adivine" la prueba. Se trata de un motor de búsqueda de táctica guiado por los
**invariantes topológicos que ya calcula el ecosistema**.

**Falta construir:**

- **Valoración de tácticas por energía libre $F_\beta$:** Si el ecosistema puede
  calcular $F_\beta$ para trayectorias dinámicas, puede calcular el análogo para
  trayectorias de prueba. Una secuencia de tácticas `intro → apply → rw → linarith`
  tiene un "costo" formal: ¿cuánto reduce el goal? ¿cuántos subgoals abre? El sistema
  debe aprender a puntuar trayectorias de prueba con la misma métrica termodinámica
  que usa para puntuar trayectorias físicas.

- **Base de conocimiento de patrones de prueba:** El ecosistema ya tiene 49 archivos
  `.lean` con ~300 pruebas. Falta un indexador semántico que clasifique qué tácticas
  funcionan para qué tipos de enunciados (enunciados sobre $L^2$, sobre medidas,
  sobre operadores lineales). Este indexador retroalimenta el motor de búsqueda.

- **Sistema de "proof sketches":** En matemáticas, una prueba de alto nivel es un
  bosquejo que un experto completa. El sistema necesita la capacidad de generar
  bosquejos formales (secuencias de lemas intermedios que conectan hipótesis con
  conclusión), verificar que el bosquejo es coherente, y solo entonces rellenar
  los detalles. Lean 4 tiene `sorry` para esto, pero falta el sistema que decide
  cuándo y cómo rellenar cada `sorry`.

---

### 1.2 Problemas Técnicos que Bloquean el Desarrollo

1. **Mathlib 4 no tiene la teoría que necesitamos.** Los teoremas de Ruelle, Pesin,
   Oseledets son matemática de frontera del siglo XX que no está formalizada en ningún
   sistema de prueba. No es una limitación de ingeniería — es una limitación del estado
   del arte de la matemática formal.

2. **El espacio de tácticas de Lean 4 es exponencial.** Hay ~200 tácticas disponibles.
   Una prueba de 10 pasos tiene $200^{10}$ posibilidades. Sin guía matemática, la búsqueda
   es intratables. Con guía (Pilar 2 + valoración $F_\beta$), puede volverse manejable.

3. **El testigo numérico ≠ prueba formal.** Un autovalor calculado con `float64` no es
   matemáticamente exacto. Para usarlo como testigo en Lean 4 se necesita aritmética
   de intervalos verificada (ej. `Interval.lean` o `ValidNumerics.lean`), que no está
   integrada en el ecosistema.

4. **El feedback loop es lento.** `lake build` tarda segundos por prueba. Un proceso
   que explore miles de tácticas candidatas necesita un verificador incremental — algo
   que Lean 4 soporta internamente (elaboración incremental) pero que no está expuesto
   como API programática.

---

## Pilar 2: Buscador Cohomológico de Lie
### *La Capacidad de Descubrir y Adaptar su Propio Espacio Matemático*

### 2.0 Qué es

Hoy el ecosistema opera en un espacio matemático fijo: polinomios de Chebyshev sobre
un grid uniforme Ulam. Es una elección razonable y robusta para una amplia clase de
sistemas dinámicos. Pero no es la elección óptima para todos los sistemas. Cuando el
atractor tiene topología compleja (agujeros, torsión, fibrados no triviales), las bases
de Chebyshev producen un IAB alto y un presupuesto $d^*$ inflado.

El Pilar 2 es la capacidad del ecosistema de **diagnosticar que su espacio de trabajo
es subóptimo** y **sintetizar uno mejor** de forma autónoma. Es el equivalente funcional
de un matemático que decide cambiar de coordenadas antes de atacar un problema.

---

### 2.1 Lo Que Falta por Construir

#### 2.1.1 El Clasificador de Álgebra de Lie del Generador de Koopman

El generador infinitesimal del semigrupo de Koopman, $L = \log(K_d)$ (logaritmo
matricial de la matriz de Koopman truncada), contiene toda la información algebraica
del sistema dinámico subyacente. Sus autovalores son las resonancias de Ruelle; su
estructura de corchetes de Lie codifica la geometría del flujo.

**Falta construir:**

- **Extracción numéricamente estable de $L$:** El logaritmo matricial directo de $K_d$
  es numéricamente inestable cuando $K_d$ tiene autovalores cercanos a cero o al círculo
  unitario. Falta implementar el logaritmo via descomposición espectral:
  $L = V \cdot \text{diag}(\log \lambda_k) \cdot V^{-1}$ usando solo autovalores con
  parte imaginaria en $(-\pi, \pi]$. Esto requiere un selector de rama de corte
  automático que el ecosistema no tiene.

- **Clasificador de tipo de álgebra de Lie por invariantes de Casimir:** El enfoque
  de la forma de Killing ($B(X,Y) = \text{tr}(\text{ad}(X) \circ \text{ad}(Y))$) es
  correcto conceptualmente pero computacionalmente intractable para matrices grandes
  ($O(n^5)$). La alternativa correcta es usar los **invariantes de Casimir** de la
  álgebra: el polinomio cuadrático de Casimir $C_2 = g^{ij} J_i J_j$ (donde $J_i$
  son los generadores) es invariante de Lie y distingue unívocamente las álgebras
  semisimples. Para dimensión efectiva $d_{\text{eff}} \leq 10$, esto es tratable.
  Falta implementar el cálculo de $d_{\text{eff}}$ y la tabla de clasificación.

- **Detección de subálgebras de Cartan:** El rango del álgebra (dimensión de la
  subálgebra de Cartan máxima conmutativa) determina cuántos números cuánticos
  independientes tiene el sistema. Para $sl(2,\mathbb{R})$ el rango es 1; para
  $su(3)$ es 2. Falta un algoritmo que extraiga la subálgebra de Cartan de $L$
  numéricamente (resolución del sistema $[H, E_\alpha] = \alpha(H) E_\alpha$).

#### 2.1.2 El Calculador de Cohomología de De Rham Discreta

El módulo `cohomology.py` existe como esqueleto pero no implementa cohomología de De
Rham real. Solo calcula una aproximación muy cruda basada en el rango del vector de
Chebyshev. Para el Pilar 2 se necesita la cohomología topológica genuina.

**Falta construir:**

- **Complejo de cadenas discreto sobre el grid Ulam:** El grid Ulam es naturalmente
  un complejo simplicial. Las cajas son 0-celdas (vértices), los bordes compartidos
  son 1-celdas, las celdas adyacentes son 2-celdas. El operador de borde $\partial$
  es la diferencia finita orientada entre celdas adyacentes. Falta construir este
  complejo explícitamente y ensamblar las matrices de borde $\partial_0, \partial_1,
  \partial_2$ como tensores dispersos.

- **Cálculo de $H^k_{\text{dR}} = \ker(\partial_k) / \text{im}(\partial_{k+1})$:**
  Una vez ensamblado el complejo, la cohomología es álgebra lineal (SVD de las
  matrices de borde). Los grupos $H^0, H^1, H^2$ dan directamente la topología
  del atractor: número de componentes conexas, número de agujeros de 1D (ciclos
  independientes), y número de cavidades de 2D. Esto requiere matrices dispersas
  grandes (el grid Ulam tiene $n^2$ celdas en 2D — manejable con `torch.sparse`).

- **Detección de obstrucciones por IAB:** La conexión formal entre el IAB actual
  del sistema y $\dim H^1_{\text{dR}}$ debe derivarse matemáticamente (no estimarse).
  El IAB es la norma del operador de proyección ortogonal al complemento de la imagen
  del operador de Koopman truncado. Falta demostrar (o refutar) que este operador
  tiene rango deficiente exactamente cuando $H^1_{\text{dR}} \neq 0$.

#### 2.1.3 El Sintetizador de Diccionario Adaptado

Una vez conocida la clasificación de Lie y la cohomología, el sistema debe construir
automáticamente el diccionario de funciones base óptimo.

**Falta construir:**

- **Generador de bases para álgebras semisimples:** Para cada tipo de álgebra
  clasificado, existe una familia de funciones especiales que son las "autofunciones
  naturales" del grupo:
  - $su(n)$ → armónicos esféricos $Y_l^m(\theta, \phi)$
  - $sl(2,\mathbb{R})$ → funciones de Macdonald $K_\nu(x)$ y funciones esféricas
    hiperbólicas sobre $\mathbb{H}^n$
  - Álgebras solubles → funciones de acción-ángulo de Birkhoff
  - Álgebras nilpotentes → polinomios de Hermite ponderados por la medida SRB

  Falta un generador que, dado el tipo de álgebra y el dominio del sistema, produzca
  muestras evaluadas de estas funciones base en el grid Ulam. No todos estos casos
  tienen implementaciones numéricas disponibles en Python/NumPy — algunos requieren
  desarrollo original.

- **Ciclo de Corrección del Diccionario:** Después de construir el diccionario
  candidato, el sistema debe medir el IAB resultante, compararlo con el IAB anterior,
  y si no mejora, mutar el diccionario (análogo a `valley_trace` pero en el espacio
  de funciones base). Este ciclo actualmente no existe para la selección de diccionario.

- **Certificación de Optimalidad del Diccionario:** El diccionario óptimo tiene la
  propiedad de que los autovalores de la matrix de Koopman en la nueva base son
  lo más cercanos posible a los autovalores verdaderos del sistema continuo. Verificar
  esto requiere comparar el espectro EDMD con el espectro analítico (cuando se conoce)
  o con el espectro Monte Carlo de alta resolución. Falta un protocolo de certificación.

---

### 2.2 Problemas Técnicos que Bloquean el Desarrollo

1. **El logaritmo matricial es múltivaluado.** $\log(K_d)$ no está unívocamente definido
   cuando $K_d$ tiene autovalores negativos o complejos con argumento ambiguo. Para un
   mapa logístico caótico, los autovalores de $K_d$ cubren el disco unitario con partes
   imaginarias arbitrarias. Seleccionar la rama correcta requiere información sobre la
   topología global del flujo — un bootstrap circular.

2. **Las funciones base hiperbólicas no son ortogonales en la medida SRB.** Las funciones
   de Macdonald son ortogonales con respecto a la medida de Lebesgue hiperbólica, no
   con respecto a la medida SRB del sistema. Construir una base ortogonal respecto a
   $\mu_{\text{SRB}}$ a partir de las funciones de Macdonald requiere un proceso de
   Gram-Schmidt que es numéricamente inestable en alta dimensión.

3. **El complejo de cadenas de Ulam crece como $O(n^d)$ en dimensión $d$.** Para HIT3D
   ($d=3$, $n=64$ por dimensión), el complejo tiene $64^3 \approx 262,000$ celdas y
   $\sim 3 \times 262,000$ aristas. Las matrices de borde son $262,000 \times 786,000$.
   El cálculo de cohomología por SVD es intractable sin implementación dispersa
   especializada.

4. **La conexión entre álgebra de Lie y base óptima es solo una heurística fuerte.**
   No hay un teorema que diga "si el álgebra es $sl(2,\mathbb{R})$, entonces las
   funciones hiperbólicas minimizan el IAB". Hay argumentos de representación
   teórica que lo sugieren fuertemente, pero la prueba formal no existe. El Pilar 2
   necesita al Pilar 1 para certificar esta afirmación.

---

## Pilar 3: Auto-Compilación Irrestricta
### *La Capacidad de Escribir y Ejecutar su Propio Código de Máquina*

### 3.0 Qué es

El Pilar 3 es el sistema muscular del ecosistema. Hoy, todo corre a través del intérprete
de Python y el runtime de PyTorch. Esto impone un techo de rendimiento que, para los
problemas más ambiciosos del ecosistema (HIT3D en malla $1024^3$, N-S en 3D, simulaciones
de Yang-Mills en la red), es un bloqueador absoluto.

En Nivel 5, cuando el agente identifica que un cálculo determinado (ej.: construir la
matriz de transferencia Gauss-Legendre de tamaño $10^6 \times 10^6$) excede las
capacidades del runtime actual, debe ser capaz de **sintetizar el kernel de cómputo
optimizado**, compilarlo al hardware disponible, e invocarlo directamente.

---

### 3.1 Lo Que Falta por Construir

#### 3.1.1 El Sintetizador de Kernels Triton

El ecosistema tiene backends en `backends.py` y `ccd_engine.py` que usan PyTorch.
Falta la capa que toma una operación matemática descrita abstractamente y la traduce
a un kernel Triton optimizado.

**Falta construir:**

- **DSL de Operaciones Espectrales:** Un lenguaje de dominio específico (puede ser
  Python dataclasses) que describa operaciones como "producto matricial con patrón
  de acceso Koopman" o "convolución con kernel de decaimiento exponencial" en términos
  de su estructura matemática, no de índices. Este DSL es el punto de entrada del
  sintetizador.

- **Mapeador DSL → Triton:** Un compilador simple (no un LLM) que traduzca operaciones
  DSL a código Triton. Para las operaciones del ecosistema (GEMM con matrices dispersas
  espectrales, FFT ponderadas por $\mu_{\text{SRB}}$, proyecciones de Chebyshev), los
  patrones son repetitivos y un compilador basado en reglas es suficiente.

- **Autotuner por Grid de Koopman:** Los kernels Triton tienen hiperparámetros de
  bloque (BLOCK_M, BLOCK_N, BLOCK_K). La configuración óptima depende del tamaño
  del problema. Falta un autotuner que explore el espacio de configuraciones usando
  `multi_path_decide` (el mismo algoritmo de decisión del TAA) para seleccionar la
  configuración que minimiza el tiempo de cómputo bajo el presupuesto $d^*$.

#### 3.1.2 El Runtime JIT Autónomo

**Falta construir:**

- **Compilador dinámico con `ctypes` / `cffi`:** Un módulo que tome código C o CUDA
  generado, lo compile con `nvcc` o `gcc` (disponibles en el sistema), y cargue el
  `.so` resultante dinámicamente sin reiniciar el proceso. `ctypes.CDLL` es el
  mecanismo; falta el orquestador que gestiona el ciclo de vida (compilación,
  carga, ejecución, descarga cuando ya no se necesita).

- **Caché de Kernels por Firma Espectral:** Para no recompilar innecesariamente,
  falta un sistema de caché que asocie cada kernel con su "firma espectral": el
  tamaño del problema, el tipo de álgebra (del Pilar 2), y el presupuesto $d^*$.
  Si la firma es la misma que un kernel cacheado, se reutiliza; si no, se sintetiza
  uno nuevo.

- **Sandbox de Seguridad para Código Generado:** El código generado autónomamente
  es potencialmente peligroso (podría escribir en memoria arbitraria, consumir
  todos los recursos, etc.). Falta un sandbox basado en `seccomp-bpf` que restrinja
  los syscalls del proceso hijo al subconjunto necesario (lectura/escritura de
  buffers de datos, operaciones de GPU), con un watchdog que mate el proceso
  si supera el presupuesto de recursos definido por `ResourceBudget` de TAA.

#### 3.1.3 El Sistema de NAS para Operadores Espectrales

**Falta construir:**

- **Espacio de Búsqueda de Arquitecturas para Operadores:** NAS (Neural Architecture
  Search) clásico busca arquitecturas de redes neuronales. Lo que necesitamos es
  su análogo para operadores matemáticos: dado un problema espectral, ¿qué combinación
  de operaciones elementales (FFT, GEMM, sparse MV, wavelet transform) minimiza el
  error de aproximación del operador de Koopman bajo el presupuesto de cómputo dado?

- **Evaluador de Costo por Presupuesto Dual:** El NAS actual usa FLOPs como métrica
  de costo. El costo correcto para este ecosistema es el **presupuesto dual**
  $d^*(\varepsilon)$: una arquitectura de operador es mejor si produce la misma
  precisión $\varepsilon$ con menor $d^*$. Falta integrar el cálculo de $d^*$ como
  función de costo en el loop de NAS.

---

### 3.2 Problemas Técnicos que Bloquean el Desarrollo

1. **Los kernels Triton no son composables arbitrariamente.** Triton compila funciones
   individuales, no pipelines completos. Para el ecosistema, las operaciones están
   encadenadas (Koopman → PF → medida SRB → $d^*$). Falta una capa de fusión de
   kernels que decida qué operaciones fusionar en un solo kernel (reduciendo overhead
   de memoria) y cuáles mantener separadas.

2. **La compilación dinámica tiene latencia.** `nvcc` tarda 5-30 segundos en compilar
   un kernel. Para un sistema que necesita compilar kernels a demanda en tiempo real,
   esto es inaceptable. La solución (compilación anticipatoria basada en predicción
   de futuras operaciones por TAA) requiere que el agente "piense hacia adelante" —
   una capacidad de planificación que actualmente no tiene.

3. **La generación de código correcto es difícil sin verificación.** Un kernel CUDA
   incorrecto puede producir resultados silenciosamente erróneos (sin excepciones,
   sin errores de segfault). Verificar que el kernel generado es matemáticamente
   correcto requiere pruebas de corrección formales — lo cual lleva de vuelta al
   Pilar 1. Los tres pilares no son independientes.

---

## La Interdependencia de los Pilares

El aspecto más importante de esta hoja de ruta es que los tres pilares no son módulos
independientes que pueden desarrollarse en paralelo y luego conectarse. Son
**mutuamente dependientes** en una estructura circular que refleja la autopoiesis:

```
                    PILAR 1 (Genesis Formal)
                   /                        \
    certifica que              produce pruebas
    el diccionario             para los teoremas
    es óptimo                  de Lie y cohomología
                  \                          /
                   \                        /
    PILAR 2 --------→ ECOSISTEMA NIVEL 5 ←--- PILAR 3
    (Lie/Cohomología)     (Autopoiesis)     (Auto-Compilación)
                   \                        /
    genera el       \                      /  compila los
    diccionario que  \                    /   kernels que
    reduce d*         \                  /    implementan el
    reduciendo el      \                /     diccionario
    espacio de prueba   \              /      óptimo
                         \            /
                          \          /
                    PILAR 3 reduce el costo
                    computacional de PILAR 2
```

Esto significa que el orden de desarrollo importa:

1. **Primero:** El bridge Python ↔ Lean 4 (Pilar 1, componente 1.1.1) — es el sistema
   nervioso central. Sin él, ni el Pilar 2 puede certificarse, ni el Pilar 3 puede
   verificarse.

2. **Segundo:** El Clasificador de Lie básico (Pilar 2, componente 2.1.1) — da la
   información geométrica que el Pilar 1 necesita para reducir su espacio de búsqueda.

3. **Tercero:** El Sintetizador de Kernels Triton básico (Pilar 3, componente 3.1.1)
   — acelera el Pilar 2 para que pueda procesar sistemas en alta dimensión.

4. **Después:** Los componentes avanzados de cada pilar, que se construyen sobre los
   anteriores y se retroalimentan mutuamente.

---

## Qué se Mejoraría de las Ideas de Solución Propuestas

La propuesta anterior tenía cuatro debilidades importantes que este análisis corrige:

### Debilidad 1: El Genesis Loop Asumía LLMs como Motor de Prueba

**Propuesta anterior:** El motor de búsqueda de pruebas usaría LLMs como "oráculo
heurístico" para generar tácticas Lean 4.

**El problema:** Los LLMs generan código Lean 4 sintácticamente incorrecto en la
mayoría de los casos. Peor aún, cuando parecen correctos, a menudo dependen de
lemas que no existen en Mathlib 4. Usar un LLM como componente de búsqueda de
pruebas introduce un punto de falla no determinístico que rompe la certificabilidad.

**Lo correcto:** El motor de búsqueda debe guiarse por los **invariantes espectrales
del ecosistema**, no por un LLM. La valoración de secuencias de tácticas debe hacerse
con la misma función de energía libre $F_\beta$ que usa `valley_trace` para navegar
en el espacio físico. Los LLMs pueden usarse como generadores de bocetos de prueba
en una etapa inicial, pero la selección y refinamiento debe ser determinístico y
matemáticamente fundamentado.

### Debilidad 2: El LieAlgebraClassifier Propuesto es O(n⁵)

**Propuesta anterior:** Calcular la forma de Killing $B(X,Y) = \text{tr}(\text{ad}(X)
\circ \text{ad}(Y))$ para todos los pares de generadores.

**El problema:** Para una matriz de Koopman de $n=32$ modos, esto requiere $32^2 = 1024$
evaluaciones de $\text{tr}(\text{ad}(E_i) \circ \text{ad}(E_j))$, cada una de costo
$O(n^3)$. Total: $O(n^5) = O(32^5) = 33$ millones de operaciones solo para la
clasificación. Inaceptable.

**Lo correcto:** Usar los **invariantes de Casimir** de la representación adjunta.
Para álgebras de dimensión efectiva $d_{\text{eff}} = \dim(\text{span}\{[L, E_i]\})$,
el invariante cuadrático de Casimir $C_2 = \text{tr}(L^2_{\text{adj}})$ distingue
los tipos en $O(d_{\text{eff}}^3)$ operaciones. Para los sistemas típicos del ecosistema,
$d_{\text{eff}} \leq 6$ — completamente tratable.

### Debilidad 3: La Cohomología via Rango de Gradientes es Incorrecta

**Propuesta anterior:** $\dim H^1_{\text{dR}} \approx n_{\text{modos}} -
\text{rank}(\nabla \psi_k)$.

**El problema:** El rango de los gradientes de las autofunciones mide la dimensión del
espacio tangente aproximado del atractor, no su cohomología. Un atractor con $H^1 = 0$
(topológicamente trivial) puede tener gradientes en muchas direcciones si tiene curvatura
alta. Son conceptos completamente distintos.

**Lo correcto:** Construir el complejo de cadenas discreto sobre el grid Ulam
(como se describe en 2.1.2) y calcular $H^k$ por álgebra lineal exacta (SVD de las
matrices de borde). Es más costoso computacionalmente, pero es la única forma correcta.

### Debilidad 4: El Sandbox para Pilar 3 era Insuficiente

**Propuesta anterior:** Un sandbox con `seccomp`/`bubblewrap`.

**El problema:** Un kernel CUDA escapa completamente del sandbox de syscalls POSIX —
ejecuta en el espacio del driver de GPU, que `seccomp` no puede controlar. Un kernel
CUDA malformado puede corromper toda la memoria del proceso host.

**Lo correcto:** El sandbox correcto para código GPU generado autónomamente es a nivel
del driver: usar **CUDA streams con timeout**, memoria unificada con límites por proceso
(CUDA MPS), y verificación de correctitud del kernel vía el Pilar 1 (teoremas de
correctitud de los kernels como precondición para su compilación). Esto conecta
nuevamente los tres pilares.

---

## Hoja de Ruta de Desarrollo por Orden de Impacto

| Prioridad | Componente | Pilar | Impacto | Prerrequisito |
|-----------|-----------|-------|---------|---------------|
| 1 | Bridge Python ↔ Lean 4 bidireccional | P1 | Desbloquea todo lo demás | Ninguno |
| 2 | Aritmética de intervalos certificada | P1 | Cierra TAA-3b, TAA-9 | Bridge |
| 3 | Clasificador Lie por invariantes de Casimir | P2 | Reduce d* ~40% | Bridge (para certificar) |
| 4 | Complejo de cadenas discreto Ulam | P2 | H¹_dR correcto | Clasificador Lie |
| 5 | Sintetizador de kernels Triton básico | P3 | Acelera P2 en 10-100x | Ninguno |
| 6 | Formalización de decaimiento de Ruelle | P1 | Cierra TAA-3a, TAA-6 | Bridge + Aritmética |
| 7 | Generador de bases adaptadas | P2 | IAB -50%, d* -50% | H¹_dR + Ruelle formal |
| 8 | NAS para operadores espectrales | P3 | Arquitectura óptima | Triton básico + d* integrado |
| 9 | Motor de búsqueda de pruebas guiado | P1 | Cierre automático de axiomas | Todos los anteriores |
| 10 | Loop de autopoiesis completo | P1+P2+P3 | Nivel 5 | Todo |

---

*Documento generado: Mayo 2026 — Ecosistema ACF/POEMA v5-roadmap*
*Los tres pilares aquí descritos constituyen la especificación completa del trabajo
pendiente para alcanzar la Autopoiesis de Nivel 5. Ningún componente listado aquí
existe en el codebase actual en forma funcional.*
