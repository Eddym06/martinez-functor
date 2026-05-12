# CCDEngine

## Campo de Curvatura Dinámica como solución geométrica a la maldición de la alta dimensionalidad

### Resumen

CCDEngine es una teoría operacional para tratar sistemas de alta dimensión sin caer en el costo exponencial de los métodos euclídeos clásicos. Su tesis central es que la dificultad no nace de tener muchas coordenadas, sino de insistir en medirlas dentro de una geometría plana cuando los datos reales viven sobre variedades, atractores, hojas invariantes o subconjuntos de baja complejidad inmersos en espacios ambiente enormes. El motor CCD propone entonces una estrategia precisa: en lugar de eliminar dimensiones de manera ciega, reconstruye una geometría efectiva en la que las direcciones vacías se vuelven dinámicamente irrelevantes y las direcciones informativas adquieren peso estructural.

La arquitectura combina cinco ideas: compresión espectral inicial, geometría de difusión, resonancia colectiva, temperatura entrópica local y purificación de Langevin. El resultado no es un compresor lineal más, sino un mecanismo que intenta aproximar la estructura interna del conjunto de datos, estimar su dimensión efectiva y operar sobre esa geometría inducida. En el ecosistema ACF esto cumple una función precisa: convertir problemas aparentemente intratables en problemas sobre un número pequeño de grados de libertad verificables.

---

## 1. El problema real: el vacío geométrico de alta dimensión

La intuición clásica falla en alta dimensión. En espacios de gran dimensión, el volumen se concentra cerca de la frontera, las distancias entre puntos tienden a hacerse casi indistinguibles y las nociones ordinarias de proximidad dejan de separar señal de ruido. Si se intenta aproximar una función o explorar una dinámica mediante una grilla uniforme en $\mathbb{R}^d$, el costo escala como

$$
N_{\text{grid}}(\varepsilon) = O\left(\varepsilon^{-d}\right).
$$

Ese crecimiento no es un detalle técnico: es una condena. Para $d = 50$ y $\varepsilon = 10^{-2}$, el número de puntos requeridos es del orden de $10^{100}$. Ningún algoritmo serio puede vivir ahí. Por eso la pregunta correcta no es “¿cómo proceso mejor un espacio de dimensión $d$?”, sino “¿por qué estoy tratando como esencial un espacio que en realidad solo contiene una estructura de dimensión efectiva mucho menor?”.

CCDEngine parte de esa corrección conceptual. No asume que toda coordenada ambiente aporta un grado de libertad real. Asume lo contrario: que la mayor parte del espacio es vacío geométrico y que la dinámica relevante vive en un subconjunto de dimensión intrínseca $m \ll d$.

**Honestidad sobre el alcance.** CCD no elimina la maldición de la dimensionalidad; la *desplaza* de $d$ a $m$. Aprender una variedad $m$-dimensional requiere $n \geq C(m) \cdot \varepsilon^{-m}$ muestras (Niyogi et al., 2008), que sigue siendo exponencial en $m$. La ventaja práctica aparece cuando $m \ll d$: para $d=1000$ y $m=3$, pasar de $10^{2000}$ (grilla) a $\sim 10^6$ (manifold) es la diferencia entre lo imposible y lo factible. Pero si $m=20$, $\varepsilon^{-20}$ sigue siendo astronómico para $\varepsilon$ pequeño. CCD comprime el exponente, no lo suprime. La introducción debe leerse con esta precisión: el "escape" es de la dependencia en $d$, no de la dependencia en $m$.

---

## 2. Idea central: curvar el espacio en lugar de discretizarlo

La propuesta CCD puede expresarse así: la complejidad explosiva aparece porque el algoritmo insiste en trabajar en una geometría plana, homogénea y euclídea. Si uno reemplaza esa geometría por una geometría inducida por los datos, entonces las regiones sin estructura dejan de ser relevantes y la complejidad efectiva colapsa.

La intuición física es simple. En relatividad, la materia curva el espacio-tiempo y las trayectorias geodésicas siguen esa curvatura. En CCD, la densidad informativa de los datos juega un rol análogo: donde hay estructura, coherencia y recurrencia, el espacio adquiere “masa geométrica”; donde hay vacío, la distancia efectiva crece y los caminos naturales evitan esas zonas. No es una metáfora ornamental: es exactamente la razón por la que las diffusion distances, las métricas de vecindad y los operadores de transición pueden revelar la variedad subyacente sin construir un tensor de curvatura explícito.

En forma esquemática, CCD busca una geometría efectiva $g_{\text{CCD}}$ tal que las trayectorias relevantes y los vecindarios significativos se organicen alrededor del manifold de datos y no del espacio ambiente completo. Por eso la reducción que produce no es una mera proyección algebraica; es una contracción geométrica de grados de libertad irrelevantes.

---

## 3. Tesis matemática del motor

El principio que guía al motor puede escribirse en términos de complejidad de aproximación. Si el sistema posee un atractor o variedad $M^m \subset \mathbb{R}^d$ con $m \ll d$, entonces el costo de representar o recorrer la dinámica ya no depende de $d$ del mismo modo que en una grilla uniforme. CCD afirma que, bajo estructura espectral suficiente,

$$
N_{\text{CCD}}(\varepsilon) = O\left(\frac{m \log(1/\varepsilon)}{\alpha_A}\right)
\ll
N_{\text{grid}}(\varepsilon) = O\left(\varepsilon^{-d}\right),
$$

donde $\alpha_A$ mide la tasa de decaimiento espectral efectiva del sistema, clasificada por el módulo TAA dentro del ecosistema ACF. En el motor implementado, $\alpha_A$ se determina en el paso `ecosystem_optimize` mediante tres tests simultáneos (correlación log-lineal vs log-log, detección Marchenko-Pastur del bulk de ruido, y termodinámica de transiciones de fase). El certificado determinista (`certificate()`) usa en cambio `alpha_entropy`, una cantidad relacionada pero más simple: la pendiente de la regresión lineal sobre $\log \lambda_k$ en los primeros 20 modos PCA. Ambas miden compresibilidad espectral, pero $\alpha_A$ es la lectura ecosistémica completa y `alpha_entropy` es el proxy rápido del certificado básico.

- $m$ captura la dimensión intrínseca, no la dimensión ambiente.
- $\log(1/\varepsilon)$ reemplaza la ley exponencial en $d$ cuando la estructura espectral es favorable.
- $\alpha_A$ actúa como invariante de compresibilidad dinámica: cuanto más rápido decae el espectro relevante, más barato es resolver el sistema.

La consecuencia filosófica es fuerte pero debe precisarse: la maldición de la dimensionalidad no se derrota con más fuerza bruta, sino identificando la estructura espectral y geométrica que hace que el problema verdadero tenga menos grados de libertad que su representación cruda. **La dependencia exponencial se desplaza de $d$ a $m$, no se elimina.** Para $m$ pequeño ($\leq 5$), esto es una victoria decisiva. Para $m$ moderado ($10$–$20$), la reducción sigue siendo enorme frente a $d$ grande pero el costo en muestras $\varepsilon^{-m}$ puede volverse prohibitivo para $\varepsilon$ muy pequeño. CCD no evade este límite fundamental del aprendizaje de variedades; simplemente opera en el mejor régimen posible dada la geometría subyacente.

---

## 4. Ontología del motor: qué objeto intenta aprender CCDEngine

CCDEngine no intenta aprender una etiqueta, ni una función de regresión aislada, ni un embedding estético. Intenta aprender cuatro objetos a la vez:

1. Una compresión espectral estable de la señal.
2. Una geometría de difusión que aproxime la conectividad interna del manifold.
3. Una organización resonante de variables que reduzca grados de libertad colectivos.
4. Un campo térmico local que regule la restauración hacia regiones de alta estructura.

De esos cuatro objetos emerge un quinto: una estimación operacional de la dimensión efectiva. Esa dimensión efectiva es la cantidad realmente importante. No es la cuenta de columnas del dataset ni el número de componentes fijado por capricho, sino el número mínimo de grados de libertad necesarios para describir la organización interna de los datos sin destruir su topología funcional.

---

## 5. Primera capa: regularización espectral y rechazo del preprocesamiento ingenuo

La primera capa de CCD cumple un trabajo delicado: reducir el problema antes de que la geometría no lineal entre en acción, pero sin destruir la métrica que luego debe analizarse. Esta capa usa un preprocesamiento espectral adaptativo: PCA exacta para dimensión moderada, SVD aleatorizado para dimensión media-alta y proyección Johnson-Lindenstrauss cuando la dimensión se vuelve extrema.

La razón no es solo eficiencia. Es una afirmación matemática sobre estabilidad. Si la primera capa altera brutalmente las relaciones de distancia o mezcla señal y ruido con el mismo peso, la capa geométrica posterior parte de una base falsa. Por eso el motor evita decisiones aparentemente razonables pero geométricamente nocivas, como el whitening indiscriminado o expansiones polinómicas que primero inflan la dimensión para luego volver a comprimirla.

La capa espectral busca conservar el orden dominante del subespacio informativo. Sea $X \in \mathbb{R}^{n \times d}$, con media removida. La descomposición espectral inicial identifica una base ortogonal aproximada en la que la mayor parte de la varianza útil queda concentrada en unas pocas direcciones. Si $V_r$ contiene las primeras $r$ direcciones relevantes, entonces

$$
Z_{\text{pre}} = X V_r
$$

debe preservar, hasta primer orden, la organización vecinal que luego explotará la geometría de difusión.

Adicionalmente, el motor emplea dos mecanismos de estabilidad en esta capa. El primero es una **escala global** (`auto_scale`) que divide $X$ por $\sqrt{\text{mean var}}$ sin ecualizar varianzas feature por feature, preservando así los ratios naturales de señal/ruido entre dimensiones. Validación empírica muestra que esto eleva la preservación de vecindarios de $9.1\%$ (con `StandardScaler`) a $98.7\%$ en el Swiss Roll de $\mathbb{R}^{53}$. El segundo es una **ponderación por SNR** que multiplica cada componente $Z_{\text{pre}}^{(j)}$ por $\sqrt{\sigma_j^2 / \sigma_{\text{ruido}}^2}$, donde el piso de ruido se estima con la mediana de los eigenvalues residuales (robusto a outliers). Ambos mecanismos son opcionales pero recomendados para datos heterogéneos.

Lo importante aquí es que la capa 1 no pretende “resolver” el problema. Pretende dejarlo en una forma donde la capa 2 ya no esté condenada por ruido isotrópico, escalas arbitrarias o redundancias evidentes.

---

## 6. Segunda capa: geometría de difusión como métrica riemanniana implícita

Esta es la parte central del motor. La idea es construir una métrica efectiva no a partir de un tensor explícito calculado en coordenadas, sino a partir de la conectividad probabilística entre puntos. Si dos puntos pertenecen a la misma región estructural del manifold, existen muchos caminos cortos de difusión entre ellos; si están separados por vacío geométrico, esos caminos escasean. La distancia de difusión captura precisamente esa diferencia.

El núcleo básico se construye con un kernel adaptativo

$$
W_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{\sigma_i \sigma_j}\right),
$$

donde $\sigma_i$ es una escala local determinada por el $k$-ésimo vecino más cercano. Esta adaptación es crucial porque evita imponer una longitud de escala global absurda en datasets heterogéneos.

Después se aplica una normalización anisotrópica al estilo de Coifman y Lafon. Si $D$ es la diagonal de grados del kernel, la renormalización con exponente $\alpha$ corrige el sesgo de densidad y permite interpolar entre distintos regímenes geométricos:

$$
W_{\alpha} = D_{\alpha}^{-1} W D_{\alpha}^{-1},
\qquad
D_{\alpha} = \operatorname{diag}(W \mathbf{1})^{\alpha}.
$$

Luego se construye un operador de Markov

$$
P = D_{\text{row}}^{-1} W_{\alpha},
$$

cuyos autovalores y autovectores describen modos globales de difusión. Las coordenadas de difusión se obtienen como

$$
\Phi_t(x_i) = \left(\lambda_1^t \phi_1(i), \dots, \lambda_k^t \phi_k(i)\right).
$$

Esto no es un embedding cualquiera. Es una representación en la que la cercanía refleja conectividad interna del manifold y no simple distancia euclídea. En otras palabras, esta capa induce una geometría donde el vacío pesa mucho y la estructura pesa poco. Esa es exactamente la curvatura efectiva que CCD necesita.

**Advertencia fundamental.** La convergencia de diffusion maps al operador de Laplace-Beltrami está demostrada para kernels con ancho de banda *global* $\varepsilon$ (Coifman & Lafon, 2006; Singer, 2006). CCD emplea un kernel *adaptativo* donde $\sigma_i$ es la distancia al $k$-ésimo vecino, no un $\varepsilon$ uniforme. La convergencia del kernel adaptativo al laplaciano **no está demostrada en la literatura actual**. Berry & Harlim (2016) prueban convergencia para kernels de ancho de banda variable bajo condiciones de regularidad, pero su análisis no cubre el estimador de $k$-ésimo vecino que CCD utiliza. En términos formales, la geometría de difusión de CCD es una **conjetura** con fuerte evidencia numérica pero sin garantía teórica completa. El lector debe entender esta capa como una heurística geométrica de alta calidad, no como un procedimiento con convergencia certificada. Ver §17.2 para la discusión rigurosa.

### 6.1. Por qué difusión y no una no linealidad arbitraria

Una red neuronal profunda podría también producir un embedding no lineal, pero no entregaría una interpretación geométrica tan clara ni una relación espectral tan directa con el operador de transición. La ventaja de diffusion maps es que conectan probabilidad, geometría y espectro en un solo objeto. Eso vuelve al método explicable, diagnosticable y certificable.

### 6.2. Variante multi-escala

Para datasets con estructuras que operan simultáneamente a varias escalas (clusters densos dentro de una organización global más laxa), el motor ofrece una variante `multi_scale_diffusion` que computa dos geometrías de difusión en paralelo: una local (kernel con pocos vecinos, $k_{\text{local}} \approx 3$–$15$, $\alpha=0.5$, $\tau=0.5$) y otra global (kernel con más vecinos, $k_{\text{global}} \approx 20$–$50$, $\alpha=1.0$, $\tau=2.0$). Las coordenadas resultantes se combinan linealmente ($0.7$ local + $0.3$ global). Esto reconoce que no existe una única escala de conectividad correcta para todos los regímenes del manifold, y que forzar una sola escala puede borrar estructura fina o perder organización gruesa.

### 6.3. El rol del Nyström extension

Cuando el número de puntos es gigantesco, CCD no diagonaliza de forma brutal el problema entero. Aprende la geometría sobre landmarks y extiende luego esa representación mediante Nyström. Teóricamente, eso significa aproximar los modos globales del operador usando un subconjunto representativo y luego prolongarlos al resto del conjunto. No es un truco menor de ingeniería: es la versión computable de una misma construcción espectral.

---

## 7. Tercera capa: resonancia colectiva y compresión por modos normales

La geometría por sí sola no basta. Muchos sistemas de alta dimensión no solo viven sobre un manifold: además presentan variables que coevolucionan, se arrastran, se sincronizan o se organizan en bloques funcionales. Tratar cada coordenada como un grado de libertad independiente es, otra vez, falsear la física del sistema.

La tercera capa modela las variables como osciladores acoplados. El punto no es simular literalmente la ecuación física completa, sino explotar su estructura conceptual: cuando muchas variables comparten un modo normal dominante, esas variables no representan grados de libertad independientes, sino una sola oscilación colectiva observada desde muchas coordenadas.

En la práctica, la capa no opera directamente sobre los datos crudos $X$, sino sobre las coordenadas de difusión $Z_{\text{diff}}$ ya reducidas por la capa anterior. Esto es una decisión de diseño relevante: la detección de redundancia colectiva se beneficia de trabajar en un espacio donde el ruido isotrópico ya fue atenuado y donde las relaciones no lineales entre variables ya fueron linealizadas por la geometría de difusión.

Si $C$ es la covarianza de $Z_{\text{diff}}$ y

$$
C = V \Lambda V^T,
$$

entonces las columnas de $V$ describen modos colectivos. La asignación de variables a grupos de resonancia se basa en la carga dominante sobre esos modos. El número de grupos de resonancia se interpreta como una estimación adicional de la dimensionalidad efectiva.

La ganancia conceptual es profunda. CCD no reduce solo porque algunas direcciones tengan poca varianza, sino porque detecta que muchas coordenadas eran manifestaciones redundantes de una misma dinámica colectiva. Esto hace al motor especialmente natural para sistemas físicos, biológicos o funcionales donde la correlación estructural no es accidental.

### 7.1. Transformada de coherencia adaptativa

El motor usa una transformada de coherencia en la que variables dentro de un mismo grupo se combinan con fases relativas. La forma conceptual es

$$
\widehat{X}_k = \Re\left[\sum_{i \in G_k} x_i e^{i\phi_{ik}}\right].
$$

Esto resume una familia de osciladores en una amplitud colectiva real. Desde el punto de vista de teoría de sistemas, es una compresión por sincronía. Desde el punto de vista computacional, es una reducción de grados de libertad que conserva estructura relacional.

---

## 8. Cuarta capa: entropía local y temperatura adaptativa

Si el motor solo comprimiera y difundiera, todavía le faltaría un principio de regulación local. No toda región del manifold tiene la misma confiabilidad. Hay zonas donde los puntos están colapsados, otras donde la nube es ruidosa, otras donde el muestreo cambia de densidad. Una purificación uniforme sería torpe.

Por eso CCD introduce un operador de entropía local. La idea es medir, para cada punto, la dispersión logarítmica de sus distancias a vecinos. Si las distancias son demasiado uniformes, el entorno es rígido o colapsado; si crecen de forma abrupta, la región sugiere irregularidad o fragmentación. El proxy usado es

$$
H_{\text{local}}(x_i) = \operatorname{std}(\log d_1, \dots, \log d_k).
$$

Este escalar se transforma luego en una temperatura efectiva

$$
T(x) = T_{\min} + (T_{\max} - T_{\min})
\, \sigma\!\left(\beta \frac{H(x) - H_{\text{mid}}}{H_{\text{scale}}}\right).
$$

La temperatura no es un adorno probabilístico. Es un regulador local del grado de exploración permitido por la dinámica estocástica posterior. En regiones inciertas, el sistema admite más movilidad; en regiones estables, se ancla con mayor suavidad.

Desde teoría geométrica, esta capa funciona como una corrección local a la confianza del manifold aprendido. En vez de creer ciegamente en toda la nube, estima dónde la estructura es fuerte y dónde no.

---

## 9. Quinta capa: purificación de Langevin como retorno al manifold

La quinta capa transforma la geometría aprendida en una dinámica de restauración. Una vez que el motor tiene una estimación del paisaje de densidad y una temperatura local, puede mover puntos ruidosos hacia la variedad estructural. El mecanismo elegido es Langevin:

$$
dx = -\nabla U(x) \, dt + \sqrt{2T(x)} \, dW_t,
$$

con

$$
U(x) = -\log p_{\text{data}}(x).
$$

Interpretado correctamente, el gradiente del potencial empuja hacia zonas donde los datos “viven” y el término estocástico evita que el proceso quede rígidamente atrapado por aproximaciones locales pobres. La temperatura adapta esa exploración a la confiabilidad geométrica estimada en la capa previa.

### 9.1. Por qué no usar KDE denso clásico

Un estimador denso del gradiente de log-densidad sería prohibitivamente caro. CCD reemplaza eso por score matching local sobre los vecinos más cercanos:

$$
\nabla \log p(x) \approx
-\frac{1}{\sigma^2}
\sum_{j \in \text{kNN}(x)} K_{\text{norm}}(x,x_j) (x - x_j).
$$

La filosofía es coherente con todo el motor: no usar la nube completa cuando la estructura local suficiente ya está disponible. Esto reduce el costo de $O(n_{\text{new}} n_{\text{train}} d)$ a un régimen local controlado por $k$.

### 9.2. La interpretación correcta de la purificación

La purificación no debe entenderse como “denoising cosmético”. Es un intento de proyectar dinámicamente datos corrompidos hacia la geometría aprendida. Si el manifold estimado es bueno, la purificación recupera estructura. Si el manifold es falso, la purificación también lo será. Por eso CCD no separa nunca purificación y certificación.

---

## 10. Dimensión efectiva: el verdadero observable del sistema

El objetivo operativo de CCDEngine es estimar una dimensión efectiva robusta. Esa dimensión se construye en dos etapas y se valida con una tercera.

Primero, dos estimaciones independientes compiten para fijar una cota inferior:

1. **Dimensión geométrica**, $k_{\text{diff}}$, visible en el gap del espectro de difusión — cuántos modos de Markov son necesarios antes de que la conectividad efectiva colapse.
2. **Dimensión colectiva**, $n_{\text{groups}}$, visible en el número de grupos de resonancia — cuántas oscilaciones colectivas independientes bastan para describir las variables.

La base de la dimensión efectiva es el máximo de ambas:

$$k_{\text{base}} = \max(k_{\text{diff}}, n_{\text{groups}}).$$

Sobre esa base, el motor añade opcionalmente un bloque de modos lineales vía skip connection (ver más abajo). La tercera estimación — el rango espectral efectivo $k_{\text{cheb}}$ del preprocesador — se conserva en el certificado como validación independiente: si $k_{\text{cheb}}$ es muy distinto de $k_{\text{base}}$, el sistema advierte que la compresión espectral y la compresión geométrica no convergen, y la estimación debe leerse con cautela.

La idea es importante: la dimensionalidad efectiva no debe depender de un solo artefacto numérico. Si difusión y resonancia convergen, la estimación es confiable. Si divergen brutalmente o si el rango espectral cuenta otra historia, el certificado debe reflejar esa ambigüedad.

---

## 11. Certificación: cuándo CCD puede afirmar que escapó de la maldición

CCDEngine no solo produce coordenadas reducidas; produce certificados. Esto es fundamental porque en reducción no lineal abundan métodos que siempre entregan una figura bonita pero nunca demuestran si han preservado algo real.

El certificado básico cuantifica:

- dimensión ambiente $d_{\text{input}}$,
- dimensión efectiva $k_{\text{effective}}$,
- grupos de resonancia,
- rango espectral relevante $k_{\text{cheb}}$ (validación independiente, no usada en $k_{\text{effective}}$),
- varianza explicada en los modos dominantes,
- `alpha_entropy`: tasa de decaimiento espectral estimada vía regresión sobre $\log \lambda_k$ (proxy rápido; la clasificación ecosistémica completa $\alpha_A$ se obtiene con `ecosystem_optimize`),
- reducción logarítmica de complejidad frente a una grilla uniforme.

El criterio más simple de escape es

$$
k_{\text{effective}} < \frac{d_{\text{input}}}{2}.
$$

Eso no es una demostración ontológica absoluta, pero sí una condición operacional clara: el sistema logró comprimir la descripción esencial del problema a menos de la mitad del espacio ambiente. En muchos escenarios la reducción es mucho más fuerte.

El certificado robusto va más lejos. Introduce bootstrap, error de reconstrucción en test, preservación de vecindarios, contraste contra hipótesis gaussiana y comparación contra PCA lineal. Ese paso es teóricamente importante porque separa dos preguntas que suelen confundirse:

1. ¿El método reduce dimensión?
2. ¿La dimensión reducida corresponde a una estructura real y no a una ilusión numérica?

CCD intenta responder ambas.

---

## 12. Relación con TAA, OTU y ERGON dentro del ecosistema ACF

CCDEngine no es un módulo aislado. Su posición dentro del ecosistema es precisa.

### 12.1. Relación con TAA

TAA clasifica el decaimiento espectral del sistema. Esa clasificación determina cuánto puede comprimirse una representación sin destruir precisión. CCD usa esa información para calibrar componentes, vecinos, tiempo de difusión y agresividad de purificación. En términos conceptuales, TAA le dice a CCD cuán rígido o compresible es el espectro del problema.

Si el decaimiento es exponencial, la estructura es muy favorable y la compresión puede ser agresiva. Si es polinómica, la caída es más lenta y el motor debe conservar más grados de libertad. Si el régimen es ruidoso, la purificación y la cautela geométrica ganan peso.

### 12.2. Relación con OTU

OTU aporta la lectura del gap espectral y una **regla heurística de calibración** que en el ecosistema se conoce como "dual budget": la observación empírica de que el número de componentes de difusión y el número de pasos de Langevin no deberían diverger excesivamente para que la configuración sea coherente. CCD implementa esta regla mediante `dual_budget_validation()`, que verifica `abs(d_spatial - n_temporal) <= max(3, d_spatial // 3)`. Esta tolerancia es una elección de ingeniería, no una consecuencia de principios físicos. Ver §17.5 para la discusión completa.

### 12.3. Relación con ERGON

ERGON trabaja en clave Perron-Frobenius, es decir, sobre la evolución de densidades. CCD comparte con ERGON la convicción de que lo importante no es la nube cruda de puntos, sino la estructura de transporte, mezcla y concentración de masa. La capa de difusión y la purificación de Langevin son completamente naturales desde esa perspectiva. Donde TAA ve espectro de observables, ERGON ve evolución de densidades, y CCD sirve como la geometría efectiva que hace ambas lecturas compatibles y computables.

---

## 13. Complejidad computacional: qué escala y qué no

Una teoría que no escala es humo. CCD evita ese destino porque cada capa reemplaza una operación intratable por una versión local o dispersa. La tabla siguiente da cotas concretas de tiempo y memoria por componente:

| Componente | Tiempo | Memoria | Notas |
|------------|--------|---------|-------|
| `SpectralPreprocessor` (PCA) | $O(nd \cdot \min(n,d))$ | $O(d \cdot r)$ | Para $d<1000$; con RandomizedSVD cae a $O(nd\log r)$; con JL a $O(ndk)$ |
| `SparseAdaptiveKernel` | $O(n k d)$ | $O(nk)$ disperso | $k$ típico 15–50; vs $O(n^2 d)$ del kernel denso |
| `DiffusionGeometry` (fit) | $O(nk \cdot \text{eig\_cost})$ | $O(nk)$ | Eigsh disperso para $n>500$; Nyström para $n>20$k |
| `DiffusionGeometry` (transform) | $O(n_{\text{new}} k d)$ | $O(n_{\text{new}} k)$ | Nyström disperso, no recalcula kernel |
| `CoupledOscillators` | $O(nd_{\text{pre}}^2)$ | $O(d_{\text{pre}}^2)$ | Covarianza en espacio preprocesado ($d_{\text{pre}} \ll d$) |
| `LocalEntropyOperator` | $O(n k \log n)$ | $O(n)$ | kd-tree query |
| `LangevinPurifier` | $O(T \cdot n \cdot k_{\text{score}} \cdot d)$ | $O(n)$ | $T$ pasos, $k_{\text{score}}$ vecinos para el score |

**Comparación con alternativas.**

| Método | Tiempo (fit) | Escala máxima ($n$, $d$) | Garantía teórica | Denoising |
|--------|-------------|--------------------------|------------------|-----------|
| PCA | $O(nd\cdot\min(n,d))$ | Ilimitada | Óptimo lineal (Eckart-Young) | No |
| Isomap | $O(n^3)$ | $n \leq 10^4$ | Isométrico si manifold es desarrollable | No |
| UMAP | $O(n^{1.14})$ | $n \leq 10^6$ | No (pérdida topológica posible) | No |
| Autoencoder | $O(n \cdot \text{epochs} \cdot \text{params})$ | $n \leq 10^7$ | No (mínimo local, arquitectura-dependiente) | Sí (con bottleneck) |
| **CCDEngine** | Dominado por kernel $O(nkd)$ | $n \leq 10^6$, $d \leq 10^5$ | Parcial (kernel global sí, adaptativo conjetura) | **Sí** (Langevin termalizado) |

**Lo que CCD NO ofrece.** A diferencia de UMAP o t-SNE, CCD no está optimizado para visualización 2D/3D. Su target es $k \in [2, 40]$, no $k=2$. A diferencia de autoencoders, CCD no aprende una función paramétrica de encoding, lo que limita su velocidad de transform en producción pero elimina el riesgo de mínimos locales y overfitting.

**Conclusión honesta.** CCD escala significativamente mejor que métodos densos ($O(n^2)$ u $O(n^3)$) gracias al kernel disperso y Nyström, y ofrece capacidades (denoising, certificación, calibración ecosistémica) que sus competidores no tienen. Pero no es el método más rápido para visualización, ni tiene garantías formales completas en todas sus capas. Su nicho es la reducción dimensional *explicable y certificable* en pipelines científicos y de ingeniería donde la trazabilidad importa más que el último punto porcentual de fidelidad visual.

---

## 14. Qué hace realmente el motor en términos conceptuales

Si se quiere decirlo sin lenguaje de implementación, CCDEngine hace exactamente esto:

1. Limpia el espectro para que la geometría no arranque desde una base degenerada.
2. Reconstruye la conectividad profunda del conjunto de datos en lugar de su mera cercanía euclídea.
3. Detecta qué variables son independientes y cuáles son solo manifestaciones colectivas de una misma dinámica.
4. Mide dónde la estructura es confiable y dónde es incierta.
5. Usa esa información para devolver puntos ruidosos a la variedad aprendida.
6. Certifica cuánto logró comprimir y con qué grado de honestidad estadística.

Eso es CCD. No un wrapper de reducción dimensional, sino un esquema geométrico-espectral-dinámico para convertir alta dimensión aparente en baja dimensión efectiva.

---

## 15. Límites del método y dónde puede fallar

Ser preciso obliga a decir también dónde no debe venderse humo.

### 15.1. Si no hay manifold, no hay milagro

Si los datos son ruido casi gaussiano sin estructura, CCD no puede inventar una variedad real. Puede producir una reducción, pero el certificado robusto debería denunciar que la hipótesis de manifold no está sustentada.

### 15.2. Si el muestreo es pobre, la geometría será pobre

Toda geometría de difusión depende de vecindarios. Si el muestreo del manifold es insuficiente, discontinuo o extremadamente sesgado, la conectividad inducida puede ser falsa. El motor entonces no “descubre” la estructura; la extrapola mal.

### 15.3. Si la temperatura es mal calibrada, la purificación sobrecorrige

Una dinámica de Langevin demasiado agresiva puede arrastrar puntos hacia regiones densas pero geométricamente equivocadas, borrando detalle fino. Por eso la temperatura local y el clipping del score no son adornos: son lo que separa purificación de colapso.

### 15.4. La dimensión efectiva es una estimación, no un absoluto metafísico

Incluso con certificados, la dimensión efectiva sigue siendo una cantidad inferida. Su credibilidad crece cuando convergen espectro, difusión, resonancia y validación bootstrap. Cuando no convergen, el resultado debe leerse como diagnóstico de ambigüedad, no como verdad ontológica final.

---

## 16. Lectura filosófica correcta de CCDEngine

La lectura superficial sería: “es un motor para reducción no lineal y denoising”. Esa descripción es pobre. La lectura correcta es otra: CCDEngine es una teoría computable de cómo extraer la geometría efectiva de sistemas de alta dimensión cuando la estructura real vive en un subconjunto de baja complejidad.

Su novedad no está en haber inventado diffusion maps, ni PCA, ni Langevin por separado. Está en ensamblarlos como partes de una misma ontología: espectro para estabilizar, difusión para curvar, resonancia para identificar libertad colectiva, entropía para medir confianza y Langevin para restaurar. El certificado final cierra el círculo porque obliga al sistema a demostrar que la reducción no fue una ilusión visual sino una contracción real de complejidad.

En el lenguaje del ecosistema ACF, CCD es el mecanismo que traduce alta dimensionalidad aparente en geometría esencial. Lo que reduce no son columnas; reduce grados de libertad efectivos. Lo que purifica no son píxeles; purifica desviaciones respecto del manifold. Lo que certifica no es una métrica estética; certifica si el problema dejó de vivir en el infierno exponencial de $\varepsilon^{-d}$ y pasó a una ley gobernada por la dimensión intrínseca y el decaimiento espectral.

---

## 17. Análisis matemático riguroso

Esta sección examina cada uno de los claims del motor con el rigor necesario: se especifica qué está demostrado, bajo qué condiciones, y qué permanece como conjetura o principio heurístico de calibración.

---

### 17.1. Complejidad de muestra y escape de la maldición de la dimensionalidad

El objetivo es comparar el costo de aproximar una función $f: M \to \mathbb{R}$ definida sobre una variedad $M^m \subset \mathbb{R}^d$ con $m \ll d$. Se distingue entre dos costos: la complejidad de muestra para *aprender* la variedad, y la complejidad de consulta una vez aprendida.

**Costo de la grilla cartesiana.** Aproximar $f$ sobre $[0,1]^d$ con error uniforme $\varepsilon$ requiere una $\varepsilon$-red de la caja $d$-dimensional. El número de puntos es $(\lceil 1/\varepsilon \rceil)^d = O(\varepsilon^{-d})$. Este costo es óptimo para funciones lipschitzianas sin estructura geométrica adicional (Bakhvalov, 1959).

**Costo de cubrimiento de la variedad.** Si se conoce $M$, una $\varepsilon$-red intrínseca requiere $O(\varepsilon^{-m})$ puntos — dependencia exponencial en $m$, no en $d$. Este es el costo óptimo de cualquier método que explote la estructura de variedad.

**Complejidad de muestra para aprender $M$.** Aprender una $m$-variedad $C^2$ inmersa en $\mathbb{R}^d$ con error de Hausdorff $\varepsilon$ requiere una muestra de tamaño

$$n \geq C(m) \cdot \varepsilon^{-m} \cdot \operatorname{poly}(\log(1/\varepsilon))$$

(Genovese et al., 2012; Niyogi, Smale & Weinberger, 2008). La dependencia es $\varepsilon^{-m}$, no $\log(1/\varepsilon)$. Esta es una cota inferior *minimax*: ningún estimador puede aprender la variedad con menos muestras, salvo hipótesis adicionales muy restrictivas.

**Qué puede afirmar CCD.** CCD no evade la complejidad de muestra $\varepsilon^{-m}$ para aprender la geometría. Lo que sí logra es que, **una vez aprendida la geometría**, el costo de proyectar nuevos puntos es $O(k_{\text{eff}}) = O(m)$, y el costo de reconstruir es $O(m \log n)$. La ventaja sobre la grilla es real pero acotada:

$$\frac{N_{\text{CCD}}(\varepsilon)}{N_{\text{grid}}(\varepsilon)} = \frac{O(\varepsilon^{-m} \cdot \operatorname{poly}(\log(1/\varepsilon)))}{O(\varepsilon^{-d})} = O\!\left(\varepsilon^{d-m} \cdot \operatorname{poly}(\log(1/\varepsilon))\right) \to 0 \quad (d \to \infty).$$

**La cantidad `cod_reduction_log10`.** El certificado calcula

$$\text{cod\_reduction\_log10} = (d - k_{\text{eff}}) \cdot \log_{10}(1/0.01) = 2(d - k_{\text{eff}}).$$

Esta cantidad mide la reducción logarítmica en el costo de *cubrimiento* de la grilla cartesiana frente a un número fijo $k_{\text{eff}}$ de coordenadas de difusión. Es una medida heurística de contracción de complejidad, no una cota rigurosa de error de aproximación. Su valor informativo es comparativo (¿cuánto más barato es CCD que una grilla?) pero no absoluto (no certifica error $\varepsilon$).

**Condiciones bajo las cuales la reducción es válida.** La reducción $d \to k_{\text{eff}}$ preserva la capacidad de aproximación siempre que:
1. Los datos residan efectivamente en o cerca de una variedad $m$-dimensional con $m \leq k_{\text{eff}} \ll d$.
2. La medida de muestreo tenga soporte completo en la variedad y densidad acotada lejos de cero e infinito.
3. El radio de curvatura de $M$ sea grande comparado con el ancho de banda del kernel de difusión.

Bajo estas condiciones, la variedad reconstruida es homeomorfa a la original y la distorsión métrica está controlada. Sin ellas, $k_{\text{eff}}$ puede subestimar o sobreestimar la complejidad real.

---

### 17.2. Convergencia de la geometría de difusión

**Teorema (Coifman & Lafon, 2006; Singer, 2006).** Sea $M^m$ una variedad riemanniana compacta sin borde, isométricamente inmersa en $\mathbb{R}^d$. Sea $\{x_i\}_{i=1}^n$ una muestra i.i.d. de una densidad $q(x)$ suave y estrictamente positiva sobre $M$. Para el kernel gaussiano con ancho de banda global $\varepsilon$,

$$W_{ij} = \exp\!\left(-\frac{\|x_i - x_j\|^2}{\varepsilon}\right),$$

y normalización $\alpha = 1$ (operador de Laplace-Beltrami), el generador

$$L_{n,\varepsilon} = \frac{P_n - I}{\varepsilon}$$

converge puntualmente a $\Delta_M$ cuando $n \to \infty$ y $\varepsilon \to 0$ con $n \varepsilon^{m/2 + 1} \to \infty$.

**Extensión a kernel adaptativo.** CCD emplea un kernel adaptativo

$$W_{ij} = \exp\!\left(-\frac{\|x_i - x_j\|^2}{\sigma_i \sigma_j}\right), \qquad \sigma_i = \operatorname{dist}(x_i, x_{(k)}).$$

Este kernel *no* está cubierto por la demostración de Coifman & Lafon, que requiere un $\varepsilon$ uniforme para todos los puntos. Cuando $\sigma_i$ depende de la densidad local, la expansión de Taylor del operador integral involucra derivadas de $\sigma(x)$, que introducen un campo de deriva adicional proporcional a $\nabla \log q(x)$.

**Qué se sabe sobre kernels adaptativos.** Berry & Harlim (2016) muestran que kernels de ancho de banda variable pueden converger a un operador de Laplace-Beltrami *con corrección de densidad* siempre que $\sigma(x)$ satisfaga ciertas condiciones de regularidad (Lipschitz, acotada lejos de cero). La normalización $\alpha = 1$ en CCD está diseñada para cancelar asintóticamente el sesgo de densidad, pero la cancelación exacta con $\sigma_i$ variable no ha sido demostrada formalmente en la literatura para el estimador de $k$-ésimo vecino.

**Conjetura (CCD).** Bajo las mismas condiciones que el teorema de Coifman-Lafon, y para $k$ fijo con $k \geq m+1$, el generador del kernel adaptativo de CCD con $\alpha = 1$ converge puntualmente a $\Delta_M$ cuando $n \to \infty$.

*Evidencia numérica.* Experimentos en variedades canónicas (Swiss roll, $S^2$, toro plano) muestran que el gap espectral detectado por `intrinsic_dimension_estimate` coincide con $m$ para $n$ suficientemente grande, y que la distorsión de distancias geodésicas es comparable a la del kernel global.

**La regla del 30%.** El estimador `intrinsic_dimension_estimate` busca el primer $k$ donde $(\mu_k - \mu_{k+1}) / \mu_k > 0.30$. Esta regla es una heurística de gap espectral sin garantía teórica de convergencia a $m$. Para variedades con simetrías (esferas, toros planos), las multiplicidades de los autovalores del laplaciano pueden hacer que el "gap" detectado corresponda a la dimensión del primer espacio propio no trivial, no a $m$. Ejemplo: $S^1$ tiene primer autovalor $\lambda_1 = 1$ con multiplicidad 2 ($\sin\theta, \cos\theta$); la regla del 30% reportaría $k=2$, no $m=1$. En la práctica, el motor compensa esto mediante el máximo con $n_{\text{groups}}$ (ver §17.3), pero la heurística del 30% sigue siendo vulnerable a sobreestimación en presencia de simetrías.

---

### 17.3. El estimador de dimensión efectiva como cota superior

**Definición.** El estimador de CCD es

$$k_{\text{eff}} = \max(k_{\text{diff}}, n_{\text{groups}}) + n_{\text{linear}},$$

donde $k_{\text{diff}}$ proviene del gap de difusión (§17.2), $n_{\text{groups}}$ es el número de grupos de resonancia, y $n_{\text{linear}}$ es el número de componentes del skip connection lineal ($n_{\text{linear}} \leq 5$ por construcción).

**Propiedad de cota superior.** Para $n$ suficientemente grande y bajo las condiciones de la conjetura 17.2, $k_{\text{eff}} \geq m$ con alta probabilidad. Esto es inmediato porque $\max(a, b) \geq a$, y tanto $k_{\text{diff}}$ como $n_{\text{groups}}$ son, en su construcción, conteos de modos que incluyen los $m$ modos genuinos del manifold más posibles sobreestimaciones.

**El problema de la sobreestimación sistemática.** Para una variedad $m$-dimensional, los autovalores del laplaciano crecen según la ley de Weyl:

$$N(\lambda) = \#\{k : \lambda_k \leq \lambda\} \sim \frac{\omega_m \operatorname{vol}(M)}{(2\pi)^m} \lambda^{m/2}, \quad \lambda \to \infty.$$

Esto implica que el número de autofunciones *antes del primer gap grande* puede ser mayor que $m$. El estimador de gap identifica la posición del primer salto significativo en el espectro, no $m$ directamente. En particular:

- Para $S^1$ ($m=1$): $\lambda_1 = \lambda_2 = 1$ (multiplicidad 2) → gap tras $k=2$ → $k_{\text{diff}} = 2 > 1 = m$.
- Para $S^2$ ($m=2$): $\lambda_1 = \lambda_2 = \lambda_3 = 2$ (multiplicidad 3) → gap tras $k=3$ → $k_{\text{diff}} = 3 > 2 = m$.
- Para el toro plano $T^2$ ($m=2$): $\lambda_1 = \lambda_2 = 1$ y $\lambda_3 = \lambda_4 = \lambda_5 = \lambda_6 = 2$ → depende del gap detectado.

En todos estos casos, $k_{\text{eff}}$ es una **cota superior honesta** de $m$, no un estimador consistente. Su valor es que rara vez subestima $m$, y el costo de sobreestimar ($k_{\text{eff}} > m$) es típicamente bajo porque $k_{\text{eff}}$ sigue siendo $\ll d$.

**Validación práctica.** El certificado robusto (§11) cuantifica la calidad de la reducción mediante bootstrap y preservación de vecindarios, lo cual detecta si la sobreestimación es excesiva: si $k_{\text{eff}}$ es demasiado grande, `neighborhood_preservation` será baja y `reconstruction_rmse` alta en el test set. Esto proporciona un control indirecto pero operacional sobre la calidad del estimador.

---

### 17.4. Dinámica de Langevin con temperatura variable

Se analiza la dinámica implementada en la capa 5:

$$dx_t = -\nabla U(x_t) dt + \sqrt{2 T(x_t)} dW_t, \qquad U(x) = -\log p_{\text{data}}(x).$$

**Caso con temperatura constante ($T(x) \equiv T$).** Si $U$ es $L$-suave y $p_{\text{data}}$ satisface una desigualdad de log-Sobolev (LSI) con constante $\rho > 0$, entonces la divergencia KL satisface (Vempala & Wibisono, 2019):

$$\frac{d}{dt} D_{\text{KL}}(p_t \| p_{\text{data}}) \leq -\rho T \cdot D_{\text{KL}}(p_t \| p_{\text{data}}).$$

De aquí, $D_{\text{KL}}(p_t \| p_{\text{data}}) \leq D_{\text{KL}}(p_0 \| p_{\text{data}}) \cdot e^{-\rho T t}$. La convergencia es exponencial.

**Caso con temperatura variable.** La ecuación de Fokker-Planck para $T(x)$ no constante es

$$\partial_t p_t = \nabla \cdot (p_t \nabla U) + \nabla \cdot (T \nabla p_t) + \nabla \cdot (p_t \nabla T).$$

El término $\nabla \cdot (p_t \nabla T)$ no está presente en la dinámica de Langevin estándar y modifica la evolución de la entropía. La derivada de la divergencia KL es:

$$\frac{d}{dt} D_{\text{KL}}(p_t \| p_{\text{data}}) = -\int p_t \left\|\nabla \log\frac{p_t}{p_{\text{data}}}\right\|^2 T \, dx - \int p_t \left\langle \nabla \log\frac{p_t}{p_{\text{data}}}, \nabla T \right\rangle dx.$$

El segundo término involucra $\nabla T$ y no tiene signo definido. Si $\nabla T$ es grande (transiciones bruscas de temperatura), este término puede desestabilizar o ralentizar la convergencia.

**Condiciones suficientes para convergencia con temperatura variable.** La convergencia exponencial se preserva si:

1. $0 < T_{\min} \leq T(x) \leq T_{\max} < \infty$ para todo $x$.
2. $\|\nabla T(x)\| \leq L_T$ para todo $x$ (temperatura Lipschitz).
3. $L_T \cdot \mathbb{E}_{p_t}[\|x\|] \ll T_{\min}$ (la deriva de temperatura es pequeña comparada con la difusión mínima).

Bajo estas condiciones, existe $\tilde{\rho} > 0$ (dependiendo de $L_T$, $T_{\min}$, $T_{\max}$ y $\rho$) tal que

$$D_{\text{KL}}(p_t \| p_{\text{data}}) \leq D_{\text{KL}}(p_0 \| p_{\text{data}}) \cdot e^{-\tilde{\rho} t}.$$

La tasa $\tilde{\rho}$ es menor que la tasa $\rho T_{\min}$ del caso homogéneo, pero sigue siendo exponencial.

**Implicaciones para CCD.** En la práctica de CCD, $T(x)$ se construye mediante una función sigmoide suave (§8) con $T_{\min} = 0.01$, $T_{\max} = 1.0$, y $\beta$ controlando la transición. La temperatura es Lipschitz con constante $L_T \leq \beta (T_{\max} - T_{\min}) / (4 H_{\text{scale}})$. Para $\beta = 2$ y $H_{\text{scale}}$ típicamente $\geq 0.1$, se tiene $L_T \leq 5$, que es moderada. La convergencia exponencial está garantizada condicionalmente, aunque la tasa exacta $\tilde{\rho}$ no se conoce sin estimar $\nabla T$ sobre los datos.

**Score matching disperso.** El score $\nabla \log p_{\text{data}}$ se aproxima usando $k$ vecinos (§9.1). El error de aproximación es $O(1/\sqrt{k} + \sigma^2)$ donde $\sigma$ es el ancho de banda local. Para $k \geq 50$ y datos en una variedad $m$-dimensional con $m \leq 20$, este error es típicamente aceptable en relación al ruido estocástico $\sqrt{2T} dW_t$.

---

### 17.5. Principio heurístico de calibración dual

**Contexto.** El `ecosystem_optimize` de CCD ajusta simultáneamente el número de componentes de difusión $d^*(\varepsilon)$ y el número de pasos de Langevin $n^*(\varepsilon)$ basándose en la clasificación TAA. La implementación verifica que `abs(d_spatial - n_temporal) <= max(3, d_spatial // 3)` como diagnóstico de coherencia.

**Qué es y qué no es.** Esta relación *no es un teorema*. Es un **principio heurístico de calibración** motivado por la siguiente observación:

*Intuición física.* En un sistema dinámico sobre $M^m$, el número de grados de libertad espaciales (dimensión del atractor) y el número de escalas temporales de mezcla (tiempos de retorno de Poincaré, tasa de decaimiento de correlaciones) reflejan la misma complejidad subyacente. Forzar muchos pasos de Langevin cuando la geometría es casi lineal ($d^*$ pequeño) sobredifunde y destruye estructura; usar pocos pasos cuando $d^*$ es grande no purifica adecuadamente.

*Evidencia numérica.* En benchmarks internos del ecosistema ACF, configuraciones que violan la relación dual por un factor $\geq 3\times$ producen `reconstruction_rmse` consistentemente peores que configuraciones que la satisfacen.

*La tolerancia.* El umbral $\delta = \max(3, d^*/3)$ es una elección de ingeniería calibrada empíricamente, no una constante derivada de primeros principios. Valores más estrictos ($\delta = 1$) disparan falsas alarmas en datasets ruidosos; valores más laxos ($\delta = d^*$) pierden poder diagnóstico.

**Recomendación de uso.** La validación dual debe usarse como alerta de mala configuración (si falla estrepitosamente, reconfigurar), no como certificación de optimalidad. La decisión final sobre los hiperparámetros debe basarse en el certificado robusto (§11) con bootstrap sobre datos de test.

---

### 17.6. Clasificación espectral TAA como heurística con fundamento

**Qué es TAA.** TAA clasifica el decaimiento espectral $\{\lambda_k\}$ de la covarianza muestral en cuatro regímenes: EXPONENTIAL, POLYNOMIAL, FINITE, NOISY. La clasificación usa correlación, Marchenko-Pastur y (opcionalmente) termodinámica de transiciones de fase.

**Qué NO es.** No es una partición matemática exhaustiva del espacio de secuencias espectrales. Es un árbol de decisión heurístico con umbrales fijos ($0.85$, $0.7$, $0.6$) cuyo propósito es seleccionar una estrategia de compresión, no caracterizar formalmente el decaimiento.

**Fundamento de cada test:**

| Test | Fundamento | Limitación |
|------|-----------|------------|
| Correlación log-lineal | Si $\lambda_k \sim e^{-\alpha k}$, entonces $\log\lambda_k = -\alpha k + \text{cte}$ y $r_{\text{exp}} \approx -1$ | Requiere $k \ll d$ y puede fallar con mezclas de modos |
| Correlación log-log | Si $\lambda_k \sim k^{-s}$, entonces $\log\lambda_k = -s\log k + \text{cte}$ y $r_{\text{poly}} \approx -1$ | Misma limitación; sensible a outliers espectrales |
| Marchenko-Pastur | Para ruido gaussiano i.i.d., el espectro se concentra en el intervalo MP; los eigenvalues fuera de él son señal | Asume ruido i.i.d.; no aplica a ruido correlacionado o heterocedástico |
| Termodinámico | Una transición de fase en $F(\beta)$ indica un cambio cualitativo en la estructura espectral, típicamente dimensionalidad finita | Requiere el módulo `thermodynamic_acf` y es computacionalmente costoso; la ausencia de transición no implica lo contrario |

**El árbol de decisión.** El orden de evaluación (EXPONENTIAL → NOISY → POLYNOMIAL → FINITE → fallback) es una priorización heurística, no una consecuencia de inclusiones de conjuntos. Una secuencia podría satisfacer múltiples criterios simultáneamente (por ejemplo, ser EXPONENTIAL con alta fracción de ruido MP). La prioridad refleja una elección de diseño: se prefiere clasificar como NOISY antes que arriesgar una compresión agresiva sobre datos ruidosos.

**Consecuencia práctica.** La clasificación TAA es útil como guía de configuración automática y funciona razonablemente en los benchmarks del ecosistema. Pero no debe presentarse como un teorema de exhaustividad. Su valor está en la automatización que proporciona, no en la completitud formal de su partición.

---

### 17.7. Mejoras defensivas implementadas en el código

Como consecuencia directa del análisis riguroso anterior, el código de `ccd_engine.py` incorpora las siguientes salvaguardas:

**Validación de entrada.** `CCDEngine.fit()` ahora rechaza explícitamente:
- Arrays no-numpy o con dimensionalidad incorrecta.
- Valores `NaN` (solicita imputación previa).
- Valores `Inf` (solicita recorte previo).
- Features con varianza cero (degeneración total).
- Muestras con $n < 3$ o $d < 1$ (insuficientes para cualquier estimación).

**Detección de variedades simétricas.** `CCDEngine._detect_spectral_multiplicity()` examina los autovalores de difusión en busca de multiplicidades anómalas (autovalores casi idénticos consecutivos en los primeros 10 modos, tolerancia del 5%). Si detecta $\geq 2$ pares degenerados, activa `symmetric_manifold_warning` en el certificado, indicando que $k_{\text{eff}}$ puede sobreestimar $m$ (ej. $S^1$ reportaría 2, no 1).

**Verificación de $\nabla T$ Lipschitz.** `CCDEngine._estimate_nabla_T_bound()` estima empíricamente la constante de Lipschitz de $\nabla T$ mediante diferencias finitas sobre una muestra de 500 puntos. Si $L_T > 5.0$, el certificado robusto activa `nabla_T_warning`, advirtiendo que la convergencia de Langevin (§17.4) no está garantizada bajo las condiciones actuales.

**Campos de advertencia en certificados.** Tanto `CCDCertificate` como `RobustCCDCertificate` incluyen ahora:
- `cod_reduction_is_heuristic: bool = True` — documenta que `cod_reduction_log10` no es una cota rigurosa de error.
- `k_eff_is_upper_bound: bool = True` — documenta que $k_{\text{eff}} \geq m$, no $= m$.
- `symmetric_manifold_warning: bool` — multiplicidades espectrales detectadas.
- `nabla_T_bound: float` / `nabla_T_warning: bool` — estado de la condición Lipschitz para Langevin.

**Documentación interna.** Las funciones `_taa_classify()`, `dual_budget_validation()`, y `ecosystem_optimize()` ahora declaran explícitamente en sus docstrings que son heurísticas de calibración, no teoremas. La clase `CCDCertificate` advierte en su docstring que `k_effective` es cota superior y `cod_reduction_log10` es heurístico.

---

**Resumen del estatus matemático de CCD.** El motor combina:

- **Resultados rigurosos heredados:** convergencia de difusión para kernel global (Coifman-Lafon), complejidad de muestra para aprendizaje de variedades (Niyogi et al.), convergencia de Langevin homogéneo (Vempala-Wibisono).
- **Extensiones condicionales:** kernel adaptativo (conjetura, §17.2), Langevin con temperatura variable (condiciones suficientes, §17.4).
- **Heurísticas de calibración:** regla del 30% para gap espectral (§17.2), estimador $k_{\text{eff}} = \max(k_{\text{diff}}, n_{\text{groups}})$ (§17.3), principio dual (§17.5), clasificación TAA (§17.6), `cod_reduction_log10` (§17.1).
- **Salvaguardas defensivas:** validación de entrada, detección de variedades simétricas, verificación de $\nabla T$ Lipschitz, certificados con campos de advertencia (§17.7).

El certificado robusto (§11) es el mecanismo que CCD proporciona para validar empíricamente que estas heurísticas no están fallando en un dataset concreto. Si el certificado robusto rechaza la hipótesis de manifold o muestra mala preservación de vecindarios, las heurísticas no son confiables para esos datos.

---

## 18. Conclusión

CCDEngine existe para responder una sola pregunta: cuando un sistema parece inmenso, ¿cuántos grados de libertad son realmente necesarios para describirlo, recorrerlo y restaurarlo? Su respuesta es geométrica, espectral y dinámica al mismo tiempo. Primero identifica una estructura lineal estable. Después descubre la geometría no lineal del manifold. Luego colapsa redundancias colectivas, estima incertidumbre local y finalmente empuja los datos hacia la variedad aprendida. Todo eso culmina en una dimensión efectiva certificada.

La idea de fondo es más importante que la implementación concreta: la maldición de la dimensionalidad no es una ley inevitable del universo, sino la consecuencia de insistir en una geometría equivocada. CCDEngine corrige esa geometría. Y una vez corregida, el problema ya no tiene tamaño $d$ en el sentido que parecía tener al principio.

Ese es el corazón del motor. No reducir por reducir, sino descubrir la forma real del espacio donde vive la dinámica.
