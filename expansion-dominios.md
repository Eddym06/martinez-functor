I. ✅ COMPLETADO — Tensor ACF (implementado en `acf_functor/tensor_acf.py`)
   - TT-SVD, Tucker HOSVD, evaluación zipper, invariantes α tensoriales
   - 20 tests (15 unitarios + 3 integración + 2 masivos) — todos pasan
   - Lean 4: `MathTest/TensorACFCertificates.lean`
   - Documentación: Paper.md §36, Poema.md §17, Poema-manual.md §29

Problema actual: el functor opera sobre 
f
:
[
−
1
,
1
]
→
R
f:[−1,1]→R. Toda la ciencia computacional necesita 
f
:
[
−
1
,
1
]
d
→
R
f:[−1,1] 
d
 →R.

Extensión natural:

Φ
(
f
)
(
x
1
,
…
,
x
d
)
=
∑
k
1
,
…
,
k
d
c
k
1
⋯
k
d
 
T
k
1
(
x
1
)
⋯
T
k
d
(
x
d
)
Φ(f)(x 
1
​
 ,…,x 
d
​
 )= 
k 
1
​
 ,…,k 
d
​
 
∑
​
 c 
k 
1
​
 ⋯k 
d
​
 
​
 T 
k 
1
​
 
​
 (x 
1
​
 )⋯T 
k 
d
​
 
​
 (x 
d
​
 )

Pero esto crece como 
O
(
n
d
)
O(n 
d
 ) — maldición de la dimensionalidad. La solución es la descomposición en tren tensorial (TT/MPS):

f
(
x
1
,
…
,
x
d
)
≈
∑
α
1
,
…
,
α
d
−
1
A
1
[
x
1
]
α
1
A
2
[
x
2
]
α
1
α
2
⋯
A
d
[
x
d
]
α
d
−
1
f(x 
1
​
 ,…,x 
d
​
 )≈ 
α 
1
​
 ,…,α 
d−1
​
 
∑
​
 A 
1
​
 [x 
1
​
 ] 
α 
1
​
 
 A 
2
​
 [x 
2
​
 ] 
α 
1
​
 α 
2
​
 
 ⋯A 
d
​
 [x 
d
​
 ] 
α 
d−1
​
 
 

donde cada 
A
k
A 
k
​
  es un tensor de rango 
r
r ("rango TT"). La reducción ACF de cada factor 
A
k
A 
k
​
  es una reducción 1D aplicada capa a capa.

Conexión directa con código existente:

graph_acf.py: un grafo de 
n
n nodos con tensor de señal es ya el caso 
d
=
1
d=1 de señal sobre rejillas 
d
d-dimensionales.
meta_compiler.py: el espacio de gramáticas se extiende añadiendo BasisFamily.TENSOR_TRAIN, BasisFamily.TUCKER.
koopman_adaptive.py: Koopman multidimensional via EDMD con observables tensoriados.
Nuevo módulo: tensor_acf.py con TensorTrainReducer, TuckerACF, CPDecompositionACF, TensorACFInvariants(alpha_per_mode, tt_rank, tucker_multilinear_rank).

Aplicaciones directas: PDEs en dominio 3D, funciones de valor en juegos, funciones de onda cuántica 
ψ
(
x
1
,
…
,
x
N
)
ψ(x 
1
​
 ,…,x 
N
​
 ).

II. Operadores sobre Espacios de Funciones — ACF de EDPs
Problema: el functor reduce 
f
:
R
→
R
f:R→R. Una EDP es un operador 
L
:
L
2
(
Ω
)
→
L
2
(
Ω
)
L:L 
2
 (Ω)→L 
2
 (Ω), que es un objeto de orden superior al que el ACF aún no accede directamente.

Extensión: reducir la función de Green 
G
(
x
,
y
)
G(x,y) del operador:

(
L
u
)
(
x
)
=
∫
Ω
G
(
x
,
y
)
 
u
(
y
)
 
d
y
(Lu)(x)=∫ 
Ω
​
 G(x,y)u(y)dy

G
:
Ω
×
Ω
→
R
G:Ω×Ω→R es una función de dos variables — caso especial de Tensor ACF con 
d
=
2
d=2. La reducción ACF de 
G
G produce una aproximación de rango bajo:

G
(
x
,
y
)
≈
∑
k
=
1
r
ϕ
k
(
x
)
 
ψ
k
(
y
)
G(x,y)≈ 
k=1
∑
r
​
 ϕ 
k
​
 (x)ψ 
k
​
 (y)

que convierte el operador integral en 
r
r productos exterior — exactamente una secuencia FMA matricial.

Conexión directa:

El kernel de Koopman 
K
(
x
,
y
)
=
∑
j
λ
j
ϕ
j
(
x
)
ψ
j
(
y
)
K(x,y)=∑ 
j
​
 λ 
j
​
 ϕ 
j
​
 (x)ψ 
j
​
 (y) ya tiene esta forma en koopman_adaptive.py.
mixed_composition.py ya tiene la teoría de composición operador-polinomio.
La entropía de Kolmogorov de 
L
L ya está formalizada en kolmogorov_entropy.py.
Nuevo módulo: operator_acf.py con GreenFunctionReducer, IntegralOperatorACF, PDEKernelACF, OperatorACFInvariants(spectral_decay, green_rank, operator_alpha).

Aplicaciones: solver de EDP con operador ACF-comprimido, aceleración de métodos de elementos de frontera, compresión de matrices de atención en transformers.

III. Funciones Estocásticas — ACF de Incertidumbre (Polynomial Chaos)
Problema: el functor opera sobre funciones deterministas. Los sistemas reales tienen incertidumbre paramétrica 
f
(
x
;
ξ
)
f(x;ξ) donde 
ξ
∼
μ
ξ∼μ es aleatorio.

Extensión — Expansión en Caos Polinomial (PCE):

f
(
x
;
ξ
)
=
∑
∣
α
∣
≤
p
c
α
(
x
)
 
H
α
(
ξ
)
f(x;ξ)= 
∣α∣≤p
∑
​
 c 
α
​
 (x)H 
α
​
 (ξ)

donde 
H
α
H 
α
​
  son los polinomios de Hermite/Legendre ortogonales respecto a 
μ
μ (base estocástica), y 
c
α
(
x
)
c 
α
​
 (x) son funciones deterministas cada una reducible por ACF estándar.

La reducción ACF del campo estocástico completo es la composición:

Φ
stoch
(
f
)
=
∑
α
Φ
(
c
α
)
 
H
α
Φ 
stoch
​
 (f)= 
α
∑
​
 Φ(c 
α
​
 )H 
α
​
 

El invariante 
α
α del campo estocástico mide el decaimiento de 
∥
c
α
∥
∥c 
α
​
 ∥ en función de 
∣
α
∣
∣α∣ — directamente legible como "cuánta incertidumbre es efectiva".

Conexión directa:

thermodynamic_acf.py: la temperatura 
β
β del ACF termodinámico es análoga a la temperatura inversa de la distribución 
μ
μ.
information_geometry.py: la métrica de Fisher sobre 
μ
μ ya está formalizada.
superposition.py: la superposición de estados SuperpositionState ya maneja combinaciones lineales pesadas.
Nuevo módulo: stochastic_acf.py con PolynomialChaosACF, StochasticReducer, UncertaintyBound(alpha_stoch, effective_dimension, confidence_band).

Aplicaciones: UQ científico, inferencia bayesiana acelerada, reducción de modelos en ingeniería probabilística.

IV. Ecuaciones Diferenciales Ordinarias / Control — ACF de Campos Vectoriales
Problema: el functor reduce una función 
f
:
R
→
R
f:R→R. Un sistema dinámico 
x
˙
=
f
(
x
)
x
˙
 =f(x) tiene el campo vectorial 
f
:
R
n
→
R
n
f:R 
n
 →R 
n
  como su objeto generador.

Extensión 1 — Reducción del campo vectorial:

Φ
(
f
)
=
(
Φ
(
f
1
)
,
…
,
Φ
(
f
n
)
)
Φ(f)=(Φ(f 
1
​
 ),…,Φ(f 
n
​
 ))

genera un sistema reducido 
x
˙
=
Φ
(
f
)
(
x
)
x
˙
 =Φ(f)(x) cuyas trayectorias difieren de las originales en 
O
(
ε
⋅
e
L
t
)
O(ε⋅e 
Lt
 ) (Gronwall).

Extensión 2 — Función de valor HJB:

La función de valor 
V
(
x
)
=
min
⁡
u
∫
0
∞
ℓ
(
x
,
u
)
 
d
t
V(x)=min 
u
​
 ∫ 
0
∞
​
 ℓ(x,u)dt satisface la ecuación Hamilton-Jacobi-Bellman:

0
=
min
⁡
u
 
[
ℓ
(
x
,
u
)
+
∇
V
(
x
)
⊤
f
(
x
,
u
)
]
0= 
u
min
​
 [ℓ(x,u)+∇V(x) 
⊤
 f(x,u)]

V
:
R
n
→
R
V:R 
n
 →R es una función reducible directamente por ACF. La política óptima 
u
∗
(
x
)
=
−
1
2
R
−
1
g
(
x
)
⊤
∇
Φ
(
V
)
(
x
)
u 
∗
 (x)=− 
2
1
​
 R 
−1
 g(x) 
⊤
 ∇Φ(V)(x) sale gratis del gradiente del polinomio Chebyshev.

Extensión 3 — Funciones de Lyapunov certificadas:

Si 
V
V es una función de Lyapunov para 
x
˙
=
f
(
x
)
x
˙
 =f(x), y 
Φ
(
V
)
Φ(V) es su reducción, entonces 
Φ
(
V
)
Φ(V) también es una función de Lyapunov con la misma región de atracción si 
ε
<
∥
V
˙
∥
∞
/
∥
f
∥
∞
ε<∥ 
V
˙
 ∥ 
∞
​
 /∥f∥ 
∞
​
 . Esto da certificados de estabilidad exportables a Lean 4.

Conexión directa:

koopman_adaptive.py: Koopman para sistemas dinámicos ya existe — pero solo como análisis espectral.
acf_inverse.py: 
Φ
−
1
Φ 
−1
  sobre el campo vectorial reconstruye el sistema original.
genesis.py con TheoremSeeds: candidatos a funciones de Lyapunov.
Nuevo módulo: ode_acf.py con VectorFieldReducer, HJBReducer, LyapunovACF(certificate, region_of_attraction, epsilon_stability), TrajectoryACF.

V. 🧊 CONGELADO — Funciones Complejas — ACF de Análisis Complejo
Problema: el functor opera sobre 
R
R. Las funciones analíticas 
f
:
C
→
C
f:C→C tienen estructura más rica: series de Laurent, polos, residuos.

Extensión — Aproximantes de Padé como ACF racional:

Los aproximantes de Padé 
[
m
/
n
]
(
z
)
=
P
m
(
z
)
/
Q
n
(
z
)
[m/n](z)=P 
m
​
 (z)/Q 
n
​
 (z) son la generalización natural de Chebyshev para funciones con polos: el denominador 
Q
n
Q 
n
​
  captura los polos, el numerador 
P
m
P 
m
​
  los residuos.

Φ
Pad
e
ˊ
(
f
)
=
P
m
(
z
)
Q
n
(
z
)
,
ε
=
∥
f
−
P
m
Q
n
∥
H
2
Φ 
Pad 
e
ˊ
 
​
 (f)= 
Q 
n
​
 (z)
P 
m
​
 (z)
​
 ,ε= 
​
 f− 
Q 
n
​
 
P 
m
​
 
​
  
​
  
H 
2
 
​
 

La secuencia FMA para evaluar 
P
m
(
z
)
/
Q
n
(
z
)
P 
m
​
 (z)/Q 
n
​
 (z) es dos cadenas Horner con una división — evaluación 
O
(
m
+
n
)
O(m+n) en lugar de 
O
(
2
n
)
O(2 
n
 ).

El invariante 
α
α complejo mide la velocidad de decaimiento de los residuos de 
f
f en sus polos — directamente relacionado con los coeficientes de Laurent.

Extensión adicional — Transformaciones conformes:

Una transformación conforme 
w
=
ϕ
(
z
)
w=ϕ(z) es una función 
ϕ
:
C
→
C
ϕ:C→C holomorfa biyectiva. ACF puede usarla para mapear el dominio: en vez de reducir 
f
f en 
[
a
,
b
]
[a,b], reduce 
f
∘
ϕ
−
1
f∘ϕ 
−1
  en el disco unitario donde la convergencia de Chebyshev es exponencial.

Conexión directa:

meta_compiler.py: añadir BasisFamily.PADE, BasisFamily.HARDY_H2, BasisFamily.RATIONAL.
galois_symmetry.py: la simetría de Galois sobre el campo complejo 
C
C ya está formalizada.
Nuevo módulo: complex_acf.py con PadeReducer, HardySpaceACF, ConformalACF, LaurentACF, ComplexACFInvariants(pole_count, residue_decay, hardy_alpha).

VI. 🧊 CONGELADO — Medidas y Transporte Óptimo — ACF de Wasserstein
Problema: el functor opera sobre funciones. Las distribuciones de probabilidad 
μ
,
ν
μ,ν sobre 
R
n
R 
n
  son objetos en un espacio métrico completamente diferente.

Extensión — Mapa de Brenier como ACF:

Por el Teorema de Brenier, existe un único mapa de transporte óptimo 
T
:
R
n
→
R
n
T:R 
n
 →R 
n
  tal que 
T
#
μ
=
ν
T 
#
​
 μ=ν y 
T
=
∇
ϕ
T=∇ϕ para algún potencial convexo 
ϕ
ϕ. El potencial 
ϕ
:
R
n
→
R
ϕ:R 
n
 →R es reducible por ACF.

Φ
(
ϕ
)
  
⟹
  
Φ
(
T
)
=
∇
Φ
(
ϕ
)
(gradiente del polinomio Chebyshev — exacto)
Φ(ϕ)⟹Φ(T)=∇Φ(ϕ)(gradiente del polinomio Chebyshev — exacto)

La distancia de Wasserstein 
W
2
(
μ
,
ν
)
2
=
∫
∥
x
−
T
(
x
)
∥
2
d
μ
(
x
)
W 
2
​
 (μ,ν) 
2
 =∫∥x−T(x)∥ 
2
 dμ(x) se convierte en una integral sobre la reducción ACF: se puede evaluar analíticamente usando la secuencia FMA.

Conexión directa:

information_geometry.py: geometría de información Fisher-Rao es dual a la geometría de Wasserstein (Otto-Villani).
La métrica de Wasserstein sobre el espacio de reducciones ACF daría una nueva métrica en 
M
f
M 
f
​
  (espacio de móduli).
moduli_spaces.py: el camino geodésico en el espacio de móduli es un transporte óptimo entre reducciones.
Nuevo módulo: wasserstein_acf.py con BrenierPotentialReducer, WassersteinACF, OptimalTransportCertificate.

VII. ✅ COMPLETADO — Matrix ACF (implementado en `acf_functor/matrix_acf.py`)
   - ChebyshevMatrixReducer, Clenshaw, 5 funciones matriciales built-in
   - 20 tests — todos pasan. Lean 4 en TensorACFCertificates.lean
   - Paper.md §37, Poema.md §18, Poema-manual.md §30
Problema: el functor reduce funciones escalares. Las matrices 
A
∈
R
n
×
n
A∈R 
n×n
  son, vía funciones de matrices, objetos que el ACF puede abordar directamente.

Extensión — Funciones de matrices via Chebyshev:

f
(
A
)
=
∑
k
=
0
d
c
k
T
k
(
A
)
(Chebyshev de matriz)
f(A)= 
k=0
∑
d
​
 c 
k
​
 T 
k
​
 (A)(Chebyshev de matriz)

Si el espectro de 
A
A está en 
[
λ
min
⁡
,
λ
max
⁡
]
[λ 
min
​
 ,λ 
max
​
 ] (mapeado a 
[
−
1
,
1
]
[−1,1]), la secuencia FMA para evaluar 
f
(
A
)
f(A) es 
d
d multiplicaciones matriziales — el caso matricial exacto del HornerReducer.

Aplicaciones directas:

e
t
A
e 
tA
 : exponencial de matriz (heat flow), 
d
≈
10
d≈10 para precisión doble a tiempos cortos.
(
A
+
σ
I
)
−
1
(A+σI) 
−1
 : resolución de sistemas lineales via serie de Neumann cuando 
σ
σ es grande.
A
1
/
2
A 
1/2
 : raíz de matriz via Chebyshev en 
[
λ
min
⁡
,
λ
max
⁡
]
[λ 
min
​
 ,λ 
max
​
 ].
Conexión directa:

neural_acf.py: ya reduce 
W
∈
R
m
×
n
W∈R 
m×n
  via SVD, extensión natural a funciones de matrices.
lie_analysis.py: el álgebra de Lie 
g
⊂
g
l
(
n
)
g⊂gl(n) ya opera en matrices.
koopman_adaptive.py: el operador de Koopman 
K
K ya es una matriz en la representación EDMD.
Nuevo módulo: matrix_acf.py con MatrixFunctionACF, ChebyshevMatrixReducer, MatrixFunctionInvariants(matrix_alpha, spectral_range, condition_number_bound).

VIII. 🧊 CONGELADO — Series Temporales No Estacionarias — Temporal ACF
Problema: el functor reduce 
f
(
x
)
f(x) en un dominio estático. Las series temporales 
{
f
t
}
t
≥
0
{f 
t
​
 } 
t≥0
​
  tienen parámetros que derivan en el tiempo.

Extensión — Reducción deslizante con ventana adaptativa:

Φ
[
t
−
w
,
t
]
(
f
)
con 
w
=
w
(
t
)
 adaptativo seg
u
ˊ
n 
∥
ε
(
w
)
∥
Φ 
[t−w,t]
​
 (f)con w=w(t) adaptativo seg 
u
ˊ
 n ∥ε(w)∥

El invariante 
α
(
t
)
α(t) se convierte en un proceso estocástico que mide la complejidad instantánea de la señal. Cuando 
α
(
t
)
α(t) salta, el sistema detectó un punto de cambio (change-point detection vía ACF).

Conexión directa:

auto_evolution.py: AdaptiveRefinement ya subdivide intervalos; aplicar en la dimensión temporal.
kolmogorov_entropy.py: la entropía de Kolmogorov de una serie temporal es exactamente lo que mide.
Koopman ya opera sobre trayectorias — esta extensión lo hace online/incremental.
Nuevo módulo: temporal_acf.py con SlidingWindowACF, NonStationaryReducer, AlphaProcess(alpha_trajectory, change_points), IncrementalKoopman.

IX. 🧊 CONGELADO — Geometría Diferencial — ACF en Variedades
Problema: el functor opera en 
R
n
R 
n
 . Las variedades riemannianas 
M
M (esferas, toros, grupos de Lie) son dominios donde muchas funciones en física y aprendizaje automático están definidas naturalmente.

Extensión — ACF espectral en variedades:

El operador de Laplace-Beltrami 
Δ
M
Δ 
M
​
  en una variedad 
M
M tiene eigenfunciones 
{
ϕ
j
}
{ϕ 
j
​
 } que son la base de Fourier adaptada a la geometría. La señal 
f
:
M
→
R
f:M→R se expande:

f
=
∑
j
f
^
j
ϕ
j
,
f
^
j
=
∫
M
f
 
ϕ
j
 
d
vol
f= 
j
∑
​
  
f
^
​
  
j
​
 ϕ 
j
​
 , 
f
^
​
  
j
​
 =∫ 
M
​
 fϕ 
j
​
 dvol

El ACF espectral reduce 
f
f truncando esta expansión con un filtro polinomial 
H
(
λ
j
)
H(λ 
j
​
 ) — exactamente el mismo mecanismo que graph_acf.py en el límite de grafo fino.

graph_acf.py ya es el caso discreto de este dominio. El continuo es la generalización natural.

Nuevo módulo: manifold_acf.py con LaplaceBeltramiACF, SphericalHarmonicsACF, RiemannianReducer, ManifoldACFInvariants(volume, laplacian_alpha, heat_kernel_decay).

X. 🧊 CONGELADO — Circuitos Cuánticos — ACF en Espacio de Hilbert
Problema: el functor opera en 
R
R. Pero ya existe superposition.py. La extensión lógica es el espacio de Hilbert 
H
H.

Extensión:

Un estado cuántico 
∣
ψ
⟩
∈
L
2
(
R
)
∣ψ⟩∈L 
2
 (R) tiene función de Wigner 
W
ψ
(
x
,
p
)
∈
R
2
n
W 
ψ
​
 (x,p)∈R 
2n
  (representación fase-espacio) reducible por ACF bidimensional. La función de Wigner de un estado gaussiano es precisamente un polinomio de grado 2 — 
Φ
(
W
ψ
)
Φ(W 
ψ
​
 ) con 
ε
=
0
ε=0.

Para estados no gaussianos, el ACF reduce 
W
ψ
W 
ψ
​
  dando un indicador de no-gaussianidad 
ε
Wigner
ε 
Wigner
​
  que mide la cantidad de entrelazamiento cuántico representable como polinomio Chebyshev.

Conexión directa:

superposition.py: SuperpositionState ya maneja mezclas cuánticas de reducciones ACF.
field_action.py: acción de campo 
S
=
∫
L
 
d
4
x
S=∫Ld 
4
 x (mecánica cuántica de campos).






 Los 4 vectores de expansión más potentes
1. Tensor ACF — desbloquea toda la ciencia computacional multivariable. El ACF pasa de ser un compresor de funciones 1D a un motor de aproximación general. Rango TT como "grado multidimensional" — el invariante 
α
α se convierte en un tensor de invariantes por modo.

2. ODE/Control ACF — el campo vectorial 
f
(
x
)
f(x) como objeto ACF primario. Las funciones de Lyapunov como certificados formales exportables a Lean 4. La política óptima 
u
∗
(
x
)
=
∇
Φ
(
V
)
(
x
)
u 
∗
 (x)=∇Φ(V)(x) gratis del gradiente del polinomio.

3. Stochastic/PCE ACF — el functor opera sobre funciones aleatorias. 
α
efect
α 
efect
​
  mide la dimensionalidad efectiva de la incertidumbre. El meta-compilador elige automáticamente entre Hermite, Legendre, Laguerre según la distribución de 
ξ
ξ.

4. Matrix Function ACF — extiende neural_acf.py a álgebra lineal numérica general. 
f
(
A
)
=
∑
k
c
k
T
k
(
A
)
f(A)=∑ 
k
​
 c 
k
​
 T 
k
​
 (A) convierte la exponencial matricial, la raíz cuadrada y los resolvidores lineales en cadenas FMA matriciales con cotas 
ε
ε formales.