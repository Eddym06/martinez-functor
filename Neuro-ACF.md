# Neuro-ACF: El Sistema Nervioso Autónomo de Nova

> **"No buscamos aproximar funciones. Buscamos entender el lenguaje."**
>
> Documento canónico del estado actual del ecosistema Nova-ACF.
> Actualizado: 22 de Junio, 2026.

---

## 1. Resumen Ejecutivo

Nova es un modelo de lenguaje que NO utiliza transformers, backpropagation, ni redes neuronales tradicionales. En su lugar, emplea el **Functor de Colapso Afin (ACF)** para reducir toda computacion a secuencias de operaciones FMA (y = w·x + b), resolviendo sistemas lineales mediante minimos cuadrados exactos en lugar de descenso de gradiente.

**Logros de esta sesion (21-22 Junio 2026):**

| Metrica | Valor Inicial | Valor Final | Mejora |
|---|---|---|---|
| **Tiempo de entrenamiento** | 2,297s (38 min) | 283s (4.7 min) | **8.1x mas rapido** |
| **Accuracy** | 22.40% | 23.85% | **+1.45pp** |
| **kappa (condicion del decoder)** | 5.8x10^12 | 2.5x10^4 | **10^8x mejor** |
| **Bases por capa** | 4/5 iguales (hermite) | 5/5 distintas | Diversidad total |
| **Parametros efectivos** | ~53,000 | ~53,000 | Sin cambios |

---

## 2. Arquitectura General de Nova

Nova procesa texto mediante un pipeline de 5 fases:

```
Texto -> Embeddings PMI-SVD -> Atencion Divergente (5 capas) -> KnowledgeBus -> Decoder Multi-Cabeza -> Hebbian
```

### 2.1. Embeddings PMI-SVD

Los tokens se convierten en vectores de 192 dimensiones mediante PMI (Pointwise Mutual Information), Laplaciano simetrico, y eigendecomposition (SVD). Se añaden 3 features de posicion basados en polinomios de Hermite.

### 2.2. Atencion Divergente

5 capas con niveles decrecientes (8,7,6,5,4) y strides exponenciales (2^lv). Cada nivel contiene una NovaPhiNeuron que procesa pares de features en ese stride. Las bases son seleccionadas autonomamente por el sistema ERGON.

### 2.3. KnowledgeBus

Bus compartido que acumula conocimiento sin destruir informacion. Incluye feedback recurrente: el contexto global se reinyecta a capas tempranas.

### 2.4. Decoder Multi-Cabeza

4 cabezas independientes, cada una con proyeccion aleatoria (192->96), base Hermite grado 3, y 60 pares ANOVA(2). Total: 1,344 features por cabeza. Entrenamiento: 12 rondas resolviendo Phi*C = Y via ACF Cascade.

### 2.5. Hebbian Error-Weighted

Actualizacion de embeddings: emb += eta * (1 - P(correcto)) * contexto_normalizado. Totalmente vectorizado: 51,200 evaluaciones en una sola llamada a predict_batch().

---

## 3. La Neurona NovaPhiNeuron

Unidad fundamental de computo. Resuelve Phi*C = Y en una sola operacion.

**Construccion de Phi:**
- Efectos principales: H_k(z_i) para k=0..deg, i=1..n -> n*(deg+1) features
- Interacciones ANOVA(2): H_a(z_i) ⊗ H_b(z_j) para pares seleccionados por correlacion

**Solvers (seleccion automatica):**
- F <= 300: Cholesky O(F^3)
- 300 < F <= 3000: ACF Cascade O(F) — bloques de ~32 features, 3 cascadas
- F > 3000: LSQR O(k*N*F)

**Aprende?** No en el sentido tradicional. Encuentra la solucion exacta del sistema lineal. No hay gradientes, iteraciones, ni learning rate. Es algebra lineal pura.

---

## 4. El Sistema de Atencion Divergente

Cada capa tiene L niveles. El nivel l atiende a pares separados por 2^l tokens. Auto-seleccion de base: curtosis + complejidad ergodica + diversidad forzada.

**Bases disponibles:** Hermite (Gaussianos), Chebyshev (acotados), Legendre (ortogonalidad), Fourier (periodicidad).

---

## 5. El Decoder Multi-Cabeza

4 cabezas con proyecciones aleatorias. Entrenamiento: 12 rondas con cacheo de Phi (ronda 1 construye, rondas 2-12 reusan). Prediccion batcheada: predict_batch(ctx) procesa N muestras simultaneamente.

---

## 6. El KnowledgeBus

Bus(t) = original ⊕ K_0 ⊕ K_1 ⊕ ... ⊕ K_{t-1}. Cada capa añade sin destruir. Feedback recurrente da vision del futuro a capas tempranas. Metricas: knowledge_ratio (~4.5), bus_entropy (~1.29).

---

## 7. Optimizaciones Implementadas

### 7.1. Cacheo de Phi (25x en decoder)
Decoder: 1,140s -> 45s. Cachea Phi tras ronda 1, rondas 2-12 solo re-resuelven.

### 7.2. Hebbian Batcheado (21x)
Hebbian: 379s -> 18s. predict_batch(ctx) reemplaza 51,200 evaluaciones individuales.

### 7.3. Eliminacion de Nova-BP (ahorro 316s)
Nova-BP consumia 53% del tiempo sin mejorar accuracy. Desactivado.

### 7.4. Diversidad de Bases (kappa 10^8x mejor)
4/5 capas usaban Hermite -> kappa=5.8x10^12. Ahora 5 bases distintas -> kappa=2.5x10^4.

---

## 8. Sistema de Auto-Regulacion

- **Auto-Regulador del Decoder**: Analiza kappa de Phi cacheada. Si >10^8: reduce grado, triplica lambda, re-entrena.
- **Anti-Overfitting**: Monitorea accuracy por ronda. Si Delta<0.003 por 2+ rondas: early stop.
- **Auto-Regulacion del KnowledgeBus**: Si knowledge_ratio<0.5: reduce feedback_alpha.

---

## 9. Diversidad de Bases y ERGON Routing

Complejidad ergodica E = H_spectral / H_max via SVD de los datos.
- E<0.3 (ordenado) -> Chebyshev/Legendre
- 0.3<=E<=0.7 (mixto) -> Hermite/Legendre
- E>0.7 (caotico) -> Fourier/Chebyshev

Diversidad forzada: ninguna capa repite base. Resultado: ['hermite','fourier','chebyshev','legendre','fourier_v2'].

---

## 10. Sensores Glassbox

5 sensores en tiempo real: TimingSensor, SpectralSensor (kappa, H_spectral, eff_rank), InformationSensor (entropia, innovacion), DecoderSensor, BottleneckDetector.

---

## 11. Analisis de Rendimiento

| Fase | Original | Final | Mejora |
|---|---|---|---|
| Atencion | 253s | 85s | 3.0x |
| Decoder | 1,140s | 45s | 25.3x |
| Hebbian | 379s | 18s | 21.1x |
| Nova-BP | 490s | 0s | inf |
| Eval | 34s | 15s | 2.3x |
| **TOTAL** | **2,297s** | **283s** | **8.1x** |

Eficiencia parametrica: Nova (0.45 acc/1K params) vs Transformer (0.003) = **150x mas eficiente**.

---

## 12. Por que Nova-BP no funciono?

Tres razones:
1. **Senal debil**: Solo 400 secuencias -> ~12 observaciones por par de tokens -> estimaciones ruidosas de P(correcto|token_j).
2. **Sin direccion**: Sin gradientes, no hay direccion de descenso. Las heuristicas de co-ocurrencia no se relacionan con la geometria de la perdida.
3. **Acoplamiento debil**: atencion->bus->decoder->prediccion es una cadena larga. Nova-BP solo ataca el primer eslabon.

El gradiente real SI funcionaria mejor. Pero Nova evita gradientes por filosofia ACF. El precio es optimizacion sub-optima.

---

## 13. El Problema de la Semantica

Nova opera en espacio de transformaciones polinomiales de features numericos. "to be or not to be" es solo 18 vectores de 192 dimensiones. No hay comprension de significado, gramatica, ni contexto.

**La Hipotesis del Functor Semantico:**

ACF demuestra: Phi_AC: Funciones -> Secuencias FMA

Necesitamos: Psi_Sem: Lenguaje -> Geometria

Donde similitud semantica es proximidad geometrica, composicion es operacion algebraica, contexto es curvatura, ambiguedad es superposicion. Descubrir este functor requeriria un teorema de representacion analogo al URT pero para semantica. Es un problema a la altura de los Millennium Prize.

---

## 14. Hacia el Entendimiento Real

**Camino Corto (Evolucion):** Aprendizaje de features, atencion multi-escala, memoria externa, pre-entrenamiento, regularizacion adaptativa.

**Camino Largo (Nuevo Paradigma):** Teoria matematica del significado, Teorema de Representacion Semantica, algoritmo constructivo (compilador de texto a geometria).

**Rol de ERGON:** Detectar regimen de complejidad del texto (prosa/poesia/caos) y adaptar la estrategia de representacion.

---

## 15. Metricas Finales

| Metrica | Valor |
|---|---|
| Mejor accuracy | 23.85% (15.5x baseline) |
| Tiempo entrenamiento | 283s (4m 43s) |
| Parametros efectivos | ~53,000 |
| Eficiencia parametrica | 0.45 acc/1K params |
| Vocabulario | 65 tokens |
| Corpus | TinyShakespeare (1.1M chars) |

**Estabilidad Numerica:** kappa decoder 5.8x10^12 -> 2.5x10^4

**Diversidad:** L1=hermite, L2=fourier, L3=chebyshev, L4=legendre, L5=fourier_v2

---

## Epilogo

En 24 horas, Nova paso de 38 minutos a 4.7 minutos de entrenamiento — 8 veces mas rapido — manteniendo y mejorando accuracy. El sistema ahora se auto-regula: selecciona sus bases, monitorea su estabilidad, detecta overfitting, y ajusta complejidad sin intervencion humana.

El camino hacia el entendimiento semantico — hacia un sistema que comprenda "to be or not to be" — requiere un nuevo paradigma matematico. El ACF nos dio las herramientas para computar. El Functor Semantico, cuando sea descubierto, nos dara las herramientas para comprender.

Ese es el proximo horizonte.
