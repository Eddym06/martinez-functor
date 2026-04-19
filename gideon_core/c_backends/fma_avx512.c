/*
 * fma_avx512.c — Kernels FMA bare-metal para Gideon
 *
 * Jerarquía de implementaciones (el compilador selecciona según -m flags):
 *   1. AVX-512   (8 doubles/ciclo) — si -mavx512f -mavx512dq disponibles
 *   2. AVX2+FMA3 (4 doubles/ciclo) — si -mavx2 -mfma disponibles
 *   3. Escalar   (1 double/ciclo)  — baseline, siempre funciona
 *
 * Todos los punteros deben ser no-solapados (__restrict__).
 * El caller Rust garantiza alineación y longitud.
 *
 * Parámetros configRurables en tiempo de compilación:
 *   GIDEON_PREFETCH_DIST  — distancia de prefetch en palabras de 8 bytes (default 8)
 *                           build.rs la calcula desde el perfil de HW (gideon_autotune)
 *   GIDEON_FMA_UNROLL     — factor de desenrollado del loop AVX-512 (1, 2 ó 4)
 *                           2 es óptimo para Meteor Lake, 4 para Zen4/5
 */

#include "cpu_kernels.h"
#include <string.h>  /* memcpy */

/* ── Helpers de detección en tiempo de compilación ─────────────────────────── */
#if defined(GIDEON_AVX512) && defined(__AVX512F__) && defined(__AVX512DQ__)
  #include <immintrin.h>
  #define USE_AVX512 1
#elif defined(GIDEON_AVX2) && defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
  #define USE_AVX2_FMA 1
#endif

/* ── Parámetros de optimización ─────────────────────────────────────────────── */
#ifndef GIDEON_PREFETCH_DIST
  #define GIDEON_PREFETCH_DIST 8   /* 8 × 8 bytes = 64 bytes = 1 línea de caché */
#endif

#ifndef GIDEON_FMA_UNROLL
  #define GIDEON_FMA_UNROLL 1      /* build.rs lo sobreescribe con fma_unroll_factor */
#endif


/* ═══════════════════════════════════════════════════════════════════════════
 * gideon_fma_scalar — baseline sin SIMD
 * ═══════════════════════════════════════════════════════════════════════════ */
void gideon_fma_scalar(
    double* __restrict__ out,
    const double* __restrict__ x,
    double w,
    double b,
    size_t n
) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = w * x[i] + b;
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 * gideon_fma_vector — selecciona la ruta SIMD más rápida disponible
 * ═══════════════════════════════════════════════════════════════════════════ */
void gideon_fma_vector(
    double* __restrict__ out,
    const double* __restrict__ x,
    double w,
    double b,
    size_t n
) {
#ifdef USE_AVX512
    /* AVX-512: 8 doubles por iteración.
     *
     * Cuando GIDEON_FMA_UNROLL >= 2 desenrollamos x2 (16 doubles/iter),
     * manteniendo la ALU ocupada durante la latencia del VFMADD (4 ciclos).
     * Prefetch anticipa GIDEON_PREFETCH_DIST × 8 bytes de lectura futura.
     */
    __m512d vw = _mm512_set1_pd(w);
    __m512d vb = _mm512_set1_pd(b);
    size_t i = 0;

#if GIDEON_FMA_UNROLL >= 2
    /* Bucle desenrollado ×2: 16 doubles por iteración */
    for (; i + 16 <= n; i += 16) {
        _mm_prefetch((const char*)(x   + i + GIDEON_PREFETCH_DIST * 8), _MM_HINT_T0);
        __m512d va = _mm512_loadu_pd(x + i);
        __m512d vb2 = _mm512_loadu_pd(x + i + 8);
        _mm512_storeu_pd(out + i,     _mm512_fmadd_pd(vw, va,  vb));
        _mm512_storeu_pd(out + i + 8, _mm512_fmadd_pd(vw, vb2, vb));
    }
#endif  /* GIDEON_FMA_UNROLL >= 2 */

    /* Bucle base: 8 doubles por iteración */
    for (; i + 8 <= n; i += 8) {
        _mm_prefetch((const char*)(x + i + GIDEON_PREFETCH_DIST * 8), _MM_HINT_T0);
        __m512d vx = _mm512_loadu_pd(x + i);
        _mm512_storeu_pd(out + i, _mm512_fmadd_pd(vw, vx, vb));
    }
    /* Epílogo escalar para el resto */
    for (; i < n; ++i) {
        out[i] = w * x[i] + b;
    }

#elif defined(USE_AVX2_FMA)
    /* AVX2 + FMA3: 4 doubles por iteración */
    __m256d vw = _mm256_set1_pd(w);
    __m256d vb = _mm256_set1_pd(b);
    size_t i = 0;

#if GIDEON_FMA_UNROLL >= 2
    for (; i + 8 <= n; i += 8) {
        _mm_prefetch((const char*)(x + i + GIDEON_PREFETCH_DIST * 4), _MM_HINT_T0);
        __m256d va  = _mm256_loadu_pd(x + i);
        __m256d va2 = _mm256_loadu_pd(x + i + 4);
        _mm256_storeu_pd(out + i,     _mm256_fmadd_pd(vw, va,  vb));
        _mm256_storeu_pd(out + i + 4, _mm256_fmadd_pd(vw, va2, vb));
    }
#endif

    for (; i + 4 <= n; i += 4) {
        _mm_prefetch((const char*)(x + i + GIDEON_PREFETCH_DIST * 4), _MM_HINT_T0);
        __m256d vx = _mm256_loadu_pd(x + i);
        _mm256_storeu_pd(out + i, _mm256_fmadd_pd(vw, vx, vb));
    }
    for (; i < n; ++i) {
        out[i] = w * x[i] + b;
    }

#else
    /* Fallback escalar */
    gideon_fma_scalar(out, x, w, b, n);
#endif
}


/* ═══════════════════════════════════════════════════════════════════════════
 * gideon_fma_chain — aplica N transformaciones afines en secuencia
 *
 * Equivalente a: x_k+1 = weights[k] * x_k + biases[k]  para k in [0, n_fma)
 * Aprovecha que la iteración sobre la cadena es el bucle exterior (por cache).
 * ═══════════════════════════════════════════════════════════════════════════ */
void gideon_fma_chain(
    double* __restrict__ out,
    const double* __restrict__ x,
    const double* __restrict__ weights,
    const double* __restrict__ biases,
    size_t n_elements,
    size_t n_fma
) {
    if (n_fma == 0) {
        /* Sin transformación: copiar entrada a salida */
        memcpy(out, x, n_elements * sizeof(double));
        return;
    }

    /* Primera pasada: aplica la primera FMA de x → out */
    gideon_fma_vector(out, x, weights[0], biases[0], n_elements);

    /* Pasadas intermedias: aplica en-place sobre out.
     * No podemos pasar out==out a gideon_fma_vector (viola __restrict__).
     * Usamos el loop SIMD directamente aquí, con prefetch. */
    for (size_t k = 1; k < n_fma; ++k) {
        double w = weights[k];
        double b = biases[k];
#ifdef USE_AVX512
        {
            __m512d vw = _mm512_set1_pd(w);
            __m512d vb = _mm512_set1_pd(b);
            size_t i = 0;
#if GIDEON_FMA_UNROLL >= 2
            for (; i + 16 <= n_elements; i += 16) {
                _mm_prefetch((const char*)(out + i + GIDEON_PREFETCH_DIST * 8), _MM_HINT_T0);
                __m512d va  = _mm512_loadu_pd(out + i);
                __m512d va2 = _mm512_loadu_pd(out + i + 8);
                _mm512_storeu_pd(out + i,     _mm512_fmadd_pd(vw, va,  vb));
                _mm512_storeu_pd(out + i + 8, _mm512_fmadd_pd(vw, va2, vb));
            }
#endif
            for (; i + 8 <= n_elements; i += 8) {
                _mm_prefetch((const char*)(out + i + GIDEON_PREFETCH_DIST * 8), _MM_HINT_T0);
                __m512d v = _mm512_loadu_pd(out + i);
                _mm512_storeu_pd(out + i, _mm512_fmadd_pd(vw, v, vb));
            }
            for (; i < n_elements; ++i) { out[i] = w * out[i] + b; }
        }
#elif defined(USE_AVX2_FMA)
        {
            __m256d vw = _mm256_set1_pd(w);
            __m256d vb = _mm256_set1_pd(b);
            size_t i = 0;
#if GIDEON_FMA_UNROLL >= 2
            for (; i + 8 <= n_elements; i += 8) {
                _mm_prefetch((const char*)(out + i + GIDEON_PREFETCH_DIST * 4), _MM_HINT_T0);
                __m256d va  = _mm256_loadu_pd(out + i);
                __m256d va2 = _mm256_loadu_pd(out + i + 4);
                _mm256_storeu_pd(out + i,     _mm256_fmadd_pd(vw, va,  vb));
                _mm256_storeu_pd(out + i + 4, _mm256_fmadd_pd(vw, va2, vb));
            }
#endif
            for (; i + 4 <= n_elements; i += 4) {
                _mm_prefetch((const char*)(out + i + GIDEON_PREFETCH_DIST * 4), _MM_HINT_T0);
                __m256d v = _mm256_loadu_pd(out + i);
                _mm256_storeu_pd(out + i, _mm256_fmadd_pd(vw, v, vb));
            }
            for (; i < n_elements; ++i) { out[i] = w * out[i] + b; }
        }
#else
        for (size_t i = 0; i < n_elements; ++i) { out[i] = w * out[i] + b; }
#endif
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 * gideon_fma_fold — aplica la cadena colapsada W_total·x + B_total
 *                   Camino óptimo cuando engine.py ha calculado el fold.
 * ═══════════════════════════════════════════════════════════════════════════ */
void gideon_fma_fold(
    double* __restrict__ out,
    const double* __restrict__ x,
    double W_total,
    double B_total,
    size_t n
) {
    /* Idéntico a gideon_fma_vector — la distinción es semántica */
    gideon_fma_vector(out, x, W_total, B_total, n);
}
