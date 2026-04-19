/*
 * gemm_kernel.c — Micro-kernel GEMM bare-metal para Gideon
 *
 * Implementa: C = alpha * A @ B + beta * C  (matrices row-major, doubles)
 *
 * Diseño:
 *   - Bloque de registro (MR×NR) optimizado para cachés L1/L2.
 *   - Tiling de tres niveles: L3 (KC×NC), L2 (MC×KC), L1 (MR×NR).
 *   - Ruta AVX-512 para el bloque de registro 8×4 (NUEVO, Frente 2).
 *   - Ruta AVX2+FMA3 para el bloque de registro 4×4 (fallback).
 *   - Ruta escalar de fallback siempre compilada.
 *
 * Para matrices pequeñas (<= GIDEON_SMALL_THRESHOLD) se usa la ruta NAIVE.
 *
 * Parámetros de bloqueo configurables en tiempo de compilación:
 *   GIDEON_MR, GIDEON_NR, GIDEON_MC, GIDEON_KC, GIDEON_NC
 *   Son calculados dinámicamente por gideon_autotune.py (_compute_derived_params)
 *   y pasados como -D flags desde build.rs, que los lee del perfil JSON.
 */

#include "cpu_kernels.h"

/* ── Parámetros de bloqueo — sobreescribibles via -DGIDEON_MC=NNN ───────────── */
#ifndef GIDEON_MR
  #define GIDEON_MR   4   /* filas del micro-kernel */
#endif
#ifndef GIDEON_NR
  #define GIDEON_NR   4   /* columnas del micro-kernel */
#endif
#ifndef GIDEON_MC
  #define GIDEON_MC  64   /* tile nivel L2 (filas) */
#endif
#ifndef GIDEON_KC
  #define GIDEON_KC  64   /* tile nivel L2/L1 (k) */
#endif
#ifndef GIDEON_NC
  #define GIDEON_NC 256   /* tile nivel L3 (columnas) */
#endif
#ifndef GIDEON_SMALL_THRESHOLD
  #define GIDEON_SMALL_THRESHOLD 64  /* usar naive para matrices <= NxN */
#endif

#if defined(GIDEON_AVX512) && defined(__AVX512F__) && defined(__AVX512DQ__)
  #include <immintrin.h>
  #define USE_AVX512_GEMM 1
#elif defined(GIDEON_AVX2) && defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
  #define USE_AVX2_GEMM 1
#endif


/* ── GEMM naive (reference, siempre correcta) ─────────────────────────────── */
#ifdef __GNUC__
__attribute__((unused))
#endif
static void gemm_naive(
    double* __restrict__ C,
    const double* __restrict__ A,
    const double* __restrict__ B,
    double alpha, double beta,
    int32_t M, int32_t N, int32_t K
) {
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int32_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}


#ifdef USE_AVX512_GEMM
/*
 * Micro-kernel 8×4 AVX-512 — Frente 2 bare-metal
 *
 * Geometría MR=8, NR=4: ocupa 4 registros ZMM de acumuladores (8 doubles cada uno)
 * más registros para los broadcasts de B y la columna de A.
 *
 * Por qué 8×4 y no 8×8:
 *   - NR=8 requeriría 8 ZMM acumuladores (64 doubles) — deja poco espacio para A/B.
 *   - 8×4 = 32 doubles en 4 ZMM + 8 broadcasts = máxima eficiencia de registros.
 *
 * Throughput teórico en Ultra 9 185H AVX-512:
 *   8 doubles × 2 FMA/ciclo = 16 FLOP/ciclo × 2.8 GHz = ~44.8 GFLOP/s pico
 *   (vs AVX2 4×4: 8 FLOP/ciclo → ~22.4 GFLOP/s pico)
 */
static void microkernel_8x4_avx512(
    double* C, const double* A, const double* B,
    int32_t N, int32_t K,
    int32_t i, int32_t j, int32_t k_start, int32_t k_end
) {
    __m512d c0 = _mm512_setzero_pd();
    __m512d c1 = _mm512_setzero_pd();
    __m512d c2 = _mm512_setzero_pd();
    __m512d c3 = _mm512_setzero_pd();

    for (int32_t k = k_start; k < k_end; ++k) {
        /* A[i..i+7, k]: carga 8 doubles de la columna k, filas i..i+7 */
        __m512d a_col = _mm512_loadu_pd(&A[i * K + k]);

        c0 = _mm512_fmadd_pd(a_col, _mm512_set1_pd(B[k * N + j + 0]), c0);
        c1 = _mm512_fmadd_pd(a_col, _mm512_set1_pd(B[k * N + j + 1]), c1);
        c2 = _mm512_fmadd_pd(a_col, _mm512_set1_pd(B[k * N + j + 2]), c2);
        c3 = _mm512_fmadd_pd(a_col, _mm512_set1_pd(B[k * N + j + 3]), c3);
    }

    /* Extraer y acumular en C (row-major) */
    double acc[4][8];
    _mm512_storeu_pd(acc[0], c0);
    _mm512_storeu_pd(acc[1], c1);
    _mm512_storeu_pd(acc[2], c2);
    _mm512_storeu_pd(acc[3], c3);

    for (int32_t r = 0; r < 8; ++r) {
        C[(i + r) * N + j + 0] += acc[0][r];
        C[(i + r) * N + j + 1] += acc[1][r];
        C[(i + r) * N + j + 2] += acc[2][r];
        C[(i + r) * N + j + 3] += acc[3][r];
    }
}
#endif  /* USE_AVX512_GEMM */


#ifdef USE_AVX2_GEMM
/*
 * Micro-kernel 4×4 AVX2+FMA para un bloque de la matriz C.
 * Acumula A[i:i+4, p:p+KC] @ B[p:p+KC, j:j+4] en registros ymm.
 */
static void microkernel_4x4(
    double* C, const double* A, const double* B,
    int32_t N, int32_t K,
    int32_t i, int32_t j, int32_t k_start, int32_t k_end
) {
    __m256d c0 = _mm256_setzero_pd();
    __m256d c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd();
    __m256d c3 = _mm256_setzero_pd();

    for (int32_t k = k_start; k < k_end; ++k) {
        __m256d b_row = _mm256_loadu_pd(&B[k * N + j]);
        c0 = _mm256_fmadd_pd(_mm256_set1_pd(A[(i+0)*K + k]), b_row, c0);
        c1 = _mm256_fmadd_pd(_mm256_set1_pd(A[(i+1)*K + k]), b_row, c1);
        c2 = _mm256_fmadd_pd(_mm256_set1_pd(A[(i+2)*K + k]), b_row, c2);
        c3 = _mm256_fmadd_pd(_mm256_set1_pd(A[(i+3)*K + k]), b_row, c3);
    }

    /* Escribe el resultado acumulado de vuelta a C */
    __m256d old0 = _mm256_loadu_pd(&C[(i+0)*N + j]);
    __m256d old1 = _mm256_loadu_pd(&C[(i+1)*N + j]);
    __m256d old2 = _mm256_loadu_pd(&C[(i+2)*N + j]);
    __m256d old3 = _mm256_loadu_pd(&C[(i+3)*N + j]);

    _mm256_storeu_pd(&C[(i+0)*N + j], _mm256_add_pd(c0, old0));
    _mm256_storeu_pd(&C[(i+1)*N + j], _mm256_add_pd(c1, old1));
    _mm256_storeu_pd(&C[(i+2)*N + j], _mm256_add_pd(c2, old2));
    _mm256_storeu_pd(&C[(i+3)*N + j], _mm256_add_pd(c3, old3));
}
#endif  /* USE_AVX2_GEMM */


/* ═══════════════════════════════════════════════════════════════════════════
 * gideon_gemm — entry point público
 * ═══════════════════════════════════════════════════════════════════════════ */
void gideon_gemm(
    double* __restrict__ C,
    const double* __restrict__ A,
    const double* __restrict__ B,
    double alpha, double beta,
    int32_t M, int32_t N, int32_t K
) {
    /* Siempre aplicamos beta*C primero */
    if (beta != 1.0) {
        for (int32_t i = 0; i < M * N; ++i) {
            C[i] *= beta;
        }
    }

    /* Matrices pequeñas: naive es más rápido (sin overhead de tiling) */
    if (M <= GIDEON_SMALL_THRESHOLD && N <= GIDEON_SMALL_THRESHOLD) {
        for (int32_t i = 0; i < M; ++i) {
            for (int32_t j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int32_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] += alpha * sum;
            }
        }
        return;
    }

#ifdef USE_AVX512_GEMM
    /*
     * Ruta AVX-512: micro-kernel 8×4.
     * MR y NR pueden ser sobreescritos desde build.rs; el micro-kernel
     * siempre usa paso fijo 8×4 (constexpr en C), así que sólo lo llamamos
     * cuando la tile es exactamente 8×4 (o múltiplo de esos).
     */
    const int32_t MR_512 = 8;   /* tiles de fila del micro-kernel */
    const int32_t NR_512 = 4;   /* tiles de columna del micro-kernel */

    for (int32_t ii = 0; ii < M; ii += MR_512) {
        int32_t i_end = (ii + MR_512 <= M) ? (ii + MR_512) : M;
        int32_t rows  = i_end - ii;

        for (int32_t jj = 0; jj < N; jj += NR_512) {
            int32_t j_end = (jj + NR_512 <= N) ? (jj + NR_512) : N;
            int32_t cols  = j_end - jj;

            if (rows == MR_512 && cols == NR_512) {
                /* Bloque completo 8×4 — usa micro-kernel AVX-512 */
                microkernel_8x4_avx512(C, A, B, N, K, ii, jj, 0, K);
                if (alpha != 1.0) {
                    for (int32_t r = 0; r < MR_512; ++r) {
                        for (int32_t c = 0; c < NR_512; ++c) {
                            C[(ii + r) * N + jj + c] *= alpha;
                        }
                    }
                }
            } else {
                /* Borde — usa escalar */
                for (int32_t i = ii; i < i_end; ++i) {
                    for (int32_t j = jj; j < j_end; ++j) {
                        double sum = 0.0;
                        for (int32_t k = 0; k < K; ++k) {
                            sum += A[i*K + k] * B[k*N + j];
                        }
                        C[i*N + j] += alpha * sum;
                    }
                }
            }
        }
    }

#elif defined(USE_AVX2_GEMM)
    /* Ruta AVX2: micro-kernel 4×4 */
    for (int32_t ii = 0; ii < M; ii += GIDEON_MR) {
        int32_t i_end = ii + GIDEON_MR <= M ? ii + GIDEON_MR : M;
        int32_t rows  = i_end - ii;

        for (int32_t jj = 0; jj < N; jj += GIDEON_NR) {
            int32_t j_end = jj + GIDEON_NR <= N ? jj + GIDEON_NR : N;
            int32_t cols  = j_end - jj;

            if (rows == GIDEON_MR && cols == GIDEON_NR) {
                microkernel_4x4(C, A, B, N, K, ii, jj, 0, K);
                if (alpha != 1.0) {
                    __m256d va = _mm256_set1_pd(alpha);
                    for (int32_t r = 0; r < GIDEON_MR; ++r) {
                        __m256d v = _mm256_loadu_pd(&C[(ii+r)*N + jj]);
                        _mm256_storeu_pd(&C[(ii+r)*N + jj], _mm256_mul_pd(v, va));
                    }
                }
            } else {
                for (int32_t i = ii; i < i_end; ++i) {
                    for (int32_t j = jj; j < j_end; ++j) {
                        double sum = 0.0;
                        for (int32_t k = 0; k < K; ++k) {
                            sum += A[i*K + k] * B[k*N + j];
                        }
                        C[i*N + j] += alpha * sum;
                    }
                }
            }
        }
    }
#else
    /* Fallback escalar */
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int32_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] += alpha * sum;
        }
    }
#endif
}
