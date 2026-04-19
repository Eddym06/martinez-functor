/*
 * cpu_kernels.h — Cabeceras para backends numéricos Gideon
 *
 * Diseño:
 *   - Todos los kernels trabajan sobre arrays f64 (double).
 *   - El llamante Rust garantiza que los punteros son válidos y no se solapan.
 *   - Los kernels son C puro: sin dependencias externas.
 *
 * Convención:
 *   - out, x: arrays de salida/entrada de `n` elementos.
 *   - w, b:   escalares de peso y sesgo (para FMA y chain).
 *   - Retornan void; errores se comunican por valor de salida conocido (-inf).
 */

#ifndef GIDEON_CPU_KERNELS_H
#define GIDEON_CPU_KERNELS_H

#include <stddef.h>  /* size_t */
#include <stdint.h>  /* int32_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ── FMA escalar (baseline — siempre disponible) ──────────────────────────── */
void gideon_fma_scalar(
    double* __restrict__ out,
    const double* __restrict__ x,
    double w,
    double b,
    size_t n
);

/* ── FMA vectorial — el compilador elige AVX2 o AVX-512 según flags ──────── */
void gideon_fma_vector(
    double* __restrict__ out,
    const double* __restrict__ x,
    double w,
    double b,
    size_t n
);

/* ── Cadena FMA completa: aplica N pasos (w_i, b_i) secuencialmente ──────── */
/*    x_out[i] = b_N + w_N*(b_{N-1} + w_{N-1}*(... + w_1*x[i] + b_1 ...))  */
void gideon_fma_chain(
    double* __restrict__ out,
    const double* __restrict__ x,
    const double* __restrict__ weights, /* longitud n_fma */
    const double* __restrict__ biases,  /* longitud n_fma */
    size_t n_elements,
    size_t n_fma
);

/* ── Fold afín: colapsa toda la cadena a W_total·x + B_total ──────────────── */
void gideon_fma_fold(
    double* __restrict__ out,
    const double* __restrict__ x,
    double W_total,
    double B_total,
    size_t n
);

/* ── GEMM: C = alpha * A @ B + beta * C  (row-major) ─────────────────────── */
void gideon_gemm(
    double* __restrict__ C,
    const double* __restrict__ A,
    const double* __restrict__ B,
    double alpha,
    double beta,
    int32_t M,   /* filas de A y C */
    int32_t N,   /* columnas de B y C */
    int32_t K    /* columnas de A, filas de B */
);

#ifdef __cplusplus
}
#endif

#endif /* GIDEON_CPU_KERNELS_H */
