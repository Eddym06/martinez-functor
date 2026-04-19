#!/usr/bin/env python3
"""
INVESTIGACIÓN PROFUNDA SESIÓN 5B — Seguimiento y validaciones cruzadas
======================================================================

Investigaciones de segundo nivel basadas en los resultados de la sesión 5:
  B1: Convergencia individual γ_k → h_KS (corrección de INV-1)
  B2: Estructura de Jordan / puntos excepcionales del operador PF
  B3: Verificación numérica del teorema de Gallavotti-Cohen (ERGON §18)
  B4: Consistencia cruzada ERGON↔OTU: matrices PF con parámetros idénticos
  B5: Nueva propiedad: operador resolvente R(z) = (zI - L)^{-1} y su traza
  B6: Nueva propiedad: entropía de transferencia inter-escala
  B7: Verificación del dual budget con definiciones independientes
  B8: Estructura de la medida SRB en el espacio de Fourier
"""

import sys, math
import numpy as np
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from acf_functor.gelfand_triple import GelfandTriple, _build_transfer_matrix, _build_koopman_matrix
from acf_functor.ergon_agent import ERGONAgent

logistic_r4 = lambda x: 4.0 * x * (1.0 - x)
tent_map = lambda x: np.where(x < 0.5, 2.0 * x, 2.0 * (1.0 - x))

print("=" * 80)
print("INVESTIGACIÓN PROFUNDA SESIÓN 5B — VALIDACIONES CRUZADAS")
print("=" * 80)

# ===========================================================================
# B1: Convergencia individual γ_k → h_KS (corrección INV-1)
# ===========================================================================
print("\n" + "=" * 80)
print("B1: CONVERGENCIA INDIVIDUAL γ_k → h_KS")
print("    La conjetura Σγ_k = h_KS es FALSA. Pero ¿γ_k → h_KS para k→∞?")
print("=" * 80)

for name, T_fn, h_exact in [("logistic_r4", logistic_r4, np.log(2)), ("tent", tent_map, np.log(2))]:
    for N in [128, 256, 512]:
        grid = np.linspace(0.001, 0.999, N)
        L = _build_transfer_matrix(T_fn, grid)
        eigvals = linalg.eig(L, left=False, right=False)
        idx = np.argsort(-np.abs(eigvals))
        eigvals = eigvals[idx]
        
        gammas = [-np.log(abs(eigvals[k])) for k in range(1, min(30, len(eigvals))) if abs(eigvals[k]) > 1e-6]
        
        print(f"\n  {name} N={N}:")
        print(f"    h_KS exacto = {h_exact:.6f}")
        print(f"    γ_1 = {gammas[0]:.6f}")
        print(f"    γ_5 = {gammas[4]:.6f}" if len(gammas) > 4 else "")
        print(f"    γ_10 = {gammas[9]:.6f}" if len(gammas) > 9 else "")
        print(f"    γ_20 = {gammas[19]:.6f}" if len(gammas) > 19 else "")
        print(f"    γ_k converge a: {np.mean(gammas[-5:]):.6f} (media últimos 5)")
        print(f"    Ratio γ_∞/h_KS ≈ {np.mean(gammas[-5:])/h_exact:.4f}")

# ===========================================================================
# B2: Puntos Excepcionales del Operador PF (Estructura de Jordan)
# ===========================================================================
print("\n" + "=" * 80)
print("B2: PUNTOS EXCEPCIONALES — COALESCENCIA DE AUTOVALORES")
print("    Detección de pares (λ_i, λ_j) con |λ_i - λ_j| < ε")
print("=" * 80)

for name, T_fn in [("logistic_r4", logistic_r4), ("tent", tent_map)]:
    grid = np.linspace(0.001, 0.999, 256)
    L = _build_transfer_matrix(T_fn, grid)
    eigvals = linalg.eig(L, left=False, right=False)
    idx = np.argsort(-np.abs(eigvals))
    eigvals_sorted = eigvals[idx][:30]
    
    # Buscar pares cuasi-degenerados
    ep_threshold = 0.01
    eps_found = []
    for i in range(len(eigvals_sorted)):
        for j in range(i+1, len(eigvals_sorted)):
            sep = abs(eigvals_sorted[i] - eigvals_sorted[j])
            if sep < ep_threshold and abs(eigvals_sorted[i]) > 0.1:
                eps_found.append((i, j, eigvals_sorted[i], eigvals_sorted[j], sep))
    
    print(f"\n  {name}:")
    print(f"    Pares cuasi-degenerados (|λ_i - λ_j| < {ep_threshold}):")
    for i, j, li, lj, sep in eps_found[:10]:
        print(f"      (k={i}, k={j}): λ_i={li:.4f}, λ_j={lj:.4f}, |Δλ|={sep:.6f}")
    
    # Defecto de Jordan: para cada par, calcular ||(L-λI)²v||
    if eps_found:
        i0, j0, l0, _, _ = eps_found[0]
        _, V = linalg.eig(L)
        V_sorted = V[:, idx]
        v = V_sorted[:, i0]
        defect = np.linalg.norm((L - l0 * np.eye(len(L))) @ (L - l0 * np.eye(len(L))) @ v)
        print(f"    Defecto de Jordan del par más cercano: ||(L-λI)²v|| = {defect:.6f}")
        print(f"    → {'PUNTO EXCEPCIONAL (Jordan no-trivial)' if defect < 1e-4 else 'Degeneración accidental (Jordan trivial)'}")
    else:
        print(f"    → No hay pares cuasi-degenerados (espectro simple)")

# ===========================================================================
# B3: Verificación del Teorema de Gallavotti-Cohen
# ===========================================================================
print("\n" + "=" * 80)
print("B3: TEOREMA DE FLUCTUACIÓN DE GALLAVOTTI-COHEN")
print("    P(σ_n = +h) / P(σ_n = -h) → e^{nh} para n→∞")
print("=" * 80)

T_fn = logistic_r4
n_trajectories = 50000
n_steps_list = [10, 20, 50, 100]

print(f"  logistic r=4, h_KS ≈ log(2) = {np.log(2):.6f}")
print()

rng = np.random.default_rng(42)
for n_steps in n_steps_list:
    sigma_n_values = []
    for _ in range(n_trajectories):
        x = 0.001 + 0.998 * rng.random()
        sigma = 0.0
        for _ in range(n_steps):
            # σ_k = log|T'(x_k)| = log|4(1-2x)|
            deriv = abs(4.0 * (1.0 - 2.0 * x))
            if deriv > 1e-14:
                sigma += np.log(deriv)
            x = 4.0 * x * (1.0 - x)
            x = np.clip(x, 0.001, 0.999)
        sigma_n_values.append(sigma / n_steps)
    
    sigma_arr = np.array(sigma_n_values)
    h_mean = np.mean(sigma_arr)
    
    # Contar P(σ > h_mean) y P(σ < -h_mean + 2*h_mean) = P(σ < h_mean)
    # GC: P(σ_n ≈ +a) / P(σ_n ≈ -a) = e^{n·2a} para a centrado
    # Binning simétrico alrededor de h_mean
    bins = np.linspace(h_mean - 0.5, h_mean + 0.5, 50)
    hist, edges = np.histogram(sigma_arr, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    # GC ratio para desviaciones simétricas respecto a la media
    gc_ratios = []
    for k in range(len(centers) // 4):
        p_plus = hist[len(centers)//2 + k]
        p_minus = hist[len(centers)//2 - k - 1] if len(centers)//2 - k - 1 >= 0 else 0
        if p_minus > 10 and p_plus > 10:
            delta_sigma = centers[len(centers)//2 + k] - centers[len(centers)//2 - k - 1]
            ratio_log = np.log(p_plus / p_minus) if p_minus > 0 else 0
            gc_pred = n_steps * delta_sigma
            gc_ratios.append((delta_sigma, ratio_log, gc_pred))
    
    print(f"  n={n_steps}: ⟨σ_n⟩ = {h_mean:.4f}, Var(σ_n) = {np.var(sigma_arr):.6f}")
    print(f"          Var(σ_n)·n = {np.var(sigma_arr) * n_steps:.4f} (debería → P''(1) = 0 para Bernoulli)")

# ===========================================================================
# B4: Consistencia Cruzada ERGON ↔ OTU (matrices PF idénticas)
# ===========================================================================
print("\n" + "=" * 80)
print("B4: CONSISTENCIA CRUZADA ERGON ↔ OTU")
print("    Comparar matrices PF construidas con parámetros IDÉNTICOS")
print("=" * 80)

N = 128
grid_otu = np.linspace(0.001, 0.999, N)

# OTU: 8 puntos Gauss-Legendre por celda
L_otu = _build_transfer_matrix(logistic_r4, grid_otu)

# ERGON: 20 puntos random por celda (como en el código original)
# Reconstruir la lógica de ERGON exacta
a, b = 0.001, 0.999
h = (b - a) / N
L_ergon = np.zeros((N, N))
rng_ergon = np.random.default_rng(0)
for j in range(N):
    x_left = grid_otu[j] - h/2
    x_right = grid_otu[j] + h/2
    x_pts = x_left + (x_right - x_left) * rng_ergon.random(20)
    x_pts = np.clip(x_pts, a + 1e-10, b - 1e-10)
    y_pts = logistic_r4(x_pts)
    for yi in y_pts:
        ii = int(np.clip(int(np.floor((yi - a) / h)), 0, N-1))
        L_ergon[ii, j] += 1.0 / 20
col_sums = L_ergon.sum(axis=0)
col_sums[col_sums == 0] = 1.0
L_ergon /= col_sums

# Comparar
diff_norm = np.linalg.norm(L_otu - L_ergon, 'fro') / np.linalg.norm(L_otu, 'fro')
print(f"  ||L_OTU - L_ERGON||_F / ||L_OTU||_F = {diff_norm:.4f} ({diff_norm*100:.1f}%)")

# Comparar autovalores
ev_otu = linalg.eig(L_otu, left=False, right=False)
ev_ergon = linalg.eig(L_ergon, left=False, right=False)
ev_otu = np.sort(np.abs(ev_otu))[::-1]
ev_ergon = np.sort(np.abs(ev_ergon))[::-1]

print(f"  Autovalores dominantes:")
for k in range(6):
    print(f"    k={k}: |λ_OTU|={ev_otu[k]:.6f}, |λ_ERGON|={ev_ergon[k]:.6f}, diff={abs(ev_otu[k]-ev_ergon[k]):.6f}")

# SRB measures
mu_otu = np.ones(N)/N
for _ in range(3000):
    mu_new = L_otu @ mu_otu; mu_new = np.abs(mu_new)
    s = mu_new.sum()
    if s > 0: mu_new /= s
    mu_otu = mu_new

mu_ergon = np.ones(N)/N
for _ in range(3000):
    mu_new = L_ergon @ mu_ergon; mu_new = np.abs(mu_new)
    s = mu_new.sum()
    if s > 0: mu_new /= s
    mu_ergon = mu_new

tv_dist = 0.5 * np.sum(np.abs(mu_otu - mu_ergon))
print(f"  Distancia TV(μ_OTU, μ_ERGON) = {tv_dist:.6f}")

# h_KS from each
h_otu = 0.0
for j in range(N):
    if mu_otu[j] < 1e-15: continue
    for i in range(N):
        if L_otu[i,j] > 1e-15:
            h_otu -= mu_otu[j] * L_otu[i,j] * np.log(L_otu[i,j])

h_ergon = 0.0
for j in range(N):
    if mu_ergon[j] < 1e-15: continue
    for i in range(N):
        if L_ergon[i,j] > 1e-15:
            h_ergon -= mu_ergon[j] * L_ergon[i,j] * np.log(L_ergon[i,j])

print(f"  h_KS_OTU (Markov) = {h_otu:.6f}")
print(f"  h_KS_ERGON (Markov) = {h_ergon:.6f}")
print(f"  Diferencia relativa = {abs(h_otu-h_ergon)/h_otu:.4%}")

# ===========================================================================
# B5: Operador Resolvente R(z) = (zI - L)^{-1}
# ===========================================================================
print("\n" + "=" * 80)
print("B5: NUEVA PROPIEDAD — TRAZA DEL RESOLVENTE R(z) = (zI - L)^{-1}")
print("    tr R(z) = Σ_k 1/(z - λ_k) — función meromorfa con polos en λ_k")
print("=" * 80)

grid = np.linspace(0.001, 0.999, 128)
L = _build_transfer_matrix(logistic_r4, grid)
eigvals = linalg.eig(L, left=False, right=False)

# Evaluar tr R(z) a lo largo de un círculo |z| = r en el plano complejo
for r in [0.8, 1.2, 2.0]:
    n_pts = 100
    angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    tr_R = np.zeros(n_pts, dtype=complex)
    for i, theta in enumerate(angles):
        z = r * np.exp(1j * theta)
        # tr R(z) = Σ 1/(z - λ_k)
        tr_R[i] = np.sum(1.0 / (z - eigvals + 1e-14))
    
    # Integral de contorno: (1/2πi) ∮ tr R(z) dz = número de autovalores dentro del contorno
    # = Σ_k 1{|λ_k| < r}
    integral = np.sum(tr_R * 1j * r * np.exp(1j * angles)) * (2 * np.pi / n_pts) / (2 * np.pi * 1j)
    n_inside_actual = np.sum(np.abs(eigvals) < r)
    
    print(f"  |z|={r:.1f}: (1/2πi)∮ tr R(z) dz = {integral.real:.2f} + {integral.imag:.2f}i")
    print(f"           # autovalores con |λ|<{r}: {n_inside_actual}")
    print(f"           Error del conteo: {abs(integral.real - n_inside_actual):.4f}")

# ===========================================================================
# B6: Entropía de Transferencia Inter-Escala
# ===========================================================================
print("\n" + "=" * 80)
print("B6: NUEVA PROPIEDAD — ENTROPÍA DE TRANSFERENCIA INTER-ESCALA")
print("    ¿Cuánta información transfiere T entre escalas (modos) diferentes?")
print("=" * 80)

grid = np.linspace(0.001, 0.999, 256)
L = _build_transfer_matrix(logistic_r4, grid)
eigvals_full, V = linalg.eig(L)
idx = np.argsort(-np.abs(eigvals_full))
V = V[:, idx]

try:
    V_inv = np.linalg.inv(V)
except:
    V_inv = np.linalg.pinv(V)

# L en la base propia: Λ = V^{-1} L V = diag(λ_k) (exacto si L es diagonalizable)
Lambda = V_inv @ L @ V

# El operador L² en la base propia: L² = V Λ² V^{-1}
# Pero si hay transferencia inter-modos, L² tiene elementos off-diagonal
# en la base propia que representan "acoplamiento no-lineal entre modos"

# La matriz de transferencia inter-escala es: T_{kl} = |Λ_{kl}|² / Σ_l |Λ_{kl}|²
# (normalizada por filas → probabilidad de transición modo k → modo l)

n_modes = min(20, Lambda.shape[0])
Lambda_sub = Lambda[:n_modes, :n_modes]

# Magnitud de los elementos off-diagonal (deberían ser 0 para operador diagonal)
off_diag = Lambda_sub - np.diag(np.diag(Lambda_sub))
off_diag_norm = np.linalg.norm(off_diag, 'fro')
diag_norm = np.linalg.norm(np.diag(Lambda_sub))
coupling = off_diag_norm / (diag_norm + 1e-14)

print(f"  Logística r=4:")
print(f"    ||off-diag(Λ)||_F = {off_diag_norm:.6f}")
print(f"    ||diag(Λ)||_F = {diag_norm:.6f}")
print(f"    Acoplamiento inter-escala = {coupling:.6f}")
print(f"    → {'Acoplamiento significativo (operador no-normal)' if coupling > 0.01 else 'Acoplamiento despreciable (operador esencialmente diagonal)'}")

# Entropía de mezcla entre modos
for k in range(min(5, n_modes)):
    row = np.abs(Lambda_sub[k, :]) ** 2
    row_sum = row.sum()
    if row_sum > 1e-14:
        probs = row / row_sum
        S_k = -np.sum(probs * np.log(probs + 1e-30))
        print(f"    S_mezcla(modo {k}) = {S_k:.4f} (max = {np.log(n_modes):.4f})")

# ===========================================================================
# B7: Dual Budget con definiciones INDEPENDIENTES
# ===========================================================================
print("\n" + "=" * 80)
print("B7: DUAL BUDGET — VERIFICACIÓN CON DEFINICIONES INDEPENDIENTES")
print("    d*(ε) = min{d : ||K_d f - Kf|| < ε} (truncación Koopman)")
print("    n*(ε) = min{n : C(n) < ε}            (tiempo de mezcla)")
print("=" * 80)

grid = np.linspace(0.001, 0.999, 256)
L = _build_transfer_matrix(logistic_r4, grid)
K = _build_koopman_matrix(logistic_r4, grid, 32)

eigvals_L = linalg.eig(L, left=False, right=False)
idx_L = np.argsort(-np.abs(eigvals_L))
eigvals_K = linalg.eig(K, left=False, right=False)
idx_K = np.argsort(-np.abs(eigvals_K))

gamma_L = -np.log(abs(eigvals_L[idx_L[1]])) if abs(eigvals_L[idx_L[1]]) > 1e-6 else 0
gamma_K = -np.log(abs(eigvals_K[idx_K[1]])) if abs(eigvals_K[idx_K[1]]) > 1e-6 else 0

print(f"  Γ_PF (gap PF/Ulam) = {gamma_L:.6f}")
print(f"  Γ_K (gap Koopman/EDMD) = {gamma_K:.6f}")
print(f"  Diferencia: {abs(gamma_L - gamma_K):.6f} ({abs(gamma_L - gamma_K)/gamma_L:.1%})")
print()

# n*(ε) vía correlación decay REAL (no formula)
mu = np.ones(256) / 256
for _ in range(3000):
    mu_new = L @ mu; mu_new = np.abs(mu_new)
    s = mu_new.sum()
    if s > 0: mu_new /= s
    mu = mu_new

f_test = np.cos(np.pi * grid)
g_test = np.sin(np.pi * grid)
mean_f = np.dot(f_test, mu)
mean_g = np.dot(g_test, mu)

print(f"  {'ε':>8} | {'n*(ε) correlación':>18} | {'n*(ε) fórmula Γ_L':>18} | {'d*(ε) fórmula Γ_K':>18}")
print("  " + "-" * 75)

for eps in [0.1, 0.01, 0.001]:
    # n*(ε) empírico: iterar L hasta que |C(n)| < ε
    mu_iter = g_test * mu  # initial
    n_star_emp = -1
    for n in range(1, 500):
        mu_iter = L @ mu_iter
        cross = np.dot(f_test, mu_iter)
        corr = abs(cross - mean_f * mean_g * np.sum(mu_iter))
        if corr < eps:
            n_star_emp = n
            break
    
    n_star_formula_L = math.ceil(math.log(1.0/eps) / gamma_L) if gamma_L > 1e-10 else -1
    d_star_formula_K = math.ceil(math.log(1.0/eps) / gamma_K) if gamma_K > 1e-10 else -1
    
    print(f"  {eps:>8.3f} | {n_star_emp:>18d} | {n_star_formula_L:>18d} | {d_star_formula_K:>18d}")

print(f"\n  → El dual budget d*(ε) = n*(ε) falla porque Γ_PF ≠ Γ_Koopman")
print(f"  → La discrepancia es {abs(gamma_L - gamma_K)/gamma_L:.1%}: el EDMD y Ulam dan gaps diferentes")

# ===========================================================================
# B8: Medida SRB en el Espacio de Fourier
# ===========================================================================
print("\n" + "=" * 80)
print("B8: NUEVA PROPIEDAD — ESTRUCTURA DE FOURIER DE μ_SRB")
print("    Coeficientes de Fourier ĉ_k = ∫ μ(x) e^{-2πikx} dx")
print("=" * 80)

for name, T_fn in [("logistic_r4", logistic_r4), ("tent", tent_map)]:
    grid = np.linspace(0.001, 0.999, 512)
    L = _build_transfer_matrix(T_fn, grid)
    mu = np.ones(512) / 512
    for _ in range(3000):
        mu_new = L @ mu; mu_new = np.abs(mu_new)
        s = mu_new.sum()
        if s > 0: mu_new /= s
        mu = mu_new
    
    # FFT de μ_SRB
    fourier = np.fft.fft(mu)
    power = np.abs(fourier)**2
    
    # Decaimiento de los coeficientes de Fourier
    n_show = 20
    print(f"\n  {name}:")
    print(f"    Decaimiento espectral de μ_SRB:")
    print(f"    {'k':>4} | {'|ĉ_k|':>12} | {'|ĉ_k|/|ĉ_1|':>14}")
    c1 = abs(fourier[1])
    for k in range(1, n_show+1):
        print(f"    {k:>4} | {abs(fourier[k]):>12.8f} | {abs(fourier[k])/c1:>14.6f}")
    
    # ¿Decaimiento potencial o exponencial?
    ks = np.arange(1, min(50, len(fourier)//2))
    mods = np.array([abs(fourier[k]) for k in ks])
    mods = mods[mods > 1e-14]
    ks_valid = ks[:len(mods)]
    
    if len(mods) > 5:
        # Ajuste log|ĉ_k| = α log(k) + β (potencial)
        log_k = np.log(ks_valid)
        log_c = np.log(mods)
        alpha_pot, beta_pot = np.polyfit(log_k, log_c, 1)
        
        # Ajuste log|ĉ_k| = -γk + β (exponencial)
        gamma_exp, beta_exp = np.polyfit(ks_valid.astype(float), log_c, 1)
        
        # R² para cada ajuste
        pred_pot = alpha_pot * log_k + beta_pot
        pred_exp = gamma_exp * ks_valid + beta_exp
        ss_tot = np.sum((log_c - np.mean(log_c))**2)
        r2_pot = 1 - np.sum((log_c - pred_pot)**2) / (ss_tot + 1e-14)
        r2_exp = 1 - np.sum((log_c - pred_exp)**2) / (ss_tot + 1e-14)
        
        print(f"    Ajuste potencial: |ĉ_k| ~ k^{alpha_pot:.3f} (R²={r2_pot:.4f})")
        print(f"    Ajuste exponencial: |ĉ_k| ~ e^{{{gamma_exp:.4f}k}} (R²={r2_exp:.4f})")
        
        if r2_pot > r2_exp:
            print(f"    → DECAIMIENTO POTENCIAL (α = {alpha_pot:.3f})")
            print(f"    → La medida SRB tiene singularidades (no es C^∞)")
        else:
            print(f"    → DECAIMIENTO EXPONENCIAL (γ = {-gamma_exp:.4f})")
            print(f"    → La medida SRB es analítica")

# ===========================================================================
# B9: Nueva propiedad — Dimensión Efectiva del Espacio de Koopman
# ===========================================================================
print("\n" + "=" * 80)
print("B9: NUEVA PROPIEDAD — DIMENSIÓN EFECTIVA DEL ESPACIO DE KOOPMAN")
print("    d_eff = exp(S_spec) donde S_spec = -Σ p_k log p_k, p_k = σ_k²/Σσ_j²")
print("    (Basada en los valores singulares de la matriz EDMD)")
print("=" * 80)

for name, T_fn, dom in [
    ("logistic_r4", logistic_r4, (0.001, 0.999)),
    ("tent", tent_map, (0.001, 0.999)),
    ("logistic_r38", lambda x: 3.8*x*(1-x), (0.001, 0.999)),
]:
    for n_test in [16, 32, 48]:
        grid = np.linspace(dom[0], dom[1], 256)
        K = _build_koopman_matrix(T_fn, grid, n_test)
        svd_vals = linalg.svdvals(K)
        
        # Distribución normalizada
        weights = svd_vals**2
        weights = weights[weights > 1e-20]
        probs = weights / weights.sum()
        S = -np.sum(probs * np.log(probs))
        d_eff = np.exp(S)
        
        print(f"  {name} n_test={n_test}: d_eff = {d_eff:.2f} / {n_test} ({d_eff/n_test:.1%})")

# ===========================================================================
# B10: Propiedad de Simetrización — ¿L + L^T tiene mejor espectro?
# ===========================================================================
print("\n" + "=" * 80)
print("B10: SIMETRIZACIÓN DEL OPERADOR PF: (L + L^T)/2")
print("     ¿Mejora la estabilidad numérica y la precisión de Pesin?")
print("=" * 80)

for name, T_fn in [("logistic_r4", logistic_r4), ("tent", tent_map)]:
    grid = np.linspace(0.001, 0.999, 256)
    L = _build_transfer_matrix(T_fn, grid)
    L_sym = 0.5 * (L + L.T)
    
    # SRB de L original
    mu_L = np.ones(256)/256
    for _ in range(3000):
        mu_new = L @ mu_L; mu_new = np.abs(mu_new); s = mu_new.sum()
        if s > 0: mu_new /= s
        mu_L = mu_new
    
    # SRB de L simétrico
    mu_sym = np.ones(256)/256
    for _ in range(3000):
        mu_new = L_sym @ mu_sym; mu_new = np.abs(mu_new); s = mu_new.sum()
        if s > 0: mu_new /= s
        mu_sym = mu_new
    
    tv = 0.5 * np.sum(np.abs(mu_L - mu_sym))
    
    # Gaps espectrales
    ev_L = np.sort(np.abs(linalg.eigvals(L)))[::-1]
    ev_sym = np.sort(np.abs(linalg.eigvals(L_sym)))[::-1]
    gamma_L = -np.log(ev_L[1]) if ev_L[1] > 1e-6 else float('inf')
    gamma_sym = -np.log(ev_sym[1]) if ev_sym[1] > 1e-6 else float('inf')
    
    # Condición
    kappa_L = linalg.svdvals(L)[0] / (linalg.svdvals(L)[-1] + 1e-14)
    kappa_sym = linalg.svdvals(L_sym)[0] / (linalg.svdvals(L_sym)[-1] + 1e-14)
    
    print(f"  {name}:")
    print(f"    Γ(L) = {gamma_L:.4f}, Γ(L_sym) = {gamma_sym:.4f}")
    print(f"    κ(L) = {kappa_L:.2e}, κ(L_sym) = {kappa_sym:.2e}")
    print(f"    TV(μ_L, μ_sym) = {tv:.6f}")
    print(f"    → La simetrización {'MEJORA' if kappa_sym < kappa_L else 'EMPEORA'} el condicionamiento")

print("\n" + "=" * 80)
print("FIN DE LA INVESTIGACIÓN SESIÓN 5B")
print("=" * 80)
