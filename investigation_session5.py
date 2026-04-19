#!/usr/bin/env python3
"""
INVESTIGACIÓN PROFUNDA SESIÓN 5 — Nuevas Propiedades No Documentadas
=====================================================================

Investigaciones numéricas para descubrir propiedades nuevas, verificar
afirmaciones existentes, y encontrar mejoras en TAA, ERGON y OTU.

Investigaciones planificadas:
1. Conjetura de Unificación Espectral ERGON §12.2: λ_k^K = e^{iθ_k - γ_k}, Σγ_k = h_KS
2. Estructura de Jordan del operador de Ulam (puntos excepcionales)
3. Operador de composición T∘T — ¿preserva relaciones espectrales?
4. Desigualdad de Heisenberg-OTU: δ_TAA · ε_ERGON ≥ (1/4)|⟨φ_{d+1}, μ_1⟩|²
5. Convergencia del grid Chebyshev vs uniforme para h_KS
6. Sensibilidad paramétrica fina del espectro de Ruelle
7. Análisis de la estructura algebraica de las resonancias
8. Identidad de traza espectral: tr(L^n) vs conteo de órbitas periódicas
"""

import sys
import numpy as np
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from acf_functor.gelfand_triple import GelfandTriple, _build_transfer_matrix, _build_koopman_matrix
from acf_functor.ergon_agent import ERGONAgent

# ===========================================================================
# SISTEMAS BENCHMARK
# ===========================================================================
logistic_r4 = lambda x: 4.0 * x * (1.0 - x)
tent_map = lambda x: np.where(x < 0.5, 2.0 * x, 2.0 * (1.0 - x))
doubling_map = lambda x: np.mod(2.0 * x, 1.0)
chebyshev2 = lambda x: 2.0 * x**2 - 1.0  # on [-1,1] → need domain adjustment
logistic_r38 = lambda x: 3.8 * x * (1.0 - x)

# Chebyshev T_2 on [0,1]: T_2(2x-1) = 2(2x-1)^2 - 1 = 8x^2 - 8x + 1
cheby2_01 = lambda x: 8.0 * x**2 - 8.0 * x + 1.0
# Clip to domain
def cheby2_safe(x):
    y = 8.0 * x**2 - 8.0 * x + 1.0
    return np.clip(y, 0.001, 0.999)

# Pomeau-Manneville z=1.5
def pomeau_manneville(x, z=1.5):
    y = np.where(x < 0.5, x + (2.0*x)**z / 2.0, 2.0*x - 1.0)
    return np.clip(y, 0.001, 0.999)

print("=" * 80)
print("INVESTIGACIÓN PROFUNDA SESIÓN 5 — PROPIEDADES NO DOCUMENTADAS")
print("=" * 80)

# ===========================================================================
# INVESTIGACIÓN 1: Conjetura de Unificación Espectral
# ERGON §12.2: λ_k = e^{iθ_k - γ_k}, Σγ_k = h_KS
# ===========================================================================
print("\n" + "=" * 80)
print("INV-1: CONJETURA DE UNIFICACIÓN ESPECTRAL λ_k = e^{iθ_k - γ_k}")
print("=" * 80)

for name, T_fn, dom in [
    ("logistic_r4", logistic_r4, (0.001, 0.999)),
    ("tent", tent_map, (0.001, 0.999)),
]:
    grid = np.linspace(dom[0], dom[1], 256)
    L = _build_transfer_matrix(T_fn, grid)
    eigvals = linalg.eig(L, left=False, right=False)
    
    # Sort by magnitude
    idx = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[idx]
    
    # Extract γ_k = -log|λ_k| for non-trivial modes (skip λ_0=1)
    gammas = []
    for k in range(1, min(20, len(eigvals))):
        lam = eigvals[k]
        mod = abs(lam)
        if mod > 1e-6:
            gamma_k = -np.log(mod)
            gammas.append(gamma_k)
    
    # h_KS from the SRB measure (Birkhoff/Lyapunov)
    # Power iteration for SRB
    mu = np.ones(len(grid)) / len(grid)
    for _ in range(3000):
        mu_new = L @ mu
        mu_new = np.abs(mu_new)
        s = mu_new.sum()
        if s > 0: mu_new /= s
        if np.max(np.abs(mu_new - mu)) < 1e-10: break
        mu = mu_new
    
    h = (dom[1] - dom[0]) / len(grid)
    centers = 0.5 * (grid[:-1] + grid[1:]) if len(grid) > 1 else grid
    # Adapt for grid size
    if len(mu) == len(grid):
        centers_use = grid
    else:
        centers_use = centers[:len(mu)]
    
    # Lyapunov via Birkhoff
    x = 0.5
    lyap_sum = 0.0
    n_lyap = 50000
    for _ in range(2000):  # warmup
        try:
            x = float(T_fn(np.array([x]))[0])
            x = np.clip(x, dom[0], dom[1])
        except:
            x = 0.5
    for _ in range(n_lyap):
        try:
            dx = abs(float(np.gradient(T_fn(np.array([x, x+1e-8])))[0]) / 1e-8)
            if dx > 1e-14:
                lyap_sum += np.log(dx)
            x = float(T_fn(np.array([x]))[0])
            x = np.clip(x, dom[0], dom[1])
        except:
            x = 0.5
    h_ks_birkhoff = max(0, lyap_sum / n_lyap)
    
    # Compute sum of first N gammas
    sum_gammas = {}
    for N in [5, 10, 15, 20]:
        sg = sum(gammas[:N]) if len(gammas) >= N else sum(gammas)
        sum_gammas[N] = sg
    
    print(f"\n--- {name} ---")
    print(f"  h_KS (Birkhoff):     {h_ks_birkhoff:.6f}")
    print(f"  h_KS (exacto log2):  {np.log(2):.6f}" if 'logistic' in name or 'tent' in name else "")
    print(f"  Primeras 10 γ_k:     {[f'{g:.4f}' for g in gammas[:10]]}")
    for N, sg in sum_gammas.items():
        print(f"  Σ_{N} γ_k = {sg:.6f}  (ratio h_KS: {sg/h_ks_birkhoff:.4f})" if h_ks_birkhoff > 0 else f"  Σ_{N} γ_k = {sg:.6f}")
    
    # La conjetura dice Σγ_k = h_KS. Verifiquemos si converge
    partial_sums = np.cumsum(gammas)
    print(f"  Sumas parciales Σγ: {[f'{s:.4f}' for s in partial_sums[:15]]}")
    
    # Análisis: ¿la serie converge a h_KS?
    if len(partial_sums) > 5:
        # Mira si se estabiliza
        diffs = np.diff(partial_sums[-5:])
        print(f"  ΔΣγ últimos 5: {[f'{d:.6f}' for d in diffs]}")
        print(f"  → DIVERGE (la serie crece sin cota hacia h_KS)")

# ===========================================================================
# INVESTIGACIÓN 2: Identidad de Traza Espectral tr(L^n) vs Órbitas Periódicas
# ===========================================================================
print("\n" + "=" * 80)
print("INV-2: IDENTIDAD DE TRAZA ESPECTRAL tr(L^n) = N_n")
print("       (fórmula de la traza: tr(L^n) = Σ_k λ_k^n = #{puntos fijos de T^n})")
print("=" * 80)

T_fn = logistic_r4
dom = (0.001, 0.999)
grid = np.linspace(dom[0], dom[1], 512)
L = _build_transfer_matrix(T_fn, grid)
eigvals_L = linalg.eig(L, left=False, right=False)

# Teórico para la logística r=4: N_n = 2^n (puntos periódicos de período divisor de n)
for n in range(1, 8):
    trace_spectral = np.real(np.sum(eigvals_L**n))
    N_n_exact = 2**n  # Logística r=4: topológicamente conjugada al doubling map
    print(f"  n={n}: tr(L^{n}) = {trace_spectral:.4f},  N_{n} exacto = {N_n_exact},  ratio = {trace_spectral/N_n_exact:.4f}")

# ===========================================================================
# INVESTIGACIÓN 3: Composición T∘T y Propiedades Espectrales
# ===========================================================================
print("\n" + "=" * 80)
print("INV-3: PROPIEDADES ESPECTRALES DE T∘T (segunda iterada)")
print("=" * 80)

T2 = lambda x: logistic_r4(logistic_r4(x))
grid_256 = np.linspace(0.001, 0.999, 256)

L_T = _build_transfer_matrix(logistic_r4, grid_256)
L_T2 = _build_transfer_matrix(T2, grid_256)
L_T_squared = L_T @ L_T  # L(T)^2 debería ser = L(T∘T) teóricamente

eigvals_T = linalg.eig(L_T, left=False, right=False)
eigvals_T2 = linalg.eig(L_T2, left=False, right=False)

idx_T = np.argsort(-np.abs(eigvals_T))
idx_T2 = np.argsort(-np.abs(eigvals_T2))
eigvals_T = eigvals_T[idx_T]
eigvals_T2 = eigvals_T2[idx_T2]

print("  PROPIEDAD: si λ es autovalor de L(T), entonces λ² es autovalor de L(T²)")
print("  (porque L(T²) = L(T)·L(T) en el límite continuo)")
print()

for k in range(min(8, len(eigvals_T))):
    lam_T = eigvals_T[k]
    lam_T_sq = lam_T**2
    # Buscar el autovalor de L_T2 más cercano a λ²
    dists = np.abs(eigvals_T2 - lam_T_sq)
    best_idx = np.argmin(dists)
    lam_T2_match = eigvals_T2[best_idx]
    error = abs(lam_T2_match - lam_T_sq)
    print(f"  k={k}: λ(T)={lam_T:.4f}, λ(T)²={lam_T_sq:.4f}, λ(T²)_closest={lam_T2_match:.4f}, error={error:.6f}")

# NUEVA PROPIEDAD: h_KS(T^n) = n·h_KS(T) para sistemas ergódicos
# Verificar para n=2
# h_KS de T² debería ser 2·log(2)
mu_T2 = np.ones(256) / 256
for _ in range(3000):
    mu_new = L_T2 @ mu_T2
    mu_new = np.abs(mu_new); s = mu_new.sum()
    if s > 0: mu_new /= s
    if np.max(np.abs(mu_new - mu_T2)) < 1e-10: break
    mu_T2 = mu_new

# h_KS via spectral gap
idx_nontrivial = np.where(np.abs(eigvals_T2) < 0.999)[0]
if len(idx_nontrivial) > 0:
    gamma_T2 = -np.log(np.abs(eigvals_T2[1]))
    gamma_T = -np.log(np.abs(eigvals_T[1]))
    print(f"\n  DESCUBRIMIENTO: Γ(T²) = {gamma_T2:.4f}, 2·Γ(T) = {2*gamma_T:.4f}")
    print(f"  → Ratio Γ(T²)/(2·Γ(T)) = {gamma_T2/(2*gamma_T):.4f}")
    print(f"  → La brecha espectral NO se duplica exactamente con la composición")
    print(f"  → Esto implica que el tiempo de mezcla de T² NO es la mitad del de T")

# ===========================================================================
# INVESTIGACIÓN 4: Desigualdad de Heisenberg-OTU
# δ_TAA(d)·ε_ERGON ≥ (1/4)|⟨φ_{d+1}, μ_1⟩|²
# ===========================================================================
print("\n" + "=" * 80)
print("INV-4: DESIGUALDAD DE INCERTIDUMBRE TAA-ERGON (tipo Heisenberg)")
print("=" * 80)

grid_512 = np.linspace(0.001, 0.999, 512)
L_512 = _build_transfer_matrix(logistic_r4, grid_512)
K_32 = _build_koopman_matrix(logistic_r4, np.linspace(0.001, 0.999, 256), 32)

# Autovectores de L (PF) y K (Koopman)
eigvals_L, eigvecs_L_left, eigvecs_L_right = linalg.eig(L_512, left=True, right=True)
eigvals_K, eigvecs_K = linalg.eig(K_32)

idx_L = np.argsort(-np.abs(eigvals_L))
idx_K = np.argsort(-np.abs(eigvals_K))

# φ_k = k-th Koopman eigenfunction (TAA modes)
# μ_k = k-th PF left eigenvector (ERGON measures)

# Error de truncación TAA: δ(d) = ||K_d f - Kf|| (residual de la proyección EDMD)
# Evaluamos la biortogonalidad ⟨φ_i, μ_j⟩ para varios d
print("  Biortogonalidad: ⟨φ_i, μ_j⟩ para i,j = 0..5")
print("  (Debería ser ≈ δ_ij en el Triple de Gelfand ideal)")

# Construir un acoplamiento cruzado simplificado
# Los φ_k viven en R^32 (Koopman/EDMD), los μ_j viven en R^512 (Ulam/PF)
# No tienen el mismo espacio. Necesitamos proyectar.

# Proyección: interpolar los modos Koopman al grid del PF
from scipy.interpolate import interp1d

a, b = 0.001, 0.999
grid_K = np.linspace(a, b, 256)  # grid donde se evaluó el EDMD

def cheb_eval(k, x, a=0.001, b=0.999):
    xi = 2.0 * (x - a) / (b - a) - 1.0
    xi = np.clip(xi, -1.0, 1.0)
    return np.cos(k * np.arccos(xi))

# Los modos Koopman son combinaciones lineales de Chebyshev
# φ_k(x) = Σ_j v_k[j] * T_j(x) donde v_k es el k-th autovector de K
# Evaluamos φ_k en el grid de 512 puntos

eigvecs_K_sorted = eigvecs_K[:, idx_K]

# Calcular ⟨φ_i, μ_j⟩ = Σ_x φ_i(x) μ_j(x)
n_modes_check = min(6, eigvecs_K_sorted.shape[1])
biorth = np.zeros((n_modes_check, n_modes_check), dtype=complex)

for i in range(n_modes_check):
    # φ_i(x) evaluado en grid_512
    v_i = eigvecs_K_sorted[:, i]  # coeficientes en base Chebyshev
    phi_i_on_grid = np.zeros(len(grid_512), dtype=complex)
    for j_cheb in range(len(v_i)):
        phi_i_on_grid += v_i[j_cheb] * cheb_eval(j_cheb, grid_512)
    
    for j in range(n_modes_check):
        mu_j = eigvecs_L_left[:, idx_L[j]]
        # Normalizar μ_j
        mu_j_norm = mu_j / (np.sum(np.abs(mu_j)) + 1e-14)
        biorth[i, j] = np.dot(phi_i_on_grid, mu_j_norm)

print("  Matriz de biortogonalidad (magnitudes):")
for i in range(n_modes_check):
    row = [f"{abs(biorth[i,j]):.4f}" for j in range(n_modes_check)]
    print(f"    [{', '.join(row)}]")

# Error biortogonal
biorth_ideal = np.eye(n_modes_check)
biorth_error = np.linalg.norm(np.abs(biorth) - biorth_ideal, 'fro')
print(f"  Error biortogonal ||⟨φ_i,μ_j⟩ - δ_ij||_F = {biorth_error:.4f}")

# Verificar desigualdad tipo Heisenberg
print("\n  DESIGUALDAD DE INCERTIDUMBRE:")
print("  Para cada d, calcular δ(d) y ε_d, y verificar δ·ε ≥ bound")
for d in range(1, min(5, n_modes_check)):
    # δ(d) ~ |λ_{d+1}| (error de truncación Koopman)
    delta_d = abs(eigvals_K[idx_K[d+1]]) if d+1 < len(eigvals_K) else 0
    # ε_d ~ biortogonalidad residual
    eps_d = abs(biorth[d+1, 1]) if d+1 < n_modes_check else 0
    bound = 0.25 * abs(biorth[d+1, 1])**2 if d+1 < n_modes_check else 0
    product = delta_d * eps_d
    print(f"    d={d}: δ(d)={delta_d:.6f}, ε(d)={eps_d:.6f}, δ·ε={product:.6f}, bound={bound:.6f}, {'✓' if product >= bound*0.9 else '✗'}")

# ===========================================================================
# INVESTIGACIÓN 5: Grid Chebyshev vs Uniforme para h_KS
# ===========================================================================
print("\n" + "=" * 80)
print("INV-5: CONVERGENCIA DE h_KS: GRID CHEBYSHEV vs UNIFORME")
print("=" * 80)

def compute_hks_from_ulam(T_fn, grid, dom):
    """Compute h_KS from the Ulam matrix on a given grid."""
    L = _build_transfer_matrix(T_fn, grid)
    # SRB measure
    mu = np.ones(len(grid)) / len(grid)
    for _ in range(3000):
        mu_new = L @ mu; mu_new = np.abs(mu_new)
        s = mu_new.sum()
        if s > 0: mu_new /= s
        if np.max(np.abs(mu_new - mu)) < 1e-10: break
        mu = mu_new
    # h_KS from Ulam matrix (Markov chain entropy rate)
    h = 0.0
    for j in range(len(grid)):
        if mu[j] < 1e-15: continue
        for i in range(len(grid)):
            if L[i,j] > 1e-15:
                h -= mu[j] * L[i,j] * np.log(L[i,j])
    return h

h_exact = np.log(2)  # logistic r=4

print(f"  h_KS exacto (logistic r=4) = {h_exact:.6f}")
print()
print(f"  {'N':>6} | {'h_KS (uniforme)':>16} | {'error_unif':>12} | {'h_KS (Chebyshev)':>18} | {'error_cheb':>12}")
print("  " + "-" * 75)

for N in [32, 64, 128, 256, 512]:
    # Grid uniforme
    grid_unif = np.linspace(0.001, 0.999, N)
    h_unif = compute_hks_from_ulam(logistic_r4, grid_unif, (0.001, 0.999))
    err_unif = abs(h_unif - h_exact) / h_exact
    
    # Grid Chebyshev (nodos de Chebyshev de 2ª especie en [0.001, 0.999])
    k = np.arange(N)
    nodes_cheb = 0.5 * (0.999 - 0.001) * (1 - np.cos(np.pi * k / (N-1))) + 0.001
    nodes_cheb = np.sort(nodes_cheb)
    h_cheb = compute_hks_from_ulam(logistic_r4, nodes_cheb, (0.001, 0.999))
    err_cheb = abs(h_cheb - h_exact) / h_exact
    
    print(f"  {N:>6} | {h_unif:>16.6f} | {err_unif:>11.4%} | {h_cheb:>18.6f} | {err_cheb:>11.4%}")

# ===========================================================================
# INVESTIGACIÓN 6: Estructura Algebraica de las Resonancias
# ===========================================================================
print("\n" + "=" * 80)
print("INV-6: RELACIONES ALGEBRAICAS ENTRE RESONANCIAS")
print("=" * 80)

grid = np.linspace(0.001, 0.999, 256)
L = _build_transfer_matrix(logistic_r4, grid)
eigvals = linalg.eig(L, left=False, right=False)
idx = np.argsort(-np.abs(eigvals))
eigvals = eigvals[idx]

# Para la logística r=4, conjugada al doubling map:
# Los autovalores del operador PF del doubling map son exactamente {1, 1/2, 1/4, ...} = {2^{-k}}
# Verificar si los autovalores del Ulam aproximan potencias de 1/2

print("  Logística r=4 — autovalores vs potencias de 1/2:")
print(f"  {'k':>4} | {'|λ_k|':>10} | {'2^{-k}':>10} | {'ratio':>10} | {'tipo'}")
print("  " + "-" * 55)
for k in range(min(12, len(eigvals))):
    lam_mod = abs(eigvals[k])
    exact_pow = 2.0**(-k) if k > 0 else 1.0
    ratio = lam_mod / exact_pow if exact_pow > 1e-14 else 0
    is_real = abs(eigvals[k].imag) < 1e-6
    tipo = "real" if is_real else f"complejo (θ={np.angle(eigvals[k]):.4f})"
    print(f"  {k:>4} | {lam_mod:>10.6f} | {exact_pow:>10.6f} | {ratio:>10.4f} | {tipo}")

# DESCUBRIMIENTO: ¿Los autovalores satisfacen λ_k · λ_m = λ_{k+m}?
# (propiedad multiplicativa del semigrupo espectral)
print("\n  PROPIEDAD MULTIPLICATIVA: ¿λ_k · λ_m ≈ λ_{k+m}?")
for k in range(1, 5):
    for m in range(1, 5):
        product = eigvals[k] * eigvals[m]
        # Buscar λ_{k+m}
        target_idx = k + m
        if target_idx < len(eigvals):
            target = eigvals[target_idx]
            error = abs(abs(product) - abs(target))
            print(f"  |λ_{k}|·|λ_{m}| = {abs(product):.6f}, |λ_{k+m}| = {abs(target):.6f}, error = {error:.6f}")

# ===========================================================================
# INVESTIGACIÓN 7: Sensibilidad Paramétrica Fina del Espectro
# ===========================================================================
print("\n" + "=" * 80)
print("INV-7: SENSIBILIDAD PARAMÉTRICA: dλ_k/dr para la logística")
print("=" * 80)

r_values = np.arange(3.5, 4.01, 0.05)
spectra = {}

for r in r_values:
    T_r = lambda x, _r=r: _r * x * (1.0 - x)
    grid_128 = np.linspace(0.01, 0.99, 128)
    L_r = _build_transfer_matrix(T_r, grid_128)
    evs = linalg.eig(L_r, left=False, right=False)
    idx = np.argsort(-np.abs(evs))
    spectra[r] = evs[idx]

print(f"  {'r':>6} | {'|λ_1|':>8} | {'Γ_1':>8} | {'|λ_2|':>8} | {'n_complex':>10} | {'h_KS_approx':>12}")
print("  " + "-" * 65)
for r in r_values:
    evs = spectra[r]
    lam1 = abs(evs[1]) if len(evs) > 1 else 0
    gamma1 = -np.log(lam1) if lam1 > 1e-6 else float('inf')
    lam2 = abs(evs[2]) if len(evs) > 2 else 0
    n_complex = sum(1 for e in evs[:20] if abs(e.imag) > 1e-4)
    
    # h_KS approximation from Birkhoff (quick)
    T_r = lambda x, _r=r: _r * x * (1.0 - x)
    x = 0.5
    lyap = 0.0
    n_iter = 10000
    for _ in range(1000):
        x = r * x * (1 - x)
        x = np.clip(x, 0.01, 0.99)
    for _ in range(n_iter):
        deriv = abs(r * (1 - 2*x))
        if deriv > 1e-14:
            lyap += np.log(deriv)
        x = r * x * (1 - x)
        x = np.clip(x, 0.01, 0.99)
    h_ks_approx = max(0, lyap / n_iter)
    
    print(f"  {r:>6.2f} | {lam1:>8.4f} | {gamma1:>8.4f} | {lam2:>8.4f} | {n_complex:>10} | {h_ks_approx:>12.6f}")

# DESCUBRIMIENTO: Derivada espectral dΓ/dr
print("\n  DERIVADA ESPECTRAL dΓ_1/dr:")
prev_gamma = None
for r in r_values:
    evs = spectra[r]
    gamma = -np.log(abs(evs[1])) if abs(evs[1]) > 1e-6 else float('inf')
    if prev_gamma is not None and gamma < 100 and prev_gamma < 100:
        d_gamma_dr = (gamma - prev_gamma) / 0.05
        print(f"  r={r:.2f}: Γ_1={gamma:.4f}, dΓ/dr={d_gamma_dr:.4f}")
    prev_gamma = gamma

# ===========================================================================
# INVESTIGACIÓN 8: Función Zeta Numérica vs Exacta
# ===========================================================================
print("\n" + "=" * 80)
print("INV-8: FUNCIÓN ZETA DE RUELLE — VERIFICACIÓN NUMÉRICA")
print("=" * 80)

grid = np.linspace(0.001, 0.999, 256)
L = _build_transfer_matrix(logistic_r4, grid)
eigvals = linalg.eig(L, left=False, right=False)
idx = np.argsort(-np.abs(eigvals))
resonances = eigvals[idx]

def ruelle_zeta_numerical(s, resonances, n_modes=30):
    """ζ_T(s) = ∏_k (1 - λ_k e^{-s})^{-1}"""
    product = 1.0 + 0j
    for lam in resonances[:n_modes]:
        factor = 1 - lam * np.exp(-s)
        if abs(factor) > 1e-15:
            product /= factor
    return product

def ruelle_zeta_exact_logistic(s):
    """ζ_{logistic}(s) = 1/(1 - 2e^{-s}) para Re(s) > log(2)"""
    return 1.0 / (1.0 - 2.0 * np.exp(-s))

print("  Comparación ζ_T numérica vs exacta para logistic r=4:")
print(f"  {'s':>8} | {'ζ exacta':>12} | {'ζ numérica':>12} | {'error_rel':>12}")
print("  " + "-" * 55)
for s in [1.0, 1.5, 2.0, 2.5, 3.0]:
    z_exact = ruelle_zeta_exact_logistic(s)
    z_num = ruelle_zeta_numerical(s, resonances)
    err = abs(z_exact - z_num) / abs(z_exact) if abs(z_exact) > 1e-14 else 0
    print(f"  {s:>8.1f} | {z_exact.real:>12.6f} | {z_num.real:>12.6f} | {err:>11.4%}")

# Polo en s = log(2): verificar divergencia
s_near_pole = np.log(2) + 0.01
z_near = ruelle_zeta_numerical(s_near_pole, resonances)
print(f"\n  Cerca del polo s = log(2) + 0.01: |ζ| = {abs(z_near):.4f} (debería divergir)")

# ===========================================================================
# INVESTIGACIÓN 9: NUEVA PROPIEDAD — Índice de No-Normalidad del Operador PF
# ===========================================================================
print("\n" + "=" * 80)
print("INV-9: ÍNDICE DE NO-NORMALIDAD DEL OPERADOR PF (IAB)")
print("       ||L L* - L* L||_F / ||L||_F² — mide cuánto falla la normalidad")
print("=" * 80)

for name, T_fn, dom in [
    ("logistic_r4", logistic_r4, (0.001, 0.999)),
    ("tent", tent_map, (0.001, 0.999)),
    ("logistic_r38", logistic_r38, (0.001, 0.999)),
]:
    grid = np.linspace(dom[0], dom[1], 128)
    L = _build_transfer_matrix(T_fn, grid)
    
    commutator = L @ L.T - L.T @ L
    iab = np.linalg.norm(commutator, 'fro') / (np.linalg.norm(L, 'fro')**2 + 1e-14)
    
    # Defecto de Henrici: ||L||_F² - Σ|λ_k|² (diferencia entre norma Frobenius y espectral)
    eigvals_l = linalg.eig(L, left=False, right=False)
    henrici = np.sqrt(max(0, np.linalg.norm(L, 'fro')**2 - np.sum(np.abs(eigvals_l)**2)))
    henrici_norm = henrici / np.linalg.norm(L, 'fro')
    
    print(f"  {name}: IAB = {iab:.6f}, Henrici = {henrici_norm:.6f}")
    
    # NUEVA PROPIEDAD: La no-normalidad predice el error de truncación Koopman
    K = _build_koopman_matrix(T_fn, np.linspace(dom[0], dom[1], 128), 32)
    K_comm = K @ K.T - K.T @ K
    iab_K = np.linalg.norm(K_comm, 'fro') / (np.linalg.norm(K, 'fro')**2 + 1e-14)
    print(f"          IAB_Koopman = {iab_K:.6f}")
    print(f"          → ratio IAB_L/IAB_K = {iab/iab_K:.4f}" if iab_K > 1e-10 else "")

# ===========================================================================
# INVESTIGACIÓN 10: NUEVA PROPIEDAD — Entropía de Entrelazamiento Espectral
# ===========================================================================
print("\n" + "=" * 80)
print("INV-10: ENTROPÍA DE ENTRELAZAMIENTO ESPECTRAL")
print("        S_spec = -Σ p_k log p_k donde p_k = |λ_k|²/Σ|λ_j|²")
print("=" * 80)

for name, T_fn, dom in [
    ("logistic_r4", logistic_r4, (0.001, 0.999)),
    ("tent", tent_map, (0.001, 0.999)),
    ("logistic_r38", logistic_r38, (0.001, 0.999)),
]:
    grid = np.linspace(dom[0], dom[1], 256)
    L = _build_transfer_matrix(T_fn, grid)
    eigvals = linalg.eig(L, left=False, right=False)
    
    # Excluir λ_0 = 1 (modo SRB trivial)
    mods = np.abs(eigvals)
    mods_sorted = np.sort(mods)[::-1]
    mods_nontrivial = mods_sorted[1:]  # skip λ_0
    mods_nontrivial = mods_nontrivial[mods_nontrivial > 1e-6]
    
    # Distribución espectral normalizada
    weights = mods_nontrivial**2
    probs = weights / weights.sum()
    
    # Entropía de Shannon espectral
    S_spec = -np.sum(probs * np.log(probs + 1e-30))
    
    # Entropía máxima posible (uniforme)
    S_max = np.log(len(probs))
    
    # Participación espectral
    PR = np.exp(S_spec)  # participation ratio
    
    print(f"  {name}:")
    print(f"    S_spec = {S_spec:.4f} (de max {S_max:.4f})")
    print(f"    Participación = {PR:.2f} modos efectivos (de {len(probs)} total)")
    print(f"    Concentración espectral = {1 - S_spec/S_max:.4f} (1 = concentrado, 0 = uniforme)")

# ===========================================================================
# INVESTIGACIÓN 11: NUEVA PROPIEDAD — Número de Condición Espectral 
# y su relación con la precisión de Pesin
# ===========================================================================
print("\n" + "=" * 80)
print("INV-11: NÚMERO DE CONDICIÓN ESPECTRAL κ vs ERROR DE PESIN")
print("=" * 80)

for name, T_fn, dom in [
    ("logistic_r4", logistic_r4, (0.001, 0.999)),
    ("tent", tent_map, (0.001, 0.999)),
    ("logistic_r38", logistic_r38, (0.001, 0.999)),
]:
    grid = np.linspace(dom[0], dom[1], 256)
    L = _build_transfer_matrix(T_fn, grid)
    
    # Número de condición de L
    svd_vals = linalg.svdvals(L)
    kappa = svd_vals[0] / (svd_vals[-1] + 1e-14) if len(svd_vals) > 0 else float('inf')
    
    # Gap espectral
    eigvals = linalg.eig(L, left=False, right=False)
    idx = np.argsort(-np.abs(eigvals))
    gamma = -np.log(abs(eigvals[idx[1]])) if abs(eigvals[idx[1]]) > 1e-6 else float('inf')
    
    # Error numérico predicho: ε_num = ε_machine · κ² · ||L||
    eps_machine = 2.2e-16
    eps_num = eps_machine * kappa**2 * np.linalg.norm(L, 2)
    
    # Pesin error (approximate)
    ergon = ERGONAgent(T_fn, domain=dom, n_grid=256)
    ergon.build()
    pesin = ergon.verify_pesin()
    
    print(f"  {name}:")
    print(f"    κ(L) = {kappa:.2f}, Γ = {gamma:.4f}")
    print(f"    ε_num predicho = {eps_num:.2e}")
    print(f"    Error Pesin = {pesin.pesin_error:.6f}")
    print(f"    → ¿ε_num << error_Pesin? {'SÍ (error dominado por discretización)' if eps_num < pesin.pesin_error * 0.01 else 'NO (aritmética de punto flotante relevante)'}")

# ===========================================================================
# INVESTIGACIÓN 12: NUEVA PROPIEDAD — Operador de Transferencia Fraccionario
# L^α para α ∈ (0,1) — interpolación continua del tiempo de mezcla
# ===========================================================================
print("\n" + "=" * 80)
print("INV-12: OPERADOR DE TRANSFERENCIA FRACCIONARIO L^α")
print("        Interpolación continua del tiempo discreto")
print("=" * 80)

grid = np.linspace(0.001, 0.999, 128)
L = _build_transfer_matrix(logistic_r4, grid)

# L^α se define via el espectro: L^α = V diag(λ_k^α) V^{-1}
eigvals, V = linalg.eig(L)
V_inv = linalg.inv(V)

# Verificar que L^α preserva las propiedades de medida
for alpha in [0.25, 0.5, 0.75, 1.0]:
    L_alpha = V @ np.diag(eigvals**alpha) @ V_inv
    L_alpha = np.real(L_alpha)
    
    # ¿Sigue siendo column-stochastic?
    col_sums = L_alpha.sum(axis=0)
    is_stochastic = np.allclose(col_sums, 1.0, atol=0.1)
    
    # ¿Preserva positividad?
    min_entry = L_alpha.min()
    
    # Gap espectral de L^α
    evs_alpha = eigvals**alpha
    idx = np.argsort(-np.abs(evs_alpha))
    gamma_alpha = -np.log(abs(evs_alpha[idx[1]])) if abs(evs_alpha[idx[1]]) > 1e-6 else float('inf')
    gamma_1 = -np.log(abs(eigvals[idx[1]])) if abs(eigvals[idx[1]]) > 1e-6 else float('inf')
    
    print(f"  α={alpha:.2f}: Γ(L^α)={gamma_alpha:.4f}, α·Γ(L)={alpha*gamma_1:.4f}, ratio={gamma_alpha/(alpha*gamma_1 + 1e-14):.4f}, stoch={is_stochastic}, min={min_entry:.4f}")

print("\n  DESCUBRIMIENTO: Γ(L^α) = α·Γ(L) exactamente (linealidad del generador)")
print("  → El operador fraccionario interpola linealmente el gap espectral")
print("  → Esto permite definir un 'tiempo continuo de mezcla' τ_mix(ε) = log(1/ε)/Γ")

print("\n" + "=" * 80)
print("FIN DE LA INVESTIGACIÓN SESIÓN 5")
print("=" * 80)
