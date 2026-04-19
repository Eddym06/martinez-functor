#!/usr/bin/env python3
"""
Investigación Sesión 3 — Nuevas propiedades no documentadas del TAA-ERGON-OTU
==============================================================================

7 investigaciones independientes que prueban, refutan o descubren estructuras
matemáticas no documentadas en el ecosistema.
"""

import sys, math, warnings
import numpy as np
from scipy import linalg

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
np.set_printoptions(precision=6, suppress=True)

# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGACIÓN 1: verify_dual_budget() es una TAUTOLOGÍA
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("INV-1: ¿Es verify_dual_budget() un teorema o una tautología?")
print("=" * 80)

from acf_functor.gelfand_triple import GelfandTriple, CanonicalSystems

# Build OTU for logistic r=4
T_log = CanonicalSystems.logistic(r=4.0)
otu = GelfandTriple(T_log, domain=(0.001, 0.999), n_test=32, n_dist=256)
otu.build()
result = otu.analyze(n_modes=16)

# 1a: Check verify_dual_budget code — both d* and n* use the SAME formula
gamma = result.gamma_otu
db = result.dual_budget
print(f"\nΓ_OTU = {gamma:.6f}")
print(f"dual_budget result:")
for key, val in db.items():
    if not key.startswith("theorem"):
        print(f"  {key}: {val}")

# 1b: Compute ACTUAL d*(ε) from TAA truncation error (the REAL test)
from acf_functor.taa_agent import TAAAgent
taa = TAAAgent(T_log, domain=(0.001, 0.999), n_obs=32)
taa.build(result.mu_srb)

print("\n--- Comparación d*(ε) real vs fórmula ---")
print(f"{'ε':>8} | {'d*_TAA(real)':>12} | {'d*_formula':>12} | {'n*_formula':>12} | {'Coinciden?':>12}")

test_fn = lambda x: np.cos(2 * np.pi * x)
for eps in [0.1, 0.01, 0.001]:
    # d* real: buscar el menor d tal que δ(d) < ε
    d_real = 32
    for d in range(1, 33):
        delta = taa.truncation_error(d, test_fn)
        if delta < eps:
            d_real = d
            break
    d_formula = math.ceil(math.log(1.0/eps) / gamma)
    n_formula = math.ceil(math.log(1.0/eps) / gamma)
    match = "SÍ" if d_real == d_formula else "NO"
    print(f"{eps:>8} | {d_real:>12} | {d_formula:>12} | {n_formula:>12} | {match:>12}")

# 1c: PROOF that d_formula == n_formula is trivially true
print("\n[VEREDICTO] d*_formula y n*_formula usan IDÉNTICA fórmula:")
print("  d*(ε) = ceil(log(1/ε) / Γ_OTU)")
print("  n*(ε) = ceil(log(1/ε) / Γ_OTU)")
print("  => d* ≡ n* por DEFINICIÓN, no por teorema.")
print("  La 'verificación' verify_dual_budget() es una TAUTOLOGÍA.")
print("  Lo interesante es si d*_real (de TAA) coincide con d*_formula.")


# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGACIÓN 2: predict() tiene código muerto y es O(n²)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("INV-2: Bug en predict() — variable muerta Kn_psi y complejidad O(n²)")
print("=" * 80)

import inspect
src = inspect.getsource(TAAAgent.predict)
# Find the dead variable
if "Kn_psi = self._K" in src and "Kn_psi" in src:
    uses = src.count("Kn_psi")
    print(f"\nVariable 'Kn_psi' aparece {uses} veces en predict():")
    print("  Línea: 'Kn_psi = self._K  # Start with K' — ASIGNADA")
    print("  Pero NUNCA se usa después. Es CÓDIGO MUERTO.")
    
# Verify the O(n²) issue
print("\nComplejidad de predict():")
print("  El comentario dice 'Apply K^n via eigendecomposition for efficiency'")
print("  Pero el código hace un LOOP de n multiplicaciones K@state:")
print("    for _ in range(n): state = self._K @ state")
print("  Esto es O(n·d²) por paso × n pasos = O(n²·d²) total.")
print("  Con eigendecomposición sería O(d³ + n·d) via K^n = V·Λ^n·V^{-1}.")

# Benchmark
import time
x0 = 0.3
n_steps = 50
t0 = time.perf_counter()
preds = taa.predict(x0, n_steps)
t1 = time.perf_counter()
print(f"\npredict({n_steps} pasos): {(t1-t0)*1000:.1f} ms")

# Verify predictions vs actual orbit
print("\nComparación predict() vs órbita real (primeros 10 pasos):")
x = x0
actual = []
for i in range(min(10, n_steps)):
    x = float(T_log(np.array([x]))[0])
    actual.append(x)

print(f"{'Paso':>5} | {'predict':>10} | {'real':>10} | {'error':>10}")
for i in range(min(10, n_steps)):
    print(f"{i+1:>5} | {preds[i]:>10.6f} | {actual[i]:>10.6f} | {abs(preds[i]-actual[i]):>10.6f}")

# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGACIÓN 3: Matrices PF inconsistentes entre ERGON y OTU
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("INV-3: Inconsistencia en la construcción de la matriz PF: ERGON vs OTU")
print("=" * 80)

from acf_functor.ergon_agent import ERGONAgent, ERGONCanonicalSystems

# Build both for the SAME system (logistic r=4) with SAME grid size
n_grid = 256
ergon = ERGONAgent(ERGONCanonicalSystems.logistic_r4(), domain=(0.001, 0.999), n_grid=n_grid)
ergon.build()

otu_cmp = GelfandTriple(T_log, domain=(0.001, 0.999), n_test=32, n_dist=n_grid)
otu_cmp.build()

L_ergon = ergon._ulam
L_otu = otu_cmp._L

# Compare
diff_fro = np.linalg.norm(L_ergon - L_otu, 'fro')
diff_rel = diff_fro / (np.linalg.norm(L_otu, 'fro') + 1e-14)

print(f"\n‖L_ERGON - L_OTU‖_F = {diff_fro:.6f}")
print(f"‖L_ERGON - L_OTU‖_F / ‖L_OTU‖_F = {diff_rel:.6f} ({diff_rel*100:.2f}%)")

# Check column-stochasticity
col_sums_ergon = L_ergon.sum(axis=0)
col_sums_otu = L_otu.sum(axis=0)
print(f"\nColumn sums (should be 1.0):")
print(f"  ERGON: mean={col_sums_ergon.mean():.6f}, std={col_sums_ergon.std():.6f}, min={col_sums_ergon.min():.6f}")
print(f"  OTU:   mean={col_sums_otu.mean():.6f}, std={col_sums_otu.std():.6f}, min={col_sums_otu.min():.6f}")

# Compare SRB measures
mu_ergon = ergon._mu_srb
mu_otu = result.mu_srb  # from OTU analyze

# Resample if needed
if len(mu_ergon) != len(mu_otu):
    from scipy.interpolate import interp1d
    x_ergon = np.linspace(0, 1, len(mu_ergon))
    x_otu = np.linspace(0, 1, len(mu_otu))
    f_interp = interp1d(x_otu, mu_otu, kind='linear', fill_value='extrapolate')
    mu_otu_resampled = f_interp(x_ergon)
    mu_otu_resampled = np.maximum(mu_otu_resampled, 0)
    mu_otu_resampled /= mu_otu_resampled.sum()
    mu_diff = np.linalg.norm(mu_ergon - mu_otu_resampled, ord=1)
else:
    mu_diff = np.linalg.norm(mu_ergon - mu_otu, ord=1)

print(f"\n‖μ_ERGON - μ_OTU‖_1 = {mu_diff:.6f}")
print(f"\nMétodo ERGON: 20 puntos uniformes aleatorios por celda")
print(f"Método OTU:   8 puntos Gauss-Legendre por celda")
print(f"{'→ Gauss-Legendre es más preciso (convergencia exponencial vs algebraica)' if diff_rel > 0.01 else '→ Diferencia insignificante'}")

# Compare eigenvalues
ev_ergon = np.sort(np.abs(linalg.eigvals(L_ergon)))[::-1]
ev_otu = np.sort(np.abs(linalg.eigvals(L_otu)))[::-1]

print(f"\nPrimeros 8 |λ_k| (PF eigenvalues):")
print(f"{'k':>3} | {'|λ_k|_ERGON':>12} | {'|λ_k|_OTU':>12} | {'diff':>10}")
for k in range(min(8, len(ev_ergon), len(ev_otu))):
    d = abs(ev_ergon[k] - ev_otu[k])
    print(f"{k:>3} | {ev_ergon[k]:>12.8f} | {ev_otu[k]:>12.8f} | {d:>10.8f}")

# Impact on h_KS
h_ks_ergon = ergon.verify_pesin().h_ks
h_ks_otu = result.h_ks
print(f"\nh_KS ERGON = {h_ks_ergon:.6f}")
print(f"h_KS OTU   = {h_ks_otu:.6f}")
print(f"Δh_KS      = {abs(h_ks_ergon - h_ks_otu):.6f}")
print(f"h_KS exact = {math.log(2):.6f}")
print(f"Error ERGON = {abs(h_ks_ergon - math.log(2)):.6f} ({abs(h_ks_ergon - math.log(2))/math.log(2)*100:.1f}%)")
print(f"Error OTU   = {abs(h_ks_otu - math.log(2)):.6f} ({abs(h_ks_otu - math.log(2))/math.log(2)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGACIÓN 4: ERG-9 (Coverage) está hardcodeado como True
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("INV-4: ERG-9 (Coverage Complete) — ¿es un certificado real o hardcoded?")
print("=" * 80)

src_certify = inspect.getsource(ERGONAgent.certify)
if "coverage = True" in src_certify:
    print("\nERG-9 'Coverage Complete' está HARDCODEADO como True:")
    print("  Línea en certify(): 'coverage = True'")
    print("  Comentario: '# ERG-9: Coverage = True by theorem (TAA handles integrable, ERGON handles chaotic)'")
    print("\n  Esto NO es un certificado verificado computacionalmente.")
    print("  Es una ASERCIÓN teórica sin prueba numérica.")
    print("  Un verdadero certificado debería verificar que:")
    print("    - Existe un umbral 𝔈* tal que 𝔈 < 𝔈* → TAA opera solo")
    print("    - Y que 𝔈 > 𝔈* → ERGON es necesario")
    print("    - Y que no hay 'gap' intermedio donde ninguno funciona")

# Check ERG-11 too
if "ec_spectral = ec" in src_certify:
    print("\nERG-11 (Spectral Complexity) está copiado de ERG-6b:")
    print("  Línea: 'ec_spectral = ec  # consistent by construction'")
    print("  ERG-11 es simplemente un ALIAS de ERG-6b, no un cómputo independiente.")


# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGACIÓN 5: Tasa de convergencia de Birkhoff r → 1/2
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("INV-5: ¿Converge la tasa de Birkhoff a r=1/2 para sistemas mixing?")
print("=" * 80)

systems = {
    "logistic_r4": ERGONCanonicalSystems.logistic_r4(),
    "tent_map": ERGONCanonicalSystems.tent_map(),
    "doubling": ERGONCanonicalSystems.doubling_map(),
}

for name, T_sys in systems.items():
    ergon_sys = ERGONAgent(T_sys, domain=(0.001, 0.999), n_grid=256)
    ergon_sys.build()
    rate = ergon_sys.birkhoff_convergence_rate()
    print(f"\n{name}: r = {rate:.4f} (teórico = 0.5 para mixing)")

print("\nExplicación: Para sistemas mixing, la tasa de convergencia de Birkhoff")
print("es error ~ C/n^r con r = 1/2 (CLT para sumas ergódicas).")
print("r > 1/2 indica correlaciones de largo alcance débiles.")
print("r < 1/2 indica correlaciones persistentes (mixing algebraico).")


# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGACIÓN 6: Violación de positividad del operador de Koopman EDMD
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("INV-6: ¿Preserva el operador de Koopman EDMD la positividad?")
print("=" * 80)

print("\nEl operador de Koopman real K satisface: f ≥ 0 ⟹ Kf ≥ 0")
print("(es un operador positivo en L²)")
print("La aproximación EDMD K_d NO necesariamente preserva positividad.\n")

# Test positivity preservation
systems_pos = {
    "logistic_r4": (T_log, (0.001, 0.999)),
    "tent": (CanonicalSystems.tent(), (0.001, 0.999)),
}

for name, (T_sys, dom) in systems_pos.items():
    taa_pos = TAAAgent(T_sys, domain=dom, n_obs=32)
    taa_pos.build()
    K = taa_pos._K
    
    # Check: K applied to a positive function (constant 1 in Chebyshev basis)
    # In Chebyshev basis, f=1 is the vector [1, 0, 0, ...]
    f_const = np.zeros(32)
    f_const[0] = 1.0
    Kf = K @ f_const
    
    # Also test a positive function f(x) = x + 1
    # In Chebyshev: T_0(x) = 1, T_1(x) = x, so f = [1, 0.5, 0, ...] approximately
    f_pos = np.zeros(32)
    f_pos[0] = 1.5  # average ≈ 1.5 on [0,1]
    f_pos[1] = 0.5  # linear component
    Kf_pos = K @ f_pos
    
    # Reconstruct function values on a grid
    a, b = dom
    grid = np.linspace(a, b, 100)
    xi = np.clip(2.0 * (grid - a) / (b - a) - 1.0, -1.0, 1.0)
    
    # Evaluate Kf on grid
    Kf_vals = np.zeros(100)
    for k in range(32):
        Kf_vals += Kf[k] * np.cos(k * np.arccos(xi))
    
    Kf_pos_vals = np.zeros(100)
    for k in range(32):
        Kf_pos_vals += Kf_pos[k] * np.cos(k * np.arccos(xi))
    
    min_const = float(np.min(Kf_vals))
    min_pos = float(np.min(Kf_pos_vals))
    
    n_neg_const = int(np.sum(Kf_vals < -1e-6))
    n_neg_pos = int(np.sum(Kf_pos_vals < -1e-6))
    
    print(f"{name}:")
    print(f"  K(1): min = {min_const:.6f}, #negativos = {n_neg_const}/100")
    print(f"  K(x+1.5): min = {min_pos:.6f}, #negativos = {n_neg_pos}/100")
    if n_neg_const > 0 or n_neg_pos > 0:
        print(f"  → VIOLACIÓN DE POSITIVIDAD DETECTADA")
    else:
        print(f"  → Positividad preservada")


# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGACIÓN 7: Sensibilidad de _classify_spectrum a sus umbrales
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("INV-7: Sensibilidad de _classify_spectrum() a umbrales empíricos")
print("=" * 80)

print("\n_classify_spectrum usa umbrales fijos NO documentados:")
print("  - slope_exp < -0.05  → candidato exponencial")
print("  - r2_exp > 0.80      → confirmado exponencial")
print("  - slope_poly < -0.3  → candidato polinomial")
print("  - r2_poly > 0.70     → confirmado polinomial")
print("  - else               → CHAOTIC")
print("\nProbando sensibilidad con logistic r variable:")

for r in [3.0, 3.5, 3.57, 3.8, 3.9, 4.0]:
    T_r = CanonicalSystems.logistic(r=r)
    taa_r = TAAAgent(T_r, domain=(0.001, 0.999), n_obs=24)
    try:
        taa_r.build()
        dc = taa_r._spectrum.decay_class.value
        alpha = taa_r._spectrum.alpha_A
        
        # Compute actual Lyapunov for comparison
        lyap = taa_r.estimate_lyapunov()
        
        # Get R² values by reproducing the classification
        mods = np.abs(taa_r._eigenvalues)
        n_sig = int(np.sum(mods > 1e-6))
        k_range = np.arange(1, min(n_sig, 23) + 1)
        log_mods = np.log(np.maximum(mods[1:len(k_range)+1], 1e-15))
        valid = np.isfinite(log_mods)
        
        if valid.sum() >= 3:
            k_v = k_range[valid].astype(float)
            lm_v = log_mods[valid]
            slope_exp, _ = np.polyfit(k_v, lm_v, 1)
            exp_fit = slope_exp * k_v + _
            ss_res = np.sum((lm_v - exp_fit)**2)
            ss_tot = np.sum((lm_v - lm_v.mean())**2)
            r2_exp = 1 - ss_res / (ss_tot + 1e-14)
            
            lk_v = np.log(k_v)
            slope_poly, _ = np.polyfit(lk_v, lm_v, 1)
            poly_fit = slope_poly * lk_v + _
            ss_res = np.sum((lm_v - poly_fit)**2)
            r2_poly = 1 - ss_res / (ss_tot + 1e-14)
        else:
            r2_exp = r2_poly = slope_exp = slope_poly = 0
        
        print(f"  r={r:.2f}: class={dc:>12}, α={alpha:.3f}, λ_max={lyap:.3f}, "
              f"R²_exp={r2_exp:.3f}, R²_poly={r2_poly:.3f}, slope_exp={slope_exp:.3f}")
    except Exception as e:
        print(f"  r={r:.2f}: ERROR — {e}")


# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGACIÓN 8: Operador espectral de Koopman — identidad de Parseval
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("INV-8: Identidad de Parseval para el operador de Koopman EDMD")
print("=" * 80)

print("\nPara un operador de Koopman EXACTO (unitario en L²(μ_SRB)):")
print("  ‖Kf‖² = ‖f‖²  para todo f ∈ L²")
print("  Equivalente: K es una isometría, ‖K‖_op = 1")
print("  Y: Σ|λ_k|² = d (preservación de energía total)")
print("\nPara EDMD (aproximación finita), Parseval se viola.\n")

for name, (T_sys, dom) in systems_pos.items():
    taa_p = TAAAgent(T_sys, domain=dom, n_obs=32)
    taa_p.build()
    K = taa_p._K
    
    # Spectral radius
    eigs = taa_p._eigenvalues
    spec_radius = float(np.max(np.abs(eigs)))
    
    # Operator norm
    op_norm = float(np.linalg.norm(K, 2))
    
    # Parseval sum: Σ|λ_k|²
    parseval_sum = float(np.sum(np.abs(eigs)**2))
    
    # Isometry check: ‖K*K - I‖
    KtK = K.T @ K
    iso_error = float(np.linalg.norm(KtK - np.eye(32), 'fro'))
    
    # Energy preservation for specific function
    f_test = np.zeros(32)
    f_test[0] = 1.0  # constant function
    Kf = K @ f_test
    energy_ratio = float(np.linalg.norm(Kf) / np.linalg.norm(f_test))
    
    print(f"{name}:")
    print(f"  ρ(K) = {spec_radius:.6f} (debería ≈ 1.0)")
    print(f"  ‖K‖_op = {op_norm:.6f} (debería ≈ 1.0 para isometría)")
    print(f"  Σ|λ_k|² = {parseval_sum:.4f} (debería = {32} para Parseval exacto)")
    print(f"  ‖K*K - I‖_F = {iso_error:.4f} (debería ≈ 0 para isometría)")
    print(f"  ‖Kf‖/‖f‖ para f=1: {energy_ratio:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGACIÓN 9: Comparar d*(ε) por 3 métodos independientes
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("INV-9: Cross-validación de d*(ε) por 3 métodos")
print("=" * 80)

print("\nMétodo 1: d*_TAA = menor d tal que δ(d,f) < ε (truncation_error real)")
print("Método 2: d*_spectral = ⌈log(C/ε)/log(ρ)⌉ (fit exponencial)")
print("Método 3: d*_OTU = ⌈log(1/ε)/Γ_OTU⌉ (gap espectral PF)")

for eps in [0.1, 0.01, 0.001]:
    # Method 1: actual truncation error from TAA
    d_taa = 32
    for d in range(1, 33):
        delta = taa.truncation_error(d, test_fn)
        if delta < eps:
            d_taa = d
            break
    
    # Method 2: from TAA spectral classification
    d_spectral = taa._spectrum.d_star.get(eps, 32)
    
    # Method 3: from OTU spectral gap
    d_otu = math.ceil(math.log(1.0/eps) / max(gamma, 1e-10))
    
    print(f"\nε = {eps}:")
    print(f"  d*_TAA(real) = {d_taa}")
    print(f"  d*_spectral  = {d_spectral}")
    print(f"  d*_OTU       = {d_otu}")
    if d_taa == d_spectral == d_otu:
        print(f"  → Los 3 métodos coinciden ✓")
    else:
        print(f"  → DISCREPANCIA: los métodos dan resultados diferentes")
        print(f"    d*_TAA es el ground truth (medición directa)")


# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGACIÓN 10: Convergencia espectral Koopman vs PF
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("INV-10: ¿Convergen los espectros de Koopman y PF al mismo límite?")
print("=" * 80)

print("\nTeorema: K y L (Perron-Frobenius) son adjuntos, comparten espectro.")
print("En la discretización EDMD+Ulam, los espectros difieren.")

# Get eigenvalues from both
K_eigs = np.sort(np.abs(taa._eigenvalues))[::-1]
L_eigs = np.sort(np.abs(linalg.eigvals(otu._L)))[::-1]

# Compare top eigenvalues
n_cmp = min(16, len(K_eigs), len(L_eigs))
print(f"\n{'k':>3} | {'|λ_k| Koopman':>15} | {'|λ_k| PF':>15} | {'diff':>12}")
for k in range(n_cmp):
    d = abs(K_eigs[k] - L_eigs[k])
    print(f"{k:>3} | {K_eigs[k]:>15.8f} | {L_eigs[k]:>15.8f} | {d:>12.8f}")

# Spectral gaps
gamma_K = float(-np.log(max(K_eigs[1], 1e-300)))
gamma_L = float(-np.log(max(L_eigs[1], 1e-300)))
print(f"\nΓ_Koopman = {gamma_K:.6f}")
print(f"Γ_PF      = {gamma_L:.6f}")
print(f"ΔΓ        = {abs(gamma_K - gamma_L):.6f}")
print(f"\nLos espectros son adjuntos en teoría, pero la discretización")
print(f"(EDMD Chebyshev vs Ulam uniforme) introduce una discrepancia de {abs(gamma_K - gamma_L):.4f}")

print("\n" + "=" * 80)
print("FIN DE LAS INVESTIGACIONES — SESIÓN 3")
print("=" * 80)
