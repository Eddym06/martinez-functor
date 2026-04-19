"""
INVESTIGACIÓN 3: El Umbral Crítico 𝔈* y la Jerarquía de Tiempos No-Cloning vs Dual Budget
=============================================================================================

HIPÓTESIS COMBINADA:
A) La fórmula 𝔈* = 1 - Γ_OTU/h_KS es un PREDICTOR correcto de cuándo TAA necesita ERGON.
B) La cota no-cloning n_min(ε) ≥ C·ε^{-D₂} SIEMPRE domina la cota dual n*(ε) = log(1/ε)/Γ.

Estas dos hipótesis juntas establecen una JERARQUÍA DE ESCALAS TEMPORALES:
  n_dual < n_crossover < n_cloning

MÉTODO: 
- Verificar 𝔈* para 4 sistemas distintos
- Computar ambas cotas de tiempo y verificar la dominancia
- Usar el ecosistema TAA+ERGON+OTU completo
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import math
from acf_functor.taa_agent import TAAAgent, TAACanonicalSystems
from acf_functor.ergon_agent import ERGONAgent, ERGONCanonicalSystems
from acf_functor.gelfand_triple import GelfandTriple, CanonicalSystems
from acf_functor.deep_problems import compute_no_clone_bound

print("=" * 80)
print("INVESTIGACIÓN 3: UMBRAL CRÍTICO 𝔈* Y JERARQUÍA DE ESCALAS TEMPORALES")
print("=" * 80)

# ============ PARTE A: Verificación de 𝔈* ============
print("\n" + "=" * 60)
print("PARTE A: Umbral Crítico 𝔈* = 1 - Γ_OTU / h_KS")
print("=" * 60)

test_systems = [
    ("Logistic r=4", CanonicalSystems.logistic(r=4.0), (0.0, 1.0)),
    ("Tent map", CanonicalSystems.tent(), (0.0, 1.0)),
    ("Logistic r=3.8", CanonicalSystems.logistic(r=3.8), (0.0, 1.0)),
    ("Chebyshev n=2", CanonicalSystems.chebyshev(n=2), (-1.0, 1.0)),
]

results = []
for name, T, domain in test_systems:
    print(f"\n  --- {name} ---")
    
    # OTU analysis
    triple = GelfandTriple(T, domain=domain, n_test=32, n_dist=256, srb_tol=1e-8)
    triple.build()
    otu_result = triple.analyze(n_modes=16, n_orbit=30000)
    
    gamma_otu = otu_result.gamma_otu
    h_ks = otu_result.h_ks
    
    # TAA analysis — pass OTU-derived h_ks and lyapunov_sum
    taa = TAAAgent(T, domain=domain, n_obs=24, n_traj=1000)
    taa.build()
    # Use h_ks as lyapunov_sum (Pesin identity for 1D maps)
    state = taa.diagnose(h_ks=h_ks, lyapunov_sum=h_ks)
    
    # Compute 𝔈* 
    if h_ks > 1e-6:
        e_star = 1.0 - gamma_otu / h_ks
        e_star = max(0.0, min(1.0, e_star))
    else:
        e_star = 0.0
    
    # Compute current 𝔈 from TAA state
    e_current = state.ergodic_complexity
    
    # Prediction: should TAA defer?
    taa_should_defer = e_current > e_star and h_ks > 0.05
    taa_actually_defers = state.defer_to_ergon
    
    correct = taa_should_defer == taa_actually_defers or (not taa_should_defer and not taa_actually_defers)
    
    print(f"    Γ_OTU = {gamma_otu:.4f}")
    print(f"    h_KS  = {h_ks:.4f}")
    print(f"    𝔈*    = {e_star:.4f}")
    print(f"    𝔈     = {e_current:.4f}")
    print(f"    𝔈 {'>' if e_current > e_star else '<'} 𝔈* → TAA {'NECESITA' if taa_should_defer else 'NO necesita'} ERGON")
    print(f"    TAA dice defer={taa_actually_defers}")
    print(f"    {'✅ PREDICCIÓN CORRECTA' if correct else '⚠️ DISCREPANCIA'}")
    
    results.append({
        "name": name, "gamma": gamma_otu, "h_ks": h_ks,
        "e_star": e_star, "e_current": e_current,
        "prediction_correct": correct
    })

n_correct = sum(1 for r in results if r["prediction_correct"])
print(f"\n  RESULTADO: {n_correct}/{len(results)} predicciones correctas")

# ============ PARTE B: Jerarquía de Escalas Temporales ============
print("\n" + "=" * 60)
print("PARTE B: Jerarquía n_dual < n_crossover < n_cloning")
print("=" * 60)

for name, T, domain in test_systems[:2]:  # logistic y tent
    print(f"\n  --- {name} ---")
    
    # Get OTU data
    triple = GelfandTriple(T, domain=domain, n_test=32, n_dist=256, srb_tol=1e-8)
    triple.build()
    otu_result = triple.analyze(n_modes=16, n_orbit=30000)
    
    gamma_otu = otu_result.gamma_otu
    
    # Get multifractal D_2
    if otu_result.multifractal is not None:
        D_2 = otu_result.multifractal.D_2
    else:
        D_2 = 0.9  # default
    
    # Get crossover time from damping profile
    if otu_result.spectrum.damping_profile is not None:
        n_crossover = otu_result.spectrum.damping_profile.n_crossover
    else:
        n_crossover = float('inf')
    
    epsilons = [0.1, 0.01, 0.001, 0.0001]
    
    print(f"    Γ_OTU = {gamma_otu:.4f}, D_2 = {D_2:.4f}")
    print(f"    n_crossover = {n_crossover:.1f}")
    print()
    print(f"    {'ε':>10s} | {'n_dual':>10s} | {'n_cross':>10s} | {'n_clone':>10s} | {'dual<clone?':>12s}")
    print(f"    {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12}")
    
    all_hierarchical = True
    for eps in epsilons:
        # n_dual = log(1/ε) / Γ_OTU
        if gamma_otu > 1e-10:
            n_dual = math.ceil(math.log(1.0 / eps) / gamma_otu)
        else:
            n_dual = 999999
        
        # n_cloning from no-clone bound: n_min ≥ C · ε^{-D_2}
        h_ks = otu_result.h_ks
        clone_cert = compute_no_clone_bound(D_2=D_2, h_ks=h_ks, epsilon=eps)
        n_clone = clone_cert.n_min
        
        n_cross_val = n_crossover if np.isfinite(n_crossover) else -1
        
        hierarchical = n_dual <= n_clone
        if not hierarchical:
            all_hierarchical = False
        
        print(f"    {eps:>10.4f} | {n_dual:>10d} | {n_cross_val:>10.0f} | {n_clone:>10d} | {'✅' if hierarchical else '❌'}")
    
    print(f"\n    {'✅ JERARQUÍA VERIFICADA' if all_hierarchical else '❌ JERARQUÍA VIOLADA'}: n_dual ≤ n_cloning para todo ε")

# ============ PARTE C: Discrepancia en la definición de 𝔈 ============
print("\n" + "=" * 60)
print("PARTE C: Discrepancia 𝔈(T) entre TAA y ERGON")
print("=" * 60)

T = CanonicalSystems.logistic(r=4.0)
domain = (0.0, 1.0)

# ERGON computation
T_erg = ERGONCanonicalSystems.logistic_r4()
ergon = ERGONAgent(T_erg, domain=(0.001, 0.999), n_grid=256, n_power_iter=3000)
ergon.build()
lyap = ergon.compute_lyapunov_field()
h_ks_erg = lyap.lyapunov_sum
lyap_sum = lyap.lyapunov_sum

# ERGON formula: 𝔈 = h_KS / log(1 + Σλ⁺)
e_ergon = h_ks_erg / math.log(1.0 + lyap_sum + 1e-14)

# TAA formula: 𝔈 = h_KS / Σλ⁺
e_taa = h_ks_erg / (lyap_sum + 1e-14)

print(f"  h_KS = {h_ks_erg:.6f}")
print(f"  Σλ⁺  = {lyap_sum:.6f}")
print(f"  log(1+Σλ⁺) = {math.log(1.0 + lyap_sum):.6f}")
print(f"\n  𝔈_ERGON = h_KS / log(1+Σλ⁺)  = {e_ergon:.6f}")
print(f"  𝔈_TAA   = h_KS / Σλ⁺           = {e_taa:.6f}")
print(f"  Δ𝔈      = {abs(e_ergon - e_taa):.6f}")

# For Pesin-saturated systems (h_KS = Σλ⁺):
print(f"\n  Para h_KS = Σλ⁺ (Pesin exacto):")
print(f"    𝔈_ERGON = Σλ⁺ / log(1+Σλ⁺) = {lyap_sum / math.log(1.0 + lyap_sum):.6f} (> 1.0 → clamped a 1.0)")
print(f"    𝔈_TAA   = 1.0 (exacto)")
print(f"\n  ⚠️ DISCREPANCIA DOCUMENTADA: Las dos fórmulas son DIFERENTES objetos matemáticos.")
print(f"  → TAA usa el ratio de Pesin puro: 𝔈 = 1 ↔ Pesin saturado")
print(f"  → ERGON usa una versión regularizada: nunca diverge, pero pierde la propiedad 𝔈 = 1 ↔ Pesin")

print("\n" + "=" * 80)
print("CONCLUSIÓN INVESTIGACIÓN 3")
print("=" * 80)
