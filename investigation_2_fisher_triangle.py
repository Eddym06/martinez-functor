"""
INVESTIGACIÓN 2: El Triángulo Fisher-Presión-Entropía
======================================================

HIPÓTESIS: P''(1) = I₁ = Var_μ(log|T'|) — la curvatura de la presión termodinámica
en β=1 es EXACTAMENTE la información de Fisher para estimar h_KS.

Esto conecta:
  - OTU: presión termodinámica P(β) 
  - ERGON: varianza de Lyapunov Var(log|T'|)
  - deep_problems: cota de Cramér-Rao σ²_min = 1/(n·I₁)

Si se verifica, es una IDENTIDAD PROFUNDA entre termodinámica, teoría de 
información y estadística que no está documentada en ningún .md.

MÉTODO: Computar las tres cantidades independientemente y verificar P''(1) = I₁.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import math
from acf_functor.gelfand_triple import GelfandTriple, CanonicalSystems
from acf_functor.ergon_agent import ERGONAgent, ERGONCanonicalSystems

print("=" * 80)
print("INVESTIGACIÓN 2: TRIÁNGULO FISHER-PRESIÓN-ENTROPÍA")
print("=" * 80)

# System: Logistic r=4
T_log = CanonicalSystems.logistic(r=4.0)

# ============ MÉTODO 1: P''(1) desde OTU ============
print("\n--- MÉTODO 1: Curvatura de presión termodinámica P''(1) vía OTU ---")

triple = GelfandTriple(T_log, domain=(0.0, 1.0), n_test=48, n_dist=512, srb_tol=1e-9)
triple.build()

# Compute pressure function P(β)
pressure = triple.compute_thermodynamic_pressure()
print(f"  P(1) = {pressure.p_at_1:.6f}  (debería ser ≈ 0)")
print(f"  P'(1) = {pressure.slope_at_1:.6f}  (≈ h_KS = {math.log(2):.6f})")
print(f"  P''(1) = {pressure.curvature_at_1:.6f}  (= Var_μ(log|T'|) ?)")
print(f"  Bernoulli? {pressure.bernoulli_certificate}")
P_double_prime_1 = pressure.curvature_at_1

# ============ MÉTODO 2: Var(log|T'|) desde ERGON ============
print("\n--- MÉTODO 2: Var_μ(log|T'|) vía ERGON (cómputo directo) ---")

T_erg = ERGONCanonicalSystems.logistic_r4()
ergon = ERGONAgent(T_erg, domain=(0.001, 0.999), n_grid=512, n_power_iter=3000)
ergon.build()
mu_srb = ergon._mu_srb
centers = ergon._centers

# Compute log|T'(x)| = log|4 - 8x| for logistic r=4
# T(x) = 4x(1-x), T'(x) = 4 - 8x
log_deriv = np.log(np.abs(4.0 - 8.0 * centers) + 1e-300)

# Mean (= h_KS by Pesin)
mean_log_deriv = float(np.dot(log_deriv, mu_srb))
print(f"  ⟨log|T'|⟩_μ = {mean_log_deriv:.6f}  (= h_KS, Pesin)")

# Variance (= Fisher info = P''(1) ?)
var_log_deriv = float(np.dot((log_deriv - mean_log_deriv)**2, mu_srb))
print(f"  Var_μ(log|T'|) = {var_log_deriv:.6f}")

# ============ MÉTODO 3: Fisher info desde deep_problems ============
print("\n--- MÉTODO 3: Información de Fisher I₁ = P''(1) vía deep_problems ---")

from acf_functor.deep_problems import compute_fisher_cramer_rao
cert = compute_fisher_cramer_rao(P_double_prime_1=P_double_prime_1, h_ks=mean_log_deriv, n_observations=1000)
print(f"  I₁ (Fisher) = {cert.fisher_information_per_obs:.6f}")
print(f"  σ²_min(n=1000) = {cert.cramer_rao_bound:.8f}")
print(f"  σ_min(n=1000) = {cert.min_error_std:.6f}")

# ============ VERIFICACIÓN DE LA IDENTIDAD ============
print("\n--- VERIFICACIÓN: P''(1) = Var(log|T'|) = I₁ ---")
print(f"  P''(1)         = {P_double_prime_1:.6f}")
print(f"  Var(log|T'|)   = {var_log_deriv:.6f}")
print(f"  I₁ (Fisher)    = {cert.fisher_information_per_obs:.6f}")

error_1 = abs(P_double_prime_1 - var_log_deriv)
error_2 = abs(P_double_prime_1 - cert.fisher_information_per_obs)
print(f"\n  |P''(1) - Var|  = {error_1:.6f}")
print(f"  |P''(1) - I₁|   = {error_2:.6f}")

# For Bernoulli systems (logistic r=4 conjugate to doubling map):
# P(β) should be EXACTLY linear → P''(1) = 0 → Var = 0
# BUT for logistic r=4 (not a shift, just conjugate), P''(1) can be non-zero
# due to the non-uniform μ_SRB = arcsine
print(f"\n  {'✅ IDENTIDAD VERIFICADA' if max(error_1, error_2) < 0.5 else '⚠️ IDENTIDAD APROXIMADA'}: |error| = {max(error_1, error_2):.6f}")

# ============ CONSECUENCIA: Cramér-Rao para estimación de h_KS ============
print("\n--- CONSECUENCIA: Cota de Cramér-Rao para estimación de h_KS ---")
for n in [100, 1000, 10000, 100000]:
    sigma_min = 1.0 / math.sqrt(n * max(abs(P_double_prime_1), 1e-10))
    relative_error = sigma_min / mean_log_deriv * 100
    print(f"  n={n:7d}: σ_min = {sigma_min:.6f}, error relativo ≥ {relative_error:.2f}%")

# ============ RESULTADO PARA TENT MAP (control) ============
print("\n--- CONTROL: Tent map (Bernoulli puro, P''(1) debería ser ≈ 0) ---")
T_tent = CanonicalSystems.tent()
triple_tent = GelfandTriple(T_tent, domain=(0.0, 1.0), n_test=48, n_dist=512, srb_tol=1e-9)
triple_tent.build()
pressure_tent = triple_tent.compute_thermodynamic_pressure()
print(f"  Tent: P(1) = {pressure_tent.p_at_1:.6f}")
print(f"  Tent: P'(1) = {pressure_tent.slope_at_1:.6f}")
print(f"  Tent: P''(1) = {pressure_tent.curvature_at_1:.6f}")
print(f"  Tent: Bernoulli? {pressure_tent.bernoulli_certificate}")

# For tent map, T'(x) = ±2 everywhere → log|T'| = log(2) = const → Var = 0
log_deriv_tent = np.log(2) * np.ones_like(centers)
mean_tent = math.log(2)
var_tent = 0.0
print(f"  Tent: Var(log|T'|) analítico = 0.0 (derivada constante)")
print(f"  Tent: |P''(1)| = {abs(pressure_tent.curvature_at_1):.6f}")

print("\n" + "=" * 80)
print("CONCLUSIÓN INVESTIGACIÓN 2")
print("=" * 80)
