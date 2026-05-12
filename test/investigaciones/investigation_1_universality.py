"""
INVESTIGACIÓN 1: Constante Universal de No-Normalidad N(K) ≈ 0.684
====================================================================

HIPÓTESIS: La no-normalidad del operador de Koopman EDMD con base de Chebyshev
es INDEPENDIENTE del sistema dinámico y de la medida de referencia.

Si esto es cierto, es un TEOREMA DE UNIVERSALIDAD de teoría de matrices aleatorias
aplicado a EDMD: el sesgo de proyección es una constante computable.

MÉTODO: Computar N(K) para 6 sistemas dinámicos con propiedades muy diferentes,
con y sin μ_SRB, variando n_obs. Si N(K) ≈ const → universalidad confirmada.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from acf_functor.taa_agent import TAAAgent, TAACanonicalSystems

def compute_non_normality(K):
    """N(K) = ‖KK* - K*K‖_F / ‖K‖_F²"""
    KKt = K @ K.T
    KtK = K.T @ K
    norm_sq = float(np.linalg.norm(K, 'fro') ** 2)
    if norm_sq < 1e-14:
        return 0.0
    return float(np.linalg.norm(KKt - KtK, 'fro')) / norm_sq

# Define 6 test systems with VERY different dynamics
systems = {
    "Logistic r=4 (caótico)": TAACanonicalSystems.logistic(r=4.0),
    "Logistic r=3.5 (período 4)": TAACanonicalSystems.logistic(r=3.5),
    "Contracción 0.3": TAACanonicalSystems.linear_contraction(rate=0.3),
    "Contracción 0.7": TAACanonicalSystems.linear_contraction(rate=0.7),
    "Tent map": TAACanonicalSystems.tent(),
    "Coseno (rotación)": lambda x: np.cos(0.3 * x),
}

domains = {
    "Logistic r=4 (caótico)": (0.01, 0.99),
    "Logistic r=3.5 (período 4)": (0.01, 0.99),
    "Contracción 0.3": (-1.0, 1.0),
    "Contracción 0.7": (-1.0, 1.0),
    "Tent map": (0.01, 0.99),
    "Coseno (rotación)": (-1.0, 1.0),
}

print("=" * 80)
print("INVESTIGACIÓN 1: UNIVERSALIDAD DE N(K) EN EDMD CON BASE CHEBYSHEV")
print("=" * 80)

# Test 1: N(K) across systems with fixed n_obs=24
print("\n--- TEST 1: N(K) vs sistema dinámico (n_obs=24, n_traj=1000) ---")
nk_values = []
for name, T in systems.items():
    domain = domains[name]
    agent = TAAAgent(T, domain=domain, n_obs=24, n_traj=1000)
    agent.build()
    nk = compute_non_normality(agent._K)
    nk_values.append(nk)
    print(f"  {name:35s}  N(K) = {nk:.6f}")

mean_nk = np.mean(nk_values)
std_nk = np.std(nk_values)
cv = std_nk / mean_nk  # coeficiente de variación
print(f"\n  Media N(K) = {mean_nk:.6f} ± {std_nk:.6f}  (CV = {cv:.4f})")
print(f"  {'✅ UNIVERSALIDAD CONFIRMADA' if cv < 0.15 else '❌ UNIVERSALIDAD REFUTADA'}: CV {'<' if cv < 0.15 else '>'} 0.15")

# Test 2: N(K) vs n_obs for logistic r=4
print("\n--- TEST 2: N(K) vs dimensión n_obs (logistic r=4) ---")
T_log = TAACanonicalSystems.logistic(r=4.0)
nobs_vals = [8, 12, 16, 20, 24, 32, 48, 64]
for n_obs in nobs_vals:
    agent = TAAAgent(lambda x: np.clip(T_log(x), 0.01, 0.99), 
                     domain=(0.01, 0.99), n_obs=n_obs, n_traj=2000)
    agent.build()
    nk = compute_non_normality(agent._K)
    nk_random = 1.0 - 1.0/n_obs  # predicción de matrices aleatorias
    iab = nk / nk_random
    print(f"  n_obs={n_obs:3d}  N(K)={nk:.6f}  N_random={nk_random:.4f}  IAB={iab:.4f}")

# Test 3: N(K) con y sin μ_SRB
print("\n--- TEST 3: Efecto de μ_SRB sobre N(K) ---")
from acf_functor.ergon_agent import ERGONAgent, ERGONCanonicalSystems

T_erg = ERGONCanonicalSystems.logistic_r4()
ergon = ERGONAgent(T_erg, domain=(0.001, 0.999), n_grid=128, n_power_iter=2000)
ergon.build()
mu_srb = ergon._mu_srb

# Sin μ_SRB
agent_no_srb = TAAAgent(lambda x: np.clip(T_log(x), 0.01, 0.99),
                         domain=(0.01, 0.99), n_obs=24, n_traj=1000)
agent_no_srb.build(mu_srb=None)
nk_no = compute_non_normality(agent_no_srb._K)

# Con μ_SRB
agent_with_srb = TAAAgent(lambda x: np.clip(T_log(x), 0.01, 0.99),
                           domain=(0.01, 0.99), n_obs=24, n_traj=1000)
# Resample μ_SRB to grid
mu_grid = np.interp(np.linspace(0.01, 0.99, 24), np.linspace(0.001, 0.999, 128), mu_srb)
mu_grid = mu_grid / mu_grid.sum()
agent_with_srb.build(mu_srb=mu_grid)
nk_with = compute_non_normality(agent_with_srb._K)

print(f"  Sin μ_SRB:   N(K) = {nk_no:.6f}")
print(f"  Con μ_SRB:   N(K) = {nk_with:.6f}")
print(f"  Diferencia:  |ΔN| = {abs(nk_no - nk_with):.6f}")
print(f"  {'✅ INDEPENDIENTE de μ' if abs(nk_no - nk_with) / nk_no < 0.1 else '❌ DEPENDIENTE de μ'}")

# Test 4: Comparación con matrices aleatorias gaussianas
print("\n--- TEST 4: N(K_EDMD) vs N(K_random) para n=24 ---")
rng = np.random.default_rng(42)
nk_randoms = []
for _ in range(100):
    K_rand = rng.standard_normal((24, 24))
    nk_randoms.append(compute_non_normality(K_rand))
mean_random = np.mean(nk_randoms)
std_random = np.std(nk_randoms)
print(f"  N(K_EDMD) para logistic r=4:     {nk_no:.6f}")
print(f"  N(K_random) Gaussiana (n=24):     {mean_random:.6f} ± {std_random:.6f}")
print(f"  Predicción teórica 1-1/n:         {1.0 - 1.0/24:.6f}")
print(f"  Ratio EDMD/Random:                {nk_no/mean_random:.4f}")

print("\n" + "=" * 80)
print("CONCLUSIÓN INVESTIGACIÓN 1")
print("=" * 80)
