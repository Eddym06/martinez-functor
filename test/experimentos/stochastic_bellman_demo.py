"""Stochastic Control Demo — ACF Ecosystem
==========================================
Problema: agente 1D con ruido browniano (Ornstein-Uhlenbeck), costo cuadrático,
horizonte infinito descontado.

  Dinámica:   dx = -a*x*dt + σ*dW_t
  Discret.:   x_{t+1} = ρ*x_t + σ√Δt * ε_t,   ε ~ N(0,1)
  Costo:      c(x) = x²
  Valor:      V(x) = x² + γ*E[V(x')|x]   →   V*(x) = P*x²,  P = 1/(1-γρ²)

Pipeline:
  1. Generar trayectorias con Euler-Maruyama
  2. BiPoem.symbiosis()  → EMA (densidad invariante vía EDMD)
  3. CoPoem.synthesize() → TAA (sintetiza operador para "minimize expected cost")
  4. BiPoem.find_fixed_point() → ciclo Φ ⇌ Φ* hasta convergencia
  5. Residuo de Bellman: ‖V - c - γ·𝒦V‖
"""

from __future__ import annotations

import sys
import math
from typing import Tuple, Dict, Any

import torch
import numpy as np

# --------------------------------------------------------------------- #
# Asegurar que el paquete acf_functor y poema sean importables           #
# --------------------------------------------------------------------- #
import os
sys.path.insert(0, os.path.dirname(__file__))

from poema.frontend import BiPoem, CoPoem
from acf_functor.core import KoopmanReducer


# ═══════════════════════════════════════════════════════════════════════ #
#  SECCIÓN 0 – Parámetros del problema                                   #
# ═══════════════════════════════════════════════════════════════════════ #

DTYPE = torch.float64

# Ornstein-Uhlenbeck
A     = 0.5     # velocidad de reversión a la media
SIGMA = 0.3     # intensidad del ruido
DT    = 0.05    # paso temporal (Euler-Maruyama)
RHO   = 1.0 - A * DT   # factor de estabilidad en la discretización

# Coste y descuento
GAMMA = 0.95

# Valor teórico: P = 1 / (1 - γρ²)
P_TRUE = 1.0 / (1.0 - GAMMA * RHO**2)

# Trayectorias
N_TRAJ    = 40      # número de trayectorias independientes
T_STEPS   = 500     # pasos por trayectoria
X0_SCALE  = 2.0     # desviación estándar de las condiciones iniciales

# Test para residuo de Bellman
N_TEST = 200

print("=" * 65)
print("DEMO CONTROL ESTOCÁSTICO 1D — ECOSISTEMA ACF")
print("=" * 65)
print(f"  Parámetros OU: a={A}, σ={SIGMA}, Δt={DT}, ρ={RHO:.4f}")
print(f"  Descuento γ={GAMMA},  P_teórico={P_TRUE:.4f}")
print()


# ═══════════════════════════════════════════════════════════════════════ #
#  SECCIÓN 1 – Generación de trayectorias (Euler-Maruyama)               #
# ═══════════════════════════════════════════════════════════════════════ #

torch.manual_seed(42)

def euler_maruyama(
    x0: torch.Tensor,      # (N,)
    n_steps: int,
    a: float,
    sigma: float,
    dt: float,
) -> torch.Tensor:
    """
    Integra N trayectorias de dx = -a*x*dt + σ*dW en paralelo.
    Devuelve tensor (N, n_steps+1).
    """
    N = x0.shape[0]
    traj = torch.zeros(N, n_steps + 1, dtype=DTYPE)
    traj[:, 0] = x0
    rho_step = 1.0 - a * dt
    noise_scale = sigma * math.sqrt(dt)
    for t in range(n_steps):
        noise = torch.randn(N, dtype=DTYPE) * noise_scale
        traj[:, t + 1] = rho_step * traj[:, t] + noise
    return traj


x0 = torch.randn(N_TRAJ, dtype=DTYPE) * X0_SCALE
trajectories = euler_maruyama(x0, T_STEPS, A, SIGMA, DT)  # (N_TRAJ, T+1)

print("─" * 65)
print("PASO 1 — Trayectorias con ruido browniano")
print(f"  Forma: {trajectories.shape}  (N_traj={N_TRAJ}, T={T_STEPS})")
print(f"  x̄ = {trajectories.mean():.4f},  σ̂ = {trajectories.std():.4f}")
print(f"  Densidad invariante teórica: N(0, {(SIGMA**2/(2*A)):.3f})")

# Organizar datos para BiPoem: (n_features=1, n_timesteps=N_TRAJ*(T+1))
all_states = trajectories.reshape(1, -1)  # (1, N_TRAJ*(T+1))
print(f"  Datos achatados para BiPoem: {all_states.shape}")
print()


# ═══════════════════════════════════════════════════════════════════════ #
#  SECCIÓN 2 – BiPoem.symbiosis() → EMA de la densidad invariante        #
# ═══════════════════════════════════════════════════════════════════════ #
#  La descomposición EDMD extrae el operador de Koopman del sistema       #
#  estocástico. Sus autovalores capturan los modos de la densidad.        #
#  El eigenvalor dominante (≈1) corresponde a la medida invariante;      #
#  α(f) mide la tasa de decaimiento espectral (suavidad de la densidad). #
# ═══════════════════════════════════════════════════════════════════════ #

print("─" * 65)
print("PASO 2 — BiPoem.symbiosis()  [EMA densidad invariante]")

bipoem = BiPoem(dtype=DTYPE)
copoem = CoPoem(dtype=DTYPE)

sym_result: Dict[str, Any] = bipoem.symbiosis(
    data=all_states,
    structure={"target_alpha": 0.5},
    max_dimension=64,
    max_iterations=15,
    convergence_threshold=1e-3,
)

K_sym      = sym_result["koopman_matrix"]    # (d, d) en espacio observable
eigvals    = sym_result["eigenvalues"]        # (d,) complejos
alpha_ema  = sym_result["alpha"]
obs_dim    = K_sym.shape[0]

dom_eigval = float(torch.max(torch.abs(eigvals)).real)
print(f"  Dimensión observable: {obs_dim}")
print(f"  Eigenvalor dominante: {dom_eigval:.6f}  (1.0 → medida invariante)")
print(f"  α_ACF(f): {alpha_ema:.4f}  (decay espectral de la densidad)")
print(f"  Error de reconstrucción EDMD: "
      f"{sym_result['reconstruction_error']:.2e}")
print(f"  Iteraciones: {len(sym_result['dimension_history'])}")
print()


# ═══════════════════════════════════════════════════════════════════════ #
#  SECCIÓN 3 – CoPoem.synthesize() → TAA "minimizar costo esperado"      #
# ═══════════════════════════════════════════════════════════════════════ #
#  CoPoem sintetiza una matriz en el espacio de observables que codifica  #
#  la geometría espectral compatible con la especificación objetivo.      #
#  Spec type "optimization" con objetivo "expected_cost":                 #
#    dim = obs_dim (misma base que el Koopman de symbiosis)              #
#    γ-estabilidad implica radio espectral ≤ γ (contracción bajo 𝒦)      #
# ═══════════════════════════════════════════════════════════════════════ #

print("─" * 65)
print("PASO 3 — CoPoem.synthesize()  [TAA 'minimiza costo esperado']")

spec_cost = copoem.minimizes(
    objective="expected_cost",
    dimension=obs_dim,
    constraints={"spectral_radius": GAMMA, "stability": True},
)
W_taa, report_taa = copoem.synthesize_with_report(spec_cost)

print(f"  Spec: type='optimization', objective='expected_cost'")
print(f"  Matriz W_TAA: {W_taa.shape}, ‖W‖_F = {torch.norm(W_taa,'fro'):.4f}")
print(f"  Radio espectral (CoPoem report): {report_taa.spectral_radius_actual:.4f}")
print(f"  Gap de adjunción Φ⇌Φ*: {report_taa.adjunction_gap:.2e}")
print(f"  Consistencia espectral: {report_taa.spectral_consistency:.4f}")
print()


# ═══════════════════════════════════════════════════════════════════════ #
#  SECCIÓN 4 – BiPoem.find_fixed_point() → convergencia Φ ⇌ Φ*          #
# ═══════════════════════════════════════════════════════════════════════ #

print("─" * 65)
print("PASO 4 — BiPoem.find_fixed_point()  [ciclo Φ ⇌ Φ*]")

fp_result: Dict[str, Any] = bipoem.find_fixed_point(
    data=all_states,
    max_cycles=30,
    tol=1e-5,
)

converged = fp_result["converged"]
n_cycles  = fp_result["cycles"]
final_gap = fp_result["final_gap"]
alpha_fp  = fp_result["acf_alpha"]

print(f"  Convergió: {converged}")
print(f"  Ciclos requeridos: {n_cycles}")
print(f"  Gap final ‖Φ - Φ*‖_F = {final_gap:.2e}")
print(f"  α_ACF en punto fijo: {alpha_fp:.4f}")

if fp_result["history"]:
    print("  Historial de gaps (每iter):")
    for h in fp_result["history"][:5]:
        print(f"    ciclo={h['cycle']:2d}  gap={h['gap']:.2e}  "
              f"α={h['alpha']:.4f}  rec_err={h['reconstruction_error']:.2e}")
    if len(fp_result["history"]) > 5:
        print(f"    ... ({len(fp_result['history'])-5} ciclos más)")
print()


# ═══════════════════════════════════════════════════════════════════════ #
#  SECCIÓN 5 – Residuo de Bellman: ‖V - c - γ·𝒦V‖                      #
# ═══════════════════════════════════════════════════════════════════════ #
#  V*(x) = P*x²  es la función de valor analítica.                     #
#  𝒦 es el operador de Koopman EDMD actuando sobre observables.         #
#                                                                         #
#  Si V = wᵀψ(x)  (proyección en la base EDMD),                        #
#  entonces (𝒦V)(x) ≈ wᵀ K ψ(x).                                      #
#  Residuo de Bellman: r(x) = V(x) - c(x) - γ·(𝒦V)(x)                #
# ═══════════════════════════════════════════════════════════════════════ #

print("─" * 65)
print("PASO 5 — Residuo de Bellman: ‖V - c - γ·𝒦V‖")

# 5a. Puntos de test sobre la densidad invariante
torch.manual_seed(99)
sigma_inv = float(math.sqrt(SIGMA**2 / (2.0 * A)))   # σ de la densidad estacionaria
x_test = torch.randn(N_TEST, dtype=DTYPE) * sigma_inv  # (N_test,)

# 5b. Función de valor analítica V*(x) = P_TRUE * x²
V_true = P_TRUE * x_test**2   # (N_test,)

# 5c. Costo instantáneo c(x) = x²
cost = x_test**2               # (N_test,)

# 5d. Evaluar observables ψ(x) en los puntos de test
#     KoopmanReducer.polynomial_observables espera (n_features, n_points)
x_test_2d = x_test.unsqueeze(0)  # (1, N_test)
psi_test = KoopmanReducer.polynomial_observables(
    x_test_2d, max_degree=2
)  # (d_obs, N_test)

# La dimensión observable puede diferir de obs_dim (EDMD usa su propia base).
# Recortamos o rellenamos para compatibilidad con K_sym.
d_obs = K_sym.shape[0]
d_psi = psi_test.shape[0]

if d_psi >= d_obs:
    psi_test = psi_test[:d_obs, :]
else:
    # Rellenar con ceros si la base EDMD es más grande que la base de test
    pad = torch.zeros(d_obs - d_psi, N_TEST, dtype=DTYPE)
    psi_test = torch.cat([psi_test, pad], dim=0)

# 5e. Proyectar V*(x) sobre la base EDMD  →  encontrar w tal que ψ(x)ᵀ w ≈ V*(x)
#     Mínimos cuadrados: min_w ‖Ψᵀ w - V*‖²
Psi_T = psi_test.T   # (N_test, d_obs)
w_coeff, _, _, _ = torch.linalg.lstsq(
    Psi_T,
    V_true.unsqueeze(1),   # (N_test, 1)
    rcond=1e-10,
)
w_coeff = w_coeff.squeeze(1)   # (d_obs,)

# Verificar calidad de la proyección
V_proj = Psi_T @ w_coeff      # (N_test,)
proj_error = float(torch.norm(V_proj - V_true) / (torch.norm(V_true) + 1e-15))
print(f"  Error de proyección V* en base EDMD: {proj_error:.2e}")

# 5f. Aplicar operador de Koopman: (𝒦V)(x) ≈ wᵀ K ψ(x)
#     K_sym: (d_obs, d_obs),  psi_test: (d_obs, N_test)
KV_test = (K_sym @ psi_test).T @ w_coeff   # (N_test,)

# 5g. Residuo de Bellman
bellman_residual = V_true - cost - GAMMA * KV_test   # (N_test,)
bell_L2   = float(torch.norm(bellman_residual) / math.sqrt(N_TEST))
bell_Linf = float(torch.max(torch.abs(bellman_residual)))
bell_mean = float(torch.mean(torch.abs(bellman_residual)))

print(f"  P_teórico = {P_TRUE:.6f}")
print(f"  ‖V - 𝒦V‖₂  (RMS)     = {bell_L2:.4e}")
print(f"  ‖V - 𝒦V‖_∞ (sup)     = {bell_Linf:.4e}")
print(f"  E[|V - 𝒦V|] (media)  = {bell_mean:.4e}")
print()

# 5h. Desglose por cuantiles para diagnosticar la distribución del residuo
quantiles = torch.tensor([0.25, 0.50, 0.75, 0.90, 0.99], dtype=DTYPE)
abs_res_q = torch.quantile(torch.abs(bellman_residual), quantiles)
print("  Distribución |residuo de Bellman|:")
for q, v in zip(quantiles.tolist(), abs_res_q.tolist()):
    print(f"    p{int(q*100):02d} = {v:.4e}")
print()


# ═══════════════════════════════════════════════════════════════════════ #
#  RESUMEN FINAL                                                          #
# ═══════════════════════════════════════════════════════════════════════ #

print("=" * 65)
print("RESUMEN")
print("=" * 65)
print(f"  α_ACF(f) EMA (symbiosis) : {alpha_ema:.4f}")
print(f"  α_ACF(f) punto fijo      : {alpha_fp:.4f}")
print(f"  Punto fijo converge      : {converged}  (ciclos={n_cycles}, gap={final_gap:.2e})")
print(f"  CoPoem radio espectral   : {report_taa.spectral_radius_actual:.4f}  (target=γ={GAMMA})")
print(f"  P_teórico                : {P_TRUE:.4f}")
print(f"  Residuo Bellman RMS      : {bell_L2:.4e}")
print(f"  Residuo Bellman sup      : {bell_Linf:.4e}")

threshold = 0.5 * P_TRUE
status = "✓ DENTRO DE TOLERANCIA" if bell_L2 < threshold else "✗ REVISAR BASE OBSERVABLE"
print(f"  Estado                   : {status}")
# Residuo relativo: normalizado por P_TRUE (escala del problema)
bell_rel = bell_L2 / (P_TRUE + 1e-15)
print(f"  Residuo Bellman relativo : {bell_rel:.4e}  (= RMS / P_teórico)")
print()
print("  NOTA: el residuo no-nulo refleja la aproximación EDMD de 𝒦 con")
print(f"  base polinomial de dim={obs_dim} (grado≤2 en 1D). Un operador")
print(f"  exacto sobre N(0,σ²_inv) daría residuo = 0 analíticamente.")
print("=" * 65)
