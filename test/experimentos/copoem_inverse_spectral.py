#!/usr/bin/env python3
"""
copoem_inverse_spectral.py — El Último Paso para Alcanzar el Nivel Dios.

CoPoem Inverse Spectral Solver: Dado un espectro objetivo E(k) ~ k^{-3},
el Co-Functor Φ* diseña autónomamente la amplitud de cada modo k para
alcanzar la cascada de enstrofía k^{-3}.

  Forzamiento: Banda ancha k ∈ [4, 10]
  Control: CoPoemSpectralDesigner optimiza A(k) per-modo + ν(k) per-escala
  Objetivo: E(k) ~ k^{-3} en el rango inercial [6, 50]

Martínez's Invariant — Abril 2026
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Setup paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from poema.backends.gideon.ns_acf_coupled import CoupledNSACFSolver, CoupledNSACFConfig

# ═════════════════════════════════════════════════════════════════════════════
# Configuración del Experimento
# ═════════════════════════════════════════════════════════════════════════════

config = CoupledNSACFConfig(
    N=256,
    Re=10000.0,
    nu4_coeff=1e-12,
    k_force=4,                    # Legacy: initial forcing scale
    force_amplitude=15.0,         # Per-mode initial amplitude
    friction=0.02,
    dt_init=1e-3,
    cfl_target=0.3,
    T_total=25.0,                 # 25 time units
    analysis_interval=1.5,        # CoPoem control every 1.5s
    n_koopman_modes=60,
    adaptive_hyper=True,
    auto_evolve=True,
    snapshot_interval=0.05,
    spinup_fraction=0.3,

    # Multi-Oracle AMR (runs alongside CoPoem for diagnostics)
    use_multi_oracle=True,
    d_target=15,
    h_ks_target=0.1,

    # CoPoem Inverse Spectral Design — THE KEY
    use_copoem=True,
    k_force_min=4,
    k_force_max=10,
    copoem_learning_rate=0.4,
    copoem_target_slope=-3.0,
    copoem_max_power=5000.0,
)


# ═════════════════════════════════════════════════════════════════════════════
# Ejecutar Simulación
# ═════════════════════════════════════════════════════════════════════════════

print("╔" + "═" * 68 + "╗")
print("║  COPOEM INVERSE SPECTRAL SOLVER — EL ÚLTIMO PASO AL NIVEL DIOS   ║")
print("║  Co-Functor Φ*: E_target(k) ~ k^{-3} → F(k), ν(k) autónomos    ║")
print("╚" + "═" * 68 + "╝")
print()
print(f"  Grid: {config.N}×{config.N}, Re = {config.Re:.0f}")
print(f"  Broadband forcing: k ∈ [{config.k_force_min}, {config.k_force_max}]")
print(f"  Target slope: {config.copoem_target_slope}")
print(f"  Total power budget: {config.copoem_max_power}")
print(f"  T_total = {config.T_total}, control interval = {config.analysis_interval}")
print()

t_start = time.time()
solver = CoupledNSACFSolver(config)
result = solver.simulate()
t_total = time.time() - t_start


# ═════════════════════════════════════════════════════════════════════════════
# Análisis de Resultados
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  ANÁLISIS DE RESULTADOS — CoPoem Inverse Spectral Design")
print("=" * 70)

# Energy spectrum
k_bins = result["k_bins"]
E_k = result["E_k"]

# Compute final slope in inertial range [6, 50]
mask_ir = (k_bins >= 6) & (k_bins <= 50) & (E_k > 1e-20)
if np.sum(mask_ir) > 5:
    log_k = np.log(k_bins[mask_ir])
    log_E = np.log(E_k[mask_ir])
    slope_final, intercept = np.polyfit(log_k, log_E, 1)
else:
    slope_final = -99.0
    intercept = 0.0

# CoPoem history analysis
copoem_hist = result.get("copoem_history", [])
designer_hist = result.get("copoem_designer_history", [])

if copoem_hist:
    misfits = [h["misfit"] for h in copoem_hist]
    slopes = [h["slope"] for h in copoem_hist]
    gaps = [h["adjunction_gap"] for h in copoem_hist]
    powers = [h["total_power"] for h in copoem_hist]
    d95s = [h["d_95"] for h in copoem_hist]
    phases = [h["phase"] for h in copoem_hist]

    print(f"\n  CoPoem control cycles: {len(copoem_hist)}")
    print(f"  Misfit J: {misfits[0]:.3f} → {misfits[-1]:.3f} "
          f"(reduction: {misfits[0]/max(misfits[-1], 1e-10):.1f}×)")
    print(f"  Slope: {slopes[0]:.2f} → {slopes[-1]:.2f} (target: -3.00)")
    print(f"  Adjunction gap: {gaps[0]:.3f} → {gaps[-1]:.3f}")
    print(f"  d_95: {d95s[0]} → {d95s[-1]}")
    print(f"  Total power: {powers[0]:.0f} → {powers[-1]:.0f}")
    print(f"  Final phase: {phases[-1]}")

    # Show amplitude evolution
    if copoem_hist[-1].get("amplitudes"):
        amps_final = copoem_hist[-1]["amplitudes"]
        k_shells = list(range(config.k_force_min, config.k_force_max + 1))
        print(f"\n  Amplitudes per-mode A(k) [final]:")
        for k, a in zip(k_shells, amps_final):
            bar = "█" * int(a / 3)
            print(f"    k={k:2d}: A={a:6.1f} {bar}")

# Koopman results
koopman_results = result.get("koopman_results", [])
if koopman_results:
    last_k = koopman_results[-1]
    print(f"\n  Koopman analysis: {len(koopman_results)} total")
    print(f"  Final d_95 = {last_k.d_95}, n_coherent = {last_k.n_coherent}")
    if hasattr(last_k, 'eigenvalues') and last_k.eigenvalues is not None:
        spec_rad = np.max(np.abs(last_k.eigenvalues))
        print(f"  Spectral radius: {spec_rad:.4f}")

# Energy and enstrophy
E_final = result["energy"][-1] if len(result["energy"]) > 0 else 0
Z_final = result["enstrophy"][-1] if len(result["enstrophy"]) > 0 else 0
print(f"\n  Energy: {E_final:.4f}")
print(f"  Enstrophy: {Z_final:.4f}")
print(f"  Spectral slope (inertial [6,50]): {slope_final:.2f}")
print(f"  Total time: {t_total:.1f}s ({result['n_steps']} steps)")


# ═════════════════════════════════════════════════════════════════════════════
# Visualización — 10 paneles
# ═════════════════════════════════════════════════════════════════════════════

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(
        "CoPoem Inverse Spectral Solver — Co-Functor Φ*: E(k) ~ k⁻³",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel 1: Final vorticity field ──
    ax1 = fig.add_subplot(gs[0, 0])
    if len(result["snapshots"]) > 0:
        omega_final = result["snapshots"][-1]
        vmax = np.percentile(np.abs(omega_final), 99)
        ax1.imshow(omega_final, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax1.set_title("Vorticity ω(x,y) [final]", fontsize=11)
    else:
        ax1.text(0.5, 0.5, "No snapshots", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # ── Panel 2: Energy spectrum E(k) with k^{-3} reference ──
    ax2 = fig.add_subplot(gs[0, 1])
    valid = E_k > 1e-25
    ax2.loglog(k_bins[valid], E_k[valid], "b-", lw=2, label="E(k) actual")
    # k^{-3} reference
    k_ref = k_bins[(k_bins >= 5) & (k_bins <= 60)]
    if len(k_ref) > 0:
        E_ref = np.exp(intercept) * k_ref ** slope_final
        ax2.loglog(k_ref, E_ref, "r--", lw=1.5, alpha=0.7,
                   label=f"Fit: k^{{{slope_final:.2f}}}")
        # True k^{-3}
        C_target = np.exp(intercept) * k_ref[0] ** (slope_final + 3) * k_ref[0] ** (-3)
        E_target_ref = C_target * k_ref ** (-3)
        ax2.loglog(k_ref, E_target_ref, "g:", lw=1.5, alpha=0.7, label="k⁻³ target")
    # Shade forcing band
    ax2.axvspan(config.k_force_min, config.k_force_max, alpha=0.1, color="orange",
                label=f"Forcing k∈[{config.k_force_min},{config.k_force_max}]")
    ax2.set_xlabel("k")
    ax2.set_ylabel("E(k)")
    ax2.set_title(f"Energy Spectrum (slope = {slope_final:.2f})", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Per-mode amplitudes A(k) ──
    ax3 = fig.add_subplot(gs[0, 2])
    if designer_hist:
        k_shells = designer_hist[-1]["k_shells"]
        # Plot evolution of amplitudes
        n_hist = len(designer_hist)
        colors = plt.cm.viridis(np.linspace(0, 1, n_hist))
        for i, dh in enumerate(designer_hist):
            alpha = 0.3 + 0.7 * (i / max(n_hist - 1, 1))
            ax3.plot(dh["k_shells"], dh["amplitudes"], "o-", color=colors[i],
                     alpha=alpha, markersize=4, lw=1)
        # Highlight final
        ax3.plot(k_shells, designer_hist[-1]["amplitudes"], "ro-", lw=2.5,
                 markersize=8, label="Final A(k)", zorder=10)
        ax3.set_xlabel("k (shell)")
        ax3.set_ylabel("A(k)")
        ax3.set_title("Per-mode Amplitudes A(k)", fontsize=11)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No CoPoem data", ha="center", va="center", transform=ax3.transAxes)

    # ── Panel 4: Misfit J(t) ──
    ax4 = fig.add_subplot(gs[1, 0])
    if copoem_hist:
        t_cp = [h["t"] for h in copoem_hist]
        ax4.semilogy(t_cp, misfits, "b-o", lw=2, markersize=5)
        ax4.axhline(0.3, ls="--", color="green", alpha=0.5, label="Threshold")
        ax4.set_xlabel("t")
        ax4.set_ylabel("Misfit J")
        ax4.set_title("Spectral Misfit J(t)", fontsize=11)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

    # ── Panel 5: Slope evolution ──
    ax5 = fig.add_subplot(gs[1, 1])
    if copoem_hist:
        ax5.plot(t_cp, slopes, "b-o", lw=2, markersize=5, label="Actual slope")
        ax5.axhline(-3.0, ls="--", color="red", lw=2, label="Target: -3.0")
        ax5.fill_between(t_cp, -3.5, -2.5, alpha=0.1, color="green", label="±0.5 band")
        ax5.set_xlabel("t")
        ax5.set_ylabel("Spectral slope")
        ax5.set_title("Slope → k⁻³ Convergence", fontsize=11)
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

    # ── Panel 6: Adjunction gap ──
    ax6 = fig.add_subplot(gs[1, 2])
    if copoem_hist:
        ax6.semilogy(t_cp, gaps, "m-o", lw=2, markersize=5)
        ax6.set_xlabel("t")
        ax6.set_ylabel("Adjunction gap ||E-E*||/||E*||")
        ax6.set_title("CoPoem Adjunction Gap Φ ⇌ Φ*", fontsize=11)
        ax6.grid(True, alpha=0.3)

    # ── Panel 7: d_95(t) ──
    ax7 = fig.add_subplot(gs[2, 0])
    if copoem_hist:
        ax7.plot(t_cp, d95s, "g-o", lw=2, markersize=5)
        ax7.axhline(config.d_target, ls="--", color="red", alpha=0.5, label=f"d_target={config.d_target}")
        ax7.set_xlabel("t")
        ax7.set_ylabel("d_95")
        ax7.set_title("Koopman Dimension d₉₅(t)", fontsize=11)
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)

    # ── Panel 8: ν(k) profile ──
    ax8 = fig.add_subplot(gs[2, 1])
    if designer_hist:
        k_shells = designer_hist[-1]["k_shells"]
        nu_prof = designer_hist[-1]["nu_profile"]
        ax8.bar(k_shells, nu_prof, color="steelblue", alpha=0.7, width=0.6)
        ax8.axhline(1.0, ls="--", color="gray", alpha=0.5, label="ν_base")
        ax8.set_xlabel("k (shell)")
        ax8.set_ylabel("ν_mult(k)")
        ax8.set_title("Viscosity Profile ν(k)/ν₄_base", fontsize=11)
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3)

    # ── Panel 9: Phase diagram A(k) × misfit ──
    ax9 = fig.add_subplot(gs[2, 2])
    if designer_hist:
        mean_amps = [np.mean(dh["amplitudes"]) for dh in designer_hist]
        mf = [dh["misfit"] for dh in designer_hist]
        # Color by phase
        phase_colors = {
            "ramping": "blue",
            "sculpting": "orange",
            "converged": "green",
        }
        for i in range(len(designer_hist)):
            c = phase_colors.get(designer_hist[i]["phase"], "gray")
            ax9.scatter(mean_amps[i], mf[i], c=c, s=40 + i * 5, zorder=3)
        ax9.set_xlabel("Mean A(k)")
        ax9.set_ylabel("Misfit J")
        ax9.set_title("Phase Diagram: Ā × J", fontsize=11)
        # Add phase labels
        for ph, c in phase_colors.items():
            ax9.scatter([], [], c=c, label=ph)
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3)

    # ── Panel 10: Energy & Enstrophy evolution ──
    ax10 = fig.add_subplot(gs[3, 0])
    times = result["times"]
    if len(times) > 0:
        ax10.plot(times, result["energy"], "b-", lw=1.5, label="Energy E(t)")
        ax10_twin = ax10.twinx()
        ax10_twin.plot(times, result["enstrophy"], "r-", lw=1.5, label="Enstrophy Z(t)")
        ax10.set_xlabel("t")
        ax10.set_ylabel("Energy E", color="blue")
        ax10_twin.set_ylabel("Enstrophy Z", color="red")
        ax10.set_title("Energy & Enstrophy", fontsize=11)
        lines1, labels1 = ax10.get_legend_handles_labels()
        lines2, labels2 = ax10_twin.get_legend_handles_labels()
        ax10.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax10.grid(True, alpha=0.3)

    # ── Panel 11: Total power evolution ──
    ax11 = fig.add_subplot(gs[3, 1])
    if copoem_hist:
        ax11.plot(t_cp, powers, "k-o", lw=2, markersize=5)
        ax11.axhline(config.copoem_max_power, ls="--", color="red", alpha=0.5,
                     label=f"P_max={config.copoem_max_power}")
        ax11.set_xlabel("t")
        ax11.set_ylabel("Σ A(k)²")
        ax11.set_title("Total Forcing Power", fontsize=11)
        ax11.legend(fontsize=9)
        ax11.grid(True, alpha=0.3)

    # ── Panel 12: Summary text ──
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.axis("off")
    summary_text = (
        f"CoPoem Inverse Spectral Solver\n"
        f"{'─' * 35}\n"
        f"Grid: {config.N}² = {config.N**2:,} DOFs\n"
        f"Re = {config.Re:.0f}, T = {config.T_total}\n"
        f"Forcing: k ∈ [{config.k_force_min}, {config.k_force_max}]\n"
        f"{'─' * 35}\n"
        f"Target slope: {config.copoem_target_slope}\n"
        f"Final slope: {slope_final:.3f}\n"
        f"Slope error: {abs(slope_final - config.copoem_target_slope):.3f}\n"
    )
    if copoem_hist:
        summary_text += (
            f"{'─' * 35}\n"
            f"Misfit: {misfits[0]:.3f} → {misfits[-1]:.3f}\n"
            f"Adj. gap: {gaps[0]:.3f} → {gaps[-1]:.3f}\n"
            f"d_95: {d95s[0]} → {d95s[-1]}\n"
            f"Final phase: {phases[-1]}\n"
            f"Control cycles: {len(copoem_hist)}\n"
        )
    summary_text += (
        f"{'─' * 35}\n"
        f"Time: {t_total:.1f}s ({result['n_steps']} steps)\n"
    )
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.savefig(str(ROOT / "copoem_inverse_spectral_results.png"), dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: copoem_inverse_spectral_results.png")
    plt.close()

except ImportError:
    print("  matplotlib not available — skipping plots")
except Exception as e:
    print(f"  Plot error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# Guardar resultados JSON
# ═════════════════════════════════════════════════════════════════════════════

json_result = {
    "experiment": "CoPoem Inverse Spectral Solver",
    "config": {
        "N": config.N,
        "Re": config.Re,
        "T_total": config.T_total,
        "k_force_min": config.k_force_min,
        "k_force_max": config.k_force_max,
        "target_slope": config.copoem_target_slope,
        "max_power": config.copoem_max_power,
        "learning_rate": config.copoem_learning_rate,
        "analysis_interval": config.analysis_interval,
        "d_target": config.d_target,
    },
    "results": {
        "slope_final": float(slope_final),
        "slope_error": float(abs(slope_final - config.copoem_target_slope)),
        "energy_final": float(E_final),
        "enstrophy_final": float(Z_final),
        "n_koopman_analyses": len(koopman_results),
        "n_copoem_cycles": len(copoem_hist),
        "elapsed_seconds": float(t_total),
        "n_steps": result["n_steps"],
    },
    "copoem_evolution": copoem_hist,
    "designer_evolution": designer_hist,
}

if copoem_hist:
    json_result["results"]["misfit_initial"] = float(misfits[0])
    json_result["results"]["misfit_final"] = float(misfits[-1])
    json_result["results"]["adjunction_gap_final"] = float(gaps[-1])
    json_result["results"]["d_95_final"] = int(d95s[-1])
    json_result["results"]["final_phase"] = phases[-1]
    json_result["results"]["final_amplitudes"] = copoem_hist[-1].get("amplitudes", [])

with open(ROOT / "copoem_inverse_spectral_results.json", "w") as f:
    json.dump(json_result, f, indent=2, default=str)
print(f"  JSON saved: copoem_inverse_spectral_results.json")


# ═════════════════════════════════════════════════════════════════════════════
# Reporte Final
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "╔" + "═" * 68 + "╗")
print("║  RESULTADO FINAL — CoPoem Inverse Spectral Solver                ║")
print("╚" + "═" * 68 + "╝")
print(f"  Espectro objetivo: E(k) ~ k^{{{config.copoem_target_slope}}}")
print(f"  Espectro alcanzado: E(k) ~ k^{{{slope_final:.3f}}}")
print(f"  Error de pendiente: {abs(slope_final - config.copoem_target_slope):.3f}")
if copoem_hist:
    print(f"  Misfit espectral: {misfits[-1]:.4f}")
    print(f"  Adjunction gap Φ ⇌ Φ*: {gaps[-1]:.4f}")
    print(f"  Fase final: {phases[-1]}")
print(f"  Tiempo total: {t_total:.1f}s")
print()
