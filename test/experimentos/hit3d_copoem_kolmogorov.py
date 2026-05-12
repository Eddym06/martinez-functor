#!/usr/bin/env python3
"""
hit3d_copoem_kolmogorov.py — 3D HIT con CoPoem → k^{-5/3}

El experimento definitivo: Cascada directa de energía en 3D con
CoPoem esculpiendo autónomamente el espectro de Kolmogorov.

Grid: 64³ (262,144 DOFs)
Re = 1600, Forzamiento broadband k ∈ [2, 6]
Target: E(k) ~ k^{-5/3}

Martínez's Invariant — Abril 2026
"""

import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ─── Paths ───
sys.path.insert(0, "/home/Martínez's Invariant")

from poema.backends.gideon.ns3d_hit_solver import HIT3DSolver, HIT3DConfig


def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  3D HIT + CoPoem — KOLMOGOROV k^{-5/3} SPECTRAL DESIGN          ║")
    print("║  Cascada directa de energía en turbulencia 3D isótropa           ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    cfg = HIT3DConfig(
        N=64,
        Re=1600.0,
        nu4_coeff=1e-14,
        T_total=10.0,
        cfl_target=0.4,
        force_amplitude=100.0,
        # CoPoem
        use_copoem=True,
        k_force_min=2,
        k_force_max=6,
        copoem_target_slope=-5.0 / 3.0,
        copoem_learning_rate=0.3,
        copoem_max_power=100000.0,
        # Analysis
        analysis_interval=1.0,
        snapshot_interval=0.1,
        n_koopman_modes=40,
    )

    print(f"  Grid: {cfg.N}³ = {cfg.N**3:,} DOFs")
    print(f"  Re = {cfg.Re:.0f}, ν = {1/cfg.Re:.6f}")
    print(f"  Target: E(k) ~ k^{{-5/3}} = k^{{{cfg.copoem_target_slope:.4f}}}")
    print(f"  Broadband forcing: k ∈ [{cfg.k_force_min}, {cfg.k_force_max}]")
    print(f"  T_total = {cfg.T_total}, P_max = {cfg.copoem_max_power}")
    print()

    solver = HIT3DSolver(config=cfg)
    results = solver.simulate()

    # ── Analysis ──
    print()
    print("=" * 70)
    print("  ANÁLISIS DE RESULTADOS — 3D HIT CoPoem Kolmogorov")
    print("=" * 70)

    k_bins = results["k_bins"]
    E_k = results["E_k"]

    # Compute final slope in inertial range [k_force_max+2, N/3]
    k_ir_min = cfg.k_force_max + 2
    k_ir_max = cfg.N // 3
    ir_mask = (k_bins >= k_ir_min) & (k_bins <= k_ir_max) & (E_k > 1e-25)
    slope_final = -99.0
    if np.sum(ir_mask) >= 5:
        log_k = np.log(k_bins[ir_mask])
        log_E = np.log(E_k[ir_mask])
        slope_final, intercept = np.polyfit(log_k, log_E, 1)

    # CoPoem history
    ch = results["copoem_history"]
    n_cycles = len(ch)

    if n_cycles > 0:
        misfits = [c["misfit"] for c in ch]
        slopes = [c["slope"] for c in ch]
        gaps = [c["adjunction_gap"] for c in ch]
        d95s = [c["d_95"] for c in ch]
        powers = [c["total_power"] for c in ch]
        phases = [c["phase"] for c in ch]
        amps_final = ch[-1]["amplitudes"]

        print(f"\n  CoPoem control cycles: {n_cycles}")
        print(f"  Misfit J: {misfits[0]:.3f} → {misfits[-1]:.3f} (reduction: {misfits[0]/max(misfits[-1],1e-10):.1f}×)")
        print(f"  Slope: {slopes[0]:.2f} → {slopes[-1]:.2f} (target: {cfg.copoem_target_slope:.4f})")
        print(f"  Adjunction gap: {gaps[0]:.3f} → {gaps[-1]:.3f}")
        print(f"  d_95: {d95s[0]} → {d95s[-1]}")
        print(f"  Total power: {powers[0]:.0f} → {powers[-1]:.0f}")
        print(f"  Final phase: {phases[-1]}")

        slope_error = abs(slopes[-1] - cfg.copoem_target_slope)
        print(f"\n  Slope error |s - (-5/3)|: {slope_error:.3f}")

        print(f"\n  Amplitudes per-mode A(k) [final]:")
        for i, k_shell in enumerate(range(cfg.k_force_min, cfg.k_force_max + 1)):
            if i < len(amps_final):
                bar = "█" * max(1, int(amps_final[i] / 1.5))
                print(f"    k={k_shell:2d}: A={amps_final[i]:6.1f} {bar}")
    else:
        misfits = []
        slopes = []
        gaps = []
        d95s = []
        powers = []
        phases = []
        amps_final = []

    print(f"\n  Re_λ (Taylor-scale): {results['Re_lambda']:.1f}")
    print(f"  Energy: {results['energy'][-1]:.6f}" if len(results['energy']) > 0 else "")
    print(f"  Enstrophy: {results['enstrophy'][-1]:.6f}" if len(results['enstrophy']) > 0 else "")
    print(f"  Spectral slope (inertial [{k_ir_min},{k_ir_max}]): {slope_final:.3f}")
    print(f"  Total time: {results['elapsed_total']:.1f}s ({results['n_steps']} steps)")

    # ── Visualization ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(22, 16))
        fig.suptitle(
            f"3D HIT CoPoem — Kolmogorov k$^{{-5/3}}$ Spectral Design\n"
            f"N={cfg.N}³, Re={cfg.Re:.0f}, k∈[{cfg.k_force_min},{cfg.k_force_max}], "
            f"T={cfg.T_total}",
            fontsize=16, fontweight="bold",
        )
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

        # 1. Energy spectrum E(k) with Kolmogorov reference
        ax1 = fig.add_subplot(gs[0, 0])
        valid = E_k > 1e-30
        ax1.loglog(k_bins[valid], E_k[valid], "b-", lw=2, label="E(k) final")
        # k^{-5/3} reference
        k_ref = k_bins[(k_bins >= k_ir_min) & (k_bins <= k_ir_max)]
        if len(k_ref) > 0 and slope_final > -50:
            E_ref = np.exp(intercept) * k_ref ** slope_final if slope_final > -50 else k_ref ** (-5/3)
            ax1.loglog(k_ref, E_ref, "r--", lw=1.5, alpha=0.7, label=f"fit: k$^{{{slope_final:.2f}}}$")
        # Kolmogorov line
        C_K = E_k[valid][len(E_k[valid])//4] * k_bins[valid][len(k_bins[valid])//4]**(5/3) if np.any(valid) else 1.0
        k_kol = k_bins[(k_bins >= 2) & (k_bins <= k_ir_max)]
        ax1.loglog(k_kol, C_K * k_kol ** (-5/3), "g:", lw=1, alpha=0.5, label="k$^{-5/3}$")
        ax1.set_xlabel("k")
        ax1.set_ylabel("E(k)")
        ax1.set_title("Energy Spectrum")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Compensated spectrum: E(k) * k^{5/3}
        ax2 = fig.add_subplot(gs[0, 1])
        comp = E_k * k_bins ** (5/3)
        valid_comp = comp > 1e-30
        ax2.semilogx(k_bins[valid_comp], comp[valid_comp], "b-", lw=2)
        ax2.axhline(y=np.median(comp[(k_bins >= k_ir_min) & (k_bins <= k_ir_max) & valid_comp]) if np.any((k_bins >= k_ir_min) & valid_comp) else 0,
                     color="r", ls="--", alpha=0.5, label="median plateau")
        ax2.set_xlabel("k")
        ax2.set_ylabel("E(k) · k$^{5/3}$")
        ax2.set_title("Compensated Spectrum")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Amplitudes A(k) evolution
        ax3 = fig.add_subplot(gs[0, 2])
        if n_cycles > 1:
            t_cp = [c["t"] for c in ch]
            all_amps = np.array([c["amplitudes"] for c in ch])
            for i in range(all_amps.shape[1]):
                k_shell = cfg.k_force_min + i
                ax3.plot(t_cp, all_amps[:, i], "-o", ms=3, label=f"k={k_shell}")
            ax3.set_xlabel("t")
            ax3.set_ylabel("A(k)")
            ax3.set_title("Per-mode Amplitudes")
            ax3.legend(fontsize=7, ncol=2)
        ax3.grid(True, alpha=0.3)

        # 4. Misfit J(t)
        ax4 = fig.add_subplot(gs[0, 3])
        if n_cycles > 1:
            ax4.semilogy([c["t"] for c in ch], misfits, "r-o", ms=3)
        ax4.set_xlabel("t")
        ax4.set_ylabel("J (misfit)")
        ax4.set_title("Spectral Misfit J(t)")
        ax4.grid(True, alpha=0.3)

        # 5. Slope convergence
        ax5 = fig.add_subplot(gs[1, 0])
        if n_cycles > 1:
            ax5.plot([c["t"] for c in ch], slopes, "b-o", ms=3, label="slope(t)")
            ax5.axhline(y=-5/3, color="g", ls="--", lw=2, label="−5/3 (Kolmogorov)")
            ax5.fill_between([ch[0]["t"], ch[-1]["t"]],
                             -5/3 - 0.3, -5/3 + 0.3, alpha=0.1, color="g")
        ax5.set_xlabel("t")
        ax5.set_ylabel("Spectral slope")
        ax5.set_title("Slope → k$^{-5/3}$")
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # 6. Adjunction gap
        ax6 = fig.add_subplot(gs[1, 1])
        if n_cycles > 1:
            ax6.plot([c["t"] for c in ch], gaps, "m-o", ms=3)
        ax6.set_xlabel("t")
        ax6.set_ylabel("||Φ∘Φ* − Id||")
        ax6.set_title("Adjunction Gap")
        ax6.grid(True, alpha=0.3)

        # 7. d_95
        ax7 = fig.add_subplot(gs[1, 2])
        if n_cycles > 1:
            ax7.plot([c["t"] for c in ch], d95s, "k-o", ms=3)
        ax7.set_xlabel("t")
        ax7.set_ylabel("d_95")
        ax7.set_title("Koopman d$_{95}$")
        ax7.grid(True, alpha=0.3)

        # 8. Energy & Enstrophy
        ax8 = fig.add_subplot(gs[1, 3])
        t_arr = results["times"]
        if len(t_arr) > 0:
            ax8_twin = ax8.twinx()
            ax8.plot(t_arr, results["energy"], "b-", lw=1, label="E(t)")
            ax8_twin.plot(t_arr, results["enstrophy"], "r-", lw=1, label="Z(t)")
            ax8.set_xlabel("t")
            ax8.set_ylabel("Energy E", color="b")
            ax8_twin.set_ylabel("Enstrophy Z", color="r")
            ax8.set_title("Energy & Enstrophy")
        ax8.grid(True, alpha=0.3)

        # 9. Total power
        ax9 = fig.add_subplot(gs[2, 0])
        if n_cycles > 1:
            ax9.plot([c["t"] for c in ch], powers, "g-o", ms=3)
            ax9.axhline(y=cfg.copoem_max_power, color="r", ls="--", alpha=0.5, label=f"P_max={cfg.copoem_max_power}")
        ax9.set_xlabel("t")
        ax9.set_ylabel("Σ A(k)²")
        ax9.set_title("Total Forcing Power")
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3)

        # 10. Phase diagram
        ax10 = fig.add_subplot(gs[2, 1])
        if n_cycles > 1:
            mean_amps = [np.mean(c["amplitudes"]) for c in ch]
            colors = {"ramping": "orange", "sculpting": "blue", "converged": "green"}
            for i in range(len(ch)):
                c = colors.get(phases[i], "gray")
                ax10.scatter(mean_amps[i], misfits[i], c=c, s=30, zorder=5)
            # Legend
            for ph, col in colors.items():
                ax10.scatter([], [], c=col, s=30, label=ph)
            ax10.set_xlabel("Mean A(k)")
            ax10.set_ylabel("Misfit J")
            ax10.set_title("Phase Diagram")
            ax10.legend(fontsize=8)
        ax10.grid(True, alpha=0.3)

        # 11. Re_lambda evolution (approximate from E, Z series)
        ax11 = fig.add_subplot(gs[2, 2])
        if len(t_arr) > 10:
            E_arr = results["energy"]
            Z_arr = results["enstrophy"]
            nu = 1.0 / cfg.Re
            Re_l = []
            for i in range(len(E_arr)):
                e, z = E_arr[i], Z_arr[i]
                if z > 1e-30:
                    u2 = 2 * e / 3
                    eps = 2 * nu * z
                    Re_l.append(np.sqrt(15 * u2 / max(nu * eps, 1e-30)) * np.sqrt(u2))
                else:
                    Re_l.append(0)
            ax11.plot(t_arr, Re_l, "b-", lw=1)
            ax11.set_xlabel("t")
            ax11.set_ylabel("Re$_\\lambda$")
            ax11.set_title("Taylor-scale Reynolds")
        ax11.grid(True, alpha=0.3)

        # 12. Summary text
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.axis("off")
        summary = (
            f"RESULTADO FINAL\n"
            f"{'─'*30}\n"
            f"Grid: {cfg.N}³ = {cfg.N**3:,} DOFs\n"
            f"Re = {cfg.Re:.0f}\n"
            f"Re_λ = {results['Re_lambda']:.1f}\n"
            f"Target: k^{{-5/3}} = k^{{{-5/3:.4f}}}\n"
            f"Slope final: {slope_final:.3f}\n"
            f"Slope error: {abs(slope_final - cfg.copoem_target_slope):.3f}\n"
            f"Misfit J: {misfits[-1]:.3f}\n" if misfits else ""
            f"Gap Φ⇌Φ*: {gaps[-1]:.3f}\n" if gaps else ""
            f"d_95: {d95s[-1]}\n" if d95s else ""
            f"Phase: {phases[-1]}\n" if phases else ""
            f"Cycles: {n_cycles}\n"
            f"Time: {results['elapsed_total']:.1f}s\n"
            f"Steps: {results['n_steps']}"
        )
        ax12.text(0.05, 0.95, summary, transform=ax12.transAxes,
                  fontsize=10, verticalalignment="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        plt.savefig("hit3d_copoem_kolmogorov_results.png", dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved: hit3d_copoem_kolmogorov_results.png")
        plt.close()
    except Exception as e:
        print(f"\n  ⚠ Plot error: {e}")

    # ── JSON ──
    json_data = {
        "experiment": "3D HIT CoPoem Kolmogorov",
        "config": {
            "N": cfg.N,
            "Re": cfg.Re,
            "T_total": cfg.T_total,
            "k_force_min": cfg.k_force_min,
            "k_force_max": cfg.k_force_max,
            "target_slope": cfg.copoem_target_slope,
            "max_power": cfg.copoem_max_power,
            "learning_rate": cfg.copoem_learning_rate,
        },
        "results": {
            "slope_final": slope_final,
            "slope_error": abs(slope_final - cfg.copoem_target_slope),
            "Re_lambda": results["Re_lambda"],
            "energy_final": float(results["energy"][-1]) if len(results["energy"]) > 0 else 0,
            "enstrophy_final": float(results["enstrophy"][-1]) if len(results["enstrophy"]) > 0 else 0,
            "n_copoem_cycles": n_cycles,
            "misfit_initial": misfits[0] if misfits else None,
            "misfit_final": misfits[-1] if misfits else None,
            "adjunction_gap_final": gaps[-1] if gaps else None,
            "d_95_final": d95s[-1] if d95s else None,
            "final_phase": phases[-1] if phases else None,
            "final_amplitudes": amps_final,
            "elapsed_seconds": results["elapsed_total"],
            "n_steps": results["n_steps"],
        },
        "copoem_evolution": results["copoem_history"],
        "designer_evolution": results.get("copoem_designer_history", []),
    }
    with open("hit3d_copoem_kolmogorov_results.json", "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  JSON saved: hit3d_copoem_kolmogorov_results.json")

    # ── Final banner ──
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  RESULTADO FINAL — 3D HIT CoPoem Kolmogorov                     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"  Target: E(k) ~ k^{{{cfg.copoem_target_slope:.4f}}}")
    print(f"  Achieved: E(k) ~ k^{{{slope_final:.3f}}}")
    print(f"  Slope error: {abs(slope_final - cfg.copoem_target_slope):.3f}")
    if misfits:
        print(f"  Misfit: {misfits[-1]:.3f}")
    if gaps:
        print(f"  Adjunction gap: {gaps[-1]:.3f}")
    print(f"  Re_λ: {results['Re_lambda']:.1f}")
    print(f"  Time: {results['elapsed_total']:.1f}s")


if __name__ == "__main__":
    main()
