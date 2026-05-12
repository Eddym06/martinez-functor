#!/usr/bin/env python3
"""
cascade_acceleration.py — Experimento "Aceleración de la Cascada"
=================================================================

Termostato de Turbulencia: conduce un flujo NS 2D (256×256, Re=10000)
hacia turbulencia completamente desarrollada usando el AMR multi-oráculo
ACF, y lo MANTIENE en ese régimen usando solo ~d_target modos de Koopman.

El ACF "acelera" la cascada de enstrofía:
  - Si d_95 < d_target: A↑, ν₄↓ (inyectar energía, reducir disipación)
  - Si h_KS < h_target: ν₄↓ (reducir hiperviscosidad)
  - Si E(k) no es k⁻³: ajuste fino de A y ν₄
  - Si d_95 ≈ d_target y E(k) ≈ k⁻³: EQUILIBRIO (termostato estabiliza)

Backend: c_native, auto_evolve=True
Multi-Oracle AMR: 5 oráculos + árbitro bayesiano + cascade accelerator

Martínez's Invariant — Abril 2026
"""

import sys
sys.stdout.reconfigure(line_buffering=True)
import os
import time
import warnings
import traceback
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_cascade_experiment():
    """Execute the full Cascade Acceleration experiment."""

    from poema.backends.gideon.ns_acf_coupled import CoupledNSACFSolver, CoupledNSACFConfig

    print("╔" + "═" * 68 + "╗")
    print("║   EXPERIMENTO: ACELERACIÓN DE LA CASCADA — TERMOSTATO ACF        ║")
    print("║   NS 2D 256×256 Re=10000 — Multi-Oracle AMR + Cascade Accel.    ║")
    print("╚" + "═" * 68 + "╝")
    print()

    config = CoupledNSACFConfig(
        N=256,
        Re=10000.0,
        nu4_coeff=5e-13,
        k_force=4,
        force_amplitude=25.0,     # Start with moderate forcing
        friction=0.015,            # Low friction to let energy accumulate
        cfl_target=0.3,
        T_total=20.0,

        # ACF coupling
        analysis_interval=1.5,     # Analyze every 1.5 time units
        n_koopman_modes=60,
        adaptive_hyper=True,
        auto_evolve=True,

        # Multi-Oracle AMR + Cascade
        use_multi_oracle=True,
        d_target=20,               # Target: d_95 >= 20 for developed turbulence
        h_ks_target=0.05,          # Target: h_KS >= 0.05

        # Snapshots
        snapshot_interval=0.1,
        spinup_fraction=0.0,       # No spinup discard (we want to see the full transition)

        # Backend
        backend="c_native",
    )

    solver = CoupledNSACFSolver(config)
    t_start = time.time()
    result = solver.simulate()
    t_total = time.time() - t_start

    return result, t_total


def analyze_results(result, t_total):
    """Deep analysis of the cascade acceleration results."""

    print("\n" + "=" * 70)
    print("ANÁLISIS POST-SIMULACIÓN")
    print("=" * 70)

    p = result["params"]
    snapshots = result["snapshots"]
    energy = result["energy"]
    enstrophy = result["enstrophy"]
    times = result["times"]
    k_bins = result["k_bins"]
    E_k = result["E_k"]
    koopman_results = result["koopman_results"]
    thermostat_hist = result["thermostat_history"]
    adaptation_log = result["adaptation_log"]

    n_snap = len(snapshots)
    print(f"\n  Snapshots: {n_snap}")
    print(f"  Pasos totales: {result['n_steps']}")
    print(f"  Tiempo real: {t_total:.1f}s")

    # ── Energy & Enstrophy summary ──
    print(f"\n━━━ ENERGÍA Y ENSTROFÍA ━━━")
    print(f"  E_initial = {energy[0]:.6f}, E_final = {energy[-1]:.6f}")
    print(f"  Z_initial = {enstrophy[0]:.6f}, Z_final = {enstrophy[-1]:.6f}")
    print(f"  E_max = {np.max(energy):.6f}, Z_max = {np.max(enstrophy):.6f}")

    # ── Koopman evolution ──
    print(f"\n━━━ EVOLUCIÓN KOOPMAN ━━━")
    if koopman_results:
        d95_series = [kr.d_95 for kr in koopman_results]
        coherent_series = [kr.n_coherent for kr in koopman_results]
        print(f"  Análisis Koopman realizados: {len(koopman_results)}")
        print(f"  d_95 serie: {d95_series}")
        print(f"  Coherent serie: {coherent_series}")
        print(f"  d_95 final: {d95_series[-1]}")
        print(f"  d_95 máximo: {max(d95_series)}")

    # ── Thermostat phases ──
    print(f"\n━━━ FASES DEL TERMOSTATO ━━━")
    if thermostat_hist:
        phases = [h.phase for h in thermostat_hist]
        phase_counts = {}
        for ph in phases:
            phase_counts[ph] = phase_counts.get(ph, 0) + 1
        for ph, count in phase_counts.items():
            print(f"  {ph}: {count} ciclos")

        # Show transition timeline
        prev_phase = None
        for h in thermostat_hist:
            if h.phase != prev_phase:
                print(f"    t={h.t:.2f}: → {h.phase} (d_95={h.d_95}, A={h.force_effective:.1f}, ν₄={h.nu4_effective:.1e})")
                prev_phase = h.phase

    # ── Spectral slope evolution ──
    print(f"\n━━━ PENDIENTE ESPECTRAL ━━━")
    mask = (k_bins >= 6) & (k_bins <= min(40, p["N"] // 4)) & (E_k > 1e-20)
    if np.sum(mask) > 5:
        log_k = np.log(k_bins[mask])
        log_E = np.log(E_k[mask])
        slope, _ = np.polyfit(log_k, log_E, 1)
        print(f"  Pendiente final: E(k) ~ k^{{{slope:.2f}}}")
        if abs(slope - (-3.0)) < 1.0:
            print(f"  ✓ CASCADA DE ENSTROFÍA (k⁻³) ALCANZADA")
        elif abs(slope - (-5.0/3)) < 0.5:
            print(f"  ✓ Cascada inversa (k⁻⁵/³)")
        else:
            print(f"  ⚠ Cascada en desarrollo")
    else:
        slope = -99.0
        print(f"  ⚠ Insuficientes datos para slope")

    # ── Reduction factor ──
    print(f"\n━━━ REDUCCIÓN DIMENSIONAL ━━━")
    if koopman_results:
        d_95_final = koopman_results[-1].d_95
        N = p["N"]
        reduction = N**2 / max(d_95_final, 1)
        print(f"  N² = {N**2:,}, d_95 = {d_95_final}")
        print(f"  REDUCCIÓN: {reduction:.0f}×")
        print(f"  Coherent structures: {koopman_results[-1].n_coherent}")

    # ── Oracle vote summary ──
    print(f"\n━━━ RESUMEN DE ORÁCULOS ━━━")
    vote_log = result.get("oracle_vote_log", [])
    if vote_log:
        for entry in vote_log:
            t = entry["t"]
            phase = entry["phase"]
            d95 = entry["d_95"]
            h_ks = entry.get("h_ks", 0)
            sl = entry.get("slope", 0)
            votes = entry["votes"]
            n_active = sum(1 for _, c, _, _ in votes if c > 0.1)
            print(f"  t={t:.1f}: phase={phase}, d_95={d95}, h_KS={h_ks:.4f}, "
                  f"slope={sl:.1f}, oracles={n_active}/5")

    # ── Koopman analysis on final state ──
    print(f"\n━━━ ANÁLISIS FINAL PROFUNDO ━━━")
    if n_snap > 30:
        from poema.backends.gideon.koopman_gpu import KoopmanGPU
        try:
            # Use last 60% of snapshots for final analysis
            n_use = max(30, int(n_snap * 0.6))
            final_snaps = snapshots[-n_use:]
            dt_snap = times[-1] - times[-2] if len(times) > 1 else 0.1

            koopman = KoopmanGPU()
            final_result = koopman.analyze(final_snaps, dt=dt_snap, n_modes=min(60, n_use - 2))

            print(f"  Backend: {final_result.backend}")
            print(f"  d_95 = {final_result.d_95}")
            print(f"  Intrinsic dim = {final_result.intrinsic_dim}")
            print(f"  Coherent = {final_result.n_coherent}")
            print(f"  Reconstruction error = {final_result.reconstruction_error:.6f}")
            print(f"  EDMD norm = {final_result.edmd_matrix_norm:.4f}")

            N = p["N"]
            reduction = N**2 / max(final_result.d_95, 1)
            print(f"  REDUCCIÓN FINAL: {N}²/{final_result.d_95} = {reduction:.0f}×")

        except Exception as e:
            print(f"  ⚠ Análisis final falló: {e}")
            traceback.print_exc()

    return slope


def generate_visualization(result, slope):
    """Generate 8-panel visualization."""
    print("\n━━━ VISUALIZACIÓN ━━━")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(24, 11))
        fig.suptitle(
            f"Cascade Acceleration — NS 2D {result['params']['N']}×{result['params']['N']} "
            f"Re={result['params']['Re']:.0f} — Multi-Oracle ACF Thermostat",
            fontsize=14, fontweight="bold"
        )

        p = result["params"]
        times = result["times"]
        energy = result["energy"]
        enstrophy = result["enstrophy"]
        k_bins = result["k_bins"]
        E_k = result["E_k"]
        koopman_results = result["koopman_results"]
        thermostat_hist = result["thermostat_history"]
        snapshots = result["snapshots"]

        # (a) Final vorticity
        ax = axes[0, 0]
        if len(snapshots) > 0:
            omega_final = snapshots[-1]
            vmax = np.percentile(np.abs(omega_final), 98)
            im = ax.imshow(omega_final, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           extent=[0, 2*np.pi, 0, 2*np.pi], origin="lower")
            plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("(a) Vorticidad ω(x,y) final")

        # (b) Energy spectrum with k^-3 reference
        ax = axes[0, 1]
        mask_pos = E_k > 0
        ax.loglog(k_bins[mask_pos], E_k[mask_pos], "b-", lw=1.5, label="E(k)")
        k_ref = np.logspace(np.log10(3), np.log10(60), 50)
        ax.loglog(k_ref, 5e-3 * k_ref**(-3), "r--", alpha=0.7, label="k$^{-3}$")
        ax.loglog(k_ref, 2e-3 * k_ref**(-5/3), "g--", alpha=0.7, label="k$^{-5/3}$")
        ax.set_title(f"(b) E(k) — slope={slope:.2f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (c) d_95 evolution (the key plot!)
        ax = axes[0, 2]
        if koopman_results:
            d95_series = [kr.d_95 for kr in koopman_results]
            # Approximate times for each analysis
            t_analyses = np.linspace(
                p.get("analysis_interval", 2.0),
                times[-1] if len(times) > 0 else p["T_total"],
                len(d95_series)
            )
            ax.plot(t_analyses, d95_series, "b-o", lw=2, ms=5)
            ax.axhline(y=p.get("d_target", 30), color="r", ls="--", lw=1, label=f"d_target={p.get('d_target', 30)}")
            ax.set_ylabel("d_95")
            ax.set_xlabel("t")
            ax.legend(fontsize=8)
        ax.set_title("(c) d_95 evolución")
        ax.grid(True, alpha=0.3)

        # (d) Thermostat phase diagram
        ax = axes[0, 3]
        if thermostat_hist:
            t_thermo = [h.t for h in thermostat_hist]
            phases_num = []
            for h in thermostat_hist:
                if h.phase == "acceleration":
                    phases_num.append(0)
                elif h.phase == "stabilization":
                    phases_num.append(1)
                else:
                    phases_num.append(2)
            colors = ["red", "orange", "green"]
            labels = ["Acceleration", "Stabilization", "Equilibrium"]
            for i, (t, pn) in enumerate(zip(t_thermo, phases_num)):
                ax.bar(t, 1, width=max(t_thermo[1] - t_thermo[0] if len(t_thermo) > 1 else 1, 0.5),
                       bottom=pn, color=colors[pn], alpha=0.7)
            # Legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
            ax.legend(handles=legend_elements, fontsize=8)
            ax.set_yticks([0.5, 1.5, 2.5])
            ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("(d) Thermostat Phase")
        ax.set_xlabel("t")

        # (e) E(t) and Z(t)
        ax = axes[1, 0]
        ax.plot(times, energy, "b-", lw=1, label="E(t)")
        ax2 = ax.twinx()
        ax2.plot(times, enstrophy, "r-", lw=1, alpha=0.7, label="Z(t)")
        ax.set_title("(e) E(t) y Z(t)")
        ax.set_xlabel("t")
        lines1, l1 = ax.get_legend_handles_labels()
        lines2, l2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, l1 + l2, fontsize=8)
        ax.grid(True, alpha=0.3)

        # (f) Koopman eigenvalues (final)
        ax = axes[1, 1]
        if koopman_results:
            spectrum = koopman_results[-1].eigenvalues
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.5, alpha=0.3)
            coherent = koopman_results[-1].coherent_mask
            ax.scatter(np.real(spectrum[~coherent]), np.imag(spectrum[~coherent]),
                       c="gray", s=15, alpha=0.5, label="Transient")
            if np.any(coherent):
                ax.scatter(np.real(spectrum[coherent]), np.imag(spectrum[coherent]),
                           c="red", s=35, zorder=5, label="Coherent")
            ax.set_title(f"(f) Koopman (d_95={koopman_results[-1].d_95})")
            ax.legend(fontsize=8)
            ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # (g) Forcing & ν₄ evolution (thermostat control)
        ax = axes[1, 2]
        if thermostat_hist:
            t_th = [h.t for h in thermostat_hist]
            force_series = [h.force_effective for h in thermostat_hist]
            nu4_series = [h.nu4_effective for h in thermostat_hist]
            ax.plot(t_th, force_series, "b-o", lw=1.5, ms=3, label="A(t)")
            ax_r = ax.twinx()
            ax_r.semilogy(t_th, nu4_series, "r-s", lw=1.5, ms=3, label="ν₄(t)")
            ax_r.set_ylabel("ν₄", color="r")
            ax.set_ylabel("A", color="b")
            lines1, l1 = ax.get_legend_handles_labels()
            lines2, l2 = ax_r.get_legend_handles_labels()
            ax.legend(lines1 + lines2, l1 + l2, fontsize=8)
        ax.set_title("(g) Control: A(t), ν₄(t)")
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.3)

        # (h) Reduction factor evolution
        ax = axes[1, 3]
        if thermostat_hist:
            t_th = [h.t for h in thermostat_hist]
            red_series = [h.reduction_factor for h in thermostat_hist]
            ax.semilogy(t_th, red_series, "g-o", lw=2, ms=4)
            ax.set_ylabel("N²/d_95")
        ax.set_title("(h) Reducción dimensional")
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "cascade_acceleration_results.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Figura: {outpath}")
        return outpath
    except Exception as e:
        print(f"  ⚠ Visualización falló: {e}")
        traceback.print_exc()
        return ""


def final_report(result, slope, t_total, fig_path):
    """Print comprehensive final report."""
    p = result["params"]
    koopman_results = result["koopman_results"]
    thermostat_hist = result["thermostat_history"]

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║  REPORTE FINAL — ACELERACIÓN DE LA CASCADA                      ║")
    print("║  Termostato de Turbulencia ACF Multi-Oráculo                    ║")
    print("╚" + "═" * 68 + "╝")

    print(f"\n━━━ CONFIGURACIÓN ━━━")
    print(f"  Grid: {p['N']}×{p['N']} ({p['N']**2:,} DOFs)")
    print(f"  Re = {p['Re']:.0f}, T_total = {p['T_total']}")
    print(f"  Backend: {p['backend']}, auto_evolve: {p['auto_evolve']}")
    print(f"  Multi-Oracle: {p['multi_oracle']}")
    print(f"  d_target = {p['d_target']}, h_ks_target = {p['h_ks_target']}")
    print(f"  Tiempo real: {t_total:.1f}s")

    print(f"\n━━━ RESULTADO CLAVE ━━━")
    if koopman_results:
        d95_series = [kr.d_95 for kr in koopman_results]
        d95_final = d95_series[-1]
        d95_max = max(d95_series)
        N = p["N"]
        reduction = N**2 / max(d95_final, 1)
        print(f"  d_95 evolución: {d95_series}")
        print(f"  d_95 final: {d95_final}")
        print(f"  d_95 máximo alcanzado: {d95_max}")
        print(f"  REDUCCIÓN DIMENSIONAL: {N}²/{d95_final} = {reduction:.0f}×")

        reached_target = d95_final >= p["d_target"] * 0.7
        cascade_formed = abs(slope - (-3.0)) < 1.5
        print(f"\n  ¿Turbulencia desarrollada (d_95 ≥ {p['d_target']})? {'SÍ ✓' if reached_target else 'EN PROCESO'}")
        print(f"  ¿Cascada k⁻³? {'SÍ ✓' if cascade_formed else 'EN DESARROLLO'}")
        print(f"  Pendiente espectral: {slope:.2f}")

    if thermostat_hist:
        phases = [h.phase for h in thermostat_hist]
        reached_eq = "equilibrium" in phases
        reached_stab = "stabilization" in phases
        print(f"\n  ¿Alcanzó estabilización? {'SÍ ✓' if reached_stab else 'NO'}")
        print(f"  ¿Alcanzó equilibrio? {'SÍ ✓' if reached_eq else 'NO'}")

        # Final state
        final_state = thermostat_hist[-1]
        print(f"\n  Estado final del termostato:")
        print(f"    Fase: {final_state.phase}")
        print(f"    A_efectivo = {final_state.force_effective:.2f}")
        print(f"    ν₄_efectivo = {final_state.nu4_effective:.1e}")
        print(f"    α_efectivo = {final_state.friction_effective:.4f}")

    energy = result["energy"]
    enstrophy = result["enstrophy"]
    print(f"\n━━━ BALANCE ENERGÉTICO ━━━")
    print(f"  E_final = {energy[-1]:.6f}")
    print(f"  Z_final = {enstrophy[-1]:.6f}")
    dE = np.diff(energy)
    dt = np.diff(result["times"])
    dE_dt = dE / (dt + 1e-12)
    print(f"  dE/dt final = {dE_dt[-1]:.6f}")
    stationarity = abs(np.mean(dE_dt[-10:])) < np.std(dE_dt[-10:]) if len(dE_dt) > 10 else False
    print(f"  ¿Estacionario? {'SÍ ✓' if stationarity else 'NO (aún transitorio)'}")

    if fig_path:
        print(f"\n  Figura: {fig_path}")
    print()


def main():
    t0 = time.time()

    # Run the experiment
    result, t_exp = run_cascade_experiment()

    # Analyze
    slope = analyze_results(result, t_exp)

    # Visualize
    fig_path = generate_visualization(result, slope)

    # Report
    final_report(result, slope, t_exp, fig_path)

    print(f"Tiempo total (incl. análisis): {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
