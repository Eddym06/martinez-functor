"""Visualization dashboard driven by real artifacts and reproducible computations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure package imports resolve to the repository package, not local helper modules.
ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)

from acf_functor.certified_koopman import CertifiedKoopman
from acf_functor.invariant_unified import AlphaCombinatorial


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.dpi": 140,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def plot_alpha_convergence(output_dir: Path) -> None:
    eps = [10 ** (-k) for k in range(1, 8)]
    calc = AlphaCombinatorial(epsilon_range=eps)

    poly = lambda x: x**3 + 2 * x + 1
    sfn = np.sin
    e_poly = [calc.minimum_fma_count(poly, e, domain=(-1.0, 1.0)) for e in eps]
    e_sin = [calc.minimum_fma_count(sfn, e, domain=(-np.pi / 2, np.pi / 2)) for e in eps]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(eps, e_poly, "o-", label="x^3+2x+1")
    ax.loglog(eps, e_sin, "s-", label="sin(x)")
    ax.invert_xaxis()
    ax.set_xlabel("epsilon")
    ax.set_ylabel("E(f, epsilon)")
    ax.set_title("Complejidad FMA vs precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_convergence.png")
    plt.close(fig)


def plot_koopman_branch_errors(output_dir: Path) -> None:
    ck = CertifiedKoopman()
    cases = [
        ("poly", lambda x: 0.8 * x + 0.1, -1.0, 1.0),
        ("logistic", lambda x: 3.5 * x * (1 - x), 0.0, 1.0),
        ("tanh", lambda x: np.tanh(1.2 * x), -1.0, 1.0),
    ]

    names = []
    empirical = []
    bounds = []
    for name, g, a, b in cases:
        x0 = 0.25
        n = 5
        out = ck.predict(g, x0=x0, n_steps=n, domain=(a, b), target_error=1e-3)
        x_true = x0
        for _ in range(n):
            x_true = float(g(x_true))
        names.append(name)
        empirical.append(abs(out.predicted_value - x_true))
        bounds.append(out.error_bound)

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, empirical, w, label="error empirico")
    ax.bar(x + w / 2, bounds, w, label="cota certificada")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_yscale("log")
    ax.set_title("Koopman: error empirico vs cota")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "koopman_branch_errors.png")
    plt.close(fig)


def generate_all_plots(output: str) -> None:
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_alpha_convergence(out_dir)
    plot_koopman_branch_errors(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="plots")
    args = parser.parse_args()
    generate_all_plots(args.output)
    print(f"Plots written to: {args.output}")
