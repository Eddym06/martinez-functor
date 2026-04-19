#!/usr/bin/env python3
"""
Periodic Table of Spectrums: robust analytical engine over BiPoem spectral signatures.

This script upgrades the previous single-pass report into a policy-aware analyzer:
1. Monte Carlo trials with uncertainty estimates,
2. Expanded phase-space coverage (chaotic, discontinuous, stiff, oscillatory),
3. Domain-calibrated family thresholds,
4. Structured exports (Markdown + JSON + Parquet),
5. Optional bridge to cluster proxy benchmark.

Refs: Paper §18.4, §22.5; BiPoem Bi2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, cast

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised via subprocess in tests
    pd = None
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poema import BiPoem


TensorGen = Callable[[torch.Generator], torch.Tensor]


@dataclass
class TrialMetrics:
    alpha: float
    spectral_gap: float
    dominant_lambda: float
    reconstruction_error: float


@dataclass
class AggregateMetrics:
    alpha_mean: float
    alpha_std: float
    alpha_var: float
    alpha_ci95: float
    spectral_gap_mean: float
    spectral_gap_std: float
    spectral_gap_var: float
    spectral_gap_ci95: float
    dominant_lambda_mean: float
    dominant_lambda_std: float
    dominant_lambda_var: float
    dominant_lambda_ci95: float
    reconstruction_error_mean: float
    reconstruction_error_std: float
    reconstruction_error_var: float
    reconstruction_error_ci95: float
    drift_score: float
    drift_flag: bool


DEFAULT_DOMAIN_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "general": (1.5, 3.0),
    "finance": (1.8, 3.4),
    "fluids": (1.4, 2.8),
    "signals": (1.3, 2.6),
}


POLICY_LIBRARY: Dict[str, Dict[str, object]] = {
    "fast": {
        "degree": 16,
        "precision": "fp32",
        "observable_family": "polynomial",
        "max_dimension": 32,
        "rank_budget": "low",
    },
    "algebraic": {
        "degree": 32,
        "precision": "fp32",
        "observable_family": "hybrid-polynomial-trig",
        "max_dimension": 48,
        "rank_budget": "medium",
    },
    "slow": {
        "degree": 64,
        "precision": "fp64",
        "observable_family": "rich-lifted",
        "max_dimension": 64,
        "rank_budget": "high",
    },
}


def _to_time_series(sample: torch.Tensor) -> torch.Tensor:
    """Normalize samples to shape (n_states, n_steps) expected by BiPoem."""
    if sample.dim() == 1:
        return sample.unsqueeze(0)
    if sample.dim() == 2:
        # Input often comes as (steps, features). Convert to (features, steps).
        if sample.shape[0] >= sample.shape[1]:
            return sample.T.contiguous()
        return sample
    raise ValueError(f"unsupported sample shape: {tuple(sample.shape)}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean_std_var_ci95(values: List[float]) -> Tuple[float, float, float, float]:
    t = torch.tensor(values, dtype=torch.float64)
    if t.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(t.mean().item())
    var = float(t.var(unbiased=False).item())
    std = float(t.std(unbiased=False).item())
    ci95 = 1.96 * std / (t.numel() ** 0.5)
    return mean, std, var, ci95


def _drift_score(alpha_values: List[float], rec_values: List[float]) -> float:
    alpha_t = torch.tensor(alpha_values, dtype=torch.float64)
    rec_t = torch.tensor(rec_values, dtype=torch.float64)
    if alpha_t.numel() < 2:
        return 0.0
    alpha_cv = float(alpha_t.std(unbiased=False) / (alpha_t.abs().mean() + 1e-12))
    rec_cv = float(rec_t.std(unbiased=False) / (rec_t.abs().mean() + 1e-12))
    return max(alpha_cv, rec_cv)


def _classify_alpha(alpha: float, domain: str) -> str:
    low, high = DEFAULT_DOMAIN_THRESHOLDS.get(domain, DEFAULT_DOMAIN_THRESHOLDS["general"])
    if alpha < low:
        return "fast"
    if alpha < high:
        return "algebraic"
    return "slow"


def _policy_for_family(family: str, drift_flag: bool) -> Dict[str, object]:
    policy = dict(POLICY_LIBRARY.get(family, POLICY_LIBRARY["algebraic"]))
    if drift_flag:
        # Conservative fallback under instability.
        policy["precision"] = "fp64"
        policy["max_dimension"] = int(_safe_float(policy.get("max_dimension", 48), default=48.0)) + 8
        policy["stability_mode"] = "conservative"
    return policy


def _markdown_table(rows: List[dict]) -> str:
    headers = [
        "Case",
        "Domain",
        "Family",
        "alpha_mean",
        "alpha_std",
        "alpha_ci95",
        "SpectralGap_mean",
        "DominantLambda_mean",
        "ReconstructionErr_mean",
        "DriftScore",
        "DriftFlag",
    ]
    md = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        md.append(
            "| "
            + " | ".join(
                [
                    str(row["Case"]),
                    str(row["Domain"]),
                    str(row["Family"]),
                    f"{row['alpha_mean']:.4f}",
                    f"{row['alpha_std']:.4f}",
                    f"{row['alpha_ci95']:.4f}",
                    f"{row['Spectral Gap mean']:.4f}",
                    f"{row['Dominant lambda mean']:.4f}",
                    f"{row['Reconstruction Error mean']:.4f}",
                    f"{row['Drift Score']:.4f}",
                    str(row["Drift Flag"]),
                ]
            )
            + " |"
        )
    return "\n".join(md)


def _sample_lorenz(generator: torch.Generator, steps: int = 300, dt: float = 0.01) -> torch.Tensor:
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    x = torch.randn(3, generator=generator, dtype=torch.float64) * 0.05 + torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    out = torch.empty(steps, 3, dtype=torch.float64)
    for i in range(steps):
        dx = sigma * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - beta * x[2]
        x = x + dt * torch.tensor([dx, dy, dz], dtype=torch.float64)
        out[i] = x
    return out


def _sample_rossler(generator: torch.Generator, steps: int = 300, dt: float = 0.02) -> torch.Tensor:
    a, b, c = 0.2, 0.2, 5.7
    x = torch.randn(3, generator=generator, dtype=torch.float64) * 0.05 + torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
    out = torch.empty(steps, 3, dtype=torch.float64)
    for i in range(steps):
        dx = -x[1] - x[2]
        dy = x[0] + a * x[1]
        dz = b + x[2] * (x[0] - c)
        x = x + dt * torch.tensor([dx, dy, dz], dtype=torch.float64)
        out[i] = x
    return out


def _sample_logistic_map(generator: torch.Generator, steps: int = 300, r: float = 3.9) -> torch.Tensor:
    x = torch.rand(1, generator=generator, dtype=torch.float64) * 0.5 + 0.25
    out = torch.empty(steps, 1, dtype=torch.float64)
    for i in range(steps):
        x = r * x * (1.0 - x)
        out[i] = x
    return out


def _sample_sine(generator: torch.Generator, steps: int = 300) -> torch.Tensor:
    _ = generator
    t = torch.linspace(0.0, 12.0, steps, dtype=torch.float64)
    return torch.sin(t).unsqueeze(-1)


def _sample_exp_decay(generator: torch.Generator, steps: int = 300) -> torch.Tensor:
    _ = generator
    t = torch.linspace(0.0, 10.0, steps, dtype=torch.float64)
    return torch.exp(-0.5 * t).unsqueeze(-1)


def _sample_linear_stable(generator: torch.Generator, steps: int = 300) -> torch.Tensor:
    _ = generator
    t = torch.linspace(-1.0, 1.0, steps, dtype=torch.float64)
    return (0.9 * t).unsqueeze(-1)


def _sample_rotation(generator: torch.Generator, steps: int = 300) -> torch.Tensor:
    phase = float(torch.rand(1, generator=generator, dtype=torch.float64).item()) * 0.5
    t = torch.linspace(0.0, 6.0 * torch.pi, steps, dtype=torch.float64) + phase
    return torch.stack([torch.cos(t), torch.sin(t)], dim=-1)


def _sample_relu(generator: torch.Generator, steps: int = 300) -> torch.Tensor:
    _ = generator
    t = torch.linspace(-2.0, 2.0, steps, dtype=torch.float64)
    return torch.relu(t).unsqueeze(-1)


def _sample_step(generator: torch.Generator, steps: int = 300) -> torch.Tensor:
    _ = generator
    t = torch.linspace(-1.0, 1.0, steps, dtype=torch.float64)
    return (t >= 0.0).to(torch.float64).unsqueeze(-1)


def _sample_square_wave(generator: torch.Generator, steps: int = 300) -> torch.Tensor:
    phase = float(torch.rand(1, generator=generator, dtype=torch.float64).item()) * 0.5
    t = torch.linspace(0.0, 8.0 * torch.pi, steps, dtype=torch.float64) + phase
    return torch.sign(torch.sin(t)).unsqueeze(-1)


def _sample_stiff_two_scale(generator: torch.Generator, steps: int = 350, dt: float = 0.005) -> torch.Tensor:
    # Two-scale linear stiff dynamics: one slow and one very fast decaying mode.
    x = torch.randn(2, generator=generator, dtype=torch.float64) * 0.1 + torch.tensor([1.0, 1.0], dtype=torch.float64)
    out = torch.empty(steps, 2, dtype=torch.float64)
    for i in range(steps):
        dx = -0.1 * x[0]
        dy = -25.0 * x[1]
        x = x + dt * torch.tensor([dx, dy], dtype=torch.float64)
        out[i] = x
    return out


def _case_bank() -> Dict[str, Dict[str, object]]:
    return {
        "sin": {"domain": "signals", "generator": _sample_sine},
        "exp_decay": {"domain": "general", "generator": _sample_exp_decay},
        "logistic": {"domain": "general", "generator": _sample_logistic_map},
        "linear_stable": {"domain": "general", "generator": _sample_linear_stable},
        "rotation": {"domain": "signals", "generator": _sample_rotation},
        "lorenz": {"domain": "fluids", "generator": _sample_lorenz},
        "rossler": {"domain": "fluids", "generator": _sample_rossler},
        "relu": {"domain": "signals", "generator": _sample_relu},
        "step": {"domain": "signals", "generator": _sample_step},
        "square_wave": {"domain": "signals", "generator": _sample_square_wave},
        "stiff_two_scale": {"domain": "fluids", "generator": _sample_stiff_two_scale},
    }


def _extract_trial_metrics(out: Dict[str, object]) -> TrialMetrics:
    spectrum = out.get("bifunctorial_spectrum", {})
    if not isinstance(spectrum, dict):
        spectrum = {}
    return TrialMetrics(
        alpha=_safe_float(out.get("acf_alpha", 0.0)),
        spectral_gap=_safe_float(spectrum.get("spectral_gap", 0.0)),
        dominant_lambda=_safe_float(spectrum.get("dominant_eigenvalue", 0.0)),
        reconstruction_error=_safe_float(out.get("reconstruction_error", 0.0)),
    )


def _aggregate_trials(trials: List[TrialMetrics], drift_threshold: float) -> AggregateMetrics:
    alpha_vals = [t.alpha for t in trials]
    gap_vals = [t.spectral_gap for t in trials]
    dl_vals = [t.dominant_lambda for t in trials]
    rec_vals = [t.reconstruction_error for t in trials]

    a_mean, a_std, a_var, a_ci95 = _mean_std_var_ci95(alpha_vals)
    g_mean, g_std, g_var, g_ci95 = _mean_std_var_ci95(gap_vals)
    d_mean, d_std, d_var, d_ci95 = _mean_std_var_ci95(dl_vals)
    r_mean, r_std, r_var, r_ci95 = _mean_std_var_ci95(rec_vals)
    drift = _drift_score(alpha_vals, rec_vals)
    return AggregateMetrics(
        alpha_mean=a_mean,
        alpha_std=a_std,
        alpha_var=a_var,
        alpha_ci95=a_ci95,
        spectral_gap_mean=g_mean,
        spectral_gap_std=g_std,
        spectral_gap_var=g_var,
        spectral_gap_ci95=g_ci95,
        dominant_lambda_mean=d_mean,
        dominant_lambda_std=d_std,
        dominant_lambda_var=d_var,
        dominant_lambda_ci95=d_ci95,
        reconstruction_error_mean=r_mean,
        reconstruction_error_std=r_std,
        reconstruction_error_var=r_var,
        reconstruction_error_ci95=r_ci95,
        drift_score=drift,
        drift_flag=drift > drift_threshold,
    )


def _run_case_trials(
    bi: BiPoem,
    case_gen: TensorGen,
    n_trials: int,
    noise_std: float,
    seed: int,
    max_dimension: int,
) -> List[TrialMetrics]:
    trials: List[TrialMetrics] = []
    for trial_idx in range(n_trials):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + trial_idx)
        sample = case_gen(gen)
        if noise_std > 0.0:
            sample = sample + noise_std * torch.randn(sample.shape, generator=gen, dtype=sample.dtype)
        x = _to_time_series(sample)
        out = bi.symbiosis_with_report(x, max_dimension=max_dimension)
        trials.append(_extract_trial_metrics(out))
    return trials


def _emit_cluster_plan(records: List[Dict[str, object]], plan_output: Path) -> None:
    plan = {
        "schema": "periodic-cluster-plan-v1",
        "items": [
            {
                "case": r["Case"],
                "domain": r["Domain"],
                "family": r["Family"],
                "policy": r["Compile Policy"],
                "drift_flag": r["Drift Flag"],
            }
            for r in records
        ],
    }
    plan_output.parent.mkdir(parents=True, exist_ok=True)
    plan_output.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def _run_cluster_bridge(plan_output: Path, steps: int, cluster_output: Path) -> None:
    # Bridge: run cluster proxy with policy arguments emitted by this table.
    print(f"Running cluster proxy bridge with plan: {plan_output}")
    plan = json.loads(plan_output.read_text(encoding="utf-8"))
    items = plan.get("items", []) if isinstance(plan, dict) else []
    if not isinstance(items, list) or not items:
        raise RuntimeError("cluster plan has no items")

    cluster_output.parent.mkdir(parents=True, exist_ok=True)
    case_metrics: List[Dict[str, object]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        case = str(item.get("case", "unknown"))
        policy = item.get("policy", {})
        if not isinstance(policy, dict):
            policy = {}

        case_file = cluster_output.parent / f"cluster_proxy_{case}.json"
        cmd = [
            sys.executable,
            str(ROOT / "benchmarks" / "cluster_proxy.py"),
            "--steps",
            str(steps),
            "--profile-name",
            f"{item.get('family', 'unknown')}-{case}",
            "--degree",
            str(int(_safe_float(policy.get("degree", 32), 32.0))),
            "--max-dimension",
            str(int(_safe_float(policy.get("max_dimension", 48), 48.0))),
            "--precision",
            str(policy.get("precision", "fp32")),
            "--observable-family",
            str(policy.get("observable_family", "hybrid-polynomial-trig")),
            "--rank-budget",
            str(policy.get("rank_budget", "medium")),
            "--output",
            str(case_file),
        ]
        subprocess.run(cmd, check=True)
        case_metrics.append(json.loads(case_file.read_text(encoding="utf-8")))

    bridge_report = {
        "schema": "periodic-cluster-bridge-v1",
        "plan": str(plan_output),
        "items": case_metrics,
        "cases_executed": len(case_metrics),
    }
    cluster_output.write_text(json.dumps(bridge_report, indent=2), encoding="utf-8")


def generate_periodic_table(
    output_md: bool = True,
    output_path: str = "periodic_table.md",
    json_output_path: str = "artifacts/periodic_table.json",
    parquet_output_path: str = "artifacts/periodic_table.parquet",
    n_trials: int = 10,
    noise_std: float = 0.01,
    seed: int = 7,
    domain: str = "general",
    drift_threshold: float = 0.20,
    max_dimension: int = 32,
    emit_cluster_plan: bool = False,
    cluster_plan_path: str = "artifacts/periodic_cluster_plan.json",
    run_cluster_proxy: bool = False,
    cluster_steps: int = 10,
    cluster_output_path: str = "artifacts/cluster_proxy_metrics.json",
) -> List[Dict[str, object]]:
    random.seed(seed)
    torch.manual_seed(seed)

    if n_trials < 2:
        raise ValueError("n_trials must be >= 2 for uncertainty estimation")

    bi = BiPoem(dtype=torch.float64)
    test_cases = _case_bank()
    records: List[Dict[str, object]] = []

    for name, meta in test_cases.items():
        case_domain = str(meta.get("domain", "general"))
        if domain != "all" and case_domain != domain:
            continue

        case_gen = cast(TensorGen, meta["generator"])
        if not callable(case_gen):
            continue

        trials = _run_case_trials(
            bi=bi,
            case_gen=case_gen,
            n_trials=n_trials,
            noise_std=noise_std,
            seed=seed + abs(hash(name)) % 10_000,
            max_dimension=max_dimension,
        )
        agg = _aggregate_trials(trials, drift_threshold=drift_threshold)
        family = _classify_alpha(agg.alpha_mean, case_domain)
        policy = _policy_for_family(family, agg.drift_flag)

        records.append(
            {
                "Case": name,
                "Domain": case_domain,
                "Trials": n_trials,
                "Family": family,
                "alpha_mean": agg.alpha_mean,
                "alpha_std": agg.alpha_std,
                "alpha_var": agg.alpha_var,
                "alpha_ci95": agg.alpha_ci95,
                "Spectral Gap mean": agg.spectral_gap_mean,
                "Spectral Gap std": agg.spectral_gap_std,
                "Spectral Gap var": agg.spectral_gap_var,
                "Spectral Gap ci95": agg.spectral_gap_ci95,
                "Dominant lambda mean": agg.dominant_lambda_mean,
                "Dominant lambda std": agg.dominant_lambda_std,
                "Dominant lambda var": agg.dominant_lambda_var,
                "Dominant lambda ci95": agg.dominant_lambda_ci95,
                "Reconstruction Error mean": agg.reconstruction_error_mean,
                "Reconstruction Error std": agg.reconstruction_error_std,
                "Reconstruction Error var": agg.reconstruction_error_var,
                "Reconstruction Error ci95": agg.reconstruction_error_ci95,
                "Drift Score": agg.drift_score,
                "Drift Flag": agg.drift_flag,
                "Compile Policy": policy,
            }
        )

    records.sort(key=lambda r: (str(r["Domain"]), _safe_float(r["alpha_mean"])))
    if not records:
        raise RuntimeError(f"no records generated for domain={domain}")

    md = _markdown_table(records)
    print(md)

    json_path = Path(json_output_path)
    parquet_path = Path(parquet_output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if output_md:
        Path(output_path).write_text(md + "\n", encoding="utf-8")

    if pd is None:
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        parquet_status = "skipped (ImportError: pandas not installed)"
        print("[warn] pandas not available: JSON written via stdlib, parquet export skipped")
    else:
        df = pd.DataFrame(records)
        df.to_json(json_path, orient="records", indent=2)
        try:
            df.to_parquet(parquet_path, index=False)
            parquet_status = "written"
        except Exception as exc:
            parquet_status = f"skipped ({exc.__class__.__name__}: {exc})"
            print(f"[warn] Parquet export not available: {parquet_status}")

    plan_output = Path(cluster_plan_path)
    if emit_cluster_plan:
        _emit_cluster_plan(records, plan_output)

    if run_cluster_proxy:
        if not emit_cluster_plan:
            _emit_cluster_plan(records, plan_output)
        _run_cluster_bridge(plan_output, steps=cluster_steps, cluster_output=Path(cluster_output_path))

    print(
        "\nGenerated periodic analytics with uncertainty bands; "
        f"JSON={json_path}, Parquet={parquet_status}, records={len(records)}"
    )
    print("Periodic Table generated")
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate periodic spectral analytics and policies")
    parser.add_argument("--output", default="artifacts/periodic_table.md", help="Output markdown file path")
    parser.add_argument("--json-output", default="artifacts/periodic_table.json", help="Output JSON file path")
    parser.add_argument("--parquet-output", default="artifacts/periodic_table.parquet", help="Output parquet file path")
    parser.add_argument("--no-write", action="store_true", help="Do not write markdown file")
    parser.add_argument("--n-trials", type=int, default=10, help="Monte Carlo runs per case")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Gaussian noise std added per trial")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed")
    parser.add_argument(
        "--domain",
        default="all",
        choices=["all", "general", "finance", "fluids", "signals"],
        help="Domain filter and threshold calibration key",
    )
    parser.add_argument("--drift-threshold", type=float, default=0.20, help="Drift flag threshold")
    parser.add_argument("--max-dimension", type=int, default=32, help="BiPoem max_dimension")
    parser.add_argument("--emit-cluster-plan", action="store_true", help="Write plan JSON for cluster proxy")
    parser.add_argument("--cluster-plan-output", default="artifacts/periodic_cluster_plan.json", help="Cluster plan JSON path")
    parser.add_argument("--run-cluster-proxy", action="store_true", help="Run cluster proxy benchmark bridge")
    parser.add_argument("--cluster-steps", type=int, default=10, help="Steps used for cluster proxy bridge")
    parser.add_argument(
        "--cluster-output",
        default="artifacts/cluster_proxy_metrics.json",
        help="Cluster proxy metrics output path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_periodic_table(
        output_md=not args.no_write,
        output_path=args.output,
        json_output_path=args.json_output,
        parquet_output_path=args.parquet_output,
        n_trials=args.n_trials,
        noise_std=args.noise_std,
        seed=args.seed,
        domain=args.domain,
        drift_threshold=args.drift_threshold,
        max_dimension=args.max_dimension,
        emit_cluster_plan=args.emit_cluster_plan,
        cluster_plan_path=args.cluster_plan_output,
        run_cluster_proxy=args.run_cluster_proxy,
        cluster_steps=args.cluster_steps,
        cluster_output_path=args.cluster_output,
    )

