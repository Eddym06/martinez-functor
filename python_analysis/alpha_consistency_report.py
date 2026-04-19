from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np

# Ensure imports resolve to repository package, not local helper module names.
ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)

from acf_functor.invariant_unified import (
    AlphaCombinatorial,
    AlphaGeometric,
    AlphaSpectral,
    ACFInvariantUnified,
)


CanonicalFn = Tuple[str, Callable[[float], float], Tuple[float, float]]


def _canonical_suite() -> List[CanonicalFn]:
    return [
        ("poly_cubic", lambda x: x**3 + 2.0 * x + 1.0, (-1.0, 1.0)),
        ("sin", np.sin, (-math.pi / 2.0, math.pi / 2.0)),
        ("exp", np.exp, (-1.0, 1.0)),
        ("tanh", np.tanh, (-2.0, 2.0)),
    ]


def build_alpha_consistency_report(
    *,
    consistency_threshold: float = 0.5,
    skip_geometric: bool = False,
    fast_mode: bool = False,
) -> Dict[str, object]:
    """Compute a reproducible alpha-consistency report over canonical functions.

    The report does not assert theoretical equivalence of definitions. It quantifies
    empirical agreement/disagreement and keeps the discrepancy explicit.
    """
    inv = ACFInvariantUnified(consistency_threshold=consistency_threshold)

    if fast_mode:
        inv.comb = AlphaCombinatorial(epsilon_range=[1e-1, 1e-2, 1e-3, 1e-4])
        inv.spec = AlphaSpectral(n_observables=20, n_trajectory=1200)
        inv.geo = AlphaGeometric(n_trajectory=8000, n_scales=10)

    rows: List[Dict[str, object]] = []
    for name, fn, domain in _canonical_suite():
        est = inv.compute(fn, domain=domain, function_name=name, skip_geometric=skip_geometric)
        rows.append(
            {
                "function": name,
                "domain": [domain[0], domain[1]],
                "alpha_combinatorial": est.alpha_combinatorial,
                "alpha_spectral": est.alpha_spectral,
                "alpha_geometric": est.alpha_geometric,
                "alpha_comb_ci": [est.alpha_comb_ci[0], est.alpha_comb_ci[1]],
                "alpha_spec_ci": [est.alpha_spec_ci[0], est.alpha_spec_ci[1]],
                "alpha_geo_ci": [est.alpha_geo_ci[0], est.alpha_geo_ci[1]],
                "definitions_consistent": est.definitions_consistent,
                "max_discrepancy": est.max_discrepancy,
                "best_estimate": est.best_estimate,
                "best_definition": est.best_definition.value,
                "n_fma_evaluations": est.n_fma_evaluations,
                "computation_time_s": est.computation_time_s,
            }
        )

    discrepancies = [float(r["max_discrepancy"]) for r in rows]
    consistent_count = sum(1 for r in rows if bool(r["definitions_consistent"]))

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "consistency_threshold": consistency_threshold,
        "skip_geometric": skip_geometric,
        "fast_mode": fast_mode,
        "summary": {
            "n_functions": len(rows),
            "n_consistent": consistent_count,
            "consistency_rate": (consistent_count / len(rows)) if rows else 0.0,
            "max_discrepancy_global": max(discrepancies) if discrepancies else 0.0,
            "mean_discrepancy_global": float(np.mean(discrepancies)) if discrepancies else 0.0,
        },
        "results": rows,
        "interpretation": {
            "note": (
                "This report quantifies empirical alignment between alpha definitions; "
                "it is not a formal proof of equivalence."
            )
        },
    }


def write_report(report: Dict[str, object], output_json: Path, output_md: Path | None = None) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if output_md is not None:
        lines: List[str] = []
        lines.append("# Alpha Consistency Report")
        lines.append("")
        lines.append(f"- timestamp: {report['timestamp_utc']}")
        lines.append(f"- threshold: {report['consistency_threshold']}")
        lines.append(f"- skip_geometric: {report['skip_geometric']}")
        lines.append(f"- fast_mode: {report['fast_mode']}")
        lines.append("")
        summary = report["summary"]
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- n_functions: {summary['n_functions']}")
        lines.append(f"- n_consistent: {summary['n_consistent']}")
        lines.append(f"- consistency_rate: {summary['consistency_rate']:.3f}")
        lines.append(f"- max_discrepancy_global: {summary['max_discrepancy_global']:.6f}")
        lines.append(f"- mean_discrepancy_global: {summary['mean_discrepancy_global']:.6f}")
        lines.append("")
        lines.append("## Per-function Results")
        lines.append("")
        lines.append("| function | alpha_comb | alpha_spec | alpha_geo | discrepancy | consistent | best |")
        lines.append("| --- | ---: | ---: | ---: | ---: | :---: | --- |")
        for row in report["results"]:
            lines.append(
                "| {function} | {alpha_combinatorial:.6f} | {alpha_spectral:.6f} | {alpha_geometric:.6f} | {max_discrepancy:.6f} | {definitions_consistent} | {best_definition} |".format(
                    **row
                )
            )
        lines.append("")
        lines.append(report["interpretation"]["note"])
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate alpha-consistency report")
    p.add_argument("--output-json", default="artifacts/alpha_consistency_report.json")
    p.add_argument("--output-md", default="artifacts/alpha_consistency_report.md")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--skip-geometric", action="store_true")
    p.add_argument("--fast", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_alpha_consistency_report(
        consistency_threshold=args.threshold,
        skip_geometric=args.skip_geometric,
        fast_mode=args.fast,
    )
    write_report(report, Path(args.output_json), Path(args.output_md))
    print(f"alpha consistency report written: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
