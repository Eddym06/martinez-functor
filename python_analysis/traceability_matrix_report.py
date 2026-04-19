#!/usr/bin/env python3
"""Generate a repository traceability matrix artifact.

The matrix links claims to implementation paths, validation tests, and artifacts.
This keeps documentation synchronized with executable evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceabilityEntry:
    claim_id: str
    claim: str
    rigor_layer: str
    status: str
    implementation_refs: list[str]
    validation_refs: list[str]
    artifact_refs: list[str]
    commands: list[str]
    current_limit: str


def build_entries() -> list[TraceabilityEntry]:
    return [
        TraceabilityEntry(
            claim_id="C-EXACT-POLY",
            claim="Polynomial reduction/evaluation path is exact under Horner/FMA branch assumptions.",
            rigor_layer="Formal + runtime",
            status="Certified",
            implementation_refs=[
                "poema/compiler.py",
                "MathTest/HornerExact.lean",
                "MathTest/HornerExtract.lean",
            ],
            validation_refs=[
                "tests/test_poema.py",
                "tests/test_functor_engine.py",
            ],
            artifact_refs=[
                "artifacts/poema_hardening_metrics.json",
            ],
            commands=[
                "PYTHONPATH=. python3 -m pytest tests/test_poema.py tests/test_functor_engine.py -q",
                "./lean-4.29.0-rc6-linux/bin/lake build",
            ],
            current_limit="Certified on implemented branch assumptions; does not claim unrestricted global theorem for all runtimes.",
        ),
        TraceabilityEntry(
            claim_id="C-TRANSCENDENTAL-CERT",
            claim="Canonical transcendental branches use constructive interval certificates synchronized to runtime.",
            rigor_layer="Constructive + runtime",
            status="Certified",
            implementation_refs=[
                "python_analysis/generate_interval_certificates.py",
                "MathTest/TranscendentalCertificates.lean",
            ],
            validation_refs=[
                "python_analysis/test_transcendental_integration.py",
                "tests/test_jit_onnx_export.py",
            ],
            artifact_refs=[
                "python_analysis/certificates/transcendental_runtime.py",
            ],
            commands=[
                "PYTHONPATH=. python3 -m pytest python_analysis/test_transcendental_integration.py -q",
            ],
            current_limit="Coverage is explicit for canonical domains/functions; not a claim of universal transcendental closure.",
        ),
        TraceabilityEntry(
            claim_id="C-KOOPMAN-ADAPTIVE",
            claim="Adaptive Koopman diagnostics are implemented and empirically validated on covered benchmark classes.",
            rigor_layer="Empirical systems",
            status="Validated",
            implementation_refs=[
                "acf_functor/koopman_adaptive.py",
                "acf_functor/kolmogorov_entropy.py",
            ],
            validation_refs=[
                "tests/test_invariant_unified.py",
                "tests/test_certified_koopman_extended.py",
            ],
            artifact_refs=[
                "artifacts/cluster_bridge_metrics.json",
                "artifacts/periodic_table.md",
            ],
            commands=[
                "PYTHONPATH=. python3 -m pytest tests/test_invariant_unified.py tests/test_certified_koopman_extended.py -q",
                "python3 benchmarks/periodic_table.py --output artifacts/periodic_table.md",
            ],
            current_limit="General finite-dimensional truncation theorem for broad nonlinear families remains open.",
        ),
        TraceabilityEntry(
            claim_id="C-ALPHA-CONSISTENCY",
            claim="Alpha estimators are tracked with executable discrepancy reporting across canonical functions.",
            rigor_layer="Empirical systems",
            status="Validated",
            implementation_refs=[
                "python_analysis/alpha_consistency_report.py",
            ],
            validation_refs=[
                "tests/test_alpha_consistency_report.py",
            ],
            artifact_refs=[
                "artifacts/alpha_consistency_report.json",
                "artifacts/alpha_consistency_report.md",
            ],
            commands=[
                "PYTHONPATH=. python3 python_analysis/alpha_consistency_report.py --fast --skip-geometric --output-json artifacts/alpha_consistency_report.json --output-md artifacts/alpha_consistency_report.md",
                "PYTHONPATH=. python3 -m pytest tests/test_alpha_consistency_report.py -q",
            ],
            current_limit="Estimator agreement is measured; full theorem-level equivalence is explicitly open.",
        ),
        TraceabilityEntry(
            claim_id="C-SECTION18-CLOSURE",
            claim="Section 18 engineering closure is implemented with reproducible closure tests and benchmarks.",
            rigor_layer="Empirical systems",
            status="Validated",
            implementation_refs=[
                "benchmarks/periodic_table.py",
                "benchmarks/cluster_proxy.py",
                "poema/frontend.py",
            ],
            validation_refs=[
                "tests/test_section18_closure.py",
                "tests/test_symbiotic_convergence.py",
            ],
            artifact_refs=[
                "artifacts/periodic_table.json",
                "artifacts/cluster_proxy_metrics.json",
            ],
            commands=[
                "PYTHONPATH=. python3 -m pytest tests/test_section18_closure.py -q",
                "python3 benchmarks/cluster_proxy.py --steps 5 --output artifacts/cluster_proxy_metrics.json",
            ],
            current_limit="Global theorem closure for all nonlinear classes is outside the Section 18 closure scope.",
        ),
    ]


def build_matrix() -> dict[str, Any]:
    entries = build_entries()
    status_counts: dict[str, int] = {}
    for entry in entries:
        status_counts[entry.status] = status_counts.get(entry.status, 0) + 1

    return {
        "generated_on": date.today().isoformat(),
        "entry_count": len(entries),
        "status_counts": status_counts,
        "entries": [asdict(entry) for entry in entries],
        "note": "This matrix tracks implemented evidence and explicit limits; it does not elevate open research fronts to theorem status.",
    }


def render_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# Traceability Matrix",
        "",
        f"Generated on: {matrix['generated_on']}",
        "",
        "## Summary",
        "",
        f"- Total entries: {matrix['entry_count']}",
    ]

    for status, count in sorted(matrix["status_counts"].items()):
        lines.append(f"- {status}: {count}")

    lines.extend([
        "",
        "## Entries",
        "",
        "| ID | Claim | Rigor | Status |",
        "| --- | --- | --- | --- |",
    ])

    for entry in matrix["entries"]:
        lines.append(
            f"| {entry['claim_id']} | {entry['claim']} | {entry['rigor_layer']} | {entry['status']} |"
        )

    lines.extend([
        "",
        "## Detailed Evidence",
        "",
    ])

    for entry in matrix["entries"]:
        lines.extend(
            [
                f"### {entry['claim_id']}",
                f"- Claim: {entry['claim']}",
                f"- Rigor layer: {entry['rigor_layer']}",
                f"- Status: {entry['status']}",
                "- Implementation:",
            ]
        )
        for item in entry["implementation_refs"]:
            lines.append(f"  - {item}")

        lines.append("- Validation:")
        for item in entry["validation_refs"]:
            lines.append(f"  - {item}")

        lines.append("- Artifacts:")
        for item in entry["artifact_refs"]:
            lines.append(f"  - {item}")

        lines.append("- Reproducible commands:")
        for item in entry["commands"]:
            lines.append(f"  - {item}")

        lines.append(f"- Current limit: {entry['current_limit']}")
        lines.append("")

    return "\n".join(lines)


def write_outputs(matrix: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(matrix), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate claim-to-evidence traceability matrix artifacts.")
    parser.add_argument(
        "--output-json",
        default="artifacts/traceability_matrix.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/traceability_matrix.md",
        help="Output Markdown path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = build_matrix()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    write_outputs(matrix, output_json, output_md)
    print(f"Traceability matrix written: {output_json}")
    print(f"Traceability matrix written: {output_md}")


if __name__ == "__main__":
    main()
