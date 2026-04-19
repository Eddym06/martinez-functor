"""Closure checks for Paper Section 18 artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from poema.ast_nodes import AffineNode, ComposeNode, GeometricType
from poema.compiler import CompilationReport, GeometricTypeChecker


ROOT = Path(__file__).resolve().parents[1]


def test_required_closure_files_exist() -> None:
    required = [
        ROOT / "VALIDATION_STATUS.md",
        ROOT / "SECTION18_CLOSURE.md",
        ROOT / "benchmarks" / "periodic_table.py",
        ROOT / "benchmarks" / "cluster_proxy.py",
        ROOT / "python_analysis" / "cluster_proxy_benchmark.py",
    ]
    for p in required:
        assert p.exists(), f"missing closure artifact: {p}"


def test_paper_has_closed_section18_and_section22() -> None:
    text = (ROOT / "Paper.md").read_text(encoding="utf-8")
    assert "## 18. Five Deepenings Closed Inside the Existing Framework" in text
    assert "## 22. Closure of Section 18 Investigations" in text


def test_periodic_table_script_generates_output(tmp_path: Path) -> None:
    out_file = tmp_path / "periodic_table.md"
    cmd = [
        sys.executable,
        str(ROOT / "benchmarks" / "periodic_table.py"),
        "--output",
        str(out_file),
    ]
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    assert out_file.exists()
    assert "Case" in out_file.read_text(encoding="utf-8")
    assert "Periodic Table generated" in completed.stdout


def test_cluster_proxy_runs_with_fallback(tmp_path: Path) -> None:
    out_file = tmp_path / "cluster_proxy_metrics.json"
    cmd = [
        sys.executable,
        str(ROOT / "benchmarks" / "cluster_proxy.py"),
        "--steps",
        "1",
        "--output",
        str(out_file),
    ]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    assert out_file.exists()
    payload = out_file.read_text(encoding="utf-8")
    assert "baseline_params" in payload
    assert "phi_params" in payload


def test_geometric_checker_emits_stub_warnings() -> None:
    outer = AffineNode(
        scale_factor=2.0,
        shift_value=1.0,
        geometric_type=GeometricType(1, 1, continuity=3, symmetry_group="SO2"),
    )
    inner = AffineNode(
        scale_factor=3.0,
        shift_value=2.0,
        geometric_type=GeometricType(1, 1, continuity=0, symmetry_group="SO3"),
    )
    node = ComposeNode(outer=outer, inner=inner)

    checker = GeometricTypeChecker(target_precision="fp64", auto_compensate=False)
    report = CompilationReport()
    checker.check(node, report)

    assert report.lie_bracket_depth >= 1
    assert any("cohomology stub" in w for w in report.warnings)
