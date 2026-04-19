from __future__ import annotations

from python_analysis.alpha_consistency_report import build_alpha_consistency_report


def test_alpha_consistency_report_fast_mode_structure() -> None:
    report = build_alpha_consistency_report(
        consistency_threshold=0.5,
        skip_geometric=True,
        fast_mode=True,
    )

    assert "summary" in report
    assert "results" in report
    assert report["summary"]["n_functions"] >= 3
    assert len(report["results"]) == report["summary"]["n_functions"]

    for row in report["results"]:
        assert "function" in row
        assert "alpha_combinatorial" in row
        assert "alpha_spectral" in row
        assert "max_discrepancy" in row
        assert "best_definition" in row
