from pathlib import Path

from python_analysis.traceability_matrix_report import build_matrix, render_markdown, write_outputs


def test_traceability_matrix_structure(tmp_path: Path) -> None:
    matrix = build_matrix()

    assert matrix["entry_count"] >= 5
    assert "entries" in matrix
    assert isinstance(matrix["entries"], list)

    ids = {entry["claim_id"] for entry in matrix["entries"]}
    assert "C-EXACT-POLY" in ids
    assert "C-TRANSCENDENTAL-CERT" in ids
    assert "C-ALPHA-CONSISTENCY" in ids

    total_status = sum(matrix["status_counts"].values())
    assert total_status == matrix["entry_count"]

    md = render_markdown(matrix)
    assert "# Traceability Matrix" in md
    assert "## Detailed Evidence" in md

    json_out = tmp_path / "traceability.json"
    md_out = tmp_path / "traceability.md"
    write_outputs(matrix, json_out, md_out)

    assert json_out.exists()
    assert md_out.exists()
    assert "C-SECTION18-CLOSURE" in md_out.read_text(encoding="utf-8")
