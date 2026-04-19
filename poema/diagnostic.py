"""
Poema Diagnostic Tool.

Provides human-readable diagnosis of compilation results with
traffic-light severity indicators and actionable recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .ast_nodes import ASTNode
from .compiler import CompilationReport, PoemCompiler


class Severity(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass
class DiagnosticIssue:
    severity: Severity
    category: str
    message: str
    recommendation: str


@dataclass
class DiagnosticReport:
    semaforo_global: Severity = Severity.GREEN
    problemas: List[DiagnosticIssue] = field(default_factory=list)
    recomendaciones: List[str] = field(default_factory=list)
    metricas: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        color = {
            Severity.GREEN: "🟢",
            Severity.YELLOW: "🟡",
            Severity.RED: "🔴",
        }[self.semaforo_global]

        lines = [
            f"{color} POEMA DIAGNOSTIC REPORT",
            f"  Severity: {self.semaforo_global.value}",
            f"  Issues: {len(self.problemas)}",
            f"  Recommendations: {len(self.recomendaciones)}",
            "",
        ]

        if self.problemas:
            lines.append("  Issues:")
            for issue in self.problemas:
                icon = {"GREEN": "✓", "YELLOW": "⚠", "RED": "✗"}[issue.severity.value]
                lines.append(f"    [{icon}] {issue.category}: {issue.message}")
                lines.append(f"        → {issue.recommendation}")
            lines.append("")

        if self.metricas:
            lines.append("  Metrics:")
            for key, val in self.metricas.items():
                lines.append(f"    {key}: {val}")

        return "\n".join(lines)


def diagnose(
    ast: ASTNode,
    domain: Tuple[float, float],
    compiler_options: Optional[Dict[str, Any]] = None,
    test_points: int = 100,
) -> DiagnosticReport:
    """
    Ejecuta análisis completo y produce reporte human-readable.

    Args:
        ast: AST Poema a diagnosticar
        domain: Dominio de entrada
        compiler_options: Opciones para PoemCompiler
        test_points: Número de puntos de prueba para verificación rápida

    Returns:
        DiagnosticReport con semáforo, problemas, recomendaciones y métricas.
    """
    compiler_options = compiler_options or {}
    report = DiagnosticReport()

    # Compile
    compiler = PoemCompiler(
        target=compiler_options.get("target", "pytorch"),
        precision=compiler_options.get("precision", "fp64"),
        auto_domain_repair=compiler_options.get("auto_domain_repair", True),
        enable_self_modulation=compiler_options.get("enable_self_modulation", False),
    )

    try:
        fn, comp_report = compiler.compile(ast, domain=domain)
    except Exception as e:
        report.semaforo_global = Severity.RED
        report.problemas.append(DiagnosticIssue(
            severity=Severity.RED,
            category="compilation",
            message=f"Compilation failed: {e}",
            recommendation="Review AST structure and type compatibility",
        ))
        return report

    # Collect metrics
    report.metricas = {
        "fma_ops": comp_report.total_fma_ops,
        "epsilon": f"{comp_report.total_epsilon:.3e}",
        "domain_guard_violations": comp_report.domain_guard_violations,
        "domain_guard_max_overshoot": f"{comp_report.domain_guard_max_overshoot:.3e}",
        "simplifications": comp_report.simplifications_applied,
        "compilation_time_ms": f"{comp_report.compilation_time_ms:.2f}",
        "certificate_source": comp_report.certificate_source or "none",
    }

    # Check domain guard violations
    if comp_report.domain_guard_violations > 0:
        overshoot = comp_report.domain_guard_max_overshoot
        if overshoot > 0.5:
            report.semaforo_global = Severity.RED
            report.problemas.append(DiagnosticIssue(
                severity=Severity.RED,
                category="domain_guard",
                message=f"{comp_report.domain_guard_violations} violations, overshoot={overshoot:.3f}",
                recommendation="Reduce input domain or increase polynomial degree",
            ))
        else:
            if report.semaforo_global == Severity.GREEN:
                report.semaforo_global = Severity.YELLOW
            report.problemas.append(DiagnosticIssue(
                severity=Severity.YELLOW,
                category="domain_guard",
                message=f"{comp_report.domain_guard_violations} violations, overshoot={overshoot:.3f}",
                recommendation="Consider auto_domain_repair=True or narrower domain",
            ))

    # Check epsilon
    eps = comp_report.total_epsilon
    if eps > 1e-3:
        if report.semaforo_global == Severity.GREEN:
            report.semaforo_global = Severity.YELLOW
        report.problemas.append(DiagnosticIssue(
            severity=Severity.YELLOW,
            category="precision",
            message=f"High epsilon: {eps:.3e}",
            recommendation="Increase polynomial degree or narrow domain",
        ))
    elif eps > 1e-6:
        if report.semaforo_global == Severity.GREEN:
            report.semaforo_global = Severity.YELLOW
        report.problemas.append(DiagnosticIssue(
            severity=Severity.YELLOW,
            category="precision",
            message=f"Moderate epsilon: {eps:.3e}",
            recommendation="Acceptable for most uses; increase degree for higher precision",
        ))

    # Check certificate source
    cert_source = comp_report.certificate_source or ""
    if cert_source == "local_estimate":
        if report.semaforo_global == Severity.GREEN:
            report.semaforo_global = Severity.YELLOW
        report.problemas.append(DiagnosticIssue(
            severity=Severity.YELLOW,
            category="certification",
            message="No Lean certificate (local estimate only)",
            recommendation="Use canonical domain for certified compilation",
        ))

    # Quick runtime test
    try:
        a, b = domain
        x = torch.linspace(a, b, test_points, dtype=torch.float64)
        y = fn(x)

        if not torch.all(torch.isfinite(y)):
            report.semaforo_global = Severity.RED
            nans = int(torch.sum(~torch.isfinite(y)).item())
            report.problemas.append(DiagnosticIssue(
                severity=Severity.RED,
                category="runtime",
                message=f"{nans}/{test_points} non-finite outputs",
                recommendation="Enable auto_domain_repair or check domain bounds",
            ))
    except Exception as e:
        report.semaforo_global = Severity.RED
        report.problemas.append(DiagnosticIssue(
            severity=Severity.RED,
            category="runtime",
            message=f"Runtime error: {e}",
            recommendation="Check input tensor dtype and device",
        ))

    # Check for warnings
    for warning in comp_report.warnings:
        if report.semaforo_global == Severity.GREEN:
            report.semaforo_global = Severity.YELLOW
        report.problemas.append(DiagnosticIssue(
            severity=Severity.YELLOW,
            category="warning",
            message=warning,
            recommendation="Review warning context",
        ))

    # Generate recommendations
    if report.semaforo_global == Severity.GREEN:
        report.recomendaciones.append("Compilation is healthy. No action needed.")
    else:
        if comp_report.total_fma_ops > 100:
            report.recomendaciones.append(
                f"High FMA count ({comp_report.total_fma_ops}). "
                "Consider simplifying the expression or using target='triton' for GPU execution."
            )
        if comp_report.domain_guard_violations > 0:
            report.recomendaciones.append(
                "Domain guard violations detected. "
                "Narrow the input domain or increase the polynomial degree."
            )
        if cert_source == "local_estimate":
            report.recomendaciones.append(
                "No Lean certificate available. "
                "Use a canonical domain for certified compilation."
            )

    return report
