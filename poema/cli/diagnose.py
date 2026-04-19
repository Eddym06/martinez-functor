"""
CLI de diagnóstico para compilaciones Poema.

Uso:
    python3 -m poema.cli.diagnose "sin(cos(x))" --domain "-pi,pi" --degree 24
    python3 -m poema.cli.diagnose --file my_program.json
"""

import argparse
import math
import sys
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Diagnóstico de compilaciones Poema"
    )
    parser.add_argument("expression", nargs="?", help="Expresión a compilar")
    parser.add_argument("--file", help="Archivo JSON con AST serializado")
    parser.add_argument("--domain", default="-5,5",
                        help="Dominio como 'a,b' (default: -5,5)")
    parser.add_argument("--degree", type=int, default=24,
                        help="Grado para trascendentales (default: 24)")
    parser.add_argument("--precision", default="fp64",
                        choices=["fp32", "fp64"])
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Parsear dominio
    a_str, b_str = args.domain.split(",")
    a = eval(a_str.replace("pi", str(math.pi)).replace("e", str(math.e)))
    b = eval(b_str.replace("pi", str(math.pi)).replace("e", str(math.e)))
    domain = (a, b)
    
    # Importar Poema
    from poema import Poem, PoemCompiler
    from poema.diagnostic import diagnose as poema_diagnose
    
    dtype = torch.float64 if args.precision == "fp64" else torch.float32
    P = Poem(dtype=dtype)
    
    # Compilar
    if args.expression:
        ast = P.continuous_flow(args.expression, domain=domain, degree=args.degree)
    elif args.file:
        from poema.ast_serialization import ast_load
        ast, _ = ast_load(args.file)
    else:
        print("Error: proporcionar expresión o --file", file=sys.stderr)
        sys.exit(1)
    
    compiler = PoemCompiler(target="pytorch", precision=args.precision)
    fn, report = compiler.compile(ast, domain=domain)
    diag = poema_diagnose(ast, domain=domain)
    
    # Imprimir reporte visual
    _print_visual_report(ast, report, diag, args.verbose)


def _print_visual_report(ast, report, diag, verbose):
    """Imprime reporte visual en terminal."""
    
    # Semáforo
    colors = {
        "GREEN":  "\033[92m",
        "YELLOW": "\033[93m",
        "RED":    "\033[91m",
        "RESET":  "\033[0m",
    }
    
    severity = diag.semaforo_global.name
    color = colors.get(severity, "")
    reset = colors["RESET"]
    
    icons = {"GREEN": "✅", "YELLOW": "⚠️ ", "RED": "❌"}
    icon = icons.get(severity, "?")
    
    print(f"\n{'='*60}")
    print(f"  {color}{icon} POEMA DIAGNOSTIC REPORT{reset}")
    print(f"{'='*60}")
    print(f"  Severity:    {color}{severity}{reset}")
    print(f"  ε certified: {report.epsilon_certified:.3e}")
    print(f"  FMA ops:     {report.total_fma_ops}")
    print(f"  Certificate: {report.certificate_source}")
    print(f"  Domain:      violations={report.domain_guard_violations}")
    
    if report.domain_guard_violations > 0:
        print(f"\n  {colors['RED']}⚠ Domain Guard Alerts:{reset}")
        for alert in report.domain_guard_alerts:
            print(f"    • {alert}")
    
    if verbose and report.node_profiles:
        print(f"\n  Node Profiles:")
        print(f"  {'Type':<25} {'FMA':>6} {'ε contrib':>12} {'Domain status'}")
        print(f"  {'-'*25} {'-'*6} {'-'*12} {'-'*15}")
        for profile in report.node_profiles:
            status_color = colors["RED"] if profile.domain_guard_status in ["violation", "warning"] else ""
            print(f"  {profile.node_type:<25} {profile.fma_contribution:>6} "
                  f"{profile.epsilon_contribution:>12.3e} "
                  f"{status_color}{profile.domain_guard_status}{reset}")
    
    if diag.recomendaciones:
        print(f"\n  Recommendations:")
        for rec in diag.recomendaciones:
            print(f"    → {rec}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
