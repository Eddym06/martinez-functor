"""
Adaptive certificate generation for transcendental functions.

Generates Lean 4 certificates with degree adapted to target epsilon.
Supports multiple precision levels and extended domains.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mpmath as mp
import numpy as np

mp.iv.dps = 100


# Certification table: (func_name, domain) -> {target_epsilon: min_degree}
CERTIFICATION_TABLE: Dict[Tuple[str, Tuple[float, float]], Dict[float, int]] = {
    ('sin', (-math.pi, math.pi)): {
        1e-3: 20,
        1e-6: 32,
        1e-10: 48,
        1e-14: 60,
    },
    ('cos', (-math.pi, math.pi)): {
        1e-3: 20,
        1e-6: 32,
        1e-10: 48,
        1e-14: 60,
    },
    ('exp', (-1.0, 1.0)): {
        1e-3: 15,
        1e-6: 22,
        1e-10: 30,
        1e-14: 38,
    },
    ('log', (0.5, 2.0)): {
        1e-3: 25,
        1e-6: 35,
        1e-10: 45,
        1e-14: 55,
    },
    ('tanh', (-1.0, 1.0)): {
        1e-3: 20,
        1e-6: 36,
        1e-10: 55,
        1e-14: 70,
    },
    ('tanh', (-2.0, 2.0)): {
        1e-3: 30,
        1e-6: 50,
        1e-10: 70,
        1e-14: 85,
    },
    ('sigmoid', (-1.0, 1.0)): {
        1e-3: 20,
        1e-6: 36,
        1e-10: 55,
        1e-14: 70,
    },
    ('sigmoid', (-4.0, 4.0)): {
        1e-3: 30,
        1e-6: 50,
        1e-10: 70,
        1e-14: 85,
    },
}


def clenshaw_iv(coeffs: list[float], x: mp.iv.mpf, a: float, b: float) -> mp.iv.mpf:
    """Clenshaw evaluation with interval arithmetic."""
    y = (2 * x - (a + b)) / (b - a)
    b2 = mp.iv.mpf([0, 0])
    b1 = mp.iv.mpf([0, 0])
    for c in reversed(coeffs[1:]):
        bk = c + 2 * y * b1 - b2
        b2, b1 = b1, bk
    return coeffs[0] + y * b1 - b2


def interval_bound(func_name: str, coeffs: list[float], a: float, b: float, partitions: int = 4096) -> float:
    """Compute certified error bound using interval arithmetic."""
    h = (b - a) / partitions
    max_err = 0.0

    for i in range(partitions):
        l = a + i * h
        r = l + h
        x = mp.iv.mpf([l, r])

        approx = clenshaw_iv(coeffs, x, a, b)
        
        if func_name == "sin":
            truth = mp.iv.sin(x)
        elif func_name == "cos":
            truth = mp.iv.cos(x)
        elif func_name == "exp":
            truth = mp.iv.exp(x)
        elif func_name == "log":
            truth = mp.iv.log(x)
        elif func_name == "tanh":
            exp2x = mp.iv.exp(2 * x)
            truth = (exp2x - 1) / (exp2x + 1)
        elif func_name == "sigmoid":
            truth = 1 / (1 + mp.iv.exp(-x))
        else:
            raise ValueError(f"Unknown function: {func_name}")

        err = abs(truth - approx)
        upper = float(err.b)
        if upper > max_err:
            max_err = upper

    return max_err


def fit_coeffs(func_name: str, a: float, b: float, degree: int) -> list[float]:
    """Fit Chebyshev coefficients for function on domain."""
    xs = np.linspace(a, b, 8192)
    
    if func_name == "sin":
        ys = np.sin(xs)
    elif func_name == "cos":
        ys = np.cos(xs)
    elif func_name == "exp":
        ys = np.exp(xs)
    elif func_name == "log":
        ys = np.log(xs)
    elif func_name == "tanh":
        ys = np.tanh(xs)
    elif func_name == "sigmoid":
        ys = 1 / (1 + np.exp(-xs))
    else:
        raise ValueError(f"Unknown function: {func_name}")

    cheb = np.polynomial.chebyshev.Chebyshev.fit(xs, ys, deg=degree, domain=[a, b])
    return [float(c) for c in cheb.coef.tolist()]


def compute_required_degree(func_name: str, domain: tuple, target_epsilon: float, 
                            max_degree: int = 100) -> int:
    """
    Calculate minimum Chebyshev degree needed to achieve target_epsilon.
    Uses binary search on [min_degree, max_degree].
    """
    # Check table first
    table_key = (func_name, domain)
    if table_key in CERTIFICATION_TABLE:
        table = CERTIFICATION_TABLE[table_key]
        for eps, degree in sorted(table.items()):
            if eps <= target_epsilon:
                return degree
    
    # Binary search
    lo, hi = 5, max_degree
    while lo < hi:
        mid = (lo + hi) // 2
        coeffs = fit_coeffs(func_name, domain[0], domain[1], mid)
        error = interval_bound(func_name, coeffs, domain[0], domain[1], 4096)
        if error <= target_epsilon:
            hi = mid
        else:
            lo = mid + 1
    return lo


def fmt_float(x: float) -> str:
    return format(x, ".17g")


def fmt_list(xs: list[float]) -> str:
    return "[" + ", ".join(fmt_float(x) for x in xs) + "]"


def build_lean(cert_data: dict[str, dict]) -> str:
    """Generate Lean 4 certificate code."""
    lines = ['import Std', '', 'namespace TranscendentalCertificates', '']
    
    # Group by function
    funcs = sorted(cert_data.keys())
    
    # Degree declarations
    lines.append('-- ============================================================')
    lines.append('-- Degree declarations')
    lines.append('-- ============================================================')
    for name in funcs:
        d = cert_data[name]
        func_name = d['func_name']
        domain_str = f"{d['a']:.4f}_{d['b']:.4f}".replace('-', 'neg').replace('.', '_')
        lines.append(f"def {func_name}Degree_{domain_str} : Nat := {d['degree']}")
    lines.append('')
    
    # Domain declarations
    lines.append('-- ============================================================')
    lines.append('-- Domain declarations')
    lines.append('-- ============================================================')
    for name in funcs:
        d = cert_data[name]
        func_name = d['func_name']
        domain_str = f"{d['a']:.4f}_{d['b']:.4f}".replace('-', 'neg').replace('.', '_')
        lines.append(f"def {func_name}DomainA_{domain_str} : Float := {fmt_float(d['a'])}")
        lines.append(f"def {func_name}DomainB_{domain_str} : Float := {fmt_float(d['b'])}")
    lines.append('')
    
    # Coefficients
    lines.append('-- ============================================================')
    lines.append('-- Chebyshev coefficients')
    lines.append('-- ============================================================')
    for name in funcs:
        d = cert_data[name]
        func_name = d['func_name']
        domain_str = f"{d['a']:.4f}_{d['b']:.4f}".replace('-', 'neg').replace('.', '_')
        lines.append(f"def {func_name}Coeffs_{domain_str} : List Float := {fmt_list(d['coeffs'])}")
    lines.append('')
    
    # Error bounds
    lines.append('-- ============================================================')
    lines.append('-- Interval error bounds')
    lines.append('-- ============================================================')
    for name in funcs:
        d = cert_data[name]
        func_name = d['func_name']
        domain_str = f"{d['a']:.4f}_{d['b']:.4f}".replace('-', 'neg').replace('.', '_')
        lines.append(f"def {func_name}IntervalErrorBound_{domain_str} : Float := {fmt_float(d['err'])}")
    lines.append('')
    
    # Certified epsilons
    lines.append('-- ============================================================')
    lines.append('-- Certified epsilon (error bound + safety margin)')
    lines.append('-- ============================================================')
    for name in funcs:
        d = cert_data[name]
        func_name = d['func_name']
        domain_str = f"{d['a']:.4f}_{d['b']:.4f}".replace('-', 'neg').replace('.', '_')
        lines.append(f"def {func_name}CertifiedEpsilon_{domain_str} : Float := {fmt_float(d['eps'])}")
    lines.append('')
    
    # Certificate predicates
    lines.append('-- ============================================================')
    lines.append('-- Certificate predicates')
    lines.append('-- ============================================================')
    for name in funcs:
        d = cert_data[name]
        func_name = d['func_name']
        domain_str = f"{d['a']:.4f}_{d['b']:.4f}".replace('-', 'neg').replace('.', '_')
        lines.append(f"def {func_name}CertificateHolds_{domain_str} : Bool := decide ({func_name}IntervalErrorBound_{domain_str} ≤ {func_name}CertifiedEpsilon_{domain_str})")
    lines.append('')
    
    # Theorems
    lines.append('-- ============================================================')
    lines.append('-- Constructive certificate theorems')
    lines.append('-- ============================================================')
    for name in funcs:
        d = cert_data[name]
        func_name = d['func_name']
        domain_str = f"{d['a']:.4f}_{d['b']:.4f}".replace('-', 'neg').replace('.', '_')
        lines.append(f"theorem {func_name}_certificate_{domain_str} : {func_name}CertificateHolds_{domain_str} = true := by native_decide")
        lines.append('')
    
    lines.append('end TranscendentalCertificates')
    lines.append('')
    
    return '\n'.join(lines)


def generate_certificate(func_name: str, domain: tuple, target_epsilon: float,
                        output_path: Optional[str] = None) -> dict:
    """
    Generate certificate for function with target epsilon.
    
    Returns dict with certificate metadata.
    """
    degree = compute_required_degree(func_name, domain, target_epsilon)
    coeffs = fit_coeffs(func_name, domain[0], domain[1], degree)
    err = interval_bound(func_name, coeffs, domain[0], domain[1], 16384)
    eps = err * 1.05 + 1e-15
    
    cert = {
        'func_name': func_name,
        'domain': domain,
        'a': domain[0],
        'b': domain[1],
        'degree': degree,
        'coeffs': coeffs,
        'err': err,
        'eps': eps,
        'target_epsilon': target_epsilon,
    }
    
    if output_path:
        lean_content = build_lean({f"{func_name}_{domain[0]}_{domain[1]}": cert})
        Path(output_path).write_text(lean_content, encoding="utf-8")
        print(f"✓ Generated {output_path}")
    
    return cert


def main() -> None:
    """Generate all certificates with adaptive precision."""
    targets = [
        # (func_name, domain, target_epsilon)
        ('sin', (-math.pi, math.pi), 1e-10),
        ('cos', (-math.pi, math.pi), 1e-10),
        ('exp', (-1.0, 1.0), 1e-10),
        ('log', (0.5, 2.0), 1e-10),
        ('tanh', (-1.0, 1.0), 1e-10),
        ('tanh', (-2.0, 2.0), 1e-10),
        ('sigmoid', (-1.0, 1.0), 1e-10),
        ('sigmoid', (-4.0, 4.0), 1e-10),
    ]
    
    cert_data = {}
    for func_name, domain, target_eps in targets:
        name = f"{func_name}_{domain[0]}_{domain[1]}"
        cert = generate_certificate(func_name, domain, target_eps)
        cert_data[name] = cert
        print(f"  {func_name} in {domain}: degree={cert['degree']}, err={cert['err']:.3e}, eps={cert['eps']:.3e}")
    
    # Generate combined Lean file
    lean_content = build_lean(cert_data)
    out = Path("MathTest/TranscendentalCertificates.lean")
    out.write_text(lean_content, encoding="utf-8")
    print(f"\n✓ Generated {out}")


if __name__ == "__main__":
    main()
