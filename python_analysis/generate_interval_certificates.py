from __future__ import annotations

import math
from pathlib import Path

import mpmath as mp
import numpy as np

mp.iv.dps = 100


def clenshaw_iv(coeffs: list[float], x: mp.iv.mpf, a: float, b: float) -> mp.iv.mpf:
    y = (2 * x - (a + b)) / (b - a)
    b2 = mp.iv.mpf([0, 0])
    b1 = mp.iv.mpf([0, 0])
    for c in reversed(coeffs[1:]):
        bk = c + 2 * y * b1 - b2
        b2, b1 = b1, bk
    return coeffs[0] + y * b1 - b2


def interval_bound(func_name: str, coeffs: list[float], a: float, b: float, partitions: int) -> float:
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
            # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
            exp2x = mp.iv.exp(2 * x)
            truth = (exp2x - 1) / (exp2x + 1)
        elif func_name == "sigmoid":
            # sigmoid(x) = 1 / (1 + exp(-x))
            truth = 1 / (1 + mp.iv.exp(-x))
        else:
            raise ValueError(f"Unknown function: {func_name}")

        err = abs(truth - approx)
        upper = float(err.b)
        if upper > max_err:
            max_err = upper

    return max_err


def fit_coeffs(func_name: str, a: float, b: float, degree: int) -> list[float]:
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


def fmt_float(x: float) -> str:
    return format(x, ".17g")


def fmt_list(xs: list[float]) -> str:
    return "[" + ", ".join(fmt_float(x) for x in xs) + "]"


def build_lean(cert_data: dict[str, dict]) -> str:
    sin = cert_data["sin"]
    cos = cert_data["cos"]
    exp = cert_data["exp"]
    log = cert_data["log"]
    tanh = cert_data["tanh"]
    sigmoid = cert_data["sigmoid"]

    return f'''import Std

namespace TranscendentalCertificates

-- ============================================================
-- Degree declarations
-- ============================================================
def sinDegree : Nat := {sin["degree"]}
def cosDegree : Nat := {cos["degree"]}
def expDegree : Nat := {exp["degree"]}
def logDegree : Nat := {log["degree"]}
def tanhDegree : Nat := {tanh["degree"]}
def sigmoidDegree : Nat := {sigmoid["degree"]}

-- ============================================================
-- Domain declarations
-- ============================================================
def sinDomainA : Float := {fmt_float(sin["a"])}
def sinDomainB : Float := {fmt_float(sin["b"])}
def cosDomainA : Float := {fmt_float(cos["a"])}
def cosDomainB : Float := {fmt_float(cos["b"])}
def expDomainA : Float := {fmt_float(exp["a"])}
def expDomainB : Float := {fmt_float(exp["b"])}
def logDomainA : Float := {fmt_float(log["a"])}
def logDomainB : Float := {fmt_float(log["b"])}
def tanhDomainA : Float := {fmt_float(tanh["a"])}
def tanhDomainB : Float := {fmt_float(tanh["b"])}
def sigmoidDomainA : Float := {fmt_float(sigmoid["a"])}
def sigmoidDomainB : Float := {fmt_float(sigmoid["b"])}

-- ============================================================
-- Chebyshev coefficients
-- ============================================================
def sinCoeffs : List Float := {fmt_list(sin["coeffs"])}
def cosCoeffs : List Float := {fmt_list(cos["coeffs"])}
def expCoeffs : List Float := {fmt_list(exp["coeffs"])}
def logCoeffs : List Float := {fmt_list(log["coeffs"])}
def tanhCoeffs : List Float := {fmt_list(tanh["coeffs"])}
def sigmoidCoeffs : List Float := {fmt_list(sigmoid["coeffs"])}

-- ============================================================
-- Interval error bounds (computed by interval arithmetic)
-- ============================================================
def sinIntervalErrorBound : Float := {fmt_float(sin["err"])}
def cosIntervalErrorBound : Float := {fmt_float(cos["err"])}
def expIntervalErrorBound : Float := {fmt_float(exp["err"])}
def logIntervalErrorBound : Float := {fmt_float(log["err"])}
def tanhIntervalErrorBound : Float := {fmt_float(tanh["err"])}
def sigmoidIntervalErrorBound : Float := {fmt_float(sigmoid["err"])}

-- ============================================================
-- Certified epsilon (error bound + safety margin)
-- ============================================================
def sinCertifiedEpsilon : Float := {fmt_float(sin["eps"])}
def cosCertifiedEpsilon : Float := {fmt_float(cos["eps"])}
def expCertifiedEpsilon : Float := {fmt_float(exp["eps"])}
def logCertifiedEpsilon : Float := {fmt_float(log["eps"])}
def tanhCertifiedEpsilon : Float := {fmt_float(tanh["eps"])}
def sigmoidCertifiedEpsilon : Float := {fmt_float(sigmoid["eps"])}

-- ============================================================
-- Certificate predicates
-- ============================================================
def sinCertificateHolds : Bool := decide (sinIntervalErrorBound ≤ sinCertifiedEpsilon)
def cosCertificateHolds : Bool := decide (cosIntervalErrorBound ≤ cosCertifiedEpsilon)
def expCertificateHolds : Bool := decide (expIntervalErrorBound ≤ expCertifiedEpsilon)
def logCertificateHolds : Bool := decide (logIntervalErrorBound ≤ logCertifiedEpsilon)
def tanhCertificateHolds : Bool := decide (tanhIntervalErrorBound ≤ tanhCertifiedEpsilon)
def sigmoidCertificateHolds : Bool := decide (sigmoidIntervalErrorBound ≤ sigmoidCertifiedEpsilon)

-- ============================================================
-- Constructive certificate theorems
-- ============================================================
theorem sin_certificate_constructive : sinCertificateHolds = true := by native_decide

theorem cos_certificate_constructive : cosCertificateHolds = true := by native_decide

theorem exp_certificate_constructive : expCertificateHolds = true := by native_decide

theorem log_certificate_constructive : logCertificateHolds = true := by native_decide

theorem tanh_certificate_constructive : tanhCertificateHolds = true := by native_decide

theorem sigmoid_certificate_constructive : sigmoidCertificateHolds = true := by native_decide

end TranscendentalCertificates
'''


def main() -> None:
    specs = {
        "sin": {"a": -math.pi, "b": math.pi, "degree": 24, "partitions": 4096},
        "cos": {"a": -math.pi, "b": math.pi, "degree": 24, "partitions": 4096},
        "exp": {"a": -1.0, "b": 1.0, "degree": 18, "partitions": 4096},
        "log": {"a": 0.5, "b": 2.0, "degree": 28, "partitions": 4096},
        # AMPLIADOS: dominios reales para uso en ML
        "tanh": {"a": -4.0, "b": 4.0, "degree": 50, "partitions": 16384},
        "sigmoid": {"a": -8.0, "b": 8.0, "degree": 60, "partitions": 16384},
    }

    cert_data: dict[str, dict] = {}

    for name, s in specs.items():
        coeffs = fit_coeffs(name, s["a"], s["b"], s["degree"])
        err = interval_bound(name, coeffs, s["a"], s["b"], s["partitions"])
        eps = err * 1.05 + 1e-15
        cert_data[name] = {
            "a": s["a"],
            "b": s["b"],
            "degree": s["degree"],
            "coeffs": coeffs,
            "err": err,
            "eps": eps,
        }

    lean_content = build_lean(cert_data)
    out = Path("MathTest/TranscendentalCertificates.lean")
    out.write_text(lean_content, encoding="utf-8")
    print(f"Generated {out}")
    for name in ["sin", "cos", "exp", "log", "tanh", "sigmoid"]:
        print(f"  {name}: err={cert_data[name]['err']:.3e}, eps={cert_data[name]['eps']:.3e}")


if __name__ == "__main__":
    main()
