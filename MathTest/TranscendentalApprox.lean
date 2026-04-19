import Std
import MathTest.TranscendentalCertificates

namespace TranscendentalApprox

open TranscendentalCertificates

/-
  Stronger certification layer:
  - bounds are no longer axiomatic in this module,
  - they are imported from constructive interval certificates,
  - certificate booleans are proven by `native_decide` in
    MathTest/TranscendentalCertificates.lean.
-/

def sin_bound_canonical : Bool := sinCertificateHolds
def exp_bound_canonical : Bool := expCertificateHolds
def log_bound_canonical : Bool := logCertificateHolds

theorem sin_bound_canonical_proved : sin_bound_canonical = true :=
  sin_certificate_constructive

theorem exp_bound_canonical_proved : exp_bound_canonical = true :=
  exp_certificate_constructive

theorem log_bound_canonical_proved : log_bound_canonical = true :=
  log_certificate_constructive

def pythonModule : String :=
  String.intercalate "" [
    "# AUTO-GENERADO desde MathTest/TranscendentalApprox.lean\n",
    "# NO MODIFICAR MANUALMENTE\n",
    "\n",
    "from dataclasses import dataclass\n",
    "from typing import List, Tuple\n",
    "import math\n",
    "import torch\n",
    "\n",
    "@dataclass(frozen=True)\n",
    "class ChebyshevApproximation:\n",
    "    function_name: str\n",
    "    coeffs: List[float]\n",
    "    domain_a: float\n",
    "    domain_b: float\n",
    "    degree: int\n",
    "    epsilon: float\n",
    "    lean_theorem: str\n",
    "    lean_source: str = \"MathTest/TranscendentalApprox.lean\"\n",
    "\n",
    "def clenshaw_eval(coeffs: List[float], x: float, domain: Tuple[float, float]) -> float:\n",
    "    if not coeffs:\n",
    "        return 0.0\n",
    "    a, b = domain\n",
    "    y = (2.0 * x - (a + b)) / (b - a)\n",
    "    y = max(-1.0, min(1.0, y))\n",
    "    b_k2 = 0.0\n",
    "    b_k1 = 0.0\n",
    "    for c in reversed(coeffs[1:]):\n",
    "        b_k = c + 2.0 * y * b_k1 - b_k2\n",
    "        b_k2 = b_k1\n",
    "        b_k1 = b_k\n",
    "    return coeffs[0] + y * b_k1 - b_k2\n",
    "\n",
    "def clenshaw_eval_tensor(coeffs: List[float], x: torch.Tensor, domain: Tuple[float, float]) -> torch.Tensor:\n",
    "    a, b = domain\n",
    "    y = (2.0 * x - (a + b)) / (b - a)\n",
    "    y = torch.clamp(y, -1.0, 1.0)\n",
    "    b_k2 = torch.zeros_like(x)\n",
    "    b_k1 = torch.zeros_like(x)\n",
    "    for c in reversed(coeffs[1:]):\n",
    "        b_k = c + 2.0 * y * b_k1 - b_k2\n",
    "        b_k2 = b_k1\n",
    "        b_k1 = b_k\n",
    "    return coeffs[0] + y * b_k1 - b_k2\n",
    "\n",
    s!"_SIN_COEFFS = [{String.intercalate ", " (sinCoeffs.map (fun x => toString x))}]\n",
    s!"_EXP_COEFFS = [{String.intercalate ", " (expCoeffs.map (fun x => toString x))}]\n",
    s!"_LOG_COEFFS = [{String.intercalate ", " (logCoeffs.map (fun x => toString x))}]\n",
    s!"_SIN_EPS = {toString sinCertifiedEpsilon}\n",
    s!"_EXP_EPS = {toString expCertifiedEpsilon}\n",
    s!"_LOG_EPS = {toString logCertifiedEpsilon}\n",
    "\n",
    "def chebyshev_sin(n: int = 20, epsilon: float = 1e-6) -> ChebyshevApproximation:\n",
    "    return ChebyshevApproximation(\n",
    "        function_name=\"sin\",\n",
    "        coeffs=_SIN_COEFFS,\n",
    "        domain_a=-math.pi,\n",
    "        domain_b=math.pi,\n",
    "        degree=20,\n",
    "        epsilon=_SIN_EPS,\n",
    "        lean_theorem=\"sin_bound_canonical_proved\",\n",
    "    )\n",
    "\n",
    "def chebyshev_exp(n: int = 15, epsilon: float = 1e-6) -> ChebyshevApproximation:\n",
    "    return ChebyshevApproximation(\n",
    "        function_name=\"exp\",\n",
    "        coeffs=_EXP_COEFFS,\n",
    "        domain_a=-1.0,\n",
    "        domain_b=1.0,\n",
    "        degree=15,\n",
    "        epsilon=_EXP_EPS,\n",
    "        lean_theorem=\"exp_bound_canonical_proved\",\n",
    "    )\n",
    "\n",
    "def chebyshev_log(n: int = 25, epsilon: float = 1e-6) -> ChebyshevApproximation:\n",
    "    return ChebyshevApproximation(\n",
    "        function_name=\"log\",\n",
    "        coeffs=_LOG_COEFFS,\n",
    "        domain_a=0.5,\n",
    "        domain_b=2.0,\n",
    "        degree=25,\n",
    "        epsilon=_LOG_EPS,\n",
    "        lean_theorem=\"log_bound_canonical_proved\",\n",
    "    )\n",
    "\n",
    "def reduce_sin_domain(x: float):\n",
    "    two_pi = 2.0 * math.pi\n",
    "    k = round(x / two_pi)\n",
    "    x_red = x - k * two_pi\n",
    "    if x_red > math.pi:\n",
    "        x_red -= two_pi\n",
    "    elif x_red < -math.pi:\n",
    "        x_red += two_pi\n",
    "    return x_red, int(k)\n",
    "\n",
    "def eval_sin_complete(x: float, approx: ChebyshevApproximation) -> float:\n",
    "    xr, _ = reduce_sin_domain(x)\n",
    "    return clenshaw_eval(approx.coeffs, xr, (approx.domain_a, approx.domain_b))\n",
    "\n",
    "def eval_exp_complete(x: float, approx: ChebyshevApproximation) -> float:\n",
    "    k = round(x)\n",
    "    frac = x - k\n",
    "    return (math.e ** k) * clenshaw_eval(approx.coeffs, frac, (approx.domain_a, approx.domain_b))\n"
  ]

def emitPythonModule : IO Unit := do
  IO.FS.writeFile "python_analysis/transcendental_generated.py" pythonModule
  IO.println "Generated python_analysis/transcendental_generated.py"

end TranscendentalApprox

def main : IO Unit :=
  TranscendentalApprox.emitPythonModule
