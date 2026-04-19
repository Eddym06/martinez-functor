# AUTO-GENERADO desde MathTest/TranscendentalApprox.lean
# NO MODIFICAR MANUALMENTE

from dataclasses import dataclass
from typing import List, Tuple
import math
import torch

@dataclass(frozen=True)
class ChebyshevApproximation:
    function_name: str
    coeffs: List[float]
    domain_a: float
    domain_b: float
    degree: int
    epsilon: float
    lean_theorem: str
    lean_source: str = "MathTest/TranscendentalApprox.lean"

def clenshaw_eval(coeffs: List[float], x: float, domain: Tuple[float, float]) -> float:
    if not coeffs:
        return 0.0
    a, b = domain
    y = (2.0 * x - (a + b)) / (b - a)
    y = max(-1.0, min(1.0, y))
    b_k2 = 0.0
    b_k1 = 0.0
    for c in reversed(coeffs[1:]):
        b_k = c + 2.0 * y * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k
    return coeffs[0] + y * b_k1 - b_k2

def clenshaw_eval_tensor(coeffs: List[float], x: torch.Tensor, domain: Tuple[float, float]) -> torch.Tensor:
    a, b = domain
    y = (2.0 * x - (a + b)) / (b - a)
    y = torch.clamp(y, -1.0, 1.0)
    b_k2 = torch.zeros_like(x)
    b_k1 = torch.zeros_like(x)
    for c in reversed(coeffs[1:]):
        b_k = c + 2.0 * y * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k
    return coeffs[0] + y * b_k1 - b_k2

_SIN_COEFFS = [0.000000, 0.569231, -0.000000, -0.666917, 0.000000, 0.104282, 0.000000, -0.006841, 0.000000, 0.000250, -0.000000, -0.000006, 0.000000, 0.000000, 0.000000, -0.000000, 0.000000, 0.000000, 0.000000, -0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
_EXP_COEFFS = [1.266066, 1.130318, 0.271495, 0.044337, 0.005474, 0.000543, 0.000045, 0.000003, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.000000, -0.000000, 0.000000, -0.000000]
_LOG_COEFFS = [0.117783, 0.666667, -0.111111, 0.024691, -0.006173, 0.001646, -0.000457, 0.000131, -0.000038, 0.000011, -0.000003, 0.000001, -0.000000, 0.000000, -0.000000, 0.000000, -0.000000, 0.000000, -0.000000, 0.000000, -0.000000, 0.000000, -0.000000, 0.000000, -0.000000, 0.000000, -0.000000, 0.000000, -0.000000]
_SIN_EPS = 0.004255
_EXP_EPS = 0.001479
_LOG_EPS = 0.000944

def chebyshev_sin(n: int = 20, epsilon: float = 1e-6) -> ChebyshevApproximation:
    return ChebyshevApproximation(
        function_name="sin",
        coeffs=_SIN_COEFFS,
        domain_a=-math.pi,
        domain_b=math.pi,
        degree=20,
        epsilon=_SIN_EPS,
        lean_theorem="sin_bound_canonical_proved",
    )

def chebyshev_exp(n: int = 15, epsilon: float = 1e-6) -> ChebyshevApproximation:
    return ChebyshevApproximation(
        function_name="exp",
        coeffs=_EXP_COEFFS,
        domain_a=-1.0,
        domain_b=1.0,
        degree=15,
        epsilon=_EXP_EPS,
        lean_theorem="exp_bound_canonical_proved",
    )

def chebyshev_log(n: int = 25, epsilon: float = 1e-6) -> ChebyshevApproximation:
    return ChebyshevApproximation(
        function_name="log",
        coeffs=_LOG_COEFFS,
        domain_a=0.5,
        domain_b=2.0,
        degree=25,
        epsilon=_LOG_EPS,
        lean_theorem="log_bound_canonical_proved",
    )

def reduce_sin_domain(x: float):
    two_pi = 2.0 * math.pi
    k = round(x / two_pi)
    x_red = x - k * two_pi
    if x_red > math.pi:
        x_red -= two_pi
    elif x_red < -math.pi:
        x_red += two_pi
    return x_red, int(k)

def eval_sin_complete(x: float, approx: ChebyshevApproximation) -> float:
    xr, _ = reduce_sin_domain(x)
    return clenshaw_eval(approx.coeffs, xr, (approx.domain_a, approx.domain_b))

def eval_exp_complete(x: float, approx: ChebyshevApproximation) -> float:
    k = round(x)
    frac = x - k
    return (math.e ** k) * clenshaw_eval(approx.coeffs, frac, (approx.domain_a, approx.domain_b))
