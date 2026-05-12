import math
import torch
import json
from poema.formal_verification import FormalVerificationSuite, Interval
import poema.ast_nodes as ast_n
from poema.ast_nodes import ASTNode, InputNode, ScaleNode, ShiftNode, TranscendentalNode, PolynomialNode
from poema.compiler import FMAInstruction

print("\n" + "="*60)
print("  EXECUTING ENHANCED FORMAL VERIFICATION SUITE")
print("="*60)

# Initialize formal suite over bounding domain
suite = FormalVerificationSuite(domain=(-math.pi/2, math.pi/2), resolution=5000)

print("\n[1.1] URT WITH HIGH-PRECISION REFERENCE:")
def sin_torch(x): return torch.sin(x)
def sin_taylor(x): return x - (x**3)/6.0 + (x**5)/120.0
res = suite.verify_urt_truncation(sin_torch, sin_taylor, use_high_precision=True)
print("  * sin(x) vs Taylor(d=5) with mpmath reference:")
print(json.dumps(res, indent=4))

print("\n[1.2] FMA CONSERVATION WITH CONDITION NUMBERS:")
n_in = InputNode("x")
n_c1 = ScaleNode(torch.tensor([2.0]), n_in)
n_s1 = ShiftNode(torch.tensor([1.0]), n_c1)
n_c2 = ScaleNode(torch.tensor([3.0]), n_s1)

# Matrix with high condition number
W_bad = torch.tensor([[1e6, 0], [0, 1e-6]], dtype=torch.float64)
fma_seq = [
    FMAInstruction(torch.tensor([[2.0]]), torch.tensor([1.0])),
    FMAInstruction(W_bad, torch.zeros(2))
]
res2 = suite.verify_fma_conservation(n_c2, fma_seq)
print("  * AST conservation with condition monitoring:")
print(json.dumps(res2, indent=4))

print("\n[1.3] FUNCTORIAL COMPOSITION WITH LIPSCHITZ ERROR:")
def f(x): return torch.exp(x)
def g(x): return x**2
def phi_f(x): return 1 + x + x**2/2
def phi_g(x): return x**2
res3 = suite.verify_functorial_composition(f, g, phi_f, phi_g, error_f=0.01, error_g=0.005)
print("  * Composition with Lipschitz error propagation:")
print(json.dumps(res3, indent=4))

print("\n[1.4] ALPHA INDICES WITH MATRIX COMPOSITION:")
W1 = torch.tensor([[1.0, 0.5], [0.0, 2.0]], dtype=torch.float64)
W2 = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.float64)
fma_matrix_seq = [FMAInstruction(W1, torch.zeros(2)), FMAInstruction(W2, torch.zeros(2))]
res4 = suite.calculate_alpha_indices(fma_matrix_seq)
print("  * Alpha indices via matrix composition (correct):")
print(json.dumps(res4, indent=4))

print("\n[1.5] REVERSIBILITY & STRATIFIED CONTINUITY:")
def forward(x): return 2.0 * x
def inverse(x): return 0.50001 * x
res5 = suite.verify_reversibility(forward, inverse)
print("  * Reversibility bounds:")
print(json.dumps(res5, indent=4))

# Piecewise function test
def piecewise_test(x):
    return torch.where(x > 0, x**2, -x**2)
res6 = suite.verify_stratified_continuity(piecewise_test, boundaries=[0.0])
print("  * Stratified continuity verification:")
print(json.dumps(res6, indent=4))

print("\n[STRESS TEST] COMPOSITIONAL FUZZING:")
res7 = suite.stress_test_composition(depth=10, n_trials=50)
print("  * Deep composition stress test:")
print(json.dumps(res7, indent=4))

print("\n[INTERVAL PROPAGATION] DOMAIN VIOLATION:")
input_interval = Interval(-2.0, 2.0)
# Build AST: sin(2*x + 1)
n_in = InputNode("x")
n_scale = ScaleNode(torch.tensor([2.0]), n_in)
n_shift = ShiftNode(torch.tensor([1.0]), n_scale)
n_poly = PolynomialNode([1.0], input_node=n_shift)
final_poly_int = suite.interval_propagator(n_poly, input_interval) # Dummy propagate
n_sin = TranscendentalNode("sin", polynomial=n_poly, certified_epsilon=1e-5, original_domain=(-math.pi, math.pi))
final_interval = suite.interval_propagator(n_sin, input_interval)
print(f"  * Interval propagation: input [{input_interval.lower}, {input_interval.upper}]")
print(f"    -> sin(2*x+1) output [{final_interval.lower}, {final_interval.upper}]")

print("\n[LEAN CERTIFICATES] VERIFICATION:")
res8 = suite.verify_lean_certificates()
print("  * Lean certificate validation:")
print(json.dumps(res8, indent=4))

print("\n" + "="*60)
