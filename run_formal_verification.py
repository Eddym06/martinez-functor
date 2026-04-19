import torch
import json
from poema.formal_verification import FormalVerificationSuite
import poema.ast_nodes as ast_n
from poema.ast_nodes import ASTNode, InputNode, ScaleNode, ShiftNode
from poema.compiler import FMAInstruction

# Dummy setup for execution
print("\n" + "="*60)
print("  EXECUTING ACF FORMAL VERIFICATION MEASUREMENTS")
print("="*60)

# Initialize formal suite over bounding domain
suite = FormalVerificationSuite(domain=(-2.0, 2.0), resolution=10000)

print("\n[1.1] UNIVERSAL REDUCTION THEOREM (URT) BOUNDS:")
# Case: Truncating a non-linear transcendental function (e.g. sin(x)) with Taylor
def target_function(x): return torch.sin(x)
def phi_koopman_reduction(x): return x - (x**3)/6.0 + (x**5)/120.0
res = suite.verify_urt_truncation(target_function, phi_koopman_reduction)
print("  * Evaluated on sin(x) vs Taylor(d=5):")
print(json.dumps(res, indent=4))

print("\n[1.2] FMA CONSERVATION LAW (STRUCTURAL COUNT):")
# Create AST: 3.0 * (2.0 * x + 1.0)
n_in = InputNode("x")
n_c1 = ScaleNode(torch.tensor([2.0]), n_in)
n_s1 = ShiftNode(torch.tensor([1.0]), n_c1)
n_c2 = ScaleNode(torch.tensor([3.0]), n_s1)

# Lowered conceptually to 2 FMA blocks
fma_seq = [
    FMAInstruction(torch.tensor([[2.0]]), torch.tensor([1.0])),
    FMAInstruction(torch.tensor([[3.0]]), torch.tensor([0.0]))
]
res2 = suite.verify_fma_conservation(n_c2, fma_seq)
print("  * AST graph traversal vs Lowered instructions:")
print(json.dumps(res2, indent=4))

print("\n[1.3] FUNCTORIAL COMPOSITION COMMUTATIVITY (HOMOMORPHISM DELTA):")
# Test Phi(f o g) vs (Phi(f) o Phi(g)) diverges over approximations
def f_true(x): return torch.exp(x)
def g_true(x): return x**2
# Phi equivalents (low degree approximations)
def phi_f(x): return 1 + x + x**2/2
def phi_g(x): return x**2
delta = suite.verify_functorial_composition(f_true, g_true, phi_f, phi_g)
print(f"  * L_inf Divergence ||Phi(f o g) - (Phi(f) o Phi(g))|| : {delta:.5f}")
print("  * (Values > 0 represent categorical breakdown boundary.)")

print("\n[1.4] ALPHA(f) UNIFICATION MEASUREMENT:")
# Generate a sequence of GEMM arrays to expose singular values vs combinatorial depths
block1 = torch.tensor([[1.0, 0.5], [0.0, 2.0]], dtype=torch.float64) # SVD -> max S = 2.06, det = 2.0
block2 = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.float64) # SVD -> max S = 0.5, det = 0.25
fma_matrix_seq = [FMAInstruction(block1, torch.zeros(2)), FMAInstruction(block2, torch.zeros(2))]
res4 = suite.calculate_alpha_indices(fma_matrix_seq)
print("  * Calculated empirical divergencies across equivalent heuristic indexes:")
print(json.dumps(res4, indent=4))

print("\n[1.5] INEXACT REVERSIBILITY BOUNDS:")
# Phi inverse on Koopman approx breaks down structurally
def forward_phi(x): return 2.0 * x
def inverse_phi_approx(x): return 0.50001 * x # Inexact Koopman observable inversion
res5 = suite.verify_reversibility(forward_phi, inverse_phi_approx)
print("  * Evaluated isomorphism exactness on structural bounds:")
print(json.dumps(res5, indent=4))

print("\n" + "="*60)
