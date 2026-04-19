import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poema.ast_nodes import InputNode, PolynomialNode, AffineNode, ComposeNode, StratifiedNode, TranscendentalNode
from poema.compiler import PoemCompiler

def test_deep_propagation_certification():
    print("Testing Analytical Error Propagation across multi-layered graphs...")
    poly = PolynomialNode(coefficients=torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64))
    
    # affine scales outer error by 3.0
    affine = AffineNode(scale_factor=torch.tensor([3.0], dtype=torch.float64), shift_value=torch.tensor([1.0], dtype=torch.float64))
    
    # Base certified node has epsilon 1e-5 and local constant lip=1 (sin)
    trans = TranscendentalNode(
        name="sin",
        polynomial=poly,
        certified_epsilon=1.0e-5,
        original_domain=(-1.0, 1.0)
    )
    
    # Outer composed scale applies affine transformation to transcendental output. 
    # Propagated err: e = 3.0 * (1e-5) = 3e-5
    composed = ComposeNode(outer=affine, inner=trans)
    
    # Same bound on both stratified paths
    stratified = StratifiedNode(branches=[
        StratifiedNode.Branch(selector_ast=InputNode("x"), body_ast=composed, domain=(-5.0, 5.0))
    ])
    
    compiler = PoemCompiler(target="pytorch", precision="fp64")
    _, report = compiler.compile(stratified)
    
    print(f"Deep Tree Computed Epsilon: {report.total_epsilon}")
    
    assert abs(report.total_epsilon - (3.0 * 1.0e-5)) < 1e-12, "Compilation failed to propagate exact scaled error bounds!"
    print("[SUCCESS] Ast Error Propagation correctly tracks topological amplification of error!")
    
if __name__ == "__main__":
    test_deep_propagation_certification()
