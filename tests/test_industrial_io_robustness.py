import os
import sys
import torch
import numpy as np
import tempfile
import onnx
import onnxruntime

# Configuracion de determinismo para tests numericos
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_num_threads(1)

# Ensure poema is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poema.ast_nodes import InputNode, PolynomialNode, AffineNode, ComposeNode, StratifiedNode, TranscendentalNode
from poema.compiler import PoemCompiler
from poema.onnx_export import PoemONNXExporter
from poema.ast_serialization import ASTSerializer

def create_complex_ast():
    poly = PolynomialNode(coefficients=torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64))
    affine = AffineNode(scale_factor=torch.tensor([3.0], dtype=torch.float64), shift_value=torch.tensor([1.0], dtype=torch.float64))
    composed = ComposeNode(outer=affine, inner=poly)
    
    trans = TranscendentalNode(
        name="sin",
        polynomial=poly,
        certified_epsilon=1e-5,
        original_domain=(-1.0, 1.0)
    )
    
    stratified = StratifiedNode(branches=[
        StratifiedNode.Branch(selector_ast=InputNode("x"), body_ast=trans, domain=(-5.0, 0.0)),
        StratifiedNode.Branch(selector_ast=InputNode("x"), body_ast=composed, domain=(0.0, 5.0))
    ])
    
    return stratified

def test_ast_serialization_robustness():
    print("Testing AST Serialization Robustness...")
    ast = create_complex_ast()
    
    # Serialize to JSON
    serialized = ASTSerializer.to_dict(ast)
    
    # Deserialize back to AST
    deserialized_ast = ASTSerializer.from_dict(serialized)
    
    # Compile both and assert absolute equivalence in execution
    compiler = PoemCompiler(target="pytorch", precision="fp64")
    
    executable_orig, _ = compiler.compile(ast)
    executable_recon, _ = compiler.compile(deserialized_ast)
    
    test_tensor = torch.linspace(-5.0, 5.0, 100, dtype=torch.float64)
    
    out_orig = executable_orig(test_tensor)
    out_recon = executable_recon(test_tensor)
    
    # Reemplazo de assertions estrictas imposibles por tolerancias realistas
    torch.testing.assert_close(out_orig, out_recon, rtol=0.0, atol=1e-15, msg="Serialization lost precision")
    print("[SUCCESS] Serialization and Deserialization mathematically robust. Diff <= 1e-15")

def test_onnx_execution_parity_robustness():
    print("Testing ONNX Execution Parity Robustness...")
    ast = create_complex_ast()
    
    compiler = PoemCompiler(target="pytorch", precision="fp64")
    torch_executable, _ = compiler.compile(ast)
    
    # Export to ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "complex_poema_graph.onnx")
        
        # Test inputs to trace bounds if necessary
        exporter = PoemONNXExporter()
        exporter.export(ast, input_shape=(103,), output_path=onnx_path)
        
        
        # Load in ONNXRuntime
        session = onnxruntime.InferenceSession(onnx_path)
        
        # Generate random inputs and explicit critical boundary inputs (Gap fixes)
        boundary_test = torch.tensor([0.0, -5.0, 5.0], dtype=torch.float64)
        test_inputs = torch.cat([
            torch.linspace(-10.0, -0.1, 50, dtype=torch.float64),
            torch.linspace(0.1, 10.0, 50, dtype=torch.float64),
            boundary_test
        ])
        
        # PyTorch Reference Execution
        torch_output = torch_executable(test_inputs).numpy()
        
        # ONNX Runtime Execution
        onnx_input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {onnx_input_name: test_inputs.numpy()})[0]
        
        # Tolerancias realistas asumiendo implementaciones y optimizaciones SIMD distintas, 
        # y alineando con la varianza declarada en el epsilon certificado.
        np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-10, atol=1e-12, err_msg="ONNX Engine Parity Failure")
        max_diff = np.max(np.abs(torch_output - onnx_output))
        print(f"[SUCCESS] ONNX Execution matches PyTorch within realistic limits. Max discrepancy: {max_diff}")

if __name__ == "__main__":
    test_ast_serialization_robustness()
    test_onnx_execution_parity_robustness()
