import torch
from poema.compiler import TritonBackend, FMAInstruction
from poema.frontend import Poem

def test_triton_gemm():
    print("Testing Triton GEMM...", flush=True)
    try:
        import triton
    except ImportError:
        print("Triton not installed.")
        return
        
    W = torch.rand(16, 16)
    b = torch.rand(16)
    
    # We create an instruction
    inst = FMAInstruction(W, b)
    wrapper = TritonBackend._compile_vectorial([inst], "gemm_test")
    
    x = torch.rand(16, 8)
    y_expected = W @ x + b.unsqueeze(1)
    
    # Call the wrapper (which moves things to GPU)
    y_actual = wrapper(x)
    diff = torch.max(torch.abs(y_expected.cuda() - y_actual))
    print(f"Max diff: {diff.item()}")
    assert diff < 1e-4

def test_parser_turing():
    print("Testing Parser Turing completeness...")
    # let's test if parser works for "if(x>0) x else -x"
    try:
        poema = Poem(domain=(-1.0, 1.0), degree=5)
        # Using Poema's direct parse... the parser is `_RecursiveDescentParser` called via Poema methods?
        # Actually in `frontend.py`: `_parse_expr` inside `_RecursiveDescentParser`.
        from poema.frontend import _RecursiveDescentParser
        parser = _RecursiveDescentParser("if (x>0) x else -x", poema, (-1.0, 1.0), 5)
        ast = parser.parse()
        print("Parsed if/else:", type(ast))
    except Exception as e:
        print("Error parsing:", e)

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_triton_gemm()
    else:
        print("No CUDA, skipping GEMM.")
    test_parser_turing()
