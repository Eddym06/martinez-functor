import torch
from poema.compiler import TritonBackend, FMAInstruction

W = torch.ones(2, 2)
b = torch.zeros(2)
inst = FMAInstruction(W, b)
wrapper = TritonBackend._compile_vectorial([inst], "gemm_test")

x = torch.ones(2, 2)
y_expected = W @ x + b.unsqueeze(1)
y_actual = wrapper(x)
print("Expected:\n", y_expected)
print("Actual:\n", y_actual)
