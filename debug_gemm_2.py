import torch
import torch.nn.functional as F
from poema.compiler import TritonBackend, FMAInstruction

W = torch.rand(16, 16)
b = torch.rand(16)
inst = FMAInstruction(W, b)
wrapper = TritonBackend._compile_vectorial([inst], "gemm_test")
x = torch.rand(16, 8)
y_expected = (W @ x).cuda() + b.cuda().unsqueeze(1)
y_actual = wrapper(x)

print("Diff max:", (y_expected - y_actual).abs().max().item())
print("Expected first row:\n", y_expected[0])
print("Actual first row:\n", y_actual[0])
