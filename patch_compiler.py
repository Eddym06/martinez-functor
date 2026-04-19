import re

with open("poema/compiler.py", "r") as f:
    code = f.read()

vector_compile_code = """
        # Massive Matrix / Vectorial GEMM Operations via Triton natively
        import triton
        import triton.language as tl
        
        W_dev = W_total.contiguous().to(device='cuda', dtype=torch.float32)
        b_dev = b_total.contiguous().view(-1).to(device='cuda', dtype=torch.float32)
        
        @triton.jit
        def _gemm_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr, stride_cm: tl.constexpr, stride_cn: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            pid_m = pid % num_pid_m
            pid_n = pid // num_pid_m
            
            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk
                
            c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
            c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
            tl.store(c_ptrs, accumulator, mask=c_mask)

        def wrapper(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 1:
                x = x.unsqueeze(1)
            x_dev = x.contiguous().to(device='cuda', dtype=torch.float32)
            
            M, K = W_dev.shape
            K_x, N = x_dev.shape
            assert K == K_x, "Incompatible dimensions"
            
            y_dev = torch.empty((M, N), device='cuda', dtype=torch.float32)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
            _gemm_kernel[grid](W_dev, x_dev, y_dev, M, N, K, W_dev.stride(0), W_dev.stride(1), x_dev.stride(0), x_dev.stride(1), y_dev.stride(0), y_dev.stride(1), BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32)
            
            return y_dev + b_dev.unsqueeze(1)
            
        wrapper.__name__ = kernel_name
        return wrapper
"""

code = re.sub(
    r"# Use torch.matmul-based wrapper.*?return wrapper",
    vector_compile_code.strip(),
    code,
    flags=re.DOTALL
)

with open("poema/compiler.py", "w") as f:
    f.write(code)
