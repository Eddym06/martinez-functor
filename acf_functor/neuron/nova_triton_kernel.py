"""
nova_triton_kernel.py — Kernels Triton para Nova (RTX 4050)
=============================================================

Kernels GPU funcionales: Hermite, predicción, y actualización BESD.

Autor: AXIOM-1
"""

import numpy as np

try:
    import triton
    import triton.language as tl
    import torch
    HAS_TRITON = torch.cuda.is_available()
except (ImportError, AssertionError):
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def hermite_batch_kernel(
        X_ptr, H_ptr, n_samples: int, n_input: int, max_degree: int,
        stride_x: int, BLOCK_S: tl.constexpr = 256,
    ):
        """Evaluar Hermite: (n_input, n_samples) → (deg+1, n_input, n_samples)."""
        dim_idx = tl.program_id(0)
        block_idx = tl.program_id(1)
        s_offs = block_idx * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = s_offs < n_samples
        x = tl.load(X_ptr + dim_idx * stride_x + s_offs, mask=mask, other=0.0)
        deg_stride = n_input * n_samples
        tl.store(H_ptr + 0 * deg_stride + dim_idx * n_samples + s_offs, 1.0, mask=mask)
        if max_degree >= 1:
            tl.store(H_ptr + 1 * deg_stride + dim_idx * n_samples + s_offs, x, mask=mask)
        hp = tl.full((BLOCK_S,), 1.0, dtype=tl.float32)  # He_0
        hc = x  # He_1
        for k in range(2, max_degree + 1):
            hn = x * hc - (k - 1.0) * hp
            tl.store(H_ptr + k * deg_stride + dim_idx * n_samples + s_offs, hn, mask=mask)
            hp = hc
            hc = hn


    @triton.jit
    def update_4x4_kernel(
        H_ptr, C_ptr, Y_ptr, lam: float,
        n_samples: int, n_input: int, n_output: int, d1: int,
        BLOCK_S: tl.constexpr = 128,
    ):
        """
        Actualizar TODOS los coeficientes principales (un paso BESD en GPU).
        Grid: (n_input,) — cada programa = una dimensión.
        Resuelve ridge 4×4 exacto con Cholesky en registros.
        """
        dim_idx = tl.program_id(0)
        deg_stride = n_input * n_samples
        
        # Construir Gram A = ΦᵀΦ + λI
        a00=lam; a01=0.0; a02=0.0; a03=0.0
        a11=lam; a12=0.0; a13=0.0
        a22=lam; a23=0.0; a33=lam
        
        for s_start in range(0, n_samples, BLOCK_S):
            s = s_start + tl.arange(0, BLOCK_S); m = s < n_samples
            off = dim_idx * n_samples
            q0 = tl.load(H_ptr + 0*deg_stride + off + s, mask=m, other=0.0)
            q1 = tl.load(H_ptr + 1*deg_stride + off + s, mask=m, other=0.0)
            q2 = tl.load(H_ptr + 2*deg_stride + off + s, mask=m, other=0.0)
            q3 = tl.load(H_ptr + 3*deg_stride + off + s, mask=m, other=0.0)
            a00+=tl.sum(q0*q0); a01+=tl.sum(q0*q1); a02+=tl.sum(q0*q2); a03+=tl.sum(q0*q3)
            a11+=tl.sum(q1*q1); a12+=tl.sum(q1*q2); a13+=tl.sum(q1*q3)
            a22+=tl.sum(q2*q2); a23+=tl.sum(q2*q3); a33+=tl.sum(q3*q3)
        
        # Cholesky 4×4: A = L·Lᵀ
        L00=tl.sqrt(a00)
        L10=a01/L00; L11=tl.sqrt(a11-L10*L10)
        L20=a02/L00; L21=(a12-L20*L10)/L11; L22=tl.sqrt(a22-L20*L20-L21*L21)
        L30=a03/L00; L31=(a13-L30*L10)/L11; L32=(a23-L30*L20-L31*L21)/L22
        L33=tl.sqrt(a33-L30*L30-L31*L31-L32*L32)
        
        for j_out in range(n_output):
            # b = Φᵀ·Y (usando Y como proxy del residual)
            b0=0.0; b1=0.0; b2=0.0; b3=0.0
            for s_start in range(0, n_samples, BLOCK_S):
                s = s_start + tl.arange(0, BLOCK_S); m = s < n_samples
                off = dim_idx * n_samples
                q0 = tl.load(H_ptr + 0*deg_stride + off + s, mask=m, other=0.0)
                q1 = tl.load(H_ptr + 1*deg_stride + off + s, mask=m, other=0.0)
                q2 = tl.load(H_ptr + 2*deg_stride + off + s, mask=m, other=0.0)
                q3 = tl.load(H_ptr + 3*deg_stride + off + s, mask=m, other=0.0)
                r = tl.load(Y_ptr + j_out*n_samples + s, mask=m, other=0.0)
                b0+=tl.sum(q0*r); b1+=tl.sum(q1*r); b2+=tl.sum(q2*r); b3+=tl.sum(q3*r)
            
            y0=b0/L00; y1=(b1-L10*y0)/L11; y2=(b2-L20*y0-L21*y1)/L22
            y3=(b3-L30*y0-L31*y1-L32*y2)/L33
            c3=y3/L33; c2=(y2-L32*c3)/L22; c1=(y1-L21*c2-L31*c3)/L11
            c0=(y0-L10*c1-L20*c2-L30*c3)/L00
            
            base = j_out * n_input * d1 + dim_idx * d1
            tl.store(C_ptr + base + 0, c0); tl.store(C_ptr + base + 1, c1)
            tl.store(C_ptr + base + 2, c2); tl.store(C_ptr + base + 3, c3)


# ═══════════════════════════════════════════════════════════════════════════════
# NumPy fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _numpy_hermite(X, d):
    H = np.zeros((X.shape[0], X.shape[1], d+1), dtype=X.dtype)
    H[:,:,0] = 1.0
    if d >= 1: H[:,:,1] = X
    for k in range(2, d+1): H[:,:,k] = X*H[:,:,k-1] - (k-1)*H[:,:,k-2]
    return H

def _numpy_update(H, C, Y, lam):
    """BESD NumPy corregido: einsum ->bj para predicción por output."""
    ns, ni, d1 = H.shape; no = Y.shape[1]
    for _ in range(2):
        pred_full = np.einsum('jik,bik->bj', C, H)  # (n_samples, n_output) CORREGIDO
        for jo in range(no):
            pred = pred_full[:, jo].copy()
            for i in range(ni):
                old = H[:,i,:] @ C[jo,i,:]; pred -= old
                r = Y[:,jo] - pred
                Phi = H[:,i,:]; A = Phi.T@Phi + lam*np.eye(d1); b = Phi.T@(r+old)
                try: C[jo,i,:] = np.linalg.solve(A,b)
                except: C[jo,i,:] = np.linalg.lstsq(A,b,rcond=None)[0]
                pred += Phi @ C[jo,i,:]
            pred_full[:, jo] = pred
    return C


# ═══════════════════════════════════════════════════════════════════════════════
# API unificada
# ═══════════════════════════════════════════════════════════════════════════════

def hermite_batch(X, max_degree, use_gpu=True):
    if use_gpu and HAS_TRITON and X.shape[1] >= 32:
        ns, ni = X.shape; d = max_degree
        Xg = torch.from_numpy(X.astype(np.float32)).cuda().T.contiguous()
        Hg = torch.empty(d+1, ni, ns, dtype=torch.float32, device='cuda')
        B = 256
        hermite_batch_kernel[(ni, triton.cdiv(ns, B))](Xg, Hg, ns, ni, d, Xg.stride(0), BLOCK_S=B)
        return Hg.permute(2,1,0).cpu().numpy().astype(np.float64)
    return _numpy_hermite(X, max_degree)


def update_main_coefficients(H, C, Y, lam, use_gpu=True):
    if use_gpu and HAS_TRITON and H.shape[1] >= 32:
        ns, ni, d1 = H.shape; no = C.shape[0]
        Hg = torch.from_numpy(H.astype(np.float32)).cuda().permute(2,1,0).contiguous()
        Cg = torch.from_numpy(C.astype(np.float32)).cuda().contiguous()
        Yg = torch.from_numpy(Y.astype(np.float32)).cuda().T.contiguous()
        B = 128
        update_4x4_kernel[(ni,)](Hg, Cg, Yg, lam, ns, ni, no, d1, BLOCK_S=B)
        torch.cuda.synchronize()
        return Cg.cpu().numpy().astype(np.float64)
    return _numpy_update(H, C, Y, lam)


def predict_main(C, H, use_gpu=True):
    if use_gpu and HAS_TRITON and H.shape[1] >= 64:
        Cg = torch.from_numpy(C.astype(np.float32)).cuda()
        Hg = torch.from_numpy(H.astype(np.float32)).cuda()
        out = torch.einsum('jik,bik->bj', Cg, Hg)
        return out.cpu().numpy().astype(np.float64)
    return np.einsum('jik,bik->bj', C, H)


def is_gpu_available():
    return HAS_TRITON


def get_backend_name():
    if HAS_TRITON:
        return f"triton (GPU: {torch.cuda.get_device_name(0)})"
    return "numpy (CPU)"


# ── Benchmark ─────────────────────────────────────────────────────────────

def benchmark(n_input=2048, n_samples=400, n_output=3, n_iter=3):
    """Comparar NumPy vs Triton en operaciones clave."""
    import time
    np.random.seed(42)
    X = np.random.randn(n_samples, n_input).astype(np.float64) * 0.5
    Y = np.random.randn(n_samples, n_output).astype(np.float64) * 0.5
    lam = 0.05
    
    print(f"  Benchmark: n_input={n_input}, n_samples={n_samples}, n_output={n_output}")
    print(f"  Backend: {get_backend_name()}")
    
    # Hermite
    t0 = time.perf_counter()
    H_np = _numpy_hermite(X, 3)
    t_np_h = time.perf_counter() - t0
    
    if HAS_TRITON:
        t0 = time.perf_counter()
        H_tr = hermite_batch(X, 3, use_gpu=True)
        t_tr_h = time.perf_counter() - t0
        print(f"  Hermite:  NumPy={t_np_h:.4f}s  Triton={t_tr_h:.4f}s  speedup={t_np_h/max(t_tr_h,1e-6):.1f}×")
    else:
        print(f"  Hermite:  NumPy={t_np_h:.4f}s  (Triton no disponible)")
    
    # Update main coefs
    C = np.zeros((n_output, n_input, 4))
    t0 = time.perf_counter()
    for _ in range(n_iter):
        C = _numpy_update(H_np, C, Y, lam)
    t_np_u = time.perf_counter() - t0
    
    if HAS_TRITON and n_input >= 128:
        C2 = np.zeros((n_output, n_input, 4))
        t0 = time.perf_counter()
        for _ in range(n_iter):
            C2 = update_main_coefficients(H_np, C2, Y, lam, use_gpu=True)
        t_tr_u = time.perf_counter() - t0
        print(f"  BESD upd: NumPy={t_np_u:.4f}s  Triton={t_tr_u:.4f}s  speedup={t_np_u/max(t_tr_u,1e-6):.1f}×")
        print(f"  Coef match: {np.allclose(C, C2, atol=1e-3)}")
    else:
        print(f"  BESD upd: NumPy={t_np_u:.4f}s  (Triton no disponible)")
    
    return {"hermite_np": t_np_h, "update_np": t_np_u}


if __name__ == "__main__":
    print("=" * 60)
    print("  NOVA TRITON KERNEL — BENCHMARK GPU")
    print("=" * 60)
    for dims in [512, 1024, 2048, 4096]:
        benchmark(n_input=dims)
        print()
    print("  ✓ Benchmark completo")
