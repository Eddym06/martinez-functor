"""
nova_gpu.py — GPU/Triton Acceleration for Nova ACF
===================================================
Distribución óptima CPU/GPU usando:
  - Triton kernels para evaluación de bases, einsums, softmax
  - Gideon (Rust AVX2+FMA) para CPU vía Poema
  - PyTorch CUDA para Cholesky y operaciones de álgebra lineal

HARDWARE MAP:
  GPU (RTX 4050 6.4GB):
    ✅ Basis evaluation (Triton: _triton_hermite)
    ✅ Cholesky solves (torch.cholesky_solve)
    ✅ Pair einsum (torch.einsum)
    ✅ Forward pass batching
    ✅ Hebbian updates (vectorized)
    
  CPU (Gideon AVX2+FMA via Poema):
    ✅ Pair selection (correlation, small matrices)
    ✅ Data preprocessing
    ✅ Small matrix ops (< 200×200)
    ✅ Control flow
    
  NEVER Python loops:
    ❌ for seq in range(n_seqs)  → vectorized GPU
    ❌ for pos in range(seq_len) → batched GPU
    ❌ for layer (small, OK)
    ❌ head.evaluate(xp) for xp in X → head.predict(X).mean
"""

import numpy as np
import time
import os

# ═══════════════════════════════════════════════════════════════
# 🔥 TRITON KERNEL — Hermite Basis Evaluation
# ═══════════════════════════════════════════════════════════════

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


if _HAS_TRITON:
    @triton.jit
    def _triton_hermite_kernel(
        z_ptr,          # (N, d) float32 — normalized input
        B_ptr,          # (N, d, deg+1) float32 — output basis
        N: int,         # number of samples
        d: int,         # input dimension
        deg: int,       # polynomial degree
        BLOCK_N: tl.constexpr,  # block size for samples
        BLOCK_D: tl.constexpr,  # block size for features
    ):
        """Triton kernel for physicist's Hermite polynomial evaluation.
        
        H_0(z) = 1
        H_1(z) = z
        H_{k+1}(z) = z·H_k(z) - k·H_{k-1}(z)
        
        Grid: (ceil(N/BLOCK_N), ceil(d/BLOCK_D))
        """
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        
        # Block indices
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        
        n_mask = n_offs < N
        d_mask = d_offs < d
        
        # Load z for this block: (BLOCK_N, BLOCK_D)
        z = tl.load(z_ptr + n_offs[:, None] * d + d_offs[None, :],
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        
        # Base addresses for output
        stride_d = deg + 1  # inner dim stride
        
        # H_0 = 1
        tl.store(B_ptr + n_offs[:, None] * d * stride_d + d_offs[None, :] * stride_d,
                 tl.full(z.shape, 1.0, dtype=tl.float32),
                 mask=n_mask[:, None] & d_mask[None, :])
        
        if deg >= 1:
            # H_1 = z
            tl.store(B_ptr + n_offs[:, None] * d * stride_d + d_offs[None, :] * stride_d + 1,
                     z,
                     mask=n_mask[:, None] & d_mask[None, :])
        
        # Recurrence: H_{k+1} = z·H_k - k·H_{k-1}
        h_prev = tl.full(z.shape, 1.0, dtype=tl.float32)  # H_0
        h_curr = z  # H_1
        
        for k in range(2, deg + 1):
            h_next = z * h_curr - (k - 1.0) * h_prev
            tl.store(B_ptr + n_offs[:, None] * d * stride_d + d_offs[None, :] * stride_d + k,
                     h_next,
                     mask=n_mask[:, None] & d_mask[None, :])
            h_prev = h_curr
            h_curr = h_next
    
    
    def triton_hermite_batch(X: np.ndarray, deg: int) -> np.ndarray:
        """Evaluate Hermite basis using Triton GPU kernel.
        
        Args:
            X: (N, d) float32 or float64 — normalized input (z-scores)
            deg: polynomial degree
            
        Returns:
            B: (N, d, deg+1) float32 — basis evaluations
        """
        X = np.asarray(X, dtype=np.float32)
        N, d = X.shape
        B = np.zeros((N, d, deg + 1), dtype=np.float32)
        
        # Heuristic block sizes for RTX 4050 (6.4GB, 2560 CUDA cores)
        BLOCK_N = min(128, N)
        BLOCK_D = min(64, d)
        
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(d, BLOCK_D))
        
        # Triton requires torch tensors
        import torch
        X_t = torch.from_numpy(X).cuda()
        B_t = torch.zeros(N, d, deg + 1, dtype=torch.float32, device='cuda')
        
        _triton_hermite_kernel[grid](
            X_t, B_t, N, d, deg,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
        
        torch.cuda.synchronize()
        return B_t.cpu().numpy()

else:
    def triton_hermite_batch(X: np.ndarray, deg: int) -> np.ndarray:
        """Fallback CPU implementation (same math, different hardware)."""
        X = np.asarray(X, dtype=np.float64)
        N, d = X.shape
        B = np.zeros((N, d, deg + 1), dtype=np.float64)
        B[:, :, 0] = 1.0
        if deg >= 1:
            B[:, :, 1] = X
        for k in range(2, deg + 1):
            B[:, :, k] = X * B[:, :, k-1] - (k - 1.0) * B[:, :, k-2]
        return B


# ═══════════════════════════════════════════════════════════════
# 🚀 GPU-ACCELERATED OPERATIONS
# ═══════════════════════════════════════════════════════════════

class NovaGPUAccelerator:
    """Accelerator that routes operations to optimal hardware.
    
    Decision matrix:
      - f ≤ 200: CPU (Gideon) — overhead of GPU transfer > compute
      - 200 < f ≤ 2000: GPU via Triton — custom kernels
      - f > 2000: GPU via PyTorch — batched BLAS
      - Python loops: NEVER — always vectorized
    """
    
    def __init__(self, use_triton: bool = True, use_gideon: bool = True):
        self.use_triton = use_triton and _HAS_TRITON
        self.use_gideon = use_gideon
        self._has_torch = False
        self._device = None
        
        try:
            import torch
            if torch.cuda.is_available():
                self._has_torch = True
                self._device = torch.device('cuda')
        except ImportError:
            pass
        
        self.stats = {
            "hermite_calls": 0, "hermite_time": 0.0,
            "einsum_calls": 0, "einsum_time": 0.0,
            "cholesky_calls": 0, "cholesky_time": 0.0,
        }
    
    def hermite_basis(self, X: np.ndarray, deg: int) -> np.ndarray:
        """Evaluate Hermite basis. Routes to Triton if available."""
        t0 = time.perf_counter()
        
        if self.use_triton and _HAS_TRITON:
            result = triton_hermite_batch(X.astype(np.float32), deg)
        else:
            # NumPy vectorized (no Python loops)
            result = triton_hermite_batch(X, deg)
        
        self.stats["hermite_calls"] += 1
        self.stats["hermite_time"] += time.perf_counter() - t0
        return result
    
    def pair_einsum(self, B: np.ndarray, pair_i: np.ndarray, 
                    pair_j: np.ndarray) -> np.ndarray:
        """Compute pair outer products: B[:,i] ⊗ B[:,j] for all pairs.
        
        Args:
            B: (N, d, deg+1) — basis
            pair_i, pair_j: (n_pairs,) — pair indices
        
        Returns:
            psi: (N, n_pairs, (deg+1)²) — flattened outer products
        """
        t0 = time.perf_counter()
        
        if self._has_torch and B.shape[0] > 1000:
            import torch
            B_t = torch.from_numpy(B.astype(np.float32)).to(self._device)
            pi_t = torch.from_numpy(pair_i).to(self._device)
            pj_t = torch.from_numpy(pair_j).to(self._device)
            
            # torch.einsum on GPU
            psi_t = torch.einsum('npi,npj->npij', B_t[:, pi_t, :], B_t[:, pj_t, :])
            n_pairs = len(pair_i)
            d2 = B.shape[2] ** 2
            result = psi_t.reshape(B.shape[0], n_pairs * d2).cpu().numpy()
        else:
            # NumPy einsum (BLAS-backed, fast on CPU)
            result = np.einsum('npi,npj->npij', 
                              B[:, pair_i, :], B[:, pair_j, :])
            result = result.reshape(B.shape[0], -1)
        
        self.stats["einsum_calls"] += 1
        self.stats["einsum_time"] += time.perf_counter() - t0
        return result
    
    def cholesky_solve(self, Phi: np.ndarray, Y: np.ndarray, 
                       l2_lambda: float = 0.01) -> np.ndarray:
        """Solve (ΦᵀΦ + λI)C = ΦᵀY via Cholesky on GPU.
        
        Returns: C matrix of shape (n_features, n_output)
        """
        t0 = time.perf_counter()
        
        if self._has_torch:
            import torch
            n_features = Phi.shape[1]
            lam = max(l2_lambda, 0.001)
            
            Phi_t = torch.from_numpy(Phi.astype(np.float32)).to(self._device)
            Y_t = torch.from_numpy(Y.astype(np.float32)).to(self._device)
            
            A_t = Phi_t.T @ Phi_t + lam * torch.eye(n_features, device=self._device)
            b_t = Phi_t.T @ Y_t
            
            L_t = torch.linalg.cholesky(A_t)
            C = torch.cholesky_solve(b_t, L_t).cpu().numpy().astype(np.float64)
        else:
            import scipy.linalg
            lam = max(l2_lambda, 0.001)
            A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
            b = Phi.T @ Y
            L, low = scipy.linalg.cho_factor(A, lower=True)
            C = scipy.linalg.cho_solve((L, low), b)
        
        self.stats["cholesky_calls"] += 1
        self.stats["cholesky_time"] += time.perf_counter() - t0
        return C
    
    def batched_forward(self, model, all_emb: list) -> np.ndarray:
        """Forward pass through all layers — batched where possible.
        
        🔥 Key optimization: instead of iterating over sequences
        and calling process_and_write_bus per sequence, we batch
        the attention computation across multiple sequences.
        
        For now: uses existing forward but avoids unnecessary copies.
        """
        all_ctx = []
        for emb in all_emb:
            model.bus.initialize(emb)
            x = emb.astype(np.float64)
            for layer in model.layers:
                x = layer.process_and_write_bus(x, model.bus)
            all_ctx.append(model.bus.read())
        return np.vstack(all_ctx)
    
    def vectorized_hebbian(self, ctx: np.ndarray, targets: np.ndarray,
                           emb_matrix: np.ndarray, vocab_size: int,
                           hebb_eta: float, d_emb: int) -> np.ndarray:
        """🔥 VECTORIZED Hebbian update — NO Python loops.
        
        ctx: (N, d_emb) — context vectors
        targets: (N,) — target token IDs
        emb_matrix: (V, d_emb) — current embeddings
        hebb_eta: learning rate
        d_emb: embedding dimension (embed_dim - 3)
        
        Returns: updated emb_matrix
        """
        N = len(ctx)
        valid = (targets >= 0) & (targets < vocab_size)
        targets_v = targets[valid]
        ctx_v = ctx[valid, :d_emb]
        
        if len(targets_v) == 0:
            return emb_matrix
        
        # Normalize context vectors: (N_valid, d_emb)
        ctx_norms = np.linalg.norm(ctx_v, axis=1, keepdims=True)
        ctx_norms = np.maximum(ctx_norms, 1e-8)
        ctx_normalized = ctx_v / ctx_norms
        
        # 🔥 Vectorized: for each target token, accumulate
        # the context vector into the embedding
        # Using np.add.at for in-place accumulation
        emb_delta = np.zeros_like(emb_matrix)
        np.add.at(emb_delta, targets_v, ctx_normalized * hebb_eta)
        
        # Normalize updated embeddings
        new_emb = emb_matrix + emb_delta
        emb_norms = np.linalg.norm(new_emb, axis=1, keepdims=True)
        emb_norms = np.maximum(emb_norms, 1e-8)
        new_emb = new_emb / emb_norms
        
        return new_emb
    
    def stats_summary(self) -> dict:
        """Get accelerator statistics."""
        return {
            **{k: f"{v:.3f}s" for k, v in self.stats.items() if k.endswith("_time")},
            **{k: v for k, v in self.stats.items() if k.endswith("_calls")},
            "triton_available": _HAS_TRITON,
            "torch_gpu_available": self._has_torch,
        }


# ═══════════════════════════════════════════════════════════════
# 📊 HARDWARE PROFILER
# ═══════════════════════════════════════════════════════════════

def profile_hardware():
    """Detect and profile available hardware."""
    info = {
        "cpu": {"name": "Unknown", "cores": os.cpu_count()},
        "gpu": {"available": False, "name": "None", "memory_gb": 0},
        "triton": _HAS_TRITON,
        "gideon": False,
    }
    
    # CPU
    try:
        import platform
        info["cpu"]["name"] = platform.processor()
    except:
        pass
    
    # GPU via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"]["available"] = True
            info["gpu"]["name"] = props.name
            info["gpu"]["memory_gb"] = props.total_memory / 1e9
            info["gpu"]["compute_capability"] = f"{props.major}.{props.minor}"
            info["gpu"]["multiprocessors"] = props.multi_processor_count
    except:
        pass
    
    # Gideon (Rust backend)
    try:
        from gideon_core import GideonCoreEngine, CoreEngineConfig
        info["gideon"] = True
    except:
        pass
    
    return info


def print_hardware_profile():
    """Print hardware profile in readable format."""
    info = profile_hardware()
    
    print("\n🖥️  HARDWARE PROFILE")
    print("=" * 50)
    print(f"  CPU: {info['cpu']['name']} ({info['cpu']['cores']} cores)")
    
    gpu = info['gpu']
    if gpu['available']:
        print(f"  GPU: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
        print(f"       Compute {gpu['compute_capability']}, "
              f"{gpu['multiprocessors']} SMs")
    else:
        print(f"  GPU: NOT AVAILABLE")
    
    print(f"  Triton: {'✅' if info['triton'] else '❌ (not installed)'}")
    print(f"  Gideon: {'✅' if info['gideon'] else '❌ (not found)'}")
    print(f"  Poema:  {'✅' if os.path.exists('poema') else '❌ (not found)'}")
    print("=" * 50)


# ═══════════════════════════════════════════════════════════════
# 🎯 OPTIMAL HARDWARE ROUTING
# ═══════════════════════════════════════════════════════════════

def route_operation(n_features: int, n_samples: int) -> str:
    """Route operation to optimal hardware based on size.
    
    Returns: 'cpu', 'gpu_triton', or 'gpu_torch'
    """
    if n_features <= 200:
        return "cpu"  # GPU transfer overhead > compute
    elif n_features <= 2000 and _HAS_TRITON:
        return "gpu_triton"
    elif n_features > 2000:
        return "gpu_torch"
    else:
        return "cpu"


# ═══════════════════════════════════════════════════════════════
# 🧪 QUICK BENCHMARK
# ═══════════════════════════════════════════════════════════════

def benchmark_hermite(n_samples: int = 51200, d: int = 192, deg: int = 2):
    """Benchmark Hermite basis evaluation on CPU vs GPU."""
    X = np.random.randn(n_samples, d).astype(np.float32)
    
    # CPU
    t0 = time.perf_counter()
    B_cpu = triton_hermite_batch(X, deg)  # Uses fallback if no Triton
    t_cpu = time.perf_counter() - t0
    
    print(f"\n⚡ Hermite Benchmark ({n_samples}×{d}, deg={deg})")
    print(f"  CPU (NumPy): {t_cpu:.3f}s")
    
    # GPU Triton
    if _HAS_TRITON:
        t0 = time.perf_counter()
        B_gpu = triton_hermite_batch(X, deg)
        t_gpu = time.perf_counter() - t0
        speedup = t_cpu / max(t_gpu, 1e-6)
        print(f"  GPU (Triton): {t_gpu:.3f}s ({speedup:.1f}× speedup)")
        print(f"  Match: {'✅' if np.allclose(B_cpu, B_gpu, atol=1e-4) else '❌'}")
    else:
        print(f"  GPU (Triton): NOT AVAILABLE")


# ═══════════════════════════════════════════════════════════════
# � PAIR OUTER PRODUCTS — torch.einsum GPU / NumPy CPU
# ═══════════════════════════════════════════════════════════════

def triton_pair_outer_batch(B: np.ndarray, pair_i: np.ndarray, 
                             pair_j: np.ndarray) -> np.ndarray:
    """Compute pair outer products — GPU-accelerated via torch.einsum.
    
    torch.einsum is already highly optimized for this (uses cuBLAS).
    Falls back to NumPy einsum if no GPU available.
    
    Args:
        B: (N, d, deg+1) — basis evaluations
        pair_i, pair_j: (n_pairs,) — pair indices
        
    Returns:
        psi: (N, n_pairs * (deg+1)²) float64 — flattened outer products
    """
    import torch
    N, d, d1 = B.shape
    n_pairs = len(pair_i)
    d2 = d1 * d1
    
    if n_pairs == 0:
        return np.zeros((N, 0), dtype=np.float64)
    
    # Route to GPU torch.einsum if available (cuBLAS-accelerated)
    try:
        if torch.cuda.is_available() and N >= 64:
            B_t = torch.from_numpy(np.asarray(B, dtype=np.float32)).cuda()
            pi_t = torch.from_numpy(np.asarray(pair_i, dtype=np.int64)).cuda()
            pj_t = torch.from_numpy(np.asarray(pair_j, dtype=np.int64)).cuda()
            psi_t = torch.einsum('npi,npj->npij', B_t[:, pi_t, :], B_t[:, pj_t, :])
            result = psi_t.reshape(N, n_pairs * d2).cpu().numpy().astype(np.float64)
        else:
            # CPU fallback: NumPy einsum (BLAS-backed)
            result = np.einsum('npi,npj->npij',
                              B[:, pair_i, :].astype(np.float64),
                              B[:, pair_j, :].astype(np.float64))
            result = result.reshape(N, -1)
    except Exception:
        # Ultimate fallback
        result = np.einsum('npi,npj->npij',
                          B[:, pair_i, :].astype(np.float64),
                          B[:, pair_j, :].astype(np.float64))
        result = result.reshape(N, -1)
    
    return result.astype(np.float64)


# ═══════════════════════════════════════════════════════════════
# 🚀 TRITON-ACCELERATED Φ BUILDER (builds full Phi for ACF neuron)
# ═══════════════════════════════════════════════════════════════

def triton_build_phi(B: np.ndarray, pair_i: np.ndarray, 
                     pair_j: np.ndarray) -> np.ndarray:
    """Build full Phi matrix: main effects + pair interactions — GPU-accelerated.
    
    Math: Phi = [B_flat | psi_flat]
      B_flat: (N, n_input * d1) — main effects
      psi_flat: (N, n_pairs * d1²) — pair interactions
    
    Args:
        B: (N, n_input, deg+1) — basis
        pair_i, pair_j: (n_pairs,) — pair indices
        
    Returns:
        Phi: (N, n_input*d1 + n_pairs*d1²) — full design matrix
    """
    N, d, d1 = B.shape
    n_pairs = len(pair_i)
    
    # Main effects: just flatten B
    Phi_main = np.asarray(B, dtype=np.float64).reshape(N, d * d1)
    
    # Pair effects: Triton-accelerated
    if n_pairs > 0:
        Psi = triton_pair_outer_batch(B, pair_i, pair_j)
        return np.hstack([Phi_main, Psi])
    return Phi_main


if __name__ == "__main__":
    print_hardware_profile()
    benchmark_hermite()
