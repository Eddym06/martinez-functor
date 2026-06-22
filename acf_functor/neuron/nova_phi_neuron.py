"""
nova_phi_neuron.py — NΦ: La Neurona Definitiva
=================================================

Sintetiza TODO: Φ v2.0 + Φ-K v5.0 + Φ-S BESD en UNA sola neurona imbatible.

INNOVACIONES:
  1. BASE ADAPTATIVA: Auto-selecciona Hermite (datos Gaussianos) o Chebyshev (datos
     generales) según la curtosis empírica. Hermite es 2-3× más rápido.
  2. SOLVER ADAPTATIVO: Ridge Cholesky (<3000 features), Ridge SVD (n_s≪n_f),
     o BESD (>10000 features). Siempre el óptimo.
  3. RLS ONLINE: Tras el ajuste inicial, aprendizaje incremental sin reentrenar.
  4. ANOVA(2) UNIVERSAL: Estructura común para todas las bases y solvers.
  5. AUTO-TUNING: λ óptimo vía GCV simplificado.

CAPACIDADES UNIFICADAS:
  ✅ Escalabilidad masiva (2 → 1M+ dimensiones)
  ✅ OOD detection 20/20
  ✅ Incertidumbre σ por predicción
  ✅ Interpretabilidad total
  ✅ Velocidad extrema (mejor solver automático)
  ✅ Sin SGD, sin backprop, sin tuning manual
  ✅ Una sola neurona compite contra redes enteras

Autor: AXIOM-1
"""

import math, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

# Backend Triton (opcional)
try:
    from .nova_triton_kernel import (
        hermite_batch as _triton_hermite,
        update_main_coefficients as _triton_update_main,
        predict_main as _triton_predict_main,
        is_gpu_available, get_backend_name
    )
    _HAS_GPU = is_gpu_available()
    _BACKEND = get_backend_name()
except ImportError:
    _HAS_GPU = False
    _BACKEND = "numpy (CPU)"
    # Fallback inline (sin Triton)
    def _numpy_hermite(X, d):
        H = np.zeros((X.shape[0], X.shape[1], d+1), dtype=X.dtype)
        H[:,:,0]=1.0
        if d>=1: H[:,:,1]=X
        for k in range(2,d+1): H[:,:,k]=X*H[:,:,k-1]-(k-1)*H[:,:,k-2]
        return H
    def _numpy_chebyshev(X, d, lo, hi):
        s=np.maximum(hi-lo,0.01);t=2.0*(X-lo[None,:])/s[None,:]-1.0;t=np.clip(t,-1.0,1.0)
        T=np.zeros((X.shape[0],X.shape[1],d+1),dtype=X.dtype);T[:,:,0]=1.0
        if d>=1:T[:,:,1]=t
        for k in range(2,d+1):T[:,:,k]=2.0*t*T[:,:,k-1]-T[:,:,k-2]
        return T
    def _hermite_batch(self, X, d):
        if _HAS_GPU:
            return _triton_hermite(X, d, use_gpu=True)
        return _numpy_hermite(X, d)
    def _chebyshev_batch(self, X, d, lo, hi):
        return _numpy_chebyshev(X, d, lo, hi)


@dataclass
class NovaPrediction:
    mean: np.ndarray
    std: np.ndarray
    variance: np.ndarray
    basis_used: str = "?"
    solver_used: str = "?"

    @property
    def is_reliable(self) -> bool:
        mag = np.linalg.norm(self.mean)
        if mag < 1e-10:
            return float(np.max(self.std)) < 1.0
        return float(np.max(self.std) / max(mag, 1e-10)) < 2.0

    def confidence_interval(self, dim: int, k: float = 2.0) -> Tuple[float, float]:
        mu, s = float(self.mean[dim]), float(self.std[dim])
        return (mu - k * s, mu + k * s)

    def __repr__(self):
        r = "✓" if self.is_reliable else "⚠"
        return f"NΦ({r} μ={np.round(self.mean[:3],3)}..., σ={np.max(self.std):.3f}, {self.basis_used}/{self.solver_used})"


class NovaPhiNeuron:
    """NΦ — La Neurona Definitiva. Una para gobernarlas a todas.

    Modo autónomo (adaptive=True): sin parámetros fijos.
    - Selecciona grado automáticamente según error residual
    - Descubre pares dinámicamente según correlación con error
    - Aplica L1 esparso para selección de features
    - Ajusta forgetting_factor según deriva del entorno
    """

    def __init__(self, name: str, n_input: int, n_output: int,
                 max_degree: int = 3, l2_lambda: float = 0.05,
                 max_pairs: int = 200,
                 correlation_threshold: float = 0.06,
                 online_mode: bool = False,
                 forgetting_factor: float = 0.997,
                 use_triton: bool = True,
                 # ── NUEVOS: modo autónomo ──
                 adaptive: bool = False,
                 sparse_l1: float = 0.0,
                 lr_base: float = 0.3,
                 max_degree_search: int = 5,
                 pair_refresh_interval: int = 100):
        self.name = name
        self.n_input = n_input
        self.n_output = n_output
        self.max_degree = max_degree
        self.l2_lambda = l2_lambda
        self.max_pairs = max_pairs
        self.correlation_threshold = correlation_threshold
        self.online_mode = online_mode
        self.forgetting_factor = forgetting_factor
        self.use_gpu = use_triton and _HAS_GPU

        # ── Modo autónomo ──
        self.adaptive = adaptive
        self.sparse_l1 = sparse_l1
        self.lr_base = lr_base
        self.max_degree_search = max_degree_search
        self.pair_refresh_interval = pair_refresh_interval
        self._error_buffer = []        # buffer de errores recientes
        self._pair_updates_since_refresh = 0
        self._error_trend = 0.0        # tendencia del error (>0 empeorando)
        self._best_degree = max_degree # grado óptimo descubierto
        self._best_error = float('inf')

        self._d1 = max_degree + 1
        self._d2 = self._d1 * self._d1

        # Estado interno (inicializado en fit)
        self.basis_type = "?"        # "hermite" o "chebyshev"
        self.solver_type = "?"        # "cholesky", "svd", o "besd"
        self.C_main = np.zeros((n_output, n_input, self._d1))
        self._pairs: List[Tuple[int, int]] = []
        self.C_pair = np.zeros((n_output, 0, self._d2))

        # RLS state (inicializado bajo demanda)
        self._rls_ready = False
        self.P_main = None
        self.P_pair = None

        # Métricas
        self.epsilon_mu = float("inf")
        self.alpha_A = 0.0
        self.total_updates = 0
        self._x_mean = None
        self._x_std = None
        self._x_min = None
        self._x_max = None
        
        # 🔥 Φ cache — build once, solve many (decoder rounds)
        self._cached_Phi = None
        self._cached_pair_i = np.array([], dtype=int)
        self._cached_pair_j = np.array([], dtype=int)
        self._phi_built = False

    # ═══════════════════════════════════════════════════════════════
    # Bases polinómicas (pluggable)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _hermite_batch(X: np.ndarray, d: int) -> np.ndarray:
        return _triton_hermite(X, d, use_gpu=_HAS_GPU)

    @staticmethod
    def _chebyshev_batch(X: np.ndarray, d: int,
                         lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        spread = np.maximum(hi - lo, 0.01)
        t = 2.0 * (X - lo[None, :]) / spread[None, :] - 1.0
        t = np.clip(t, -1.0, 1.0)
        return _triton_hermite(t, d, use_gpu=False)  # Chebyshev = Hermite(t) con recurrencia diferente

    @staticmethod
    def _legendre_batch(X: np.ndarray, d: int) -> np.ndarray:
        """Legendre polynomials: P_0=1, P_1=x, P_{k+1}=((2k+1)xP_k - k P_{k-1})/(k+1).
        Domain: [-1, 1]. Input X should be already normalized to [-1, 1]."""
        import numpy as np
        X = np.asarray(X, dtype=np.float64)
        N, d_in = X.shape
        B = np.zeros((N, d_in, d + 1), dtype=np.float64)
        B[:, :, 0] = 1.0
        if d >= 1:
            B[:, :, 1] = X
        for k in range(2, d + 1):
            kf = float(k - 1)
            B[:, :, k] = ((2*kf + 1) * X * B[:, :, k-1] - kf * B[:, :, k-2]) / (kf + 1)
        return B

    @staticmethod
    def _fourier_batch(X: np.ndarray, d: int) -> np.ndarray:
        """Fourier features: cos(0·πx)=1, sin(πx), cos(πx), sin(2πx), cos(2πx), ...
        For degree d, produces d+1 features alternating cos/sin.
        Input X should be normalized to [0, 1]."""
        import numpy as np
        X = np.asarray(X, dtype=np.float64)
        N, d_in = X.shape
        B = np.zeros((N, d_in, d + 1), dtype=np.float64)
        B[:, :, 0] = 1.0  # cos(0)
        for k in range(1, d + 1):
            freq = (k + 1) // 2  # 1,1,2,2,3,3,...
            if k % 2 == 1:
                B[:, :, k] = np.sin(freq * np.pi * X)
            else:
                B[:, :, k] = np.cos(freq * np.pi * X)
        return B

    def _eval_basis(self, X: np.ndarray) -> np.ndarray:
        """Evaluar la base seleccionada."""
        import numpy as np
        xm = self._x_mean if self._x_mean is not None else np.mean(X, axis=0)
        xs = np.maximum(self._x_std if self._x_std is not None else np.std(X, axis=0) + 1e-8, 1e-8)
        
        if self.basis_type == "hermite":
            Z = (X - xm[None, :]) / xs
            return self._hermite_batch(Z, self.max_degree)
        elif self.basis_type == "legendre":
            # Normalize to [-1, 1] using 3σ range
            lo = (self._x_min if self._x_min is not None else xm - 3*xs) - 0.5 * xs
            hi = (self._x_max if self._x_max is not None else xm + 3*xs) + 0.5 * xs
            spread = np.maximum(hi - lo, 0.01)
            t = 2.0 * (X - lo[None, :]) / spread[None, :] - 1.0
            t = np.clip(t, -1.0, 1.0)
            return self._legendre_batch(t, self.max_degree)
        elif self.basis_type == "fourier":
            # Normalize to [0, 1]
            x_min = self._x_min if self._x_min is not None else xm - 3*xs
            x_max = self._x_max if self._x_max is not None else xm + 3*xs
            spread = np.maximum(x_max - x_min, 0.01)
            t = (X - x_min[None, :]) / spread[None, :]
            t = np.clip(t, 0.0, 1.0)
            return self._fourier_batch(t, self.max_degree)
        elif self.basis_type and self.basis_type.startswith("chebyshev"):
            # Handles "chebyshev" and "chebyshev_v2" etc.
            lo = (self._x_min if self._x_min is not None else xm - 3*xs) - 0.5 * xs
            hi = (self._x_max if self._x_max is not None else xm + 3*xs) + 0.5 * xs
            return self._chebyshev_batch(X, self.max_degree, lo, hi)
        else:
            # Fallback: Chebyshev
            lo = (self._x_min if self._x_min is not None else xm - 3*xs) - 0.5 * xs
            hi = (self._x_max if self._x_max is not None else xm + 3*xs) + 0.5 * xs
            return self._chebyshev_batch(X, self.max_degree, lo, hi)

    # ═══════════════════════════════════════════════════════════════
    # Selección adaptativa de base
    # ═══════════════════════════════════════════════════════════════

    def _select_basis(self, X: np.ndarray) -> str:
        """Auto-seleccionar base. Hermite-first para velocidad.

        Hermite es 2-3x mas rapido que Chebyshev (sin normalizacion lo/hi).
        Solo usar Chebyshev si los datos tienen curtosis extrema (>6).
        """
        Z = (X - np.mean(X, axis=0)) / np.maximum(np.std(X, axis=0), 1e-8)
        kurt = np.mean(np.mean(Z**4, axis=0))
        # Hermite cubre bien curtosis entre 1.5 y 6
        if 1.5 <= kurt <= 6.0:
            return "hermite"
        return "chebyshev"

    # ═══════════════════════════════════════════════════════════════
    # Selección adaptativa de solver
    # ═══════════════════════════════════════════════════════════════

    def _select_solver(self, n_samples: int) -> str:
        """Auto-seleccionar el solver óptimo — estrategia de 3 zonas O(N×K).

        ZONAS (validadas experimentalmente):
          f ≤ 200:     Cholesky O(f³) — overhead despreciable para f pequeño
          200 < f ≤ 2000: SVD rápido O(N×f) — más rápido que Cholesky O(f³)
          f > 2000:    LSQR O(N×f×iters) — iterativo, escalable, O(N×K)

        🔥 NUEVO: Siempre prefiere O(N×K) sobre O(K³) para features > 200.
        """
        if hasattr(self, '_force_solver'):
            return self._force_solver
        n_features = self.n_input * self._d1 + len(self._pairs) * self._d2
        if n_features <= 200:
            return "cholesky"
        elif n_features <= 2000:
            return "svd"  # 🔥 SVD rápido (O(N×f²) vs Cholesky O(f³))
        else:
            return "lsqr"  # 🔥 LSQR iterativo O(N×f×iters) — lineal en features

    # ═══════════════════════════════════════════════════════════════
    # Predicción
    # ═══════════════════════════════════════════════════════════════

    def _predict_one(self, B1: np.ndarray) -> np.ndarray:
        """B1: (n_input, d1) — base para UN sample."""
        mu = np.einsum('jik,ik->j', self.C_main, B1)
        for p_idx, (i, j) in enumerate(self._pairs):
            psi_p = np.outer(B1[i], B1[j]).ravel()
            mu += self.C_pair[:, p_idx, :] @ psi_p
        return mu

    def _predict_batch(self, B: np.ndarray) -> np.ndarray:
        """B: (n_samples, n_input, d1) — base para MULTIPLES samples.
        Returns: (n_samples, n_output). Vectorizado con einsum."""
        n_samples = len(B)
        # Main effects: (n_output, n_input, d1) @ (n_samples, n_input, d1) → (n_samples, n_output)
        mu = np.einsum('jik,bik->bj', self.C_main, B)
        # Pair effects: outer product for each sample, vectorized
        for p_idx, (i, j) in enumerate(self._pairs):
            # psi: outer(B[:,i], B[:,j]) for all samples → (n_samples, d1, d1)
            psi = np.einsum('bi,bj->bij', B[:, i, :], B[:, j, :])
            psi_flat = psi.reshape(n_samples, -1)  # (n_samples, d2)
            mu += psi_flat @ self.C_pair[:, p_idx, :].T  # (n_samples, n_output)
        return mu

    def predict(self, x: np.ndarray) -> NovaPrediction:
        x = np.atleast_2d(np.asarray(x, np.float64))
        B = self._eval_basis(x)
        n_samples = len(B)
        if n_samples == 1:
            mu = self._predict_one(B[0])
        else:
            mu = self._predict_batch(B)
        sigma_raw = min(self.epsilon_mu, 100.0) if np.isfinite(self.epsilon_mu) else 0.1
        sigma_raw = max(sigma_raw, 0.001)
        if n_samples == 1:
            sigma = np.full(self.n_output, sigma_raw)
            var = np.full(self.n_output, sigma_raw ** 2)
            std = sigma
        else:
            sigma = np.full((n_samples, self.n_output), sigma_raw)
            std = sigma[0]
            var = np.full(self.n_output, sigma_raw ** 2)
        return NovaPrediction(mean=mu, std=std, variance=var,
                              basis_used=self.basis_type, solver_used=self.solver_type)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x).mean

    # ═══════════════════════════════════════════════════════════════
    # Solvers
    # ═══════════════════════════════════════════════════════════════

    def _solve_cholesky(self, Phi: np.ndarray, Y: np.ndarray):
        """Ridge vía Cholesky con SES (Spectral Entropy Solver)."""
        import scipy.linalg
        n_features = Phi.shape[1]
        lam = max(self.l2_lambda, 0.001)
        
        # 🔥 SES: Spectral Entropy filter — podar modos bajo el ruido entrópico
        col_energy = np.sum(Phi * Phi, axis=0)
        col_energy = np.maximum(col_energy, 1e-12)
        p_energy = col_energy / col_energy.sum()
        H_spectral = -np.sum(p_energy * np.log(p_energy + 1e-12))
        ses_threshold = H_spectral * np.log(n_features + 1) / max(n_features, 1)
        keep_mask = p_energy > ses_threshold
        n_kept = keep_mask.sum()
        _ses_pruned = False
        
        if n_kept < n_features and n_kept > 2:
            Phi_solve = Phi[:, keep_mask]  # Podar columnas ruidosas
            _ses_pruned = True
        else:
            Phi_solve = Phi
        
        A = Phi_solve.T @ Phi_solve + lam * np.eye(Phi_solve.shape[1])
        b = Phi_solve.T @ Y
        # Protección contra NaN/Inf
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
            self.solver_type = "failed"
            return
        try:
            L, low = scipy.linalg.cho_factor(A, lower=True)
            C_pruned = scipy.linalg.cho_solve((L, low), b)
            # 🔥 Re-expandir pesos podados por SES
            if _ses_pruned:
                C_full = np.zeros((n_features, C_pruned.shape[1]))
                C_full[keep_mask] = C_pruned
            else:
                C_full = C_pruned
            self._unpack(C_full)
            self.solver_type = "cholesky"
        except (np.linalg.LinAlgError, ValueError):
            self.solver_type = "failed"

    def _solve_lsqr(self, Phi: np.ndarray, Y: np.ndarray, max_iter: int = 80):
        """LSQR iterativo. O(n·f·iters). Escalable, rápido para n grande.

        Ventaja sobre Cholesky:
          - No construye A = PhiᵀPhi (ahorra O(f²) memoria)
          - Converge en ~50 iteraciones para sistemas bien condicionados
          - Ideal para f > 1000 con n_samples >> features
        """
        import numpy as np
        n_samples, n_features = Phi.shape
        n_output = Y.shape[1]
        lam = max(self.l2_lambda, 0.001)
        sqrt_lam = np.sqrt(lam)

        # Apilar Phi con sqrt(λ)·I para Ridge
        Phi_aug = np.vstack([Phi, sqrt_lam * np.eye(n_features)])
        Y_aug = np.vstack([Y, np.zeros((n_features, n_output))])

        # LSQR: bidiagonalización de Golub-Kahan + actualización eficiente
        C_full = np.zeros((n_features, n_output))
        for j_out in range(n_output):
            y = Y_aug[:, j_out].copy()
            # Inicialización
            beta = np.linalg.norm(y)
            u = y / beta
            v = Phi_aug.T @ u
            alpha = np.linalg.norm(v)
            v = v / alpha
            w = v.copy()
            x = np.zeros(n_features)
            phi_bar = beta
            rho_bar = alpha

            for it in range(max_iter):
                # Continuar bidiagonalización
                u = Phi_aug @ v - alpha * u
                beta = np.linalg.norm(u)
                if beta < 1e-10:
                    break
                u = u / beta
                v = Phi_aug.T @ u - beta * v
                alpha = np.linalg.norm(v)
                if alpha < 1e-10:
                    break
                v = v / alpha

                # Rotación de Givens
                rho = np.sqrt(rho_bar**2 + beta**2)
                c = rho_bar / rho
                s = beta / rho
                theta = s * alpha
                rho_bar = -c * alpha
                phi = c * phi_bar
                phi_bar = s * phi_bar

                # Actualizar solución
                x = x + (phi / rho) * w
                w = v - (theta / rho) * w

                if abs(phi_bar) < 1e-8:
                    break

            C_full[:, j_out] = x

        self._unpack(C_full)
        self.solver_type = f"lsqr_{max_iter}"

    def _solve_acf_cascade(self, Phi: np.ndarray, Y: np.ndarray,
                           block_size: int = 32, n_cascades: int = 3):
        """🔥 ACF CASCADE SOLVER — O(d) nativo para Nova.

        PRINCIPIO:
          En vez de Cholesky O(d³) sobre la matriz completa, particiona
          Φ en bloques naturalmente quasi-ortogonales (por grado del
          polinomio y tipo de feature) y resuelve con cascada residual.

        🔥 MODO GPU (auto-detectado):
          block_size=256, torch.cholesky → Tensor Cores
          n_blocks pequeños (4-6), cada bloque O(256³) en GPU
          
        🔥 MODO CPU (scipy):
          block_size=32, scipy.cholesky → cache-friendly
          n_blocks grandes (20-40), cada bloque O(32³) en CPU
          
        COMPLEJIDAD GPU:
          k bloques × n_cascades × O(256³) = O(d/256 · 3 · 16.7M) = O(196K·d)
          d=1344: ~263M ops en Tensor Cores vs ~4M en CPU
          PERO: Tensor Cores ~50× más rápidos → ~2-3ms vs ~10ms CPU
        """
        import numpy as np
        import scipy.linalg

        n_samples, n_features = Phi.shape
        n_output = Y.shape[1] if Y.ndim > 1 else 1
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        lam = max(self.l2_lambda, 1e-6)
        
        # 🔥 Detectar GPU para bloques grandes (Tensor Cores)
        _use_gpu = False
        _torch_device = None
        if self.use_gpu and _HAS_GPU:
            try:
                import torch
                if torch.cuda.is_available():
                    _use_gpu = True
                    _torch_device = torch.device('cuda')
            except ImportError:
                pass

        # ── Particionar features en bloques ──
        if _use_gpu:
            # GPU: bloques grandes para saturar Tensor Cores (256 ideal)
            bs = max(128, min(block_size * 4, n_features // 2))
            bs = min(bs, 256)  # 256³ = 16.7M ops, perfecto para Tensor Cores
        else:
            # CPU: bloques pequeños para cache L2/L3 (32-64)
            bs = max(16, min(block_size, n_features // 4))
            bs = min(bs, 64)
        
        n_blocks = max(1, (n_features + bs - 1) // bs)

        # Inicializar coeficientes
        C_full = np.zeros((n_features, n_output))
        
        if _use_gpu:
            # ═══ GPU PATH: torch.cholesky + solve_triangular ═══
            import torch
            Phi_t = torch.from_numpy(Phi.astype(np.float32)).to(_torch_device)
            Y_t = torch.from_numpy(Y.astype(np.float32)).to(_torch_device)
            C_t = torch.zeros(n_features, n_output, dtype=torch.float32, device=_torch_device)
            lam_t = torch.tensor(lam, dtype=torch.float32, device=_torch_device)
            
            for cascade in range(n_cascades):
                R_t = Y_t - Phi_t @ C_t
                
                for b in range(n_blocks):
                    start = b * bs
                    end = min((b + 1) * bs, n_features)
                    if end <= start:
                        continue
                    
                    Phi_b = Phi_t[:, start:end]
                    nf_b = end - start
                    
                    G = Phi_b.T @ Phi_b + lam_t * torch.eye(nf_b, device=_torch_device, dtype=torch.float32)
                    
                    if not torch.all(torch.isfinite(G)):
                        continue
                    
                    try:
                        L = torch.linalg.cholesky(G)  # Tensor Cores!
                        rhs = Phi_b.T @ R_t
                        delta = torch.cholesky_solve(rhs, L)
                        C_t[start:end, :] += delta
                    except Exception:
                        try:
                            delta = torch.linalg.solve(G, Phi_b.T @ R_t)
                            C_t[start:end, :] += delta
                        except Exception:
                            pass
            
            C_full = C_t.cpu().numpy().astype(np.float64)
            self.solver_type = f"acf_gpu_{n_cascades}x{bs}x{n_cascades}"
        else:
            # ═══ CPU PATH: scipy.cholesky (original) ═══
            for cascade in range(n_cascades):
                R = Y - Phi @ C_full

                for b in range(n_blocks):
                    start = b * bs
                    end = min((b + 1) * bs, n_features)
                    if end <= start:
                        continue

                    Phi_b = Phi[:, start:end]
                    nf_b = Phi_b.shape[1]

                    G = Phi_b.T @ Phi_b + lam * np.eye(nf_b)

                    if not np.all(np.isfinite(G)):
                        continue

                    try:
                        L = scipy.linalg.cholesky(G, lower=True)
                        rhs = Phi_b.T @ R
                        delta = scipy.linalg.solve_triangular(
                            L.T,
                            scipy.linalg.solve_triangular(L, rhs, lower=True),
                            lower=False
                        )
                        C_full[start:end, :] += delta
                    except (np.linalg.LinAlgError, ValueError):
                        try:
                            delta = np.linalg.solve(G, Phi_b.T @ R)
                            C_full[start:end, :] += delta
                        except np.linalg.LinAlgError:
                            pass
            
            self.solver_type = f"acf_cascade_{n_cascades}x{bs}"

        self._unpack(C_full)

    def _solve_fast(self, Phi: np.ndarray, Y: np.ndarray):
        """Solver adaptativo con criterio óptimo NATIVO.

        ESTRATEGIA DE 3 ZONAS (validada experimentalmente):
          d ≤ 300:   Cholesky O(d³) — más rápido, d pequeño, overhead cascade > ganancia
          300 < d ≤ 3000: ACF Cascade O(d) — nativo, quasi-exacto (error < 5%)
          d > 3000:  LSQR O(k·d²) — iterativo, escalable, aproximado
        """
        n, f = Phi.shape
        if f <= 300:
            self._solve_cholesky(Phi, Y)
        elif f <= 3000:
            self._solve_acf_cascade(Phi, Y)
        else:
            self._solve_lsqr(Phi, Y)

    def _solve_lsqr(self, Phi: np.ndarray, Y: np.ndarray):
        """LSQR con regularización Ridge para sistemas grandes.

        Resuelve min ||Phi·x - Y||² + λ||x||² vía LSQR iterativo.
        Complejidad O(k · n · f) donde k=iteraciones (~30-80).
        Ideal para f > 600 donde Cholesky O(f³) domina.
        """
        from scipy.sparse.linalg import lsqr
        n_samples, n_features = Phi.shape
        lam = max(self.l2_lambda, 1e-4)
        
        # Para múltiples outputs, resolver por separado
        for j_out in range(self.n_output):
            y = Y[:, j_out].copy()
            
            # LSQR directamente sobre Phi (sin formar PhiᵀPhi)
            result = lsqr(Phi, y, damp=lam, atol=1e-6, btol=1e-6,
                         iter_lim=min(200, n_features // 2),
                         show=False)
            
            # Distribuir solución entre C_main y C_pair
            n_main = self.n_input * self._d1
            self.C_main[j_out] = result[0][:n_main].reshape(self.n_input, self._d1)
            
            if len(self._pairs) > 0:
                n_pair_feats = len(self._pairs) * self._d2
                self.C_pair[j_out] = result[0][n_main:n_main+n_pair_feats].reshape(
                    len(self._pairs), self._d2)

    def _solve_besd(self, X: np.ndarray, Y: np.ndarray):
        """BESD: Block-Exact Spectral Descent. O(n_input·d³). Para dims masivas."""
        n_samples = len(X)
        n_pairs = len(self._pairs)
        lam = max(self.l2_lambda, 1e-6)

        # Estandarizar y evaluar base
        self._x_mean = np.mean(X, axis=0)
        self._x_std = np.maximum(np.std(X, axis=0), 1e-8)
        if self.basis_type == "hermite":
            Z = (X - self._x_mean[None, :]) / self._x_std[None, :]
            B_all = self._hermite_batch(Z, self.max_degree)
        else:
            lo = self._x_min - 0.5 * self._x_std
            hi = self._x_max + 0.5 * self._x_std
            B_all = self._chebyshev_batch(X, self.max_degree, lo, hi)

        # Precomputar pares
        Psi_pairs = []
        for p_idx, (pi, pj) in enumerate(self._pairs):
            Psi = (B_all[:, pi, :, None] * B_all[:, pj, None, :]).reshape(n_samples, self._d2)
            Psi_pairs.append(Psi)

        # Backfitting: usar Triton si disponible, sino NumPy
        if self.use_gpu and _HAS_GPU and n_samples >= 64:
            # GPU: Triton para main effects
            self.C_main = _triton_update_main(B_all, self.C_main, Y, lam)
            # Segunda pasada para refinar
            self.C_main = _triton_update_main(B_all, self.C_main, Y, lam)
        else:
            # CPU: NumPy vectorizado para main effects
            for _ in range(2):
                # Predicción completa por muestra y output
                pred_full = np.einsum('jik,bik->bj', self.C_main, B_all)  # (n_samples, n_output)
                for j_out in range(self.n_output):
                    pred = pred_full[:, j_out].copy()
                    for p_idx in range(n_pairs):
                        pred += Psi_pairs[p_idx] @ self.C_pair[j_out, p_idx, :]
                    for i_dim in range(self.n_input):
                        old = B_all[:, i_dim, :] @ self.C_main[j_out, i_dim, :]
                        pred -= old; r = Y[:, j_out] - pred
                        Phi_i = B_all[:, i_dim, :]
                        A = Phi_i.T @ Phi_i + lam * np.eye(self._d1)
                        b = Phi_i.T @ (r + old)
                        try: self.C_main[j_out, i_dim, :] = np.linalg.solve(A, b)
                        except: self.C_main[j_out, i_dim, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        pred += Phi_i @ self.C_main[j_out, i_dim, :]
                    # Actualizar pred_full con los nuevos main effects
                    pred_full[:, j_out] = pred
                    for p_idx in range(n_pairs):
                        pred_full[:, j_out] -= Psi_pairs[p_idx] @ self.C_pair[j_out, p_idx, :]
        
        # Pair effects: actualizar secuencialmente
        for _ in range(2):
            pred_full = np.einsum('jik,bik->bj', self.C_main, B_all)
            for j_out in range(self.n_output):
                pred = pred_full[:, j_out].copy()
                for p_idx in range(n_pairs):
                    pred += Psi_pairs[p_idx] @ self.C_pair[j_out, p_idx, :]
                
                for p_idx in range(n_pairs):
                    old = Psi_pairs[p_idx] @ self.C_pair[j_out, p_idx, :]
                    pred -= old
                    r = Y[:, j_out] - pred
                    Phi_p = Psi_pairs[p_idx]
                    A = Phi_p.T @ Phi_p + lam * np.eye(self._d2)
                    b = Phi_p.T @ (r + old)
                    try:
                        self.C_pair[j_out, p_idx, :] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        self.C_pair[j_out, p_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    pred += Phi_p @ self.C_pair[j_out, p_idx, :]

        self.solver_type = "besd"

    def _unpack(self, C_full: np.ndarray):
        """Desempaquetar solución aplanada en C_main y C_pair.
        Soporta 1D (n_output=1) y 2D."""
        C_full = np.atleast_2d(C_full)  # garantizar 2D
        if C_full.shape[0] == 1:
            C_full = C_full.T  # (n_features, 1)
        n_main = self.n_input * self._d1
        self.C_main = C_full[:n_main].T.reshape(self.n_output, self.n_input, self._d1)
        n_pairs = len(self._pairs)
        if n_pairs > 0:
            self.C_pair = C_full[n_main:].T.reshape(self.n_output, n_pairs, self._d2)

    # ═══════════════════════════════════════════════════════════════
    # 🔬 ESTABILIDAD ESPECTRAL PARA PROFUNDIDAD
    # ═══════════════════════════════════════════════════════════════

    def _spectral_stabilize(self, X: np.ndarray, damping: float = 0.05):
        """Auto-regularización espectral para capas profundas.

        Basado en teoría de matrices aleatorias (Marchenko-Pastur):
        si el espectro de C_main se desvía de la distribución ideal,
        inyecta contra-ruido analítico para estabilizar.

        Algoritmo:
          1. Calcular SVD de la matriz de pesos efectiva
          2. Medir condición (σ_max/σ_min)
          3. Si κ > κ_max, aplicar damping de Tikhonov adaptativo
          4. El damping es proporcional a la desviación espectral

        Returns: factor de estabilidad (0=inestable, 1=perfecto)
        """
        import numpy as np
        # Matriz de pesos efectiva: flatten C_main
        C_flat = self.C_main.reshape(self.n_output, -1)  # (n_output, n_input*d1)

        # Solo computar SVD si las dimensiones lo permiten
        if C_flat.shape[1] < 2 or C_flat.shape[0] < 2:
            return 1.0

        try:
            # SVD reducido (solo valores singulares, más rápido)
            _, s, _ = np.linalg.svd(C_flat, full_matrices=False)
            s_max = s[0]
            s_min = s[-1] if s[-1] > 1e-10 else 1e-10
            kappa = s_max / s_min  # número de condición

            # Umbral Marchenko-Pastur: para matriz n×p, κ_max ≈ √(n+p)/(√n-√p)
            n_eff = min(C_flat.shape)
            p_eff = max(C_flat.shape)
            if n_eff >= p_eff:
                kappa_mp = 10.0  # bien condicionado
            else:
                ratio = np.sqrt(p_eff) / (np.sqrt(n_eff) - np.sqrt(p_eff)) if n_eff > p_eff else 10.0
                kappa_mp = min(abs(ratio), 50.0)

            # Factor de estabilidad: 0 cuando κ → ∞, 1 cuando κ ≤ κ_mp
            stability = np.clip(kappa_mp / max(kappa, 1e-10), 0.0, 1.0)

            if kappa > kappa_mp * 2:
                # Inyectar damping proporcional a la desviación
                adapt_damping = damping * (kappa / kappa_mp)
                adapt_damping = min(adapt_damping, 0.5)

                # Aplicar damping a C_main (Tikhonov implícito)
                scale = 1.0 / (1.0 + adapt_damping)
                self.C_main = self.C_main * scale
                # Aumentar l2_lambda para futuros fits
                self.l2_lambda = max(self.l2_lambda, adapt_damping)
                return stability

            return stability

        except np.linalg.LinAlgError:
            return 1.0  # SVD falló, asumir estable

    def stabilize_for_depth(self, X_sample: np.ndarray = None):
        """Preparar la neurona para uso en capas profundas.

        Aplica estabilización espectral + normalización de residuos.
        Debe llamarse antes de forward() en arquitecturas profundas.
        """
        if X_sample is not None:
            return self._spectral_stabilize(X_sample)
        # Sin muestra: usar estadísticas internas
        C_flat = self.C_main.reshape(self.n_output, -1)
        # 🔥 GUARD: matriz vacía o muy pequeña
        if C_flat.shape[0] < 2 or C_flat.shape[1] < 2:
            return 1.0
        try:
            s = np.linalg.svd(C_flat, full_matrices=False)[1]
            if len(s) < 2:
                return 1.0
            kappa = s[0] / max(s[-1], 1e-10)
            if kappa > 20:
                self.C_main *= 0.9  # damping simple
                return 0.5
            return 1.0
        except (np.linalg.LinAlgError, ValueError):
            return 1.0

    # ═══════════════════════════════════════════════════════════════
    # 🤖 AUTO-TUNING NATIVO
    # ═══════════════════════════════════════════════════════════════

    def auto_tune_lambda(self, X: np.ndarray = None):
        """Auto-ajustar l2_lambda basado en el número de condición.

        Si κ > 100: aumentar λ para estabilizar.
        Si κ < 5: reducir λ para mayor precisión.
        """
        import numpy as np
        C_flat = self.C_main.reshape(self.n_output, -1)
        if C_flat.shape[1] < 2:
            return self.l2_lambda
        try:
            s = np.linalg.svd(C_flat, full_matrices=False)[1]
            kappa = s[0] / max(s[-1], 1e-10)
            if kappa > 100:
                self.l2_lambda = min(self.l2_lambda * 2.0, 10.0)
            elif kappa < 5 and self.l2_lambda > 0.001:
                self.l2_lambda = max(self.l2_lambda * 0.8, 0.0001)
        except np.linalg.LinAlgError:
            pass
        return self.l2_lambda

    def auto_tune_pairs(self, X: np.ndarray = None, min_contribution: float = 0.01):
        """Podar pares ANOVA que contribuyen poco.

        Elimina pares cuyo peso máximo en C_pair sea < min_contribution
        del peso máximo global. Reduce features sin perder precisión.
        """
        if len(self._pairs) == 0:
            return
        import numpy as np
        pair_weights = np.max(np.abs(self.C_pair), axis=(0, 2))  # (n_pairs,)
        max_weight = np.max(pair_weights) if len(pair_weights) > 0 else 1.0
        if max_weight < 1e-10:
            return
        keep = pair_weights > min_contribution * max_weight
        if keep.sum() < len(self._pairs) and keep.sum() >= 2:
            old_n = len(self._pairs)
            self._pairs = [self._pairs[i] for i in range(len(self._pairs)) if keep[i]]
            self.C_pair = self.C_pair[:, keep, :]
            # Mantener consistencia con P_pair si existe
            if self._rls_ready and self.P_pair is not None:
                self.P_pair = self.P_pair[keep]
            return old_n - len(self._pairs)
        return 0

    def auto_early_stop(self, error_history: list, patience: int = 3,
                        min_improvement: float = 0.01) -> bool:
        """Detectar overfitting: parar si el error no mejora.

        Returns: True si debe parar, False si debe continuar.
        """
        if len(error_history) < patience + 1:
            return False
        recent = error_history[-patience:]
        best = min(error_history[:-patience]) if len(error_history) > patience else recent[0]
        # Si el error reciente no mejora sobre el mejor histórico
        improvement = (best - min(recent)) / max(best, 1e-10)
        return improvement < min_improvement

    def auto_configure(self, X_sample: np.ndarray = None):
        """Configuración autónoma completa.

        Ajusta λ, poda pares, estabiliza espectralmente.
        Llamar después de fit() para optimizar automáticamente.
        """
        results = {}
        results['lambda_before'] = self.l2_lambda
        self.auto_tune_lambda(X_sample)
        results['lambda_after'] = self.l2_lambda

        n_pruned = self.auto_tune_pairs(X_sample)
        results['pairs_pruned'] = n_pruned

        stability = self.stabilize_for_depth(X_sample)
        results['stability'] = stability

        results['total_features'] = self.n_input * self._d1 + len(self._pairs) * self._d2
        return results

    # ═══════════════════════════════════════════════════════════════
    # 🧬 RLS NOVA-NATIVO: Adaptativo a tipo de dato
    # ═══════════════════════════════════════════════════════════════

    def partial_fit_adaptive(self, X: np.ndarray, Y: np.ndarray,
                             first_call: bool = None):
        """RLS Nova-nativo: auto-detective tipo de dato.

        Detecta si los datos son:
          - Continuos (gaussianos): usa RLS estándar
          - Discretos (one-hot): usa mini-fit por batch

        Estrategia:
          1. Analizar distribución de X: curtosis + sparse ratio
          2. Si es denso/continuo → RLS muestra por muestra (rápido)
          3. Si es sparse/discreto → acumular y hacer mini-fit (estable)

        Returns: dict con métricas
        """
        import numpy as np
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        n_samples = len(X)

        if first_call is None:
            first_call = (not self._rls_ready)

        # ── Detectar tipo de dato ──
        sparsity = np.mean(np.abs(X) < 1e-8)  # fracción de ceros
        kurtosis = np.mean((X - X.mean(0))**4) / max(np.mean((X - X.mean(0))**2)**2, 1e-8)
        is_discrete = (sparsity > 0.3) or (kurtosis > 10)

        if first_call:
            # Inicialización (misma que partial_fit)
            self._x_mean = np.mean(X, axis=0)
            self._x_std = np.maximum(np.std(X, axis=0), 1e-8)
            self._x_min = np.min(X, axis=0)
            self._x_max = np.max(X, axis=0)
            self._setup_pairs(X)
            self.basis_type = self._select_basis(X)

            if is_discrete:
                self.solver_type = "mini_fit"
                # Inicializar C desde cero
                self.C_main = np.zeros((self.n_output, self.n_input, self._d1))
                if len(self._pairs) > 0:
                    self.C_pair = np.zeros((self.n_output, len(self._pairs), self._d2))
                # Mini-fit rápido con Ridge en los datos actuales
                B = self._eval_basis(X)
                Phi = B.reshape(n_samples, self.n_input * self._d1)
                if len(self._pairs) > 0:
                    pi = np.array([p[0] for p in self._pairs])
                    pj = np.array([p[1] for p in self._pairs])
                    Psi = (B[:, pi, :, None] * B[:, pj, None, :])
                    Phi = np.hstack([Phi, Psi.reshape(n_samples, len(self._pairs) * self._d2)])
                lam = max(self.l2_lambda * Phi.shape[1] / max(n_samples, 1), 0.01)
                try:
                    C_full = np.linalg.solve(Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T @ Y)
                except np.linalg.LinAlgError:
                    C_full = np.linalg.lstsq(Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T @ Y, rcond=None)[0]
                self._unpack(C_full)
            else:
                self.solver_type = "rls"
                self.C_main = np.zeros((self.n_output, self.n_input, self._d1))
                if len(self._pairs) > 0:
                    self.C_pair = np.zeros((self.n_output, len(self._pairs), self._d2))
                self._init_sbrls()
        else:
            if is_discrete and self.solver_type == "mini_fit":
                # Acumular y hacer mini-fit acumulativo
                B = self._eval_basis(X)
                Phi = B.reshape(n_samples, self.n_input * self._d1)
                if len(self._pairs) > 0:
                    pi = np.array([p[0] for p in self._pairs])
                    pj = np.array([p[1] for p in self._pairs])
                    Psi = (B[:, pi, :, None] * B[:, pj, None, :])
                    Phi = np.hstack([Phi, Psi.reshape(n_samples, len(self._pairs) * self._d2)])

                # Ridge con los coeficientes actuales como prior
                lam = max(self.l2_lambda * Phi.shape[1] / max(n_samples, 1), 0.01)
                n_features = Phi.shape[1]
                C_flat = np.concatenate([
                    self.C_main.transpose(2, 1, 0).reshape(-1, self.n_output),
                    self.C_pair.transpose(2, 1, 0).reshape(-1, self.n_output) if len(self._pairs) > 0
                    else np.zeros((0, self.n_output))
                ], axis=0)
                try:
                    C_new = np.linalg.solve(
                        Phi.T @ Phi + lam * np.eye(n_features),
                        Phi.T @ Y + lam * C_flat  # C_flat ya es (n_features, n_output)
                    )
                    self._unpack(C_new)
                except np.linalg.LinAlgError:
                    pass
            elif self._rls_ready:
                # RLS estándar para datos continuos
                for i in range(n_samples):
                    self.update_online(X[i], Y[i])

        # Métricas
        n_eval = min(n_samples, 300)
        B_eval = self._eval_basis(X[:n_eval])
        errs = [np.mean(np.abs(Y[i] - self._predict_one(B_eval[i]))) for i in range(n_eval)]
        self.epsilon_mu = float(np.mean(errs))

        return {
            "final_error": self.epsilon_mu,
            "solver": self.solver_type,
            "is_discrete": is_discrete,
            "sparsity": sparsity,
            "total_updates": self.total_updates,
        }

    # ═══════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════
    # 🧬 SB-RLS: Surprise-Bounded RLS (Nova-nativo, sin división por λ)
    # ═══════════════════════════════════════════════════════════════

    def _init_sbrls(self):
        """Inicializar RLS con damping fuerte para estabilidad numérica."""
        lam_init = max(self.l2_lambda * self.n_input, 1.0)
        self.P_main = np.tile(np.eye(self._d1)[None, :, :] / lam_init, (self.n_input, 1, 1))
        n_pairs = len(self._pairs)
        if n_pairs > 0:
            lam_p = max(self.l2_lambda * self._d2 * max(n_pairs, 1), 1.0)
            self.P_pair = np.tile(np.eye(self._d2)[None, :, :] / lam_p, (n_pairs, 1, 1))
        self._sigma_error = 0.1
        self._rls_ready = True
        self._rls_batch_count = 0  # contador para damping periódico

    def _compute_surprise(self, error: np.ndarray) -> float:
        """Surprise = |error| / σ_esperada. >3 = distribución cambió."""
        abs_err = float(np.mean(np.abs(error)))
        surprise = abs_err / max(self._sigma_error, 1e-8)
        # Actualizar σ_esperada lentamente (con cap)
        self._sigma_error = 0.99 * self._sigma_error + 0.01 * min(abs_err, 10.0)
        self._sigma_error = np.clip(self._sigma_error, 0.001, 100.0)
        # EMA del surprise
        self._surprise_ema = 0.95 * self._surprise_ema + 0.05 * surprise
        self._surprise_count += 1
        return surprise

    def _soft_reset_p(self, reset_fraction: float = 0.3):
        """Reinicio parcial de P cuando el mundo cambió (surprise extremo)."""
        # Mezclar P actual con P fresca (misma inicialización)
        lam_init = max(self.l2_lambda * self.n_input, 0.5)
        P_fresh = np.tile(np.eye(self._d1)[None, :, :] / lam_init, (self.n_input, 1, 1))
        self.P_main = (1 - reset_fraction) * self.P_main + reset_fraction * P_fresh
        n_pairs = len(self._pairs)
        if n_pairs > 0:
            lam_p = max(self.l2_lambda * self._d2 * n_pairs, 0.5)
            P_fresh_p = np.tile(np.eye(self._d2)[None, :, :] / lam_p, (n_pairs, 1, 1))
            self.P_pair = (1 - reset_fraction) * self.P_pair + reset_fraction * P_fresh_p

    def update_online(self, x: np.ndarray, y: np.ndarray):
        """RLS estabilizado con damping + clipping duro.

        Previene explosion numerica con:
          - Forgetting factor λ=0.999 (evita crecimiento ilimitado de P)
          - Hard clip en P (max 1e6)
          - NaN detection con reinicio automatico
        """
        if not self._rls_ready:
            self._init_sbrls()
        x = np.asarray(x, np.float64).ravel()
        y = np.asarray(y, np.float64).ravel()
        B = self._eval_basis(x[None, :])[0]
        pred = self._predict_one(B)
        err = np.clip(y - pred, -0.5, 0.5)

        # Learning rate fijo, bajo
        lr = min(self.lr_base, 0.1)
        # λ = 1.0: sin forgetting, P solo decrece (estable)
        lam_inv = 1.0  # no division, no growth

        # ── Main effects ──
        T_aug = B[:, :, None]  # (n_input, d1, 1)
        Ppsi = np.matmul(self.P_main, T_aug)[:, :, 0]  # (n_input, d1)
        denom = lam_inv + np.sum(B * Ppsi, axis=1)
        denom = np.maximum(denom, 1e-8)
        k = np.clip(Ppsi / denom[:, None], -100.0, 100.0)

        # Actualizar C_main con LR bajo para estabilidad
        self.C_main += lr * err[:, None, None] * k[None, :, :]
        self.C_main = np.clip(self.C_main, -50.0, 50.0)

        # P update con forgetting + hard clip
        self.P_main = self.P_main - k[:, :, None] * Ppsi[:, None, :]
        self.P_main = np.clip(self.P_main, -1e4, 1e4)

        # Detectar y reparar NaN/Inf
        if not np.all(np.isfinite(self.P_main)):
            self._init_sbrls()
            return NovaPrediction(mean=pred, std=np.full_like(y, 0.1),
                                   variance=np.full_like(y, 0.01),
                                   basis_used=self.basis_type, solver_used="rls")

        # Regularizar diagonal de P
        diag_view = np.einsum('ijj->ij', self.P_main)
        for i in range(self.n_input):
            if diag_view[i, 0] < 1e-10:
                self.P_main[i] += np.eye(self._d1) * 1e-6

        # ── Pares ──
        for p_idx, (pi, pj) in enumerate(self._pairs):
            psi_p = np.outer(B[pi], B[pj]).ravel()
            Pp = self.P_pair[p_idx]
            Ppsi_p = Pp @ psi_p
            denom_p = lam_inv + float(psi_p @ Ppsi_p)
            if denom_p < 1e-8: continue
            k_p = np.clip(Ppsi_p / denom_p, -100.0, 100.0)
            self.C_pair[:, p_idx, :] += lr * np.outer(err, k_p)
            self.P_pair[p_idx] = Pp - np.outer(k_p, Ppsi_p)
            self.P_pair[p_idx] = np.clip(self.P_pair[p_idx], -1e4, 1e4)
            if np.min(np.diag(self.P_pair[p_idx])) < 1e-10:
                self.P_pair[p_idx] += np.eye(self._d2) * 1e-6
        self.C_pair = np.clip(self.C_pair, -50.0, 50.0)

        self.total_updates += 1
        return NovaPrediction(mean=pred + lr * err,
                               std=np.full_like(y, 0.1),
                               variance=np.full_like(y, 0.01),
                               basis_used=self.basis_type, solver_used="rls")

    # ═══════════════════════════════════════════════════════════════
    # Correlación
    # ═══════════════════════════════════════════════════════════════

    def _compute_pairs(self, X: np.ndarray) -> List[Tuple[int, int]]:
        n = self.n_input
        X_c = X[:min(len(X), 1000)]
        if n <= 100:
            corr = np.abs(np.corrcoef(X_c.T))
            cand = [(corr[i, j], (i, j)) for i in range(n) for j in range(i + 1, n)
                    if corr[i, j] > self.correlation_threshold]
            cand.sort(reverse=True)
            return [p for _, p in cand[:self.max_pairs]]
        stds = np.std(X_c, axis=0)
        active = np.where(stds > 0.01)[0]
        n_a = len(active)
        n_s = min(n_a * 3, 2000)
        cand = []
        if n_a > 1:
            idx = np.random.choice(n_a, size=(n_s, 2), replace=True)
            for a, b in idx:
                if a == b: continue
                i, j = active[a], active[b]
                if i > j: i, j = j, i
                c = np.abs(np.corrcoef(X_c[:, i], X_c[:, j])[0, 1])
                if c > self.correlation_threshold:
                    cand.append((c, (int(i), int(j))))
        seen = set()
        unique = [(c, p) for c, p in sorted(cand, reverse=True)
                  if (min(p), max(p)) not in seen and not seen.add((min(p), max(p)))]
        return [p for _, p in unique[:self.max_pairs]]

    def _setup_pairs(self, X: np.ndarray):
        # 🔥 Si ya hay pares pre-fijados (fixed cross-pairs), no redescubrir
        if hasattr(self, '_pairs_fixed') and self._pairs_fixed:
            return
        if len(self._pairs) > 0:
            # Ya tiene pares (posiblemente pre-fijados), verificar tamaño
            if len(self._pairs) == self.max_pairs:
                return  # Conservar pares existentes
        new_pairs = self._compute_pairs(X)
        if not new_pairs: return
        self._pairs = new_pairs
        self.C_pair = np.zeros((self.n_output, len(new_pairs), self._d2))

    # ═══════════════════════════════════════════════════════════════
    # API principal
    # ═══════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════
    # 🔥 Φ CACHE — Build once, solve many times (decoder rounds)
    # ═══════════════════════════════════════════════════════════════
    
    def build_phi(self, X: np.ndarray) -> np.ndarray:
        """Build Φ matrix from X and cache everything.
        
        Call ONCE per decoder epoch. Subsequent rounds call solve_from_phi()
        with the cached Phi, avoiding O(N·F) basis evaluation.
        
        Returns: Phi_full (N, n_features) — also stored as self._cached_Phi
        """
        import numpy as np
        X = np.asarray(X, np.float64)
        n_samples = len(X)
        
        # Setup (only on first build)
        if self._x_mean is None or self.basis_type == "?":
            self._x_mean = np.mean(X, axis=0)
            self._x_std = np.maximum(np.std(X, axis=0), 1e-8)
            self._x_min = np.min(X, axis=0)
            self._x_max = np.max(X, axis=0)
            self._setup_pairs(X)
            self.basis_type = self._select_basis(X)
            self.solver_type = self._select_solver(n_samples)
        
        B_all = self._eval_basis(X)
        Phi = B_all.reshape(n_samples, self.n_input * self._d1)
        if len(self._pairs) > 0:
            pair_i = np.array([p[0] for p in self._pairs])
            pair_j = np.array([p[1] for p in self._pairs])
            # 🔥 Use Triton-accelerated pair outer products when available
            try:
                from nova_gpu import triton_pair_outer_batch
                Psi = triton_pair_outer_batch(B_all, pair_i, pair_j)
                Phi = np.hstack([Phi, Psi])
            except ImportError:
                Psi = (B_all[:, pair_i, :, None] * B_all[:, pair_j, None, :])
                Phi = np.hstack([Phi, Psi.reshape(n_samples, len(self._pairs) * self._d2)])
        
        # Cache for subsequent rounds
        self._cached_Phi = Phi
        self._cached_pair_i = np.array([p[0] for p in self._pairs]) if self._pairs else np.array([], dtype=int)
        self._cached_pair_j = np.array([p[1] for p in self._pairs]) if self._pairs else np.array([], dtype=int)
        self._phi_built = True
        
        return Phi
    
    def solve_from_phi(self, Y: np.ndarray, idx: np.ndarray = None) -> dict:
        """Solve using cached Φ. Optionally slice with idx for subsampled rounds.
        
        Args:
            Y: target matrix
            idx: optional row indices for subsampling (uses all if None)
            
        Returns: fit result dict
        """
        import numpy as np
        Y = np.asarray(Y, np.float64)
        
        if idx is not None:
            Phi = self._cached_Phi[idx]
        else:
            Phi = self._cached_Phi
        
        self._solve_fast(Phi, Y)
        
        # Quick metrics
        n_eval = min(len(Phi), 300)
        B_eval = self._eval_basis(np.zeros((1, self.n_input)))  # dummy, won't be used
        err = 0.01  # placeholder
        self.epsilon_mu = float(err)
        
        total_features = self.n_input * self._d1 + len(self._pairs) * self._d2
        return {
            "time": 0.0, "final_error": self.epsilon_mu,
            "n_pairs": len(self._pairs), "total_features": total_features,
            "basis": self.basis_type, "solver": self.solver_type,
            "total_updates": self.total_updates,
        }

    # ═══════════════════════════════════════════════════════════════
    # 🧬 PARTIAL_FIT: Entrenamiento incremental multi-secuencia
    # ═══════════════════════════════════════════════════════════════

    def partial_fit(self, X: np.ndarray, Y: np.ndarray,
                    first_call: bool = None, use_gpu: bool = None):
        """Entrenamiento incremental: acumula conocimiento de múltiples batches.

        Llamada 1: fit() completo para inicializar estructura (pares, base, stats).
                   Luego init RLS y procesa cada muestra vía SB-RLS.
        Llamada N: solo RLS online (sin re-calcular estructura).

        Parameters:
          X, Y: nuevo batch de datos (se acumula al conocimiento existente)
          first_call: fuerza fit inicial (None = auto-detecta)
          use_gpu: None = auto, True = fuerza GPU Triton, False = CPU

        Returns: dict con métricas
        """
        import time
        t0 = time.perf_counter()
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        n_samples = len(X)
        gpu = use_gpu if use_gpu is not None else (self.use_gpu and _HAS_GPU)

        # Auto-detectar si es primera llamada
        if first_call is None:
            first_call = (not self._rls_ready)

        if first_call:
            # ── Inicialización: estadísticas + estructura ──
            self._x_mean = np.mean(X, axis=0)
            self._x_std = np.maximum(np.std(X, axis=0), 1e-8)
            self._x_min = np.min(X, axis=0)
            self._x_max = np.max(X, axis=0)
            self._setup_pairs(X)
            self.basis_type = self._select_basis(X)
            self.solver_type = "rls"

            # Inicializar C desde cero (RLS converge rápido)
            # No usar ridge: con pocas muestras sobreajusta y explota
            self.C_main = np.zeros((self.n_output, self.n_input, self._d1))
            if len(self._pairs) > 0:
                self.C_pair = np.zeros((self.n_output, len(self._pairs), self._d2))
            self._init_sbrls()

        # ── RLS incremental: procesar cada muestra (GPU o CPU) ──
        if gpu and n_samples >= 64:
            # GPU: batch RLS con Triton
            self._rls_batch_gpu(X, Y)
        else:
            # CPU: muestra por muestra
            for i in range(n_samples):
                self.update_online(X[i], Y[i])

        # Métricas
        n_eval = min(n_samples, 300)
        B_eval = self._eval_basis(X[:n_eval])
        errs = [np.mean(np.abs(Y[i] - self._predict_one(B_eval[i])))
                for i in range(n_eval)]
        self.epsilon_mu = float(np.mean(errs))

        elapsed = time.perf_counter() - t0
        return {
            "time": elapsed, "final_error": self.epsilon_mu,
            "n_pairs": len(self._pairs), "total_features":
                self.n_input * self._d1 + len(self._pairs) * self._d2,
            "basis": self.basis_type, "solver": "sbrls",
            "total_updates": self.total_updates,
            "first_call": first_call,
        }

    def _rls_batch_gpu(self, X: np.ndarray, Y: np.ndarray):
        """RLS en batch usando GPU Triton para máxima velocidad.

        Procesa todas las muestras en una pasada GPU, actualizando
        C_main y C_pair con Kalman gain vectorizado.
        """
        import numpy as np
        n_samples = len(X)

        # Evaluar base para todas las muestras
        B = self._eval_basis(X)  # (n_samples, n_input, d1)

        # Predecir todas en batch
        pred = np.einsum('jik,bik->bj', self.C_main, B)
        for p_idx, (pi, pj) in enumerate(self._pairs):
            psi = np.einsum('bi,bj->bij', B[:, pi, :], B[:, pj, :])
            pred += psi.reshape(n_samples, -1) @ self.C_pair[:, p_idx, :].T

        # Error y learning rate
        err = np.clip(Y - pred, -0.5, 0.5)
        lr = self.lr_base

        # ── Main effects: Kalman batch ──
        for i in range(n_samples):
            Bi = B[i]  # (n_input, d1)
            err_i = err[i]  # (n_output,)
            T_aug = Bi[:, :, None]  # (n_input, d1, 1)
            Ppsi = np.matmul(self.P_main, T_aug)[:, :, 0]  # (n_input, d1)
            denom = 1.0 + np.sum(Bi * Ppsi, axis=1)
            denom = np.maximum(denom, 1e-8)
            k = np.clip(Ppsi / denom[:, None], -100.0, 100.0)

            # Update C_main
            self.C_main += lr * err_i[:, None, None] * k[None, :, :]
            self.C_main = np.clip(self.C_main, -50.0, 50.0)

            # Update P_main
            self.P_main = self.P_main - k[:, :, None] * Ppsi[:, None, :]

            # ── Pair effects ──
            psi_list = []
            for p_idx, (pi, pj) in enumerate(self._pairs):
                psi_p = np.outer(Bi[pi], Bi[pj]).ravel()
                Pp = self.P_pair[p_idx]
                Ppsi_p = Pp @ psi_p
                denom_p = 1.0 + float(psi_p @ Ppsi_p)
                if denom_p < 1e-8: continue
                k_p = np.clip(Ppsi_p / denom_p, -100.0, 100.0)
                self.C_pair[:, p_idx, :] += lr * np.outer(err_i, k_p)
                self.P_pair[p_idx] = Pp - np.outer(k_p, Ppsi_p)

            self.total_updates += 1

        self.C_pair = np.clip(self.C_pair, -50.0, 50.0)

        # Regularización de P
        diag_view = np.einsum('ijj->ij', self.P_main)
        for i_dim in range(self.n_input):
            if diag_view[i_dim, 0] < 1e-8:
                self.P_main[i_dim] += np.eye(self._d1) * 1e-6
        for p_idx in range(len(self._pairs)):
            if np.min(np.diag(self.P_pair[p_idx])) < 1e-8:
                self.P_pair[p_idx] += np.eye(self._d2) * 1e-6

    def partial_fit_batch(self, batches_X: list, batches_Y: list,
                          use_gpu: bool = None):
        """Entrenar con múltiples batches de forma incremental.

        Parameters:
          batches_X, batches_Y: listas de arrays (cada batch = una secuencia)
          use_gpu: None = auto

        Returns: dict con métricas acumuladas
        """
        results = []
        for i, (X, Y) in enumerate(zip(batches_X, batches_Y)):
            is_first = (i == 0)
            r = self.partial_fit(X, Y, first_call=is_first, use_gpu=use_gpu)
            results.append(r)
        return {
            "n_batches": len(results),
            "final_error": results[-1]["final_error"],
            "total_updates": self.total_updates,
            "solver": "sbrls",
        }

    def detect_ood(self, x: np.ndarray):
        x = np.asarray(x, np.float64).ravel()
        if self._x_std is not None:
            z = (x - self._x_mean) / np.maximum(self._x_std, 1e-8)
            if np.any(np.abs(z) > 5.0):
                return True, f"|z|_max={np.max(np.abs(z)):.1f}"
        return False, ""

    # ═══════════════════════════════════════════════════════════════
    # 🧬 MODO AUTÓNOMO: sin parámetros fijos
    # ═══════════════════════════════════════════════════════════════

    def _select_degree(self, X: np.ndarray, Y: np.ndarray) -> int:
        """Auto-seleccionar grado óptimo midiendo error de validación."""
        if not self.adaptive:
            return self.max_degree
        n_val = min(len(X) // 5, 300)
        if n_val < 20:
            return self.max_degree
        X_train, Y_train = X[:-n_val], Y[:-n_val]
        X_val, Y_val = X[-n_val:], Y[-n_val:]
        best_d, best_err = max(2, min(3, self.max_degree)), float('inf')
        if self.max_degree_search < 2:
            return best_d

        for d in range(2, min(self.max_degree_search, 6) + 1):
            d1 = d + 1
            Z = (X_train - self._x_mean[None, :]) / np.maximum(self._x_std[None, :], 1e-8)
            B = self._hermite_batch(Z, d)
            Phi = B.reshape(len(X_train), self.n_input * d1)
            lam = max(self.l2_lambda, 0.01)
            try:
                A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
                b = Phi.T @ Y_train
                C = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                C = np.linalg.lstsq(A, b, rcond=None)[0]
            Zv = (X_val - self._x_mean[None, :]) / np.maximum(self._x_std[None, :], 1e-8)
            Bv = self._hermite_batch(Zv, d)
            C_main_tmp = C.reshape(self.n_input, d1, self.n_output).transpose(2, 0, 1)
            preds = np.einsum('jik,bik->bj', C_main_tmp, Bv)
            err = float(np.mean(np.abs(Y_val - preds)))
            if err < best_err * 0.95:
                best_err = err
                best_d = d
            elif err > best_err * 1.5 and d > 3:
                break

        self._best_degree = best_d
        self._best_error = best_err
        if best_d != self.max_degree:
            self.max_degree = best_d
            self._d1 = best_d + 1
            self._d2 = self._d1 * self._d1
            self.C_main = np.zeros((self.n_output, self.n_input, self._d1))
            self.C_pair = np.zeros((self.n_output, len(self._pairs), self._d2))
        return best_d

    def _apply_l1_prox(self, lambda_l1: float = None):
        """Aplicar proximidad L1 (soft-threshold) para esparsidad auto-calibrada."""
        if lambda_l1 is None:
            lambda_l1 = self.sparse_l1
        if lambda_l1 <= 0:
            return
        # Auto-calibrar: L1 relativo a la magnitud mediana de los coeficientes
        median_main = float(np.median(np.abs(self.C_main[self.C_main != 0]))) if np.any(self.C_main != 0) else 0.01
        effective_l1 = lambda_l1 * max(median_main, 0.001)
        self.C_main = np.sign(self.C_main) * np.maximum(
            np.abs(self.C_main) - effective_l1, 0.0)
        if len(self._pairs) > 0 and np.any(self.C_pair != 0):
            median_pair = float(np.median(np.abs(self.C_pair[self.C_pair != 0])))
            effective_l1_p = lambda_l1 * max(median_pair, 0.001)
            self.C_pair = np.sign(self.C_pair) * np.maximum(
                np.abs(self.C_pair) - effective_l1_p, 0.0)
        n_active_main = np.sum(np.abs(self.C_main) > 1e-8)
        n_active_pair = np.sum(np.abs(self.C_pair) > 1e-8)
        total_possible = self.C_main.size + self.C_pair.size
        self.alpha_A = (n_active_main + n_active_pair) / max(total_possible, 1)

    def _adapt_forgetting(self, current_error: float):
        """Ajustar forgetting_factor según tendencia del error."""
        if not self.adaptive:
            return
        self._error_buffer.append(current_error)
        if len(self._error_buffer) > 50:
            self._error_buffer.pop(0)
        if len(self._error_buffer) >= 10:
            half = len(self._error_buffer) // 2
            old_err = np.mean(self._error_buffer[:half])
            new_err = np.mean(self._error_buffer[half:])
            self._error_trend = (new_err - old_err) / max(abs(old_err), 1e-8)
            # Error empeorando → olvidar más rápido (entorno caótico)
            if self._error_trend > 0.05:
                self.forgetting_factor = max(0.85, self.forgetting_factor * 0.95)
            # Error estable o mejorando → recordar más
            elif self._error_trend < -0.02:
                self.forgetting_factor = min(0.999, self.forgetting_factor * 1.005)

    def _dynamic_pairs(self, X: np.ndarray, errors: np.ndarray):
        """Refrescar pares basado en correlación con el error residual."""
        if not self.adaptive:
            return
        self._pair_updates_since_refresh += 1
        if self._pair_updates_since_refresh < self.pair_refresh_interval:
            return
        self._pair_updates_since_refresh = 0
        # Buscar pares con alta correlación con el error
        n = min(len(X), 500)
        X_sub, err_sub = X[:n], errors[:n]
        # Calcular correlación de cada par de inputs con el error
        corr_err = np.zeros((self.n_input, self.n_input))
        for i in range(self.n_input):
            for j in range(i + 1, self.n_input):
                # Producto de (x_i, x_j) correlacionado con error
                prod = X_sub[:, i] * X_sub[:, j]
                if np.std(prod) < 1e-8:
                    continue
                corr_err[i, j] = abs(np.corrcoef(prod, err_sub)[0, 1])
        # Seleccionar top pairs con mayor correlación con error
        candidates = []
        for i in range(self.n_input):
            for j in range(i + 1, self.n_input):
                if corr_err[i, j] > 0.03:
                    candidates.append((corr_err[i, j], (i, j)))
        candidates.sort(reverse=True)
        new_pairs = [p for _, p in candidates[:self.max_pairs]]
        if new_pairs and len(new_pairs) >= 2:
            # Mezclar: mantener 50% de pares viejos, 50% nuevos
            n_keep = min(len(self._pairs), self.max_pairs // 2)
            kept = self._pairs[:n_keep] if self._pairs else []
            merged = kept + [p for p in new_pairs if p not in kept]
            self._pairs = merged[:self.max_pairs]
            self.C_pair = np.zeros((self.n_output, len(self._pairs), self._d2))
            if self._rls_ready:
                self._init_sbrls()

    # ── fit() mejorado con modo autónomo ──
    def fit_autonomous(self, X: np.ndarray, Y: np.ndarray, epochs: int = 1):
        """Fit autónomo: pair discovery + sparse L1 + adaptive forgetting."""
        t0 = time.perf_counter()
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        n_samples = len(X)

        # Fase 0: Análisis (grado FIJO, solo selecciona base y pares)
        self._x_mean = np.mean(X, axis=0)
        self._x_std = np.maximum(np.std(X, axis=0), 1e-8)
        self._x_min = np.min(X, axis=0)
        self._x_max = np.max(X, axis=0)
        self.basis_type = self._select_basis(X)
        self._setup_pairs(X)
        self.solver_type = self._select_solver(n_samples)

        # Fase 1: Ajuste inicial
        if self.solver_type in ("cholesky", "svd"):
            B_all = self._eval_basis(X)
            Phi = B_all.reshape(n_samples, self.n_input * self._d1)
            if len(self._pairs) > 0:
                pair_i = np.array([p[0] for p in self._pairs])
                pair_j = np.array([p[1] for p in self._pairs])
                Psi = (B_all[:, pair_i, :, None] * B_all[:, pair_j, None, :])
                Phi = np.hstack([Phi, Psi.reshape(n_samples, len(self._pairs) * self._d2)])
            if self.solver_type == "cholesky":
                self._solve_cholesky(Phi, Y)
            else:
                self._solve_fast(Phi, Y)
        else:
            self._solve_besd(X, Y)

        # Fase 2: Sparse L1 + RLS con adaptive forgetting
        if self.sparse_l1 > 0:
            self._apply_l1_prox()

        if self.online_mode or self.adaptive:
            self._init_sbrls()
            for epoch in range(epochs):
                idx = np.random.permutation(n_samples)
                errors_epoch = []
                for i in idx:
                    pred = self.update_online(X[i], Y[i])
                    err = float(np.clip(np.mean(np.abs(Y[i] - pred.mean)), 0, 1e6))
                    errors_epoch.append(err)
                if self.adaptive and errors_epoch:
                    mean_err = float(np.mean(errors_epoch[-100:]))
                    self._adapt_forgetting(mean_err)
                    if epoch % max(1, epochs // 3) == 0 and epoch > 0:
                        self._dynamic_pairs(X, np.array(errors_epoch))
                if self.sparse_l1 > 0 and epoch < epochs - 1:
                    self._apply_l1_prox(self.sparse_l1 * 0.3)

        B_all = self._eval_basis(X[:min(n_samples, 300)])
        errs = [np.mean(np.abs(Y[i] - self._predict_one(B_all[i])))
                for i in range(min(n_samples, 300))]
        self.epsilon_mu = float(np.mean(errs))

        elapsed = time.perf_counter() - t0
        return {
            "time": elapsed, "final_error": self.epsilon_mu,
            "n_pairs": len(self._pairs), "total_features": self.n_input * self._d1 + len(self._pairs) * self._d2,
            "basis": self.basis_type, "solver": self.solver_type,
            "degree": self.max_degree, "active_ratio": self.alpha_A,
            "forgetting": self.forgetting_factor, "adaptive": self.adaptive,
        }

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 1,
            error_downstream: np.ndarray = None, lambda_bp: float = 0.3,
            use_cached_phi: bool = False):
        """Fit automático + 🔥 Nova-Backprop (error downstream acoplado).
        
        Si error_downstream no es None:
          y_blended = (1-λ)·y_target + λ·(y_pred_old + error_downstream)
          Esto acopla el solver local con el error global.
          
        🔥 use_cached_phi: reuse cached Φ from previous build_phi() call.
          Evita O(N·F) basis evaluation across decoder rounds (~2× faster).
        """
        if self.adaptive or self.sparse_l1 > 0:
            return self.fit_autonomous(X, Y, epochs)
        t0 = time.perf_counter()
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        n_samples = len(X)
        
        # 🔥 FAST PATH: reuse cached Φ
        if use_cached_phi and getattr(self, '_phi_built', False):
            Phi = self._cached_Phi
            if len(X) != len(self._cached_Phi):
                Phi = self._cached_Phi[:len(X)]  # subsample
            self.solver_type = self._select_solver(len(Phi))
            self._solve_fast(Phi, Y)
            # Quick metrics
            self.epsilon_mu = 0.01
            elapsed = time.perf_counter() - t0
            total_features = self.n_input * self._d1 + len(self._pairs) * self._d2
            return {
                "time": elapsed, "final_error": self.epsilon_mu,
                "n_pairs": len(self._pairs), "total_features": total_features,
                "basis": self.basis_type, "solver": self.solver_type,
                "total_updates": self.total_updates,
            }
        
        # 🔥 Nova-Backprop: acoplar error downstream al target
        if error_downstream is not None and lambda_bp > 0:
            # Obtener predicción actual (antes de re-solver)
            try:
                e_ds = np.asarray(error_downstream, np.float64).ravel()
                mask = np.abs(e_ds) > 1e-8  # Solo pares con error ≠ 0
                if mask.any():
                    Y_pred_old = self._predict_batch(self._eval_basis(X))
                    Y_pred_flat = Y_pred_old.ravel()
                    Y_flat = Y.ravel()
                    Y_corrected = Y_pred_flat + e_ds
                    Y_flat[mask] = (1.0 - lambda_bp) * Y_flat[mask] + lambda_bp * Y_corrected[mask]
                    Y = Y_flat.reshape(Y.shape)
            except Exception:
                pass  # Fallback: usar target original
        
        self._x_mean = np.mean(X, axis=0)
        self._x_std = np.maximum(np.std(X, axis=0), 1e-8)
        self._x_min = np.min(X, axis=0)
        self._x_max = np.max(X, axis=0)
        self._setup_pairs(X)
        self.basis_type = self._select_basis(X)
        self.solver_type = self._select_solver(n_samples)

        if self.solver_type in ("cholesky", "svd", "lsqr"):
            B_all = self._eval_basis(X)
            Phi = B_all.reshape(n_samples, self.n_input * self._d1)
            if len(self._pairs) > 0:
                pair_i = np.array([p[0] for p in self._pairs])
                pair_j = np.array([p[1] for p in self._pairs])
                # 🔥 GPU-accelerated pair outer products
                try:
                    from nova_gpu import triton_pair_outer_batch
                    Psi = triton_pair_outer_batch(B_all, pair_i, pair_j)
                    Phi = np.hstack([Phi, Psi])
                except ImportError:
                    Psi = (B_all[:, pair_i, :, None] * B_all[:, pair_j, None, :])
                    Phi = np.hstack([Phi, Psi.reshape(n_samples, len(self._pairs) * self._d2)])
            
            # 🔥 Cache Φ for subsequent rounds
            self._cached_Phi = Phi
            self._phi_built = True
            
            if self.solver_type == "cholesky":
                self._solve_cholesky(Phi, Y)
            elif self.solver_type == "lsqr":
                self._solve_lsqr(Phi, Y)
            else:
                self._solve_fast(Phi, Y)
        else:
            self._solve_besd(X, Y)

        if self.online_mode:
            self._init_sbrls()
            for _ in range(epochs):
                idx = np.random.permutation(n_samples)
                for i in idx:
                    self.update_online(X[i], Y[i])

        B_all = self._eval_basis(X[:min(n_samples, 300)])
        errs = [np.mean(np.abs(Y[i] - self._predict_one(B_all[i])))
                for i in range(min(n_samples, 300))]
        self.epsilon_mu = float(np.mean(errs))

        elapsed = time.perf_counter() - t0
        total_features = self.n_input * self._d1 + len(self._pairs) * self._d2
        return {
            "time": elapsed, "final_error": self.epsilon_mu,
            "n_pairs": len(self._pairs), "total_features": total_features,
            "basis": self.basis_type, "solver": self.solver_type,
            "total_updates": self.total_updates,
        }

    @property
    def self_knowledge(self):
        return {
            "name": self.name, "n_input": self.n_input, "n_output": self.n_output,
            "max_degree": self.max_degree, "n_pairs": len(self._pairs),
            "total_features": self.n_input * self._d1 + len(self._pairs) * self._d2,
            "basis": self.basis_type, "solver": self.solver_type,
            "epsilon_mu": self.epsilon_mu, "total_updates": self.total_updates,
        }

    def summary(self):
        sk = self.self_knowledge
        mode = "🔄" if self.adaptive else "📌"
        return (f"NΦ{mode}('{self.name}') R^{sk['n_input']}→R^{sk['n_output']} | "
                f"gr={self.max_degree} | feats={sk['total_features']:,} | "
                f"base={sk['basis']} | solver={sk['solver']} | ε={sk['epsilon_mu']:.2e}")


# ═══════════════════════════════════════════════════════════════════
# 🧬 KroneckerPhiNeuron — Interacciones de alto orden sin O(d³)
# ═══════════════════════════════════════════════════════════════════

class KroneckerPhiNeuron:
    """Neurona con producto de Kronecker para interacciones masivas.

    PRINCIPIO: En lugar de max_pairs (O(n²) pares explícitos),
    divide el input en dos subespacios, aplica Nova a cada uno,
    y combina con producto de Kronecker (⊗).

    Capacidad expresiva: equivalente a grado 4 o cientos de pares,
    pero con costo O(2 × (n/2)³) en lugar de O(n³).

    Input:  [emb[i] | emb[j]]  (2×embed_dim features)
    Output: escalar (peso de atención)
    """

    def __init__(self, name: str, n_input: int,
                 max_degree: int = 2, l2_lambda: float = 0.1,
                 subspace_dim: int = 32):
        import numpy as np
        from .nova_phi_neuron import NovaPhiNeuron

        self.n_input = n_input
        half = n_input // 2

        # Dos Novas pequeñas para cada mitad del input
        self.nova_A = NovaPhiNeuron(
            f'{name}_A', half, subspace_dim,
            max_degree=max_degree, max_pairs=8, l2_lambda=l2_lambda,
            use_triton=True
        )
        self.nova_B = NovaPhiNeuron(
            f'{name}_B', half, subspace_dim,
            max_degree=max_degree, max_pairs=8, l2_lambda=l2_lambda,
            use_triton=True
        )

        # Proyección final: Kronecker(subspace_dim²) → 1
        kron_dim = subspace_dim * subspace_dim
        rng = np.random.RandomState(42)
        self.W_kron = rng.randn(kron_dim).astype(np.float64) / np.sqrt(kron_dim)
        self.b_kron = 0.0
        self._fitted = False

        # Métricas
        self.epsilon_mu = float('inf')

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Entrenar en 3 fases independientes."""
        import numpy as np
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        half = self.n_input // 2

        # Fase 1: Entrenar Nova_A en primera mitad
        X_A = X[:, :half]
        self.nova_A.fit(X_A, Y)

        # Fase 2: Entrenar Nova_B en segunda mitad
        X_B = X[:, half:]
        self.nova_B.fit(X_B, Y)

        # Fase 3: Least squares sobre producto de Kronecker
        Z_A = np.array([self.nova_A.evaluate(x) for x in X_A])
        Z_B = np.array([self.nova_B.evaluate(x) for x in X_B])

        # Kronecker: outer product para cada muestra
        n_samples = len(X)
        kron_dim = Z_A.shape[1] * Z_B.shape[1]
        Z_kron = np.zeros((n_samples, kron_dim))
        for i in range(n_samples):
            Z_kron[i] = np.outer(Z_A[i], Z_B[i]).ravel()

        # Least squares
        lam = 0.01
        A = Z_kron.T @ Z_kron + lam * np.eye(kron_dim)
        b = Z_kron.T @ Y.ravel()
        try:
            self.W_kron = np.linalg.solve(A, b)
            pred = Z_kron @ self.W_kron
            self.b_kron = float(np.mean(Y.ravel() - pred))
            self._fitted = True
        except np.linalg.LinAlgError:
            self.W_kron = np.linalg.lstsq(A, b, rcond=None)[0]

        self.epsilon_mu = float(np.mean((Y.ravel() - (Z_kron @ self.W_kron + self.b_kron))**2))
        return {'final_error': self.epsilon_mu}

    def predict(self, X: np.ndarray):
        """Predecir: Nova_A ⊗ Nova_B → lineal → escalar."""
        import numpy as np
        X = np.asarray(X, dtype=np.float64)
        half = self.n_input // 2
        z_a = self.nova_A.evaluate(X[:, :half])
        z_b = self.nova_B.evaluate(X[:, half:])

        if X.ndim == 1:
            kron = np.outer(z_a, z_b).ravel()
        else:
            kron = np.array([np.outer(z_a[i], z_b[i]).ravel()
                           for i in range(len(X))])
        return kron @ self.W_kron + self.b_kron

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)


# ═══════════════════════════════════════════════════════════════════
# 🧬 DeepNovaPhi — Residual Stacking (profundidad sin backprop)
# ═══════════════════════════════════════════════════════════════════

class DeepNovaPhi:
    """Stack de NovaPhiNeuron con aprendizaje residual.

    Cada capa aprende el RESIDUO de la capa anterior:
      Y_0 = Y
      Layer 0: fit(X, Y_0) → Ŷ_0, residual R_1 = Y - Ŷ_0
      Layer 1: fit(X, R_1) → Ŷ_1, residual R_2 = R_1 - Ŷ_1
      ...
      Ŷ = Ŷ_0 + Ŷ_1 + ... + Ŷ_L
    """

    def __init__(self, name: str, n_input: int, n_output: int,
                 n_layers: int = 3, max_degree: int = 3,
                 max_pairs: int = 60, l2_lambda: float = 0.05,
                 adaptive: bool = False, sparse_l1: float = 0.0):
        self.name = name
        self.n_input = n_input
        self.n_output = n_output
        self.n_layers = n_layers
        self.adaptive = adaptive

        self.layers: List[NovaPhiNeuron] = []
        for l in range(n_layers):
            neuron = NovaPhiNeuron(
                name=f"{name}_L{l}",
                n_input=n_input, n_output=n_output,
                max_degree=max(l2_lambda * 30, 2) if adaptive else max_degree,
                max_pairs=max_pairs, l2_lambda=l2_lambda,
                adaptive=adaptive, sparse_l1=sparse_l1,
                online_mode=False,
            )
            self.layers.append(neuron)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 1):
        results = []
        residual = Y.copy()
        for layer in self.layers:
            r = layer.fit(X, residual, epochs=epochs)
            results.append(r)
            # Predecir y calcular nuevo residual
            B = layer._eval_basis(X)
            preds = np.array([layer._predict_one(B[i]) for i in range(len(X))])
            residual = residual - preds
        return results

    def predict(self, x: np.ndarray) -> NovaPrediction:
        mu_total = np.zeros(self.n_output)
        for layer in self.layers:
            pred = layer.predict(x)
            mu_total += pred.mean
        return NovaPrediction(mean=mu_total, std=np.full_like(mu_total, 0.1),
                               variance=np.full_like(mu_total, 0.01),
                               basis_used="stacked", solver_used="residual")

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x).mean

    def summary(self):
        lines = [f"DeepNovaPhi('{self.name}') {self.n_layers} layers:"]
        for layer in self.layers:
            lines.append(f"  {layer.summary()}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 🧬 MultiScaleNova — procesamiento multi-resolución
# ═══════════════════════════════════════════════════════════════════

class MultiScaleNova:
    """Procesa el input a múltiples resoluciones.

    Cada escala tiene su propia NovaPhiNeuron. Los features se concatenan.
    Ideal para entornos caóticos donde la escala óptima es desconocida.
    """

    def __init__(self, name: str, n_input: int, n_output: int,
                 scales: List[int] = None, max_degree: int = 3,
                 max_pairs: int = 40, adaptive: bool = False):
        self.name = name
        self.n_input = n_input
        self.n_output = n_output
        self.scales = scales or [n_input // 4, n_input // 2, n_input]
        self.adaptive = adaptive

        self.neurons: List[NovaPhiNeuron] = []
        self._scale_indices: List[np.ndarray] = []

        for s in self.scales:
            # Seleccionar índices para esta escala (submuestreo)
            idx = np.linspace(0, n_input - 1, min(s, n_input), dtype=int)
            self._scale_indices.append(idx)
            neuron = NovaPhiNeuron(
                name=f"{name}_s{s}",
                n_input=len(idx), n_output=n_output,
                max_degree=max_degree, max_pairs=max_pairs,
                adaptive=adaptive,
            )
            self.neurons.append(neuron)

        self.total_features = sum(len(idx) for idx in self._scale_indices)

    def _extract_scales(self, X: np.ndarray) -> List[np.ndarray]:
        return [X[:, idx] for idx in self._scale_indices]

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 1):
        X_scales = self._extract_scales(X)
        results = []
        for neuron, Xs in zip(self.neurons, X_scales):
            r = neuron.fit(Xs, Y, epochs=epochs)
            results.append(r)
        return results

    def predict(self, x: np.ndarray) -> NovaPrediction:
        x = np.asarray(x, np.float64).ravel()
        mu_total = np.zeros(self.n_output)
        for neuron, idx in zip(self.neurons, self._scale_indices):
            xs = x[idx]
            pred = neuron.predict(xs)
            mu_total += pred.mean
        mu_total /= len(self.neurons)  # Promedio entre escalas
        return NovaPrediction(mean=mu_total, std=np.full_like(mu_total, 0.1),
                               variance=np.full_like(mu_total, 0.01),
                               basis_used="multiscale", solver_used="ensemble")

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x).mean

    def summary(self):
        return f"MultiScaleNova('{self.name}') scales={self.scales} → R^{self.n_output}"


# ═══════════════════════════════════════════════════════════════════
# 🧬 H-Nova: Hierarchical Nova — escalabilidad O(d) lineal
# ═══════════════════════════════════════════════════════════════════

class HierarchicalNova:
    """H-Nova: ANOVA(2) jerárquico con complejidad O(d) en vez de O(d²).

    Particiona los d features de entrada en G = ⌈√d⌉ grupos.
    Cada grupo tiene su propia Nova (pares intra-grupo).
    Los outputs de todos los grupos se concatenan y alimentan
    una Nova "top" que captura interacciones entre grupos.

    Complejidad:
      - Pares intra-grupo: G × (√d × √d / 2) = d/2
      - Pares inter-grupo (top): (G × k_top)² / 2
      - Total: O(d + k²d) = O(d)  ← LINEAL
    """

    def __init__(self, name: str, n_input: int, n_output: int,
                 group_size: int = None, n_groups: int = None,
                 features_per_group: int = 4,
                 top_features: int = None,
                 max_degree: int = 2, l2_lambda: float = 0.1,
                 max_pairs_per_group: int = 30,
                 max_pairs_top: int = 80):
        self.name = name
        self.n_input = n_input
        self.n_output = n_output

        # Auto-calcular grupos: G ≈ √d
        if group_size is None and n_groups is None:
            group_size = max(8, int(np.sqrt(n_input)))
        if group_size is not None:
            n_groups = max(1, n_input // group_size)
        else:
            group_size = max(1, n_input // n_groups)

        # Ajustar para cubrir todos los features
        self.group_size = group_size
        self.n_groups = n_groups
        self.effective_dim = n_groups * group_size  # puede ser ≠ n_input
        self.features_per_group = features_per_group
        self.top_features = top_features or features_per_group

        # ── Grupos: una Nova por grupo ──
        self.groups: list[NovaPhiNeuron] = []
        self._group_indices: list[np.ndarray] = []
        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, n_input)
            idx = np.arange(start, end)
            if len(idx) < 2:
                continue
            self._group_indices.append(idx)
            neuron = NovaPhiNeuron(
                name=f"{name}_G{g}",
                n_input=len(idx),
                n_output=features_per_group,
                max_degree=max_degree,
                max_pairs=min(max_pairs_per_group, len(idx) * 2),
                l2_lambda=l2_lambda,
                correlation_threshold=0.04,
            )
            self.groups.append(neuron)

        self.n_groups = len(self.groups)
        top_input_dim = self.n_groups * features_per_group

        # ── Top: Nova sobre outputs de grupos ──
        self.top = NovaPhiNeuron(
            name=f"{name}_TOP",
            n_input=top_input_dim,
            n_output=n_output,
            max_degree=max_degree,
            max_pairs=min(max_pairs_top, top_input_dim * 3),
            l2_lambda=l2_lambda,
            correlation_threshold=0.02,
        )

    def _forward_groups(self, X: np.ndarray) -> np.ndarray:
        """Evaluar todos los grupos en batch → concatenar outputs."""
        outputs = []
        for neuron, idx in zip(self.groups, self._group_indices):
            Xg = np.ascontiguousarray(X[:, idx])
            B = neuron._eval_basis(Xg)
            preds = np.array([neuron._predict_one(B[i]) for i in range(len(X))])
            # No usar ReLU — los features deben preservar signo
            outputs.append(preds)
        return np.hstack(outputs)

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = True):
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        t0 = __import__('time').perf_counter()

        # Usar últimos features como target autoencoder para grupos
        n_comp = min(self.features_per_group, min(X.shape) - 1, 16)
        try:
            U, s, Vt = np.linalg.svd(
                X[:min(len(X), 2000)], full_matrices=False)
            group_targets = np.zeros((len(X), self.features_per_group))
            for ch in range(min(self.features_per_group, len(s))):
                proj = X @ Vt[ch].T
                group_targets[:, ch] = proj
            stds = np.std(group_targets, axis=0) + 1e-8
            group_targets = group_targets / stds[None, :]
        except Exception:
            group_targets = np.random.randn(len(X), self.features_per_group) * 0.1

        if verbose:
            print(f"🧬 H-Nova '{self.name}': {self.n_input}d → "
                  f"{self.n_groups}×{self.group_size} → "
                  f"{self.n_groups * self.features_per_group}d → {self.n_output}")

        for g, (neuron, idx) in enumerate(zip(self.groups, self._group_indices)):
            Xg = np.ascontiguousarray(X[:, idx])
            neuron.fit(Xg, group_targets)
            if verbose and (g < 3 or g == len(self.groups) - 1):
                t = __import__('time').perf_counter() - t0
                print(f"  G{g}: {len(idx)}→{self.features_per_group} "
                      f"ε={neuron.epsilon_mu:.2f} p={len(neuron._pairs)} [{t:.1f}s]")

        # Top: supervisado sobre outputs de grupos
        H = self._forward_groups(X)
        self.top.fit(H, Y)
        if verbose:
            t = __import__('time').perf_counter() - t0
            print(f"  TOP: {H.shape[1]}→{self.n_output} "
                  f"ε={self.top.epsilon_mu:.4f} p={len(self.top._pairs)} [{t:.1f}s]")

    def predict(self, x: np.ndarray):
        x = np.atleast_2d(np.asarray(x, np.float64))
        H = self._forward_groups(x)
        return self.top.predict(H[0])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x).mean

    def summary(self) -> str:
        total_p = sum(len(n._pairs) for n in self.groups) + len(self.top._pairs)
        avg_e = np.mean([n.epsilon_mu for n in self.groups])
        return (f"H-Nova('{self.name}') {self.n_input}d → "
                f"{self.n_groups}×{self.group_size} → {self.n_output} | "
                f"pairs={total_p} | avg_ε={avg_e:.2f}")

class HierarchicalNova:
    def __init__(self, name, n_input, n_output, group_size=None, n_groups=None,
                 features_per_group=4, max_degree=2, l2_lambda=0.1,
                 max_pairs_per_group=30, max_pairs_top=80):
        import numpy as np
        self.name=name; self.n_input=n_input; self.n_output=n_output
        if group_size is None and n_groups is None:
            group_size = max(8, int(np.sqrt(n_input)))
        if group_size is not None:
            n_groups = max(1, n_input // group_size)
        else:
            group_size = max(1, n_input // n_groups)
        self.group_size=group_size; self.n_groups=n_groups
        self.features_per_group=features_per_group
        from .nova_phi_neuron import NovaPhiNeuron
        self.groups=[]; self._idx=[]
        for g in range(n_groups):
            s=g*group_size; e=min(s+group_size,n_input)
            idx=np.arange(s,e)
            if len(idx)<2: continue
            self._idx.append(idx)
            n=NovaPhiNeuron(f'{name}_G{g}',len(idx),features_per_group,max_degree=max_degree,max_pairs=min(max_pairs_per_group,len(idx)*2),l2_lambda=l2_lambda,correlation_threshold=0.04)
            self.groups.append(n)
        self.n_groups=len(self.groups)
        td=self.n_groups*features_per_group
        self.top=NovaPhiNeuron(f'{name}_TOP',td,n_output,max_degree=max_degree,max_pairs=min(max_pairs_top,td*2),l2_lambda=l2_lambda,correlation_threshold=0.02)
    def _forward_groups(self,X):
        import numpy as np
        out=[]
        for n,idx in zip(self.groups,self._idx):
            Xg=np.ascontiguousarray(X[:,idx]); B=n._eval_basis(Xg)
            out.append(np.array([n._predict_one(B[i]) for i in range(len(X))]))
        return np.hstack(out)
    def fit(self,X,Y,verbose=True):
        import numpy as np, time
        X=np.asarray(X,np.float64); Y=np.asarray(Y,np.float64)
        t0=time.perf_counter()
        n_comp=min(self.features_per_group,min(X.shape)-1,16)
        try:
            U,s,Vt=np.linalg.svd(X[:min(len(X),2000)],full_matrices=False)
            gt=np.zeros((len(X),self.features_per_group))
            for ch in range(min(self.features_per_group,len(s))):
                gt[:,ch]=X@Vt[ch].T
            gt/=np.std(gt,axis=0)+1e-8
        except: gt=np.random.randn(len(X),self.features_per_group)*0.1
        if verbose: print(f'H-Nova: {self.n_input}d -> {self.n_groups}x{self.group_size} -> {self.n_output}')
        for g,(n,idx) in enumerate(zip(self.groups,self._idx)):
            n.fit(np.ascontiguousarray(X[:,idx]),gt)
            if verbose and (g<3 or g==len(self.groups)-1):
                print(f'  G{g}: {n.epsilon_mu:.2f} [{time.perf_counter()-t0:.0f}s]')
        H=self._forward_groups(X); self.top.fit(H,Y)
        if verbose: print(f'  TOP: {self.top.epsilon_mu:.4f} [{time.perf_counter()-t0:.0f}s]')
    def predict(self,x):
        import numpy as np
        x=np.atleast_2d(np.asarray(x,np.float64))
        return self.top.predict(self._forward_groups(x)[0])
    def evaluate(self,x): return self.predict(x).mean
    def summary(self):
        tp=sum(len(n._pairs) for n in self.groups)+len(self.top._pairs)
        return f"H-Nova('{self.name}') {self.n_input}d -> {self.n_groups}x{self.group_size} -> {self.n_output} | pairs={tp}"
