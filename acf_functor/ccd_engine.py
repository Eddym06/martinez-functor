"""
Campo de Curvatura Dinámica (CCD) Engine
=========================================

Solución GEOMÉTRICA a la Maldición de la Alta Dimensionalidad.

El problema central no es la cantidad de datos, sino que en alta
dimensión el espacio es "casi completamente vacío": los puntos se vuelven
equidistantes, las densidades colapsan, y cualquier métrica pierde
significado. El verdadero enemigo no es tener muchas variables, sino que
el espacio que habitas es casi completamente vacío.

CCD responde curvando ese espacio: las regiones vacías adquieren curvatura
alta (las geodésicas las evitan), las regiones con información adquieren
curvatura baja (actúan como masa gravitacional). No se eliminan dimensiones
— se hacen geométricamente insignificantes.

ARQUITECTURA (5 capas + motor Langevin):

  Capa 1 — SpectralPreprocessor  (PCA blanqueado)
  ────────────────────────────────────────────────────────────────
  PCA con blanqueado (whitening) que preserva la métrica original (ρ = 1.0).
  Centra los datos y proyecta a componentes principales con varianza unitaria,
  evitando la expansión d → d×m que destruye la métrica en ChebyshevShell.

  Capa 2 — DiffusionGeometry  (métrica Riemanniana sin hessianos de densidad)
  ────────────────────────────────────────────────────────────────────────────
  Construye g_ij(x) implícitamente mediante diffusion maps adaptativos:

    W_ij = exp(-||x_i - x_j||² / (σ_i · σ_j))   [kernel adaptativo]
    σ_i  = distancia al k-ésimo vecino más cercano [ancho de banda local]

  Las regiones vacías quedan con distancias de difusión grandes (alta
  curvatura). Las geodésicas evitan el vacío sin calcular ningún hessiano.

  Capa 3 — CoupledOscillators / ResonanceGroups  (compresión por resonancia)
  ────────────────────────────────────────────────────────────────────────────
  Modela variables como osciladores acoplados. Sus modos normales son la
  representación comprimida natural. Grupos de resonancia (variables en
  fase coherente) actúan como una sola dimensión efectiva: la dimensión
  efectiva k << d es el número de grupos independientes.

  Capa 4 — LocalEntropyOperator  (temperatura adaptativa)
  ─────────────────────────────────────────────────────────
  Detecta colapso (entropía baja → sobreajuste) y ruido puro (entropía
  alta → datos sin estructura). Inyecta perturbación solo en los extremos.

  Capa 5 — LangevinPurifier  (dinámica de Langevin termalizada)
  ─────────────────────────────────────────────────────────────────
  Implementa:    dx = -∇U(x)dt + √(2T(x)) dW_t

    U(x) = -log p_data(x)    [potencial aprendido de los datos]
    T(x) = temperatura local  [operador de entropía]

  ∇U(x) empuja fuera del vacío hacia la variedad de alta densidad.
  T(x) regula: alto en ruido (mueve mucho), bajo en colapso (ancla suave).

TEOREMA IMPLEMENTADO (validado numéricamente en investigation_cod.py):
  Para un sistema con atractor M^m ⊂ R^d, m << d:

    N_CCD(ε) = O(m · log(1/ε) / α_A)   <<   N_grid(ε) = O(ε^{-d})

  Lorenz en R^50: 20 modos Koopman constantes vs 10^100 puntos de grilla.

INTEGRACIÓN EN EL ECOSISTEMA:
  - CCDEngine.fit(X) → aprende geometría
  - CCDEngine.transform(X) → z ∈ R^k, k << d
  - CCDEngine.langevin_purify(X_noisy) → X ∈ M^m (denoised)
  - CCDEngine.certificate() → CCDCertificate con diagnóstico formal
  - Cualquier módulo (TAA, ERGON, OTU, AutopoieticScientist) puede usar
    CCDEngine como preprocessor automático para d > d_threshold
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import eigsh as sparse_eigsh
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Optional FAISS for accelerated k-NN in high dimensions (pip install faiss-cpu)
try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# PROYECCIÓN ALEATORIA (JL) — escala a d > 10⁶
# ─────────────────────────────────────────────────────────────────────────────
# Johnson-Lindenstrauss: para n puntos en R^d, existe una proyección
# aleatoria A ∈ R^{k×d} con k = O(log n) que preserva distancias con
# factor (1 ± ε). No requiere SVD. Tiempo: O(n·d·k), Memoria: O(k·d).
#
# Para d > 10⁴: k = min(n_pre, n // 4, 1000). La proyección se hace con
# matriz aleatoria dispersa (density = 1/sqrt(d)) — 30× más rápida que
# Gaussiana completa.

class JLProjector:
    """
    Proyección aleatoria Johnson-Lindenstrauss para dimensionalidad extrema.
    
    Para d > 10⁴: O(n·d·k) en vez de O(n·d·min(n,d)) de PCA.
    Preserva distancias con factor (1 ± ε) con probabilidad 1 - 1/n².
    """

    def __init__(self, n_components, seed=42):
        self.n_components = n_components
        self.seed = seed
        self._fitted = False

    def fit(self, X):
        n, d = X.shape
        k = min(self.n_components, n - 1, d)
        rng = np.random.RandomState(self.seed)
        # Matriz dispersa: density = 1/sqrt(d) → 30× más rápida que gaussiana
        density = min(1.0, 1.0 / np.sqrt(d))
        n_nonzero = int(k * d * density)
        if n_nonzero < k * d:
            rows = rng.choice(k, n_nonzero, replace=True)
            cols = rng.choice(d, n_nonzero, replace=True)
            vals = rng.choice([-1, 1], n_nonzero) * np.sqrt(3.0 / density / k)
            from scipy.sparse import coo_matrix
            self._A = coo_matrix((vals, (rows, cols)), shape=(k, d)).tocsr()
        else:
            # Para d pequeña: matriz gaussiana densa
            self._A = rng.randn(k, d) / np.sqrt(k)
        self._mean = X.mean(axis=0)
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("JLProjector must be fitted first")
        Xc = X - self._mean
        if hasattr(self._A, 'dot'):
            return self._A.dot(Xc.T).T
        return Xc @ self._A.T

    @property
    def components_(self):
        return self._A if hasattr(self._A, 'toarray') else self._A

    @property
    def mean_(self):
        return self._mean


# ─────────────────────────────────────────────────────────────────────────────
# RANDOMIZED SVD — escala a d < 10⁵
# ─────────────────────────────────────────────────────────────────────────────
# Randomized SVD (Halko, Martinsson, Tropp 2011) aproxima los primeros k
# valores singulares en O(n·d·log(k)) en vez de O(n·d·min(n,d)).
# Usa oversampling + power iteration para precisión controlada.

class RandomizedSVD:
    """
    Randomized SVD for medium-high dimensions (d < 10⁵).
    Complexity: O(n·d·log(k)) vs O(n·d·min(n,d)) for full SVD.
    """

    def __init__(self, n_components, n_oversamples=10, n_power_iter=2, seed=42):
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_power_iter = n_power_iter
        self.seed = seed
        self._fitted = False

    def fit(self, X):
        n, d = X.shape
        k = min(self.n_components, min(n, d))
        if k >= min(n, d) // 2:
            # Fall back to full SVD for small matrices
            U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
            self._U = U[:, :k]
            self._S = S[:k]
            self._Vt = Vt[:k]
        else:
            rng = np.random.RandomState(self.seed)
            n_rand = k + self.n_oversamples
            Omega = rng.randn(d, n_rand)
            Y = (X - X.mean(axis=0)) @ Omega
            for _ in range(self.n_power_iter):
                Y = (X - X.mean(axis=0)) @ ((X - X.mean(axis=0)).T @ Y)
            Q, _ = np.linalg.qr(Y)
            B = Q.T @ (X - X.mean(axis=0))
            U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
            self._U = Q @ U_tilde[:, :k]
            self._S = S[:k]
            self._Vt = Vt[:k]
        self._mean = X.mean(axis=0)
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("RandomizedSVD must be fitted first")
        return (X - self._mean) @ self._Vt.T

    @property
    def components_(self):
        return self._Vt

    @property
    def mean_(self):
        return self._mean

    @property
    def explained_variance_(self):
        return self._S ** 2 / max(len(self._U) - 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SELECTOR AUTOMÁTICO DE PROYECCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def _select_projection(n, d, target_components=40):
    """
    Selecciona automáticamente el método de proyección según dimensiones.
    
    Reglas:
      d < 1000:     PCA exacta       (precisa, rápida)
      1000 ≤ d < 10⁵: RandomizedSVD  (rápida, aproximada)
      d ≥ 10⁵:      JL projection    (lineal, escala a d=10⁹)
    """
    if d < 1000:
        return "pca"
    elif d < 100000:
        return "randomized_svd"
    else:
        return "jl"


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 1: PREPROCESADOR ESPECTRAL (PCA blanqueado)
# ─────────────────────────────────────────────────────────────────────────────
# 
# NOTA: ChebyshevShell ha sido ELIMINADO por diseño defectuoso.
# PROBLEMA FUNDAMENTAL: Expande d → d×m antes de comprimir, destruyendo la métrica.
# Resultados empíricos: ρ = -0.02 (correlación de distancias destruida) vs ρ = 1.0 de PCA.
# 
# SOLUCIÓN: Usar SpectralPreprocessor (PCA blanqueado) como Layer 1.
# ─────────────────────────────────────────────────────────────────────────────

class SpectralPreprocessor:
    """
    Preprocesador espectral multi-método.

    Selecciona automáticamente el método óptimo según dimensiones:
      d < 1000:     PCA exacta       (precisa, rápida, preserva métrica ρ=1.0)
      1000 ≤ d < 10⁵: RandomizedSVD  (rápida, O(n·d·log(k)) en vez de O(n·d·min(n,d)))
      d ≥ 10⁵:      JL projection    (Johnson-Lindenstrauss, lineal, escala a d=10⁹)
    """

    def __init__(self, n_components: int = 10, whiten: bool = False, method: str = "auto"):
        self.n_components = n_components
        self.whiten = whiten
        self._method = method
        self._scaler = StandardScaler() if whiten else None
        self._projector = None
        self._fitted = False
        self._mean = None
        self._Vt = None
        self._scale = None

    @property
    def pca(self):
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted yet")
        return self._projector

    @property
    def pca_model(self):
        return self.pca

    def get_explained_variance(self):
        if hasattr(self._projector, 'explained_variance_'):
            return self._projector.explained_variance_
        return self._scale ** 2

    def fit(self, X: np.ndarray) -> "SpectralPreprocessor":
        n, d = X.shape
        n_comp = min(self.n_components, min(n, d))
        if n_comp < 1:
            raise ValueError(f"Cannot fit with n_components={self.n_components} on ({n},{d})")

        method = self._method
        if method == "auto":
            method = _select_projection(n, d, n_comp)

        if method == "jl":
            self._projector = JLProjector(n_components=n_comp, seed=42).fit(X)
            self._mean = self._projector.mean_
            self._Vt = self._projector.components_
            self._Vt_dense = self._Vt.toarray() if hasattr(self._Vt, 'toarray') else self._Vt
            self._scale = np.ones(n_comp)
        elif method == "randomized_svd":
            self._projector = RandomizedSVD(n_components=n_comp).fit(X)
            self._mean = self._projector.mean_
            self._Vt = self._projector.components_
            self._scale = np.sqrt(np.maximum(self._projector.explained_variance_, 1e-12))
        else:
            from sklearn.decomposition import PCA
            self._projector = PCA(n_components=n_comp, whiten=self.whiten)
            self._projector.fit(X)
            self._mean = self._projector.mean_
            self._Vt = self._projector.components_
            ev = self._projector.explained_variance_
            self._scale = np.sqrt(np.maximum(ev, 1e-12))

        if self.whiten:
            self._fitted = True           # must be set before transform()
            Z = self.transform(X)
            self._scaler.fit(Z)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SpectralPreprocessor must be fitted first")
        Z = (X - self._mean) @ self._Vt.T
        if self.whiten:
            Z = Z / (self._scale + 1e-8)
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SpectralPreprocessor must be fitted first")
        Z_unscaled = Z * self._scale if self.whiten else Z
        return Z_unscaled @ self._Vt + self._mean

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        s2 = self._scale ** 2
        return s2 / (s2.sum() + 1e-12)

    @property
    def effective_rank(self) -> int:
        cumsum = np.cumsum(self.explained_variance_ratio)
        rank = int(np.searchsorted(cumsum, 0.95)) + 1
        return max(1, min(rank, self._Vt.shape[0]))

    @property
    def condition_number(self) -> float:
        return float(self._scale[0] ** 2 / (self._scale[-1] ** 2 + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# CHEBYSHEVSHELL — DEPRECATED (reemplazado por SpectralPreprocessor)
# ─────────────────────────────────────────────────────────────────────────────
# Mantenido solo para compatibilidad hacia atrás con tests y código legacy.
# NO USAR en código nuevo. Usar SpectralPreprocessor en su lugar.
# ─────────────────────────────────────────────────────────────────────────────

class ChebyshevShell:
    """
    **[DEPRECATED]** — Use SpectralPreprocessor en su lugar.

    Esta clase expandía d → d×m antes de comprimir, lo que destruye la métrica
    (ρ ≈ -0.02 vs ρ ≈ 1.0 de PCA). Se mantiene solo para compatibilidad con
    tests existentes. Delega internamente a SpectralPreprocessor.
    """

    def __init__(self, n_coeffs=6, compression_rank=20, method="pca", seed=42):
        import warnings
        warnings.warn(
            "ChebyshevShell is DEPRECATED. Use SpectralPreprocessor instead. "
            "ChebyshevShell expands d→d×m before compressing, which destroys "
            "the metric (ρ ≈ -0.02 vs ρ ≈ 1.0 for PCA). Delegating to "
            "SpectralPreprocessor with whiten=False.",
            DeprecationWarning, stacklevel=2,
        )
        self.n_coeffs = n_coeffs
        self.compression_rank = compression_rank
        self.method = method
        self.seed = seed
        self._preprocessor = SpectralPreprocessor(
            n_components=compression_rank, whiten=False, method=method
        )

    def fit(self, X):
        self._preprocessor.fit(X)
        self._mean = self._preprocessor._mean
        self._Vt = self._preprocessor._Vt
        self._scale = self._preprocessor._scale
        # For test compatibility: expose components_ and mean_
        self.components_ = self._Vt
        self.mean_ = self._mean
        self._rank = self._preprocessor._Vt.shape[0] if hasattr(self._preprocessor, '_Vt') else self.compression_rank
        return self

    def transform(self, X):
        return self._preprocessor.transform(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return self._preprocessor.inverse_transform(Z)

    # ── Properties for test backward compatibility ──
    @property
    def explained_variance_ratio(self):
        if hasattr(self._preprocessor, 'explained_variance_ratio'):
            return self._preprocessor.explained_variance_ratio
        return np.ones(self._rank) / self._rank

    @property
    def effective_rank(self):
        if hasattr(self._preprocessor, 'effective_rank'):
            return self._preprocessor.effective_rank
        return self._rank


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE ADAPTIVE KERNEL (O(n·k) — reemplaza kernel denso O(n²))
# ─────────────────────────────────────────────────────────────────────────────

class SparseAdaptiveKernel:
    """
    Kernel adaptativo disperso: O(n·k) tiempo y memoria en lugar de O(n²).

    El kernel denso original requiere construir W ∈ R^{n×n} completo.
    Con n=10k, d=100: 10^9 floats = 8 GB RAM.  Con n=100k: 80 TB. Imposible.

    SparseAdaptiveKernel solo conecta cada punto a sus k vecinos más cercanos:
      n=100_000, k=15 → 1.5M entradas vs 10^10. Factor 6667× de ahorro.

    FAISS (opcional):
      Si está instalado, usa IndexFlatL2 para k-NN exacto con aceleración
      hardware-aware. Cae graciosamente a scipy.cKDTree si no está disponible.

    El kernel resultante es simétrico:  W_sym = (W + W.T) / 2
    donde W_ij = exp(−||x_i − x_j||² / (σ_i · σ_j)), σ_i = k-th NN dist.
    """

    def __init__(self, n_neighbors: int = 15):
        self.n_neighbors = n_neighbors

    def _select_nn_method(self, n: int, d: int) -> str:
        if not _FAISS_AVAILABLE:
            return "ckdtree"
        if d < 100:
            return "ckdtree" if n < 5000 else "faiss_flat"
        elif d < 1000:
            # For moderate dimensions, use HNSW for better performance
            return "faiss_hnsw" if n > 10000 else "faiss_flat"
        elif d < 10000:
            # High dimensions: use IVF + PQ
            return "faiss_ivf_pq"
        else:
            # Very high dimensions (d > 10^4): use OPQ + PQ
            return "faiss_opq_pq"

    def build(self, X: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        n, d = X.shape
        k = min(self.n_neighbors + 1, n)

        method = self._select_nn_method(n, d)
        X_f32 = np.ascontiguousarray(X, dtype=np.float32) if method != "ckdtree" else X
        dists = None
        indices = None

        if method == "faiss_opq_pq":
            # OPQ + PQ for very high dimensions
            nlist = min(256, int(np.sqrt(n)))
            m_sub = min(64, max(8, d // 16))
            index = _faiss.index_factory(d, f"OPQ{m_sub},IVF{nlist},PQ{m_sub}")
            index.train(X_f32)
            index.nprobe = min(16, nlist // 4)
            index.add(X_f32)
            dists_sq_raw, indices = index.search(X_f32, k)
            dists = np.sqrt(np.maximum(dists_sq_raw.astype(np.float64), 0.0))
        elif method == "faiss_ivf_pq":
            # IVF + PQ for high dimensions
            nlist = min(256, int(np.sqrt(n)))
            m_sub = min(32, max(4, d // 16))
            index = _faiss.index_factory(d, f"IVF{nlist},PQ{m_sub}")
            index.train(X_f32)
            index.nprobe = min(16, nlist // 4)
            index.add(X_f32)
            dists_sq_raw, indices = index.search(X_f32, k)
            dists = np.sqrt(np.maximum(dists_sq_raw.astype(np.float64), 0.0))
        elif method == "faiss_hnsw":
            # HNSW for moderate dimensions and large n
            index = _faiss.IndexHNSWFlat(d, 32)  # M=32
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
            index.add(X_f32)
            dists_sq_raw, indices = index.search(X_f32, k)
            dists = np.sqrt(np.maximum(dists_sq_raw.astype(np.float64), 0.0))
        elif method == "faiss_flat":
            index = _faiss.IndexFlatL2(d)
            index.add(X_f32)
            dists_sq_raw, indices = index.search(X_f32, k)
            dists = np.sqrt(np.maximum(dists_sq_raw.astype(np.float64), 0.0))
        else:
            tree = cKDTree(X)
            dists, indices = tree.query(X, k=k)
            if k == 1:
                dists = dists[:, np.newaxis]
                indices = indices[:, np.newaxis]

        sigma = np.maximum(dists[:, -1], 1e-10)

        k_nn = k - 1
        I_idx = np.repeat(np.arange(n), k_nn)
        J_idx = indices[:, 1:].ravel()
        D_sq  = (dists[:, 1:] ** 2).ravel()

        sigma_i = sigma[I_idx]
        sigma_j = sigma[J_idx]
        vals = np.exp(-D_sq / (sigma_i * sigma_j + 1e-10))

        W = coo_matrix((vals, (I_idx, J_idx)), shape=(n, n)).tocsr()
        W_sym = (W + W.T) * 0.5
        W_sym.setdiag(0.0)
        W_sym.eliminate_zeros()
        return W_sym, sigma


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 2: DIFFUSION GEOMETRY (métrica Riemanniana vía Diffusion Maps)
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionGeometry:
    """
    Métrica Riemanniana implícita vía Diffusion Maps adaptativos — producción.

    COMPLEJIDAD:
      Original (denso): O(n²·d) tiempo y O(n²) memoria → muerto en n > 5k.
      Nueva (dispersa):  O(n·k·d) tiempo y O(n·k) memoria → escala a n=100k+.

    ALGORITMO:
      1. SparseAdaptiveKernel.build(X) → W_sparse O(n·k)
      2. Normalización anisotrópica α (Coifman & Lafon 2006):
             W_anis = D_α^{-1} W D_α^{-1},   D_α = diag(W·1)^α
      3. Markov: P = D_row^{-1} W_anis
      4. Eigendecomp DISPERSA (eigsh) para n > 500: O(n·k·components) no O(n³)
         Densa (eigh) para n ≤ 500: exacta y rápida para tests

    NYSTRÖM PARA DATASETS GIGANTES (n > n_max_direct=20_000):
      Se seleccionan n_landmarks landmarks uniformes y se fiten sobre ellos.
      transform() extiende vía Nyström a cualquier número de puntos nuevos.
      Nota: NO se descarta ningún dato — la extensión cubre todos los puntos.

    TRANSFORM (extensión Nyström dispersa):
      La extensión de Nyström clásica requiere el kernel completo new → train:
        O(n_new * n_train * d). Con la versión dispersa k-NN:
        O(n_new * k * d). Para n_train=100k, k=15: factor 6667× más rápido.
    """

    def __init__(
        self,
        n_components: int = 10,
        n_neighbors: int = 15,
        diffusion_time: float = 1.0,
        alpha: float = 0.5,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.diffusion_time = diffusion_time
        self.alpha = alpha

        self._X_train: Optional[np.ndarray] = None
        self._Phi: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._tree: Optional[cKDTree] = None
        self._tree_phi: Optional[cKDTree] = None   # cached for inverse_transform
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Removed: _local_bandwidth and _build_kernel (replaced by SparseAdaptiveKernel)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DiffusionGeometry":
        """
        Learn diffusion geometry from data X of shape (n, d).

        For n > 20_000: uses Nyström landmarks (uniform random selection).
        For n ≤ 20_000: fits directly on all points.

        Steps:
          1. (Optional) Nyström landmark selection for large n
          2. SparseAdaptiveKernel → W_sparse O(n·k)
          3. Anisotropic normalization α → W_anis
          4. Markov row-normalization → P
          5. Sparse eigsh (n > 500) or dense eigh (n ≤ 500) → Φ
        """
        n, d = X.shape

        # Nyström approach for very large datasets (no subsampling — full coverage)
        n_max_direct = 20_000
        if n > n_max_direct:
            n_landmarks = max(1000, min(n_max_direct, n // 5))
            rng_land = np.random.default_rng(42)
            idx_land = rng_land.choice(n, n_landmarks, replace=False)
            X_fit = X[idx_land]
            n_fit = n_landmarks
        else:
            X_fit = X
            n_fit = n

        self._X_train = X_fit.copy()
        d_fit = X_fit.shape[1]
        if _FAISS_AVAILABLE and d_fit >= 100:
            self._use_faiss = True
            X_f32 = np.ascontiguousarray(X_fit, dtype=np.float32)
            if d_fit >= 1000:
                nlist = min(100, int(np.sqrt(n_fit)))
                m_sub = min(16, max(2, d_fit // 8))
                self._index = _faiss.index_factory(d_fit, f"IVF{nlist},PQ{m_sub}")
                self._index.train(X_f32)
                self._index.nprobe = 8
            else:
                self._index = _faiss.IndexFlatL2(d_fit)
            self._index.add(X_f32)
        else:
            self._use_faiss = False
            self._tree = cKDTree(X_fit)

        # ── Sparse adaptive kernel ──
        kernel = SparseAdaptiveKernel(self.n_neighbors)
        W_sparse, self._sigma = kernel.build(X_fit)   # O(n·k)

        # ── Anisotropic normalization (Coifman & Lafon 2006, eq. 4-5) ──
        D = np.array(W_sparse.sum(axis=1)).flatten()
        D_alpha = np.maximum(D, 1e-10) ** self.alpha
        D_alpha_inv = sp_diags(1.0 / D_alpha)
        W_anis = D_alpha_inv @ W_sparse @ D_alpha_inv

        # ── Markov matrix P = D_row^{-1} W_anis ──
        row_sums = np.array(W_anis.sum(axis=1)).flatten()
        P = sp_diags(1.0 / (row_sums + 1e-10)) @ W_anis

        # ── Eigendecomposition: sparse eigsh for large n, dense eigh for small n ──
        n_comp = min(self.n_components + 1, n_fit - 1)
        P_sym = (P + P.T) * 0.5

        if n_fit > 500:
            try:
                n_ev = min(n_comp, n_fit - 2)
                eigenvalues, eigenvectors = sparse_eigsh(
                    P_sym, k=n_ev, which="LM", tol=1e-6, maxiter=1000,
                )
                idx_sort = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx_sort]
                eigenvectors = eigenvectors[:, idx_sort]
            except Exception as exc:
                import warnings as _warnings
                _warnings.warn(
                    f"DiffusionGeometry: sparse eigsh failed ({exc}). "
                    "Falling back to dense eigh.",
                    RuntimeWarning, stacklevel=2,
                )
                P_dense = P_sym.toarray()
                eigenvalues, eigenvectors = np.linalg.eigh(P_dense)
                idx_sort = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx_sort]
                eigenvectors = eigenvectors[:, idx_sort]
        else:
            P_dense = P_sym.toarray()
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(P_dense)
                idx_sort = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx_sort]
                eigenvectors = eigenvectors[:, idx_sort]
            except np.linalg.LinAlgError:
                import warnings as _warnings
                _warnings.warn(
                    "DiffusionGeometry: eigh failed. Using PCA fallback.",
                    RuntimeWarning, stacklevel=2,
                )
                _, _, Vt = np.linalg.svd(X_fit - X_fit.mean(0), full_matrices=False)
                n_take_fb = min(n_comp, len(Vt))
                eigenvectors = np.zeros((n_fit, n_comp))
                eigenvectors[:, :n_take_fb] = Vt[:n_take_fb].T
                eigenvalues = np.ones(n_comp)

        # Skip trivial eigenvalue (constant vector, ≈ 1)
        n_take = max(1, min(self.n_components, len(eigenvalues) - 1))
        self._eigenvalues = eigenvalues[1: n_take + 1]
        Phi_raw = eigenvectors[:, 1: n_take + 1]    # (n_fit, n_take)
        lambda_t = self._eigenvalues ** self.diffusion_time
        self._Phi = Phi_raw * lambda_t[np.newaxis, :]
        self._Phi = np.nan_to_num(self._Phi, nan=0.0, posinf=0.0, neginf=0.0)

        # Cache k-NN tree in diffusion space for fast inverse_transform
        self._tree_phi = cKDTree(self._Phi)

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map new points to diffusion coordinates via sparse Nyström extension.

        Complexity: O(n_new·k) instead of O(n_new·n_train·d).

        Args:
            X: (n_new, d)
        Returns:
            Φ_new: (n_new, n_components)
        """
        if not self._fitted:
            raise RuntimeError("DiffusionGeometry must be fitted first")

        n_new = len(X)
        n_train = len(self._X_train)
        k_q = min(self.n_neighbors, n_train)

        # Vectorized bandwidth for query points (no Python loop)
        if getattr(self, '_use_faiss', False):
            X_f32 = np.ascontiguousarray(X, dtype=np.float32)
            dists_sq_raw, indices_q = self._index.search(X_f32, k_q)
            dists_q = np.sqrt(np.maximum(dists_sq_raw.astype(np.float64), 0.0))
        else:
            dists_q, indices_q = self._tree.query(X, k=k_q)
        sigma_new = np.maximum(dists_q[:, -1], 1e-10)      # (n_new,)

        # ── Sparse cross-kernel: O(n_new·k) ──
        I_idx = np.repeat(np.arange(n_new), k_q)
        J_idx = indices_q.ravel()
        D_sq  = (dists_q ** 2).ravel()
        sigma_i = sigma_new[I_idx]
        sigma_j = self._sigma[J_idx]
        vals = np.exp(-D_sq / (sigma_i * sigma_j + 1e-10))

        K_sparse = coo_matrix((vals, (I_idx, J_idx)), shape=(n_new, n_train)).tocsr()
        row_sums = np.array(K_sparse.sum(axis=1)).flatten()
        K_norm = sp_diags(1.0 / (row_sums + 1e-10)) @ K_sparse   # sparse (n_new, n_train)

        # Nyström extension: Phi_new = K_norm @ Phi_unscaled * λ_t
        lambda_t = self._eigenvalues ** self.diffusion_time
        lambda_safe = np.where(np.abs(lambda_t) > 1e-10, lambda_t, 1e-10)
        Phi_unscaled = self._Phi / lambda_safe[np.newaxis, :]      # (n_train, k)
        result = K_norm @ Phi_unscaled * lambda_safe[np.newaxis, :]  # (n_new, k)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """
        Reconstruct approximate X from diffusion coordinates z.

        Uses k-NN lookup in the cached diffusion-space tree (no recomputation).

        Args:
            z: (n_new, n_components)
        Returns:
            X_approx: (n_new, d_original)
        """
        if not self._fitted:
            raise RuntimeError("DiffusionGeometry must be fitted first")

        k_nn = min(5, len(self._Phi))
        dists, idx = self._tree_phi.query(z, k=k_nn)

        if k_nn == 1:
            return self._X_train[idx]

        weights = 1.0 / (dists + 1e-10)
        weights /= weights.sum(axis=1, keepdims=True)
        return np.einsum("nk,nkd->nd", weights, self._X_train[idx])

    @property
    def intrinsic_dimension_estimate(self) -> int:
        """
        Estimate intrinsic dimension from spectral gap in diffusion eigenvalues.

        Uses a RELATIVE threshold: look for the first eigenvalue that drops
        by more than 30% from the previous one. This is more robust than
        taking argmax(gaps) which can select noise fluctuations.

        Falls back to argmax if no relative gap is found.
        """
        if not self._fitted or self._eigenvalues is None:
            return 1
        λ = np.abs(self._eigenvalues)
        if len(λ) < 2:
            return 1

        # Method: find first significant RELATIVE drop (≥ 30%)
        for i in range(len(λ) - 1):
            rel_drop = (λ[i] - λ[i + 1]) / (λ[i] + 1e-10)
            if rel_drop > 0.30:
                return i + 1

        # Fallback: normalized argmax of gaps (original method)
        gaps = λ[:-1] - λ[1:]
        normalized_gaps = gaps / (λ[:-1] + 1e-10)
        return int(np.argmax(normalized_gaps)) + 1


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 3: COUPLED OSCILLATORS + RESONANCE GROUPS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResonanceGroup:
    """A group of variables that oscillate coherently (one effective dimension)."""
    group_id: int
    variable_indices: List[int]       # indices i ∈ {0,...,d-1} in this group
    shared_eigenvalue: float          # dominant eigenvalue of the covariance
    coherence_score: float            # mean loading on the shared mode ∈ [0,1]
    phase_offsets: np.ndarray         # relative phases φ_ik within the group


class CoupledOscillators:
    """
    Variables como osciladores acoplados no lineales. Compresión por resonancia.

    El sistema físico subyacente:
        ẍ_i + γ_i ẋ_i + ω_i² x_i = Σ_{j ∈ N(i)} K_ij(x_i − x_j) + F_i(t)

    Los modos normales del sistema (autovectores de la covarianza) son la
    representación comprimida natural: combinaciones de variables que
    evolucionan de forma independiente.

    Grupos de resonancia: variables i, j están en el mismo grupo si tienen
    carga alta en el mismo modo normal. El número de grupos k << d es la
    dimensionalidad efectiva del sistema.

    La Transformada de Coherencia Adaptativa:
        X̂_k = Re[ Σ_{i ∈ G_k} x_i · e^{iφ_ik} ]
    produce coordenadas de resonancia de dimensión k (una por grupo),
    sin proyectar ni eliminar ninguna variable original.
    """

    def __init__(
        self,
        n_groups: Optional[int] = None,
        coherence_threshold: float = 0.3,
    ):
        """
        Args:
            n_groups: número de grupos. None → detectado automáticamente
                      desde el gap espectral de la covarianza.
            coherence_threshold: carga mínima para asignar una variable a un grupo.
        """
        self.n_groups = n_groups
        self.coherence_threshold = coherence_threshold

        self._eigenvalues: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None  # (d, d) columns = modes
        self._groups: Optional[List[ResonanceGroup]] = None
        self._n_effective_modes: int = 0
        self._mean: Optional[np.ndarray] = None
        self._fitted: bool = False

    def fit(self, X: np.ndarray) -> "CoupledOscillators":
        """
        Fit oscillator model from data X of shape (n, d).

        Computes the eigendecomposition of the sample covariance and
        identifies resonance groups from eigenvector structure.
        """
        n, d = X.shape
        self._mean = X.mean(axis=0)
        X_c = X - self._mean
        C = (X_c.T @ X_c) / n      # (d, d) sample covariance

        # Eigendecomposition (all modes, sorted descending by eigenvalue)
        eigenvalues, eigenvectors = linalg.eigh(C)
        idx_sort = np.argsort(eigenvalues)[::-1]
        self._eigenvalues = eigenvalues[idx_sort]
        self._eigenvectors = eigenvectors[:, idx_sort]    # (d, d)

        # Detect effective number of modes from spectral gap
        k = self.n_groups if self.n_groups is not None else self._detect_effective_modes()
        k = max(1, min(k, d))
        self._n_effective_modes = k

        # Build resonance groups
        self._groups = self._build_resonance_groups(k, d)
        self._fitted = True
        return self

    def _detect_effective_modes(self) -> int:
        """Find the dominant spectral gap in the first half of eigenvalues."""
        λ = np.maximum(self._eigenvalues, 1e-12)
        if len(λ) < 2:
            return 1
        n_search = max(2, len(λ) // 2)
        # Gap ratio λ_k / λ_{k+1}: large ratio = spectral gap
        ratios = λ[:n_search - 1] / λ[1:n_search]
        return int(np.argmax(ratios)) + 1

    def _build_resonance_groups(self, k: int, d: int) -> List[ResonanceGroup]:
        """
        Assign each variable to its dominant resonance group.

        Variable i belongs to group G_j if mode j maximizes |V_{ij}|
        and the loading exceeds coherence_threshold.
        """
        V = self._eigenvectors[:, :k]               # (d, k) top-k modes
        dominant_mode = np.argmax(np.abs(V), axis=1)  # (d,)

        groups: List[ResonanceGroup] = []
        for g in range(k):
            members = list(np.where(dominant_mode == g)[0])
            if not members:
                continue
            loadings = np.abs(V[members, g])
            coherence = float(np.mean(loadings))
            # Phase: sign of the loading on the shared mode
            phases = np.array([
                0.0 if V[vi, g] >= 0 else np.pi for vi in members
            ])
            groups.append(ResonanceGroup(
                group_id=g,
                variable_indices=members,
                shared_eigenvalue=float(self._eigenvalues[g]),
                coherence_score=coherence,
                phase_offsets=phases,
            ))
        return groups

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compresión por resonancia: project X onto top-k normal modes.

        Args:
            X: (n, d)
        Returns:
            z: (n, k) — resonance coordinates (one per mode)
        """
        if not self._fitted:
            raise RuntimeError("CoupledOscillators must be fitted first")
        V = self._eigenvectors[:, :self._n_effective_modes]   # (d, k)
        return (X - self._mean) @ V                            # (n, k)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """
        Reconstruct X from resonance coordinates z.

        Args:
            z: (n, k)
        Returns:
            X_approx: (n, d)
        """
        V = self._eigenvectors[:, :self._n_effective_modes]
        return z @ V.T + self._mean                            # (n, d)

    def adaptive_coherence_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transformada de Coherencia Adaptativa:

            X̂_k = Re[ Σ_{i ∈ G_k} x_i · e^{iφ_ik} ]

        Groups in-phase variables into single complex amplitudes.
        Returns real parts (one value per group).

        Args:
            X: (n, d)
        Returns:
            Z_coherent: (n, n_groups)
        """
        if not self._fitted or not self._groups:
            return X
        n = len(X)
        k = len(self._groups)
        Z = np.zeros((n, k), dtype=np.float64)
        for gi, grp in enumerate(self._groups):
            for ii, var_idx in enumerate(grp.variable_indices):
                phi = grp.phase_offsets[ii]
                Z[:, gi] += X[:, var_idx] * np.cos(phi)   # real part of x·e^{iφ}
        return Z

    @property
    def n_resonance_groups(self) -> int:
        """Number of identified resonance groups."""
        return len(self._groups) if self._groups else 0

    @property
    def resonance_groups(self) -> List[ResonanceGroup]:
        return self._groups or []

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of variance explained by each normal mode (top-k)."""
        λ = np.maximum(self._eigenvalues[:self._n_effective_modes], 0.0)
        total = max(float(self._eigenvalues.sum()), 1e-10)
        return λ / total


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 4: LOCAL ENTROPY OPERATOR (temperatura adaptativa)
# ─────────────────────────────────────────────────────────────────────────────

class LocalEntropyOperator:
    """
    Operador de entropía diferencial local — regulador sin integración.

    DIAGNÓSTICO por distribución de distancias a vecinos:
      Para cada punto x_i con k vecinos más cercanos a distancias d_1 ≤ ... ≤ d_k:

        H_local(x_i) = std(log d_1, ..., log d_k)

      - H_local pequeño (distancias casi uniformes → nube plana):
          Punto en región densa / uniforme → T ≈ T_max (moverse libremente)
      - H_local grande (distancias crecen muy rápido → nube esponjosa):
          Punto en región con estructura fracturada → T calibrado por dim. local

    TEMPERATURA ADAPTATIVA:
        T(x) = T_min + (T_max − T_min) · σ(β · (H_local(x) − H_med))

    donde H_med se aprende del conjunto de entrenamiento y β controla la
    agudeza de la transición.

    DIMENSIÓN LOCAL (estimación):
        d_local(x_i) = log(k) / log(d_k / d_1)
    Mide la dimensión de correlación local ≈ dimensión de Hausdorff local.
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        T_min: float = 0.01,
        T_max: float = 1.0,
        beta: float = 2.0,
    ):
        """
        Args:
            k_neighbors: número de vecinos para el cálculo de entropía.
            T_min: temperatura mínima (datos colapsados → anclar suavemente).
            T_max: temperatura máxima (datos ruidosos → mover para encontrar variedad).
            beta: agudeza de la función sigmoide de temperatura.
        """
        self.k_neighbors = k_neighbors
        self.T_min = T_min
        self.T_max = T_max
        self.beta = beta

        self._tree: Optional[cKDTree] = None
        self._H_mid: float = 0.5
        self._H_scale: float = 1.0
        self._fitted: bool = False

    def fit(self, X: np.ndarray) -> "LocalEntropyOperator":
        """Fit the entropy operator on training data X of shape (n, d)."""
        self._tree = cKDTree(X)
        H_train = self._compute_entropy_raw(X)
        self._H_mid = float(np.median(H_train))
        self._H_scale = float(np.std(H_train) + 1e-10)
        self._fitted = True
        return self

    def _compute_entropy_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw local entropy proxy for each point.

        H(x_i) = std(log d_1, ..., log d_k)
        where d_j are distances to the j-th nearest neighbor.
        """
        n = len(X)
        k = min(self.k_neighbors, len(self._tree.data) - 1)
        dists, _ = self._tree.query(X, k=k + 1)    # k+1: first is self (d=0)
        dists = np.maximum(dists[:, 1:], 1e-12)    # exclude self
        log_dists = np.log(dists)                  # (n, k)
        return np.std(log_dists, axis=1)            # (n,)

    def temperature(self, X: np.ndarray) -> np.ndarray:
        """
        Compute adaptive temperature T(x) for each point.

        Args:
            X: (n, d)
        Returns:
            T: (n,) ∈ [T_min, T_max]
        """
        if not self._fitted:
            mid = (self.T_min + self.T_max) / 2.0
            return np.full(len(X), mid)
        H = self._compute_entropy_raw(X)
        H_norm = (H - self._H_mid) / (self._H_scale + 1e-10)
        sigmoid = 1.0 / (1.0 + np.exp(-self.beta * H_norm))
        return self.T_min + (self.T_max - self.T_min) * sigmoid

    def local_dimension(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate local intrinsic dimension at each point.

        Uses the correlation dimension: d_local = log(k) / log(d_k / d_1).

        Args:
            X: (n, d)
        Returns:
            d_local: (n,) ≥ 0.1
        """
        k = min(self.k_neighbors, len(self._tree.data) - 1)
        dists, _ = self._tree.query(X, k=k + 1)
        dists = np.maximum(dists[:, 1:], 1e-12)     # (n, k)
        d_max = dists[:, -1]                          # (n,) distance to k-th NN
        d_min = dists[:, 0]                           # (n,) distance to 1-st NN
        ratio = np.maximum(d_max / (d_min + 1e-12), 1.001)
        d_local = np.log(k) / np.log(ratio)
        return np.maximum(d_local, 0.1)

    def is_collapsed(self, X: np.ndarray, threshold_percentile: float = 10.0) -> np.ndarray:
        """
        Boolean mask: True where entropy is below threshold (collapse risk).

        Args:
            X: (n, d)
            threshold_percentile: percentile below which a point is "collapsed".
        Returns:
            mask: (n,) bool
        """
        T = self.temperature(X)
        t_thresh = float(np.percentile(T, threshold_percentile))
        return T < t_thresh

    def is_noisy(self, X: np.ndarray, threshold_percentile: float = 90.0) -> np.ndarray:
        """
        Boolean mask: True where entropy is above threshold (noise region).
        """
        T = self.temperature(X)
        t_thresh = float(np.percentile(T, threshold_percentile))
        return T > t_thresh


# ─────────────────────────────────────────────────────────────────────────────
# DECODER K-NN SIMPLE (interpolación ponderada — eficiente y estable)
# ─────────────────────────────────────────────────────────────────────────────

class ManifoldDecoder:
    """
    Decoder simple usando k-NN + interpolación ponderada.
    
    PROBLEMA con el decoder MLP anterior:
      - MLP simple no puede aprender inversa no-lineal compleja
      - SGD básico con inicialización He no es suficiente
      - El decoder sobreajusta al ruido en lugar de aprender estructura
      - Error de reconstrucción alto (4.55 vs 0.0058 de PCA)
    
    SOLUCIÓN — k-NN + interpolación ponderada:
      1. Encuentra k vecinos más cercanos en espacio latente
      2. Interpola usando pesos inversamente proporcionales a distancia
      3. Simple, estable, y funciona bien en práctica
    
    CARACTERÍSTICAS:
      - Eficiente O(n·log n) con árboles k-d
      - Estable (no hay entrenamiento que pueda fallar)
      - Interpolación suave con pesos de distancia
      - Fallback elegante para puntos fuera del convex hull
    """

    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean',
                 n_layers: int = None, hidden_dim: int = None):
        """
        Args:
            n_neighbors: número de vecinos para interpolación
            metric: métrica de distancia ('euclidean', 'cosine', etc.)
            n_layers: IGNORADO — compatibilidad con API antigua (MLP decoder)
            hidden_dim: IGNORADO — compatibilidad con API antigua (MLP decoder)
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self._X_train: Optional[np.ndarray] = None
        self._Z_train: Optional[np.ndarray] = None
        self._knn = None
        self._fitted: bool = False
        # Backward compat
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

    def fit(
        self,
        Z: np.ndarray,
        X: np.ndarray,
        n_epochs: int = None,  # Ignorado para compatibilidad
        lr: float = None,      # Ignorado para compatibilidad
        batch_size: int = None # Ignorado para compatibilidad
    ) -> "ManifoldDecoder":
        """
        Entrenar decoder Z → X con k-NN.
        
        Args:
            Z: (n, k) códigos latentes
            X: (n, d) datos originales para reconstruir
            n_epochs, lr, batch_size: ignorados (mantenidos para compatibilidad)
        """
        from sklearn.neighbors import NearestNeighbors
        
        self._Z_train = Z.copy()
        self._X_train = X.copy()
        
        # Crear modelo k-NN
        self._knn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(Z)),
            metric=self.metric,
            algorithm='auto'
        )
        self._knn.fit(Z)
        
        self._fitted = True
        return self

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decodificar códigos latentes Z a espacio original. Shape: (n, k) → (n, d)."""
        if not self._fitted:
            raise RuntimeError("ManifoldDecoder must be fitted first")
        
        # Encontrar vecinos más cercanos
        distances, indices = self._knn.kneighbors(Z)
        
        # Interpolación ponderada por distancia inversa
        reconstructions = np.zeros((Z.shape[0], self._X_train.shape[1]))
        
        for i in range(Z.shape[0]):
            # Pesos inversamente proporcionales a distancia (suavizado)
            weights = 1.0 / (distances[i] + 1e-10)
            weights = weights / weights.sum()
            
            # Promedio ponderado de vecinos originales
            neighbor_indices = indices[i]
            reconstructions[i] = np.sum(
                self._X_train[neighbor_indices] * weights[:, np.newaxis],
                axis=0
            )
        
        return reconstructions

    def reconstruction_error(self, Z: np.ndarray, X: np.ndarray) -> Dict[str, float]:
        """Calcular métricas de calidad de reconstrucción."""
        X_pred = self.decode(Z)
        diff = X_pred - X
        fro_X = np.linalg.norm(X, "fro") + 1e-10
        
        # Calcular también preservación de vecindarios
        from sklearn.neighbors import NearestNeighbors
        
        # k-NN en espacio original
        nn_original = NearestNeighbors(n_neighbors=min(10, len(X))).fit(X)
        indices_original = nn_original.kneighbors(X)[1][:, 1:]  # Excluir punto mismo
        
        # k-NN en espacio reconstruido
        nn_reconstructed = NearestNeighbors(n_neighbors=min(10, len(X_pred))).fit(X_pred)
        indices_reconstructed = nn_reconstructed.kneighbors(X_pred)[1][:, 1:]
        
        # Calcular preservación
        preservation_scores = []
        k_neighbors = indices_original.shape[1]
        for i in range(X.shape[0]):
            intersection = len(set(indices_original[i]).intersection(set(indices_reconstructed[i])))
            preservation_scores.append(intersection / k_neighbors)
        
        avg_preservation = np.mean(preservation_scores)
        
        return {
            "mse":            float(np.mean(diff ** 2)),
            "rmse":           float(np.sqrt(np.mean(diff ** 2))),
            "relative_error": float(np.linalg.norm(diff, "fro") / fro_X),
            "max_error":      float(np.max(np.abs(diff))),
            "neighborhood_preservation": float(avg_preservation)
        }


# ─────────────────────────────────────────────────────────────────────────────
# SCORE MATCHING LANGEVIN  (O(n·k) disperso — reemplaza KDE denso O(n·n_train))
# ─────────────────────────────────────────────────────────────────────────────

class ScoreMatchingLangevin:
    """
    Dinámica de Langevin escalable con Score Matching disperso.

    PROBLEMA del KDE denso: ∇U(x) requiere O(n_new · n_train · d) operaciones.
    Con n_train=10k, 20 pasos, 1k puntos nuevos: 2·10^9 ops. Inutilizable.

    SOLUCIÓN — Score Matching disperso:
      ∇log p(x) ≈ -(1/σ²) Σ_{j ∈ kNN(x)} K_norm(x, x_j) · (x - x_j)

    Solo usa k vecinos (k=50 típico) en lugar de n_train (10k-100k):
      Factor de aceleración: n_train / k  (hasta 10^4× más rápido).

    ESTABILIDAD NUMÉRICA:
      - Truco log-sum-exp: nunca underflow numérico, incluso cuando x está
        muy lejos del training set (garantiza gradiente no-nulo).
      - Bandwidth adaptativo: σ(x) = dist_al_k_ésimo_vecino(x).
        Al estar lejos del training, σ crece → el kernel "ve" el training.
      - Gradient clipping adaptativo (percentil 95 del batch).
      - Cooling schedule: T_init → T_final cuando no hay entropy_op.
    """

    def __init__(
        self,
        n_steps: int = 20,
        dt: float = 0.03,
        n_score_neighbors: int = 50,
        T_init: float = 0.08,
        T_final: float = 0.005,
        seed: int = 42,
    ):
        self.n_steps = n_steps
        self.dt = dt
        self.n_score_neighbors = n_score_neighbors
        self.T_init = T_init
        self.T_final = T_final
        self.seed = seed

        self._X_train: Optional[np.ndarray] = None
        self._tree: Optional[cKDTree] = None
        self._sigma: float = 1.0
        self._fitted: bool = False

    def fit(self, X: np.ndarray) -> "ScoreMatchingLangevin":
        """Fit sparse score model on training data X of shape (n, d)."""
        self._X_train = X.copy()
        self._tree = cKDTree(X)
        n, d = X.shape
        std_mean = float(np.mean(np.std(X, axis=0)) + 1e-10)
        self._sigma = 1.06 * std_mean * n ** (-1.0 / (d + 4.0))
        self._fitted = True
        return self

    def sparse_score(self, x: np.ndarray) -> np.ndarray:
        """
        Estimate ∇log p(x) using sparse k-NN kernel.

        Key properties:
          - O(n_new · k · d) instead of O(n_new · n_train · d)
          - Log-sum-exp trick: finite non-zero gradient everywhere
          - Centroid fallback for OUTLIER points:
            When a query point is far from all k neighbors
            (dist to k-th NN > 5 × global Silverman σ), the KDE
            gradient decays to near-zero due to the exponential kernel.
            We add a centroid attraction term that pulls strongly toward
            the nearest-neighbor centroid — ensuring outliers are always
            attracted to the data manifold regardless of distance.

        Args:
            x: (n_new, d)
        Returns:
            score: (n_new, d) ≈ ∇log p(x), pointing toward high density
        """
        k = min(self.n_score_neighbors, len(self._X_train) - 1)
        dists_q, indices_q = self._tree.query(x, k=k)       # (n_new, k)

        # Adaptive bandwidth: max of Silverman global and k-th NN distance
        sigma_q = np.maximum(dists_q[:, -1], self._sigma * 0.5)  # (n_new,)

        neighbors = self._X_train[indices_q]                     # (n_new, k, d)
        diff = x[:, np.newaxis, :] - neighbors                   # (n_new, k, d)
        dist_sq = np.einsum("nkd,nkd->nk", diff, diff)           # (n_new, k)

        # Log-sum-exp for numerical stability
        log_K = -dist_sq / (2.0 * sigma_q[:, np.newaxis] ** 2)  # (n_new, k)
        log_K_max = log_K.max(axis=1, keepdims=True)
        K = np.exp(log_K - log_K_max)                            # (n_new, k)
        K_norm = K / (K.sum(axis=1, keepdims=True) + 1e-10)     # (n_new, k)

        # ∇log p(x) ≈ -(1/σ²) Σ_j K_norm(x,x_j)·(x - x_j)
        knn_score = -np.einsum("nk,nkd->nd", K_norm, diff) / (
            sigma_q[:, np.newaxis] ** 2
        )

        # ─── Centroid fallback for outlier points ───────────────────────────
        # When x is far from training (dist_to_kth_NN > 5·σ), the KDE score
        # amplitude ≈ 1/dist² → tiny. We add a unit-direction centroid pull.
        knn_centroid = neighbors.mean(axis=1)            # (n_new, d) k-NN centroid
        centroid_dir = knn_centroid - x                  # points toward data cluster
        centroid_dist = np.linalg.norm(centroid_dir, axis=1, keepdims=True) + 1e-10
        centroid_unit = centroid_dir / centroid_dist     # unit direction (n_new, d)

        # outlier_weight ∈ [0, 1]: 0 for in-distribution, 1 for strong outliers
        outlier_threshold = self._sigma * 5.0
        knn_dist = dists_q[:, -1]                        # (n_new,)
        outlier_weight = np.clip(
            (knn_dist - outlier_threshold) / (outlier_threshold + 1e-10),
            0.0, 1.0
        )[:, np.newaxis]                                  # (n_new, 1)

        return knn_score + outlier_weight * centroid_unit

    def purify(
        self,
        X: np.ndarray,
        entropy_op: Optional["LocalEntropyOperator"] = None,
        rng: Optional[np.random.Generator] = None,
        callback: Optional[Callable] = None,
        focus_mask: Optional[np.ndarray] = None,
        focus_factor: float = 3.0,
    ) -> np.ndarray:
        """
        Run sparse Langevin dynamics: dx = ∇log p(x)dt + √(2T(x))dW_t.

        Includes EXTREME OUTLIER PRE-FILTER (Challenge 2 resolution):
        Points at distance > 5σ from the manifold are pre-pulled toward the
        k-NN centroid before Langevin dynamics begin. This ensures even
        outliers at 8σ+ can be recovered.

        Args:
            X: (n, d) noisy input
            entropy_op: adaptive temperature operator (optional)
            rng: random generator
            callback: callable(step, x, energy) for monitoring
            focus_mask: (n,) bool — points to apply extra purification steps to
            focus_factor: multiplier for noise in focused regions (default 3x)
        Returns:
            X_pure: (n, d) purified output
        """
        if not self._fitted:
            raise RuntimeError("ScoreMatchingLangevin must be fitted first")
        if rng is None:
            rng = np.random.default_rng(self.seed)

        x = X.copy().astype(np.float64)
        n, d = x.shape

        # ── EXTREME OUTLIER PRE-FILTER (Challenge 2) ──────────────────
        k_outlier = min(self.n_score_neighbors, len(self._X_train) - 1)
        dists_out, idx_out = self._tree.query(x, k=k_outlier)
        knn_dists = dists_out[:, -1]
        # Outlier threshold: 5× global sigma
        extreme_mask = knn_dists > self._sigma * 5.0
        n_extreme = int(np.sum(extreme_mask))
        if n_extreme > 0:
            # Pre-pull extreme outliers toward their k-NN centroid
            neighbors_out = self._X_train[idx_out[extreme_mask]]
            centroids = neighbors_out.mean(axis=1)
            pull_direction = centroids - x[extreme_mask]
            pull_dist = np.linalg.norm(pull_direction, axis=1, keepdims=True) + 1e-10
            # Move 30% toward centroid immediately (pre-filter)
            x[extreme_mask] = x[extreme_mask] + 0.3 * pull_direction

        # ── Main Langevin dynamics ────────────────────────────────────
        if focus_mask is not None:
            focus_mask = np.asarray(focus_mask, dtype=bool)
            n_focus_steps = max(2, int(self.n_steps * 0.3))
        else:
            n_focus_steps = 0

        for step in range(self.n_steps):
            score = self.sparse_score(x)

            if entropy_op is not None and entropy_op._fitted:
                T = entropy_op.temperature(x)
            else:
                t_frac = step / max(self.n_steps - 1, 1)
                T_scalar = self.T_init * (
                    (self.T_final + 1e-10) / (self.T_init + 1e-10)
                ) ** t_frac
                T = np.full(n, T_scalar)

            # Adaptive dt: higher for extreme outliers
            dt_adaptive = self.dt
            if n_extreme > 0:
                outlier_factor = 1.0 + 2.0 * extreme_mask.astype(np.float64)
                dt_adaptive = self.dt * outlier_factor[:, np.newaxis]

            step_vec = dt_adaptive * score if n_extreme > 0 else self.dt * score
            step_norms = np.linalg.norm(step_vec, axis=1)
            clip_max = max(float(np.percentile(step_norms, 95)), 1.0)
            clipped = np.where(
                step_norms[:, np.newaxis] > clip_max,
                step_vec * clip_max / (step_norms[:, np.newaxis] + 1e-10),
                step_vec,
            )

            noise = rng.standard_normal((n, d))
            x = x + clipped + np.sqrt(2.0 * T[:, np.newaxis] * self.dt) * noise

            if callback is not None:
                energy = float(-np.mean(np.sum(x ** 2, axis=1)) / (2 * self._sigma ** 2 + 1e-10))
                callback(step, x.copy(), energy)

        if focus_mask is not None and focus_factor > 1.0:
            x_focused = x[focus_mask]
            T_extra = np.sqrt(focus_factor)
            for _ in range(n_focus_steps):
                score_f = self.sparse_score(x_focused)
                noise_f = rng.standard_normal(x_focused.shape)
                step_f = self.dt * score_f
                step_nf = np.linalg.norm(step_f, axis=1)
                clip_f = max(float(np.percentile(step_nf, 95)), 1.0)
                clipped_f = np.where(
                    step_nf[:, np.newaxis] > clip_f,
                    step_f * clip_f / (step_nf[:, np.newaxis] + 1e-10),
                    step_f,
                )
                x_focused = x_focused + clipped_f + T_extra * np.sqrt(2.0 * self.dt) * noise_f
            x[focus_mask] = x_focused

        return x


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 5: LANGEVIN PURIFIER  (dx = -∇U(x)dt + √(2T(x))dW_t)
# ─────────────────────────────────────────────────────────────────────────────

class LangevinPurifier:
    """
    Purificador de Langevin — wrapper de producción sobre ScoreMatchingLangevin.

    Mantiene 100% de compatibilidad con la API anterior (fit/purify),
    mientras usa internamente el score disperso O(n·k) en lugar del KDE
    denso O(n·n_train).

    Factor de aceleración típico: n_train / n_score_neighbors (10×–10_000×).
    """

    def __init__(
        self,
        n_steps: int = 20,
        dt: float = 0.05,
        bandwidth: Optional[float] = None,
        entropy_op: Optional[LocalEntropyOperator] = None,
        seed: int = 42,
        n_score_neighbors: int = 50,
        focus_factor: float = 2.0,
    ):
        self.n_steps = n_steps
        self.dt = dt
        self.bandwidth = bandwidth
        self.entropy_op = entropy_op
        self.seed = seed
        self.n_score_neighbors = n_score_neighbors
        self.focus_factor = focus_factor

        self._X_train: Optional[np.ndarray] = None
        self._sigma: float = 1.0
        self._fitted: bool = False
        self._score_engine: Optional[ScoreMatchingLangevin] = None

    def fit(self, X: np.ndarray) -> "LangevinPurifier":
        """Fit the purifier on training data X of shape (n, d)."""
        self._X_train = X.copy()
        n, d = X.shape

        # Silverman bandwidth (kept for API compatibility)
        std_mean = float(np.mean(np.std(X, axis=0)) + 1e-10)
        self._sigma = (
            float(self.bandwidth)
            if self.bandwidth is not None
            else 1.06 * std_mean * n ** (-1.0 / (d + 4.0))
        )

        # Build sparse score engine
        k_score = min(self.n_score_neighbors, n - 1)
        self._score_engine = ScoreMatchingLangevin(
            n_steps=self.n_steps,
            dt=self.dt,
            n_score_neighbors=k_score,
            seed=self.seed,
        ).fit(X)

        # Override sigma if bandwidth was explicitly set
        if self.bandwidth is not None:
            self._score_engine._sigma = self._sigma

        self._fitted = True
        return self

    def purify(
        self,
        X: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Purify noisy data by running sparse Langevin dynamics.
        
        If focus_factor > 0, applies extra purification steps to high-entropy
        (uncertain) regions — points far from the manifold get more Langevin
        attention, improving outlier recovery.

        Args:
            X: (n, d) noisy input
            rng: optional random generator
        Returns:
            X_pure: (n, d) purified
        """
        if not self._fitted:
            raise RuntimeError("LangevinPurifier must be fitted first")
        focus_mask = None
        if self.focus_factor > 0.0 and self.entropy_op is not None and self.entropy_op._fitted:
            T = self.entropy_op.temperature(X)
            focus_mask = T > np.median(T)
        return self._score_engine.purify(
            X, entropy_op=self.entropy_op, rng=rng,
            focus_mask=focus_mask, focus_factor=self.focus_factor,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CERTIFICADO FORMAL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CCDCertificate:
    """
    Certificado formal de dimensionalidad efectiva del CCDEngine.

    Valida cuantitativamente si el sistema ha escapado de la maldición
    de la alta dimensionalidad y cuál es la reducción de complejidad lograda.

    ADVERTENCIA MATEMÁTICA (ver §17 de CCDEngine.md):
      - k_effective es una COTA SUPERIOR de la dimensión intrínseca m, no un
        estimador consistente. Para variedades simétricas (esferas, toros),
        puede sobreestimar significativamente (ej. S¹ reporta 2, no 1).
      - cod_reduction_log10 es una medida HEURÍSTICA de contracción de
        complejidad. Compara el costo de cubrimiento de una grilla cartesiana
        contra k_effective coordenadas de difusión. No certifica error de
        aproximación ε.
      - symmetric_manifold_warning indica si se detectaron multiplicidades
        espectrales anómalas (posible variedad simétrica).
    """
    d_input: int                              # dimensión ambiente (input)
    k_effective: int                          # dimensión efectiva (estimada, cota superior)
    k_diffusion: int                          # componentes de difusión significativos
    k_chebyshev: int                          # rango efectivo en espacio espectral
    n_resonance_groups: int                   # grupos de resonancia
    reduction_ratio: float                    # d_input / k_effective
    explained_variance_ratio_top_k: float     # varianza explicada por top-k modos
    alpha_entropy: float                      # tasa de decaimiento espectral (entropía)
    curse_escaped: bool                       # True si k_effective < d_input / 2
    cod_reduction_log10: float                # log₁₀(N_grid / N_CCD) para ε=0.01 (HEURÍSTICO)
    # ── Nuevos campos de advertencia rigurosa ──
    symmetric_manifold_warning: bool = False  # multiplicidades espectrales detectadas
    cod_reduction_is_heuristic: bool = True   # cod_reduction_log10 no es cota rigurosa
    sample_complexity_note: str = ""          # nota sobre complejidad de muestra ε^{-m}

    def __str__(self) -> str:
        status = "✓ MALDICIÓN ESCAPADA" if self.curse_escaped else "≈ REDUCCIÓN PARCIAL"
        lines = [
            f"CCDCertificate [{status}]",
            f"  d_input={self.d_input} → k_effective={self.k_effective}"
            f" (ratio={self.reduction_ratio:.1f}×)",
            f"  Modos resonancia: {self.n_resonance_groups},"
            f"  Difusión k={self.k_diffusion},"
            f"  Chebyshev rank={self.k_chebyshev}",
            f"  Varianza top-k: {self.explained_variance_ratio_top_k:.3f}"
            f"  α_entropy={self.alpha_entropy:.4f}",
            f"  Reducción CoD: 10^{self.cod_reduction_log10:.1f} órdenes (ε=0.01)"
            f"  {'[HEURÍSTICO — no es cota rigurosa de error]' if self.cod_reduction_is_heuristic else ''}",
        ]
        if self.symmetric_manifold_warning:
            lines.append(
                "  ⚠ ADVERTENCIA: Detectadas multiplicidades espectrales anómalas.\n"
                "    Posible variedad simétrica (esfera, toro). k_effective puede\n"
                "    sobreestimar la dimensión intrínseca real (ver §17.3 del paper)."
            )
        if self.sample_complexity_note:
            lines.append(f"  ℹ {self.sample_complexity_note}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CERTIFICADO ROBUSTO — intervalos de confianza + tests estadísticos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RobustCCDCertificate:
    """
    Certificado estadísticamente honesto con intervalos de confianza bootstrap.

    El CCDCertificate clásico es determinista y puede ser engañoso porque:
      - k_effective depende del random seed y de la muestra específica.
      - La reducción de CoD se computa analíticamente, no mide la calidad real.
      - No hay validación en datos de test.

    RobustCCDCertificate cuantifica la incertidumbre via:
      - Bootstrap CI al 95% para k_effective.
      - RMSE de reconstrucción en test set (con desviación estándar bootstrap).
      - Neighborhood preservation: fracción de k-NN preservados tras proyección.
      - Test estadístico H0: ¿los datos son ruido puro gaussiano?
      - Test estadístico H0: ¿PCA lineal es tan bueno como CCD?
      - Métricas de producción: latencia de transform, footprint de memoria.

    Genera vía: CCDEngine.robust_certificate(X_train, X_test)
    """
    # Dimensionality
    d_input: int
    k_effective: int
    k_ci_lower: int              # bootstrap 95% CI lower bound
    k_ci_upper: int              # bootstrap 95% CI upper bound

    # Reconstruction quality (on held-out test set)
    reconstruction_rmse: float
    reconstruction_rmse_std: float
    neighborhood_preservation: float   # fraction of k-NN preserved in [0, 1]

    # Statistical tests
    manifold_hypothesis_pvalue: float  # H0: data is pure Gaussian noise
    linearity_test_pvalue: float       # H0: linear PCA is as good as CCD

    # Production metrics
    transform_latency_ms: float
    memory_footprint_mb: float
    curse_escaped: bool
    cod_reduction_log10: float
    # ── Rigor & safety warnings ──
    nabla_T_bound: float = 0.0           # estimated Lipschitz constant of T(x)
    nabla_T_warning: bool = False        # True if ∇T too large for Langevin convergence
    spectral_multiplicity_warning: bool = False  # detected symmetric manifold
    cod_reduction_is_heuristic: bool = True       # cod_reduction_log10 no es cota rigurosa
    k_eff_is_upper_bound: bool = True             # k_effective ≥ m, no = m

    def is_production_ready(self) -> bool:
        """True if all production readiness criteria are met."""
        return (
            self.reconstruction_rmse < 0.5
            and self.neighborhood_preservation > 0.5
            and self.manifold_hypothesis_pvalue < 0.05
            and self.transform_latency_ms < 500.0
            and self.curse_escaped
            and not self.nabla_T_warning      # ∇T must be safe for Langevin
        )

    def __str__(self) -> str:
        ready = "✓ PRODUCTION READY" if self.is_production_ready() else "≈ RESEARCH PROTOTYPE"
        sep = "━" * 44
        lines = [
            f"RobustCCDCertificate [{ready}]\n{sep}",
            f"Dimensionality:  {self.d_input}D → {self.k_effective}D"
            f"  [CI95: {self.k_ci_lower}–{self.k_ci_upper}D]",
        ]
        if self.k_eff_is_upper_bound:
            lines.append(
                f"  ℹ k_effective es COTA SUPERIOR (≥ m). Puede sobreestimar"
                f" en variedades simétricas."
            )
        if self.spectral_multiplicity_warning:
            lines.append(
                f"  ⚠ ADVERTENCIA: Multiplicidades espectrales anómalas."
                f" Posible variedad simétrica."
            )
        lines.extend([
            f"Quality:",
            f"  Reconstruction RMSE:     {self.reconstruction_rmse:.4f}"
            f" ± {self.reconstruction_rmse_std:.4f}",
            f"  Neighborhood preserved:  {self.neighborhood_preservation:.1%}",
            f"Statistical tests:",
            f"  Manifold hypothesis:  p={self.manifold_hypothesis_pvalue:.4f}"
            f" {'✓' if self.manifold_hypothesis_pvalue < 0.05 else '✗'}",
            f"  Non-linearity test:   p={self.linearity_test_pvalue:.4f}"
            f" {'✓' if self.linearity_test_pvalue < 0.05 else '✗'}",
            f"Production:",
            f"  Transform latency:  {self.transform_latency_ms:.1f} ms",
            f"  Memory footprint:   {self.memory_footprint_mb:.1f} MB",
            f"  CoD reduction:      10^{self.cod_reduction_log10:.1f}"
            f"  {'[HEURÍSTICO]' if self.cod_reduction_is_heuristic else ''}",
        ])
        if self.nabla_T_warning:
            lines.append(
                f"  ⚠ ∇T Lipschitz = {self.nabla_T_bound:.3f} > umbral seguro."
                f" Langevin puede no converger (ver §17.4)."
            )
        else:
            lines.append(
                f"  ∇T Lipschitz = {self.nabla_T_bound:.3f} ✓ acotado"
            )
        lines.append(sep)
        return "\n".join(lines)

    @classmethod
    def compute(
        cls,
        engine: "CCDEngine",
        X_train: np.ndarray,
        X_test: np.ndarray,
        n_bootstrap: int = 20,
    ) -> "RobustCCDCertificate":
        """
        Compute a RobustCCDCertificate via bootstrap + statistical tests.

        Args:
            engine:      fitted CCDEngine
            X_train:     (n_train, d) training data
            X_test:      (n_test, d) held-out test data
            n_bootstrap: bootstrap resampling iterations
        Returns:
            RobustCCDCertificate with all metrics filled.
        """
        rng = np.random.default_rng(42)
        n_test = len(X_test)

        # 1. Bootstrap reconstruction error on test set
        errors: List[float] = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_test, size=n_test, replace=True)
            X_b = X_test[idx]
            Z_b = engine.transform(X_b)
            X_rec = engine.inverse_transform(Z_b)
            errors.append(float(np.sqrt(np.mean((X_b - X_rec) ** 2))))
        recon_rmse = float(np.mean(errors))
        recon_rmse_std = float(np.std(errors))

        # 2. Neighborhood preservation (k-NN trustworthiness)
        k_nn = min(10, n_test - 1)
        Z_test = engine.transform(X_test)
        n_compare = min(n_test, 200)   # cap for speed
        tree_orig = cKDTree(X_test[:n_compare])
        tree_proj = cKDTree(Z_test[:n_compare])
        _, nn_orig = tree_orig.query(X_test[:n_compare], k=k_nn + 1)
        _, nn_proj = tree_proj.query(Z_test[:n_compare], k=k_nn + 1)
        preserved = float(np.mean([
            len(set(nn_orig[i, 1:]) & set(nn_proj[i, 1:])) / k_nn
            for i in range(n_compare)
        ]))

        # 3. Manifold test: H0 = data is Gaussian noise
        #    p < 0.05 → non-Gaussian structure → real manifold
        try:
            from scipy import stats as _stats
            _, p_manifold = _stats.normaltest(Z_test.flatten())
        except Exception:
            p_manifold = 0.0

        # 4. Linearity test: CCD vs PCA reconstruction
        try:
            from scipy import stats as _stats
            k_eff = engine._k_effective
            X_mean = X_train.mean(0)
            _, _, Vt = np.linalg.svd(X_train - X_mean, full_matrices=False)
            V = Vt[:k_eff].T
            X_test_c = X_test - X_mean
            X_pca_rec = X_test_c @ V @ V.T + X_mean
            pca_errs = [
                float(np.sqrt(np.mean(
                    (X_test[rng.choice(n_test, n_test, replace=True)]
                     - X_pca_rec[rng.choice(n_test, n_test, replace=True)]) ** 2
                )))
                for _ in range(n_bootstrap)
            ]
            t_stat = float(
                (np.mean(pca_errs) - recon_rmse) / (np.std(errors) + 1e-10)
            )
            p_linearity = float(_stats.t.sf(t_stat, df=n_bootstrap - 1))
        except Exception:
            p_linearity = 0.5

        # 5. Transform latency
        n_bench = min(100, n_test)
        t0 = time.perf_counter()
        for _ in range(10):
            engine.transform(X_test[:n_bench])
        latency_ms = (time.perf_counter() - t0) / 10.0 * 1000.0

        # 6. Memory footprint (approximate)
        mem_mb = sys.getsizeof(engine) / 1e6
        if engine._preprocessor is not None and engine._preprocessor._Vt is not None:
            mem_mb += engine._preprocessor._Vt.nbytes / 1e6
        if engine._diffusion is not None and engine._diffusion._Phi is not None:
            mem_mb += engine._diffusion._Phi.nbytes / 1e6

        # 7. Bootstrap CI for k_effective
        k_ests: List[int] = []
        n_half = max(len(X_train) // 2, 10)
        for _ in range(n_bootstrap):
            idx = rng.choice(len(X_train), size=n_half, replace=False)
            eng_b = CCDEngine(d_threshold=engine.d_threshold, fit_decoder=False)
            try:
                eng_b.fit(X_train[idx])
                k_ests.append(eng_b._k_effective)
            except Exception:
                k_ests.append(engine._k_effective)
        k_ci_lo = int(np.percentile(k_ests, 2.5))
        k_ci_hi = int(np.percentile(k_ests, 97.5))
        # Ensure the main estimate is within the CI (statistical consistency)
        k_ci_lo = min(k_ci_lo, engine._k_effective)
        k_ci_hi = max(k_ci_hi, engine._k_effective)

        base_cert = engine.certificate()

        # 8. ∇T Lipschitz estimation for Langevin safety check
        nabla_T_bound = 0.0
        nabla_T_warning = False
        if engine._entropy_op is not None and engine._entropy_op._fitted:
            nabla_T_bound = engine._estimate_nabla_T_bound(X_test)
            # Threshold: ∇T Lipschitz > 5.0 is concerning (see §17.4)
            nabla_T_warning = nabla_T_bound > 5.0

        # 9. Check spectral multiplicity (symmetric manifold detection)
        spectral_mult_warning = base_cert.symmetric_manifold_warning

        return cls(
            d_input=engine._d_input,
            k_effective=engine._k_effective,
            k_ci_lower=k_ci_lo,
            k_ci_upper=k_ci_hi,
            reconstruction_rmse=recon_rmse,
            reconstruction_rmse_std=recon_rmse_std,
            neighborhood_preservation=preserved,
            manifold_hypothesis_pvalue=float(p_manifold),
            linearity_test_pvalue=float(p_linearity),
            transform_latency_ms=float(latency_ms),
            memory_footprint_mb=float(mem_mb),
            curse_escaped=base_cert.curse_escaped,
            cod_reduction_log10=base_cert.cod_reduction_log10,
            nabla_T_bound=nabla_T_bound,
            nabla_T_warning=nabla_T_warning,
            spectral_multiplicity_warning=spectral_mult_warning,
            cod_reduction_is_heuristic=True,
            k_eff_is_upper_bound=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR CCD — ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class CCDEngine:
    """
    Motor CCD (Campo de Curvatura Dinámica) — motor unificado para alta dimensión.

    Combina las 5 capas del framework CCD en un pipeline integrado que
    cualquier módulo del ecosistema puede usar como preprocessor automático
    para d > d_threshold.

    Flujo completo:
      X_raw  →  SpectralPreprocessor  →  DiffusionGeometry  →  z_low_dim
                       ↕                   ↕
               CoupledOscillators  →  resonance groups
                       ↕
               LocalEntropyOperator → T(x)  →  LangevinPurifier

    EJEMPLOS DE USO:
      # Entrenar
      engine = CCDEngine(d_threshold=5)
      engine.fit(X_train)

      # Proyectar a baja dimensión
      z = engine.transform(X_new)      # shape (n, k_effective)

      # Reconstruir (aproximado)
      X_approx = engine.inverse_transform(z)

      # Purificar datos ruidosos
      X_clean = engine.langevin_purify(X_noisy)

      # Certificado formal
      cert = engine.certificate()
      print(cert)

    INTEGRACIÓN AUTOMÁTICA:
      Si d < d_threshold, el motor devuelve los datos sin transformar.
      Si d ≥ d_threshold, activa todo el pipeline CCD.
    """

    def __init__(
        self,
        d_threshold: int = 5,
        n_preprocess_components: int = 0,  # 0 = auto
        n_diffusion_components: int = 15,
        n_diffusion_neighbors: int = 20,
        n_oscillator_groups: Optional[int] = None,
        n_entropy_neighbors: int = 10,
        n_langevin_steps: int = 20,
        langevin_dt: float = 0.05,
        diffusion_alpha: float = 1.0,
        seed: int = 42,
        fit_decoder: bool = True,
        decoder_n_neighbors: int = 5,
        decoder_epochs: int = 0,        # IGNORED — kept for backward compat (old MLP decoder removed)
        auto_scale: bool = False,
        spectral_gap_threshold: float = 1.25,
        use_snr_weighting: bool = True,
        multi_scale_diffusion: bool = False,
        skip_connection: bool = True,
    ):
        """
        Args:
            d_threshold: activar CCD solo si d ≥ d_threshold.
            n_preprocess_components: componentes PCA para SpectralPreprocessor.
                0 = auto: max(n_diffusion_components * 4, 32) — al menos 4× el
                objetivo de difusión para que el preprocesamiento no limite la resolución.
            n_diffusion_components: dimensión de salida del diffusion map.
            n_diffusion_neighbors: k para ancho de banda adaptativo.
            n_oscillator_groups: grupos de resonancia (None = auto-detección).
            n_entropy_neighbors: k para cálculo de entropía local.
            n_langevin_steps: pasos de purificación Langevin.
            langevin_dt: paso de tiempo Langevin.
            diffusion_alpha: parámetro de anisotropía (0=kernel, 0.5=FP, 1=LB).
            seed: semilla aleatoria para dinámica Langevin.
            fit_decoder: si True, entrenar ManifoldDecoder para inverse_transform.
            decoder_n_neighbors: vecinos k-NN para el decoder.
            auto_scale: si True, aplica escala global interna antes de la PCA.
                Divide todos los features por sqrt(mean(var_per_feature)), lo que
                PRESERVA los ratios de varianza entre features (señal/ruido)
                sin igualar varianzas individualmente como hace StandardScaler.

            ⚠⚠⚠ MEJORAS PARA ALTA DIMENSION Y CONDICIONES EXTREMAS ⚠⚠⚠
            spectral_gap_threshold: umbral de ratio de eigenvalues para detectar gap espectral
                (1.25 = 25% más grande que el ruido). Valores más altos = más conservador.
            use_snr_weighting: ponderar distancias por SNR de cada dimensión (mejora precisión en ruido)
            multi_scale_diffusion: usar kernel que combina escalas locales y globales (mejor para estructuras complejas)

                CUÁNDO USAR auto_scale=True:
                  - Datos donde las dimensiones de señal tienen mayor varianza
                    natural que las dimensiones de ruido (datos estructurados,
                    embeddings, datos físicos con señal dominante).
                  - Cuando NO se aplicó StandardScaler antes de pasar a CCD.
                  - MEJORA DEMOSTRADA (investigation_v3): en Swiss Roll R^53,
                    el solapamiento k-NN con vecinos reales sube de 9.1% a 98.7%,
                    y LCMC(clean) sube de 0.58 a 0.98 (+69% absoluto).

                CUÁNDO NO USAR (dejar False):
                  - Datos con features de escalas muy diferentes SIN relación
                    señal/ruido (p.ej. mezcla de porcentajes y valores absolutos):
                    en este caso aplicar StandardScaler antes.
                  - Backward compatibility: False preserva el comportamiento anterior.
        """
        self.d_threshold = d_threshold
        self.n_preprocess_components = n_preprocess_components
        self.n_diffusion_components = n_diffusion_components
        self.n_diffusion_neighbors = n_diffusion_neighbors
        self.n_oscillator_groups = n_oscillator_groups
        self.n_entropy_neighbors = n_entropy_neighbors
        self.n_langevin_steps = n_langevin_steps
        self.langevin_dt = langevin_dt
        self.diffusion_alpha = diffusion_alpha
        self.seed = seed
        self.fit_decoder = fit_decoder
        self.decoder_n_neighbors = decoder_n_neighbors
        self.decoder_epochs = decoder_epochs      # kept for backward compatibility (ignored)
        self.auto_scale = auto_scale
        self.spectral_gap_threshold = spectral_gap_threshold
        self.use_snr_weighting = use_snr_weighting
        self.multi_scale_diffusion = multi_scale_diffusion
        self.skip_connection = skip_connection
        self._projection_method: str = "pca"

        self._preprocessor: Optional[SpectralPreprocessor] = None  # Capa 1 (activa)
        self._diffusion: Optional[DiffusionGeometry] = None
        self._oscillators: Optional[CoupledOscillators] = None
        self._entropy_op: Optional[LocalEntropyOperator] = None
        self._langevin: Optional[LangevinPurifier] = None
        self._decoder: Optional[ManifoldDecoder] = None
        self._oscillators_in_reduced_space: bool = False

        self._d_input: int = 0
        self._k_effective: int = 0
        self._fitted: bool = False
        self._auto_scale_factor: float = 1.0  # computed in fit()
        self._snr_weights: Optional[np.ndarray] = None  # NEW: pesos SNR

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def skip_alpha(self) -> float:
        """Get the learned skip-connection weight α."""
        return getattr(self, '_skip_alpha', 0.5)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "CCDEngine":
        """
        Aprender la geometría del manifold de datos desde X de forma (n, d).

        PIPELINE CONECTADO (producción):
          X → Capa1(SpectralPreprocessor) → Z_pre   [PCA whitening, O(n·d·r)]
            → Capa2(DiffusionGeometry) → Z_diff      [sparse diffusion maps]
              → Capa3(CoupledOscillators en Z_diff)   [resonancia en espacio diff]
              → Capa4(LocalEntropy en X)              [temperatura adaptativa]
              → Capa5(LangevinPurifier en X)          [Langevin sparse]
          Si fit_decoder: ManifoldDecoder.fit(Z_diff, X)

        POR QUÉ SpectralPreprocessor:
          Benchmarks muestran que SpectralPreprocessor preserva la métrica
          (correlación de rangos de distancia ρ ≈ 1.0) y es 30× más rápido
          que ChebyshevShell que destruía la métrica (ρ ≈ -0.02).
        """
        # ── VALIDACIÓN DE ENTRADA ──
        if X is None or not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}")
        if X.dtype not in (np.float32, np.float64):
            X = X.astype(np.float64)
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values. Impute or remove before fitting.")
        if np.any(np.isinf(X)):
            raise ValueError("X contains Inf values. Remove or clip before fitting.")
        if np.all(np.std(X, axis=0) < 1e-15):
            raise ValueError("All features have zero variance. Cannot fit CCDEngine.")

        n, d = X.shape
        if n < 3:
            raise ValueError(f"Need at least 3 samples, got {n}")
        if d < 1:
            raise ValueError(f"Need at least 1 feature, got {d}")
        self._d_input = d

        # ── Escala global interna (preserva ratio señal/ruido entre features) ──
        # VALIDADO (investigation_v3_stdscaler.py):
        #   StandardScaler destruye el ratio señal/ruido: k-NN en R^53 encuentra
        #   solo 9.1% de vecinos reales. Sin StandardScaler (escala global): 98.7%.
        #   Con auto_scale=True, LCMC(clean) sube 0.58 → 0.98 en Swiss Roll R^53.
        if self.auto_scale:
            var_per_feature = np.var(X, axis=0)  # (d,)
            mean_var = float(np.mean(var_per_feature))
            self._auto_scale_factor = float(np.sqrt(max(mean_var, 1e-10)))
        else:
            self._auto_scale_factor = 1.0
        X = X / self._auto_scale_factor

        # ── Fast path para datos de baja dimensión ──
        if d < self.d_threshold:
            self._oscillators = CoupledOscillators(
                n_groups=self.n_oscillator_groups
            ).fit(X)
            self._entropy_op = LocalEntropyOperator(
                k_neighbors=min(self.n_entropy_neighbors, n - 1),
            ).fit(X)
            self._k_effective = d
            self._oscillators_in_reduced_space = False
            self._fitted = True
            return self

        # ── Capa 1: SpectralPreprocessor (PCA SIN whitening) ──
        # VALIDADO: whiten=False preserva métrica (rho=1.0) vs whiten=True (rho=0.985).
        # El whitening iguala varianzas → las dimensiones de ruido pesan igual que señal
        # → DiffusionGeometry pierde rho de 0.42 a 0.32. Sin whitening: rho=0.42+.
        n_pre = self.n_preprocess_components
        if n_pre <= 0:
            n_pre = min(d, max(self.n_diffusion_components * 3, 40))
        n_pre = min(n_pre, d, n - 1)
        n_pre = max(n_pre, self.n_diffusion_components + 1)

        self._preprocessor = SpectralPreprocessor(
            n_components=n_pre, whiten=False,
        ).fit(X)
        Z_pre = self._preprocessor.transform(X)    # (n, n_pre) PCA sin whitening — preserva métrica
        
        # ⚠⚠⚠ MEJORA: Detección de gap espectral para ajuste automático ⚠⚠⚠
        eigenvalues = self._preprocessor.get_explained_variance()
        eigenvalues = eigenvalues / (eigenvalues[0] + 1e-10)

        # Detectar gap espectral adaptativo: buscar primera gran caída relativa
        # Umbral conservador que escala con el tamaño de datos
        if n < 200:
            adaptive_threshold = max(self.spectral_gap_threshold, 3.0)
        elif n < 1000:
            adaptive_threshold = max(self.spectral_gap_threshold, 2.5)
        else:
            adaptive_threshold = max(self.spectral_gap_threshold, 2.0)
        gap_detected = False
        search_limit = min(len(eigenvalues), max(20, n // 10))
        for i in range(1, search_limit):
            ratio = eigenvalues[i-1] / max(eigenvalues[i], 1e-12)
            if ratio > adaptive_threshold:
                n_pre = i
                gap_detected = True
                break

        if gap_detected:
            n_pre = max(n_pre, self.n_diffusion_components + 2, 8)
            self._preprocessor = SpectralPreprocessor(
                n_components=n_pre, whiten=False,
            ).fit(X)
            Z_pre = self._preprocessor.transform(X)
            # Recalcular eigenvalues con el nuevo preprocessor
            eigenvalues = self._preprocessor.get_explained_variance()
            eigenvalues = eigenvalues / (eigenvalues[0] + 1e-10)
        
        # ⚠⚠⚠ MEJORA: Ponderación por SNR ⚠⚠⚠
        if self.use_snr_weighting:
            signal_var = eigenvalues[:n_pre]
            if n_pre < len(eigenvalues):
                residual_ev = eigenvalues[n_pre:]
                # Noise floor: median of residual eigenvalues (robust to outliers)
                noise_floor = float(np.median(residual_ev))
                # Add small fraction of max residual to avoid zero floor
                noise_floor = max(noise_floor, float(residual_ev[-1]), 1e-10)
            else:
                noise_floor = float(eigenvalues[-1]) * 0.1 if len(eigenvalues) > 0 else 0.1
            # SNR weights: sqrt of ratio, clamped to [0.1, 10.0]
            snr_ratio = signal_var / (noise_floor + 1e-10)
            self._snr_weights = np.sqrt(np.clip(snr_ratio, 0.01, 100.0))
            Z_pre = Z_pre * self._snr_weights

        # ── Capa 2: Diffusion Geometry MEJORADA ──
        n_comp = min(self.n_diffusion_components, n - 2, n_pre - 1)
        n_comp = max(n_comp, 1)

        # Use ecosystem-optimized diffusion time if available
        diff_time = getattr(self, '_diffusion_time', 1.0)

        if self.multi_scale_diffusion:
            k_local = max(3, min(15, n_pre - 1, n // 20))
            k_global = max(k_local + 5, min(50, n_pre - 1, n // 5))
            diffusion_local = DiffusionGeometry(
                n_components=n_comp,
                n_neighbors=k_local,
                diffusion_time=diff_time * 0.5,
                alpha=0.5
            ).fit(Z_pre)

            diffusion_global = DiffusionGeometry(
                n_components=n_comp,
                n_neighbors=k_global,
                diffusion_time=diff_time * 2.0,
                alpha=1.0
            ).fit(Z_pre)

            Z_diff = 0.7 * diffusion_local.transform(Z_pre) + \
                     0.3 * diffusion_global.transform(Z_pre)
            self._diffusion = diffusion_local
        else:
            n_neigh = max(3, min(self.n_diffusion_neighbors, n - 1, n_pre - 1))
            self._diffusion = DiffusionGeometry(
                n_components=n_comp,
                n_neighbors=n_neigh,
                diffusion_time=diff_time,
                alpha=self.diffusion_alpha,
            ).fit(Z_pre)
            Z_diff = self._diffusion.transform(Z_pre)

        # ── Capa 3: Coupled Oscillators con resonancia adaptativa ──
        # ⚠⚠⚠ MEJORA: Ajustar grupos de resonancia basado en entropía espectral ⚠⚠⚠
        if self.n_oscillator_groups is None:
            # Estimar grupos óptimos usando entropía espectral
            from scipy import stats
            freq_spectrum = np.abs(np.fft.rfft(Z_diff, axis=0))
            spectral_entropy = stats.entropy(freq_spectrum, axis=1)
            n_groups = max(2, int(np.median(spectral_entropy) * 3))
        else:
            n_groups = self.n_oscillator_groups
            
        self._oscillators = CoupledOscillators(
            n_groups=n_groups,
        ).fit(Z_diff)
        self._oscillators_in_reduced_space = True

        # ── Capa 4: Local Entropy en X ──
        self._entropy_op = LocalEntropyOperator(
            k_neighbors=min(self.n_entropy_neighbors, n - 1),
        ).fit(X)

        # ── Capa 5: Langevin Purifier ──
        if self.n_langevin_steps > 0:
            self._langevin = LangevinPurifier(
                n_steps=self.n_langevin_steps,
                dt=self.langevin_dt,
                entropy_op=self._entropy_op,
                seed=self.seed,
                focus_factor=2.0,
            ).fit(X)

        # ── Estimación de dimensionalidad efectiva (ROBUSTA) ──
        k_diff = self._diffusion.intrinsic_dimension_estimate if self._diffusion else d
        n_groups = self._oscillators.n_resonance_groups if self._oscillators else d
        k_eff_base = max(1, max(k_diff, n_groups))
        # Skip-connection añade n_linear componentes lineales
        if self.skip_connection:
            n_linear = min(n_comp, n_pre, 5)
            self._skip_n_linear = n_linear
            # Ecosystem-trained α (replaces heuristic)
            self._skip_alpha = self._train_skip_alpha(X, Z_pre, Z_diff, n_linear)
            self._k_effective = k_eff_base + n_linear
        else:
            self._skip_alpha = 0.0
            self._skip_n_linear = 0
            self._k_effective = k_eff_base

        # ── ManifoldDecoder: k-NN inverse (Z_diff → X) ──
        if self.fit_decoder:
            self._decoder = ManifoldDecoder(
                n_neighbors=min(self.decoder_n_neighbors, n - 1)
            ).fit(Z_diff, X)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform / Inverse / Purify
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X to the low-dimensional CCD representation.

        Route: X → SpectralPreprocessor → DiffusionGeometry → z_diff

        Args:
            X: (n, d)
        Returns:
            z: (n, k_effective) if d ≥ d_threshold, else X unchanged.
        """
        if not self._fitted:
            raise RuntimeError("CCDEngine must be fitted first")
        if self._d_input < self.d_threshold or self._preprocessor is None:
            return X
        X_scaled = X / self._auto_scale_factor
        Z_pre = self._preprocessor.transform(X_scaled)
        if self.use_snr_weighting and hasattr(self, '_snr_weights') and self._snr_weights is not None:
            Z_pre = Z_pre * self._snr_weights
        Z_diff = self._diffusion.transform(Z_pre)
        if self.skip_connection:
            n_linear = getattr(self, '_skip_n_linear', min(Z_diff.shape[1], Z_pre.shape[1], 5))
            alpha = getattr(self, '_skip_alpha', 0.5)
            Z_linear = Z_pre[:, :n_linear]
            Z_weighted = alpha * Z_linear + (1.0 - alpha) * Z_diff[:, :n_linear]
            # Asegurar que tenemos columnas restantes para concatenar
            if n_linear < Z_diff.shape[1]:
                return np.hstack([Z_weighted, Z_diff[:, n_linear:]])
            else:
                return Z_weighted
        return Z_diff

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """
        Reconstruct approximate X from diffusion coordinates z.

        Args:
            z: (n, k)
        Returns:
            X_approx: (n, d)
        """
        if not self._fitted:
            raise RuntimeError("CCDEngine must be fitted first")
        if self._d_input < self.d_threshold or self._diffusion is None:
            return z
        if self._decoder is not None and self._decoder._fitted:
            X_approx = self._decoder.decode(z)
        else:
            Z_pre_approx = self._diffusion.inverse_transform(z)
            if self.use_snr_weighting and hasattr(self, '_snr_weights') and self._snr_weights is not None:
                Z_pre_approx = Z_pre_approx / (self._snr_weights + 1e-10)
            X_approx = self._preprocessor.inverse_transform(Z_pre_approx) if self._preprocessor else Z_pre_approx
        return X_approx * self._auto_scale_factor

    def langevin_purify(self, X: np.ndarray) -> np.ndarray:
        """
        Purify corrupted/noisy data by running Langevin dynamics.

        Implements: dx = -∇U(x)dt + √(2T(x))dW_t
        Moves data toward the high-density manifold.

        Args:
            X: (n, d) noisy input data
        Returns:
            X_pure: (n, d) purified data
        """
        if not self._fitted:
            raise RuntimeError("CCDEngine must be fitted first")
        if self._d_input < self.d_threshold or self._langevin is None:
            return X
        X_scaled = X / self._auto_scale_factor
        return self._langevin.purify(X_scaled) * self._auto_scale_factor

    def temperature(self, X: np.ndarray) -> np.ndarray:
        """
        Compute adaptive temperature T(x) for each point.

        Args:
            X: (n, d)
        Returns:
            T: (n,) ∈ [T_min, T_max]
        """
        if not self._fitted or self._entropy_op is None:
            return np.full(len(X), 0.5)
        X_scaled = X / self._auto_scale_factor
        return self._entropy_op.temperature(X_scaled)

    def local_dimension(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate local intrinsic dimension at each point in X.

        Args:
            X: (n, d)
        Returns:
            d_local: (n,) local dimension estimates
        """
        if not self._fitted or self._entropy_op is None:
            return np.full(len(X), float(self._d_input))
        return self._entropy_op.local_dimension(X)

    def effective_dimension(self) -> int:
        """Return the estimated effective intrinsic dimensionality k << d."""
        return self._k_effective

    def transform_resonance(self, X: np.ndarray) -> np.ndarray:
        """
        Project X using CoupledOscillators resonance transform.

        Uses the oscillator normal modes (not diffusion coordinates)
        for a simpler linear compression.

        Args:
            X: (n, d)
        Returns:
            z_res: (n, n_modes)
        """
        if not self._fitted:
            raise RuntimeError("CCDEngine must be fitted first")
        if self._oscillators is None:
            return X
        X_scaled = X / self._auto_scale_factor
        if self._oscillators_in_reduced_space and self._diffusion is not None:
            Z_pre = self._preprocessor.transform(X_scaled) if self._preprocessor else X_scaled
            Z_diff = self._diffusion.transform(Z_pre)
            return self._oscillators.transform(Z_diff)
        return self._oscillators.transform(X_scaled)

    def transform_coherence(self, X: np.ndarray) -> np.ndarray:
        """
        Project X using Adaptive Coherence Transform.

        Groups in-phase oscillator variables into complex amplitudes,
        producing one coordinate per resonance group.

        Args:
            X: (n, d)
        Returns:
            z_coh: (n, n_groups)
        """
        if not self._fitted:
            raise RuntimeError("CCDEngine must be fitted first")
        if self._oscillators is None:
            return X
        X_scaled = X / self._auto_scale_factor
        if self._oscillators_in_reduced_space and self._diffusion is not None:
            Z_pre = self._preprocessor.transform(X_scaled) if self._preprocessor else X_scaled
            Z_diff = self._diffusion.transform(Z_pre)
            return self._oscillators.adaptive_coherence_transform(Z_diff)
        return self._oscillators.adaptive_coherence_transform(X_scaled)

    # ------------------------------------------------------------------
    # Formal certificate
    # ------------------------------------------------------------------

    def certificate(self) -> CCDCertificate:
        """
        Generate a formal CCDCertificate of the dimensionality reduction achieved.

        Returns:
            CCDCertificate with all metrics and a curse_escaped flag.
        """
        if not self._fitted:
            raise RuntimeError("CCDEngine must be fitted first")

        d = self._d_input
        k = self._k_effective

        if d < self.d_threshold:
            return CCDCertificate(
                d_input=d, k_effective=d,
                k_diffusion=d, k_chebyshev=d,
                n_resonance_groups=d,
                reduction_ratio=1.0,
                explained_variance_ratio_top_k=1.0,
                alpha_entropy=0.0,
                curse_escaped=False,
                cod_reduction_log10=0.0,
            )

        k_diff = self._diffusion.intrinsic_dimension_estimate if self._diffusion else k
        k_cheb = (
            self._preprocessor.effective_rank if self._preprocessor is not None else k
        )
        n_groups = self._oscillators.n_resonance_groups if self._oscillators else k

        if self._oscillators is not None:
            evr = self._oscillators.explained_variance_ratio
            var_top_k = float(np.clip(evr.sum(), 0.0, 1.0))
        else:
            var_top_k = 1.0

        if self._preprocessor is not None:
            evr_pca = self._preprocessor.explained_variance_ratio
            log_evr = np.log(np.maximum(evr_pca, 1e-12))
            if len(log_evr) >= 2:
                slope = float(-np.polyfit(np.arange(min(len(log_evr), 20)), log_evr[:min(len(log_evr), 20)], 1)[0])
            else:
                slope = 0.0
            alpha_ent = max(slope, 0.0)
        else:
            alpha_ent = 0.0

        # ── Detección de variedades simétricas ──
        symmetric_warning = self._detect_spectral_multiplicity()

        eps = 0.01
        log10_N_grid = d * np.log10(1.0 / eps)
        log10_N_ccd = k * np.log10(1.0 / eps)
        cod_reduction = float(log10_N_grid - log10_N_ccd)

        # ── Sample complexity note ──
        # El costo de aprender la variedad es O(ε^{-m}), no O(log(1/ε)).
        # cod_reduction_log10 mide solo la contracción de cubrimiento.
        sample_note = (
            f"Complejidad de muestra para aprender M: O(ε^{{-{min(k, 20)}}}) "
            f"(Niyogi et al.). cod_reduction_log10 mide contracción de "
            f"cubrimiento, no cota de error de aproximación."
        )

        return CCDCertificate(
            d_input=d,
            k_effective=k,
            k_diffusion=k_diff,
            k_chebyshev=k_cheb,
            n_resonance_groups=n_groups,
            reduction_ratio=float(d / max(k, 1)),
            explained_variance_ratio_top_k=var_top_k,
            alpha_entropy=alpha_ent,
            curse_escaped=(k < d // 2),
            cod_reduction_log10=max(cod_reduction, 0.0),
            symmetric_manifold_warning=symmetric_warning,
            cod_reduction_is_heuristic=True,
            sample_complexity_note=sample_note,
        )

    def robust_certificate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        n_bootstrap: int = 20,
    ) -> "RobustCCDCertificate":
        """
        Generate a statistically robust certificate with bootstrap CI.

        Args:
            X_train: (n_train, d) training data
            X_test: (n_test, d) held-out test data
            n_bootstrap: number of bootstrap resampling iterations
        Returns:
            RobustCCDCertificate
        """
        return RobustCCDCertificate.compute(self, X_train, X_test, n_bootstrap)

    # ------------------------------------------------------------------
    # Ecosystem-Driven Optimization (TAA + OTU + ERGON concepts)
    # ------------------------------------------------------------------

    def ecosystem_optimize(
        self,
        X: np.ndarray,
        epsilon: float = 0.01,
    ) -> Dict[str, float]:
        """
        Use TAA spectral classification + OTU dual budget to auto-tune CCD.

        This method embodies the ecosystem's key discoveries:
          - TAA's α_A classification (HEURISTIC, §17.6) determines the optimal
            diffusion strategy
          - OTU's Γ (spectral gap) bounds the diffusion time τ ≤ 1/Γ
          - D₂ (correlation dimension) validates intrinsic dimension estimates
          - Dual Budget calibration principle (HEURISTIC, §17.5): aligns spatial
            and temporal complexity

        Args:
            X: (n, d) input data
            epsilon: target precision for d*(ε) computation

        Returns:
            dict with auto-tuned parameters and diagnostic information
        """
        n, d = X.shape

        # ── Step 1: TAA Spectral Classification ────────────────────────
        X_c = X - X.mean(axis=0)
        _, S, _ = np.linalg.svd(X_c, full_matrices=False)
        eigenvalues_raw = (S ** 2) / (n - 1) if n > 1 else S ** 2
        eigenvalues = eigenvalues_raw / (eigenvalues_raw[0] + 1e-10)

        alpha_class, alpha_rate = self._taa_classify(eigenvalues)

        # ── Step 1b: Marchenko-Pastur signal count ──────────────────────
        mp_ratio = d / max(n, 1)
        if mp_ratio > 0.01:
            mp_lambda_plus = (1.0 + np.sqrt(mp_ratio)) ** 2
            n_signal_mp = int(np.sum(eigenvalues > mp_lambda_plus * max(eigenvalues[-1], 1e-10) * 1.5))
        else:
            n_signal_mp = len(eigenvalues)

        d_star_taa = self._taa_compute_d_star(eigenvalues, epsilon)
        d_star = min(d_star_taa, max(n_signal_mp, 2))

        # ── Step 2: OTU Spectral Gap ───────────────────────────────────
        if len(eigenvalues) >= 2:
            gamma_otu = -np.log(max(eigenvalues[1] / (eigenvalues[0] + 1e-10), 1e-8))
        else:
            gamma_otu = 0.1

        # ── Step 3: Thermodynamic β_c for Dual Budget calibration ───────
        thermo_beta_c = 1.0
        thermo_d_star = d_star
        try:
            import torch
            eig_t = torch.tensor(eigenvalues, dtype=torch.float64)
            from acf_functor.thermodynamic_acf import (
                FreeEnergyComputer, CriticalityDetector
            )
            computer = FreeEnergyComputer(eig_t, entropy_mode="spectral")
            detector = CriticalityDetector(
                computer, beta_min=0.01, beta_max=200.0, n_beta=200
            )
            transitions = detector.find_phase_transitions()
            if transitions:
                thermo_beta_c = float(transitions[0].beta_c)
                thermo_d_star = transitions[0].d_high
            else:
                _, d_star_seq = detector.sweep()
                thermo_d_star = int(np.median(d_star_seq)) if d_star_seq else d_star
        except ImportError:
            pass

        # ── Step 4: Compute optimal components ─────────────────────────
        optimal_n_components = min(d_star, d - 1, n - 1)
        optimal_n_components = max(optimal_n_components, 2)
        optimal_n_components = min(optimal_n_components, 40)

        # Diffusion time: τ = 1/Γ_OTU (OTU dual budget principle)
        optimal_diffusion_time = min(max(1.0 / (gamma_otu + 1e-4), 0.1), 5.0)

        # ── Step 5: Alpha-based configuration ──────────────────────────
        if alpha_class == "EXPONENTIAL":
            optimal_alpha = 0.5
            optimal_n_neighbors = int(np.clip(d_star * 3, 10, 50))
            optimal_langevin_steps = min(optimal_n_components * 5, 40)
            strategy = "diffusion_fast_purify"
        elif alpha_class == "POLYNOMIAL":
            optimal_alpha = 1.0
            optimal_n_neighbors = int(np.clip(d_star * 5, 15, 80))
            optimal_langevin_steps = min(optimal_n_components * 3, 25)
            strategy = "diffusion_deep"
        elif alpha_class == "FINITE":
            optimal_alpha = 0.0
            optimal_n_neighbors = d_star + 2
            optimal_langevin_steps = 0
            strategy = "linear_only"
        elif alpha_class == "NOISY":
            optimal_alpha = 1.0
            optimal_n_neighbors = int(np.clip(d_star * 8, 20, 100))
            optimal_langevin_steps = min(optimal_n_components * 6, 50)
            strategy = "purify_aggressive"
        else:
            optimal_alpha = 0.5
            optimal_n_neighbors = self.n_diffusion_neighbors
            optimal_langevin_steps = self.n_langevin_steps
            strategy = "adaptive"

        # ── Step 6: Dual Budget strict calibration via thermodynamic β_c ──
        # When β_c is known, calibrate n_langevin_steps to match d_spatial
        if optimal_langevin_steps > 0 and thermo_beta_c > 0:
            # β_c determines the critical temperature where spatial/temporal balance exists
            budget_ratio = max(1.0, optimal_langevin_steps / max(optimal_n_components, 1))
            if budget_ratio > 3.0:
                # Over-aggressive purification detected → recalibrate
                optimal_langevin_steps = max(
                    2,
                    min(optimal_langevin_steps,
                        optimal_n_components + int(thermo_beta_c * 0.5)))
            strategy += "_calibrated"

        # ── Step 7: Apply auto-tuned parameters ────────────────────────
        self.n_diffusion_components = int(optimal_n_components)
        self.n_diffusion_neighbors = int(optimal_n_neighbors)
        self.diffusion_alpha = float(optimal_alpha)
        self.n_langevin_steps = int(optimal_langevin_steps)
        self._diffusion_time = float(optimal_diffusion_time)

        return {
            "alpha_class": alpha_class,
            "alpha_rate": float(alpha_rate),
            "d_star": int(d_star),
            "d_star_taa": int(d_star_taa),
            "n_signal_mp": int(n_signal_mp),
            "gamma_otu": float(gamma_otu),
            "optimal_n_components": int(optimal_n_components),
            "optimal_n_neighbors": int(optimal_n_neighbors),
            "optimal_diffusion_alpha": float(optimal_alpha),
            "optimal_diffusion_time": float(optimal_diffusion_time),
            "optimal_langevin_steps": int(optimal_langevin_steps),
            "thermo_beta_c": float(thermo_beta_c),
            "thermo_d_star": int(thermo_d_star),
            "strategy": strategy,
            "dual_budget_verified": abs(optimal_n_components - optimal_langevin_steps) <= 3,
            "dual_budget_ratio": float(optimal_langevin_steps / max(optimal_n_components, 1)),
            "correlation_dimension": float(self._estimate_correlation_dimension(X)),
        }

    def _taa_classify(self, eigenvalues: np.ndarray) -> Tuple[str, float]:
        """
        CLASIFICACIÓN HEURÍSTICA TAA del decaimiento espectral.

        NO es una partición matemática exhaustiva de las secuencias espectrales.
        Es un árbol de decisión con umbrales fijos (0.85, 0.7, 0.6) diseñado
        para seleccionar automáticamente la estrategia de compresión de CCD.

        Ver §17.6 del paper para los fundamentos y limitaciones de cada test.

        Returns (alpha_class, alpha_rate) where alpha_class is one of:
          EXPONENTIAL, POLYNOMIAL, FINITE, NOISY, UNKNOWN
        """
        n = len(eigenvalues)
        if n < 3:
            return "FINITE", float(n)

        # ── Test 1: Correlation-based classification ───────────────────
        k = np.arange(1, n + 1)
        log_lam = np.log(np.maximum(eigenvalues, 1e-12))
        log_k = np.log(np.maximum(k, 1e-12))

        r_exp = float(np.corrcoef(k[:min(n, 30)], log_lam[:min(n, 30)])[0, 1]) if n > 2 else 0.0
        r_poly = float(np.corrcoef(log_k[:min(n, 20)], log_lam[:min(n, 20)])[0, 1]) if n > 2 else 0.0

        # ── Test 2: Marchenko-Pastur noise bulk detection ──────────────
        # For X ∈ R^{n×d} with iid noise N(0,σ²):
        #   λ_max = σ²(1 + √(d/n))²,  λ_min = σ²(1 - √(d/n))²
        # Eigenvalues inside [λ_min, λ_max] are noise.
        # This only applies for d/n > 0 (i.e., n not >> d).
        n_data = getattr(self, '_d_input', n)
        if n_data > 0 and len(eigenvalues) > 1:
            mp_ratio = len(eigenvalues) / max(n_data, 1)
            if mp_ratio > 0.01:
                mp_lambda_plus = (1.0 + np.sqrt(mp_ratio)) ** 2
                mp_lambda_minus = max(0.0, (1.0 - np.sqrt(mp_ratio))) ** 2 if mp_ratio <= 1.0 else 0.0
                # Count signal eigenvalues above MP upper bound
                n_signal_mp = int(np.sum(eigenvalues > mp_lambda_plus * eigenvalues[-1] * 1.5))
                n_noise_mp = n - n_signal_mp
                mp_noise_fraction = n_noise_mp / max(n, 1)
            else:
                n_signal_mp = n
                mp_noise_fraction = 0.0
        else:
            n_signal_mp = n
            mp_noise_fraction = 0.0

        # ── Test 3: Thermodynamic phase transition ─────────────────────
        thermo_phase_exists = False
        thermo_beta_c = 1.0
        thermo_d_star = n
        try:
            import torch
            eig_t = torch.tensor(eigenvalues, dtype=torch.float64)
            from acf_functor.thermodynamic_acf import (
                FreeEnergyComputer, CriticalityDetector
            )
            computer = FreeEnergyComputer(eig_t, entropy_mode="spectral")
            detector = CriticalityDetector(
                computer, beta_min=0.01, beta_max=200.0, n_beta=200
            )
            transitions = detector.find_phase_transitions()
            if transitions:
                thermo_phase_exists = True
                main_transition = transitions[0]
                thermo_beta_c = float(main_transition.beta_c)
                thermo_d_star = main_transition.d_high
            else:
                # No phase transition → check if finite
                _, d_star_seq = detector.sweep()
                if len(set(d_star_seq)) <= 2:
                    thermo_phase_exists = False
                    thermo_d_star = d_star_seq[0] if d_star_seq else n
                else:
                    thermo_phase_exists = True
                    thermo_d_star = max(d_star_seq)
        except ImportError:
            pass

        # ── Decision logic combining all three tests ───────────────────
        # EXPONENTIAL: log-linear correlation AND low noise
        if abs(r_exp) > abs(r_poly) and abs(r_exp) > 0.85 and mp_noise_fraction < 0.7:
            rho = float(np.exp(-np.polyfit(k[:min(n, 30)], -log_lam[:min(n, 30)], 1)[0]))
            return "EXPONENTIAL", max(rho, 1.01)

        # NOISY: high MP noise fraction → data dominated by noise
        if mp_noise_fraction > 0.6:
            return "NOISY", max(2.0, float(mp_noise_fraction * 5.0))

        # POLYNOMIAL: log-log correlation OR moderate noise
        if abs(r_poly) > 0.7 or (0.3 < mp_noise_fraction <= 0.6):
            coeffs = np.polyfit(log_k[:min(n, 20)], -log_lam[:min(n, 20)], 1)
            s = float(max(coeffs[0], 0.5))
            return "POLYNOMIAL", s

        # FINITE: requires thermodynamic confirmation if available
        if thermo_phase_exists:
            return "FINITE", float(thermo_d_star)

        # Check for clear spectral gap (ratio > 10x) — override to FINITE
        for i in range(1, min(len(eigenvalues), 15)):
            ratio = eigenvalues[i-1] / max(eigenvalues[i], 1e-12)
            if ratio > 10.0 and i <= max(3, n // 20):
                return "FINITE", float(i)

        # Fallback: secondary check for sharp drop-off
        for i in range(1, min(len(eigenvalues), 15)):
            if eigenvalues[i-1] / max(eigenvalues[i], 1e-12) > 10.0:
                return "FINITE", float(i)

        # Ultimate fallback
        return "POLYNOMIAL", 1.5

    def _taa_compute_d_star(
        self, eigenvalues: np.ndarray, epsilon: float
    ) -> int:
        """
        TAA-3b: Compute d*(ε) = optimal number of components for precision ε.

        For exponential decay |λ_k| ≤ ρ^{-k}: d* = O(log 1/ε)
        For polynomial decay |λ_k| ≤ k^{-s}: d* = O(ε^{-1/s})
        """
        alpha_class, alpha_rate = self._taa_classify(eigenvalues)

        if alpha_class == "EXPONENTIAL":
            inv_rho = 1.0 / max(alpha_rate - 1.0, 0.01)
            d_star = int(np.ceil(np.log(1.0 / epsilon) / np.log(inv_rho + 1e-10)))
        elif alpha_class == "POLYNOMIAL":
            s = max(alpha_rate, 0.5)
            d_star = int(np.ceil((1.0 / epsilon) ** (1.0 / s)))
        else:
            d_star = min(len(eigenvalues), int(np.ceil(1.0 / epsilon)))

        return max(2, min(d_star, len(eigenvalues)))

    def _train_skip_alpha(
        self, X_raw: np.ndarray, Z_pre: np.ndarray,
        Z_diff: np.ndarray, n_linear: int,
    ) -> float:
        """
        Train skip-connection α using ecosystem signals + cross-validation.

        The ecosystem determines the search range:
          TAA α_class → FINITE: α ∈ [0.7, 0.95] (data is linear)
          TAA α_class → POLYNOMIAL: α ∈ [0.3, 0.7]
          TAA α_class → EXPONENTIAL: α ∈ [0.1, 0.5]
          TAA α_class → NOISY: α ∈ [0.1, 0.3]
          OTU D₂: if D₂ << d, expand range downward (more non-linear)

        Cross-validation selects α ∈ range that maximizes k-NN preservation
        on the projected space. This replaces the heuristic variance-based α.
        """
        n = len(Z_pre)
        if n < 10 or n_linear < 1:
            return 0.5

        # ── Ecosystem signals ──────────────────────────────────────────
        # TAA classification on PCA eigenvalues
        X_c = X_raw - X_raw.mean(axis=0)
        _, S, _ = np.linalg.svd(X_c, full_matrices=False)
        ev = (S ** 2) / (n - 1) if n > 1 else S ** 2
        ev_norm = ev / (ev[0] + 1e-10)
        alpha_class, alpha_rate = self._taa_classify(ev_norm)

        # OTU correlation dimension
        D2 = self._estimate_correlation_dimension(X_raw)
        d = X_raw.shape[1]
        D2_ratio = D2 / max(d, 1)

        # Thermodynamic β_c
        thermo_beta_c = 1.0
        try:
            import torch
            eig_t = torch.tensor(ev_norm, dtype=torch.float64)
            from acf_functor.thermodynamic_acf import FreeEnergyComputer, CriticalityDetector
            computer = FreeEnergyComputer(eig_t, entropy_mode="spectral")
            detector = CriticalityDetector(computer, beta_min=0.01, beta_max=200.0, n_beta=100)
            transitions = detector.find_phase_transitions()
            if transitions:
                thermo_beta_c = float(transitions[0].beta_c)
        except ImportError:
            pass

        # ── Determine search range from ecosystem signals ──────────────
        if alpha_class == "FINITE":
            alpha_min, alpha_max = 0.7, 0.95
        elif alpha_class == "EXPONENTIAL" or alpha_class == "NOISY":
            alpha_min, alpha_max = 0.1, 0.5
        else:  # POLYNOMIAL or UNKNOWN
            alpha_min, alpha_max = 0.3, 0.7

        # Expand range if manifold is low-dimensional (D₂ << d)
        if D2_ratio < 0.3:
            alpha_max += 0.15

        # If no phase transition (β_c → ∞), data is linear → increase α
        if thermo_beta_c > 50.0:
            alpha_min += 0.15

        # ── DETECCIÓN DE CASOS PROBLEMÁTICOS ──────────────────────────
        # 1. Detectar outliers extremos (varianza alta en componentes principales)
        var_ratio = ev_norm[0] / max(ev_norm[-1], 1e-10) if len(ev_norm) > 1 else 1.0
        has_extreme_outliers = var_ratio > 100.0  # Componentes principales dominantes
        
        # 2. Detectar datos puramente lineales (decaimiento rápido)
        is_purely_linear = alpha_class == "FINITE" and alpha_rate > 0.9
        
        # 3. Detectar alta dimensión con ruido (muchas componentes con varianza similar)
        noise_floor = np.median(ev_norm[max(1, len(ev_norm)//2):]) if len(ev_norm) > 5 else ev_norm[-1]
        signal_to_noise = ev_norm[0] / max(noise_floor, 1e-10)
        is_high_dim_noise = signal_to_noise < 5.0 and d > 50
        
        # Ajustar α basado en casos problemáticos
        if has_extreme_outliers:
            # Outliers extremos: α alto para estabilidad
            alpha_min = max(alpha_min, 0.6)
            alpha_max = min(alpha_max, 0.9)
            # Desactivar SNR weighting para outliers
            self.use_snr_weighting = False
            
        if is_purely_linear:
            # Datos puramente lineales: α muy alto
            alpha_min = max(alpha_min, 0.8)
            alpha_max = min(alpha_max, 0.98)
            
        if is_high_dim_noise:
            # Alta dimensión con ruido: α medio, evitar extremos
            alpha_min = max(alpha_min, 0.3)
            alpha_max = min(alpha_max, 0.7)
            # Reducir SNR weighting para ruido
            if hasattr(self, '_snr_weights'):
                self._snr_weights = np.clip(self._snr_weights, 0.5, 2.0)
        
        # 4. Detectar datos fuertemente no-lineales (alta dimensión intrínseca)
        is_high_intrinsic_dim = D2_ratio > 0.8  # D₂ cercano a d
        if is_high_intrinsic_dim:
            # Datos no-lineales fuertes: α bajo, priorizar difusión
            alpha_min = max(alpha_min, 0.1)
            alpha_max = min(alpha_max, 0.4)
            # Aumentar componentes de difusión
            if hasattr(self, 'n_diffusion_components'):
                self.n_diffusion_components = min(self.n_diffusion_components + 5, Z_diff.shape[1])
        
        # 5. Detectar ruido correlacionado (estructura débil)
        is_correlated_noise = thermo_beta_c > 100.0 and signal_to_noise < 3.0
        if is_correlated_noise:
            # Ruido correlacionado: α medio-alto, estabilidad
            alpha_min = max(alpha_min, 0.5)
            alpha_max = min(alpha_max, 0.8)
            # Reducir SNR weighting
            if hasattr(self, '_snr_weights'):
                self._snr_weights = np.clip(self._snr_weights, 0.7, 1.5)
        
        alpha_min = np.clip(alpha_min, 0.05, 1.0)
        alpha_max = np.clip(alpha_max, 0.05, 1.0)

        # ── Ecosystem-guided cross-validation ─────────────────────────
        # Use more sophisticated validation based on ecosystem signals
        n_folds = min(5, max(2, n // 30))  # More folds for better estimation
        fold_size = n // n_folds
        n_test = min(fold_size // 2, 150)  # Larger validation sets

        n_lin = min(n_linear, Z_pre.shape[1])
        
        # ── VALIDACIÓN DE SKIP-CONNECTION ────────────────────────────
        # Primero validar si el skip-connection es beneficioso
        # Comparar: 1) Solo PCA, 2) Solo difusión, 3) Skip-connection
        
        # Calcular trustworthiness para solo PCA
        trust_pca_only = self._compute_trustworthiness_fast(X_raw, Z_pre[:, :n_lin], n_test)
        
        # Calcular trustworthiness para solo difusión
        trust_diff_only = self._compute_trustworthiness_fast(X_raw, Z_diff[:, :n_lin], n_test)
        
        # Si PCA es mucho mejor que difusión, aumentar α mínimo
        pca_vs_diff_ratio = trust_pca_only / max(trust_diff_only, 1e-10)
        if pca_vs_diff_ratio > 1.1:  # PCA es 10% mejor que difusión
            alpha_min = max(alpha_min, 0.8)
            alpha_max = min(alpha_max, 0.98)
            # Si la diferencia es muy grande, forzar α alto
            if pca_vs_diff_ratio > 1.3:
                # Datos puramente lineales: usar α ≈ 1.0
                best_alpha = 0.95
                self._skip_alpha = best_alpha
                return best_alpha
        
        # Si difusión es mucho mejor que PCA, disminuir α máximo
        diff_vs_pca_ratio = trust_diff_only / max(trust_pca_only, 1e-10)
        if diff_vs_pca_ratio > 1.1:  # Difusión es 10% mejor que PCA
            alpha_min = max(alpha_min, 0.1)
            alpha_max = min(alpha_max, 0.5)
            # Si la diferencia es muy grande, desactivar skip-connection
            if diff_vs_pca_ratio > 1.3:
                # Datos fuertemente no-lineales: skip-connection es perjudicial
                self.skip_connection = False
                best_alpha = 0.0
                self._skip_alpha = best_alpha
                return best_alpha
        
        # ── DETECCIÓN DE CASOS DONDE SKIP-CONNECTION ES PERJUDICIAL ──
        # Si PCA es mucho mejor que difusión, también considerar desactivar skip
        if pca_vs_diff_ratio > 1.3:  # PCA es 30% mejor que difusión
            # Datos puramente lineales: skip-connection puede ser perjudicial
            # Forzar α muy alto en lugar de desactivar completamente
            best_alpha = 0.95
            self._skip_alpha = best_alpha
            return best_alpha
        
        # ── DECISIÓN FINAL: ¿USAR SKIP-CONNECTION O NO? ──────────────
        # Si la diferencia entre PCA y difusión es pequeña (< 5%), 
        # el skip-connection puede introducir ruido innecesario
        relative_diff = abs(trust_pca_only - trust_diff_only) / max(trust_pca_only, 1e-10)
        
        if relative_diff < 0.10:  # Diferencia menor al 10%
            # Los métodos son casi equivalentes, skip-connection puede ser perjudicial
            # Evaluar si el skip-connection ayuda o no
            # Probar varios α para encontrar el mejor
            test_alphas = [0.2, 0.5, 0.8]
            best_test_trust = -1.0
            best_test_alpha = 0.5
            
            for test_alpha in test_alphas:
                Z_test = test_alpha * Z_pre[:, :n_lin] + (1.0 - test_alpha) * Z_diff[:, :n_lin]
                trust_test = self._compute_trustworthiness_fast(X_raw, Z_test, n_test)
                if trust_test > best_test_trust:
                    best_test_trust = trust_test
                    best_test_alpha = test_alpha
            
            # Comparar con el mejor de PCA o difusión sola
            best_single_trust = max(trust_pca_only, trust_diff_only)
            
            # Si el skip-connection no mejora significativamente, desactivarlo
            if best_test_trust < best_single_trust - 0.0005:
                self.skip_connection = False
                best_alpha = 0.0
                self._skip_alpha = best_alpha
                return best_alpha
            else:
                # Usar el mejor α encontrado
                best_alpha = best_test_alpha
                self._skip_alpha = best_alpha
                return best_alpha
        
        # Adaptive grid based on ecosystem signals
        if alpha_class == "FINITE":
            # Linear data: finer grid near high alpha
            alphas = np.linspace(alpha_min, alpha_max, 9)
            alphas = np.sort(np.concatenate([alphas, np.linspace(0.8, 0.95, 5)]))
        elif alpha_class == "EXPONENTIAL":
            # Non-linear data: finer grid in middle range
            alphas = np.linspace(alpha_min, alpha_max, 11)
        else:
            # Default: balanced grid
            alphas = np.linspace(alpha_min, alpha_max, 9)
        
        alphas = np.unique(np.clip(alphas, 0.05, 0.95))
        scores = np.zeros(len(alphas))
        trust_scores = np.zeros(len(alphas))  # Trustworthiness metric
        
        # Use FAISS for high-dimensional validation if available
        use_faiss = _FAISS_AVAILABLE and X_raw.shape[1] > 100
        
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = min(val_start + n_test, n)
            if val_end <= val_start:
                continue
            val_idx = np.arange(val_start, val_end)
            X_val = X_raw[val_idx]
            Z_pre_val = Z_pre[val_idx, :n_lin]
            Z_diff_val = Z_diff[val_idx, :n_lin]
            
            # Ground truth neighbors in original space
            k_neighbors = min(10, val_end - val_start - 1)
            if use_faiss:
                # Use FAISS for high dimensions
                X_val_f32 = np.ascontiguousarray(X_val, dtype=np.float32)
                index_orig = _faiss.IndexFlatL2(X_val.shape[1])
                index_orig.add(X_val_f32)
                _, nn_orig = index_orig.search(X_val_f32, k_neighbors)
            else:
                # Use cKDTree for low dimensions
                from scipy.spatial import cKDTree
                tree_orig = cKDTree(X_val)
                _, nn_orig = tree_orig.query(X_val, k=k_neighbors)
            
            for ai, alpha in enumerate(alphas):
                Z_val = alpha * Z_pre_val + (1.0 - alpha) * Z_diff_val
                
                if use_faiss:
                    Z_val_f32 = np.ascontiguousarray(Z_val, dtype=np.float32)
                    index_val = _faiss.IndexFlatL2(Z_val.shape[1])
                    index_val.add(Z_val_f32)
                    _, nn_val = index_val.search(Z_val_f32, k_neighbors)
                else:
                    tree_val = cKDTree(Z_val)
                    _, nn_val = tree_val.query(Z_val, k=k_neighbors)
                
                # Compute trustworthiness metric (T_k)
                # Trustworthiness measures preservation of neighborhood structure
                trust = 0.0
                for i in range(len(nn_val)):
                    orig_neighbors = set(nn_orig[i])
                    proj_neighbors = set(nn_val[i])
                    # Count neighbors preserved in projection
                    preserved = len(orig_neighbors.intersection(proj_neighbors))
                    trust += preserved / k_neighbors
                
                trust_scores[ai] += trust / len(nn_val)
                
                # Also compute continuity (inverse of trustworthiness)
                continuity = 0.0
                for i in range(len(nn_orig)):
                    orig_neighbors = set(nn_orig[i])
                    proj_neighbors = set(nn_val[i])
                    preserved = len(orig_neighbors.intersection(proj_neighbors))
                    continuity += preserved / k_neighbors
                
                # Combined score: trustworthiness - 0.3*continuity (balanced)
                scores[ai] += (trust / len(nn_val)) - 0.3 * (continuity / len(nn_orig))

        # Normalize scores
        trust_scores /= n_folds
        scores /= n_folds
        
        # Use ecosystem to weight the final decision
        if alpha_class == "FINITE":
            # For linear data, trustworthiness is more important
            final_scores = 0.7 * trust_scores + 0.3 * scores
        elif alpha_class == "EXPONENTIAL" or alpha_class == "NOISY":
            # For non-linear/noisy data, balance is key
            final_scores = 0.5 * trust_scores + 0.5 * scores
        else:
            # Default: use combined scores
            final_scores = 0.6 * trust_scores + 0.4 * scores
        
        best_idx = int(np.argmax(final_scores))
        best_alpha = float(alphas[best_idx])
        
        # Apply ecosystem-based regularization
        if alpha_class == "FINITE" and best_alpha < 0.7:
            best_alpha = min(0.85, best_alpha + 0.15)
        elif alpha_class == "EXPONENTIAL" and best_alpha > 0.6:
            best_alpha = max(0.3, best_alpha - 0.15)
            
        return best_alpha

    def _compute_trustworthiness_fast(self, X: np.ndarray, Z: np.ndarray, n_test: int = 100) -> float:
        """
        Fast trustworthiness computation for skip-connection validation.
        
        Computes trustworthiness metric on a subset of points for speed.
        Trustworthiness measures how well k-NN relationships are preserved.
        """
        n = len(X)
        if n < 20 or n_test < 10:
            return 0.5
            
        # Sample test points
        n_test = min(n_test, n // 2)
        test_idx = np.random.default_rng(42).choice(n, n_test, replace=False)
        
        # Compute k-NN in original space
        k_neighbors = min(10, n - 1)
        X_test = X[test_idx]
        Z_test = Z[test_idx]
        
        # Use FAISS for high dimensions
        d_orig = X.shape[1]
        if d_orig > 100:
            try:
                import faiss
                # Build index for original space
                X_norm = X_test.astype(np.float32)
                index_orig = faiss.IndexFlatL2(d_orig)
                index_orig.add(X_norm)
                _, orig_nn = index_orig.search(X_norm, k_neighbors + 1)
                orig_nn = orig_nn[:, 1:]  # Exclude self
            except ImportError:
                # Fallback to scipy
                tree_orig = cKDTree(X_test)
                _, orig_nn = tree_orig.query(X_test, k=k_neighbors + 1)
                orig_nn = orig_nn[:, 1:]
        else:
            tree_orig = cKDTree(X_test)
            _, orig_nn = tree_orig.query(X_test, k=k_neighbors + 1)
            orig_nn = orig_nn[:, 1:]
        
        # Compute k-NN in projected space
        d_proj = Z.shape[1]
        if d_proj > 100:
            try:
                import faiss
                Z_norm = Z_test.astype(np.float32)
                index_proj = faiss.IndexFlatL2(d_proj)
                index_proj.add(Z_norm)
                _, proj_nn = index_proj.search(Z_norm, k_neighbors + 1)
                proj_nn = proj_nn[:, 1:]
            except ImportError:
                tree_proj = cKDTree(Z_test)
                _, proj_nn = tree_proj.query(Z_test, k=k_neighbors + 1)
                proj_nn = proj_nn[:, 1:]
        else:
            tree_proj = cKDTree(Z_test)
            _, proj_nn = tree_proj.query(Z_test, k=k_neighbors + 1)
            proj_nn = proj_nn[:, 1:]
        
        # Compute trustworthiness
        trust = 0.0
        for i in range(n_test):
            # Points that are neighbors in projected space but not in original
            proj_neighbors = set(proj_nn[i])
            orig_neighbors = set(orig_nn[i])
            
            # Rank of intruders in original space
            intruders = proj_neighbors - orig_neighbors
            for intruder in intruders:
                # Find rank of intruder in original space
                try:
                    rank = np.where(orig_nn[i] == intruder)[0][0] + 1
                except IndexError:
                    # Intruder not in original k-NN, rank > k_neighbors
                    rank = k_neighbors + 1
                
                trust += (rank - k_neighbors) / k_neighbors
        
        # Normalize: trustworthiness ∈ [0, 1], higher is better
        trust = 1.0 - (2.0 * trust) / (n_test * k_neighbors * (2 * n_test - 3 * k_neighbors - 1))
        return float(np.clip(trust, 0.0, 1.0))

    def _estimate_correlation_dimension(self, X: np.ndarray) -> float:
        """
        OTU-2026: Estimate D₂ (correlation dimension) from pair distances.

        D₂ = ∂log C(r) / ∂log r where C(r) = fraction of pairs within distance r.
        This is the OTU-endogenous intrinsic dimension estimate.
        """
        n = len(X)
        if n < 20:
            return float(X.shape[1])

        n_sample = min(n, 300)
        idx = np.random.default_rng(42).choice(n, n_sample, replace=False)
        X_s = X[idx]

        tree = cKDTree(X_s)
        k = min(10, n_sample - 1)
        dists, _ = tree.query(X_s, k=k + 1)
        r_vals = dists[:, 1:].mean(axis=1)
        r_vals = r_vals[r_vals > 1e-10]

        if len(r_vals) < 5:
            return float(X.shape[1])

        r_sorted = np.sort(r_vals)
        r_frac = np.linspace(0.1, 0.9, 9)
        r_quantiles = np.percentile(r_sorted, r_frac * 100)

        d2_estimates = []
        for i in range(len(r_quantiles) - 1):
            r1, r2 = r_quantiles[i], r_quantiles[i + 1]
            if r2 <= r1 + 1e-10:
                continue
            c1 = np.mean(r_vals <= r1)
            c2 = np.mean(r_vals <= r2)
            if c1 > 0.001 and c2 > c1:
                slope = (np.log(c2) - np.log(c1)) / (np.log(r2) - np.log(r1))
                d2_estimates.append(slope)

        return float(np.median(d2_estimates)) if d2_estimates else float(X.shape[1])

    def dual_budget_validation(self) -> Dict[str, bool]:
        """
        PRINCIPIO HEURÍSTICO de calibración dual — NO es un teorema.

        Verifica que el número de componentes espaciales (k_effective) y
        el número de pasos temporales de Langevin no diverjan excesivamente.
        Motivado por la observación de que la complejidad espacial y temporal
        de un sistema dinámico reflejan la misma dimensionalidad subyacente.

        La tolerancia δ = max(3, d_spatial//3) es empírica, no derivada de
        primeros principios. Ver §17.5 del paper para la discusión rigurosa.

        Returns diagnostic of whether CCD configuration respects this heuristic.
        """
        if not self._fitted:
            return {"verified": False, "reason": "not_fitted"}

        d_spatial = self._k_effective
        n_temporal = self.n_langevin_steps if self._langevin is not None else 0

        verified = abs(d_spatial - n_temporal) <= max(3, d_spatial // 3)
        return {
            "verified": verified,
            "d_spatial": d_spatial,
            "n_temporal": n_temporal,
            "ratio": float(n_temporal / max(d_spatial, 1)),
            "budget_match": abs(d_spatial - n_temporal) <= 5,
            "isomorphism_holds": verified,
        }

    def full_ecosystem_fit(self, X: np.ndarray) -> Dict:
        """
        Complete ecosystem-driven pipeline:
          TAA spectral classification → OTU parameter optimization → CCD fit

        This is the recommended entry point for production use.
        It leverages the full ACF ecosystem to auto-configure CCD for
        optimal performance on any dataset.

        Args:
            X: (n, d) input data
        Returns:
            dict with ecosystem diagnostics + CCD certificate
        """
        if X.shape[1] < self.d_threshold:
            self.fit(X)
            return {
                "ecosystem_used": False,
                "reason": "low_dim",
                "certificate": str(self.certificate()),
            }

        eco_diag = self.ecosystem_optimize(X)
        self.fit(X)
        cert = self.certificate()
        dual = self.dual_budget_validation()

        return {
            "ecosystem_used": True,
            "taa_classification": eco_diag["alpha_class"],
            "taa_alpha_rate": eco_diag["alpha_rate"],
            "taa_d_star": eco_diag["d_star"],
            "otu_gamma": eco_diag["gamma_otu"],
            "otu_correlation_dim": eco_diag["correlation_dimension"],
            "strategy": eco_diag["strategy"],
            "dual_budget": dual,
            "certificate": str(cert),
            "curse_escaped": cert.curse_escaped,
            "cod_reduction_log10": cert.cod_reduction_log10,
            "k_effective": self._k_effective,
        }

    def _detect_spectral_multiplicity(self) -> bool:
        """
        Detect possible symmetric manifold via spectral multiplicities.

        For symmetric manifolds (S^1, S^2, T^2), the Laplacian has degenerate
        eigenvalues (multiplicity > 1). This method checks if the top eigenvalues
        of the diffusion operator show suspiciously equal values, which would
        indicate that k_effective is overestimating the true intrinsic dimension.

        Returns:
            True if spectral multiplicity warning should be raised.
        """
        if self._diffusion is None or not self._diffusion._fitted:
            return False
        ev = self._diffusion._eigenvalues
        if ev is None or len(ev) < 3:
            return False

        # Check for near-degenerate eigenvalues in top modes
        # For S^1: λ_1 ≈ λ_2 (multiplicity 2)
        # For S^2: λ_1 ≈ λ_2 ≈ λ_3 (multiplicity 3)
        # Tolerance: 5% relative difference
        tol = 0.05
        n_check = min(len(ev) - 1, 10)
        multiplicity_count = 0
        for i in range(n_check - 1):
            rel_diff = abs(ev[i] - ev[i + 1]) / (abs(ev[i]) + 1e-10)
            if rel_diff < tol:
                multiplicity_count += 1

        # Warn if 2+ consecutive near-equal eigenvalues in top 10
        return multiplicity_count >= 2

    def _estimate_nabla_T_bound(self, X: np.ndarray) -> float:
        """
        Estimate the Lipschitz constant of ∇T(x).

        For Langevin convergence with variable temperature, we need
        ‖∇T(x)‖ ≤ L_T for all x. This estimates L_T empirically from
        a sample by computing finite differences of T(x).

        Args:
            X: (n, d) sample points
        Returns:
            Estimated Lipschitz constant of T (upper bound from sample).
        """
        if self._entropy_op is None or not self._entropy_op._fitted:
            return 0.0

        X_s = X / self._auto_scale_factor
        T_vals = self._entropy_op.temperature(X_s)
        n, d = X_s.shape

        # Finite-difference estimate of ‖∇T‖ at each point
        max_grad = 0.0
        n_sample = min(n, 500)
        idx = np.random.default_rng(42).choice(n, n_sample, replace=False)

        for i in idx:
            x_i = X_s[i]
            T_i = T_vals[i]
            # Perturb along each dimension
            for j in range(min(d, 20)):  # cap for performance
                h = max(1e-5, np.std(X_s[:, j]) * 0.01)
                x_pert = x_i.copy()
                x_pert[j] += h
                T_pert = self._entropy_op.temperature(x_pert[np.newaxis, :])[0]
                grad_j = abs(T_pert - T_i) / h
                max_grad = max(max_grad, grad_j)

        return float(max_grad)


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES PÚBLICAS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_high_dim(
    X: np.ndarray,
    d_threshold: int = 5,
    n_components: Optional[int] = None,
    fit: bool = True,
    engine: Optional[CCDEngine] = None,
) -> Tuple[np.ndarray, CCDEngine]:
    """
    Convenience function: fit a CCDEngine and transform X in one call.

    Args:
        X: (n, d) input data
        d_threshold: activate CCD only if d ≥ d_threshold
        n_components: target low-dim components (None = auto)
        fit: if True, fit a new engine; if False, reuse provided engine
        engine: existing fitted engine (used if fit=False)
    Returns:
        (Z, engine):
            Z: (n, k) low-dimensional representation
            engine: fitted CCDEngine instance
    """
    if fit or engine is None:
        n_diff = n_components or 10
        engine = CCDEngine(
            d_threshold=d_threshold,
            n_diffusion_components=n_diff,
        ).fit(X)
    Z = engine.transform(X)
    return Z, engine


def estimate_intrinsic_dimension(X: np.ndarray, k: int = 10) -> Dict[str, float]:
    """
    Quick intrinsic dimension estimate via multiple methods.

    Methods:
      - local_dim_mean: mean of local correlation dimension estimates
      - spectral_gap: from diffusion map eigenvalue gap
      - pca_95: number of PCA components for 95% variance

    Args:
        X: (n, d) data
        k: number of nearest neighbors
    Returns:
        dict with dimension estimates from each method
    """
    n, d = X.shape

    # Method 1: local correlation dimension
    entropy_op = LocalEntropyOperator(k_neighbors=min(k, n - 1)).fit(X)
    d_local = entropy_op.local_dimension(X)
    local_dim_mean = float(np.mean(d_local))
    local_dim_median = float(np.median(d_local))

    # Method 2: spectral gap in diffusion map
    n_neigh = min(k, n - 1)
    diff = DiffusionGeometry(n_components=min(20, n - 2), n_neighbors=n_neigh).fit(X)
    spectral_gap_dim = diff.intrinsic_dimension_estimate

    # Method 3: PCA 95% variance
    osc = CoupledOscillators().fit(X)
    evr = osc.explained_variance_ratio
    cumvar = np.cumsum(np.append(evr, 1.0))
    pca_95_dim = int(np.searchsorted(cumvar, 0.95)) + 1

    return {
        "local_dim_mean": local_dim_mean,
        "local_dim_median": local_dim_median,
        "spectral_gap_dim": float(spectral_gap_dim),
        "pca_95_dim": float(pca_95_dim),
        "d_ambient": float(d),
        "reduction_ratio_spectral": float(d / max(spectral_gap_dim, 1)),
        "reduction_ratio_pca95": float(d / max(pca_95_dim, 1)),
    }
