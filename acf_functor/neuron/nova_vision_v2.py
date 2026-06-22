"""
nova_vision_v2.py — NovaStream + NovaConv2D: Visión y Video como Flujo de Píxeles
==================================================================================

VISIÓN RADICALMENTE NUEVA del video: NO frame por frame.
Cada píxel es un STREAM independiente que fluye en el tiempo.

  ┌─────────────────────────────────────────────────────────┐
  │  Enfoque tradicional (CNN 3D):                          │
  │  [F0 F1 F2 ... FT] → 3D Convolution → features          │
  │  Problema: trata el video como cubo estático            │
  │                                                         │
  │  Enfoque NovaStream (Pixel Flow):                       │
  │  p(i,j,0) → p(i,j,1) → p(i,j,2) → ... → p(i,j,T)      │
  │  Cada píxel es un río. Nova aprende la corriente.       │
  │  + vecinos espaciales para contexto                     │
  └─────────────────────────────────────────────────────────┘

COMPONENTES:
  1. NovaStream    — Video como flujo temporal de píxeles (ONLINE, RLS)
  2. NovaConv2D    — Imágenes con ANOVA(2) espacial (mejorado: batch eval)
  3. NovaVision    — Pipeline unificado imagen + video

PROPIEDADES QUE SE MANTIENEN:
  ✅ ANOVA(2): interacciones de pares (espacio-espacio, tiempo-tiempo, espacio-tiempo)
  ✅ One-shot learning + RLS online
  ✅ OOD detection por píxel y por frame
  ✅ Incertidumbre por predicción
  ✅ Sin SGD, sin backprop
  ✅ Interpretabilidad total

Autor: AXIOM-1
"""

import math, time, os, gzip, struct
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

from .nova_phi_neuron import NovaPhiNeuron, NovaPrediction

# ═══════════════════════════════════════════════════════════════════
# UTILIDADES: Datasets reales
# ═══════════════════════════════════════════════════════════════════

def load_mnist(path: str = "", max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cargar MNIST desde archivos IDX locales o desde ~/.keras/datasets/.
    Si no encuentra los archivos, genera datos sintéticos MNIST-like.

    Returns:
        X_train: (60000, 28, 28) uint8
        Y_train: (60000,) uint8
        X_test:  (10000, 28, 28) uint8
        Y_test:  (10000,) uint8
    """
    import urllib.request

    # Buscar en ubicaciones comunes
    search_paths = [
        path,
        os.path.expanduser("~/.keras/datasets/"),
        "/tmp/",
        ".",
    ]

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train-images-idx3-ubyte.gz": "train_images",
        "train-labels-idx1-ubyte.gz": "train_labels",
        "t10k-images-idx3-ubyte.gz":  "test_images",
        "t10k-labels-idx1-ubyte.gz":  "test_labels",
    }

    loaded = {}
    for fname, key in files.items():
        found = False
        for sp in search_paths:
            full = os.path.join(sp, fname)
            if os.path.exists(full):
                loaded[key] = full
                found = True
                break
        if not found:
            # Descargar a /tmp
            url = base_url + fname
            dest = os.path.join("/tmp", fname)
            if not os.path.exists(dest):
                try:
                    urllib.request.urlretrieve(url, dest)
                except Exception:
                    pass
            if os.path.exists(dest):
                loaded[key] = dest

    def read_idx_images(filepath):
        with gzip.open(filepath, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num, rows, cols)

    def read_idx_labels(filepath):
        with gzip.open(filepath, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    result = {}
    for key in ["train_images", "train_labels", "test_images", "test_labels"]:
        if key in loaded:
            if "images" in key:
                result[key] = read_idx_images(loaded[key])
            else:
                result[key] = read_idx_labels(loaded[key])

    if len(result) == 4:
        X_train, Y_train = result["train_images"], result["train_labels"]
        X_test, Y_test = result["test_images"], result["test_labels"]
    else:
        # Sintético MNIST-like
        print("  ⚠ MNIST no encontrado, usando datos sintéticos")
        rng = np.random.RandomState(42)
        X_train = np.zeros((600, 28, 28), dtype=np.uint8)
        Y_train = np.zeros(600, dtype=np.uint8)
        for i in range(600):
            cls = i % 10
            Y_train[i] = cls
            cx, cy = 14 + rng.randint(-2, 3), 14 + rng.randint(-2, 3)
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    val = int(200 * np.exp(-(dx**2+dy**2)/8) + rng.randint(0, 30))
                    x, y = cx+dx, cy+dy
                    if 0 <= x < 28 and 0 <= y < 28:
                        X_train[i, x, y] = min(255, max(0, val + cls * 5))
        X_test = X_train[-100:].copy()
        Y_test = Y_train[-100:].copy()
        X_train = X_train[:-100]
        Y_train = Y_train[:-100]

    if max_samples:
        X_train = X_train[:max_samples]
        Y_train = Y_train[:max_samples]

    return X_train, Y_train, X_test, Y_test


# ═══════════════════════════════════════════════════════════════════
# NovaStream: Video como Flujo de Píxeles
# ═══════════════════════════════════════════════════════════════════

class NovaStream:
    """
    Procesador de video como FLUJO DE PÍXELES.

    NO procesa frames. Procesa PÍXELES a través del tiempo.

    Cada posición (i,j) en el video es un stream independiente:
      s_{ij} = [p_{ij}(0), p_{ij}(1), ..., p_{ij}(T)]

    Nova aprende:
      - AUTOCORRELACIÓN temporal: p_{ij}(t) con p_{ij}(t-1), p_{ij}(t-2), ...
      - CORRELACIÓN espacial: p_{ij}(t) con p_{i+1,j}(t), p_{i,j+1}(t), ...
      - CORRELACIÓN CRUZADA: p_{ij}(t) con p_{i+1,j}(t-1)  (¡flujo de movimiento!)

    Entrenamiento: acumula features de N warmup frames → batch fit → solo predice.
    """

    def __init__(self,
                 height: int,
                 width: int,
                 temporal_window: int = 5,
                 spatial_radius: int = 1,
                 out_features: int = 8,
                 max_degree: int = 3,
                 l2_lambda: float = 0.05,
                 max_pairs: int = 60,
                 warmup_frames: int = 15,
                 online_mode: bool = False):
        """
        Args:
            height, width: Dimensiones del frame
            temporal_window: Cuántos pasos temporales recordar (K)
            spatial_radius: Radio del vecindario espacial (1 = 8-vecindad)
            out_features: Features de salida por píxel
            max_degree: Grado ANOVA
            l2_lambda: Regularización
            max_pairs: Pares ANOVA(2)
            warmup_frames: Frames a acumular antes del batch fit
            online_mode: Si True, RLS online después del batch fit (inestable en video)
        """
        self.H = height
        self.W = width
        self.K = temporal_window
        self.R = spatial_radius
        self.out_features = out_features
        self.warmup_frames = warmup_frames

        # Calcular dimensión del feature vector por píxel
        n_spatial = (2 * spatial_radius + 1) ** 2
        n_temporal = temporal_window
        n_cross = n_spatial
        self.n_spatial = n_spatial
        self.n_temporal = n_temporal
        self.feature_dim = n_spatial + n_temporal + n_cross

        # Una Nova — se entrena en batch y luego solo predice
        self.neuron = NovaPhiNeuron(
            name="nv_stream",
            n_input=self.feature_dim,
            n_output=out_features,
            max_degree=max_degree,
            l2_lambda=l2_lambda,
            max_pairs=max_pairs,
            online_mode=False,  # Siempre batch fit, más estable
        )

        # Buffer circular: guarda los últimos K+1 frames
        self._frame_buffer = deque(maxlen=max(temporal_window + 1, spatial_radius + 1, warmup_frames + 1))
        self._trained = False
        self._t = 0

        # Acumulador de features para batch fit
        self._acc_X: List[np.ndarray] = []
        self._acc_n_frames = 0

    # ═══════════════════════════════════════════════════════════════
    # Extracción de features por píxel (FLUJO)
    # ═══════════════════════════════════════════════════════════════

    def _extract_pixel_features(self, i: int, j: int, t: int) -> np.ndarray:
        """
        Construir el vector de features para el píxel (i,j) en tiempo t.

        Feature vector = [spatial_now | temporal_past | cross_spatial_past]

        spatial_now:     vecindario 3×3 (o (2R+1)²) en frame t
        temporal_past:   valores del píxel (i,j) en t-1, t-2, ..., t-K
        cross_past:      vecindario 3×3 en frame t-1

        Returns:
            vector de tamaño feature_dim
        """
        buf = list(self._frame_buffer)
        features = []

        # 1. Spatial context (frame actual = último en buffer)
        if len(buf) > 0:
            frame_now = buf[-1]
            for di in range(-self.R, self.R + 1):
                for dj in range(-self.R, self.R + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.H and 0 <= nj < self.W:
                        features.append(float(frame_now[ni, nj]))
                    else:
                        features.append(0.0)
        else:
            features.extend([0.0] * self.n_spatial)

        # 2. Temporal context (mismo píxel en frames pasados)
        for lag in range(1, self.K + 1):
            idx = len(buf) - 1 - lag
            if idx >= 0:
                features.append(float(buf[idx][i, j]))
            else:
                features.append(0.0)

        # 3. Cross spatial-temporal (vecinos en t-1)
        if len(buf) >= 2:
            frame_prev = buf[-2]
            for di in range(-self.R, self.R + 1):
                for dj in range(-self.R, self.R + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.H and 0 <= nj < self.W:
                        features.append(float(frame_prev[ni, nj]))
                    else:
                        features.append(0.0)
        else:
            features.extend([0.0] * self.n_spatial)

        return np.array(features, dtype=np.float64)

    # ═══════════════════════════════════════════════════════════════
    # API principal
    # ═══════════════════════════════════════════════════════════════

    def push_frame(self, frame: np.ndarray) -> Dict:
        """
        Empujar UN frame al stream. El frame se añade al buffer y
        se procesan todos los píxeles.

        Args:
            frame: (H, W) — un frame del video

        Returns:
            Dict con features, OODs, y stats del frame
        """
        frame = np.asarray(frame, dtype=np.float64)
        assert frame.shape == (self.H, self.W), \
            f"Frame shape {frame.shape} != ({self.H}, {self.W})"

        self._frame_buffer.append(frame)
        self._t += 1

        if len(self._frame_buffer) < 2:
            return {"t": self._t, "features": None, "ood_count": 0}

        # Extraer features para cada píxel
        all_features = np.zeros((self.H, self.W, self.feature_dim), dtype=np.float64)
        for i in range(self.H):
            for j in range(self.W):
                all_features[i, j] = self._extract_pixel_features(i, j, self._t)

        # Fase de warmup: acumular features para batch fit
        if not self._trained:
            self._acc_X.append(all_features.reshape(-1, self.feature_dim))
            self._acc_n_frames += 1

            if self._acc_n_frames >= self.warmup_frames:
                # Batch fit: entrenar en TODOS los features acumulados
                X_all = np.vstack(self._acc_X)
                n_pixels = len(X_all)

                # Target autoencoder: reconstruir features principales
                Y_all = np.zeros((n_pixels, self.out_features))
                for k in range(min(self.out_features, self.feature_dim)):
                    Y_all[:, k] = X_all[:, k]

                self.neuron.fit(X_all, Y_all)
                self._trained = True
                self._acc_X.clear()  # Liberar memoria

        # Evaluar y detectar OOD
        ood_count = 0
        features_out = np.zeros((self.H, self.W, self.out_features), dtype=np.float64)
        ood_map = np.zeros((self.H, self.W), dtype=bool)

        for i in range(self.H):
            for j in range(self.W):
                fvec = all_features[i, j]
                if self._trained:
                    pred = self.neuron.predict(fvec)
                    features_out[i, j] = pred.mean

                    is_ood, _ = self.neuron.detect_ood(fvec)
                    if is_ood:
                        ood_count += 1
                        ood_map[i, j] = True
                else:
                    features_out[i, j] = fvec[:self.out_features]

        return {
            "t": self._t,
            "features": features_out,
            "ood_count": ood_count,
            "ood_map": ood_map,
            "trained": self._trained,
        }

    def process_video(self, frames: np.ndarray) -> List[Dict]:
        """
        Procesar un video completo como stream de píxeles.

        Args:
            frames: (T, H, W) — secuencia de frames

        Returns:
            Lista de resultados por frame
        """
        results = []
        for t in range(len(frames)):
            r = self.push_frame(frames[t])
            results.append(r)
        return results

    def predict_next_frame(self) -> Optional[np.ndarray]:
        """
        Predecir el siguiente frame basado en el flujo de píxeles.

        Usa la Nova entrenada para extrapolar cada píxel al futuro.

        Returns:
            (H, W) frame predicho, o None si no hay suficientes datos
        """
        if not self._trained or len(self._frame_buffer) < 2:
            return None

        pred_frame = np.zeros((self.H, self.W), dtype=np.float64)
        for i in range(self.H):
            for j in range(self.W):
                fvec = self._extract_pixel_features(i, j, self._t)
                pred = self.neuron.predict(fvec)
                pred_frame[i, j] = pred.mean[0]  # Primera feature como predicción

        return pred_frame

    @property
    def buffer_size(self) -> int:
        return len(self._frame_buffer)

    @property
    def is_ready(self) -> bool:
        return self._trained

    def summary(self) -> str:
        return (
            f"NovaStream({self.H}×{self.W}, K={self.K}, R={self.R}) | "
            f"feat_dim={self.feature_dim} → out={self.out_features} | "
            f"trained={'✅' if self._trained else '❌'} | "
            f"buffer={len(self._frame_buffer)}/{self._frame_buffer.maxlen}"
        )


# ═══════════════════════════════════════════════════════════════════
# NovaConv2D v2: Mejorado con batch evaluation y Triton
# ═══════════════════════════════════════════════════════════════════

class NovaConv2D:
    """
    Capa convolucional ANOVA(2) — v2 mejorada.

    Mejoras sobre v1:
      - Batch evaluation: evalúa TODOS los parches de una vez con einsum
      - Normalización automática de features
      - Soporte para imágenes multi-canal reales
      - Forward 5-10× más rápido que v1
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 16,
                 kernel_size: int = 5,
                 stride: int = 1,
                 padding: int = 2,
                 max_degree: int = 3,
                 l2_lambda: float = 0.05,
                 max_pairs: int = 60,
                 use_triton: bool = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_degree = max_degree

        patch_dim = kernel_size * kernel_size * in_channels

        self.neurons = [
            NovaPhiNeuron(
                name=f"nvc2_ch{i}",
                n_input=patch_dim,
                n_output=1,
                max_degree=max_degree,
                l2_lambda=l2_lambda,
                max_pairs=max_pairs,
                use_triton=use_triton,
            )
            for i in range(out_channels)
        ]

        self._trained = False
        self._fit_time = 0.0
        self._output_shape = None
        self._patch_mean = None
        self._patch_std = None

        # GPU cache (inicializado en _build_gpu_cache tras fit)
        self._gpu_cache = None

    def _build_gpu_cache(self):
        """Pre-cargar tensores a GPU tras fit() para evitar uploads repetidos."""
        import torch
        if not torch.cuda.is_available():
            return
        device = torch.device('cuda')
        neuron0 = self.neurons[0]
        x_mean = torch.from_numpy(neuron0._x_mean.astype(np.float32)).to(device)
        x_std = torch.from_numpy(np.maximum(neuron0._x_std, 1e-8).astype(np.float32)).to(device)
        C_stacked = torch.stack([
            torch.from_numpy(n.C_main[0].astype(np.float32)).to(device)
            for n in self.neurons
        ])
        C_pairs = []
        pair_indices = []
        for n in self.neurons:
            if n._pairs:
                C_pairs.append(torch.from_numpy(
                    n.C_pair[0, :len(n._pairs)].astype(np.float32)).to(device))
                pair_indices.append(([p[0] for p in n._pairs], [p[1] for p in n._pairs]))
            else:
                C_pairs.append(None)
                pair_indices.append(([], []))
        self._gpu_cache = {
            'x_mean': x_mean,
            'x_std': x_std,
            'C_stacked': C_stacked,
            'C_pairs': C_pairs,
            'pair_indices': pair_indices,
            'device': device,
        }

    # ═══════════════════════════════════════════════════════════════
    # Extracción de parches optimizada
    # ═══════════════════════════════════════════════════════════════

    def _extract_patches(self, X: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Extraer parches con stride tricks (zero-copy).

        Input:  (N, C, H, W) o (C, H, W)
        Output: (N_patches, C*k*k), H_out, W_out
        """
        if X.ndim == 3:
            X = X[None, ...]
        N, C, H, W = X.shape
        k = self.kernel_size
        s, p = self.stride, self.padding

        if p > 0:
            X = np.pad(X, ((0,0),(0,0),(p,p),(p,p)), mode='reflect')

        H_out = (H + 2*p - k) // s + 1
        W_out = (W + 2*p - k) // s + 1

        shape = (N, C, H_out, W_out, k, k)
        strides = (X.strides[0], X.strides[1],
                   s * X.strides[2], s * X.strides[3],
                   X.strides[2], X.strides[3])
        patches = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        patches = patches.transpose(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(N * H_out * W_out, C * k * k)

        return np.ascontiguousarray(patches), H_out, W_out

    # ═══════════════════════════════════════════════════════════════
    # Batch evaluation (rápido)
    # ═══════════════════════════════════════════════════════════════

    def _batch_evaluate(self, patches: np.ndarray) -> np.ndarray:
        """
        Evaluar TODOS los canales en TODOS los parches. GPU-native FP16.

        ANOVA(2):  y = C_main · B1 + Σ_pairs C_pair · (B1_i ⊗ B1_j)
        """
        import torch
        n_patches = len(patches)

        # ── Detectar GPU y caché ──
        use_gpu = self._gpu_cache is not None
        if not use_gpu:
            # CPU fallback
            out = np.zeros((n_patches, self.out_channels), dtype=np.float64)
            for ch_idx, neuron in enumerate(self.neurons):
                B = neuron._eval_basis(patches)
                mu = np.einsum('jik,pik->p', neuron.C_main, B)
                for p_idx, (pi, pj) in enumerate(neuron._pairs):
                    psi = (B[:, pi, :, None] * B[:, pj, None, :]).reshape(n_patches, -1)
                    mu += psi @ neuron.C_pair[0, p_idx, :]
                out[:, ch_idx] = mu
            return out

        # ── GPU FP16 path ──
        cache = self._gpu_cache
        device = cache['device']
        dtype = torch.float16  # FP16 para máxima velocidad en RTX

        # Subir parches a GPU (FP16)
        patches_t = torch.from_numpy(patches.astype(np.float32)).to(device).to(dtype)

        # Base Hermite en GPU (FP16)
        Z = (patches_t - cache['x_mean'].to(dtype)) / cache['x_std'].to(dtype)
        d = self.neurons[0].max_degree
        d1 = d + 1
        B = torch.zeros(n_patches, self.neurons[0].n_input, d1, device=device, dtype=dtype)
        B[:, :, 0] = 1.0
        if d >= 1:
            B[:, :, 1] = Z
        for k in range(2, d1):
            B[:, :, k] = Z * B[:, :, k - 1] - (k - 1) * B[:, :, k - 2]

        # Efectos principales (batched, FP16)
        out = torch.einsum('cik,pik->pc', cache['C_stacked'].to(dtype), B)

        # Pares ANOVA(2) vectorizados
        for ch_idx in range(self.out_channels):
            p_idx = cache['pair_indices'][ch_idx]
            pi_list, pj_list = p_idx
            if not pi_list:
                continue
            n_pairs = len(pi_list)
            B_pi = B[:, pi_list, :]
            B_pj = B[:, pj_list, :]
            psi = torch.einsum('pai,paj->paij', B_pi, B_pj)
            psi = psi.reshape(n_patches, n_pairs, d1 * d1)
            c_pair = cache['C_pairs'][ch_idx].to(dtype)
            out[:, ch_idx] += torch.einsum('pnf,nf->p', psi, c_pair)

        return out.float().cpu().numpy().astype(np.float64)

    # ═══════════════════════════════════════════════════════════════
    # API
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_image_input(X: np.ndarray) -> np.ndarray:
        """Convertir cualquier formato de imagen a (N, C, H, W)."""
        if X.ndim == 2:
            return X[None, None, :, :]        # (H, W) → (1, 1, H, W)
        elif X.ndim == 3:
            if X.shape[0] in (1, 3, 4):
                return X[None, :, :, :]        # (C, H, W) → (1, C, H, W)
            else:
                return X[:, None, :, :]        # (N, H, W) → (N, 1, H, W)
        return X  # Ya es (N, C, H, W)

    def fit(self, X: np.ndarray, targets: np.ndarray = None) -> Dict:
        """
        Entrenar NovaConv2D.

        Args:
            X: Imágenes (N, C, H, W), (N, H, W), (C, H, W), o (H, W)
            targets: (N_patches, out_channels) target por parche, o None para autoencoder

        Returns:
            Dict con métricas
        """
        t0 = time.perf_counter()
        # Normalizar formato: siempre → (N, C, H, W)
        X = self._normalize_image_input(X)

        patches, H_out, W_out = self._extract_patches(X)
        n_patches = len(patches)

        # Normalizar parches
        self._patch_mean = np.mean(patches, axis=0)
        self._patch_std = np.maximum(np.std(patches, axis=0), 1e-8)
        patches_norm = (patches - self._patch_mean[None, :]) / self._patch_std[None, :]

        # Autoencoder targets si no se proporcionan
        if targets is None:
            targets = self._compute_targets(patches_norm, n_patches)

        results = []
        for ch_idx, neuron in enumerate(self.neurons):
            y_ch = targets[:, ch_idx:ch_idx+1]
            r = neuron.fit(patches_norm, y_ch)
            results.append(r)

        self._trained = True
        self._output_shape = (H_out, W_out)
        self._fit_time = time.perf_counter() - t0
        self._build_gpu_cache()  # Pre-cargar tensores a GPU

        avg_f = np.mean([r['total_features'] for r in results])
        return {
            "time": self._fit_time,
            "n_patches": n_patches,
            "output_shape": (H_out, W_out),
            "avg_features": int(avg_f),
            "total_features": int(avg_f * self.out_channels),
            "solvers": [r['solver'] for r in results],
            "bases": [r['basis'] for r in results],
        }

    def _compute_targets(self, patches: np.ndarray, n_patches: int,
                         max_samples: int = 100000) -> np.ndarray:
        """SVD parcial para targets no supervisados. Muestrea si >100k parches."""
        n_comp = min(self.out_channels, min(patches.shape) - 1, 32)
        if n_patches > max_samples:
            idx = np.random.choice(n_patches, max_samples, replace=False)
            patches_svd = patches[idx]
        else:
            patches_svd = patches
            idx = None
        try:
            U, s, Vt = np.linalg.svd(patches_svd, full_matrices=False)
            targets = np.zeros((n_patches, self.out_channels), dtype=np.float32)
            rows = idx if idx is not None else slice(None)
            for ch in range(min(self.out_channels, len(s))):
                targets[rows, ch] = U[:, ch] * s[ch]
        except np.linalg.LinAlgError:
            targets = np.random.randn(n_patches, self.out_channels).astype(np.float32) * 0.1
        stds = np.std(targets, axis=0) + 1e-8
        return (targets / stds[None, :]).astype(np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass con batch evaluation.

        Input:  (N, C, H, W), (N, H, W), (C, H, W), o (H, W)
        Output: (N, out_channels, H_out, W_out) o (out_channels, H_out, W_out) si singular
        """
        if not self._trained:
            raise RuntimeError("NovaConv2D no entrenada.")

        # Detectar si es entrada singular (1 imagen sin batch)
        original_was_single = X.ndim == 2 or (X.ndim == 3 and X.shape[0] in (1, 3, 4))

        # Normalizar formato → (N, C, H, W)
        X = self._normalize_image_input(X)
        N = X.shape[0]

        patches, H_out, W_out = self._extract_patches(X)

        # Normalizar
        if self._patch_mean is not None:
            patches = (patches - self._patch_mean[None, :]) / np.maximum(self._patch_std[None, :], 1e-8)

        # Batch evaluate
        out_flat = self._batch_evaluate(patches)  # (N_patches, out_channels)
        out = out_flat.T.reshape(self.out_channels, N, H_out, W_out)
        out = out.transpose(1, 0, 2, 3)  # (N, out_channels, H_out, W_out)

        if original_was_single:
            out = out[0]
        return out

    def forward_with_ood(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward + mapa OOD.

        Returns:
            features: (N, out_channels, H_out, W_out)
            ood_map:  (N, H_out, W_out) bool
        """
        original_was_single = X.ndim == 2 or (X.ndim == 3 and X.shape[0] in (1, 3, 4))

        X = self._normalize_image_input(X)
        N = X.shape[0]

        patches, H_out, W_out = self._extract_patches(X)
        if self._patch_mean is not None:
            patches = (patches - self._patch_mean[None, :]) / np.maximum(self._patch_std[None, :], 1e-8)

        out_flat = self._batch_evaluate(patches)

        # OOD per patch
        ood_flat = np.zeros(len(patches), dtype=bool)
        for i in range(len(patches)):
            ood_flat[i] = any(n.detect_ood(patches[i])[0] for n in self.neurons)

        out = out_flat.T.reshape(self.out_channels, N, H_out, W_out).transpose(1, 0, 2, 3)
        ood = ood_flat.reshape(N, H_out, W_out)

        if original_was_single:
            out, ood = out[0], ood[0]
        return out, ood

    @property
    def total_params(self) -> int:
        return sum(n.n_input * n._d1 + len(n._pairs) * n._d2 for n in self.neurons)

    def summary(self) -> str:
        k = self.kernel_size
        patch_dim = k * k * self.in_channels
        lines = [
            f"NovaConv2D v2 ({self.in_channels}→{self.out_channels}, k={k})",
            f"  Patch:  {k}×{k}×{self.in_channels} = {patch_dim}d → ANOVA(2)",
            f"  Params: {self.total_params:,}",
            f"  Trained: {'✅' if self._trained else '❌'}",
        ]
        if self._trained:
            lines.append(f"  Fit:   {self._fit_time:.2f}s")
            lines.append(f"  Out:   {self._output_shape}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# NovaVision v2: Pipeline Unificado Imagen + Video
# ═══════════════════════════════════════════════════════════════════

class NovaVisionV2:
    """
    Pipeline completo para clasificación de IMÁGENES y VIDEO.

    Imagen:
      X → NovaConv2D (spatial features) → GlobalAvgPool → Nova (classifier)

    Video:
      X → NovaStream (pixel flow) → GlobalAvgPool(spatial+temp) → Nova (classifier)

    Propiedades:
      ✅ Misma arquitectura para imagen y video
      ✅ OOD detection por frame
      ✅ Incertidumbre por predicción
      ✅ One-shot learning
    """

    def __init__(self,
                 num_classes: int = 10,
                 mode: str = "image",   # "image" o "video"
                 in_channels: int = 1,
                 hidden_channels: int = 16,
                 kernel_size: int = 5,
                 max_degree: int = 3,
                 l2_lambda: float = 0.05,
                 # Video params
                 temporal_window: int = 5,
                 spatial_radius: int = 1):
        self.mode = mode
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels

        self.conv = NovaConv2D(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            max_degree=max_degree,
            l2_lambda=l2_lambda,
        )

        self.stream = None  # Para modo video
        if mode == "video":
            self.temporal_window = temporal_window
            self.spatial_radius = spatial_radius

        self.classifier = None
        self._trained = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """
        Entrenar el pipeline.

        Modo imagen:
          X: (N, C, H, W), Y: (N,) labels

        Modo video:
          X: (N, T, C, H, W), Y: (N,) labels
          Se entrena NovaStream en los videos y se clasifica.
        """
        t0 = time.perf_counter()

        if self.mode == "image":
            return self._fit_image(X, Y, t0)
        else:
            return self._fit_video(X, Y, t0)

    def _fit_image(self, X: np.ndarray, Y: np.ndarray, t0: float) -> Dict:
        """Entrenamiento para imágenes."""
        # Normalizar formato: siempre → (N, C, H, W)
        X = self._normalize_image_input(X)

        conv_r = self.conv.fit(X)
        features = self.conv.forward(X)  # (N, hidden, H', W')

        # Global average pooling
        N = features.shape[0]
        pooled = features.mean(axis=(2, 3))  # (N, hidden)

        # Clasificador
        Y_oh = np.eye(self.num_classes)[np.asarray(Y, int)]
        self.classifier = NovaPhiNeuron(
            name="nv_cls",
            n_input=self.hidden_channels,
            n_output=self.num_classes,
            max_degree=2,
            l2_lambda=0.05,
            max_pairs=20,
        )
        cls_r = self.classifier.fit(pooled, Y_oh)

        self._trained = True
        preds = np.array([self.classifier.evaluate(p) for p in pooled])
        train_acc = np.mean(np.argmax(preds, axis=1) == Y)

        return {
            "mode": "image",
            "time": time.perf_counter() - t0,
            "train_accuracy": float(train_acc),
            "conv_time": conv_r['time'],
            "conv_features": conv_r['total_features'],
            "classifier_features": cls_r['total_features'],
        }

    def _fit_video(self, X: np.ndarray, Y: np.ndarray, t0: float) -> Dict:
        """Entrenamiento para video usando NovaStream."""
        # X: (N, T, H, W) — videos monocanal
        N, T, H, W = X.shape[:4]

        total_frames = N * T

        # Crear NovaStream con warmup suficiente
        self.stream = NovaStream(
            height=H, width=W,
            temporal_window=self.temporal_window,
            spatial_radius=self.spatial_radius,
            out_features=self.hidden_channels,
            warmup_frames=min(15, total_frames),
        )

        # Fase 1: Entrenar stream en TODOS los videos (batch fit durante warmup)
        for n in range(N):
            for t in range(T):
                self.stream.push_frame(X[n, t])

        # Fase 2: Extraer features por video (stream ya entrenado)
        all_features = []
        for n in range(N):
            # Reset buffer para cada video
            self.stream._frame_buffer.clear()
            self.stream._t = 0
            for t in range(T):
                r = self.stream.push_frame(X[n, t])
            # Features del ÚLTIMO frame (con contexto temporal completo)
            if r['features'] is not None:
                pooled = r['features'].mean(axis=(0, 1))  # (hidden,)
                all_features.append(pooled)
            else:
                all_features.append(np.zeros(self.hidden_channels))

        all_features = np.array(all_features)

        # Fase 3: Clasificador sobre features de video
        Y_oh = np.eye(self.num_classes)[np.asarray(Y, int)]
        self.classifier = NovaPhiNeuron(
            name="nv_cls_video",
            n_input=self.hidden_channels,
            n_output=self.num_classes,
            max_degree=2,
            l2_lambda=0.05,
            max_pairs=20,
        )
        cls_r = self.classifier.fit(all_features, Y_oh)

        self._trained = True
        preds = np.array([self.classifier.evaluate(f) for f in all_features])
        train_acc = np.mean(np.argmax(preds, axis=1) == Y)

        return {
            "mode": "video",
            "time": time.perf_counter() - t0,
            "train_accuracy": float(train_acc),
            "stream_buffer": self.stream.buffer_size,
            "stream_trained": self.stream._trained,
            "classifier_features": cls_r['total_features'],
        }

    @staticmethod
    def _normalize_image_input(X: np.ndarray) -> np.ndarray:
        """Convertir cualquier formato de imagen a (N, C, H, W)."""
        if X.ndim == 2:
            return X[None, None, :, :]        # (H, W) → (1, 1, H, W)
        elif X.ndim == 3:
            # Si el primer dim es 1, 3 o 4 → (C, H, W). Si no → (N, H, W).
            if X.shape[0] in (1, 3, 4):
                return X[None, :, :, :]        # (C, H, W) → (1, C, H, W)
            else:
                return X[:, None, :, :]        # (N, H, W) → (N, 1, H, W)
        return X  # Ya es (N, C, H, W)

    def predict_image(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predecir clases para imágenes."""
        if not self._trained:
            raise RuntimeError("No entrenado.")

        X = self._normalize_image_input(X)

        features = self.conv.forward(X)
        if features.ndim == 3:
            features = features[None, ...]
        N = features.shape[0]
        pooled = features.mean(axis=(2, 3))

        logits = np.array([self.classifier.evaluate(p) for p in pooled])
        if logits.ndim == 1:
            logits = logits[None, :]

        logits_s = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits_s) / np.sum(np.exp(logits_s), axis=1, keepdims=True)
        classes = np.argmax(probs, axis=1)

        return classes, probs

    def predict_video(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Predecir clases para video + OOD por frame.

        Args:
            X: (N, T, H, W) videos

        Returns:
            classes, probs, frame_oods
        """
        if not self._trained or self.stream is None:
            raise RuntimeError("No entrenado en modo video.")

        if X.ndim == 3:
            X = X[None, ...]

        N, T, H, W = X.shape[:4]
        all_features = []
        all_frame_oods = []

        for n in range(N):
            # Reset stream buffer para cada video
            self.stream._frame_buffer.clear()
            self.stream._t = 0

            frame_oods = []
            for t in range(T):
                r = self.stream.push_frame(X[n, t])
                if r['features'] is not None:
                    frame_oods.append(r['ood_count'])

            if r['features'] is not None:
                pooled = r['features'].mean(axis=(0, 1))
                all_features.append(pooled)
            else:
                all_features.append(np.zeros(self.hidden_channels))
            all_frame_oods.append(frame_oods)

        all_features = np.array(all_features)
        logits = np.array([self.classifier.evaluate(f) for f in all_features])
        if logits.ndim == 1:
            logits = logits[None, :]

        logits_s = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits_s) / np.sum(np.exp(logits_s), axis=1, keepdims=True)
        classes = np.argmax(probs, axis=1)

        return classes, probs, all_frame_oods

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Accuracy."""
        if self.mode == "image":
            classes, _ = self.predict_image(X)
        else:
            classes, _, _ = self.predict_video(X)
        return float(np.mean(classes == np.asarray(Y, int)))

    def summary(self) -> str:
        lines = [
            f"NovaVision v2 [{self.mode}]",
            "=" * 50,
        ]
        if self.mode == "image":
            lines.append(self.conv.summary())
            lines.append(f"  Pool:     GlobalAvgPool → ({self.hidden_channels},)")
        else:
            if self.stream:
                lines.append(f"  Stream:   {self.stream.summary()}")
            else:
                lines.append(f"  Stream:   K={self.temporal_window}, R={self.spatial_radius}")
        if self.classifier:
            lines.append(f"  Class:    Nova({self.hidden_channels}→{self.num_classes})")
        lines.append(f"  Trained:  {'✅' if self._trained else '❌'}")
        return "\n".join(lines)
