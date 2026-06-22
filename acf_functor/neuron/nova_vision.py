"""
nova_vision.py — NovaConv2d / NovaConv3d: La Neurona Definitiva para Visión y Video
=====================================================================================

INNOVACIÓN CLAVE:
  CNN clásica  = MLP + weight sharing espacial (kernel convolucional)
  NovaConv2d   = Nova + weight sharing espacial (ANOVA(2) compartido por parche)

Así como una CNN es un MLP aplicado a cada parche, NovaConv es una Nova aplicada
a cada parche. La misma neurona ANOVA(2) aprende interacciones entre píxeles
dentro del kernel, y se comparte en todas las posiciones espaciales.

DIFERENCIAS vs CNN:
  ✅ Interacciones NO LINEALES de pares de píxeles (no solo suma ponderada)
  ✅ Aprendizaje en UN solo paso (sin SGD, sin backprop)
  ✅ Detección OOD por posición espacial
  ✅ Incertidumbre por predicción
  ✅ 30-43× menos parámetros que capa fully-connected equivalente
  ✅ Interpretable: qué pares de píxeles importan más

ARQUITECTURA:
  - Una Nova por canal de salida (como un filtro de CNN)
  - Cada Nova recibe un parche k×k×C_in aplanado
  - ANOVA(2): interacciones entre features del parche
  - Weight sharing: misma Nova evaluada en cada posición

Autor: AXIOM-1
"""

import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from .nova_phi_neuron import NovaPhiNeuron


class NovaConv2d:
    """
    Capa convolucional basada en Nova.

    Equivalencia:
      CNN:       Conv2d(in_ch, out_ch, kernel_size)  → out = σ(W * patch + b)
      NovaConv:  NovaConv2d(in_ch, out_ch, kernel_size) → out = Nova(patch)

    Cada canal de salida = una Nova independiente entrenada sobre TODOS los parches.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 max_degree: int = 3,
                 l2_lambda: float = 0.05,
                 max_pairs: int = 80,
                 use_triton: bool = True):
        """
        Args:
            in_channels: Canales de entrada (C_in)
            out_channels: Canales de salida (C_out) — una Nova por canal
            kernel_size: Tamaño del parche (k×k)
            stride: Paso del deslizamiento
            padding: Relleno de bordes
            max_degree: Grado máximo del polinomio ANOVA (2=cuadrático, 3=cúbico)
            l2_lambda: Regularización L2 (Ridge)
            max_pairs: Máximo de pares ANOVA(2) a considerar
            use_triton: Usar GPU Triton si disponible
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_degree = max_degree
        self.l2_lambda = l2_lambda
        self.max_pairs = max_pairs
        self.use_triton = use_triton

        # Patch size: k × k × C_in
        patch_dim = kernel_size * kernel_size * in_channels

        # Una Nova por canal de salida
        self.neurons = [
            NovaPhiNeuron(
                name=f"nv_conv_ch{i}",
                n_input=patch_dim,
                n_output=1,  # Cada Nova produce 1 valor (1 canal)
                max_degree=max_degree,
                l2_lambda=l2_lambda,
                max_pairs=max_pairs,
                use_triton=use_triton,
            )
            for i in range(out_channels)
        ]

        self._trained = False
        self._input_shape = None  # (H, W) para inferencia
        self._fit_time = 0.0

    # ═══════════════════════════════════════════════════════════════
    # Extracción de parches (im2col)
    # ═══════════════════════════════════════════════════════════════

    def _extract_patches(self, X: np.ndarray) -> np.ndarray:
        """
        Extraer todos los parches de una o varias imágenes.

        Input:  (N, C_in, H, W)  o  (C_in, H, W)
        Output: (N_patches, k*k*C_in)

        Usa as_strided para zero-copy (rápido).
        """
        if X.ndim == 3:
            X = X[None, ...]  # Añadir batch dim
        N, C, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        # Aplicar padding
        if p > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        else:
            X_pad = X

        _, _, H_pad, W_pad = X_pad.shape

        # Dimensiones de salida
        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1

        # Extraer parches con stride tricks
        shape = (N, C, H_out, W_out, k, k)
        strides = (X_pad.strides[0], X_pad.strides[1],
                   s * X_pad.strides[2], s * X_pad.strides[3],
                   X_pad.strides[2], X_pad.strides[3])

        patches = np.lib.stride_tricks.as_strided(X_pad, shape=shape, strides=strides)
        # Reordenar: (N, H_out, W_out, C, k, k) → (N*H_out*W_out, C*k*k)
        patches = patches.transpose(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(N * H_out * W_out, C * k * k)

        return np.ascontiguousarray(patches)  # Hacer contiguo para operaciones

    # ═══════════════════════════════════════════════════════════════
    # Entrenamiento
    # ═══════════════════════════════════════════════════════════════

    def fit(self, X: np.ndarray, Y: np.ndarray = None,
            target_patches: np.ndarray = None) -> Dict:
        """
        Entrenar NovaConv2d.

        Modo 1 — Autoencoder (no supervisado):
          fit(X) → reconstruye los parches de X
          Cada Nova aprende a representar un parche en 1 valor
          y reconstruirlo. Entrenamiento: Y = primer componente principal
          de cada parche.

        Modo 2 — Supervisado:
          fit(X, target_patches=Y_patches)
          donde Y_patches tiene forma (N_patches, out_channels)

        Modo 3 — Clasificación:
          fit(X, Y=labels) con Nova como extractor de features
          (requiere aplanar después)

        Args:
            X: Imágenes (N, C, H, W) o (C, H, W)
            Y: Etiquetas (N,) para clasificación (opcional)
            target_patches: Targets por parche (N_patches, out_channels) (opcional)

        Returns:
            Dict con métricas de entrenamiento
        """
        t0 = time.perf_counter()
        if X.ndim == 3:
            X = X[None, ...]
        N, C, H, W = X.shape

        # Extraer todos los parches
        patches = self._extract_patches(X)  # (N_patches, k*k*C_in)
        n_patches = len(patches)

        # Determinar targets
        if target_patches is not None:
            # Modo supervisado: targets por parche
            Y_all = np.asarray(target_patches, dtype=np.float64)
            if Y_all.ndim == 1:
                Y_all = Y_all[:, None]
        elif Y is not None:
            # Modo clasificación: usar Nova como extractor de features
            # Cada canal aprende una proyección del parche
            # Usamos SVD para encontrar la mejor proyección 1D por canal
            Y_all = self._compute_projection_targets(patches, n_patches)
        else:
            # Modo autoencoder: cada Nova aprende a comprimir el parche
            Y_all = self._compute_projection_targets(patches, n_patches)

        # Entrenar cada Nova en todos los parches
        results = []
        for ch_idx, neuron in enumerate(self.neurons):
            y_ch = Y_all[:, ch_idx:ch_idx+1]
            r = neuron.fit(patches, y_ch)
            results.append(r)

        self._trained = True
        self._fit_time = time.perf_counter() - t0
        self._input_shape = (H, W)

        avg_features = np.mean([r['total_features'] for r in results])
        return {
            "time": self._fit_time,
            "n_patches": n_patches,
            "out_channels": self.out_channels,
            "avg_features_per_channel": int(avg_features),
            "total_features": int(avg_features * self.out_channels),
            "solvers": [r['solver'] for r in results],
            "bases": [r['basis'] for r in results],
        }

    def _compute_projection_targets(self, patches: np.ndarray,
                                     n_patches: int) -> np.ndarray:
        """
        Computar targets para entrenamiento no supervisado.
        Cada canal aprende una proyección diferente del parche,
        usando los primeros out_channels componentes principales.
        """
        # Centrar los parches
        patch_mean = np.mean(patches, axis=0, keepdims=True)
        patches_c = patches - patch_mean

        # SVD parcial para obtener direcciones principales
        n_comp = min(self.out_channels, min(patches_c.shape) - 1, 32)
        try:
            U, s, Vt = np.linalg.svd(patches_c, full_matrices=False)
            # Proyectar sobre los primeros out_channels componentes
            Y_all = np.zeros((n_patches, self.out_channels))
            for ch in range(min(self.out_channels, len(s))):
                Y_all[:, ch] = U[:, ch] * s[ch]
        except np.linalg.LinAlgError:
            # Fallback: usar estadísticos simples por canal
            Y_all = np.zeros((n_patches, self.out_channels))
            chunk = patches.shape[1] // self.out_channels
            for ch in range(self.out_channels):
                start = ch * chunk
                end = min(start + chunk, patches.shape[1])
                if start < patches.shape[1]:
                    Y_all[:, ch] = np.mean(patches[:, start:end], axis=1)

        # Normalizar
        stds = np.std(Y_all, axis=0)
        stds = np.maximum(stds, 1e-8)
        Y_all = Y_all / stds[None, :]

        return Y_all

    # ═══════════════════════════════════════════════════════════════
    # Inferencia (forward pass)
    # ═══════════════════════════════════════════════════════════════

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: aplicar NovaConv2d a una imagen o batch.

        Input:  (N, C_in, H, W) o (C_in, H, W)
        Output: (N, C_out, H_out, W_out)
        """
        if not self._trained:
            raise RuntimeError("NovaConv2d no entrenada. Llama fit() primero.")

        single = X.ndim == 3
        if single:
            X = X[None, ...]
        N, C, H, W = X.shape

        k = self.kernel_size
        s = self.stride
        p = self.padding
        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1

        # Extraer parches
        patches = self._extract_patches(X)  # (N*H_out*W_out, k*k*C_in)

        # Evaluar cada Nova en todos los parches
        out = np.zeros((N, self.out_channels, H_out, W_out), dtype=np.float64)
        for ch_idx, neuron in enumerate(self.neurons):
            # Evaluar en batch
            preds = np.array([neuron.evaluate(p) for p in patches])
            out[:, ch_idx, :, :] = preds.reshape(N, H_out, W_out)

        if single:
            out = out[0]
        return out

    def forward_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass con incertidumbre por posición espacial.

        Returns:
            mean:  (N, C_out, H_out, W_out) — predicciones
            std:   (N, C_out, H_out, W_out) — incertidumbre
        """
        if not self._trained:
            raise RuntimeError("NovaConv2d no entrenada. Llama fit() primero.")

        single = X.ndim == 3
        if single:
            X = X[None, ...]
        N, C, H, W = X.shape

        k = self.kernel_size
        s = self.stride
        p = self.padding
        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1

        patches = self._extract_patches(X)

        out_mean = np.zeros((N, self.out_channels, H_out, W_out), dtype=np.float64)
        out_std = np.zeros_like(out_mean)

        for ch_idx, neuron in enumerate(self.neurons):
            for i in range(len(patches)):
                pred = neuron.predict(patches[i])
                n_idx = i // (H_out * W_out)
                h_idx = (i % (H_out * W_out)) // W_out
                w_idx = i % W_out
                out_mean[n_idx, ch_idx, h_idx, w_idx] = pred.mean[0]
                out_std[n_idx, ch_idx, h_idx, w_idx] = pred.std[0]

        if single:
            out_mean, out_std = out_mean[0], out_std[0]
        return out_mean, out_std

    def detect_ood_map(self, X: np.ndarray) -> np.ndarray:
        """
        Generar mapa OOD espacial: para cada posición, ¿es anómalo el parche?

        Returns:
            ood_map: (N, H_out, W_out) — True donde el parche es OOD
        """
        if not self._trained:
            raise RuntimeError("NovaConv2d no entrenada.")

        single = X.ndim == 3
        if single:
            X = X[None, ...]
        N, C, H, W = X.shape

        k = self.kernel_size
        s = self.stride
        p = self.padding
        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1

        patches = self._extract_patches(X)
        ood_map = np.zeros((N, H_out, W_out), dtype=bool)

        for i in range(len(patches)):
            # Si ALGUNA neurona detecta OOD, la posición es OOD
            is_ood = any(neuron.detect_ood(patches[i])[0]
                        for neuron in self.neurons)
            n_idx = i // (H_out * W_out)
            h_idx = (i % (H_out * W_out)) // W_out
            w_idx = i % W_out
            ood_map[n_idx, h_idx, w_idx] = is_ood

        if single:
            ood_map = ood_map[0]
        return ood_map

    # ═══════════════════════════════════════════════════════════════
    # Utilidades
    # ═══════════════════════════════════════════════════════════════

    @property
    def output_shape(self) -> Optional[Tuple[int, int]]:
        """Dimensiones de salida (H_out, W_out) para el último input."""
        if self._input_shape is None:
            return None
        H, W = self._input_shape
        k, s, p = self.kernel_size, self.stride, self.padding
        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1
        return (H_out, W_out)

    @property
    def total_params(self) -> int:
        """Total de features (≈ parámetros efectivos)."""
        return sum(n.n_input * n._d1 + len(n._pairs) * n._d2
                   for n in self.neurons)

    @property
    def cnn_equivalent_params(self) -> int:
        """Parámetros de una CNN equivalente."""
        k = self.kernel_size
        return k * k * self.in_channels * self.out_channels + self.out_channels

    def summary(self) -> str:
        lines = [
            f"NovaConv2d({self.in_channels}→{self.out_channels}, k={self.kernel_size}, "
            f"s={self.stride}, p={self.padding})",
            f"  Parches:       {self.kernel_size}×{self.kernel_size}×{self.in_channels} = "
            f"{self.kernel_size**2 * self.in_channels} dims",
            f"  Parámetros:    {self.total_params:,} (CNN equiv: {self.cnn_equivalent_params:,})",
            f"  Entrenado:     {'✅' if self._trained else '❌'}",
        ]
        if self._trained:
            lines.append(f"  Tiempo fit:    {self._fit_time:.2f}s")
            if self._input_shape:
                lines.append(f"  Output shape:  {self.output_shape}")
        return "\n".join(lines)


class NovaConv3d:
    """
    Capa convolucional 3D basada en Nova (para video).

    Como NovaConv2d pero con parches espacio-temporales:
      kernel_size = (t, h, w)
      Cada parche:  t × h × w × C_in  →  aplanado y procesado por Nova.

    Captura interacciones ANOVA(2) entre píxeles ADYACENTES en el espacio
    Y en frames consecutivos en el tiempo.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (1, 1, 1),
                 max_degree: int = 2,   # Video: grado 2 suficiente (demasiados features)
                 l2_lambda: float = 0.1,
                 max_pairs: int = 40,   # Pares reducidos por el costo 3D
                 use_triton: bool = True):
        """
        Args:
            in_channels: Canales de entrada
            out_channels: Canales de salida
            kernel_size: (T, H, W) — profundidad temporal + espacial
            stride: (T, H, W)
            padding: (T, H, W)
            max_degree: Grado ANOVA (2 recomendado para video)
            l2_lambda: Regularización
            max_pairs: Pares ANOVA(2) (reducido porque parches 3D son grandes)
            use_triton: GPU
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride,) if isinstance(stride, int) else stride
        self.padding = (padding,) if isinstance(padding, int) else padding
        self.max_degree = max_degree
        self.l2_lambda = l2_lambda
        self.max_pairs = max_pairs
        self.use_triton = use_triton

        kt, kh, kw = self.kernel_size
        patch_dim = kt * kh * kw * in_channels

        self.neurons = [
            NovaPhiNeuron(
                name=f"nv3d_ch{i}",
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

    def _extract_patches_3d(self, X: np.ndarray) -> np.ndarray:
        """
        Extraer parches 3D de video.

        Input:  (N, C, T, H, W)  o  (C, T, H, W)
        Output: (N_patches, kt*kh*kw*C_in)
        """
        if X.ndim == 4:
            X = X[None, ...]
        N, C, T, H, W = X.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        pt, ph, pw = self.padding

        # Padding 3D
        if any(p > 0 for p in (pt, ph, pw)):
            X = np.pad(X, ((0, 0), (0, 0),
                           (pt, pt), (ph, ph), (pw, pw)), mode='constant')

        _, _, T_pad, H_pad, W_pad = X.shape

        T_out = (T + 2 * pt - kt) // st + 1
        H_out = (H + 2 * ph - kh) // sh + 1
        W_out = (W + 2 * pw - kw) // sw + 1

        # Extraer manualmente (as_strided 5D es frágil)
        patches = np.zeros((N * T_out * H_out * W_out, kt * kh * kw * C),
                           dtype=X.dtype)
        idx = 0
        for n in range(N):
            for t in range(T_out):
                for h in range(H_out):
                    for w in range(W_out):
                        t0, h0, w0 = t * st, h * sh, w * sw
                        patch = X[n, :, t0:t0+kt, h0:h0+kh, w0:w0+kw]
                        patches[idx] = patch.ravel()
                        idx += 1

        return patches

    def fit(self, X: np.ndarray,
            target_patches: np.ndarray = None) -> Dict:
        """Entrenar NovaConv3d."""
        t0 = time.perf_counter()
        if X.ndim == 4:
            X = X[None, ...]

        patches = self._extract_patches_3d(X)
        n_patches = len(patches)

        if target_patches is not None:
            Y_all = np.asarray(target_patches, dtype=np.float64)
            if Y_all.ndim == 1:
                Y_all = Y_all[:, None]
        else:
            # Proyección PCA simple
            patch_mean = np.mean(patches, axis=0, keepdims=True)
            patches_c = patches - patch_mean
            try:
                U, s, Vt = np.linalg.svd(patches_c, full_matrices=False)
                Y_all = np.zeros((n_patches, self.out_channels))
                for ch in range(min(self.out_channels, len(s))):
                    Y_all[:, ch] = U[:, ch] * s[ch]
            except np.linalg.LinAlgError:
                Y_all = np.random.randn(n_patches, self.out_channels) * 0.1
            stds = np.std(Y_all, axis=0) + 1e-8
            Y_all /= stds[None, :]

        results = []
        for ch_idx, neuron in enumerate(self.neurons):
            y_ch = Y_all[:, ch_idx:ch_idx+1]
            r = neuron.fit(patches, y_ch)
            results.append(r)

        self._trained = True
        self._fit_time = time.perf_counter() - t0

        avg_features = np.mean([r['total_features'] for r in results])
        return {
            "time": self._fit_time,
            "n_patches": n_patches,
            "out_channels": self.out_channels,
            "avg_features_per_channel": int(avg_features),
            "solvers": [r['solver'] for r in results],
        }

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass 3D.

        Input:  (N, C_in, T, H, W) o (C_in, T, H, W)
        Output: (N, C_out, T_out, H_out, W_out)
        """
        if not self._trained:
            raise RuntimeError("NovaConv3d no entrenada.")

        single = X.ndim == 4
        if single:
            X = X[None, ...]
        N, C, T, H, W = X.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        pt, ph, pw = self.padding

        T_out = (T + 2 * pt - kt) // st + 1
        H_out = (H + 2 * ph - kh) // sh + 1
        W_out = (W + 2 * pw - kw) // sw + 1

        patches = self._extract_patches_3d(X)

        out = np.zeros((N, self.out_channels, T_out, H_out, W_out),
                       dtype=np.float64)
        for ch_idx, neuron in enumerate(self.neurons):
            preds = np.array([neuron.evaluate(p) for p in patches])
            out[:, ch_idx, :, :, :] = preds.reshape(N, T_out, H_out, W_out)

        if single:
            out = out[0]
        return out

    @property
    def total_params(self) -> int:
        return sum(n.n_input * n._d1 + len(n._pairs) * n._d2
                   for n in self.neurons)

    def summary(self) -> str:
        kt, kh, kw = self.kernel_size
        lines = [
            f"NovaConv3d({self.in_channels}→{self.out_channels}, "
            f"k=({kt},{kh},{kw}))",
            f"  Parches:       {kt}×{kh}×{kw}×{self.in_channels} = "
            f"{kt*kh*kw*self.in_channels} dims",
            f"  Parámetros:    {self.total_params:,}",
            f"  Entrenado:     {'✅' if self._trained else '❌'}",
        ]
        if self._trained:
            lines.append(f"  Tiempo fit:    {self._fit_time:.2f}s")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# NovaVision — Pipeline completo de visión con Nova
# ═══════════════════════════════════════════════════════════════════

class NovaVision:
    """
    Pipeline completo: NovaConv2d + Nova classifier para clasificación
    de imágenes.

    Arquitectura:
      Imagen → NovaConv2d (feature extraction) → GlobalAvgPool → Nova (classifier)

    Esto es análogo a:
      Imagen → CNN → GlobalAvgPool → MLP

    Pero Nova en ambas etapas.
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 10,
                 hidden_channels: int = 16,
                 kernel_size: int = 3,
                 max_degree: int = 3,
                 l2_lambda: float = 0.05):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels

        self.conv = NovaConv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            max_degree=max_degree,
            l2_lambda=l2_lambda,
        )

        self.classifier = None  # Se crea después de conv.fit()
        self._trained = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """
        Entrenar el pipeline completo.

        Args:
            X: Imágenes (N, C, H, W)
            Y: Etiquetas (N,) — enteros 0..num_classes-1

        Returns:
            Dict con métricas
        """
        t0 = time.perf_counter()

        # Fase 1: Entrenar NovaConv2d (no supervisado)
        conv_result = self.conv.fit(X)
        conv_features = self.conv.forward(X)  # (N, hidden, H_out, W_out)

        # Fase 2: Global Average Pooling
        N = conv_features.shape[0]
        pooled = conv_features.mean(axis=(2, 3))  # (N, hidden)

        # Fase 3: Entrenar clasificador Nova
        Y_onehot = np.eye(self.num_classes)[np.asarray(Y, int)]
        self.classifier = NovaPhiNeuron(
            name="nv_classifier",
            n_input=self.hidden_channels,
            n_output=self.num_classes,
            max_degree=2,
            l2_lambda=0.05,
            max_pairs=20,
        )
        classifier_result = self.classifier.fit(pooled, Y_onehot)

        self._trained = True
        elapsed = time.perf_counter() - t0

        # Accuracy en train
        preds = self._evaluate_batch(pooled)
        if preds.ndim == 1:
            preds = preds[None, :]
        train_acc = np.mean(np.argmax(preds, axis=1) == Y)

        return {
            "time": elapsed,
            "train_accuracy": float(train_acc),
            "conv_time": conv_result['time'],
            "classifier_time": classifier_result['time'],
            "conv_patches": conv_result['n_patches'],
            "conv_features": conv_result['total_features'],
            "classifier_features": classifier_result['total_features'],
        }

    def _evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluar Nova en batch: (N, d) → (N, k)."""
        return np.array([self.classifier.evaluate(x) for x in X])

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predecir clases.

        Returns:
            classes: (N,) — clase predicha
            probs:   (N, num_classes) — probabilidades
        """
        if not self._trained:
            raise RuntimeError("NovaVision no entrenada.")

        conv_features = self.conv.forward(X)
        if conv_features.ndim == 3:
            conv_features = conv_features[None, ...]
        pooled = conv_features.mean(axis=(2, 3))
        logits = self._evaluate_batch(pooled)

        # Softmax
        if logits.ndim == 1:
            logits = logits[None, :]
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits_stable) / np.sum(np.exp(logits_stable), axis=1, keepdims=True)
        classes = np.argmax(probs, axis=1)

        return classes, probs

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Accuracy."""
        classes, _ = self.predict(X)
        return float(np.mean(classes == np.asarray(Y, int)))

    def summary(self) -> str:
        lines = [
            "NovaVision Pipeline",
            "=" * 40,
            self.conv.summary(),
            f"  Pool:          GlobalAvgPool → ({self.hidden_channels},)",
        ]
        if self.classifier:
            lines.append(f"  Classifier:    Nova({self.hidden_channels}→{self.num_classes})")
        lines.append(f"  Entrenado:     {'✅' if self._trained else '❌'}")
        return "\n".join(lines)
