"""
nova_webcam_tracker.py — NovaStream Webcam: Rastrea movimiento en tiempo real
===============================================================================

Modo tracking con webcam usando NovaConv2D features:
  1. Capturas un frame de referencia con tu puño/mano
  2. Seleccionas la región del puño (ROI)
  3. Nova extrae features de esa región
  4. En cada frame nuevo, busca dónde está el puño comparando features
  5. Dibuja un círculo/rectángulo siguiendo el movimiento

Sin CNN, sin optical flow, sin kalman filter — solo ANOVA(2) features.
"""

import cv2
import numpy as np
import time
from collections import deque

# Importar Nova
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acf_functor.neuron.nova_vision_v2 import NovaConv2D


class NovaWebcamTracker:
    """
    Rastreador de objetos por webcam usando features de NovaConv2D.

    Flujo:
      1. reference_frame → NovaConv2D → ref_features
      2. new_frame → sliding window → NovaConv2D → candidate_features
      3. cosine_similarity(ref_features, candidate_features) → posición
    """

    def __init__(self, roi_size: int = 32, feature_dim: int = 16):
        self.roi_size = roi_size

        # NovaConv2D ligera para features de región
        self.conv = NovaConv2D(
            in_channels=1,
            out_channels=feature_dim,
            kernel_size=5,
            stride=2,
            padding=2,
            max_degree=2,
            l2_lambda=0.1,
            max_pairs=20,
        )

        self.ref_features = None
        self.ref_roi = None
        self._trained = False
        self._trail = deque(maxlen=30)  # Estela de posiciones

    def train_on_reference(self, gray_frame: np.ndarray, roi: tuple = None):
        """
        Entrenar en el frame de referencia.

        Args:
            gray_frame: (H, W) uint8 imagen en escala de grises
            roi: (x, y, w, h) región del puño. Si None, usar frame completo.
        """
        if roi is not None:
            x, y, w, h = roi
            # Asegurar que el ROI es cuadrado y del tamaño correcto
            size = max(w, h)
            cx, cy = x + w//2, y + h//2
            x1 = max(0, cx - size//2)
            y1 = max(0, cy - size//2)
            x2 = min(gray_frame.shape[1], x1 + size)
            y2 = min(gray_frame.shape[0], y1 + size)
            self.ref_roi = (x1, y1, x2-x1, y2-y1)
            patch = gray_frame[y1:y2, x1:x2]
        else:
            h, w = gray_frame.shape
            size = min(h, w)
            y1, x1 = (h-size)//2, (w-size)//2
            self.ref_roi = (x1, y1, size, size)
            patch = gray_frame[y1:y1+size, x1:x1+size]

        # Redimensionar al tamaño de ROI
        patch = cv2.resize(patch, (self.roi_size, self.roi_size))
        patch_f = patch.astype(np.float64) / 255.0

        # Entrenar NovaConv2D en el patch (one-shot)
        # Usamos el patch como dato de entrenamiento (autoencoder)
        X_train = np.tile(patch_f[None, :, :], (8, 1, 1))  # 8 copias con ruido
        X_train += np.random.randn(*X_train.shape) * 0.02
        self.conv.fit(X_train)
        self._trained = True

        # Extraer features de referencia
        self.ref_features = self._extract_features(patch_f)
        return self.ref_features

    def _extract_features(self, patch: np.ndarray) -> np.ndarray:
        """Extraer features NovaConv2D de un parche (roi_size, roi_size) float64."""
        if patch.shape != (self.roi_size, self.roi_size):
            patch = cv2.resize(patch, (self.roi_size, self.roi_size))
        conv_out = self.conv.forward(patch)  # (feature_dim, H', W')
        features = conv_out.mean(axis=(1, 2))  # (feature_dim,)
        return features

    def track(self, gray_frame: np.ndarray, search_radius: int = 40):
        """
        Buscar el objeto en un nuevo frame.

        Args:
            gray_frame: (H, W) uint8
            search_radius: píxeles alrededor de la última posición para buscar

        Returns:
            (cx, cy, score) — centro del objeto detectado y score de similitud
        """
        if not self._trained or self.ref_features is None:
            return None, None, 0.0

        H, W = gray_frame.shape
        s = self.roi_size
        step = 4  # Paso de búsqueda (4 píxeles para velocidad)

        best_score = -1
        best_pos = None

        # Si tenemos posición previa, buscar alrededor
        if len(self._trail) > 0:
            last_cx, last_cy = self._trail[-1]
            x_start = max(0, last_cx - search_radius)
            x_end = min(W - s, last_cx + search_radius)
            y_start = max(0, last_cy - search_radius)
            y_end = min(H - s, last_cy + search_radius)
        else:
            x_start, x_end = 0, W - s
            y_start, y_end = 0, H - s

        # Sliding window
        for y in range(y_start, y_end, step):
            for x in range(x_start, x_end, step):
                patch = gray_frame[y:y+s, x:x+s]
                if patch.shape != (s, s):
                    continue

                patch_f = patch.astype(np.float64) / 255.0
                features = self._extract_features(patch_f)

                # Cosine similarity con referencia
                score = float(np.dot(self.ref_features, features) /
                            (np.linalg.norm(self.ref_features) *
                             np.linalg.norm(features) + 1e-10))

                if score > best_score:
                    best_score = score
                    best_pos = (x + s//2, y + s//2)

        if best_pos is not None:
            self._trail.append(best_pos)

        return (best_pos[0] if best_pos else None,
                best_pos[1] if best_pos else None,
                best_score)


def run_webcam_tracker():
    """
    Ejecutar el tracker de puño/mano con webcam.

    Controles:
      ESPACIO → Capturar frame de referencia (muestra tu puño)
      ESC     → Salir
      R       → Resetear referencia
    """
    print("╔══════════════════════════════════════════════╗")
    print("║   NovaStream Webcam Tracker                 ║")
    print("║   Rastrea tu puño con features ANOVA(2)     ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    print("Controles:")
    print("  ESPACIO → Capturar referencia (muestra tu puño)")
    print("  R       → Resetear referencia")
    print("  ESC     → Salir")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = NovaWebcamTracker(roi_size=48, feature_dim=16)
    ref_set = False
    selecting_roi = False
    roi_start = None
    roi_current = None
    roi_final = None

    print("🎥 Webcam abierta. Pulsa ESPACIO para capturar referencia.")
    print()

    fps_counter = deque(maxlen=30)
    last_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Espejo
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        # FPS
        now = time.perf_counter()
        fps_counter.append(1.0 / max(now - last_time, 0.001))
        last_time = now
        fps = sum(fps_counter) / len(fps_counter)

        if not ref_set:
            # Modo: esperando referencia
            if selecting_roi and roi_start and roi_current:
                x1, y1 = roi_start
                x2, y2 = roi_current
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

            cv2.putText(display, "ESPACIO: capturar referencia",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Modo: trackeando
            cx, cy, score = tracker.track(gray)

            if cx is not None and cy is not None:
                s = tracker.roi_size // 2
                color = (0, 255, 0) if score > 0.5 else (0, 165, 255)
                cv2.rectangle(display,
                             (cx - s, cy - s),
                             (cx + s, cy + s),
                             color, 2)
                cv2.circle(display, (cx, cy), 5, color, -1)

                # Dibujar estela
                for i, (tx, ty) in enumerate(tracker._trail):
                    alpha = (i + 1) / len(tracker._trail)
                    c = int(255 * alpha)
                    cv2.circle(display, (tx, ty), 2, (0, c, 255-c), -1)

                # Score
                cv2.putText(display, f"Score: {score:.0%}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(display, f"Trackeando | R: reset",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FPS counter
        cv2.putText(display, f"FPS: {fps:.0f}",
                   (display.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("NovaStream Webcam Tracker", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord(' '):  # ESPACIO
            if not ref_set:
                print("  📸 Capturando referencia...")
                tracker.train_on_reference(gray)
                ref_set = True
                print(f"  ✅ Referencia capturada | Features: {len(tracker.ref_features)}d")
                print(f"  🎯 Mostrando tracker en pantalla...")
        elif key == ord('r') or key == ord('R'):
            ref_set = False
            tracker = NovaWebcamTracker(roi_size=48, feature_dim=16)
            print("  🔄 Referencia reseteada")

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Tracker cerrado")


if __name__ == "__main__":
    run_webcam_tracker()
