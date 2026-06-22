"""
novastream_tracker.py — NovaStream Live Fist Tracker (FastRTC)
==============================================================
2 paneles:
  IZQ: Captura el puño de referencia (entrena NovaConv2D)
  DER: NovaStream Tracking — cámara + rastreo en vivo vía WebRTC

Usa FastRTC (WebRTC nativo UDP) para streaming de baja latencia.
"""

import sys, os, time, numpy as np, cv2, gradio as gr
from fastrtc import WebRTC, AdditionalOutputs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acf_functor.neuron.nova_vision_v2 import NovaConv2D


# ── Panel 1: capturar referencia ──

def train_reference(img, state):
    """Entrena NovaConv2D con el parche central de la imagen capturada."""
    if img is None:
        return None, "❌ Sube o captura una imagen de tu puño.", state

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    H, W = gray.shape
    s = 64
    y1, x1 = max(0, (H - s) // 2), max(0, (W - s) // 2)
    patch = gray[y1:y1 + s, x1:x1 + s].astype(np.float64) / 255.0

    if patch.shape != (s, s):
        return None, f"❌ Imagen muy pequeña ({patch.shape}). Usa al menos {s}x{s}.", state

    t0 = time.perf_counter()
    conv = NovaConv2D(1, 12, 5, 2, 2, max_degree=2, l2_lambda=0.1, max_pairs=15)
    conv.fit(np.tile(patch[None, :, :], (8, 1, 1)) + np.random.randn(8, s, s) * 0.02)
    ref = conv.forward(patch).mean(axis=(1, 2))
    elapsed = time.perf_counter() - t0

    state = state or {}
    state['ref_features'] = ref
    state['conv'] = conv
    state['roi_size'] = s
    state['trained'] = True

    disp = cv2.cvtColor((patch * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    disp = cv2.resize(disp, (128, 128), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(disp, (0, 0), (127, 127), (0, 255, 255), 2)

    return disp, f"✅ Referencia lista en {elapsed:.2f}s ({len(ref)} feat).\n\nEl tracking empieza automáticamente en **NovaStream Tracking**.", state


# ── Panel 2: tracking en vivo ──

def stream_fn(frame, state):
    """WebRTC callback. Devuelve (frame, AdditionalOutputs(score, status, state))."""
    if frame is None:
        return frame, AdditionalOutputs(0.0, "📷 Pulsa **Click to Access Webcam** en NovaStream Tracking.", state)

    if state is None or not state.get('trained'):
        d = frame.copy()
        H, W = d.shape[:2]
        cx, cy = W // 2, H // 2
        cv2.rectangle(d, (cx - 32, cy - 32), (cx + 32, cy + 32), (255, 255, 0), 2)
        cv2.putText(d, "Captura el puno en panel izq", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        return d, AdditionalOutputs(0.0, "⬅️ Captura tu puño en **Captura de Puño** (panel izq)", state)

    # ── tracking ──
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    s = state['roi_size']
    conv = state['conv']
    ref = state['ref_features']

    fm = conv.forward(gray.astype(np.float64) / 255.0)
    rn = ref / (np.linalg.norm(ref) + 1e-10)
    fn = fm / (np.linalg.norm(fm, axis=0) + 1e-10)[None, :, :]
    sm = np.tensordot(rn, fn, axes=([0], [0]))
    idx = np.argmax(sm)
    by, bx = np.unravel_index(idx, sm.shape)
    sc = float(sm[by, bx])

    cx = int(bx * 2 + s // 2)
    cy = int(by * 2 + s // 2)
    cx = max(s // 2, min(W - s // 2, cx))
    cy = max(s // 2, min(H - s // 2, cy))

    d = frame.copy()
    color = (0, 255, 0) if sc > 0.5 else (0, 165, 255)
    thickness = 4 if sc > 0.7 else 2
    cv2.rectangle(d, (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 2), color, thickness)
    cv2.circle(d, (cx, cy), 8, color, -1)
    cv2.putText(d, f"NovaStream: {sc:.0%}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(d, f"({cx},{cy})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return d, AdditionalOutputs(float(sc), f"🔵 Score: {sc:.0%} | ({cx}, {cy})", state)


def reset_tracker(state):
    return None, AdditionalOutputs(0.0, "📷 Captura el puño en el panel izquierdo. El tracking se activa solo.", None)


# ── UI ──

with gr.Blocks(title="NovaStream Live Fist Tracker") as app:
    gr.Markdown("# 🎯 NovaStream — Live Fist Tracker")
    gr.Markdown("### Panel izquierdo: captura → Panel derecho: rastreo en vivo")

    ts = gr.State(None)

    with gr.Row(equal_height=True):
        # ── PANEL IZQUIERDO: Captura de referencia ──
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📸 Captura de Puño")
            ref_img = gr.Image(label="Muestra tu puño centrado", type="numpy",
                               sources=["webcam", "upload"])
            train_btn = gr.Button("🎯 Entrenar NovaStream", variant="primary", size="lg")
            ref_output = gr.Image(label="Parche capturado (64×64)", height=150)
            ref_status = gr.Markdown("⬆️ Muestra tu puño centrado y pulsa **Entrenar NovaStream**")

        # ── PANEL DERECHO: Tracking en vivo ──
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 🔴 NovaStream Tracking")
            webrtc = WebRTC(label="NovaStream Tracking", mode="send-receive",
                            mirror_webcam=True, height=400,
                            full_screen=False)
            with gr.Row():
                reset_btn = gr.Button("🔁 Reset", variant="secondary")
            score = gr.Number(label="Score", value=0.0, precision=4)
            track_status = gr.Markdown("⬅️ Captura tu puño en el panel izq. El tracking se activa solo.")

    # ── Eventos ──

    train_btn.click(train_reference, inputs=[ref_img, ts],
                    outputs=[ref_output, ref_status, ts])

    webrtc.stream(stream_fn, inputs=[webrtc, ts],
                  outputs=[webrtc, score, track_status, ts],
                  time_limit=120, concurrency_limit=1)

    reset_btn.click(reset_tracker, inputs=[ts],
                    outputs=[webrtc, score, track_status, ts])

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7861, share=False)
