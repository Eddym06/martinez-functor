"""
nova_glassbox.py — Instrumentación Glassbox para Nova ACF
==========================================================
Sensores, visores y analizadores en tiempo real para
entender EXACTAMENTE qué hace Nova durante el entrenamiento.

Principio ACF: Glassbox significa que podemos ver TODO.
No hay caja negra. Cada matriz, cada valor propio, cada
operación matemática es visible y analizable.

SENSORES:
  - TimingSensor: per-fase, per-capa, per-operación
  - SpectralSensor: distribución de valores propios, condicionamiento
  - InformationSensor: flujo de información entre capas
  - BottleneckDetector: qué operaciones son más lentas
  - MemorySensor: uso de RAM/VRAM por fase

VISORES:
  - PhaseTimeline: línea de tiempo de fases
  - LayerHeatmap: actividad por capa y nivel
  - PairCorrelationMap: correlación de pares ANOVA
  - BusInformationFlow: cómo fluye info en el KnowledgeBus
"""

import time
import threading
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

def _json_safe(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 📊 GLASSBOX SENSOR CORE
# ═══════════════════════════════════════════════════════════════

class GlassboxSensor:
    """Sensor base para instrumentación glassbox.
    
    Cada sensor se engancha en puntos específicos del entrenamiento
    y recolecta datos SIN afectar el rendimiento (operaciones O(1)).
    """
    def __init__(self, name: str):
        self.name = name
        self.data = []
        self._start_time = None
        
    def start(self):
        self._start_time = time.perf_counter()
        
    def record(self, **kwargs):
        kwargs['timestamp'] = time.perf_counter()
        self.data.append(kwargs)
        
    def summary(self) -> dict:
        return {"name": self.name, "n_records": len(self.data)}


class TimingSensor(GlassboxSensor):
    """Mide tiempo de cada fase, capa, y operación.
    
    Jerarquía: Phase > Layer > Operation
    """
    def __init__(self):
        super().__init__("timing")
        self._phase_stack = []  # [(name, start_time)]
        self._layer_times = defaultdict(list)
        self._op_times = defaultdict(list)
        self.phases = {}
        
    def enter_phase(self, name: str):
        self._phase_stack.append((name, time.perf_counter()))
        
    def exit_phase(self):
        if self._phase_stack:
            name, t0 = self._phase_stack.pop()
            elapsed = time.perf_counter() - t0
            self.phases[name] = self.phases.get(name, 0) + elapsed
            self.record(phase=name, duration=elapsed)
            
    def record_layer(self, layer_idx: int, duration: float, 
                     basis: str = "?", n_levels: int = 0):
        self._layer_times[layer_idx].append(duration)
        self.record(type="layer", layer=layer_idx, duration=duration,
                    basis=basis, n_levels=n_levels)
        
    def record_op(self, op_name: str, duration: float, hardware: str = "?"):
        self._op_times[op_name].append(duration)
        self.record(type="op", op=op_name, duration=duration, hardware=hardware)
        
    def summary(self) -> dict:
        base = super().summary()
        base["phases"] = {k: f"{v:.2f}s" for k, v in self.phases.items()}
        base["layers_avg"] = {k: f"{np.mean(v):.3f}s" for k, v in self._layer_times.items()}
        base["ops_avg"] = {k: f"{np.mean(v)*1000:.2f}ms" for k, v in self._op_times.items()}
        base["total"] = f"{sum(self.phases.values()):.2f}s"
        
        # Encontrar bottleneck
        if self._op_times:
            total_ops = {k: sum(v) for k, v in self._op_times.items()}
            bottleneck = max(total_ops, key=total_ops.get)
            base["bottleneck"] = f"{bottleneck} ({total_ops[bottleneck]:.2f}s total)"
        return base


class SpectralSensor(GlassboxSensor):
    """Analiza propiedades espectrales de matrices Phi.
    
    Mide:
      - Condicionamiento κ = σ_max / σ_min
      - Entropía espectral H = -Σ p_i log p_i
      - Rango efectivo (nº de valores propios > umbral)
      - Distribución de valores singulares
    """
    def __init__(self):
        super().__init__("spectral")
        self.condition_numbers = []
        self.spectral_entropies = []
        self.effective_ranks = []
        
    def record_phi(self, name: str, Phi: np.ndarray, 
                   max_svd: int = 50):
        """Analizar matriz Phi (N, F)."""
        try:
            # SVD rápido (primeros max_svd valores singulares)
            if Phi.shape[0] > 2000:
                idx = np.random.choice(Phi.shape[0], 2000, replace=False)
                Phi_s = Phi[idx]
            else:
                Phi_s = Phi
                
            _, S, _ = np.linalg.svd(Phi_s.astype(np.float64), full_matrices=False)
            S = S[:min(len(S), max_svd)]
            
            # Condicionamiento
            kappa = float(S[0] / max(S[-1], 1e-12))
            self.condition_numbers.append(kappa)
            
            # Entropía espectral
            p = S / (S.sum() + 1e-12)
            H = float(-np.sum(p * np.log(p + 1e-12)))
            self.spectral_entropies.append(H)
            
            # Rango efectivo (valores > 1% del máximo)
            eff_rank = int(np.sum(S > 0.01 * S[0]))
            self.effective_ranks.append(eff_rank)
            
            self.record(
                name=name, kappa=kappa, H_spectral=H,
                eff_rank=eff_rank, sigma_max=float(S[0]),
                sigma_min=float(S[-1]), n_singular=len(S)
            )
        except Exception as e:
            self.record(name=name, error=str(e))
            
    def summary(self) -> dict:
        base = super().summary()
        if self.condition_numbers:
            base["kappa"] = {
                "mean": f"{np.mean(self.condition_numbers):.1f}",
                "max": f"{np.max(self.condition_numbers):.1f}",
                "min": f"{np.min(self.condition_numbers):.1f}",
                "stable": "✅" if np.mean(self.condition_numbers) < 1e6 else "⚠️ NUMERICALLY UNSTABLE"
            }
        if self.spectral_entropies:
            base["H_spectral"] = {
                "mean": f"{np.mean(self.spectral_entropies):.2f}",
                "max": f"{np.max(self.spectral_entropies):.2f}",
            }
        if self.effective_ranks:
            base["eff_rank"] = {
                "mean": np.mean(self.effective_ranks),
                "max": np.max(self.effective_ranks),
            }
        return base


class InformationSensor(GlassboxSensor):
    """Mide flujo de información entre capas.
    
    - Entropía de estados del bus
    - Información mutua entre capa y output
    - Ratio de innovación (cuánto añade cada capa)
    """
    def __init__(self):
        super().__init__("information")
        self.bus_entropies = []
        self.layer_innovations = []
        self.bus_knowledge_ratios = []
        
    def record_bus_state(self, phase: str, bus_state: np.ndarray):
        """Analizar estado del KnowledgeBus."""
        x = bus_state.ravel()
        # Protección NaN: si hay NaNs, saltar histograma
        if not np.all(np.isfinite(x)):
            self.record(type="bus", phase=phase, entropy=0.0, norm=float(np.nan_to_num(np.linalg.norm(bus_state))))
            return
        # Entropía por histograma
        hist, _ = np.histogram(x, bins=50, density=True)
        hist = hist[hist > 0]
        H = float(-np.sum(hist * np.log(hist + 1e-12)))
        self.bus_entropies.append(H)
        
        # Norma
        norm = float(np.linalg.norm(bus_state))
        
        self.record(type="bus", phase=phase, entropy=H, norm=norm)
        
    def record_layer_innovation(self, layer_idx: int, 
                                 input_norm: float, output_norm: float,
                                 knowledge_ratio: float):
        """Cuánto cambió la información al pasar por la capa."""
        innovation = abs(output_norm - input_norm) / max(input_norm, 1e-8)
        self.layer_innovations.append((layer_idx, innovation))
        self.bus_knowledge_ratios.append(knowledge_ratio)
        self.record(type="innovation", layer=layer_idx, 
                    innovation=innovation, knowledge_ratio=knowledge_ratio)
        
    def summary(self) -> dict:
        base = super().summary()
        if self.bus_entropies:
            ent_mean = float(np.mean(self.bus_entropies))
            base["bus_entropy"] = f"{ent_mean:.2f}" if np.isfinite(ent_mean) else "NaN"
        if self.bus_knowledge_ratios:
            krs = self.bus_knowledge_ratios
            kr_mean = float(np.mean(krs))
            kr_final = float(krs[-1])
            base["knowledge_ratio"] = {
                "mean": f"{kr_mean:.3f}" if np.isfinite(kr_mean) else "NaN",
                "final": f"{kr_final:.3f}" if np.isfinite(kr_final) else "NaN",
                "growing": "✅" if kr_final > krs[0] else "⚠️ SATURATED"
            }
        return base


class DecoderSensor(GlassboxSensor):
    """Analiza el comportamiento del decoder.
    
    - Acuerdo entre cabezas (head agreement)
    - Confianza promedio del decoder
    - Distribución de predicciones (¿colapsó a pocas clases?)
    - Per-token difficulty (¿qué tokens son más difíciles?)
    """
    def __init__(self, vocab_size: int):
        super().__init__("decoder")
        self.vocab_size = vocab_size
        self.head_agreements = []
        self.confidences = []
        self.prediction_counts = np.zeros(vocab_size, dtype=np.float64)
        self.token_errors = np.zeros(vocab_size, dtype=np.float64)
        self.token_counts = np.zeros(vocab_size, dtype=np.float64)
        
    def record_prediction(self, target: int, predicted: int, 
                          confidence: float, head_votes: List[np.ndarray] = None):
        """Registrar una predicción del decoder."""
        self.prediction_counts[predicted] += 1
        self.token_counts[target] += 1
        if target != predicted:
            self.token_errors[target] += 1
        self.confidences.append(confidence)
        
        if head_votes and len(head_votes) >= 2:
            # Acuerdo entre cabezas: correlación de rankings
            rankings = [np.argsort(-v)[:5] for v in head_votes]
            agreements = []
            for i in range(len(rankings)):
                for j in range(i+1, len(rankings)):
                    agree = len(set(rankings[i]) & set(rankings[j])) / 5.0
                    agreements.append(agree)
            if agreements:
                self.head_agreements.append(float(np.mean(agreements)))
                
    def summary(self) -> dict:
        base = super().summary()
        if self.confidences:
            base["avg_confidence"] = f"{np.mean(self.confidences):.3f}"
        if self.head_agreements:
            base["head_agreement"] = f"{np.mean(self.head_agreements):.3f}"
        if self.token_counts.sum() > 0:
            error_rates = self.token_errors / (self.token_counts + 1)
            hardest = int(np.argmax(error_rates))
            easiest = int(np.argmin(error_rates[self.token_counts > 10]))
            base["hardest_token"] = f"token {hardest} ({error_rates[hardest]:.1%} error)"
            base["pred_collapse"] = "⚠️" if np.max(self.prediction_counts) / max(self.prediction_counts.sum(), 1) > 0.5 else "✅"
        return base


# ═══════════════════════════════════════════════════════════════
# 📈 GLASSBOX VISUALIZER
# ═══════════════════════════════════════════════════════════════

class GlassboxVisualizer:
    """Genera reportes visuales del estado de Nova durante entrenamiento.
    
    Formato: JSON + resumen texto. Listo para dashboards o notebooks.
    """
    
    def __init__(self, sensors: List[GlassboxSensor],
                 output_dir: str = "glassbox_reports"):
        self.sensors = {s.name: s for s in sensors}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._checkpoint_counter = 0
        
    def checkpoint(self, label: str = ""):
        """Guardar snapshot de todos los sensores."""
        self._checkpoint_counter += 1
        report = {
            "checkpoint": self._checkpoint_counter,
            "label": label,
            "timestamp": time.time(),
            "sensors": {name: s.summary() for name, s in self.sensors.items()}
        }
        
        # 🔥 Convert numpy types to Python native for JSON
        report = _json_safe(report)
        
        filename = f"glassbox_{self._checkpoint_counter:04d}"
        if label:
            filename += f"_{label}"
        filepath = os.path.join(self.output_dir, f"{filename}.json")
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
    
    def print_summary(self):
        """Imprimir resumen legible de todos los sensores."""
        print("\n" + "=" * 60)
        print("🔍 GLASSBOX ANALYSIS — Nova Internal State")
        print("=" * 60)
        
        for name, sensor in self.sensors.items():
            summary = sensor.summary()
            print(f"\n📡 {name.upper()}:")
            for k, v in summary.items():
                if k not in ("name", "n_records", "data"):
                    if isinstance(v, dict):
                        print(f"  {k}:")
                        for kk, vv in v.items():
                            print(f"    {kk}: {vv}")
                    else:
                        print(f"  {k}: {v}")
        
        print("\n" + "=" * 60)
    
    def plot_phase_timeline(self) -> str:
        """Generar timeline ASCII de fases."""
        timing = self.sensors.get("timing")
        if not timing or not timing.phases:
            return "No timing data"
            
        total = sum(timing.phases.values())
        lines = ["\n📊 Phase Timeline:"]
        lines.append("-" * 50)
        
        bar_width = 40
        for name, duration in sorted(timing.phases.items(), 
                                      key=lambda x: -x[1]):
            pct = duration / total
            bar = "█" * int(pct * bar_width)
            lines.append(f"  {name:25s} {bar} {duration:6.1f}s ({pct:4.0%})")
            
        lines.append("-" * 50)
        lines.append(f"  {'TOTAL':25s} {total:.1f}s")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 🎯 BOTTLENECK DETECTOR
# ═══════════════════════════════════════════════════════════════

class BottleneckDetector:
    """Detecta cuellos de botella analizando datos de sensores.
    
    Clasifica bottlenecks en:
      - COMPUTE: operación lenta pero vectorizable → mover a GPU/Triton
      - LOOP: Python loop → vectorizar
      - MEMORY: RAM/VRAM saturada → reducir batch
      - NUMERICAL: mal condicionamiento → regularizar
      - INFORMATION: poca info en features → mejorar arquitectura
    """
    
    def analyze(self, timing: TimingSensor, spectral: SpectralSensor,
                info: InformationSensor, decoder: DecoderSensor) -> dict:
        findings = []
        
        # 1. Analizar tiempos
        if timing._op_times:
            total_by_op = {k: sum(v) for k, v in timing._op_times.items()}
            for op, total_t in sorted(total_by_op.items(), key=lambda x: -x[1])[:5]:
                avg_t = np.mean(timing._op_times[op])
                findings.append({
                    "type": "COMPUTE" if total_t > 10 else "OK",
                    "op": op,
                    "total_time": f"{total_t:.1f}s",
                    "avg_time": f"{avg_t*1000:.1f}ms",
                    "suggestion": "Move to GPU/Triton" if total_t > 10 else None
                })
        
        # 2. Analizar condicionamiento
        if spectral.condition_numbers:
            mean_kappa = np.mean(spectral.condition_numbers)
            if mean_kappa > 1e8:
                findings.append({
                    "type": "NUMERICAL",
                    "issue": f"κ = {mean_kappa:.1e} — extremely ill-conditioned",
                    "suggestion": "Increase l2_lambda or reduce feature count"
                })
            elif mean_kappa > 1e6:
                findings.append({
                    "type": "NUMERICAL",
                    "issue": f"κ = {mean_kappa:.1e} — moderately ill-conditioned",
                    "suggestion": "Consider SES filter or higher λ"
                })
        
        # 3. Analizar información
        if info.bus_knowledge_ratios and len(info.bus_knowledge_ratios) > 1:
            first = info.bus_knowledge_ratios[0]
            last = info.bus_knowledge_ratios[-1]
            if last < first * 1.1:
                findings.append({
                    "type": "INFORMATION",
                    "issue": f"Knowledge ratio stagnant ({first:.2f}→{last:.2f})",
                    "suggestion": "Layers not adding new info — increase depth or pairs"
                })
        
        # 4. Analizar decoder
        if decoder.head_agreements:
            mean_agree = np.mean(decoder.head_agreements)
            if mean_agree > 0.9:
                findings.append({
                    "type": "INFORMATION",
                    "issue": f"Head agreement {mean_agree:.2%} — heads redundant",
                    "suggestion": "Reduce n_heads or increase projection diversity"
                })
            elif mean_agree < 0.3:
                findings.append({
                    "type": "INFORMATION",
                    "issue": f"Head agreement {mean_agree:.2%} — heads disagree too much",
                    "suggestion": "Aggregator may be underfitting"
                })
        
        return {"n_findings": len(findings), "findings": findings}


# ═══════════════════════════════════════════════════════════════
# 🧪 QUICK DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════

def diagnose_model(model, X_sample: np.ndarray, Y_sample: np.ndarray):
    """Diagnóstico rápido: analiza condición numérica, 
    distribución de features, y bottlenecks potenciales."""
    
    print("\n🔬 QUICK DIAGNOSTIC")
    print("=" * 50)
    
    # 1. Analizar contexto de atención
    if hasattr(model, 'layers'):
        for li, layer in enumerate(model.layers):
            print(f"\n  Layer {li+1}:")
            attn = layer.attention
            print(f"    n_levels: {attn.n_levels}")
            for lv in range(attn.n_levels):
                if attn._level_trained[lv]:
                    n = attn.level_neurons[lv]
                    print(f"    L{lv}: {n.n_input}→{n.n_output}, "
                          f"{len(n._pairs)} pairs, "
                          f"deg={n.max_degree}, "
                          f"basis={n.basis_type}, "
                          f"solver={n.solver_type}")
    
    # 2. Forward pass + spectral analysis
    if hasattr(model, 'layers') and len(model.layers) > 0:
        # Forward rápido
        x = X_sample[:min(128, len(X_sample))]
        if hasattr(model, 'bus'):
            model.bus.initialize(x)
        for layer in model.layers:
            if hasattr(layer, 'process_and_write_bus'):
                x = layer.process_and_write_bus(x, model.bus)
            else:
                x = layer.forward(x)
        
        # Spectral
        try:
            _, S, _ = np.linalg.svd(x.astype(np.float64), full_matrices=False)
            kappa = float(S[0] / max(S[-1], 1e-12))
            p = S / S.sum()
            H = float(-np.sum(p * np.log(p + 1e-12)))
            print(f"\n  📐 Context vectors: κ={kappa:.1f}, H={H:.2f}, "
                  f"eff_rank={int(np.sum(S > 0.01*S[0]))}")
        except:
            pass
    
    # 3. Decoder analysis
    if hasattr(model, 'decoder'):
        d = model.decoder
        if hasattr(d, 'n_heads'):
            print(f"\n  🎯 Decoder: {d.n_heads} heads × {d.proj_dim}d, "
                  f"T={d.temperature:.2f}")
        elif hasattr(d, 'n_features'):
            print(f"\n  🎯 SpectralDecoder: {d.n_features} features, "
                  f"{len(d._pairs)} pairs")
    
    print("\n" + "=" * 50)


# ═══════════════════════════════════════════════════════════════
# 🏭 FACTORY
# ═══════════════════════════════════════════════════════════════

def create_glassbox_suite(vocab_size: int = 65) -> Tuple[
    TimingSensor, SpectralSensor, InformationSensor, 
    DecoderSensor, GlassboxVisualizer, BottleneckDetector
]:
    """Crear suite completa de sensores glassbox."""
    timing = TimingSensor()
    spectral = SpectralSensor()
    info = InformationSensor()
    decoder = DecoderSensor(vocab_size)
    visualizer = GlassboxVisualizer([timing, spectral, info, decoder])
    detector = BottleneckDetector()
    
    return timing, spectral, info, decoder, visualizer, detector
