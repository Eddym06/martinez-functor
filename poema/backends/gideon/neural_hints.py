"""
GideonNeuralHints — Base de Heurísticas para Descubrimiento de Arquitecturas IA.

Este módulo establece la INFRAESTRUCTURA BASE para el futuro motor de
búsqueda de arquitecturas de IA de Poema/Gideon. Por ahora:

  - Define blueprints de arquitecturas (transformers, CNNs, MLPs, GNNs)
  - Implementa métricas de complejidad algebraica (α-complejidad de ACF)
  - Proporciona grafos de búsqueda de hiperparámetros como GideonGraph
  - Define la interfaz ArchitectureBlueprint que Gideon usará para NAS futuro

NOTA: El motor de NAS completo (búsqueda con RL/evolución) requiere
componentes adicionales planificados en el roadmap de Poema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# ArchitectureKind
# ─────────────────────────────────────────────────────────────────────────────

from enum import Enum

class ArchKind(Enum):
    MLP        = "mlp"
    CNN        = "cnn"
    TRANSFORMER = "transformer"
    GNN        = "gnn"          # Graph Neural Network
    RECURRENT  = "recurrent"
    HYBRID     = "hybrid"
    UNKNOWN    = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# LayerSpec — especificación de una capa
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerSpec:
    """Especificación de una capa de IA como nodo en el grafo Gideon."""
    kind: str                       # linear | conv | attention | norm | activation
    in_features: int = 0
    out_features: int = 0
    params: Dict[str, Any] = field(default_factory=dict)

    # Métricas de complejidad
    flops: int = 0
    params_count: int = 0
    fma_equivalent: int = 0        # equivalente FMA para Gideon

    def compute_flops(self) -> int:
        if self.kind == "linear":
            return 2 * self.in_features * self.out_features
        if self.kind == "attention":
            seq_len = self.params.get("seq_len", 512)
            d_model = self.in_features
            return 4 * seq_len * d_model * d_model
        if self.kind == "conv":
            kernel = self.params.get("kernel_size", 3)
            return 2 * self.in_features * self.out_features * kernel * kernel
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# ArchitectureBlueprint
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArchitectureBlueprint:
    """
    Blueprint de arquitectura de IA que Gideon puede analizar, optimizar
    y (en el futuro) buscar automáticamente.

    Cada blueprint es un grafo de LayerSpec + metadatos de complejidad.
    """
    name: str
    kind: ArchKind
    layers: List[LayerSpec] = field(default_factory=list)

    # Métricas globales
    total_params: int = 0
    total_flops: int = 0
    alpha_complexity: float = 0.0  # Índice ACF de complejidad

    # Restricciones del espacio de búsqueda
    search_space: Dict[str, Any] = field(default_factory=dict)

    def add_layer(self, spec: LayerSpec) -> None:
        spec.flops = spec.compute_flops()
        spec.params_count = spec.in_features * spec.out_features
        spec.fma_equivalent = spec.flops // 2
        self.layers.append(spec)
        self.total_flops += spec.flops
        self.total_params += spec.params_count

    def compute_alpha_complexity(self) -> float:
        """
        Calcula el Índice de Complejidad ACF (α) del blueprint.

        α = log(total_flops) / log(total_params + 1)
        Representa la eficiencia computacional de la arquitectura.
        Mayor α = más cómputo por parámetro.
        """
        if self.total_params == 0:
            return 0.0
        self.alpha_complexity = math.log(max(1, self.total_flops)) / math.log(max(2, self.total_params + 1))
        return self.alpha_complexity

    def to_gideon_fma_count(self) -> int:
        """Convierte el FLOP count total a equivalente FMA para Gideon IR."""
        return self.total_flops // 2

    def summary(self) -> str:
        alpha = self.compute_alpha_complexity()
        return (
            f"Blueprint('{self.name}', kind={self.kind.value})\n"
            f"  Capas:       {len(self.layers)}\n"
            f"  Parámetros:  {self.total_params:,}\n"
            f"  FLOPs:       {self.total_flops:,}\n"
            f"  α-complejidad: {alpha:.4f}\n"
            f"  FMA-equiv:   {self.to_gideon_fma_count():,}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GideonNeuralHints — catálogo + análisis
# ─────────────────────────────────────────────────────────────────────────────

class GideonNeuralHints:
    """
    Catálogo de heurísticas y blueprints de arquitecturas IA para Gideon.

    Funciones:
      - Crear blueprints estándar (MLP, Transformer, CNN)
      - Analizar complejidad ACF de cualquier blueprint
      - Generar espacio de búsqueda discreto para NAS futuro
      - Convertir blueprints a GideonGraph (para análisis de paralelismo)
    """

    # ── Fábricas de blueprints estándar ──────────────────────────────────────

    @staticmethod
    def mlp(
        layer_dims: List[int],
        activation: str = "relu",
        name: str = "mlp",
    ) -> ArchitectureBlueprint:
        """Blueprint de perceptrón multicapa."""
        bp = ArchitectureBlueprint(name=name, kind=ArchKind.MLP)
        for i in range(len(layer_dims) - 1):
            bp.add_layer(LayerSpec(
                kind="linear",
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
            ))
            if i < len(layer_dims) - 2:
                bp.add_layer(LayerSpec(
                    kind="activation",
                    in_features=layer_dims[i + 1],
                    out_features=layer_dims[i + 1],
                    params={"fn": activation},
                ))
        return bp

    @staticmethod
    def transformer(
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        seq_len: int = 512,
        ffn_dim: int = 2048,
        name: str = "transformer",
    ) -> ArchitectureBlueprint:
        """Blueprint de Transformer estándar."""
        bp = ArchitectureBlueprint(name=name, kind=ArchKind.TRANSFORMER)
        for _ in range(n_layers):
            # Self-attention
            bp.add_layer(LayerSpec(
                kind="attention",
                in_features=d_model,
                out_features=d_model,
                params={"n_heads": n_heads, "seq_len": seq_len},
            ))
            # FFN
            bp.add_layer(LayerSpec(kind="linear", in_features=d_model, out_features=ffn_dim))
            bp.add_layer(LayerSpec(kind="linear", in_features=ffn_dim, out_features=d_model))
            bp.add_layer(LayerSpec(kind="norm", in_features=d_model, out_features=d_model))
        return bp

    @staticmethod
    def cnn_resnet_block(
        channels: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 4,
        name: str = "resnet_block",
    ) -> ArchitectureBlueprint:
        """Blueprint de bloque ResNet."""
        bp = ArchitectureBlueprint(name=name, kind=ArchKind.CNN)
        for _ in range(n_blocks):
            bp.add_layer(LayerSpec(
                kind="conv",
                in_features=channels,
                out_features=channels,
                params={"kernel_size": kernel_size},
            ))
            bp.add_layer(LayerSpec(kind="norm", in_features=channels, out_features=channels))
            bp.add_layer(LayerSpec(kind="activation", in_features=channels, out_features=channels))
        return bp

    # ── Análisis de blueprints ────────────────────────────────────────────────

    @staticmethod
    def analyse_blueprint(bp: ArchitectureBlueprint) -> Dict[str, Any]:
        """Análisis completo de un blueprint con métricas ACF."""
        alpha = bp.compute_alpha_complexity()
        fma_eq = bp.to_gideon_fma_count()

        # Índice de profundidad efectiva
        seq_layers = [l for l in bp.layers if l.kind in ("linear", "conv", "attention")]
        depth = len(seq_layers)

        # Ratio parámetros / FLOPs
        param_flop_ratio = bp.total_params / max(1, bp.total_flops)

        return {
            "name": bp.name,
            "kind": bp.kind.value,
            "total_params": bp.total_params,
            "total_flops": bp.total_flops,
            "fma_equivalent": fma_eq,
            "alpha_complexity": alpha,
            "depth": depth,
            "param_flop_ratio": param_flop_ratio,
            "layers": len(bp.layers),
            "gideon_fma_budget": fma_eq,
        }

    # ── Generación de espacios de búsqueda (base para NAS) ───────────────────

    @staticmethod
    def generate_mlp_search_space(
        min_hidden: int = 64,
        max_hidden: int = 1024,
        min_depth: int = 2,
        max_depth: int = 8,
    ) -> Dict[str, Any]:
        """
        Genera el espacio de búsqueda para NAS de MLPs.
        Formato compatible con futuros agentes de búsqueda de Gideon.
        """
        import math
        # Dimensiones en potencias de 2
        dims = [2 ** i for i in range(int(math.log2(min_hidden)), int(math.log2(max_hidden)) + 1)]
        return {
            "type": "mlp_search_space",
            "hidden_dims": dims,
            "depths": list(range(min_depth, max_depth + 1)),
            "activations": ["relu", "tanh", "sigmoid", "gelu"],
            "total_configs": len(dims) ** max_depth * (max_depth - min_depth + 1),
        }

    @staticmethod
    def generate_transformer_search_space() -> Dict[str, Any]:
        """Espacio de búsqueda base para Transformers."""
        return {
            "type": "transformer_search_space",
            "d_model": [128, 256, 512, 1024],
            "n_heads": [2, 4, 8, 16],
            "n_layers": [2, 4, 6, 8, 12],
            "ffn_multipliers": [2, 4, 8],
            "seq_lengths": [128, 256, 512, 1024],
        }
