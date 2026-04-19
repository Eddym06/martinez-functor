"""
GideonGraph — Grafo de Cómputo Heterogéneo de Gideon.

Un GideonGraph extiende GideonProgram con capacidades de:
  - Paralelización de ramas independientes
  - Fusión de kernels (kernel fusion) cross-backend
  - Análisis de dependencias y planificación de ejecución
  - Soporte para grafos de IA (DAGs de capas, attention, conv)
  - Soporte para grafos de búsqueda de teoremas (proof trees)

Clases:
  GraphEdge           — arista tipada con metadatos de tensor
  GideonGraphNode     — nodo con capacidad de paralelización
  ExecutionPlan       — plan de ejecución con fases paralelas
  GideonGraph         — grafo principal con análisis y optimización
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .ir import GideonProgram, IRNode, IRNodeKind


# ─────────────────────────────────────────────────────────────────────────────
# GraphEdge
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphEdge:
    """Arista tipada en el grafo de cómputo."""
    src_id: str
    dst_id: str
    tensor_shape: Tuple[int, ...] = ()
    dtype: str = "fp64"
    requires_grad: bool = False
    # Cotas del tensor propagado
    lo: float = float("-inf")
    hi: float = float("inf")
    epsilon: float = 0.0

    def __repr__(self) -> str:
        return f"Edge({self.src_id!r} → {self.dst_id!r}, shape={self.tensor_shape})"


# ─────────────────────────────────────────────────────────────────────────────
# GideonGraphNode
# ─────────────────────────────────────────────────────────────────────────────

class NodeStatus(enum.Enum):
    PENDING   = "pending"
    READY     = "ready"      # todas las entradas disponibles
    RUNNING   = "running"
    DONE      = "done"
    ERROR     = "error"


@dataclass
class GideonGraphNode:
    """
    Nodo en el grafo de ejecución.  
    Envuelve un IRNode y agrega estado de ejecución y kernel asociado.
    """
    ir_node: IRNode
    status: NodeStatus = NodeStatus.PENDING
    assigned_backend: str = ""          # qué backend ejecuta este nodo
    kernel_fn: Optional[Callable] = None  # función compilada
    elapsed_ms: float = 0.0

    @property
    def node_id(self) -> str:
        return self.ir_node.node_id

    @property
    def kind(self) -> IRNodeKind:
        return self.ir_node.kind

    def mark_ready(self) -> None:
        self.status = NodeStatus.READY

    def mark_done(self, elapsed_ms: float = 0.0) -> None:
        self.status = NodeStatus.DONE
        self.elapsed_ms = elapsed_ms

    def __repr__(self) -> str:
        return (
            f"GideonGraphNode({self.node_id!r}, "
            f"kind={self.kind.value}, status={self.status.value}, "
            f"backend={self.assigned_backend!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionPlan
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionPhase:
    """Fase de ejecución: conjunto de nodos que pueden ejecutarse en paralelo."""
    phase_idx: int
    node_ids: List[str]
    is_parallelizable: bool = True

    def __repr__(self) -> str:
        return f"Phase({self.phase_idx}, nodes={len(self.node_ids)}, parallel={self.is_parallelizable})"


@dataclass
class ExecutionPlan:
    """
    Plan de ejecución producido por GideonGraph.analyse().

    Las fases son niveles topológicos: nodos en el mismo nivel no tienen
    dependencias entre sí y pueden ejecutarse en paralelo.
    """
    phases: List[ExecutionPhase] = field(default_factory=list)
    total_nodes: int = 0
    parallelizable_ratio: float = 0.0   # fracción de nodos en fases >1 nodo
    critical_path_length: int = 0       # longitud del camino crítico (# fases)
    estimated_fma_ops: int = 0

    def summary(self) -> str:
        lines = [
            "ExecutionPlan",
            f"  Fases:            {len(self.phases)}",
            f"  Nodos totales:    {self.total_nodes}",
            f"  Paralelizables:  {self.parallelizable_ratio:.1%}",
            f"  Camino crítico:   {self.critical_path_length} fases",
            f"  FMA estimados:   {self.estimated_fma_ops}",
        ]
        for ph in self.phases:
            lines.append(f"    {ph}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# GideonGraph
# ─────────────────────────────────────────────────────────────────────────────

class GideonGraph:
    """
    Grafo de cómputo heterogéneo de Gideon.

    Construido a partir de un GideonProgram, el GideonGraph:
      1. Indexa nodos y aristas para acceso O(1)
      2. Analiza dependencias y genera un ExecutionPlan con fases paralelas
      3. Detecta patrones fusionables (FMA chains, operaciones contiguas)
      4. Asigna backends según capacidades y tipo de nodo
      5. Proporciona métricas de grafos de IA y búsqueda de teoremas (bases)
    """

    def __init__(self, program: GideonProgram) -> None:
        self.program = program
        self._nodes: Dict[str, GideonGraphNode] = {}
        self._edges: List[GraphEdge] = []
        self._adj: Dict[str, List[str]] = {}     # id → lista de sucesores
        self._pred: Dict[str, List[str]] = {}    # id → lista de predecesores
        self._build()

    # ── Internal build ────────────────────────────────────────────────────────

    def _build(self) -> None:
        for nid, ir_node in self.program.nodes.items():
            gn = GideonGraphNode(ir_node=ir_node)
            self._nodes[nid] = gn
            self._adj[nid] = []
            self._pred[nid] = []

        for nid, ir_node in self.program.nodes.items():
            for src_id in ir_node.inputs:
                if src_id in self._nodes:
                    edge = GraphEdge(
                        src_id=src_id,
                        dst_id=nid,
                        lo=ir_node.meta.interval_lo,
                        hi=ir_node.meta.interval_hi,
                        epsilon=ir_node.meta.epsilon_bound,
                    )
                    self._edges.append(edge)
                    self._adj[src_id].append(nid)
                    self._pred[nid].append(src_id)

    # ── Topological sort (Kahn's BFS) ─────────────────────────────────────────

    def _topo_levels(self) -> List[List[str]]:
        """Devuelve lista de niveles topológicos (BFS)."""
        in_degree: Dict[str, int] = {nid: len(preds) for nid, preds in self._pred.items()}
        queue = [nid for nid, d in in_degree.items() if d == 0]
        levels: List[List[str]] = []

        while queue:
            levels.append(list(queue))
            next_queue: List[str] = []
            for nid in queue:
                for succ in self._adj.get(nid, []):
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        next_queue.append(succ)
            queue = next_queue

        return levels

    # ── Análisis principal ───────────────────────────────────────────────────

    def analyse(self) -> ExecutionPlan:
        """
        Analiza el grafo y devuelve un ExecutionPlan con fases paralelas.
        """
        levels = self._topo_levels()
        phases: List[ExecutionPhase] = []
        parallel_count = 0

        for i, level in enumerate(levels):
            is_para = len(level) > 1
            phases.append(ExecutionPhase(
                phase_idx=i,
                node_ids=level,
                is_parallelizable=is_para,
            ))
            if is_para:
                parallel_count += len(level)

        total = self.program.total_nodes()
        para_ratio = parallel_count / total if total > 0 else 0.0

        plan = ExecutionPlan(
            phases=phases,
            total_nodes=total,
            parallelizable_ratio=para_ratio,
            critical_path_length=len(phases),
            estimated_fma_ops=self.program.total_fma,
        )
        return plan

    # ── Detección de FMA chains fusionables ──────────────────────────────────

    def find_fusable_chains(self) -> List[List[str]]:
        """
        Detecta cadenas de nodos FMA puro consecutivos que pueden fusionarse
        en un único kernel C/AVX en el backend nativo.
        """
        chains: List[List[str]] = []
        visited: Set[str] = set()

        for nid in self.program.topo_order:
            if nid in visited:
                continue
            node = self._nodes.get(nid)
            if node is None or node.kind != IRNodeKind.FMA:
                continue

            chain = [nid]
            visited.add(nid)
            cur = nid
            while True:
                succs = self._adj.get(cur, [])
                # Solo fusionamos si hay exactamente un sucesor y también es FMA
                if len(succs) != 1:
                    break
                s = succs[0]
                s_node = self._nodes.get(s)
                if s_node is None or s_node.kind != IRNodeKind.FMA:
                    break
                # El sucesor solo tiene un predecesor (sin merge de ramas)
                if len(self._pred.get(s, [])) != 1:
                    break
                chain.append(s)
                visited.add(s)
                cur = s

            if len(chain) >= 2:
                chains.append(chain)

        return chains

    # ── Conteo de nodos de IA base ────────────────────────────────────────────

    def ai_layer_count(self) -> Dict[str, int]:
        """Cuenta nodos de tipo base para arquitecturas de IA."""
        ai_kinds = {
            IRNodeKind.MATMUL, IRNodeKind.GEMM, IRNodeKind.CONV,
            IRNodeKind.NORM, IRNodeKind.ATTENTION,
        }
        counts: Dict[str, int] = {}
        for n in self._nodes.values():
            if n.kind in ai_kinds:
                k = n.kind.value
                counts[k] = counts.get(k, 0) + 1
        return counts

    # ── Estadísticas del grafo ────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        plan = self.analyse()
        chains = self.find_fusable_chains()
        return {
            "n_nodes": len(self._nodes),
            "n_edges": len(self._edges),
            "n_phases": len(plan.phases),
            "parallel_ratio": plan.parallelizable_ratio,
            "critical_path": plan.critical_path_length,
            "fusable_chains": len(chains),
            "total_fma": self.program.total_fma,
            "global_epsilon": self.program.global_epsilon,
            "ai_layers": self.ai_layer_count(),
        }

    def __repr__(self) -> str:
        return (
            f"GideonGraph(nodes={len(self._nodes)}, "
            f"edges={len(self._edges)}, "
            f"fma={self.program.total_fma})"
        )
