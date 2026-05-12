"""
HyperGraph Engine — Every Computation is a Directed Hypergraph of FMA Nodes
============================================================================

The foundational representation for the Universal Constructor. ALL computable
structures — neural networks, graph algorithms, PDE solvers, search systems —
are modeled as directed hypergraphs where:

  - Nodes = FMA operations (y = W·x + b)
  - Hyperedges = Data dependencies (tensor flows between nodes)
  - Annotations = Spectral metadata (α, Koopman eigenvalues, entropy)

This is NOT a toy graph. It is the executable computation substrate that
every other module in the ecosystem operates on.

HIERARCHY OF VIEWS
──────────────────

  TAA  sees the hypergraph as a DYNAMICAL SYSTEM:
    Activations flow through time steps. Lyapunov exponents classify stability.
    Koopman modes lift the nonlinear flow into a linear embedding.

  OTU/ERGON sees the THERMODYNAMICS:
    Each node has computational energy E_node = cost(FMA).
    Entropy S = -Σ p_i log p_i over the activation distribution.
    Bottlenecks are nodes where entropy accumulates (high S, low throughput).

  CoPoem sees the DESIGN SPACE:
    The hypergraph topology is a point on a Riemannian manifold.
    Rewiring = geodesic flow on the manifold. Cost = E(G) - S(G)/β.

CERTIFICATES:
  HG-1: Every node in the hypergraph is a valid FMA operation
  HG-2: The hypergraph is a DAG (no cycles in the forward pass)
  HG-3: Spectral analysis produces bounded α for every subgraph
  HG-4: Rewrites preserve I/O equivalence within tolerance ε
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Node and Edge types
# ---------------------------------------------------------------------------

class NodeKind(str, Enum):
    """Classification of compute nodes."""
    FMA_SCALAR = "fma_scalar"          # y = w*x + b (scalar)
    FMA_MATRIX = "fma_matrix"          # Y = W@X + B (matrix)
    FMA_TENSOR = "fma_tensor"          # Batched GEMM
    NONLINEAR = "nonlinear"            # Activation, etc. (to be reduced)
    REDUCTION = "reduction"            # Sum, mean, max over an axis
    MEMORY = "memory"                  # Load/store (data movement)
    BRANCH = "branch"                  # Conditional dispatch
    SYNCHRONIZE = "synchronize"        # Barrier / allreduce
    SOURCE = "source"                  # Input data
    SINK = "sink"                      # Output data


class EdgeKind(str, Enum):
    """Data dependency classification."""
    DATA_FLOW = "data_flow"            # Tensor passed between nodes
    CONTROL_FLOW = "control_flow"      # Execution ordering
    GRADIENT_FLOW = "gradient_flow"    # Backward pass dependency
    MEMORY_ALIAS = "memory_alias"      # Shared memory reference


@dataclass
class HyperNode:
    """A single compute node in the hypergraph."""
    node_id: int
    kind: NodeKind
    shape_in: Tuple[int, ...]          # Input tensor shape
    shape_out: Tuple[int, ...]         # Output tensor shape
    fma_cost: int                      # Number of FMA operations
    label: str = ""                    # Human-readable label
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Spectral annotations (filled by analysis passes)
    alpha: Optional[float] = None      # ACF spectral decay index
    lyapunov: Optional[float] = None   # Local Lyapunov exponent
    entropy: Optional[float] = None    # Local activation entropy

    @property
    def is_fma(self) -> bool:
        return self.kind in (NodeKind.FMA_SCALAR, NodeKind.FMA_MATRIX, NodeKind.FMA_TENSOR)


@dataclass
class HyperEdge:
    """A directed hyperedge connecting one or more sources to one or more sinks."""
    edge_id: int
    kind: EdgeKind
    sources: List[int]                 # Source node IDs
    sinks: List[int]                   # Sink node IDs
    tensor_shape: Tuple[int, ...] = () # Shape of data on edge
    bandwidth_bytes: int = 0           # Estimated data movement


@dataclass
class SubGraph:
    """A contiguous subgraph (region) of the hypergraph."""
    region_id: int
    node_ids: Set[int]
    label: str = ""
    total_fma: int = 0
    alpha: Optional[float] = None
    classification: Optional[str] = None  # ANALYTIC, CHAOTIC, etc.


# ---------------------------------------------------------------------------
# ComputableHyperGraph — the core data structure
# ---------------------------------------------------------------------------

class ComputableHyperGraph:
    """
    Directed hypergraph representing an arbitrary computation.

    This is the universal substrate: any program, neural network, PDE solver,
    or graph algorithm can be expressed as a ComputableHyperGraph. The ACF
    ecosystem then operates on this structure to analyze, optimize, and
    reconstruct the computation.
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self._nodes: Dict[int, HyperNode] = {}
        self._edges: Dict[int, HyperEdge] = {}
        self._adjacency_out: Dict[int, List[int]] = {}  # node_id -> [edge_ids]
        self._adjacency_in: Dict[int, List[int]] = {}   # node_id -> [edge_ids]
        self._next_node_id = 0
        self._next_edge_id = 0
        self._regions: Dict[int, SubGraph] = {}

    # -- Construction API ---------------------------------------------------

    def add_node(self, kind: NodeKind, shape_in: Tuple[int, ...],
                 shape_out: Tuple[int, ...], fma_cost: int = 0,
                 label: str = "", **metadata) -> int:
        """Add a compute node. Returns the node ID."""
        nid = self._next_node_id
        self._next_node_id += 1
        self._nodes[nid] = HyperNode(
            node_id=nid, kind=kind, shape_in=shape_in, shape_out=shape_out,
            fma_cost=fma_cost, label=label, metadata=metadata,
        )
        self._adjacency_out[nid] = []
        self._adjacency_in[nid] = []
        return nid

    def add_edge(self, sources: List[int], sinks: List[int],
                 kind: EdgeKind = EdgeKind.DATA_FLOW,
                 tensor_shape: Tuple[int, ...] = ()) -> int:
        """Add a directed hyperedge. Returns the edge ID."""
        eid = self._next_edge_id
        self._next_edge_id += 1
        bw = int(np.prod(tensor_shape)) * 4 if tensor_shape else 0
        self._edges[eid] = HyperEdge(
            edge_id=eid, kind=kind, sources=sources, sinks=sinks,
            tensor_shape=tensor_shape, bandwidth_bytes=bw,
        )
        for s in sources:
            self._adjacency_out.setdefault(s, []).append(eid)
        for t in sinks:
            self._adjacency_in.setdefault(t, []).append(eid)
        return eid

    def add_fma_chain(self, n_ops: int, dim: int, label_prefix: str = "fma") -> List[int]:
        """Add a linear chain of n_ops FMA nodes. Returns node IDs."""
        shape = (dim,)
        ids = []
        for i in range(n_ops):
            nid = self.add_node(
                NodeKind.FMA_MATRIX if dim > 1 else NodeKind.FMA_SCALAR,
                shape_in=shape, shape_out=shape, fma_cost=dim * dim,
                label=f"{label_prefix}_{i}",
            )
            ids.append(nid)
            if i > 0:
                self.add_edge([ids[i - 1]], [nid], tensor_shape=shape)
        return ids

    # -- Query API ----------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    @property
    def total_fma(self) -> int:
        return sum(n.fma_cost for n in self._nodes.values())

    @property
    def total_bandwidth(self) -> int:
        return sum(e.bandwidth_bytes for e in self._edges.values())

    def node(self, nid: int) -> HyperNode:
        return self._nodes[nid]

    def edge(self, eid: int) -> HyperEdge:
        return self._edges[eid]

    def nodes(self) -> List[HyperNode]:
        return list(self._nodes.values())

    def edges(self) -> List[HyperEdge]:
        return list(self._edges.values())

    def successors(self, nid: int) -> List[int]:
        """Return node IDs reachable via outgoing edges."""
        result = []
        for eid in self._adjacency_out.get(nid, []):
            result.extend(self._edges[eid].sinks)
        return result

    def predecessors(self, nid: int) -> List[int]:
        """Return node IDs that feed into this node."""
        result = []
        for eid in self._adjacency_in.get(nid, []):
            result.extend(self._edges[eid].sources)
        return result

    def source_nodes(self) -> List[int]:
        """Nodes with no incoming data edges."""
        return [nid for nid, elist in self._adjacency_in.items()
                if not elist or all(self._edges[e].kind != EdgeKind.DATA_FLOW
                                    for e in elist)]

    def sink_nodes(self) -> List[int]:
        """Nodes with no outgoing data edges."""
        return [nid for nid, elist in self._adjacency_out.items()
                if not elist or all(self._edges[e].kind != EdgeKind.DATA_FLOW
                                    for e in elist)]

    def topological_order(self) -> List[int]:
        """Kahn's algorithm for topological sort of nodes."""
        in_degree = {nid: 0 for nid in self._nodes}
        for e in self._edges.values():
            if e.kind == EdgeKind.DATA_FLOW:
                for s in e.sinks:
                    in_degree[s] += 1
        queue = [nid for nid, d in in_degree.items() if d == 0]
        order = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for eid in self._adjacency_out.get(nid, []):
                e = self._edges[eid]
                if e.kind != EdgeKind.DATA_FLOW:
                    continue
                for s in e.sinks:
                    in_degree[s] -= 1
                    if in_degree[s] == 0:
                        queue.append(s)
        return order

    def is_dag(self) -> bool:
        """Check if the data-flow subgraph is acyclic."""
        return len(self.topological_order()) == self.n_nodes

    # -- Subgraph / Region API -----------------------------------------------

    def extract_subgraph(self, node_ids: Set[int], label: str = "") -> SubGraph:
        """Extract a subgraph as a named region."""
        total = sum(self._nodes[nid].fma_cost for nid in node_ids if nid in self._nodes)
        rid = len(self._regions)
        sg = SubGraph(region_id=rid, node_ids=node_ids, label=label, total_fma=total)
        self._regions[rid] = sg
        return sg

    def partition_by_depth(self, max_region_size: int = 64) -> List[SubGraph]:
        """Partition the graph into contiguous depth-bands."""
        order = self.topological_order()
        depth = {}
        for nid in order:
            preds = self.predecessors(nid)
            depth[nid] = max((depth.get(p, 0) for p in preds), default=0) + 1

        max_d = max(depth.values()) if depth else 1
        band_size = max(1, max_d // max(1, self.n_nodes // max_region_size))
        bands: Dict[int, Set[int]] = {}
        for nid, d in depth.items():
            band_id = d // max(1, band_size)
            bands.setdefault(band_id, set()).add(nid)

        regions = []
        for bid in sorted(bands.keys()):
            sg = self.extract_subgraph(bands[bid], label=f"depth_band_{bid}")
            regions.append(sg)
        return regions

    # -- Analysis passes ----------------------------------------------------

    def compute_adjacency_matrix(self) -> np.ndarray:
        """Return dense adjacency matrix (data-flow edges only)."""
        n = self.n_nodes
        id_map = {nid: i for i, nid in enumerate(sorted(self._nodes.keys()))}
        A = np.zeros((n, n), dtype=np.float64)
        for e in self._edges.values():
            if e.kind == EdgeKind.DATA_FLOW:
                for s in e.sources:
                    for t in e.sinks:
                        if s in id_map and t in id_map:
                            A[id_map[s], id_map[t]] = 1.0
        return A

    def spectral_analysis(self, n_modes: int = 20) -> Dict[str, Any]:
        """
        Compute spectral properties of the computation graph.

        Returns eigenvalues, spectral gap, and estimated entropy of the
        graph Laplacian — the same view TAA uses to classify dynamical systems,
        but applied to the computation graph itself.
        """
        A = self.compute_adjacency_matrix()
        n = A.shape[0]
        if n == 0:
            return {"eigenvalues": np.array([]), "spectral_gap": 0.0, "entropy": 0.0}

        # Graph Laplacian
        D = np.diag(A.sum(axis=1))
        L = D - A

        # Eigendecomposition (symmetric part for analysis)
        L_sym = 0.5 * (L + L.T)
        k = min(n_modes, n)
        if n <= 500:
            evals = np.linalg.eigvalsh(L_sym)
        else:
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csr_matrix
            evals = eigsh(csr_matrix(L_sym), k=k, which='SM', return_eigenvectors=False)
            evals = np.sort(evals)

        # Spectral gap (Fiedler value) — measures algebraic connectivity
        spectral_gap = float(evals[1]) if len(evals) > 1 else 0.0

        # Spectral entropy
        evals_pos = evals[evals > 1e-12]
        if len(evals_pos) > 0:
            p = evals_pos / evals_pos.sum()
            entropy = -float(np.sum(p * np.log(p + 1e-30)))
        else:
            entropy = 0.0

        # Node-level entropy from degree distribution
        degrees = A.sum(axis=1) + A.sum(axis=0)
        deg_sum = degrees.sum()
        if deg_sum > 0:
            p_deg = degrees / deg_sum
            degree_entropy = -float(np.sum(p_deg * np.log(p_deg + 1e-30)))
        else:
            degree_entropy = 0.0

        return {
            "eigenvalues": evals[:k],
            "spectral_gap": spectral_gap,
            "spectral_entropy": entropy,
            "degree_entropy": degree_entropy,
            "n_components": int(np.sum(evals < 1e-10)),
            "total_fma": self.total_fma,
            "total_bandwidth_bytes": self.total_bandwidth,
        }

    def identify_bottlenecks(self, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find computational bottlenecks — nodes with high FMA cost and high
        fan-in/fan-out (entropy accumulation points per ERGON).
        """
        scores = []
        for nid, node in self._nodes.items():
            fan_in = len(self.predecessors(nid))
            fan_out = len(self.successors(nid))
            # Bottleneck score: high cost × high connectivity
            score = node.fma_cost * (1.0 + np.log1p(fan_in + fan_out))
            scores.append((nid, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    # -- Graph Rewriting ----------------------------------------------------

    def replace_subgraph(self, old_node_ids: Set[int],
                         new_graph: "ComputableHyperGraph") -> None:
        """
        Replace a subgraph with a new one, preserving external connections.

        This is the fundamental operation for optimization: replace a
        detected region with its ACF-optimized equivalent.
        """
        # Find external edges entering / leaving the subgraph
        external_in_edges = []
        external_out_edges = []
        for nid in old_node_ids:
            for eid in self._adjacency_in.get(nid, []):
                e = self._edges[eid]
                if any(s not in old_node_ids for s in e.sources):
                    external_in_edges.append(eid)
            for eid in self._adjacency_out.get(nid, []):
                e = self._edges[eid]
                if any(s not in old_node_ids for s in e.sinks):
                    external_out_edges.append(eid)

        # Map old boundary nodes to new source/sink nodes
        new_sources = new_graph.source_nodes()
        new_sinks = new_graph.sink_nodes()

        # Import new nodes with remapped IDs
        id_remap = {}
        for new_node in new_graph.nodes():
            new_id = self.add_node(
                kind=new_node.kind, shape_in=new_node.shape_in,
                shape_out=new_node.shape_out, fma_cost=new_node.fma_cost,
                label=new_node.label,
            )
            id_remap[new_node.node_id] = new_id

        # Import new internal edges
        for new_edge in new_graph.edges():
            self.add_edge(
                sources=[id_remap[s] for s in new_edge.sources],
                sinks=[id_remap[s] for s in new_edge.sinks],
                kind=new_edge.kind, tensor_shape=new_edge.tensor_shape,
            )

        # Reconnect external edges to new boundary nodes
        for eid in external_in_edges:
            e = self._edges[eid]
            if new_sources:
                new_sink = id_remap[new_sources[0]]
                self.add_edge(
                    sources=[s for s in e.sources if s not in old_node_ids],
                    sinks=[new_sink],
                    kind=e.kind, tensor_shape=e.tensor_shape,
                )

        for eid in external_out_edges:
            e = self._edges[eid]
            if new_sinks:
                new_source = id_remap[new_sinks[0]]
                self.add_edge(
                    sources=[new_source],
                    sinks=[s for s in e.sinks if s not in old_node_ids],
                    kind=e.kind, tensor_shape=e.tensor_shape,
                )

        # Remove old nodes and their edges
        for nid in old_node_ids:
            edges_to_remove = set(self._adjacency_in.get(nid, []) +
                                  self._adjacency_out.get(nid, []))
            for eid in edges_to_remove:
                self._edges.pop(eid, None)
            self._adjacency_in.pop(nid, None)
            self._adjacency_out.pop(nid, None)
            self._nodes.pop(nid, None)

    def fingerprint(self) -> str:
        """Compute a topology-aware hash of the graph structure."""
        parts = []
        for nid in sorted(self._nodes.keys()):
            n = self._nodes[nid]
            parts.append(f"{n.kind.value}:{n.shape_in}:{n.shape_out}:{n.fma_cost}")
        for eid in sorted(self._edges.keys()):
            e = self._edges[eid]
            parts.append(f"{sorted(e.sources)}:{sorted(e.sinks)}")
        blob = "|".join(parts).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def summary(self) -> Dict[str, Any]:
        """Compact summary of the hypergraph."""
        kind_counts = {}
        for n in self._nodes.values():
            kind_counts[n.kind.value] = kind_counts.get(n.kind.value, 0) + 1
        return {
            "name": self.name,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "total_fma": self.total_fma,
            "total_bandwidth_bytes": self.total_bandwidth,
            "is_dag": self.is_dag(),
            "n_regions": len(self._regions),
            "kind_distribution": kind_counts,
            "fingerprint": self.fingerprint(),
        }


# ---------------------------------------------------------------------------
# HyperGraph builders for common structures
# ---------------------------------------------------------------------------

def build_linear_chain(n_layers: int, dim: int, name: str = "chain") -> ComputableHyperGraph:
    """Build a simple FMA chain (e.g., MLP-like forward pass)."""
    g = ComputableHyperGraph(name)
    src = g.add_node(NodeKind.SOURCE, shape_in=(dim,), shape_out=(dim,), label="input")
    prev = src
    for i in range(n_layers):
        nid = g.add_node(
            NodeKind.FMA_MATRIX, shape_in=(dim,), shape_out=(dim,),
            fma_cost=dim * dim, label=f"linear_{i}",
        )
        g.add_edge([prev], [nid], tensor_shape=(dim,))
        prev = nid
    sink = g.add_node(NodeKind.SINK, shape_in=(dim,), shape_out=(dim,), label="output")
    g.add_edge([prev], [sink], tensor_shape=(dim,))
    return g


def build_residual_network(n_blocks: int, dim: int,
                           name: str = "resnet") -> ComputableHyperGraph:
    """Build a residual network graph (skip connections)."""
    g = ComputableHyperGraph(name)
    src = g.add_node(NodeKind.SOURCE, shape_in=(dim,), shape_out=(dim,), label="input")
    prev = src
    for i in range(n_blocks):
        # Main path: two FMAs
        fma1 = g.add_node(NodeKind.FMA_MATRIX, (dim,), (dim,), dim*dim, f"block{i}_fma1")
        fma2 = g.add_node(NodeKind.FMA_MATRIX, (dim,), (dim,), dim*dim, f"block{i}_fma2")
        # Reduction node (addition for skip connection)
        add_node = g.add_node(NodeKind.REDUCTION, (dim,), (dim,), dim, f"block{i}_add")
        g.add_edge([prev], [fma1], tensor_shape=(dim,))
        g.add_edge([fma1], [fma2], tensor_shape=(dim,))
        g.add_edge([fma2], [add_node], tensor_shape=(dim,))
        g.add_edge([prev], [add_node], tensor_shape=(dim,))  # skip
        prev = add_node
    sink = g.add_node(NodeKind.SINK, (dim,), (dim,), label="output")
    g.add_edge([prev], [sink], tensor_shape=(dim,))
    return g


def build_attention_block(n_heads: int, dim: int, seq_len: int,
                          name: str = "attention") -> ComputableHyperGraph:
    """Build a multi-head attention computation graph."""
    g = ComputableHyperGraph(name)
    head_dim = max(1, dim // n_heads)
    src = g.add_node(NodeKind.SOURCE, (seq_len, dim), (seq_len, dim), label="input")

    head_outputs = []
    for h in range(n_heads):
        # Q, K, V projections
        q = g.add_node(NodeKind.FMA_MATRIX, (seq_len, dim), (seq_len, head_dim),
                        seq_len * dim * head_dim, f"head{h}_Q")
        k = g.add_node(NodeKind.FMA_MATRIX, (seq_len, dim), (seq_len, head_dim),
                        seq_len * dim * head_dim, f"head{h}_K")
        v = g.add_node(NodeKind.FMA_MATRIX, (seq_len, dim), (seq_len, head_dim),
                        seq_len * dim * head_dim, f"head{h}_V")
        g.add_edge([src], [q], tensor_shape=(seq_len, dim))
        g.add_edge([src], [k], tensor_shape=(seq_len, dim))
        g.add_edge([src], [v], tensor_shape=(seq_len, dim))

        # Attention scores: Q @ K^T
        attn = g.add_node(NodeKind.FMA_MATRIX, (seq_len, head_dim), (seq_len, seq_len),
                           seq_len * seq_len * head_dim, f"head{h}_attn")
        g.add_edge([q, k], [attn], tensor_shape=(seq_len, head_dim))

        # Softmax (nonlinear — to be ACF-reduced)
        soft = g.add_node(NodeKind.NONLINEAR, (seq_len, seq_len), (seq_len, seq_len),
                           seq_len * seq_len * 5, f"head{h}_softmax")
        g.add_edge([attn], [soft], tensor_shape=(seq_len, seq_len))

        # Weighted values: softmax(QK^T) @ V
        weighted = g.add_node(NodeKind.FMA_MATRIX, (seq_len, seq_len), (seq_len, head_dim),
                               seq_len * seq_len * head_dim, f"head{h}_weighted")
        g.add_edge([soft, v], [weighted], tensor_shape=(seq_len, head_dim))
        head_outputs.append(weighted)

    # Concatenate + project
    proj = g.add_node(NodeKind.FMA_MATRIX, (seq_len, dim), (seq_len, dim),
                       seq_len * dim * dim, "output_proj")
    for ho in head_outputs:
        g.add_edge([ho], [proj], tensor_shape=(seq_len, head_dim))

    sink = g.add_node(NodeKind.SINK, (seq_len, dim), (seq_len, dim), label="output")
    g.add_edge([proj], [sink], tensor_shape=(seq_len, dim))
    return g


def build_stencil_grid(nx: int, ny: int, stencil_width: int = 3,
                       name: str = "stencil") -> ComputableHyperGraph:
    """Build a stencil computation graph for PDE solvers on a 2D grid."""
    g = ComputableHyperGraph(name)

    # Source: full grid
    src = g.add_node(NodeKind.SOURCE, (nx, ny), (nx, ny), label="grid_input")

    # Each interior point applies a stencil (FMA chain)
    n_interior = max(1, (nx - 2) * (ny - 2))
    fma_per_point = stencil_width * stencil_width

    # Model as a single batched FMA node (practical representation)
    stencil = g.add_node(
        NodeKind.FMA_TENSOR, (nx, ny), (nx, ny),
        fma_cost=n_interior * fma_per_point,
        label=f"stencil_{stencil_width}x{stencil_width}",
    )
    g.add_edge([src], [stencil], tensor_shape=(nx, ny))

    sink = g.add_node(NodeKind.SINK, (nx, ny), (nx, ny), label="grid_output")
    g.add_edge([stencil], [sink], tensor_shape=(nx, ny))
    return g


def from_torch_module(module: "torch.nn.Module", input_shape: Tuple[int, ...],
                      name: str = "torch_module") -> ComputableHyperGraph:
    """
    Build a ComputableHyperGraph from a PyTorch module by tracing.

    This enables the Universal Constructor to ingest ANY PyTorch model
    and represent it as an FMA hypergraph for analysis and optimization.
    """
    g = ComputableHyperGraph(name)
    src = g.add_node(NodeKind.SOURCE, input_shape, input_shape, label="input")
    prev = src

    for layer_name, layer in module.named_children():
        if isinstance(layer, torch.nn.Linear):
            in_f = layer.in_features
            out_f = layer.out_features
            nid = g.add_node(
                NodeKind.FMA_MATRIX,
                shape_in=(in_f,), shape_out=(out_f,),
                fma_cost=in_f * out_f,
                label=layer_name,
                weight_shape=(out_f, in_f),
            )
        elif isinstance(layer, (torch.nn.ReLU, torch.nn.GELU, torch.nn.Sigmoid,
                                torch.nn.Tanh, torch.nn.SiLU)):
            prev_shape = g.node(prev).shape_out
            nid = g.add_node(
                NodeKind.NONLINEAR,
                shape_in=prev_shape, shape_out=prev_shape,
                fma_cost=int(np.prod(prev_shape)) * 5,
                label=layer_name,
                activation_type=type(layer).__name__,
            )
        elif isinstance(layer, (torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
            prev_shape = g.node(prev).shape_out
            nid = g.add_node(
                NodeKind.REDUCTION,
                shape_in=prev_shape, shape_out=prev_shape,
                fma_cost=int(np.prod(prev_shape)) * 3,
                label=layer_name,
            )
        elif isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d)):
            prev_shape = g.node(prev).shape_out
            out_ch = layer.out_channels
            k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            nid = g.add_node(
                NodeKind.FMA_TENSOR,
                shape_in=prev_shape, shape_out=(out_ch,),
                fma_cost=layer.in_channels * out_ch * k * k,
                label=layer_name,
            )
        else:
            prev_shape = g.node(prev).shape_out
            nid = g.add_node(
                NodeKind.FMA_SCALAR,
                shape_in=prev_shape, shape_out=prev_shape,
                fma_cost=int(np.prod(prev_shape)),
                label=layer_name,
            )
        g.add_edge([prev], [nid], tensor_shape=g.node(prev).shape_out)
        prev = nid

    sink = g.add_node(NodeKind.SINK, g.node(prev).shape_out, g.node(prev).shape_out,
                       label="output")
    g.add_edge([prev], [sink], tensor_shape=g.node(prev).shape_out)
    return g


def build_butterfly_fft(N: int, name: str = "butterfly_fft") -> ComputableHyperGraph:
    """
    Build a Cooley-Tukey butterfly DAG for FFT of size N.

    The graph encodes the STRUCTURE of the FFT — log2(N) butterfly stages,
    each with N/2 butterfly operations (6 FMA each: 4 real muls + 2 adds
    for the complex multiply-add).

    This is NOT a callable FFT — it's the computation graph that the
    BackendSynthesizer analyzes to generate optimized native code.

    Parameters
    ----------
    N : int
        FFT size (must be a power of 2, >= 2)
    name : str
        Graph name

    Returns
    -------
    ComputableHyperGraph with butterfly structure annotations
    """
    import math
    assert N >= 2 and (N & (N - 1)) == 0, f"N must be a power of 2, got {N}"
    log_n = int(math.log2(N))

    g = ComputableHyperGraph(name)

    # Source: input signal (N complex values)
    src = g.add_node(NodeKind.SOURCE, (N,), (N,), label="input_signal")

    # Bit-reversal permutation (data movement, no FMA)
    perm = g.add_node(NodeKind.MEMORY, (N,), (N,), fma_cost=0,
                       label="bit_reversal_permutation")
    g.add_edge([src], [perm], tensor_shape=(N,))

    prev = perm
    # log2(N) butterfly stages
    fma_per_butterfly = 6  # complex multiply-add = 4 real muls + 2 adds
    n_butterflies = N // 2

    for stage in range(log_n):
        half = 1 << stage
        n_groups = N // (2 * half)
        stage_node = g.add_node(
            NodeKind.FMA_TENSOR, (N,), (N,),
            fma_cost=n_butterflies * fma_per_butterfly,
            label=f"butterfly_stage_{stage}",
            stage=stage,
            half_size=half,
            n_groups=n_groups,
            n_butterflies=n_butterflies,
            pattern="butterfly",
        )
        g.add_edge([prev], [stage_node], tensor_shape=(N,))
        prev = stage_node

    # Sink: output spectrum
    sink = g.add_node(NodeKind.SINK, (N,), (N,), label="output_spectrum")
    g.add_edge([prev], [sink], tensor_shape=(N,))

    return g
