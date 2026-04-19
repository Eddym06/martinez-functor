"""
GideonIR — Representación Intermedia (Intermediate Representation) de Poema.

GideonIR es la capa de abstracción entre el compilador frontend de Poema y los
backends de ejecución. Análogo a LLVM-IR / MLIR, pero especializado para:

  - Cadenas FMA con tipos geométricos verificados
  - Aritmética de intervalos propagada
  - Grafos de cómputo simbólico + numérico mezclados
  - Anotaciones formales (certificados epsilon, cotas Lean 4)
  - Soporte futuro: nodos de búsqueda de arquitecturas de IA y teoremas

Diseño:
  IRNodeKind   — enum de 28 tipos de nodos
  IRNode       — nodo tipado con metadatos de seguridad numérica
  GideonIR     — constructor/parser/serializer
  GideonProgram — programa compilado listo para despacho
"""

from __future__ import annotations

import enum
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# IRNodeKind
# ─────────────────────────────────────────────────────────────────────────────

class IRNodeKind(enum.Enum):
    """Tipos de nodos en Gideon IR."""
    # --- Primitivos ---
    CONST       = "const"        # Constante escalar
    INPUT       = "input"        # Nodo de entrada
    FMA         = "fma"          # y = w*x + b  (unidad fundamental)
    IDENTITY    = "identity"     # f(x) = x

    # --- Afines ---
    SCALE       = "scale"        # f(x) = α·x
    SHIFT       = "shift"        # f(x) = x + β
    AFFINE      = "affine"       # f(x) = αx + β

    # --- Composición ---
    COMPOSE     = "compose"      # f ∘ g
    PARALLEL    = "parallel"     # (f, g) en paralelo sobre canales distintos
    BRANCH      = "branch"       # piecewise / stratified

    # --- Polinomios ---
    POLY_HORNER = "poly_horner"  # Evaluación por método de Horner
    POLY_CHEB   = "poly_cheb"    # Aproximación Chebyshev

    # --- Trascendentales ---
    SIN         = "sin"
    COS         = "cos"
    EXP         = "exp"
    LOG         = "log"
    TANH        = "tanh"
    SIGMOID     = "sigmoid"

    # --- Álgebra lineal / tensorial ---
    MATMUL      = "matmul"       # Multiplicación de matrices
    GEMM        = "gemm"         # α·A@B + β·C (BLAS nivel 3 general)
    CONV        = "conv"         # Convolución (base arquitecturas IA)
    NORM        = "norm"         # Normalización (LayerNorm / BatchNorm)
    ATTENTION   = "attention"    # Scaled-dot-product attention (base IA)

    # --- Control de flujo simbólico ---
    LOOP        = "loop"         # Bucle fijo (Fixed-trip)
    RECURSIVE   = "recursive"   # Composición recursiva / autorreferencia

    # --- Semilla: IA y teoremas (bases futuras) ---
    ARCH_PROBE  = "arch_probe"   # Sonda de búsqueda de arquitectura
    THEOREM_SEED = "theorem_seed" # Semilla de candidato a teorema


# ─────────────────────────────────────────────────────────────────────────────
# IRNode
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IRNodeMetadata:
    """Metadatos de seguridad numérica y trazabilidad de un nodo IR."""
    epsilon_bound: float = 0.0          # Cota de error certificada
    domain_lo: float = -math.inf        # Límite inferior del dominio válido
    domain_hi: float = math.inf         # Límite superior del dominio válido
    interval_lo: float = -math.inf      # Intervalo de salida propagado (lo)
    interval_hi: float = math.inf       # Intervalo de salida propagado (hi)
    continuity: str = "unknown"         # C0 | C1 | Cω | unknown
    lean_certificate: Optional[str] = None  # Certificado Lean 4 adjunto
    source_node_id: Optional[str] = None    # Referencia AST original
    fma_cost: int = 0                   # Operaciones FMA contadas
    verified: bool = False              # Verificado formalmente
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epsilon_bound": self.epsilon_bound,
            "domain": [self.domain_lo, self.domain_hi],
            "interval_output": [self.interval_lo, self.interval_hi],
            "continuity": self.continuity,
            "lean_certificate": self.lean_certificate,
            "fma_cost": self.fma_cost,
            "verified": self.verified,
            "tags": self.tags,
        }


@dataclass
class IRNode:
    """
    Nodo en Gideon IR.

    Cada nodo representa una operación atómica o compuesta con tipos
    geométricos verificados, metadatos de seguridad y anotaciones formales.
    """
    kind: IRNodeKind
    node_id: str
    inputs: List[str] = field(default_factory=list)   # IDs de nodos predecesores
    params: Dict[str, Any] = field(default_factory=dict)  # Parámetros del nodo
    meta: IRNodeMetadata = field(default_factory=IRNodeMetadata)

    # Forma tensorial
    input_shape: Tuple[int, ...] = ()
    output_shape: Tuple[int, ...] = ()

    def __repr__(self) -> str:
        return (
            f"IRNode({self.kind.value!r}, id={self.node_id!r}, "
            f"inputs={self.inputs}, ε={self.meta.epsilon_bound:.2e})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "id": self.node_id,
            "inputs": self.inputs,
            "params": self.params,
            "meta": self.meta.to_dict(),
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
        }


# ─────────────────────────────────────────────────────────────────────────────
# GideonProgram
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GideonProgram:
    """
    Programa Gideon compilado y listo para despacho.

    Un GideonProgram es el resultado de bajar el AST de Poema a GideonIR.
    Contiene la topología completa de nodos, la secuencia de ejecución
    topológicamente ordenada, el FMA total y las cotas de error globales.
    """
    name: str = "gideon_program"
    nodes: Dict[str, IRNode] = field(default_factory=dict)
    output_ids: List[str] = field(default_factory=list)
    input_ids: List[str] = field(default_factory=list)
    topo_order: List[str] = field(default_factory=list)   # sorted execution order

    # Métricas globales
    total_fma: int = 0
    global_epsilon: float = 0.0
    creation_time: float = field(default_factory=time.time)

    # Cotas formales
    lean_theorems: List[str] = field(default_factory=list)
    domain: Tuple[float, float] = (-1.0, 1.0)

    def add_node(self, node: IRNode) -> None:
        self.nodes[node.node_id] = node

    def total_nodes(self) -> int:
        return len(self.nodes)

    def node_kinds(self) -> Dict[str, int]:
        """Cuenta de nodos por tipo."""
        counts: Dict[str, int] = {}
        for n in self.nodes.values():
            k = n.kind.value
            counts[k] = counts.get(k, 0) + 1
        return counts

    def summary(self) -> str:
        lines = [
            f"GideonProgram('{self.name}')",
            f"  Nodos:       {self.total_nodes()}",
            f"  FMA total:   {self.total_fma}",
            f"  ε global:    {self.global_epsilon:.4e}",
            f"  Outputs:     {self.output_ids}",
            f"  Tipos:       {self.node_kinds()}",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        data = {
            "name": self.name,
            "total_fma": self.total_fma,
            "global_epsilon": self.global_epsilon,
            "domain": list(self.domain),
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "topo_order": self.topo_order,
            "input_ids": self.input_ids,
            "output_ids": self.output_ids,
            "lean_theorems": self.lean_theorems,
        }
        return json.dumps(data, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# GideonIR — constructor / lowering
# ─────────────────────────────────────────────────────────────────────────────

class GideonIR:
    """
    Constructor de GideonProgram a partir de secuencias FMA o AST de Poema.

    Responsabilidades:
      - Bajar (lower) secuencias FMA a IRNodes tipados
      - Propagar intervalos a través del grafo
      - Calcular cotas epsilon acumuladas
      - Ordenar topológicamente para ejecución
      - Serializar / deserializar programas (JSON)
    """

    def __init__(self) -> None:
        self._counter: int = 0

    # ── ID generator ─────────────────────────────────────────────────────────

    def _new_id(self, prefix: str = "n") -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:04d}"

    # ── Lowering from FMA sequence ────────────────────────────────────────────

    def from_fma_sequence(
        self,
        fma_seq: List[Any],
        domain: Tuple[float, float] = (-1.0, 1.0),
        precision: str = "fp64",
        name: str = "fma_program",
    ) -> GideonProgram:
        """
        Convierte una secuencia FMA (lista de objetos con .weight y .bias)
        a un GideonProgram completamente tipado.
        """
        prog = GideonProgram(name=name, domain=domain)

        # Epsilon base por precisión
        eps_map = {
            "fp64": 2.22e-16,
            "fp32": 1.19e-7,
            "fp16": 9.77e-4,
            "bf16": 3.91e-3,
        }
        unit_eps = eps_map.get(precision, 2.22e-16)

        # Nodo de entrada
        inp_id = self._new_id("inp")
        inp_node = IRNode(
            kind=IRNodeKind.INPUT,
            node_id=inp_id,
            params={"precision": precision},
        )
        inp_node.meta.interval_lo = domain[0]
        inp_node.meta.interval_hi = domain[1]
        prog.add_node(inp_node)
        prog.input_ids.append(inp_id)
        prog.topo_order.append(inp_id)

        prev_id = inp_id
        prev_lo, prev_hi = float(domain[0]), float(domain[1])
        cumulative_eps = 0.0
        total_fma = 0

        for fma_instr in fma_seq:
            w = float(getattr(fma_instr, "weight", 1.0))
            b = float(getattr(fma_instr, "bias", 0.0))

            nid = self._new_id("fma")
            node = IRNode(
                kind=IRNodeKind.FMA,
                node_id=nid,
                inputs=[prev_id],
                params={"weight": w, "bias": b},
            )

            # Propagación de intervalos: y = w*x + b
            new_lo = w * prev_lo + b if w >= 0 else w * prev_hi + b
            new_hi = w * prev_hi + b if w >= 0 else w * prev_lo + b
            node.meta.interval_lo = new_lo
            node.meta.interval_hi = new_hi

            # Acumulación de error: ε_i ≈ |w|·ε_{i-1} + unit_eps
            cumulative_eps = abs(w) * cumulative_eps + unit_eps
            node.meta.epsilon_bound = cumulative_eps
            node.meta.fma_cost = 1

            prog.add_node(node)
            prog.topo_order.append(nid)
            prev_id = nid
            prev_lo, prev_hi = new_lo, new_hi
            total_fma += 1

        prog.output_ids = [prev_id]
        prog.total_fma = total_fma
        prog.global_epsilon = cumulative_eps
        return prog

    # ── Lowering from Poema AST ───────────────────────────────────────────────

    def from_ast(
        self,
        ast_node: Any,
        domain: Tuple[float, float] = (-1.0, 1.0),
        precision: str = "fp64",
        name: str = "ast_program",
    ) -> GideonProgram:
        """
        Baja (lower) un AST de Poema (ASTNode) a GideonProgram.
        Hace un recorrido DFS y emite IRNodes.
        """
        prog = GideonProgram(name=name, domain=domain)
        eps_map = {"fp64": 2.22e-16, "fp32": 1.19e-7, "fp16": 9.77e-4}
        unit_eps = eps_map.get(precision, 2.22e-16)

        inp_id = self._new_id("inp")
        inp_node = IRNode(kind=IRNodeKind.INPUT, node_id=inp_id,
                          params={"precision": precision})
        inp_node.meta.interval_lo = domain[0]
        inp_node.meta.interval_hi = domain[1]
        prog.add_node(inp_node)
        prog.input_ids.append(inp_id)
        prog.topo_order.append(inp_id)

        out_id = self._lower_ast_node(ast_node, prog, inp_id, unit_eps)
        prog.output_ids = [out_id]

        # Calcular métricas globales
        prog.total_fma = sum(n.meta.fma_cost for n in prog.nodes.values())
        prog.global_epsilon = max(
            (n.meta.epsilon_bound for n in prog.nodes.values()), default=0.0
        )
        return prog

    def _lower_ast_node(
        self,
        ast_node: Any,
        prog: GideonProgram,
        prev_id: str,
        unit_eps: float,
    ) -> str:
        """DFS lowering; devuelve el ID del nodo de salida producido."""
        node_type = type(ast_node).__name__

        if node_type == "IdentityNode":
            return prev_id

        if node_type == "ConstantNode":
            nid = self._new_id("const")
            val = float(getattr(ast_node, "value", 0.0))
            n = IRNode(kind=IRNodeKind.CONST, node_id=nid,
                       inputs=[prev_id], params={"value": val})
            n.meta.interval_lo = val
            n.meta.interval_hi = val
            n.meta.fma_cost = 1
            prog.add_node(n); prog.topo_order.append(nid)
            return nid

        if node_type in ("ScaleNode",):
            nid = self._new_id("scale")
            alpha = float(getattr(ast_node, "scale_factor", 1.0))
            n = IRNode(kind=IRNodeKind.SCALE, node_id=nid,
                       inputs=[prev_id], params={"alpha": alpha})
            n.meta.fma_cost = 1
            prog.add_node(n); prog.topo_order.append(nid)
            return nid

        if node_type in ("ShiftNode",):
            nid = self._new_id("shift")
            beta = float(getattr(ast_node, "shift_value", 0.0))
            n = IRNode(kind=IRNodeKind.SHIFT, node_id=nid,
                       inputs=[prev_id], params={"beta": beta})
            n.meta.fma_cost = 1
            prog.add_node(n); prog.topo_order.append(nid)
            return nid

        if node_type == "AffineNode":
            nid = self._new_id("affine")
            alpha = float(getattr(ast_node, "scale_factor", 1.0))
            beta  = float(getattr(ast_node, "shift_value", 0.0))
            n = IRNode(kind=IRNodeKind.AFFINE, node_id=nid,
                       inputs=[prev_id], params={"alpha": alpha, "beta": beta})
            n.meta.fma_cost = 1
            prog.add_node(n); prog.topo_order.append(nid)
            return nid

        if node_type == "ComposeNode":
            inner = getattr(ast_node, "inner", None)
            outer = getattr(ast_node, "outer", None)
            mid_id = self._lower_ast_node(inner, prog, prev_id, unit_eps) if inner else prev_id
            return self._lower_ast_node(outer, prog, mid_id, unit_eps) if outer else mid_id

        if node_type == "PolynomialNode":
            coeffs = list(getattr(ast_node, "coefficients", [1.0, 0.0]))
            nid = self._new_id("poly")
            n = IRNode(kind=IRNodeKind.POLY_HORNER, node_id=nid,
                       inputs=[prev_id], params={"coefficients": coeffs})
            n.meta.fma_cost = max(0, len(coeffs) - 1)
            prog.add_node(n); prog.topo_order.append(nid)
            return nid

        if node_type == "TranscendentalNode":
            fn_name = str(getattr(ast_node, "fn_name", "sin")).lower()
            kind_map = {
                "sin": IRNodeKind.SIN, "cos": IRNodeKind.COS,
                "exp": IRNodeKind.EXP, "log": IRNodeKind.LOG,
                "tanh": IRNodeKind.TANH, "sigmoid": IRNodeKind.SIGMOID,
            }
            kind = kind_map.get(fn_name, IRNodeKind.SIN)
            nid = self._new_id(fn_name)
            degree = int(getattr(ast_node, "chebyshev_degree", 8))
            eps = float(getattr(ast_node, "epsilon", unit_eps))
            n = IRNode(kind=kind, node_id=nid, inputs=[prev_id],
                       params={"fn_name": fn_name, "chebyshev_degree": degree})
            n.meta.epsilon_bound = eps
            n.meta.fma_cost = degree
            prog.add_node(n); prog.topo_order.append(nid)
            return nid

        if node_type == "StratifiedNode":
            nid = self._new_id("branch")
            n = IRNode(kind=IRNodeKind.BRANCH, node_id=nid,
                       inputs=[prev_id], params={})
            n.meta.fma_cost = 2
            prog.add_node(n); prog.topo_order.append(nid)
            return nid

        # Fallback genérico
        nid = self._new_id("generic")
        n = IRNode(kind=IRNodeKind.FMA, node_id=nid, inputs=[prev_id],
                   params={"weight": 1.0, "bias": 0.0})
        prog.add_node(n); prog.topo_order.append(nid)
        return nid

    # ── Serialización ─────────────────────────────────────────────────────────

    @staticmethod
    def to_json(prog: GideonProgram) -> str:
        return prog.to_json()

    @staticmethod
    def from_json(data: str) -> GideonProgram:
        d = json.loads(data)
        prog = GideonProgram(
            name=d.get("name", "restored"),
            domain=tuple(d.get("domain", [-1.0, 1.0])),
            total_fma=d.get("total_fma", 0),
            global_epsilon=d.get("global_epsilon", 0.0),
        )
        for ndata in d.get("nodes", []):
            _interval = ndata["meta"].get("interval_output", [-math.inf, math.inf])
            meta = IRNodeMetadata(
                epsilon_bound=ndata["meta"]["epsilon_bound"],
                domain_lo=ndata["meta"]["domain"][0],
                domain_hi=ndata["meta"]["domain"][1],
                interval_lo=_interval[0],
                interval_hi=_interval[1],
                fma_cost=ndata["meta"]["fma_cost"],
                verified=ndata["meta"]["verified"],
            )
            node = IRNode(
                kind=IRNodeKind(ndata["kind"]),
                node_id=ndata["id"],
                inputs=ndata["inputs"],
                params=ndata["params"],
                meta=meta,
            )
            prog.add_node(node)
        prog.topo_order = d.get("topo_order", [])
        prog.input_ids = d.get("input_ids", [])
        prog.output_ids = d.get("output_ids", [])
        prog.lean_theorems = d.get("lean_theorems", [])
        return prog

    # ── Optimization passes ───────────────────────────────────────────────────

    @staticmethod
    def fold_affine_chain(prog: "GideonProgram") -> Optional[Tuple[float, float]]:
        """
        Pasa de optimización: reduce una cadena puramente afín a un único AFFINE.

        FMA(wN,bN) ∘ … ∘ FMA(w1,b1)(x)  =  W·x + B

        Recorre el orden topológico y acumula la composición.
        Devuelve (W, B) si toda la cadena es afín. Devuelve None si contiene
        nodos no lineales (SIN, EXP, ATTENTION, CONV, etc.) que impiden el
        pliegue estático.

        Uso:
            wb = GideonIR.fold_affine_chain(prog)
            if wb:
                W, B = wb
                y = W * x + B   # equivalente exacto de toda la cadena
        """
        _NON_AFFINE = frozenset({
            IRNodeKind.SIN, IRNodeKind.COS, IRNodeKind.EXP, IRNodeKind.LOG,
            IRNodeKind.TANH, IRNodeKind.SIGMOID,
            IRNodeKind.MATMUL, IRNodeKind.GEMM, IRNodeKind.CONV,
            IRNodeKind.NORM, IRNodeKind.ATTENTION,
            IRNodeKind.POLY_HORNER, IRNodeKind.POLY_CHEB,
            IRNodeKind.PARALLEL, IRNodeKind.BRANCH, IRNodeKind.COMPOSE,
            IRNodeKind.LOOP, IRNodeKind.RECURSIVE,
            IRNodeKind.ARCH_PROBE, IRNodeKind.THEOREM_SEED,
        })
        W, B = 1.0, 0.0
        for nid in prog.topo_order:
            node = prog.nodes.get(nid)
            if node is None:
                continue
            k, p = node.kind, node.params
            if k in _NON_AFFINE:
                return None
            elif k in (IRNodeKind.INPUT, IRNodeKind.IDENTITY):
                pass
            elif k == IRNodeKind.FMA:
                w, b = p.get("weight", 1.0), p.get("bias", 0.0)
                W, B = w * W, w * B + b
            elif k == IRNodeKind.AFFINE:
                a, b = p.get("alpha", 1.0), p.get("beta", 0.0)
                W, B = a * W, a * B + b
            elif k == IRNodeKind.SCALE:
                a = p.get("alpha", 1.0)
                W, B = a * W, a * B
            elif k == IRNodeKind.SHIFT:
                B = B + p.get("beta", 0.0)
            elif k == IRNodeKind.CONST:
                W, B = 0.0, p.get("value", 0.0)
        return (W, B)

    @staticmethod
    def chain_hash(fma_sequence: "List[Any]") -> str:
        """
        Calcula un hash SHA-256 estable (16 chars) de una cadena FMA.

        Sirve como clave para el caché de compilación en GideonEngine.
        El hash es determinista: mismas cadenas → mismo hash.
        """
        import hashlib
        if not fma_sequence:
            return "empty_chain_0000"
        parts = ";".join(
            f"{float(getattr(f, 'weight', 1.0)):.17g},{float(getattr(f, 'bias', 0.0)):.17g}"
            for f in fma_sequence
        )
        return hashlib.sha256(parts.encode()).hexdigest()[:16]
