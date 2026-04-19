"""
Poema AST.

Free algebra over the three primitive generators:
- scale(a): x -> a * x
- shift(b): x -> x + b
- compose(f, g): x -> f(g(x))

Nodes carry geometric types so composition validity is enforced at construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
import math
from typing import Any, List, Optional, Tuple, Union

import torch


@dataclass(frozen=True)
class GeometricType:
    input_dim: int
    output_dim: int
    continuity: int = -1
    domain_bounds: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
    symmetry_group: Optional[str] = None

    def is_composable_with(self, inner: "GeometricType") -> bool:
        return inner.output_dim == self.input_dim

    def compose_type(self, inner: "GeometricType") -> "GeometricType":
        if not self.is_composable_with(inner):
            raise TopologicalObstructionError(
                f"cannot compose: outer expects input_dim={self.input_dim}, "
                f"inner returns output_dim={inner.output_dim}"
            )
        return GeometricType(
            input_dim=inner.input_dim,
            output_dim=self.output_dim,
            continuity=min(self.continuity, inner.continuity),
            domain_bounds=inner.domain_bounds,
        )


def Scalar() -> GeometricType:
    return GeometricType(1, 1)


def Vector(n: int) -> GeometricType:
    return GeometricType(n, n)


def Morphism(n: int, m: int) -> GeometricType:
    return GeometricType(n, m)


def Flow(n: int) -> GeometricType:
    return GeometricType(n, n, continuity=-1)


def Form(k: int, n: int) -> GeometricType:
    return GeometricType(n, math.comb(n, k))


class TopologicalObstructionError(TypeError):
    def __init__(self, message: str, node: Optional["ASTNode"] = None):
        self.node = node
        super().__init__(
            f"TopologicalObstruction: {message}"
            + (f" at node={node}" if node is not None else "")
        )


class PrecisionDegradationWarning(UserWarning):
    pass


class NodeTag(Enum):
    SCALE = auto()
    SHIFT = auto()
    COMPOSE = auto()
    AFFINE = auto()
    IDENTITY = auto()
    POLYNOMIAL = auto()
    TRANSCENDENTAL = auto()
    KOOPMAN = auto()
    STRATIFIED = auto()
    SEQUENCE = auto()
    PARALLEL = auto()
    FEEDBACK = auto()
    PARAMETER = auto()
    CONSTANT = auto()
    INPUT = auto()
    CONSTRAINT = auto()
    OBJECTIVE = auto()
    COUPLING = auto()


@dataclass
class CompilationMetadata:
    fma_count: int = 0
    estimated_epsilon: float = 0.0
    requires_sheaf: bool = False
    lie_bracket_depth: int = 0
    parallelizable: bool = True
    compensation_nodes_injected: int = 0
    warnings: List[str] = field(default_factory=list)


class ASTNode(ABC):
    def __init__(
        self,
        tag: NodeTag,
        geometric_type: GeometricType,
        children: Optional[List["ASTNode"]] = None,
    ):
        self.tag = tag
        self.geometric_type = geometric_type
        self.children = children or []
        self.metadata = CompilationMetadata()

    @abstractmethod
    def simplify(self) -> "ASTNode":
        raise NotImplementedError

    @abstractmethod
    def estimate_fma_cost(self) -> int:
        raise NotImplementedError

    def compose_with(self, inner: "ASTNode") -> "ComposeNode":
        return ComposeNode(outer=self, inner=inner)

    def __matmul__(self, other: "ASTNode") -> "ComposeNode":
        return self.compose_with(other)


class ScaleNode(ASTNode):
    def __init__(
        self,
        factor: Union[float, torch.Tensor],
        child: Optional[ASTNode] = None,
        geometric_type: Optional[GeometricType] = None,
    ):
        self.factor = torch.as_tensor(factor, dtype=torch.float64)
        gt = geometric_type or (child.geometric_type if child is not None else Scalar())
        super().__init__(NodeTag.SCALE, gt, [child] if child is not None else [])

    def simplify(self) -> ASTNode:
        one = torch.tensor(1.0, dtype=self.factor.dtype)
        zero = torch.tensor(0.0, dtype=self.factor.dtype)
        if torch.allclose(self.factor, one):
            if self.children:
                return self.children[0].simplify()
            return IdentityNode(self.geometric_type)
        if torch.allclose(self.factor, zero):
            return ConstantNode(0.0, self.geometric_type)
        if self.children and isinstance(self.children[0], ScaleNode):
            inner = self.children[0]
            return ScaleNode(
                factor=self.factor * inner.factor,
                child=inner.children[0] if inner.children else None,
                geometric_type=self.geometric_type,
            ).simplify()
        return self

    def estimate_fma_cost(self) -> int:
        return (self.children[0].estimate_fma_cost() if self.children else 0) + 1


class ShiftNode(ASTNode):
    def __init__(
        self,
        value: Union[float, torch.Tensor],
        child: Optional[ASTNode] = None,
        geometric_type: Optional[GeometricType] = None,
    ):
        self.value = torch.as_tensor(value, dtype=torch.float64)
        gt = geometric_type or (child.geometric_type if child is not None else Scalar())
        super().__init__(NodeTag.SHIFT, gt, [child] if child is not None else [])

    def simplify(self) -> ASTNode:
        zero = torch.tensor(0.0, dtype=self.value.dtype)
        if torch.allclose(self.value, zero):
            if self.children:
                return self.children[0].simplify()
            return IdentityNode(self.geometric_type)
        if self.children and isinstance(self.children[0], ShiftNode):
            inner = self.children[0]
            return ShiftNode(
                value=self.value + inner.value,
                child=inner.children[0] if inner.children else None,
                geometric_type=self.geometric_type,
            ).simplify()
        return self

    def estimate_fma_cost(self) -> int:
        return (self.children[0].estimate_fma_cost() if self.children else 0) + 1


class ComposeNode(ASTNode):
    def __init__(self, outer: ASTNode, inner: ASTNode):
        self.outer = outer
        self.inner = inner
        gt = outer.geometric_type.compose_type(inner.geometric_type)
        super().__init__(NodeTag.COMPOSE, gt, [outer, inner])

    def simplify(self) -> ASTNode:
        outer = self.outer.simplify()
        inner = self.inner.simplify()

        if isinstance(outer, IdentityNode):
            return inner
        if isinstance(inner, IdentityNode):
            return outer

        if isinstance(outer, ScaleNode) and isinstance(inner, ShiftNode):
            return AffineNode(
                scale_factor=outer.factor,
                shift_value=outer.factor * inner.value,
                child=inner.children[0] if inner.children else None,
                geometric_type=self.geometric_type,
            ).simplify()

        if isinstance(outer, AffineNode) and isinstance(inner, AffineNode):
            return AffineNode(
                scale_factor=outer.scale_factor * inner.scale_factor,
                shift_value=outer.scale_factor * inner.shift_value + outer.shift_value,
                child=inner.children[0] if inner.children else None,
                geometric_type=self.geometric_type,
            ).simplify()

        return ComposeNode(outer, inner)

    def estimate_fma_cost(self) -> int:
        return self.outer.estimate_fma_cost() + self.inner.estimate_fma_cost()


class IdentityNode(ASTNode):
    def __init__(self, geometric_type: Optional[GeometricType] = None):
        super().__init__(NodeTag.IDENTITY, geometric_type or Scalar())

    def simplify(self) -> ASTNode:
        return self

    def estimate_fma_cost(self) -> int:
        return 0


class ConstantNode(ASTNode):
    def __init__(
        self,
        value: Union[float, torch.Tensor],
        geometric_type: Optional[GeometricType] = None,
    ):
        self.value = torch.as_tensor(value, dtype=torch.float64)
        super().__init__(NodeTag.CONSTANT, geometric_type or Scalar())

    def simplify(self) -> ASTNode:
        return self

    def estimate_fma_cost(self) -> int:
        return 1


class AffineNode(ASTNode):
    def __init__(
        self,
        scale_factor: Union[float, torch.Tensor],
        shift_value: Union[float, torch.Tensor],
        child: Optional[ASTNode] = None,
        geometric_type: Optional[GeometricType] = None,
    ):
        self.scale_factor = torch.as_tensor(scale_factor, dtype=torch.float64)
        self.shift_value = torch.as_tensor(shift_value, dtype=torch.float64)
        gt = geometric_type or (child.geometric_type if child is not None else Scalar())
        super().__init__(NodeTag.AFFINE, gt, [child] if child is not None else [])

    def simplify(self) -> ASTNode:
        one = torch.tensor(1.0, dtype=self.scale_factor.dtype)
        zero = torch.tensor(0.0, dtype=self.scale_factor.dtype)
        if torch.allclose(self.scale_factor, one) and torch.allclose(self.shift_value, zero):
            if self.children:
                return self.children[0].simplify()
            return IdentityNode(self.geometric_type)
        if torch.allclose(self.scale_factor, zero):
            return ConstantNode(self.shift_value, self.geometric_type)
        if self.children and isinstance(self.children[0], AffineNode):
            inner = self.children[0]
            return AffineNode(
                scale_factor=self.scale_factor * inner.scale_factor,
                shift_value=self.scale_factor * inner.shift_value + self.shift_value,
                child=inner.children[0] if inner.children else None,
                geometric_type=self.geometric_type,
            ).simplify()
        return self

    def estimate_fma_cost(self) -> int:
        return (self.children[0].estimate_fma_cost() if self.children else 0) + 1


class InputNode(ASTNode):
    def __init__(self, name: str = "x", geometric_type: Optional[GeometricType] = None):
        self.name = name
        super().__init__(NodeTag.INPUT, geometric_type or Scalar())

    def simplify(self) -> ASTNode:
        return self

    def estimate_fma_cost(self) -> int:
        return 0


class PolynomialNode(ASTNode):
    def __init__(
        self,
        coefficients: Union[List[float], torch.Tensor],
        input_node: Optional[InputNode] = None,
        geometric_type: Optional[GeometricType] = None,
    ):
        self.coefficients = torch.as_tensor(coefficients, dtype=torch.float64)
        self.input_node = input_node or InputNode()
        super().__init__(NodeTag.POLYNOMIAL, geometric_type or Scalar())
        self.metadata.fma_count = max(1, int(self.coefficients.numel()) - 1)

    def simplify(self) -> ASTNode:
        coeffs = self.coefficients
        while coeffs.numel() > 1 and torch.abs(coeffs[-1]).item() < 1e-30:
            coeffs = coeffs[:-1]
        if coeffs.numel() == 1:
            return ConstantNode(coeffs[0], self.geometric_type)
        self.coefficients = coeffs
        return self

    def estimate_fma_cost(self) -> int:
        return max(1, int(self.coefficients.numel()) - 1)


class TranscendentalNode(ASTNode):
    def __init__(
        self,
        name: str,
        polynomial: PolynomialNode,
        certified_epsilon: float,
        original_domain: Tuple[float, float],
        geometric_type: Optional[GeometricType] = None,
        chebyshev_coefficients: Optional[Union[List[float], torch.Tensor]] = None,
        evaluation_mode: str = "horner",
    ):
        self.name = name
        self.polynomial = polynomial
        self.certified_epsilon = float(certified_epsilon)
        self.original_domain = original_domain
        self.chebyshev_coefficients = (
            torch.as_tensor(chebyshev_coefficients, dtype=torch.float64)
            if chebyshev_coefficients is not None
            else None
        )
        self.evaluation_mode = evaluation_mode
        super().__init__(
            NodeTag.TRANSCENDENTAL,
            geometric_type or Scalar(),
            children=[polynomial],
        )
        self.metadata.estimated_epsilon = self.certified_epsilon

    def simplify(self) -> ASTNode:
        simplified = self.polynomial.simplify()
        if isinstance(simplified, PolynomialNode):
            self.polynomial = simplified
        elif isinstance(simplified, ConstantNode):
            self.polynomial = PolynomialNode([float(simplified.value.item())])
        return self

    def estimate_fma_cost(self) -> int:
        return self.polynomial.estimate_fma_cost()


class StratifiedNode(ASTNode):
    @dataclass
    class Branch:
        selector_ast: ASTNode
        body_ast: ASTNode
        domain: Tuple[float, float]
        tear_type: Optional[str] = None

    def __init__(
        self,
        branches: List["StratifiedNode.Branch"],
        geometric_type: Optional[GeometricType] = None,
    ):
        self.branches = branches
        children: List[ASTNode] = []
        for b in branches:
            children.extend([b.selector_ast, b.body_ast])
        super().__init__(NodeTag.STRATIFIED, geometric_type or Scalar(), children)
        self.metadata.requires_sheaf = True

    def simplify(self) -> ASTNode:
        for b in self.branches:
            b.body_ast = b.body_ast.simplify()
        return self

    def estimate_fma_cost(self) -> int:
        return sum(b.body_ast.estimate_fma_cost() + 1 for b in self.branches)


class ParameterNode(ASTNode):
    def __init__(
        self,
        name: str,
        initial_value: Union[float, torch.Tensor],
        requires_grad: bool = True,
        geometric_type: Optional[GeometricType] = None,
    ):
        self.name = name
        self.param_value = torch.as_tensor(initial_value, dtype=torch.float64)
        self.requires_grad = requires_grad
        super().__init__(NodeTag.PARAMETER, geometric_type or Scalar())

    def simplify(self) -> ASTNode:
        return self

    def estimate_fma_cost(self) -> int:
        return 0


class ConstraintNode(ASTNode):
    def __init__(
        self,
        constraint_type: str,
        value: Any,
        geometric_type: Optional[GeometricType] = None,
    ):
        self.constraint_type = constraint_type
        self.constraint_value = value
        super().__init__(NodeTag.CONSTRAINT, geometric_type or Scalar())

    def simplify(self) -> ASTNode:
        return self

    def estimate_fma_cost(self) -> int:
        return 0


@dataclass
class FMAInstruction:
    weight: torch.Tensor
    bias: torch.Tensor
    source_node: Optional[ASTNode] = None
    precision: str = "fp64"

    def execute(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(dtype=x.dtype, device=x.device)
        b = self.bias.to(dtype=x.dtype, device=x.device)
        if w.dim() >= 2:
            return torch.matmul(x, w) + b
        return b.expand_as(x) + w.expand_as(x) * x

class PiecewiseNode(ASTNode):
    def __init__(self, cond: ASTNode, true_expr: ASTNode, false_expr: ASTNode):
        from .ast_nodes import NodeTag, Scalar
        super().__init__(NodeTag.STRATIFIED, Scalar())
        self.children = [cond, true_expr, false_expr]
        self.cond = cond
        self.true_expr = true_expr
        self.false_expr = false_expr
        self._fma_cost = 2
        
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        cond_val = self.cond.evaluate(x)
        mask = cond_val > 0
        res = torch.empty_like(cond_val, dtype=x.dtype)
        if mask.any():
            res[mask] = self.true_expr.evaluate(x)[mask]
        if (~mask).any():
            res[~mask] = self.false_expr.evaluate(x)[~mask]
        return res

    def simplify(self) -> 'ASTNode': return self
    def estimate_fma_cost(self) -> int: return 2

class LoopNode(ASTNode):
    def __init__(self, init: ASTNode, cond: ASTNode, body: ASTNode):
        super().__init__(NodeTag.SEQUENCE, Scalar())
        self.children = [init, cond, body]
        self.init = init
        self.cond = cond
        self.body = body
        
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        val = self.init.evaluate(x)
        for _ in range(1000):
            if torch.all(self.cond.evaluate(val) <= 0):
                break
            val = self.body.evaluate(val)
        return val

    def simplify(self) -> 'ASTNode': return self
    def estimate_fma_cost(self) -> int: return 50

class DefNode(ASTNode):
    def __init__(self, name: str, args: list, body: ASTNode):
        super().__init__(NodeTag.FEEDBACK, Scalar())
        self.children = [body]
        self.name = name
        self.args = args
        self.body = body
        
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return self.body.evaluate(x)

    def simplify(self) -> 'ASTNode': return self
    def estimate_fma_cost(self) -> int: return 1

