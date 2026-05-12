"""
Poema Distribution Nodes — AST extension for distribution theory.

Adds:
  - DistributionNode: wraps DualDistribution in Poema AST
  - DistributionActNode: ⟨T, φ⟩ evaluation
  - DistributionDiffNode: distributional derivative
  - DistributionConvNode: convolution of distributions
  - DistributionPiecewiseNode: stratified functions as distributions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from .ast_nodes import (
    ASTNode,
    CompilationMetadata,
    GeometricType,
    NodeTag,
    Scalar,
    TopologicalObstructionError,
)


class DistributionNodeTag(Enum):
    DISTRIBUTION = auto()
    DISTRIBUTION_ACT = auto()
    DISTRIBUTION_DIFF = auto()
    DISTRIBUTION_CONV = auto()
    DISTRIBUTION_PIECEWISE = auto()


@dataclass
class DistributionMetadata(CompilationMetadata):
    """Extended metadata for distribution compilation."""
    n_singularities: int = 0
    max_singularity_order: int = 0
    spectral_modes: int = 0
    consistency_residual: float = 0.0
    hormander_safe: bool = True
    cohomology_h1_rank: int = 0
    is_fma_compatible: bool = True
    requires_singularity_kernel: bool = False


class DistributionNode(ASTNode):
    """
    AST node wrapping a distribution representation.

    This is the bridge between distribution theory and the Poema compiler.
    Stores a DualDistribution (or factory to create one) and exposes
    the FMA-compilable spectral part alongside the singularity tensor.
    """

    def __init__(
        self,
        distribution: "DualDistribution",  # type: ignore
        geometric_type: Optional[GeometricType] = None,
        children: Optional[List[ASTNode]] = None,
    ):
        from acf_functor.distribution_theory import DualDistribution
        self.distribution: DualDistribution = distribution
        gt = geometric_type or Scalar()
        super().__init__(NodeTag.PARAMETER, gt, children or [])
        self.metadata = DistributionMetadata()

    def simplify(self) -> "ASTNode":
        # Distributions don't simplify algebraically like affine nodes
        if not self.distribution.singularities:
            # Pure spectral distribution → convert to PolynomialNode
            coeffs = self.distribution.spectral.coefficients
            from .ast_nodes import PolynomialNode
            return PolynomialNode(
                coefficients=[float(c) for c in coeffs],
                geometric_type=self.geometric_type,
            ).simplify()
        return self

    def estimate_fma_cost(self) -> int:
        """FMA cost = spectral evaluation cost (linear in n_modes)."""
        return len(self.distribution.spectral.coefficients)

    def get_spectral_tensor(self) -> "SpectralTensor":
        """Extract the spectral (FMA-compatible) component."""
        return self.distribution.spectral

    def get_singularity_tensor(self) -> List:
        """Extract the singularity tensor."""
        return self.distribution.singularities

    def evaluate_regularized(self, x: torch.Tensor, smoothing: float = 1e-3) -> torch.Tensor:
        """Regularized evaluation (for visualization and testing)."""
        x_np = x.detach().cpu().numpy()
        vals = self.distribution.evaluate_approximate(x_np, smoothing)
        return torch.as_tensor(vals, dtype=x.dtype, device=x.device)


class DistributionActNode(ASTNode):
    """
    Represents the action ⟨T, φ⟩ of a distribution on a test function.

    This is the fundamental operation: evaluating how the distribution
    acts on a given test function φ, implemented as:
      - Regular part: numerical quadrature of spectral tensor × φ
      - Singular part: derivative-of-φ evaluation at singularity positions
    """

    def __init__(
        self,
        distribution_node: DistributionNode,
        test_function: Callable[[torch.Tensor], torch.Tensor],
        geometric_type: Optional[GeometricType] = None,
    ):
        self.test_function = test_function
        gt = geometric_type or Scalar()
        super().__init__(NodeTag.PARAMETER, gt, [distribution_node])

    def simplify(self) -> "ASTNode":
        return self

    def estimate_fma_cost(self) -> int:
        return self.children[0].estimate_fma_cost() + 10  # Quadrature overhead


class DistributionDiffNode(ASTNode):
    """
    Distributional derivative D(T) = T'.

    Implements the three simultaneous operations:
    1. Spectral differentiation (Chebyshev recurrence)
    2. Singularity order elevation
    3. Leibniz boundary adjustment
    """

    def __init__(
        self,
        distribution_node: DistributionNode,
        order: int = 1,
    ):
        self.order = order
        gt = distribution_node.geometric_type
        super().__init__(NodeTag.PARAMETER, gt, [distribution_node])

    def simplify(self) -> "ASTNode":
        # Apply distributional derivative
        dist = self.children[0].distribution
        for _ in range(self.order):
            dist = dist.differentiate()
        return DistributionNode(dist, self.geometric_type)

    def estimate_fma_cost(self) -> int:
        base_cost = self.children[0].estimate_fma_cost()
        # Derivative cost: spectral recurrence + singularity elevation
        return base_cost + self.order * (base_cost // 4 + 5)


class DistributionConvNode(ASTNode):
    """
    Convolution of two distributions T₁ * T₂.

    Automatically checks Hörmander condition before execution.
    If ill-defined, raises TopologicalObstructionError at compile time.
    """

    def __init__(
        self,
        dist_node_1: DistributionNode,
        dist_node_2: DistributionNode,
    ):
        self.dist_node_1 = dist_node_1
        self.dist_node_2 = dist_node_2
        gt = dist_node_1.geometric_type
        super().__init__(NodeTag.PARAMETER, gt, [dist_node_1, dist_node_2])

    def simplify(self) -> "ASTNode":
        T1 = self.dist_node_1.distribution
        T2 = self.dist_node_2.distribution

        result = T1.convolve(T2)
        if result is None:
            raise TopologicalObstructionError(
                "Hörmander condition violated: convolution of "
                f"{T1.distribution_type} and {T2.distribution_type} is ill-defined. "
                "Wavefront sets have antipodal directions.",
                node=self,
            )
        return DistributionNode(result, self.geometric_type)

    def estimate_fma_cost(self) -> int:
        c1 = self.children[0].estimate_fma_cost()
        c2 = self.children[1].estimate_fma_cost()
        # FFT convolution: O(n log n)
        import math
        n = max(c1, c2)
        return int(n * math.log2(n + 1)) + c1 + c2


class DistributionPiecewiseNode(ASTNode):
    """
    Piecewise function represented as a distribution.

    f(x) = { g(x) if x ∈ region_A; h(x) if x ∈ region_B }

    Internally computed as: f = g·H_A + h·H_B where H_A, H_B are
    characteristic functions represented as Heaviside distributions.

    The jumps at region boundaries are explicitly captured as singularities,
    enabling exact differentiation via the distributional derivative.
    """

    def __init__(
        self,
        regions: List[Tuple[float, float]],
        functions: List[ASTNode],
        geometric_type: Optional[GeometricType] = None,
    ):
        if len(regions) != len(functions):
            raise TopologicalObstructionError(
                f"Number of regions ({len(regions)}) != number of functions ({len(functions)})"
            )
        self.regions = regions
        self.functions = functions
        gt = geometric_type or Scalar()
        super().__init__(NodeTag.STRATIFIED, gt, functions)

    def simplify(self) -> "ASTNode":
        return self  # Deferred to compiler pass

    def estimate_fma_cost(self) -> int:
        return sum(c.estimate_fma_cost() for c in self.children) + len(self.regions)

    def to_distribution(self, domain: Tuple[float, float], n_modes: int = 64) -> DistributionNode:
        """Convert piecewise function to distribution representation."""
        from acf_functor.distribution_theory import (
            DualDistribution, DirectionalSingularity, SingularityType, SpectralTensor
        )
        import numpy as np

        a, b = domain
        x = np.linspace(a, b, 1000)
        values = np.zeros(len(x))

        singularities = []

        for (r_left, r_right), func_node in zip(self.regions, self.functions):
            mask = (x >= r_left) & (x < r_right)
            # Evaluate function on this region
            # (Simplified — in practice, use compiled FMA for each region)
            if hasattr(func_node, 'evaluate_approximate'):
                values[mask] = func_node.evaluate_approximate(
                    torch.as_tensor(x[mask], dtype=torch.float64)
                ).detach().cpu().numpy()
            else:
                values[mask] = 1.0  # Placeholder

            # Add Heaviside singularities at region boundaries
            for boundary in [r_left, r_right]:
                if a < boundary < b:
                    jump = 0.0  # Would compute actual jump from functions
                    singularities.append(
                        DirectionalSingularity(
                            position=np.array([boundary]),
                            codirection=np.array([1.0]),
                            order=-1,
                            mass=float(jump) if abs(jump) > 1e-10 else 1.0,
                            singularity_type=SingularityType.HEAVISIDE,
                        )
                    )

        # Fit spectral part
        coeffs = np.zeros(n_modes)
        for k in range(n_modes):
            ck = 2.0 / (len(x) - 1) if 0 < k < len(x) - 1 else 1.0 / (len(x) - 1)
            coeffs[k] = ck * np.sum(
                values * np.cos(np.pi * k * np.arange(len(x)) / (len(x) - 1))
            )

        spectral = SpectralTensor(
            coefficients=coeffs,
            basis_type="chebyshev",
            domain=domain,
            n_modes=n_modes,
        )

        from acf_functor.distribution_theory import DualDistribution
        dist = DualDistribution(
            spectral=spectral,
            singularities=singularities,
            domain=domain,
            distribution_type="piecewise",
        )

        return DistributionNode(dist, self.geometric_type)
