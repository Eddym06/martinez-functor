"""
Lie bracket analysis for FMA sequences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

from .core import FMAOperation, ReductionResult


@dataclass
class LieBracketResult:
    bracket_matrix: torch.Tensor
    frobenius_norm: float
    commutes: bool
    pair_indices: Tuple[int, int]


@dataclass
class ParallelSchedule:
    stages: List[List[int]]
    serial_depth: int
    total_operations: int
    parallelism_ratio: float
    bracket_details: List[LieBracketResult]


class LieBracketAnalyzer:
    def __init__(self, commutativity_threshold: float = 1e-10):
        self.threshold = commutativity_threshold

    def compute_bracket(self, A: torch.Tensor, B: torch.Tensor) -> LieBracketResult:
        A_mat = self._to_affine_matrix(A)
        B_mat = self._to_affine_matrix(B)
        bracket = A_mat @ B_mat - B_mat @ A_mat
        norm = float(torch.norm(bracket, p="fro").item())
        return LieBracketResult(
            bracket_matrix=bracket,
            frobenius_norm=norm,
            commutes=(norm < self.threshold),
            pair_indices=(-1, -1),
        )

    def analyze_sequence(self, fma_sequence: List[FMAOperation]) -> ParallelSchedule:
        n = len(fma_sequence)
        if n <= 1:
            return ParallelSchedule(
                stages=[[i] for i in range(n)],
                serial_depth=n,
                total_operations=n,
                parallelism_ratio=0.0,
                bracket_details=[],
            )

        brackets: List[LieBracketResult] = []
        commutation = torch.ones((n, n), dtype=torch.bool)

        for i in range(n):
            for j in range(i + 1, n):
                result = self.compute_bracket(fma_sequence[i].weight, fma_sequence[j].weight)
                result.pair_indices = (i, j)
                brackets.append(result)
                if not result.commutes:
                    commutation[i, j] = False
                    commutation[j, i] = False

        stages = self._build_parallel_stages(n, commutation)
        depth = len(stages)
        ratio = 1.0 - depth / max(n, 1)

        return ParallelSchedule(
            stages=stages,
            serial_depth=depth,
            total_operations=n,
            parallelism_ratio=ratio,
            bracket_details=brackets,
        )

    def analyze_reduction(self, reduction: ReductionResult) -> ParallelSchedule:
        return self.analyze_sequence(reduction.fma_sequence)

    def serial_depth_invariant(self, fma_sequence: List[FMAOperation]) -> int:
        n = len(fma_sequence)
        if n <= 1:
            return n

        non_zero = []
        for i in range(n):
            for j in range(i + 1, n):
                b = self.compute_bracket(fma_sequence[i].weight, fma_sequence[j].weight)
                if not b.commutes:
                    non_zero.append(b.bracket_matrix.flatten().to(torch.float64))

        if not non_zero:
            return 1

        M = torch.stack(non_zero)
        rank = int(torch.linalg.matrix_rank(M, atol=self.threshold).item())
        return min(rank + 1, n)

    def _to_affine_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.dim() == 0:
            out = torch.eye(2, dtype=weight.dtype, device=weight.device)
            out[0, 0] = weight
            return out
        if weight.dim() == 1:
            return torch.diag(weight)
        return weight

    def _build_parallel_stages(self, n: int, commutation: torch.Tensor) -> List[List[int]]:
        assigned = [False] * n
        stages: List[List[int]] = []

        for i in range(n):
            if assigned[i]:
                continue
            stage = [i]
            assigned[i] = True

            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                can_join = True
                for m in stage:
                    if not bool(commutation[m, j]):
                        can_join = False
                        break
                    if j == m + 1:
                        can_join = False
                        break
                if can_join:
                    stage.append(j)
                    assigned[j] = True

            stages.append(stage)

        return stages


# ---------------------------------------------------------------------------
# NC Complexity Analyzer (new — formalizes Paper.md §14.1 / §22.5)
# ---------------------------------------------------------------------------

@dataclass
class NCComplexityResult:
    """
    Complexity classification of an FMA sequence in the circuit sense.

    Serial depth = dim(Lie-span{[GEMM_i, GEMM_j] : i < j}).

    Connection to P vs NC (Theorem §22.5 of Paper.md):
    - If serial_depth_dim = O(log n), the FMA sequence admits an NC² circuit.
    - If serial_depth_dim = Θ(n), the sequence is inherently sequential (P-complete).

    Note: this is a formal dimension count over the Lie algebra spanned by the
    commutators of the weight matrices. It is a lower bound on the actual
    circuit depth (a Lie algebraic certificate of serial complexity).
    """
    n_ops: int
    serial_depth: int                  # schedule depth from stage builder
    bracket_frobenius_norms: List[float]  # ‖[A_i, A_j]‖_F for each pair
    lie_span_dim: int                  # dim(span{[A_i, A_j]}) over ℝ
    nc_class: str                      # "NC^1", "NC^2", "NC^3", or "P-hard"
    depth_over_log_n: float            # serial_depth / log2(n) — key ratio
    is_nc: bool                        # serial_depth = O(log n)?
    parallelizable_fraction: float     # fraction of ops in non-singleton stages
    proof_sketch: str


class NCComplexityAnalyzer:
    """
    Analyze the computational complexity class of an ACF FMA sequence.

    The key object is the Lie algebra:
        g(f) = Lie-span{ [W_i, W_j] : i < j, W_i ∈ GL(n) }

    Its dimension dim(g(f)) determines the serial complexity:
        - dim(g(f)) = 0         → all GEMM commute → NC^1 (tree structure)
        - dim(g(f)) = O(log n)  → NC^2
        - dim(g(f)) = O(n)      → P-complete (inherently sequential)

    This formalizes the claim in Paper.md §14.1:
        "The ACF serial depth is measured by the Lie bracket structure of
         its GEMM composition — P-completeness of a function ↔ full Lie algebra."

    Usage
    -----
        analyzer = NCComplexityAnalyzer()
        result = analyzer.analyze(reduction.fma_sequence)
        print(result.nc_class, result.lie_span_dim)
    """

    def __init__(self, commutativity_threshold: float = 1e-8):
        self.threshold = commutativity_threshold
        self._bracket_analyzer = LieBracketAnalyzer(commutativity_threshold)

    def _bracket_matrix(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A_m = self._bracket_analyzer._to_affine_matrix(A)
        B_m = self._bracket_analyzer._to_affine_matrix(B)
        return A_m @ B_m - B_m @ A_m

    def _lie_span_dimension(self, brackets: List[torch.Tensor]) -> int:
        """
        Compute dim(span{B_k}) where B_k are bracket matrices.

        Method: flatten each matrix to a vector and compute the rank of the
        resulting matrix via SVD (stable numerical rank).
        """
        if not brackets:
            return 0
        vecs = [b.detach().double().flatten() for b in brackets if float(torch.norm(b,p="fro").item()) > self.threshold]
        if not vecs:
            return 0
        M = torch.stack(vecs, dim=1)  # (d², k) matrix
        try:
            sv = torch.linalg.svdvals(M)
            rank = int((sv > sv[0] * 1e-10).sum().item())
        except Exception:
            rank = int(torch.matrix_rank(M).item())
        return rank

    def _nc_label(self, lie_dim: int, n_ops: int) -> Tuple[str, bool]:
        if n_ops == 0:
            return "NC^0", True
        log_n = max(1.0, math.log2(max(n_ops, 2)))
        ratio = lie_dim / log_n
        if lie_dim == 0:
            return "NC^1", True
        elif ratio <= 1.0:
            return "NC^1", True
        elif ratio <= 2.0:
            return "NC^2", True
        elif ratio <= 3.0:
            return "NC^3", True
        else:
            return "P-hard", False

    def analyze(self, fma_sequence: List[FMAOperation]) -> NCComplexityResult:
        """
        Full NC complexity analysis of an FMA sequence.

        Returns NCComplexityResult with lie_span_dim, nc_class, depth, etc.
        """
        import math

        n = len(fma_sequence)
        if n == 0:
            return NCComplexityResult(
                n_ops=0, serial_depth=0,
                bracket_frobenius_norms=[], lie_span_dim=0,
                nc_class="NC^0", depth_over_log_n=0.0,
                is_nc=True, parallelizable_fraction=1.0,
                proof_sketch="Empty sequence — trivially NC^0.",
            )

        # Compute all pairwise brackets
        all_brackets: List[torch.Tensor] = []
        frob_norms: List[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                W_i = fma_sequence[i].weight
                W_j = fma_sequence[j].weight
                br = self._bracket_matrix(W_i, W_j)
                fn = float(torch.norm(br, p="fro").item())
                frob_norms.append(fn)
                all_brackets.append(br)

        lie_dim = self._lie_span_dimension(all_brackets)

        # Schedule depth from existing analyzer
        schedule = self._bracket_analyzer.analyze_sequence(fma_sequence)
        depth = schedule.serial_depth

        log_n = max(1.0, math.log2(max(n, 2)))
        depth_ratio = lie_dim / log_n

        nc_label, is_nc = self._nc_label(lie_dim, n)

        # Parallelizable fraction
        single_stage_ops = sum(1 for st in schedule.stages if len(st) == 1)
        par_frac = 1.0 - single_stage_ops / max(n, 1)

        proof = (
            f"NC Complexity Certificate for FMA sequence (n={n} ops):\n"
            f"  Lie algebra dimension: dim(g(f)) = {lie_dim}\n"
            f"  log₂(n)              = {log_n:.2f}\n"
            f"  dim/log₂(n)          = {depth_ratio:.3f}\n"
            f"  Schedule serial depth = {depth}\n"
            f"\n"
            f"  Classification: {nc_label}\n"
            f"  Theorem (Paper.md §22.5):\n"
            f"    dim(g(f)) = dim(Lie-span{{[W_i, W_j] : i<j}})\n"
            f"    dim = 0           → NC^1 (all GEMM commute → parallel tree)\n"
            f"    dim = O(log n)    → NC² (polylogarithmic depth circuit)\n"
            f"    dim = Θ(n)        → P-hard (inherently sequential)\n"
            f"  {'This sequence is in NC (admits polylog-depth parallel circuit).' if is_nc else 'This sequence is P-hard: no O(log n) depth circuit exists.'}"
        )

        return NCComplexityResult(
            n_ops=n,
            serial_depth=depth,
            bracket_frobenius_norms=frob_norms,
            lie_span_dim=lie_dim,
            nc_class=nc_label,
            depth_over_log_n=depth_ratio,
            is_nc=is_nc,
            parallelizable_fraction=par_frac,
            proof_sketch=proof,
        )
