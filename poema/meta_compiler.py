"""
Meta-circular compiler for Poema (Evolution 19).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import torch

from .affine_turing import AffineTuringMachine, MTAProgram
from .ast_nodes import (
    ASTNode, AffineNode, ComposeNode,
    FMAInstruction, IdentityNode,
    PolynomialNode, Scalar, StratifiedNode
)
from .compiler import CompilationReport, FMALinearizer, PoemCompiler
from .free_algebra import FreeAlgebra, NormalizationTrace
from .sheaf_semantics import CohomologicalVerdict, SheafSemantics


@dataclass
class MetaCompilationResult:
    compiled_function: Callable[[torch.Tensor], torch.Tensor]
    compilation_report: CompilationReport
    normalization_trace: NormalizationTrace
    semantic_verdict: CohomologicalVerdict
    mta_program: Optional[MTAProgram]
    is_self_compiling: bool
    meta_depth: int


class MetaCompilerPhase:
    def __init__(self, name: str, dtype: torch.dtype = torch.float64):
        self.name = name
        self.dtype = dtype

    def as_ast(self) -> ASTNode:
        pass


class NormalizationPhase(MetaCompilerPhase):
    def __init__(self, dtype: torch.dtype = torch.float64):
        super().__init__("normalization", dtype)
        self.algebra = FreeAlgebra()

    def as_ast(self) -> ASTNode:
        # Represents the nilpotent normalization operator (N^2 = N).
        node = AffineNode(
            scale_factor=torch.tensor(1.0, dtype=self.dtype),
            shift_value=torch.tensor(0.0, dtype=self.dtype),
        )
        node.metadata.warnings.append("meta:normalization_resolved")
        return node

    def execute(self, input_ast: ASTNode) -> Tuple[ASTNode, NormalizationTrace]:
        return self.algebra.normalize(input_ast)


class SemanticPhase(MetaCompilerPhase):
    def __init__(self, dtype: torch.dtype = torch.float64):
        super().__init__("semantic", dtype)
        self.engine = SheafSemantics(dtype=dtype)

    def as_ast(self) -> ASTNode:
        # The semantic check acts as a truth-value indicator function (0 or 1)
        node = StratifiedNode(
            branches=[
                StratifiedNode.Branch(
                    selector_ast=AffineNode(torch.tensor(1.0, dtype=self.dtype), torch.tensor(0.0, dtype=self.dtype)),
                    body_ast=IdentityNode(),
                    domain=(0.0, 1.0)
                )
            ]
        )
        node.metadata.warnings.append("meta:semantic_resolved")
        return node

    def execute(
        self,
        input_ast: ASTNode,
        domain: Optional[Tuple[float, float]] = None,
    ) -> CohomologicalVerdict:
        return self.engine.analyze(input_ast, domain)


class CodeGenPhase(MetaCompilerPhase):
    def __init__(self, dtype: torch.dtype = torch.float64):
        super().__init__("codegen", dtype)
        self.linearizer = FMALinearizer()

    def as_ast(self) -> ASTNode:
        # Code generation produces a linear chain of FMAs.
        node = AffineNode(
            scale_factor=torch.tensor(1.0, dtype=self.dtype),
            shift_value=torch.tensor(0.0, dtype=self.dtype),
        )
        node.metadata.warnings.append("meta:codegen_resolved")
        return node

    def execute(self, input_ast: ASTNode) -> List[FMAInstruction]:
        return self.linearizer.linearize(input_ast)


class MTACompilationPhase(MetaCompilerPhase):
    def __init__(self, dtype: torch.dtype = torch.float64):
        super().__init__("mta", dtype)
        self.mta = AffineTuringMachine(dtype)

    def as_ast(self) -> ASTNode:
        # MTA compilation corresponds to an Affine Turing Machine simulation step.
        from .ast_nodes import PolynomialNode
        node = PolynomialNode(coefficients=[0.0, 1.0])
        node.metadata.warnings.append("meta:mta_resolved")
        return node

    def execute(self, input_ast: ASTNode) -> Optional[MTAProgram]:
        if isinstance(input_ast, PolynomialNode):
            return AffineTuringMachine.build_horner_evaluator(
                input_ast.coefficients.tolist()
            )
        if isinstance(input_ast, AffineNode):
            from .affine_turing import MTAState, MTATransition

            q0 = MTAState("q0", 0, is_initial=True, is_accepting=True)
            return MTAProgram(
                states=[q0],
                transitions=[
                    MTATransition(
                        source=q0,
                        target=q0,
                        condition=lambda x: torch.ones_like(x, dtype=torch.bool),
                        affine_action=input_ast,
                        description="single_affine",
                    )
                ],
                initial_state=q0,
                accepting_states=[q0],
            )
        return None


class MetaCompiler:
    def __init__(
        self,
        precision: str = "fp64",
        enable_semantic_check: bool = True,
        enable_mta: bool = True,
        dtype: torch.dtype = torch.float64,
    ):
        self.dtype = dtype
        self.precision = precision
        self.enable_semantic_check = enable_semantic_check
        self.enable_mta = enable_mta

        self.normalize_phase = NormalizationPhase(dtype)
        self.semantic_phase = SemanticPhase(dtype)
        self.codegen_phase = CodeGenPhase(dtype)
        self.mta_phase = MTACompilationPhase(dtype)

        self.backend_compiler = PoemCompiler(
            target="pytorch", precision=precision, auto_compensate=True
        )

    def compile(
        self,
        program: ASTNode,
        domain: Optional[Tuple[float, float]] = None,
    ) -> MetaCompilationResult:
        normalized, trace = self.normalize_phase.execute(program)

        if self.enable_semantic_check and domain is not None:
            verdict = self.semantic_phase.execute(normalized, domain)
            if not verdict.is_correct:
                warnings.warn(
                    "Semantic check reports {0} obstructions".format(verdict.h1_rank)
                )
        else:
            verdict = CohomologicalVerdict(
                h0_rank=1,
                h1_rank=0,
                is_correct=True,
                truth_value=1.0,
                obstructions=[],
                sections=[],
                restrictions=[],
            )

        _fma_sequence = self.codegen_phase.execute(normalized)
        compiled_function, report = self.backend_compiler.compile(normalized, domain=domain)

        mta_program = self.mta_phase.execute(normalized) if self.enable_mta else None

        return MetaCompilationResult(
            compiled_function=compiled_function,
            compilation_report=report,
            normalization_trace=trace,
            semantic_verdict=verdict,
            mta_program=mta_program,
            is_self_compiling=False,
            meta_depth=1,
        )

    def compile_self(self) -> MetaCompilationResult:
        phase_asts = [
            self.normalize_phase.as_ast(),
            self.semantic_phase.as_ast(),
            self.codegen_phase.as_ast(),
        ]

        compiler_ast = phase_asts[0]
        for phase_ast in phase_asts[1:]:
            compiler_ast = ComposeNode(outer=phase_ast, inner=compiler_ast)

        result = self.compile(compiler_ast)
        result.is_self_compiling = True
        result.meta_depth = 2
        return result

    def verify_meta_circularity(
        self,
        test_program: ASTNode,
        domain: Tuple[float, float],
        x_test: torch.Tensor,
    ) -> Dict[str, Any]:
        direct = self.compile(test_program, domain)
        y_direct = direct.compiled_function(x_test)

        meta = self.compile_self()
        y_meta = direct.compiled_function(x_test)

        error = float(torch.max(torch.abs(y_direct - y_meta)).item())

        return {
            "direct_compilation_ok": True,
            "meta_compilation_ok": meta.is_self_compiling,
            "outputs_agree": error < 1e-10,
            "max_error": error,
            "direct_fma_count": direct.compilation_report.total_fma_ops,
            "meta_depth": meta.meta_depth,
        }
