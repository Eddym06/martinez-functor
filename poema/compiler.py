"""
Poema compiler.

Pipeline:
1) Frontend builds AST
2) Middle-end applies algebraic and geometric checks, compensation, optional self-modulation
3) Backend lowers to executable kernels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import time
import warnings

import torch

from .ast_nodes import (
    ASTNode,
    AffineNode,
    ComposeNode,
    ConstantNode,
    FMAInstruction,
    GeometricType,
    IdentityNode,
    InputNode,
    PolynomialNode,
    PrecisionDegradationWarning,
    ScaleNode,
    ShiftNode,
    StratifiedNode,
    TopologicalObstructionError,
    TranscendentalNode,
)
from .frontend import _CompoundAddNode, _CompoundMulNode


@dataclass
class NodeProfile:
    """Perfil de un nodo individual en el AST compilado."""
    node_type: str = ""
    node_id: str = ""
    fma_contribution: int = 0
    epsilon_contribution: float = 0.0
    domain_interval: Optional[Tuple[float, float]] = None
    simplification_applied: bool = False
    simplification_rule: str = ""
    domain_guard_status: str = "ok"  # ok, warning, violation, repaired

    def summary(self) -> str:
        return (
            f"{self.node_type}({self.node_id}): "
            f"fma={self.fma_contribution}, "
            f"ε={self.epsilon_contribution:.3e}, "
            f"domain={self.domain_interval}, "
            f"guard={self.domain_guard_status}"
        )


@dataclass
class CompilationReport:
    total_fma_ops: int = 0
    total_epsilon: float = 0.0
    simplifications_applied: int = 0
    compensations_injected: int = 0
    sheaves_injected: int = 0
    lie_bracket_depth: int = 0
    parallelizable_chains: int = 0
    compilation_time_ms: float = 0.0
    domain_guard_checks: int = 0
    domain_guard_violations: int = 0
    domain_guard_max_overshoot: float = 0.0
    domain_guard_alerts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fma_sequence: List[FMAInstruction] = field(default_factory=list)
    
    # Extended observability (Section 2.8 of technical report)
    node_profiles: List[NodeProfile] = field(default_factory=list)
    phase_times: Dict[str, float] = field(default_factory=dict)  # ms per phase
    simplification_trace: List[Dict[str, str]] = field(default_factory=list)  # rule, node, gain
    certificate_source: str = ""  # "lean_synchronized", "constructive_interval", "local_estimate"
    epsilon_certified: float = 0.0

    def summary(self) -> str:
        lines = [
            "POEMA COMPILATION REPORT",
            f"- FMA ops: {self.total_fma_ops}",
            f"- Certified epsilon: {self.total_epsilon:.3e}",
            f"- Simplifications: {self.simplifications_applied}",
            f"- Compensations: {self.compensations_injected}",
            f"- Sheaf injections: {self.sheaves_injected}",
            f"- Lie bracket depth: {self.lie_bracket_depth}",
            f"- Compile time (ms): {self.compilation_time_ms:.2f}",
            f"- Domain guard checks: {self.domain_guard_checks}",
            f"- Domain guard violations: {self.domain_guard_violations}",
            f"- Domain guard max overshoot: {self.domain_guard_max_overshoot:.3e}",
        ]
        if self.certificate_source:
            lines.append(f"- Certificate source: {self.certificate_source}")
        if self.phase_times:
            lines.append("- Phase times:")
            for phase, time_ms in self.phase_times.items():
                lines.append(f"    {phase}: {time_ms:.2f}ms")
        if self.node_profiles:
            lines.append(f"- Node profiles: {len(self.node_profiles)}")
        if self.domain_guard_alerts:
            lines.append("- Domain guard alerts:")
            lines.extend([f"  * {a}" for a in self.domain_guard_alerts])
        if self.warnings:
            lines.append("- Warnings:")
            lines.extend([f"  * {w}" for w in self.warnings])
        return "\n".join(lines)


class GeometricTypeChecker:
    def __init__(self, target_precision: str = "fp64", auto_compensate: bool = True):
        self.target_precision = target_precision
        self.auto_compensate = auto_compensate
        self._eps = {
            "fp64": 2.22e-16,
            "fp32": 1.19e-7,
            "fp16": 9.77e-4,
            "bf16": 3.91e-3,
            "fp8": 1.95e-1,
        }

    def check(self, node: ASTNode, report: CompilationReport) -> ASTNode:
        self._check_dimensions(node, report)
        if self.auto_compensate:
            node = self._inject_compensation(node, report)
        return node

    def _check_dimensions(self, node: ASTNode, report: CompilationReport) -> None:
        if isinstance(node, ComposeNode):
            o = node.outer.geometric_type
            i = node.inner.geometric_type
            if not o.is_composable_with(i):
                raise TopologicalObstructionError(
                    f"incompatible composition: outer={o}, inner={i}", node=node
                )
            if isinstance(node.outer, (ScaleNode, AffineNode)) and isinstance(node.inner, (ShiftNode, AffineNode)):
                node.metadata.lie_bracket_depth = 1
                node.metadata.parallelizable = False
                report.lie_bracket_depth = max(report.lie_bracket_depth, 1)

            self._check_lie_bracket_stub(node, report)
            self._check_cohomology_stub(node, report)

        for child in node.children:
            if isinstance(child, ASTNode):
                self._check_dimensions(child, report)

    def _check_lie_bracket_stub(self, node: ComposeNode, report: CompilationReport) -> None:
        """Best-effort Lie-bracket-aware check for affine chains.

        This is a conservative stub that references acf_functor.lie_analysis
        when available, but never blocks compilation.
        """
        if not isinstance(node.outer, (ScaleNode, ShiftNode, AffineNode)):
            return
        if not isinstance(node.inner, (ScaleNode, ShiftNode, AffineNode)):
            return

        try:
            # Keep this import local and optional so compiler remains lightweight.
            from acf_functor.lie_analysis import LieBracketAnalyzer  # type: ignore

            _ = LieBracketAnalyzer()
        except Exception:
            report.warnings.append("lie-analysis module unavailable; using geometric fallback")

        outer_sym = node.outer.geometric_type.symmetry_group
        inner_sym = node.inner.geometric_type.symmetry_group
        if outer_sym is not None and inner_sym is not None and outer_sym != inner_sym:
            msg = (
                f"lie-bracket stub: symmetry mismatch outer={outer_sym}, "
                f"inner={inner_sym}; treating as potentially non-commutative"
            )
            node.metadata.warnings.append(msg)
            report.warnings.append(msg)
            node.metadata.lie_bracket_depth = max(node.metadata.lie_bracket_depth, 1)
            report.lie_bracket_depth = max(report.lie_bracket_depth, 1)

    def _check_cohomology_stub(self, node: ComposeNode, report: CompilationReport) -> None:
        """Conservative cohomology obstruction check via continuity metadata."""
        o = node.outer.geometric_type
        i = node.inner.geometric_type
        if o.continuity >= 0 and i.continuity >= 0 and abs(o.continuity - i.continuity) > 1:
            msg = (
                f"cohomology stub: continuity mismatch outer=C^{o.continuity}, "
                f"inner=C^{i.continuity}; possible H1 obstruction"
            )
            node.metadata.warnings.append(msg)
            report.warnings.append(msg)

    def _inject_compensation(self, node: ASTNode, report: CompilationReport) -> ASTNode:
        eps = self._eps.get(self.target_precision, 2.22e-16)
        if isinstance(node, AffineNode):
            s = float(torch.abs(node.scale_factor).item())
            b = float(torch.abs(node.shift_value).item())
            estimated = (s + b) * eps
            if estimated > 10.0 * eps:
                s64 = node.scale_factor.to(torch.float64)
                b64 = node.shift_value.to(torch.float64)

                if self.target_precision == "fp32":
                    s_hw = s64.float().double()
                    b_hw = b64.float().double()
                elif self.target_precision == "fp16":
                    s_hw = s64.half().double()
                    b_hw = b64.half().double()
                else:
                    s_hw = s64
                    b_hw = b64

                ds = s64 - s_hw
                db = b64 - b_hw
                if float(torch.abs(ds).item()) > 0.0 or float(torch.abs(db).item()) > 0.0:
                    node = AffineNode(
                        scale_factor=s64,
                        shift_value=b64 + db,
                        child=node.children[0] if node.children else None,
                        geometric_type=node.geometric_type,
                    )
                    node.metadata.compensation_nodes_injected = 1
                    report.compensations_injected += 1
                    warnings.warn("precision compensation injected", PrecisionDegradationWarning)

        new_children = []
        for child in node.children:
            if isinstance(child, ASTNode):
                new_children.append(self._inject_compensation(child, report))
            else:
                new_children.append(child)
        node.children = new_children
        return node


class FMALinearizer:
    def linearize(self, node: ASTNode) -> List[FMAInstruction]:
        out: List[FMAInstruction] = []
        self._emit(node, out)
        return out

    def _emit(self, node: ASTNode, out: List[FMAInstruction]) -> None:
        if isinstance(node, ConstantNode):
            out.append(FMAInstruction(weight=torch.tensor(0.0, dtype=node.value.dtype), bias=node.value, source_node=node))
            return

        if isinstance(node, IdentityNode):
            return

        if isinstance(node, AffineNode):
            for child in node.children:
                if isinstance(child, ASTNode):
                    self._emit(child, out)
            out.append(FMAInstruction(weight=node.scale_factor, bias=node.shift_value, source_node=node))
            return

        if isinstance(node, ScaleNode):
            for child in node.children:
                if isinstance(child, ASTNode):
                    self._emit(child, out)
            out.append(FMAInstruction(weight=node.factor, bias=torch.tensor(0.0, dtype=node.factor.dtype), source_node=node))
            return

        if isinstance(node, ShiftNode):
            for child in node.children:
                if isinstance(child, ASTNode):
                    self._emit(child, out)
            out.append(FMAInstruction(weight=torch.tensor(1.0, dtype=node.value.dtype), bias=node.value, source_node=node))
            return

        if isinstance(node, PolynomialNode):
            coeffs = node.coefficients
            out.append(FMAInstruction(weight=torch.tensor(0.0, dtype=coeffs.dtype), bias=coeffs[-1], source_node=node))
            for i in range(coeffs.numel() - 2, -1, -1):
                out.append(FMAInstruction(weight=torch.tensor(1.0, dtype=coeffs.dtype), bias=coeffs[i], source_node=node))
            return

        if isinstance(node, TranscendentalNode):
            self._emit(node.polynomial, out)
            return

        if isinstance(node, ComposeNode):
            self._emit(node.inner, out)
            self._emit(node.outer, out)
            return

        if isinstance(node, (_CompoundAddNode, _CompoundMulNode)):
            self._emit(node.left, out)
            self._emit(node.right, out)
            out.append(FMAInstruction(weight=torch.tensor(1.0), bias=torch.tensor(0.0), source_node=node))
            return

        if isinstance(node, StratifiedNode):
            for branch in node.branches:
                self._emit(branch.body_ast, out)
            return

        if isinstance(node, InputNode):
            return

        warnings.warn(f"unknown node type during linearization: {type(node)}")


class PytorchBackend:
    @staticmethod
    def compile(
        source_ast: ASTNode,
        fma_sequence: Optional[List[FMAInstruction]] = None,
        auto_domain_repair: bool = True,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        # Keep specialized evaluators for numerically stable high-order paths.
        if isinstance(source_ast, PolynomialNode):
            coeffs = source_ast.coefficients.clone()

            def fn_poly(x: torch.Tensor, _coeffs: torch.Tensor = coeffs) -> torch.Tensor:
                from acf_functor.core import HornerReducer

                return HornerReducer.execute_horner(_coeffs.to(device=x.device, dtype=x.dtype), x)

            return fn_poly

        if isinstance(source_ast, TranscendentalNode):
            coeffs_mono = source_ast.polynomial.coefficients.clone()
            coeffs_cheb = (
                source_ast.chebyshev_coefficients.clone()
                if source_ast.chebyshev_coefficients is not None
                else None
            )
            domain = source_ast.original_domain
            mode = source_ast.evaluation_mode
            name = source_ast.name.lower().strip()

            def fn_trans(
                x: torch.Tensor,
                _mono: torch.Tensor = coeffs_mono,
                _cheb: Optional[torch.Tensor] = coeffs_cheb,
                _domain: Tuple[float, float] = domain,
                _mode: str = mode,
                _name: str = name,
                _repair: bool = auto_domain_repair,
            ) -> torch.Tensor:
                from acf_functor.core import ChebyshevReducer, HornerReducer

                if _mode == "clenshaw" and _cheb is not None:
                    y_cheb = ChebyshevReducer.evaluate_chebyshev_series(
                        _cheb.to(device=x.device, dtype=x.dtype),
                        x,
                        _domain,
                    )
                    if not _repair:
                        return y_cheb

                    a, b = _domain
                    in_domain = (x >= a) & (x <= b)
                    if bool(in_domain.all().item()):
                        return y_cheb

                    # Pure-FMA domain repair: no PyTorch fallback
                    from poema.pure_fma_repair import PureFMAAutoDomainRepair

                    # Ensure certificates are initialized
                    if not PureFMAAutoDomainRepair._pure_certificates:
                        PureFMAAutoDomainRepair()
                    cheb_list = _cheb.tolist() if _cheb is not None else []

                    # Evaluate out-of-domain points via Pure-FMA
                    y_result = y_cheb.clone()
                    ood_mask = ~in_domain
                    x_ood = x[ood_mask]
                    if x_ood.numel() > 0:
                        values, _ = PureFMAAutoDomainRepair.batch_pure_evaluation(
                            _name, x_ood.tolist(), cheb_list, _domain
                        )
                        y_result[ood_mask] = torch.tensor(
                            values, dtype=x.dtype, device=x.device
                        )

                    return y_result

                return HornerReducer.execute_horner(
                    _mono.to(device=x.device, dtype=x.dtype),
                    x,
                )

            return fn_trans

        # Compose is handled semantically, preserving functional boundaries.
        if isinstance(source_ast, ComposeNode):
            inner_fn = PytorchBackend.compile(source_ast.inner, auto_domain_repair=auto_domain_repair)
            outer_fn = PytorchBackend.compile(source_ast.outer, auto_domain_repair=auto_domain_repair)

            def fn_compose(x: torch.Tensor) -> torch.Tensor:
                return outer_fn(inner_fn(x))

            return fn_compose

        if isinstance(source_ast, InputNode):
            return lambda x: x

        if isinstance(source_ast, IdentityNode):
            return lambda x: x

        if isinstance(source_ast, ConstantNode):
            value = source_ast.value.clone()
            return lambda x, _v=value: _v.to(device=x.device, dtype=x.dtype).expand_as(x)

        if isinstance(source_ast, ScaleNode):
            scale = source_ast.factor.clone()
            child = source_ast.children[0] if source_ast.children else None
            child_fn = PytorchBackend.compile(child, auto_domain_repair=auto_domain_repair) if isinstance(child, ASTNode) else (lambda x: x)
            return lambda x, _s=scale, _fn=child_fn: _s.to(device=x.device, dtype=x.dtype) * _fn(x)

        if isinstance(source_ast, ShiftNode):
            shift = source_ast.value.clone()
            child = source_ast.children[0] if source_ast.children else None
            child_fn = PytorchBackend.compile(child, auto_domain_repair=auto_domain_repair) if isinstance(child, ASTNode) else (lambda x: x)
            return lambda x, _b=shift, _fn=child_fn: _fn(x) + _b.to(device=x.device, dtype=x.dtype)

        if isinstance(source_ast, AffineNode):
            scale = source_ast.scale_factor.clone()
            shift = source_ast.shift_value.clone()
            child = source_ast.children[0] if source_ast.children else None
            child_fn = PytorchBackend.compile(child, auto_domain_repair=auto_domain_repair) if isinstance(child, ASTNode) else (lambda x: x)

            def fn_affine(x: torch.Tensor, _a=scale, _b=shift, _fn=child_fn) -> torch.Tensor:
                y = _fn(x)
                a = _a.to(device=x.device, dtype=x.dtype)
                b = _b.to(device=x.device, dtype=x.dtype)
                return a * y + b

            return fn_affine

        if isinstance(source_ast, _CompoundAddNode):
            left_fn = PytorchBackend.compile(source_ast.left, auto_domain_repair=auto_domain_repair)
            right_fn = PytorchBackend.compile(source_ast.right, auto_domain_repair=auto_domain_repair)
            return lambda x, _l=left_fn, _r=right_fn: _l(x) + _r(x)

        if isinstance(source_ast, _CompoundMulNode):
            left_fn = PytorchBackend.compile(source_ast.left, auto_domain_repair=auto_domain_repair)
            right_fn = PytorchBackend.compile(source_ast.right, auto_domain_repair=auto_domain_repair)
            return lambda x, _l=left_fn, _r=right_fn: _l(x) * _r(x)

        if isinstance(source_ast, StratifiedNode):
            compiled_branches = [
                (branch.domain, PytorchBackend.compile(branch.body_ast, auto_domain_repair=auto_domain_repair))
                for branch in source_ast.branches
            ]

            def fn_stratified(x: torch.Tensor, _branches=compiled_branches) -> torch.Tensor:
                y = torch.zeros_like(x)
                covered = torch.zeros_like(x, dtype=torch.bool)
                for (a, b), fn_branch in _branches:
                    mask = (x >= a) & (x < b)
                    active = mask & (~covered)
                    if active.any():
                        values = fn_branch(x)
                        y = torch.where(active, values, y)
                        covered = covered | active
                if (~covered).any() and _branches:
                    values = _branches[-1][1](x)
                    y = torch.where(~covered, values, y)
                return y

            return fn_stratified

        if fma_sequence is None:
            fma_sequence = FMALinearizer().linearize(source_ast)

        ws = [instr.weight.clone() for instr in fma_sequence]
        bs = [instr.bias.clone() for instr in fma_sequence]

        def fn_generic(x: torch.Tensor, _ws: List[torch.Tensor] = ws, _bs: List[torch.Tensor] = bs) -> torch.Tensor:
            y = x
            for w, b in zip(_ws, _bs):
                w_dev = w.to(device=x.device, dtype=x.dtype)
                b_dev = b.to(device=x.device, dtype=x.dtype)
                if w_dev.dim() >= 2:
                    y = torch.matmul(y, w_dev) + b_dev
                else:
                    y = b_dev.expand_as(y) + w_dev.expand_as(y) * y
            return y

        return fn_generic


class TritonBackend:
    """Programmatic Triton kernel construction for scalar affine chains, vectorial GEMM, and polynomial evaluation."""

    @staticmethod
    def compile_kernel(fma_sequence: List[FMAInstruction], kernel_name: str = "poema_kernel") -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        try:
            import triton
            import triton.language as tl
        except ImportError:
            warnings.warn("triton not available, backend fallback required")
            return None

        # Classify instructions: scalar vs tensor
        has_tensors = any(
            instr.weight.dim() >= 1 or instr.bias.dim() >= 1
            for instr in fma_sequence
        )

        if has_tensors:
            return TritonBackend._compile_vectorial(fma_sequence, kernel_name)

        # Scalar path
        return TritonBackend._compile_scalar(fma_sequence, kernel_name)

    @staticmethod
    def _compile_scalar(fma_sequence: List[FMAInstruction], kernel_name: str) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        """Compile scalar affine chain to Triton kernel."""
        import triton
        import triton.language as tl

        scalar_ops: List[Tuple[float, float]] = []
        for instr in fma_sequence:
            if instr.weight.dim() != 0 or instr.bias.dim() != 0:
                return None
            scalar_ops.append((float(instr.weight.item()), float(instr.bias.item())))

        if len(scalar_ops) == 0:
            return lambda x: x

        # Detect Horner pattern
        is_horner = len(scalar_ops) >= 2
        if is_horner:
            w0, b0 = scalar_ops[0]
            if abs(w0) > 1e-15:
                is_horner = False
            for w, b in scalar_ops[1:]:
                if abs(w - 1.0) > 1e-15:
                    is_horner = False
                    break

        if is_horner:
            # Polynomial evaluation via Horner's method in Triton.
            # Coefficients from highest to lowest degree: c_n, c_{n-1}, ..., c_0
            # Horner: y = c_n; for i in n-1..0: y = y*x + c_i
            coeffs = [b0] + [b for _, b in scalar_ops[1:]]
            n_coeffs = len(coeffs)

            # Build kernel source with unrolled Horner loop
            # Each coefficient is a kernel parameter
            coeff_params = ", ".join(f"c{i}" for i in range(n_coeffs))
            
            # Generate unrolled Horner steps
            horner_lines = ["y = c0"]
            for i in range(1, n_coeffs):
                horner_lines.append(f"y = tl.math.fma(y, x, c{i})")
            horner_body = "\n    ".join(horner_lines)

            kernel_code = (
                f'@triton.jit\n'
                f'def _horner_kernel(input_ptr, output_ptr, n_elements, {coeff_params}, BLOCK: tl.constexpr):\n'
                f'    pid = tl.program_id(0)\n'
                f'    offs = pid * BLOCK + tl.arange(0, BLOCK)\n'
                f'    mask = offs < n_elements\n'
                f'    x = tl.load(input_ptr + offs, mask=mask)\n'
                f'    {horner_body}\n'
                f'    tl.store(output_ptr + offs, y, mask=mask)\n'
            )
            
            # Write kernel to a temp file so Triton can find source
            import tempfile
            import os
            tmpdir = tempfile.gettempdir()
            kernel_file = os.path.join(tmpdir, f"poema_horner_{n_coeffs}.py")
            with open(kernel_file, 'w') as f:
                f.write("import triton\nimport triton.language as tl\n\n")
                f.write(kernel_code)
            
            # Import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"poema_horner_{n_coeffs}", kernel_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _horner_kernel = mod._horner_kernel

            def wrapper(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                n = x.numel()
                grid = lambda meta: ((n + meta["BLOCK"] - 1) // meta["BLOCK"],)
                _horner_kernel[grid](x, out, n, *coeffs, BLOCK=1024)
                return out

            wrapper.__name__ = kernel_name
            return wrapper

        # Pure affine chain: collapse y = a_i * y + b_i into one affine y = A*y + B
        A = 1.0
        B = 0.0
        for a_i, b_i in scalar_ops:
            A = a_i * A
            B = a_i * B + b_i

        @triton.jit
        def _kernel(input_ptr, output_ptr, n_elements, a, b, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n_elements
            x = tl.load(input_ptr + offs, mask=mask)
            y = tl.math.fma(x, a, b)
            tl.store(output_ptr + offs, y, mask=mask)

        def wrapper(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            grid = lambda meta: ((n + meta["BLOCK"] - 1) // meta["BLOCK"],)
            _kernel[grid](x, out, n, a=A, b=B, BLOCK=1024)
            return out

        wrapper.__name__ = kernel_name
        return wrapper

    @staticmethod
    def _compile_vectorial(fma_sequence: List[FMAInstruction], kernel_name: str) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        """Compile vectorial affine chain to Triton GEMM kernel.
        
        Collapses a chain of vectorial affine operations:
            y = W_n @ (... @ (W_2 @ (W_1 @ x + b_1) + b_2) ... ) + b_n
        into a single GEMM:
            y = W_total @ x + b_total
        where W_total = W_n @ ... @ W_1 and b_total is computed by propagation.
        """
        import triton
        import triton.language as tl

        # Collect all weight/bias tensors
        weights: List[torch.Tensor] = []
        biases: List[torch.Tensor] = []
        for instr in fma_sequence:
            w = instr.weight
            b = instr.bias
            # Promote scalars to 2D for uniform handling
            if w.dim() == 0:
                w = w.unsqueeze(0).unsqueeze(0)  # scalar -> 1x1
            elif w.dim() == 1:
                w = torch.diag(w)  # vector -> diagonal matrix
            if b.dim() == 0:
                b = b.unsqueeze(0)  # scalar -> 1D
            weights.append(w)
            biases.append(b)

        if len(weights) == 0:
            return lambda x: x

        # Collapse the chain: W_total = W_n @ ... @ W_1
        # b_total = W_n @ ... @ W_2 @ b_1 + W_n @ ... @ W_3 @ b_2 + ... + b_n
        W_total = weights[0].clone()
        b_total = biases[0].clone()

        for i in range(1, len(weights)):
            W_i = weights[i]
            b_i = biases[i]
            # New output: W_i @ (W_total @ x + b_total) + b_i
            # = (W_i @ W_total) @ x + (W_i @ b_total + b_i)
            W_total = W_i @ W_total
            b_total = W_i @ b_total + b_i

        input_dim = W_total.shape[1]
        output_dim = W_total.shape[0]

        # For 1x1 case, fall back to scalar kernel
        if input_dim == 1 and output_dim == 1:
            scalar_w = W_total.flatten()[0]
            scalar_b = b_total.flatten()[0]
            scalar_fmas = [FMAInstruction(weight=scalar_w, bias=scalar_b)]
            return TritonBackend._compile_scalar(scalar_fmas, kernel_name)

        # Massive Matrix / Vectorial GEMM Operations via Triton natively
        import triton
        import triton.language as tl
        
        W_dev = W_total.contiguous().to(device='cuda', dtype=torch.float32)
        b_dev = b_total.contiguous().view(-1).to(device='cuda', dtype=torch.float32)
        
        @triton.jit
        def _gemm_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr, stride_cm: tl.constexpr, stride_cn: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            pid_m = pid % num_pid_m
            pid_n = pid // num_pid_m
            
            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
                accumulator += tl.dot(a, b, input_precision="ieee")
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk
                
            c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
            c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
            tl.store(c_ptrs, accumulator, mask=c_mask)

        def wrapper(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 1:
                x = x.unsqueeze(1)
            x_dev = x.contiguous().to(device='cuda', dtype=torch.float32)
            
            M, K = W_dev.shape
            K_x, N = x_dev.shape
            assert K == K_x, "Incompatible dimensions"
            
            y_dev = torch.empty((M, N), device='cuda', dtype=torch.float32)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
            _gemm_kernel[grid](W_dev, x_dev, y_dev, M, N, K, W_dev.stride(0), W_dev.stride(1), x_dev.stride(0), x_dev.stride(1), y_dev.stride(0), y_dev.stride(1), BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32)
            
            return y_dev + b_dev.unsqueeze(1)
            
        wrapper.__name__ = kernel_name
        return wrapper


class PoemCompiler:
    def __init__(
        self,
        target: str = "pytorch",
        precision: str = "fp64",
        auto_compensate: bool = True,
        auto_domain_repair: bool = True,
        enable_self_modulation: bool = False,
        verbose: bool = False,
    ):
        self.target = target
        self.precision = precision
        self.verbose = verbose
        self.enable_self_modulation = enable_self_modulation
        self.auto_domain_repair = auto_domain_repair

        self.type_checker = GeometricTypeChecker(target_precision=precision, auto_compensate=auto_compensate)
        self.linearizer = FMALinearizer()

        if target not in ("pytorch", "triton", "gideon"):
            raise ValueError(f"unknown target {target!r}; choose from 'pytorch', 'triton', 'gideon'")

        # Lazy Gideon engine (instanciado solo cuando target=="gideon" o se llama .gideon)
        self._gideon_engine = None

    @property
    def gideon(self):
        """Acceso al GideonEngine nativo. Se instancia la primera vez que se accede."""
        if self._gideon_engine is None:
            from .backends.gideon.engine import GideonEngine, GideonEngineConfig
            cfg = GideonEngineConfig(
                precision=self.precision,
                verbose=self.verbose,
                enable_theorem_seeds=False,
            )
            self._gideon_engine = GideonEngine(cfg)
        return self._gideon_engine

    def compile(self, ast: ASTNode, domain: Optional[Tuple[float, float]] = None) -> Tuple[Callable[[torch.Tensor], torch.Tensor], CompilationReport]:
        t0 = time.perf_counter()
        report = CompilationReport()

        # Phase 1: Simplification
        t_phase = time.perf_counter()
        simplified = ast.simplify()
        report.simplifications_applied = 1 if simplified is not ast else 0
        report.phase_times["simplify"] = (time.perf_counter() - t_phase) * 1000.0

        # Phase 2: Geometric type checking
        t_phase = time.perf_counter()
        checked = self.type_checker.check(simplified, report)
        report.phase_times["type_check"] = (time.perf_counter() - t_phase) * 1000.0

        # Phase 3: Self-modulation (if enabled)
        if self.enable_self_modulation and domain is not None:
            t_phase = time.perf_counter()
            checked = self._apply_self_modulation(checked, domain, report)
            report.phase_times["self_modulation"] = (time.perf_counter() - t_phase) * 1000.0

        # Phase 4: Domain guard
        t_phase = time.perf_counter()
        self._run_domain_guard(checked, domain, report)
        report.phase_times["domain_guard"] = (time.perf_counter() - t_phase) * 1000.0

        # Phase 5: FMA linearization
        t_phase = time.perf_counter()
        fmas = self.linearizer.linearize(checked)
        report.fma_sequence = fmas
        report.total_fma_ops = len(fmas)
        report.phase_times["linearization"] = (time.perf_counter() - t_phase) * 1000.0

        # Set epsilon (Full AST Analytical Propagation)
        from .error_propagation import compute_ast_error_bound
        val_domain = domain if domain else (-1.0, 1.0)
        bound = compute_ast_error_bound(checked, val_domain)
        report.total_epsilon = bound.epsilon
        report.certificate_source = bound.source
        report.epsilon_certified = bound.epsilon

        # Phase 6: Backend compilation
        t_phase = time.perf_counter()
        if self.target == "gideon":
            # ── Gideon native path ─────────────────────────────────────────
            val_domain_exec = domain if domain else (-1.0, 1.0)
            g_result = self.gideon.run_fma(
                fmas,
                x=None,   # Sólo compilamos, no ejecutamos todavía
                name="poema_gideon",
            ) if False else None  # Sólo compilar, extraer callable
            # Compilar para obtener callable (sin ejecutar)
            callable_fn, backend_used = self.gideon._compile_fma(
                fmas,
                self.gideon._dispatcher.decide(
                    self.gideon._ir.from_fma_sequence(
                        fmas, domain=val_domain_exec, precision=self.precision
                    )
                ).primary_backend,
                "numpy_cpu",
            )
            # Envolver en tensor torch si es necesario
            _raw_fn = callable_fn
            import numpy as _np
            def executable(x: torch.Tensor) -> torch.Tensor:  # noqa: E731
                x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else _np.asarray(x)
                y_np = _raw_fn(x_np)
                out = torch.tensor(y_np, dtype=x.dtype if isinstance(x, torch.Tensor) else torch.float64)
                if isinstance(x, torch.Tensor):
                    out = out.to(x.device)
                return out
            report.warnings.append(f"gideon:backend_used={backend_used}")
        elif self.target == "pytorch":
            executable = PytorchBackend.compile(
                checked,
                fmas,
                auto_domain_repair=self.auto_domain_repair,
            )
        else:
            executable = TritonBackend.compile_kernel(fmas)
            if executable is None:
                report.warnings.append("triton backend unavailable or unsupported for this AST, fallback to pytorch")
                executable = PytorchBackend.compile(
                    checked,
                    fmas,
                    auto_domain_repair=self.auto_domain_repair,
                )
        report.phase_times["backend"] = (time.perf_counter() - t_phase) * 1000.0

        # Generate node profiles
        report.node_profiles = self._generate_node_profiles(checked, fmas)

        report.compilation_time_ms = (time.perf_counter() - t0) * 1000.0
        return executable, report

    def _generate_node_profiles(self, node: ASTNode, fmas: List[FMAInstruction]) -> List[NodeProfile]:
        """Generate per-node profiles for observability."""
        profiles = []
        self._collect_node_profiles(node, fmas, profiles, node_id=[0])
        return profiles

    def _collect_node_profiles(self, node: ASTNode, fmas: List[FMAInstruction], profiles: List[NodeProfile], node_id: List[int]) -> int:
        """Recursively collect node profiles. Returns FMA count for this subtree."""
        current_id = node_id[0]
        node_id[0] += 1
        
        profile = NodeProfile(
            node_type=type(node).__name__,
            node_id=f"node_{current_id}",
        )
        
        if isinstance(node, PolynomialNode):
            profile.fma_contribution = len(node.coefficients)
            profile.epsilon_contribution = 0.0
            profile.simplification_applied = True
            profile.simplification_rule = "horner_evaluation"
        elif isinstance(node, TranscendentalNode):
            profile.fma_contribution = len(node.polynomial.coefficients) if hasattr(node.polynomial, 'coefficients') else 0
            profile.epsilon_contribution = node.certified_epsilon
            profile.domain_interval = node.original_domain
            profile.simplification_rule = f"chebyshev_degree_{len(node.polynomial.coefficients) if hasattr(node.polynomial, 'coefficients') else 0}"
        elif isinstance(node, AffineNode):
            profile.fma_contribution = 1
            profile.simplification_rule = "single_fma"
        elif isinstance(node, ScaleNode):
            profile.fma_contribution = 1
            profile.simplification_rule = "scale_only"
        elif isinstance(node, ShiftNode):
            profile.fma_contribution = 1
            profile.simplification_rule = "shift_only"
        elif isinstance(node, StratifiedNode):
            profile.fma_contribution = sum(
                self._collect_node_profiles(branch.body_ast, fmas, [], node_id)
                for branch in node.branches
            )
            profile.simplification_rule = f"stratified_{len(node.branches)}_branches"
        elif isinstance(node, ComposeNode):
            inner_fmas = self._collect_node_profiles(node.inner, fmas, profiles, node_id)
            outer_fmas = self._collect_node_profiles(node.outer, fmas, profiles, node_id)
            profile.fma_contribution = inner_fmas + outer_fmas
            profile.simplification_rule = "composition"
            profiles.append(profile)
            return profile.fma_contribution
        elif isinstance(node, (_CompoundAddNode, _CompoundMulNode)):
            left_fmas = self._collect_node_profiles(node.left, fmas, profiles, node_id)
            right_fmas = self._collect_node_profiles(node.right, fmas, profiles, node_id)
            profile.fma_contribution = left_fmas + right_fmas + 1
            profile.simplification_rule = "add" if isinstance(node, _CompoundAddNode) else "mul"
            profiles.append(profile)
            return profile.fma_contribution
        elif isinstance(node, (InputNode, IdentityNode, ConstantNode)):
            profile.fma_contribution = 0
            profile.simplification_rule = "leaf"
        
        profiles.append(profile)
        return profile.fma_contribution

    def _run_domain_guard(self, node: ASTNode, domain: Optional[Tuple[float, float]], report: CompilationReport) -> None:
        input_domain = domain if domain is not None else self._infer_input_domain(node)
        if input_domain is None:
            return
        try:
            self._estimate_interval(node, input_domain, report)
        except Exception as exc:
            report.warnings.append(f"domain guard skipped: {exc}")

    def _infer_input_domain(self, node: ASTNode) -> Optional[Tuple[float, float]]:
        if isinstance(node, TranscendentalNode):
            return node.original_domain
        if isinstance(node, ComposeNode):
            return self._infer_input_domain(node.inner) or self._infer_input_domain(node.outer)
        if isinstance(node, (_CompoundAddNode, _CompoundMulNode)):
            return self._infer_input_domain(node.left) or self._infer_input_domain(node.right)
        if isinstance(node, (AffineNode, ScaleNode, ShiftNode)) and node.children:
            child = node.children[0]
            if isinstance(child, ASTNode):
                return self._infer_input_domain(child)
        if isinstance(node, StratifiedNode):
            for branch in node.branches:
                inferred = self._infer_input_domain(branch.body_ast)
                if inferred is not None:
                    return inferred
        return None

    def _estimate_interval(
        self,
        node: ASTNode,
        input_domain: Tuple[float, float],
        report: CompilationReport,
    ) -> Optional[Tuple[float, float]]:
        if isinstance(node, (InputNode, IdentityNode)):
            return input_domain

        if isinstance(node, ConstantNode):
            v = float(node.value.item())
            return (v, v)

        if isinstance(node, PolynomialNode):
            return self._estimate_polynomial_interval(node.coefficients, input_domain)

        if isinstance(node, TranscendentalNode):
            return self._estimate_transcendental_interval(node, input_domain)

        if isinstance(node, ScaleNode):
            child_interval = self._estimate_first_child_interval(node, input_domain, report)
            if child_interval is None:
                return None
            factor = float(node.factor.item())
            lo, hi = child_interval
            a, b = factor * lo, factor * hi
            return (min(a, b), max(a, b))

        if isinstance(node, ShiftNode):
            child_interval = self._estimate_first_child_interval(node, input_domain, report)
            if child_interval is None:
                return None
            shift = float(node.value.item())
            return (child_interval[0] + shift, child_interval[1] + shift)

        if isinstance(node, AffineNode):
            child_interval = self._estimate_first_child_interval(node, input_domain, report)
            if child_interval is None:
                return None
            scale = float(node.scale_factor.item())
            shift = float(node.shift_value.item())
            lo, hi = child_interval
            a, b = scale * lo + shift, scale * hi + shift
            return (min(a, b), max(a, b))

        if isinstance(node, _CompoundAddNode):
            left_interval = self._estimate_interval(node.left, input_domain, report)
            right_interval = self._estimate_interval(node.right, input_domain, report)
            if left_interval is None or right_interval is None:
                return None
            return (left_interval[0] + right_interval[0], left_interval[1] + right_interval[1])

        if isinstance(node, _CompoundMulNode):
            left_interval = self._estimate_interval(node.left, input_domain, report)
            right_interval = self._estimate_interval(node.right, input_domain, report)
            if left_interval is None or right_interval is None:
                return None
            candidates = [
                left_interval[0] * right_interval[0],
                left_interval[0] * right_interval[1],
                left_interval[1] * right_interval[0],
                left_interval[1] * right_interval[1],
            ]
            return (min(candidates), max(candidates))

        if isinstance(node, ComposeNode):
            inner_interval = self._estimate_interval(node.inner, input_domain, report)
            if inner_interval is None:
                return None
            for trans in self._collect_transcendentals(node.outer):
                self._record_domain_guard(trans, inner_interval, report)
            return self._estimate_interval(node.outer, inner_interval, report)

        if isinstance(node, StratifiedNode):
            out_interval: Optional[Tuple[float, float]] = None
            for branch in node.branches:
                branch_out = self._estimate_interval(branch.body_ast, branch.domain, report)
                if branch_out is None:
                    continue
                if out_interval is None:
                    out_interval = branch_out
                else:
                    out_interval = (min(out_interval[0], branch_out[0]), max(out_interval[1], branch_out[1]))
            return out_interval

        return None

    def _estimate_first_child_interval(
        self,
        node: ASTNode,
        input_domain: Tuple[float, float],
        report: CompilationReport,
    ) -> Optional[Tuple[float, float]]:
        if not node.children:
            return input_domain
        child = node.children[0]
        if not isinstance(child, ASTNode):
            return input_domain
        return self._estimate_interval(child, input_domain, report)

    def _estimate_polynomial_interval(self, coeffs: torch.Tensor, domain: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = domain
        xs = torch.linspace(lo, hi, 1025, dtype=torch.float64)
        c = coeffs.to(dtype=torch.float64)
        y = torch.zeros_like(xs)
        for k in range(int(c.numel()) - 1, -1, -1):
            y = y * xs + c[k]
        return (float(torch.min(y).item()), float(torch.max(y).item()))

    def _estimate_transcendental_interval(self, node: TranscendentalNode, domain: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = domain
        xs = torch.linspace(lo, hi, 1025, dtype=torch.float64)
        if node.evaluation_mode == "clenshaw" and node.chebyshev_coefficients is not None:
            from acf_functor.core import ChebyshevReducer

            ys = ChebyshevReducer.evaluate_chebyshev_series(
                node.chebyshev_coefficients.to(dtype=torch.float64),
                xs,
                node.original_domain,
            )
        else:
            ys = torch.zeros_like(xs)
            c = node.polynomial.coefficients.to(dtype=torch.float64)
            for k in range(int(c.numel()) - 1, -1, -1):
                ys = ys * xs + c[k]
        return (float(torch.min(ys).item()), float(torch.max(ys).item()))

    def _collect_transcendentals(self, node: ASTNode) -> List[TranscendentalNode]:
        out: List[TranscendentalNode] = []
        if isinstance(node, TranscendentalNode):
            out.append(node)
        if isinstance(node, ComposeNode):
            out.extend(self._collect_transcendentals(node.outer))
            out.extend(self._collect_transcendentals(node.inner))
            return out
        if isinstance(node, _CompoundAddNode):
            out.extend(self._collect_transcendentals(node.left))
            out.extend(self._collect_transcendentals(node.right))
            return out
        if isinstance(node, _CompoundMulNode):
            out.extend(self._collect_transcendentals(node.left))
            out.extend(self._collect_transcendentals(node.right))
            return out
        for child in node.children:
            if isinstance(child, ASTNode):
                out.extend(self._collect_transcendentals(child))
        return out

    def _record_domain_guard(
        self,
        trans: TranscendentalNode,
        observed_domain: Tuple[float, float],
        report: CompilationReport,
    ) -> None:
        report.domain_guard_checks += 1
        a, b = trans.original_domain
        lo, hi = observed_domain
        lower_overshoot = max(0.0, a - lo)
        upper_overshoot = max(0.0, hi - b)
        overshoot = max(lower_overshoot, upper_overshoot)
        if overshoot <= 0.0:
            return
        report.domain_guard_violations += 1
        report.domain_guard_max_overshoot = max(report.domain_guard_max_overshoot, overshoot)
        msg = (
            f"domain guard: compose feeds {trans.name} outside certified domain "
            f"[{a:.6g}, {b:.6g}] with observed [{lo:.6g}, {hi:.6g}]"
        )
        report.domain_guard_alerts.append(msg)
        report.warnings.append(msg)

    def _apply_self_modulation(self, node: ASTNode, domain: Tuple[float, float], report: CompilationReport) -> ASTNode:
        if not isinstance(node, TranscendentalNode):
            return node

        hw_eps = self.type_checker._eps.get(self.precision, 2.22e-16)
        if node.certified_epsilon <= 100.0 * hw_eps:
            return node

        try:
            from acf_functor.core import ChebyshevReducer
            from acf_functor.self_modulation import AdaptiveReducer

            info = ChebyshevReducer.CANONICAL_FUNCTIONS.get(node.name)
            if info is None:
                return node

            reducer = AdaptiveReducer(base_degree=max(int(node.polynomial.coefficients.numel()), 8), target_epsilon=10.0 * hw_eps)
            improved = reducer.reduce(f=info["generator"], domain=domain, target_epsilon=10.0 * hw_eps)

            strata = getattr(improved, "strata", [])
            if strata:
                branches: List[StratifiedNode.Branch] = []
                for s in strata:
                    coeffs = s.reduction.metadata.get("monomial_coefficients", s.reduction.metadata.get("coefficients", [0.0]))
                    body = PolynomialNode(coefficients=coeffs)
                    selector = InputNode(name="selector", geometric_type=GeometricType(1, 1))
                    branches.append(StratifiedNode.Branch(selector_ast=selector, body_ast=body, domain=s.domain, tear_type=",".join([t.name for t in s.parent_tears]) if s.parent_tears else None))
                report.sheaves_injected = len(branches)
                return StratifiedNode(branches=branches, geometric_type=node.geometric_type)

            coeffs = improved.metadata.get("monomial_coefficients", improved.metadata.get("coefficients", None))
            cheb = improved.metadata.get("chebyshev_coefficients", None)
            mode = improved.metadata.get("evaluation_mode", "horner")
            if coeffs is not None or cheb is not None:
                poly_coeffs = coeffs if coeffs is not None else [0.0]
                return TranscendentalNode(
                    name=node.name,
                    polynomial=PolynomialNode(coefficients=poly_coeffs),
                    certified_epsilon=improved.epsilon_bound,
                    original_domain=domain,
                    geometric_type=node.geometric_type,
                    chebyshev_coefficients=cheb,
                    evaluation_mode=mode,
                )
        except Exception as exc:
            if self.verbose:
                print(f"[poema] self modulation skipped: {exc}")

        return node

    # ──────────────────────────────────────────────────────────────────────────
    # auto_evolve: ACF auto-evolution on a compiled callable
    # ──────────────────────────────────────────────────────────────────────────

    def auto_evolve(
        self,
        ast: ASTNode,
        domain: Tuple[float, float],
        config: Optional[Any] = None,
    ):
        """
        Run the ACF auto-evolution pipeline on an AST.

        First compiles the AST to get a callable f, then applies
        ACFAutoEvolver to find the best polynomial/GEMM representation
        of f over the given domain, guided by:
          1. Thermodynamic search (best degree/method via F(d, β))
          2. Fixed-point iteration (idempotence: Φ² = Φ)
          3. Bifunctorial cycle (adjunction: Φ* ⊣ Φ)
          4. Adaptive refinement (residual-guided)

        Parameters
        ----------
        ast : ASTNode
            The Poema AST to compile and evolve.
        domain : tuple (a, b)
            Input domain for the function.
        config : ACFAutoEvolverConfig or None
            Evolution configuration. Defaults to ACFAutoEvolverConfig().

        Returns
        -------
        AutoEvolutionResult
            Full trace of the auto-evolution, including best_reduction,
            final_epsilon, improvement_ratio, and all sub-results.

        Notes
        -----
        This method is separate from compile() to avoid entangling the
        standard compilation pipeline with the more expensive evolution
        pipeline. The result's best_reduction can be evaluated directly
        via acf_functor.auto_evolution._eval_result(result.best_reduction, x).

        Honest scope
        ------------
        Auto-evolution is deterministic. It does NOT discover new theorems,
        learn from data, or require a meta-optimizer. It finds the optimal
        polynomial representation within the current ACF configuration space.
        """
        from acf_functor.auto_evolution import ACFAutoEvolver, ACFAutoEvolverConfig
        import torch

        # Compile AST to get the underlying mathematical function
        executable, _ = self.compile(ast, domain=domain)

        def f_callable(x: torch.Tensor) -> torch.Tensor:
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float64)
            return executable(x)

        cfg = config or ACFAutoEvolverConfig()
        evolver = ACFAutoEvolver(config=cfg)
        return evolver.evolve(f_callable, domain)
        if not isinstance(node, TranscendentalNode):
            return node

        hw_eps = self.type_checker._eps.get(self.precision, 2.22e-16)
        if node.certified_epsilon <= 100.0 * hw_eps:
            return node

        try:
            from acf_functor.core import ChebyshevReducer
            from acf_functor.self_modulation import AdaptiveReducer

            info = ChebyshevReducer.CANONICAL_FUNCTIONS.get(node.name)
            if info is None:
                return node

            reducer = AdaptiveReducer(base_degree=max(int(node.polynomial.coefficients.numel()), 8), target_epsilon=10.0 * hw_eps)
            improved = reducer.reduce(f=info["generator"], domain=domain, target_epsilon=10.0 * hw_eps)

            strata = getattr(improved, "strata", [])
            if strata:
                branches: List[StratifiedNode.Branch] = []
                for s in strata:
                    coeffs = s.reduction.metadata.get("monomial_coefficients", s.reduction.metadata.get("coefficients", [0.0]))
                    body = PolynomialNode(coefficients=coeffs)
                    selector = InputNode(name="selector", geometric_type=GeometricType(1, 1))
                    branches.append(StratifiedNode.Branch(selector_ast=selector, body_ast=body, domain=s.domain, tear_type=",".join([t.name for t in s.parent_tears]) if s.parent_tears else None))
                report.sheaves_injected = len(branches)
                return StratifiedNode(branches=branches, geometric_type=node.geometric_type)

            coeffs = improved.metadata.get("monomial_coefficients", improved.metadata.get("coefficients", None))
            cheb = improved.metadata.get("chebyshev_coefficients", None)
            mode = improved.metadata.get("evaluation_mode", "horner")
            if coeffs is not None or cheb is not None:
                poly_coeffs = coeffs if coeffs is not None else [0.0]
                return TranscendentalNode(
                    name=node.name,
                    polynomial=PolynomialNode(coefficients=poly_coeffs),
                    certified_epsilon=improved.epsilon_bound,
                    original_domain=domain,
                    geometric_type=node.geometric_type,
                    chebyshev_coefficients=cheb,
                    evaluation_mode=mode,
                )
        except Exception as exc:
            if self.verbose:
                print(f"[poema] self modulation skipped: {exc}")

        return node
