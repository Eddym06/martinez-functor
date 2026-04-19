"""
BackendProtocol — interface that every Poema backend must satisfy.
BackendCapabilities — static description of what a backend can do.
BackendResult — unified result container.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple
import abc


@dataclass
class BackendCapabilities:
    name: str
    supports_gpu: bool = False
    supports_cpu: bool = True
    supports_batched: bool = True
    supports_gradient: bool = False
    supports_verilog: bool = False
    supports_cpp_emit: bool = False
    hardware_vendor: str = "generic"   # nvidia | amd | intel | fpga | generic
    max_fma_depth: int = 10_000
    precision_formats: List[str] = field(default_factory=lambda: ["fp64", "fp32"])
    notes: str = ""


@dataclass
class BackendResult:
    """Unified result from any backend compilation."""
    callable_fn: Optional[Callable] = None     # Python-callable evaluation function
    emitted_code: Optional[str] = None         # Source code (C++, Verilog, etc.)
    emitted_path: Optional[str] = None         # Path to emitted file
    fma_count: int = 0
    epsilon_bound: float = 0.0
    backend_name: str = ""
    extra: dict = field(default_factory=dict)


class BackendProtocol(abc.ABC):
    """Abstract base that all Poema backends must implement."""

    @property
    @abc.abstractmethod
    def capabilities(self) -> BackendCapabilities:
        ...

    @abc.abstractmethod
    def compile(
        self,
        fma_sequence: List[Any],           # List[FMAInstruction]
        source_ast: Any,                   # ASTNode
        domain: Tuple[float, float] = (-1.0, 1.0),
        precision: str = "fp64",
        **kwargs,
    ) -> BackendResult:
        """
        Compile an FMA sequence (and optionally the AST) to a backend-specific artifact.
        Must return a BackendResult with at minimum callable_fn or emitted_code set.
        """
        ...

    @abc.abstractmethod
    def verify_available(self) -> bool:
        """Return True if the required runtime/toolchain is available."""
        ...

    def describe(self) -> str:
        c = self.capabilities
        lines = [
            f"Backend: {c.name}",
            f"  Vendor: {c.hardware_vendor}",
            f"  GPU support: {c.supports_gpu}",
            f"  CPU support: {c.supports_cpu}",
            f"  Precision: {', '.join(c.precision_formats)}",
            f"  Verilog emit: {c.supports_verilog}",
            f"  C++ emit:     {c.supports_cpp_emit}",
            f"  Max FMA depth: {c.max_fma_depth}",
        ]
        if c.notes:
            lines.append(f"  Notes: {c.notes}")
        return "\n".join(lines)
