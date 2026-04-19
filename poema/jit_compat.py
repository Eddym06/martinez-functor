"""
PyTorch JIT compatibility for Poema functions.

Wraps compiled Poema functions as torch.nn.Module compatible with torch.jit.script.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .ast_nodes import ASTNode
from .compiler import CompilationReport, PoemCompiler


class PoemJITWrapper(torch.nn.Module):
    """
    Wrapper que hace funciones Poema compatibles con torch.jit.script.
    
    Permite usar funciones Poema en modelos PyTorch existentes
    que serán compilados con TorchScript.
    
    Example:
        P = Poem(dtype=torch.float64)
        ast = P.continuous_flow("sin(x) + x^2")
        
        module = PoemJITWrapper(ast, domain=(-2, 2))
        scripted = module.to_torchscript()
        
        # Integrar en modelo existente
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.activation = PoemJITWrapper(ast)
            
            def forward(self, x):
                return self.activation(x)
    """
    
    epsilon_certified: float
    certificate_source: str
    total_fma_ops: int
    
    def __init__(
        self,
        ast: ASTNode,
        domain: Optional[Tuple[float, float]] = None,
        precision: str = "fp64",
        auto_domain_repair: bool = True,
    ):
        super().__init__()
        
        # Compilar la función Poema
        compiler = PoemCompiler(
            target='pytorch',
            precision=precision,
            auto_domain_repair=auto_domain_repair,
        )
        self._fn, self._report = compiler.compile(ast, domain=domain)
        
        # Guardar metadatos como atributos del módulo
        self.epsilon_certified = float(self._report.total_epsilon)
        self.certificate_source = self._report.certificate_source or "none"
        self.total_fma_ops = int(self._report.total_fma_ops)
        
        # Registrar dominio como buffer para serialización
        if domain is not None:
            self.register_buffer('domain_min', torch.tensor(domain[0], dtype=torch.float64))
            self.register_buffer('domain_max', torch.tensor(domain[1], dtype=torch.float64))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x)
    
    def to_torchscript(self) -> torch.jit.ScriptModule:
        """Compila el módulo a TorchScript."""
        return torch.jit.script(self)
    
    def extra_repr(self) -> str:
        return (
            f"epsilon_certified={self.epsilon_certified:.3e}, "
            f"certificate_source='{self.certificate_source}', "
            f"total_fma_ops={self.total_fma_ops}"
        )


class PoemActivation(torch.nn.Module):
    """
    Función de activación Poema compatible con PyTorch.
    
    Ejemplo:
        # GELU aproximado via Poema
        gelu = PoemActivation("0.5*x*(1 + tanh(0.797885*x + 0.035677*x^3))")
        
        # Swish
        swish = PoemActivation("x * sigmoid(x)")
    """
    
    def __init__(
        self,
        expression: str,
        domain: Tuple[float, float] = (-5.0, 5.0),
        degree: int = 24,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        
        from .frontend import Poem
        
        P = Poem(dtype=dtype)
        ast = P.continuous_flow(expression, domain=domain, degree=degree)
        
        self._wrapper = PoemJITWrapper(ast, domain=domain)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._wrapper(x)
