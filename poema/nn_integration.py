"""
Integración de Poema con PyTorch nn.Module.

Permite reemplazar funciones de activación estándar con
versiones compiladas por Poema que tienen cotas de error certificadas.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict


class PoemActivationLayer(nn.Module):
    """
    Función de activación compilada por Poema, usable como nn.Module.
    
    Ejemplo:
        # Reemplazar nn.GELU() con versión certificada
        model.activation = PoemActivationLayer.gelu(domain=(-4.0, 4.0))
        
        # Usar en un modelo personalizado
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = PoemActivationLayer.swish(domain=(-6.0, 6.0))
    """
    
    def __init__(
        self,
        expression: str,
        domain: Tuple[float, float],
        degree: int = 30,
        precision: str = "fp64",
        auto_domain_repair: bool = True,
        dtype=torch.float64,
    ):
        super().__init__()
        
        from poema import Poem, PoemCompiler
        
        self.expression = expression
        self.domain = domain
        self.precision = precision
        
        P = Poem(dtype=dtype)
        ast = P.continuous_flow(expression, domain=domain, degree=degree)
        
        compiler = PoemCompiler(
            target="pytorch",
            precision=precision,
            auto_domain_repair=auto_domain_repair,
        )
        
        self._fn, self._report = compiler.compile(ast, domain=domain)
        self.epsilon_certified = self._report.epsilon_certified
        self.certificate_source = self._report.certificate_source
        self.total_fma_ops = self._report.total_fma_ops
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x)
    
    def extra_repr(self) -> str:
        return (f"expr='{self.expression}', domain={self.domain}, "
                f"ε={self.epsilon_certified:.2e}, "
                f"cert='{self.certificate_source}', "
                f"fma_ops={self.total_fma_ops}")
    
    @classmethod
    def gelu(cls, domain=(-4.0, 4.0), degree=40, **kwargs):
        """GELU aproximado con certificado de error."""
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        expr = f"0.5 * x * (1 + tanh({sqrt_2_over_pi:.10f} * (x + 0.044715 * x^3)))"
        return cls(expr, domain=domain, degree=degree, **kwargs)
    
    @classmethod
    def swish(cls, domain=(-6.0, 6.0), degree=30, **kwargs):
        """Swish/SiLU: x * sigmoid(x)."""
        return cls("x * sigmoid(x)", domain=domain, degree=degree, **kwargs)
    
    @classmethod
    def tanh_act(cls, domain=(-4.0, 4.0), degree=50, **kwargs):
        """tanh con dominio extendido certificado."""
        return cls("tanh(x)", domain=domain, degree=degree, **kwargs)
    
    @classmethod
    def relu(cls, domain=(-4.0, 4.0), **kwargs):
        """ReLU exacto via piecewise."""
        return cls("piecewise(x >= 0, x, 0)", domain=domain, degree=1, **kwargs)
    
    @classmethod
    def mish(cls, domain=(-4.0, 4.0), degree=30, **kwargs):
        """Mish: x * tanh(log(1 + exp(x)))."""
        return cls("x * tanh(log(1 + exp(x)))", domain=domain, degree=degree, **kwargs)
    
    @classmethod
    def from_expression(cls, expr: str, domain: Tuple[float, float], **kwargs):
        """Crear desde expresión arbitraria."""
        return cls(expr, domain=domain, **kwargs)


def replace_activations_in_model(
    model: nn.Module,
    activation_map: dict,
    verbose: bool = True,
) -> nn.Module:
    """
    Reemplaza funciones de activación en un modelo existente.
    
    Args:
        model: modelo PyTorch
        activation_map: diccionario {tipo_original: PoemActivationLayer}
        verbose: si imprimir qué se reemplazó
    
    Ejemplo:
        model = replace_activations_in_model(
            model,
            {nn.GELU: PoemActivationLayer.gelu(),
             nn.SiLU: PoemActivationLayer.swish()}
        )
    """
    replacements = 0
    
    def _replace_recursive(module, name, parent):
        nonlocal replacements
        for child_name, child in list(module.named_children()):
            child_type = type(child)
            if child_type in activation_map:
                replacement = activation_map[child_type]
                setattr(module, child_name, replacement)
                replacements += 1
                if verbose:
                    print(f"  Replaced {child_name} ({child_type.__name__}) "
                          f"→ PoemActivationLayer(ε={replacement.epsilon_certified:.2e})")
            else:
                _replace_recursive(child, child_name, module)
    
    if verbose:
        print(f"Replacing activations in {type(model).__name__}...")
    
    _replace_recursive(model, "root", None)
    
    if verbose:
        print(f"Total replacements: {replacements}")
    
    return model
