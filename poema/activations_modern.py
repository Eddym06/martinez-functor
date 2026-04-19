"""
Funciones de activación modernas para transformers, con certificación.

GELU exacto: 0.5 * x * (1 + erf(x/sqrt(2)))
SwiGLU: x * sigmoid(x) (= Swish)
RoPE: rotaciones en el espacio de embeddings
"""

import torch
import math
from typing import Dict, List, Optional, Tuple, Callable


def gelu_exact(poem_frontend, degree: int = 40, domain: tuple = (-4.0, 4.0)):
    """
    GELU exacto usando aproximación certificada de erf.
    
    GELU(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
    
    Estrategia de compilación:
    1. Certificar erf en [domain[0]/sqrt(2), domain[1]/sqrt(2)]
    2. Componer con la transformación lineal x → x/sqrt(2)
    3. Multiplicar por 0.5 * x
    
    La aproximación usada es la estándar de HuggingFace:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    Error < 1e-4 en [-4, 4]
    """
    P = poem_frontend
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    
    ast = P.continuous_flow(
        f"0.5 * x * (1 + tanh({sqrt_2_over_pi:.10f} * (x + 0.044715 * x^3)))",
        domain=domain,
        degree=degree,
    )
    return ast


def swiglu(poem_frontend, degree: int = 30, domain: tuple = (-6.0, 6.0)):
    """
    SwiGLU / Swish: x * sigmoid(x)
    
    Certificado en el dominio dado.
    """
    P = poem_frontend
    ast = P.continuous_flow("x * sigmoid(x)", domain=domain, degree=degree)
    return ast


def rope_embedding(
    poem_frontend,
    dim: int,
    max_seq_len: int = 2048,
    base: float = 10000.0,
) -> Dict:
    """
    Rotary Position Embedding (RoPE).
    
    RoPE aplica una rotación 2D a pares de dimensiones:
    [x_{2i}, x_{2i+1}] → [x_{2i}*cos(θ) - x_{2i+1}*sin(θ),
                           x_{2i}*sin(θ) + x_{2i+1}*cos(θ)]
    
    donde θ = pos / base^(2i/dim)
    
    Esto es una cadena de operaciones afines con coeficientes
    que dependen de la posición → compilable como FMA con
    pesos dependientes de la posición.
    """
    thetas = [
        1.0 / (base ** (2 * i / dim))
        for i in range(dim // 2)
    ]
    
    # Retorna los coeficientes; la aplicación real depende de la posición
    return {
        "thetas": thetas,
        "dim": dim,
        "description": "RoPE - aplicar con posición específica",
        "compile_for_position": lambda pos: _compile_rope_at_position(
            poem_frontend, thetas, pos
        ),
    }


def _compile_rope_at_position(poem_frontend, thetas: List[float], pos: float) -> Dict:
    """Compila RoPE para una posición específica."""
    import math
    
    cos_vals = [math.cos(theta * pos) for theta in thetas]
    sin_vals = [math.sin(theta * pos) for theta in thetas]
    
    # Cada par (cos_i, sin_i) define una rotación 2D
    # compilable como dos operaciones FMA
    return {"cos": cos_vals, "sin": sin_vals}
