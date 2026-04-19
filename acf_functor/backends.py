"""
Affine Collapse Functor (ACF) - Framework integration backends.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple
import warnings

import torch
import torch.fx

from .core import ChebyshevReducer, HornerReducer, ACFFunctor


_TORCH_OP_MAP: Dict[Callable, Tuple[str, Tuple[float, float]]] = {
    torch.sin: ("sin", (-3.141592653589793, 3.141592653589793)),
    torch.cos: ("cos", (-3.141592653589793, 3.141592653589793)),
    torch.exp: ("exp", (-1.0, 1.0)),
    torch.tanh: ("tanh", (-3.0, 3.0)),
    torch.sigmoid: ("sigmoid", (-6.0, 6.0)),
}


class FunctorGraphTransformer:
    def __init__(self, degree: int = 16, target_epsilon: Optional[float] = 1e-6):
        self.degree = degree
        self.target_epsilon = target_epsilon
        self.functor = ACFFunctor(default_dtype=torch.float64)
        self._generated: Dict[str, Callable] = {}

    def _get_horner_fn(self, func_name: str, domain: Tuple[float, float]) -> Callable:
        key = f"{func_name}_{self.degree}_{domain}"
        if key in self._generated:
            return self._generated[key]

        reduction = self.functor.reduce_transcendental(
            func_name,
            degree=self.degree,
            domain=domain,
            target_epsilon=self.target_epsilon,
        )
        coeffs = torch.tensor(reduction.metadata["monomial_coefficients"], dtype=torch.float64)

        def horner_eval(x, _coeffs=coeffs):
            return HornerReducer.execute_horner(_coeffs.to(device=x.device, dtype=x.dtype), x)

        self._generated[key] = horner_eval
        return horner_eval

    def transform(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        graph = gm.graph
        replacements = 0

        for node in list(graph.nodes):
            if node.op == "call_function" and node.target in _TORCH_OP_MAP:
                func_name, domain = _TORCH_OP_MAP[node.target]
                fn = self._get_horner_fn(func_name, domain)
                with graph.inserting_after(node):
                    new_node = graph.call_function(fn, args=node.args, kwargs={})
                    node.replace_all_uses_with(new_node)
                graph.erase_node(node)
                replacements += 1

        if replacements:
            graph.lint()
            gm.recompile()
        return gm


def acf_backend(
    gm: torch.fx.GraphModule,
    example_inputs,
    degree: int = 16,
    target_epsilon: float = 1e-6,
):
    try:
        transformer = FunctorGraphTransformer(degree=degree, target_epsilon=target_epsilon)
        transformed = transformer.transform(gm)
        return transformed.forward
    except Exception as exc:
        warnings.warn(f"ACF backend failed ({exc}), falling back to eager.")
        return gm.forward


# Canonical ACF alias (preferred name)


try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


if HAS_JAX:

    class JaxACFFunctor:
        def __init__(self, degree: int = 20):
            self.degree = degree
            self._cache: Dict[str, jnp.ndarray] = {}
            self._pt_functor = ACFFunctor(default_dtype=torch.float64)

        def _coeffs(self, func_name: str, domain: Tuple[float, float]) -> jnp.ndarray:
            key = f"{func_name}_{self.degree}_{domain}"
            if key not in self._cache:
                red = self._pt_functor.reduce_transcendental(func_name, degree=self.degree, domain=domain)
                self._cache[key] = jnp.asarray(red.metadata["monomial_coefficients"])
            return self._cache[key]

        @staticmethod
        def horner_jax(coefficients: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            out = jnp.full_like(x, coefficients[-1])
            for i in range(coefficients.shape[0] - 2, -1, -1):
                out = coefficients[i] + x * out
            return out

        def sin(self, x: jnp.ndarray) -> jnp.ndarray:
            return self.horner_jax(self._coeffs("sin", (-3.141592653589793, 3.141592653589793)), x)

        def cos(self, x: jnp.ndarray) -> jnp.ndarray:
            return self.horner_jax(self._coeffs("cos", (-3.141592653589793, 3.141592653589793)), x)

        def exp(self, x: jnp.ndarray) -> jnp.ndarray:
            return self.horner_jax(self._coeffs("exp", (-1.0, 1.0)), x)

        def tanh(self, x: jnp.ndarray) -> jnp.ndarray:
            return self.horner_jax(self._coeffs("tanh", (-3.0, 3.0)), x)

    def create_jax_functor(degree: int = 20) -> JaxACFFunctor:
        return JaxACFFunctor(degree=degree)

else:

    def create_jax_functor(*args, **kwargs):
        raise ImportError("JAX is not installed. Install jax and jaxlib.")
