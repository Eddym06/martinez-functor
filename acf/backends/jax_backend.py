try:
    from acf_functor.backends import create_jax_functor

    _jax_functor = create_jax_functor(degree=20)

    def acf_primitive(tensor):
        """JAX-compatible ACF primitive using Horner-reduced sin approximation."""
        return _jax_functor.sin(tensor)

    # Backward-compat alias
    acf_primitive = acf_primitive

except ImportError:
    def acf_primitive(*args, **kwargs):
        raise ImportError("JAX is not installed. Please install JAX/JAXLIB to use this backend.")

    # Backward-compat alias
    acf_primitive = acf_primitive
