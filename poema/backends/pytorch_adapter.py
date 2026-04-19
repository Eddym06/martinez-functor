"""
PytorchBackendAdapter — wraps existing PytorchBackend in the BackendProtocol interface.
"""
from __future__ import annotations
from typing import Any, List, Tuple

from .protocol import BackendCapabilities, BackendProtocol, BackendResult


class PytorchBackendAdapter(BackendProtocol):

    @property
    def capabilities(self) -> BackendCapabilities:
        gpu = self._has_cuda()
        return BackendCapabilities(
            name="pytorch",
            supports_gpu=gpu,
            supports_cpu=True,
            supports_batched=True,
            supports_gradient=True,
            hardware_vendor="nvidia" if gpu else "generic",
            precision_formats=["fp64", "fp32", "fp16", "bf16"],
            notes="PyTorch backend — GPU if CUDA available, falls back to CPU.",
        )

    def verify_available(self) -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def compile(
        self,
        fma_sequence: List[Any],
        source_ast: Any,
        domain: Tuple[float, float] = (-1.0, 1.0),
        precision: str = "fp64",
        **kwargs,
    ) -> BackendResult:
        import torch
        import numpy as np

        dtype_map = {
            "fp64": torch.float64, "double": torch.float64,
            "fp32": torch.float32, "single": torch.float32,
            "fp16": torch.float16, "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(precision, torch.float64)

        # If we have a concrete FMA sequence, build a direct PyTorch chain.
        # Do NOT delegate to PytorchBackend.compile(source_ast=None) — that
        # path falls through to an empty FMA list and returns identity.
        if fma_sequence:
            weights = [float(getattr(f, "weight", 1.0)) for f in fma_sequence]
            biases  = [float(getattr(f, "bias",   0.0)) for f in fma_sequence]
            _dtype  = dtype

            def _fma_torch(x, _ws=weights, _bs=biases, _dt=_dtype):
                import torch as _torch
                import numpy as _np
                arr = _np.asarray(x, dtype=np.float64)
                t = _torch.as_tensor(arr, dtype=_dt)
                for w, b in zip(_ws, _bs):
                    t = w * t + b
                return t.numpy()

            return BackendResult(
                callable_fn=_fma_torch,
                fma_count=len(fma_sequence),
                epsilon_bound=0.0,
                backend_name="pytorch",
            )

        # Fallback: delegate to PytorchBackend.compile when we have a real AST.
        from ..compiler import PytorchBackend  # existing implementation
        auto_repair = kwargs.get("auto_domain_repair", True)
        fn = PytorchBackend.compile(source_ast, auto_domain_repair=auto_repair)

        def _wrapped(x):
            import torch as _torch
            import numpy as _np
            if isinstance(x, _np.ndarray):
                t = _torch.as_tensor(x, dtype=dtype)
                return fn(t).numpy()
            return fn(x)

        return BackendResult(
            callable_fn=_wrapped,
            fma_count=0,
            epsilon_bound=0.0,
            backend_name="pytorch",
        )


class ROCmBackendAdapter(BackendProtocol):
    """
    AMD ROCm adapter.  Uses PyTorch with HIP device instead of CUDA.
    Falls back to CPU transparently if ROCm is unavailable.
    """

    @property
    def capabilities(self) -> BackendCapabilities:
        rocm = self._has_rocm()
        return BackendCapabilities(
            name="rocm",
            supports_gpu=rocm,
            supports_cpu=not rocm,
            supports_batched=True,
            supports_gradient=True,
            hardware_vendor="amd" if rocm else "generic",
            precision_formats=["fp64", "fp32", "fp16"],
            notes="AMD ROCm/HIP — GPU if ROCm available, otherwise CPU fallback.",
        )

    def verify_available(self) -> bool:
        return self._has_pytorch()

    @staticmethod
    def _has_pytorch() -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _has_rocm() -> bool:
        try:
            import torch
            return torch.cuda.is_available() and "hip" in torch.version.hip if hasattr(torch.version, "hip") else False
        except Exception:
            return False

    def compile(
        self,
        fma_sequence: List[Any],
        source_ast: Any,
        domain: Tuple[float, float] = (-1.0, 1.0),
        precision: str = "fp64",
        **kwargs,
    ) -> BackendResult:
        # Delegate to PyTorch adapter — ROCm runs through PyTorch's HIP device
        adapter = PytorchBackendAdapter()
        result = adapter.compile(fma_sequence, source_ast, domain, precision, **kwargs)
        result.backend_name = "rocm"

        # Wrap to move tensor to AMD device if available
        if self._has_rocm():
            import torch
            base_fn = result.callable_fn
            device = torch.device("cuda")  # PyTorch uses "cuda" for ROCm too

            def _rocm_wrapped(x):
                if hasattr(x, "to"):
                    x = x.to(device)
                out = base_fn(x)
                return out.cpu() if hasattr(out, "cpu") else out

            result.callable_fn = _rocm_wrapped

        return result
