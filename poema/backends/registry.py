"""
BackendRegistry — centralized registry for Poema backends.
Auto-detects available backends at import time.
"""
from __future__ import annotations
from typing import Dict, Optional, Type

from .protocol import BackendProtocol


class BackendRegistry:
    """
    Registry that maps backend names to instances.
    Auto-detects availability at construction time.
    """

    _REGISTERED: Dict[str, BackendProtocol] = {}
    _initialized: bool = False

    @classmethod
    def _init_defaults(cls) -> None:
        if cls._initialized:
            return
        cls._initialized = True

        from .numpy_backend import NumpyBackend
        from .pytorch_adapter import PytorchBackendAdapter, ROCmBackendAdapter
        from .verilog_backend import VerilogBackend
        from .c_engine import CNativeEngine
        from .wasm_backend import WasmBackend
        from .onnx_backend import ONNXBackend

        for backend in [
            NumpyBackend(),
            CNativeEngine(),
            WasmBackend(),
            ONNXBackend(),
            PytorchBackendAdapter(),
            ROCmBackendAdapter(),
            VerilogBackend(),
        ]:
            cls._REGISTERED[backend.capabilities.name] = backend

    @classmethod
    def get(cls, name: str) -> BackendProtocol:
        cls._init_defaults()
        if name not in cls._REGISTERED:
            available = list(cls._REGISTERED.keys())
            raise KeyError(
                f"Backend '{name}' not found. Available: {available}"
            )
        return cls._REGISTERED[name]

    @classmethod
    def available(cls) -> Dict[str, bool]:
        cls._init_defaults()
        return {name: b.verify_available() for name, b in cls._REGISTERED.items()}

    @classmethod
    def register(cls, backend: BackendProtocol) -> None:
        cls._init_defaults()
        cls._REGISTERED[backend.capabilities.name] = backend

    @classmethod
    def describe_all(cls) -> str:
        cls._init_defaults()
        lines = ["Poema Backend Registry", "=" * 40]
        for name, b in cls._REGISTERED.items():
            avail = "✓ AVAILABLE" if b.verify_available() else "✗ unavailable"
            lines.append(f"\n[{name}]  {avail}")
            lines.append(b.describe())
        return "\n".join(lines)

    @classmethod
    def best_for_cpu(cls) -> BackendProtocol:
        """Return the best available CPU backend (prefers native C engine)."""
        cls._init_defaults()
        for name in ("c_native", "pytorch", "numpy_cpu"):
            b = cls._REGISTERED.get(name)
            if b and b.verify_available():
                return b
        raise RuntimeError("No CPU backend available.")

    @classmethod
    def best_for_gpu(cls) -> BackendProtocol:
        """Return the best available GPU backend."""
        cls._init_defaults()
        for name in ("rocm", "pytorch"):
            b = cls._REGISTERED.get(name)
            if b and b.verify_available() and b.capabilities.supports_gpu:
                return b
        raise RuntimeError("No GPU backend available.")

    # ── Gideon Engine integration ─────────────────────────────────────────────

    @classmethod
    def gideon(cls):
        """
        Devuelve una instancia del motor unificado Gideon.
        Gideon orquesta todos los backends de forma inteligente.
        """
        from .gideon import GideonEngine
        return GideonEngine()

    @classmethod
    def gideon_with_config(cls, **kwargs):
        """Devuelve Gideon con configuración personalizada."""
        from .gideon import GideonEngine, GideonEngineConfig
        cfg = GideonEngineConfig(**kwargs)
        return GideonEngine(config=cfg)
