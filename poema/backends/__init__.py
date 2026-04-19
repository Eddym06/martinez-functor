"""
Poema Backend Abstraction Layer.

Provides a hardware-agnostic interface over multiple backend targets:
  - NumpyBackend         : Pure NumPy CPU (no torch dependency)
  - CNativeEngine        : Native C + AVX2/AVX-512 + OpenMP (titan engine)
  - WasmBackend          : WebAssembly WAT/WASM + JS loader
  - ONNXBackend          : ONNX graph (TensorRT, OpenVINO, CoreML, ORT)
  - PytorchBackendAdapter: CPU/NVIDIA via PyTorch
  - ROCmBackendAdapter   : AMD GPU via PyTorch ROCm
  - VerilogBackend       : Synthesisable RTL/HDL
  - GideonEngine         : Motor Unificado — orquesta todos los anteriores
"""

from .protocol import BackendProtocol, BackendCapabilities, BackendResult
from .numpy_backend import NumpyBackend
from .pytorch_adapter import PytorchBackendAdapter
from .rocm_adapter import ROCmBackendAdapter
from .verilog_backend import VerilogBackend
from .c_engine import CNativeEngine, CEngineCapabilities, CEngineBenchmark
from .wasm_backend import WasmBackend, WasmArtifact
from .onnx_backend import ONNXBackend, ONNXBuildInfo
from .registry import BackendRegistry
from .gideon import (
    GideonEngine,
    GideonIR,
    IRNode,
    IRNodeKind,
    GideonProgram,
    GideonGraph,
    GideonGraphNode,
    GraphEdge,
    ExecutionPlan,
    GideonDispatcher,
    DispatchDecision,
    BackendHint,
    GideonNeuralHints,
    ArchitectureBlueprint,
    GideonTheoremSeeds,
    TheoremCandidate,
)

__all__ = [
    "BackendProtocol",
    "BackendCapabilities",
    "BackendResult",
    "NumpyBackend",
    "CNativeEngine",
    "CEngineCapabilities",
    "CEngineBenchmark",
    "WasmBackend",
    "WasmArtifact",
    "ONNXBackend",
    "ONNXBuildInfo",
    "PytorchBackendAdapter",
    "ROCmBackendAdapter",
    "VerilogBackend",
    "BackendRegistry",
    # Gideon — Motor Unificado
    "GideonEngine",
    "GideonIR",
    "IRNode",
    "IRNodeKind",
    "GideonProgram",
    "GideonGraph",
    "GideonGraphNode",
    "GraphEdge",
    "ExecutionPlan",
    "GideonDispatcher",
    "DispatchDecision",
    "BackendHint",
    "GideonNeuralHints",
    "ArchitectureBlueprint",
    "GideonTheoremSeeds",
    "TheoremCandidate",
]

