"""
Gideon — Motor Unificado de Poema.

Gideon es el corazón computacional de Poema: un motor de ejecución matemática
de grado industrial que orquesta todos los backends bajo una interfaz unificada,
y sienta las bases para:

  1. Procesamiento matemático-computacional de alta potencia (FMA nativo, SIMD,
     GPU, FPGA, WASM).
  2. Descubrimiento y entrenamiento autónomo de arquitecturas de IA (base).
  3. Descubrimiento de teoremas matemáticos asistido por computador (base).

Exports principales:
  GideonEngine          — motor unificado principal (v1.2.0)
  GideonIR              — representación intermedia nativa de Poema
  GideonGraph           — grafo de computación heterogéneo
  GideonDispatcher      — despacho inteligente de backends (heurístico)
  MLDispatcher          — dispatcher con aprendizaje de ejecuciones previas
  GideonTelemetry       — base de datos persistente de ejecuciones
  GideonHardwareProfiler — perfilador de hardware con micro-benchmarks
  HardwareCapabilities  — perfil completo de CPU + GPU
  GideonNeuralHints     — heurísticas para arquitecturas de IA
  GideonTheoremSeeds    — semillas para descubrimiento de teoremas
"""

from .engine import GideonEngine, FrozenGraphError
from .ir import GideonIR, IRNode, IRNodeKind, GideonProgram
from .graph import GideonGraph, GideonGraphNode, GraphEdge, ExecutionPlan
from .dispatcher import GideonDispatcher, DispatchDecision, BackendHint
from .neural_hints import GideonNeuralHints, ArchitectureBlueprint
from .theorem_seeds import GideonTheoremSeeds, TheoremCandidate
from .gideon_autotune import GideonHardwareProfiler, HardwareCapabilities
from .ml_dispatcher import GideonTelemetry, MLDispatcher, ExecutionRecord
from .koopman_gpu import KoopmanGPU, KoopmanGPUResult
from .ns_acf_coupled import CoupledNSACFSolver, CoupledNSACFConfig, AdaptiveMeshController
from .turbulence_thermostat import (
    MultiOracleAMR,
    CascadeAccelerator,
    BayesianArbiter,
    KoopmanOracle,
    RuelleOracle,
    ErgonOracle,
    ThermodynamicOracle,
    SpectralOracle,
    OracleVote,
    ThermostatState,
)
from .copoem_spectral_designer import (
    CoPoemSpectralDesigner,
    CoPoemOracle,
    DesignerConfig,
    SpectralDesignState,
)
from .ns3d_hit_solver import HIT3DSolver, HIT3DConfig
from .rom_executor import GideonROMExecutor, ROMExecutionResult, EnsembleResult

__all__ = [
    "GideonEngine",
    "FrozenGraphError",
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
    "GideonHardwareProfiler",
    "HardwareCapabilities",
    "GideonTelemetry",
    "MLDispatcher",
    "ExecutionRecord",
    "KoopmanGPU",
    "KoopmanGPUResult",
    "CoupledNSACFSolver",
    "CoupledNSACFConfig",
    "AdaptiveMeshController",
    "MultiOracleAMR",
    "CascadeAccelerator",
    "BayesianArbiter",
    "KoopmanOracle",
    "RuelleOracle",
    "ErgonOracle",
    "ThermodynamicOracle",
    "SpectralOracle",
    "OracleVote",
    "ThermostatState",
    "CoPoemSpectralDesigner",
    "CoPoemOracle",
    "DesignerConfig",
    "SpectralDesignState",
    "HIT3DSolver",
    "HIT3DConfig",
    "GideonROMExecutor",
    "ROMExecutionResult",
    "EnsembleResult",
]

__version__ = "1.6.0"
__engine__ = "Gideon"
