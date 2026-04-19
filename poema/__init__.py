"""Poema language package for the Affine Collapse Functor (ACF) ecosystem."""

from .ast_nodes import (
    ASTNode,
    AffineNode,
    ComposeNode,
    ConstantNode,
    ConstraintNode,
    FMAInstruction,
    Flow,
    Form,
    GeometricType,
    IdentityNode,
    InputNode,
    Morphism,
    ParameterNode,
    PolynomialNode,
    PrecisionDegradationWarning,
    Scalar,
    ScaleNode,
    ShiftNode,
    StratifiedNode,
    TopologicalObstructionError,
    TranscendentalNode,
    Vector,
)
from .compiler import (
    CompilationReport,
    FMALinearizer,
    GeometricTypeChecker,
    PoemCompiler,
    PytorchBackend,
    TritonBackend,
)
from .frontend import BiPoem, CoPoem, Poem
from .free_algebra import FreeAlgebra, NormalizationTrace, RelationType, RewriteRule
from .sheaf_semantics import (
    CohomologicalVerdict,
    SemanticDomain,
    SemanticSection,
    SheafSemantics,
)
from .affine_turing import (
    AffineTuringMachine,
    MTAExecution,
    MTAProgram,
    MTAState,
    MTATransition,
)
from .meta_compiler import (
    CodeGenPhase,
    MTACompilationPhase,
    MetaCompilationResult,
    MetaCompiler,
    NormalizationPhase,
    SemanticPhase,
)
from .error_propagation import (
    ErrorBound,
    compose_error_bounds,
    affine_error_propagation,
    sum_error_bounds,
    LIPSCHITZ_CONSTANTS,
)
from .multivariate import (
    MultivariateExpr,
    JacobianExpr,
    parse_multivariate,
)
from .activations_modern import (
    gelu_exact,
    swiglu,
    rope_embedding,
)
from .nn_integration import (
    PoemActivationLayer,
    replace_activations_in_model,
)
from .gemm_collider import (
    GEMMCollider,
    KahanHornerKernel,
    GEMMBlock,
    ColliderReport,
)
from .auto_domain_repair import (
    ExpandedDomainRepair,
    ExpandedDomainCert,
)
from .copoem_multiobjective import (
    CoPoemMultiObjective,
    AndersonState,
    IncompatibilityReport,
)
from .backends import (
    BackendProtocol,
    BackendCapabilities,
    BackendResult,
    BackendRegistry,
    NumpyBackend,
    CNativeEngine,
    CEngineCapabilities,
    CEngineBenchmark,
    WasmBackend,
    WasmArtifact,
    ONNXBackend,
    ONNXBuildInfo,
    PytorchBackendAdapter,
    ROCmBackendAdapter,
    VerilogBackend,
)
from .lean_live_verifier import (
    LeanLiveVerifier,
    VerificationStatus,
    VerificationResult,
    FullSuiteReport,
)
from .canonical_alpha import (
    compute_canonical_alpha,
    CanonicalAlpha,
    AlphaEstimates,
    AlphaEstimator,
    AlphaCanonicalizer,
)
from .taa_agent import (
    TAAAgent,
    TAAReport,
    AlphaClass,
)
from .ergon import (
    ERGONAgent,
    ERGONReport,
    SRBMeasure,
)

__version__ = "2.4.0"

__all__ = [
    "Poem",
    "CoPoem",
    "BiPoem",
    "PoemCompiler",
    "CompilationReport",
    "MetaCompiler",
    "MetaCompilationResult",
    "GeometricTypeChecker",
    "FMALinearizer",
    "PytorchBackend",
    "TritonBackend",
    "GeometricType",
    "Scalar",
    "Vector",
    "Morphism",
    "Flow",
    "Form",
    "ASTNode",
    "ScaleNode",
    "ShiftNode",
    "ComposeNode",
    "AffineNode",
    "IdentityNode",
    "ConstantNode",
    "InputNode",
    "PolynomialNode",
    "TranscendentalNode",
    "StratifiedNode",
    "ParameterNode",
    "ConstraintNode",
    "FreeAlgebra",
    "NormalizationTrace",
    "RelationType",
    "RewriteRule",
    "SheafSemantics",
    "CohomologicalVerdict",
    "SemanticSection",
    "SemanticDomain",
    "AffineTuringMachine",
    "MTAProgram",
    "MTAState",
    "MTATransition",
    "MTAExecution",
    "NormalizationPhase",
    "SemanticPhase",
    "CodeGenPhase",
    "MTACompilationPhase",
    "FMAInstruction",
    "TopologicalObstructionError",
    "PrecisionDegradationWarning",
    # Error propagation
    "ErrorBound",
    "compose_error_bounds",
    "affine_error_propagation",
    "sum_error_bounds",
    "LIPSCHITZ_CONSTANTS",
    # Multivariate
    "MultivariateExpr",
    "JacobianExpr",
    "parse_multivariate",
    # Modern activations
    "gelu_exact",
    "swiglu",
    "rope_embedding",
    # NN integration
    "PoemActivationLayer",
    "replace_activations_in_model",
    # GEMM-Triton Collider
    "GEMMCollider",
    "KahanHornerKernel",
    "GEMMBlock",
    "ColliderReport",
    # Auto-Domain Repair mejorado
    "ExpandedDomainRepair",
    "ExpandedDomainCert",
    # CoPoem Multiobjetivo
    "CoPoemMultiObjective",
    "AndersonState",
    "IncompatibilityReport",
    # TAA: Tensor Autocomputable Agent (Koopman)
    "TAAAgent",
    "TAAReport",
    "AlphaClass",
    # ERGON: Perron-Frobenius Agent (SRB / Pesin)
    "ERGONAgent",
    "ERGONReport",
    "SRBMeasure",
]
