// ir.rs — Representación Intermedia de Gideon (Rust core)
//
// Espejo Rust-nativo del Python GideonIR.
// Ventajas sobre la versión Python:
//   - Pattern matching exhaustivo con `match` — el compilador avisa si falta un caso.
//   - Zero-cost abstractions: los enums no tienen overhead de boxing.
//   - Serializable/deserializable con serde sin código adicional.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tipos de nodos soportados — espejo de Python IRNodeKind.
///
/// El compilador Rust garantiza que cualquier `match` sobre este enum
/// sea exhaustivo: si se añade un nuevo variant y se olvida manejarlo,
/// el código no compila.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IRNodeKind {
    // Primitivos
    Const,
    Input,
    Fma,
    Identity,
    // Afines
    Scale,
    Shift,
    Affine,
    // Composición
    Compose,
    Parallel,
    Branch,
    // Polinomios
    PolyHorner,
    PolyCheb,
    // Trascendentales
    Sin,
    Cos,
    Exp,
    Log,
    Tanh,
    Sigmoid,
    // Álgebra lineal
    Matmul,
    Gemm,
    Conv,
    Norm,
    Attention,
    // Control de flujo
    Loop,
    Recursive,
    // IA / Teoremas
    ArchProbe,
    TheoremSeed,
}

impl IRNodeKind {
    /// Si este nodo es una operación FMA o afín (compone cadenas).
    pub fn is_affine(&self) -> bool {
        matches!(
            self,
            Self::Fma | Self::Scale | Self::Shift | Self::Affine
            | Self::Identity | Self::Const
        )
    }

    /// Si este nodo puede beneficiarse del backend GPU.
    pub fn prefers_gpu(&self) -> bool {
        matches!(
            self,
            Self::Matmul | Self::Gemm | Self::Conv | Self::Attention
        )
    }

    /// Coste estimado en operaciones FMA (para el dispatcher heurístico).
    pub fn fma_cost_estimate(&self) -> usize {
        match self {
            Self::Fma | Self::Affine | Self::Scale | Self::Shift => 1,
            Self::Identity | Self::Const | Self::Input          => 0,
            Self::Compose                                        => 2,
            Self::Parallel | Self::Branch                       => 2,
            Self::PolyHorner | Self::PolyCheb                   => 8,
            Self::Sin | Self::Cos | Self::Tanh | Self::Sigmoid  => 12,
            Self::Exp | Self::Log                               => 10,
            Self::Matmul | Self::Gemm                           => 100,
            Self::Conv                                          => 200,
            Self::Attention                                     => 300,
            Self::Norm                                          => 20,
            Self::Loop | Self::Recursive                        => 50,
            Self::ArchProbe | Self::TheoremSeed                 => 1,
        }
    }
}

/// Metadatos de seguridad numérica de un nodo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRNodeMetadata {
    pub epsilon_bound: f64,
    pub domain_lo:     f64,
    pub domain_hi:     f64,
    pub interval_lo:   f64,
    pub interval_hi:   f64,
    pub continuity:    String,
    pub fma_cost:      usize,
    pub verified:      bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lean_certificate: Option<String>,
    pub tags:          Vec<String>,
}

impl Default for IRNodeMetadata {
    fn default() -> Self {
        Self {
            epsilon_bound:    0.0,
            domain_lo:        f64::NEG_INFINITY,
            domain_hi:        f64::INFINITY,
            interval_lo:      f64::NEG_INFINITY,
            interval_hi:      f64::INFINITY,
            continuity:       "unknown".into(),
            fma_cost:         0,
            verified:         false,
            lean_certificate: None,
            tags:             Vec::new(),
        }
    }
}

/// Un nodo en el grafo IR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRNode {
    pub node_id:  String,
    pub kind:     IRNodeKind,
    /// Parámetros numéricos del nodo (pesos, biases, coeficientes...).
    pub params:   Vec<f64>,
    /// Inputs: IDs de nodos predecesores.
    pub inputs:   Vec<String>,
    pub metadata: IRNodeMetadata,
}

impl IRNode {
    pub fn new(node_id: impl Into<String>, kind: IRNodeKind) -> Self {
        Self {
            node_id:  node_id.into(),
            kind,
            params:   Vec::new(),
            inputs:   Vec::new(),
            metadata: IRNodeMetadata::default(),
        }
    }

    /// FMA rápido: y = w * x + b
    pub fn fma(id: impl Into<String>, w: f64, b: f64, input: impl Into<String>) -> Self {
        let mut n = Self::new(id, IRNodeKind::Fma);
        n.params = vec![w, b];
        n.inputs = vec![input.into()];
        n.metadata.fma_cost = 1;
        n
    }
}

/// Programa compilado: lista ordenada de nodos + metadata del programa.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GideonProgram {
    pub nodes:      Vec<IRNode>,
    pub chain_hash: String,
    pub n_fma:      usize,
    pub precision:  String,
    /// Mapa opcional de salidas nombradas: nombre → node_id
    pub outputs:    HashMap<String, String>,
}

impl GideonProgram {
    pub fn new(nodes: Vec<IRNode>, precision: impl Into<String>) -> Self {
        let n_fma = nodes.iter().map(|n| n.metadata.fma_cost).sum();
        let hash  = compute_chain_hash(&nodes);
        Self {
            n_fma,
            chain_hash: hash,
            nodes,
            precision: precision.into(),
            outputs: HashMap::new(),
        }
    }

    /// Si todos los nodos son afines, el programa es plegable (foldable).
    pub fn is_fully_affine(&self) -> bool {
        self.nodes.iter().all(|n| n.kind.is_affine())
    }

    /// Número de elementos del programa (útil para elegir ruta GPU).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Hash determinista de la cadena de nodos (para fold cache).
/// Se usa el contenido de `kind + params` — no los IDs.
fn compute_chain_hash(nodes: &[IRNode]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut h = DefaultHasher::new();
    for node in nodes {
        // Hash del tipo de nodo
        format!("{:?}", node.kind).hash(&mut h);
        // Hash de parámetros con representación de bits (NaN-safe)
        for &p in &node.params {
            p.to_bits().hash(&mut h);
        }
    }
    format!("{:016x}", h.finish())
}
