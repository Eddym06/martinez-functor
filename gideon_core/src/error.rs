// error.rs — Tipos de error unificados de GideonCore
//
// Rust obliga a manejar todos los errores en tiempo de compilación.
// Este módulo define la jerarquía de errores para el core.

use std::fmt;

/// Error principal de GideonCore.
#[derive(Debug, Clone, PartialEq)]
pub enum GideonError {
    /// El programa IR está vacío o malformado.
    InvalidProgram(String),
    /// El backend solicitado no está disponible en este hardware.
    BackendUnavailable(String),
    /// Fallo durante la ejecución de un kernel.
    ExecutionFailed(String),
    /// Error de I/O (lectura/escritura de telemetría, caché).
    Io(String),
    /// Deserialización de JSON fallida.
    Deserialize(String),
    /// Condición de carrera detectada (no debería ocurrir — indica bug).
    ConcurrencyError(String),
    /// El nodo IR referenciado no existe en el grafo.
    NodeNotFound(String),
    /// Dimensiones incompatibles en una operación (GEMM, etc.).
    DimensionMismatch { expected: String, got: String },
}

impl fmt::Display for GideonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProgram(s)       => write!(f, "InvalidProgram: {s}"),
            Self::BackendUnavailable(s)   => write!(f, "BackendUnavailable: {s}"),
            Self::ExecutionFailed(s)      => write!(f, "ExecutionFailed: {s}"),
            Self::Io(s)                   => write!(f, "IO: {s}"),
            Self::Deserialize(s)          => write!(f, "Deserialize: {s}"),
            Self::ConcurrencyError(s)     => write!(f, "ConcurrencyError: {s}"),
            Self::NodeNotFound(s)         => write!(f, "NodeNotFound: {s}"),
            Self::DimensionMismatch { expected, got } =>
                write!(f, "DimensionMismatch: expected {expected}, got {got}"),
        }
    }
}

impl std::error::Error for GideonError {}

// Conversiones desde errores estándar
impl From<std::io::Error> for GideonError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

impl From<serde_json::Error> for GideonError {
    fn from(e: serde_json::Error) -> Self {
        Self::Deserialize(e.to_string())
    }
}

/// Alias de conveniencia.
pub type GideonResult<T> = Result<T, GideonError>;
