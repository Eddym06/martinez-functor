// dispatcher.rs — GideonDispatcher con concurrencia segura
//
// La clave Rust:
//   - `latency_history` usa Arc<RwLock<...>>: múltiples lectores concurrentes,
//     un escritor a la vez. El compilador garantiza que no hay acceso simultáneo
//     escritor+lector — imposible en Python/C sin sincronización manual.
//   - Pattern matching sobre (n_fma, has_cuda, has_avx512) es exhaustivo
//     y legible: el compilador Rust alerta ante casos no cubiertos.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::ir::GideonProgram;

/// Pista al dispatcher sobre qué clase de backend prefiere el nodo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendHint {
    CpuScalar,
    CpuSimd,    // AVX2 obligatorio
    CpuAvx512,
    GpuCuda,
    GpuRocm,
    Auto,       // El dispatcher decide sin restricciones
}

impl std::fmt::Display for BackendHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CpuScalar  => write!(f, "cpu_scalar"),
            Self::CpuSimd    => write!(f, "cpu_simd"),
            Self::CpuAvx512  => write!(f, "cpu_avx512"),
            Self::GpuCuda    => write!(f, "gpu_cuda"),
            Self::GpuRocm    => write!(f, "gpu_rocm"),
            Self::Auto       => write!(f, "auto"),
        }
    }
}

/// Decisión de despacho producida por el dispatcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchDecision {
    pub backend:            String,
    pub hint:               BackendHint,
    pub reason:             String,
    pub estimated_speedup:  f64,    // Relativo a CPU escalar (1.0 = sin mejora)
    pub use_fold:           bool,   // Si el motor debe intentar fold afín
    pub use_parallel:       bool,   // Si Rayon debe paralelizar las fases
}

impl DispatchDecision {
    pub fn summary(&self) -> String {
        format!(
            "{} ({}×) — {}",
            self.backend, self.estimated_speedup, self.reason
        )
    }
}

/// Perfil de hardware detectado en este host.
///
/// Nota: En Rust no hay herencia. Usamos composición con `Default`
/// para construir perfiles parciales en tests.
#[derive(Debug, Clone, Default)]
pub struct HardwareProfile {
    pub has_avx2:    bool,
    pub has_avx512:  bool,
    pub has_cuda:    bool,
    pub has_rocm:    bool,
    pub cpu_cores:   usize,
    pub gpu_name:    String,
    /// Capacidad de cómputo GPU como (mayor, menor).
    pub gpu_cc:      Option<(u32, u32)>,
}

impl HardwareProfile {
    /// Detecta el hardware disponible en tiempo de ejecución.
    pub fn detect() -> Self {
        let mut p = Self::default();
        p.cpu_cores = num_cpus();

        // Detección de flags CPU desde /proc/cpuinfo (Linux)
        #[cfg(target_os = "linux")]
        if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
            p.has_avx2   = content.contains(" avx2 ");
            p.has_avx512 = content.contains(" avx512f ");
        }

        p
    }
}

/// Obtiene el número de CPUs lógicas de forma portátil.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// GideonDispatcher — toma decisiones de despacho sin condiciones de carrera.
///
/// Seguridad: `latency_history` está protegido por `RwLock` de parking_lot:
///   - Múltiples hilos pueden leer historial simultáneamente.
///   - Solo un hilo puede escribir a la vez.
///   - El compilador Rust impide acceso sin el guard — imposible olvidarlo.
pub struct GideonDispatcher {
    pub hw_profile:      HardwareProfile,
    /// Backend_name → lista de latencias en ms (últimas 200 muestras).
    latency_history: Arc<RwLock<std::collections::HashMap<String, Vec<f64>>>>,
    gpu_min_elements: usize,
}

impl GideonDispatcher {
    pub fn new(hw_profile: HardwareProfile) -> Self {
        Self {
            hw_profile,
            latency_history: Arc::new(RwLock::new(Default::default())),
            gpu_min_elements: 2_000_000,
        }
    }

    pub fn with_gpu_threshold(mut self, threshold: usize) -> Self {
        self.gpu_min_elements = threshold;
        self
    }

    /// Registra una latencia medida para el historial de ranking dinámico.
    /// Thread-safe: el RwLock garantiza que no hay data races.
    pub fn record_latency(&self, backend: &str, elapsed_ms: f64) {
        let mut guard = self.latency_history.write();
        let history = guard.entry(backend.to_string()).or_default();
        history.push(elapsed_ms);
        // Ventana deslizante: mantener sólo las últimas 200 muestras
        if history.len() > 200 {
            history.remove(0);
        }
    }

    /// Media de latencias registradas para un backend.
    pub fn avg_latency(&self, backend: &str) -> Option<f64> {
        let guard = self.latency_history.read();
        guard.get(backend).filter(|v| !v.is_empty()).map(|v| {
            v.iter().copied().sum::<f64>() / v.len() as f64
        })
    }

    /// Decide qué backend ejecuta el programa dado.
    ///
    /// Lógica de decisión (en orden de prioridad):
    ///   1. Si el historial de latencias favorece un backend concreto → ese backend.
    ///   2. Si n_fma > umbral GPU y CUDA disponible → gpu_cuda.
    ///   3. Si AVX-512 disponible → cpu_avx512.
    ///   4. Si AVX2 disponible → cpu_simd.
    ///   5. Fallback → cpu_scalar.
    ///
    /// El pattern matching de Rust garantiza cobertura total de casos.
    pub fn decide(&self, program: &GideonProgram, n_elements: usize) -> DispatchDecision {
        // 1. Verificar si el historial tiene una preferencia clara
        if let Some(best) = self.best_from_history() {
            return DispatchDecision {
                backend:           best.clone(),
                hint:              BackendHint::Auto,
                reason:            "latency history".into(),
                estimated_speedup: self.speedup_estimate(&best),
                use_fold:          program.is_fully_affine(),
                use_parallel:      self.hw_profile.cpu_cores >= 4,
            };
        }

        // 2. Análisis del programa para decidir heurísticamente
        let any_gpu_node = program.nodes.iter().any(|n| n.kind.prefers_gpu());
        let large_workload = n_elements >= self.gpu_min_elements;

        match (
            any_gpu_node || large_workload,
            self.hw_profile.has_cuda,
            self.hw_profile.has_avx512,
            self.hw_profile.has_avx2,
        ) {
            // GPU disponible y carga suficiente
            (true, true, _, _) => DispatchDecision {
                backend:           "gpu_cuda".into(),
                hint:              BackendHint::GpuCuda,
                reason:            "large workload + CUDA available".into(),
                estimated_speedup: 8.0,
                use_fold:          program.is_fully_affine(),
                use_parallel:      false,  // GPU ya paralela internamente
            },
            // AVX-512
            (_, false, true, _) => DispatchDecision {
                backend:           "cpu_avx512".into(),
                hint:              BackendHint::CpuAvx512,
                reason:            "AVX-512 available".into(),
                estimated_speedup: 4.0,
                use_fold:          program.is_fully_affine(),
                use_parallel:      self.hw_profile.cpu_cores >= 4,
            },
            // AVX2 + FMA
            (_, _, _, true) => DispatchDecision {
                backend:           "cpu_simd".into(),
                hint:              BackendHint::CpuSimd,
                reason:            "AVX2+FMA available".into(),
                estimated_speedup: 2.5,
                use_fold:          program.is_fully_affine(),
                use_parallel:      self.hw_profile.cpu_cores >= 2,
            },
            // Escalar
            _ => DispatchDecision {
                backend:           "cpu_scalar".into(),
                hint:              BackendHint::CpuScalar,
                reason:            "baseline scalar".into(),
                estimated_speedup: 1.0,
                use_fold:          program.is_fully_affine(),
                use_parallel:      false,
            },
        }
    }

    /// Devuelve el backend más rápido según historial, si hay suficiente data.
    fn best_from_history(&self) -> Option<String> {
        let guard = self.latency_history.read();
        if guard.len() < 2 {
            return None; // No hay suficientes backends para comparar
        }
        guard
            .iter()
            .filter(|(_, v)| v.len() >= 5)
            .min_by(|(_, a), (_, b)| {
                let avg_a = a.iter().copied().sum::<f64>() / a.len() as f64;
                let avg_b = b.iter().copied().sum::<f64>() / b.len() as f64;
                avg_a.partial_cmp(&avg_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone())
    }

    /// Speedup estimado para un backend (basado en historial o heurística).
    fn speedup_estimate(&self, backend: &str) -> f64 {
        let guard = self.latency_history.read();
        let scalar_avg = guard
            .get("cpu_scalar")
            .filter(|v| !v.is_empty())
            .map(|v| v.iter().copied().sum::<f64>() / v.len() as f64)
            .unwrap_or(1.0);
        let backend_avg = guard
            .get(backend)
            .filter(|v| !v.is_empty())
            .map(|v| v.iter().copied().sum::<f64>() / v.len() as f64)
            .unwrap_or(scalar_avg);
        if backend_avg > 0.0 {
            (scalar_avg / backend_avg).max(1.0)
        } else {
            1.0
        }
    }

    /// Resumen de hardware para logging.
    pub fn hardware_summary(&self) -> String {
        format!(
            "HW: cores={}, avx2={}, avx512={}, cuda={}, gpu={}",
            self.hw_profile.cpu_cores,
            self.hw_profile.has_avx2,
            self.hw_profile.has_avx512,
            self.hw_profile.has_cuda,
            if self.hw_profile.gpu_name.is_empty() { "none" } else { &self.hw_profile.gpu_name },
        )
    }
}
