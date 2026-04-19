// lib.rs — Punto de entrada del crate gideon_core
//
// Expone dos superficies:
//   1. API Rust pura (rlib) — para tests unitarios en Rust y benchmarks.
//   2. Módulo Python via pyo3 (cdylib) — para `import gideon_core` desde Python.
//
// pyo3 convierte panics de Rust en excepciones Python (no mata el intérprete).
// Esto es imposible con cffi/ctypes donde un segfault en C mata todo el proceso.

pub mod error;
pub mod ir;
pub mod dispatcher;
pub mod telemetry;
pub mod scheduler;
pub mod engine;

// Re-exports para uso desde Rust
pub use error::{GideonError, GideonResult};
pub use ir::{IRNode, IRNodeKind, GideonProgram, IRNodeMetadata};

/// Versión del core Rust.
pub use dispatcher::{GideonDispatcher, DispatchDecision, BackendHint, HardwareProfile};
pub use telemetry::{GideonTelemetry, ExecutionRecord, BackendStats};
pub use scheduler::{GideonScheduler, ExecutionPlan, PhaseResult};
pub use engine::{GideonCoreEngine, CoreEngineConfig, CoreExecutionResult};

// ── Módulo Python (solo se compila cuando se genera la extensión cdylib) ─────
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
const CORE_VERSION: &str = "1.0.0";

// ── Clases Python expuestas ───────────────────────────────────────────────────

/// Wrapper Python sobre CoreEngineConfig.
#[pyclass(name = "CoreEngineConfig")]
#[derive(Clone)]
struct PyCoreEngineConfig {
    inner: CoreEngineConfig,
}

#[pymethods]
impl PyCoreEngineConfig {
    #[new]
    #[pyo3(signature = (
        precision     = "fp64",
        fold_affine   = true,
        gpu_min_elements = 2_000_000,
        use_parallel  = true,
        max_threads   = None,
        telemetry_path = None,
    ))]
    fn new(
        precision:        &str,
        fold_affine:      bool,
        gpu_min_elements: usize,
        use_parallel:     bool,
        max_threads:      Option<usize>,
        telemetry_path:   Option<&str>,
    ) -> Self {
        Self {
            inner: CoreEngineConfig {
                precision:        precision.into(),
                fold_affine,
                gpu_min_elements,
                use_parallel,
                max_threads,
                telemetry_path:   telemetry_path.map(std::path::PathBuf::from),
                fold_cache_path:  None,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CoreEngineConfig(precision={:?}, fold={}, gpu_min={})",
            self.inner.precision, self.inner.fold_affine, self.inner.gpu_min_elements
        )
    }
}

/// Wrapper Python sobre GideonCoreEngine.
#[pyclass(name = "GideonCoreEngine")]
struct PyGideonCoreEngine {
    inner: engine::GideonCoreEngine,
}

#[pymethods]
impl PyGideonCoreEngine {
    /// Crea el motor con una configuración opcional.
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<PyCoreEngineConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self { inner: GideonCoreEngine::new(cfg) }
    }

    /// Ejecuta una cadena FMA sobre un array de entrada.
    ///
    /// Args:
    ///     weights: Lista de pesos (float) de la cadena FMA.
    ///     biases:  Lista de biases correspondientes.
    ///     x:       Array de entrada (list[float]).
    ///     precision: "fp64" o "fp32" (default: "fp64").
    ///
    /// Returns:
    ///     dict con keys: output, elapsed_ms, backend, folded, total_fma, success.
    #[pyo3(signature = (weights, biases, x, precision = "fp64"))]
    fn run_fma<'py>(
        &self,
        py:        Python<'py>,
        weights:   Vec<f64>,
        biases:    Vec<f64>,
        x:         Vec<f64>,
        precision: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        if weights.len() != biases.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "weights y biases deben tener la misma longitud"
            ));
        }

        // Construir un GideonProgram mínimo desde los parámetros
        let nodes: Vec<IRNode> = weights
            .iter()
            .zip(biases.iter())
            .enumerate()
            .map(|(i, (&w, &b))| IRNode::fma(format!("fma_{i}"), w, b, format!("in_{i}")))
            .collect();
        let program = GideonProgram::new(nodes, precision);

        let result = self.inner.run_fma(&program, &x).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })?;

        let d = PyDict::new_bound(py);
        d.set_item("output",     &result.output)?;
        d.set_item("elapsed_ms", result.elapsed_ms)?;
        d.set_item("backend",    &result.backend)?;
        d.set_item("folded",     result.folded)?;
        d.set_item("gpu_used",   result.gpu_used)?;
        d.set_item("total_fma",  result.total_fma)?;
        d.set_item("success",    result.success)?;
        Ok(d)
    }

    /// Ejecuta GEMM: C = alpha·A@B + beta·C (matrices row-major, f64).
    ///
    /// Returns: (C_result, elapsed_ms)
    #[pyo3(signature = (a, b, c, m, n, k, alpha = 1.0, beta = 0.0))]
    fn run_gemm(
        &self,
        a:     Vec<f64>,
        b:     Vec<f64>,
        mut c: Vec<f64>,
        m:     i32,
        n:     i32,
        k:     i32,
        alpha: f64,
        beta:  f64,
    ) -> PyResult<(Vec<f64>, f64)> {
        let elapsed = self.inner.run_gemm(&a, &b, &mut c, alpha, beta, m, n, k)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok((c, elapsed))
    }

    /// Estadísticas de telemetría.
    fn telemetry_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.inner.telemetry().backend_stats();
        let d = PyDict::new_bound(py);
        d.set_item("total_records", self.inner.telemetry().len())?;
        d.set_item("fold_cache_size", self.inner.fold_cache_size())?;
        d.set_item("hardware", self.inner.dispatcher().hardware_summary())?;

        let backends = PyDict::new_bound(py);
        for (k, v) in &stats {
            let bs = PyDict::new_bound(py);
            bs.set_item("count",  v.count)?;
            bs.set_item("avg_ms", v.avg_ms)?;
            bs.set_item("min_ms", v.min_ms)?;
            bs.set_item("p50_ms", v.p50_ms)?;
            bs.set_item("p95_ms", v.p95_ms)?;
            bs.set_item("max_ms", v.max_ms)?;
            backends.set_item(k, bs)?;
        }
        d.set_item("backends", backends)?;
        Ok(d)
    }

    /// Exporta calibración para el bucle cerrado ACF.
    fn export_acf_calibration<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let cal = self.inner.telemetry().export_acf_calibration();
        let d = PyDict::new_bound(py);

        macro_rules! set_from_json {
            ($key:expr) => {
                if let Some(v) = cal.get($key) {
                    d.set_item($key, v.to_string())?;
                }
            };
        }
        set_from_json!("n_records");
        set_from_json!("fastest_backend");
        set_from_json!("fold_effectiveness_pct");
        set_from_json!("gpu_usage_pct");
        set_from_json!("cache_hit_rate_pct");
        set_from_json!("acf_notes");
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "GideonCoreEngine(v{}, hw={})",
            CORE_VERSION,
            self.inner.dispatcher().hardware_summary()
        )
    }
}

/// Wrapper Python sobre ExecutionPlan.
#[pyclass(name = "ExecutionPlan")]
struct PyExecutionPlan {
    inner: ExecutionPlan,
}

#[pymethods]
impl PyExecutionPlan {
    /// Construye un plan desde una descripción JSON del programa.
    ///
    /// Args:
    ///     nodes_json: lista de dicts con keys "node_id", "kind", "inputs", "params".
    ///     precision:  "fp64" o "fp32".
    #[staticmethod]
    #[pyo3(signature = (nodes_json, precision = "fp64"))]
    fn from_nodes(nodes_json: Vec<std::collections::HashMap<String, String>>, precision: &str) -> Self {
        let nodes: Vec<IRNode> = nodes_json
            .into_iter()
            .enumerate()
            .map(|(i, m)| {
                let node_id = m.get("node_id").cloned().unwrap_or_else(|| format!("n{i}"));
                let mut node = IRNode::new(node_id, IRNodeKind::Fma);
                if let Some(inp) = m.get("input") {
                    node.inputs = vec![inp.clone()];
                }
                node
            })
            .collect();
        let program = GideonProgram::new(nodes, precision);
        Self { inner: ExecutionPlan::from_program(&program) }
    }

    fn n_phases(&self) -> usize {
        self.inner.phases.len()
    }

    fn phase_widths<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let widths: Vec<usize> = self.inner.phases.iter().map(|p| p.len()).collect();
        PyList::new_bound(py, widths)
    }

    fn __repr__(&self) -> String {
        let (n_phases, total, max_w) = GideonScheduler::parallelism_stats(&self.inner);
        format!("ExecutionPlan(phases={n_phases}, nodes={total}, max_width={max_w})")
    }
}

// ── Módulo Python ─────────────────────────────────────────────────────────────

/// Define el módulo `gideon_core` importable desde Python.
///
/// `#[pymodule]` es la macro pyo3 que registra el módulo.
/// Todos los `#[pyclass]` que se añadan aquí se convierten en clases Python.
#[pymodule]
fn gideon_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", CORE_VERSION)?;
    m.add("__doc__",
        "GideonEngine Rust core — concurrencia segura + backends C AVX-512/GEMM"
    )?;

    m.add_class::<PyCoreEngineConfig>()?;
    m.add_class::<PyGideonCoreEngine>()?;
    m.add_class::<PyExecutionPlan>()?;

    Ok(())
}
