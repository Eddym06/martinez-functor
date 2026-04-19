// engine.rs — GideonEngineCore: el corazón en Rust
//
// Orquesta el pipeline completo:
//   IR → Plan → Dispatch → Kernel C (FFI) → Resultado → Telemetría
//
// La gestión de memoria es completamente determinista (RAII):
//   - Los buffers de cómputo se libran al salir del scope.
//   - No hay GC que pause la ejecución a mitad de un cómputo crítico.
//   - Arc<> gestiona la propiedad compartida de telemetría y dispatcher.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::dispatcher::{GideonDispatcher, HardwareProfile};
use crate::error::{GideonError, GideonResult};
use crate::ir::GideonProgram;
use crate::scheduler::{ExecutionPlan, GideonScheduler, PhaseResult};
use crate::telemetry::{ExecutionRecord, GideonTelemetry};

// ── Buffer de salida alineado a 64 bytes (Frente 4 — bare-metal) ─────────────
//
// Los kernels C AVX-512 usan `_mm512_loadu_pd` (carga no alineada), que es
// equivalente a `_mm512_load_pd` cuando los datos SÍ están alineados a 64 bytes.
// Garantizar 64-byte alignment:
//   1. Evita cruzar múltiples líneas de caché en la primera carga.
//   2. Permite al hardware pre-fetcher predecir accesos alineados con mayor eficiencia.
//   3. Habilita transiciones futuras a `_mm512_load_pd` (instrucción más simple → menor latencia).
//
// AlignedF64Buffer es un wrapper RAII + Deref que gestiona la alineación correctamente.
// No usa Vec<f64> porque Vec::drop usa dealloc con alignment=8 (align_of::<f64>),
// lo que violaría el contrato del allocador si allocamos con alignment=64.

struct AlignedF64Buffer {
    ptr:    *mut f64,
    len:    usize,
    layout: Layout,
}

impl AlignedF64Buffer {
    /// Aloca n doubles con alineación de 64 bytes (1 línea de caché), inicializados a 0.0.
    fn new(n: usize) -> Self {
        if n == 0 {
            return Self { ptr: std::ptr::NonNull::dangling().as_ptr(), len: 0, layout: Layout::new::<u8>() };
        }
        let layout = Layout::from_size_align(n * std::mem::size_of::<f64>(), 64)
            .expect("AlignedF64Buffer: layout inválido");
        // Safety: layout es válido y tiene size > 0.
        let ptr = unsafe { alloc_zeroed(layout) } as *mut f64;
        assert!(!ptr.is_null(), "AlignedF64Buffer: alloc_zeroed falló (OOM)");
        Self { ptr, len: n, layout }
    }

    fn as_mut_ptr(&mut self) -> *mut f64 { self.ptr }
    fn as_ptr(&self)     -> *const f64  { self.ptr }

    /// Copia el contenido a un Vec<f64> estándar (para la API de retorno).
    fn to_vec(&self) -> Vec<f64> {
        // Safety: ptr es válido, len es correcto, datos son f64 válidos.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }.to_vec()
    }

    fn as_slice(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for AlignedF64Buffer {
    fn drop(&mut self) {
        if self.len > 0 {
            // Safety: ptr fue alocado por alloc_zeroed con este mismo layout.
            unsafe { dealloc(self.ptr as *mut u8, self.layout) };
        }
    }
}

// Safety: AlignedF64Buffer no comparte el ptr entre threads sin sincronización.
unsafe impl Send for AlignedF64Buffer {}
unsafe impl Sync for AlignedF64Buffer {}

// ── FFI declarations — llama a los kernels C compilados con build.rs ────────
// Safety: los punteros son válidos, lon-overlapping, y de longitud garantizada.
extern "C" {
    fn gideon_fma_vector(
        out:      *mut f64,
        x:        *const f64,
        w:        f64,
        b:        f64,
        n:        usize,
    );

    fn gideon_fma_chain(
        out:      *mut f64,
        x:        *const f64,
        weights:  *const f64,
        biases:   *const f64,
        n_elems:  usize,
        n_fma:    usize,
    );

    fn gideon_fma_fold(
        out: *mut f64,
        x:   *const f64,
        w:   f64,
        b:   f64,
        n:   usize,
    );

    fn gideon_gemm(
        C:     *mut f64,
        A:     *const f64,
        B:     *const f64,
        alpha: f64,
        beta:  f64,
        M:     i32,
        N:     i32,
        K:     i32,
    );
}

/// Resultado de una ejecución del motor Rust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreExecutionResult {
    pub output:      Vec<f64>,
    pub elapsed_ms:  f64,
    pub backend:     String,
    pub total_fma:   usize,
    pub success:     bool,
    pub error:       String,
    pub cache_hit:   bool,
    pub folded:      bool,
    pub gpu_used:    bool,
    pub phase_results: Vec<PhaseResult>,
}

/// Configuración del motor Rust.
#[derive(Debug, Clone)]
pub struct CoreEngineConfig {
    pub precision:        String,
    pub fold_affine:      bool,
    pub gpu_min_elements: usize,
    pub use_parallel:     bool,
    pub max_threads:      Option<usize>,
    pub telemetry_path:   Option<PathBuf>,
    pub fold_cache_path:  Option<PathBuf>,
}

impl Default for CoreEngineConfig {
    fn default() -> Self {
        Self {
            precision:        "fp64".into(),
            fold_affine:      true,
            gpu_min_elements: 2_000_000,
            use_parallel:     true,
            max_threads:      None,
            telemetry_path:   None,
            fold_cache_path:  None,
        }
    }
}

/// Caché de fold: chain_hash → (W_total, B_total) o None si la cadena
/// no es plegable (tiene nodos no-afines).
type FoldCache = Arc<Mutex<HashMap<String, Option<(f64, f64)>>>>;

/// Motor principal de GideonCore en Rust.
pub struct GideonCoreEngine {
    config:     CoreEngineConfig,
    dispatcher: Arc<GideonDispatcher>,
    telemetry:  GideonTelemetry,
    scheduler:  GideonScheduler,
    fold_cache: FoldCache,
}

impl GideonCoreEngine {
    /// Construye el motor y carga la caché de fold desde disco.
    pub fn new(config: CoreEngineConfig) -> Self {
        let hw = HardwareProfile::detect();
        let dispatcher = Arc::new(
            GideonDispatcher::new(hw).with_gpu_threshold(config.gpu_min_elements)
        );
        let telemetry = GideonTelemetry::new(config.telemetry_path.as_deref());
        let scheduler = {
            let mut s = GideonScheduler::new(telemetry.clone());
            if let Some(n) = config.max_threads {
                s = s.with_threads(n);
            }
            s
        };

        let fold_cache = Arc::new(Mutex::new(HashMap::new()));
        Self::load_fold_cache(&fold_cache, config.fold_cache_path.as_deref());

        Self { config, dispatcher, telemetry, scheduler, fold_cache }
    }

    /// Ejecuta una cadena FMA sobre un array de entrada.
    ///
    /// Pipeline:
    ///   1. Intentar fold afín (si habilitado y cadena afín)
    ///   2. Si fold exitoso → kernel C `gideon_fma_fold` (1 op)
    ///   3. Si no → kernel C `gideon_fma_chain` (n_fma ops)
    ///   4. Registrar en telemetría (sin bloquear el resultado)
    pub fn run_fma(
        &self,
        program: &GideonProgram,
        x: &[f64],
    ) -> GideonResult<CoreExecutionResult> {
        let t0   = Instant::now();
        let n    = x.len();
        // Buffer de salida alineado a 64 bytes (Frente 4): permite al prefetcher
        // operar sobre líneas de caché enteras desde el primer acceso.
        let mut out = AlignedF64Buffer::new(n);

        // 1. Dispatch decision
        let dec = self.dispatcher.decide(program, n);

        // 2. Intentar fold afín
        let fold = if self.config.fold_affine && program.is_fully_affine() {
            self.try_fold(program)
        } else {
            None
        };

        let (folded, gpu_used) = (fold.is_some(), dec.backend.starts_with("gpu"));

        // 3. Ejecutar kernel C apropiado
        if let Some((w, b)) = fold {
            // Ruta óptima: una sola operación
            unsafe {
                gideon_fma_fold(out.as_mut_ptr(), x.as_ptr(), w, b, n);
            }
        } else {
            // Extraer pesos y biases de los nodos FMA
            let (weights, biases) = extract_fma_params(program);
            if weights.is_empty() {
                return Err(GideonError::InvalidProgram(
                    "ningún nodo FMA encontrado en el programa".into()
                ));
            }
            unsafe {
                gideon_fma_chain(
                    out.as_mut_ptr(),
                    x.as_ptr(),
                    weights.as_ptr(),
                    biases.as_ptr(),
                    n,
                    weights.len(),
                );
            }
        }

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // 4. Registrar en telemetría (sin bloquear el resultado)
        self.telemetry.record(ExecutionRecord {
            chain_hash: program.chain_hash.clone(),
            n_elements: n,
            n_fma:      program.n_fma,
            backend:    dec.backend.clone(),
            elapsed_ms,
            folded,
            gpu_used,
            timestamp:  ExecutionRecord::now(),
            precision:  self.config.precision.clone(),
            success:    true,
            cache_hit:  false,
        });

        // 5. Actualizar historial del dispatcher
        self.dispatcher.record_latency(&dec.backend, elapsed_ms);

        Ok(CoreExecutionResult {
            output:        out.to_vec(),  // copia desde buffer 64B-aligned a Vec<f64> estándar
            elapsed_ms,
            backend:       dec.backend,
            total_fma:     program.n_fma,
            success:       true,
            error:         String::new(),
            cache_hit:     false,
            folded,
            gpu_used,
            phase_results: Vec::new(),
        })
    }

    /// Ejecuta GEMM: C = alpha * A @ B + beta * C
    pub fn run_gemm(
        &self,
        a: &[f64], b: &[f64], c: &mut [f64],
        alpha: f64, beta: f64,
        m: i32, n: i32, k: i32,
    ) -> GideonResult<f64> {
        // Validar dimensiones
        if a.len() != (m * k) as usize {
            return Err(GideonError::DimensionMismatch {
                expected: format!("A: {}×{} = {}", m, k, m*k),
                got:      format!("{}", a.len()),
            });
        }
        if b.len() != (k * n) as usize {
            return Err(GideonError::DimensionMismatch {
                expected: format!("B: {}×{} = {}", k, n, k*n),
                got:      format!("{}", b.len()),
            });
        }
        let t0 = Instant::now();
        unsafe {
            gideon_gemm(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), alpha, beta, m, n, k);
        }
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    }

    /// Ejecuta el plan de fases con paralelismo Rayon.
    pub fn run_plan(
        &self,
        plan: &ExecutionPlan,
        program: &GideonProgram,
    ) -> GideonResult<Vec<PhaseResult>> {
        let dec = self.dispatcher.decide(program, 0);
        self.scheduler.execute_plan(plan, program, &dec.backend)
    }

    /// Telemetría compartida (para exportar calibración ACF).
    pub fn telemetry(&self) -> &GideonTelemetry {
        &self.telemetry
    }

    pub fn dispatcher(&self) -> &GideonDispatcher {
        &self.dispatcher
    }

    // ── Fold cache ─────────────────────────────────────────────────────────

    /// Intenta colapsar la cadena afín a (W_total, B_total).
    /// Usa la caché: si el chain_hash ya fue calculado, devuelve el resultado
    /// guardado sin recomputar.
    fn try_fold(&self, program: &GideonProgram) -> Option<(f64, f64)> {
        let hash = &program.chain_hash;

        // Consultar caché primero (lock mínimo)
        {
            let guard = self.fold_cache.lock();
            if let Some(cached) = guard.get(hash) {
                return *cached;
            }
        }

        // Calcular el fold
        let result = compute_affine_fold(program);

        // Guardar en caché
        {
            let mut guard = self.fold_cache.lock();
            guard.insert(hash.clone(), result);
        }

        result
    }

    /// Número de entradas en la caché de fold.
    pub fn fold_cache_size(&self) -> usize {
        self.fold_cache.lock().len()
    }

    fn load_fold_cache(cache: &FoldCache, path: Option<&Path>) {
        let Some(p) = path else { return };
        let Ok(data) = std::fs::read_to_string(p) else { return };
        let Ok(parsed): Result<HashMap<String, Option<[f64; 2]>>, _> =
            serde_json::from_str(&data)
        else { return };

        let mut guard = cache.lock();
        for (k, v) in parsed {
            guard.insert(k, v.map(|[w, b]| (w, b)));
        }
    }
}

/// Colapsa una cadena de FMAs a una única transformación afín (W, B).
///
/// Para la cadena x → w1*x+b1 → w2*(w1*x+b1)+b2 → ...
/// W_total = w_n * ... * w_1
/// B_total = b_n + w_n*(b_{n-1} + w_{n-1}*(...))
fn compute_affine_fold(program: &GideonProgram) -> Option<(f64, f64)> {
    if !program.is_fully_affine() {
        return None;
    }

    let fma_nodes: Vec<_> = program
        .nodes
        .iter()
        .filter(|n| n.kind.is_affine() && n.params.len() >= 2)
        .collect();

    if fma_nodes.is_empty() {
        return Some((1.0, 0.0)); // identidad
    }

    let mut w = 1.0f64;
    let mut b = 0.0f64;
    for node in &fma_nodes {
        let wi = node.params[0];
        let bi = node.params[1];
        // Composición: y = wi*(w*x + b) + bi = (wi*w)*x + (wi*b + bi)
        b = wi * b + bi;
        w = wi * w;
    }

    Some((w, b))
}

/// Extrae vectores de pesos y biases de los nodos FMA del programa.
fn extract_fma_params(program: &GideonProgram) -> (Vec<f64>, Vec<f64>) {
    let mut weights = Vec::new();
    let mut biases  = Vec::new();
    for node in &program.nodes {
        if node.kind.is_affine() && node.params.len() >= 2 {
            weights.push(node.params[0]);
            biases.push(node.params[1]);
        }
    }
    (weights, biases)
}
