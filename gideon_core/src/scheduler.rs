// scheduler.rs — Scheduler distribuido con Rayon
//
// Este módulo es el que hace posibles los beneficios de Rust sobre Python/C:
//
//   - Rayon usa un thread pool basado en work-stealing. Cada hilo tiene su
//     propia cola local; cuando termina, "roba" trabajo de otros hilos.
//     El resultado: balanceo automático sin programar explícitamente los hilos.
//
//   - El compilador Rust garantiza que los tipos enviados entre hilos
//     implementan `Send + Sync`. Si intentas compartir un tipo no thread-safe
//     (como un Rc<> sin Arc<>), el compilador rechaza el código.
//     En C, este error solo aparece en producción (segfault o data race).
//
//   - `PhaseResult` es `Send` porque todos sus campos son tipos básicos.
//     El compilador verifica esto automáticamente.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::error::{GideonError, GideonResult};
use crate::ir::{GideonProgram, IRNode};
use crate::telemetry::GideonTelemetry;

/// Resultado de ejecutar una fase del grafo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseResult {
    pub phase_id:   usize,
    pub node_ids:   Vec<String>,
    pub elapsed_ms: f64,
    pub backend:    String,
    pub success:    bool,
    pub error:      Option<String>,
}

/// Plan de ejecución: fases ordenadas topológicamente.
/// Las fases del mismo nivel pueden ejecutarse en paralelo.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// phases[i] = lista de nodos que pueden ejecutarse en paralelo.
    pub phases: Vec<Vec<String>>,  // node_ids por fase
}

impl ExecutionPlan {
    /// Construye un plan de ejecución con análisis topológico del programa.
    pub fn from_program(program: &GideonProgram) -> Self {
        // Algoritmo: BFS topológico por niveles.
        // Nodos sin dependencias → nivel 0; sus dependientes → nivel 1; etc.
        use std::collections::{HashMap, VecDeque};

        // Construir mapa de dependencias: node_id → set de inputs
        // Calcular in-degree
        let mut in_degree: HashMap<&str, usize> = program
            .nodes
            .iter()
            .map(|n| (n.node_id.as_str(), n.inputs.len()))
            .collect();

        // Cola inicial: nodos sin dependencias
        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        // Asignar nivel a cada nodo
        let mut level: HashMap<&str, usize> = HashMap::new();
        let mut max_level = 0usize;

        // Construir grafo inverso: quién depende de node_id
        let mut dependents_of: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in &program.nodes {
            for inp in &node.inputs {
                dependents_of.entry(inp.as_str()).or_default().push(node.node_id.as_str());
            }
        }

        while let Some(id) = queue.pop_front() {
            let lv = *level.get(id).unwrap_or(&0);
            max_level = max_level.max(lv);

            if let Some(deps) = dependents_of.get(id) {
                for &dep in deps {
                    let entry = in_degree.get_mut(dep).unwrap();
                    *entry = entry.saturating_sub(1);

                    let new_lv = lv + 1;
                    let current = level.entry(dep).or_insert(0);
                    *current = (*current).max(new_lv);

                    if *entry == 0 {
                        queue.push_back(dep);
                    }
                }
            }
        }

        // Agrupar por nivel
        let mut phases: Vec<Vec<String>> = vec![Vec::new(); max_level + 1];
        for node in &program.nodes {
            let lv = *level.get(node.node_id.as_str()).unwrap_or(&0);
            phases[lv].push(node.node_id.clone());
        }

        // Eliminar fases vacías
        phases.retain(|p| !p.is_empty());

        Self { phases }
    }
}

/// Scheduler de ejecución en paralelo.
///
/// Usa Rayon para ejecutar fases paralelas con work-stealing automático.
/// La `GideonTelemetry` se comparte de forma segura mediante Arc<Mutex>.
pub struct GideonScheduler {
    telemetry:  GideonTelemetry,
    /// Número máximo de hilos Rayon (None = automático = n_cpus).
    max_threads: Option<usize>,
}

impl GideonScheduler {
    pub fn new(telemetry: GideonTelemetry) -> Self {
        Self {
            telemetry,
            max_threads: None,
        }
    }

    pub fn with_threads(mut self, n: usize) -> Self {
        self.max_threads = Some(n);
        self
    }

    /// Ejecuta un plan de ejecución usando paralelismo inter-fase con Rayon.
    ///
    /// Cada fase es una `par_iter` — Rayon distribuye los nodos entre los
    /// hilos del pool automáticamente.
    ///
    /// Garantía de corrección: una fase solo empieza cuando la anterior
    /// termina (`collect()` es una barrera implícita).
    pub fn execute_plan(
        &self,
        plan: &ExecutionPlan,
        program: &GideonProgram,
        backend: &str,
    ) -> GideonResult<Vec<PhaseResult>> {
        // Mapa rápido node_id → IRNode
        let node_map: std::collections::HashMap<&str, &IRNode> = program
            .nodes
            .iter()
            .map(|n| (n.node_id.as_str(), n))
            .collect();

        let pool = self.build_pool()?;

        let mut all_results = Vec::with_capacity(plan.phases.len());

        for (phase_idx, phase_nodes) in plan.phases.iter().enumerate() {
            // Par_iter ejecuta los nodos de esta fase en paralelo
            let phase_results: Vec<PhaseResult> = pool.install(|| {
                phase_nodes
                    .par_iter()
                    .map(|node_id| {
                        let node = node_map.get(node_id.as_str());
                        execute_node_stub(phase_idx, node_id, node, backend)
                    })
                    .collect()
            });

            // Verificar si algún nodo de la fase falló
            if phase_results.iter().any(|r| !r.success) {
                let first_err = phase_results
                    .iter()
                    .find(|r| !r.success)
                    .and_then(|r| r.error.clone())
                    .unwrap_or_else(|| "fase fallida".into());
                return Err(GideonError::ExecutionFailed(
                    format!("fase {phase_idx}: {first_err}")
                ));
            }

            all_results.extend(phase_results);
        }

        Ok(all_results)
    }

    /// Estadísticas de paralelismo de la última ejecución.
    pub fn parallelism_stats(plan: &ExecutionPlan) -> (usize, usize, usize) {
        let total_nodes: usize = plan.phases.iter().map(|p| p.len()).sum();
        let max_width = plan.phases.iter().map(|p| p.len()).max().unwrap_or(0);
        let n_phases  = plan.phases.len();
        (n_phases, total_nodes, max_width)
    }

    fn build_pool(&self) -> GideonResult<rayon::ThreadPool> {
        let mut builder = rayon::ThreadPoolBuilder::new();
        if let Some(n) = self.max_threads {
            builder = builder.num_threads(n);
        }
        builder
            .build()
            .map_err(|e| GideonError::ConcurrencyError(e.to_string()))
    }
}

/// Simula la ejecución de un nodo (el cálculo real lo hace engine.rs).
/// En el futuro, este stub se reemplaza con la llamada al C kernel apropiado.
fn execute_node_stub(
    phase_id: usize,
    node_id: &str,
    node: Option<&&IRNode>,
    backend: &str,
) -> PhaseResult {
    let t0 = Instant::now();

    let success = node.is_some();
    let error   = if success { None } else {
        Some(format!("nodo {node_id} no encontrado en el programa"))
    };

    PhaseResult {
        phase_id,
        node_ids:   vec![node_id.to_string()],
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        backend:    backend.to_string(),
        success,
        error,
    }
}
