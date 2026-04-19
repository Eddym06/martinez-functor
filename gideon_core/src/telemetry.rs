// telemetry.rs — Telemetría persistente thread-safe
//
// La ventaja de Rust aquí:
//   - `records` está envuelto en Arc<Mutex<Vec<...>>>:
//     múltiples hilos pueden tener una referencia Arc (conteo de referencias
//     atómico), pero solo uno puede tener el Mutex abierto a la vez.
//   - El compilador garantiza que NO se puede acceder a `records.data`
//     sin haber adquirido el lock. En C, esto es responsabilidad del programador.
//   - `flush()` libera el lock ANTES de la operación de disco (I/O lenta),
//     evitando bloquear otros hilos durante la escritura.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{GideonError, GideonResult};

const MAX_RECORDS: usize = 10_000;
const AUTO_SAVE_INTERVAL: usize = 50;
const DB_VERSION: &str = "1.2";

/// Un registro de una ejecución pasada.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub chain_hash:  String,
    pub n_elements:  usize,
    pub n_fma:       usize,
    pub backend:     String,
    pub elapsed_ms:  f64,
    pub folded:      bool,
    pub gpu_used:    bool,
    pub timestamp:   f64,   // UNIX timestamp en segundos
    pub precision:   String,
    pub success:     bool,
    pub cache_hit:   bool,
}

impl ExecutionRecord {
    pub fn now() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
}

/// Estadísticas por backend (computable sin lock prolongado).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendStats {
    pub count:    usize,
    pub avg_ms:   f64,
    pub min_ms:   f64,
    pub p50_ms:   f64,
    pub p95_ms:   f64,
    pub max_ms:   f64,
}

impl BackendStats {
    fn from_latencies(mut lats: Vec<f64>) -> Self {
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = lats.len();
        let p95_idx = ((n as f64 * 0.95) as usize).min(n.saturating_sub(1));
        Self {
            count:  n,
            avg_ms: lats.iter().sum::<f64>() / n as f64,
            min_ms: lats[0],
            p50_ms: lats[n / 2],
            p95_ms: lats[p95_idx],
            max_ms: lats[n - 1],
        }
    }
}

/// Base de datos de telemetría compartible entre hilos.
///
/// Internamente usa `Arc<Mutex<TelemetryInner>>` para que múltiples
/// componentes del scheduler puedan compartirla sin clonarla.
#[derive(Clone)]
pub struct GideonTelemetry {
    inner:   Arc<Mutex<TelemetryInner>>,
    db_path: Option<PathBuf>,
}

struct TelemetryInner {
    records:       Vec<ExecutionRecord>,
    records_since_save: usize,
    version:       String,
}

impl GideonTelemetry {
    /// Crea una telemetría nueva, cargando desde disco si existe.
    pub fn new(db_path: Option<impl AsRef<Path>>) -> Self {
        let path = db_path.map(|p| p.as_ref().to_path_buf());
        let inner = Self::load_from_disk(path.as_deref()).unwrap_or_else(|_| TelemetryInner {
            records:             Vec::new(),
            records_since_save:  0,
            version:             DB_VERSION.into(),
        });
        Self {
            inner:   Arc::new(Mutex::new(inner)),
            db_path: path,
        }
    }

    /// Registra una nueva ejecución.
    ///
    /// Thread-safe: el lock se adquiere minimamente (solo para push + check).
    /// Si se alcanza AUTO_SAVE_INTERVAL, el flush se hace FUERA del lock.
    pub fn record(&self, rec: ExecutionRecord) {
        let should_flush = {
            let mut guard = self.inner.lock();
            // Ventana deslizante: descartar el más antiguo si excede MAX_RECORDS
            if guard.records.len() >= MAX_RECORDS {
                guard.records.remove(0);
            }
            guard.records.push(rec);
            guard.records_since_save += 1;
            guard.records_since_save >= AUTO_SAVE_INTERVAL
        };
        // Lock liberado aquí — el flush de disco no bloquea a otros hilos
        if should_flush {
            let _ = self.flush();
        }
    }

    /// Estadísticas agrupadas por backend.
    pub fn backend_stats(&self) -> HashMap<String, BackendStats> {
        let guard = self.inner.lock();
        let mut by_backend: HashMap<String, Vec<f64>> = HashMap::new();
        for rec in guard.records.iter().filter(|r| r.success) {
            by_backend.entry(rec.backend.clone()).or_default().push(rec.elapsed_ms);
        }
        by_backend
            .into_iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, v)| (k, BackendStats::from_latencies(v)))
            .collect()
    }

    /// Recomienda el mejor backend para un workload similar al dado.
    ///
    /// Filtra por: mismo `n_fma`, `n_elements` dentro de ±10×, mismo `precision`.
    /// Requiere al menos `min_samples` registros para dar recomendación.
    pub fn get_best_backend(
        &self,
        n_elements: usize,
        n_fma: usize,
        precision: &str,
        min_samples: usize,
    ) -> Option<String> {
        let guard = self.inner.lock();

        // Filtro estricto: mismo n_fma + tamaño similar
        let lo = (n_elements / 10).max(1);
        let hi = n_elements * 10;

        let mut by_backend: HashMap<String, Vec<f64>> = HashMap::new();
        for rec in guard.records.iter().filter(|r| {
            r.success
                && r.n_fma == n_fma
                && r.n_elements >= lo
                && r.n_elements <= hi
                && r.precision == precision
        }) {
            by_backend.entry(rec.backend.clone()).or_default().push(rec.elapsed_ms);
        }

        // Si no hay suficientes datos con filtro estricto, relajar a solo n_fma
        if by_backend.values().all(|v| v.len() < min_samples) {
            by_backend.clear();
            for rec in guard.records.iter().filter(|r| {
                r.success && r.n_fma == n_fma && r.precision == precision
            }) {
                by_backend.entry(rec.backend.clone()).or_default().push(rec.elapsed_ms);
            }
        }

        // Elegir el backend con menor media de latencia
        by_backend
            .into_iter()
            .filter(|(_, v)| v.len() >= min_samples)
            .min_by(|(_, a), (_, b)| {
                let avg_a = a.iter().sum::<f64>() / a.len() as f64;
                let avg_b = b.iter().sum::<f64>() / b.len() as f64;
                avg_a.partial_cmp(&avg_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k)
    }

    /// Número total de registros.
    pub fn len(&self) -> usize {
        self.inner.lock().records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Persiste la telemetría en disco atómicamente (.tmp → rename).
    pub fn flush(&self) -> GideonResult<()> {
        let Some(ref path) = self.db_path else {
            return Ok(());
        };

        // Copiar los datos con el lock mínimo
        let snapshot = {
            let mut guard = self.inner.lock();
            guard.records_since_save = 0;
            guard.records.clone()
        };

        // Escritura atómica: primero a .tmp, luego rename
        let parent = path.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;

        let tmp = path.with_extension("json.tmp");
        let db = serde_json::json!({
            "version": DB_VERSION,
            "records": snapshot,
        });
        std::fs::write(&tmp, serde_json::to_string_pretty(&db)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Exporta calibración para el bucle ACF → Poema → Gideon.
    pub fn export_acf_calibration(&self) -> serde_json::Value {
        let stats    = self.backend_stats();
        let n_total  = self.len();
        let guard    = self.inner.lock();

        // Calcular métricas globales
        let n_folded = guard.records.iter().filter(|r| r.folded).count();
        let n_gpu    = guard.records.iter().filter(|r| r.gpu_used).count();
        let n_cache  = guard.records.iter().filter(|r| r.cache_hit).count();
        let n_ok     = guard.records.iter().filter(|r| r.success).count();

        let fold_pct  = if n_ok > 0 { n_folded as f64 / n_ok as f64 * 100.0 } else { 0.0 };
        let gpu_pct   = if n_ok > 0 { n_gpu    as f64 / n_ok as f64 * 100.0 } else { 0.0 };
        let cache_pct = if n_ok > 0 { n_cache  as f64 / n_ok as f64 * 100.0 } else { 0.0 };

        // Backend con menor latencia media
        let fastest = stats.iter()
            .min_by(|(_, a), (_, b)| {
                a.avg_ms.partial_cmp(&b.avg_ms).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone())
            .unwrap_or_default();

        drop(guard); // liberar lock antes de construir JSON

        let backend_latencies: serde_json::Value = stats.iter()
            .map(|(k, v)| (k.clone(), serde_json::json!(v.avg_ms)))
            .collect::<serde_json::Map<_, _>>()
            .into();

        let acf_notes = if n_total < 10 {
            "Insuficientes datos: usar configuración por defecto.".into()
        } else if fold_pct > 80.0 {
            format!("Alta efectividad de fold ({fold_pct:.0}%): aumentar agresividad de fusión afín.")
        } else if gpu_pct > 50.0 {
            format!("GPU dominante ({gpu_pct:.0}%): considerar offload temprano.")
        } else {
            format!("Backend óptimo: {fastest}. Fold: {fold_pct:.0}%, GPU: {gpu_pct:.0}%.")
        };

        serde_json::json!({
            "n_records":              n_total,
            "backend_latencies":      backend_latencies,
            "fastest_backend":        fastest,
            "fold_effectiveness_pct": fold_pct,
            "gpu_usage_pct":          gpu_pct,
            "cache_hit_rate_pct":     cache_pct,
            "acf_notes":              acf_notes,
        })
    }

    /// Carga desde disco, valida la versión de esquema.
    fn load_from_disk(path: Option<&Path>) -> GideonResult<TelemetryInner> {
        let p = path.ok_or_else(|| GideonError::Io("no path".into()))?;
        let data = std::fs::read_to_string(p)?;
        let parsed: serde_json::Value = serde_json::from_str(&data)?;

        // Validar versión del schema
        let ver = parsed["version"].as_str().unwrap_or("");
        if ver != DB_VERSION {
            return Err(GideonError::Deserialize(
                format!("telemetry schema v{ver} != expected v{DB_VERSION}")
            ));
        }

        let records: Vec<ExecutionRecord> =
            serde_json::from_value(parsed["records"].clone())?;

        Ok(TelemetryInner {
            records,
            records_since_save: 0,
            version: ver.into(),
        })
    }
}
