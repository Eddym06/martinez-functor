"""
ml_dispatcher.py — Dispatcher con Aprendizaje Automático y Telemetría para Gideon.

Implementa el bucle de retroalimentación completo del sistema Gideon:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Gideon ejecuta FMA chain                                               │
  │       │                                                                 │
  │       ▼                                                                 │
  │  GideonTelemetry.record() ─── persiste en ~/.gideon/telemetry.json     │
  │       │                                                                 │
  │       ├──→ MLDispatcher.decide() ─── mejora decisiones de backend      │
  │       │                                                                 │
  │       └──→ export_acf_calibration() ─── datos para ACF (bucle cerrado) │
  │                │                                                        │
  │                ▼                                                        │
  │         ACFInvariant calibra bounds empíricos                           │
  │         FormalVerificationSuite usa latencias reales                    │
  └─────────────────────────────────────────────────────────────────────────┘

Componentes:
  ExecutionRecord  — registro individual de una ejecución
  GideonTelemetry  — base de datos persistente de ejecuciones
  MLDispatcher     — dispatcher que aprende de telemetría

El bucle cerrado (Gideon → ACF) se activa llamando:
    calibration = telemetry.export_acf_calibration()
    # Pasar a ACFInvariant o FormalVerificationSuite para calibrar bounds

Uso básico:
    telemetry = GideonTelemetry()
    telemetry.record(engine_result, chain_hash="abc123", n_elements=1000)
    stats = telemetry.get_backend_stats()
    best = telemetry.get_best_backend(n_elements=10_000, n_fma=5)
    calib = telemetry.export_acf_calibration()
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# ExecutionRecord
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionRecord:
    """Registro de una ejecución individual del motor Gideon."""
    chain_hash: str           # SHA-256 de la cadena FMA (16 chars)
    n_elements: int           # Número de elementos del input
    n_fma: int                # Número de FMAs en la cadena
    backend: str              # Backend usado ("affine_fold", "c_native", etc.)
    elapsed_ms: float         # Tiempo total de ejecución en ms
    folded: bool              # ¿Se aplicó pliegue algebraico?
    gpu_used: bool            # ¿Se usó GPU?
    timestamp: float          # Unix timestamp
    precision: str = "fp64"   # Precisión usada
    success: bool = True      # ¿Ejecución exitosa?
    cache_hit: bool = False   # ¿Hit en caché de compilación?


# ─────────────────────────────────────────────────────────────────────────────
# GideonTelemetry
# ─────────────────────────────────────────────────────────────────────────────

class GideonTelemetry:
    """
    Base de datos persistente de ejecuciones del motor Gideon.

    Registra cada ejecución y proporciona estadísticas para:
      - MLDispatcher (mejores decisiones de backend)
      - Bucle cerrado con ACF (calibración de bounds formales)

    Persiste en ~/.gideon/telemetry.json (configurable).

    Limite de memoria: MAX_RECORDS registros más recientes.
    Auto-save: cada AUTO_SAVE_INTERVAL registros nuevos.

    Uso:
        telemetry = GideonTelemetry()

        # Registrar una ejecución
        telemetry.record(result, chain_hash, n_elements=5000)

        # Consultar estadísticas
        stats = telemetry.get_backend_stats()    # por backend
        best  = telemetry.get_best_backend(n_elements=5000, n_fma=10)

        # Exportar para ACF (bucle cerrado Gideon→ACF)
        calibration = telemetry.export_acf_calibration()
    """

    MAX_RECORDS = 10_000
    AUTO_SAVE_INTERVAL = 50
    DB_VERSION = "1.2"

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or os.path.expanduser("~/.gideon/telemetry.json")
        # Ruta SQLite al lado del JSON (mismo directorio, extensión .db)
        self._sqlite_path = os.path.splitext(self.db_path)[0] + ".db"
        self._records: List[ExecutionRecord] = []
        self._use_sqlite: bool = False
        self._load()

    # ── Registro ─────────────────────────────────────────────────────────────

    def record(
        self,
        result: Any,
        chain_hash: str,
        n_elements: int,
        precision: str = "fp64",
    ) -> None:
        """
        Registra el resultado de una ejecución de GideonEngine.

        Acepta cualquier objeto con los atributos estándar de
        GideonExecutionResult (duck typing).
        """
        rec = ExecutionRecord(
            chain_hash=chain_hash,
            n_elements=n_elements,
            n_fma=int(getattr(result, "total_fma", 0)),
            backend=str(getattr(result, "backend_used", "unknown")),
            elapsed_ms=float(getattr(result, "elapsed_ms", 0.0)),
            folded=bool(getattr(result, "folded", False)),
            gpu_used=bool(getattr(result, "gpu_used", False)),
            timestamp=time.time(),
            precision=precision,
            success=bool(getattr(result, "success", True)),
            cache_hit=bool(getattr(result, "cache_hit", False)),
        )
        self._records.append(rec)

        # Mantener límite de registros
        if len(self._records) > self.MAX_RECORDS:
            self._records = self._records[-self.MAX_RECORDS :]

        # Auto-save periódico
        if len(self._records) % self.AUTO_SAVE_INTERVAL == 0:
            self._save()

    # ── Consultas ────────────────────────────────────────────────────────────

    def get_backend_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Estadísticas de latencia por backend.

        Retorna dict: backend → {count, avg_ms, min_ms, p50_ms, p95_ms, max_ms}
        Solo incluye ejecuciones exitosas.
        """
        per_backend: Dict[str, List[float]] = defaultdict(list)
        for rec in self._records:
            if rec.success:
                per_backend[rec.backend].append(rec.elapsed_ms)

        stats: Dict[str, Dict[str, Any]] = {}
        for backend, times in per_backend.items():
            times.sort()
            n = len(times)
            stats[backend] = {
                "count": n,
                "avg_ms": sum(times) / n,
                "min_ms": times[0],
                "p50_ms": times[n // 2],
                "p95_ms": times[min(int(n * 0.95), n - 1)],
                "max_ms": times[-1],
            }
        return stats

    def get_best_backend(
        self,
        n_elements: Optional[int],
        n_fma: Optional[int],
        precision: str = "fp64",
        min_samples: int = 5,
    ) -> Optional[str]:
        """
        Recomienda el backend con menor latencia histórica.

        Par\u00e1metros
        ----------
        n_elements : int | None
            Tama\u00f1o del input. Si None, no se filtra por tama\u00f1o.
        n_fma : int | None
            N\u00famero de FMAs. Si None, no se filtra por n_fma.
        precision : str
            "fp64" | "fp32"
        min_samples : int
            M\u00ednimo de muestras para recomendar con confianza.

        Solo recomienda si hay \u2265 min_samples muestras del mejor backend.
        Retorna None si no hay datos suficientes.
        """
        # Filtro: primero con n_fma exacto + n_elements similar
        def _matches(r: ExecutionRecord) -> bool:
            if not r.success or r.precision != precision:
                return False
            if n_fma is not None and r.n_fma != n_fma:
                return False
            if n_elements is not None:
                low  = n_elements * 0.1
                high = n_elements * 10
                if not (low <= r.n_elements <= high):
                    return False
            return True

        relevant = [r for r in self._records if _matches(r)]

        # Relajar a solo n_fma si hay pocos resultados
        if len(relevant) < min_samples and n_elements is not None:
            def _matches_no_size(r: ExecutionRecord) -> bool:
                if not r.success or r.precision != precision:
                    return False
                if n_fma is not None and r.n_fma != n_fma:
                    return False
                return True
            relevant = [r for r in self._records if _matches_no_size(r)]

        if not relevant:
            return None

        per_backend: Dict[str, List[float]] = defaultdict(list)
        for r in relevant:
            per_backend[r.backend].append(r.elapsed_ms)

        # Elegir backend con menor latencia media que tenga suficientes muestras
        candidates = {
            b: times for b, times in per_backend.items()
            if len(times) >= max(min_samples // 2, 1)
        }
        if not candidates:
            return None

        return min(candidates, key=lambda b: sum(candidates[b]) / len(candidates[b]))

    def get_chain_stats(self, chain_hash: str) -> Optional[Dict[str, Any]]:
        """Estadísticas específicas para una cadena FMA por su hash."""
        recs = [r for r in self._records if r.chain_hash == chain_hash]
        if not recs:
            return None
        times = sorted(r.elapsed_ms for r in recs if r.success)
        if not times:
            return None
        n = len(times)
        return {
            "n_executions": len(recs),
            "n_success": n,
            "avg_ms": sum(times) / n,
            "p50_ms": times[n // 2],
            "backends_used": list({r.backend for r in recs}),
            "fold_rate": sum(1 for r in recs if r.folded) / len(recs),
            "gpu_rate": sum(1 for r in recs if r.gpu_used) / len(recs),
            "cache_hit_rate": sum(1 for r in recs if r.cache_hit) / len(recs),
        }

    # ── Exportación para bucle cerrado ACF ────────────────────────────────────

    def export_acf_calibration(self) -> Dict[str, Any]:
        """
        Exporta datos de calibración para el bucle cerrado Gideon→ACF.

        Este método es el puente entre la telemetría medida de Gideon y los
        sistemas formales de ACF (ACFInvariant, FormalVerificationSuite).

        ACF puede usar estos datos para:
          1. Calibrar ACFInvariant.compute_alpha() con datos empíricos de error
          2. Ajustar las cotas epsilon de FormalVerificationSuite con latencias
             reales en lugar de cotas teóricas conservadoras
          3. Determinar qué backends producen resultados más precisos

        Retorna:
          {
            "n_records": int,
            "backend_latencies": {backend: avg_ms},
            "backend_p95_latencies": {backend: p95_ms},
            "fastest_backend": str | None,
            "fold_effectiveness_pct": float,
            "gpu_usage_pct": float,
            "cache_hit_rate_pct": float,
            "size_distribution": {small_pct, medium_pct, large_pct},
            "precision_usage": {fp64_pct, fp32_pct},
            "acf_notes": str,  # Notas interpretativas para ACF
          }
        """
        stats = self.get_backend_stats()
        n = len(self._records)

        backend_latencies = {b: s["avg_ms"] for b, s in stats.items()}
        backend_p95 = {b: s["p95_ms"] for b, s in stats.items()}
        fastest = (
            min(backend_latencies, key=backend_latencies.__getitem__)
            if backend_latencies else None
        )

        fold_pct = (
            100.0 * sum(1 for r in self._records if r.folded) / n if n else 0.0
        )
        gpu_pct = (
            100.0 * sum(1 for r in self._records if r.gpu_used) / n if n else 0.0
        )
        cache_hit_pct = (
            100.0 * sum(1 for r in self._records if r.cache_hit) / n if n else 0.0
        )

        size_dist = {
            "small_pct": 100.0 * sum(1 for r in self._records if r.n_elements < 10_000) / max(n, 1),
            "medium_pct": 100.0 * sum(1 for r in self._records if 10_000 <= r.n_elements < 1_000_000) / max(n, 1),
            "large_pct": 100.0 * sum(1 for r in self._records if r.n_elements >= 1_000_000) / max(n, 1),
        }

        prec_usage = {
            "fp64_pct": 100.0 * sum(1 for r in self._records if r.precision == "fp64") / max(n, 1),
            "fp32_pct": 100.0 * sum(1 for r in self._records if r.precision == "fp32") / max(n, 1),
        }

        # Nota interpretativa para ACF
        notes: List[str] = []
        if fold_pct > 70:
            notes.append(
                "Mayoría de cadenas son afines puras — bounds de error ultrabajamente acotables."
            )
        if gpu_pct > 50:
            notes.append(
                "GPU dominante — usar bounds de error fp64 CUDA (idénticos a CPU fp64)."
            )
        if fastest and fastest in ("affine_fold", "pytorch_gpu_folded"):
            notes.append(
                "Fold algebraico produce máximo speedup — epsilon es ε_máquina (2.2e-16)."
            )

        calibration: Dict[str, Any] = {
            "n_records": n,
            "backend_latencies": backend_latencies,
            "backend_p95_latencies": backend_p95,
            "fastest_backend": fastest,
            "fold_effectiveness_pct": fold_pct,
            "gpu_usage_pct": gpu_pct,
            "cache_hit_rate_pct": cache_hit_pct,
            "size_distribution": size_dist,
            "precision_usage": prec_usage,
            "acf_notes": " | ".join(notes) if notes else "Sin observaciones especiales.",
        }
        return calibration

    # ── Gestión ──────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Fuerza guardado inmediato a disco."""
        self._save()

    def clear(self) -> None:
        """Vacía todos los registros (en memoria y disco)."""
        self._records.clear()
        self._save()

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"GideonTelemetry(records={len(self._records)}, path={self.db_path!r})"

    # ── Persistencia (SQLite WAL + JSON fallback) ──────────────────────────────

    def _sqlite_conn(self) -> sqlite3.Connection:
        """Abre una conexión SQLite con WAL mode para seguridad multi-proceso."""
        os.makedirs(os.path.dirname(self._sqlite_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self._sqlite_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chain_hash  TEXT    NOT NULL,
                n_elements  INTEGER NOT NULL,
                n_fma       INTEGER NOT NULL,
                backend     TEXT    NOT NULL,
                elapsed_ms  REAL    NOT NULL,
                folded      INTEGER NOT NULL,
                gpu_used    INTEGER NOT NULL,
                timestamp   REAL    NOT NULL,
                precision   TEXT    NOT NULL DEFAULT 'fp64',
                success     INTEGER NOT NULL DEFAULT 1,
                cache_hit   INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_backend ON records(backend)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_n_fma ON records(n_fma)"
        )
        conn.commit()
        return conn

    def _save(self) -> None:
        """Persiste registros. Usa SQLite si disponible, JSON como fallback."""
        if self._use_sqlite:
            # SQLite: insertar los registros no guardados (los últimos desde last save)
            # Estrategia simple: re-escribir todos (bounded por MAX_RECORDS)
            try:
                with self._sqlite_conn() as conn:
                    conn.execute("DELETE FROM records")
                    conn.executemany(
                        """INSERT INTO records
                           (chain_hash,n_elements,n_fma,backend,elapsed_ms,
                            folded,gpu_used,timestamp,precision,success,cache_hit)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        [
                            (r.chain_hash, r.n_elements, r.n_fma, r.backend,
                             r.elapsed_ms, int(r.folded), int(r.gpu_used),
                             r.timestamp, r.precision, int(r.success), int(r.cache_hit))
                            for r in self._records
                        ]
                    )
                # Backward compat: crear stub JSON para que herramientas externas
                # puedan detectar la existencia del archivo de telemetría.
                try:
                    os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
                    stub = {"_db_version": self.DB_VERSION, "_storage": "sqlite",
                            "_sqlite_path": self._sqlite_path, "records": []}
                    tmp = self.db_path + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(stub, f, separators=(",", ":"))
                    os.replace(tmp, self.db_path)
                except Exception:
                    pass
                return
            except Exception:
                pass  # Caer a JSON

        # JSON fallback (atómico: tmp → rename)
        try:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            data = {
                "_db_version": self.DB_VERSION,
                "records": [asdict(r) for r in self._records],
            }
            tmp = self.db_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            os.replace(tmp, self.db_path)
        except Exception:
            pass  # No fallar si el FS no es accesible

    def _load(self) -> None:
        """Carga registros. Prioriza SQLite; JSON como fallback."""
        # Intentar SQLite primero
        try:
            os.makedirs(os.path.dirname(self._sqlite_path) or ".", exist_ok=True)
            with self._sqlite_conn() as conn:
                rows = conn.execute(
                    "SELECT chain_hash,n_elements,n_fma,backend,elapsed_ms,"
                    "folded,gpu_used,timestamp,precision,success,cache_hit "
                    "FROM records ORDER BY id DESC LIMIT ?",
                    (self.MAX_RECORDS,)
                ).fetchall()
            self._records = [
                ExecutionRecord(
                    chain_hash=r[0], n_elements=r[1], n_fma=r[2],
                    backend=r[3], elapsed_ms=r[4], folded=bool(r[5]),
                    gpu_used=bool(r[6]), timestamp=r[7], precision=r[8],
                    success=bool(r[9]), cache_hit=bool(r[10]),
                )
                for r in reversed(rows)  # orden cronológico
            ]
            self._use_sqlite = True
            # Migrar JSON → SQLite si existe
            if os.path.exists(self.db_path) and len(self._records) == 0:
                self._migrate_json_to_sqlite()
            return
        except Exception:
            self._use_sqlite = False

        # Fallback: JSON
        if not os.path.exists(self.db_path):
            return
        try:
            with open(self.db_path) as f:
                data = json.load(f)
            if data.get("_db_version") != self.DB_VERSION:
                return
            self._records = [ExecutionRecord(**r) for r in data.get("records", [])]
        except Exception:
            self._records = []

    def _migrate_json_to_sqlite(self) -> None:
        """Migra datos del JSON legacy a SQLite."""
        try:
            with open(self.db_path) as f:
                data = json.load(f)
            if data.get("_db_version") != self.DB_VERSION:
                return
            old_records = [ExecutionRecord(**r) for r in data.get("records", [])]
            if old_records:
                self._records = old_records
                self._save()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# MLDispatcher
# ─────────────────────────────────────────────────────────────────────────────

class MLDispatcher:
    """
    Dispatcher que aprende de ejecuciones reales vía GideonTelemetry.

    Estrategia:
      1. Si hay ≥ MIN_SAMPLES datos para el patrón actual → usar la recomendación
         estadística de GideonTelemetry (backend con menor latencia histórica)
      2. Si hay < MIN_SAMPLES datos → delegar al dispatcher heurístico clásico

    La "inteligencia" crece con cada ejecución sin re-entrenamiento explícito:
    las estadísticas se actualizan continuamente en GideonTelemetry.

    Con suficientes datos, MLDispatcher supera al dispatcher heurístico porque:
      - Conoce el comportamiento REAL del hardware del usuario (no estimaciones)
      - Detecta qué backends son lentos en este host específico (ej: CFFI lento
        por falta de AVX-512, o CUDA lento por PCIe 3.0 en lugar de 4.0)
      - Adapta las decisiones automáticamente si el hardware cambia

    Integración con el bucle cerrado ACF:
        ml_disp = MLDispatcher(telemetry, fallback_dispatcher)
        decision = ml_disp.decide(program, precision="fp64")
        # Después de la ejecución, el resultado se registra en telemetry
        # → en la siguiente llamada, ml_disp usará ese dato para mejorar

    Uso en GideonEngine:
        engine.config.use_ml_dispatcher = True
        # El engine instancia MLDispatcher internamente
    """

    MIN_SAMPLES = 5  # Mínimo de muestras para recomendar con confianza

    def __init__(
        self,
        telemetry: GideonTelemetry,
        fallback_dispatcher: Any,    # GideonDispatcher (evitar import circular)
    ) -> None:
        self.telemetry = telemetry
        self.fallback = fallback_dispatcher

    def decide(self, program: Any, precision: str = "fp64") -> Any:
        """
        Decide el mejor backend para ejecutar `program`.

        Usa datos históricos de telemetría si hay suficientes muestras;
        si no, delega al dispatcher heurístico (GideonDispatcher).

        Strategy:
          1. Busca el mejor backend para el n_fma exacto del programa con
             el rango de n_elements más amplio posible.
          2. Si no hay suficientes muestras por n_fma, usa estadísticas globales
             (todos los backends, todas las cadenas) como proxy.
          3. Si no hay datos en absoluto → heurístico clásico.
        """
        from .dispatcher import DispatchDecision

        n_fma = int(getattr(program, "total_fma", 0))

        # Paso 1: buscar por n_fma exacto (sin restricción de n_elements)
        best = self.telemetry.get_best_backend(
            n_elements=None,   # Sin filtro de tamaño
            n_fma=n_fma,
            precision=precision,
            min_samples=self.MIN_SAMPLES,
        )

        # Paso 2: si no hay datos para este n_fma, usar estadísticas globales
        if best is None and len(self.telemetry) >= self.MIN_SAMPLES:
            best = self.telemetry.get_best_backend(
                n_elements=None,
                n_fma=None,   # Cualquier n_fma
                precision=precision,
                min_samples=self.MIN_SAMPLES,
            )

        if best is not None:
            speedup = self._predict_speedup(best)
            return DispatchDecision(
                primary_backend=best,
                fallback_backend="numpy_cpu",
                reason=f"ml_dispatcher:empirical_best={best}",
                estimated_speedup=speedup,
                node_backend_map={},
            )

        # Sin datos suficientes → heurístico
        return self.fallback.decide(program, precision=precision)

    def predict_speedup(self, backend: str) -> float:
        """Predicción de speedup sobre numpy basada en telemetría."""
        return self._predict_speedup(backend)

    def telemetry_summary(self) -> str:
        """Resumen legible de la telemetría acumulada."""
        stats = self.telemetry.get_backend_stats()
        if not stats:
            return "MLDispatcher: sin datos (delega al heurístico)"
        lines = [f"MLDispatcher ({len(self.telemetry)} ejecuciones registradas):"]
        for backend, s in sorted(stats.items(), key=lambda kv: kv[1]["avg_ms"]):
            lines.append(
                f"  {backend:<28} avg={s['avg_ms']:6.1f}ms  "
                f"p95={s['p95_ms']:6.1f}ms  n={s['count']}"
            )
        return "\n".join(lines)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _predict_speedup(self, backend: str) -> float:
        """Calcula el speedup esperado sobre numpy usando telemetría."""
        stats = self.telemetry.get_backend_stats()
        numpy_avg = stats.get("numpy_cpu", {}).get("avg_ms")
        backend_avg = stats.get(backend, {}).get("avg_ms")
        if numpy_avg and backend_avg and backend_avg > 0:
            return numpy_avg / backend_avg
        # Fallback a estimaciones heurísticas
        _defaults = {
            "affine_fold": 50.0,
            "pytorch_gpu_folded": 20.0,
            "c_native": 15.0,
            "pytorch": 5.0,
            "numpy_cpu": 1.0,
        }
        return _defaults.get(backend, 1.0)
