"""
rust_bridge.py — Puente Python → Rust Core de Gideon

Importar código Rust desde Python es tan simple como:

    from .rust_bridge import RUST_CORE_AVAILABLE, get_rust_engine

Si `gideon_core` (la extensión Rust compilada con maturin) está disponible:
  - `get_rust_engine()` devuelve un `GideonCoreEngine` Rust
  - Las operaciones FMA, GEMM y el scheduler paralelo usan C + Rayon

Si no está disponible (entorno sin Rust, CI básico, etc.):
  - `RUST_CORE_AVAILABLE = False`
  - Las llamadas caen al motor Python existente (engine.py)

Estrategia de integración:
  GideonEngine (Python/engine.py)
    └── si RUST_CORE_AVAILABLE
          └── delega run_fma() al GideonCoreEngine Rust via esta capa

Ventajas de la delegación selectiva:
  - Cero riesgo de regresión: Python siempre disponible como fallback.
  - Tests existentes (206 hasta v1.2.0) siguen pasando sin cambios.
  - El motor Rust se adopta incrementalmente, módulo por módulo.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ── Importación condicional del módulo Rust ───────────────────────────────────
try:
    import gideon_core as _gc  # type: ignore[import]

    RUST_CORE_AVAILABLE: bool = True
    RUST_CORE_VERSION:   str  = getattr(_gc, "__version__", "unknown")

    # Clases Rust expuestas directamente
    RustCoreEngine:  Any = _gc.GideonCoreEngine
    RustCoreConfig:  Any = _gc.CoreEngineConfig
    RustExecPlan:    Any = _gc.ExecutionPlan

except (ImportError, AttributeError):
    RUST_CORE_AVAILABLE = False
    RUST_CORE_VERSION   = "n/a"
    RustCoreEngine      = None
    RustCoreConfig      = None
    RustExecPlan        = None


# ── Instancia global única (singleton por proceso) ────────────────────────────
_GLOBAL_ENGINE: Optional[Any] = None


def get_rust_engine(
    *,
    precision: str = "fp64",
    fold_affine: bool = True,
    gpu_min_elements: int = 2_000_000,
    use_parallel: bool = True,
    telemetry_path: Optional[str] = None,
    force_new: bool = False,
) -> Optional[Any]:
    """
    Devuelve la instancia global del motor Rust.

    Parametros
    ----------
    precision : str
        "fp64" o "fp32".
    fold_affine : bool
        Si el motor debe intentar colapsar cadenas afines a W·x+B.
    gpu_min_elements : int
        Umbral de elementos para preferir GPU sobre CPU.
    use_parallel : bool
        Si el scheduler Rayon puede usar múltiples hilos.
    telemetry_path : str | None
        Ruta alternativa para la DB de telemetría.
    force_new : bool
        Fuerza la creación de una instancia nueva (descarta la global).

    Returns
    -------
    GideonCoreEngine | None
        Motor Rust listo, o None si no está disponible.
    """
    global _GLOBAL_ENGINE

    if not RUST_CORE_AVAILABLE:
        return None

    if _GLOBAL_ENGINE is not None and not force_new:
        return _GLOBAL_ENGINE

    cfg_kwargs: Dict[str, Any] = {
        "precision":        precision,
        "fold_affine":      fold_affine,
        "gpu_min_elements": gpu_min_elements,
        "use_parallel":     use_parallel,
    }
    if telemetry_path is not None:
        cfg_kwargs["telemetry_path"] = telemetry_path

    try:
        cfg = RustCoreConfig(**cfg_kwargs)
        _GLOBAL_ENGINE = RustCoreEngine(cfg)
    except Exception:
        _GLOBAL_ENGINE = None

    return _GLOBAL_ENGINE


def reset_rust_engine() -> None:
    """Descarta la instancia global (útil en tests)."""
    global _GLOBAL_ENGINE
    _GLOBAL_ENGINE = None


# ── API de alto nivel ─────────────────────────────────────────────────────────

def rust_run_fma(
    weights: List[float],
    biases:  List[float],
    x:       List[float],
    precision: str = "fp64",
) -> Optional[Dict[str, Any]]:
    """
    Ejecuta una cadena FMA con el motor Rust.

    Returns
    -------
    dict con keys: output, elapsed_ms, backend, folded, gpu_used, total_fma, success
    — o None si el motor Rust no está disponible.
    """
    engine = get_rust_engine()
    if engine is None:
        return None
    try:
        return engine.run_fma(weights, biases, x, precision)
    except Exception:
        return None


def rust_run_gemm(
    a: List[float],
    b: List[float],
    c: List[float],
    m: int, n: int, k: int,
    alpha: float = 1.0,
    beta:  float = 0.0,
) -> Optional[Tuple[List[float], float]]:
    """
    Ejecuta GEMM: C = alpha·A@B + beta·C con el motor Rust.

    Returns
    -------
    (C_result, elapsed_ms) — o None si el motor no está disponible.
    """
    engine = get_rust_engine()
    if engine is None:
        return None
    try:
        return engine.run_gemm(a, b, c, m, n, k, alpha, beta)
    except Exception:
        return None


def rust_telemetry_stats() -> Optional[Dict[str, Any]]:
    """
    Estadísticas de telemetría del motor Rust.

    Returns
    -------
    dict con keys: total_records, fold_cache_size, hardware, backends
    — o None si el motor no está disponible.
    """
    engine = get_rust_engine()
    if engine is None:
        return None
    try:
        return engine.telemetry_stats()
    except Exception:
        return None


def rust_export_acf_calibration() -> Optional[Dict[str, Any]]:
    """
    Exporta datos de calibración para el bucle cerrado ACF ← Gideon.

    Retorna el análisis de qué backend domina, tasa de fold, y notas
    para que ACFInvariant ajuste sus parámetros internos.
    """
    engine = get_rust_engine()
    if engine is None:
        return None
    try:
        return engine.export_acf_calibration()
    except Exception:
        return None


def rust_status() -> Dict[str, Any]:
    """
    Resumen del estado del Rust core.

    Returns
    -------
    dict con: available, version, engine_repr
    """
    engine = get_rust_engine()
    return {
        "available":    RUST_CORE_AVAILABLE,
        "version":      RUST_CORE_VERSION,
        "engine_repr":  repr(engine) if engine is not None else None,
    }
