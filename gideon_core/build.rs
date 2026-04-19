// build.rs — Compila los backends numéricos en C durante `cargo build`
//
// Estrategia:
//   1. fma_avx512.c  — kernel FMA con AVX-512 / AVX2 (prefetch + unroll)
//   2. gemm_kernel.c — GEMM micro-kernel con AVX-512 8×4 / AVX2 4×4
//
// Parámetros de optimización desde el perfil de hardware (gideon_autotune v2.0):
//   GIDEON_PREFETCH_DIST — líneas de prefetch (fma_unroll = cpu_prefetch_dist)
//   GIDEON_FMA_UNROLL    — factor de desenrollado del loop FMA
//   GIDEON_MC, GIDEON_KC, GIDEON_NC — tiles GEMM derivados de L1/L2/L3
//   GIDEON_MR, GIDEON_NR             — micro-kernel rows/cols
//
// El perfil se lee de $GIDEON_PROFILE_PATH (JSON generado por GideonHardwareProfiler).
// Si no existe, se usan defaults conservadores que funcionan en cualquier hardware.

use std::env;
use std::path::Path;

fn main() {
    // ── Leer parámetros del perfil de hardware ────────────────────────────────
    let (prefetch_dist, fma_unroll, mc, kc, nc, mr, nr) = read_hw_profile();

    let mut build = cc::Build::new();
    build
        .file("c_backends/fma_avx512.c")
        .file("c_backends/gemm_kernel.c")
        .include("c_backends")
        .opt_level(3)
        .flag_if_supported("-std=c11")
        .flag_if_supported("-ffast-math")
        .flag_if_supported("-funroll-loops")
        .flag_if_supported("-fomit-frame-pointer")
        // Parámetros de optimización derivados del perfil de hardware
        .define("GIDEON_PREFETCH_DIST", Some(prefetch_dist.to_string().as_str()))
        .define("GIDEON_FMA_UNROLL",    Some(fma_unroll.to_string().as_str()))
        .define("GIDEON_MC",            Some(mc.to_string().as_str()))
        .define("GIDEON_KC",            Some(kc.to_string().as_str()))
        .define("GIDEON_NC",            Some(nc.to_string().as_str()))
        .define("GIDEON_MR",            Some(mr.to_string().as_str()))
        .define("GIDEON_NR",            Some(nr.to_string().as_str()));

    // ── AVX-512 — habilitado por feature flag O por detección automática ───────
    let use_avx512 = cfg!(feature = "avx512") || detect_avx512_support();
    if use_avx512 {
        build
            .flag_if_supported("-mavx512f")
            .flag_if_supported("-mavx512dq")
            .flag_if_supported("-mavx512bw")
            .flag_if_supported("-mavx512vl")
            .define("GIDEON_AVX512", None);
        println!("cargo:rustc-cfg=avx512_detected");
    }

    // ── AVX2+FMA3 — siempre habilitado si disponible (Haswell 2013+) ─────────
    build
        .flag_if_supported("-mavx2")
        .flag_if_supported("-mfma")
        .define("GIDEON_AVX2", None);

    build.compile("gideon_c_kernels");

    // Recompilar si cambian las fuentes C o el perfil de hardware
    println!("cargo:rerun-if-changed=c_backends/fma_avx512.c");
    println!("cargo:rerun-if-changed=c_backends/gemm_kernel.c");
    println!("cargo:rerun-if-changed=c_backends/cpu_kernels.h");
    if let Ok(p) = env::var("GIDEON_PROFILE_PATH") {
        println!("cargo:rerun-if-changed={}", p);
    }
}

/// Lee los parámetros de optimización del perfil JSON de GideonHardwareProfiler.
///
/// El perfil es generado por gideon_autotune.py y contiene los valores
/// calculados dinámicamente para el hardware actual.
///
/// Campos leídos (todos tienen defaults conservadores si el perfil no existe):
///   cpu_prefetch_dist → GIDEON_PREFETCH_DIST
///   fma_unroll_factor → GIDEON_FMA_UNROLL
///   gemm_mc           → GIDEON_MC
///   gemm_kc           → GIDEON_KC
///   gemm_nc           → GIDEON_NC
///   gemm_mr           → GIDEON_MR
///   gemm_nr           → GIDEON_NR
fn read_hw_profile() -> (u32, u32, u32, u32, u32, u32, u32) {
    // Defaults seguros para cualquier hardware moderno (x86-64 sin AVX-512)
    let defaults = (8u32, 1u32, 64u32, 64u32, 256u32, 4u32, 4u32);

    // Buscar el perfil en la ruta configurada
    let profile_path = env::var("GIDEON_PROFILE_PATH").unwrap_or_else(|_| {
        // Búsqueda por convención: junto al Cargo.toml o en home
        let cwd = env::current_dir().unwrap_or_default();
        let candidates = [
            cwd.join("hw_profile.json"),
            cwd.parent().map(|p| p.join("hw_profile.json")).unwrap_or_default(),
            Path::new("/tmp/gideon_hw_profile.json").to_path_buf(),
        ];
        candidates
            .iter()
            .find(|p| p.exists())
            .map(|p| p.display().to_string())
            .unwrap_or_default()
    });

    if profile_path.is_empty() || !Path::new(&profile_path).exists() {
        return defaults;
    }

    let Ok(data) = std::fs::read_to_string(&profile_path) else { return defaults };

    // Parser JSON mínimo sin dependencias externas
    // Busca patrones "clave": valor numérico en la cadena JSON plana
    let prefetch_dist = json_find_u32(&data, "cpu_prefetch_dist").unwrap_or(defaults.0);
    let fma_unroll    = json_find_u32(&data, "fma_unroll_factor").unwrap_or(defaults.1);
    let mc            = json_find_u32(&data, "gemm_mc").unwrap_or(defaults.2);
    let kc            = json_find_u32(&data, "gemm_kc").unwrap_or(defaults.3);
    let nc            = json_find_u32(&data, "gemm_nc").unwrap_or(defaults.4);
    let mr            = json_find_u32(&data, "gemm_mr").unwrap_or(defaults.5);
    let nr            = json_find_u32(&data, "gemm_nr").unwrap_or(defaults.6);

    // Validar rangos razonables
    let prefetch_dist = prefetch_dist.max(2).min(32);
    let fma_unroll    = fma_unroll.max(1).min(4);
    let kc            = kc.max(32).min(512);
    let mc            = mc.max(4).min(512);
    let nc            = nc.max(64).min(2048);
    let mr            = mr.max(1).min(16);
    let nr            = nr.max(1).min(16);

    (prefetch_dist, fma_unroll, mc, kc, nc, mr, nr)
}

/// Extrae el primer entero positivo asociado a una clave en un JSON plano.
/// Ejemplo: `"gemm_kc": 256` → Some(256)
fn json_find_u32(json: &str, key: &str) -> Option<u32> {
    let needle = format!("\"{}\":", key);
    let pos    = json.find(&needle)?;
    let after  = json[pos + needle.len()..].trim_start();
    // Leer dígitos hasta el primer no-dígito
    let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().ok()
}

/// Detecta soporte AVX-512 leyendo /proc/cpuinfo en Linux.
/// En Windows y macOS siempre devuelve false (se usa el feature flag "avx512" explícito).
fn detect_avx512_support() -> bool {
    #[cfg(target_os = "linux")]
    {
        let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") else { return false };
        // Buscar "avx512f" en las flags de cualquier CPU
        return info.lines()
            .filter(|l| l.starts_with("flags"))
            .any(|l| l.contains("avx512f"));
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}
