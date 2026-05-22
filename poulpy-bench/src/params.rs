//! JSON-configurable benchmark parameters.
//!
//! Load the active parameter set with [`BenchParams::get`].  All benchmark
//! binaries call this once; a [`std::sync::OnceLock`] ensures the JSON file
//! is only parsed once per process.
//!
//! # Environment variable
//!
//! Set `POULPY_BENCH_PARAMS` to either:
//! - a path to a JSON file (`/path/to/params.json`), or
//! - an inline JSON string (`{"core":{"n":2048}}`).
//!
//! Any field omitted from the JSON falls back to its default value.
//!
//! # Example JSON file
//!
//! ```json
//! {
//!   "hal":  { "sweeps": [[10,2,2],[12,2,8],[14,2,32]] },
//!   "cnv":  { "sweeps": [[10,1],[12,4],[14,16]] },
//!   "vmp":  { "sweeps": [[10,2,1,2,3],[12,7,1,2,8]] },
//!   "svp_prepare": { "log_n_values": [10,12,14] },
//!   "core": { "n": 4096, "base2k": 18, "k": 54, "rank": 1, "dsize": 1 }
//! }
//! ```

use serde::{Deserialize, Serialize};

/// HAL sweep parameters for `vec_znx*`, `vec_znx_dft`, and `svp` benchmarks.
///
/// Each entry is `[log_n, cols, size]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HalSweepParams {
    pub sweeps: Vec<[usize; 3]>,
}

impl Default for HalSweepParams {
    fn default() -> Self {
        Self {
            sweeps: vec![[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]],
        }
    }
}

/// Sweep parameters for convolution benchmarks.
///
/// Each entry is `[log_n, size]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CnvSweepParams {
    pub sweeps: Vec<[usize; 2]>,
}

impl Default for CnvSweepParams {
    fn default() -> Self {
        Self {
            sweeps: vec![[10, 1], [11, 2], [12, 4], [13, 8], [14, 16], [15, 32], [16, 64]],
        }
    }
}

/// Sweep parameters for VMP benchmarks.
///
/// Each entry is `[log_n, rows, cols_in, cols_out, size]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VmpSweepParams {
    pub sweeps: Vec<[usize; 5]>,
}

impl Default for VmpSweepParams {
    fn default() -> Self {
        Self {
            sweeps: vec![
                [10, 2, 1, 2, 3],
                [11, 4, 1, 2, 5],
                [12, 7, 1, 2, 8],
                [13, 15, 1, 2, 16],
                [14, 31, 1, 2, 32],
            ],
        }
    }
}

/// Sweep parameters for `svp_prepare` (just a list of `log_n` values).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SvpPrepareParams {
    pub log_n: Vec<usize>,
}

impl Default for SvpPrepareParams {
    fn default() -> Self {
        Self {
            log_n: vec![10, 11, 12, 13, 14],
        }
    }
}

/// Core GLWE layout parameters used by all core-layer and scheme-layer benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CoreParams {
    pub n: u32,
    pub base2k: u32,
    pub k: u32,
    pub rank: u32,
    pub dsize: u32,
}

impl Default for CoreParams {
    fn default() -> Self {
        Self {
            n: 1 << 12,
            base2k: 18,
            k: 54,
            rank: 1,
            dsize: 1,
        }
    }
}

impl CoreParams {
    pub fn dnum(&self) -> u32 {
        self.k.div_ceil(self.dsize * self.base2k)
    }
}

/// Top-level container for all configurable benchmark parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchParams {
    /// Backend labels to benchmark (used by the shell wrapper script).
    ///
    /// Available: `fft64-ref`, `ntt120-ref`, `fft64-avx`, `ntt120-avx`,
    /// `fft64-avx512`, `ntt120-avx512`, `ntt-ifma`.
    /// If any AVX backend is listed, `--features enable-avx` is added
    /// automatically; if any AVX-512F backend is listed, `--features
    /// enable-avx512f` is added automatically; if any IFMA backend is listed,
    /// `--features enable-ifma` is added automatically. Omit or leave empty to
    /// run all compiled-in backends.
    #[serde(default)]
    pub backends: Vec<String>,
    /// List of bench binaries to run (used by the shell wrapper script).
    ///
    /// When empty or absent, the script runs its built-in default set.
    /// Available names: `vec_znx`, `vec_znx_big`, `vec_znx_dft`, `convolution`,
    /// `svp`, `vmp`, `fft`, `ntt`, `operations`, `encryption`, `decryption`,
    /// `glwe_tensor`,
    /// `automorphism`, `external_product`, `keyswitch`,
    /// `blind_rotate`, `circuit_bootstrapping`, `bdd_prepare`, `bdd_arithmetic`,
    /// `ckks_leveled`, `standard`.
    #[serde(default)]
    pub run: Vec<String>,
    #[serde(default)]
    pub hal: HalSweepParams,
    #[serde(default)]
    pub cnv: CnvSweepParams,
    #[serde(default)]
    pub vmp: VmpSweepParams,
    #[serde(default)]
    pub svp_prepare: SvpPrepareParams,
    #[serde(default)]
    pub core: CoreParams,
}

impl BenchParams {
    /// Return the process-wide parameter set, loading it on first call.
    ///
    /// Reads `POULPY_BENCH_PARAMS` as a JSON file path or inline JSON string.
    /// Falls back silently to [`Default`] if the variable is unset or the
    /// content cannot be parsed (a warning is printed to stderr in that case).
    pub fn get() -> &'static Self {
        static PARAMS: std::sync::OnceLock<BenchParams> = std::sync::OnceLock::new();
        PARAMS.get_or_init(Self::load)
    }

    fn load() -> Self {
        let Ok(val) = std::env::var("POULPY_BENCH_PARAMS") else {
            return Self::default();
        };
        // Try as a file path first, then as inline JSON.
        let json = std::fs::read_to_string(&val).unwrap_or(val.clone());
        serde_json::from_str(&json).unwrap_or_else(|e| {
            eprintln!("POULPY_BENCH_PARAMS: failed to parse: {e}");
            Self::default()
        })
    }
}
