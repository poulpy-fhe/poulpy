use std::fmt::Display;

use poulpy_ckks::SlotsKind;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct HalSweepParms {
    pub n: usize,
    pub cols: usize,
    pub size: usize,
}

impl Display for HalSweepParms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x({}x{})", self.n, self.cols, self.size)
    }
}

pub struct VmpSweepParms {
    pub n: usize,
    pub rows: usize,
    pub cols_in: usize,
    pub cols_out: usize,
    pub size: usize,
}

impl Display for VmpSweepParms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x({}x{})x({}x{})",
            self.n, self.rows, self.cols_in, self.cols_out, self.size
        )
    }
}

pub struct CnvSweepParms {
    pub n: usize,
    pub size: usize,
}

impl Display for CnvSweepParms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.n, self.size)
    }
}

/// Sweep parameters for the negacyclic reim FFT/IFFT (`m` is the transform
/// half-length passed to `NegacyclicFFTNew::new`; the transformed data has
/// length `2 * m`).
#[derive(Debug, Clone)]
pub struct ReimSweepParams {
    pub m: usize,
}

impl Display for ReimSweepParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "m={}", self.m)
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

impl Display for CoreParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(n={},base2k={},k={},rank={},dsize={})",
            self.n, self.base2k, self.k, self.rank, self.dsize
        )
    }
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

pub fn key_dnum_k_aux(k: u32, base2k: u32, dsize: u32) -> (u32, u32) {
    let digit: u32 = dsize * base2k;
    assert!(
        k >= 2 * digit,
        "k ({k}) must hold at least one gadget digit plus one digit of guard ({digit} bits each)"
    );
    let dnum: u32 = k / digit - 1;
    (dnum, k - dnum * digit)
}

/// One point of the CKKS benchmark sweep.
///
/// The number of limbs (`k = limbs * base2k`) and the gadget split (`dsize`,
/// `dnum`) are scaled down with `n`, so the benchmark shape stays representative
/// across sizes (smaller rings support smaller moduli / fewer limbs). `dnum` is
/// derived as `⌈k / (dsize * base2k)⌉`, matching `tsk_layout`.
#[derive(Clone, Copy)]
pub struct CkksBenchParams {
    pub n: usize,
    pub base2k: usize,
    pub k: usize,
    pub log_delta: usize,
    pub dsize: usize,
    pub slots: SlotsKind,
}

impl Display for CkksBenchParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(n={},base2k={},k={},log_delta={},dsize={})",
            self.n, self.base2k, self.k, self.log_delta, self.dsize
        )
    }
}

/// Parameters shared by the `poulpy-bin-fhe` blind-rotation and
/// circuit-bootstrapping benchmarks — enough to build the `Module`, the GLWE
/// layout, and the LWE layout. Everything specific to one benchmark (its
/// per-key gadget-decomposition shape, message encoding, ...) lives in that
/// benchmark's own params struct — see [`BlindRotateBenchParams`] and
/// [`CircuitBootstrappingBenchParam`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinFheBenchParams {
    pub n_glwe: u32,
    pub n_lwe: u32,
    pub base2k: u32,
    pub k_aux: u32,
    pub rank: u32, // Same rank for GLWE and GGLWE keys, for now. Could be split if needed.
}

impl Display for BinFheBenchParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(n_glwe={},n_lwe={},base2k={},k_aux={},rank={})",
            self.n_glwe, self.n_lwe, self.base2k, self.k_aux, self.rank
        )
    }
}

/// Parameters for the blind-rotation benchmark: the shared GLWE/LWE shape
/// from [`BinFheBenchParams`], plus the constants specific to this benchmark
/// (LWE secret block size, look-up-table extension factor, and the log2 of
/// the plaintext message modulus).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindRotateBenchParams {
    pub bin_fhe_params: BinFheBenchParams,
    pub block_size: usize,
    pub extension_factor: usize,
    pub log_message_modulus: usize,
}

impl Display for BlindRotateBenchParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{},block_size={},extension_factor={},log_message_modulus={}",
            self.bin_fhe_params, self.block_size, self.extension_factor, self.log_message_modulus
        )
    }
}

/// Parameters for the circuit-bootstrapping benchmark: the shared GLWE/LWE
/// shape from [`BinFheBenchParams`], the gadget-decomposition shape
/// (`dnum`/`dsize`) for each key involved (which can differ per key — the
/// blind-rotation key `brk` has no `dsize` of its own, its layout only
/// carries a `dnum`), and the `log_domain`/`extension_factor` passed to
/// `execute_to_constant`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBootstrappingBenchParam {
    pub bin_fhe_params: BinFheBenchParams,
    pub brk_dnum: u32,
    pub atk_dnum: u32,
    pub atk_dsize: u32,
    pub tsk_dnum: u32,
    pub tsk_dsize: u32,
    pub ggsw_dnum: u32,
    pub ggsw_dsize: u32,
    pub log_domain: usize,
    pub extension_factor: usize,
}

impl Display for CircuitBootstrappingBenchParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{},brk_dnum={},atk_dnum={},atk_dsize={},tsk_dnum={},tsk_dsize={},ggsw_dnum={},ggsw_dsize={},log_domain={},extension_factor={}",
            self.bin_fhe_params,
            self.brk_dnum,
            self.atk_dnum,
            self.atk_dsize,
            self.tsk_dnum,
            self.tsk_dsize,
            self.ggsw_dnum,
            self.ggsw_dsize,
            self.log_domain,
            self.extension_factor
        )
    }
}

pub fn default_bench_params_hal() -> Vec<HalSweepParms> {
    vec![
        HalSweepParms {
            n: 1 << 10,
            cols: 2,
            size: 2,
        },
        HalSweepParms {
            n: 1 << 11,
            cols: 2,
            size: 4,
        },
        HalSweepParms {
            n: 1 << 12,
            cols: 2,
            size: 8,
        },
        HalSweepParms {
            n: 1 << 13,
            cols: 2,
            size: 16,
        },
        HalSweepParms {
            n: 1 << 14,
            cols: 2,
            size: 32,
        },
        HalSweepParms {
            n: 1 << 15,
            cols: 2,
            size: 64,
        },
    ]
}

pub fn default_bench_params_vmp() -> Vec<VmpSweepParms> {
    vec![
        VmpSweepParms {
            n: 1 << 10,
            rows: 2,
            cols_in: 1,
            cols_out: 2,
            size: 3,
        },
        VmpSweepParms {
            n: 1 << 11,
            rows: 4,
            cols_in: 1,
            cols_out: 2,
            size: 5,
        },
        VmpSweepParms {
            n: 1 << 12,
            rows: 7,
            cols_in: 1,
            cols_out: 2,
            size: 8,
        },
        VmpSweepParms {
            n: 1 << 13,
            rows: 15,
            cols_in: 1,
            cols_out: 2,
            size: 16,
        },
        VmpSweepParms {
            n: 1 << 14,
            rows: 31,
            cols_in: 1,
            cols_out: 2,
            size: 32,
        },
        VmpSweepParms {
            n: 1 << 15,
            rows: 63,
            cols_in: 1,
            cols_out: 2,
            size: 64,
        },
    ]
}

pub fn default_bench_params_cnv() -> Vec<CnvSweepParms> {
    vec![
        CnvSweepParms { n: 1 << 10, size: 2 },
        CnvSweepParms { n: 1 << 11, size: 4 },
        CnvSweepParms { n: 1 << 12, size: 8 },
        CnvSweepParms { n: 1 << 13, size: 16 },
        CnvSweepParms { n: 1 << 14, size: 32 },
        CnvSweepParms { n: 1 << 15, size: 64 },
    ]
}

pub fn default_bench_params_core() -> Vec<CoreParams> {
    vec![
        CoreParams {
            n: 1 << 12,
            base2k: 52,
            k: 54 * 2,
            rank: 1,
            dsize: 1,
        },
        CoreParams {
            n: 1 << 13,
            base2k: 52,
            k: 54 * 3,
            rank: 1,
            dsize: 1,
        },
        CoreParams {
            n: 1 << 14,
            base2k: 52,
            k: 54 * 6,
            rank: 1,
            dsize: 1,
        },
        CoreParams {
            n: 1 << 15,
            base2k: 52,
            k: 54 * 12,
            rank: 1,
            dsize: 3,
        },
        CoreParams {
            n: 1 << 16,
            base2k: 52,
            k: 54 * 24,
            rank: 1,
            dsize: 6,
        },
    ]
}

pub fn default_bench_params_ckks() -> Vec<CkksBenchParams> {
    vec![
        CkksBenchParams {
            n: 1 << 12,
            base2k: 52,
            k: 52,
            log_delta: 20,
            dsize: 1,
            slots: SlotsKind::Complex,
        },
        CkksBenchParams {
            n: 1 << 13,
            base2k: 52,
            k: 52 * 3,
            log_delta: 40,
            dsize: 1,
            slots: SlotsKind::Complex,
        },
        CkksBenchParams {
            n: 1 << 14,
            base2k: 52,
            k: 52 * 6,
            log_delta: 40,
            dsize: 1,
            slots: SlotsKind::Complex,
        },
        CkksBenchParams {
            n: 1 << 15,
            base2k: 52,
            k: 52 * 12,
            log_delta: 40,
            dsize: 3,
            slots: SlotsKind::Complex,
        },
        CkksBenchParams {
            n: 1 << 16,
            base2k: 52,
            k: 52 * 24,
            log_delta: 40,
            dsize: 6,
            slots: SlotsKind::Complex,
        },
    ]
}

/// Single representative blind-rotation benchmark point. Unlike the other
/// `default_bench_params_*` sweeps, bin-fhe params aren't indexed by a single
/// `log_n` (the GLWE ring, LWE dimension, and per-key gadget shapes vary
/// independently), so this is one fixed, non-swept parameter set.
pub fn default_bench_params_blind_rotate() -> BlindRotateBenchParams {
    BlindRotateBenchParams {
        bin_fhe_params: BinFheBenchParams {
            n_glwe: 1 << 12,
            n_lwe: 630,
            base2k: 18,
            k_aux: 54,
            rank: 1,
        },
        block_size: 7,
        extension_factor: 1,
        log_message_modulus: 2,
    }
}

/// Single representative circuit-bootstrapping benchmark point (see
/// [`default_bench_params_blind_rotate`] for why this isn't a `log_n` sweep).
pub fn default_bench_params_circuit_bootstrapping() -> CircuitBootstrappingBenchParam {
    CircuitBootstrappingBenchParam {
        bin_fhe_params: BinFheBenchParams {
            n_glwe: 1 << 12,
            n_lwe: 630,
            base2k: 18,
            k_aux: 90,
            rank: 1,
        },
        brk_dnum: 4,
        atk_dnum: 4,
        atk_dsize: 1,
        tsk_dnum: 4,
        tsk_dsize: 1,
        ggsw_dnum: 3,
        ggsw_dsize: 1,
        log_domain: 1,
        extension_factor: 1,
    }
}
