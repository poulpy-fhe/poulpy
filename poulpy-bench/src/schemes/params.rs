use std::fmt::Display;

use poulpy_ckks::SlotsKind;
use poulpy_hal::layouts::Backend;
use serde::{Deserialize, Serialize};

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

#[derive(Clone, Copy)]
pub struct CkksBootstrappingBenchParams {
    pub preset: CkksBootstrappingPreset,
    pub base2k: usize,
    pub dsize: usize,
    pub dense_to_sparse_dsize: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CkksBootstrappingPreset {
    C2S16Levels,
    S2C16Levels,
}

impl Display for CkksBootstrappingPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::C2S16Levels => write!(f, "c2s_16_levels"),
            Self::S2C16Levels => write!(f, "s2c_16_levels"),
        }
    }
}

impl Display for CkksBootstrappingBenchParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(preset={},n={},base2k={},dsize={},dense_to_sparse_dsize={})",
            self.preset,
            1 << 16,
            self.base2k,
            self.dsize,
            self.dense_to_sparse_dsize
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

pub fn default_bench_params_ckks_bootstrapping<BE: Backend>() -> [CkksBootstrappingBenchParams; 2] {
    let (base2k, dsize, dense_to_sparse_dsize) = if BE::DFT_IS_EXACT { (52, 4, 3) } else { (19, 7, 7) };
    [CkksBootstrappingPreset::C2S16Levels, CkksBootstrappingPreset::S2C16Levels].map(|preset| CkksBootstrappingBenchParams {
        preset,
        base2k,
        dsize,
        dense_to_sparse_dsize,
    })
}

/// Blind-rotation benchmark points. Unlike the other `default_bench_params_*`
/// sweeps, bin-fhe params aren't indexed by a single `log_n` (the GLWE ring,
/// LWE dimension, and per-key gadget shapes vary independently) — currently
/// a single representative parameter set, but more will be added as
/// additional bin-fhe parameter regimes become relevant to benchmark.
pub fn default_bench_params_blind_rotate() -> Vec<BlindRotateBenchParams> {
    vec![BlindRotateBenchParams {
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
    }]
}

/// Circuit-bootstrapping benchmark points (see
/// [`default_bench_params_blind_rotate`] for why this isn't a `log_n` sweep).
pub fn default_bench_params_circuit_bootstrapping() -> Vec<CircuitBootstrappingBenchParam> {
    vec![CircuitBootstrappingBenchParam {
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
    }]
}
