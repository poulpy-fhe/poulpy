use std::fmt::Display;

use poulpy_core::layouts::{Base2K, Dsize, GGLWELayout, TorusPrecision};
use serde::{Deserialize, Serialize};

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

/// Splits a total key width `k` into `(dnum, k_aux)` so that
/// `dnum * dsize * base2k + k_aux == k` exactly: the guard `k_aux` is one gadget
/// digit plus the remainder of `k` modulo the digit width, and `dnum` covers
/// the rest via [`GGLWELayout::dnum_for_input`].
pub fn key_dnum_k_aux(k: u32, base2k: u32, dsize: u32) -> (u32, u32) {
    let digit: u32 = dsize * base2k;
    assert!(
        k >= 2 * digit,
        "k ({k}) must hold at least one gadget digit plus one digit of guard ({digit} bits each)"
    );
    let k_aux: u32 = digit + k % digit;
    let dnum: u32 = GGLWELayout::dnum_for_input(Base2K(base2k), TorusPrecision(k - k_aux), Dsize(dsize)).0;
    (dnum, k_aux)
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
