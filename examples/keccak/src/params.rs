//! The parameters the client and the server agree on: the ring, the scales and widths of a round, the security bounds,
//! the shape of every key, and the maps of a round.
//!
//! The 128-bit bounds are those `poulpy-ckks` uses for its presets, from the table of
//! [sparse-key-estimate](https://github.com/jdumezy/sparse-key-estimate/blob/aa2c40d4eee4551e6fba06c55d7432f6918e1597/Precomputed-Tables/128bits_security.md)
//! at `N = 2^15` and an error of deviation 3.2.

use anyhow::{Result, ensure};
use poulpy_ckks::api::LinearTransformationStrategy;
use poulpy_core::layouts::{
    Degree, Diagonals, GGLWELayout, GLWEAutomorphismKeyLayout, GLWETensorKeyLayout, LinearTransformationLayout, Rank,
    TorusPrecision, optimal_bsgs_giant_step,
};

use crate::{
    bootstrapping::Bootstrapping,
    layout::{SLOTS, SlotMap, linear_layer, shift_x},
};

/// Degree `2^15`: its 2^14 slots hold 8 states of 2^11 positions.
pub const LOG_N: usize = 15;
/// Bits per limb of the torus representation.
pub const BASE2K: usize = 52;
/// Scale of the integers the bootstrapping reduces. Each bit of it halves the error of the bits it returns.
pub const LOG_DELTA: usize = 34;
/// One bit between the modulus and the scale at the raise, so the bootstrapping reads integers mod 2.
pub const LOG_MSG_RATIO: usize = 1;
/// Scale of the bits the bootstrapping returns.
pub const OUTPUT_LOG_DELTA: usize = LOG_DELTA + LOG_MSG_RATIO;
/// Scale of the diagonals of the round's maps.
pub const LOG_DELTA_MAPS: usize = 20;
/// Bits a round keeps above the bootstrapping input.
const MARGIN: usize = 3;

/// A secret's Hamming weight and the largest modulus, in bits, under it at 128 bits of security.
pub struct SecretBound {
    pub hamming_weight: usize,
    pub max_log_modulus: usize,
}

pub const DENSE_SECRET: SecretBound = SecretBound {
    hamming_weight: 1024,
    max_log_modulus: 851,
};
pub const SPARSE_SECRET: SecretBound = SecretBound {
    hamming_weight: 32,
    max_log_modulus: 162,
};

/// Bits a key keeps past its input and one digit, for its own error to stay below the input's last bit.
const KEY_NOISE_MARGIN: usize = 2;

/// The key of fewest digits that switches an input of `input_k` bits under a modulus of at most `limit` bits.
///
/// A key of `d`-bit digits covers its input with `ceil(input_k / d)` digits, its gadget, and keeps behind them an
/// auxiliary modulus of a digit and `log N` bits, so the product of a digit and a key row loses nothing. Fewer digits
/// mean fewer products, but wider digits need a wider auxiliary modulus: the widest digit that fits the bound wins.
/// Digits are whole limbs.
pub fn key_layout(input_k: usize, limit: usize) -> Result<GGLWELayout> {
    let shape = |k_digit: usize| {
        let gadget = input_k.div_ceil(k_digit) * k_digit;
        let k_aux = (k_digit + LOG_N).min(limit.saturating_sub(gadget));
        let fits = k_aux >= k_digit && gadget + k_aux >= input_k + k_digit + KEY_NOISE_MARGIN;
        fits.then_some((input_k.div_ceil(k_digit), k_digit, k_aux))
    };
    let (dnum, k_digit, k_aux) = (1..=input_k.div_ceil(BASE2K))
        .filter_map(|limbs| shape(limbs * BASE2K))
        .min_by_key(|&(dnum, k_digit, _)| (dnum, k_digit))
        .unwrap_or((0, 0, 0));
    ensure!(dnum > 0, "no key switches {input_k} bits under {limit}");
    Ok(GGLWELayout {
        n: Degree(1 << LOG_N),
        base2k: BASE2K.into(),
        dnum: dnum.into(),
        k_aux: TorusPrecision(k_aux as u32),
        rank_in: Rank(1),
        rank_out: Rank(1),
        k_digit: TorusPrecision(k_digit as u32),
        k_offset: TorusPrecision(0),
        stride: 1,
    })
}

/// Everything the client and the server agree on.
pub struct Params {
    pub bootstrapping: Bootstrapping,
    /// theta, rho and pi, then the `x + 1` and `x + 2` reads of chi.
    pub maps: [SlotMap; 3],
    /// The rotation keys and the tensor key of the round, over the bootstrapping's output.
    pub rotation_key: GLWEAutomorphismKeyLayout,
    pub tensor_key: GLWETensorKeyLayout,
}

impl Params {
    pub fn new() -> Result<Self> {
        // A round spends `LOG_DELTA_MAPS` bits of modulus on `L` and on the shifts of chi, `OUTPUT_LOG_DELTA` on the
        // product of chi, and the margin.
        let bootstrapping = Bootstrapping::new(MARGIN + 2 * LOG_DELTA_MAPS + OUTPUT_LOG_DELTA)?;
        let key = key_layout(bootstrapping.output_k, DENSE_SECRET.max_log_modulus)?;
        Ok(Self {
            bootstrapping,
            maps: [linear_layer(), shift_x(1), shift_x(2)],
            rotation_key: key.into(),
            tensor_key: key.into(),
        })
    }

    /// The Galois elements of the rotation keys the round's maps take.
    pub fn round_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        let mut elements: Vec<i64> = self
            .maps
            .iter()
            .flat_map(|map| bsgs_layout(&diagonals(map)).index().galois_elements(cyclotomic_order))
            .collect();
        elements.sort_unstable();
        elements.dedup();
        elements
    }
}

/// The real diagonals of a slot map.
pub fn diagonals(map: &SlotMap) -> Diagonals<f64> {
    Diagonals::<f64>::from_slot_map(SLOTS, SLOTS, map.iter().map(|&(o, i, w)| (o, i, w as f64)))
}

/// Their baby-step giant-step schedule.
pub fn bsgs_layout(diagonals: &Diagonals<f64>) -> LinearTransformationLayout {
    let indexes = diagonals.indexes();
    let giant_step = optimal_bsgs_giant_step(indexes.iter().copied(), SLOTS);
    LinearTransformationLayout {
        indexes,
        slots: SLOTS,
        strategy: LinearTransformationStrategy::Bsgs { giant_step },
    }
}
