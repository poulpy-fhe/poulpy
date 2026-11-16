//! The parameters the client and the server agree on: the ring, the scales and widths of a round, the security bounds,
//! the shape of every key, and the maps of a round.
//!
//! The 128-bit bounds are those `poulpy-ckks` certifies its presets with: the rlwe-params table vendored in
//! `poulpy-core`, read through [`poulpy_core::security`], at `N = 2^15` and an error of deviation 3.2.

use anyhow::{Result, ensure};
use poulpy_ckks::api::LinearTransformationStrategy;
use poulpy_core::{
    Distribution,
    layouts::{
        Degree, Diagonals, GGLWELayout, GLWEAutomorphismKeyLayout, GLWETensorKeyLayout, LinearTransformationLayout, Rank,
        TorusPrecision, optimal_bsgs_giant_step,
    },
    security::{SecretSpec, Security},
};

use crate::{
    bootstrapping::Bootstrapping,
    layout::{SLOTS, SlotMap, linear_layer, shift_x},
};

/// Degree `2^15`: its 2^14 slots hold 8 states of 2^11 positions.
pub(crate) const LOG_N: usize = 15;
/// Bits per limb of the torus representation.
pub(crate) const BASE2K: usize = 52;
/// Scale of the integers the bootstrapping reduces. Each bit of it halves the error of the bits it returns.
pub(crate) const LOG_DELTA: usize = 34;
/// One bit between the modulus and the scale at the raise, so the bootstrapping reads integers mod 2.
pub(crate) const LOG_MSG_RATIO: usize = 1;
/// Scale of the bits the bootstrapping returns.
pub(crate) const OUTPUT_LOG_DELTA: usize = LOG_DELTA + LOG_MSG_RATIO;
/// Scale of the diagonals of the round's maps.
pub(crate) const LOG_DELTA_MAPS: usize = 20;
/// Bits a round keeps above the bootstrapping input.
const MARGIN: usize = 3;

/// The level every key and ciphertext is certified at: their moduli stay under the bound of their secret.
pub(crate) const SECURITY: Security = Security::Bits128;
/// Hamming weight of the client's ternary secret.
pub(crate) const DENSE_HAMMING_WEIGHT: usize = 1024;
/// Hamming weight of the ternary secret the raise switches to and back.
pub(crate) const SPARSE_HAMMING_WEIGHT: usize = 32;
/// The client's secret, as the bounds see it.
pub(crate) const DENSE_SECRET: SecretSpec = SecretSpec::new(1 << LOG_N, Distribution::TernaryFixed(DENSE_HAMMING_WEIGHT));
/// The secret of the raise, on the same ring.
pub(crate) const SPARSE_SECRET: SecretSpec = SecretSpec::new(1 << LOG_N, Distribution::TernaryFixed(SPARSE_HAMMING_WEIGHT));

/// Bits a key keeps past its input and one digit, for its own error to stay below the input's last bit.
const KEY_NOISE_MARGIN: usize = 2;

/// The key of fewest digits that switches an input of `input_k` bits under a modulus of at most `limit` bits.
///
/// A key of `d`-bit digits covers its input with `ceil(input_k / d)` digits, its gadget, and keeps behind them an
/// auxiliary modulus of a digit and `log N` bits, so the product of a digit and a key row loses nothing. Fewer digits
/// mean fewer products, but wider digits need a wider auxiliary modulus: the widest digit that fits the bound wins.
/// Digits are whole limbs.
pub(crate) fn key_layout(input_k: usize, limit: usize) -> Result<GGLWELayout> {
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
    pub(crate) bootstrapping: Bootstrapping,
    /// theta, rho and pi, then the `x + 1` and `x + 2` reads of chi.
    pub(crate) maps: [SlotMap; 3],
    /// The rotation keys and the tensor key of the round, over the bootstrapping's output.
    pub(crate) rotation_key: GLWEAutomorphismKeyLayout,
    pub(crate) tensor_key: GLWETensorKeyLayout,
}

impl Params {
    pub fn new() -> Result<Self> {
        // A round spends `LOG_DELTA_MAPS` bits of modulus on `L` and on the shifts of chi, `OUTPUT_LOG_DELTA` on the
        // product of chi, and the margin.
        let bootstrapping = Bootstrapping::new(MARGIN + 2 * LOG_DELTA_MAPS + OUTPUT_LOG_DELTA)?;
        let key = key_layout(bootstrapping.output_k, SECURITY.max_log_q(DENSE_SECRET)?)?;
        SECURITY.check(DENSE_SECRET, "round key", &key)?;
        Ok(Self {
            bootstrapping,
            maps: [linear_layer(), shift_x(1), shift_x(2)],
            rotation_key: key.into(),
            tensor_key: key.into(),
        })
    }

    /// The Galois elements of the rotation keys the round's maps take.
    pub(crate) fn round_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
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
pub(crate) fn diagonals(map: &SlotMap) -> Diagonals<f64> {
    Diagonals::<f64>::from_slot_map(SLOTS, SLOTS, map.iter().map(|&(o, i, w)| (o, i, w as f64)))
}

/// Their baby-step giant-step schedule.
pub(crate) fn bsgs_layout(diagonals: &Diagonals<f64>) -> LinearTransformationLayout {
    let indexes = diagonals.indexes();
    let giant_step = optimal_bsgs_giant_step(indexes.iter().copied(), SLOTS);
    LinearTransformationLayout {
        indexes,
        slots: SLOTS,
        strategy: LinearTransformationStrategy::Bsgs { giant_step },
    }
}
