//! Shaping polynomials of the [`ReplicatedLayout`].
//!
//! A replicated layout of period `R = 2^log_r` shapes errors by
//! `g = U(X^(N/(2R)))` with `U ∈ Z[Y]/(Y^(2R)+1)`. Slot `t` sees `g` at
//! `U(ω^(5^(t mod R)))`, `ω = e^(iπ/(2R))`, so `U` acts on `R` slot classes,
//! class `0` carrying the data and the others the guards.
//!
//! [`ShapingProfile`] derives what shaped rounding needs from `U`: its inverse
//! modulo `2^64`, the per-class magnitudes and the largest exact pass.
//! [`ShapingUnit`] lists units with the smallest guard amplification for a
//! prescribed data suppression.

use poulpy_core::layouts::ReplicatedLayout;

use num_traits::{FromPrimitive, ToPrimitive, Zero};

use crate::{Quad, numerics::CKKSFloat};

/// Exact data of a shaping polynomial `U`, and its slot-class profile.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapingProfile {
    log_r: usize,
    inverse: Vec<u64>,
    log_magnitudes: Vec<f64>,
    l1: u64,
}

impl ShapingProfile {
    pub fn new(layout: &ReplicatedLayout) -> Self {
        Self::from_coeffs(layout.log_r(), layout.shaping())
    }

    fn from_coeffs(log_r: usize, coeffs: &[i16]) -> Self {
        let wide: Vec<u64> = coeffs.iter().map(|&c| c as i64 as u64).collect();
        Self {
            log_r,
            inverse: inverse_mod_2_64(&wide),
            log_magnitudes: (0..1 << log_r).map(|j| log2_abs_at_class(coeffs, log_r, j)).collect(),
            l1: coeffs.iter().map(|&c| c.unsigned_abs() as u64).sum(),
        }
    }

    /// Base-2 logarithm of the slot period `R`.
    pub fn log_r(&self) -> usize {
        self.log_r
    }

    /// The `2R` coefficients of `U^(-1) mod 2^64`.
    pub fn inverse(&self) -> &[u64] {
        &self.inverse
    }

    /// `log2 |U(ω^(5^j))|` for each slot class `j < R`, the data class first.
    pub fn class_log_magnitudes(&self) -> &[f64] {
        &self.log_magnitudes
    }

    /// Bits by which `U` suppresses errors on the data class.
    pub fn suppression_bits(&self) -> f64 {
        -self.log_magnitudes[0]
    }

    /// Bits by which `U` amplifies errors on the worst guard class.
    pub fn guard_bits(&self) -> f64 {
        self.log_magnitudes[1..].iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// `||U||_1`.
    pub fn l1_norm(&self) -> u64 {
        self.l1
    }

    /// Largest drop `b` whose shaped-rounding numerator fits a signed 64-bit
    /// word: `b <= 64 - ceil(log2(2 + ||U||_1))`.
    pub fn max_pass_bits(&self) -> usize {
        64 - (2 + self.l1).next_power_of_two().trailing_zeros() as usize
    }

    /// Shaping depth `min(log_delta + w, max_pass_bits)` of a rescale by `2^log_delta`.
    pub fn shaping_depth(&self, log_delta: usize, w: usize) -> usize {
        (log_delta + w).min(self.max_pass_bits())
    }
}

/// A shaping unit `U = ∏ ((1 - Y^a)/(1 - Y))^(e_a)`, odd `1 < a < 2R`,
/// minimizing the largest guard magnitude `z` subject to `ℓ_0(U) <= -T` and
/// `ℓ_j(U) <= z` for `1 <= j < R`, where `ℓ_j(U) = log2 |U(ω^(5^j))|` is linear
/// in the exponents. The presets solve this integer program exactly over
/// `|e_a| <= 40`, a range their optimal exponents do not reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapingUnit {
    pub log_r: usize,
    pub target_bits: u32,
    pub coeffs: &'static [i16],
}

impl ShapingUnit {
    /// `R = 8`, 20.08 bits of suppression, guards at most 4.00 bits.
    pub const U16_20: Self = Self {
        log_r: 3,
        target_bits: 20,
        coeffs: &[0, -6, 1, 1, 1, 1, -1, 1, 1, 1, 1, -6, 0, 2, 0, -2],
    };

    /// `R = 8`, 40.84 bits of suppression, guards at most 7.06 bits.
    pub const U16_40: Self = Self {
        log_r: 3,
        target_bits: 40,
        coeffs: &[21, 8, -48, 2, 0, -2, 48, -8, -21, 2, -2, 0, 3, 0, -2, 2],
    };

    /// `R = 8`, 62.00 bits of suppression, guards at most 9.90 bits.
    pub const U16_60: Self = Self {
        log_r: 3,
        target_bits: 60,
        coeffs: &[-30, -101, 16, 284, -122, 0, 122, -284, -16, 101, 30, -140, 91, 75, 91, -140],
    };

    /// `R = 16`, 23.50 bits of suppression, guards at most 3.14 bits.
    pub const U32_20: Self = Self {
        log_r: 4,
        target_bits: 20,
        coeffs: &[
            -1, 0, 1, -1, 0, 0, 0, 1, 1, 1, 0, 0, 0, -1, 1, 0, -1, 0, -1, -1, 0, 1, 1, -2, 0, 2, -1, -1, 0, 1, 1, 0,
        ],
    };

    /// `R = 16`, 40.99 bits of suppression, guards at most 4.25 bits.
    pub const U32_40: Self = Self {
        log_r: 4,
        target_bits: 40,
        coeffs: &[
            0, 3, -1, 2, 0, -2, -3, -4, 3, 0, 0, 2, -1, -1, 3, 0, -3, 1, 1, -2, 0, 0, -3, 4, 3, 2, 0, -2, 1, -3, 0, -1,
        ],
    };

    /// `R = 16`, 45.95 bits of suppression, guards at most 4.59 bits.
    pub const U32_45: Self = Self {
        log_r: 4,
        target_bits: 45,
        coeffs: &[
            1, 1, 1, -5, -1, -4, 0, 2, -1, -3, 5, 1, 3, 1, 5, -3, -1, 2, 0, -4, -1, -5, 1, 1, 1, 1, -1, 1, 0, -1, 1, -1,
        ],
    };

    /// `R = 16`, 50.21 bits of suppression, guards at most 4.74 bits.
    pub const U32_50: Self = Self {
        log_r: 4,
        target_bits: 50,
        coeffs: &[
            -1, 2, 3, -2, -1, 0, 1, -2, 1, -2, 1, 0, -1, -2, 3, 2, -1, 0, 6, 0, -5, -2, -5, 0, 0, 0, 5, 2, 5, 0, -6, 0,
        ],
    };

    /// `R = 16`, 67.07 bits of suppression, guards at most 5.69 bits.
    pub const U32_60: Self = Self {
        log_r: 4,
        target_bits: 60,
        coeffs: &[
            -9, 0, 4, 1, 0, -2, -3, 7, 1, -3, -6, 15, -1, -7, -7, -3, 0, 3, 7, 7, 1, -15, 6, 3, -1, -7, 3, 2, 0, -1, -4, 0,
        ],
    };

    pub const ALL: [Self; 8] = [
        Self::U16_20,
        Self::U16_40,
        Self::U16_60,
        Self::U32_20,
        Self::U32_40,
        Self::U32_45,
        Self::U32_50,
        Self::U32_60,
    ];

    /// The replicated layout of this unit for a batch of `2^log_k` values.
    pub fn layout(&self, log_k: u8, w: u8) -> poulpy_core::Result<ReplicatedLayout> {
        ReplicatedLayout::new(log_k, self.log_r as u8, w, self.coeffs)
    }
}

/// `5^j mod 4R`, the exponent of `ω` at slot class `j`.
fn class_exponent(log_r: usize, j: usize) -> u64 {
    let modulus = 4u64 << log_r;
    (0..j).fold(1, |acc, _| acc * 5 % modulus)
}

fn log2_abs_at_class(coeffs: &[i16], log_r: usize, j: usize) -> f64 {
    // Strong suppression cancels far below the coefficients' magnitude, so the
    // sum runs in binary128.
    let e = class_exponent(log_r, j);
    let (mut re, mut im) = (Quad::zero(), Quad::zero());
    for (i, &c) in coeffs.iter().enumerate() {
        let (cos, sin) = Quad::ckks_root_of_unity((e * i as u64) % (4 << log_r), log_r as u32 + 2);
        let c = Quad::from_i16(c).unwrap();
        re += c * cos;
        im += c * sin;
    }
    ((re * re + im * im).ckks_log2() / Quad::from_f64(2.0).unwrap())
        .to_f64()
        .unwrap()
}

/// Product in `(Z/2^64)[Y]/(Y^m + 1)`.
fn negacyclic_mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    let m = a.len();
    let mut out = vec![0u64; m];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            let p = x.wrapping_mul(y);
            let k = i + j;
            if k < m {
                out[k] = out[k].wrapping_add(p);
            } else {
                out[k - m] = out[k - m].wrapping_sub(p);
            }
        }
    }
    out
}

/// Inverse in `(Z/2^64)[Y]/(Y^m + 1)` of `u` with `u(1)` odd, by Newton's
/// iteration `v ← v(2 - uv)` from `v = 1`. Since `(Y+1)^m ∈ 2Z[Y]/(Y^m + 1)`,
/// the error `1 - uv` reaches zero after `log2(64m)` squarings.
fn inverse_mod_2_64(u: &[u64]) -> Vec<u64> {
    let m = u.len();
    let mut one = vec![0u64; m];
    one[0] = 1;
    let mut v = one.clone();
    loop {
        let uv = negacyclic_mul(u, &v);
        if uv == one {
            return v;
        }
        let mut correction: Vec<u64> = uv.iter().map(|x| x.wrapping_neg()).collect();
        correction[0] = correction[0].wrapping_add(2);
        v = negacyclic_mul(&v, &correction);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_profiles() {
        for (unit, suppression, guard, l1) in [
            (ShapingUnit::U16_20, 20.076, 4.005, 25),
            (ShapingUnit::U16_40, 40.842, 7.058, 169),
            (ShapingUnit::U16_60, 61.996, 9.903, 1643),
            (ShapingUnit::U32_20, 23.498, 3.138, 21),
            (ShapingUnit::U32_40, 40.994, 4.253, 51),
            (ShapingUnit::U32_45, 45.955, 4.594, 59),
            (ShapingUnit::U32_50, 50.207, 4.742, 61),
            (ShapingUnit::U32_60, 67.067, 5.692, 129),
        ] {
            let profile = ShapingProfile::new(&unit.layout(2, 24).unwrap());
            assert!((profile.suppression_bits() - suppression).abs() < 5e-4, "{unit:?}");
            assert!(profile.suppression_bits() >= unit.target_bits as f64, "{unit:?}");
            assert!((profile.guard_bits() - guard).abs() < 5e-4, "{unit:?}");
            assert_eq!(profile.l1_norm(), l1, "{unit:?}");
            // A unit has norm one, so its class profile sums to zero.
            assert!(profile.class_log_magnitudes().iter().sum::<f64>().abs() < 1e-9, "{unit:?}");
        }
    }

    #[test]
    fn pass_size_and_shaping_depth() {
        let profile = ShapingProfile::new(&ShapingUnit::U32_45.layout(4, 24).unwrap());
        assert_eq!(profile.max_pass_bits(), 58);
        assert_eq!(profile.shaping_depth(20, 24), 44);
        assert_eq!(profile.shaping_depth(40, 24), 58);
        assert_eq!(
            ShapingProfile::new(&ShapingUnit::U16_60.layout(4, 24).unwrap()).max_pass_bits(),
            53
        );
    }

    #[test]
    fn inverse_modulo_2_64() {
        let non_unit = ReplicatedLayout::new(2, 1, 8, &[3, 0, 2, 0]).unwrap();
        let layouts = ShapingUnit::ALL.map(|unit| unit.layout(2, 8).unwrap());
        for layout in layouts.iter().chain([&non_unit]) {
            let u: Vec<u64> = layout.shaping().iter().map(|&c| c as i64 as u64).collect();
            let product = negacyclic_mul(&u, ShapingProfile::new(layout).inverse());
            assert_eq!(product[0], 1);
            assert!(product[1..].iter().all(|&c| c == 0));
        }
        // `3 + 2Y^2` has norm `13^2` over the four embeddings of `Z[Y]/(Y^4+1)`.
        let sum: f64 = ShapingProfile::new(&non_unit).class_log_magnitudes().iter().sum();
        assert!((sum - 13f64.ckks_log2()).abs() < 1e-9);
    }
}
