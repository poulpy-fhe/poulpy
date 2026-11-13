//! Shaping polynomials of the [`ReplicatedLayout`] of
//! [ePrint 2026/2513](https://eprint.iacr.org/2026/2513).
//!
//! A replicated layout of period `R = 2^log_r` shapes errors by
//! `g = U(X^(N/(2R)))` with `U ∈ Z[Y]/(Y^(2R)+1)`. Slot `t` sees `g` at
//! `U(ω^(5^(t mod R)))`, `ω = e^(iπ/(2R))`, so `U` acts on `R` slot classes,
//! class `0` carrying the data and the others the guards.
//!
//! On the conjugate-invariant ring of degree `N`, the subring of
//! `Z[X]/(X^(2N)+1)` fixed by `X -> X^-1`, `g = U(X^(N/R))` must be invariant
//! too: `U` lies in `Z[Y+Y^-1]`. Its `N` real slots fall into the same `R`
//! classes, slot `t` in class `t mod R`. Every unit of `Z[Y]/(Y^(2R)+1)` is a
//! power of `Y` times an invariant unit, which has the same class magnitudes,
//! so the conjugate-invariant ring shapes with `Y^s*U` for the shift `s` that
//! makes it invariant.
//!
//! [`ShapingProfile`] derives what shaped rounding needs from `U`: its inverse
//! modulo `2^64`, the per-class magnitudes and the largest exact pass.
//! [`ShapingUnit`] lists units with the smallest guard amplification for a
//! prescribed data suppression.

use std::{marker::PhantomData, sync::OnceLock};

use anyhow::Context;
use poulpy_core::layouts::{ReplicatedLayout, TorusPrecision};
use poulpy_hal::layouts::{Ring, Standard};

use num_traits::{FromPrimitive, ToPrimitive, Zero};

use crate::{CKKSCompositionError, CKKSInfos, CKKSResult, Quad, SlotLayout, ckks_offset_unary, numerics::CKKSFloat};

/// Largest [`ShapingProfile::max_pass_bits`], reached by `U = ±1`.
pub const MAX_PASS_BITS: usize = 62;

/// Bits a shaped product or automorphism computes under its scale beyond the scale itself,
/// the working bits of ePrint 2026/2513: [`ShapingProfile::shaping_depth`] is
/// `min(log_delta + SHAPED_WORKING_BITS, max_pass_bits)`.
pub const SHAPED_WORKING_BITS: usize = 24;

/// Largest number of coefficients of a shaping polynomial, `2R`.
pub(crate) const MAX_COEFFS: usize = 2 << ReplicatedLayout::MAX_LOG_R;

/// Whether `Rg` is the conjugate-invariant ring of degree `N`, where `Y = X^(N/R)` in
/// `Z[X]/(X^(2N)+1)` and polynomials hold their `N` invariant coefficients, rather than
/// `Z[X]/(X^N+1)`, where `Y = X^(N/(2R))`. A constant of the ring type.
pub const fn conjugate_invariant<Rg: Ring>() -> bool {
    Rg::CYCLOTOMIC_ORDER_FACTOR != Standard::CYCLOTOMIC_ORDER_FACTOR
}

/// Exact data of a shaping polynomial `U` on the ring `Rg`, and its slot-class
/// profile. The ring is part of the type, so a profile of one ring cannot reach
/// the shaped rounding of the other.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapingProfile<Rg: Ring = Standard> {
    log_r: usize,
    shift: usize,
    unit: [i16; MAX_COEFFS],
    coeffs: [i16; MAX_COEFFS],
    shaping: [u64; MAX_COEFFS],
    inverse: OnceLock<[u64; MAX_COEFFS]>,
    log_magnitudes: OnceLock<Vec<f64>>,
    l1: u64,
    _ring: PhantomData<Rg>,
}

impl ShapingProfile<Standard> {
    /// The profile of `layout` on the standard ring, which shapes with `U` itself.
    pub fn new(layout: &ReplicatedLayout) -> Self {
        Self::from_coeffs(layout.log_r(), 0, layout.shaping(), layout.shaping())
    }
}

impl<Rg: Ring> ShapingProfile<Rg> {
    /// The profile of `layout` on the ring `Rg`. The conjugate-invariant
    /// ring shapes with `Y^s*U`, the smallest shift `s < 2R` that makes `U`
    /// invariant, and refuses a `U` that has none.
    pub fn try_new(layout: &ReplicatedLayout) -> CKKSResult<Self> {
        Self::for_op("ShapingProfile::try_new", layout)
    }

    /// [`Self::try_new`], refusing under `op`, the public operation that asked for the profile.
    pub(crate) fn for_op(op: &'static str, layout: &ReplicatedLayout) -> CKKSResult<Self> {
        let u = layout.shaping();
        if !conjugate_invariant::<Rg>() {
            return Ok(Self::from_coeffs(layout.log_r(), 0, u, u));
        }
        let m = u.len();
        let shifted = (0..m).map(|s| (s, monomial_shift(u, s))).find(|(_, v)| is_invariant(&v[..m]));
        let Some((shift, v)) = shifted else {
            return Err(CKKSCompositionError::InvalidOperand {
                op,
                reason: "no power of Y shifts the shaping polynomial onto a conjugate-invariant one".into(),
            }
            .into());
        };
        Ok(Self::from_coeffs(layout.log_r(), shift, u, &v[..m]))
    }

    fn from_coeffs(log_r: usize, shift: usize, unit: &[i16], coeffs: &[i16]) -> Self {
        let (mut padded, mut shaping, mut unshifted) = ([0; MAX_COEFFS], [0; MAX_COEFFS], [0; MAX_COEFFS]);
        unshifted[..unit.len()].copy_from_slice(unit);
        for (i, &c) in coeffs.iter().enumerate() {
            padded[i] = c;
            shaping[i] = c as i64 as u64;
        }
        Self {
            log_r,
            shift,
            unit: unshifted,
            coeffs: padded,
            shaping,
            inverse: OnceLock::new(),
            log_magnitudes: OnceLock::new(),
            l1: coeffs.iter().map(|&c| c.unsigned_abs() as u64).sum(),
            _ring: PhantomData,
        }
    }

    fn len(&self) -> usize {
        2 << self.log_r
    }

    /// Base-2 logarithm of the slot period `R`.
    pub fn log_r(&self) -> usize {
        self.log_r
    }

    /// The shift `s` of the shaping polynomial `Y^s*U`, zero on the standard ring.
    pub fn shift(&self) -> usize {
        self.shift
    }

    /// The `2R` coefficients of the layout's `U`, before the shift: an independent check
    /// recomputes [`Self::shift`], [`Self::coefficients`] and [`Self::inverse`] from them.
    pub fn unit(&self) -> &[i16] {
        &self.unit[..self.len()]
    }

    /// Smallest number of coefficients a shaped polynomial has, one period of
    /// `Y`: `2R` on the standard ring and `R` invariant ones on the
    /// conjugate-invariant ring. Degrees are multiples of it.
    pub fn period(&self) -> usize {
        if conjugate_invariant::<Rg>() {
            self.len() / 2
        } else {
            self.len()
        }
    }

    /// The `2R` signed coefficients of the shaping polynomial `Y^s*U`, in
    /// `Z[Y]/(Y^(2R)+1)`. On the conjugate-invariant ring coefficient `2R - j`
    /// is the opposite of coefficient `j` and coefficient `R` is zero.
    pub fn coefficients(&self) -> &[i16] {
        &self.coeffs[..self.len()]
    }

    /// The `2R` coefficients of `(Y^s*U)^(-1) mod 2^64`, invariant when `Y^s*U` is.
    pub fn inverse(&self) -> &[u64] {
        &self.inverse.get_or_init(|| inverse_mod_2_64(&self.shaping[..self.len()]))[..self.len()]
    }

    /// `log2 |U(ω^(5^j))|` for each slot class `j < R`, the data class first.
    pub fn class_log_magnitudes(&self) -> &[f64] {
        self.log_magnitudes.get_or_init(|| {
            (0..1 << self.log_r)
                .map(|j| log2_abs_at_class(&self.coeffs[..self.len()], self.log_r, j))
                .collect()
        })
    }

    /// Bits by which `U` suppresses errors on the data class.
    pub fn suppression_bits(&self) -> f64 {
        -self.class_log_magnitudes()[0]
    }

    /// Bits by which `U` amplifies errors on the worst guard class.
    pub fn guard_bits(&self) -> f64 {
        self.class_log_magnitudes()[1..]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
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

    /// Shaping depth `min(log_delta + SHAPED_WORKING_BITS, max_pass_bits)` of a rescale by `2^log_delta`.
    pub fn shaping_depth(&self, log_delta: usize) -> usize {
        (log_delta + SHAPED_WORKING_BITS).min(self.max_pass_bits())
    }

    /// Writes `g*h` into `gh` for `h = [U^(-1)*c]_(2^lambda)`, centered, where
    /// `c` is read modulo `2^64`, so that `c - g*h` is divisible by
    /// `2^lambda`. `h` is a buffer as long as `c`. `c` holds the coefficients of a polynomial of
    /// the profile's ring, whose degree is a multiple of [`Self::period`]: on
    /// the conjugate-invariant ring its invariant coefficients, and `h` and
    /// `g*h` are invariant too. Exact for `lambda <= max_pass_bits`.
    pub fn shaped_multiple_host(&self, lambda: usize, c: &[i64], h: &mut [i64], gh: &mut [i64]) {
        assert!(
            !c.is_empty() && c.len().is_multiple_of(self.period()) && h.len() == c.len() && gh.len() == c.len(),
            "shaped_multiple_host: {} coefficients do not hold whole periods of {}",
            c.len(),
            self.period()
        );
        let mask = u64::MAX >> (64 - lambda);
        self.product(self.inverse(), c, h);
        for x in h.iter_mut() {
            let reduced = *x as u64 & mask;
            *x = if reduced > mask >> 1 {
                reduced as i64 - (1i64 << lambda)
            } else {
                reduced as i64
            };
        }
        self.product(&self.shaping[..self.len()], h, gh);
    }

    /// `p(Y)*x` in the profile's ring for `p` of `2R` coefficients, invariant
    /// on the conjugate-invariant ring.
    fn product(&self, p: &[u64], x: &[i64], out: &mut [i64]) {
        let stride = x.len() / self.period();
        if conjugate_invariant::<Rg>() {
            invariant_sparse(&p[..self.len() / 2], stride, x, out)
        } else {
            negacyclic_sparse(p, stride, x, out)
        }
    }
}

/// `Y^s*u` in `Z[Y]/(Y^m+1)`, `m = u.len()`, in the first `m` entries.
fn monomial_shift(u: &[i16], s: usize) -> [i16; MAX_COEFFS] {
    let m = u.len();
    let mut out = [0; MAX_COEFFS];
    for (i, &c) in u.iter().enumerate() {
        let k = i + s;
        out[k % m] = if (k / m).is_multiple_of(2) { c } else { -c };
    }
    out
}

/// Whether `v in Z[Y]/(Y^m+1)` is fixed by `Y -> Y^-1 = -Y^(m-1)`: `v_(m/2) = 0`
/// and `v_(m-j) = -v_j`.
fn is_invariant(v: &[i16]) -> bool {
    let m = v.len();
    v[m / 2] == 0 && (1..m / 2).all(|j| v[m - j] == -v[j])
}

/// A shaping unit `U = ∏ ((1 - Y^a)/(1 - Y))^(e_a)`, odd `1 < a < 2R`,
/// minimizing the largest guard magnitude `z` subject to `ℓ_0(U) <= -T` and
/// `ℓ_j(U) <= z` for `1 <= j < R`, where `ℓ_j(U) = log2 |U(ω^(5^j))|` is linear
/// in the exponents. The presets up to `R = 16` solve this integer program
/// exactly over `|e_a| <= 40`, a range their optimal exponents do not reach;
/// the `R = 32` presets, over 31 exponents, are the best found in 15 minutes
/// of branch and bound, each within 0.81 bits of the proven bound on `z`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapingUnit {
    pub log_r: usize,
    pub target_bits: u32,
    pub coeffs: &'static [i16],
}

impl ShapingUnit {
    /// `R = 2`, 10.17 bits of suppression, guard at most 10.17 bits: the one
    /// guard class takes the whole suppression.
    pub const U4_10: Self = Self {
        log_r: 1,
        target_bits: 10,
        coeffs: &[577, -408, 0, 408],
    };

    /// `R = 2`, 15.26 bits of suppression, guard at most 15.26 bits, the most
    /// whose coefficients fit in `i16`.
    pub const U4_15: Self = Self {
        log_r: 1,
        target_bits: 15,
        coeffs: &[-19601, 13860, 0, -13860],
    };

    /// `R = 4`, 20.68 bits of suppression, guards at most 7.29 bits.
    pub const U8_20: Self = Self {
        log_r: 2,
        target_bits: 20,
        coeffs: &[9, -70, 0, 70, -9, -1, -39, -1],
    };

    /// `R = 4`, 30.21 bits of suppression, guards at most 11.14 bits.
    pub const U8_30: Self = Self {
        log_r: 2,
        target_bits: 30,
        coeffs: &[190, 0, -190, 8, 689, -1139, 689, 8],
    };

    /// `R = 4`, 41.36 bits of suppression, guards at most 14.58 bits. Past a
    /// target of 45 bits, the coefficients of `R = 4` units leave `i16`.
    pub const U8_40: Self = Self {
        log_r: 2,
        target_bits: 40,
        coeffs: &[0, 4182, 4337, 1164, -11485, 1164, 4337, 4182],
    };

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

    /// `R = 32`, 21.43 bits of suppression, guards at most 3.02 bits, over a
    /// proven lower bound of 2.40 bits.
    pub const U64_20: Self = Self {
        log_r: 5,
        target_bits: 20,
        coeffs: &[
            -1, -1, 1, 1, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0,
            -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
        ],
    };

    /// `R = 32`, 40.27 bits of suppression, guards at most 3.68 bits, over a
    /// proven lower bound of 2.88 bits.
    pub const U64_40: Self = Self {
        log_r: 5,
        target_bits: 40,
        coeffs: &[
            -1, 1, -1, 1, 1, 0, 0, -1, 0, -2, -1, 0, -1, 2, 0, 1, 0, 0, 0, 1, 0, 1, 0, -1, 1, 0, -1, 0, 0, 0, 1, 0, -1, 1, 0, -1,
            0, -1, 0, 0, 0, -1, 0, -2, 1, 0, 1, 2, 0, 1, 0, 0, -1, -1, 1, -1, 1, -1, 0, 0, 1, 0, 0, -1,
        ],
    };

    /// `R = 32`, 61.34 bits of suppression, guards at most 4.24 bits, over a
    /// proven lower bound of 3.58 bits.
    pub const U64_60: Self = Self {
        log_r: 5,
        target_bits: 60,
        coeffs: &[
            0, -1, 1, 1, -1, -1, 2, -1, -1, 1, 0, -2, 0, 2, 0, -1, 1, 1, -2, 1, 1, -1, -1, 1, 0, 0, 0, -1, 0, 0, -2, -2, -2, -2,
            1, 2, 1, 0, 1, 1, -1, 0, 0, 1, 3, 1, 0, 0, -1, 1, 1, 0, 1, 2, 1, -2, -2, -2, -2, 0, 0, -1, 0, 0,
        ],
    };

    /// `R = 32`, 80.32 bits of suppression, guards at most 4.90 bits, over a
    /// proven lower bound of 4.13 bits.
    pub const U64_80: Self = Self {
        log_r: 5,
        target_bits: 80,
        coeffs: &[
            0, 0, -2, 0, -1, -1, 0, 2, 3, 0, 1, 1, -1, 0, 0, 0, 2, 2, -5, -1, 1, 1, 2, -3, 2, 1, 1, -1, -5, 2, 2, 0, 0, 0, -1, 1,
            1, 0, 3, 2, 0, -1, -1, 0, -2, 0, 0, -3, 1, -5, 0, -2, 0, 0, 1, 0, -1, 0, 0, 2, 0, 5, -1, 3,
        ],
    };

    pub const ALL: [Self; 17] = [
        Self::U4_10,
        Self::U4_15,
        Self::U8_20,
        Self::U8_30,
        Self::U8_40,
        Self::U16_20,
        Self::U16_40,
        Self::U16_60,
        Self::U32_20,
        Self::U32_40,
        Self::U32_45,
        Self::U32_50,
        Self::U32_60,
        Self::U64_20,
        Self::U64_40,
        Self::U64_60,
        Self::U64_80,
    ];

    /// The replicated layout of this unit for a batch of `2^log_k` values.
    pub fn layout(&self, log_k: u8) -> poulpy_core::Result<ReplicatedLayout> {
        ReplicatedLayout::new(log_k, self.log_r as u8, self.coeffs)
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

/// Writes `Σ_t coeffs[t] · X^(t·stride) · x` in `(Z/2^64)[X]/(X^N + 1)` into `out`.
fn negacyclic_sparse(coeffs: &[u64], stride: usize, x: &[i64], out: &mut [i64]) {
    let n = x.len();
    out.fill(0);
    for (t, &coeff) in coeffs.iter().enumerate().filter(|&(_, &c)| c != 0) {
        let coeff = coeff as i64;
        let shift = t * stride;
        let (wrapped, direct) = out.split_at_mut(shift);
        for (o, &x) in direct.iter_mut().zip(&x[..n - shift]) {
            *o = o.wrapping_add(coeff.wrapping_mul(x));
        }
        for (o, &x) in wrapped.iter_mut().zip(&x[n - shift..]) {
            *o = o.wrapping_sub(coeff.wrapping_mul(x));
        }
    }
}

/// Writes `p(Y)*x` into `out` on the conjugate-invariant ring of degree
/// `N = x.len()`, `Y = X^stride` in `(Z/2^64)[X]/(X^(2N) + 1)`, for invariant
/// `p` and `x` given by their invariant coefficients: `p_0` and `p_t` for
/// `0 < t < R`, `R*stride = N`, standing for `p_t*(Y^t + Y^-t)`. Coefficient
/// `j` of `x` stands for `x_j*X^j` and, when `j > 0`, `x_j*X^-j`, so
/// `X^-j = -X^(2N-j)` reads `x_(2N-j) = -x_j` and `x_N = 0`.
fn invariant_sparse(p: &[u64], stride: usize, x: &[i64], out: &mut [i64]) {
    let n = x.len();
    let p0 = p[0] as i64;
    for (o, &x) in out.iter_mut().zip(x) {
        *o = p0.wrapping_mul(x);
    }
    for (t, &coeff) in p.iter().enumerate().skip(1).filter(|&(_, &c)| c != 0) {
        let coeff = coeff as i64;
        let m = t * stride;
        // `X^m * x`: coefficient `i` reads `x_(i-m)`, which is `x_(m-i)` below `m`.
        for (o, &x) in out[m..].iter_mut().zip(x) {
            *o = o.wrapping_add(coeff.wrapping_mul(x));
        }
        for (i, o) in out[..m].iter_mut().enumerate() {
            *o = o.wrapping_add(coeff.wrapping_mul(x[m - i]));
        }
        // `X^-m * x`: coefficient `i` reads `x_(i+m)`, zero at `i + m = N` and
        // `-x_(2N-i-m)` above it.
        for (o, &x) in out[..n - m].iter_mut().zip(&x[m..]) {
            *o = o.wrapping_add(coeff.wrapping_mul(x));
        }
        for (i, o) in out.iter_mut().enumerate().skip(n - m + 1) {
            *o = o.wrapping_sub(coeff.wrapping_mul(x[2 * n - i - m]));
        }
    }
}

/// Product in `(Z/2^64)[Y]/(Y^m + 1)`, for `m <= MAX_COEFFS`.
fn negacyclic_mul(a: &[u64], b: &[u64]) -> [u64; MAX_COEFFS] {
    let m = a.len();
    let mut out = [0u64; MAX_COEFFS];
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
fn inverse_mod_2_64(u: &[u64]) -> [u64; MAX_COEFFS] {
    let m = u.len();
    let mut one = [0u64; MAX_COEFFS];
    one[0] = 1;
    let mut v = one;
    loop {
        let uv = negacyclic_mul(u, &v[..m]);
        if uv == one {
            return v;
        }
        let mut correction = uv.map(u64::wrapping_neg);
        correction[0] = correction[0].wrapping_add(2);
        v = negacyclic_mul(&v[..m], &correction[..m]);
    }
}

/// Shaping depth of a product dividing by `2^log_delta` in `slot_layout`, `None` when it rounds plainly.
pub(crate) fn product_depth(slot_layout: SlotLayout, log_delta: usize) -> Option<usize> {
    match slot_layout {
        SlotLayout::Replicated(layout) => {
            Some(ShapingProfile::new(&layout).shaping_depth(log_delta)).filter(|&lambda| lambda > 0)
        }
        SlotLayout::Full => None,
    }
}

/// Input width consumed by the automorphism, including a shifted shaping temporary.
pub(crate) fn automorphism_input_k<Dst: CKKSInfos, Src: CKKSInfos>(dst: &Dst, src: &Src) -> TorusPrecision {
    if ckks_offset_unary(dst, src) != 0 {
        let lambda = product_depth(src.slot_layout(), src.log_delta()).unwrap_or(0);
        (dst.k().as_usize() + lambda).into()
    } else {
        src.k()
    }
}

/// Quantizes `coeffs` at `log_delta + λ` bits into `out`, then divides by `2^λ` with
/// shaped rounding, so that the quantization error is a multiple of the
/// layout's shaping polynomial up to an unshaped `2^-λ` remainder. `λ` is the
/// largest exact pass that keeps `log_delta + log_budget + λ` within `i128`;
/// a plaintext that leaves none is rejected. `low`, `h` and `gh` are buffers as
/// long as `coeffs`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn shaped_quantize_into<F, Rg: Ring>(
    op: &'static str,
    coeffs: &[F],
    layout: &ReplicatedLayout,
    log_delta: usize,
    log_budget: usize,
    out: &mut [i128],
    low: &mut [i64],
    h: &mut [i64],
    gh: &mut [i64],
) -> anyhow::Result<()>
where
    F: crate::api::CKKSEncodingScalar,
{
    let (profile, lambda) = shaping_pass::<Rg>(op, layout, coeffs.len(), log_delta, log_budget)?;
    for ((q, l), &x) in out.iter_mut().zip(low.iter_mut()).zip(coeffs) {
        *q = x
            .ckks_quantize(log_delta + lambda)
            .context("CKKS coefficient is not representable as an i128 at the plaintext scale")?;
        *l = *q as i64;
    }
    profile.shaped_multiple_host(lambda, low, h, gh);
    for (q, &m) in out.iter_mut().zip(gh.iter()) {
        *q = (*q - m as i128) >> lambda;
    }
    Ok(())
}

/// The shaping pass of [`shaped_quantize_into`] for `coeffs` and its shaping
/// depth, checked before any quantization and refused under `op`.
fn shaping_pass<Rg: Ring>(
    op: &'static str,
    layout: &ReplicatedLayout,
    n: usize,
    log_delta: usize,
    log_budget: usize,
) -> CKKSResult<(ShapingProfile<Rg>, usize)> {
    let invalid = |reason: String| -> crate::CKKSError { CKKSCompositionError::InvalidOperand { op, reason }.into() };
    let profile = ShapingProfile::<Rg>::for_op(op, layout)?;
    let period = profile.period();
    if !n.is_multiple_of(period) {
        return Err(invalid(format!(
            "a replicated layout of period {} needs a multiple of {period} coefficients",
            1usize << layout.log_r()
        )));
    }
    let lambda = profile.max_pass_bits().min(126usize.saturating_sub(log_delta + log_budget));
    if lambda == 0 {
        return Err(invalid(format!(
            "a replicated plaintext of {} torus bits leaves no shaping pass within i128",
            log_delta + log_budget
        )));
    }
    Ok((profile, lambda))
}

/// [`shaped_quantize_into`] class by class, on stack buffers: with `Y = X^stride`,
/// `stride = coeffs.len() / period`, the shaping product acts on each class
/// `r < stride` of coefficients `r + stride*t` as a negacyclic product of `2R`
/// coefficients in `Y`. On the conjugate-invariant ring the class holds its own
/// `R` invariant coefficients and, above them, the negated mirror ones `-x_(2N-i)`,
/// whose centered reduction is the negated reduction of the mirror, as the
/// whole-vector pass computes it. Every coefficient is quantized and checked
/// before the first `emit(r, values)`, which receives the `period` results of
/// class `r` in order of `t`. Bit for bit [`shaped_quantize_into`].
pub(crate) fn shaped_quantize_by_class<F, Rg: Ring>(
    coeffs: &[F],
    layout: &ReplicatedLayout,
    log_delta: usize,
    log_budget: usize,
    mut emit: impl FnMut(usize, &mut [i128]),
) -> anyhow::Result<()>
where
    F: crate::api::CKKSEncodingScalar,
{
    let n = coeffs.len();
    let (profile, lambda) = shaping_pass::<Rg>("ckks_quantize_coeffs_into", layout, n, log_delta, log_budget)?;
    let quantize = |x: F| {
        x.ckks_quantize(log_delta + lambda)
            .context("CKKS coefficient is not representable as an i128 at the plaintext scale")
    };
    for &x in coeffs {
        quantize(x)?;
    }
    let (period, len) = (profile.period(), profile.len());
    let stride = n / period;
    let mask = u64::MAX >> (64 - lambda);
    let centered = |x: i64| {
        let reduced = x as u64 & mask;
        if reduced > mask >> 1 {
            reduced as i64 - (1i64 << lambda)
        } else {
            reduced as i64
        }
    };
    let (mut q, mut low, mut h, mut gh) = (
        [0i128; MAX_COEFFS],
        [0i64; MAX_COEFFS],
        [0i64; MAX_COEFFS],
        [0i64; MAX_COEFFS],
    );
    for r in 0..stride {
        for (t, q) in q[..len].iter_mut().enumerate() {
            let i = r + stride * t;
            *q = if i < n {
                quantize(coeffs[i]).expect("quantized above")
            } else if i == n {
                0
            } else {
                -quantize(coeffs[2 * n - i]).expect("quantized above")
            };
        }
        for (l, &q) in low[..len].iter_mut().zip(&q[..len]) {
            *l = q as i64;
        }
        negacyclic_sparse(profile.inverse(), 1, &low[..len], &mut h[..len]);
        for (t, h) in h[..len].iter_mut().enumerate() {
            *h = if t < period {
                centered(*h)
            } else {
                centered(h.wrapping_neg()).wrapping_neg()
            };
        }
        negacyclic_sparse(&profile.shaping[..len], 1, &h[..len], &mut gh[..len]);
        for (q, &m) in q[..period].iter_mut().zip(&gh[..period]) {
            *q = (*q - m as i128) >> lambda;
        }
        emit(r, &mut q[..period]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_hal::layouts::ConjugateInvariant;

    fn pseudo_random(n: usize, seed: u64) -> Vec<i64> {
        (0..n as u64)
            .map(|i| (i + seed).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(i as u32 % 64) as i64)
            .collect()
    }

    #[test]
    fn conjugate_invariant_profiles() {
        for (unit, shift) in ShapingUnit::ALL
            .into_iter()
            .zip([0, 0, 2, 3, 4, 10, 4, 3, 24, 1, 20, 24, 0, 17, 4, 20, 41])
        {
            let layout = unit.layout(2).unwrap();
            let standard = ShapingProfile::new(&layout);
            let profile = ShapingProfile::<ConjugateInvariant>::try_new(&layout).unwrap();
            assert_eq!(ShapingProfile::<Standard>::try_new(&layout).unwrap(), standard);
            assert_eq!(profile.shift(), shift, "{unit:?}");
            assert_eq!(profile.period(), 1 << unit.log_r);
            assert!(is_invariant(profile.coefficients()), "{unit:?}");
            assert_eq!(profile.l1_norm(), standard.l1_norm());
            assert_eq!(profile.max_pass_bits(), standard.max_pass_bits());
            // Binary128 resolves the 80 bits of suppression of `U64_80` to a few 1e-9 bits.
            for (a, b) in profile.class_log_magnitudes().iter().zip(standard.class_log_magnitudes()) {
                assert!((a - b).abs() < 1e-8, "{unit:?}");
            }
            let u: Vec<u64> = profile.coefficients().iter().map(|&c| c as i64 as u64).collect();
            let product = negacyclic_mul(&u, profile.inverse());
            assert_eq!(product[0], 1);
            assert!(product[1..u.len()].iter().all(|&c| c == 0));
            // The inverse of an invariant unit is invariant.
            let (m, inverse) = (u.len(), profile.inverse());
            assert!(inverse[m / 2] == 0 && (1..m / 2).all(|j| inverse[m - j] == inverse[j].wrapping_neg()));
        }
    }

    #[test]
    fn conjugate_invariant_refuses_shapes_without_invariant_shift() {
        let refused = ReplicatedLayout::new(2, 1, &[3, 0, 2, 0]).unwrap();
        assert!(ShapingProfile::<ConjugateInvariant>::try_new(&refused).is_err());
        let invariant = ReplicatedLayout::new(2, 1, &[3, 1, 0, -1]).unwrap();
        assert_eq!(ShapingProfile::<ConjugateInvariant>::try_new(&invariant).unwrap().shift(), 0);
    }

    /// The invariant coefficients of `x` unfolded onto `Z[X]/(X^(2N)+1)`.
    fn unfold(x: &[i64]) -> Vec<i64> {
        let n = x.len();
        let mut ambient = vec![0i64; 2 * n];
        ambient[..n].copy_from_slice(x);
        for j in 1..n {
            ambient[2 * n - j] = x[j].wrapping_neg();
        }
        ambient
    }

    #[test]
    fn invariant_product_matches_ambient() {
        for unit in [ShapingUnit::U16_40, ShapingUnit::U32_45] {
            let profile = ShapingProfile::<ConjugateInvariant>::try_new(&unit.layout(2).unwrap()).unwrap();
            let r = profile.period();
            let shaping: Vec<u64> = profile.coefficients().iter().map(|&c| c as i64 as u64).collect();
            for p in [&shaping[..], profile.inverse()] {
                for n in [r, 4 * r] {
                    let x = pseudo_random(n, n as u64);
                    let mut ambient = vec![0i64; 2 * n];
                    negacyclic_sparse(p, n / r, &unfold(&x), &mut ambient);
                    let mut got = vec![0i64; n];
                    invariant_sparse(&p[..r], n / r, &x, &mut got);
                    assert_eq!(unfold(&got), ambient, "{unit:?}, n = {n}");
                }
            }
        }
    }

    /// The class-by-class quantizer equals the whole-vector one on both rings,
    /// down to shaping passes of one and two bits, whose centered reductions tie.
    #[test]
    fn shaped_quantize_by_class_matches_whole_vector() {
        by_class_matches_whole_vector::<Standard>();
        by_class_matches_whole_vector::<ConjugateInvariant>();
    }

    fn by_class_matches_whole_vector<Rg: Ring>() {
        let layout = ShapingUnit::U32_45.layout(2).unwrap();
        let period = ShapingProfile::<Rg>::try_new(&layout).unwrap().period();
        for n in [period, 8 * period] {
            let coeffs: Vec<f64> = pseudo_random(n, n as u64 + 7)
                .iter()
                .map(|&x| (x >> 40) as f64 / (1u64 << 23) as f64)
                .collect();
            // A full pass, then passes of two bits and one bit.
            for (log_delta, log_budget) in [(40, 20), (60, 64), (60, 65)] {
                let mut want = vec![0i128; n];
                let (mut low, mut h, mut gh) = (vec![0i64; n], vec![0i64; n], vec![0i64; n]);
                shaped_quantize_into::<_, Rg>(
                    "test", &coeffs, &layout, log_delta, log_budget, &mut want, &mut low, &mut h, &mut gh,
                )
                .unwrap();
                let mut got = vec![0i128; n];
                let stride = n / period;
                shaped_quantize_by_class::<_, Rg>(&coeffs, &layout, log_delta, log_budget, |r, values| {
                    for (t, &v) in values.iter().enumerate() {
                        got[r + stride * t] = v;
                    }
                })
                .unwrap();
                assert!(
                    got == want,
                    "invariant = {}, n = {n}, log_delta = {log_delta}, log_budget = {log_budget}",
                    conjugate_invariant::<Rg>()
                );
            }
        }
    }

    #[test]
    fn shaped_multiple_is_divisible() {
        let layout = ShapingUnit::U32_45.layout(2).unwrap();
        divisible(&ShapingProfile::new(&layout));
        divisible(&ShapingProfile::<ConjugateInvariant>::try_new(&layout).unwrap());
    }

    fn divisible<Rg: Ring>(profile: &ShapingProfile<Rg>) {
        let lambda = 40;
        let n = 4 * profile.period();
        let c = pseudo_random(n, 7);
        let (mut h, mut gh) = (vec![0; n], vec![0; n]);
        profile.shaped_multiple_host(lambda, &c, &mut h, &mut gh);
        assert!(h.iter().all(|&x| x.unsigned_abs() <= 1 << (lambda - 1)));
        assert!(
            c.iter()
                .zip(&gh)
                .all(|(&c, &gh)| c.wrapping_sub(gh).trailing_zeros() as usize >= lambda),
            "invariant = {}",
            conjugate_invariant::<Rg>()
        );
    }

    #[test]
    fn preset_profiles() {
        for (unit, suppression, guard, l1) in [
            (ShapingUnit::U4_10, 10.172, 10.172, 1393),
            (ShapingUnit::U4_15, 15.259, 15.259, 47321),
            (ShapingUnit::U8_20, 20.680, 7.289, 199),
            (ShapingUnit::U8_30, 30.212, 11.139, 2913),
            (ShapingUnit::U8_40, 41.360, 14.578, 30851),
            (ShapingUnit::U16_20, 20.076, 4.005, 25),
            (ShapingUnit::U16_40, 40.842, 7.058, 169),
            (ShapingUnit::U16_60, 61.996, 9.903, 1643),
            (ShapingUnit::U32_20, 23.498, 3.138, 21),
            (ShapingUnit::U32_40, 40.994, 4.253, 51),
            (ShapingUnit::U32_45, 45.955, 4.594, 59),
            (ShapingUnit::U32_50, 50.207, 4.742, 61),
            (ShapingUnit::U32_60, 67.067, 5.692, 129),
            (ShapingUnit::U64_20, 21.425, 3.024, 17),
            (ShapingUnit::U64_40, 40.270, 3.676, 39),
            (ShapingUnit::U64_60, 61.337, 4.240, 61),
            (ShapingUnit::U64_80, 80.318, 4.897, 79),
        ] {
            let profile = ShapingProfile::new(&unit.layout(2).unwrap());
            assert!((profile.suppression_bits() - suppression).abs() < 5e-4, "{unit:?}");
            assert!(profile.suppression_bits() >= unit.target_bits as f64, "{unit:?}");
            assert!((profile.guard_bits() - guard).abs() < 5e-4, "{unit:?}");
            assert_eq!(profile.l1_norm(), l1, "{unit:?}");
            // A unit has norm one, so its class profile sums to zero, up to the
            // few 1e-9 bits to which binary128 resolves 80 bits of suppression.
            assert!(profile.class_log_magnitudes().iter().sum::<f64>().abs() < 1e-8, "{unit:?}");
        }
    }

    #[test]
    fn pass_size_and_shaping_depth() {
        let profile = ShapingProfile::new(&ShapingUnit::U32_45.layout(4).unwrap());
        assert_eq!(profile.max_pass_bits(), 58);
        assert_eq!(profile.shaping_depth(20), 44);
        assert_eq!(profile.shaping_depth(40), 58);
        assert_eq!(
            ShapingProfile::new(&ShapingUnit::U16_60.layout(4).unwrap()).max_pass_bits(),
            53
        );
    }

    fn negacyclic_mul_dense(a: &[u64], b: &[u64]) -> Vec<u64> {
        let m = a.len();
        let mut out = vec![0u64; m];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                let p = x.wrapping_mul(y);
                if i + j < m {
                    out[i + j] = out[i + j].wrapping_add(p);
                } else {
                    out[i + j - m] = out[i + j - m].wrapping_sub(p);
                }
            }
        }
        out
    }

    #[test]
    fn sparse_product_matches_dense() {
        let n = 64;
        let x: Vec<i64> = (0..n as i64)
            .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64).rotate_left(i as u32))
            .collect();
        let xu: Vec<u64> = x.iter().map(|&v| v as u64).collect();
        for (coeffs, stride) in [
            (vec![3u64, 0, u64::MAX, 7], 16),
            (vec![1, 2, 0, 5, 0, 0, 9, u64::MAX - 4], 8),
            (vec![0, 11], 1),
        ] {
            let mut dense = vec![0u64; n];
            for (t, &c) in coeffs.iter().enumerate() {
                dense[t * stride] = c;
            }
            let expected: Vec<i64> = negacyclic_mul_dense(&dense, &xu).into_iter().map(|v| v as i64).collect();
            let mut got = vec![0; n];
            negacyclic_sparse(&coeffs, stride, &x, &mut got);
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn inverse_modulo_2_64() {
        let non_unit = ReplicatedLayout::new(2, 1, &[3, 0, 2, 0]).unwrap();
        let layouts = ShapingUnit::ALL.map(|unit| unit.layout(2).unwrap());
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
