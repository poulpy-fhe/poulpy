//! Portable quad-precision (IEEE 754 binary128) scalar for high-precision CKKS.
//!
//! [`Quad`] is a `#[repr(transparent)]` newtype over the nightly primitive
//! `f128`, implementing the `num_traits` surface (`Float`, `FloatConst`,
//! `From`/`ToPrimitive`) that `num_traits` does not provide for `f128` itself.
//! It replaces the `f128` crate (libquadmath, x86_64 Linux GNU only) as the
//! *type*, so storage and basic arithmetic work on every supported CPU target.
//!
//! `+ - * /`, comparisons and conversions use compiler-builtins soft float.
//! Linux GNU targets with a native binary128 libm use the primitive operations;
//! other targets use pure-Rust `libm` for exact/algebraic operations and a
//! guarded arbitrary-precision implementation for transcendentals. This avoids
//! Darwin's absent binary128 libm without changing the public type or its ABI.
//! `backing` is the single seam where that routing is decided.
//!
//! Under the `libquadmath` feature on x86_64 Linux GNU, the same set is routed
//! through libquadmath. The feature is deliberately a no-op elsewhere.

use core::cmp::Ordering;
use core::fmt;
use core::iter::{Product, Sum};
use core::num::FpCategory;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use num_traits::{Float, FloatConst, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};

/// IEEE 754 binary128 (quadruple precision) float, portable across the
/// `poulpy` CPU backends (x86_64 and aarch64).
#[derive(Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Quad(pub f128);

// `Quad` is transparent over IEEE-754 binary128. Every bit pattern is a valid
// floating-point value, so it is safe to store and transfer as plain bytes.
unsafe impl bytemuck::Zeroable for Quad {}
unsafe impl bytemuck::Pod for Quad {}

impl Quad {
    /// Wraps a primitive `f128`.
    #[inline]
    pub const fn new(x: f128) -> Self {
        Quad(x)
    }

    /// Returns the inner primitive `f128`.
    #[inline]
    pub const fn get(self) -> f128 {
        self.0
    }

    /// Raw IEEE 754 binary128 bit pattern.
    #[inline]
    pub fn to_bits(self) -> u128 {
        self.0.to_bits()
    }

    /// Builds a value from a raw IEEE 754 binary128 bit pattern.
    #[inline]
    pub fn from_bits(bits: u128) -> Self {
        Quad(f128::from_bits(bits))
    }
}

impl Default for Quad {
    #[inline]
    fn default() -> Self {
        Quad(0.0)
    }
}

// `f128` has no `Display`/`Debug` yet; render via `f64` plus the exact bits.
impl fmt::Debug for Quad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quad({:e} | {:#034x})", self.0 as f64, self.to_bits())
    }
}

impl fmt::Display for Quad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0 as f64)
    }
}

macro_rules! bin_op {
    ($Trait:ident, $method:ident, $op:tt) => {
        impl $Trait for Quad {
            type Output = Quad;
            #[inline]
            fn $method(self, rhs: Quad) -> Quad {
                Quad(self.0 $op rhs.0)
            }
        }
    };
}
bin_op!(Add, add, +);
bin_op!(Sub, sub, -);
bin_op!(Mul, mul, *);
bin_op!(Div, div, /);

// `%` is the one operator that lowers to a libm call (`fmodf128` / `fmodl`)
// rather than a soft-float builtin, so it goes through `backing` too.
impl Rem for Quad {
    type Output = Quad;
    #[inline]
    fn rem(self, rhs: Quad) -> Quad {
        Quad(backing::rem(self.0, rhs.0))
    }
}

impl Neg for Quad {
    type Output = Quad;
    #[inline]
    fn neg(self) -> Quad {
        Quad(-self.0)
    }
}

macro_rules! assign_op {
    ($Trait:ident, $method:ident, $op:tt) => {
        impl $Trait for Quad {
            #[inline]
            fn $method(&mut self, rhs: Quad) {
                self.0 $op rhs.0;
            }
        }
    };
}
assign_op!(AddAssign, add_assign, +=);
assign_op!(SubAssign, sub_assign, -=);
assign_op!(MulAssign, mul_assign, *=);
assign_op!(DivAssign, div_assign, /=);

impl RemAssign for Quad {
    #[inline]
    fn rem_assign(&mut self, rhs: Quad) {
        self.0 = backing::rem(self.0, rhs.0);
    }
}

impl Sum for Quad {
    #[inline]
    fn sum<I: Iterator<Item = Quad>>(iter: I) -> Quad {
        iter.fold(Quad(0.0), Add::add)
    }
}
impl<'a> Sum<&'a Quad> for Quad {
    #[inline]
    fn sum<I: Iterator<Item = &'a Quad>>(iter: I) -> Quad {
        iter.fold(Quad(0.0), |acc, &x| acc + x)
    }
}
impl Product for Quad {
    #[inline]
    fn product<I: Iterator<Item = Quad>>(iter: I) -> Quad {
        iter.fold(Quad(1.0), Mul::mul)
    }
}
impl<'a> Product<&'a Quad> for Quad {
    #[inline]
    fn product<I: Iterator<Item = &'a Quad>>(iter: I) -> Quad {
        iter.fold(Quad(1.0), |acc, &x| acc * x)
    }
}

impl Zero for Quad {
    #[inline]
    fn zero() -> Quad {
        Quad(0.0)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl One for Quad {
    #[inline]
    fn one() -> Quad {
        Quad(1.0)
    }
}

impl Num for Quad {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;
    /// Parses via `f64` then widens (not full-precision); only here to satisfy `Num`.
    #[inline]
    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(src, radix).map(|x| Quad(x as f128))
    }
}

impl NumCast for Quad {
    #[inline]
    fn from<T: ToPrimitive>(n: T) -> Option<Quad> {
        n.to_f64().map(|x| Quad(x as f128))
    }
}

// Cast directly: the `num_traits` defaults would route through `i64`/`f64` and
// truncate. Float `as` int casts saturate, which is fine for `poulpy`'s in-range values.
impl ToPrimitive for Quad {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        Some(self.0 as i64)
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        Some(self.0 as u64)
    }
    #[inline]
    fn to_i128(&self) -> Option<i128> {
        Some(self.0 as i128)
    }
    #[inline]
    fn to_u128(&self) -> Option<u128> {
        Some(self.0 as u128)
    }
    #[inline]
    fn to_isize(&self) -> Option<isize> {
        Some(self.0 as isize)
    }
    #[inline]
    fn to_usize(&self) -> Option<usize> {
        Some(self.0 as usize)
    }
    #[inline]
    fn to_i32(&self) -> Option<i32> {
        Some(self.0 as i32)
    }
    #[inline]
    fn to_u32(&self) -> Option<u32> {
        Some(self.0 as u32)
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        Some(self.0 as f32)
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        Some(self.0 as f64)
    }
}

impl FromPrimitive for Quad {
    #[inline]
    fn from_i64(n: i64) -> Option<Quad> {
        Some(Quad(n as f128))
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Quad> {
        Some(Quad(n as f128))
    }
    #[inline]
    fn from_i128(n: i128) -> Option<Quad> {
        Some(Quad(n as f128))
    }
    #[inline]
    fn from_u128(n: u128) -> Option<Quad> {
        Some(Quad(n as f128))
    }
    #[inline]
    fn from_isize(n: isize) -> Option<Quad> {
        Some(Quad(n as f128))
    }
    #[inline]
    fn from_usize(n: usize) -> Option<Quad> {
        Some(Quad(n as f128))
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Quad> {
        Some(Quad(n as f128))
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Quad> {
        Some(Quad(n as f128))
    }
}

/// Lossy `integer_decode`: binary128's 113-bit significand can't fit the 64-bit
/// mantissa, so the low 49 bits are dropped. Only here to satisfy `Float`; unused.
fn integer_decode_f128(x: f128) -> (u64, i16, i8) {
    const SHIFT: i16 = 49; // 113 significand bits - 64
    let bits = x.to_bits();
    let sign: i8 = if (bits >> 127) & 1 == 1 { -1 } else { 1 };
    let exponent_bits = ((bits >> 112) & 0x7fff) as i16;
    let fraction = bits & ((1u128 << 112) - 1);
    let (significand, exponent) = if exponent_bits == 0 {
        (fraction, -16382 - 112)
    } else {
        (fraction | (1u128 << 112), exponent_bits - 16383 - 112)
    };
    (((significand >> SHIFT) as u64), exponent + SHIFT, sign)
}

macro_rules! fwd_unary {
    ($($method:ident),+ $(,)?) => {
        $(
            #[inline]
            fn $method(self) -> Quad {
                Quad(self.0.$method())
            }
        )+
    };
}

/// Single routing seam for every `Quad` method that lowers to a libm call.
///
/// The target predicate is written here and nowhere else. Callers go through
/// [`routed_unary`] / [`routed_binary`], so adding a third backing is local to
/// this module.
mod backing {
    #[cfg(all(feature = "libquadmath", target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    pub(super) use quadmath::*;

    #[cfg(all(
        target_os = "linux",
        target_env = "gnu",
        any(target_arch = "x86_64", target_arch = "aarch64"),
        not(all(feature = "libquadmath", target_arch = "x86_64")),
    ))]
    pub(super) use primitive::*;

    // Fail closed: only targets positively known to provide binary128 libm use
    // the primitive methods. In particular, both Darwin architectures land here.
    #[cfg(not(all(
        target_os = "linux",
        target_env = "gnu",
        any(target_arch = "x86_64", target_arch = "aarch64"),
    )))]
    pub(super) use portable::*;

    /// Nightly primitive `f128`. Correct wherever `long double` is binary128.
    #[cfg(all(
        target_os = "linux",
        target_env = "gnu",
        any(target_arch = "x86_64", target_arch = "aarch64"),
        not(all(feature = "libquadmath", target_arch = "x86_64")),
    ))]
    mod primitive {
        macro_rules! unary {
            ($($f:ident),+ $(,)?) => {
                $(
                    #[inline(always)]
                    pub fn $f(x: f128) -> f128 {
                        x.$f()
                    }
                )+
            };
        }

        macro_rules! binary {
            ($($f:ident),+ $(,)?) => {
                $(
                    #[inline(always)]
                    pub fn $f(x: f128, y: f128) -> f128 {
                        x.$f(y)
                    }
                )+
            };
        }

        unary!(
            floor, ceil, round, trunc, fract, sqrt, cbrt, exp, exp2, ln, log2, log10, sin, cos, tan, asin, acos, atan, exp_m1,
            ln_1p, sinh, cosh, tanh, asinh, acosh, atanh,
        );
        binary!(powf, log, hypot, atan2);

        #[inline(always)]
        pub fn rem(x: f128, y: f128) -> f128 {
            x % y
        }

        #[inline(always)]
        pub fn mul_add(x: f128, a: f128, b: f128) -> f128 {
            x.mul_add(a, b)
        }

        #[inline(always)]
        pub fn sin_cos(x: f128) -> (f128, f128) {
            x.sin_cos()
        }
    }

    /// libquadmath, reached by bit-casting to the `f128` crate's binary128 type
    /// (identical layout) and back. Storage never leaves the primitive.
    #[cfg(all(feature = "libquadmath", target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    mod quadmath {
        use num_traits::Float;

        type Lq = ::f128::f128;

        #[inline(always)]
        fn to_lq(x: f128) -> Lq {
            // `Lq` is a transparent 16-byte binary128; every bit pattern is valid.
            unsafe { core::mem::transmute::<u128, Lq>(x.to_bits()) }
        }

        #[inline(always)]
        fn from_lq(x: Lq) -> f128 {
            f128::from_bits(unsafe { core::mem::transmute::<Lq, u128>(x) })
        }

        macro_rules! unary {
            ($($f:ident),+ $(,)?) => {
                $(
                    #[inline(always)]
                    pub fn $f(x: f128) -> f128 {
                        from_lq(Float::$f(to_lq(x)))
                    }
                )+
            };
        }

        macro_rules! binary {
            ($($f:ident),+ $(,)?) => {
                $(
                    #[inline(always)]
                    pub fn $f(x: f128, y: f128) -> f128 {
                        from_lq(Float::$f(to_lq(x), to_lq(y)))
                    }
                )+
            };
        }

        unary!(
            floor, ceil, round, trunc, fract, sqrt, cbrt, exp, exp2, ln, log2, log10, sin, cos, tan, asin, acos, atan, exp_m1,
            ln_1p, sinh, cosh, tanh, asinh, acosh, atanh,
        );
        binary!(powf, log, hypot, atan2);

        #[inline(always)]
        pub fn rem(x: f128, y: f128) -> f128 {
            from_lq(to_lq(x) % to_lq(y))
        }

        #[inline(always)]
        pub fn mul_add(x: f128, a: f128, b: f128) -> f128 {
            from_lq(Float::mul_add(to_lq(x), to_lq(a), to_lq(b)))
        }

        #[inline(always)]
        pub fn sin_cos(x: f128) -> (f128, f128) {
            let (s, c) = Float::sin_cos(to_lq(x));
            (from_lq(s), from_lq(c))
        }
    }

    /// Pure-Rust binary128 math for targets without a binary128 system libm.
    ///
    /// `libm` supplies the correctly-rounded algebraic operations. Astro-float
    /// evaluates the transcendental surface with 79 guard bits, after which
    /// `from_big` performs one round-to-nearest-even conversion to binary128.
    /// It is used instead of the existing Dashu dependency because it also
    /// covers the hyperbolic `Float` surface without bespoke approximations.
    /// The module is also built by tests on Linux so CI can validate it against
    /// libquadmath without needing to emulate Darwin.
    #[cfg(any(
        test,
        not(all(
            target_os = "linux",
            target_env = "gnu",
            any(target_arch = "x86_64", target_arch = "aarch64"),
        )),
    ))]
    #[cfg_attr(test, allow(dead_code))]
    pub(super) mod portable {
        use std::cell::RefCell;

        use astro_float_num::{BigFloat, Consts, INF_NEG, INF_POS, NAN, RoundingMode, Sign, WORD_BIT_SIZE};

        const WORK_PRECISION: usize = 192;
        const _: () = assert!(WORK_PRECISION >= 113);
        const ROUNDING: RoundingMode = RoundingMode::ToEven;
        const SIGN_MASK: u128 = 1 << 127;
        const FRACTION_MASK: u128 = (1 << 112) - 1;

        std::thread_local! {
            static CONSTANTS: RefCell<Consts> = RefCell::new(
                Consts::new().expect("failed to initialize portable f128 constants"),
            );
        }

        #[inline]
        fn with_constants<T>(f: impl FnOnce(&mut Consts) -> T) -> T {
            CONSTANTS.with(|constants| f(&mut constants.borrow_mut()))
        }

        /// Exact conversion from an IEEE binary128 value to an Astro value.
        fn to_big(x: f128, precision: usize) -> BigFloat {
            let bits = x.to_bits();
            let negative = bits & SIGN_MASK != 0;
            let exponent = ((bits >> 112) & 0x7fff) as i32;
            let fraction = bits & FRACTION_MASK;

            if exponent == 0x7fff {
                return if fraction != 0 {
                    NAN.clone()
                } else if negative {
                    INF_NEG.clone()
                } else {
                    INF_POS.clone()
                };
            }

            let (significand, scale) = if exponent == 0 {
                (fraction, -16494)
            } else {
                ((1u128 << 112) | fraction, exponent - 16383 - 112)
            };

            let mut value = BigFloat::from_u128(significand, precision.max(128));
            if significand != 0 {
                let exponent = value.exponent().expect("finite Astro value") + scale;
                value.set_exponent(exponent);
            }
            value.set_sign(if negative { Sign::Neg } else { Sign::Pos });
            value
        }

        #[inline]
        fn mantissa_bit(words: &[astro_float_num::Word], bit: usize) -> bool {
            let word = bit / WORD_BIT_SIZE;
            word < words.len() && (words[word] & (1 << (bit % WORD_BIT_SIZE))) != 0
        }

        /// Shift a little-endian mantissa right and round it to nearest-even.
        fn rounded_shift(words: &[astro_float_num::Word], shift: usize) -> u128 {
            let mut value = 0u128;
            for bit in 0..128 {
                if mantissa_bit(words, shift + bit) {
                    value |= 1u128 << bit;
                }
            }

            if shift != 0 {
                let halfway = mantissa_bit(words, shift - 1);
                let sticky = (0..shift - 1).any(|bit| mantissa_bit(words, bit));
                if halfway && (sticky || value & 1 != 0) {
                    value += 1;
                }
            }
            value
        }

        /// Round an Astro value once into the IEEE binary128 interchange format.
        fn from_big(value: &BigFloat) -> f128 {
            if value.is_nan() {
                return f128::NAN;
            }
            if value.is_inf_pos() {
                return f128::INFINITY;
            }
            if value.is_inf_neg() {
                return f128::NEG_INFINITY;
            }

            let (words, _significant, sign, mut exponent, _) = value.as_raw_parts().expect("finite Astro value");
            let sign_bits = if sign == Sign::Neg { SIGN_MASK } else { 0 };
            if value.is_zero() {
                return f128::from_bits(sign_bits);
            }
            if exponent > 16384 {
                return f128::from_bits(sign_bits | (0x7fffu128 << 112));
            }

            let precision = words.len() * WORD_BIT_SIZE;
            if exponent >= -16381 {
                let mut significand = rounded_shift(words, precision - 113);
                if significand == 1u128 << 113 {
                    significand >>= 1;
                    exponent += 1;
                }
                if exponent > 16384 {
                    return f128::from_bits(sign_bits | (0x7fffu128 << 112));
                }
                let biased = (exponent + 16382) as u128;
                f128::from_bits(sign_bits | (biased << 112) | (significand & FRACTION_MASK))
            } else {
                // Subnormal value in units of 2^-16494. The shift is nonnegative
                // for every Astro precision produced by this module.
                let shift = (precision as i32 - 16494 - exponent).max(0) as usize;
                let significand = rounded_shift(words, shift);
                if significand >= 1u128 << 112 {
                    f128::from_bits(sign_bits | (1u128 << 112))
                } else {
                    f128::from_bits(sign_bits | significand)
                }
            }
        }

        #[cfg(test)]
        pub(crate) fn conversion_roundtrip(x: f128) -> f128 {
            from_big(&to_big(x, WORK_PRECISION))
        }

        macro_rules! unary_with_constants {
            ($($name:ident),+ $(,)?) => {
                $(
                    #[inline]
                    pub fn $name(x: f128) -> f128 {
                        let x = to_big(x, WORK_PRECISION);
                        from_big(&with_constants(|constants| {
                            x.$name(WORK_PRECISION, ROUNDING, constants)
                        }))
                    }
                )+
            };
        }

        unary_with_constants!(
            exp, ln, log2, log10, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh
        );

        #[inline]
        pub fn floor(x: f128) -> f128 {
            libm::floorf128(x)
        }

        #[inline]
        pub fn ceil(x: f128) -> f128 {
            libm::ceilf128(x)
        }

        #[inline]
        pub fn round(x: f128) -> f128 {
            libm::roundf128(x)
        }

        #[inline]
        pub fn trunc(x: f128) -> f128 {
            libm::truncf128(x)
        }

        #[inline]
        pub fn fract(x: f128) -> f128 {
            libm::fmodf128(x, 1.0)
        }

        #[inline]
        pub fn sqrt(x: f128) -> f128 {
            libm::sqrtf128(x)
        }

        #[inline]
        pub fn cbrt(x: f128) -> f128 {
            from_big(&to_big(x, WORK_PRECISION).cbrt(WORK_PRECISION, ROUNDING))
        }

        #[inline]
        pub fn exp2(x: f128) -> f128 {
            powf(2.0, x)
        }

        /// Cancellation-safe `exp(x) - 1` for small `x`.
        pub fn exp_m1(x: f128) -> f128 {
            if x == 0.0 {
                return x;
            }
            let x_big = to_big(x, WORK_PRECISION);
            if x.abs() >= 0.5 {
                let one = BigFloat::from_u8(1, WORK_PRECISION);
                return from_big(&with_constants(|constants| {
                    x_big
                        .exp(WORK_PRECISION, ROUNDING, constants)
                        .sub(&one, WORK_PRECISION, ROUNDING)
                }));
            }

            let mut term = x_big.clone();
            let mut sum = x_big.clone();
            for n in 2..=512u64 {
                let divisor = BigFloat::from_u64(n, WORK_PRECISION);
                term = term
                    .mul(&x_big, WORK_PRECISION, ROUNDING)
                    .div(&divisor, WORK_PRECISION, ROUNDING);
                let next = sum.add(&term, WORK_PRECISION, ROUNDING);
                if next.cmp(&sum) == Some(0) {
                    break;
                }
                sum = next;
            }
            from_big(&sum)
        }

        /// Cancellation-safe `ln(1 + x)`, using `2*atanh(x/(2+x))` near zero.
        pub fn ln_1p(x: f128) -> f128 {
            if x == 0.0 {
                return x;
            }
            let x_big = to_big(x, WORK_PRECISION);
            let one = BigFloat::from_u8(1, WORK_PRECISION);
            if x.abs() >= 0.5 {
                return from_big(&with_constants(|constants| {
                    one.add(&x_big, WORK_PRECISION, ROUNDING)
                        .ln(WORK_PRECISION, ROUNDING, constants)
                }));
            }

            let two = BigFloat::from_u8(2, WORK_PRECISION);
            let z = x_big.div(&two.add(&x_big, WORK_PRECISION, ROUNDING), WORK_PRECISION, ROUNDING);
            let z2 = z.mul(&z, WORK_PRECISION, ROUNDING);
            let mut power = z.clone();
            let mut sum = z;
            for denominator in (3..=1025u64).step_by(2) {
                power = power.mul(&z2, WORK_PRECISION, ROUNDING);
                let term = power.div(&BigFloat::from_u64(denominator, WORK_PRECISION), WORK_PRECISION, ROUNDING);
                let next = sum.add(&term, WORK_PRECISION, ROUNDING);
                if next.cmp(&sum) == Some(0) {
                    break;
                }
                sum = next;
            }
            from_big(&sum.mul(&two, WORK_PRECISION, ROUNDING))
        }

        #[inline]
        pub fn powf(x: f128, y: f128) -> f128 {
            // Match the IEEE special cases before Astro propagates a NaN base.
            if y == 0.0 || x == 1.0 {
                return 1.0;
            }
            if x.is_nan() || y.is_nan() {
                return f128::NAN;
            }
            if y.is_infinite() {
                let magnitude = x.abs();
                if magnitude == 1.0 {
                    return 1.0;
                }
                return if (magnitude > 1.0) == y.is_sign_positive() {
                    f128::INFINITY
                } else {
                    0.0
                };
            }
            if x.is_infinite() {
                let odd_integer = y == trunc(y) && rem(y, 2.0).abs() == 1.0;
                let negative = x.is_sign_negative() && odd_integer;
                if y.is_sign_positive() {
                    return if negative { f128::NEG_INFINITY } else { f128::INFINITY };
                }
                return if negative { f128::from_bits(SIGN_MASK) } else { 0.0 };
            }
            let x = to_big(x, WORK_PRECISION);
            let y = to_big(y, WORK_PRECISION);
            from_big(&with_constants(|constants| x.pow(&y, WORK_PRECISION, ROUNDING, constants)))
        }

        #[inline]
        pub fn log(x: f128, base: f128) -> f128 {
            let x = to_big(x, WORK_PRECISION);
            let base = to_big(base, WORK_PRECISION);
            from_big(&with_constants(|constants| {
                // Astro's two-argument `log` can fail to converge for simple
                // exact powers (for example, log(4, 2)). Its unary `ln`
                // implementation does not have that failure mode.
                let numerator = x.ln(WORK_PRECISION, ROUNDING, constants);
                let denominator = base.ln(WORK_PRECISION, ROUNDING, constants);
                numerator.div(&denominator, WORK_PRECISION, ROUNDING)
            }))
        }

        pub fn hypot(x: f128, y: f128) -> f128 {
            if x.is_infinite() || y.is_infinite() {
                return f128::INFINITY;
            }
            if x.is_nan() || y.is_nan() {
                return f128::NAN;
            }
            let x = to_big(x, WORK_PRECISION);
            let y = to_big(y, WORK_PRECISION);
            let sum = x
                .mul(&x, WORK_PRECISION, ROUNDING)
                .add(&y.mul(&y, WORK_PRECISION, ROUNDING), WORK_PRECISION, ROUNDING);
            from_big(&sum.sqrt(WORK_PRECISION, ROUNDING))
        }

        pub fn atan2(y: f128, x: f128) -> f128 {
            if x.is_nan() || y.is_nan() {
                return f128::NAN;
            }
            if y == 0.0 {
                return if x.is_sign_negative() {
                    core::f128::consts::PI.copysign(y)
                } else {
                    y
                };
            }
            if x == 0.0 {
                return core::f128::consts::FRAC_PI_2.copysign(y);
            }
            if y.is_infinite() {
                if x.is_infinite() {
                    let angle = if x.is_sign_negative() {
                        3.0 * core::f128::consts::FRAC_PI_4
                    } else {
                        core::f128::consts::FRAC_PI_4
                    };
                    return angle.copysign(y);
                }
                return core::f128::consts::FRAC_PI_2.copysign(y);
            }
            if x.is_infinite() {
                return if x.is_sign_negative() {
                    core::f128::consts::PI.copysign(y)
                } else {
                    0.0f128.copysign(y)
                };
            }

            let y_big = to_big(y, WORK_PRECISION);
            let x_big = to_big(x, WORK_PRECISION);
            from_big(&with_constants(|constants| {
                let angle = y_big
                    .div(&x_big, WORK_PRECISION, ROUNDING)
                    .atan(WORK_PRECISION, ROUNDING, constants);
                if x.is_sign_negative() {
                    let pi = constants.pi(WORK_PRECISION, ROUNDING);
                    if y.is_sign_negative() {
                        angle.sub(&pi, WORK_PRECISION, ROUNDING)
                    } else {
                        angle.add(&pi, WORK_PRECISION, ROUNDING)
                    }
                } else {
                    angle
                }
            }))
        }

        #[inline]
        pub fn rem(x: f128, y: f128) -> f128 {
            libm::fmodf128(x, y)
        }

        #[inline]
        pub fn mul_add(x: f128, a: f128, b: f128) -> f128 {
            libm::fmaf128(x, a, b)
        }

        #[inline]
        pub fn sin_cos(x: f128) -> (f128, f128) {
            let x = to_big(x, WORK_PRECISION);
            with_constants(|constants| {
                (
                    from_big(&x.sin(WORK_PRECISION, ROUNDING, constants)),
                    from_big(&x.cos(WORK_PRECISION, ROUNDING, constants)),
                )
            })
        }
    }
}

/// Unary methods that lower to a libm call, routed through [`backing`].
macro_rules! routed_unary {
    ($($method:ident),+ $(,)?) => {
        $(
            #[inline]
            fn $method(self) -> Quad {
                Quad(backing::$method(self.0))
            }
        )+
    };
}

/// Binary methods that lower to a libm call, routed through [`backing`].
macro_rules! routed_binary {
    ($($method:ident),+ $(,)?) => {
        $(
            #[inline]
            fn $method(self, other: Quad) -> Quad {
                Quad(backing::$method(self.0, other.0))
            }
        )+
    };
}

macro_rules! fwd_predicate {
    ($($method:ident),+ $(,)?) => {
        $(
            #[inline]
            fn $method(self) -> bool {
                self.0.$method()
            }
        )+
    };
}

macro_rules! fwd_const {
    ($($method:ident => $value:expr),+ $(,)?) => {
        $(
            #[inline]
            fn $method() -> Quad {
                Quad($value)
            }
        )+
    };
}

impl Float for Quad {
    fwd_const! {
        nan => f128::NAN,
        infinity => f128::INFINITY,
        neg_infinity => f128::NEG_INFINITY,
        neg_zero => -0.0,
        min_value => f128::MIN,
        min_positive_value => f128::MIN_POSITIVE,
        epsilon => f128::EPSILON,
        max_value => f128::MAX,
    }

    fwd_predicate!(is_nan, is_infinite, is_finite, is_normal, is_sign_positive, is_sign_negative);

    // Pure bit manipulation: no libm call, always the primitive.
    fwd_unary!(abs, signum, recip);

    // Everything that lowers to a libm call, including the rounding family.
    routed_unary!(
        floor, ceil, round, trunc, fract, sqrt, cbrt, exp, exp2, ln, log2, log10, sin, cos, tan, asin, acos, atan, exp_m1, ln_1p,
        sinh, cosh, tanh, asinh, acosh, atanh,
    );

    #[inline]
    fn classify(self) -> FpCategory {
        self.0.classify()
    }

    #[inline]
    fn mul_add(self, a: Quad, b: Quad) -> Quad {
        Quad(backing::mul_add(self.0, a.0, b.0))
    }

    // `__powitf2`, a compiler_builtins soft-float routine: correct everywhere.
    #[inline]
    fn powi(self, n: i32) -> Quad {
        Quad(self.0.powi(n))
    }

    routed_binary!(powf, log);

    #[inline]
    fn max(self, other: Quad) -> Quad {
        Quad(self.0.max(other.0))
    }

    #[inline]
    fn min(self, other: Quad) -> Quad {
        Quad(self.0.min(other.0))
    }

    /// Positive difference; `f128` has no `abs_sub`, so implemented directly.
    #[inline]
    fn abs_sub(self, other: Quad) -> Quad {
        if self.0 > other.0 { Quad(self.0 - other.0) } else { Quad(0.0) }
    }

    routed_binary!(hypot, atan2);

    #[inline]
    fn sin_cos(self) -> (Quad, Quad) {
        let (s, c) = backing::sin_cos(self.0);
        (Quad(s), Quad(c))
    }

    #[inline]
    fn copysign(self, sign: Quad) -> Quad {
        Quad(self.0.copysign(sign.0))
    }

    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        integer_decode_f128(self.0)
    }
}

impl FloatConst for Quad {
    fwd_const! {
        E => core::f128::consts::E,
        FRAC_1_PI => core::f128::consts::FRAC_1_PI,
        FRAC_1_SQRT_2 => core::f128::consts::FRAC_1_SQRT_2,
        FRAC_2_PI => core::f128::consts::FRAC_2_PI,
        FRAC_2_SQRT_PI => core::f128::consts::FRAC_2_SQRT_PI,
        FRAC_PI_2 => core::f128::consts::FRAC_PI_2,
        FRAC_PI_3 => core::f128::consts::FRAC_PI_3,
        FRAC_PI_4 => core::f128::consts::FRAC_PI_4,
        FRAC_PI_6 => core::f128::consts::FRAC_PI_6,
        FRAC_PI_8 => core::f128::consts::FRAC_PI_8,
        LN_10 => core::f128::consts::LN_10,
        LN_2 => core::f128::consts::LN_2,
        LOG10_E => core::f128::consts::LOG10_E,
        LOG2_E => core::f128::consts::LOG2_E,
        PI => core::f128::consts::PI,
        SQRT_2 => core::f128::consts::SQRT_2,
        TAU => core::f128::consts::TAU,
        LOG10_2 => core::f128::consts::LOG10_2,
        LOG2_10 => core::f128::consts::LOG2_10,
    }
}

impl Quad {
    /// IEEE 754 `totalOrder` over the binary128 bits; mirrors `f64::total_cmp`.
    #[inline]
    pub fn total_cmp(&self, other: &Quad) -> Ordering {
        let mut a = self.0.to_bits() as i128;
        let mut b = other.0.to_bits() as i128;
        a ^= (((a >> 127) as u128) >> 1) as i128;
        b ^= (((b >> 127) as u128) >> 1) as i128;
        a.cmp(&b)
    }
}

#[cfg(test)]
mod tests {
    use super::Quad;
    use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};

    #[test]
    fn conversions_roundtrip() {
        // i128 must survive full width (113-bit significand => exact below 2^113).
        let v: i128 = (1i128 << 100) + 12345;
        let q = Quad::from_i128(v).unwrap();
        assert_eq!(q.to_i128().unwrap(), v);

        assert_eq!(Quad::from_f64(1.5).unwrap().to_f64().unwrap(), 1.5);
        assert_eq!(Quad::from_i64(-7).unwrap().to_i64().unwrap(), -7);
    }

    #[test]
    fn arithmetic_and_consts() {
        let two = Quad::from_f64(2.0).unwrap();
        assert_eq!(two.sqrt().powi(2).to_f64().unwrap(), 2.0);
        // sin(pi) ~ 0 to far beyond f64 precision
        assert!(<Quad as FloatConst>::PI().sin().abs().to_f64().unwrap() < 1e-30);
        assert_eq!((two + two).to_f64().unwrap(), 4.0);
    }

    /// This runs the fallback directly even on Linux, where the native backing
    /// is selected. It therefore catches Darwin regressions without emulation.
    #[test]
    fn portable_binary128_roundtrip() {
        use super::backing::portable;

        let cases = [
            0,
            1u128 << 127,
            1,
            (1u128 << 112) - 1,
            1u128 << 112,
            (0x3fffu128 << 112) | 1,
            (0x4000u128 << 112) | ((1u128 << 111) + 17),
            (0x7ffeu128 << 112) | ((1u128 << 112) - 1),
            0x7fffu128 << 112,
            (1u128 << 127) | (0x7fffu128 << 112),
        ];
        for bits in cases {
            assert_eq!(portable::conversion_roundtrip(f128::from_bits(bits)).to_bits(), bits);
        }

        let mut state = 0x9e37_79b9_7f4a_7c15_d1b5_4a32_d192_ed03u128;
        for _ in 0..1024 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Avoid NaNs: their payload is intentionally canonicalized.
            let bits = state & !(0x7fffu128 << 112) | ((state >> 64) % 0x7fff) << 112;
            assert_eq!(portable::conversion_roundtrip(f128::from_bits(bits)).to_bits(), bits);
        }
    }

    /// Value assertions for the complete portable `Float` surface. In
    /// particular, the sqrt assertion rejects the Intel-Darwin `sqrt(2) == 2`
    /// failure rather than merely checking that the result is finite.
    #[test]
    fn portable_math_surface_values() {
        use super::backing::portable as p;

        let close = |actual: f128, expected: f128, tolerance: f128| {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{} != {}",
                actual as f64,
                expected as f64
            );
        };
        let tolerance = 1e-30f128;
        let half = 0.5f128;
        let one = 1.0f128;
        let two = 2.0f128;

        assert_eq!(p::floor(-1.25), -2.0);
        assert_eq!(p::ceil(-1.25), -1.0);
        assert_eq!(p::round(-1.5), -2.0);
        assert_eq!(p::trunc(-1.75), -1.0);
        assert_eq!(p::fract(-1.75), -0.75);
        assert_eq!(p::rem(5.5, 2.0), 1.5);
        assert_eq!(p::mul_add(2.0, 3.0, 4.0), 10.0);

        close(p::sqrt(two), core::f128::consts::SQRT_2, tolerance);
        close(p::cbrt(8.0), 2.0, tolerance);
        close(p::exp(p::ln(two)), two, tolerance);
        close(p::exp2(3.0), 8.0, tolerance);
        close(p::log2(8.0), 3.0, tolerance);
        close(p::log10(100.0), 2.0, tolerance);
        close(p::log(8.0, 2.0), 3.0, tolerance);
        close(p::exp_m1(1e-20), 1e-20, 1e-39);
        close(p::ln_1p(1e-20), 1e-20, 1e-39);

        let (sin, cos) = p::sin_cos(one);
        close(sin, p::sin(one), tolerance);
        close(cos, p::cos(one), tolerance);
        close(sin * sin + cos * cos, one, tolerance);
        close(p::tan(core::f128::consts::FRAC_PI_4), one, tolerance);
        close(p::asin(half), core::f128::consts::FRAC_PI_6, tolerance);
        close(p::acos(half), core::f128::consts::FRAC_PI_3, tolerance);
        close(p::atan(one), core::f128::consts::FRAC_PI_4, tolerance);
        close(p::atan2(one, -one), 3.0 * core::f128::consts::FRAC_PI_4, tolerance);

        close(p::asinh(p::sinh(one)), one, tolerance);
        close(p::acosh(p::cosh(one)), one, tolerance);
        close(p::atanh(p::tanh(half)), half, tolerance);
        close(p::hypot(3.0, 4.0), 5.0, tolerance);
        close(p::powf(9.0, half), 3.0, tolerance);
    }

    #[test]
    fn portable_log_regressions_terminate() {
        use super::backing::portable as p;
        use std::sync::mpsc;
        use std::time::Duration;

        // Astro's two-argument BigFloat::log does not terminate for these
        // exact powers. Keep the watchdog here so a regression fails CI
        // promptly instead of hanging the test process.
        let (sender, receiver) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            sender.send([p::log(4.0, 2.0), p::log(100.0, 10.0)]).unwrap();
        });
        let [log_4_base_2, log_100_base_10] = receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("portable logarithms did not terminate within two seconds");
        worker.join().expect("portable logarithm worker panicked");

        assert!((log_4_base_2 - 2.0).abs() <= 1e-30);
        assert!((log_100_base_10 - 2.0).abs() <= 1e-30);
    }

    #[test]
    fn portable_special_value_semantics() {
        use super::backing::portable as p;

        let neg_zero = f128::from_bits(1u128 << 127);
        let is_neg_zero = |x: f128| x.to_bits() == neg_zero.to_bits();

        assert!(p::sqrt(-1.0).is_nan());
        assert!(is_neg_zero(p::sqrt(neg_zero)));
        assert!(p::sin(f128::INFINITY).is_nan());
        assert!(is_neg_zero(p::sin(neg_zero)));
        assert!(is_neg_zero(p::tan(neg_zero)));
        assert!(is_neg_zero(p::atan(neg_zero)));

        assert_eq!(p::ln(0.0), f128::NEG_INFINITY);
        assert_eq!(p::ln_1p(-1.0), f128::NEG_INFINITY);
        assert!(p::ln_1p(-2.0).is_nan());
        assert_eq!(p::exp(f128::NEG_INFINITY), 0.0);
        assert_eq!(p::exp2(f128::NEG_INFINITY).to_bits(), 0.0f128.to_bits());
        assert_eq!(p::exp2(f128::INFINITY), f128::INFINITY);
        assert_eq!(p::exp_m1(f128::NEG_INFINITY), -1.0);
        assert_eq!(p::log(0.0, 2.0), f128::NEG_INFINITY);
        assert_eq!(p::log(neg_zero, 2.0), f128::NEG_INFINITY);

        assert!(p::asin(2.0).is_nan());
        assert!(p::acos(2.0).is_nan());
        assert!(p::acosh(0.0).is_nan());
        assert_eq!(p::atanh(1.0), f128::INFINITY);
        assert_eq!(p::atanh(-1.0), f128::NEG_INFINITY);

        assert_eq!(p::hypot(f128::INFINITY, f128::NAN), f128::INFINITY);
        assert_eq!(p::powf(f128::NAN, 0.0), 1.0);
        assert_eq!(p::powf(1.0, f128::NAN), 1.0);
        assert!(p::powf(f128::NAN, f128::INFINITY).is_nan());
        assert_eq!(p::powf(-1.0, f128::INFINITY), 1.0);
        assert_eq!(p::powf(-2.0, f128::INFINITY), f128::INFINITY);
        assert_eq!(p::powf(-2.0, f128::NEG_INFINITY), 0.0);
        assert_eq!(p::powf(f128::NEG_INFINITY, 3.0), f128::NEG_INFINITY);
        assert!(is_neg_zero(p::powf(f128::NEG_INFINITY, -3.0)));
        assert_eq!(p::powf(f128::NEG_INFINITY, 2.0), f128::INFINITY);
        assert_eq!(p::powf(f128::NEG_INFINITY, -2.0).to_bits(), 0.0f128.to_bits());
        assert!(p::rem(f128::INFINITY, 1.0).is_nan());

        assert_eq!(p::atan2(0.0, -1.0), core::f128::consts::PI);
        assert_eq!(p::atan2(neg_zero, -1.0), -core::f128::consts::PI);
        assert!(is_neg_zero(p::atan2(neg_zero, 1.0)));
    }

    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    #[test]
    fn portable_transcendentals_match_libquadmath() {
        use super::backing::portable as p;
        use f128::f128 as Lq;
        use num_traits::Float as LqFloat;

        fn lq_bits(x: Lq) -> u128 {
            unsafe { core::mem::transmute::<Lq, u128>(x) }
        }
        fn lq(x: f64) -> Lq {
            <Lq as FromPrimitive>::from_f64(x).unwrap()
        }
        fn ulp_diff(a: u128, b: u128) -> u128 {
            fn key(bits: u128) -> i128 {
                let bits = bits as i128;
                bits ^ (((bits >> 127) as u128 >> 1) as i128)
            }
            (key(a) - key(b)).unsigned_abs()
        }

        const MAX_ULP: u128 = 4;
        let mut checks = 0usize;
        let mut worst = 0u128;
        macro_rules! check {
            ($name:ident, $x:expr) => {{
                let xf = $x;
                let distance = ulp_diff(p::$name(xf as f128).to_bits(), lq_bits(LqFloat::$name(lq(xf))));
                worst = worst.max(distance);
                assert!(
                    distance <= MAX_ULP,
                    "{}({}) differs by {} ULP",
                    stringify!($name),
                    xf,
                    distance
                );
                checks += 1;
            }};
        }

        for x in [-3.0, -1.0, -0.25, 0.0, 0.125, 0.5, 1.0, 3.0] {
            check!(sin, x);
            check!(cos, x);
            check!(tan, x);
            check!(atan, x);
            check!(exp, x);
            check!(exp2, x);
            check!(exp_m1, x);
            check!(sinh, x);
            check!(cosh, x);
            check!(tanh, x);
            check!(asinh, x);
            check!(cbrt, x);
            if x > 0.0 {
                check!(ln, x);
                check!(log2, x);
                check!(log10, x);
                check!(sqrt, x);
            }
            if (-1.0..=1.0).contains(&x) {
                check!(asin, x);
                check!(acos, x);
            }
            if x > -1.0 {
                check!(ln_1p, x);
            }
            if x >= 1.0 {
                check!(acosh, x);
            }
            if (-1.0..1.0).contains(&x) {
                check!(atanh, x);
            }
        }

        macro_rules! check_binary {
            ($name:ident, $x:expr, $y:expr) => {{
                let (xf, yf) = ($x, $y);
                let distance = ulp_diff(
                    p::$name(xf as f128, yf as f128).to_bits(),
                    lq_bits(LqFloat::$name(lq(xf), lq(yf))),
                );
                worst = worst.max(distance);
                assert!(
                    distance <= MAX_ULP,
                    "{}({}, {}) differs by {} ULP",
                    stringify!($name),
                    xf,
                    yf,
                    distance
                );
                checks += 1;
            }};
        }
        for (x, y) in [(0.25, 0.5), (2.0, 3.0), (9.0, 0.5), (12.5, -2.0)] {
            check_binary!(powf, x, y);
        }
        for (x, base) in [(0.25, 2.0), (2.0, 10.0), (12.5, 3.0)] {
            check_binary!(log, x, base);
        }
        for (x, y) in [(3.0, 4.0), (1e100, 1e-100), (-7.0, 11.0)] {
            check_binary!(hypot, x, y);
            check_binary!(atan2, x, y);
        }
        assert!(checks > 100);
        eprintln!("portable f128: {checks} libquadmath comparisons, worst {worst} ULP");
    }

    /// `Quad` (primitive `f128`) must match libquadmath: arithmetic/`sqrt`/rounding
    /// bit-for-bit, transcendentals to ≤ 2 ULP (glibc and libquadmath are
    /// independent libms; an `f64`-then-widened bug would be off by ~2^60 ULP).
    ///
    /// Only runs on x86_64 Linux GNU, where the `f128` crate's C shim and
    /// libquadmath are known to be available.
    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    #[test]
    fn matches_libquadmath_precision() {
        use f128::f128 as Lq;
        use num_traits::Float as LqFloat;

        fn lq_bits(x: Lq) -> u128 {
            unsafe { core::mem::transmute::<Lq, u128>(x) }
        }
        fn lq(x: f64) -> Lq {
            <Lq as FromPrimitive>::from_f64(x).unwrap()
        }
        // ULP distance via the monotone total-order key.
        fn ulp_diff(a: u128, b: u128) -> u128 {
            fn key(bits: u128) -> i128 {
                let b = bits as i128;
                b ^ (((b >> 127) as u128 >> 1) as i128)
            }
            (key(a) - key(b)).unsigned_abs()
        }

        // a spread of inputs incl. negatives, fractions, large/small magnitudes
        let inputs: &[f64] = &[
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            2.0,
            3.0,
            7.0,
            0.1,
            -0.1,
            1234.5,
            -9876.25,
            1e-12,
            1e12,
            core::f64::consts::PI,
            core::f64::consts::E,
            0.25,
            100.0,
            -100.0,
        ];

        const MAX_ULP: u128 = 2;
        let mut exact = 0usize;
        let mut transc = 0usize;
        let mut worst_ulp = 0u128;

        // correctly-rounded => identical bits
        macro_rules! check_exact {
            ($name:literal, $qm:ident, $q:expr, $r:expr, $xf:expr) => {{
                assert_eq!(
                    Quad::$qm($q).to_bits(),
                    lq_bits(LqFloat::$qm($r)),
                    "{} not bit-identical for input {}",
                    $name,
                    $xf
                );
                exact += 1;
            }};
        }
        // transcendental => within MAX_ULP
        macro_rules! check_ulp {
            ($name:literal, $qm:ident, $q:expr, $r:expr, $xf:expr) => {{
                let d = ulp_diff(Quad::$qm($q).to_bits(), lq_bits(LqFloat::$qm($r)));
                worst_ulp = worst_ulp.max(d);
                assert!(
                    d <= MAX_ULP,
                    "{} off by {} ULP for input {} (> {})",
                    $name,
                    d,
                    $xf,
                    MAX_ULP
                );
                transc += 1;
            }};
        }

        for &xf in inputs {
            let q = Quad::from_f64(xf).unwrap();
            let r = lq(xf);

            check_exact!("floor", floor, q, r, xf);
            check_exact!("ceil", ceil, q, r, xf);
            check_exact!("round", round, q, r, xf);
            check_exact!("trunc", trunc, q, r, xf);
            check_exact!("abs", abs, q, r, xf);

            check_ulp!("sin", sin, q, r, xf);
            check_ulp!("cos", cos, q, r, xf);
            check_ulp!("exp2", exp2, q, r, xf);

            // binary arithmetic is correctly rounded => identical bits
            let half = Quad::from_f64(0.5).unwrap();
            let half_r = lq(0.5);
            assert_eq!(
                (q * half).to_bits(),
                lq_bits(r * half_r),
                "mul not bit-identical for input {xf}"
            );
            assert_eq!(
                (q + half).to_bits(),
                lq_bits(r + half_r),
                "add not bit-identical for input {xf}"
            );
            assert_eq!(
                (q - half).to_bits(),
                lq_bits(r - half_r),
                "sub not bit-identical for input {xf}"
            );
            // `%` is `fmod`, exact and correctly rounded in both libms.
            assert_eq!(
                (q % half).to_bits(),
                lq_bits(r % half_r),
                "rem not bit-identical for input {xf}"
            );
            exact += 4;

            // powf transcendental. A negative base with non-integer exponent
            // is a domain error: both sides must return NaN, whose sign and
            // payload are unspecified (libquadmath yields -NaN, primitive
            // f128 +NaN), so the ULP metric only applies to the valid domain.
            let pow_ours = q.powf(half);
            let pow_theirs = LqFloat::powf(r, half_r);
            if xf < 0.0 {
                assert!(
                    pow_ours.is_nan() && LqFloat::is_nan(pow_theirs),
                    "powf must be NaN for negative base {xf}"
                );
            } else {
                let d = ulp_diff(pow_ours.to_bits(), lq_bits(pow_theirs));
                worst_ulp = worst_ulp.max(d);
                assert!(d <= MAX_ULP, "powf off by {d} ULP for input {xf} (> {MAX_ULP})");
            }
            transc += 1;

            // sqrt (correctly rounded) and ln, only where defined
            if xf > 0.0 {
                check_exact!("sqrt", sqrt, q, r, xf);
                check_ulp!("ln", ln, q, r, xf);
            }
        }

        // ---- full-mantissa operands ----
        //
        // Every input above is f64-representable, so its low 60 significand
        // bits are zero — an implementation that silently rounded through f64
        // would pass all of it. Compose operands as `hi + lo` with `lo` far
        // below f64's reach of `hi` (populating the deep mantissa) on both
        // sides identically, and re-run the exact/transcendental checks.
        let composed: &[(f64, f64)] = &[
            (1.0, 1.0e-25),
            (core::f64::consts::PI, -3.7e-24),
            (0.5, 1.0e-30),
            (-2.0, 5.0e-26),
            (1234.5, -1.0e-20),
            (1e-12, 1e-40),
        ];
        for &(hi, lo) in composed {
            let xf = hi;
            let q = Quad::from_f64(hi).unwrap() + Quad::from_f64(lo).unwrap();
            let r = lq(hi) + lq(lo);
            assert_eq!(q.to_bits(), lq_bits(r), "composition differs for ({hi}, {lo})");
            // The composition genuinely populated sub-f64 mantissa.
            assert!(
                (q - Quad::from_f64(q.to_f64().unwrap()).unwrap()).abs() > Quad::from_f64(0.0).unwrap(),
                "operand ({hi}, {lo}) carries no sub-f64 mantissa"
            );

            check_exact!("floor", floor, q, r, xf);
            check_exact!("abs", abs, q, r, xf);
            let half = Quad::from_f64(0.5).unwrap();
            let half_r = lq(0.5);
            assert_eq!(
                (q * half).to_bits(),
                lq_bits(r * half_r),
                "mul not bit-identical for composed ({hi}, {lo})"
            );
            assert_eq!(
                (q + half).to_bits(),
                lq_bits(r + half_r),
                "add not bit-identical for composed ({hi}, {lo})"
            );
            assert_eq!(
                (q / half).to_bits(),
                lq_bits(r / half_r),
                "div not bit-identical for composed ({hi}, {lo})"
            );
            exact += 5;

            check_ulp!("sin", sin, q, r, xf);
            check_ulp!("cos", cos, q, r, xf);
            check_ulp!("exp2", exp2, q, r, xf);
            if hi > 0.0 {
                check_exact!("sqrt", sqrt, q, r, xf);
                check_ulp!("ln", ln, q, r, xf);
            }
        }

        // ---- subnormals ----
        //
        // f64 subnormals are normal in binary128; reach true binary128
        // subnormals by scaling min-positive down, identically on both sides.
        fn lq_from_bits(bits: u128) -> Lq {
            unsafe { core::mem::transmute::<u128, Lq>(bits) }
        }
        let tiny = Quad::min_positive_value();
        // The f128 crate's `min_positive_value` is not binary128's MIN_POSITIVE;
        // build the reference from the same bit pattern.
        let tiny_r = lq_from_bits(tiny.to_bits());
        let half = Quad::from_f64(0.5).unwrap();
        let half_r = lq(0.5);
        let sub = tiny * half;
        let sub_r = tiny_r * half_r;
        assert_eq!(sub.to_bits(), lq_bits(sub_r), "subnormal halving differs");
        assert!(sub.to_bits() != 0, "subnormal collapsed to zero");
        assert_eq!(sub.classify(), core::num::FpCategory::Subnormal);
        assert_eq!((sub + sub).to_bits(), lq_bits(sub_r + sub_r), "subnormal add differs");
        assert_eq!(
            (sub * Quad::from_f64(4.0).unwrap()).to_bits(),
            lq_bits(sub_r * lq(4.0)),
            "subnormal renormalizing mul differs"
        );
        exact += 3;

        // ---- specials ----
        let inf = Quad::infinity();
        let inf_r = <Lq as LqFloat>::infinity();
        assert_eq!((inf + inf).to_bits(), lq_bits(inf_r + inf_r));
        assert!((inf - inf).is_nan() && (inf_r - inf_r).is_nan(), "inf − inf must be NaN");
        assert!(
            (Quad::from_f64(0.0).unwrap() / Quad::from_f64(0.0).unwrap()).is_nan(),
            "0/0 must be NaN"
        );
        let neg_zero = Quad::neg_zero();
        assert!(neg_zero.is_sign_negative() && neg_zero == Quad::from_f64(0.0).unwrap());
        assert_eq!(
            Quad::from_f64(3.0).unwrap().copysign(neg_zero).to_f64().unwrap(),
            -3.0,
            "copysign must honor -0.0"
        );
        // NaN propagates through exact ops and transcendentals alike.
        assert!((Quad::nan() * Quad::from_f64(2.0).unwrap()).is_nan());
        assert!(Quad::nan().sin().is_nan());

        // Constants agree to ≤ MAX_ULP (libquadmath's may be ~1 ULP off).
        macro_rules! check_const_ulp {
            ($name:literal, $c:ident) => {{
                let d = ulp_diff(<Quad as FloatConst>::$c().to_bits(), lq_bits(<Lq as FloatConst>::$c()));
                worst_ulp = worst_ulp.max(d);
                assert!(d <= MAX_ULP, "const {} off by {} ULP", $name, d);
            }};
        }
        check_const_ulp!("PI", PI);
        check_const_ulp!("E", E);
        check_const_ulp!("SQRT_2", SQRT_2);
        check_const_ulp!("LN_2", LN_2);

        assert!(
            exact > 100 && transc > 55,
            "expected many comparisons (exact={exact}, transc={transc})"
        );
        eprintln!("Quad vs libquadmath: {exact} ops bit-identical, {transc} transcendentals within {worst_ulp} ULP");
    }

    /// Arch-independent full-mantissa invariants: `Quad` arithmetic must be
    /// exact well past the 53-bit boundary an `f64`-roundtripping bug would
    /// impose. All identities below are exactly representable in binary128, so
    /// every assertion is bit-exact.
    #[test]
    fn full_mantissa_arithmetic_is_exact() {
        let one = Quad::from_f64(1.0).unwrap();
        let eps56 = Quad::from_f64(2f64.powi(-56)).unwrap();

        // (1 + 2^-56)·(1 + 2^-56) = 1 + 2^-55 + 2^-112: all three terms fit the
        // 113-bit significand, so the product is exact — and 2^-112 is far
        // below anything an f64 path could carry.
        let a = one + eps56;
        let product = a * a;
        let expected = one + Quad::from_f64(2f64.powi(-55)).unwrap() + Quad::from_f64(2f64.powi(-112)).unwrap();
        assert_eq!(product.to_bits(), expected.to_bits(), "113-bit product not exact");

        // Double-double recomposition round-trips: (hi + lo) − hi == lo when
        // the combined significand span fits 113 bits (`lo` is a power of two —
        // one significand bit — at relative 2^-61, far below f64's reach).
        let hi = Quad::from_f64(core::f64::consts::PI).unwrap();
        let lo = Quad::from_f64(2f64.powi(-60)).unwrap();
        assert_eq!(
            ((hi + lo) - hi).to_bits(),
            lo.to_bits(),
            "double-double recomposition lost bits"
        );

        // Sterbenz: subtraction of close values is exact.
        let b = one + eps56 + eps56;
        assert_eq!((b - a).to_bits(), eps56.to_bits(), "Sterbenz subtraction not exact");

        // Powers of two divide exactly all the way into the subnormal range,
        // and climb back losslessly.
        let mut x = Quad::min_positive_value();
        let half = Quad::from_f64(0.5).unwrap();
        let two = Quad::from_f64(2.0).unwrap();
        for _ in 0..10 {
            x *= half;
        }
        assert_eq!(x.classify(), core::num::FpCategory::Subnormal);
        for _ in 0..10 {
            x *= two;
        }
        assert_eq!(
            x.to_bits(),
            Quad::min_positive_value().to_bits(),
            "subnormal round-trip lost bits"
        );

        // integer_decode keeps the top 64 significand bits; a residue at
        // relative 2^-60 — below f64's 53-bit reach but inside the top 64 —
        // must show up in the decoded mantissa.
        let lo60 = Quad::from_f64(1.9e-18).unwrap();
        let (mantissa_composed, _, _) = Float::integer_decode(hi + lo60);
        let (mantissa_pi, _, _) = Float::integer_decode(hi);
        assert_ne!(mantissa_composed, mantissa_pi, "integer_decode blind to sub-f64 mantissa");
    }
}
