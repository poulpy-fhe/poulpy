//! Portable quad-precision (IEEE 754 binary128) scalar for high-precision CKKS.
//!
//! [`Quad`] is a `#[repr(transparent)]` newtype over the nightly primitive
//! `f128`, implementing the `num_traits` surface (`Float`, `FloatConst`,
//! `From`/`ToPrimitive`) that `num_traits` does not provide for `f128` itself.
//! It replaces the `f128` crate (libquadmath, x86_64-only) so the binary128
//! path works on both x86_64 and aarch64-linux-gnu. The `f128` math symbols are
//! only guaranteed on targets with "reliable" f128 math (those two).

use core::cmp::Ordering;
use core::fmt;
use core::iter::{Product, Sum};
use core::num::FpCategory;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use rand_distr::num_traits::{Float, FloatConst, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};

/// IEEE 754 binary128 (quadruple precision) float, portable across the
/// `poulpy` CPU backends (x86_64 and aarch64).
#[derive(Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Quad(pub f128);

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
bin_op!(Rem, rem, %);

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
assign_op!(RemAssign, rem_assign, %=);

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

    fwd_unary!(
        floor, ceil, round, trunc, fract, abs, signum, recip, sqrt, exp, exp2, ln, log2, log10, cbrt, sin, cos, tan, asin, acos,
        atan, exp_m1, ln_1p, sinh, cosh, tanh, asinh, acosh, atanh,
    );

    #[inline]
    fn classify(self) -> FpCategory {
        self.0.classify()
    }

    #[inline]
    fn mul_add(self, a: Quad, b: Quad) -> Quad {
        Quad(self.0.mul_add(a.0, b.0))
    }

    #[inline]
    fn powi(self, n: i32) -> Quad {
        Quad(self.0.powi(n))
    }

    #[inline]
    fn powf(self, n: Quad) -> Quad {
        Quad(self.0.powf(n.0))
    }

    #[inline]
    fn log(self, base: Quad) -> Quad {
        Quad(self.0.log(base.0))
    }

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

    #[inline]
    fn hypot(self, other: Quad) -> Quad {
        Quad(self.0.hypot(other.0))
    }

    #[inline]
    fn atan2(self, other: Quad) -> Quad {
        Quad(self.0.atan2(other.0))
    }

    #[inline]
    fn sin_cos(self) -> (Quad, Quad) {
        let (s, c) = self.0.sin_cos();
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
    use rand_distr::num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};

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

    /// `Quad` (primitive `f128`) must match libquadmath: arithmetic/`sqrt`/rounding
    /// bit-for-bit, transcendentals to ≤ 2 ULP (glibc and libquadmath are
    /// independent libms; an `f64`-then-widened bug would be off by ~2^60 ULP).
    ///
    /// Only runs on x86_64, where the `f128` crate (libquadmath) is available.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn matches_libquadmath_precision() {
        use f128::f128 as Lq;
        use rand_distr::num_traits::Float as LqFloat;

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

        for &xf in inputs {
            let q = Quad::from_f64(xf).unwrap();
            let r = lq(xf);

            // correctly-rounded => identical bits
            macro_rules! check_exact {
                ($name:literal, $qm:ident) => {{
                    assert_eq!(
                        Quad::$qm(q).to_bits(),
                        lq_bits(LqFloat::$qm(r)),
                        "{} not bit-identical for input {}",
                        $name,
                        xf
                    );
                    exact += 1;
                }};
            }
            // transcendental => within MAX_ULP
            macro_rules! check_ulp {
                ($name:literal, $qm:ident) => {{
                    let d = ulp_diff(Quad::$qm(q).to_bits(), lq_bits(LqFloat::$qm(r)));
                    worst_ulp = worst_ulp.max(d);
                    assert!(
                        d <= MAX_ULP,
                        "{} off by {} ULP for input {} (> {})",
                        $name,
                        d,
                        xf,
                        MAX_ULP
                    );
                    transc += 1;
                }};
            }

            check_exact!("floor", floor);
            check_exact!("ceil", ceil);
            check_exact!("round", round);
            check_exact!("trunc", trunc);
            check_exact!("abs", abs);

            check_ulp!("sin", sin);
            check_ulp!("cos", cos);
            check_ulp!("exp2", exp2);

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
            exact += 3;

            // powf transcendental
            worst_ulp = worst_ulp.max(ulp_diff(q.powf(half).to_bits(), lq_bits(LqFloat::powf(r, half_r))));
            assert!(
                ulp_diff(q.powf(half).to_bits(), lq_bits(LqFloat::powf(r, half_r))) <= MAX_ULP,
                "powf off by too many ULP for input {xf}"
            );
            transc += 1;

            // sqrt (correctly rounded) and ln, only where defined
            if xf > 0.0 {
                check_exact!("sqrt", sqrt);
                check_ulp!("ln", ln);
            }
        }

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
            exact > 80 && transc > 40,
            "expected many comparisons (exact={exact}, transc={transc})"
        );
        eprintln!("Quad vs libquadmath: {exact} ops bit-identical, {transc} transcendentals within {worst_ulp} ULP");
    }
}
