//! Portable quad-precision (IEEE 754 binary128) scalar for high-precision CKKS.
//!
//! [`Quad`] is a `#[repr(transparent)]` newtype over the nightly primitive
//! `f128`, implementing the `num_traits` surface (`Float`, `FloatConst`,
//! `From`/`ToPrimitive`) that `num_traits` does not provide for `f128` itself.
//! It replaces the `f128` crate (libquadmath, x86_64-only) as the *type* so the
//! binary128 path works on both x86_64 and aarch64-linux-gnu.
//!
//! The type, its storage and its arithmetic are the primitive throughout, so
//! `Quad` is the same `Pod` newtype in every configuration. What varies is the
//! set of operations Rust lowers to `*f128` libm calls — `sqrt`, `%`, the
//! rounding family, `mul_add` and the transcendentals. Those symbols exist only
//! where Rust considers `f128` math reliable (x86_64- and aarch64-linux-gnu);
//! elsewhere they are evaluated with dashu-float, and under the `libquadmath`
//! feature on x86_64-linux the transcendentals go to libquadmath instead. See
//! the backing modules below.

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

// `%` is the one arithmetic operator that lowers to a libm call (`fmodf128`),
// so unlike its neighbours it goes through the backing.
impl Rem for Quad {
    type Output = Quad;
    #[inline]
    fn rem(self, rhs: Quad) -> Quad {
        Quad(libm::rem(self.0, rhs.0))
    }
}

impl RemAssign for Quad {
    #[inline]
    fn rem_assign(&mut self, rhs: Quad) {
        self.0 = libm::rem(self.0, rhs.0);
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

// ─── libm backings ──────────────────────────────────────────────────────────
//
// Rust lowers `f128`'s `sqrt`, `%`, the rounding family, `mul_add` and every
// transcendental to `*f128` libm calls. Which of those a target supplies
// varies, and the failure modes differ sharply:
//
//   Linux — glibc ≥ 2.26 on x86_64 and aarch64, the targets Rust lists as
//     having "reliable" `f128` math — supplies all of them.
//   macOS supplies *none* of them; it has no binary128 libm at all. Some, like
//     `asinf128`, fail to link outright. The rest — `sqrtf128`, `fmodf128`,
//     `truncf128`, `sinf128`, `expf128` … — resolve to stubs that return their
//     argument untouched, so `sqrt(2.0)` quietly evaluates to `2.0`. Nothing
//     warns, at compile time or at run time.
//
// So outside that guarantee every operation below is evaluated with
// dashu-float rather than trusted to the platform. What is *not* here —
// `+ - * /`, comparisons, conversions, `abs`/`signum`/`recip`/`min`/`max` —
// is compiler-builtins or pure bit manipulation, correct everywhere, and stays
// on the primitive.
//
//   [`quadmath`]  libquadmath through the `f128` crate; x86_64-linux, opt-in
//                 via the `libquadmath` feature. Transcendentals only, since
//                 the platform's own are already correct there.
//   [`native`]    the target's binary128 libm. Linux only.
//   [`softfloat`] dashu-float. Everywhere else.
//
// All three expose the same free functions, so the impls below are one set of
// `libm::*` calls regardless of target.

#[cfg(all(feature = "libquadmath", target_arch = "x86_64", target_os = "linux"))]
use quadmath as libm;

#[cfg(all(target_os = "linux", not(all(feature = "libquadmath", target_arch = "x86_64"))))]
use native as libm;

#[cfg(not(target_os = "linux"))]
use softfloat as libm;

/// Forwards to the primitive's inherent method.
#[cfg(target_os = "linux")]
macro_rules! primitive_unary {
    ($($method:ident),+ $(,)?) => {$(
        #[inline]
        pub fn $method(x: f128) -> f128 {
            x.$method()
        }
    )+};
}

/// The operations that are exact (or correctly rounded) by definition, in
/// terms of the primitive. Shared by the two backings that run on a target
/// whose libm is trustworthy.
#[cfg(target_os = "linux")]
macro_rules! primitive_exact {
    () => {
        primitive_unary!(sqrt, floor, ceil, round, trunc, fract);

        #[inline]
        pub fn mul_add(x: f128, y: f128, z: f128) -> f128 {
            x.mul_add(y, z)
        }

        #[inline]
        pub fn rem(x: f128, y: f128) -> f128 {
            x % y
        }
    };
}

/// libquadmath backing: primitive `f128` values are bit-cast to the `f128`
/// crate's binary128 type (both are IEEE-754 binary128, identical bit layout),
/// evaluated by libquadmath, and bit-cast back. Storage and exact arithmetic
/// never leave the primitive type.
#[cfg(all(feature = "libquadmath", target_arch = "x86_64", target_os = "linux"))]
mod quadmath {
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
        ($($method:ident),+ $(,)?) => {$(
            #[inline]
            pub fn $method(x: f128) -> f128 {
                from_lq(num_traits::Float::$method(to_lq(x)))
            }
        )+};
    }

    macro_rules! binary {
        ($($method:ident),+ $(,)?) => {$(
            #[inline]
            pub fn $method(x: f128, y: f128) -> f128 {
                from_lq(num_traits::Float::$method(to_lq(x), to_lq(y)))
            }
        )+};
    }

    primitive_exact!();

    unary!(
        exp, exp2, ln, log2, log10, cbrt, sin, cos, tan, asin, acos, atan, exp_m1, ln_1p, sinh, cosh, tanh, asinh, acosh, atanh,
    );
    binary!(powf, log, hypot, atan2);

    #[inline]
    pub fn sin_cos(x: f128) -> (f128, f128) {
        let (s, c) = num_traits::Float::sin_cos(to_lq(x));
        (from_lq(s), from_lq(c))
    }
}

/// The target's own binary128 libm, reached through the primitive's inherent
/// methods.
#[cfg(all(target_os = "linux", not(all(feature = "libquadmath", target_arch = "x86_64"))))]
mod native {
    macro_rules! binary {
        ($($method:ident),+ $(,)?) => {$(
            #[inline]
            pub fn $method(x: f128, y: f128) -> f128 {
                x.$method(y)
            }
        )+};
    }

    primitive_exact!();

    primitive_unary!(
        exp, exp2, ln, log2, log10, cbrt, sin, cos, tan, asin, acos, atan, exp_m1, ln_1p, sinh, cosh, tanh, asinh, acosh, atanh,
    );
    binary!(powf, log, hypot, atan2);

    #[inline]
    pub fn sin_cos(x: f128) -> (f128, f128) {
        x.sin_cos()
    }
}

/// dashu-float backing for targets with no binary128 libm. Each argument is
/// converted exactly to an [`FBig`], evaluated at [`PREC`] bits, and rounded
/// once to binary128's 113.
///
/// Identities are chosen for their conditioning, not their brevity — the
/// hyperbolics go through `exp_m1` and the inverse hyperbolics through `ln_1p`
/// so that small arguments keep full relative precision, which is the whole
/// point of computing in binary128.
///
/// Non-finite arguments and out-of-domain values take an `f64` detour instead:
/// `f64` already implements IEEE's special-case tables, and widening its
/// ±inf/NaN/±0 back to binary128 is exact. That path never carries precision.
#[cfg(not(target_os = "linux"))]
mod softfloat {
    use dashu_float::math::FpResult;
    use dashu_float::round::mode::HalfEven;
    use dashu_float::{Context, FBig};

    type F = FBig<HalfEven, 2>;

    /// Working precision. binary128's significand is 113 bits; the headroom
    /// keeps the final rounding in [`from_f`] from being a double rounding.
    const PREC: usize = 160;

    /// `ln(f128::MAX) ≈ 11356.5`, so past this every exponential has already
    /// saturated to ±inf or flushed to zero. Clamping keeps dashu-float from
    /// building exponents that could never be returned — and that would not
    /// fit an `isize` for arguments like `1e30`.
    const EXP_LIMIT: f128 = 12_000.0;

    fn ctx() -> Context<HalfEven> {
        Context::new(PREC)
    }

    /// A small integer at working precision.
    fn cst(n: i32) -> F {
        F::from_parts(n.into(), 0).with_precision(PREC).value()
    }

    /// Exact: every finite binary128 is a dyadic rational.
    fn to_f(x: f128) -> F {
        let bits = x.to_bits();
        let biased = ((bits >> 112) & 0x7fff) as i32;
        let fraction = bits & ((1u128 << 112) - 1);
        // Subnormals carry no implicit leading bit and a fixed exponent.
        let (significand, exponent) = if biased == 0 {
            (fraction, -16382 - 112)
        } else {
            (fraction | (1u128 << 112), biased - 16383 - 112)
        };
        let magnitude = F::from_parts(significand.into(), exponent as isize)
            .with_precision(PREC)
            .value();
        if bits >> 127 == 1 { -magnitude } else { magnitude }
    }

    /// Rounds to binary128's 113-bit significand, then scales. Results landing
    /// in the subnormal range round twice; nothing in CKKS reaches `2^-16382`.
    fn from_f(v: F) -> f128 {
        let rounded = v.with_precision(113).value();
        let significand: i128 = rounded
            .repr()
            .significand()
            .try_into()
            .expect("a 113-bit significand fits in i128");
        ldexp(significand as f128, rounded.repr().exponent())
    }

    /// `m · 2^e`, in steps small enough that every factor is a normal binary128.
    fn ldexp(m: f128, e: isize) -> f128 {
        const STEP: isize = 16382;
        let mut out = m;
        let mut e = e;
        while e > STEP {
            out *= pow2(STEP);
            e -= STEP;
        }
        while e < -STEP {
            out *= pow2(-STEP);
            e += STEP;
        }
        out * pow2(e)
    }

    /// `2^e` as a normal binary128; requires `|e| <= 16382`.
    fn pow2(e: isize) -> f128 {
        f128::from_bits(((e + 16383) as u128) << 112)
    }

    /// Unwraps a dashu-float result, or `None` where it reports NaN, infinity
    /// or over-/underflow rather than a finite value.
    fn finite(r: FpResult<2>) -> Option<F> {
        r.ok(&ctx()).map(|rounded| rounded.value())
    }

    /// Values dashu-float cannot carry: the non-finite ones, and zero, whose
    /// sign it does not model — `sin(-0.0)` must stay `-0.0`. `f64` gives the
    /// IEEE answer for every function here at these arguments, and each such
    /// answer (`±0`, `1`, `±inf`, NaN) widens to binary128 exactly.
    fn is_special(x: f128) -> bool {
        !x.is_finite() || x == 0.0
    }

    /// `ln(x)` to within ±0.7, straight off the exponent field. Used only to
    /// decide whether a power has left binary128's range, where that is ample.
    fn ln_estimate(x: f128) -> f64 {
        let exponent = ((x.to_bits() >> 112) & 0x7fff) as i32 - 16383;
        f64::from(exponent) * core::f64::consts::LN_2
    }

    /// Base-2 exponent of a finite non-zero `|x|`, subnormals included.
    fn ilog2(x: f128) -> i32 {
        let bits = x.to_bits() & !(1u128 << 127);
        let biased = (bits >> 112) as i32;
        if biased > 0 {
            biased - 16383
        } else {
            // No implicit leading bit: find the highest one actually set.
            127 - bits.leading_zeros() as i32 - 112 - 16382
        }
    }

    // ─── exact operations ───────────────────────────────────────────────────
    //
    // Rounding and remainder are exact by construction, so they are done on the
    // representation directly; only `sqrt` and `mul_add` need dashu-float.

    pub fn trunc(x: f128) -> f128 {
        let bits = x.to_bits();
        let exponent = ((bits >> 112) & 0x7fff) as i32 - 16383;
        if exponent < 0 {
            // |x| < 1, so the integer part is a zero of the same sign.
            return f128::from_bits(bits & (1u128 << 127));
        }
        if exponent >= 112 {
            // Already integral — and this is also the infinity/NaN path.
            return x;
        }
        f128::from_bits(bits & !((1u128 << (112 - exponent)) - 1))
    }

    pub fn floor(x: f128) -> f128 {
        let t = trunc(x);
        if t > x { t - 1.0 } else { t }
    }

    pub fn ceil(x: f128) -> f128 {
        let t = trunc(x);
        if t < x { t + 1.0 } else { t }
    }

    pub fn round(x: f128) -> f128 {
        // Half away from zero, matching the primitive. `x - t` is exact: `t` is
        // `x` with its low bits cleared.
        let t = trunc(x);
        let fraction = x - t;
        if fraction >= 0.5 {
            t + 1.0
        } else if fraction <= -0.5 {
            t - 1.0
        } else {
            t
        }
    }

    pub fn fract(x: f128) -> f128 {
        x - trunc(x)
    }

    pub fn rem(x: f128, y: f128) -> f128 {
        // IEEE `fmod`: `x − n·y` for the `n` that truncates toward zero.
        if x.is_nan() || y.is_nan() || !x.is_finite() || y == 0.0 {
            return f128::NAN;
        }
        if !y.is_finite() {
            return x;
        }
        let negative = x.is_sign_negative();
        let mut r = if negative { -x } else { x };
        let d = if y.is_sign_negative() { -y } else { y };
        // Each step subtracts the largest `2^k·d` that fits, which lies in
        // `[r/2, r]` — Sterbenz's lemma, so every subtraction is exact and the
        // loop is an exact reduction, not an approximation. It runs once per
        // bit of exponent separation.
        while r >= d {
            let k = ilog2(r) - ilog2(d);
            let mut step = ldexp(d, k as isize);
            if step > r {
                step = ldexp(d, (k - 1) as isize);
            }
            r -= step;
        }
        if negative { -r } else { r }
    }

    pub fn sqrt(x: f128) -> f128 {
        if !x.is_finite() || x <= 0.0 {
            // Covers ±0 (which keeps its sign), negatives (NaN) and ±inf.
            return (x as f64).sqrt() as f128;
        }
        from_f(to_f(x).nth_root(2))
    }

    pub fn mul_add(x: f128, y: f128, z: f128) -> f128 {
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return (x as f64).mul_add(y as f64, z as f64) as f128;
        }
        // The product needs 226 bits to be exact; at 256 the single rounding
        // that matters is the final one back to binary128.
        const WIDE: usize = 256;
        let wide = |v: f128| to_f(v).with_precision(WIDE).value();
        from_f(wide(x) * wide(y) + wide(z))
    }

    pub fn exp(x: f128) -> f128 {
        if is_special(x) || x > EXP_LIMIT || x < -EXP_LIMIT {
            return (x as f64).exp() as f128;
        }
        from_f(to_f(x).exp())
    }

    pub fn exp2(x: f128) -> f128 {
        // `log2(f128::MAX) = 16384`.
        if is_special(x) || x > 17_000.0 || x < -17_000.0 {
            return (x as f64).exp2() as f128;
        }
        from_f((to_f(x) * cst(2).ln()).exp())
    }

    pub fn exp_m1(x: f128) -> f128 {
        if is_special(x) || x > EXP_LIMIT || x < -EXP_LIMIT {
            return (x as f64).exp_m1() as f128;
        }
        from_f(to_f(x).exp_m1())
    }

    pub fn ln(x: f128) -> f128 {
        if !x.is_finite() || x <= 0.0 {
            return (x as f64).ln() as f128;
        }
        from_f(to_f(x).ln())
    }

    pub fn log2(x: f128) -> f128 {
        if !x.is_finite() || x <= 0.0 {
            return (x as f64).log2() as f128;
        }
        from_f(to_f(x).ln() / cst(2).ln())
    }

    pub fn log10(x: f128) -> f128 {
        if !x.is_finite() || x <= 0.0 {
            return (x as f64).log10() as f128;
        }
        from_f(to_f(x).ln() / cst(10).ln())
    }

    pub fn ln_1p(x: f128) -> f128 {
        if is_special(x) || x <= -1.0 {
            return (x as f64).ln_1p() as f128;
        }
        from_f(to_f(x).ln_1p())
    }

    pub fn log(x: f128, base: f128) -> f128 {
        if !x.is_finite() || !base.is_finite() || x <= 0.0 || base <= 0.0 || base == 1.0 {
            return (x as f64).log(base as f64) as f128;
        }
        from_f(to_f(x).ln() / to_f(base).ln())
    }

    pub fn cbrt(x: f128) -> f128 {
        if is_special(x) {
            return (x as f64).cbrt() as f128;
        }
        from_f(to_f(x).nth_root(3))
    }

    pub fn sin(x: f128) -> f128 {
        if is_special(x) {
            return (x as f64).sin() as f128;
        }
        from_f(to_f(x).sin())
    }

    pub fn cos(x: f128) -> f128 {
        if is_special(x) {
            return (x as f64).cos() as f128;
        }
        from_f(to_f(x).cos())
    }

    pub fn sin_cos(x: f128) -> (f128, f128) {
        if !x.is_finite() {
            let (s, c) = (x as f64).sin_cos();
            return (s as f128, c as f128);
        }
        let (s, c) = to_f(x).sin_cos();
        (from_f(s), from_f(c))
    }

    pub fn tan(x: f128) -> f128 {
        if !is_special(x)
            && let Some(v) = finite(to_f(x).tan())
        {
            return from_f(v);
        }
        (x as f64).tan() as f128
    }

    pub fn asin(x: f128) -> f128 {
        if !is_special(x)
            && x >= -1.0
            && x <= 1.0
            && let Some(v) = finite(to_f(x).asin())
        {
            return from_f(v);
        }
        (x as f64).asin() as f128
    }

    pub fn acos(x: f128) -> f128 {
        if !is_special(x)
            && x >= -1.0
            && x <= 1.0
            && let Some(v) = finite(to_f(x).acos())
        {
            return from_f(v);
        }
        (x as f64).acos() as f128
    }

    pub fn atan(x: f128) -> f128 {
        if is_special(x) {
            return (x as f64).atan() as f128;
        }
        from_f(to_f(x).atan())
    }

    pub fn atan2(y: f128, x: f128) -> f128 {
        if !y.is_finite() || !x.is_finite() {
            return (y as f64).atan2(x as f64) as f128;
        }
        // The axis cases are exact multiples of π, which `f64` could not carry
        // back at full width; the rest is a genuine two-argument evaluation.
        if y == 0.0 {
            let magnitude = if x.is_sign_positive() { 0.0 } else { core::f128::consts::PI };
            return if y.is_sign_negative() { -magnitude } else { magnitude };
        }
        if x == 0.0 {
            let magnitude = core::f128::consts::FRAC_PI_2;
            return if y < 0.0 { -magnitude } else { magnitude };
        }
        match finite(to_f(y).atan2(&to_f(x))) {
            Some(v) => from_f(v),
            None => (y as f64).atan2(x as f64) as f128,
        }
    }

    pub fn hypot(x: f128, y: f128) -> f128 {
        if !x.is_finite() || !y.is_finite() || (x == 0.0 && y == 0.0) {
            return (x as f64).hypot(y as f64) as f128;
        }
        let (a, b) = (to_f(x), to_f(y));
        from_f((&a * &a + &b * &b).nth_root(2))
    }

    pub fn powf(x: f128, y: f128) -> f128 {
        // IEEE `pow` has a long table of special cases around signed zeros and
        // ±inf; `f64` implements all of them. A negative base is not one of
        // them — with an integral exponent it is an ordinary full-width
        // evaluation, so only its sign is peeled off here.
        if !x.is_finite() || !y.is_finite() || x == 0.0 || y == 0.0 {
            return (x as f64).powf(y as f64) as f128;
        }
        // Note `trunc` here is this module's, not the primitive's stub.
        let negative_base = x < 0.0;
        if negative_base && trunc(y) != y {
            return f128::NAN;
        }
        let magnitude = if negative_base { -x } else { x };
        // Halving is exact, so a fractional half means an odd exponent.
        let odd_power = negative_base && trunc(y * 0.5) != y * 0.5;
        // `y·ln|x|` is the result's natural-log magnitude. The bound is far
        // outside binary128's own range (`ln(f128::MAX) ≈ 11356`) on purpose:
        // saturating is [`from_f`]'s job, which does it by rounding, and this
        // guard exists only to keep an argument like `y = 1e30` from asking
        // dashu-float for an exponent that would not fit an `isize`.
        let scale = (y as f64) * ln_estimate(magnitude);
        if !(-1.0e6..=1.0e6).contains(&scale) {
            let saturated: f128 = if scale > 0.0 { f128::INFINITY } else { 0.0 };
            return if odd_power { -saturated } else { saturated };
        }
        let out = from_f(to_f(magnitude).powf(&to_f(y)));
        if odd_power { -out } else { out }
    }

    pub fn sinh(x: f128) -> f128 {
        if is_special(x) || x > EXP_LIMIT || x < -EXP_LIMIT {
            return (x as f64).sinh() as f128;
        }
        // With `t = e^x − 1`: sinh(x) = t(t+2) / 2(1+t), free of the
        // cancellation that `(e^x − e^-x)/2` suffers as x → 0.
        let t = to_f(x).exp_m1();
        let (one, two) = (cst(1), cst(2));
        from_f(&t * (&t + &two) / (&two * (&one + &t)))
    }

    pub fn cosh(x: f128) -> f128 {
        if is_special(x) || x > EXP_LIMIT || x < -EXP_LIMIT {
            return (x as f64).cosh() as f128;
        }
        // cosh(x) = 1 + t² / 2(1+t).
        let t = to_f(x).exp_m1();
        let (one, two) = (cst(1), cst(2));
        from_f(&one + &t * &t / (&two * (&one + &t)))
    }

    pub fn tanh(x: f128) -> f128 {
        if is_special(x) || x > EXP_LIMIT || x < -EXP_LIMIT {
            return (x as f64).tanh() as f128;
        }
        // tanh(x) = t(t+2) / (t² + 2t + 2).
        let t = to_f(x).exp_m1();
        let two = cst(2);
        from_f(&t * (&t + &two) / (&t * &t + &two * &t + &two))
    }

    pub fn asinh(x: f128) -> f128 {
        if is_special(x) {
            return (x as f64).asinh() as f128;
        }
        // asinh(x) = ln1p(a + a²/(√(a²+1) + 1)) for a = |x|, which avoids both
        // the cancellation in `x + √(x²+1)` for negative x and the loss of the
        // small terms as x → 0. asinh is odd, so the sign goes back on after.
        let a = to_f(if x < 0.0 { -x } else { x });
        let one = cst(1);
        let root = (&a * &a + &one).nth_root(2);
        let out = from_f((&a + &a * &a / (&root + &one)).ln_1p());
        if x < 0.0 { -out } else { out }
    }

    pub fn acosh(x: f128) -> f128 {
        if !x.is_finite() || x < 1.0 {
            return (x as f64).acosh() as f128;
        }
        // acosh(1+u) = ln1p(u + √(u(u+2))), which keeps precision as x → 1.
        let u = to_f(x) - cst(1);
        let two = cst(2);
        from_f((&u + (&u * (&u + &two)).nth_root(2)).ln_1p())
    }

    pub fn atanh(x: f128) -> f128 {
        if is_special(x) || x <= -1.0 || x >= 1.0 {
            return (x as f64).atanh() as f128;
        }
        // atanh(x) = ln1p(2x/(1−x)) / 2.
        let a = to_f(x);
        let (one, two) = (cst(1), cst(2));
        from_f((&two * &a / (&one - &a)).ln_1p() / &two)
    }
}

/// Unary operations routed to the target's backing.
macro_rules! libm_unary {
    ($($method:ident),+ $(,)?) => {
        $(
            #[inline]
            fn $method(self) -> Quad {
                Quad(libm::$method(self.0))
            }
        )+
    };
}

/// Binary operations, same routing as [`libm_unary`].
macro_rules! libm_binary {
    ($($method:ident),+ $(,)?) => {
        $(
            #[inline]
            fn $method(self, other: Quad) -> Quad {
                Quad(libm::$method(self.0, other.0))
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

    // Sign and reciprocal are bit manipulation and division: correct on every
    // target, so they stay on the primitive.
    fwd_unary!(abs, signum, recip);

    // Exact / correctly-rounded, but libm calls all the same.
    libm_unary!(floor, ceil, round, trunc, fract, sqrt);

    // Transcendentals: see the backing modules above.
    libm_unary!(
        exp, exp2, ln, log2, log10, cbrt, sin, cos, tan, asin, acos, atan, exp_m1, ln_1p, sinh, cosh, tanh, asinh, acosh, atanh,
    );

    #[inline]
    fn classify(self) -> FpCategory {
        self.0.classify()
    }

    #[inline]
    fn mul_add(self, a: Quad, b: Quad) -> Quad {
        Quad(libm::mul_add(self.0, a.0, b.0))
    }

    /// Repeated multiplication, so it needs no libm support anywhere.
    #[inline]
    fn powi(self, n: i32) -> Quad {
        Quad(self.0.powi(n))
    }

    libm_binary!(powf, log);

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

    libm_binary!(hypot, atan2);

    #[inline]
    fn sin_cos(self) -> (Quad, Quad) {
        let (s, c) = libm::sin_cos(self.0);
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
    use num_traits::{Float, FloatConst, FromPrimitive, One, ToPrimitive, Zero};

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

    fn q(x: f64) -> Quad {
        Quad::from_f64(x).unwrap()
    }

    /// Relative agreement, in binary128 terms. The tolerance throughout these
    /// tests is `1e-30` — some 5000 ULP, deliberately loose. They exist to
    /// prove a backing is evaluating at *quad* precision at all: anything that
    /// silently drops to `f64` lands near `1e-16`, thirteen orders of magnitude
    /// away. Last-ULP conformance is [`matches_libquadmath_precision`]'s job.
    #[track_caller]
    fn assert_close(actual: Quad, expected: Quad, label: &str) {
        let error = if expected == Quad::zero() {
            actual.abs()
        } else {
            ((actual - expected) / expected).abs()
        };
        assert!(
            error < q(1e-30),
            "{label}: got {actual:?}, expected {expected:?} (relative error {:e})",
            error.to_f64().unwrap()
        );
    }

    /// `sqrt`, the rounding family, `%` and `mul_add` are libm calls too, and
    /// on a target whose libm returns its argument (macOS) every one of them is
    /// silently the identity — `sqrt(2)` gives `2`, `7 % 3` gives `7`. These
    /// are exact operations, so the assertions can be exact.
    #[test]
    fn exact_operations() {
        // Perfect squares across the exponent range, then a value that is not.
        for k in [-8000i32, -100, 0, 100, 8000] {
            let v = q(2.0).powi(2 * k);
            assert_eq!(v.sqrt(), q(2.0).powi(k), "sqrt of 2^{}", 2 * k);
        }
        assert_eq!(q(16.0).sqrt(), q(4.0));
        assert_close(q(2.0).sqrt() * q(2.0).sqrt(), q(2.0), "sqrt(2) squared");
        assert_eq!(q(0.0).sqrt(), q(0.0));
        assert!(q(-1.0).sqrt().is_nan());

        // (value, floor, ceil, trunc, round) — `round` is half away from zero.
        let cases = [
            (2.5, 2.0, 3.0, 2.0, 3.0),
            (-2.5, -3.0, -2.0, -2.0, -3.0),
            (2.7, 2.0, 3.0, 2.0, 3.0),
            (-2.7, -3.0, -2.0, -2.0, -3.0),
            (0.5, 0.0, 1.0, 0.0, 1.0),
            (-0.5, -1.0, 0.0, 0.0, -1.0),
            (7.0, 7.0, 7.0, 7.0, 7.0),
            (-7.0, -7.0, -7.0, -7.0, -7.0),
        ];
        for (x, floor, ceil, trunc, round) in cases {
            let v = q(x);
            assert_eq!(v.floor(), q(floor), "floor({x})");
            assert_eq!(v.ceil(), q(ceil), "ceil({x})");
            assert_eq!(v.trunc(), q(trunc), "trunc({x})");
            assert_eq!(v.round(), q(round), "round({x})");
            assert_eq!(v.fract(), v - q(trunc), "fract({x})");
        }
        // Signed zeros survive: `ceil(-0.5)` is −0.0, not +0.0.
        assert!(q(-0.5).ceil().is_sign_negative());
        assert!(q(-0.5).trunc().is_sign_negative());
        // Beyond 2^112 every value is already an integer.
        let big = q(2.0).powi(113);
        assert_eq!(big.trunc(), big);
        assert_eq!(big.fract(), q(0.0));

        assert_eq!(q(7.0) % q(3.0), q(1.0));
        assert_eq!(q(-7.0) % q(3.0), q(-1.0));
        assert_eq!(q(7.0) % q(-3.0), q(1.0));
        assert_eq!(q(5.5) % q(2.0), q(1.5));
        assert_eq!(q(2.0) % q(7.0), q(2.0));
        // A wide exponent gap: the reduction must stay exact over 1000 steps.
        // 2^even ≡ 1 (mod 3), so both of these are exactly 1.
        assert_eq!(q(2.0).powi(100) % q(3.0), q(1.0));
        assert_eq!(q(2.0).powi(1000) % q(3.0), q(1.0));
        let mut r = q(7.0);
        r %= q(3.0);
        assert_eq!(r, q(1.0));

        // `mul_add` must be *fused*. (2^60+1)(2^60−1) = 2^120 − 1 needs 121
        // bits, so rounding the product first snaps it to 2^120 and the
        // subtraction then cancels to nothing. Kept exact, the answer is −1.
        let (a, b) = (q(2f64.powi(60)) + Quad::one(), q(2f64.powi(60)) - Quad::one());
        let c = q(2.0).powi(120);
        assert_eq!(a.mul_add(b, -c), -Quad::one());
        assert_eq!(a * b - c, Quad::zero(), "the unfused product should cancel");
    }

    /// Odd functions must carry `-0.0` through. dashu-float has no signed
    /// zero, so its backing has to recognise the argument rather than convert
    /// it — worth pinning, since `== 0.0` hides the difference.
    #[test]
    fn signed_zero_survives() {
        let negative_zero = -Quad::zero();
        assert!(negative_zero.is_sign_negative(), "the premise");
        for (name, f) in [
            ("sin", Quad::sin as fn(Quad) -> Quad),
            ("tan", Quad::tan),
            ("asin", Quad::asin),
            ("atan", Quad::atan),
            ("sinh", Quad::sinh),
            ("tanh", Quad::tanh),
            ("asinh", Quad::asinh),
            ("atanh", Quad::atanh),
            ("exp_m1", Quad::exp_m1),
            ("ln_1p", Quad::ln_1p),
            ("cbrt", Quad::cbrt),
            ("sqrt", Quad::sqrt),
            ("trunc", Quad::trunc),
            ("round", Quad::round),
        ] {
            assert!(f(negative_zero).is_sign_negative(), "{name}(-0.0) lost its sign");
            assert!(f(Quad::zero()).is_sign_positive(), "{name}(0.0) lost its sign");
        }
        // And the even ones land on 1.0 either way.
        assert_eq!(negative_zero.cos(), Quad::one());
        assert_eq!(negative_zero.exp(), Quad::one());
        assert_eq!(negative_zero.cosh(), Quad::one());
    }

    /// Points where the result is exactly representable and the function is
    /// exact there. Every backing must land on the nose; in the dashu-float
    /// path a mistake in [`super::softfloat::to_f`], [`super::softfloat::from_f`]
    /// or the `ldexp` scaling would show up here as an outright mismatch.
    #[test]
    fn transcendentals_exact_where_exact() {
        assert_eq!(q(1.0).ln(), q(0.0));
        assert_eq!(q(0.0).exp(), q(1.0));
        assert_eq!(q(0.0).sin(), q(0.0));
        assert_eq!(q(1.0).acos(), q(0.0));
        assert_eq!(q(27.0).cbrt(), q(3.0));
        assert_eq!(q(-27.0).cbrt(), q(-3.0));
        assert_eq!(q(3.0).hypot(q(4.0)), q(5.0));
        assert_eq!(q(2.0).powf(q(10.0)), q(1024.0));
        // A negative base with an integral exponent is an ordinary evaluation,
        // not a NaN case; odd exponents keep the sign.
        assert_eq!(q(-2.0).powf(q(11.0)), q(-2048.0));
        assert_eq!(q(-2.0).powf(q(10.0)), q(1024.0));
        assert!(q(-2.0).powf(q(0.5)).is_nan());
    }

    /// Scaling across the exponent range: the dashu-float path rebuilds its
    /// result by stepping `2^e`, so a value far from 1.0 exercises code that
    /// `0.1`-sized arguments never reach.
    #[test]
    fn transcendentals_across_the_exponent_range() {
        // ln/exp round trip across the binary128 exponent field, which reaches
        // far past what an `f64` literal could name. `3^k` stays representable
        // out to k = ±10000 (about 10^±4771).
        for k in [-10_000i32, -1000, -1, 1, 1000, 10_000] {
            let v = q(3.0).powi(k);
            assert!(v.is_finite() && !v.is_zero(), "3^{k} left binary128's range");
            assert_close(v.ln().exp(), v, "exp(ln(x))");
        }
        assert_close(q(1e300).sqrt().powi(2), q(1e300), "sqrt then square");
        // 2^k is exact, and so is its base-2 logarithm.
        for k in [-16000i32, -100, 100, 16000] {
            assert_eq!(q(2.0).powi(k).log2(), q(f64::from(k)));
        }
    }

    /// Inverse pairs, round-tripped: catches a backing evaluating either
    /// direction at reduced precision. Each is composed inverse-outermost so it
    /// holds for every argument, not just on the principal branch.
    #[test]
    fn transcendental_inverses_round_trip() {
        for &x in &[0.1f64, -0.1, 0.5, -0.75, 0.9, 1.5, -2.25, 7.0, 1234.5, 1e-8, 1e8] {
            let v = q(x);
            assert_close(v.sin().asin().sin(), v.sin(), "asin(sin)");
            assert_close(v.cbrt().powi(3), v, "cbrt cubed");
            // The exponentials overflow binary128 long before x = 1e8, and a
            // round trip through infinity means nothing.
            if v.sinh().is_finite() {
                assert_close(v.sinh().asinh(), v, "asinh(sinh)");
            }
            if v.exp_m1().is_finite() {
                assert_close(v.exp_m1().ln_1p(), v, "ln_1p(exp_m1)");
            }

            if x > 0.0 {
                assert_close(v.ln().exp(), v, "exp(ln)");
                assert_close(v.log2().exp2(), v, "exp2(log2)");
                assert_close(v.log10(), v.ln() / q(10.0).ln(), "log10");
                assert_close(v.log(q(3.0)), v.ln() / q(3.0).ln(), "log base 3");
            }

            // The remaining pairs each have a saturating half — `tan`/`atan`
            // at ±π/2, `tanh`/`atanh` at ±1, `exp`/`cosh` flattening onto 1.0
            // at the origin. Outside a moderate window the inverse re-expands
            // whatever the forward direction rounded away, which is a property
            // of the round trip and not of any backing. Their behaviour in
            // those regimes is pinned by the series checks instead.
            if !(0.5..=3.0).contains(&x.abs()) {
                continue;
            }
            assert_close(v.atan().tan(), v, "tan(atan)");
            assert_close(v.tanh().atanh(), v, "atanh(tanh)");
            assert_close(v.exp().ln(), v, "ln(exp)");
            if x > 0.0 {
                assert_close(v.cosh().acosh(), v, "acosh(cosh)");
            }
        }
    }

    /// The identities the dashu-float backing is built on, at arguments where
    /// the naive formulations collapse. `2^-40` is small enough that
    /// `(e^x − e^-x)/2` would cancel away ~80 bits, yet large enough that the
    /// cubic term stays well above binary128's epsilon — so a backing that got
    /// the conditioning wrong cannot hide behind rounding.
    #[test]
    fn small_argument_relative_precision() {
        let x = q(2f64.powi(-40));
        let (x2, x3, x5) = (x * x, x * x * x, x * x * x * x * x);
        let series = |terms: [(Quad, f64); 3]| terms.iter().fold(Quad::zero(), |acc, &(t, d)| acc + t / q(d));

        assert_close(x.sinh(), series([(x, 1.0), (x3, 6.0), (x5, 120.0)]), "sinh");
        assert_close(x.tanh(), series([(x, 1.0), (-x3, 3.0), (x5 * q(2.0), 15.0)]), "tanh");
        assert_close(x.asinh(), series([(x, 1.0), (-x3, 6.0), (x5 * q(3.0), 40.0)]), "asinh");
        assert_close(x.atanh(), series([(x, 1.0), (x3, 3.0), (x5, 5.0)]), "atanh");
        assert_close(x.exp_m1(), series([(x, 1.0), (x2, 2.0), (x3, 6.0)]), "exp_m1");
        assert_close(x.ln_1p(), series([(x, 1.0), (-x2, 2.0), (x3, 3.0)]), "ln_1p");
        assert_close(x.sin(), series([(x, 1.0), (-x3, 6.0), (x5, 120.0)]), "sin");

        // acosh(1+u) = √(2u)·(1 − u/12 + 3u²/160 + …) as u → 0.
        let u = x;
        let expected = (q(2.0) * u).sqrt() * (Quad::one() - u / q(12.0) + q(3.0) * u * u / q(160.0));
        assert_close((Quad::one() + u).acosh(), expected, "acosh near 1");

        // cosh is well conditioned, so pin it with the hyperbolic identity.
        let w = q(3.0);
        assert_close(w.cosh() * w.cosh() - w.sinh() * w.sinh(), Quad::one(), "cosh² − sinh²");
        assert_close(w.tanh(), w.sinh() / w.cosh(), "tanh = sinh/cosh");
    }

    /// `atan2`'s axis cases are exact multiples of π and must come back at full
    /// width — an `f64` detour there would be off in the 17th digit.
    #[test]
    fn atan2_quadrants() {
        let pi = <Quad as FloatConst>::PI();
        let half = pi / q(2.0);
        assert_eq!(q(0.0).atan2(q(1.0)), q(0.0));
        assert_eq!(q(0.0).atan2(q(-1.0)), pi);
        assert_eq!(q(1.0).atan2(q(0.0)), half);
        assert_eq!(q(-1.0).atan2(q(0.0)), -half);
        assert_close(q(1.0).atan2(q(-1.0)), q(3.0) * pi / q(4.0), "second quadrant");
        assert_close(q(-1.0).atan2(q(-1.0)), q(-3.0) * pi / q(4.0), "third quadrant");
        assert_close(q(1e-20).atan2(q(1.0)), q(1e-20), "atan2 near the axis");
        // sin²+cos² = 1 is precision-preserving, unlike most identities.
        let t = q(0.7);
        assert_close(t.sin() * t.sin() + t.cos() * t.cos(), Quad::one(), "sin² + cos²");
        let (s, c) = t.sin_cos();
        assert_eq!((s, c), (t.sin(), t.cos()));
    }

    /// `Quad` (primitive `f128`) must match libquadmath: arithmetic/`sqrt`/rounding
    /// bit-for-bit, transcendentals to ≤ 2 ULP (glibc and libquadmath are
    /// independent libms; an `f64`-then-widened bug would be off by ~2^60 ULP).
    ///
    /// Only runs on x86_64-linux, where the `f128` crate (libquadmath) is
    /// available; macOS/clang has no <quadmath.h> to build its C shim against.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
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
