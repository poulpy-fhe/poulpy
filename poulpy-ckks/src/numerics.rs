//! Platform-independent scalar math for CKKS setup and encoding.
//!
//! Every function here returns the same bits on every target, so plaintext
//! constants and encoded coefficients do not depend on the platform libm or
//! on the backend that runs the encoding.

use num_traits::{Float, FromPrimitive};

use crate::Quad;

/// Scalar with platform-independent transcendental functions and exact
/// conversions between scalars and plaintext integers.
///
/// Arithmetic uses round to nearest, ties to even, with gradual underflow.
/// `f32` and `f64` evaluate through soft-float `libm` and `Quad` through the
/// pure-Rust binary128 implementation, whatever the target or the `Quad`
/// routing.
pub trait CKKSFloat: Float + FromPrimitive {
    fn ckks_sin(self) -> Self;
    fn ckks_cos(self) -> Self;
    fn ckks_powf(self, exponent: Self) -> Self;
    fn ckks_exp2(self) -> Self;
    fn ckks_log2(self) -> Self;
    fn ckks_sqrt(self) -> Self;

    fn ckks_sin_cos(self) -> (Self, Self) {
        (self.ckks_sin(), self.ckks_cos())
    }

    /// Integer power by binary exponentiation, with one rounding per product.
    fn ckks_powi(self, exponent: i32) -> Self {
        let mut n = exponent.unsigned_abs();
        let mut x = self;
        let mut y = Self::one();
        while n != 0 {
            if n & 1 != 0 {
                y = y * x;
            }
            n >>= 1;
            if n != 0 {
                x = x * x;
            }
        }
        if exponent < 0 { y.recip() } else { y }
    }

    /// Rounds `self * 2^log_delta` to an integer, ties away from zero.
    /// Returns `None` for non-finite values and signed integer overflow.
    fn ckks_quantize(self, log_delta: usize) -> Option<i128>;

    /// Quantizes with the same rounding, returning `None` outside `i64`.
    #[inline]
    fn ckks_quantize_i64(self, log_delta: usize) -> Option<i64> {
        self.ckks_quantize(log_delta).and_then(|value| i64::try_from(value).ok())
    }

    /// Rounds `value * 2^-log_delta` once, ties to even.
    fn ckks_dequantize(value: i128, log_delta: usize) -> Self;
}

impl CKKSFloat for f64 {
    fn ckks_sin(self) -> Self {
        libm::sin(self)
    }
    fn ckks_cos(self) -> Self {
        libm::cos(self)
    }
    fn ckks_powf(self, exponent: Self) -> Self {
        libm::pow(self, exponent)
    }
    fn ckks_exp2(self) -> Self {
        libm::exp2(self)
    }
    fn ckks_log2(self) -> Self {
        libm::log2(self)
    }
    fn ckks_sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn ckks_quantize(self, log_delta: usize) -> Option<i128> {
        quantize(self.to_bits() as u128, 52, 11, log_delta)
    }
    #[inline]
    fn ckks_quantize_i64(self, log_delta: usize) -> Option<i64> {
        if log_delta <= 1023 {
            // Scaling by a finite power of two is exact unless it overflows.
            let scale = Self::from_bits(((log_delta + 1023) as u64) << 52);
            return num_traits::ToPrimitive::to_i64(&(self * scale).round());
        }
        self.ckks_quantize(log_delta).and_then(|value| i64::try_from(value).ok())
    }
    #[inline]
    fn ckks_dequantize(value: i128, log_delta: usize) -> Self {
        Self::from_bits(dequantize(value, log_delta, 52, 11) as u64)
    }
}

impl CKKSFloat for f32 {
    fn ckks_sin(self) -> Self {
        libm::sinf(self)
    }
    fn ckks_cos(self) -> Self {
        libm::cosf(self)
    }
    fn ckks_powf(self, exponent: Self) -> Self {
        libm::powf(self, exponent)
    }
    fn ckks_exp2(self) -> Self {
        libm::exp2f(self)
    }
    fn ckks_log2(self) -> Self {
        libm::log2f(self)
    }
    fn ckks_sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn ckks_quantize(self, log_delta: usize) -> Option<i128> {
        quantize(self.to_bits() as u128, 23, 8, log_delta)
    }
    #[inline]
    fn ckks_dequantize(value: i128, log_delta: usize) -> Self {
        Self::from_bits(dequantize(value, log_delta, 23, 8) as u32)
    }
}

impl CKKSFloat for Quad {
    fn ckks_sin(self) -> Self {
        Self(crate::scalar::backing::portable::sin(self.0))
    }
    fn ckks_cos(self) -> Self {
        Self(crate::scalar::backing::portable::cos(self.0))
    }
    fn ckks_sin_cos(self) -> (Self, Self) {
        let (s, c) = crate::scalar::backing::portable::sin_cos(self.0);
        (Self(s), Self(c))
    }
    fn ckks_powf(self, exponent: Self) -> Self {
        Self(crate::scalar::backing::portable::powf(self.0, exponent.0))
    }
    fn ckks_exp2(self) -> Self {
        Self(crate::scalar::backing::portable::exp2(self.0))
    }
    fn ckks_log2(self) -> Self {
        Self(crate::scalar::backing::portable::log2(self.0))
    }
    fn ckks_sqrt(self) -> Self {
        Self(crate::scalar::backing::portable::sqrt(self.0))
    }
    #[inline]
    fn ckks_quantize(self, log_delta: usize) -> Option<i128> {
        quantize(self.to_bits(), 112, 15, log_delta)
    }
    #[inline]
    fn ckks_dequantize(value: i128, log_delta: usize) -> Self {
        Self::from_bits(dequantize(value, log_delta, 112, 15))
    }
}

/// Exact `round(x * 2^log_delta)` on the IEEE encoding of `x`.
#[inline]
fn quantize(bits: u128, fraction_bits: u32, exponent_bits: u32, log_delta: usize) -> Option<i128> {
    let exponent_mask = (1u128 << exponent_bits) - 1;
    let exponent = (bits >> fraction_bits) & exponent_mask;
    if exponent == exponent_mask {
        return None;
    }
    let negative = bits >> (fraction_bits + exponent_bits) != 0;
    let fraction = bits & ((1u128 << fraction_bits) - 1);
    let significand = fraction | (u128::from(exponent != 0) << fraction_bits);
    if significand == 0 {
        return Some(0);
    }
    let bias = (1i64 << (exponent_bits - 1)) - 1;
    let shift = (exponent.max(1) as i64 - bias - fraction_bits as i64).checked_add(i64::try_from(log_delta).ok()?)?;
    let magnitude = if shift >= 0 {
        if shift > significand.leading_zeros() as i64 {
            return None;
        }
        significand.checked_shl(u32::try_from(shift).ok()?)?
    } else {
        round_shift(significand, shift.unsigned_abs(), false)
    };
    if negative && magnitude == 1u128 << 127 {
        Some(i128::MIN)
    } else {
        let value = i128::try_from(magnitude).ok()?;
        Some(if negative { -value } else { value })
    }
}

/// `value >> shift` rounded to nearest, ties to even or away from zero.
#[inline]
fn round_shift(value: u128, shift: u64, ties_even: bool) -> u128 {
    if shift == 0 {
        return value;
    }
    if shift > 128 {
        return 0;
    }
    let half = 1u128 << (shift - 1);
    let quotient = value.checked_shr(shift as u32).unwrap_or(0);
    let remainder = value & (half | (half - 1));
    quotient + u128::from(remainder > half || (remainder == half && (!ties_even || quotient & 1 != 0)))
}

/// IEEE encoding of `value * 2^-log_delta`, rounded once to nearest even.
#[inline]
fn dequantize(value: i128, log_delta: usize, fraction_bits: u32, exponent_bits: u32) -> u128 {
    if value == 0 {
        return 0;
    }
    let sign = u128::from(value < 0) << (fraction_bits + exponent_bits);
    let magnitude = value.unsigned_abs();
    let top = 127 - magnitude.leading_zeros();
    let Ok(scale) = i64::try_from(log_delta) else {
        return sign;
    };
    let bias = (1i64 << (exponent_bits - 1)) - 1;
    let exponent = top as i64 - scale;
    if exponent < 1 - bias {
        let shift = scale.saturating_add(1 - bias - fraction_bits as i64);
        let fraction = if shift > 0 {
            round_shift(magnitude, shift as u64, true)
        } else {
            magnitude << (-shift as u32)
        };
        return sign | fraction;
    }
    let mut significand = if top > fraction_bits {
        round_shift(magnitude, (top - fraction_bits) as u64, true)
    } else {
        magnitude << (fraction_bits - top)
    };
    let mut encoded_exponent = exponent + bias;
    if significand >> (fraction_bits + 1) != 0 {
        significand >>= 1;
        encoded_exponent += 1;
    }
    sign | ((encoded_exponent as u128) << fraction_bits) | (significand & ((1u128 << fraction_bits) - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codec_edges<F: CKKSFloat>() {
        for (x, want) in [(0.0, 0), (-0.0, 0), (0.5, 1), (-0.5, -1), (1.5, 2), (-1.5, -2)] {
            assert_eq!(F::from_f64(x).unwrap().ckks_quantize(0), Some(want));
            assert_eq!(F::from_f64(x).unwrap().ckks_quantize_i64(0), Some(want as i64));
        }
        assert_eq!(F::nan().ckks_quantize(0), None);
        assert_eq!(F::infinity().ckks_quantize(0), None);
        assert_eq!(F::neg_infinity().ckks_quantize(0), None);
        assert_eq!(F::one().ckks_quantize(127), None);
        assert_eq!((-F::one()).ckks_quantize(127), Some(i128::MIN));
        assert_eq!(F::one().ckks_quantize(128), None);
        assert_eq!(F::one().ckks_quantize(usize::MAX), None);
        assert_eq!(F::zero().ckks_quantize(usize::MAX), Some(0));
        assert_eq!(F::nan().ckks_quantize_i64(0), None);
        assert_eq!(F::infinity().ckks_quantize_i64(0), None);
        assert_eq!(F::neg_infinity().ckks_quantize_i64(0), None);
        assert_eq!(F::one().ckks_quantize_i64(63), None);
        assert_eq!((-F::one()).ckks_quantize_i64(63), Some(i64::MIN));
        assert_eq!(F::one().ckks_quantize_i64(usize::MAX), None);
        assert_eq!(F::zero().ckks_quantize_i64(usize::MAX), Some(0));
    }

    #[test]
    fn exact_codec_boundaries() {
        codec_edges::<f32>();
        codec_edges::<f64>();
        codec_edges::<Quad>();
        assert_eq!(f32::from_bits(1).ckks_quantize(149), Some(1));
        assert_eq!(f32::ckks_dequantize(1, 149).to_bits(), 1);
        assert_eq!(f32::ckks_dequantize(3, 150).to_bits(), 2);
        assert_eq!(f32::ckks_dequantize((1 << 24) + 1, 0), (1u32 << 24) as f32);
        assert_eq!(f64::from_bits(1).ckks_quantize(1074), Some(1));
        assert_eq!(Quad::from_bits(1).ckks_quantize(16494), Some(1));
        assert_eq!(f64::from_bits(1).ckks_quantize_i64(1074), Some(1));
        assert_eq!(Quad::from_bits(1).ckks_quantize_i64(16494), Some(1));
        for bits in [0x43e0000000000000u64, 0xc3e0000000000000] {
            for bits in [bits - 1, bits, bits + 1] {
                let x = f64::from_bits(bits);
                assert_eq!(x.ckks_quantize_i64(0), x.ckks_quantize(0).and_then(|x| i64::try_from(x).ok()));
            }
        }
        assert_eq!(f64::ckks_dequantize(1, 1074).to_bits(), 1);
        assert_eq!(f64::ckks_dequantize(1, 1075).to_bits(), 0);
        assert_eq!(f64::ckks_dequantize(3, 1075).to_bits(), 2);
        assert_eq!(f64::ckks_dequantize(-1, 1075).to_bits(), 1 << 63);
        assert_eq!(Quad::ckks_dequantize(1, 16494).to_bits(), 1);
        assert_eq!(Quad::ckks_dequantize(3, 16495).to_bits(), 2);
        assert_eq!(
            f64::ckks_dequantize((1 << 53) + 1, 0).to_bits(),
            ((1u64 << 53) as f64).to_bits()
        );
        assert_eq!(f64::ckks_dequantize((1 << 53) - 1, 1075).to_bits(), 1 << 52);
        assert_eq!(Quad::ckks_dequantize((1 << 113) - 1, 16495).to_bits(), 1 << 112);
        assert_eq!(f64::ckks_dequantize((1 << 54) + 5, 1077).to_bits(), (1 << 51) + 1);
        assert_eq!(Quad::ckks_dequantize((1 << 114) + 5, 16497).to_bits(), (1 << 111) + 1);
    }

    /// `2^e` from its encoding. The primitive `f128` math, including `powi`
    /// and `round`, is wrong on some targets, such as Apple Silicon.
    fn pow2_f128(e: i32) -> f128 {
        f128::from_bits(((e + 16383) as u128) << 112)
    }

    #[test]
    fn quantization_matches_binary128_arithmetic() {
        let mut state = 0x853c49e6748fea9bu64;
        for _ in 0..10000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = f64::from_bits(state);
            for delta in [0, 30, 60, 110, 1023, 1024, 1074] {
                let rounded = libm::roundf128((x as f128) * pow2_f128(delta as i32));
                let expected = if !rounded.is_finite() || !(-pow2_f128(127)..pow2_f128(127)).contains(&rounded) {
                    None
                } else {
                    Some(rounded as i128)
                };
                assert_eq!(x.ckks_quantize(delta), expected, "{x:?} at {delta}");
                assert_eq!(Quad(x as f128).ckks_quantize(delta), expected, "Quad({x:?}) at {delta}");
                let narrow = x as f32;
                let rounded = libm::roundf128((narrow as f128) * pow2_f128(delta as i32));
                let expected_f32 =
                    (rounded.is_finite() && (-pow2_f128(127)..pow2_f128(127)).contains(&rounded)).then_some(rounded as i128);
                assert_eq!(narrow.ckks_quantize(delta), expected_f32, "{narrow:?} at {delta}");
                let expected_i64 = expected.and_then(|value| i64::try_from(value).ok());
                assert_eq!(x.ckks_quantize_i64(delta), expected_i64, "i64({x:?}) at {delta}");
                assert_eq!(
                    Quad(x as f128).ckks_quantize_i64(delta),
                    expected_i64,
                    "i64(Quad({x:?})) at {delta}"
                );
            }
        }
    }

    #[test]
    fn dequantization_inverts_exact_quantization() {
        let mut state = 0x2545f4914f6cdd1du64;
        for _ in 0..10000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let value = (state as i64 >> 11) as i128;
            for delta in [0, 30, 52, 60] {
                let x = f64::ckks_dequantize(value, delta);
                assert_eq!(x, value as f64 / 2f64.powi(delta as i32), "{value} at {delta}");
                assert_eq!(x.ckks_quantize(delta), Some(value), "{value} at {delta}");
            }
        }
    }
}
