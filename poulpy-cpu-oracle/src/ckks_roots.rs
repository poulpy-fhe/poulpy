//! Correctly rounded roots of unity from exact fixed-point integer arithmetic.
//!
//! Pi comes from Machin's formula and the cosine from its Taylor series, both
//! evaluated on scaled integers with a bounded truncation error. The
//! production tables of `poulpy-ckks` are checked against these values, not
//! derived from them.

use std::sync::OnceLock;

use dashu_int::{IBig, UBig, ops::BitTest};
use poulpy_ckks::numerics::CKKSFloat;

const SCALE: usize = 448;
/// The accumulated truncation error is below `2^ERROR_BITS` units of `2^-SCALE`.
const ERROR_BITS: usize = 16;
const CACHED_LOG_ORDER: u32 = 18;

fn atan_inverse(x: u64) -> IBig {
    let mut power = (UBig::ONE << SCALE) / x;
    let mut sum = IBig::ZERO;
    let mut n = 0u64;
    while power != UBig::ZERO {
        let term = IBig::from(&power / (2 * n + 1));
        if n.is_multiple_of(2) {
            sum += term;
        } else {
            sum -= term;
        }
        power /= x * x;
        n += 1;
    }
    sum
}

fn pi() -> &'static UBig {
    static PI: OnceLock<UBig> = OnceLock::new();
    PI.get_or_init(|| {
        let pi = IBig::from(16u8) * atan_inverse(5) - IBig::from(4u8) * atan_inverse(239);
        UBig::try_from(pi).unwrap()
    })
}

/// `cos(2*pi * i / 2^log_order) * 2^SCALE` for `i <= 2^(log_order - 2)`.
fn fixed_cos(i: u64, log_order: u32) -> UBig {
    let theta = (pi() * UBig::from(2 * i)) >> log_order as usize;
    let theta2 = (&theta * &theta) >> SCALE;
    let mut term = UBig::ONE << SCALE;
    let mut sum = IBig::from(term.clone());
    let mut n = 1u64;
    loop {
        term = ((&term * &theta2) >> SCALE) / ((2 * n - 1) * (2 * n));
        if term == UBig::ZERO {
            break;
        }
        if n.is_multiple_of(2) {
            sum += IBig::from(term.clone());
        } else {
            sum -= IBig::from(term.clone());
        }
        n += 1;
    }
    UBig::try_from(sum).unwrap()
}

fn cached_fixed_cos(i: u64) -> &'static UBig {
    static QUADRANT: OnceLock<Vec<UBig>> = OnceLock::new();
    &QUADRANT.get_or_init(|| {
        (0..=1u64 << (CACHED_LOG_ORDER - 2))
            .map(|i| fixed_cos(i, CACHED_LOG_ORDER))
            .collect()
    })[i as usize]
}

fn round_to(value: &UBig, bits: usize) -> UBig {
    let dropped = value.bit_len() - bits;
    let half = UBig::ONE << (dropped - 1);
    let remainder = value % (UBig::ONE << dropped);
    let significand = value >> dropped;
    if remainder > half || (remainder == half && significand.bit(0)) {
        significand + UBig::ONE
    } else {
        significand
    }
}

/// Rounds a fixed-point value, asserting that its error cannot change the result.
fn to_scalar<F: CKKSFloat>(value: &UBig) -> F {
    let bits = F::SIGNIFICAND_BITS as usize;
    let error = UBig::ONE << ERROR_BITS;
    let (low, high) = (value - &error, value + &error);
    assert_eq!(low.bit_len(), high.bit_len(), "fixed-point root crosses a binade");
    let significand = round_to(&low, bits);
    assert_eq!(
        significand,
        round_to(&high, bits),
        "fixed-point root is too close to a rounding boundary"
    );
    let mut scalar = F::from_u128(u128::try_from(&significand).unwrap()).unwrap();
    let half = F::from_f64(0.5).unwrap();
    for _ in 0..SCALE - (low.bit_len() - bits) {
        scalar = scalar * half;
    }
    scalar
}

fn quadrant_cos<F: CKKSFloat>(i: u64, log_order: u32) -> F {
    let quarter = 1u64 << (log_order - 2);
    if i == 0 {
        F::one()
    } else if i == quarter {
        F::zero()
    } else if log_order <= CACHED_LOG_ORDER {
        to_scalar(cached_fixed_cos(i << (CACHED_LOG_ORDER - log_order)))
    } else {
        to_scalar(&fixed_cos(i, log_order))
    }
}

fn cos<F: CKKSFloat>(k: u64, log_order: u32) -> F {
    let order = 1u64 << log_order;
    let k = k % order;
    let k = k.min(order - k);
    if 4 * k <= order {
        quadrant_cos(k, log_order)
    } else {
        -quadrant_cos::<F>(order / 2 - k, log_order)
    }
}

/// `(cos, sin)` of `2*pi * k / 2^log_order`, correctly rounded.
pub(crate) fn root_of_unity<F: CKKSFloat>(k: u64, log_order: u32) -> (F, F) {
    assert!(log_order >= 2);
    let order = 1u64 << log_order;
    let k = k % order;
    (cos(k, log_order), cos(order + order / 4 - k, log_order))
}

#[cfg(test)]
mod tests {
    use poulpy_ckks::{Quad, numerics::CKKSFloat};

    fn check<F: CKKSFloat + std::fmt::Debug>() {
        let log_order = super::CACHED_LOG_ORDER;
        for k in 0..1u64 << log_order {
            assert_eq!(
                super::root_of_unity::<F>(k, log_order),
                F::ckks_root_of_unity(k, log_order),
                "{k}/2^{log_order}"
            );
        }
        for k in [1u64, 5, 1 << 18, (1 << 19) - 1] {
            assert_eq!(super::root_of_unity::<F>(k, 20), F::ckks_root_of_unity(k, 20), "{k}/2^20");
        }
    }

    #[test]
    fn production_roots_match_independent_derivation() {
        check::<f32>();
        check::<f64>();
        check::<Quad>();
    }
}
