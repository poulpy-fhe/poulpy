//! Correctly rounded roots of unity `exp(2*pi*i * k / 2^log_order)`.
//!
//! Every value is the round-to-nearest-even image of the exact root.
//! Orders up to `2^TABLE_LOG_ORDER` read the checked-in quadrant table and
//! larger orders evaluate the same definition on demand.

use astro_float_num::{BigFloat, RoundingMode, WORD_BIT_SIZE};

use super::{
    CKKSFloat,
    astro::{mantissa_bit, rounded_shift, with_constants},
};

/// Base-2 logarithm of the order of the checked-in quadrant table.
pub const TABLE_LOG_ORDER: u32 = 19;
const TABLE_LEN: usize = (1 << (TABLE_LOG_ORDER - 2)) + 1;

pub(super) static COS_QUADRANT_F128: &[u8; TABLE_LEN * 16] = include_bytes!("cos_quadrant_f128.bin");

// The evaluation error is below one unit in the last place of the working
// precision, so the rounding decision must be stable across this many units.
const GUARD_BITS: usize = 8;
const INITIAL_PRECISION: usize = 320;

fn validate_quadrant(i: u64, log_order: u32) {
    assert!((2..64).contains(&log_order), "quadrant order must be in 2..64");
    assert!(i <= 1u64 << (log_order - 2), "quadrant index exceeds the first quadrant");
}

// Precisions whose rounding of the binary128 table entries is exhaustively
// checked against direct correct rounding.
const TABLE_PRECISIONS: [u32; 3] = [24, 53, 113];

fn table_bits(index: usize) -> u128 {
    u128::from_le_bytes(COS_QUADRANT_F128[16 * index..16 * index + 16].try_into().unwrap())
}

/// `cos(2*pi * i / 2^log_order)` for `0 <= i <= 2^(log_order - 2)`, correctly rounded.
fn quadrant_cos<F: CKKSFloat>(i: u64, log_order: u32) -> F {
    validate_quadrant(i, log_order);
    if log_order <= TABLE_LOG_ORDER && TABLE_PRECISIONS.contains(&F::SIGNIFICAND_BITS) {
        table_value((i << (TABLE_LOG_ORDER - log_order)) as usize)
    } else {
        generated_quadrant_cos(i, log_order)
    }
}

/// Rounds the binary128 table entry, a value in `[0, 1]`, once to `F`.
fn table_value<F: CKKSFloat>(index: usize) -> F {
    let bits = table_bits(index);
    let exponent = (bits >> 112) as usize;
    if exponent == 0 {
        return F::zero();
    }
    let significand = (bits & ((1 << 112) - 1)) | (1 << 112);
    F::ckks_dequantize(significand as i128, 16383 + 112 - exponent)
}

/// `cos(2*pi * i / 2^log_order)` for `0 <= i <= 2^(log_order - 2)`, correctly rounded.
fn generated_quadrant_cos<F: CKKSFloat>(i: u64, log_order: u32) -> F {
    // The rounded significand, possibly carried to 2^bits, must fit an i128.
    const { assert!(F::SIGNIFICAND_BITS <= 126, "CKKSFloat::SIGNIFICAND_BITS exceeds 126") };
    validate_quadrant(i, log_order);
    let quarter = 1u64 << (log_order - 2);
    if i == 0 {
        return F::one();
    }
    if i == quarter {
        return F::zero();
    }
    let (significand, exponent) = quadrant_cos_bits(i, log_order, F::SIGNIFICAND_BITS);
    F::ckks_dequantize(significand as i128, (-exponent) as usize)
}

/// `(cos, sin)` of `2*pi * k / 2^log_order`.
pub(super) fn root_of_unity<F: CKKSFloat>(k: u64, log_order: u32) -> (F, F) {
    assert!(log_order < u64::BITS, "root order 2^{log_order} exceeds u64");
    let (k, log_order) = if log_order < 2 {
        ((k & ((1u64 << log_order) - 1)) << (2 - log_order), 2)
    } else {
        (k & ((1u64 << log_order) - 1), log_order)
    };
    // Lowest terms keep the roots that the table covers off the generator.
    let shift = k.trailing_zeros().min(log_order.saturating_sub(TABLE_LOG_ORDER));
    let (k, log_order) = (k >> shift, log_order - shift);
    let order = 1u64 << log_order;
    let cos = circle_cos::<F>(k, log_order);
    let sin = circle_cos::<F>((k + order - order / 4) & (order - 1), log_order);
    (cos, sin)
}

/// `cos(2*pi * k / 2^log_order)` for `k < 2^log_order`, reduced to the first quadrant.
fn circle_cos<F: CKKSFloat>(k: u64, log_order: u32) -> F {
    let order = 1u64 << log_order;
    let k = if k > order / 2 { order - k } else { k };
    if k <= order / 4 {
        quadrant_cos::<F>(k, log_order)
    } else {
        -quadrant_cos::<F>(order / 2 - k, log_order)
    }
}

/// Significand and exponent of the correctly rounded `cos(2*pi * i / 2^log_order)`,
/// for `0 < i < 2^(log_order - 2)`.
fn quadrant_cos_bits(i: u64, log_order: u32, bits: u32) -> (u128, i64) {
    let quarter = 1u64 << (log_order - 2);
    // Past pi/4, the sine of the complement keeps the relative error bounded.
    let (j, sine) = if 2 * i <= quarter { (i, false) } else { (quarter - i, true) };
    let rm = RoundingMode::ToEven;
    with_constants(|constants| {
        let mut precision = INITIAL_PRECISION;
        loop {
            let wide = precision + 2 * WORD_BIT_SIZE;
            let mut theta = constants.pi(wide, rm).mul(&BigFloat::from_u64(2 * j, wide), wide, rm);
            theta.set_exponent(theta.exponent().expect("finite angle") - log_order as i32);
            let value = if sine {
                theta.sin(precision, rm, constants)
            } else {
                theta.cos(precision, rm, constants)
            };
            if let Some(rounded) = round_unambiguous(&value, bits) {
                return rounded;
            }
            precision *= 2;
        }
    })
}

/// Rounds a positive value to `bits` significant bits, or returns `None` if
/// the evaluation error could change the rounding decision.
fn round_unambiguous(value: &BigFloat, bits: u32) -> Option<(u128, i64)> {
    let (words, _, _, exponent, _) = value.as_raw_parts().expect("finite root");
    let bit = |index: usize| mantissa_bit(words, index);
    let precision = words.len() * WORD_BIT_SIZE;
    let dropped = precision - bits as usize;
    let half = bit(dropped - 1);
    if (GUARD_BITS..dropped - 1).all(|index| bit(index) != half) {
        return None;
    }
    Some((rounded_shift(words, dropped), exponent as i64 - bits as i64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Quad;
    use num_traits::ToPrimitive;

    fn quadrant_bits<F: CKKSFloat + bytemuck::Pod>(log_order: u32) -> Vec<u8> {
        let quarter = 1u64 << (log_order - 2);
        (0..=quarter)
            .flat_map(|i| {
                let value = generated_quadrant_cos::<F>(i, log_order);
                let mut bytes = bytemuck::bytes_of(&value).to_vec();
                if cfg!(target_endian = "big") {
                    bytes.reverse();
                }
                bytes
            })
            .collect()
    }

    #[test]
    fn quadrant_tables_match_generator() {
        assert_eq!(COS_QUADRANT_F128.as_slice(), quadrant_bits::<Quad>(TABLE_LOG_ORDER));
    }

    #[test]
    fn derived_roots_match_direct_rounding() {
        let quarter = 1u64 << (TABLE_LOG_ORDER - 2);
        for i in 0..=quarter {
            assert_eq!(
                quadrant_cos::<Quad>(i, TABLE_LOG_ORDER).to_bits(),
                table_bits(i as usize),
                "Quad {i}/2^{TABLE_LOG_ORDER}"
            );
            assert_eq!(
                quadrant_cos::<f64>(i, TABLE_LOG_ORDER).to_bits(),
                generated_quadrant_cos::<f64>(i, TABLE_LOG_ORDER).to_bits(),
                "f64 {i}/2^{TABLE_LOG_ORDER}"
            );
            assert_eq!(
                quadrant_cos::<f32>(i, TABLE_LOG_ORDER).to_bits(),
                generated_quadrant_cos::<f32>(i, TABLE_LOG_ORDER).to_bits(),
                "f32 {i}/2^{TABLE_LOG_ORDER}"
            );
        }
    }

    #[test]
    #[ignore = "rewrites the checked-in root tables"]
    fn regenerate_quadrant_tables() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/numerics");
        std::fs::write(dir.join("cos_quadrant_f128.bin"), quadrant_bits::<Quad>(TABLE_LOG_ORDER)).unwrap();
    }

    #[test]
    fn quadrant_cos_rejects_invalid_inputs() {
        fn check<F: CKKSFloat>() {
            for (i, order) in [
                (0, 0),
                (0, 1),
                (0, 64),
                (0, u32::MAX),
                (2, 2),
                (u64::MAX, TABLE_LOG_ORDER),
                (u64::MAX, TABLE_LOG_ORDER + 1),
            ] {
                assert!(std::panic::catch_unwind(|| quadrant_cos::<F>(i, order)).is_err());
                assert!(std::panic::catch_unwind(|| generated_quadrant_cos::<F>(i, order)).is_err());
            }
            for order in [2, TABLE_LOG_ORDER, TABLE_LOG_ORDER + 1, 63] {
                assert!(quadrant_cos::<F>(0, order) == F::one());
                assert!(quadrant_cos::<F>(1 << (order - 2), order) == F::zero());
            }
        }
        check::<f32>();
        check::<f64>();
        check::<Quad>();
    }

    fn check_symmetries<F: CKKSFloat + std::fmt::Debug>(log_order: u32) {
        let order = 1u64 << log_order;
        for k in 0..order {
            let (c, s) = F::ckks_root_of_unity(k, log_order);
            let (c_neg, s_neg) = F::ckks_root_of_unity(order - k, log_order);
            let (c_rot, s_rot) = F::ckks_root_of_unity(k + order / 4, log_order);
            assert_eq!((c_neg, s_neg), (c, -s), "conjugate of {k}/{order}");
            assert_eq!((c_rot, s_rot), (-s, c), "quarter turn of {k}/{order}");
            assert_eq!(
                F::ckks_root_of_unity(2 * k, log_order + 1),
                (c, s),
                "{k}/{order} in lowest terms"
            );
        }
    }

    #[test]
    fn roots_are_exactly_symmetric() {
        check_symmetries::<f32>(10);
        check_symmetries::<f64>(10);
        check_symmetries::<Quad>(8);
        assert_eq!(f64::ckks_root_of_unity(0, 0), (1.0, 0.0));
        assert_eq!(f64::ckks_root_of_unity(1, 1), (-1.0, 0.0));
        assert_eq!(f64::ckks_root_of_unity(5, 0), (1.0, 0.0));
        assert_eq!(f64::ckks_root_of_unity(3, 1), (-1.0, 0.0));
        assert_eq!(f64::ckks_root_of_unity(u64::MAX, 3), f64::ckks_root_of_unity(7, 3));
        let (c, s) = f64::ckks_root_of_unity(1, 2);
        assert_eq!((c.to_bits(), s), (0, 1.0));
    }

    #[test]
    fn roots_beyond_the_table_match_its_definition() {
        let log_order = TABLE_LOG_ORDER + 3;
        for k in [1u64, 3, 12345, (1 << (log_order - 3)) - 1, (1 << (log_order - 3)) + 7] {
            let (c, s) = f64::ckks_root_of_unity(k, log_order);
            let angle = std::f64::consts::TAU * k as f64 / (1u64 << log_order) as f64;
            assert!((c - angle.cos()).abs() <= f64::EPSILON && (s - angle.sin()).abs() <= f64::EPSILON);
            let (cq, sq) = Quad::ckks_root_of_unity(k, log_order);
            assert!((cq.to_f64().unwrap() - c).abs() <= f64::EPSILON / 2.0);
            assert!((sq.to_f64().unwrap() - s).abs() <= f64::EPSILON / 2.0);
            assert_eq!(f64::ckks_root_of_unity(k << 5, log_order + 5), (c, s));
        }
    }
}
