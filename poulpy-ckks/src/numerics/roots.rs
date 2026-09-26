//! Correctly rounded roots of unity `exp(2πi·k / 2^log_order)`.
//!
//! Every value is the IEEE round-to-nearest-even image of the exact root.
//! Orders up to `2^TABLE_LOG_ORDER` read the checked-in quadrant tables;
//! larger orders evaluate the same definition on demand.

use std::cell::RefCell;

use astro_float_num::{BigFloat, Consts, RoundingMode, WORD_BIT_SIZE, Word};

use super::CKKSFloat;

/// Base-2 logarithm of the order of the checked-in quadrant tables.
pub const TABLE_LOG_ORDER: u32 = 17;
const TABLE_LEN: usize = (1 << (TABLE_LOG_ORDER - 2)) + 1;

pub(super) static COS_QUADRANT_F64: &[u8; TABLE_LEN * 8] = include_bytes!("cos_quadrant_f64.bin");
pub(super) static COS_QUADRANT_F128: &[u8; TABLE_LEN * 16] = include_bytes!("cos_quadrant_f128.bin");

// Evaluation error is below one unit in the last place of the working
// precision; the rounding decision must be stable across this many units.
const GUARD_BITS: usize = 8;
const INITIAL_PRECISION: usize = 320;

std::thread_local! {
    static CONSTANTS: RefCell<Consts> = RefCell::new(Consts::new().expect("failed to initialize root constants"));
}

/// Table index for `cos(2π·i / 2^log_order)`, if the table covers it.
#[inline]
pub(super) fn table_index(i: u64, log_order: u32) -> Option<usize> {
    (log_order <= TABLE_LOG_ORDER).then(|| (i << (TABLE_LOG_ORDER - log_order)) as usize)
}

/// `cos(2π·i / 2^log_order)` for `0 ≤ i ≤ 2^(log_order - 2)`, correctly rounded.
pub(super) fn generated_quadrant_cos<F: CKKSFloat>(i: u64, log_order: u32) -> F {
    let quarter = 1u64 << (log_order - 2);
    assert!(i <= quarter, "quadrant index {i} exceeds 2^{}", log_order - 2);
    if i == 0 {
        return F::one();
    }
    if i == quarter {
        return F::zero();
    }
    let (significand, exponent) = quadrant_cos_bits(i, log_order, F::SIGNIFICAND_BITS);
    F::ckks_dequantize(significand as i128, (-exponent) as usize)
}

/// `(cos, sin)` of `2π·k / 2^log_order`.
pub(super) fn root_of_unity<F: CKKSFloat>(k: u64, log_order: u32) -> (F, F) {
    assert!(log_order < u64::BITS, "root order 2^{log_order} exceeds u64");
    let (k, log_order) = if log_order < 2 {
        (k << (2 - log_order), 2)
    } else {
        (k & ((1u64 << log_order) - 1), log_order)
    };
    // Lower terms keep table-covered roots off the generator.
    let shift = k.trailing_zeros().min(log_order.saturating_sub(TABLE_LOG_ORDER));
    let (k, log_order) = (k >> shift, log_order - shift);
    let order = 1u64 << log_order;
    let cos = circle_cos::<F>(k, log_order);
    let sin = circle_cos::<F>((k + order - order / 4) & (order - 1), log_order);
    (cos, sin)
}

/// `cos(2π·k / 2^log_order)` for `k < 2^log_order`, reduced to the first quadrant.
fn circle_cos<F: CKKSFloat>(k: u64, log_order: u32) -> F {
    let order = 1u64 << log_order;
    let k = if k > order / 2 { order - k } else { k };
    if k <= order / 4 {
        F::ckks_quadrant_cos(k, log_order)
    } else {
        -F::ckks_quadrant_cos(order / 2 - k, log_order)
    }
}

/// Significand and exponent of the correctly rounded `cos(2π·i / 2^log_order)`,
/// for `0 < i < 2^(log_order - 2)`.
fn quadrant_cos_bits(i: u64, log_order: u32, bits: u32) -> (u128, i64) {
    let quarter = 1u64 << (log_order - 2);
    // Past π/4, the sine of the complement keeps the relative error bounded.
    let (j, sine) = if 2 * i <= quarter { (i, false) } else { (quarter - i, true) };
    let rm = RoundingMode::ToEven;
    CONSTANTS.with(|constants| {
        let constants = &mut constants.borrow_mut();
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

/// Round a positive value to `bits` significant bits, or `None` if the
/// evaluation error could change the rounding decision.
fn round_unambiguous(value: &BigFloat, bits: u32) -> Option<(u128, i64)> {
    let (words, _, _, exponent, _) = value.as_raw_parts().expect("finite root");
    let bit = |index: usize| words[index / WORD_BIT_SIZE] & ((1 as Word) << (index % WORD_BIT_SIZE)) != 0;
    let precision = words.len() * WORD_BIT_SIZE;
    let dropped = precision - bits as usize;
    let half = bit(dropped - 1);
    if (GUARD_BITS..dropped - 1).all(|index| bit(index) != half) {
        return None;
    }
    let significand = (0..bits as usize).fold(0u128, |acc, index| acc | (u128::from(bit(dropped + index)) << index));
    Some((significand + u128::from(half), exponent as i64 - bits as i64))
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
        assert_eq!(COS_QUADRANT_F64.as_slice(), quadrant_bits::<f64>(TABLE_LOG_ORDER));
        assert_eq!(COS_QUADRANT_F128.as_slice(), quadrant_bits::<Quad>(TABLE_LOG_ORDER));
    }

    #[test]
    #[ignore = "rewrites the checked-in root tables"]
    fn regenerate_quadrant_tables() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/numerics");
        std::fs::write(dir.join("cos_quadrant_f64.bin"), quadrant_bits::<f64>(TABLE_LOG_ORDER)).unwrap();
        std::fs::write(dir.join("cos_quadrant_f128.bin"), quadrant_bits::<Quad>(TABLE_LOG_ORDER)).unwrap();
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
        check_symmetries::<f64>(10);
        check_symmetries::<Quad>(8);
        assert_eq!(f64::ckks_root_of_unity(0, 0), (1.0, 0.0));
        assert_eq!(f64::ckks_root_of_unity(1, 1), (-1.0, 0.0));
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
