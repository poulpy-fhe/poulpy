//! Galois elements of the linear-transformation layer.
//!
//! A Galois element `p`, odd, names the automorphism `X -> X^p`. The group of these modulo a
//! power of two `M >= 8` is `{1, -1} x <5>`, so `p = (-1)^c 5^j` for a conjugation bit `c` and a
//! rotation `j` modulo `M / 4`. Linear transformations hold their elements at
//! [`GALOIS_ORDER`], which every operand's order (`2n` on the standard ring, `4n` on the
//! conjugate-invariant ring) divides, in the form `(-1)^c (5^j mod GALOIS_ORDER)` that
//! [`galois_element`](poulpy_hal::layouts::galois_element) returns, and an element applies to an
//! operand of order `M` as [`galois_at`] reduces it.

use poulpy_hal::layouts::{MAX_RING_DEGREE, galois_element, mod_exp_u64};

/// The cyclotomic order Galois elements are held at: `4 * MAX_RING_DEGREE`, a multiple of the
/// order of every operand a module serves, on both rings. 5 has order [`MAX_RING_DEGREE`] modulo
/// it, so rotations are taken modulo [`MAX_RING_DEGREE`].
pub const GALOIS_ORDER: i64 = 4 * MAX_RING_DEGREE as i64;

/// The Galois element `(-1)^conjugate 5^rotation`, `rotation` of any sign.
pub fn galois_from_parts(conjugate: bool, rotation: i64) -> i64 {
    let power: i64 = galois_element(rotation.rem_euclid(MAX_RING_DEGREE as i64), GALOIS_ORDER);
    if conjugate { -power } else { power }
}

/// The Galois element of the slot rotation by `rotation`, of any sign: `X -> X^(5^rotation)`.
pub fn galois_of_rotation(rotation: i64) -> i64 {
    galois_from_parts(false, rotation)
}

/// The conjugation bit and the rotation, in `[0, MAX_RING_DEGREE)`, of the Galois element `p`.
///
/// # Algorithm
///
/// The conjugation bit `c` is set when `p = 3` modulo 4, as 5 is 1 there. The rotation is the
/// discrete logarithm in base 5 of `x = (-1)^c p`, found one bit at a time from the least
/// significant, as Pohlig-Hellman does in a group whose order is a power of two:
/// `5^(2^k) = 1 + 2^(k + 2)` modulo `2^(k + 3)`, so once the power of the lower bits equals `x`
/// modulo `2^(k + 2)`, bit `k` is set exactly when the two differ modulo `2^(k + 3)`.
///
/// # Complexity
///
/// `log2(MAX_RING_DEGREE)` iterations, each one squaring and at most one multiplication modulo
/// [`GALOIS_ORDER`], on machine words and with no table.
///
/// # Panics
///
/// When `p` is even.
pub fn galois_parts(p: i64) -> (bool, i64) {
    assert!(p & 1 == 1, "a Galois element is odd");
    let q: i64 = p.rem_euclid(GALOIS_ORDER);
    let (conjugate, x) = if q & 3 == 1 { (false, q) } else { (true, GALOIS_ORDER - q) };
    let (mut rotation, mut power, mut step): (i64, i64, i64) = (0, 1, 5);
    for k in 0..MAX_RING_DEGREE.trailing_zeros() {
        if (power - x).rem_euclid(1 << (k + 3)) != 0 {
            rotation |= 1 << k;
            power = power * step % GALOIS_ORDER;
        }
        step = step * step % GALOIS_ORDER;
    }
    (conjugate, rotation)
}

/// Whether the Galois element `p` is the identity, `p = 1` modulo [`GALOIS_ORDER`].
pub fn galois_is_identity(p: i64) -> bool {
    p.rem_euclid(GALOIS_ORDER) == 1
}

/// The slot rotation of the Galois element `p` in a map over `slots` slots, in `[0, slots)`.
///
/// # Panics
///
/// When `p` conjugates, or `slots` is not a power of two at most [`MAX_RING_DEGREE`].
pub fn galois_rotation(p: i64, slots: usize) -> i64 {
    assert!(
        slots.is_power_of_two() && slots as u64 <= MAX_RING_DEGREE,
        "the slot count must be a power of two at most MAX_RING_DEGREE"
    );
    let (conjugate, rotation) = galois_parts(p);
    assert!(!conjugate, "a slot rotation does not conjugate");
    rotation % slots as i64
}

/// The composition `X -> X^(p q)`: the product modulo [`GALOIS_ORDER`].
pub fn galois_compose(p: i64, q: i64) -> i64 {
    galois_at(p.wrapping_mul(q), GALOIS_ORDER)
}

/// The inverse of `p` modulo [`GALOIS_ORDER`].
pub fn galois_inverse(p: i64) -> i64 {
    // The group has exponent `MAX_RING_DEGREE`: the inverse is the power just below it.
    galois_at(mod_exp_u64(p as u64, MAX_RING_DEGREE as usize - 1) as i64, GALOIS_ORDER)
}

/// The Galois element `p` at the cyclotomic order of an operand, a power of two dividing
/// [`GALOIS_ORDER`]: `p` modulo that order, in the form of
/// [`galois_element`](poulpy_hal::layouts::galois_element).
///
/// # Panics
///
/// When `p` is even.
pub fn galois_at(p: i64, cyclotomic_order: i64) -> i64 {
    assert!(p & 1 == 1, "a Galois element is odd");
    assert!(
        cyclotomic_order > 0 && GALOIS_ORDER % cyclotomic_order == 0,
        "the cyclotomic order must be a power of two dividing GALOIS_ORDER"
    );
    // The conjugating elements are those equal to 3 modulo 4, as 5 is 1 there.
    if p & 3 == 3 {
        -(-p).rem_euclid(cyclotomic_order)
    } else {
        p.rem_euclid(cyclotomic_order)
    }
}

#[cfg(test)]
mod tests {
    use poulpy_hal::{GALOIS_GENERATOR_CONJUGATION, layouts::galois_element};

    use super::*;

    #[test]
    fn galois_parts_invert_the_construction() {
        for conjugate in [false, true] {
            for rotation in [0, 1, 2, 3, 100, 4095, MAX_RING_DEGREE as i64 - 1] {
                let p = galois_from_parts(conjugate, rotation);
                assert_eq!(galois_parts(p), (conjugate, rotation));
                assert_eq!(galois_parts(p + 4 * GALOIS_ORDER), (conjugate, rotation));
            }
        }
        assert_eq!(galois_parts(GALOIS_GENERATOR_CONJUGATION), (true, 0));
    }

    #[test]
    fn galois_at_matches_galois_element() {
        for order in [2_i64, 4, 16, 1024, 1 << 18, GALOIS_ORDER] {
            for rotation in [0_i64, 1, 5, 77, (order / 4).max(1) - 1] {
                assert_eq!(
                    galois_at(galois_of_rotation(rotation), order),
                    galois_element(rotation, order)
                );
                let conjugated: i64 = if rotation == 0 { -1 } else { galois_element(-rotation, order) };
                assert_eq!(galois_at(galois_from_parts(true, rotation), order), conjugated);
            }
        }
    }

    #[test]
    fn galois_arithmetic_follows_the_rotations() {
        let slots = 64;
        let (p, q) = (galois_of_rotation(40), galois_of_rotation(30));
        assert_eq!(galois_compose(p, q), galois_of_rotation(70));
        assert_eq!(galois_rotation(galois_compose(p, q), slots), 6);
        assert_eq!(galois_compose(p, galois_inverse(p)), 1);
        assert_eq!(galois_rotation(galois_inverse(p), slots), 24);
        assert_eq!(galois_rotation(galois_of_rotation(-3), slots), 61);
        let conjugated = galois_from_parts(true, 40);
        assert_eq!(galois_compose(GALOIS_GENERATOR_CONJUGATION, p), conjugated);
        assert_eq!(galois_inverse(conjugated), galois_from_parts(true, -40));
        assert_eq!(galois_compose(conjugated, galois_inverse(conjugated)), 1);
    }

    #[test]
    fn galois_arithmetic_matches_the_parts() {
        for (cp, rp) in [(false, 0_i64), (false, 3), (true, 77), (true, MAX_RING_DEGREE as i64 - 1)] {
            for (cq, rq) in [(false, 1_i64), (true, 0), (false, 4095), (true, 12_345)] {
                let (p, q) = (galois_from_parts(cp, rp), galois_from_parts(cq, rq));
                assert_eq!(galois_compose(p, q), galois_from_parts(cp != cq, rp + rq));
                assert_eq!(galois_inverse(p), galois_from_parts(cp, -rp));
            }
        }
    }
}
