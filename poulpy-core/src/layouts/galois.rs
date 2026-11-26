//! Galois elements of the linear-transformation layer.
//!
//! A Galois element `p`, odd, names the automorphism `X -> X^p`. The group of these modulo a
//! power of two `M >= 8` is `{1, -1} x <5>`, so `p = (-1)^c 5^j` for a conjugation bit `c` and a
//! rotation `j` modulo `M / 4`. Linear transformations hold their elements at
//! [`GALOIS_ORDER`], which every operand's order (`2n` on the standard ring, `4n` on the
//! conjugate-invariant ring) divides, in the form `(-1)^c (5^j mod GALOIS_ORDER)` that
//! [`galois_element`](poulpy_hal::layouts::galois_element) returns, and an element applies to an
//! operand of order `M` as [`galois_at`] reduces it.

use std::cmp::Ordering;

use poulpy_hal::layouts::MAX_RING_DEGREE;

/// The cyclotomic order Galois elements are held at: `4 * MAX_RING_DEGREE`, a multiple of the
/// order of every operand a module serves, on both rings.
pub const GALOIS_ORDER: i64 = 4 * MAX_RING_DEGREE as i64;

/// The order of 5 modulo [`GALOIS_ORDER`]: rotations are taken modulo it.
pub const GALOIS_ROTATIONS: i64 = GALOIS_ORDER / 4;

/// The Galois element of `X -> X^-1`.
pub const GALOIS_CONJUGATION: i64 = -1;

/// The Galois element `(-1)^conjugate 5^rotation`, `rotation` of any sign.
pub fn galois_from_parts(conjugate: bool, rotation: i64) -> i64 {
    let power: i64 = pow5(rotation.rem_euclid(GALOIS_ROTATIONS));
    if conjugate { -power } else { power }
}

/// The Galois element of the slot rotation by `rotation`, of any sign: `X -> X^(5^rotation)`.
pub fn galois_of_rotation(rotation: i64) -> i64 {
    galois_from_parts(false, rotation)
}

/// The conjugation bit and the rotation, in `[0, GALOIS_ROTATIONS)`, of the Galois element `p`.
///
/// # Panics
///
/// When `p` is even.
pub fn galois_parts(p: i64) -> (bool, i64) {
    assert!(p & 1 == 1, "a Galois element is odd");
    let q: i64 = p.rem_euclid(GALOIS_ORDER);
    let (conjugate, x) = if q & 3 == 1 { (false, q) } else { (true, GALOIS_ORDER - q) };
    // Bit `k` of the rotation fixes the residue modulo `2^(k + 3)`, as `5^(2^k) = 1 + 2^(k + 2)` there.
    let (mut rotation, mut power): (i64, i64) = (0, 1);
    for k in 0..GALOIS_ROTATIONS.trailing_zeros() {
        let modulus: i64 = 1 << (k + 3);
        if (power - x).rem_euclid(modulus) != 0 {
            rotation |= 1 << k;
            power = power * pow5(1 << k) % GALOIS_ORDER;
        }
    }
    (conjugate, rotation)
}

/// Whether the Galois element `p` is the identity, `p = 1` modulo [`GALOIS_ORDER`].
pub fn galois_is_identity(p: i64) -> bool {
    p.rem_euclid(GALOIS_ORDER) == 1
}

/// The slot rotation of the Galois element `p`, in `[0, slots)`.
///
/// # Panics
///
/// When `p` conjugates, or `slots` is not a power of two at most [`GALOIS_ROTATIONS`].
pub fn galois_rotation(p: i64, slots: usize) -> i64 {
    let (conjugate, rotation) = galois_parts(p);
    assert!(!conjugate, "a slot rotation does not conjugate");
    rotation % rotation_modulus(slots)
}

/// The Galois element `p` normalized for a map over `slots` slots: its rotation taken modulo
/// `slots`, its conjugation kept.
pub fn galois_mod_slots(p: i64, slots: usize) -> i64 {
    let (conjugate, rotation) = galois_parts(p);
    galois_from_parts(conjugate, rotation % rotation_modulus(slots))
}

/// The composition `X -> X^(p q)` over `slots` slots, normalized as [`galois_mod_slots`].
pub fn galois_compose(p: i64, q: i64, slots: usize) -> i64 {
    let ((cp, rp), (cq, rq)) = (galois_parts(p), galois_parts(q));
    galois_from_parts(cp != cq, (rp + rq) % rotation_modulus(slots))
}

/// The inverse of `p` over `slots` slots, normalized as [`galois_mod_slots`].
pub fn galois_inverse(p: i64, slots: usize) -> i64 {
    let (conjugate, rotation) = galois_parts(p);
    let slots: i64 = rotation_modulus(slots);
    galois_from_parts(conjugate, (slots - rotation % slots) % slots)
}

/// The Galois element `p` at the cyclotomic order of an operand, a power of two dividing
/// [`GALOIS_ORDER`]: the automorphism it applies there, in the form of
/// [`galois_element`](poulpy_hal::layouts::galois_element).
pub fn galois_at(p: i64, cyclotomic_order: i64) -> i64 {
    assert!(
        cyclotomic_order > 0 && GALOIS_ORDER % cyclotomic_order == 0,
        "the cyclotomic order must be a power of two dividing GALOIS_ORDER"
    );
    let (conjugate, rotation) = galois_parts(p);
    let power: i64 = pow5(rotation) % cyclotomic_order;
    if conjugate { -power } else { power }
}

/// The order of Galois elements by conjugation bit, then rotation: the order of the slot rotations
/// they replace.
pub fn galois_cmp(p: &i64, q: &i64) -> Ordering {
    galois_parts(*p).cmp(&galois_parts(*q))
}

/// `slots` as the modulus of rotations.
fn rotation_modulus(slots: usize) -> i64 {
    assert!(
        slots.is_power_of_two() && slots as i64 <= GALOIS_ROTATIONS,
        "the slot count must be a power of two at most GALOIS_ROTATIONS"
    );
    slots as i64
}

/// `5^e mod GALOIS_ORDER`.
fn pow5(e: i64) -> i64 {
    let (mut result, mut base, mut e): (i64, i64, i64) = (1, 5, e);
    while e > 0 {
        if e & 1 == 1 {
            result = result * base % GALOIS_ORDER;
        }
        base = base * base % GALOIS_ORDER;
        e >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use poulpy_hal::layouts::galois_element;

    use super::*;

    #[test]
    fn galois_parts_invert_the_construction() {
        for conjugate in [false, true] {
            for rotation in [0, 1, 2, 3, 100, 4095, GALOIS_ROTATIONS - 1] {
                let p = galois_from_parts(conjugate, rotation);
                assert_eq!(galois_parts(p), (conjugate, rotation));
                assert_eq!(galois_parts(p + 4 * GALOIS_ORDER), (conjugate, rotation));
            }
        }
        assert_eq!(galois_parts(GALOIS_CONJUGATION), (true, 0));
    }

    #[test]
    fn galois_at_matches_galois_element() {
        for order in [16_i64, 1024, 1 << 18, GALOIS_ORDER] {
            for rotation in [0_i64, 1, 5, 77, (order / 4) - 1] {
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
        assert_eq!(galois_compose(p, q, slots), galois_of_rotation(6));
        assert_eq!(galois_inverse(p, slots), galois_of_rotation(24));
        assert_eq!(galois_rotation(galois_of_rotation(-3), slots), 61);
        assert_eq!(galois_mod_slots(galois_of_rotation(70), slots), galois_of_rotation(6));
        assert_eq!(galois_compose(GALOIS_CONJUGATION, p, slots), galois_from_parts(true, 40));
    }
}
