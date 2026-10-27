//! Slot and Galois helpers shared by the derived drivers, the layouts and the reference bodies: the
//! fold rotation schedule, the tiled bit-reversal and the conjugate-rotate Galois element.
//!
//! Conjugation-with-rotation (seqPaCo line 8, Galois element `−5^k`) needs no
//! new operation at all: the automorphism appliers use whatever key they are
//! handed, so an automorphism key generated for the signed Galois element
//! `−5^k` applied through
//! [`CKKSConjugateOps`](crate::api::CKKSConjugateOps) performs
//! `conj(rotate(·, k))` in a single keyswitch.

use poulpy_hal::layouts::galois_element;

/// The rotation amounts of a `Tr_{a→b}` / `Pr_{a→b}` fold
/// ([`ckks_slot_trace_assign`](crate::oep::CKKSPaCoImpl::ckks_slot_trace_assign),
/// [`ckks_slot_product_assign`](crate::oep::CKKSPaCoImpl::ckks_slot_product_assign)), in application
/// order: `a/2, a/4, …, b` (`log(a/b)` entries; empty when `a == b`). Use with
/// [`galois_elements_from_rotations`](poulpy_hal::layouts::galois_elements_from_rotations)
/// to enumerate the automorphism keys a fold requires.
pub(crate) fn fold_rotations(a: usize, b: usize) -> Vec<i64> {
    (0..(a / b).trailing_zeros()).map(|l| (a >> (l + 1)) as i64).collect()
}

/// Reverses the low `log_p` bits of `x`, keeping the higher bits, the tiled
/// permutation `P` of the `BitRevLow` slot-order convention (the paper's
/// extended bit-reversal `Π^{(C/2)}` for `log_p = log(C/2)`). Self-inverse;
/// the identity for `log_p ∈ {0, 1}`. Because it only touches bits below
/// `log_p`, it commutes with every stride that is a multiple of `2^{log_p}`:
/// the fold rotations, the ψ-pairing offset `C`, and the μ/η mask strata.
pub(crate) fn ext_bitrev_low(x: usize, log_p: usize) -> usize {
    if log_p == 0 {
        return x;
    }
    let mask = (1usize << log_p) - 1;
    let rev = (x & mask).reverse_bits() >> (usize::BITS as usize - log_p);
    (x & !mask) | rev
}

/// Galois element of the fused conjugate-and-rotate-by-`k` automorphism
/// (`X ↦ X^{−5^k}`), in the crate's signed encoding (negative = composed with
/// conjugation). `k = 0` degenerates to plain conjugation (`−1`). The
/// automorphism appliers use whatever key they are handed, so a key generated
/// for this element applied through
/// [`CKKSConjugateOps`](crate::api::CKKSConjugateOps) performs
/// `conj(rotate(·, k))` in a single keyswitch, the ψ-pairing's fast tail.
pub(crate) fn conj_rotate_galois_element(k: i64, cyclotomic_order: i64) -> i64 {
    -galois_element(k, cyclotomic_order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_rotation_schedules() {
        assert_eq!(fold_rotations(64, 8), vec![32, 16, 8]);
        assert_eq!(fold_rotations(128, 64), vec![64]);
        assert_eq!(fold_rotations(16, 16), Vec::<i64>::new());
    }
}
