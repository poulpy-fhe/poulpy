//! Slot and Galois helpers shared by the derived drivers, the layouts and the reference bodies: the
//! fold rotation schedule, the tiled bit-reversal, the conjugate-rotate Galois element and the
//! constant-one plaintext.
//!
//! Conjugation-with-rotation (seqPaCo line 8, Galois element `−5^k`) needs no
//! new operation at all: the automorphism appliers use whatever key they are
//! handed, so an automorphism key generated for the signed Galois element
//! `−5^k` applied through
//! [`CKKSConjugateOps`](crate::api::CKKSConjugateOps) performs
//! `conj(rotate(·, k))` in a single keyswitch.

use crate::CKKSResult as Result;
use crate::SlotsKind;
use crate::layouts::CKKSPlaintextOwned;
use crate::{GLWEPlaintextMeta, Scale, SetCKKSInfos, ckks_log_delta, layouts::CKKSModuleAlloc};
use poulpy_core::layouts::Base2K;
use poulpy_hal::{
    AlignedBuf, alloc_aligned,
    layouts::{Backend, VecZnx, galois_element},
};

/// Builds the backend-resident constant-`1.0` plaintext used by the
/// add-one/sub-one facades.
///
/// This currently allocates and uploads a degree-1, single-limb monomial on
/// each call. The unit-operation scratch query covers the arithmetic workspace.
pub(crate) fn ckks_one_pt<BE, M>(module: &M, base2k: Base2K) -> Result<CKKSPlaintextOwned<BE>>
where
    BE: Backend,
    M: CKKSModuleAlloc<BE> + ?Sized,
{
    let meta = GLWEPlaintextMeta {
        scale: Scale::Log(1),
        slots: SlotsKind::Real,
        log_sparsity: 0,
    };

    // Monomial plaintext: total torus width is `log_delta` (budget 0).
    let k_total: usize = ckks_log_delta(&meta);

    // The constant is integer-exact: 1.0 at scale `2^log_delta` is the single
    // coefficient `1 << log_delta`, so its limbs are built on the host with the
    // integer codec and uploaded as raw bytes (no float codec, no transfer op).
    let size = k_total.div_ceil(base2k.as_usize());
    let mut limbs = VecZnx::<AlignedBuf, i64>::from_data(
        alloc_aligned::<u8>(VecZnx::<AlignedBuf, i64>::bytes_of(1, 1, size)),
        1,
        1,
        size,
    );
    limbs.encode_vec_i64(
        base2k.as_usize(),
        0,
        size * base2k.as_usize(),
        &[1i64 << ckks_log_delta(&meta)],
    );

    let mut pt = module.ckks_pt_coeffs_alloc(1, base2k, k_total.into());
    pt.set_meta(meta);
    pt.copy_from_host_bytes::<BE>(limbs.data().as_slice());
    Ok(pt)
}

/// The rotation amounts of a `Tr_{a→b}` / `Pr_{a→b}` fold
/// ([`ckks_slot_trace_assign`](crate::oep::CKKSPaCoImpl::ckks_slot_trace_assign),
/// [`ckks_slot_product_assign`](crate::oep::CKKSPaCoImpl::ckks_slot_product_assign)), in application
/// order: `a/2, a/4, …, b` (`log(a/b)` entries; empty when `a == b`). Use with
/// [`galois_elements_from_rotations`](poulpy_hal::layouts::galois_elements_from_rotations)
/// to enumerate the automorphism keys a fold requires.
pub(crate) fn fold_rotations(a: usize, b: usize) -> Vec<i64> {
    (0..(a / b).trailing_zeros()).map(|l| (a >> (l + 1)) as i64).collect()
}

/// Reverses the low `log_p` bits of `x`, keeping the higher bits — the tiled
/// permutation `P` of the `BitRevLow` slot-order convention (the paper's
/// extended bit-reversal `Π^{(C/2)}` for `log_p = log(C/2)`). Self-inverse;
/// the identity for `log_p ∈ {0, 1}`. Because it only touches bits below
/// `log_p`, it commutes with every stride that is a multiple of `2^{log_p}` —
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
/// `conj(rotate(·, k))` in a single keyswitch — the ψ-pairing's fast tail.
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
