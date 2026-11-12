//! Slot and Galois helpers shared by the derived drivers, the layouts and the reference bodies: the
//! fold rotation schedule, the tiled bit-reversal and the conjugate-rotate Galois element.
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
use crate::layouts::{CKKSPlaintextViewMut, ScratchArenaTakeCKKSPlaintext};
use crate::shaping::MAX_COEFFS;
use crate::{GLWEPlaintextMeta, Scale, SetCKKSInfos, ckks_log_delta, layouts::CKKSPlaintextAlloc};
use crate::{SlotLayout, numerics::CKKSFloat, scalar::Quad, shaping::conjugate_invariant};
use num_traits::{Float, FromPrimitive};
use poulpy_core::layouts::{Base2K, GLWEPlaintextLayout, GLWEToBackendMut, IntPolyInfos, TorusPrecision};
use poulpy_hal::{
    AlignedBuf, alloc_aligned,
    layouts::{Backend, Ring, ScratchArena, VecZnx, encode_i128_by_limb, galois_element},
};

/// Builds the backend-resident constant-`1.0` plaintext used by the
/// add-one/sub-one facades.
///
/// Each call allocates and uploads a degree-1, single-limb monomial. The
/// unit-operation scratch query covers the arithmetic workspace. A backend
/// that keeps the constant builds it here, once per module, ring and radix.
pub fn ckks_plaintext_one<BE, M, Rg: Ring>(module: &M, base2k: Base2K) -> Result<CKKSPlaintextOwned<BE, Rg>>
where
    BE: Backend,
    M: CKKSPlaintextAlloc<BE> + ?Sized,
{
    let meta = GLWEPlaintextMeta {
        scale: Scale::Log(1),
        slots: SlotsKind::Real,
        log_sparsity: 0,
        slot_layout: crate::SlotLayout::Full,
    };

    // Monomial plaintext: total torus width is `log_delta` (budget 0).
    let k_total: usize = ckks_log_delta(&meta);

    // The constant is integer-exact: 1.0 at scale `2^log_delta` is the single
    // coefficient `1 << log_delta`, so its limbs are built on the host with the
    // integer codec and uploaded as raw bytes (no float codec, no transfer op).
    let size = k_total.div_ceil(base2k.as_usize());
    let mut limbs = VecZnx::<AlignedBuf, i64, Rg>::from_data(
        alloc_aligned::<u8>(VecZnx::<AlignedBuf, i64, Rg>::bytes_of(1, 1, size)),
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

    let mut pt = module.ckks_plaintext_alloc(1usize.into(), base2k, k_total.into());
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

/// Scratch bytes of [`take_one_pt`] for a destination of `res_size` limbs.
pub(crate) fn one_pt_tmp_bytes<BE: Backend>(res_size: usize) -> usize {
    BE::scratch_aligned(BE::vec_znx_bytes_of(MAX_COEFFS, 1, res_size + 1))
}

/// The one of `slot_layout`, carved from `scratch`. On a full layout it is the
/// monomial `2` at scale `2`, added as a constant. On a replicated layout it is
/// the masked one `π_S` at scale `2^log_delta`: one on the data slots and zero
/// on the guards, a plaintext of degree `2R` in `Y = X^(N/(2R))` with
/// coefficients `cos(π·j/(2R))/R`, quantized with shaped rounding. It is
/// invariant, and the conjugate-invariant ring holds it as a plaintext of
/// degree `R`, its first `R` coefficients. Refusals carry the caller's `op`.
pub(crate) fn take_one_pt<'a, BE: Backend + 'a, Rg: Ring>(
    op: &'static str,
    base2k: Base2K,
    slot_layout: SlotLayout,
    log_delta: usize,
    scratch: ScratchArena<'a, BE>,
) -> Result<(CKKSPlaintextViewMut<'a, BE, Rg>, ScratchArena<'a, BE>)> {
    let mut data = [0i128; MAX_COEFFS];
    let (n, k, meta) = match slot_layout {
        SlotLayout::Full => {
            data[0] = 2;
            let meta = GLWEPlaintextMeta {
                scale: Scale::Log(1),
                slots: SlotsKind::Real,
                log_sparsity: 0,
                slot_layout,
            };
            (1, 1, meta)
        }
        SlotLayout::Replicated(layout) => {
            let r = 1usize << layout.log_r();
            let m = if conjugate_invariant::<Rg>() { r } else { 2 * r };
            let r_inv = Quad::from_usize(r).expect("the period is a small integer").recip();
            let mut coeffs = [Quad::from_usize(0).expect("zero"); MAX_COEFFS];
            for (j, c) in coeffs[..m].iter_mut().enumerate() {
                *c = Quad::ckks_root_of_unity(j as u64, layout.log_r() as u32 + 2).0 * r_inv;
            }
            let (mut low, mut h, mut gh) = ([0i64; MAX_COEFFS], [0i64; MAX_COEFFS], [0i64; MAX_COEFFS]);
            crate::shaping::shaped_quantize_into::<_, Rg>(
                &coeffs[..m],
                &layout,
                log_delta,
                1,
                &mut data[..m],
                &mut low[..m],
                &mut h[..m],
                &mut gh[..m],
            )?;
            // Embedded at the backend's smallest plaintext degree when `m` is below it.
            let n = m.max(BE::MIN_DEGREE);
            if n > MAX_COEFFS {
                return Err(crate::CKKSCompositionError::InvalidOperand {
                    op,
                    reason: format!("the masked one needs a plaintext of degree {n} > {MAX_COEFFS}"),
                }
                .into());
            }
            let stride = n / m;
            if stride > 1 {
                for j in (1..m).rev() {
                    data[j * stride] = data[j];
                    data[j] = 0;
                }
            }
            let meta = GLWEPlaintextMeta {
                scale: Scale::Log(log_delta),
                slots: SlotsKind::Real,
                log_sparsity: 0,
                slot_layout,
            };
            (n, log_delta + 1, meta)
        }
    };
    let layout = GLWEPlaintextLayout {
        n: n.into(),
        base2k,
        k: TorusPrecision(k as u32),
    };
    let (mut pt, scratch) = scratch.take_ckks_plaintext_scratch::<_, Rg>(&layout, meta);
    let encoded_k = pt.encoded_k().as_usize();
    let (mut carry, mut limb) = ([0i64; MAX_COEFFS], [0i64; MAX_COEFFS]);
    {
        let mut backend = pt.to_backend_mut();
        let shape = backend.data().shape();
        encode_i128_by_limb(
            base2k.as_usize(),
            encoded_k,
            &mut data[..n],
            &mut carry[..n],
            &mut limb[..n],
            |j, digits| {
                let offset = shape.scalar_offset(0, j) * size_of::<i64>();
                let mut region = BE::region_mut_ref(backend.data_mut().data_mut(), offset, size_of_val(digits));
                BE::copy_host_to_view(&mut region, bytemuck::cast_slice(digits));
            },
        );
    }
    Ok((pt, scratch))
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
