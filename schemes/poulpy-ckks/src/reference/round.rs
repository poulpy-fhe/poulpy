use crate::CKKSResult as Result;
use poulpy_core::{
    GLWENormalize,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{ScratchArenaTakeVecZnx, VecZnxAddAssign, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, ScratchArena, VecZnxToBackendMut, VecZnxToBackendRef},
};

use crate::oep::CKKSShapedRoundingImpl;
use crate::{CKKSInfos, Scale, SetCKKSInfos, SlotLayout, error::CKKSCompositionError, shaping::ShapingProfile};

/// Reference of the [`CKKSRoundImpl`](crate::oep::CKKSRoundImpl) family,
/// selected by [`impl_ckks_round_reference_full!`](crate::impl_ckks_round_reference_full).
pub trait CKKSRoundReference<BE: Backend> {
    /// Scratch of [`Self::ckks_round_reference`].
    fn ckks_round_tmp_bytes_reference<R, A>(&self, res: &R, a: &A) -> usize
    where
        Self: GLWENormalize<BE> + VecZnxNormalizeTmpBytes,
        R: GLWEInfos,
        A: GLWEInfos;

    /// [`CKKSRoundOps::ckks_round`](crate::api::CKKSRoundOps::ckks_round): rounds through the backend
    /// normalization and, for a replicated layout, the selected shaping correction
    /// [`CKKSShapedRoundingImpl::ckks_shaped_round_correction`].
    fn ckks_round_reference<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        BE: CKKSShapedRoundingImpl,
        Self: GLWENormalize<BE> + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes + VecZnxAddAssign<BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos;
}

impl<BE: Backend> CKKSRoundReference<BE> for Module<BE> {
    fn ckks_round_tmp_bytes_reference<R, A>(&self, res: &R, a: &A) -> usize
    where
        Self: GLWENormalize<BE> + VecZnxNormalizeTmpBytes,
        R: GLWEInfos,
        A: GLWEInfos,
    {
        let n = a.n().as_usize();
        let shaped = 2 * BE::scratch_aligned(BE::vec_znx_bytes_of(n, 1, 1))
            + BE::scratch_aligned(BE::vec_znx_bytes_of(n, 1, 2))
            + 2 * BE::scratch_aligned(BE::vec_znx_bytes_of(n, 1, a.size().max(res.size())))
            + self.vec_znx_normalize_tmp_bytes();
        shaped.max(self.glwe_normalize_tmp_bytes())
    }

    fn ckks_round_reference<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        BE: CKKSShapedRoundingImpl,
        Self: GLWENormalize<BE> + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes + VecZnxAddAssign<BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    {
        let invalid = |reason: String| CKKSCompositionError::InvalidOperand {
            op: "ckks_round",
            reason,
        };
        if res.rank() != a.rank() {
            return Err(invalid("source and destination ranks differ".into()).into());
        }
        if res.n() != a.n() {
            return Err(invalid("source and destination degrees differ".into()).into());
        }
        let (src_k, dst_k) = (a.k().as_usize(), res.k().as_usize());
        if dst_k > src_k {
            return Err(invalid(format!("res.k()={dst_k} exceeds a.k()={src_k}")).into());
        }
        let drop = src_k - dst_k;
        if drop > a.log_delta() {
            return Err(invalid(format!("dropping {drop} bits exceeds log_delta={}", a.log_delta())).into());
        }
        let shaping = match a.slot_layout() {
            SlotLayout::Replicated(layout) if drop > 0 => Some(ShapingProfile::for_ring::<BE::Ring>(&layout)?),
            _ => None,
        };
        if let Some(profile) = &shaping
            && profile.period() > a.n().as_usize()
        {
            return Err(invalid(format!("period 2^{} exceeds the ring", profile.log_r())).into());
        }

        crate::ckks_ensure!(
            scratch.available() >= self.ckks_round_tmp_bytes_reference(res, a),
            "ckks_round: insufficient scratch"
        );
        res.set_meta(a.meta());
        res.set_scale(Scale::Log(a.log_delta() - drop));
        res.set_k(dst_k.into());
        match shaping {
            None => self.glwe_normalize(res, a, scratch),
            Some(profile) => shaped_round(self, res, a, &profile, scratch),
        }
        res.set_noise(None);
        Ok(())
    }
}

/// Shaped rounding of every column of `src` to `dst.k()`, at depth `λ`:
/// the `λ` bits under `dst.k()`, read modulo `2^λ` after rounding at
/// `dst.k() + λ`, give `h = [U^(-1)·c]_(2^λ)` and the carry
/// `κ = (c − g·h)/2^λ`; adding `κ·2^-k − c·2^-(k+λ)` leaves a value on the
/// `2^-k` grid up to the bits under `k + λ`, which the final normalization drops.
fn shaped_round<BE, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    profile: &ShapingProfile,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: CKKSShapedRoundingImpl,
    Module<BE>: VecZnxNormalize<BE> + VecZnxAddAssign<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos,
    Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
{
    let k = dst.k().as_usize();
    let lambda = (src.k().as_usize() - k).min(profile.max_pass_bits());
    let src_base2k = src.base2k().as_usize();
    let dst_base2k = dst.base2k().as_usize();
    dst.set_canonical(true);
    let mut dst = dst.to_backend_mut();
    let src = src.to_backend_ref();
    let n = src.n().as_usize();
    let size = src.size().max(dst.size());

    let (mut low, scratch) = scratch.borrow().take_vec_znx_scratch(n, 1, 1);
    let (mut h, scratch) = scratch.take_vec_znx_scratch(n, 1, 1);
    let (mut carry, scratch) = scratch.take_vec_znx_scratch(n, 1, 2);
    let (mut correction, scratch) = scratch.take_vec_znx_scratch(n, 1, size);
    let (mut sum, mut scratch) = scratch.take_vec_znx_scratch(n, 1, size);
    for col in 0..src.rank().as_usize() + 1 {
        module.vec_znx_normalize(
            &mut low.to_backend_mut(),
            lambda,
            lambda,
            k as i64,
            0,
            src.data(),
            src_base2k,
            col,
            &mut scratch.borrow(),
        );
        BE::ckks_shaped_round_correction(
            module,
            &mut carry.to_backend_mut(),
            &mut h.to_backend_mut(),
            &low.to_backend_ref(),
            profile,
            lambda,
        );
        // `κ·2^-λ − c·2^-2λ` at radix `λ`, scaled to `κ·2^-k − c·2^-(k+λ)`.
        module.vec_znx_normalize(
            &mut correction.to_backend_mut(),
            src_base2k,
            size * src_base2k,
            lambda as i64 - k as i64,
            0,
            &carry.to_backend_ref(),
            lambda,
            0,
            &mut scratch.borrow(),
        );
        module.vec_znx_normalize(
            &mut sum.to_backend_mut(),
            src_base2k,
            size * src_base2k,
            0,
            0,
            src.data(),
            src_base2k,
            col,
            &mut scratch.borrow(),
        );
        module.vec_znx_add_assign(&mut sum.to_backend_mut(), 0, &correction.to_backend_ref(), 0);
        module.vec_znx_normalize(
            dst.data_mut(),
            dst_base2k,
            k,
            0,
            col,
            &sum.to_backend_ref(),
            src_base2k,
            0,
            &mut scratch.borrow(),
        );
    }
}
