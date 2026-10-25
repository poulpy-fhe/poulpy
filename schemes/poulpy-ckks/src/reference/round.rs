use crate::CKKSResult as Result;
use poulpy_core::{
    GLWENormalize,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{ScratchArenaTakeBasic, VecZnxAddAssign, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, ScratchArena, VecZnxToBackendMut, VecZnxToBackendRef},
};

use crate::oep::CKKSShapedRoundingImpl;
use crate::{CKKSInfos, Scale, SetCKKSInfos, SlotLayout, error::CKKSCompositionError, shaping::ShapingProfile};

/// Scratch required by [`ckks_round_into_reference`].
pub fn ckks_round_tmp_bytes_reference<BE: Backend, Dst, Src>(module: &Module<BE>, dst: &Dst, src: &Src) -> usize
where
    Module<BE>: GLWENormalize<BE> + VecZnxNormalizeTmpBytes,
    Dst: GLWEInfos,
    Src: GLWEInfos,
{
    let n = src.n().as_usize();
    let shaped = 2 * BE::scratch_aligned(BE::bytes_of_vec_znx(n, 1, 1))
        + BE::scratch_aligned(BE::bytes_of_vec_znx(n, 1, 2))
        + 2 * BE::scratch_aligned(BE::bytes_of_vec_znx(n, 1, src.size().max(dst.size())))
        + module.vec_znx_normalize_tmp_bytes();
    shaped.max(module.glwe_normalize_tmp_bytes())
}

/// Rounds using backend normalization and the selected shaping correction.
pub fn ckks_round_into_reference<BE: CKKSShapedRoundingImpl, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Module<BE>: GLWENormalize<BE> + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes + VecZnxAddAssign<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
{
    let invalid = |reason: String| CKKSCompositionError::InvalidOperand {
        op: "ckks_round",
        reason,
    };
    if dst.rank() != src.rank() {
        return Err(invalid("source and destination ranks differ".into()).into());
    }
    if dst.n() != src.n() {
        return Err(invalid("source and destination degrees differ".into()).into());
    }
    let (src_k, dst_k) = (src.k().as_usize(), dst.k().as_usize());
    if dst_k > src_k {
        return Err(invalid(format!("dst.k()={dst_k} exceeds src.k()={src_k}")).into());
    }
    let drop = src_k - dst_k;
    if drop > src.log_delta() {
        return Err(invalid(format!("dropping {drop} bits exceeds log_delta={}", src.log_delta())).into());
    }
    let shaping = match src.slot_layout() {
        SlotLayout::Replicated(layout) if drop > 0 => Some(ShapingProfile::for_ring::<BE::Ring>(&layout)?),
        _ => None,
    };
    if let Some(profile) = &shaping
        && profile.period() > src.n().as_usize()
    {
        return Err(invalid(format!("period 2^{} exceeds the ring", profile.log_r())).into());
    }

    crate::ckks_ensure!(
        scratch.available() >= ckks_round_tmp_bytes_reference(module, dst, src),
        "ckks_round: insufficient scratch"
    );
    dst.set_meta(src.meta());
    dst.set_scale(Scale::Log(src.log_delta() - drop));
    dst.set_k(dst_k.into());
    match shaping {
        None => module.glwe_normalize(dst, src, scratch),
        Some(profile) => shaped_round(module, dst, src, &profile, scratch),
    }
    dst.set_noise(None);
    Ok(())
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
        BE::ckks_shaped_round_correction_impl(
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
