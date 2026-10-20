use crate::CKKSResult as Result;
use poulpy_core::{
    GLWENormalize,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{ScratchArenaTakeBasic, VecZnxAddAssign, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, HostDataMut, HostDataRef, ScratchArena, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut},
};

use crate::{CKKSInfos, ReplicatedLayout, Scale, SetCKKSInfos, SlotLayout, error::CKKSCompositionError, shaping::ShapingProfile};

pub trait CKKSRoundReference<BE: Backend> {
    fn ckks_round_tmp_bytes_reference<Dst, Src>(&self, dst: &Dst, src: &Src) -> usize
    where
        Self: GLWENormalize<BE> + VecZnxNormalizeTmpBytes,
        Dst: GLWEInfos,
        Src: GLWEInfos,
    {
        let n = src.n().as_usize();
        let shaped = BE::scratch_aligned(BE::bytes_of_vec_znx(n, 1, 1))
            + BE::scratch_aligned(BE::bytes_of_vec_znx(n, 1, 2))
            + 2 * BE::scratch_aligned(BE::bytes_of_vec_znx(n, 1, src.size().max(dst.size())))
            + self.vec_znx_normalize_tmp_bytes();
        shaped.max(self.glwe_normalize_tmp_bytes())
    }

    fn ckks_round_into_reference<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWENormalize<BE> + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes + VecZnxAddAssign<BE>,
        BE: Backend<ZnxWord = i64>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSInfos,
    {
        let invalid = |reason: String| CKKSCompositionError::InvalidOperand {
            op: "ckks_round",
            reason,
        };
        let (src_k, dst_k) = (src.k().as_usize(), dst.k().as_usize());
        if dst_k > src_k {
            return Err(invalid(format!("dst.k()={dst_k} exceeds src.k()={src_k}")).into());
        }
        let drop = src_k - dst_k;
        if drop > src.log_delta() {
            return Err(invalid(format!("dropping {drop} bits exceeds log_delta={}", src.log_delta())).into());
        }
        let shaping = match src.slot_layout() {
            SlotLayout::Replicated(layout) if drop > 0 => Some(layout),
            _ => None,
        };
        if let Some(layout) = shaping
            && 2 << layout.log_r() > src.n().as_usize()
        {
            return Err(invalid(format!("period 2^{} exceeds the ring", layout.log_r())).into());
        }

        dst.set_meta(src.meta());
        dst.set_scale(Scale::Log(src.log_delta() - drop));
        dst.set_k(dst_k.into());
        match shaping {
            None => self.glwe_normalize(dst, src, scratch),
            Some(layout) => shaped_round(self, dst, src, &layout, scratch),
        }
        dst.set_noise(None);
        Ok(())
    }
}

impl<BE: Backend> CKKSRoundReference<BE> for poulpy_hal::layouts::Module<BE> {}

/// Shaped rounding of every column of `src` to `dst.k()`, at depth `λ`:
/// the `λ` bits under `dst.k()`, read modulo `2^λ` after rounding at
/// `dst.k() + λ`, give `h = [U^(-1)·c]_(2^λ)` and the carry
/// `κ = (c − g·h)/2^λ`; adding `κ·2^-k − c·2^-(k+λ)` leaves a value on the
/// `2^-k` grid up to the bits under `k + λ`, which the final normalization drops.
fn shaped_round<BE, M, Dst, Src>(
    module: &M,
    dst: &mut Dst,
    src: &Src,
    layout: &ReplicatedLayout,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend<ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    M: VecZnxNormalize<BE> + VecZnxAddAssign<BE> + ?Sized,
    Dst: GLWEToBackendMut<BE> + CKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSInfos,
{
    let profile = ShapingProfile::new(layout);
    let k = dst.k().as_usize();
    let lambda = (src.k().as_usize() - k).min(profile.max_pass_bits());
    let mask = u64::MAX >> (64 - lambda);
    let src_base2k = src.base2k().as_usize();
    let dst_base2k = dst.base2k().as_usize();
    dst.set_canonical(true);
    let mut dst = dst.to_backend_mut();
    let src = src.to_backend_ref();
    let n = src.n().as_usize();
    let stride = n >> (layout.log_r() + 1);
    let size = src.size().max(dst.size());

    let (mut low, scratch) = scratch.borrow().take_vec_znx_scratch(n, 1, 1);
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
        let c = low.at(0, 0).to_vec();
        let gh = profile.shaped_multiple(stride, lambda, &c);
        for (j, (&cj, &ghj)) in c.iter().zip(&gh).enumerate() {
            let numerator = cj - ghj;
            debug_assert_eq!(numerator & mask as i64, 0, "shaped numerator must be divisible by 2^λ");
            carry.at_mut(0, 0)[j] = numerator >> lambda;
            carry.at_mut(0, 1)[j] = -cj;
        }
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
