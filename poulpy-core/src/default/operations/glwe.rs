use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxAddAssignBackend, VecZnxAddIntoBackend,
        VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyTmpA, VecZnxLshAddIntoBackend, VecZnxLshAssignBackend, VecZnxLshBackend,
        VecZnxLshSubBackend, VecZnxLshTmpBytes, VecZnxMulXpMinusOneAssignBackend, VecZnxMulXpMinusOneBackend,
        VecZnxNegateAssignBackend, VecZnxNegateBackend, VecZnxNormalize, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes,
        VecZnxRotateAssignBackend, VecZnxRotateAssignTmpBytes, VecZnxRotateBackend, VecZnxRshAssignBackend, VecZnxRshTmpBytes,
        VecZnxSubAssignBackend, VecZnxSubBackend, VecZnxSubNegateAssignBackend, VecZnxZeroBackend,
    },
    layouts::{
        Backend, CnvPVecLToBackendRef, CnvPVecRToBackendRef, Module, ScratchArena, VecZnx, VecZnxBigToBackendMut,
        VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendMut, VecZnxToBackendRef,
    },
};

use crate::{
    ScratchArenaTakeCore,
    default::keyswitching::GGLWEProductDefault,
    layouts::{
        Base2K, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};

#[doc(hidden)]
pub trait GLWEMulConstDefault<BE: Backend> {
    fn glwe_mul_const_tmp_bytes_default<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_mul_const_default<'s, R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_mul_const_assign_default<'s, R, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEMulConstDefault<BE> for Module<BE>
where
    Self: Convolution<BE> + VecZnxBigBytesOf + VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes,
    Self: VecZnxCopyBackend<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_mul_const_tmp_bytes_default<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        assert_eq!(self.n() as u32, res.n());
        assert_eq!(self.n() as u32, a.n());

        let a_base2k: usize = a.base2k().as_usize();
        let res_base2k: usize = res.base2k().as_usize();
        let b_size = b.size();
        let cnv_offset = a.size().max(b_size);
        let res_size: usize = (res.size() * res_base2k).div_ceil(a_base2k);
        let res_dft_size: usize = a.size() + b_size - cnv_offset.saturating_sub(1);
        let lvl_0: usize = self.bytes_of_vec_znx_big(1, res_dft_size) + VecZnx::bytes_of(self.n(), 1, res.size());
        let lvl_1_cnv: usize = self.cnv_by_const_apply_tmp_bytes(res_size, cnv_offset, a.size(), b_size);
        let lvl_1_norm: usize = self.vec_znx_big_normalize_tmp_bytes();
        let lvl_1: usize = lvl_1_cnv.max(lvl_1_norm);

        lvl_0 + lvl_1
    }

    fn glwe_mul_const_default<'s, R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
    {
        let scratch = scratch.borrow();
        assert_eq!(res.rank(), a.rank());
        let b_size = b.size();
        assert!(
            scratch.available() >= self.glwe_mul_const_tmp_bytes_default(res, a, b),
            "scratch.available(): {} < GLWEMulConst::glwe_mul_const_tmp_bytes: {}",
            scratch.available(),
            self.glwe_mul_const_tmp_bytes_default(res, a, b)
        );

        let cols: usize = res.rank().as_usize() + 1;
        let a_base2k: usize = a.base2k().as_usize();
        let res_base2k: usize = res.base2k().as_usize();
        let a_backend = a.to_backend_ref();

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < a_base2k {
            (0, -((a_base2k - (cnv_offset % a_base2k)) as i64))
        } else {
            ((cnv_offset / a_base2k).saturating_sub(1), (cnv_offset % a_base2k) as i64)
        };

        let res_dft_size = a.size() + b_size - cnv_offset_hi;

        let (mut res_big, scratch) = scratch.take_vec_znx_big_scratch(self, 1, res_dft_size);
        let (mut res_tmp, mut scratch) = scratch.take_vec_znx_scratch(self.n(), 1, res.size());
        let b_backend = b.to_backend_ref();
        for i in 0..cols {
            {
                let mut scratch_iter = scratch.borrow();
                let mut res_big_backend = res_big.to_backend_mut();
                self.cnv_by_const_apply(
                    cnv_offset_hi,
                    &mut res_big_backend,
                    0,
                    &a_backend.data,
                    i,
                    &poulpy_hal::layouts::vec_znx_backend_ref_from_ref::<BE>(&b_backend.data),
                    0,
                    b_coeff,
                    &mut scratch_iter,
                );
            }
            let res_big_ref = res_big.to_backend_ref();
            {
                let mut scratch_iter = scratch.borrow();
                self.vec_znx_big_normalize(
                    &mut res_tmp,
                    res_base2k,
                    cnv_offset_lo,
                    0,
                    &res_big_ref,
                    a_base2k,
                    0,
                    &mut scratch_iter,
                );
            }
            let mut res_backend = res.to_backend_mut();
            let res_tmp_ref = res_tmp.to_backend_ref();
            self.vec_znx_copy_backend(&mut res_backend.data, i, &res_tmp_ref, 0);
        }
    }

    fn glwe_mul_const_assign_default<'s, R, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
    {
        let scratch = scratch.borrow();
        assert!(
            scratch.available() >= self.glwe_mul_const_tmp_bytes_default(res, res, b),
            "scratch.available(): {} < GLWEMulConst::glwe_mul_const_tmp_bytes: {}",
            scratch.available(),
            self.glwe_mul_const_tmp_bytes_default(res, res, b)
        );

        let cols: usize = res.rank().as_usize() + 1;
        let res_base2k: usize = res.base2k().as_usize();

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < res_base2k {
            (0, -((res_base2k - (cnv_offset % res_base2k)) as i64))
        } else {
            ((cnv_offset / res_base2k).saturating_sub(1), (cnv_offset % res_base2k) as i64)
        };

        let (mut res_big, scratch) = scratch.take_vec_znx_big_scratch(self, 1, res.size());
        let (mut res_tmp, mut scratch) = scratch.take_vec_znx_scratch(self.n(), 1, res.size());
        let b_backend = b.to_backend_ref();
        for i in 0..cols {
            {
                let res_backend = res.to_backend_ref();
                let mut scratch_iter = scratch.borrow();
                let mut res_big_backend = res_big.to_backend_mut();
                self.cnv_by_const_apply(
                    cnv_offset_hi,
                    &mut res_big_backend,
                    0,
                    &res_backend.data,
                    i,
                    &poulpy_hal::layouts::vec_znx_backend_ref_from_ref::<BE>(&b_backend.data),
                    0,
                    b_coeff,
                    &mut scratch_iter,
                );
            }
            let res_big_ref = res_big.to_backend_ref();
            {
                let mut scratch_iter = scratch.borrow();
                self.vec_znx_big_normalize(
                    &mut res_tmp,
                    res_base2k,
                    cnv_offset_lo,
                    0,
                    &res_big_ref,
                    res_base2k,
                    0,
                    &mut scratch_iter,
                );
            }
            let mut res_backend = res.to_backend_mut();
            let res_tmp_ref = res_tmp.to_backend_ref();
            self.vec_znx_copy_backend(&mut res_backend.data, i, &res_tmp_ref, 0);
        }
    }
}

impl<BE: Backend> GLWEMulPlainDefault<BE> for Module<BE>
where
    Self: Sized
        + ModuleN
        + CnvPVecBytesOf
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + Convolution<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopyBackend<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_mul_plain_tmp_bytes_default<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        assert_eq!(self.n() as u32, res.n());
        assert_eq!(self.n() as u32, a.n());
        assert_eq!(self.n() as u32, b.n());

        let ab_base2k: Base2K = a.base2k();
        assert_eq!(b.base2k(), ab_base2k);

        let cols: usize = res.rank().as_usize() + 1;

        let a_size: usize = a.size();
        let b_size: usize = b.size();
        let cnv_offset: usize = a_size.min(b_size);

        let lvl_0: usize = self.bytes_of_cnv_pvec_left(cols, a_size) + self.bytes_of_cnv_pvec_right(1, b_size);
        let lvl_1: usize = self
            .cnv_prepare_left_tmp_bytes(a_size, a_size)
            .max(self.cnv_prepare_right_tmp_bytes(b_size, b_size));

        let res_dft_size =
            normalize_input_limb_bound_worst_case(a_size + b_size, res.size(), res.base2k().as_usize(), ab_base2k.as_usize());
        let lvl_2_cnv_apply: usize = self.cnv_apply_dft_tmp_bytes(cnv_offset, res_dft_size, a_size, b_size);

        let lvl_2_res_dft: usize = self.bytes_of_vec_znx_dft(1, res_dft_size);
        let lvl_2_res_tmp: usize = self.bytes_of_vec_znx_big(1, res_dft_size) + VecZnx::bytes_of(self.n(), 1, res.size());
        let lvl_2_norm: usize = self.vec_znx_big_normalize_tmp_bytes();
        let lvl_2: usize = lvl_2_res_tmp + lvl_2_res_dft + lvl_2_cnv_apply.max(lvl_2_norm);

        lvl_0 + lvl_1.max(lvl_2)
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain_default<'s, R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        b: &B,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
    {
        let scratch = scratch.borrow();
        assert_eq!(res.rank(), a.rank());
        assert!(
            scratch.available() >= self.glwe_mul_plain_tmp_bytes_default(res, a, b),
            "scratch.available(): {} < GLWEMulPlain::glwe_mul_plain_tmp_bytes: {}",
            scratch.available(),
            self.glwe_mul_plain_tmp_bytes_default(res, a, b)
        );

        let ab_base2k: usize = a.base2k().as_usize();
        assert_eq!(b.base2k().as_usize(), ab_base2k);
        assert_eq!(a_effective_k.div_ceil(ab_base2k), a.size());
        assert_eq!(b_effective_k.div_ceil(ab_base2k), b.size());
        let res_base2k: usize = res.base2k().as_usize();

        let cols: usize = res.rank().as_usize() + 1;

        let (mut a_prep, scratch) = scratch.take_cnv_pvec_left_scratch(self, cols, a.size());
        let (mut b_prep, mut scratch) = scratch.take_cnv_pvec_right_scratch(self, 1, b.size());

        let a_mask = msb_mask_bottom_limb(ab_base2k, a_effective_k);
        let b_mask = msb_mask_bottom_limb(ab_base2k, b_effective_k);
        let a_backend = a.to_backend_ref();
        let b_backend = b.to_backend_ref();

        scratch = scratch.apply_mut(|scratch| self.cnv_prepare_left(&mut a_prep, &a_backend.data, a_mask, scratch));
        scratch = scratch.apply_mut(|scratch| self.cnv_prepare_right(&mut b_prep, &b_backend.data, b_mask, scratch));

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < ab_base2k {
            (0, -((ab_base2k - (cnv_offset % ab_base2k)) as i64))
        } else {
            ((cnv_offset / ab_base2k).saturating_sub(1), (cnv_offset % ab_base2k) as i64)
        };

        let res_dft_size = a.size() + b.size() - cnv_offset_hi;
        let (mut res_tmp, mut scratch) = scratch.take_vec_znx_scratch(self.n(), 1, res.size());

        for i in 0..cols {
            let (mut res_dft, mut scratch_3) = scratch.borrow().take_vec_znx_dft_scratch(self, 1, res_dft_size);
            {
                let mut res_dft_backend = res_dft.to_backend_mut();
                self.cnv_apply_dft(
                    cnv_offset_hi,
                    &mut res_dft_backend,
                    0,
                    &a_prep.to_backend_ref(),
                    i,
                    &b_prep.to_backend_ref(),
                    0,
                    &mut scratch_3,
                );
            }
            let (mut res_big, mut scratch_4) = scratch_3.take_vec_znx_big_scratch(self, 1, res_dft_size);
            {
                let mut res_big_backend = res_big.to_backend_mut();
                let mut res_dft_backend = res_dft.to_backend_mut();
                self.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
            }
            let res_big_ref = res_big.to_backend_ref();
            {
                let mut scratch_iter = scratch_4.borrow();
                self.vec_znx_big_normalize(
                    &mut res_tmp,
                    res_base2k,
                    cnv_offset_lo,
                    0,
                    &res_big_ref,
                    ab_base2k,
                    0,
                    &mut scratch_iter,
                );
            }
            let mut res_backend = res.to_backend_mut();
            let res_tmp_ref = res_tmp.to_backend_ref();
            self.vec_znx_copy_backend(&mut res_backend.data, i, &res_tmp_ref, 0);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain_assign_default<'s, R, A>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_effective_k: usize,
        a: &A,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        let scratch = scratch.borrow();
        assert!(
            scratch.available() >= self.glwe_mul_plain_tmp_bytes_default(res, res, a),
            "scratch.available(): {} < GLWEMulPlain::glwe_mul_plain_tmp_bytes: {}",
            scratch.available(),
            self.glwe_mul_plain_tmp_bytes_default(res, res, a)
        );

        let ab_base2k: usize = a.base2k().as_usize();
        assert_eq!(res.base2k().as_usize(), ab_base2k);
        assert_eq!(res_effective_k.div_ceil(ab_base2k), res.size());
        assert_eq!(a_effective_k.div_ceil(ab_base2k), a.size());

        let cols: usize = res.rank().as_usize() + 1;

        let (mut res_prep, scratch) = scratch.take_cnv_pvec_left_scratch(self, cols, res.size());
        let (mut a_prep, mut scratch) = scratch.take_cnv_pvec_right_scratch(self, 1, a.size());

        let mask_res = msb_mask_bottom_limb(ab_base2k, res_effective_k);
        let mask_a = msb_mask_bottom_limb(ab_base2k, a_effective_k);
        let a_backend = a.to_backend_ref();

        scratch = scratch.apply_mut(|scratch| {
            let res_backend = res.to_backend_ref();
            self.cnv_prepare_left(&mut res_prep, &res_backend.data, mask_res, scratch)
        });
        scratch = scratch.apply_mut(|scratch| self.cnv_prepare_right(&mut a_prep, &a_backend.data, mask_a, scratch));

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < ab_base2k {
            (0, -((ab_base2k - (cnv_offset % ab_base2k)) as i64))
        } else {
            ((cnv_offset / ab_base2k).saturating_sub(1), (cnv_offset % ab_base2k) as i64)
        };

        let res_dft_size = a.size() + res.size() - cnv_offset_hi;
        let (mut res_tmp, mut scratch) = scratch.take_vec_znx_scratch(self.n(), 1, res.size());

        for i in 0..cols {
            let (mut res_dft, mut scratch_3) = scratch.borrow().take_vec_znx_dft_scratch(self, 1, res_dft_size);
            {
                let mut res_dft_backend = res_dft.to_backend_mut();
                self.cnv_apply_dft(
                    cnv_offset_hi,
                    &mut res_dft_backend,
                    0,
                    &res_prep.to_backend_ref(),
                    i,
                    &a_prep.to_backend_ref(),
                    0,
                    &mut scratch_3,
                );
            }
            let (mut res_big, mut scratch_4) = scratch_3.take_vec_znx_big_scratch(self, 1, res_dft_size);
            {
                let mut res_big_backend = res_big.to_backend_mut();
                let mut res_dft_backend = res_dft.to_backend_mut();
                self.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
            }
            let res_big_ref = res_big.to_backend_ref();
            {
                let mut scratch_iter = scratch_4.borrow();
                self.vec_znx_big_normalize(
                    &mut res_tmp,
                    ab_base2k,
                    cnv_offset_lo,
                    0,
                    &res_big_ref,
                    ab_base2k,
                    0,
                    &mut scratch_iter,
                );
            }
            let mut res_backend = res.to_backend_mut();
            let res_tmp_ref = res_tmp.to_backend_ref();
            self.vec_znx_copy_backend(&mut res_backend.data, i, &res_tmp_ref, 0);
        }
    }
}

#[doc(hidden)]
pub trait GLWEMulPlainDefault<BE: Backend> {
    fn glwe_mul_plain_tmp_bytes_default<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain_default<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        b: &B,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_mul_plain_assign_default<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_effective_k: usize,
        a: &A,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;
}

#[doc(hidden)]
pub trait GLWETensoringDefault<BE: Backend> {
    fn glwe_tensor_square_apply_tmp_bytes_default<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    fn glwe_tensor_apply_tmp_bytes_default<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_tensor_relinearize_tmp_bytes_default<R, A, B>(&self, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    fn glwe_tensor_relinearize_default<R, A, B>(
        &self,
        res: &mut R,
        a: &A,
        tsk: &B,
        tsk_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_square_apply_default<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply_default<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        b: &B,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWETensoringDefault<BE> for Module<BE>
where
    Self: Sized
        + ModuleN
        + CnvPVecBytesOf
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + Convolution<BE>
        + VecZnxSubAssignBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxNegateBackend<BE>
        + GGLWEProductDefault<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxNormalizeTmpBytes,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_tensor_square_apply_tmp_bytes_default<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        assert_eq!(self.n() as u32, res.n());
        assert_eq!(self.n() as u32, a.n());

        let cols: usize = res.rank().as_usize() + 1;
        let a_size: usize = a.size();
        let res_size: usize = res.size();
        let cnv_offset = a_size;

        let lvl_0: usize = self.bytes_of_cnv_pvec_left(cols, a_size) + self.bytes_of_cnv_pvec_right(cols, a_size);
        let lvl_diag_cache: usize = VecZnx::bytes_of(self.n(), cols, res_size);
        let lvl_1: usize = self.cnv_prepare_self_tmp_bytes(a_size, a_size);
        let diag_dft_size =
            normalize_input_limb_bound_worst_case(2 * a_size, res_size, res.base2k().as_usize(), a.base2k().as_usize());
        let lvl_2_apply: usize = self.cnv_apply_dft_tmp_bytes(cnv_offset, diag_dft_size, a_size, a_size);
        let pairwise_dft_size =
            normalize_input_limb_bound_worst_case(2 * a_size, res_size, res.base2k().as_usize(), a.base2k().as_usize());
        let lvl_2_pairwise: usize = self.cnv_pairwise_apply_dft_tmp_bytes(cnv_offset, pairwise_dft_size, a_size, a_size);

        let lvl_2a: usize = self.bytes_of_vec_znx_dft(1, diag_dft_size)
            + self.bytes_of_vec_znx_big(1, diag_dft_size)
            + VecZnx::bytes_of(self.n(), 1, res_size)
            + lvl_2_apply.max(self.vec_znx_big_normalize_tmp_bytes());
        let lvl_2b: usize = self.bytes_of_vec_znx_dft(1, pairwise_dft_size)
            + self.bytes_of_vec_znx_big(1, pairwise_dft_size)
            + VecZnx::bytes_of(self.n(), 1, res_size)
            + lvl_2_pairwise.max(self.vec_znx_big_normalize_tmp_bytes());
        let lvl_2: usize = lvl_2a.max(lvl_2b);

        lvl_0 + lvl_diag_cache + lvl_1.max(lvl_2)
    }

    fn glwe_tensor_apply_tmp_bytes_default<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        assert_eq!(self.n() as u32, res.n());
        assert_eq!(self.n() as u32, a.n());
        assert_eq!(self.n() as u32, b.n());

        let ab_base2k: Base2K = a.base2k();
        assert_eq!(b.base2k(), ab_base2k);

        let cols: usize = res.rank().as_usize() + 1;

        let a_size: usize = a.size();
        let b_size: usize = b.size();
        let res_size: usize = res.size();
        let cnv_offset = a_size.min(b_size);

        let lvl_0: usize = self.bytes_of_cnv_pvec_left(cols, a_size) + self.bytes_of_cnv_pvec_right(cols, b_size);
        let lvl_1: usize = self
            .cnv_prepare_left_tmp_bytes(a_size, a_size)
            .max(self.cnv_prepare_right_tmp_bytes(b_size, b_size));
        let diag_dft_size =
            normalize_input_limb_bound_worst_case(a_size + b_size, res_size, res.base2k().as_usize(), ab_base2k.as_usize());
        let lvl_2_apply: usize = self.cnv_apply_dft_tmp_bytes(cnv_offset, diag_dft_size, a_size, b_size);
        let pairwise_dft_size =
            normalize_input_limb_bound_worst_case(a_size + b_size, res_size, res.base2k().as_usize(), ab_base2k.as_usize());
        let lvl_2_pairwise: usize = self.cnv_pairwise_apply_dft_tmp_bytes(cnv_offset, pairwise_dft_size, a_size, b_size);

        let lvl_2a: usize = self.bytes_of_vec_znx_dft(1, diag_dft_size)
            + self.bytes_of_vec_znx_big(1, diag_dft_size)
            + VecZnx::bytes_of(self.n(), 1, res_size)
            + lvl_2_apply.max(self.vec_znx_big_normalize_tmp_bytes());
        let lvl_2b: usize = self.bytes_of_vec_znx_dft(1, pairwise_dft_size)
            + self.bytes_of_vec_znx_big(1, pairwise_dft_size)
            + VecZnx::bytes_of(self.n(), 1, res_size)
            + lvl_2_pairwise.max(self.vec_znx_big_normalize_tmp_bytes());
        let lvl_2: usize = lvl_2a.max(lvl_2b);

        lvl_0 + lvl_1.max(lvl_2)
    }

    fn glwe_tensor_relinearize_tmp_bytes_default<R, A, B>(&self, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, res.n());
        assert_eq!(self.n() as u32, a.n());
        assert_eq!(self.n() as u32, tsk.n());

        let a_base2k: usize = a.base2k().into();
        let key_base2k: usize = tsk.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let cols: usize = tsk.rank_out().as_usize() + 1;
        let pairs: usize = tsk.rank_in().as_usize();

        let a_dft_size: usize = (a.size() * a_base2k).div_ceil(key_base2k);

        let lvl_0: usize = self.bytes_of_vec_znx_dft(pairs, a_dft_size);

        let lvl_1_pre_conv: usize = if a_base2k != key_base2k {
            VecZnx::bytes_of(self.n(), 1, a_dft_size) + self.vec_znx_normalize_tmp_bytes()
        } else {
            0
        };
        let lvl_1_res_dft: usize = self.bytes_of_vec_znx_dft(cols, tsk.size());
        let lvl_1_gglwe_product: usize = self.gglwe_product_dft_tmp_bytes_default(res.size(), a_dft_size, tsk);
        let lvl_1_post_conv: usize = if res_base2k != key_base2k {
            VecZnx::bytes_of(self.n(), 1, a_dft_size) + self.vec_znx_normalize_tmp_bytes()
        } else {
            0
        };
        let lvl_1_big_norm: usize = self.bytes_of_vec_znx_big(cols, tsk.size())
            + VecZnx::bytes_of(self.n(), 1, res.size())
            + self.vec_znx_big_normalize_tmp_bytes();
        let lvl_1_main: usize = lvl_1_res_dft + lvl_1_gglwe_product.max(lvl_1_post_conv).max(lvl_1_big_norm);
        let lvl_1: usize = lvl_1_pre_conv.max(lvl_1_main);

        lvl_0 + lvl_1
    }

    fn glwe_tensor_relinearize_default<R, A, B>(
        &self,
        res: &mut R,
        a: &A,
        tsk: &B,
        tsk_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        let scratch = scratch.borrow();
        assert!(
            scratch.available() >= self.glwe_tensor_relinearize_tmp_bytes_default(res, a, tsk),
            "scratch.available(): {} < GLWETensoring::glwe_tensor_relinearize_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_relinearize_tmp_bytes_default(res, a, tsk)
        );

        let a_base2k: usize = a.base2k().into();
        let key_base2k: usize = tsk.base2k().into();
        let res_base2k: usize = res.base2k().into();
        let a_backend = a.to_backend_ref();

        assert_eq!(res.rank(), tsk.rank_out());
        assert_eq!(a.rank(), tsk.rank_out());

        let cols: usize = tsk.rank_out().as_usize() + 1;
        let pairs: usize = tsk.rank_in().as_usize();

        let a_dft_size: usize = (a.size() * a_base2k).div_ceil(key_base2k);

        let (mut a_dft, mut scratch) = scratch.take_vec_znx_dft_scratch(self, pairs, a_dft_size);

        {
            let (mut a_conv, mut scratch_norm) = scratch.borrow().take_vec_znx_scratch(self.n(), 1, a_dft_size);
            for i in 0..pairs {
                let mut scratch_iter = scratch_norm.borrow();
                self.vec_znx_normalize(
                    &mut a_conv,
                    key_base2k,
                    0,
                    0,
                    &a_backend.data,
                    a_base2k,
                    cols + i,
                    &mut scratch_iter,
                );
                let a_conv_ref = a_conv.to_backend_ref();
                self.vec_znx_dft_apply(1, 0, &mut a_dft, i, &a_conv_ref, 0);
            }
        }

        let (mut res_dft, mut scratch_2) = scratch.borrow().take_vec_znx_dft_scratch(self, cols, tsk_size); // Todo optimise
        let tsk = tsk.to_backend_ref();

        let a_dft_ref = a_dft.to_backend_ref();
        self.gglwe_product_dft_default(&mut res_dft, &a_dft_ref, &tsk.0, &mut scratch_2);
        let (mut res_big, mut scratch_3) = scratch_2.take_vec_znx_big_scratch(self, cols, tsk_size);
        {
            let mut res_big_backend = res_big.to_backend_mut();
            let mut res_dft_backend = res_dft.to_backend_mut();
            for i in 0..cols {
                self.vec_znx_idft_apply_tmpa(&mut res_big_backend, i, &mut res_dft_backend, i);
            }
        }

        {
            let (mut a_conv, mut scratch_norm) = scratch_3.borrow().take_vec_znx_scratch(self.n(), 1, a_dft_size);
            for i in 0..cols {
                let mut scratch_iter = scratch_norm.borrow();
                self.vec_znx_normalize(&mut a_conv, key_base2k, 0, 0, &a_backend.data, a_base2k, i, &mut scratch_iter);
                let a_conv_ref = a_conv.to_backend_ref();
                self.vec_znx_big_add_small_assign(&mut res_big, i, &a_conv_ref, 0);
            }
        }

        {
            let (mut res_tmp, mut scratch_norm) = scratch_3.borrow().take_vec_znx_scratch(self.n(), 1, res.size());
            for i in 0..(res.rank() + 1).into() {
                let res_big_ref = res_big.to_backend_ref();
                let mut scratch_iter = scratch_norm.borrow();
                self.vec_znx_big_normalize(&mut res_tmp, res_base2k, 0, 0, &res_big_ref, key_base2k, i, &mut scratch_iter);
                let mut res_backend = res.to_backend_mut();
                let res_tmp_ref = res_tmp.to_backend_ref();
                self.vec_znx_copy_backend(&mut res_backend.data, i, &res_tmp_ref, 0);
            }
        }
    }

    fn glwe_tensor_square_apply_default<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        let scratch = scratch.borrow();
        assert!(
            scratch.available() >= self.glwe_tensor_square_apply_tmp_bytes_default(res, a),
            "scratch.available(): {} < GLWETensoring::glwe_tensor_square_apply_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_square_apply_tmp_bytes_default(res, a)
        );

        let a_base2k: usize = a.base2k().as_usize();

        assert_eq!(a_effective_k.div_ceil(a_base2k), a.size());

        let res_base2k: usize = res.base2k().as_usize();
        let cols: usize = res.rank().as_usize() + 1;

        let (mut a_prep, scratch) = scratch.take_cnv_pvec_left_scratch(self, cols, a.size());
        let (mut b_prep, mut scratch) = scratch.take_cnv_pvec_right_scratch(self, cols, a.size());

        let a_mask = msb_mask_bottom_limb(a_base2k, a_effective_k);
        let a_backend = a.to_backend_ref();

        let mut prep_scratch = scratch.borrow();
        self.cnv_prepare_self(&mut a_prep, &mut b_prep, &a_backend.data, a_mask, &mut prep_scratch);
        let (mut diag_terms, mut scratch) = scratch.take_vec_znx_scratch(self.n(), cols, res.size());

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < a_base2k {
            (0, -((a_base2k - (cnv_offset % a_base2k)) as i64))
        } else {
            ((cnv_offset / a_base2k).saturating_sub(1), (cnv_offset % a_base2k) as i64)
        };

        let diag_dft_size =
            normalize_input_limb_bound_with_offset(2 * a.size() - cnv_offset_hi, res.size(), res_base2k, a_base2k, cnv_offset_lo);
        let pairwise_dft_size =
            normalize_input_limb_bound_with_offset(2 * a.size() - cnv_offset_hi, res.size(), res_base2k, a_base2k, cnv_offset_lo);

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            let (mut res_dft, mut scratch_4) = scratch.borrow().take_vec_znx_dft_scratch(self, 1, diag_dft_size);
            {
                let mut res_dft_backend = res_dft.to_backend_mut();
                self.cnv_apply_dft(
                    cnv_offset_hi,
                    &mut res_dft_backend,
                    0,
                    &a_prep.to_backend_ref(),
                    i,
                    &b_prep.to_backend_ref(),
                    i,
                    &mut scratch_4,
                );
            }
            let (mut res_big, scratch_5) = scratch_4.take_vec_znx_big_scratch(self, 1, diag_dft_size);
            {
                let mut res_big_backend = res_big.to_backend_mut();
                let mut res_dft_backend = res_dft.to_backend_mut();
                self.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
            }
            let (mut tmp, mut scratch_6) = scratch_5.take_vec_znx_scratch(self.n(), 1, res.size());
            let res_big_ref = res_big.to_backend_ref();
            let mut scratch_iter = scratch_6.borrow();
            self.vec_znx_big_normalize(
                &mut tmp,
                res_base2k,
                cnv_offset_lo,
                0,
                &res_big_ref,
                a_base2k,
                0,
                &mut scratch_iter,
            );

            // TODO: Do we need 2 copies?
            {
                let mut diag_terms_mut = diag_terms.to_backend_mut();
                let tmp_ref = tmp.to_backend_ref();
                self.vec_znx_copy_backend(&mut diag_terms_mut, i, &tmp_ref, 0);
            }
            {
                let mut res_backend = res.to_backend_mut();
                let diag_terms_ref = diag_terms.to_backend_ref();
                self.vec_znx_copy_backend(&mut res_backend.data, col_i + i, &diag_terms_ref, i);
            }
        }

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            for j in i + 1..cols {
                let (mut res_dft, mut scratch_4) = scratch.borrow().take_vec_znx_dft_scratch(self, 1, pairwise_dft_size);
                {
                    let mut res_dft_backend = res_dft.to_backend_mut();
                    self.cnv_pairwise_apply_dft(
                        cnv_offset_hi,
                        &mut res_dft_backend,
                        0,
                        &a_prep.to_backend_ref(),
                        &b_prep.to_backend_ref(),
                        i,
                        j,
                        &mut scratch_4,
                    );
                }
                let (mut res_big, scratch_6) = scratch_4.take_vec_znx_big_scratch(self, 1, pairwise_dft_size);
                {
                    let mut res_big_backend = res_big.to_backend_mut();
                    let mut res_dft_backend = res_dft.to_backend_mut();
                    self.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
                }
                let (mut tmp, mut scratch_7) = scratch_6.take_vec_znx_scratch(self.n(), 1, res.size());
                let res_big_ref = res_big.to_backend_ref();
                let mut scratch_iter = scratch_7.borrow();
                self.vec_znx_big_normalize(
                    &mut tmp,
                    res_base2k,
                    cnv_offset_lo,
                    0,
                    &res_big_ref,
                    a_base2k,
                    0,
                    &mut scratch_iter,
                );
                {
                    let mut tmp_mut = tmp.to_backend_mut();
                    let diag_terms_ref = diag_terms.to_backend_ref();
                    self.vec_znx_sub_assign_backend(&mut tmp_mut, 0, &diag_terms_ref, i);
                    self.vec_znx_sub_assign_backend(&mut tmp_mut, 0, &diag_terms_ref, j);
                }

                // TODO: Do we need copy?
                let mut res_backend = res.to_backend_mut();
                let tmp_ref = tmp.to_backend_ref();
                self.vec_znx_copy_backend(&mut res_backend.data, col_i + j, &tmp_ref, 0);
            }
        }
    }

    fn glwe_tensor_apply_default<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        b: &B,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
    {
        let scratch = scratch.borrow();
        assert!(
            scratch.available() >= self.glwe_tensor_apply_tmp_bytes_default(res, a, b),
            "scratch.available(): {} < GLWETensoring::glwe_tensor_apply_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_apply_tmp_bytes_default(res, a, b)
        );

        let ab_base2k: usize = a.base2k().as_usize();
        assert_eq!(b.base2k().as_usize(), ab_base2k);
        assert_eq!(a_effective_k.div_ceil(ab_base2k), a.size());
        assert_eq!(b_effective_k.div_ceil(ab_base2k), b.size());

        let res_base2k: usize = res.base2k().as_usize();

        let cols: usize = res.rank().as_usize() + 1;

        let (mut a_prep, scratch) = scratch.take_cnv_pvec_left_scratch(self, cols, a.size());
        let (mut b_prep, mut scratch) = scratch.take_cnv_pvec_right_scratch(self, cols, b.size());

        let a_mask = msb_mask_bottom_limb(ab_base2k, a_effective_k);
        let b_mask = msb_mask_bottom_limb(ab_base2k, b_effective_k);
        let a_backend = a.to_backend_ref();
        let b_backend = b.to_backend_ref();

        let mut prep_scratch = scratch.borrow();
        self.cnv_prepare_left(&mut a_prep, &a_backend.data, a_mask, &mut prep_scratch);
        self.cnv_prepare_right(&mut b_prep, &b_backend.data, b_mask, &mut prep_scratch);
        // Example for rank=3
        //
        // (a0, a1, a2, a3) x (b0, b1, b2, a3)
        //   L   L  L   L       R   R   R   R
        //
        // c(1)    = a0 * b0 				<- (L(a0) * R(b0))
        // c(s1)   = a0 * b1 + a1 * b0 		<- (L(a0) + L(a1)) * (R(b0) + R(b1)) + NEG(L(a0) * R(b0)) + SUB(L(a1) * R(b1))
        // c(s2)   = a0 * b2 + a2 * b0		<- (L(a0) + L(a2)) * (R(b0) + R(b2)) + NEG(L(a0) * R(b0)) + SUB(L(a2) * R(b2))
        // c(s3)   = a0 * b3 + a3 * b0		<- (L(a0) + L(a3)) * (R(b0) + R(b3)) + NEG(L(a0) * R(b0)) + SUB(L(a3) * R(b3))
        // c(s1^2) = a1 * b1 				<- (L(a1) * R(b1))
        // c(s1s2) = a1 * b2 + b2 * a1		<- (L(a1) + L(a2)) * (R(b1) + R(b2)) + NEG(L(a1) * R(b1)) + SUB(L(a2) * R(b2))
        // c(s1s3) = a1 * b3 + b3 * a1		<- (L(a1) + L(a3)) * (R(b1) + R(b3)) + NEG(L(a1) * R(b1)) + SUB(L(a3) * R(b3))
        // c(s2^2) = a2 * b2 				<- (L(a2) * R(b2))
        // c(s2s3) = a2 * b3 + a3 * b2 	    <- (L(a2) + L(a3)) * (R(b2) + R(b3)) + NEG(L(a2) * R(b2)) + SUB(L(a3) * R(b3))
        // c(s3^2) = a3 * b3				<- (L(a3) * R(b3))

        // Derive the offset. If cnv_offset < a_base2k, then we shift to a negative offset
        // since the convolution doesn't support negative offset (yet).
        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < ab_base2k {
            (0, -((ab_base2k - (cnv_offset % ab_base2k)) as i64))
        } else {
            ((cnv_offset / ab_base2k).saturating_sub(1), (cnv_offset % ab_base2k) as i64)
        };

        let diag_dft_size = normalize_input_limb_bound_with_offset(
            a.size() + b.size() - cnv_offset_hi,
            res.size(),
            res_base2k,
            ab_base2k,
            cnv_offset_lo,
        );
        let pairwise_dft_size = normalize_input_limb_bound_with_offset(
            a.size() + b.size() - cnv_offset_hi,
            res.size(),
            res_base2k,
            ab_base2k,
            cnv_offset_lo,
        );

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            let (mut res_dft, mut scratch_3) = scratch.borrow().take_vec_znx_dft_scratch(self, 1, diag_dft_size);
            {
                let mut res_dft_backend = res_dft.to_backend_mut();
                self.cnv_apply_dft(
                    cnv_offset_hi,
                    &mut res_dft_backend,
                    0,
                    &a_prep.to_backend_ref(),
                    i,
                    &b_prep.to_backend_ref(),
                    i,
                    &mut scratch_3,
                );
            }
            let (mut res_big, scratch_4) = scratch_3.take_vec_znx_big_scratch(self, 1, diag_dft_size);
            {
                let mut res_big_backend = res_big.to_backend_mut();
                let mut res_dft_backend = res_dft.to_backend_mut();
                self.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
            }
            let (mut tmp, mut scratch_5) = scratch_4.take_vec_znx_scratch(self.n(), 1, res.size());
            let res_big_ref = res_big.to_backend_ref();
            let mut scratch_iter = scratch_5.borrow();
            self.vec_znx_big_normalize(
                &mut tmp,
                res_base2k,
                cnv_offset_lo,
                0,
                &res_big_ref,
                ab_base2k,
                0,
                &mut scratch_iter,
            );

            {
                let mut res_backend = res.to_backend_mut();
                let tmp_ref = tmp.to_backend_ref();
                self.vec_znx_copy_backend(&mut res_backend.data, col_i + i, &tmp_ref, 0);
            }

            // Pre-subtracts
            // res[i!=j] = NEG(a[i] * b[i]) + SUB(a[j] * b[j])
            for j in 0..cols {
                if j != i {
                    if j < i {
                        let col_j = j * cols - (j * (j + 1) / 2);
                        let mut res_backend = res.to_backend_mut();
                        let tmp_ref = tmp.to_backend_ref();
                        self.vec_znx_sub_assign_backend(&mut res_backend.data, col_j + i, &tmp_ref, 0);
                    } else {
                        let mut res_backend = res.to_backend_mut();
                        let tmp_ref = tmp.to_backend_ref();
                        self.vec_znx_negate_backend(&mut res_backend.data, col_i + j, &tmp_ref, 0);
                    }
                }
            }
        }

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            for j in i..cols {
                if j != i {
                    // res_dft = (a[i] + a[j]) * (b[i] + b[j])
                    let (mut res_dft, mut scratch_3) = scratch.borrow().take_vec_znx_dft_scratch(self, 1, pairwise_dft_size);
                    {
                        let mut res_dft_backend = res_dft.to_backend_mut();
                        self.cnv_pairwise_apply_dft(
                            cnv_offset_hi,
                            &mut res_dft_backend,
                            0,
                            &a_prep.to_backend_ref(),
                            &b_prep.to_backend_ref(),
                            i,
                            j,
                            &mut scratch_3,
                        );
                    }
                    let (mut res_big, scratch_4) = scratch_3.take_vec_znx_big_scratch(self, 1, pairwise_dft_size);
                    {
                        let mut res_big_backend = res_big.to_backend_mut();
                        let mut res_dft_backend = res_dft.to_backend_mut();
                        self.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
                    }
                    let (mut tmp, mut scratch_5) = scratch_4.take_vec_znx_scratch(self.n(), 1, res.size());
                    let res_big_ref = res_big.to_backend_ref();
                    let mut scratch_iter = scratch_5.borrow();
                    self.vec_znx_big_normalize(
                        &mut tmp,
                        res_base2k,
                        cnv_offset_lo,
                        0,
                        &res_big_ref,
                        ab_base2k,
                        0,
                        &mut scratch_iter,
                    );

                    let mut res_backend = res.to_backend_mut();
                    let tmp_ref = tmp.to_backend_ref();
                    self.vec_znx_add_assign_backend(&mut res_backend.data, col_i + j, &tmp_ref, 0);
                }
            }
        }
    }
}

#[inline]
pub fn msb_mask_bottom_limb(base2k: usize, k: usize) -> i64 {
    match k % base2k {
        0 => !0i64,
        r => (!0i64) << (base2k - r),
    }
}

#[inline]
fn normalize_input_limb_bound(
    full_size: usize,
    res_size: usize,
    res_base2k: usize,
    in_base2k: usize,
    offset_bits: usize,
) -> usize {
    full_size.min((res_size * res_base2k + offset_bits).div_ceil(in_base2k))
}

#[inline]
fn normalize_input_limb_bound_worst_case(full_size: usize, res_size: usize, res_base2k: usize, in_base2k: usize) -> usize {
    normalize_input_limb_bound(full_size, res_size, res_base2k, in_base2k, in_base2k - 1)
}

#[inline]
fn normalize_input_limb_bound_with_offset(
    full_size: usize,
    res_size: usize,
    res_base2k: usize,
    in_base2k: usize,
    res_offset: i64,
) -> usize {
    let mut offset_bits = res_offset % in_base2k as i64;
    if res_offset < 0 && offset_bits != 0 {
        offset_bits += in_base2k as i64;
    }
    normalize_input_limb_bound(full_size, res_size, res_base2k, in_base2k, offset_bits as usize)
}

#[doc(hidden)]
pub trait GLWEAddDefault<BE: Backend> {
    fn glwe_add_into_default<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>;

    fn glwe_add_assign_default<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;
}

impl<BE: Backend> GLWEAddDefault<BE> for Module<BE>
where
    Self: ModuleN + VecZnxAddIntoBackend<BE> + VecZnxCopyBackend<BE> + VecZnxAddAssignBackend<BE> + VecZnxZeroBackend<BE>,
{
    fn glwe_add_into_default<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>,
    {
        let res = &mut res.to_backend_mut();
        let a = &a.to_backend_ref();
        let b = &b.to_backend_ref();

        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(b.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.base2k(), b.base2k());
        assert_eq!(res.base2k(), b.base2k());

        if a.rank() == 0 {
            assert_eq!(res.rank(), b.rank());
        } else if b.rank() == 0 {
            assert_eq!(res.rank(), a.rank());
        } else {
            assert_eq!(res.rank(), a.rank());
            assert_eq!(res.rank(), b.rank());
        }

        let min_col: usize = (a.rank().min(b.rank()) + 1).into();
        let max_col: usize = (a.rank().max(b.rank()) + 1).into();
        let self_col: usize = (res.rank() + 1).into();

        for i in 0..min_col {
            self.vec_znx_add_into_backend(&mut res.data, i, &a.data, i, &b.data, i);
        }

        if a.rank() > b.rank() {
            for i in min_col..max_col {
                self.vec_znx_copy_backend(&mut res.data, i, &a.data, i);
            }
        } else {
            for i in min_col..max_col {
                self.vec_znx_copy_backend(&mut res.data, i, &b.data, i);
            }
        }

        for i in max_col..self_col {
            self.vec_znx_zero_backend(&mut res.data, i);
        }
    }

    fn glwe_add_assign_default<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        for i in 0..(a.rank() + 1).into() {
            self.vec_znx_add_assign_backend(&mut res.data, i, &a.data, i);
        }
    }
}

#[doc(hidden)]
pub trait GLWESubDefault<BE: Backend> {
    fn glwe_sub_default<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>;

    fn glwe_sub_assign_default<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_sub_negate_assign_default<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;
}

impl<BE: Backend> GLWESubDefault<BE> for Module<BE>
where
    Self: ModuleN
        + VecZnxSubBackend<BE>
        + VecZnxSubAssignBackend<BE>
        + VecZnxSubNegateAssignBackend<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxNegateBackend<BE>
        + VecZnxZeroBackend<BE>,
{
    fn glwe_sub_default<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>,
    {
        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();
        let b = b.to_backend_ref();
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(b.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.base2k(), res.base2k());
        assert_eq!(b.base2k(), res.base2k());

        if a.rank() == 0 {
            assert_eq!(res.rank(), b.rank());
        } else if b.rank() == 0 {
            assert_eq!(res.rank(), a.rank());
        } else {
            assert_eq!(res.rank(), a.rank());
            assert_eq!(res.rank(), b.rank());
        }

        let min_col: usize = (a.rank().min(b.rank()) + 1).into();
        let max_col: usize = (a.rank().max(b.rank()) + 1).into();
        let self_col: usize = (res.rank() + 1).into();

        for i in 0..min_col {
            self.vec_znx_sub_backend(&mut res.data, i, &a.data, i, &b.data, i);
        }

        if a.rank() > b.rank() {
            for i in min_col..max_col {
                self.vec_znx_copy_backend(&mut res.data, i, &a.data, i);
            }
        } else {
            for i in min_col..max_col {
                self.vec_znx_negate_backend(&mut res.data, i, &b.data, i);
            }
        }

        for i in max_col..self_col {
            self.vec_znx_zero_backend(&mut res.data, i);
        }
    }

    fn glwe_sub_assign_default<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() == a.rank() || a.rank() == 0);

        for i in 0..(a.rank() + 1).into() {
            self.vec_znx_sub_assign_backend(&mut res.data, i, &a.data, i);
        }
    }

    fn glwe_sub_negate_assign_default<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() == a.rank() || a.rank() == 0);

        for i in 0..(a.rank() + 1).into() {
            self.vec_znx_sub_negate_assign_backend(&mut res.data, i, &a.data, i);
        }
    }
}

#[doc(hidden)]
pub trait GLWENegateDefault<BE: Backend> {
    fn glwe_negate_default<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_negate_assign_default<R>(&self, res: &mut R)
    where
        R: GLWEToBackendMut<BE>;
}

impl<BE: Backend> GLWENegateDefault<BE> for Module<BE>
where
    Self: VecZnxNegateBackend<BE> + VecZnxNegateAssignBackend<BE> + ModuleN,
{
    fn glwe_negate_default<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();

        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.rank(), res.rank());
        let cols = res.rank().as_usize() + 1;
        for i in 0..cols {
            self.vec_znx_negate_backend(&mut res.data, i, &a.data, i);
        }
        res.base2k = a.base2k;
    }

    fn glwe_negate_assign_default<R>(&self, res: &mut R)
    where
        R: GLWEToBackendMut<BE>,
    {
        let mut res = res.to_backend_mut();

        assert_eq!(res.n(), self.n() as u32);
        let cols = res.rank().as_usize() + 1;
        for i in 0..cols {
            self.vec_znx_negate_assign_backend(&mut res.data, i);
        }
    }
}

#[doc(hidden)]
pub trait GLWEZeroDefault<BE: Backend> {
    fn glwe_zero_default<R>(&self, res: &mut R)
    where
        R: GLWEToBackendMut<BE>;
}

impl<BE: Backend> GLWEZeroDefault<BE> for Module<BE>
where
    Self: ModuleN + VecZnxZeroBackend<BE>,
{
    fn glwe_zero_default<R>(&self, res: &mut R)
    where
        R: GLWEToBackendMut<BE>,
    {
        let mut res = res.to_backend_mut();

        assert_eq!(res.n(), self.n() as u32);
        let cols = res.rank().as_usize() + 1;
        for i in 0..cols {
            self.vec_znx_zero_backend(&mut res.data, i);
        }
    }
}

#[doc(hidden)]
pub trait GLWERotateDefault<BE: Backend> {
    fn glwe_rotate_tmp_bytes_default(&self) -> usize;

    fn glwe_rotate_default<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_rotate_assign_default<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;
}

impl<BE: Backend> GLWERotateDefault<BE> for Module<BE>
where
    Self: ModuleN + VecZnxRotateBackend<BE> + VecZnxRotateAssignBackend<BE> + VecZnxRotateAssignTmpBytes + VecZnxZeroBackend<BE>,
{
    fn glwe_rotate_tmp_bytes_default(&self) -> usize {
        self.vec_znx_rotate_assign_tmp_bytes()
    }

    fn glwe_rotate_default<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();

        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert!(res.rank() == a.rank() || a.rank() == 0);

        let res_cols = (res.rank() + 1).into();
        let a_cols = (a.rank() + 1).into();

        for i in 0..a_cols {
            self.vec_znx_rotate_backend(k, &mut res.data, i, &a.data, i);
        }
        for i in a_cols..res_cols {
            self.vec_znx_zero_backend(&mut res.data, i);
        }
    }

    fn glwe_rotate_assign_default<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let mut res = res.to_backend_mut();

        assert!(
            scratch.available() >= <Self as GLWERotateDefault<BE>>::glwe_rotate_tmp_bytes_default(self),
            "scratch.available(): {} < GLWERotate::glwe_rotate_tmp_bytes: {}",
            scratch.available(),
            <Self as GLWERotateDefault<BE>>::glwe_rotate_tmp_bytes_default(self)
        );

        for i in 0..(res.rank() + 1).into() {
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_rotate_assign_backend(k, &mut res.data, i, &mut scratch_iter);
        }
    }
}

#[doc(hidden)]
pub trait GLWEMulXpMinusOneDefault<BE: Backend> {
    fn glwe_mul_xp_minus_one_default<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_mul_xp_minus_one_assign_default<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>;
}

impl<BE: Backend> GLWEMulXpMinusOneDefault<BE> for Module<BE>
where
    Self: ModuleN + VecZnxMulXpMinusOneBackend<BE> + VecZnxMulXpMinusOneAssignBackend<BE>,
{
    fn glwe_mul_xp_minus_one_default<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        let res = &mut res.to_backend_mut();
        let a = &a.to_backend_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank());

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_mul_xp_minus_one_backend(k, &mut res.data, i, &a.data, i);
        }
    }

    fn glwe_mul_xp_minus_one_assign_default<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        let res = &mut res.to_backend_mut();

        assert_eq!(res.n(), self.n() as u32);

        for i in 0..res.rank().as_usize() + 1 {
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_mul_xp_minus_one_assign_backend(k, &mut res.data, i, &mut scratch_iter);
        }
    }
}

#[doc(hidden)]
pub trait GLWECopyDefault<BE: Backend> {
    fn glwe_copy_default<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;
}

impl<BE: Backend> GLWECopyDefault<BE> for Module<BE>
where
    Self: ModuleN + VecZnxCopyBackend<BE> + VecZnxZeroBackend<BE>,
{
    fn glwe_copy_default<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert!(res.rank() == a.rank() || a.rank() == 0);

        let min_rank: usize = res.rank().min(a.rank()).as_usize() + 1;

        for i in 0..min_rank {
            self.vec_znx_copy_backend(&mut res.data, i, &a.data, i);
        }

        for i in min_rank..(res.rank() + 1).into() {
            self.vec_znx_zero_backend(&mut res.data, i);
        }
    }
}

#[doc(hidden)]
pub trait GLWEShiftDefault<BE: Backend> {
    fn glwe_shift_tmp_bytes_default(&self) -> usize;

    fn glwe_rsh_default<'s, R>(&self, k: usize, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_lsh_assign_default<'s, R>(&self, res: &mut R, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_lsh_default<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_lsh_add_default<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_lsh_sub_default<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;
}

impl<BE: Backend> GLWEShiftDefault<BE> for Module<BE>
where
    Self: ModuleN
        + VecZnxRshAssignBackend<BE>
        + VecZnxLshAddIntoBackend<BE>
        + VecZnxLshSubBackend<BE>
        + VecZnxRshTmpBytes
        + VecZnxLshTmpBytes
        + VecZnxLshAssignBackend<BE>
        + VecZnxLshBackend<BE>,
{
    fn glwe_shift_tmp_bytes_default(&self) -> usize {
        let lvl_0: usize = self.vec_znx_rsh_tmp_bytes().max(self.vec_znx_lsh_tmp_bytes());
        lvl_0
    }

    fn glwe_rsh_default<'s, R>(&self, k: usize, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let res = &mut res.to_backend_mut();
        assert!(
            scratch.available() >= <Self as GLWEShiftDefault<BE>>::glwe_shift_tmp_bytes_default(self),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            <Self as GLWEShiftDefault<BE>>::glwe_shift_tmp_bytes_default(self)
        );
        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_rsh_assign_backend(base2k, k, &mut res.data, i, &mut scratch_iter);
        }
    }

    fn glwe_lsh_assign_default<'s, R>(&self, res: &mut R, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let res = &mut res.to_backend_mut();

        assert!(
            scratch.available() >= <Self as GLWEShiftDefault<BE>>::glwe_shift_tmp_bytes_default(self),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            <Self as GLWEShiftDefault<BE>>::glwe_shift_tmp_bytes_default(self)
        );

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_lsh_assign_backend(base2k, k, &mut res.data, i, &mut scratch_iter);
        }
    }

    fn glwe_lsh_default<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let res = &mut res.to_backend_mut();
        let a = &a.to_backend_ref();
        assert!(
            scratch.available() >= <Self as GLWEShiftDefault<BE>>::glwe_shift_tmp_bytes_default(self),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            <Self as GLWEShiftDefault<BE>>::glwe_shift_tmp_bytes_default(self)
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_lsh_backend(base2k, k, &mut res.data, i, &a.data, i, &mut scratch_iter);
        }
    }

    fn glwe_lsh_add_default<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let res = &mut res.to_backend_mut();
        let a = &a.to_backend_ref();
        assert!(
            scratch.available() >= <Self as GLWEShiftDefault<BE>>::glwe_shift_tmp_bytes_default(self),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            <Self as GLWEShiftDefault<BE>>::glwe_shift_tmp_bytes_default(self)
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_lsh_add_into_backend(base2k, k, &mut res.data, i, &a.data, i, &mut scratch_iter);
        }
    }

    fn glwe_lsh_sub_default<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let res = &mut res.to_backend_mut();
        let a = &a.to_backend_ref();
        assert!(
            scratch.available() >= <Self as GLWEShiftDefault<BE>>::glwe_shift_tmp_bytes_default(self),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            <Self as GLWEShiftDefault<BE>>::glwe_shift_tmp_bytes_default(self)
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_lsh_sub_backend(base2k, k, &mut res.data, i, &a.data, i, &mut scratch_iter);
        }
    }
}

#[doc(hidden)]
pub trait GLWENormalizeDefault<BE: Backend> {
    fn glwe_normalize_tmp_bytes_default(&self) -> usize;

    fn glwe_normalize_default<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_normalize_assign_default<R>(&self, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;
}

impl<BE: Backend> GLWENormalizeDefault<BE> for Module<BE>
where
    Self: ModuleN + VecZnxNormalize<BE> + VecZnxNormalizeAssignBackend<BE> + VecZnxNormalizeTmpBytes,
{
    fn glwe_normalize_tmp_bytes_default(&self) -> usize {
        let lvl_0: usize = self.vec_znx_normalize_tmp_bytes();
        lvl_0
    }

    fn glwe_normalize_default<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank());
        assert!(
            scratch.available() >= <Self as GLWENormalizeDefault<BE>>::glwe_normalize_tmp_bytes_default(self),
            "scratch.available(): {} < GLWENormalize::glwe_normalize_tmp_bytes: {}",
            scratch.available(),
            <Self as GLWENormalizeDefault<BE>>::glwe_normalize_tmp_bytes_default(self)
        );

        let res_base2k = res.base2k().into();

        for i in 0..res.rank().as_usize() + 1 {
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_normalize(
                &mut res.data,
                res_base2k,
                0,
                i,
                &a.data,
                a.base2k().into(),
                i,
                &mut scratch_iter,
            );
        }
    }

    fn glwe_normalize_assign_default<R>(&self, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let mut res = res.to_backend_mut();

        assert!(
            scratch.available() >= <Self as GLWENormalizeDefault<BE>>::glwe_normalize_tmp_bytes_default(self),
            "scratch.available(): {} < GLWENormalize::glwe_normalize_tmp_bytes: {}",
            scratch.available(),
            <Self as GLWENormalizeDefault<BE>>::glwe_normalize_tmp_bytes_default(self)
        );
        for i in 0..res.rank().as_usize() + 1 {
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_normalize_assign_backend(res.base2k().into(), &mut res.data, i, &mut scratch_iter);
        }
    }
}
