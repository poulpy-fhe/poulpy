use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, ScratchAvailable, ScratchTakeBasic, VecZnxAddAssign, VecZnxAddInto,
        VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftApply,
        VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxLsh, VecZnxLshAddInto, VecZnxLshAssign, VecZnxLshSub, VecZnxLshTmpBytes,
        VecZnxMulXpMinusOne, VecZnxMulXpMinusOneAssign, VecZnxNegate, VecZnxNegateAssign, VecZnxNormalize, VecZnxNormalizeAssign,
        VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateAssign, VecZnxRotateAssignTmpBytes, VecZnxRshAssign,
        VecZnxRshTmpBytes, VecZnxSub, VecZnxSubAssign, VecZnxSubNegateAssign, VecZnxZero,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, VecZnx, VecZnxBig},
};

pub use crate::api::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWEMulXpMinusOne, GLWENegate, GLWENormalize, GLWERotate, GLWEShift, GLWESub,
    GLWETensoring,
};
use crate::{
    GGLWEProduct, ScratchTakeCore,
    layouts::{
        Base2K, GGLWEInfos, GLWE, GLWEInfos, GLWEPlaintext, GLWETensor, GLWETensorKeyPrepared, GLWEToMut, GLWEToRef, LWEInfos,
    },
};

#[doc(hidden)]
pub trait GLWEMulConstDefault<BE: Backend> {
    fn glwe_mul_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    fn glwe_mul_const<R, A>(&self, cnv_offset: usize, res: &mut GLWE<R>, a: &GLWE<A>, b: &[i64], scratch: &mut Scratch<BE>)
    where
        R: DataMut,
        A: DataRef;

    fn glwe_mul_const_assign<R>(&self, cnv_offset: usize, res: &mut GLWE<R>, b: &[i64], scratch: &mut Scratch<BE>)
    where
        R: DataMut;
}

impl<BE: Backend> GLWEMulConstDefault<BE> for Module<BE>
where
    Self: Convolution<BE> + VecZnxBigBytesOf + VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_mul_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        assert_eq!(self.n() as u32, res.n());
        assert_eq!(self.n() as u32, a.n());

        let a_base2k: usize = a.base2k().as_usize();
        let res_base2k: usize = res.base2k().as_usize();
        let cnv_offset = a.size().max(b_size);
        let res_size: usize = (res.size() * res_base2k).div_ceil(a_base2k);
        let lvl_0: usize = self.bytes_of_vec_znx_big(1, res_size);
        let lvl_1_cnv: usize = self.cnv_by_const_apply_tmp_bytes(res_size, cnv_offset, a.size(), b_size);
        let lvl_1_norm: usize = self.vec_znx_big_normalize_tmp_bytes();
        let lvl_1: usize = lvl_1_cnv.max(lvl_1_norm);

        lvl_0 + lvl_1
    }

    fn glwe_mul_const<R, A>(&self, cnv_offset: usize, res: &mut GLWE<R>, a: &GLWE<A>, b: &[i64], scratch: &mut Scratch<BE>)
    where
        R: DataMut,
        A: DataRef,
    {
        assert_eq!(res.rank(), a.rank());
        assert!(
            scratch.available() >= self.glwe_mul_const_tmp_bytes(res, a, b.len()),
            "scratch.available(): {} < GLWEMulConst::glwe_mul_const_tmp_bytes: {}",
            scratch.available(),
            self.glwe_mul_const_tmp_bytes(res, a, b.len())
        );

        let cols: usize = res.rank().as_usize() + 1;
        let a_base2k: usize = a.base2k().as_usize();
        let res_base2k: usize = res.base2k().as_usize();

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < a_base2k {
            (0, -((a_base2k - (cnv_offset % a_base2k)) as i64))
        } else {
            ((cnv_offset / a_base2k).saturating_sub(1), (cnv_offset % a_base2k) as i64)
        };

        let res_dft_size = a.size() + b.len() - cnv_offset_hi;

        let (mut res_big, scratch_1) = scratch.take_vec_znx_big(self, 1, res_dft_size);
        for i in 0..cols {
            self.cnv_by_const_apply(cnv_offset_hi, &mut res_big, 0, a.data(), i, b, scratch_1);
            self.vec_znx_big_normalize(res.data_mut(), res_base2k, cnv_offset_lo, i, &res_big, a_base2k, 0, scratch_1);
        }
    }

    fn glwe_mul_const_assign<R>(&self, cnv_offset: usize, res: &mut GLWE<R>, b: &[i64], scratch: &mut Scratch<BE>)
    where
        R: DataMut,
    {
        let res_ref: &GLWE<&[u8]> = &res.to_ref();
        assert!(
            scratch.available() >= self.glwe_mul_const_tmp_bytes(res_ref, res_ref, b.len()),
            "scratch.available(): {} < GLWEMulConst::glwe_mul_const_tmp_bytes: {}",
            scratch.available(),
            self.glwe_mul_const_tmp_bytes(res_ref, res_ref, b.len())
        );

        let cols: usize = res.rank().as_usize() + 1;
        let res_base2k: usize = res.base2k().as_usize();

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < res_base2k {
            (0, -((res_base2k - (cnv_offset % res_base2k)) as i64))
        } else {
            ((cnv_offset / res_base2k).saturating_sub(1), (cnv_offset % res_base2k) as i64)
        };

        let (mut res_big, scratch_1) = scratch.take_vec_znx_big(self, 1, res.size());
        for i in 0..cols {
            self.cnv_by_const_apply(cnv_offset_hi, &mut res_big, 0, res.data(), i, b, scratch_1);
            self.vec_znx_big_normalize(
                res.data_mut(),
                res_base2k,
                cnv_offset_lo,
                i,
                &res_big,
                res_base2k,
                0,
                scratch_1,
            );
        }
    }
}

impl<BE: Backend> GLWEMulPlainDefault<BE> for Module<BE>
where
    Self: Sized
        + ModuleN
        + CnvPVecBytesOf
        + VecZnxDftBytesOf
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + Convolution<BE>
        + VecZnxBigNormalizeTmpBytes,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_mul_plain_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
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
        let lvl_2_cnv_apply: usize = self.cnv_apply_dft_tmp_bytes(res_dft_size, cnv_offset, a_size, b_size);

        let lvl_2_res_dft: usize = self.bytes_of_vec_znx_dft(1, res_dft_size);
        let lvl_2_norm: usize = self.vec_znx_big_normalize_tmp_bytes();
        let lvl_2: usize = lvl_2_res_dft + lvl_2_cnv_apply.max(lvl_2_norm);

        lvl_0 + lvl_1.max(lvl_2)
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWEPlaintext<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef,
    {
        assert_eq!(res.rank(), a.rank());
        assert!(
            scratch.available() >= self.glwe_mul_plain_tmp_bytes(res, a, b),
            "scratch.available(): {} < GLWEMulPlain::glwe_mul_plain_tmp_bytes: {}",
            scratch.available(),
            self.glwe_mul_plain_tmp_bytes(res, a, b)
        );

        let ab_base2k: usize = a.base2k().as_usize();
        assert_eq!(b.base2k().as_usize(), ab_base2k);
        assert_eq!(a_effective_k.div_ceil(ab_base2k), a.size());
        assert_eq!(b_effective_k.div_ceil(ab_base2k), b.size());
        let res_base2k: usize = res.base2k().as_usize();

        let cols: usize = res.rank().as_usize() + 1;

        let (mut a_prep, scratch_1) = scratch.take_cnv_pvec_left(self, cols, a.size());
        let (mut b_prep, scratch_2) = scratch_1.take_cnv_pvec_right(self, 1, b.size());

        let a_mask = msb_mask_bottom_limb(ab_base2k, a_effective_k);
        let b_mask = msb_mask_bottom_limb(ab_base2k, b_effective_k);

        self.cnv_prepare_left(&mut a_prep, a.data(), a_mask, scratch_2);
        self.cnv_prepare_right(&mut b_prep, b.data(), b_mask, scratch_2);

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < ab_base2k {
            (0, -((ab_base2k - (cnv_offset % ab_base2k)) as i64))
        } else {
            ((cnv_offset / ab_base2k).saturating_sub(1), (cnv_offset % ab_base2k) as i64)
        };

        let res_dft_size = a.size() + b.size() - cnv_offset_hi;

        for i in 0..cols {
            let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft(self, 1, res_dft_size);
            self.cnv_apply_dft(cnv_offset_hi, &mut res_dft, 0, &a_prep, i, &b_prep, 0, scratch_3);
            let res_big = self.vec_znx_idft_apply_consume(res_dft);

            self.vec_znx_big_normalize(
                res.data_mut(),
                res_base2k,
                cnv_offset_lo,
                i,
                &res_big,
                ab_base2k,
                0,
                scratch_3,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain_assign<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        res_effective_k: usize,
        a: &GLWEPlaintext<A>,
        a_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
    {
        let res_ref: &GLWE<&[u8]> = &res.to_ref();
        assert!(
            scratch.available() >= self.glwe_mul_plain_tmp_bytes(res_ref, res_ref, a),
            "scratch.available(): {} < GLWEMulPlain::glwe_mul_plain_tmp_bytes: {}",
            scratch.available(),
            self.glwe_mul_plain_tmp_bytes(res_ref, res_ref, a)
        );

        let ab_base2k: usize = a.base2k().as_usize();
        assert_eq!(res.base2k().as_usize(), ab_base2k);
        assert_eq!(res_effective_k.div_ceil(ab_base2k), res.size());
        assert_eq!(a_effective_k.div_ceil(ab_base2k), a.size());

        let cols: usize = res.rank().as_usize() + 1;

        let (mut res_prep, scratch_1) = scratch.take_cnv_pvec_left(self, cols, res.size());
        let (mut a_prep, scratch_2) = scratch_1.take_cnv_pvec_right(self, 1, a.size());

        let mask_res = msb_mask_bottom_limb(ab_base2k, res_effective_k);
        let mask_a = msb_mask_bottom_limb(ab_base2k, a_effective_k);

        self.cnv_prepare_left(&mut res_prep, res.data(), mask_res, scratch_2);
        self.cnv_prepare_right(&mut a_prep, a.data(), mask_a, scratch_2);

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < ab_base2k {
            (0, -((ab_base2k - (cnv_offset % ab_base2k)) as i64))
        } else {
            ((cnv_offset / ab_base2k).saturating_sub(1), (cnv_offset % ab_base2k) as i64)
        };

        let res_dft_size = a.size() + res.size() - cnv_offset_hi;

        for i in 0..cols {
            let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft(self, 1, res_dft_size);
            self.cnv_apply_dft(cnv_offset_hi, &mut res_dft, 0, &res_prep, i, &a_prep, 0, scratch_3);
            let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);
            self.vec_znx_big_normalize(res.data_mut(), ab_base2k, cnv_offset_lo, i, &res_big, ab_base2k, 0, scratch_3);
        }
    }
}

#[doc(hidden)]
pub trait GLWEMulPlainDefault<BE: Backend> {
    fn glwe_mul_plain_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWEPlaintext<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    fn glwe_mul_plain_assign<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        res_effective_k: usize,
        a: &GLWEPlaintext<A>,
        a_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef;
}

#[doc(hidden)]
pub trait GLWETensoringDefault<BE: Backend> {
    fn glwe_tensor_square_apply_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    fn glwe_tensor_apply_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(&self, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    fn glwe_tensor_relinearize<R, A, B>(
        &self,
        res: &mut GLWE<R>,
        a: &GLWETensor<A>,
        tsk: &GLWETensorKeyPrepared<B, BE>,
        tsk_size: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_square_apply<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply_add_assign<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;
}

impl<BE: Backend> GLWETensoringDefault<BE> for Module<BE>
where
    Self: Sized
        + ModuleN
        + CnvPVecBytesOf
        + VecZnxDftBytesOf
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + Convolution<BE>
        + VecZnxSubAssign
        + VecZnxNegate
        + VecZnxAddAssign
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopy
        + VecZnxNormalize<BE>
        + VecZnxDftApply<BE>
        + GGLWEProduct<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxNormalizeTmpBytes,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_tensor_square_apply_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
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
        let lvl_2_apply: usize = self.cnv_apply_dft_tmp_bytes(diag_dft_size, cnv_offset, a_size, a_size);
        let pairwise_dft_size =
            normalize_input_limb_bound_worst_case(2 * a_size, res_size, res.base2k().as_usize(), a.base2k().as_usize());
        let lvl_2_pairwise: usize = self.cnv_pairwise_apply_dft_tmp_bytes(cnv_offset, pairwise_dft_size, a_size, a_size);

        let lvl_2a: usize = self.bytes_of_vec_znx_dft(1, diag_dft_size) + lvl_2_apply.max(self.vec_znx_big_normalize_tmp_bytes());
        let lvl_2b: usize =
            self.bytes_of_vec_znx_dft(1, pairwise_dft_size) + lvl_2_pairwise.max(self.vec_znx_big_normalize_tmp_bytes());
        let lvl_2: usize = lvl_2a.max(lvl_2b);

        lvl_0 + lvl_diag_cache + lvl_1.max(lvl_2)
    }

    fn glwe_tensor_apply_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
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
        let lvl_2_apply: usize = self.cnv_apply_dft_tmp_bytes(diag_dft_size, cnv_offset, a_size, b_size);
        let pairwise_dft_size =
            normalize_input_limb_bound_worst_case(a_size + b_size, res_size, res.base2k().as_usize(), ab_base2k.as_usize());
        let lvl_2_pairwise: usize = self.cnv_pairwise_apply_dft_tmp_bytes(cnv_offset, pairwise_dft_size, a_size, b_size);

        let lvl_2a: usize = self.bytes_of_vec_znx_dft(1, diag_dft_size)
            + lvl_2_apply.max(VecZnx::bytes_of(self.n(), 1, res_size) + self.vec_znx_big_normalize_tmp_bytes());
        let lvl_2b: usize = self.bytes_of_vec_znx_dft(1, pairwise_dft_size)
            + lvl_2_pairwise.max(VecZnx::bytes_of(self.n(), 1, res_size) + self.vec_znx_big_normalize_tmp_bytes());
        let lvl_2: usize = lvl_2a.max(lvl_2b);

        lvl_0 + lvl_1.max(lvl_2)
    }

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(&self, res: &R, a: &A, tsk: &B) -> usize
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
        let lvl_1_gglwe_product: usize = self.gglwe_product_dft_tmp_bytes(res.size(), a_dft_size, tsk);
        let lvl_1_post_conv: usize = if res_base2k != key_base2k {
            VecZnx::bytes_of(self.n(), 1, a_dft_size) + self.vec_znx_normalize_tmp_bytes()
        } else {
            0
        };
        let lvl_1_big_norm: usize = self.vec_znx_big_normalize_tmp_bytes();
        let lvl_1_main: usize = lvl_1_res_dft + lvl_1_gglwe_product.max(lvl_1_post_conv).max(lvl_1_big_norm);
        let lvl_1: usize = lvl_1_pre_conv.max(lvl_1_main);

        lvl_0 + lvl_1
    }

    fn glwe_tensor_relinearize<R, A, B>(
        &self,
        res: &mut GLWE<R>,
        a: &GLWETensor<A>,
        tsk: &GLWETensorKeyPrepared<B, BE>,
        tsk_size: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef,
    {
        assert!(
            scratch.available() >= self.glwe_tensor_relinearize_tmp_bytes(res, a, tsk),
            "scratch.available(): {} < GLWETensoring::glwe_tensor_relinearize_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_relinearize_tmp_bytes(res, a, tsk)
        );

        let a_base2k: usize = a.base2k().into();
        let key_base2k: usize = tsk.base2k().into();
        let res_base2k: usize = res.base2k().into();

        assert_eq!(res.rank(), tsk.rank_out());
        assert_eq!(a.rank(), tsk.rank_out());

        let cols: usize = tsk.rank_out().as_usize() + 1;
        let pairs: usize = tsk.rank_in().as_usize();

        let a_dft_size: usize = (a.size() * a_base2k).div_ceil(key_base2k);

        let (mut a_dft, scratch_1) = scratch.take_vec_znx_dft(self, pairs, a_dft_size);

        if a_base2k != key_base2k {
            let (mut a_conv, scratch_2) = scratch_1.take_vec_znx(self.n(), 1, a_dft_size);
            for i in 0..pairs {
                self.vec_znx_normalize(&mut a_conv, key_base2k, 0, 0, a.data(), a_base2k, cols + i, scratch_2);
                self.vec_znx_dft_apply(1, 0, &mut a_dft, i, &a_conv, 0);
            }
        } else {
            for i in 0..pairs {
                self.vec_znx_dft_apply(1, 0, &mut a_dft, i, a.data(), cols + i);
            }
        }

        let (mut res_dft, scratch_2) = scratch_1.take_vec_znx_dft(self, cols, tsk_size); // Todo optimise

        self.gglwe_product_dft(&mut res_dft, &a_dft, &tsk.0, scratch_2);
        let mut res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);

        if res_base2k == key_base2k {
            for i in 0..cols {
                self.vec_znx_big_add_small_assign(&mut res_big, i, a.data(), i);
            }
        } else {
            let (mut a_conv, scratch_3) = scratch_2.take_vec_znx(self.n(), 1, a_dft_size);
            for i in 0..cols {
                self.vec_znx_normalize(&mut a_conv, key_base2k, 0, 0, a.data(), a_base2k, i, scratch_3);
                self.vec_znx_big_add_small_assign(&mut res_big, i, &a_conv, 0);
            }
        }

        for i in 0..(res.rank() + 1).into() {
            self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_2);
        }
    }

    fn glwe_tensor_square_apply<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
    {
        assert!(
            scratch.available() >= self.glwe_tensor_square_apply_tmp_bytes(res, a),
            "scratch.available(): {} < GLWETensoring::glwe_tensor_square_apply_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_square_apply_tmp_bytes(res, a)
        );

        let a_base2k: usize = a.base2k().as_usize();

        assert_eq!(a_effective_k.div_ceil(a_base2k), a.size());

        let res_base2k: usize = res.base2k().as_usize();
        let cols: usize = res.rank().as_usize() + 1;

        let (mut a_prep, scratch_1) = scratch.take_cnv_pvec_left(self, cols, a.size());
        let (mut b_prep, scratch_2) = scratch_1.take_cnv_pvec_right(self, cols, a.size());

        let a_mask = msb_mask_bottom_limb(a_base2k, a_effective_k);

        self.cnv_prepare_self(&mut a_prep, &mut b_prep, a.data(), a_mask, scratch_2);
        let (mut diag_terms, scratch_3) = scratch_2.take_vec_znx(self.n(), cols, res.size());

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

            let (mut res_dft, scratch_4) = scratch_3.take_vec_znx_dft(self, 1, diag_dft_size);
            self.cnv_apply_dft(cnv_offset_hi, &mut res_dft, 0, &a_prep, i, &b_prep, i, scratch_4);
            let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);
            self.vec_znx_big_normalize(
                &mut diag_terms,
                res_base2k,
                cnv_offset_lo,
                i,
                &res_big,
                a_base2k,
                0,
                scratch_4,
            );

            self.vec_znx_copy(res.data_mut(), col_i + i, &diag_terms, i);
        }

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            for j in i + 1..cols {
                let (mut res_dft, scratch_4) = scratch_3.take_vec_znx_dft(self, 1, pairwise_dft_size);
                self.cnv_pairwise_apply_dft(cnv_offset_hi, &mut res_dft, 0, &a_prep, &b_prep, i, j, scratch_4);
                let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);
                self.vec_znx_big_normalize(
                    res.data_mut(),
                    res_base2k,
                    cnv_offset_lo,
                    col_i + j,
                    &res_big,
                    a_base2k,
                    0,
                    scratch_4,
                );
                self.vec_znx_sub_assign(res.data_mut(), col_i + j, &diag_terms, i);
                self.vec_znx_sub_assign(res.data_mut(), col_i + j, &diag_terms, j);
            }
        }
    }

    fn glwe_tensor_apply<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef,
    {
        assert!(
            scratch.available() >= self.glwe_tensor_apply_tmp_bytes(res, a, b),
            "scratch.available(): {} < GLWETensoring::glwe_tensor_apply_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_apply_tmp_bytes(res, a, b)
        );

        let ab_base2k: usize = a.base2k().as_usize();
        assert_eq!(b.base2k().as_usize(), ab_base2k);
        assert_eq!(a_effective_k.div_ceil(ab_base2k), a.size());
        assert_eq!(b_effective_k.div_ceil(ab_base2k), b.size());

        let res_base2k: usize = res.base2k().as_usize();

        let cols: usize = res.rank().as_usize() + 1;

        let (mut a_prep, scratch_1) = scratch.take_cnv_pvec_left(self, cols, a.size());
        let (mut b_prep, scratch_2) = scratch_1.take_cnv_pvec_right(self, cols, b.size());

        let a_mask = msb_mask_bottom_limb(ab_base2k, a_effective_k);
        let b_mask = msb_mask_bottom_limb(ab_base2k, b_effective_k);

        self.cnv_prepare_left(&mut a_prep, a.data(), a_mask, scratch_2);
        self.cnv_prepare_right(&mut b_prep, b.data(), b_mask, scratch_2);

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

            let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft(self, 1, diag_dft_size);
            self.cnv_apply_dft(cnv_offset_hi, &mut res_dft, 0, &a_prep, i, &b_prep, i, scratch_3);
            let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);
            let (mut tmp, scratch_4) = scratch_3.take_vec_znx(self.n(), 1, res.size());
            self.vec_znx_big_normalize(&mut tmp, res_base2k, cnv_offset_lo, 0, &res_big, ab_base2k, 0, scratch_4);

            self.vec_znx_copy(res.data_mut(), col_i + i, &tmp, 0);

            // Pre-subtracts
            // res[i!=j] = NEG(a[i] * b[i]) + SUB(a[j] * b[j])
            for j in 0..cols {
                if j != i {
                    if j < i {
                        let col_j = j * cols - (j * (j + 1) / 2);
                        self.vec_znx_sub_assign(res.data_mut(), col_j + i, &tmp, 0);
                    } else {
                        self.vec_znx_negate(res.data_mut(), col_i + j, &tmp, 0);
                    }
                }
            }
        }

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            for j in i..cols {
                if j != i {
                    // res_dft = (a[i] + a[j]) * (b[i] + b[j])
                    let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft(self, 1, pairwise_dft_size);
                    self.cnv_pairwise_apply_dft(cnv_offset_hi, &mut res_dft, 0, &a_prep, &b_prep, i, j, scratch_3);
                    let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);
                    let (mut tmp, scratch_3) = scratch_3.take_vec_znx(self.n(), 1, res.size());
                    self.vec_znx_big_normalize(&mut tmp, res_base2k, cnv_offset_lo, 0, &res_big, ab_base2k, 0, scratch_3);

                    self.vec_znx_add_assign(res.data_mut(), col_i + j, &tmp, 0);
                }
            }
        }
    }

    fn glwe_tensor_apply_add_assign<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef,
    {
        assert!(
            scratch.available() >= self.glwe_tensor_apply_tmp_bytes(res, a, b),
            "scratch.available(): {} < GLWETensoring::glwe_tensor_apply_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_apply_tmp_bytes(res, a, b)
        );

        let ab_base2k: usize = a.base2k().as_usize();
        assert_eq!(b.base2k().as_usize(), ab_base2k);
        assert_eq!(a_effective_k.div_ceil(ab_base2k), a.size());
        assert_eq!(b_effective_k.div_ceil(ab_base2k), b.size());

        let res_base2k: usize = res.base2k().as_usize();
        let cols: usize = res.rank().as_usize() + 1;

        let (mut a_prep, scratch_1) = scratch.take_cnv_pvec_left(self, cols, a.size());
        let (mut b_prep, scratch_2) = scratch_1.take_cnv_pvec_right(self, cols, b.size());

        let a_mask = msb_mask_bottom_limb(ab_base2k, a_effective_k);
        let b_mask = msb_mask_bottom_limb(ab_base2k, b_effective_k);

        self.cnv_prepare_left(&mut a_prep, a.data(), a_mask, scratch_2);
        self.cnv_prepare_right(&mut b_prep, b.data(), b_mask, scratch_2);

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

            let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft(self, 1, diag_dft_size);
            self.cnv_apply_dft(cnv_offset_hi, &mut res_dft, 0, &a_prep, i, &b_prep, i, scratch_3);
            let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);
            let (mut tmp, scratch_4) = scratch_3.take_vec_znx(self.n(), 1, res.size());
            self.vec_znx_big_normalize(&mut tmp, res_base2k, cnv_offset_lo, 0, &res_big, ab_base2k, 0, scratch_4);

            self.vec_znx_add_assign(res.data_mut(), col_i + i, &tmp, 0);

            for j in 0..cols {
                if j != i {
                    if j < i {
                        let col_j = j * cols - (j * (j + 1) / 2);
                        self.vec_znx_sub_assign(res.data_mut(), col_j + i, &tmp, 0);
                    } else {
                        self.vec_znx_sub_assign(res.data_mut(), col_i + j, &tmp, 0);
                    }
                }
            }
        }

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            for j in i..cols {
                if j != i {
                    let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft(self, 1, pairwise_dft_size);
                    self.cnv_pairwise_apply_dft(cnv_offset_hi, &mut res_dft, 0, &a_prep, &b_prep, i, j, scratch_3);
                    let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);
                    let (mut tmp, scratch_3) = scratch_3.take_vec_znx(self.n(), 1, res.size());
                    self.vec_znx_big_normalize(&mut tmp, res_base2k, cnv_offset_lo, 0, &res_big, ab_base2k, 0, scratch_3);

                    self.vec_znx_add_assign(res.data_mut(), col_i + j, &tmp, 0);
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

impl<BE: Backend> GLWEAdd for Module<BE> where Self: ModuleN + VecZnxAddInto + VecZnxCopy + VecZnxAddAssign + VecZnxZero {}

impl<BE: Backend> GLWESub for Module<BE> where
    Self: ModuleN + VecZnxSub + VecZnxCopy + VecZnxNegate + VecZnxZero + VecZnxSubAssign + VecZnxSubNegateAssign
{
}

impl<BE: Backend> GLWENegate for Module<BE> where Self: VecZnxNegate + VecZnxNegateAssign + VecZnxZero + ModuleN {}

impl<BE: Backend> GLWERotateDefault<BE> for Module<BE> where
    Self: ModuleN + VecZnxRotate + VecZnxRotateAssign<BE> + VecZnxRotateAssignTmpBytes + VecZnxZero
{
}

#[doc(hidden)]
pub trait GLWERotateDefault<BE: Backend>
where
    Self: ModuleN + VecZnxRotate + VecZnxRotateAssign<BE> + VecZnxRotateAssignTmpBytes + VecZnxZero,
{
    fn glwe_rotate_tmp_bytes(&self) -> usize {
        self.vec_znx_rotate_assign_tmp_bytes()
    }

    fn glwe_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert!(res.rank() == a.rank() || a.rank() == 0);

        let res_cols = (res.rank() + 1).into();
        let a_cols = (a.rank() + 1).into();

        for i in 0..a_cols {
            self.vec_znx_rotate(k, res.data_mut(), i, a.data(), i);
        }
        for i in a_cols..res_cols {
            self.vec_znx_zero(res.data_mut(), i);
        }
    }

    fn glwe_rotate_assign<R>(&self, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        assert!(
            scratch.available() >= self.glwe_rotate_tmp_bytes(),
            "scratch.available(): {} < GLWERotate::glwe_rotate_tmp_bytes: {}",
            scratch.available(),
            self.glwe_rotate_tmp_bytes()
        );

        for i in 0..(res.rank() + 1).into() {
            self.vec_znx_rotate_assign(k, res.data_mut(), i, scratch);
        }
    }
}

impl<BE: Backend> GLWEMulXpMinusOneDefault<BE> for Module<BE> where
    Self: ModuleN + VecZnxMulXpMinusOne + VecZnxMulXpMinusOneAssign<BE>
{
}

#[doc(hidden)]
pub trait GLWEMulXpMinusOneDefault<BE: Backend>
where
    Self: ModuleN + VecZnxMulXpMinusOne + VecZnxMulXpMinusOneAssign<BE>,
{
    fn glwe_mul_xp_minus_one<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank());

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_mul_xp_minus_one(k, res.data_mut(), i, a.data(), i);
        }
    }

    fn glwe_mul_xp_minus_one_assign<R>(&self, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        assert_eq!(res.n(), self.n() as u32);

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_mul_xp_minus_one_assign(k, res.data_mut(), i, scratch);
        }
    }
}

impl<BE: Backend> GLWECopy for Module<BE> where Self: ModuleN + VecZnxCopy + VecZnxZero {}

impl<BE: Backend> GLWEShiftDefault<BE> for Module<BE> where
    Self: ModuleN
        + VecZnxRshAssign<BE>
        + VecZnxLshAddInto<BE>
        + VecZnxLshSub<BE>
        + VecZnxRshTmpBytes
        + VecZnxLshTmpBytes
        + VecZnxLshAssign<BE>
        + VecZnxLsh<BE>
{
}

#[doc(hidden)]
pub trait GLWEShiftDefault<BE: Backend>
where
    Self: ModuleN
        + VecZnxRshAssign<BE>
        + VecZnxLshAddInto<BE>
        + VecZnxLshSub<BE>
        + VecZnxRshTmpBytes
        + VecZnxLshTmpBytes
        + VecZnxLshAssign<BE>
        + VecZnxLsh<BE>,
{
    fn glwe_shift_tmp_bytes(&self) -> usize {
        let lvl_0: usize = self.vec_znx_rsh_tmp_bytes().max(self.vec_znx_lsh_tmp_bytes());
        lvl_0
    }

    fn glwe_rsh<R>(&self, k: usize, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res = &mut res.to_mut();
        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );
        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_rsh_assign(base2k, k, res.data_mut(), i, scratch);
        }
    }

    fn glwe_lsh_assign<R>(&self, res: &mut R, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res = &mut res.to_mut();

        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_lsh_assign(base2k, k, res.data_mut(), i, scratch);
        }
    }

    fn glwe_lsh<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res = &mut res.to_mut();
        let a = &a.to_ref();
        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_lsh(base2k, k, res.data_mut(), i, a.data(), i, scratch);
        }
    }

    fn glwe_lsh_add<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res = &mut res.to_mut();
        let a = &a.to_ref();
        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_lsh_add_into(base2k, k, res.data_mut(), i, a.data(), i, scratch);
        }
    }

    fn glwe_lsh_sub<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res = &mut res.to_mut();
        let a = &a.to_ref();
        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_lsh_sub(base2k, k, res.data_mut(), i, a.data(), i, scratch);
        }
    }
}

impl<BE: Backend> GLWENormalizeDefault<BE> for Module<BE> where
    Self: ModuleN + VecZnxNormalize<BE> + VecZnxNormalizeAssign<BE> + VecZnxNormalizeTmpBytes
{
}

#[doc(hidden)]
pub trait GLWENormalizeDefault<BE: Backend>
where
    Self: ModuleN + VecZnxNormalize<BE> + VecZnxNormalizeAssign<BE> + VecZnxNormalizeTmpBytes,
{
    fn glwe_normalize_tmp_bytes(&self) -> usize {
        let lvl_0: usize = self.vec_znx_normalize_tmp_bytes();
        lvl_0
    }

    fn glwe_maybe_cross_normalize_to_ref<'a, A>(
        &self,
        glwe: &'a A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>,
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToRef + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        if glwe.base2k().as_usize() == target_base2k {
            tmp_slot.take();
            return (glwe.to_ref(), scratch);
        }

        let mut layout = glwe.glwe_layout();
        layout.base2k = target_base2k.into();

        let (tmp, scratch2) = scratch.take_glwe(&layout);
        *tmp_slot = Some(tmp);

        let tmp_ref: &mut GLWE<&mut [u8]> = tmp_slot.as_mut().expect("tmp_slot just set to Some, but found None");

        self.glwe_normalize(tmp_ref, glwe, scratch2);

        (tmp_ref.to_ref(), scratch2)
    }

    fn glwe_maybe_cross_normalize_to_mut<'a, A>(
        &self,
        glwe: &'a mut A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>,
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a mut [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToMut + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        if glwe.base2k().as_usize() == target_base2k {
            tmp_slot.take();
            return (glwe.to_mut(), scratch);
        }

        let mut layout = glwe.glwe_layout();
        layout.base2k = target_base2k.into();

        let (tmp, scratch2) = scratch.take_glwe(&layout);
        *tmp_slot = Some(tmp);

        let tmp_ref: &mut GLWE<&mut [u8]> = tmp_slot.as_mut().expect("tmp_slot just set to Some, but found None");

        self.glwe_normalize(tmp_ref, glwe, scratch2);

        (tmp_ref.to_mut(), scratch2)
    }

    fn glwe_normalize<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank());
        assert!(
            scratch.available() >= self.glwe_normalize_tmp_bytes(),
            "scratch.available(): {} < GLWENormalize::glwe_normalize_tmp_bytes: {}",
            scratch.available(),
            self.glwe_normalize_tmp_bytes()
        );

        let res_base2k = res.base2k().into();

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_normalize(res.data_mut(), res_base2k, 0, i, a.data(), a.base2k().into(), i, scratch);
        }
    }

    fn glwe_normalize_assign<R>(&self, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        assert!(
            scratch.available() >= self.glwe_normalize_tmp_bytes(),
            "scratch.available(): {} < GLWENormalize::glwe_normalize_tmp_bytes: {}",
            scratch.available(),
            self.glwe_normalize_tmp_bytes()
        );
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_normalize_assign(res.base2k().into(), res.data_mut(), i, scratch);
        }
    }
}
