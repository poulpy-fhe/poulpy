#[cfg(feature = "enable-ifma")]
use crate::NTT3x42Ifma;
use crate::{FFT64Avx512, NTT4x30Avx512};
use poulpy_core::{
    default::operations::{GLWETensoringDefault, msb_mask_bottom_limb},
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_gglwe_product_digits_strided_default, impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full,
    impl_ggsw_keyswitch_defaults_full, impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full,
    impl_glwe_keyswitch_defaults_full, impl_glwe_packing_defaults_full, impl_glwe_tensoring_default,
    impl_glwe_trace_defaults_full, impl_linear_transformation_defaults_full, impl_lwe_keyswitch_defaults_full,
    layouts::{Degree, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef},
    oep::GLWETensoringImpl,
};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxDftBytesOf, VecZnxIdftApplyTmpA, VecZnxSubAssignBackend,
    },
    layouts::{
        Backend, CnvPVecLBackendRef, CnvPVecLToBackendRef, CnvPVecRBackendRef, CnvPVecRToBackendRef, Module, ScratchArena,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendMut,
        VecZnxToBackendMut, VecZnxToBackendRef, VmpPMatBackendRef,
    },
};

impl_glwe_tensoring_default!(FFT64Avx512);
impl_gglwe_product_digits_strided_default!(FFT64Avx512);

trait RankOneTensorDft: Backend {
    fn rank_one_tensor_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize;

    fn rank_one_tensor_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        cnv_offset: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}

impl RankOneTensorDft for NTT4x30Avx512 {
    fn rank_one_tensor_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
        crate::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512_tmp_bytes(res_size, a_size, b_size)
    }

    fn rank_one_tensor_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        cnv_offset: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = Self::rank_one_tensor_dft_tmp_bytes(res.size(), a.size(), b.size());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe { crate::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512(module, res, cnv_offset, a, b, tmp) };
    }
}

#[cfg(feature = "enable-ifma")]
impl RankOneTensorDft for NTT3x42Ifma {
    fn rank_one_tensor_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
        crate::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma_tmp_bytes(res_size, a_size, b_size)
    }

    fn rank_one_tensor_dft(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        cnv_offset: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = Self::rank_one_tensor_dft_tmp_bytes(res.size(), a.size(), b.size());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe { crate::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma(res, cnv_offset, a, b, tmp) };
    }
}

/// Enforces the Core degree contract before specialized kernels size scratch
/// from `module` and index the operands.
#[inline]
fn assert_degrees<BE: Backend>(module: &Module<BE>, degrees: [Degree; 3]) {
    for n in degrees {
        let n: u32 = n.into();
        assert_eq!(module.n() as u32, n, "operand degree does not match the module");
    }
}

fn rank_one_tensor_supported<BE: Backend, R: GLWEInfos>(module: &Module<BE>, res: &R) -> bool {
    res.rank().as_usize() == 1 && matches!(module.n(), 32768 | 65536)
}

#[inline]
fn cnv_offset_to_limb_offset(cnv_offset: usize, base2k: usize) -> (usize, i64) {
    assert_ne!(base2k, 0);
    if cnv_offset < base2k {
        (0, -((base2k - (cnv_offset % base2k)) as i64))
    } else {
        ((cnv_offset / base2k).saturating_sub(1), (cnv_offset % base2k) as i64)
    }
}

#[inline]
fn normalize_input_limb_bound(full_size: usize, res_size: usize, res_base2k: usize, in_base2k: usize, res_offset: i64) -> usize {
    let mut offset_bits = res_offset % in_base2k as i64;
    if res_offset < 0 && offset_bits != 0 {
        offset_bits += in_base2k as i64;
    }
    full_size.min((res_size * res_base2k + offset_bits as usize).div_ceil(in_base2k))
}

fn rank_one_tensor_work_bytes<BE: RankOneTensorDft>(
    module: &Module<BE>,
    res_size: usize,
    dft_size: usize,
    a_size: usize,
    b_size: usize,
) -> usize
where
    Module<BE>: VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
{
    let kernel = BE::rank_one_tensor_dft_tmp_bytes(dft_size, a_size, b_size);
    let normalize = module.bytes_of_vec_znx_big(1, dft_size)
        + BE::bytes_of_vec_znx(module.n(), 1, res_size)
        + module.vec_znx_big_normalize_tmp_bytes();
    BE::bytes_of_vec_znx(module.n(), 2, res_size) + module.bytes_of_vec_znx_dft(3, dft_size) + kernel.max(normalize)
}

fn rank_one_tensor_apply_tmp_bytes<BE, R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
where
    BE: RankOneTensorDft,
    Module<BE>: ModuleN + CnvPVecBytesOf + Convolution<BE> + VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    B: GLWEInfos,
{
    assert_degrees(module, [res.n(), a.n(), b.n()]);
    let base2k = a.base2k().as_usize();
    assert_eq!(b.base2k().as_usize(), base2k);
    let a_size = a.k().as_usize().div_ceil(base2k);
    let b_size = b.k().as_usize().div_ceil(base2k);
    let dft_size = (a_size + b_size).min((res.size() * res.base2k().as_usize() + base2k - 1).div_ceil(base2k));
    let prepared = module.bytes_of_cnv_pvec_left(2, a_size) + module.bytes_of_cnv_pvec_right(2, b_size);
    let prepare = module
        .cnv_prepare_left_tmp_bytes(a_size, a_size)
        .max(module.cnv_prepare_right_tmp_bytes(b_size, b_size));
    prepared + prepare.max(rank_one_tensor_work_bytes(module, res.size(), dft_size, a_size, b_size))
}

fn rank_one_tensor_square_tmp_bytes<BE, R, A>(module: &Module<BE>, res: &R, a: &A) -> usize
where
    BE: RankOneTensorDft,
    Module<BE>: ModuleN + CnvPVecBytesOf + Convolution<BE> + VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
{
    assert_degrees(module, [res.n(), a.n(), a.n()]);
    let base2k = a.base2k().as_usize();
    let a_size = a.k().as_usize().div_ceil(base2k);
    let dft_size = (2 * a_size).min((res.size() * res.base2k().as_usize() + base2k - 1).div_ceil(base2k));
    let prepared = module.bytes_of_cnv_pvec_left(2, a_size) + module.bytes_of_cnv_pvec_right(2, a_size);
    let prepare = module.cnv_prepare_self_tmp_bytes(a_size, a_size);
    prepared + prepare.max(rank_one_tensor_work_bytes(module, res.size(), dft_size, a_size, a_size))
}

#[allow(clippy::too_many_arguments)]
fn rank_one_tensor_finish<BE, R, AP, BP>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    a_prep: &AP,
    b_prep: &BP,
    a_size: usize,
    b_size: usize,
    in_base2k: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: RankOneTensorDft,
    Module<BE>: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopyBackend<BE>
        + VecZnxSubAssignBackend<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    AP: CnvPVecLToBackendRef<BE>,
    BP: CnvPVecRToBackendRef<BE>,
{
    let res_base2k = res.base2k().as_usize();
    let (cnv_offset_hi, cnv_offset_lo) = cnv_offset_to_limb_offset(cnv_offset, in_base2k);
    let dft_size = normalize_input_limb_bound(
        a_size + b_size - cnv_offset_hi,
        res.size(),
        res_base2k,
        in_base2k,
        cnv_offset_lo,
    );
    let (mut diag_terms, scratch) = scratch.borrow().take_vec_znx_scratch(module.n(), 2, res.size());
    let (mut tensor_dft, mut work) = scratch.take_vec_znx_dft_scratch(module, 3, dft_size);
    BE::rank_one_tensor_dft(
        module,
        &mut tensor_dft.to_backend_mut(),
        cnv_offset_hi,
        &a_prep.to_backend_ref(),
        &b_prep.to_backend_ref(),
        &mut work,
    );

    for (dft_col, diag_col, res_col) in [(0, 0, 0), (2, 1, 2)] {
        let (mut product_big, mut norm_scratch) = work.borrow().take_vec_znx_big_scratch(module, 1, dft_size);
        module.vec_znx_idft_apply_tmpa(
            &mut product_big.to_backend_mut(),
            0,
            &mut tensor_dft.to_backend_mut(),
            dft_col,
        );
        module.vec_znx_big_normalize(
            &mut diag_terms.to_backend_mut(),
            res_base2k,
            cnv_offset_lo,
            diag_col,
            &product_big.to_backend_ref(),
            in_base2k,
            0,
            &mut norm_scratch,
        );
        module.vec_znx_copy_backend(
            res.to_backend_mut().data_mut(),
            res_col,
            &diag_terms.to_backend_ref(),
            diag_col,
        );
    }

    let (mut product_big, scratch) = work.borrow().take_vec_znx_big_scratch(module, 1, dft_size);
    module.vec_znx_idft_apply_tmpa(&mut product_big.to_backend_mut(), 0, &mut tensor_dft.to_backend_mut(), 1);
    let (mut pairwise, mut norm_scratch) = scratch.take_vec_znx_scratch(module.n(), 1, res.size());
    module.vec_znx_big_normalize(
        &mut pairwise.to_backend_mut(),
        res_base2k,
        cnv_offset_lo,
        0,
        &product_big.to_backend_ref(),
        in_base2k,
        0,
        &mut norm_scratch,
    );
    {
        let mut pairwise = pairwise.to_backend_mut();
        let diag_terms = diag_terms.to_backend_ref();
        module.vec_znx_sub_assign_backend(&mut pairwise, 0, &diag_terms, 0);
        module.vec_znx_sub_assign_backend(&mut pairwise, 0, &diag_terms, 1);
    }
    module.vec_znx_copy_backend(res.to_backend_mut().data_mut(), 1, &pairwise.to_backend_ref(), 0);
}

fn rank_one_tensor_apply<BE, R, A, B>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    a: &A,
    b: &B,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: RankOneTensorDft,
    Module<BE>: ModuleN
        + CnvPVecBytesOf
        + Convolution<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopyBackend<BE>
        + VecZnxSubAssignBackend<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    B: GLWEToBackendRef<BE> + GLWEInfos,
{
    assert_degrees(module, [res.n(), a.n(), b.n()]);
    assert!(scratch.available() >= rank_one_tensor_apply_tmp_bytes(module, res, a, b));
    let base2k = a.base2k().as_usize();
    assert_eq!(b.base2k().as_usize(), base2k);
    let a_size = a.k().as_usize().div_ceil(base2k);
    let b_size = b.k().as_usize().div_ceil(base2k);
    assert!(a_size <= a.size());
    assert!(b_size <= b.size());
    let (mut a_prep, scratch) = scratch.borrow().take_cnv_pvec_left_scratch(module, 2, a_size);
    let (mut b_prep, mut scratch) = scratch.take_cnv_pvec_right_scratch(module, 2, b_size);
    {
        let mut prep_scratch = scratch.borrow();
        module.cnv_prepare_left(
            &mut a_prep,
            a.to_backend_ref().data(),
            msb_mask_bottom_limb(base2k, a.k().as_usize()),
            &mut prep_scratch,
        );
        module.cnv_prepare_right(
            &mut b_prep,
            b.to_backend_ref().data(),
            msb_mask_bottom_limb(base2k, b.k().as_usize()),
            &mut prep_scratch,
        );
    }
    rank_one_tensor_finish(
        module,
        cnv_offset,
        res,
        &a_prep,
        &b_prep,
        a_size,
        b_size,
        base2k,
        &mut scratch,
    );
}

fn rank_one_tensor_square<BE, R, A>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    a: &A,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: RankOneTensorDft,
    Module<BE>: ModuleN
        + CnvPVecBytesOf
        + Convolution<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopyBackend<BE>
        + VecZnxSubAssignBackend<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    assert_degrees(module, [res.n(), a.n(), a.n()]);
    assert!(scratch.available() >= rank_one_tensor_square_tmp_bytes(module, res, a));
    let base2k = a.base2k().as_usize();
    let a_size = a.k().as_usize().div_ceil(base2k);
    assert!(a_size <= a.size());
    let (mut a_prep, scratch) = scratch.borrow().take_cnv_pvec_left_scratch(module, 2, a_size);
    let (mut b_prep, mut scratch) = scratch.take_cnv_pvec_right_scratch(module, 2, a_size);
    {
        let mut prep_scratch = scratch.borrow();
        module.cnv_prepare_self(
            &mut a_prep,
            &mut b_prep,
            a.to_backend_ref().data(),
            msb_mask_bottom_limb(base2k, a.k().as_usize()),
            &mut prep_scratch,
        );
    }
    rank_one_tensor_finish(
        module,
        cnv_offset,
        res,
        &a_prep,
        &b_prep,
        a_size,
        a_size,
        base2k,
        &mut scratch,
    );
}

macro_rules! impl_rank_one_tensoring {
    ($be:ty) => {
        unsafe impl GLWETensoringImpl<$be> for $be {
            fn glwe_tensor_apply_tmp_bytes<R, A, B>(module: &Module<$be>, res: &R, a: &A, b: &B) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
                B: GLWEInfos,
            {
                if rank_one_tensor_supported(module, res) {
                    rank_one_tensor_apply_tmp_bytes(module, res, a, b)
                } else {
                    module.glwe_tensor_apply_tmp_bytes_default(res, a, b)
                }
            }

            fn glwe_tensor_square_apply_tmp_bytes<R, A>(module: &Module<$be>, res: &R, a: &A) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
            {
                if rank_one_tensor_supported(module, res) {
                    rank_one_tensor_square_tmp_bytes(module, res, a)
                } else {
                    module.glwe_tensor_square_apply_tmp_bytes_default(res, a)
                }
            }

            fn glwe_tensor_apply<R, A, B>(
                module: &Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &A,
                b: &B,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: GLWEToBackendMut<$be> + GLWEInfos,
                A: GLWEToBackendRef<$be> + GLWEInfos,
                B: GLWEToBackendRef<$be> + GLWEInfos,
            {
                if rank_one_tensor_supported(module, res) {
                    rank_one_tensor_apply(module, cnv_offset, res, a, b, scratch)
                } else {
                    module.glwe_tensor_apply_default(cnv_offset, res, a, b, scratch)
                }
            }

            fn glwe_tensor_square_apply<R, A>(
                module: &Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &A,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: GLWEToBackendMut<$be> + GLWEInfos,
                A: GLWEToBackendRef<$be> + GLWEInfos,
            {
                if rank_one_tensor_supported(module, res) {
                    rank_one_tensor_square(module, cnv_offset, res, a, scratch)
                } else {
                    module.glwe_tensor_square_apply_default(cnv_offset, res, a, scratch)
                }
            }

            fn glwe_tensor_relinearize<R, A, T>(
                module: &Module<$be>,
                res: &mut R,
                a: &A,
                tsk: &T,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: GLWEToBackendMut<$be> + GLWEInfos,
                A: GLWEToBackendRef<$be> + GLWEInfos,
                T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<$be>,
            {
                module.glwe_tensor_relinearize_default(res, a, tsk, scratch)
            }

            fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(module: &Module<$be>, res: &R, a: &A, tsk: &B) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
                B: GGLWEInfos,
            {
                module.glwe_tensor_relinearize_tmp_bytes_default(res, a, tsk)
            }
        }
    };
}

impl_rank_one_tensoring!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_rank_one_tensoring!(NTT3x42Ifma);

unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    fn gglwe_product_digits_strided_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_cols: usize,
        a_size: usize,
        dsize: usize,
        pmat_rows: usize,
        pmat_cols_in: usize,
        _pmat_cols_out: usize,
        _pmat_size: usize,
    ) -> usize {
        crate::ntt4x30_avx512::vmp::vmp_apply_digits_strided_tmp_bytes_avx(a_cols, a_size, dsize, pmat_rows, pmat_cols_in)
    }

    fn gglwe_product_digits_strided(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = Self::gglwe_product_digits_strided_tmp_bytes(
            module,
            res.size(),
            a.cols(),
            a.size(),
            dsize,
            pmat.rows(),
            pmat.cols_in(),
            pmat.cols_out(),
            pmat.size(),
        );
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / std::mem::size_of::<u64>());
        crate::ntt4x30_avx512::vmp::vmp_apply_dft_to_dft_digits_strided_avx(module, res, a, dsize, pmat, tmp);
    }
}

#[cfg(feature = "enable-ifma")]
unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl<NTT3x42Ifma> for NTT3x42Ifma {
    fn gglwe_product_digits_strided_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_cols: usize,
        a_size: usize,
        dsize: usize,
        pmat_rows: usize,
        pmat_cols_in: usize,
        _pmat_cols_out: usize,
        _pmat_size: usize,
    ) -> usize {
        crate::ntt3x42_ifma::vmp::vmp_apply_digits_strided_tmp_bytes_ifma(a_cols, a_size, dsize, pmat_rows, pmat_cols_in)
    }

    fn gglwe_product_digits_strided(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = Self::gglwe_product_digits_strided_tmp_bytes(
            module,
            res.size(),
            a.cols(),
            a.size(),
            dsize,
            pmat.rows(),
            pmat.cols_in(),
            pmat.cols_out(),
            pmat.size(),
        );
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / std::mem::size_of::<u64>());
        crate::ntt3x42_ifma::vmp::vmp_apply_dft_to_dft_digits_strided_ifma(module, res, a, dsize, pmat, tmp);
    }
}

impl_glwe_automorphism_defaults_full!(FFT64Avx512);
impl_glwe_automorphism_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_automorphism_defaults_full!(NTT3x42Ifma);

impl_ggsw_automorphism_defaults_full!(FFT64Avx512);
impl_ggsw_automorphism_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ggsw_automorphism_defaults_full!(NTT3x42Ifma);

impl_gglwe_automorphism_defaults_full!(FFT64Avx512);
impl_gglwe_automorphism_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_gglwe_automorphism_defaults_full!(NTT3x42Ifma);

impl_decryption_defaults_full!(FFT64Avx512);
impl_decryption_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_decryption_defaults_full!(NTT3x42Ifma);

impl_glwe_trace_defaults_full!(FFT64Avx512);
impl_glwe_trace_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_trace_defaults_full!(NTT3x42Ifma);

impl_glwe_packing_defaults_full!(FFT64Avx512);
impl_glwe_packing_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_packing_defaults_full!(NTT3x42Ifma);

impl_conversion_defaults_full!(FFT64Avx512);
impl_conversion_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_conversion_defaults_full!(NTT3x42Ifma);

impl_glwe_keyswitch_defaults_full!(FFT64Avx512);
impl_glwe_keyswitch_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_keyswitch_defaults_full!(NTT3x42Ifma);

impl_gglwe_keyswitch_defaults_full!(FFT64Avx512);
impl_gglwe_keyswitch_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_gglwe_keyswitch_defaults_full!(NTT3x42Ifma);

impl_ggsw_keyswitch_defaults_full!(FFT64Avx512);
impl_ggsw_keyswitch_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ggsw_keyswitch_defaults_full!(NTT3x42Ifma);

impl_lwe_keyswitch_defaults_full!(FFT64Avx512);
impl_lwe_keyswitch_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_lwe_keyswitch_defaults_full!(NTT3x42Ifma);

impl_encryption_defaults_full!(FFT64Avx512);
impl_encryption_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_encryption_defaults_full!(NTT3x42Ifma);

impl_glwe_external_product_defaults_full!(FFT64Avx512);
impl_glwe_external_product_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_external_product_defaults_full!(NTT3x42Ifma);

impl_gglwe_external_product_defaults_full!(FFT64Avx512);
impl_gglwe_external_product_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_gglwe_external_product_defaults_full!(NTT3x42Ifma);

impl_ggsw_external_product_defaults_full!(FFT64Avx512);
impl_ggsw_external_product_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ggsw_external_product_defaults_full!(NTT3x42Ifma);

impl_linear_transformation_defaults_full!(FFT64Avx512);
impl_linear_transformation_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_linear_transformation_defaults_full!(NTT3x42Ifma);
