#[cfg(feature = "enable-ifma")]
use crate::NTT3x42Ifma;
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
use crate::NTT3x42IfmaRayon;
use crate::{FFT64Avx512, NTT4x30Avx512};
#[cfg(feature = "enable-rayon")]
use crate::{FFT64Avx512Rayon, NTT4x30Avx512Rayon};
use poulpy_core::{
    impl_conversion_reference_full, impl_decryption_reference_full, impl_encryption_reference_full,
    impl_gglwe_automorphism_reference_full, impl_gglwe_external_product_reference_full, impl_gglwe_keyswitch_reference_full,
    impl_gglwe_product_digits_strided_reference, impl_ggsw_automorphism_reference_full,
    impl_ggsw_external_product_reference_full, impl_ggsw_keyswitch_reference_full, impl_glwe_automorphism_reference_full,
    impl_glwe_external_product_reference_full, impl_glwe_keyswitch_reference_full, impl_glwe_packing_reference_full,
    impl_glwe_tensoring_reference, impl_glwe_trace_reference_full, impl_linear_transformation_reference_full,
    impl_lwe_keyswitch_reference_full,
    layouts::{Degree, GGLWEInfos, GGLWEPreparedToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
    oep::GLWETensoringImpl,
    reference::keyswitching::glwe::{GGLWEProductReference, gglwe_product_output_size},
    reference::operations::{GLWETensoringReference, cnv_offset_to_limb_offset, normalize_input_limb_bound_with_offset},
};
#[cfg(feature = "enable-ifma")]
use poulpy_hal::layouts::{DataViewMut, VecZnxDft};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyTmpA,
        VecZnxIdftNormalizeConsume, VecZnxIdftNormalizeConsumeTmpBytes, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes,
        VecZnxSubAssign,
    },
    layouts::{
        Backend, CnvPVecLBackendRef, CnvPVecLToBackendRef, CnvPVecRBackendRef, CnvPVecRToBackendRef, Module, PrepareHint,
        ScratchArena, VecZnxBackendMut, VecZnxBigToBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendMut,
        VecZnxDftToBackendRef, VecZnxToBackendMut, VecZnxToBackendRef, VmpPMatBackendRef,
    },
};

impl_glwe_tensoring_reference!(FFT64Avx512);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64Avx512Rayon);
impl_gglwe_product_digits_strided_reference!(FFT64Avx512);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64Avx512Rayon);

trait RankOneTensorDft: Backend {
    fn tensor_finish_tmp_bytes(module: &Module<Self>, n: usize, size: usize) -> usize
    where
        Module<Self>: VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
    {
        module.bytes_of_vec_znx_big(n, 1, size) + module.vec_znx_big_normalize_tmp_bytes()
    }

    #[allow(clippy::too_many_arguments)]
    fn tensor_finish(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        offset: i64,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: VecZnxIdftApplyTmpA<Self> + VecZnxBigNormalize<Self>,
    {
        let (mut big, mut scratch) = scratch.borrow().take_vec_znx_big_scratch(a.n(), 1, a.size());
        module.vec_znx_idft_apply_tmpa(&mut big, 0, a, a_col);
        module.vec_znx_big_normalize(
            res,
            res_base2k,
            res_k,
            offset,
            res_col,
            &big.to_backend_ref(),
            a_base2k,
            0,
            &mut scratch,
        );
    }

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

#[cfg(feature = "enable-ifma")]
macro_rules! ifma_tensor_finish {
    ($executor:ty, $cap:expr) => {
        fn tensor_finish_tmp_bytes(module: &Module<Self>, _n: usize, _size: usize) -> usize {
            module.vec_znx_idft_normalize_consume_tmp_bytes(0, 0)
        }

        fn tensor_finish(
            module: &Module<Self>,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_base2k: usize,
            res_k: usize,
            offset: i64,
            res_col: usize,
            a: &mut VecZnxDftBackendMut<'_, Self>,
            a_col: usize,
            a_base2k: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let n = a.n();
            let workers = poulpy_hal::execution::scratch_workers_within::<$executor>(
                a.size().min($cap),
                3 * n * size_of::<u64>(),
                scratch.available().saturating_sub(3 * n * size_of::<i128>()),
            );
            let (tmp, arena) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), workers * 3 * n);
            let (carry, _) = crate::hal_impl::take_host_typed::<Self, i128>(arena, 3 * n);
            let shape = a.shape();
            let mut a = VecZnxDft::from_shape(&mut **a.data_mut(), shape);
            crate::ntt3x42_ifma::vec_znx_dft::idft_normalize_consume_ifma::<$executor>(
                module.reinterpret(),
                res,
                res_base2k,
                res_k,
                offset,
                res_col,
                &mut a,
                a_col,
                a_base2k,
                None,
                tmp,
                carry,
            );
        }
    };
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
        unsafe {
            crate::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512::<poulpy_hal::execution::SerialTaskExecutor>(
                module, res, cnv_offset, a, b, tmp,
            )
        };
    }
}

#[cfg(feature = "enable-rayon")]
impl RankOneTensorDft for NTT4x30Avx512Rayon {
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
        unsafe {
            crate::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512::<poulpy_cpu_rayon::RayonTaskExecutor>(
                module.reinterpret(),
                &mut crate::ntt4x30_avx512::rayon::base_dft_mut(res),
                cnv_offset,
                &crate::ntt4x30_avx512::rayon::base_cnv_l_ref(a),
                &crate::ntt4x30_avx512::rayon::base_cnv_r_ref(b),
                tmp,
            )
        };
    }
}

#[cfg(feature = "enable-ifma")]
impl RankOneTensorDft for NTT3x42Ifma {
    ifma_tensor_finish!(poulpy_hal::execution::SerialTaskExecutor, 1);
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
        unsafe {
            crate::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma::<poulpy_hal::execution::SerialTaskExecutor>(
                res, cnv_offset, a, b, tmp,
            )
        };
    }
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl RankOneTensorDft for NTT3x42IfmaRayon {
    ifma_tensor_finish!(
        crate::NTT3x42IfmaRayonExecutor,
        <Self as poulpy_hal::execution::ScratchWorkers>::IDFT
    );
    fn rank_one_tensor_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::APPLY)
            * crate::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma_tmp_bytes(res_size, a_size, b_size)
    }

    fn rank_one_tensor_dft(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        cnv_offset: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = crate::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::APPLY,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            crate::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma::<crate::NTT3x42IfmaRayonExecutor>(
                &mut crate::ntt3x42_ifma::rayon::base_dft_mut(res),
                cnv_offset,
                &crate::ntt3x42_ifma::rayon::base_cnv_l_ref(a),
                &crate::ntt3x42_ifma::rayon::base_cnv_r_ref(b),
                tmp,
            )
        };
    }
}

/// Enforces the Core degree contract before specialized kernels size scratch
/// from `module` and index the operands.
#[inline]
fn assert_degrees<BE: Backend>(module: &Module<BE>, degrees: [Degree; 3]) -> usize {
    let n: usize = degrees[0].as_usize();
    poulpy_hal::layouts::check_degree::<BE>(module.n(), n);
    for other in &degrees[1..] {
        assert_eq!(other.as_usize(), n, "operand degrees do not match each other");
    }
    n
}

fn rank_one_tensor_supported<R: GLWEInfos>(res: &R) -> bool {
    res.rank().as_usize() == 1 && matches!(res.n().as_usize(), 32768 | 65536)
}

fn rank_one_tensor_work_bytes<BE: RankOneTensorDft>(
    module: &Module<BE>,
    n: usize,
    res_size: usize,
    dft_size: usize,
    a_size: usize,
    b_size: usize,
) -> usize
where
    Module<BE>: VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
{
    let kernel = BE::rank_one_tensor_dft_tmp_bytes(dft_size, a_size, b_size);
    let normalize = BE::bytes_of_vec_znx(n, 1, res_size)
        + BE::tensor_finish_tmp_bytes(module, n, dft_size).max(module.vec_znx_normalize_tmp_bytes());
    BE::bytes_of_vec_znx(n, 2, res_size) + module.bytes_of_vec_znx_dft(n, 3, dft_size) + kernel.max(normalize)
}

fn rank_one_tensor_apply_tmp_bytes<BE, R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
where
    BE: RankOneTensorDft,
    Module<BE>: ModuleN
        + CnvPVecBytesOf
        + Convolution<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    B: GLWEInfos,
{
    let n = assert_degrees(module, [res.n(), a.n(), b.n()]);
    let base2k = a.base2k().as_usize();
    assert_eq!(b.base2k().as_usize(), base2k);
    let a_size = a.k().as_usize().div_ceil(base2k);
    let b_size = b.k().as_usize().div_ceil(base2k);
    let dft_size = (a_size + b_size).min((res.size() * res.base2k().as_usize() + base2k - 1).div_ceil(base2k));
    let prepared = module.bytes_of_cnv_pvec_left(n, 2, a_size, PrepareHint::Reuse)
        + module.bytes_of_cnv_pvec_right(n, 2, b_size, PrepareHint::Reuse);
    let prepare = module
        .cnv_prepare_left_tmp_bytes(a_size, a_size)
        .max(module.cnv_prepare_right_tmp_bytes(b_size, b_size));
    prepared + prepare.max(rank_one_tensor_work_bytes(module, n, res.size(), dft_size, a_size, b_size))
}

fn rank_one_tensor_square_tmp_bytes<BE, R, A>(module: &Module<BE>, res: &R, a: &A) -> usize
where
    BE: RankOneTensorDft,
    Module<BE>: ModuleN
        + CnvPVecBytesOf
        + Convolution<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
{
    let n = assert_degrees(module, [res.n(), a.n(), a.n()]);
    let base2k = a.base2k().as_usize();
    let a_size = a.k().as_usize().div_ceil(base2k);
    let dft_size = (2 * a_size).min((res.size() * res.base2k().as_usize() + base2k - 1).div_ceil(base2k));
    let prepared = module.bytes_of_cnv_pvec_left(n, 2, a_size, PrepareHint::Reuse)
        + module.bytes_of_cnv_pvec_right(n, 2, a_size, PrepareHint::Reuse);
    let prepare = module.cnv_prepare_self_tmp_bytes(a_size, a_size);
    prepared + prepare.max(rank_one_tensor_work_bytes(module, n, res.size(), dft_size, a_size, a_size))
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
        + VecZnxCopy<BE>
        + VecZnxSubAssign<BE>
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    AP: CnvPVecLToBackendRef<BE>,
    BP: CnvPVecRToBackendRef<BE>,
{
    let res_base2k = res.base2k().as_usize();
    let res_k = res.k().as_usize();
    let (cnv_offset_hi, cnv_offset_lo) = cnv_offset_to_limb_offset(cnv_offset, in_base2k);
    let dft_size = normalize_input_limb_bound_with_offset(
        a_size + b_size - cnv_offset_hi,
        res.size(),
        res_base2k,
        in_base2k,
        cnv_offset_lo,
    );
    let n: usize = res.n().as_usize();
    poulpy_hal::layouts::check_degree::<BE>(module.n(), n);
    let (mut tensor_dft, mut work) = scratch.borrow().take_vec_znx_dft_scratch(n, 3, dft_size);
    BE::rank_one_tensor_dft(
        module,
        &mut tensor_dft.to_backend_mut(),
        cnv_offset_hi,
        &a_prep.to_backend_ref(),
        &b_prep.to_backend_ref(),
        &mut work,
    );

    for (dft_col, res_col) in [(0, 0), (2, 2)] {
        BE::tensor_finish(
            module,
            res.to_backend_mut().data_mut(),
            res_base2k,
            res_k,
            cnv_offset_lo,
            res_col,
            &mut tensor_dft,
            dft_col,
            in_base2k,
            &mut work,
        );
    }

    let (mut pairwise, mut norm_scratch) = work.borrow().take_vec_znx_scratch(n, 1, res.size());
    BE::tensor_finish(
        module,
        &mut pairwise,
        res_base2k,
        res_k,
        cnv_offset_lo,
        0,
        &mut tensor_dft,
        1,
        in_base2k,
        &mut norm_scratch,
    );
    {
        let mut pairwise = pairwise.to_backend_mut();
        let res_ref = res.to_backend_ref();
        module.vec_znx_sub_assign(&mut pairwise, 0, res_ref.data(), 0);
        module.vec_znx_sub_assign(&mut pairwise, 0, res_ref.data(), 2);
    }
    module.vec_znx_normalize_assign(res_base2k, res_k, 0, &mut pairwise.to_backend_mut(), 0, &mut norm_scratch);
    module.vec_znx_copy(res.to_backend_mut().data_mut(), 1, &pairwise.to_backend_ref(), 0);
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
        + VecZnxCopy<BE>
        + VecZnxSubAssign<BE>
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    B: GLWEToBackendRef<BE> + GLWEInfos,
{
    let n = assert_degrees(module, [res.n(), a.n(), b.n()]);
    assert!(scratch.available() >= rank_one_tensor_apply_tmp_bytes(module, res, a, b));
    let base2k = a.base2k().as_usize();
    assert_eq!(b.base2k().as_usize(), base2k);
    let a_size = a.k().as_usize().div_ceil(base2k);
    let b_size = b.k().as_usize().div_ceil(base2k);
    assert!(a_size <= a.size());
    assert!(b_size <= b.size());
    let (mut a_prep, scratch) = scratch.borrow().take_cnv_pvec_left_scratch(n, 2, a_size, PrepareHint::Reuse);
    let (mut b_prep, mut scratch) = scratch.take_cnv_pvec_right_scratch(n, 2, b_size, PrepareHint::Reuse);
    {
        let mut prep_scratch = scratch.borrow();
        module.cnv_prepare_left(&mut a_prep, a.to_backend_ref().data(), &mut prep_scratch);
        module.cnv_prepare_right(&mut b_prep, b.to_backend_ref().data(), &mut prep_scratch);
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
        + VecZnxCopy<BE>
        + VecZnxSubAssign<BE>
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let n = assert_degrees(module, [res.n(), a.n(), a.n()]);
    assert!(scratch.available() >= rank_one_tensor_square_tmp_bytes(module, res, a));
    let base2k = a.base2k().as_usize();
    let a_size = a.k().as_usize().div_ceil(base2k);
    assert!(a_size <= a.size());
    let (mut a_prep, scratch) = scratch.borrow().take_cnv_pvec_left_scratch(n, 2, a_size, PrepareHint::Reuse);
    let (mut b_prep, mut scratch) = scratch.take_cnv_pvec_right_scratch(n, 2, a_size, PrepareHint::Reuse);
    {
        let mut prep_scratch = scratch.borrow();
        module.cnv_prepare_self(&mut a_prep, &mut b_prep, a.to_backend_ref().data(), &mut prep_scratch);
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
    ($be:ty, $consume:literal) => {
        unsafe impl GLWETensoringImpl for $be {
            fn glwe_tensor_apply_tmp_bytes<R, A, B>(module: &Module<$be>, res: &R, a: &A, b: &B) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
                B: GLWEInfos,
            {
                if rank_one_tensor_supported(res) {
                    rank_one_tensor_apply_tmp_bytes(module, res, a, b)
                } else {
                    module.glwe_tensor_apply_tmp_bytes_reference(res, a, b)
                }
            }

            fn glwe_tensor_square_apply_tmp_bytes<R, A>(module: &Module<$be>, res: &R, a: &A) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
            {
                if rank_one_tensor_supported(res) {
                    rank_one_tensor_square_tmp_bytes(module, res, a)
                } else {
                    module.glwe_tensor_square_apply_tmp_bytes_reference(res, a)
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
                if rank_one_tensor_supported(res) {
                    rank_one_tensor_apply(module, cnv_offset, res, a, b, scratch)
                } else {
                    module.glwe_tensor_apply_reference(cnv_offset, res, a, b, scratch)
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
                if rank_one_tensor_supported(res) {
                    rank_one_tensor_square(module, cnv_offset, res, a, scratch)
                } else {
                    module.glwe_tensor_square_apply_reference(cnv_offset, res, a, scratch)
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
                T: poulpy_core::layouts::GetTensorKey<$be>,
            {
                if !$consume {
                    return module.glwe_tensor_relinearize_reference(res, a, tsk, scratch);
                }
                let key = tsk.get_tensor_key(a.k()).unwrap_or_else(|e| panic!("{e}"));
                if a.base2k() != key.base2k() {
                    return module.glwe_tensor_relinearize_reference(res, a, tsk, scratch);
                }
                let n = assert_degrees(module, [res.n(), a.n(), key.n()]);
                assert_eq!(res.rank(), key.rank_out());
                assert_eq!(a.rank(), key.rank_out());
                let cols = key.rank_out().as_usize() + 1;
                let pairs = key.rank_in().as_usize();
                let a_size = a.k().div_ceil(key.base2k()) as usize;
                let output_size = gglwe_product_output_size::<$be, _, _, _>(res, a, &key);
                let res_base2k = res.base2k().as_usize();
                let res_k = res.k().as_usize();
                let (mut input, scratch) = scratch.borrow().take_vec_znx_dft_scratch(n, pairs, a_size);
                let (mut output, mut scratch) = scratch.take_vec_znx_dft_scratch(n, cols, output_size);
                let a = a.to_backend_ref();
                for i in 0..pairs {
                    module.vec_znx_dft_apply(1, 0, &mut input, i, a.data(), cols + i);
                }
                module.gglwe_product_dft_reference(
                    &mut output,
                    &input.to_backend_ref(),
                    &GGLWEPreparedToBackendRef::<$be>::to_backend_ref(&&key),
                    1,
                    &mut scratch,
                );
                let mut res = res.to_backend_mut();
                for i in 0..cols {
                    module.vec_znx_idft_normalize_consume(
                        res.data_mut(),
                        res_base2k,
                        res_k,
                        i,
                        &mut output,
                        i,
                        key.base2k().as_usize(),
                        Some((a.data(), i)),
                        &mut scratch,
                    );
                }
            }

            fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(module: &Module<$be>, res: &R, a: &A, tsk: &B) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
                B: GGLWEInfos,
            {
                if !$consume || a.base2k() != tsk.base2k() {
                    return module.glwe_tensor_relinearize_tmp_bytes_reference(res, a, tsk);
                }
                let n = assert_degrees(module, [res.n(), a.n(), tsk.n()]);
                let a_size = a.k().div_ceil(tsk.base2k()) as usize;
                let output_size = gglwe_product_output_size::<$be, _, _, _>(res, a, tsk);
                module.bytes_of_vec_znx_dft(n, tsk.rank_in().as_usize(), a_size)
                    + module.bytes_of_vec_znx_dft(n, tsk.rank_out().as_usize() + 1, output_size)
                    + module
                        .gglwe_product_dft_tmp_bytes_reference(output_size, a_size, tsk)
                        .max(module.vec_znx_idft_normalize_consume_tmp_bytes(res.size(), output_size))
            }
        }
    };
}

impl_rank_one_tensoring!(NTT4x30Avx512, false);
#[cfg(feature = "enable-rayon")]
impl_rank_one_tensoring!(NTT4x30Avx512Rayon, false);
#[cfg(feature = "enable-ifma")]
impl_rank_one_tensoring!(NTT3x42Ifma, true);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl_rank_one_tensoring!(NTT3x42IfmaRayon, true);

unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl for NTT4x30Avx512 {
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
        crate::ntt4x30_avx512::vmp::vmp_apply_digits_strided_tmp_bytes_avx(a_cols, a_size, dsize, pmat_rows, pmat_cols_in, 1)
    }

    fn gglwe_product_digits_strided(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        product_limbs: usize,
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
        crate::ntt4x30_avx512::vmp::vmp_apply_dft_to_dft_digits_strided_avx::<poulpy_hal::execution::SerialTaskExecutor>(
            module,
            res,
            a,
            dsize,
            product_limbs,
            pmat,
            tmp,
        );
    }
}

#[cfg(feature = "enable-ifma")]
unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl for NTT3x42Ifma {
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
        crate::ntt3x42_ifma::vmp::vmp_apply_digits_strided_tmp_bytes_ifma(a_cols, a_size, dsize, pmat_rows, pmat_cols_in, 1)
    }

    fn gglwe_product_digits_strided(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        product_limbs: usize,
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
        crate::ntt3x42_ifma::vmp::vmp_apply_dft_to_dft_digits_strided_ifma::<poulpy_hal::execution::SerialTaskExecutor>(
            module,
            res,
            a,
            dsize,
            product_limbs,
            pmat,
            tmp,
        );
    }
}

impl_glwe_automorphism_reference_full!(FFT64Avx512);
impl_glwe_automorphism_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_automorphism_reference_full!(NTT3x42Ifma);

impl_ggsw_automorphism_reference_full!(FFT64Avx512);
impl_ggsw_automorphism_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ggsw_automorphism_reference_full!(NTT3x42Ifma);

impl_gglwe_automorphism_reference_full!(FFT64Avx512);
impl_gglwe_automorphism_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_gglwe_automorphism_reference_full!(NTT3x42Ifma);

impl_decryption_reference_full!(FFT64Avx512);
impl_decryption_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_decryption_reference_full!(NTT3x42Ifma);

impl_glwe_trace_reference_full!(FFT64Avx512);
impl_glwe_trace_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_trace_reference_full!(NTT3x42Ifma);

impl_glwe_packing_reference_full!(FFT64Avx512);
impl_glwe_packing_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_packing_reference_full!(NTT3x42Ifma);

impl_conversion_reference_full!(FFT64Avx512);
impl_conversion_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_conversion_reference_full!(NTT3x42Ifma);

impl_glwe_keyswitch_reference_full!(FFT64Avx512);
impl_glwe_keyswitch_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_keyswitch_reference_full!(NTT3x42Ifma);

impl_gglwe_keyswitch_reference_full!(FFT64Avx512);
impl_gglwe_keyswitch_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_gglwe_keyswitch_reference_full!(NTT3x42Ifma);

impl_ggsw_keyswitch_reference_full!(FFT64Avx512);
impl_ggsw_keyswitch_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ggsw_keyswitch_reference_full!(NTT3x42Ifma);

impl_lwe_keyswitch_reference_full!(FFT64Avx512);
impl_lwe_keyswitch_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_lwe_keyswitch_reference_full!(NTT3x42Ifma);

impl_encryption_reference_full!(FFT64Avx512);
poulpy_cpu_ref::impl_sampling_host!(FFT64Avx512, fft64);
impl_encryption_reference_full!(NTT4x30Avx512);
poulpy_cpu_ref::impl_sampling_host!(NTT4x30Avx512, ntt4x30);
#[cfg(feature = "enable-ifma")]
impl_encryption_reference_full!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
poulpy_cpu_ref::impl_sampling_host!(NTT3x42Ifma, ntt4x30);

impl_glwe_external_product_reference_full!(FFT64Avx512);
impl_glwe_external_product_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_external_product_reference_full!(NTT3x42Ifma);

impl_gglwe_external_product_reference_full!(FFT64Avx512);
impl_gglwe_external_product_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_gglwe_external_product_reference_full!(NTT3x42Ifma);

impl_ggsw_external_product_reference_full!(FFT64Avx512);
impl_ggsw_external_product_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ggsw_external_product_reference_full!(NTT3x42Ifma);

impl_linear_transformation_reference_full!(FFT64Avx512);
impl_linear_transformation_reference_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_linear_transformation_reference_full!(NTT3x42Ifma);

#[cfg(feature = "enable-rayon")]
impl_glwe_automorphism_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ggsw_automorphism_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_automorphism_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_decryption_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_trace_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_packing_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_conversion_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_keyswitch_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_keyswitch_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ggsw_keyswitch_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_lwe_keyswitch_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_encryption_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_sampling_host!(FFT64Avx512Rayon, fft64);
#[cfg(feature = "enable-rayon")]
impl_glwe_external_product_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_external_product_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ggsw_external_product_reference_full!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_linear_transformation_reference_full!(FFT64Avx512Rayon);

#[cfg(feature = "enable-rayon")]
mod ntt4x30_rayon_defaults {
    use super::*;

    impl_glwe_automorphism_reference_full!(NTT4x30Avx512Rayon);
    impl_ggsw_automorphism_reference_full!(NTT4x30Avx512Rayon);
    impl_gglwe_automorphism_reference_full!(NTT4x30Avx512Rayon);
    impl_decryption_reference_full!(NTT4x30Avx512Rayon);
    impl_glwe_trace_reference_full!(NTT4x30Avx512Rayon);
    impl_glwe_packing_reference_full!(NTT4x30Avx512Rayon);
    impl_conversion_reference_full!(NTT4x30Avx512Rayon);
    impl_glwe_keyswitch_reference_full!(NTT4x30Avx512Rayon);
    impl_gglwe_keyswitch_reference_full!(NTT4x30Avx512Rayon);
    impl_ggsw_keyswitch_reference_full!(NTT4x30Avx512Rayon);
    impl_lwe_keyswitch_reference_full!(NTT4x30Avx512Rayon);
    impl_encryption_reference_full!(NTT4x30Avx512Rayon);
    poulpy_cpu_ref::impl_sampling_host!(NTT4x30Avx512Rayon, ntt4x30);
    impl_glwe_external_product_reference_full!(NTT4x30Avx512Rayon);
    impl_gglwe_external_product_reference_full!(NTT4x30Avx512Rayon);
    impl_ggsw_external_product_reference_full!(NTT4x30Avx512Rayon);
    impl_linear_transformation_reference_full!(NTT4x30Avx512Rayon);
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
mod ifma_rayon_defaults {
    use super::*;

    impl_glwe_automorphism_reference_full!(NTT3x42IfmaRayon);
    impl_ggsw_automorphism_reference_full!(NTT3x42IfmaRayon);
    impl_gglwe_automorphism_reference_full!(NTT3x42IfmaRayon);
    impl_decryption_reference_full!(NTT3x42IfmaRayon);
    impl_glwe_trace_reference_full!(NTT3x42IfmaRayon);
    impl_glwe_packing_reference_full!(NTT3x42IfmaRayon);
    impl_conversion_reference_full!(NTT3x42IfmaRayon);
    impl_glwe_keyswitch_reference_full!(NTT3x42IfmaRayon);
    impl_gglwe_keyswitch_reference_full!(NTT3x42IfmaRayon);
    impl_ggsw_keyswitch_reference_full!(NTT3x42IfmaRayon);
    impl_lwe_keyswitch_reference_full!(NTT3x42IfmaRayon);
    impl_encryption_reference_full!(NTT3x42IfmaRayon);
    poulpy_cpu_ref::impl_sampling_host!(NTT3x42IfmaRayon, ntt4x30);
    impl_glwe_external_product_reference_full!(NTT3x42IfmaRayon);
    impl_gglwe_external_product_reference_full!(NTT3x42IfmaRayon);
    impl_ggsw_external_product_reference_full!(NTT3x42IfmaRayon);
}

#[cfg(all(test, feature = "enable-ifma"))]
mod relinearize_tests {
    use super::*;
    use poulpy_core::{
        GLWETensoring,
        layouts::{GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, ModuleCoreAlloc},
    };
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
        layouts::{FillUniform, ScratchOwned},
        source::Source,
    };

    macro_rules! check_relinearize {
        ($name:ident, $be:ty) => {
            #[test]
            fn $name() {
                for (n, rank) in [(256usize, 1usize), (256, 2), (65536, 1)] {
                    let module = Module::<$be>::new(n as u64);
                    let mut source = Source::new([41; 32]);
                    for (base2k, key_base2k, dsize) in [(52usize, 52usize, 1usize), (52, 52, 4), (26, 52, 4)] {
                        let layout = GLWELayout {
                            n: n.into(),
                            base2k: base2k.into(),
                            k: 415usize.into(),
                            rank: rank.into(),
                        };
                        let key_layout = GLWETensorKeyLayout {
                            n: n.into(),
                            base2k: key_base2k.into(),
                            dsize: dsize.into(),
                            dnum: 8usize.div_ceil(dsize).into(),
                            k_aux: (key_base2k * dsize + n.ilog2() as usize).into(),
                            rank: rank.into(),
                        };
                        let mut input = module.glwe_tensor_alloc_from_infos(&layout);
                        input.fill_uniform(base2k, &mut source);
                        let mut key = module.glwe_tensor_key_alloc_from_infos(&key_layout);
                        key.fill_uniform(key_base2k, &mut source);
                        let mut prepared = module.alloc_tensor_key_prepared_from_infos(&key_layout);
                        let mut prep_scratch = ScratchOwned::<$be>::alloc(module.prepare_tensor_key_tmp_bytes(&key_layout));
                        module.prepare_tensor_key(&mut prepared, &key, &mut prep_scratch.borrow());
                        for (res_base2k, res_k) in [(52usize, 311usize), (26, 415)] {
                            let output_layout = GLWELayout {
                                base2k: res_base2k.into(),
                                k: res_k.into(),
                                ..layout
                            };
                            let mut got = module.glwe_alloc_from_infos(&output_layout);
                            let mut expected = module.glwe_alloc_from_infos(&output_layout);
                            let mut scratch =
                                ScratchOwned::<$be>::alloc(module.glwe_tensor_relinearize_tmp_bytes(&got, &input, &prepared));
                            let mut reference_scratch = ScratchOwned::<$be>::alloc(
                                module.glwe_tensor_relinearize_tmp_bytes_reference(&expected, &input, &prepared),
                            );
                            module.glwe_tensor_relinearize(&mut got, &input, &prepared, &mut scratch.borrow());
                            module.glwe_tensor_relinearize_reference(
                                &mut expected,
                                &input,
                                &prepared,
                                &mut reference_scratch.borrow(),
                            );
                            assert_eq!(
                                got, expected,
                                "n={n}, rank={rank}, base2k={base2k}, dsize={dsize}, res_base2k={res_base2k}"
                            );
                        }
                    }
                }
            }
        };
    }
    check_relinearize!(consume_relinearize_serial, NTT3x42Ifma);
    #[cfg(feature = "enable-rayon")]
    check_relinearize!(consume_relinearize_parallel, NTT3x42IfmaRayon);
}
