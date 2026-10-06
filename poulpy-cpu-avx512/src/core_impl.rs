#[cfg(feature = "enable-ifma")]
use super::{NTT3x42CIIfma, NTT3x42Ifma};
#[cfg(all(feature = "enable-rayon", feature = "enable-ifma"))]
use super::{NTT3x42CIIfmaRayon, NTT3x42IfmaRayon};

use super::{FFT64Avx512, FFT64CIAvx512, NTT4x30Avx512, NTT4x30CIAvx512};
#[cfg(feature = "enable-rayon")]
use super::{FFT64Avx512Rayon, FFT64CIAvx512Rayon, NTT4x30Avx512Rayon, NTT4x30CIAvx512Rayon};
#[cfg(feature = "enable-ifma")]
use crate::ntt3x42_ifma::{
    primes::Primes42,
    tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv},
    traits::Ntt3x42IfmaDFTExecute,
};
use poulpy_core::{
    GLWEBytesOf, GLWENormalize, ScratchArenaTakeCore, impl_gglwe_product_digits_strided_reference, impl_glwe_tensoring_reference,
    layouts::{Degree, GGLWEInfos, GGLWEPreparedToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
    oep::GLWETensoringImpl,
    reference::keyswitching::glwe::{GGLWEProductReference, gglwe_product_output_size},
    reference::operations::{GLWETensoringReference, cnv_offset_to_limb_offset, normalize_input_limb_bound_with_offset},
};
use poulpy_cpu_portable::kernels::{
    ntt4x30::{
        NttDFTExecute,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
    },
    znx::ZnxAutomorphism,
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
        Backend, CnvPVecLBackendRef, CnvPVecLToBackendRef, CnvPVecRBackendRef, CnvPVecRToBackendRef, Module, PrepareHint, Ring,
        ScratchArena, VecZnxBackendMut, VecZnxBigToBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendMut,
        VecZnxDftToBackendRef, VmpPMatBackendRef,
    },
};

impl_glwe_tensoring_reference!(FFT64Avx512);
impl_glwe_tensoring_reference!(FFT64CIAvx512);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64CIAvx512Rayon);
impl_gglwe_product_digits_strided_reference!(FFT64Avx512);
impl_gglwe_product_digits_strided_reference!(FFT64CIAvx512);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64CIAvx512Rayon);

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
            crate::ntt3x42_ifma::vec_znx_dft::idft_normalize_consume_ifma::<R, $executor>(
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

impl<R: Ring> RankOneTensorDft for NTT4x30Avx512<R> {
    fn rank_one_tensor_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
        super::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512_tmp_bytes(res_size, a_size, b_size)
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
            super::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512::<_, poulpy_hal::execution::SerialTaskExecutor>(
                module, res, cnv_offset, a, b, tmp,
            )
        };
    }
}

#[cfg(feature = "enable-rayon")]
impl<R: Ring> RankOneTensorDft for NTT4x30Avx512Rayon<R> {
    fn rank_one_tensor_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
        super::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512_tmp_bytes(res_size, a_size, b_size)
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
            super::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512::<_, poulpy_cpu_rayon::RayonTaskExecutor>(
                module.reinterpret(),
                &mut super::ntt4x30_avx512::rayon::base_dft_mut::<R>(res),
                cnv_offset,
                &super::ntt4x30_avx512::rayon::base_cnv_l_ref::<R>(a),
                &super::ntt4x30_avx512::rayon::base_cnv_r_ref::<R>(b),
                tmp,
            )
        };
    }
}

#[cfg(feature = "enable-ifma")]
impl<R: Ring> RankOneTensorDft for NTT3x42Ifma<R>
where
    Self: Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, R>> + Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, R>>,
    Module<Self>: VecZnxIdftNormalizeConsumeTmpBytes,
{
    ifma_tensor_finish!(poulpy_hal::execution::SerialTaskExecutor, 1);
    fn rank_one_tensor_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
        super::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma_tmp_bytes(res_size, a_size, b_size)
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
            super::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma::<_, poulpy_hal::execution::SerialTaskExecutor>(
                res, cnv_offset, a, b, tmp,
            )
        };
    }
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl<R: Ring> RankOneTensorDft for NTT3x42IfmaRayon<R>
where
    NTT3x42Ifma<R>:
        Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, R>> + Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, R>>,
    Module<Self>: VecZnxIdftNormalizeConsumeTmpBytes,
{
    ifma_tensor_finish!(
        super::ntt3x42_ifma::NTT3x42IfmaRayonExecutor,
        <Self as poulpy_hal::execution::ScratchWorkers>::IDFT
    );
    fn rank_one_tensor_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::APPLY)
            * super::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma_tmp_bytes(res_size, a_size, b_size)
    }

    fn rank_one_tensor_dft(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        cnv_offset: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::APPLY,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            super::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma::<_, super::ntt3x42_ifma::NTT3x42IfmaRayonExecutor>(
                &mut super::ntt3x42_ifma::rayon::base_dft_mut::<R>(res),
                cnv_offset,
                &super::ntt3x42_ifma::rayon::base_cnv_l_ref::<R>(a),
                &super::ntt3x42_ifma::rayon::base_cnv_r_ref::<R>(b),
                tmp,
            )
        };
    }
}

/// Enforces the Core degree contract before specialized kernels size scratch
/// from `module` and index the operands.
#[inline]
fn assert_degrees<BE: Backend, const N: usize>(module: &Module<BE>, degrees: [Degree; N]) -> usize {
    let n: usize = degrees[0].as_usize();
    poulpy_hal::layouts::check_degree::<BE>(module.n(), n);
    for other in &degrees[1..] {
        assert_eq!(other.as_usize(), n, "operand degrees do not match each other");
    }
    n
}

/// Smallest ring degree where the rank-one tensor kernels, ordinary and
/// prepared, beat the reference composition: 1.08x to 1.5x from 2^13 up to the
/// 2^17 cap on 4 to 48 Rayon threads of a 24-core Zen 4, a tie at 2^12. Unit
/// tests lower it so the small-ring parity suites take the fast paths.
const RANK_ONE_TENSOR_MIN_DEGREE: usize = if cfg!(test) { 1 << 8 } else { 1 << 13 };

fn rank_one_tensor_supported<R: GLWEInfos>(res: &R) -> bool {
    res.rank().as_usize() == 1 && res.n().as_usize() >= RANK_ONE_TENSOR_MIN_DEGREE
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
        + VecZnxNormalizeTmpBytes
        + GLWENormalize<BE>,
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
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(a))
        + BE::scratch_aligned(module.glwe_bytes_of_from_infos(b))
        + (prepared + prepare.max(rank_one_tensor_work_bytes(module, n, res.size(), dft_size, a_size, b_size)))
            .max(module.glwe_normalize_tmp_bytes())
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
        + VecZnxNormalizeTmpBytes
        + GLWENormalize<BE>,
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
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(a))
        + (prepared + prepare.max(rank_one_tensor_work_bytes(module, n, res.size(), dft_size, a_size, a_size)))
            .max(module.glwe_normalize_tmp_bytes())
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
    let full_k = res.size() * res_base2k;
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
            full_k,
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
        full_k,
        cnv_offset_lo,
        0,
        &mut tensor_dft,
        1,
        in_base2k,
        &mut norm_scratch,
    );
    rank_one_tensor_combine(module, res, &mut pairwise, &mut norm_scratch);
}

/// Writes column 1 as the pairwise product minus both diagonals, then rounds
/// the tensor to `res.k()`. The diagonals arrive finished at full precision,
/// so they are rounded again only when `res.k()` drops limbs.
fn rank_one_tensor_combine<BE, R>(
    module: &Module<BE>,
    res: &mut R,
    pairwise: &mut VecZnxBackendMut<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxCopy<BE> + VecZnxSubAssign<BE> + VecZnxNormalizeAssign<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
{
    let base2k = res.base2k().as_usize();
    let k = res.k().as_usize();
    {
        let res_ref = res.to_backend_ref();
        module.vec_znx_sub_assign(pairwise, 0, res_ref.data(), 0);
        module.vec_znx_sub_assign(pairwise, 0, res_ref.data(), 2);
    }
    module.vec_znx_normalize_assign(base2k, k, 0, pairwise, 0, scratch);
    module.vec_znx_copy(
        res.to_backend_mut().data_mut(),
        1,
        &poulpy_hal::layouts::vec_znx_backend_ref_from_mut::<BE>(pairwise),
        0,
    );
    if k < res.size() * base2k {
        for col in [0, 2] {
            module.vec_znx_normalize_assign(base2k, k, 0, res.to_backend_mut().data_mut(), col, scratch);
        }
    }
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
        + VecZnxNormalizeTmpBytes
        + GLWENormalize<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    B: GLWEToBackendRef<BE> + GLWEInfos,
{
    let n = assert_degrees(module, [res.n(), a.n(), b.n()]);
    assert!(scratch.available() >= rank_one_tensor_apply_tmp_bytes(module, res, a, b));
    let (mut a_tmp, mut scratch) = scratch.borrow().take_glwe_scratch(a);
    let a = if a.is_canonical() {
        a.to_backend_ref()
    } else {
        module.glwe_normalize(&mut a_tmp, a, &mut scratch.borrow());
        a_tmp.to_backend_ref()
    };
    let (mut b_tmp, mut scratch) = scratch.take_glwe_scratch(b);
    let b = if b.is_canonical() {
        b.to_backend_ref()
    } else {
        module.glwe_normalize(&mut b_tmp, b, &mut scratch.borrow());
        b_tmp.to_backend_ref()
    };
    res.set_canonical(true);
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
        module.cnv_prepare_left(&mut a_prep, a.data(), &mut prep_scratch);
        module.cnv_prepare_right(&mut b_prep, b.data(), &mut prep_scratch);
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
        + VecZnxNormalizeTmpBytes
        + GLWENormalize<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let n = assert_degrees(module, [res.n(), a.n(), a.n()]);
    assert!(scratch.available() >= rank_one_tensor_square_tmp_bytes(module, res, a));
    let (mut a_tmp, mut scratch) = scratch.borrow().take_glwe_scratch(a);
    let a = if a.is_canonical() {
        a.to_backend_ref()
    } else {
        module.glwe_normalize(&mut a_tmp, a, &mut scratch.borrow());
        a_tmp.to_backend_ref()
    };
    res.set_canonical(true);
    let base2k = a.base2k().as_usize();
    let a_size = a.k().as_usize().div_ceil(base2k);
    assert!(a_size <= a.size());
    let (mut a_prep, scratch) = scratch.borrow().take_cnv_pvec_left_scratch(n, 2, a_size, PrepareHint::Reuse);
    let (mut b_prep, mut scratch) = scratch.take_cnv_pvec_right_scratch(n, 2, a_size, PrepareHint::Reuse);
    {
        let mut prep_scratch = scratch.borrow();
        module.cnv_prepare_self(&mut a_prep, &mut b_prep, a.data(), &mut prep_scratch);
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
    ($be:ty, $consume:literal, $prepared:path) => {
        unsafe impl GLWETensoringImpl for $be {
            fn glwe_tensor_apply_prepared_right_tmp_bytes<R, A>(
                module: &Module<$be>,
                res: &R,
                a: &A,
                a_size: usize,
                b_size: usize,
            ) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
            {
                module.glwe_tensor_apply_prepared_right_tmp_bytes_reference(res, a, a_size, b_size)
            }

            fn glwe_tensor_apply_prepared_right<R, A, BP>(
                module: &Module<$be>,
                offset: usize,
                res: &mut R,
                a: &A,
                b: &BP,
                b_size: usize,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: GLWEToBackendMut<$be> + GLWEInfos,
                A: GLWEToBackendRef<$be> + GLWEInfos,
                BP: CnvPVecRToBackendRef<$be>,
            {
                $prepared(module, offset, res, a, b, b_size, scratch)
            }

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
                res.set_canonical(true);
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

impl_rank_one_tensoring!(
    NTT4x30Avx512,
    false,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);
impl_rank_one_tensoring!(
    NTT4x30CIAvx512,
    false,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);
#[cfg(feature = "enable-rayon")]
impl_rank_one_tensoring!(
    NTT4x30Avx512Rayon,
    false,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);
#[cfg(feature = "enable-rayon")]
impl_rank_one_tensoring!(
    NTT4x30CIAvx512Rayon,
    false,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);
#[cfg(feature = "enable-ifma")]
impl_rank_one_tensoring!(
    NTT3x42Ifma,
    true,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);
#[cfg(feature = "enable-ifma")]
impl_rank_one_tensoring!(
    NTT3x42CIIfma,
    true,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl_rank_one_tensoring!(NTT3x42IfmaRayon, true, ifma_prepared_tensor);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl_rank_one_tensoring!(
    NTT3x42CIIfmaRayon,
    true,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);

unsafe impl<R: Ring> poulpy_core::oep::GGLWEProductDigitsStridedImpl for NTT4x30Avx512<R>
where
    Self: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
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
        super::ntt4x30_avx512::vmp::vmp_apply_digits_strided_tmp_bytes_avx(a_cols, a_size, dsize, pmat_rows, pmat_cols_in, 1)
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
        super::ntt4x30_avx512::vmp::vmp_apply_dft_to_dft_digits_strided_avx::<_, poulpy_hal::execution::SerialTaskExecutor>(
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
unsafe impl<R: Ring> poulpy_core::oep::GGLWEProductDigitsStridedImpl for NTT3x42Ifma<R>
where
    Self: Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, R>>
        + Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, R>>
        + ZnxAutomorphism,
{
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
        super::ntt3x42_ifma::vmp::vmp_apply_digits_strided_tmp_bytes_ifma(a_cols, a_size, dsize, pmat_rows, pmat_cols_in, 1)
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
        super::ntt3x42_ifma::vmp::vmp_apply_dft_to_dft_digits_strided_ifma::<_, poulpy_hal::execution::SerialTaskExecutor>(
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

poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64Avx512, fft64);
::poulpy_core::impl_conversion_reference_full!(super::FFT64Avx512);
::poulpy_core::impl_glwe_packing_derived_full!(super::FFT64Avx512);
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64Avx512);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64Avx512);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64Avx512);
::poulpy_core::impl_glwe_trace_derived_full!(super::FFT64Avx512);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64Avx512);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30Avx512, ntt4x30);
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30Avx512);
::poulpy_core::impl_glwe_packing_derived_full!(super::NTT4x30Avx512);
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30Avx512);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30Avx512);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30Avx512);
::poulpy_core::impl_glwe_trace_derived_full!(super::NTT4x30Avx512);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT3x42Ifma, ntt4x30);
#[cfg(feature = "enable-ifma")]
::poulpy_core::impl_conversion_reference_full!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_core::impl_glwe_packing_derived_full!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_core::impl_glwe_trace_derived_full!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT3x42Ifma);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64Avx512Rayon, fft64);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_conversion_reference_full!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_packing_derived_full!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_trace_derived_full!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30Avx512Rayon, ntt4x30);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_packing_derived_full!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_trace_derived_full!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30Avx512Rayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
mod ifma_rayon_defaults {
    use super::super::NTT3x42IfmaRayon;

    ::poulpy_core::impl_automorphism_reference_full!(NTT3x42IfmaRayon);
    ::poulpy_core::impl_decryption_reference_full!(NTT3x42IfmaRayon);
    ::poulpy_core::impl_ggsw_conversion_reference_full!(NTT3x42IfmaRayon);
    ::poulpy_core::impl_glwe_keyswitch_reference_full!(NTT3x42IfmaRayon);
    ::poulpy_core::impl_gglwe_keyswitch_derived_full!(NTT3x42IfmaRayon);
    ::poulpy_core::impl_ggsw_keyswitch_derived_full!(NTT3x42IfmaRayon);
    ::poulpy_core::impl_lwe_keyswitch_reference_full!(NTT3x42IfmaRayon);
    ::poulpy_core::impl_encryption_reference_full!(NTT3x42IfmaRayon);
    ::poulpy_core::impl_glwe_external_product_reference_full!(NTT3x42IfmaRayon);
    ::poulpy_core::impl_gglwe_external_product_derived_full!(NTT3x42IfmaRayon);
    ::poulpy_core::impl_ggsw_external_product_derived_full!(NTT3x42IfmaRayon);
    ::poulpy_core::impl_operations_reference_full!(NTT3x42IfmaRayon);
    ::poulpy_core::impl_polynomial_evaluation_derived_full!(NTT3x42IfmaRayon);
    poulpy_cpu_portable::impl_sampling_host!(NTT3x42IfmaRayon, ntt4x30);
}
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_core::impl_conversion_reference_full!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_core::impl_glwe_packing_derived_full!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_core::impl_glwe_trace_derived_full!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT3x42IfmaRayon);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64CIAvx512, fft64);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30CIAvx512, ntt4x30);
#[cfg(feature = "enable-ifma")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT3x42CIIfma, ntt4x30);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64CIAvx512Rayon, fft64);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30CIAvx512Rayon, ntt4x30);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT3x42CIIfmaRayon, ntt4x30);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
#[allow(clippy::too_many_arguments)]
fn ifma_prepared_tensor<R, A, BP>(
    module: &Module<NTT3x42IfmaRayon>,
    offset: usize,
    res: &mut R,
    a: &A,
    b: &BP,
    b_size: usize,
    scratch: &mut ScratchArena<'_, NTT3x42IfmaRayon>,
) where
    R: GLWEToBackendMut<NTT3x42IfmaRayon> + GLWEInfos,
    A: GLWEToBackendRef<NTT3x42IfmaRayon> + GLWEInfos,
    BP: CnvPVecRToBackendRef<NTT3x42IfmaRayon>,
{
    if !rank_one_tensor_supported(res) || a.rank().as_usize() != 1 || res.base2k() != a.base2k() {
        module.glwe_tensor_apply_prepared_right_reference(offset, res, a, b, b_size, scratch);
        return;
    }
    type BE = NTT3x42IfmaRayon;
    let n = res.n().as_usize();
    let base = a.base2k().as_usize();
    let a_size = a.k().as_usize().div_ceil(base);
    assert_degrees(module, [res.n(), a.n()]);
    // The prepared right operand may be sparse: a power of two dividing the ring degree.
    poulpy_hal::layouts::check_degree::<BE>(n, b.to_backend_ref().n());
    assert!(a_size <= a.size(), "effective input exceeds its allocation");
    assert!(scratch.available() >= module.glwe_tensor_apply_prepared_right_tmp_bytes_reference(res, a, a_size, b_size));
    let result_base = res.base2k().as_usize();
    // Preserve the prepared product's rounding before the pairwise subtraction.
    let full_k = res.size() * result_base;
    let (mut normalized, mut work) = scratch.borrow().take_glwe_scratch(a);
    let a = if a.is_canonical() {
        a.to_backend_ref()
    } else {
        module.glwe_normalize(&mut normalized, a, &mut work);
        normalized.to_backend_ref()
    };
    res.set_canonical(true);
    let (mut left, mut work) = work.take_cnv_pvec_left_scratch(n, 2, a_size, PrepareHint::Reuse);
    module.cnv_prepare_left(&mut left, a.data(), &mut work);
    let (high, low) = cnv_offset_to_limb_offset(offset, base);
    let size = normalize_input_limb_bound_with_offset(a_size + b_size - high, res.size(), result_base, base, low);
    let (mut dft, mut work) = work.take_vec_znx_dft_scratch(n, 1, size);
    for (input, output) in [(0, 0), (1, 2)] {
        module.cnv_apply_dft(
            high,
            &mut dft,
            0,
            &left.to_backend_ref(),
            input,
            &b.to_backend_ref(),
            input,
            &mut work,
        );
        BE::tensor_finish(
            module,
            res.to_backend_mut().data_mut(),
            result_base,
            full_k,
            low,
            output,
            &mut dft,
            0,
            base,
            &mut work,
        );
    }
    module.cnv_pairwise_apply_dft(
        high,
        &mut dft,
        0,
        &left.to_backend_ref(),
        &b.to_backend_ref(),
        0,
        1,
        &mut work,
    );
    let (mut pairwise, mut work) = work.take_vec_znx_scratch(n, 1, res.size());
    BE::tensor_finish(
        module,
        &mut pairwise,
        result_base,
        full_k,
        low,
        0,
        &mut dft,
        0,
        base,
        &mut work,
    );
    rank_one_tensor_combine(module, res, &mut pairwise, &mut work);
}

#[cfg(all(test, feature = "enable-ifma"))]
mod relinearize_tests {
    use super::*;
    use poulpy_core::{
        GLWETensoring,
        layouts::{GGLWEAtBackendMut, GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, ModuleCoreAlloc},
    };
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSource, VecZnxFillUniformSourceAll},
        layouts::ScratchOwned,
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
                        let input_k = input.size() * base2k;
                        module.vec_znx_fill_uniform_source_all(base2k, input_k, input.data_mut(), &mut source);
                        let mut key = module.glwe_tensor_key_alloc_from_infos(&key_layout);
                        for row in 0..key.dnum().as_usize() {
                            for col in 0..key.rank_in().as_usize() {
                                let mut view = GGLWEAtBackendMut::<$be>::at_backend_mut(&mut key, row, col);
                                let key_k = view.size() * key_base2k;
                                for out in 0..view.rank().as_usize() + 1 {
                                    module.vec_znx_fill_uniform_source(key_base2k, key_k, view.data_mut(), out, &mut source);
                                }
                            }
                        }
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
                            assert!(
                                got == expected,
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

#[cfg(all(test, feature = "enable-ifma", feature = "enable-rayon"))]
mod prepared_tensor_tests {
    use super::*;
    use poulpy_core::{
        GLWETensoring, glwe_prepare_right, glwe_tensor_apply_prepared_right,
        layouts::{GLWELayout, ModuleCoreAlloc, SetK},
    };
    use poulpy_hal::{
        api::{CnvPVecAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddAssign, VecZnxFillUniformSourceAll},
        layouts::{DataView, ScratchOwned},
        source::Source,
    };

    #[test]
    fn prepared_tensor_matches_reference() {
        type BE = NTT3x42IfmaRayon;
        let mut source = Source::new([67; 32]);
        for (n, rank, base) in [
            (65536usize, 1usize, 52usize),
            (65536, 1, 26),
            (131072, 1, 52),
            (32768, 1, 52),
            (256, 2, 26),
        ] {
            let module = Module::<BE>::new(n as u64);
            let layout = GLWELayout {
                n: n.into(),
                base2k: base.into(),
                k: (20 * base - 3).into(),
                rank: rank.into(),
            };
            let mut a = module.glwe_alloc_from_infos(&layout);
            let mut b = module.glwe_alloc_from_infos(&layout);
            b.set_k((17 * base - 1).into());
            let b_size = b.size();
            module.vec_znx_fill_uniform_source_all(base, a.k().as_usize(), a.data_mut(), &mut source);
            module.vec_znx_fill_uniform_source_all(base, b.k().as_usize(), b.data_mut(), &mut source);
            a.set_canonical(true);
            b.set_canonical(true);
            let mut right = module.cnv_pvec_right_alloc(n, rank + 1, b_size, PrepareHint::Reuse);
            let prep_bytes = module.glwe_bytes_of_from_infos(&b)
                + module
                    .cnv_prepare_right_tmp_bytes(b_size, b_size)
                    .max(module.glwe_normalize_tmp_bytes());
            let mut prep_scratch = ScratchOwned::<BE>::alloc(prep_bytes);
            glwe_prepare_right(&module, &mut right, &b, b.k().as_usize(), &mut prep_scratch.borrow());
            for lazy in [false, true] {
                if lazy {
                    for col in 0..rank + 1 {
                        module.vec_znx_add_assign(
                            GLWEToBackendMut::<BE>::to_backend_mut(&mut a).data_mut(),
                            col,
                            GLWEToBackendRef::<BE>::to_backend_ref(&b).data(),
                            col,
                        );
                    }
                    a.set_canonical(false);
                }
                for (k, offset) in [
                    (16 * base, 0),
                    (16 * base - 1, base - 1),
                    (15 * base + 1, base),
                    (16 * base + 3, base + 1),
                    (15 * base - 1, 2 * base),
                    (16 * base - 1, 23 * base + 7),
                ] {
                    // Above the fast-path threshold: one shape keeps the larger ring within the CI budget.
                    if n > 65536 && offset != base - 1 {
                        continue;
                    }
                    let output = GLWELayout { k: k.into(), ..layout };
                    let mut got = module.glwe_tensor_alloc_from_infos(&output);
                    let mut expected = module.glwe_tensor_alloc_from_infos(&output);
                    module.vec_znx_fill_uniform_source_all(base, k, got.data_mut(), &mut source);
                    let bytes = module.glwe_tensor_apply_prepared_right_tmp_bytes(&got, &a, a.size(), b_size);
                    let mut scratch = ScratchOwned::<BE>::alloc(bytes);
                    scratch.data.fill(0xa5);
                    if n == 65536 && k == 16 * base && !lazy {
                        let before = got.clone();
                        let (mut short, _) = scratch.borrow().split_at(bytes - 1);
                        assert!(
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                module.glwe_tensor_apply_prepared_right(offset, &mut got, &a, &right, b_size, &mut short);
                            }))
                            .is_err()
                        );
                        assert!(got == before, "short scratch mutated the output");
                    }
                    let before_a = a.clone();
                    let before_right = right.data().to_vec();
                    module.glwe_tensor_apply_prepared_right(offset, &mut got, &a, &right, b_size, &mut scratch.borrow());
                    assert!(a == before_a, "operand modified");
                    assert!(right.data()[..] == before_right[..], "prepared operand modified");
                    glwe_tensor_apply_prepared_right(&module, offset, &mut expected, &a, &right, b_size, &mut scratch.borrow());
                    assert!(
                        got == expected,
                        "n={n}, rank={rank}, base={base}, lazy={lazy}, k={k}, offset={offset}"
                    );
                    let bytes = module.glwe_tensor_apply_tmp_bytes(&got, &a, &b);
                    let mut scratch = ScratchOwned::<BE>::alloc(bytes);
                    // Fresh garbage, so every limb the ordinary path fails to write differs from `expected`.
                    module.vec_znx_fill_uniform_source_all(base, k, got.data_mut(), &mut source);
                    module.glwe_tensor_apply(offset, &mut got, &a, &b, &mut scratch.borrow());
                    assert!(got == expected, "ordinary/prepared n={n} k={k} offset={offset}");
                }
            }
        }
    }
}
