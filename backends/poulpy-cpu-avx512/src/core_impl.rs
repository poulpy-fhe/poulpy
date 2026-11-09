#[cfg(feature = "enable-ifma")]
use super::NTT3x42Ifma;
#[cfg(all(feature = "enable-rayon", feature = "enable-ifma"))]
use super::NTT3x42IfmaRayon;

use super::{FFT64Avx512, NTT4x30Avx512};
#[cfg(feature = "enable-rayon")]
use super::{FFT64Avx512Rayon, NTT4x30Avx512Rayon};
#[cfg(feature = "enable-ifma")]
use crate::ntt3x42_ifma::{
    primes::Primes42,
    tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv},
    traits::Ntt3x42IfmaDFTExecute,
};
use poulpy_core::{
    GLWEBytesOf, GLWENormalize, ScratchArenaTakeGLWE, cnv_offset_to_limb_offset, gglwe_gadget_dft_apply,
    gglwe_gadget_dft_apply_tmp_bytes, gglwe_gadget_input_size, gglwe_product_output_size,
    impl_gglwe_product_digits_strided_reference_full, impl_glwe_tensoring_reference_full,
    layouts::{Degree, GGLWEInfos, GGLWEPreparedToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
    normalize_input_limb_bound_with_offset,
    oep::GLWETensoringImpl,
    reference::keyswitching::glwe::GGLWEProductReference,
    reference::operations::GLWETensoringReference,
};
use poulpy_cpu_portable::kernels::{
    ntt4x30::{
        NttDFTExecute,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
    },
    znx::ZnxAutomorphism,
};
#[cfg(feature = "enable-rayon")]
use poulpy_hal::layouts::{CnvPVecL, CnvPVecR, DataView};
#[cfg(any(feature = "enable-ifma", feature = "enable-rayon"))]
use poulpy_hal::layouts::{DataViewMut, VecZnxDft};
use poulpy_hal::{
    api::{
        CnvOps, CnvPVecBytesOf, ModuleN, ScratchArenaTakeCnvPVecL, ScratchArenaTakeCnvPVecR, ScratchArenaTakeVecZnx,
        ScratchArenaTakeVecZnxBig, ScratchArenaTakeVecZnxDft, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopy, VecZnxDftBytesOf, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxIdftNormalize, VecZnxIdftNormalizeTmpBytes,
        VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxSubAssign,
    },
    layouts::{
        Backend, CnvPVecLBackendRef, CnvPVecLToBackendRef, CnvPVecRBackendRef, CnvPVecRToBackendRef, ConjugateInvariant, Module,
        PrepareHint, Ring, ScratchArena, Standard, VecZnxBackendMut, VecZnxBigToBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VmpPMatBackendRef, vec_znx_dft_backend_ref_from_mut,
    },
};

impl_glwe_tensoring_reference_full!(FFT64Avx512);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference_full!(FFT64Avx512Rayon);
impl_gglwe_product_digits_strided_reference_full!(FFT64Avx512);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference_full!(FFT64Avx512Rayon);

trait RankOneTensorDft<Rg: Ring = Standard>: Backend {
    fn tensor_finish_tmp_bytes(module: &Module<Self>, n: usize, size: usize) -> usize
    where
        Module<Self>: VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes + VecZnxIdftApplyTmpBytes,
    {
        module.vec_znx_big_bytes_of(n, 1, size)
            + module
                .vec_znx_big_normalize_tmp_bytes()
                .max(module.vec_znx_idft_apply_tmp_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    fn tensor_finish(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self, Rg>,
        res_base2k: usize,
        res_k: usize,
        offset: i64,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        a_col: usize,
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: VecZnxIdftApply<Self, Rg> + VecZnxIdftApplyTmpBytes + VecZnxBigNormalize<Self, Rg>,
    {
        let (mut big, mut scratch) = scratch.borrow().take_vec_znx_big_scratch(a.n(), 1, a.size());
        module.vec_znx_idft_apply(&mut big, 0, &vec_znx_dft_backend_ref_from_mut(a), a_col, &mut scratch);
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
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        cnv_offset: usize,
        a: &CnvPVecLBackendRef<'_, Self, Rg>,
        b: &CnvPVecRBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    );

    /// The tensor product with its cross term formed in place: `d0` and `d2` into columns 0 and 1 of `diag`,
    /// `d1` into column 0 of `cross`, within `rank_one_tensor_dft_tmp_bytes` of scratch.
    fn rank_one_cross_dft(
        module: &Module<Self>,
        diag: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        cross: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        cnv_offset: usize,
        a: &CnvPVecLBackendRef<'_, Self, Rg>,
        b: &CnvPVecRBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}

#[cfg(feature = "enable-ifma")]
macro_rules! ifma_tensor_finish {
    ($executor:ty, $cap:expr, $ring:ty) => {
        fn tensor_finish_tmp_bytes(module: &Module<Self>, _n: usize, _size: usize) -> usize {
            module.vec_znx_idft_normalize_tmp_bytes(0, 0)
        }

        fn tensor_finish(
            module: &Module<Self>,
            res: &mut VecZnxBackendMut<'_, Self, $ring>,
            res_base2k: usize,
            res_k: usize,
            offset: i64,
            res_col: usize,
            a: &mut VecZnxDftBackendMut<'_, Self, $ring>,
            a_col: usize,
            a_base2k: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let n = a.n();
            let workers = poulpy_hal::execution::scratch_workers_within::<$executor>(
                a.size().min($cap),
                3 * n * size_of::<u64>(),
                scratch.available(),
            );
            let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), workers * 3 * n);
            let shape = a.shape();
            let mut a = VecZnxDft::from_shape(&mut **a.data_mut(), shape);
            crate::ntt3x42_ifma::vec_znx_dft::idft_normalize_ifma::<$ring, $executor>(
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
            );
        }
    };
}

impl<Rg: Ring> RankOneTensorDft<Rg> for NTT4x30Avx512 {
    fn rank_one_tensor_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
        super::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512_tmp_bytes(res_size, a_size, b_size)
    }

    fn rank_one_tensor_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        cnv_offset: usize,
        a: &CnvPVecLBackendRef<'_, Self, Rg>,
        b: &CnvPVecRBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = <Self as RankOneTensorDft<Rg>>::rank_one_tensor_dft_tmp_bytes(res.size(), a.size(), b.size());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            super::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512::<_, poulpy_hal::execution::SerialTaskExecutor>(
                module, res, cnv_offset, a, b, tmp,
            )
        };
    }

    fn rank_one_cross_dft(
        module: &Module<Self>,
        diag: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        cross: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        cnv_offset: usize,
        a: &CnvPVecLBackendRef<'_, Self, Rg>,
        b: &CnvPVecRBackendRef<'_, Self, Rg>,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        unsafe {
            super::ntt4x30_avx512::convolution::cnv_tensor_rank1_cross_dft::<_, poulpy_hal::execution::SerialTaskExecutor>(
                module, diag, cross, cnv_offset, a, b,
            )
        };
    }
}

/// Rank-one tensor products of the Rayon NTT4x30 backend over one ring, on the kernels of the serial backend.
#[cfg(feature = "enable-rayon")]
macro_rules! impl_rank_one_tensor_dft_rayon {
    ($ring:ty) => {
        impl RankOneTensorDft<$ring> for NTT4x30Avx512Rayon {
            fn rank_one_tensor_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
                super::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512_tmp_bytes(res_size, a_size, b_size)
            }

            fn rank_one_tensor_dft(
                module: &Module<Self>,
                res: &mut VecZnxDftBackendMut<'_, Self, $ring>,
                cnv_offset: usize,
                a: &CnvPVecLBackendRef<'_, Self, $ring>,
                b: &CnvPVecRBackendRef<'_, Self, $ring>,
                scratch: &mut ScratchArena<'_, Self>,
            ) {
                let bytes = <Self as RankOneTensorDft<$ring>>::rank_one_tensor_dft_tmp_bytes(res.size(), a.size(), b.size());
                let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
                let res_shape = res.shape();
                unsafe {
                    super::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512::<$ring, poulpy_cpu_rayon::RayonTaskExecutor>(
                        module.reinterpret(),
                        &mut VecZnxDft::from_shape(&mut **res.data_mut(), res_shape),
                        cnv_offset,
                        &CnvPVecL::from_data(&**a.data(), a.n(), a.cols(), a.size(), a.hint()),
                        &CnvPVecR::from_data(&**b.data(), b.n(), b.cols(), b.size(), b.hint()),
                        tmp,
                    )
                };
            }

            fn rank_one_cross_dft(
                module: &Module<Self>,
                diag: &mut VecZnxDftBackendMut<'_, Self, $ring>,
                cross: &mut VecZnxDftBackendMut<'_, Self, $ring>,
                cnv_offset: usize,
                a: &CnvPVecLBackendRef<'_, Self, $ring>,
                b: &CnvPVecRBackendRef<'_, Self, $ring>,
                _scratch: &mut ScratchArena<'_, Self>,
            ) {
                let (diag_shape, cross_shape) = (diag.shape(), cross.shape());
                unsafe {
                    super::ntt4x30_avx512::convolution::cnv_tensor_rank1_cross_dft::<$ring, poulpy_cpu_rayon::RayonTaskExecutor>(
                        module.reinterpret(),
                        &mut VecZnxDft::from_shape(&mut **diag.data_mut(), diag_shape),
                        &mut VecZnxDft::from_shape(&mut **cross.data_mut(), cross_shape),
                        cnv_offset,
                        &CnvPVecL::from_data(&**a.data(), a.n(), a.cols(), a.size(), a.hint()),
                        &CnvPVecR::from_data(&**b.data(), b.n(), b.cols(), b.size(), b.hint()),
                    )
                };
            }
        }
    };
}

#[cfg(feature = "enable-rayon")]
impl_rank_one_tensor_dft_rayon!(Standard);
#[cfg(feature = "enable-rayon")]
impl_rank_one_tensor_dft_rayon!(ConjugateInvariant);

#[cfg(feature = "enable-ifma")]
impl<Rg: Ring> RankOneTensorDft<Rg> for NTT3x42Ifma
where
    Self: Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, Rg>> + Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, Rg>>,
    Module<Self>: VecZnxIdftNormalizeTmpBytes,
{
    ifma_tensor_finish!(poulpy_hal::execution::SerialTaskExecutor, 1, Rg);
    fn rank_one_tensor_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
        super::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma_tmp_bytes(res_size, a_size, b_size)
    }

    fn rank_one_tensor_dft(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        cnv_offset: usize,
        a: &CnvPVecLBackendRef<'_, Self, Rg>,
        b: &CnvPVecRBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = <Self as RankOneTensorDft<Rg>>::rank_one_tensor_dft_tmp_bytes(res.size(), a.size(), b.size());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            super::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma::<_, poulpy_hal::execution::SerialTaskExecutor>(
                res, cnv_offset, a, b, tmp,
            )
        };
    }

    fn rank_one_cross_dft(
        _module: &Module<Self>,
        diag: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        cross: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        cnv_offset: usize,
        a: &CnvPVecLBackendRef<'_, Self, Rg>,
        b: &CnvPVecRBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = <Self as RankOneTensorDft<Rg>>::rank_one_tensor_dft_tmp_bytes(diag.size(), a.size(), b.size());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            super::ntt3x42_ifma::convolution::cnv_tensor_rank1_cross_dft_ifma::<_, poulpy_hal::execution::SerialTaskExecutor>(
                diag, cross, cnv_offset, a, b, tmp,
            )
        };
    }
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl<Rg: Ring> RankOneTensorDft<Rg> for NTT3x42IfmaRayon
where
    NTT3x42Ifma: Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, Rg>> + Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, Rg>>,
    Module<Self>: VecZnxIdftNormalizeTmpBytes,
{
    ifma_tensor_finish!(
        crate::NTT3x42IfmaRayonExecutor,
        <Self as poulpy_hal::execution::ScratchWorkers>::IDFT,
        Rg
    );
    fn rank_one_tensor_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::APPLY)
            * super::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma_tmp_bytes(res_size, a_size, b_size)
    }

    fn rank_one_tensor_dft(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        cnv_offset: usize,
        a: &CnvPVecLBackendRef<'_, Self, Rg>,
        b: &CnvPVecRBackendRef<'_, Self, Rg>,
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
            super::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma::<_, crate::NTT3x42IfmaRayonExecutor>(
                &mut super::ntt3x42_ifma::rayon::base_dft_mut::<Rg>(res),
                cnv_offset,
                &super::ntt3x42_ifma::rayon::base_cnv_l_ref::<Rg>(a),
                &super::ntt3x42_ifma::rayon::base_cnv_r_ref::<Rg>(b),
                tmp,
            )
        };
    }

    fn rank_one_cross_dft(
        _module: &Module<Self>,
        diag: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        cross: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        cnv_offset: usize,
        a: &CnvPVecLBackendRef<'_, Self, Rg>,
        b: &CnvPVecRBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma_tmp_bytes(diag.size(), a.size(), b.size());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::APPLY,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            super::ntt3x42_ifma::convolution::cnv_tensor_rank1_cross_dft_ifma::<_, crate::NTT3x42IfmaRayonExecutor>(
                &mut super::ntt3x42_ifma::rayon::base_dft_mut::<Rg>(diag),
                &mut super::ntt3x42_ifma::rayon::base_dft_mut::<Rg>(cross),
                cnv_offset,
                &super::ntt3x42_ifma::rayon::base_cnv_l_ref::<Rg>(a),
                &super::ntt3x42_ifma::rayon::base_cnv_r_ref::<Rg>(b),
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

fn rank_one_tensor_work_bytes<BE: RankOneTensorDft<Rg>, Rg: Ring>(
    module: &Module<BE>,
    n: usize,
    res_size: usize,
    dft_size: usize,
    a_size: usize,
    b_size: usize,
) -> usize
where
    Module<BE>:
        VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes + VecZnxIdftApplyTmpBytes + VecZnxNormalizeTmpBytes,
{
    let kernel = BE::rank_one_tensor_dft_tmp_bytes(dft_size, a_size, b_size);
    let normalize = BE::vec_znx_bytes_of(n, 1, res_size)
        + BE::tensor_finish_tmp_bytes(module, n, dft_size).max(module.vec_znx_normalize_tmp_bytes());
    BE::vec_znx_bytes_of(n, 2, res_size) + module.vec_znx_dft_bytes_of(n, 3, dft_size) + kernel.max(normalize)
}

fn rank_one_tensor_apply_tmp_bytes<BE, R, A, B, Rg: Ring>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
where
    BE: RankOneTensorDft<Rg>,
    Module<BE>: ModuleN
        + CnvPVecBytesOf
        + CnvOps<BE, Rg>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeTmpBytes
        + GLWENormalize<BE, Rg>,
    R: GLWEInfos,
    A: GLWEInfos,
    B: GLWEInfos,
{
    let n = assert_degrees(module, [res.n(), a.n(), b.n()]);
    let base2k = a.base2k().as_usize();
    assert_eq!(b.base2k().as_usize(), base2k);
    let a_size = a.k().as_usize().div_ceil(base2k);
    let b_size = b.k().as_usize().div_ceil(base2k);
    let prepared = module.cnv_pvec_left_bytes_of(n, 2, a_size, PrepareHint::Reuse)
        + module.cnv_pvec_right_bytes_of(n, 2, b_size, PrepareHint::Reuse);
    let prepare = module
        .cnv_pvec_left_prepare_tmp_bytes(a_size, a_size)
        .max(module.cnv_pvec_right_prepare_tmp_bytes(b_size, b_size));
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(a))
        + BE::scratch_aligned(module.glwe_bytes_of_from_infos(b))
        + (prepared
            + prepare.max(rank_one_tensor_prepared_tmp_bytes::<BE, R, Rg>(
                module, res, a_size, b_size, base2k,
            )))
        .max(module.glwe_normalize_tmp_bytes())
}

/// Scratch of [`rank_one_tensor_apply_prepared`].
fn rank_one_tensor_prepared_tmp_bytes<BE, R, Rg: Ring>(
    module: &Module<BE>,
    res: &R,
    a_size: usize,
    b_size: usize,
    in_base2k: usize,
) -> usize
where
    BE: RankOneTensorDft<Rg>,
    Module<BE>:
        VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes + VecZnxIdftApplyTmpBytes + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
{
    let dft_size = (a_size + b_size).min((res.size() * res.base2k().as_usize() + in_base2k - 1).div_ceil(in_base2k));
    rank_one_tensor_work_bytes::<BE, Rg>(module, res.n().as_usize(), res.size(), dft_size, a_size, b_size)
}

/// Scratch the rank-one kernel of `glwe_mul_columns` adds to the reference one-pass products.
fn rank_one_columns_tmp_bytes<BE: RankOneTensorDft, R: GLWEInfos>(res: &R, a_size: usize, b_size: usize) -> usize {
    if rank_one_tensor_supported(res) {
        BE::rank_one_tensor_dft_tmp_bytes(a_size + b_size, a_size, b_size)
    } else {
        0
    }
}

fn rank_one_tensor_square_tmp_bytes<BE, R, A, Rg: Ring>(module: &Module<BE>, res: &R, a: &A) -> usize
where
    BE: RankOneTensorDft<Rg>,
    Module<BE>: ModuleN
        + CnvPVecBytesOf
        + CnvOps<BE, Rg>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeTmpBytes
        + GLWENormalize<BE, Rg>,
    R: GLWEInfos,
    A: GLWEInfos,
{
    let n = assert_degrees(module, [res.n(), a.n(), a.n()]);
    let base2k = a.base2k().as_usize();
    let a_size = a.k().as_usize().div_ceil(base2k);
    let dft_size = (2 * a_size).min((res.size() * res.base2k().as_usize() + base2k - 1).div_ceil(base2k));
    let prepared = module.cnv_pvec_left_bytes_of(n, 2, a_size, PrepareHint::Reuse)
        + module.cnv_pvec_right_bytes_of(n, 2, a_size, PrepareHint::Reuse);
    let prepare = module.cnv_pvec_self_prepare_tmp_bytes(a_size, a_size);
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(a))
        + (prepared
            + prepare.max(rank_one_tensor_work_bytes::<BE, Rg>(
                module,
                n,
                res.size(),
                dft_size,
                a_size,
                a_size,
            )))
        .max(module.glwe_normalize_tmp_bytes())
}

#[allow(clippy::too_many_arguments)]
fn rank_one_tensor_finish<BE, R, AP, BP, Rg: Ring>(
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
    BE: RankOneTensorDft<Rg>,
    Module<BE>: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApply<BE, Rg>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE, Rg>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopy<BE, Rg>
        + VecZnxSubAssign<BE, Rg>
        + VecZnxNormalizeAssign<BE, Rg>
        + VecZnxNormalizeTmpBytes,
    R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
    AP: CnvPVecLToBackendRef<BE, Rg>,
    BP: CnvPVecRToBackendRef<BE, Rg>,
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
fn rank_one_tensor_combine<BE, R, Rg: Ring>(
    module: &Module<BE>,
    res: &mut R,
    pairwise: &mut VecZnxBackendMut<'_, BE, Rg>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxCopy<BE, Rg> + VecZnxSubAssign<BE, Rg> + VecZnxNormalizeAssign<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
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
        &poulpy_hal::layouts::vec_znx_backend_ref_from_mut::<BE, Rg>(pairwise),
        0,
    );
    if k < res.size() * base2k {
        for col in [0, 2] {
            module.vec_znx_normalize_assign(base2k, k, 0, res.to_backend_mut().data_mut(), col, scratch);
        }
    }
}

fn rank_one_tensor_apply<BE, R, A, B, Rg: Ring>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    a: &A,
    b: &B,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: RankOneTensorDft<Rg>,
    Module<BE>: ModuleN
        + CnvPVecBytesOf
        + CnvOps<BE, Rg>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApply<BE, Rg>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE, Rg>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopy<BE, Rg>
        + VecZnxSubAssign<BE, Rg>
        + VecZnxNormalizeAssign<BE, Rg>
        + VecZnxNormalizeTmpBytes
        + GLWENormalize<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
    A: GLWEToBackendRef<BE, Rg> + GLWEInfos,
    B: GLWEToBackendRef<BE, Rg> + GLWEInfos,
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
        module.cnv_pvec_left_prepare(&mut a_prep, a.data(), &mut prep_scratch);
        module.cnv_pvec_right_prepare(&mut b_prep, b.data(), &mut prep_scratch);
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

#[allow(clippy::too_many_arguments)]
fn rank_one_tensor_apply_prepared<BE, R, AP, BP, Rg: Ring>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    a: &AP,
    a_size: usize,
    b: &BP,
    b_size: usize,
    in_base2k: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: RankOneTensorDft<Rg>,
    Module<BE>: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApply<BE, Rg>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE, Rg>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopy<BE, Rg>
        + VecZnxSubAssign<BE, Rg>
        + VecZnxNormalizeAssign<BE, Rg>
        + VecZnxNormalizeTmpBytes,
    R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
    AP: CnvPVecLToBackendRef<BE, Rg>,
    BP: CnvPVecRToBackendRef<BE, Rg>,
{
    {
        let n = res.n().as_usize();
        let (a, b) = (a.to_backend_ref(), b.to_backend_ref());
        assert!(
            (a.n(), a.cols(), a.size(), b.n(), b.cols(), b.size()) == (n, 2, a_size, n, 2, b_size),
            "prepared operands do not match res's degree and rank and the given sizes"
        );
    }
    assert!(scratch.available() >= rank_one_tensor_prepared_tmp_bytes::<BE, R, Rg>(module, res, a_size, b_size, in_base2k));
    // As the reference: the kernel writes every limb normalized.
    res.set_canonical(true);
    rank_one_tensor_finish(module, cnv_offset, res, a, b, a_size, b_size, in_base2k, scratch);
}

fn rank_one_tensor_square<BE, R, A, Rg: Ring>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    a: &A,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: RankOneTensorDft<Rg>,
    Module<BE>: ModuleN
        + CnvPVecBytesOf
        + CnvOps<BE, Rg>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApply<BE, Rg>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE, Rg>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopy<BE, Rg>
        + VecZnxSubAssign<BE, Rg>
        + VecZnxNormalizeAssign<BE, Rg>
        + VecZnxNormalizeTmpBytes
        + GLWENormalize<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
    A: GLWEToBackendRef<BE, Rg> + GLWEInfos,
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
        module.cnv_pvec_self_prepare(&mut a_prep, &mut b_prep, a.data(), &mut prep_scratch);
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
    ($be:ty, $ring:ty, $consume:literal, $prepared:path) => {
        unsafe impl GLWETensoringImpl<$ring> for $be {
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
                GLWETensoringReference::<$be, $ring>::glwe_tensor_apply_prepared_right_tmp_bytes_reference(
                    module, res, a, a_size, b_size,
                )
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
                R: GLWEToBackendMut<$be, $ring> + GLWEInfos,
                A: GLWEToBackendRef<$be, $ring> + GLWEInfos,
                BP: CnvPVecRToBackendRef<$be, $ring>,
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
                    rank_one_tensor_apply_tmp_bytes::<_, _, _, _, $ring>(module, res, a, b)
                } else {
                    GLWETensoringReference::<$be, $ring>::glwe_tensor_apply_tmp_bytes_reference(module, res, a, b)
                }
            }

            fn glwe_tensor_square_apply_tmp_bytes<R, A>(module: &Module<$be>, res: &R, a: &A) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
            {
                if rank_one_tensor_supported(res) {
                    rank_one_tensor_square_tmp_bytes::<_, _, _, $ring>(module, res, a)
                } else {
                    GLWETensoringReference::<$be, $ring>::glwe_tensor_square_apply_tmp_bytes_reference(module, res, a)
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
                R: GLWEToBackendMut<$be, $ring> + GLWEInfos,
                A: GLWEToBackendRef<$be, $ring> + GLWEInfos,
                B: GLWEToBackendRef<$be, $ring> + GLWEInfos,
            {
                if rank_one_tensor_supported(res) {
                    rank_one_tensor_apply(module, cnv_offset, res, a, b, scratch)
                } else {
                    GLWETensoringReference::<$be, $ring>::glwe_tensor_apply_reference(module, cnv_offset, res, a, b, scratch)
                }
            }

            fn glwe_tensor_square_apply<R, A>(
                module: &Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &A,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: GLWEToBackendMut<$be, $ring> + GLWEInfos,
                A: GLWEToBackendRef<$be, $ring> + GLWEInfos,
            {
                if rank_one_tensor_supported(res) {
                    rank_one_tensor_square(module, cnv_offset, res, a, scratch)
                } else {
                    GLWETensoringReference::<$be, $ring>::glwe_tensor_square_apply_reference(module, cnv_offset, res, a, scratch)
                }
            }

            fn glwe_tensor_relinearize<R, A, T>(
                module: &Module<$be>,
                res: &mut R,
                a: &A,
                tsk: &T,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: GLWEToBackendMut<$be, $ring> + GLWEInfos,
                A: GLWEToBackendRef<$be, $ring> + GLWEInfos,
                T: poulpy_core::layouts::GetTensorKey<$be, $ring>,
            {
                if !$consume {
                    return GLWETensoringReference::<$be, $ring>::glwe_tensor_relinearize_reference(module, res, a, tsk, scratch);
                }
                let key = tsk.get_tensor_key(a.k()).unwrap_or_else(|e| panic!("{e}"));
                if a.base2k() != key.base2k() {
                    return GLWETensoringReference::<$be, $ring>::glwe_tensor_relinearize_reference(module, res, a, tsk, scratch);
                }
                poulpy_core::layouts::assert_no_row_offset("glwe_tensor_relinearize", &key);
                let n = assert_degrees(module, [res.n(), a.n(), key.n()]);
                assert_eq!(res.rank(), key.rank_out());
                assert_eq!(a.rank(), key.rank_out());
                let cols = key.rank_out().as_usize() + 1;
                let pairs = key.rank_in().as_usize();
                let a_size = a.k().div_ceil(key.base2k()) as usize;
                let output_size = gglwe_product_output_size::<$be, _, _, _>(res, a, &key);
                let res_base2k = res.base2k().as_usize();
                let res_k = res.k().as_usize();
                let (base2k, k_digit, dnum) = (key.base2k(), key.k_digit(), key.dnum());
                let a_k = a.k();
                let operand_size = gglwe_gadget_input_size(base2k, k_digit, dnum, a_k, a_size);
                let (mut input, mut scratch) = scratch.borrow().take_vec_znx_dft_scratch(n, pairs, operand_size);
                let a = a.to_backend_ref();
                for i in 0..pairs {
                    gglwe_gadget_dft_apply(
                        module,
                        &mut input,
                        i,
                        a.data(),
                        cols + i,
                        base2k,
                        k_digit,
                        dnum,
                        a_k,
                        &mut scratch.borrow(),
                    );
                }
                let (mut output, mut scratch) = scratch.take_vec_znx_dft_scratch(n, cols, output_size);
                GGLWEProductReference::<$be, $ring>::gglwe_product_dft_reference(
                    module,
                    &mut output,
                    &input.to_backend_ref(),
                    &GGLWEPreparedToBackendRef::<$be, $ring>::to_backend_ref(&&key),
                    1,
                    &mut scratch,
                );
                res.set_canonical(true);
                let mut res = res.to_backend_mut();
                for i in 0..cols {
                    module.vec_znx_idft_normalize(
                        res.data_mut(),
                        res_base2k,
                        res_k,
                        0,
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
                    return GLWETensoringReference::<$be, $ring>::glwe_tensor_relinearize_tmp_bytes_reference(
                        module, res, a, tsk,
                    );
                }
                let n = assert_degrees(module, [res.n(), a.n(), tsk.n()]);
                let a_size = a.k().div_ceil(tsk.base2k()) as usize;
                let output_size = gglwe_product_output_size::<$be, _, _, _>(res, a, tsk);
                let (base2k, k_digit, dnum) = (tsk.base2k(), tsk.k_digit(), tsk.dnum());
                let operand_size = gglwe_gadget_input_size(base2k, k_digit, dnum, a.k(), a_size);
                // The digit layout runs before the output is taken, the product and
                // the fused inverse transform after.
                let layout = gglwe_gadget_dft_apply_tmp_bytes::<$be, _>(module, base2k, k_digit, dnum, a.k(), a_size);
                let product = module.vec_znx_dft_bytes_of(n, tsk.rank_out().as_usize() + 1, output_size)
                    + GGLWEProductReference::<$be, $ring>::gglwe_product_dft_tmp_bytes_reference(
                        module,
                        output_size,
                        operand_size,
                        tsk,
                    )
                    .max(module.vec_znx_idft_normalize_tmp_bytes(res.size(), output_size));
                module.vec_znx_dft_bytes_of(n, tsk.rank_in().as_usize(), operand_size) + layout.max(product)
            }

            fn glwe_mul_relinearize_tmp_bytes<R, B>(
                module: &Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                tensor_k: poulpy_core::layouts::TorusPrecision,
                tsk: &B,
            ) -> usize
            where
                R: GLWEInfos,
                B: GGLWEInfos,
            {
                // The kernel runs where the reference runs its convolutions: its scratch is added whole.
                GLWETensoringReference::<$be, $ring>::glwe_mul_relinearize_tmp_bytes_reference(
                    module, res, a_size, b_size, tensor_k, tsk,
                ) + rank_one_columns_tmp_bytes::<$be, _>(res, a_size, b_size)
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_relinearize<R, A, B, H>(
                module: &Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: poulpy_core::layouts::TorusPrecision,
                a: &A,
                b: &B,
                tsk: &H,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: poulpy_core::layouts::GLWEToBackendMut<$be, $ring> + poulpy_core::layouts::GLWEInfos,
                A: poulpy_core::layouts::GLWEToBackendRef<$be, $ring> + poulpy_core::layouts::GLWEInfos,
                B: poulpy_core::layouts::GLWEToBackendRef<$be, $ring> + poulpy_core::layouts::GLWEInfos,
                H: poulpy_core::layouts::GetTensorKey<$be, $ring>,
            {
                GLWETensoringReference::<$be, $ring>::glwe_mul_relinearize_reference(
                    module, cnv_offset, res, res_k, a, b, tsk, scratch,
                )
            }

            fn glwe_mul_relinearize_assign<R, A, H>(
                module: &Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: poulpy_core::layouts::TorusPrecision,
                a: &A,
                tsk: &H,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: poulpy_core::layouts::GLWEToBackendMut<$be, $ring> + poulpy_core::layouts::GLWEInfos,
                A: poulpy_core::layouts::GLWEToBackendRef<$be, $ring> + poulpy_core::layouts::GLWEInfos,
                H: poulpy_core::layouts::GetTensorKey<$be, $ring>,
            {
                GLWETensoringReference::<$be, $ring>::glwe_mul_relinearize_assign_reference(
                    module, cnv_offset, res, res_k, a, tsk, scratch,
                )
            }

            fn glwe_square_relinearize<R, A, H>(
                module: &Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: poulpy_core::layouts::TorusPrecision,
                a: &A,
                tsk: &H,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: poulpy_core::layouts::GLWEToBackendMut<$be, $ring> + poulpy_core::layouts::GLWEInfos,
                A: poulpy_core::layouts::GLWEToBackendRef<$be, $ring> + poulpy_core::layouts::GLWEInfos,
                H: poulpy_core::layouts::GetTensorKey<$be, $ring>,
            {
                GLWETensoringReference::<$be, $ring>::glwe_square_relinearize_reference(
                    module, cnv_offset, res, res_k, a, tsk, scratch,
                )
            }

            fn glwe_square_relinearize_assign<R, H>(
                module: &Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: poulpy_core::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: poulpy_core::layouts::GLWEToBackendMut<$be, $ring> + poulpy_core::layouts::GLWEInfos,
                H: poulpy_core::layouts::GetTensorKey<$be, $ring>,
            {
                GLWETensoringReference::<$be, $ring>::glwe_square_relinearize_assign_reference(
                    module, cnv_offset, res, res_k, tsk, scratch,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_prepared_relinearize<R, A, BP, H>(
                module: &Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: poulpy_core::layouts::TorusPrecision,
                a: &A,
                b: &BP,
                b_size: usize,
                b_k: poulpy_core::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: poulpy_core::layouts::GLWEToBackendMut<$be, $ring> + poulpy_core::layouts::GLWEInfos,
                A: poulpy_core::layouts::GLWEToBackendRef<$be, $ring> + poulpy_core::layouts::GLWEInfos,
                BP: CnvPVecRToBackendRef<$be, $ring>,
                H: poulpy_core::layouts::GetTensorKey<$be, $ring>,
            {
                GLWETensoringReference::<$be, $ring>::glwe_mul_prepared_relinearize_reference(
                    module, cnv_offset, res, res_k, a, b, b_size, b_k, tsk, scratch,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_prepared_relinearize_assign<R, AP, H>(
                module: &Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: poulpy_core::layouts::TorusPrecision,
                a: &AP,
                a_size: usize,
                a_k: poulpy_core::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: poulpy_core::layouts::GLWEToBackendMut<$be, $ring> + poulpy_core::layouts::GLWEInfos,
                AP: CnvPVecRToBackendRef<$be, $ring>,
                H: poulpy_core::layouts::GetTensorKey<$be, $ring>,
            {
                GLWETensoringReference::<$be, $ring>::glwe_mul_prepared_relinearize_assign_reference(
                    module, cnv_offset, res, res_k, a, a_size, a_k, tsk, scratch,
                )
            }

            fn glwe_mul_columns(
                module: &Module<$be>,
                cnv_offset_hi: usize,
                diag: &mut VecZnxDftBackendMut<'_, $be, $ring>,
                cross: &mut VecZnxDftBackendMut<'_, $be, $ring>,
                a: &CnvPVecLBackendRef<'_, $be, $ring>,
                b: &CnvPVecRBackendRef<'_, $be, $ring>,
                scratch: &mut ScratchArena<'_, $be>,
            ) {
                // The kernel reads two dense columns of the degree of the product.
                if diag.n() >= RANK_ONE_TENSOR_MIN_DEGREE && b.n() == diag.n() && b.cols() >= 2 {
                    <$be as RankOneTensorDft<$ring>>::rank_one_cross_dft(module, diag, cross, cnv_offset_hi, a, b, scratch)
                } else {
                    GLWETensoringReference::<$be, $ring>::glwe_mul_columns_reference(
                        module,
                        cnv_offset_hi,
                        diag,
                        cross,
                        a,
                        b,
                        scratch,
                    )
                }
            }

            fn glwe_square_operand_prepare(
                module: &::poulpy_hal::layouts::Module<$be>,
                left: &mut ::poulpy_hal::layouts::CnvPVecLBackendMut<'_, $be, $ring>,
                right: &mut ::poulpy_hal::layouts::CnvPVecRBackendMut<'_, $be, $ring>,
                a: &::poulpy_hal::layouts::VecZnxBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                GLWETensoringReference::<$be, $ring>::glwe_square_operand_prepare_reference(module, left, right, a, scratch)
            }

            fn glwe_mul_relinearize_sum_tmp_bytes<R, B>(
                module: &Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                tensor_k: poulpy_core::layouts::TorusPrecision,
                tsk: &B,
            ) -> usize
            where
                R: GLWEInfos,
                B: GGLWEInfos,
            {
                GLWETensoringReference::<$be, $ring>::glwe_mul_relinearize_sum_tmp_bytes_reference(
                    module, res, a_size, b_size, tensor_k, tsk,
                ) + rank_one_columns_tmp_bytes::<$be, _>(res, a_size, b_size)
            }

            fn glwe_mul_relinearize_sum<R, A, B, H>(
                module: &Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: poulpy_core::layouts::TorusPrecision,
                a: &[&A],
                b: &[&B],
                tsk: &H,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: GLWEToBackendMut<$be, $ring> + GLWEInfos,
                A: GLWEToBackendRef<$be, $ring> + GLWEInfos,
                B: GLWEToBackendRef<$be, $ring> + GLWEInfos,
                H: poulpy_core::layouts::GetTensorKey<$be, $ring>,
            {
                GLWETensoringReference::<$be, $ring>::glwe_mul_relinearize_sum_reference(
                    module, cnv_offset, res, res_k, a, b, tsk, scratch,
                )
            }

            fn glwe_mul_prepared_relinearize_sum_tmp_bytes<R, B>(
                module: &Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                tensor_k: poulpy_core::layouts::TorusPrecision,
                tsk: &B,
            ) -> usize
            where
                R: GLWEInfos,
                B: GGLWEInfos,
            {
                GLWETensoringReference::<$be, $ring>::glwe_mul_prepared_relinearize_sum_tmp_bytes_reference(
                    module, res, a_size, b_size, tensor_k, tsk,
                ) + rank_one_columns_tmp_bytes::<$be, _>(res, a_size, b_size)
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_prepared_relinearize_sum<'a, R, AP, BP, H>(
                module: &Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: poulpy_core::layouts::TorusPrecision,
                a: impl IntoIterator<Item = &'a AP, IntoIter: ExactSizeIterator + Clone>,
                a_size: usize,
                a_k: poulpy_core::layouts::TorusPrecision,
                b: impl IntoIterator<Item = &'a BP, IntoIter: ExactSizeIterator + Clone>,
                b_size: usize,
                b_k: poulpy_core::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: GLWEToBackendMut<$be, $ring> + GLWEInfos,
                AP: CnvPVecLToBackendRef<$be, $ring> + 'a,
                BP: CnvPVecRToBackendRef<$be, $ring> + 'a,
                H: poulpy_core::layouts::GetTensorKey<$be, $ring>,
            {
                GLWETensoringReference::<$be, $ring>::glwe_mul_prepared_relinearize_sum_reference(
                    module, cnv_offset, res, res_k, a, a_size, a_k, b, b_size, b_k, tsk, scratch,
                )
            }

            fn glwe_tensor_apply_prepared_sum_dft_tmp_bytes<R>(
                module: &Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
            ) -> usize
            where
                R: GLWEInfos,
            {
                GLWETensoringReference::<$be, $ring>::glwe_tensor_apply_prepared_sum_dft_tmp_bytes_reference(
                    module, res, a_size, b_size,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_tensor_apply_prepared_sum_dft<'a, AP>(
                module: &Module<$be>,
                cnv_offset_hi: usize,
                diag: &mut VecZnxDftBackendMut<'_, $be, $ring>,
                cross: &mut VecZnxDftBackendMut<'_, $be, $ring>,
                pairs: impl IntoIterator<Item = (&'a AP, poulpy_core::layouts::DiagonalOperand<'a, $be, $ring>)>,
                b_size: usize,
                b_k: poulpy_core::layouts::TorusPrecision,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                $be: 'a,
                AP: CnvPVecLToBackendRef<$be, $ring> + 'a,
            {
                GLWETensoringReference::<$be, $ring>::glwe_tensor_apply_prepared_sum_dft_reference(
                    module,
                    cnv_offset_hi,
                    diag,
                    cross,
                    pairs,
                    b_size,
                    b_k,
                    scratch,
                )
            }

            fn glwe_tensor_apply_prepared_tmp_bytes<R>(
                module: &Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                base2k: poulpy_core::layouts::Base2K,
            ) -> usize
            where
                R: GLWEInfos,
            {
                if rank_one_tensor_supported(res) {
                    rank_one_tensor_prepared_tmp_bytes::<_, _, $ring>(module, res, a_size, b_size, base2k.as_usize())
                } else {
                    GLWETensoringReference::<$be, $ring>::glwe_tensor_apply_prepared_tmp_bytes_reference(
                        module, res, a_size, b_size, base2k,
                    )
                }
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_tensor_apply_prepared<R, AP, BP>(
                module: &Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &AP,
                a_size: usize,
                b: &BP,
                b_size: usize,
                base2k: poulpy_core::layouts::Base2K,
                scratch: &mut ScratchArena<'_, $be>,
            ) where
                R: GLWEToBackendMut<$be, $ring> + GLWEInfos,
                AP: CnvPVecLToBackendRef<$be, $ring>,
                BP: CnvPVecRToBackendRef<$be, $ring>,
            {
                if rank_one_tensor_supported(res) {
                    rank_one_tensor_apply_prepared(
                        module,
                        cnv_offset,
                        res,
                        a,
                        a_size,
                        b,
                        b_size,
                        base2k.as_usize(),
                        scratch,
                    )
                } else {
                    GLWETensoringReference::<$be, $ring>::glwe_tensor_apply_prepared_reference(
                        module, cnv_offset, res, a, a_size, b, b_size, base2k, scratch,
                    )
                }
            }
        }
    };
}

impl_rank_one_tensoring!(
    NTT4x30Avx512,
    Standard,
    false,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);
impl_rank_one_tensoring!(
    NTT4x30Avx512,
    ConjugateInvariant,
    false,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);
#[cfg(feature = "enable-rayon")]
impl_rank_one_tensoring!(
    NTT4x30Avx512Rayon,
    Standard,
    false,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);
#[cfg(feature = "enable-rayon")]
impl_rank_one_tensoring!(
    NTT4x30Avx512Rayon,
    ConjugateInvariant,
    false,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);
#[cfg(feature = "enable-ifma")]
impl_rank_one_tensoring!(
    NTT3x42Ifma,
    Standard,
    true,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);
#[cfg(feature = "enable-ifma")]
impl_rank_one_tensoring!(
    NTT3x42Ifma,
    ConjugateInvariant,
    true,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl_rank_one_tensoring!(NTT3x42IfmaRayon, Standard, true, ifma_prepared_tensor);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl_rank_one_tensoring!(
    NTT3x42IfmaRayon,
    ConjugateInvariant,
    true,
    GLWETensoringReference::glwe_tensor_apply_prepared_right_reference
);

unsafe impl<R: Ring> poulpy_core::oep::GGLWEProductDigitsStridedImpl<R> for NTT4x30Avx512
where
    Self: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism<R>,
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
        res: &mut VecZnxDftBackendMut<'_, Self, R>,
        a: &VecZnxDftBackendRef<'_, Self, R>,
        dsize: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self, R>,
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

    ::poulpy_core::impl_gglwe_product_digits_strided_reference!(R);
}

#[cfg(feature = "enable-ifma")]
unsafe impl<R: Ring> poulpy_core::oep::GGLWEProductDigitsStridedImpl<R> for NTT3x42Ifma
where
    Self: Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, R>>
        + Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, R>>
        + ZnxAutomorphism<R>,
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
        res: &mut VecZnxDftBackendMut<'_, Self, R>,
        a: &VecZnxDftBackendRef<'_, Self, R>,
        dsize: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self, R>,
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

    ::poulpy_core::impl_gglwe_product_digits_strided_reference!(R);
}

poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64Avx512, fft64);
::poulpy_core::impl_conversion_reference_full!(super::FFT64Avx512);
::poulpy_core::impl_glwe_pack_derived_full!(super::FFT64Avx512);
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64Avx512);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64Avx512);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64Avx512);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64Avx512);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30Avx512, ntt4x30);
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30Avx512);
::poulpy_core::impl_glwe_pack_derived_full!(super::NTT4x30Avx512);
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30Avx512);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30Avx512);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30Avx512);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT3x42Ifma, ntt4x30);
#[cfg(feature = "enable-ifma")]
::poulpy_core::impl_conversion_reference_full!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_core::impl_glwe_pack_derived_full!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT3x42Ifma);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64Avx512Rayon, fft64);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_conversion_reference_full!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_pack_derived_full!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30Avx512Rayon, ntt4x30);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_pack_derived_full!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30Avx512Rayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
mod ifma_rayon_defaults {
    use super::super::NTT3x42IfmaRayon;
    use poulpy_hal::layouts::Standard;

    ::poulpy_core::impl_automorphism_reference_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_decompression_reference_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_decryption_reference_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_ggsw_conversion_reference_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_glwe_keyswitch_reference_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_gglwe_keyswitch_derived_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_ggsw_keyswitch_derived_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_lwe_keyswitch_reference_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_encryption_reference_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_glwe_external_product_reference_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_gglwe_external_product_derived_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_ggsw_external_product_derived_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_operations_reference_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_glwe_polynomial_evaluation_derived_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_preparation_reference_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_secret_reference_full!(NTT3x42IfmaRayon, Standard);
    ::poulpy_core::impl_glwe_trace_derived_full!(NTT3x42IfmaRayon, Standard);
    poulpy_cpu_portable::impl_sampling_host!(NTT3x42IfmaRayon, Standard, ntt4x30);
}
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_core::impl_conversion_reference_full!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_core::impl_glwe_pack_derived_full!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT3x42IfmaRayon, ConjugateInvariant, ntt4x30);
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
    R: GLWEToBackendMut<NTT3x42IfmaRayon, Standard> + GLWEInfos,
    A: GLWEToBackendRef<NTT3x42IfmaRayon, Standard> + GLWEInfos,
    BP: CnvPVecRToBackendRef<NTT3x42IfmaRayon, Standard>,
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
    // The prepared right operand may be embedded: its degree a power of two dividing the ring degree.
    poulpy_hal::layouts::check_degree::<BE>(n, b.to_backend_ref().n());
    assert!(a_size <= a.size(), "effective input exceeds its allocation");
    assert!(
        scratch.available()
            >= <Module<BE> as GLWETensoringReference<BE, Standard>>::glwe_tensor_apply_prepared_right_tmp_bytes_reference(
                module, res, a, a_size, b_size
            )
    );
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
    module.cnv_pvec_left_prepare(&mut left, a.data(), &mut work);
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
            1,
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
        api::GLWETensorKeyPreparedFactory,
        layouts::{GGLWEAtBackendMut, GLWEAlloc, GLWELayout, GLWETensorAlloc, GLWETensorKeyAlloc, GLWETensorKeyLayout},
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
                    // Whole-limb digits, then widths off the limb boundary, one under a limb and two above.
                    for (base2k, key_base2k, k_digit) in [
                        (52usize, 52usize, 52usize),
                        (52, 52, 208),
                        (26, 52, 208),
                        (52, 52, 49),
                        (52, 52, 105),
                        (26, 52, 159),
                    ] {
                        let layout = GLWELayout {
                            n: n.into(),
                            base2k: base2k.into(),
                            k: 415usize.into(),
                            rank: rank.into(),
                        };
                        let key_layout = GLWETensorKeyLayout {
                            n: n.into(),
                            base2k: key_base2k.into(),
                            k_digit: k_digit.into(),
                            dnum: (8 * key_base2k).div_ceil(k_digit).into(),
                            k_aux: (k_digit + n.ilog2() as usize).into(),
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
                        let mut prepared = module.glwe_tensor_key_prepared_alloc_from_infos(&key_layout);
                        let mut prep_scratch = ScratchOwned::<$be>::alloc(<Module<$be> as GLWETensorKeyPreparedFactory<
                            $be,
                            Standard,
                        >>::glwe_tensor_key_prepare_tmp_bytes(
                            &module, &key_layout
                        ));
                        module.glwe_tensor_key_prepare(&mut prepared, &key, &mut prep_scratch.borrow());
                        for (res_base2k, res_k) in [(52usize, 311usize), (26, 415)] {
                            let output_layout = GLWELayout {
                                base2k: res_base2k.into(),
                                k: res_k.into(),
                                ..layout
                            };
                            let mut got = module.glwe_alloc_from_infos(&output_layout);
                            let mut expected = module.glwe_alloc_from_infos(&output_layout);
                            let mut scratch = ScratchOwned::<$be>::alloc(
                                <Module<$be> as GLWETensoring<$be, Standard>>::glwe_tensor_relinearize_tmp_bytes(
                                    &module, &got, &input, &prepared,
                                ),
                            );
                            let mut reference_scratch = ScratchOwned::<$be>::alloc(<Module<$be> as GLWETensoringReference<
                                $be,
                                Standard,
                            >>::glwe_tensor_relinearize_tmp_bytes_reference(
                                &module, &expected, &input, &prepared
                            ));
                            module.glwe_tensor_relinearize(&mut got, &input, &prepared, &mut scratch.borrow());
                            module.glwe_tensor_relinearize_reference(
                                &mut expected,
                                &input,
                                &prepared,
                                &mut reference_scratch.borrow(),
                            );
                            assert_eq!(
                                got, expected,
                                "n={n}, rank={rank}, base2k={base2k}, k_digit={k_digit}, res_base2k={res_base2k}"
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
        GLWETensoring, glwe_right_operand_prepare,
        layouts::{GLWEAlloc, GLWELayout, GLWETensorAlloc, SetK},
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
            let prepare = CnvOps::<BE, Standard>::cnv_pvec_right_prepare_tmp_bytes(&module, b_size, b_size);
            let normalize = GLWENormalize::<BE, Standard>::glwe_normalize_tmp_bytes(&module);
            let prep_bytes = module.glwe_bytes_of_from_infos(&b) + prepare.max(normalize);
            let mut prep_scratch = ScratchOwned::<BE>::alloc(prep_bytes);
            glwe_right_operand_prepare(&module, &mut right, &b, b.k().as_usize(), &mut prep_scratch.borrow());
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
                    let bytes = GLWETensoring::<BE, Standard>::glwe_tensor_apply_prepared_right_tmp_bytes(
                        &module,
                        &got,
                        &a,
                        a.size(),
                        b_size,
                    );
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
                        assert_eq!(got, before, "short scratch mutated the output");
                    }
                    let before_a = a.clone();
                    let before_right = right.data().to_vec();
                    module.glwe_tensor_apply_prepared_right(offset, &mut got, &a, &right, b_size, &mut scratch.borrow());
                    assert_eq!(a, before_a);
                    assert!(right.data()[..] == before_right[..], "the prepared right operand changed");
                    module.glwe_tensor_apply_prepared_right_reference(
                        offset,
                        &mut expected,
                        &a,
                        &right,
                        b_size,
                        &mut scratch.borrow(),
                    );
                    assert_eq!(
                        got, expected,
                        "n={n}, rank={rank}, base={base}, lazy={lazy}, k={k}, offset={offset}"
                    );
                    let bytes = GLWETensoring::<BE, Standard>::glwe_tensor_apply_tmp_bytes(&module, &got, &a, &b);
                    let mut scratch = ScratchOwned::<BE>::alloc(bytes);
                    // Fresh garbage, so every limb the ordinary path fails to write differs from `expected`.
                    module.vec_znx_fill_uniform_source_all(base, k, got.data_mut(), &mut source);
                    module.glwe_tensor_apply(offset, &mut got, &a, &b, &mut scratch.borrow());
                    assert_eq!(got, expected, "ordinary/prepared n={n} k={k} offset={offset}");
                }
            }
        }
    }
}
