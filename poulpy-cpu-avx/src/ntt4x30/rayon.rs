//! Rayon-scheduled wrapper for the AVX2 NTT4x30 backend.

use std::mem::size_of;

use poulpy_cpu_portable::kernels::ntt4x30::mat_vec::BbcMeta;
use poulpy_cpu_rayon::{
    RayonTaskExecutor,
    ntt4x30::{PackedNtt4x30Base, PackedWord},
};
use poulpy_hal::{
    execution::TaskExecutor,
    layouts::{
        CnvDftAccTerm, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, ConjugateInvariant,
        DataView, DataViewMut, HostDataMut, HostDataRef, Module, Ring, ScratchArena, Standard, VecZnxBackendRef, VecZnxDft,
        VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMat, VmpPMatBackendRef,
    },
    oep::HalVecZnxDftImpl,
};

use super::{NTT4x30Avx, NTT4x30AvxRayon, convolution, vec_znx_dft, vmp};

/// `u64` words a limb takes per coefficient while it is transformed.
const LIMB_WORDS: usize = 4;

/// The drivers of the AVX2 backend over one ring.
macro_rules! impl_packed_base {
    ($ring:ty) => {
        #[allow(clippy::too_many_arguments)]
        impl PackedNtt4x30Base for NTT4x30Avx<$ring> {
            fn dft_tmp_words(n: usize) -> usize {
                LIMB_WORDS * n
            }

            fn idft_tmp_words(n: usize) -> usize {
                LIMB_WORDS * n
            }

            fn idft_tmpa_tmp_words(n: usize) -> usize {
                LIMB_WORDS * n
            }

            fn dft_limb<E: TaskExecutor>(module: &Module<Self>, n: usize, dst: &mut [u32], src: Option<&[i64]>, tmp: &mut [u64]) {
                vec_znx_dft::dft_limb(module, n, dst, src, tmp)
            }

            fn idft_limb<E: TaskExecutor>(module: &Module<Self>, n: usize, dst: &mut [i128], src: &[u32], tmp: &mut [u64]) {
                vec_znx_dft::idft_limb(module, n, dst, src, tmp)
            }

            fn idft_limb_tmpa<E: TaskExecutor>(
                module: &Module<Self>,
                n: usize,
                dst: &mut [i128],
                src: &mut [u32],
                tmp: &mut [u64],
            ) {
                vec_znx_dft::idft_limb(module, n, dst, src, tmp)
            }

            fn idft_limb_compact<E: TaskExecutor>(module: &Module<Self>, n: usize, slot: &mut [u32], tmp: &mut [u64]) {
                vec_znx_dft::idft_compact_limb(module, n, slot, tmp)
            }

            fn vmp_apply_tmp_bytes(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
                vmp::vmp_apply_tmp_bytes_avx(a_size, b_rows, b_cols_in)
            }

            fn vmp_apply_dft_to_dft<E: TaskExecutor>(
                module: &Module<Self>,
                res: &mut VecZnxDftBackendMut<'_, Self>,
                a: &VecZnxDftBackendRef<'_, Self>,
                pmat: &VmpPMatBackendRef<'_, Self>,
                limb_offset: usize,
                tmp: &mut [u64],
            ) {
                vmp::vmp_apply_dft_to_dft_avx::<$ring, E>(module, res, a, pmat, limb_offset, tmp)
            }

            fn vmp_apply_dft_to_dft_add<E: TaskExecutor>(
                module: &Module<Self>,
                res: &mut VecZnxDftBackendMut<'_, Self>,
                a: &VecZnxDftBackendRef<'_, Self>,
                pmat: &VmpPMatBackendRef<'_, Self>,
                limb_offset: usize,
                tmp: &mut [u64],
            ) {
                vmp::vmp_apply_dft_to_dft_add_avx::<$ring, E>(module, res, a, pmat, limb_offset, tmp)
            }

            fn vec_znx_dft_add<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
                b: &VecZnxDftBackendRef<'_, Self>,
                b_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_add::<$ring, E>(res, res_col, a, a_col, b, b_col)
            }

            fn vec_znx_dft_add_assign<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_add_assign::<$ring, E>(res, res_col, a, a_col)
            }

            fn vec_znx_dft_sub<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
                b: &VecZnxDftBackendRef<'_, Self>,
                b_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_sub::<$ring, E>(res, res_col, a, a_col, b, b_col)
            }

            fn vec_znx_dft_sub_assign<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_sub_assign::<$ring, E>(res, res_col, a, a_col)
            }

            fn vec_znx_dft_sub_negate_assign<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_sub_negate_assign::<$ring, E>(res, res_col, a, a_col)
            }

            fn vec_znx_dft_copy<E: TaskExecutor>(
                step: usize,
                offset: usize,
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_copy::<$ring, E>(step, offset, res, res_col, a, a_col)
            }

            fn vec_znx_dft_automorphism_add<E: TaskExecutor>(
                plan: &<Self as HalVecZnxDftImpl>::AutomorphismPlan,
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_automorphism_add::<$ring, E>(plan, res, res_col, a, a_col)
            }

            fn cnv_prepare_tmp_bytes(n: usize) -> usize {
                convolution::cnv_prepare_tmp_bytes(n)
            }

            fn cnv_prepare_left<BE: PackedWord, E: TaskExecutor>(
                base: &Module<Self>,
                module: &Module<BE>,
                res: &mut CnvPVecLBackendMut<'_, BE>,
                a: &VecZnxBackendRef<'_, BE>,
                tmp: &mut [u64],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                convolution::cnv_prepare_left::<BE, E>(module, res, a, tmp, |n, dst, src| {
                    vec_znx_dft::dft_limb_wide(base, n, dst, src)
                })
            }

            fn cnv_prepare_right<BE: PackedWord, E: TaskExecutor>(
                base: &Module<Self>,
                module: &Module<BE>,
                res: &mut CnvPVecRBackendMut<'_, BE>,
                a: &VecZnxBackendRef<'_, BE>,
                tmp: &mut [u64],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                convolution::cnv_prepare_right::<BE, E>(module, res, a, tmp, |n, dst, src| {
                    vec_znx_dft::dft_limb_wide(base, n, dst, src)
                })
            }

            fn cnv_prepare_self<BE: PackedWord, E: TaskExecutor>(
                base: &Module<Self>,
                module: &Module<BE>,
                left: &mut CnvPVecLBackendMut<'_, BE>,
                right: &mut CnvPVecRBackendMut<'_, BE>,
                a: &VecZnxBackendRef<'_, BE>,
                tmp: &mut [u64],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                convolution::cnv_prepare_self::<BE, E>(module, left, right, a, tmp, |n, dst, src| {
                    vec_znx_dft::dft_limb_wide(base, n, dst, src)
                })
            }

            fn cnv_apply_tmp_words(_res_size: usize) -> usize {
                0
            }

            fn cnv_apply_dft<BE: PackedWord, E: TaskExecutor>(
                module: &Module<BE>,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &CnvPVecLBackendRef<'_, BE>,
                a_col: usize,
                b: &CnvPVecRBackendRef<'_, BE>,
                b_col: usize,
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                let _ = tmp;
                unsafe {
                    convolution::cnv_apply_dft::<BE, E>(module, &BbcMeta::new(), cnv_offset, res, res_col, a, a_col, b, b_col)
                }
            }

            fn cnv_apply_dft_add<BE: PackedWord, E: TaskExecutor>(
                module: &Module<BE>,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &CnvPVecLBackendRef<'_, BE>,
                a_col: usize,
                b: &CnvPVecRBackendRef<'_, BE>,
                b_col: usize,
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                let _ = tmp;
                unsafe {
                    convolution::cnv_apply_dft_add::<BE, E>(module, &BbcMeta::new(), cnv_offset, res, res_col, a, a_col, b, b_col)
                }
            }

            fn cnv_apply_dft_sum<BE: PackedWord, E: TaskExecutor>(
                module: &Module<BE>,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                terms: &[CnvDftAccTerm<'_, BE>],
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                let _ = tmp;
                unsafe { convolution::cnv_apply_dft_sum_avx::<BE, E>(module, &BbcMeta::new(), cnv_offset, res, res_col, terms) }
            }

            fn cnv_pairwise_apply_dft<BE: PackedWord, E: TaskExecutor>(
                module: &Module<BE>,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &CnvPVecLBackendRef<'_, BE>,
                b: &CnvPVecRBackendRef<'_, BE>,
                i: usize,
                j: usize,
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                let _ = tmp;
                unsafe {
                    convolution::cnv_pairwise_apply_dft::<BE, E>(module, &BbcMeta::new(), cnv_offset, res, res_col, a, b, i, j)
                }
            }
        }
    };
}

impl_packed_base!(Standard);
impl_packed_base!(ConjugateInvariant);

mod standard {
    use poulpy_cpu_portable::kernels::znx::ZnxAutomorphismRotate;

    use super::{NTT4x30Avx, NTT4x30AvxRayon};

    poulpy_cpu_rayon::impl_ntt4x30_rayon_backend!(NTT4x30AvxRayon, NTT4x30Avx);

    unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for NTT4x30AvxRayon {
        poulpy_cpu_portable::hal_impl_vec_znx_monomial!();
    }

    unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for NTT4x30AvxRayon {
        poulpy_cpu_portable::hal_impl_vec_znx_ci!();
    }

    impl ZnxAutomorphismRotate for NTT4x30AvxRayon {
        #[inline(always)]
        fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
            <NTT4x30Avx as ZnxAutomorphismRotate>::znx_automorphism_rotate(p, k, res, a)
        }
    }
}

mod conjugate_invariant {
    use poulpy_hal::layouts::ConjugateInvariant;

    use super::{NTT4x30Avx, NTT4x30AvxRayon};

    poulpy_cpu_rayon::impl_ntt4x30_rayon_backend!(NTT4x30AvxRayon<ConjugateInvariant>, NTT4x30Avx<ConjugateInvariant>);
}

/// Interleaved-digit product of the Rayon backend over one ring, on per-worker scratch.
macro_rules! impl_digits_strided {
    ($ring:ty) => {
        unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl for NTT4x30AvxRayon<$ring> {
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
                vmp::vmp_apply_digits_strided_tmp_bytes_avx(
                    a_cols,
                    a_size,
                    dsize,
                    pmat_rows,
                    pmat_cols_in,
                    poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::VMP),
                )
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
                let metadata_bytes = 4 * dsize * size_of::<u64>();
                let per_worker =
                    vmp::vmp_apply_digits_strided_tmp_bytes_avx(a.cols(), a.size(), dsize, pmat.rows(), pmat.cols_in(), 1)
                        - metadata_bytes;
                let workers = poulpy_cpu_rayon::workers_within(
                    <Self as poulpy_hal::execution::ScratchWorkers>::VMP,
                    per_worker,
                    scratch.available().saturating_sub(metadata_bytes),
                );
                let bytes = metadata_bytes + workers * per_worker;
                let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
                let res_shape = res.shape();
                vmp::vmp_apply_dft_to_dft_digits_strided_avx::<$ring, RayonTaskExecutor>(
                    module.reinterpret(),
                    &mut VecZnxDft::from_shape(&mut **res.data_mut(), res_shape),
                    &VecZnxDft::from_shape(&**a.data(), a.shape()),
                    dsize,
                    product_limbs,
                    &VmpPMat::from_data(
                        &**pmat.data(),
                        pmat.n(),
                        pmat.rows(),
                        pmat.cols_in(),
                        pmat.cols_out(),
                        pmat.size(),
                        pmat.hint(),
                    ),
                    tmp,
                );
            }
        }
    };
}

impl_digits_strided!(Standard);
impl_digits_strided!(ConjugateInvariant);

#[cfg(feature = "enable-ckks")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_digits_strided_known_zero_prefix(
    module: &Module<NTT4x30AvxRayon>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30AvxRayon>,
    a: &VecZnxDftBackendRef<'_, NTT4x30AvxRayon>,
    dsize: usize,
    zero_prefix: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, NTT4x30AvxRayon>,
    scratch: &mut ScratchArena<'_, NTT4x30AvxRayon>,
) {
    let bytes = <NTT4x30AvxRayon as poulpy_core::oep::GGLWEProductDigitsStridedImpl>::gglwe_product_digits_strided_tmp_bytes(
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
    let (tmp, _) = crate::hal_impl::take_host_typed::<NTT4x30AvxRayon, u64>(scratch.borrow(), bytes / size_of::<u64>());
    let res_shape = res.shape();
    vmp::vmp_apply_dft_to_dft_digits_strided_avx_known_zero_prefix::<Standard, RayonTaskExecutor>(
        module.reinterpret(),
        &mut VecZnxDft::from_shape(&mut **res.data_mut(), res_shape),
        &VecZnxDft::from_shape(&**a.data(), a.shape()),
        dsize,
        product_limbs,
        &VmpPMat::from_data(
            &**pmat.data(),
            pmat.n(),
            pmat.rows(),
            pmat.cols_in(),
            pmat.cols_out(),
            pmat.size(),
            pmat.hint(),
        ),
        zero_prefix,
        tmp,
    );
}

impl<R: Ring> poulpy_hal::execution::ScratchWorkers for NTT4x30AvxRayon<R> {
    const PREPARE: usize = 4;
    const APPLY: usize = 8;
    const VMP: usize = 8;
    const IDFT: usize = 8;
}

impl<R: Ring> poulpy_cpu_rayon::RayonTuning for NTT4x30AvxRayon<R> {
    const COEFF_MIN_LEN: usize = 1 << 15;
    const COEFF_MIN_TASK: usize = 1 << 13;
    const NORMALIZE_MIN_TASK: usize = 1 << 12;
}

#[cfg(test)]
mod tests {
    use poulpy_hal::{layouts::Module, test_suite::convolution::test_convolution_by_const};

    use super::NTT4x30AvxRayon;

    #[test]
    fn convolution_by_const() {
        test_convolution_by_const(&Module::<NTT4x30AvxRayon>::new(1 << 8), 1 << 8, 50);
    }
}
