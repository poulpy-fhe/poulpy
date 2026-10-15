//! Rayon-scheduled wrapper for the AVX2 NTT4x30 backend.

use std::mem::size_of;

use poulpy_cpu_rayon::RayonTaskExecutor;
use poulpy_hal::layouts::{
    ConjugateInvariant, DataView, DataViewMut, Module, Ring, ScratchArena, Standard, VecZnxDft, VecZnxDftBackendMut,
    VecZnxDftBackendRef, VmpPMat, VmpPMatBackendRef,
};

use super::{NTT4x30Avx, NTT4x30AvxRayon, convolution, vec_znx_dft, vmp};

poulpy_cpu_rayon::impl_packed_ntt4x30_base_avx!(NTT4x30Avx, Standard);
poulpy_cpu_rayon::impl_packed_ntt4x30_base_avx!(NTT4x30Avx, ConjugateInvariant);

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
