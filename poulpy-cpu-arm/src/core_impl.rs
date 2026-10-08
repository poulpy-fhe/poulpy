use super::{FFT64CINeon, FFT64Neon, NTT4x30CINeon, NTT4x30Neon};
#[cfg(feature = "enable-rayon")]
use super::{FFT64CINeonRayon, FFT64NeonRayon, NTT4x30CINeonRayon, NTT4x30NeonRayon};
use poulpy_core::{
    impl_gglwe_product_digits_strided_reference, impl_glwe_tensoring_reference,
    reference::keyswitching::glwe::{gglwe_product_digits_strided_reference, gglwe_product_digits_strided_tmp_bytes_reference},
};
use poulpy_cpu_portable::kernels::{
    ntt4x30::{
        NttDFTExecute,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
    },
    znx::ZnxAutomorphism,
};
use poulpy_hal::layouts::{
    DataView, DataViewMut, Module, Ring, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef,
};

use crate::ntt4x30::vmp::{STRIDED_MAX_DSIZE, vmp_apply_dft_to_dft_digits_strided_neon, vmp_apply_digits_strided_tmp_bytes_neon};

impl_glwe_tensoring_reference!(FFT64Neon);
impl_glwe_tensoring_reference!(NTT4x30Neon);
impl_gglwe_product_digits_strided_reference!(FFT64Neon);
impl_glwe_tensoring_reference!(FFT64CINeon);
impl_glwe_tensoring_reference!(NTT4x30CINeon);
impl_gglwe_product_digits_strided_reference!(FFT64CINeon);

/// Interleaved-digit product hook of the NTT backends, fused up to `STRIDED_MAX_DSIZE` digits.
macro_rules! impl_ntt_digits_strided {
    ($be:ident, $executor:ty, $workers:expr) => {
        unsafe impl<R: Ring> poulpy_core::oep::GGLWEProductDigitsStridedImpl for $be<R>
        where
            NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
        {
            fn gglwe_product_digits_strided_tmp_bytes(
                module: &Module<Self>,
                res_size: usize,
                a_cols: usize,
                a_size: usize,
                dsize: usize,
                pmat_rows: usize,
                pmat_cols_in: usize,
                pmat_cols_out: usize,
                pmat_size: usize,
            ) -> usize {
                if dsize > STRIDED_MAX_DSIZE {
                    return gglwe_product_digits_strided_tmp_bytes_reference(
                        module,
                        res_size,
                        a_cols,
                        a_size,
                        dsize,
                        pmat_rows,
                        pmat_cols_in,
                        pmat_cols_out,
                        pmat_size,
                    );
                }
                $workers * vmp_apply_digits_strided_tmp_bytes_neon(a_cols, a_size)
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
                if dsize > STRIDED_MAX_DSIZE {
                    return gglwe_product_digits_strided_reference(module, res, a, dsize, product_limbs, pmat, scratch);
                }
                let per_worker = vmp_apply_digits_strided_tmp_bytes_neon(a.cols(), a.size());
                let workers = ($workers).min(scratch.available() / per_worker).max(1);
                let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), workers * per_worker / 8);
                let res_shape = res.shape();
                vmp_apply_dft_to_dft_digits_strided_neon::<R, $executor>(
                    &mut poulpy_hal::layouts::VecZnxDft::from_shape(&mut **res.data_mut(), res_shape),
                    &poulpy_hal::layouts::VecZnxDft::from_shape(&**a.data(), a.shape()),
                    dsize,
                    product_limbs,
                    &poulpy_hal::layouts::VmpPMat::from_data(
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

impl_ntt_digits_strided!(NTT4x30Neon, poulpy_hal::execution::SerialTaskExecutor, 1);
#[cfg(feature = "enable-rayon")]
impl_ntt_digits_strided!(
    NTT4x30NeonRayon,
    poulpy_cpu_rayon::RayonTaskExecutor,
    poulpy_cpu_rayon::workers(<NTT4x30NeonRayon<R> as poulpy_hal::execution::ScratchWorkers>::VMP)
);

#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(NTT4x30CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64CINeonRayon);

poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64Neon, fft64);
::poulpy_core::impl_conversion_reference_full!(super::FFT64Neon);
::poulpy_core::impl_glwe_packing_derived_full!(super::FFT64Neon);
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64Neon);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64Neon);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64Neon);
::poulpy_core::impl_glwe_trace_derived_full!(super::FFT64Neon);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64Neon);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30Neon, ntt4x30);
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30Neon);
::poulpy_core::impl_glwe_packing_derived_full!(super::NTT4x30Neon);
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30Neon);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30Neon);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30Neon);
::poulpy_core::impl_glwe_trace_derived_full!(super::NTT4x30Neon);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30Neon);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64CINeon, fft64);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30CINeon, ntt4x30);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64NeonRayon, fft64);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_conversion_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_packing_derived_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_trace_derived_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30NeonRayon, ntt4x30);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_packing_derived_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_trace_derived_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64CINeonRayon, fft64);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30CINeonRayon, ntt4x30);
