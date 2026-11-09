use super::{FFT64Neon, NTT4x30Neon};
#[cfg(feature = "enable-rayon")]
use super::{FFT64NeonRayon, NTT4x30NeonRayon};
use poulpy_core::{
    impl_gglwe_product_digits_strided_reference_full, impl_glwe_tensoring_reference_full,
    reference::keyswitching::GGLWEProductDigitsStridedReference,
};
use poulpy_hal::layouts::{
    ConjugateInvariant, DataView, DataViewMut, Module, ScratchArena, Standard, VecZnxDftBackendMut, VecZnxDftBackendRef,
    VmpPMatBackendRef,
};

use crate::ntt4x30::vmp::{STRIDED_MAX_DSIZE, vmp_apply_dft_to_dft_digits_strided_neon, vmp_apply_digits_strided_tmp_bytes_neon};

impl_glwe_tensoring_reference_full!(FFT64Neon);
impl_glwe_tensoring_reference_full!(NTT4x30Neon);
impl_gglwe_product_digits_strided_reference_full!(FFT64Neon);

/// Interleaved-digit product hook of the NTT backends, fused up to `STRIDED_MAX_DSIZE` digits.
macro_rules! impl_ntt_digits_strided {
    ($be:ident, $ring:ty, $executor:ty, $workers:expr) => {
        unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl<$ring> for $be {
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
                    return GGLWEProductDigitsStridedReference::<Self, $ring>::gglwe_product_digits_strided_tmp_bytes_reference(
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
                res: &mut VecZnxDftBackendMut<'_, Self, $ring>,
                a: &VecZnxDftBackendRef<'_, Self, $ring>,
                dsize: usize,
                product_limbs: usize,
                pmat: &VmpPMatBackendRef<'_, Self, $ring>,
                scratch: &mut ScratchArena<'_, Self>,
            ) {
                if dsize > STRIDED_MAX_DSIZE {
                    return module.gglwe_product_digits_strided_reference(res, a, dsize, product_limbs, pmat, scratch);
                }
                let per_worker = vmp_apply_digits_strided_tmp_bytes_neon(a.cols(), a.size());
                let workers = ($workers).min(scratch.available() / per_worker).max(1);
                let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), workers * per_worker / 8);
                let res_shape = res.shape();
                vmp_apply_dft_to_dft_digits_strided_neon::<$ring, $executor>(
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

            ::poulpy_core::impl_gglwe_product_digits_strided_reference!($ring);
        }
    };
}

impl_ntt_digits_strided!(NTT4x30Neon, Standard, poulpy_hal::execution::SerialTaskExecutor, 1);
impl_ntt_digits_strided!(NTT4x30Neon, ConjugateInvariant, poulpy_hal::execution::SerialTaskExecutor, 1);
#[cfg(feature = "enable-rayon")]
impl_ntt_digits_strided!(
    NTT4x30NeonRayon,
    Standard,
    poulpy_cpu_rayon::RayonTaskExecutor,
    poulpy_cpu_rayon::workers(<NTT4x30NeonRayon as poulpy_hal::execution::ScratchWorkers>::VMP)
);
#[cfg(feature = "enable-rayon")]
impl_ntt_digits_strided!(
    NTT4x30NeonRayon,
    ConjugateInvariant,
    poulpy_cpu_rayon::RayonTaskExecutor,
    poulpy_cpu_rayon::workers(<NTT4x30NeonRayon as poulpy_hal::execution::ScratchWorkers>::VMP)
);

#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference_full!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference_full!(NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference_full!(FFT64NeonRayon);

poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64Neon, fft64);
::poulpy_core::impl_conversion_reference_full!(super::FFT64Neon);
::poulpy_core::impl_glwe_pack_derived_full!(super::FFT64Neon);
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64Neon);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64Neon);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64Neon);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64Neon);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30Neon, ntt4x30);
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30Neon);
::poulpy_core::impl_glwe_pack_derived_full!(super::NTT4x30Neon);
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30Neon);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30Neon);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30Neon);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30Neon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64NeonRayon, fft64);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_conversion_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_pack_derived_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30NeonRayon, ntt4x30);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_pack_derived_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30NeonRayon);
