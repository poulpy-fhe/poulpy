use super::{FFT64CIPortable, FFT64Portable, NTT4x30CIPortable, NTT4x30Portable};
use poulpy_core::{
    impl_gglwe_product_digits_strided_reference_full, impl_glwe_tensoring_reference_full,
    reference::keyswitching::GGLWEProductDigitsStridedReference,
};
use poulpy_hal::{
    execution::SerialTaskExecutor,
    layouts::{Module, Ring, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef},
};

use crate::ntt4x30::{STRIDED_MAX_DSIZE, gglwe_product_digits_strided, gglwe_product_digits_strided_tmp_bytes};

impl_glwe_tensoring_reference_full!(FFT64Portable);
impl_glwe_tensoring_reference_full!(NTT4x30Portable);
impl_glwe_tensoring_reference_full!(FFT64CIPortable);
impl_glwe_tensoring_reference_full!(NTT4x30CIPortable);
impl_gglwe_product_digits_strided_reference_full!(FFT64Portable);
impl_gglwe_product_digits_strided_reference_full!(FFT64CIPortable);

/// Interleaved-digit product of the NTT backends, fused in one pass over the key up to `STRIDED_MAX_DSIZE` digits.
unsafe impl<R: Ring> poulpy_core::oep::GGLWEProductDigitsStridedImpl for NTT4x30Portable<R>
where
    Self: crate::kernels::ntt4x30::NttDFTExecute<
            crate::kernels::ntt4x30::ntt::NttTable<crate::kernels::ntt4x30::primes::Primes30, R>,
        > + crate::kernels::ntt4x30::NttDFTExecute<
            crate::kernels::ntt4x30::ntt::NttTableInv<crate::kernels::ntt4x30::primes::Primes30, R>,
        > + crate::kernels::znx::ZnxAutomorphism,
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
            return module.gglwe_product_digits_strided_tmp_bytes_reference(
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
        gglwe_product_digits_strided_tmp_bytes(a_cols, a_size)
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
            return module.gglwe_product_digits_strided_reference(res, a, dsize, product_limbs, pmat, scratch);
        }
        gglwe_product_digits_strided::<R, SerialTaskExecutor>(res, a, dsize, product_limbs, pmat, None, scratch);
    }

    ::poulpy_core::impl_gglwe_product_digits_strided_reference!();
}

crate::impl_cpu_core_defaults!(super::FFT64Portable, fft64);
::poulpy_core::impl_conversion_reference_full!(super::FFT64Portable);
::poulpy_core::impl_glwe_pack_derived_full!(super::FFT64Portable);
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64Portable);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64Portable);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64Portable);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64Portable);
crate::impl_cpu_core_defaults!(super::NTT4x30Portable, ntt4x30);
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30Portable);
::poulpy_core::impl_glwe_pack_derived_full!(super::NTT4x30Portable);
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30Portable);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30Portable);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30Portable);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30Portable);
crate::impl_cpu_core_defaults!(super::FFT64CIPortable, fft64);
crate::impl_cpu_core_defaults!(super::NTT4x30CIPortable, ntt4x30);
