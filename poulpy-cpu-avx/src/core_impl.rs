use super::{FFT64Avx, FFT64CIAvx, NTT4x30Avx, NTT4x30CIAvx};
#[cfg(feature = "enable-rayon")]
use super::{FFT64AvxRayon, FFT64CIAvxRayon, NTT4x30AvxRayon, NTT4x30CIAvxRayon};
use poulpy_core::{impl_gglwe_product_digits_strided_reference, impl_glwe_tensoring_reference};
use poulpy_cpu_portable::reference::{
    ntt4x30::{
        NttDFTExecute,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
    },
    znx::ZnxAutomorphism,
};
use poulpy_hal::layouts::{Module, Ring, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef};

impl_glwe_tensoring_reference!(FFT64Avx);
impl_glwe_tensoring_reference!(NTT4x30Avx);
impl_glwe_tensoring_reference!(FFT64CIAvx);
impl_glwe_tensoring_reference!(NTT4x30CIAvx);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64CIAvxRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(NTT4x30CIAvxRayon);
impl_gglwe_product_digits_strided_reference!(FFT64Avx);
impl_gglwe_product_digits_strided_reference!(FFT64CIAvx);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64CIAvxRayon);

unsafe impl<R: Ring> poulpy_core::oep::GGLWEProductDigitsStridedImpl for NTT4x30Avx<R>
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
        super::ntt4x30::vmp::vmp_apply_digits_strided_tmp_bytes_avx(a_cols, a_size, dsize, pmat_rows, pmat_cols_in, 1)
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
        super::ntt4x30::vmp::vmp_apply_dft_to_dft_digits_strided_avx::<_, poulpy_hal::execution::SerialTaskExecutor>(
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

poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64Avx, fft64);
::poulpy_core::impl_conversion_reference_full!(super::FFT64Avx);
::poulpy_core::impl_glwe_packing_derived_full!(super::FFT64Avx);
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64Avx);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64Avx);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64Avx);
::poulpy_core::impl_glwe_trace_derived_full!(super::FFT64Avx);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64Avx);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30Avx, ntt4x30);
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30Avx);
::poulpy_core::impl_glwe_packing_derived_full!(super::NTT4x30Avx);
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30Avx);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30Avx);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30Avx);
::poulpy_core::impl_glwe_trace_derived_full!(super::NTT4x30Avx);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30Avx);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64CIAvx, fft64);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30CIAvx, ntt4x30);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64AvxRayon, fft64);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_conversion_reference_full!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_packing_derived_full!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_trace_derived_full!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30AvxRayon, ntt4x30);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_packing_derived_full!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_trace_derived_full!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64CIAvxRayon, fft64);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30CIAvxRayon, ntt4x30);
