use super::{FFT64Avx, NTT4x30Avx};
#[cfg(feature = "enable-rayon")]
use super::{FFT64AvxRayon, NTT4x30AvxRayon};
use poulpy_core::{impl_gglwe_product_digits_strided_reference, impl_glwe_tensoring_reference};
use poulpy_hal::layouts::{Module, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef};

impl_glwe_tensoring_reference!(FFT64Avx);
impl_glwe_tensoring_reference!(NTT4x30Avx);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(NTT4x30AvxRayon);
impl_gglwe_product_digits_strided_reference!(FFT64Avx);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64AvxRayon);

unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl for NTT4x30Avx {
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
        super::ntt4x30::vmp::vmp_apply_dft_to_dft_digits_strided_avx::<poulpy_hal::execution::SerialTaskExecutor>(
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

poulpy_cpu_ref::impl_cpu_core_defaults!(super::FFT64Avx, fft64);
poulpy_cpu_ref::impl_cpu_core_defaults!(super::NTT4x30Avx, ntt4x30);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_core_defaults!(super::FFT64AvxRayon, fft64);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_core_defaults!(super::NTT4x30AvxRayon, ntt4x30);
