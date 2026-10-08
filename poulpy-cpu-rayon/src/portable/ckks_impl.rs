//! `poulpy-ckks` extension points of the portable Rayon backends.

use poulpy_ckks::{
    CKKSCtBounds, CKKSResult, SetCKKSInfos, impl_ckks_encapsulated_mod_up_reference, oep::CKKSEncapsulatedModUpImpl,
};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGLWEPreparedBackendRef};
use poulpy_cpu_portable::{
    ckks_encoding::{CKKSEncodingTransform, EncodingFFTTable},
    ckks_mod_up::{PackedModUp, encapsulated_mod_up, encapsulated_mod_up_tmp_bytes},
};
use poulpy_hal::layouts::{Module, Ring, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef};

use super::{FFT64PortableRayon, NTT4x30PortableRayon, core_impl::digits_strided};

impl_ckks_encapsulated_mod_up_reference!(FFT64PortableRayon);

// The portable backends encode every precision with the canonical table.
impl<R: Ring, F> CKKSEncodingTransform<F> for FFT64PortableRayon<R>
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = EncodingFFTTable<F>;
}

impl<R: Ring, F> CKKSEncodingTransform<F> for NTT4x30PortableRayon<R>
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = EncodingFFTTable<F>;
}

impl PackedModUp for NTT4x30PortableRayon {
    fn product_tmp_bytes(a_cols: usize, a_size: usize) -> usize {
        digits_strided::tmp_bytes(a_cols, a_size)
    }

    fn product_known_zero_prefix(
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        zero_prefix: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        digits_strided::apply(res, a, dsize, product_limbs, pmat, Some(zero_prefix), scratch);
    }
}

unsafe impl CKKSEncapsulatedModUpImpl for NTT4x30PortableRayon {
    fn ckks_encapsulated_mod_up_tmp_bytes<Dst, Src, D2S, S2D>(
        module: &Module<Self>,
        dst_infos: &Dst,
        src_infos: &Src,
        dense_to_sparse_infos: &D2S,
        sparse_to_dense_infos: &S2D,
    ) -> usize
    where
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        D2S: GGLWEInfos,
        S2D: GGLWEInfos,
    {
        encapsulated_mod_up_tmp_bytes(module, dst_infos, src_infos, dense_to_sparse_infos, sparse_to_dense_infos)
    }

    fn ckks_encapsulated_mod_up<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &mut Src,
        scale_up: usize,
        dense_to_sparse: &GGLWEPreparedBackendRef<'_, Self>,
        sparse_to_dense: &GGLWEPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
    {
        encapsulated_mod_up(module, dst, src, scale_up, dense_to_sparse, sparse_to_dense, scratch)
    }
}

poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::FFT64PortableRayon);
::poulpy_ckks::impl_ckks_conjugate_reference!(super::FFT64PortableRayon);
::poulpy_ckks::impl_ckks_imag_reference!(super::FFT64PortableRayon);
::poulpy_ckks::impl_ckks_bootstrapping_reference!(super::FFT64PortableRayon);
::poulpy_ckks::impl_ckks_fold_reference!(super::FFT64PortableRayon);
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference!(super::FFT64PortableRayon);
::poulpy_ckks::impl_ckks_dft_reference!(super::FFT64PortableRayon);
::poulpy_ckks::impl_ckks_eval_mod_reference!(super::FFT64PortableRayon);
poulpy_cpu_portable::impl_ckks_paco_coeff_encoding!(super::FFT64PortableRayon);
poulpy_cpu_portable::impl_ckks_ship_coeff_encoding!(super::FFT64PortableRayon);
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_ckks_conjugate_reference!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_ckks_imag_reference!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_ckks_bootstrapping_reference!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_ckks_fold_reference!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_ckks_dft_reference!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_ckks_eval_mod_reference!(super::NTT4x30PortableRayon);
poulpy_cpu_portable::impl_ckks_paco_coeff_encoding!(super::NTT4x30PortableRayon);
poulpy_cpu_portable::impl_ckks_ship_coeff_encoding!(super::NTT4x30PortableRayon);
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::FFT64CIPortableRayon);
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::NTT4x30CIPortableRayon);
