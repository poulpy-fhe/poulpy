//! `poulpy-core` extension points of the portable Rayon backends.

use poulpy_core::{
    impl_gglwe_product_digits_strided_reference_full, impl_glwe_tensoring_reference_full,
    reference::keyswitching::GGLWEProductDigitsStridedReference,
};
use poulpy_cpu_portable::{
    NTT4x30Portable,
    ntt4x30::drivers::{STRIDED_MAX_DSIZE, gglwe_product_digits_strided_tmp_bytes, gglwe_product_digits_strided_with},
};
use poulpy_hal::{
    execution::ScratchWorkers,
    layouts::{
        ConjugateInvariant, DataView, DataViewMut, Module, Ring, ScratchArena, Standard, VecZnxDft, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VmpPMat, VmpPMatBackendRef,
    },
};

use super::{FFT64PortableRayon, NTT4x30PortableRayon};
use crate::RayonTaskExecutor;

impl_glwe_tensoring_reference_full!(FFT64PortableRayon);
impl_glwe_tensoring_reference_full!(NTT4x30PortableRayon);
impl_gglwe_product_digits_strided_reference_full!(FFT64PortableRayon);

mod digits_strided {
    use super::*;

    type Rayon = NTT4x30PortableRayon;

    /// Scratch (in bytes) of the fused product: one worker's buffers for each worker of the pool.
    pub(super) fn tmp_bytes(a_cols: usize, a_size: usize) -> usize {
        crate::workers(<Rayon as ScratchWorkers>::VMP) * gglwe_product_digits_strided_tmp_bytes(a_cols, a_size)
    }

    /// The fused product of the portable backend, on as many workers as `scratch` holds buffers for.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply<Rg: Ring>(
        res: &mut VecZnxDftBackendMut<'_, Rayon, Rg>,
        a: &VecZnxDftBackendRef<'_, Rayon, Rg>,
        dsize: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Rayon, Rg>,
        zero_prefix: Option<usize>,
        scratch: &mut ScratchArena<'_, Rayon>,
    ) {
        let per_worker = gglwe_product_digits_strided_tmp_bytes(a.cols(), a.size());
        let workers = crate::workers_within(<Rayon as ScratchWorkers>::VMP, per_worker, scratch.available());
        let (tmp, _) = crate::take_scratch::<Rayon, u32>(scratch.borrow(), workers * per_worker / size_of::<u32>());
        let res_shape = res.shape();
        gglwe_product_digits_strided_with::<Rg, RayonTaskExecutor>(
            &mut VecZnxDft::<_, _, NTT4x30Portable, Rg>::from_shape(&mut **res.data_mut(), res_shape),
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
}

/// Interleaved-digit product of the NTT backends, fused in one pass over the key up to `STRIDED_MAX_DSIZE` digits.
macro_rules! impl_digits_strided {
    ($ring:ty) => {
        unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl<$ring> for NTT4x30PortableRayon {
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
                digits_strided::tmp_bytes(a_cols, a_size)
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
                digits_strided::apply(res, a, dsize, product_limbs, pmat, None, scratch);
            }

            ::poulpy_core::impl_gglwe_product_digits_strided_reference!($ring);
        }
    };
}

impl_digits_strided!(Standard);
impl_digits_strided!(ConjugateInvariant);

poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64PortableRayon, fft64);
::poulpy_core::impl_conversion_reference_full!(super::FFT64PortableRayon);
::poulpy_core::impl_lwe_keyswitch_reference_full!(super::FFT64PortableRayon);
::poulpy_core::impl_lwe_key_encryption_reference_full!(super::FFT64PortableRayon);
::poulpy_core::impl_lwe_key_preparation_derived_full!(super::FFT64PortableRayon);
::poulpy_core::impl_lwe_key_decompression_derived_full!(super::FFT64PortableRayon);
::poulpy_core::impl_glwe_pack_derived_full!(super::FFT64PortableRayon);
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64PortableRayon);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64PortableRayon);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64PortableRayon);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64PortableRayon);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30PortableRayon, ntt4x30);
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_lwe_keyswitch_reference_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_lwe_key_encryption_reference_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_lwe_key_preparation_derived_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_lwe_key_decompression_derived_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_glwe_pack_derived_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30PortableRayon);
