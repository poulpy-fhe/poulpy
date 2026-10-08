//! `poulpy-core` extension points of the portable Rayon backends.

use poulpy_core::{
    impl_gglwe_product_digits_strided_reference, impl_glwe_tensoring_reference,
    reference::keyswitching::glwe::{gglwe_product_digits_strided_reference, gglwe_product_digits_strided_tmp_bytes_reference},
};
use poulpy_cpu_portable::{
    NTT4x30Portable,
    ntt4x30::drivers::{STRIDED_MAX_DSIZE, gglwe_product_digits_strided_tmp_bytes, gglwe_product_digits_strided_with},
};
use poulpy_hal::{
    execution::ScratchWorkers,
    layouts::{
        DataView, DataViewMut, Module, ScratchArena, VecZnxDft, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMat,
        VmpPMatBackendRef,
    },
};

use super::{FFT64CIPortableRayon, FFT64PortableRayon, NTT4x30CIPortableRayon, NTT4x30PortableRayon};
use crate::RayonTaskExecutor;

impl_glwe_tensoring_reference!(FFT64PortableRayon);
impl_glwe_tensoring_reference!(NTT4x30PortableRayon);
impl_glwe_tensoring_reference!(FFT64CIPortableRayon);
impl_glwe_tensoring_reference!(NTT4x30CIPortableRayon);
impl_gglwe_product_digits_strided_reference!(FFT64PortableRayon);
impl_gglwe_product_digits_strided_reference!(FFT64CIPortableRayon);

/// Interleaved-digit product of the NTT backends over one ring, fused in one pass over the key up to
/// `STRIDED_MAX_DSIZE` digits. The helpers live in `$helpers`.
macro_rules! impl_digits_strided {
    ($ring:ty, $helpers:ident) => {
        pub(super) mod $helpers {
            use super::*;

            type Rayon = NTT4x30PortableRayon<$ring>;

            /// Scratch (in bytes) of the fused product: one worker's buffers for each worker of the pool.
            pub(in crate::portable) fn tmp_bytes(a_cols: usize, a_size: usize) -> usize {
                crate::workers(<Rayon as ScratchWorkers>::VMP) * gglwe_product_digits_strided_tmp_bytes(a_cols, a_size)
            }

            /// The fused product of the portable backend, on as many workers as `scratch` holds buffers for.
            #[allow(clippy::too_many_arguments)]
            pub(in crate::portable) fn apply(
                res: &mut VecZnxDftBackendMut<'_, Rayon>,
                a: &VecZnxDftBackendRef<'_, Rayon>,
                dsize: usize,
                product_limbs: usize,
                pmat: &VmpPMatBackendRef<'_, Rayon>,
                zero_prefix: Option<usize>,
                scratch: &mut ScratchArena<'_, Rayon>,
            ) {
                let per_worker = gglwe_product_digits_strided_tmp_bytes(a.cols(), a.size());
                let workers = crate::workers_within(<Rayon as ScratchWorkers>::VMP, per_worker, scratch.available());
                let (tmp, _) = crate::take_scratch::<Rayon, u32>(scratch.borrow(), workers * per_worker / size_of::<u32>());
                let res_shape = res.shape();
                gglwe_product_digits_strided_with::<$ring, RayonTaskExecutor>(
                    &mut VecZnxDft::<_, _, NTT4x30Portable<$ring>>::from_shape(&mut **res.data_mut(), res_shape),
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

        unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl for NTT4x30PortableRayon<$ring> {
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
                $helpers::tmp_bytes(a_cols, a_size)
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
                $helpers::apply(res, a, dsize, product_limbs, pmat, None, scratch);
            }
        }
    };
}

impl_digits_strided!(poulpy_hal::layouts::Standard, digits_strided);
impl_digits_strided!(poulpy_hal::layouts::ConjugateInvariant, digits_strided_ci);

poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64PortableRayon, fft64);
::poulpy_core::impl_conversion_reference_full!(super::FFT64PortableRayon);
::poulpy_core::impl_glwe_packing_derived_full!(super::FFT64PortableRayon);
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64PortableRayon);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64PortableRayon);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64PortableRayon);
::poulpy_core::impl_glwe_trace_derived_full!(super::FFT64PortableRayon);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64PortableRayon);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30PortableRayon, ntt4x30);
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_glwe_packing_derived_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_glwe_trace_derived_full!(super::NTT4x30PortableRayon);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30PortableRayon);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64CIPortableRayon, fft64);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30CIPortableRayon, ntt4x30);
