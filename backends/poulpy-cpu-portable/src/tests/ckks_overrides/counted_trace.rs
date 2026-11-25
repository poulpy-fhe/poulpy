//! The standard ring of [`DifferentSamplingFFT64Portable`]: the core families, its in-place trace counting its calls
//! and adding an extra its query reports, which the CKKS probes read.
use super::carry::with_marked_bytes;
use crate::hal_impl::delegating_backend::DifferentSamplingFFT64Portable;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GetAutomorphismKey, LWEInfos};
use poulpy_core::{GLWEAutomorphism, GLWEShift, GLWETrace};
use poulpy_hal::layouts::{Module, ScratchArena, Standard};
use std::cell::Cell;
use std::ops::Range;

thread_local! {
    pub(super) static TRACE_ASSIGN_CALLS: Cell<usize> = const { Cell::new(0) };
    pub(super) static TRACE_EXTRA: Cell<usize> = const { Cell::new(0) };
}

poulpy_core::impl_ggsw_conversion_reference_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_decompression_reference_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_decryption_reference_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_encryption_reference_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_operations_reference_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_glwe_polynomial_evaluation_derived_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_preparation_reference_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_secret_reference_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_gglwe_external_product_derived_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_gglwe_keyswitch_derived_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_ggsw_external_product_derived_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_ggsw_keyswitch_derived_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_automorphism_reference_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_glwe_external_product_reference_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_glwe_keyswitch_reference_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_glwe_linear_transformation_reference_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_glwe_tensoring_reference_full!(DifferentSamplingFFT64Portable, Standard);
poulpy_core::impl_gglwe_product_digits_strided_reference_full!(DifferentSamplingFFT64Portable, Standard);

// The derived body is crate-private to core, so the override runs its same-radix steps, the only ones the probes take.
unsafe impl poulpy_core::oep::GLWETraceImpl<Standard> for DifferentSamplingFFT64Portable {
    fn glwe_trace_assign_tmp_bytes<A, K>(module: &Module<Self>, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        TRACE_EXTRA.get()
            + GLWEShift::<Self, Standard>::glwe_shift_tmp_bytes(module, a_infos.size()).max(
                GLWEAutomorphism::<Self, Standard>::glwe_automorphism_tmp_bytes(module, a_infos, a_infos, key_infos),
            )
    }

    fn glwe_trace_assign<R, H>(
        module: &Module<Self>,
        res: &mut R,
        levels: Range<usize>,
        reverse: bool,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self, Standard> + GLWEInfos,
        H: GetAutomorphismKey<Self, Standard>,
    {
        TRACE_ASSIGN_CALLS.set(TRACE_ASSIGN_CALLS.get() + 1);
        with_marked_bytes(TRACE_EXTRA.get(), scratch, |scratch| {
            for p in GLWETrace::<Self, Standard>::glwe_trace_galois_elements(module, res.n(), levels, reverse) {
                let key = keys.get_automorphism_key(p, res.k()).unwrap();
                assert_eq!(res.base2k(), key.base2k(), "the counted trace runs at the keys' radix only");
                if normalize {
                    module.glwe_rsh_assign(1, res, scratch);
                }
                module.glwe_automorphism_add_assign(res, &key, scratch);
            }
        });
        res.set_noise(None);
    }
}
