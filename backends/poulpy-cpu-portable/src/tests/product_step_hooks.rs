//! Counting probes of the step hooks of `GGLWEProductDigitsStridedImpl` the linear
//! transformations reach: a transformation on powers-of-two rotations enters
//! `gglwe_products_from_dft` and, through the default of `gglwe_products_idft_normalize`,
//! `gglwe_product_idft_normalize`, on the exact budgets the queries return with the probes'
//! extra scratch, and matches `FFT64Portable`.

use crate::{FFT64Portable, hal_impl::delegating_backend::ProductStepsFFT64Portable as Probed};
use poulpy_core::{
    layouts::{GGLWEInfos, prepared::GGLWEPreparedBackendRef},
    reference::keyswitching::GGLWEProductDigitsStridedReference,
    test_suite::parity::{ParityShapes, test_linear_transformation_powers_of_two_parity},
};
use poulpy_hal::{
    layouts::{
        Backend, Module, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VmpPMatBackendRef,
    },
    test_suite::TestParams,
};
use std::cell::Cell;

/// Scratch each probe takes beside its reference.
const EXTRA: usize = 512;

thread_local! {
    static PRODUCTS_FROM_DFT: Cell<usize> = const { Cell::new(0) };
    static PRODUCT_IDFT_NORMALIZE: Cell<usize> = const { Cell::new(0) };
}

/// Counts a call in `counter`, then runs `body` on `scratch` past an `EXTRA` region whose
/// pattern it must leave intact.
fn probed(
    counter: &'static std::thread::LocalKey<Cell<usize>>,
    scratch: &mut ScratchArena<'_, Probed>,
    body: impl FnOnce(&mut ScratchArena<'_, Probed>),
) {
    counter.with(|calls| calls.set(calls.get() + 1));
    let (mut marker, mut rest) = scratch.borrow().take_region(EXTRA);
    Probed::copy_host_to_view(&mut marker, &[0x5A; EXTRA]);
    body(&mut rest);
    assert!(marker.iter().all(|&byte| byte == 0x5A), "the reference wrote past its share");
}

crate::impl_sampling_host!(Probed, fft64);
poulpy_core::impl_ggsw_conversion_reference_full!(Probed);
poulpy_core::impl_decryption_reference_full!(Probed);
poulpy_core::impl_encryption_reference_full!(Probed);
poulpy_core::impl_operations_reference_full!(Probed);
poulpy_core::impl_glwe_polynomial_evaluation_derived_full!(Probed);
poulpy_core::impl_preparation_reference_full!(Probed);
poulpy_core::impl_secret_reference_full!(Probed);
poulpy_core::impl_gglwe_external_product_derived_full!(Probed);
poulpy_core::impl_gglwe_keyswitch_derived_full!(Probed);
poulpy_core::impl_ggsw_external_product_derived_full!(Probed);
poulpy_core::impl_ggsw_keyswitch_derived_full!(Probed);
poulpy_core::impl_automorphism_reference_full!(Probed);
poulpy_core::impl_glwe_external_product_reference_full!(Probed);
poulpy_core::impl_glwe_keyswitch_reference_full!(Probed);
poulpy_core::impl_lwe_keyswitch_reference_full!(Probed);
poulpy_core::impl_glwe_tensoring_reference_full!(Probed);
poulpy_core::impl_glwe_trace_derived_full!(Probed);
poulpy_core::impl_glwe_linear_transformation_reference_full!(Probed);

unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl for Probed {
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
        module.gglwe_product_digits_strided_tmp_bytes_reference(
            res_size,
            a_cols,
            a_size,
            dsize,
            pmat_rows,
            pmat_cols_in,
            pmat_cols_out,
            pmat_size,
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
        module.gglwe_product_digits_strided_reference(res, a, dsize, product_limbs, pmat, scratch)
    }

    fn gglwe_product_fused_tmp_bytes<K>(
        module: &Module<Self>,
        res_size: usize,
        product_size: usize,
        a_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos,
    {
        EXTRA + module.gglwe_product_fused_tmp_bytes_reference(res_size, product_size, a_size, key_infos)
    }

    fn gglwe_product_fused(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        product_size: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        key: &GGLWEPreparedBackendRef<'_, Self>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, Self>, usize)>],
        p: i64,
        accumulate: bool,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        module.gglwe_product_fused_reference(res, product_size, a, key, term_count, addends, p, accumulate, scratch)
    }

    fn gglwe_products_from_dft_tmp_bytes<K>(
        module: &Module<Self>,
        count: usize,
        rank: usize,
        source_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos,
    {
        EXTRA + module.gglwe_products_from_dft_tmp_bytes_reference(count, rank, source_size, key_infos)
    }

    fn gglwe_products_from_dft(
        module: &Module<Self>,
        p: i64,
        term_count: usize,
        res: &mut [VecZnxDftBackendMut<'_, Self>],
        output_size: usize,
        accumulate: &[bool],
        sources: &mut [VecZnxDftBackendMut<'_, Self>],
        source_sizes: &[usize],
        source_base2k: usize,
        key: &GGLWEPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        probed(&PRODUCTS_FROM_DFT, scratch, |rest| {
            module.gglwe_products_from_dft_reference(
                p,
                term_count,
                res,
                output_size,
                accumulate,
                sources,
                source_sizes,
                source_base2k,
                key,
                rest,
            )
        });
    }

    fn gglwe_product_idft_normalize_tmp_bytes<K>(
        module: &Module<Self>,
        res_size: usize,
        product_size: usize,
        a_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos,
    {
        EXTRA + module.gglwe_product_idft_normalize_tmp_bytes_reference(res_size, product_size, a_size, key_infos)
    }

    fn gglwe_product_idft_normalize(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_dft: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        key: &GGLWEPreparedBackendRef<'_, Self>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, Self>, usize)>],
        body: Option<(&VecZnxBackendRef<'_, Self>, usize)>,
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        probed(&PRODUCT_IDFT_NORMALIZE, scratch, |rest| {
            module.gglwe_product_idft_normalize_reference(
                res, res_base2k, res_k, res_offset, res_dft, a, key, term_count, addends, body, a_base2k, rest,
            )
        });
    }
}

#[test]
fn powers_of_two_linear_transformation_enters_the_step_hooks() {
    let params = TestParams {
        size: 1 << 8,
        base2k: 12,
        n: 1 << 8,
    };
    let counters = [&PRODUCTS_FROM_DFT, &PRODUCT_IDFT_NORMALIZE];
    let before = counters.map(|counter| counter.get());
    test_linear_transformation_powers_of_two_parity::<FFT64Portable, Probed>(
        &params,
        &ParityShapes::default(),
        &Module::<FFT64Portable>::new(params.size as u64),
        &Module::<Probed>::new(params.size as u64),
    );
    let after = counters.map(|counter| counter.get());
    for ((hook, before), after) in ["gglwe_products_from_dft", "gglwe_product_idft_normalize"]
        .into_iter()
        .zip(before)
        .zip(after)
    {
        assert!(after > before, "the evaluation did not enter {hook}");
    }
}
