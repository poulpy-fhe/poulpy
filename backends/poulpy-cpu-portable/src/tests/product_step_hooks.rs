//! Counting probes of the step hooks of `GGLWEProductDigitsStridedImpl` the linear
//! transformations reach: a transformation on powers-of-two rotations enters
//! `gglwe_products_from_dft` and, through the default of `gglwe_products_idft_normalize`,
//! `gglwe_product_idft_normalize`, on the exact budgets the queries return with the probes'
//! extra scratch, and matches `FFT64Portable`.

use crate::{FFT64Portable, hal_impl::delegating_backend::ProductStepsFFT64Portable as Probed};
use poulpy_core::{
    layouts::{GGLWEInfos, prepared::GGLWEPreparedBackendRef},
    reference::keyswitching::glwe as reference,
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
poulpy_core::impl_polynomial_evaluation_derived_full!(Probed);
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
poulpy_core::impl_linear_transformation_reference_full!(Probed);

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
        reference::gglwe_product_digits_strided_tmp_bytes_reference(
            module,
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
        reference::gglwe_product_digits_strided_reference(module, res, a, dsize, product_limbs, pmat, scratch)
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
        EXTRA + reference::gglwe_product_fused_tmp_bytes_reference(module, res_size, product_size, a_size, key_infos)
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
        reference::gglwe_product_fused_reference(module, res, product_size, a, key, term_count, addends, p, accumulate, scratch)
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
        EXTRA + reference::gglwe_products_from_dft_tmp_bytes_reference(module, count, rank, source_size, key_infos)
    }

    fn gglwe_products_from_dft(
        module: &Module<Self>,
        res: &mut [VecZnxDftBackendMut<'_, Self>],
        sources: &mut [VecZnxDftBackendMut<'_, Self>],
        source_sizes: &[usize],
        source_base2k: usize,
        key: &GGLWEPreparedBackendRef<'_, Self>,
        output_size: usize,
        term_count: usize,
        p: i64,
        accumulate: &[bool],
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        probed(&PRODUCTS_FROM_DFT, scratch, |rest| {
            reference::gglwe_products_from_dft_reference(
                module,
                res,
                sources,
                source_sizes,
                source_base2k,
                key,
                output_size,
                term_count,
                p,
                accumulate,
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
        EXTRA + reference::gglwe_product_idft_normalize_tmp_bytes_reference(module, res_size, product_size, a_size, key_infos)
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
            reference::gglwe_product_idft_normalize_reference(
                module, res, res_base2k, res_k, res_offset, res_dft, a, key, term_count, addends, body, a_base2k, rest,
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

/// The refusals of the multi-source step hooks and of the operand layout, on `FFT64Portable`
/// with a zero key: each is made before anything is read or written.
mod refusals {
    use crate::FFT64Portable as BE;
    use poulpy_core::{
        api::GLWESwitchingKeyPreparedFactory,
        layouts::{
            Base2K, Degree, Dnum, GLWESwitchingKeyLayout, Rank, TorusPrecision,
            prepared::{GGLWEPreparedToBackendRef, GLWESwitchingKeyPrepared},
        },
        oep::GGLWEProductDigitsStridedImpl,
        reference::keyswitching::glwe::gglwe_operand_from_dft,
    };
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc},
        layouts::{Backend, Module, ScratchOwned, VecZnxDftToBackendMut, VecZnxDftToBackendRef},
    };

    const N: usize = 64;
    const BASE2K: usize = 12;

    fn key(module: &Module<BE>) -> GLWESwitchingKeyPrepared<<BE as Backend>::OwnedBuf, BE> {
        module.glwe_switching_key_prepared_alloc_from_infos(&GLWESwitchingKeyLayout {
            n: Degree(N as u32),
            base2k: Base2K(BASE2K as u32),
            dnum: Dnum(2),
            k_aux: TorusPrecision(BASE2K as u32),
            rank_in: Rank(1),
            rank_out: Rank(1),
            k_digit: TorusPrecision(BASE2K as u32),
            k_offset: TorusPrecision(0),
        })
    }

    #[test]
    #[should_panic(expected = "gglwe_products_from_dft: items of different lengths")]
    fn products_from_dft_rejects_items_of_different_lengths() {
        let module = Module::<BE>::new(N as u64);
        let key = key(&module);
        let mut res = module.vec_znx_dft_alloc(N, 2, 2);
        let mut scratch = ScratchOwned::<BE>::alloc(1 << 20);
        BE::gglwe_products_from_dft(
            &module,
            &mut [res.to_backend_mut()],
            &mut [],
            &[],
            BASE2K,
            &key.to_backend_ref(),
            2,
            1,
            1,
            &[false],
            &mut scratch.borrow(),
        );
    }

    #[test]
    #[should_panic(expected = "gglwe_operand_from_dft: the source's base2k differs from the key's")]
    fn products_from_dft_rejects_a_source_of_another_base2k() {
        let module = Module::<BE>::new(N as u64);
        let key = key(&module);
        let (mut res, mut source) = (module.vec_znx_dft_alloc(N, 2, 2), module.vec_znx_dft_alloc(N, 2, 2));
        let mut scratch = ScratchOwned::<BE>::alloc(1 << 20);
        BE::gglwe_products_from_dft(
            &module,
            &mut [res.to_backend_mut()],
            &mut [source.to_backend_mut()],
            &[2],
            BASE2K + 1,
            &key.to_backend_ref(),
            2,
            1,
            1,
            &[false],
            &mut scratch.borrow(),
        );
    }

    #[test]
    #[should_panic(expected = "gglwe_products_from_dft: scratch below its query")]
    fn products_from_dft_rejects_a_short_scratch() {
        let module = Module::<BE>::new(N as u64);
        let key = key(&module);
        let (mut res, mut source) = (module.vec_znx_dft_alloc(N, 2, 2), module.vec_znx_dft_alloc(N, 2, 2));
        let mut scratch = ScratchOwned::<BE>::alloc(64);
        BE::gglwe_products_from_dft(
            &module,
            &mut [res.to_backend_mut()],
            &mut [source.to_backend_mut()],
            &[2],
            BASE2K,
            &key.to_backend_ref(),
            2,
            1,
            1,
            &[false],
            &mut scratch.borrow(),
        );
    }

    #[test]
    #[should_panic(expected = "gglwe_operand_from_dft: scratch below its query")]
    fn operand_from_dft_rejects_a_short_scratch() {
        let module = Module::<BE>::new(N as u64);
        let key = key(&module);
        let (mut res, mut source) = (module.vec_znx_dft_alloc(N, 1, 4), module.vec_znx_dft_alloc(N, 2, 2));
        let mut scratch = ScratchOwned::<BE>::alloc(64);
        gglwe_operand_from_dft(
            &module,
            &mut res.to_backend_mut(),
            &mut source.to_backend_mut(),
            BASE2K,
            &key.to_backend_ref(),
            2,
            &mut scratch.borrow(),
        );
    }

    #[test]
    #[should_panic(expected = "gglwe_products_idft_normalize: items of different lengths")]
    fn products_idft_normalize_rejects_items_of_different_lengths() {
        let module = Module::<BE>::new(N as u64);
        let key = key(&module);
        let mut res = module.vec_znx_alloc(N, 2, 2);
        let a = module.vec_znx_dft_alloc(N, 1, 2);
        let mut scratch = ScratchOwned::<BE>::alloc(1 << 20);
        BE::gglwe_products_idft_normalize(
            &module,
            &mut [poulpy_hal::test_suite::vec_znx_backend_mut::<BE>(&mut res)],
            BASE2K,
            2 * BASE2K,
            0,
            &mut [],
            &[a.to_backend_ref()],
            &key.to_backend_ref(),
            1,
            &[],
            &[None],
            BASE2K,
            &mut scratch.borrow(),
        );
    }
}
