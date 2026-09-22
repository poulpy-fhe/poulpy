use poulpy_core::{
    api::{GGSWRotate, GLWEAdd, GLWEAutomorphism, GLWEMulXpMinusOne, GLWENormalize, GLWERotate, GLWEShift, GLWETrace},
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGSWInfos, GGSWLayout, GGSWToBackendMut, GLWE, GLWEAutomorphismKeyLayout,
        GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, ModuleCoreAlloc, Rank,
        SetGaloisElement, TorusPrecision, prepared::GLWEAutomorphismKeyPreparedFactory,
    },
    oep::{GGLWEProductDigitsStridedImpl, GGSWRotateImpl, GLWEAddImpl, GLWERotateImpl, GLWETraceImpl},
    reference::operations::{GLWEAddReference, GLWERotateReference},
};
use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, FillUniform, Module, ScratchArena, ScratchOwned, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef,
    },
    source::Source,
};

use crate::{FFT64Ref, hal_impl::delegating_backend::DelegatingFFT64Ref};
use std::{cell::Cell, collections::HashMap};

thread_local! {
    static ROTATE_OVERRIDE_CALLS: Cell<usize> = const { Cell::new(0) };
    static ROTATE_ASSIGN_CALLS: Cell<usize> = const { Cell::new(0) };
    static ADD_IMPL_CALLS: Cell<usize> = const { Cell::new(0) };
    static GGSW_ASSIGN_CALLS: Cell<usize> = const { Cell::new(0) };
    static TRACE_ASSIGN_CALLS: Cell<usize> = const { Cell::new(0) };
    static DIGIT_PRODUCT_CALLS: Cell<usize> = const { Cell::new(0) };
    static DIGIT_PRODUCT_QUERY_CALLS: Cell<usize> = const { Cell::new(0) };
}

const ROTATE_EXTRA_SCRATCH: usize = 512;
const TRACE_EXTRA_SCRATCH: usize = 4096;
const DIGIT_PRODUCT_EXTRA_SCRATCH: usize = 256;

// The backend can use reference helpers while overriding its public dispatch.
unsafe impl GLWERotateImpl for DelegatingFFT64Ref {
    fn glwe_rotate_tmp_bytes(module: &Module<Self>) -> usize {
        module.glwe_rotate_tmp_bytes_reference() + ROTATE_EXTRA_SCRATCH
    }

    fn glwe_rotate<R, A>(module: &Module<Self>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<DelegatingFFT64Ref>,
        A: GLWEToBackendRef<DelegatingFFT64Ref>,
    {
        ROTATE_OVERRIDE_CALLS.set(ROTATE_OVERRIDE_CALLS.get() + 1);
        module.glwe_rotate_reference(k, res, a);
    }

    fn glwe_rotate_assign<R>(module: &Module<Self>, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, DelegatingFFT64Ref>)
    where
        R: GLWEToBackendMut<DelegatingFFT64Ref>,
    {
        ROTATE_ASSIGN_CALLS.set(ROTATE_ASSIGN_CALLS.get() + 1);
        let (marker, mut remaining) = scratch.borrow().take_region(ROTATE_EXTRA_SCRATCH);
        marker.fill(0x6D);
        module.glwe_rotate_assign_reference(k, res, &mut remaining);
        assert!(marker.iter().all(|&byte| byte == 0x6D));
    }
}

// Selecting a backend implementation is independent of reference helper availability.
unsafe impl GLWEAddImpl for DelegatingFFT64Ref {
    fn glwe_add_into<R, A, B>(module: &Module<Self>, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>,
        B: GLWEToBackendRef<Self>,
    {
        ADD_IMPL_CALLS.set(ADD_IMPL_CALLS.get() + 1);
        module.glwe_add_into_reference(res, a, b);
    }

    fn glwe_add_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>,
    {
        module.glwe_add_assign_reference(res, a);
    }
}

// Other families independently select their portable bodies.
poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(DelegatingFFT64Ref);
poulpy_core::impl_glwe_sub_reference_full!(DelegatingFFT64Ref);
poulpy_core::impl_glwe_negate_reference_full!(DelegatingFFT64Ref);
poulpy_core::impl_glwe_copy_reference_full!(DelegatingFFT64Ref);
poulpy_core::impl_glwe_normalize_reference_full!(DelegatingFFT64Ref);
poulpy_core::impl_glwe_shift_reference_full!(DelegatingFFT64Ref);
poulpy_core::impl_glwe_keyswitch_reference_full!(DelegatingFFT64Ref);
// The pairwise suite must use a selected comparison backend's implementation
// and its own scratch query, even when that backend delegates its arithmetic.
unsafe impl GGLWEProductDigitsStridedImpl for DelegatingFFT64Ref {
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
        DIGIT_PRODUCT_QUERY_CALLS.set(DIGIT_PRODUCT_QUERY_CALLS.get() + 1);
        DIGIT_PRODUCT_EXTRA_SCRATCH
            + poulpy_core::reference::keyswitching::glwe::gglwe_product_digits_strided_tmp_bytes_reference(
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
        DIGIT_PRODUCT_CALLS.set(DIGIT_PRODUCT_CALLS.get() + 1);
        let (marker, mut remaining) = scratch.borrow().take_region(DIGIT_PRODUCT_EXTRA_SCRATCH);
        marker.fill(0x39);
        poulpy_core::reference::keyswitching::glwe::gglwe_product_digits_strided_reference(
            module,
            res,
            a,
            dsize,
            product_limbs,
            pmat,
            &mut remaining,
        );
        assert!(marker.iter().all(|&byte| byte == 0x39));
    }
}
poulpy_core::impl_conversion_reference_full!(DelegatingFFT64Ref);
poulpy_core::impl_automorphism_reference_full!(DelegatingFFT64Ref);

// Inherit derived rotate and sizing, and override only the derived assign hook.
unsafe impl GGSWRotateImpl for DelegatingFFT64Ref {
    fn ggsw_rotate_assign<R>(module: &Module<Self>, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGSWToBackendMut<Self> + GGSWInfos,
    {
        GGSW_ASSIGN_CALLS.set(GGSW_ASSIGN_CALLS.get() + 1);
        let mut res = res.to_backend_mut();
        for row in 0..res.dnum().as_usize() {
            for col in 0..res.rank().as_usize() + 1 {
                module.glwe_rotate_assign(k, &mut res.at_view_mut(row, col), &mut scratch.borrow());
            }
        }
    }
}

// The out-of-place trace inherits its derived body and scratch query. Its
// selected assign operation deliberately needs more workspace than FFT64Ref.
unsafe impl GLWETraceImpl for DelegatingFFT64Ref {
    fn glwe_trace_assign_tmp_bytes<A, K>(module: &Module<Self>, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        let layout = GLWELayout {
            n: a_infos.n(),
            base2k: key_infos.base2k(),
            k: a_infos.k(),
            rank: a_infos.rank(),
        };
        let mut bytes = module
            .glwe_shift_tmp_bytes(layout.size())
            .max(module.glwe_automorphism_tmp_bytes(&layout, &layout, key_infos));
        if a_infos.base2k() != key_infos.base2k() {
            // This test override allocates conversion storage on the heap.
            bytes = bytes.max(module.glwe_normalize_tmp_bytes());
        }
        Self::scratch_aligned(bytes) + TRACE_EXTRA_SCRATCH
    }

    fn glwe_trace_assign<R, H>(module: &Module<Self>, res: &mut R, skip: usize, keys: &H, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        H: GetAutomorphismKey<Self>,
    {
        TRACE_ASSIGN_CALLS.set(TRACE_ASSIGN_CALLS.get() + 1);
        let (marker, mut remaining) = scratch.borrow().take_region(TRACE_EXTRA_SCRATCH);
        marker.fill(0xB7);
        let rotations = module.glwe_trace_galois_elements();
        assert!(skip <= rotations.len());
        if let Some(&first) = rotations.get(skip) {
            let key = keys.get_automorphism_key(first, res.k()).unwrap();
            let mut converted = (res.base2k() != key.base2k()).then(|| {
                module.glwe_alloc_from_infos(&GLWELayout {
                    n: res.n(),
                    base2k: key.base2k(),
                    k: res.k(),
                    rank: res.rank(),
                })
            });
            if let Some(converted) = &mut converted {
                module.glwe_normalize(converted, res, &mut remaining.borrow());
            }
            {
                let mut work = match &mut converted {
                    Some(converted) => GLWEToBackendMut::<Self>::to_backend_mut(converted),
                    None => res.to_backend_mut(),
                };
                let mut work = &mut work;
                for &p in &rotations[skip..] {
                    module.glwe_rsh(1, &mut work, &mut remaining.borrow());
                    let key = keys.get_automorphism_key(p, work.k()).unwrap();
                    module.glwe_automorphism_add_assign(&mut work, &key, &mut remaining.borrow());
                }
            }
            if let Some(converted) = &converted {
                module.glwe_normalize(res, converted, &mut remaining.borrow());
            }
        }
        assert!(marker.iter().all(|&byte| byte == 0xB7));
    }
}

fn poisoned_scratch<B>(bytes: usize) -> ScratchOwned<B>
where
    B: Backend<OwnedBuf = AlignedBuf>,
    ScratchOwned<B>: ScratchOwnedAlloc<B>,
{
    let mut scratch = ScratchOwned::<B>::alloc(bytes);
    scratch.data.fill(0xA5);
    scratch.data.truncate(bytes);
    scratch
}

fn sample_glwe() -> GLWE<AlignedBuf, i64> {
    let layout = GLWELayout {
        n: Degree(256),
        base2k: Base2K(17),
        k: TorusPrecision(50),
        rank: Rank(2),
    };
    let module: Module<FFT64Ref> = Module::new(256);
    let mut ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut source = Source::new([7u8; 32]);
    ct.fill_uniform(40, &mut source);
    ct
}

#[test]
fn delegating_backend_manual_family_matches_fft64_ref() {
    let delegated_radix: Option<usize> = Module::<DelegatingFFT64Ref>::max_base2k(1 << 16, 32, 128);
    assert_eq!(delegated_radix, Some(19));
    assert_eq!(delegated_radix, Module::<FFT64Ref>::max_base2k(1 << 16, 32, 128));
    let module_delegating: Module<DelegatingFFT64Ref> = Module::new(256);
    let module_ref: Module<FFT64Ref> = Module::new(256);

    let input = sample_glwe();
    let mut delegating_out = module_delegating.glwe_alloc_from_infos(&input);
    let mut ref_out = module_ref.glwe_alloc_from_infos(&input);

    module_delegating.glwe_mul_xp_minus_one(-7, &mut delegating_out, &input);
    module_ref.glwe_mul_xp_minus_one(-7, &mut ref_out, &input);

    assert_eq!(delegating_out, ref_out);
}

#[test]
fn public_core_dispatch_reaches_override_and_forwards_assign() {
    let module_delegating: Module<DelegatingFFT64Ref> = Module::new(256);
    let module_ref: Module<FFT64Ref> = Module::new(256);

    let input = sample_glwe();
    let mut delegating_out = module_delegating.glwe_alloc_from_infos(&input);
    let mut ref_out = module_ref.glwe_alloc_from_infos(&input);

    let before = ROTATE_OVERRIDE_CALLS.get();
    module_delegating.glwe_rotate(11, &mut delegating_out, &input);
    assert_eq!(ROTATE_OVERRIDE_CALLS.get(), before + 1);
    module_ref.glwe_rotate(11, &mut ref_out, &input);

    assert_eq!(delegating_out, ref_out);
}

#[test]
fn public_core_dispatch_forwards_unchanged_method_with_own_scratch() {
    let module_delegating = Module::<DelegatingFFT64Ref>::new(256);
    let module_ref = Module::<FFT64Ref>::new(256);
    let mut actual = sample_glwe();
    let mut expected = actual.clone();
    let mut actual_scratch = ScratchOwned::<DelegatingFFT64Ref>::alloc(module_delegating.glwe_rotate_tmp_bytes());
    let mut expected_scratch = ScratchOwned::<FFT64Ref>::alloc(module_ref.glwe_rotate_tmp_bytes());
    module_delegating.glwe_rotate_assign(-11, &mut actual, &mut actual_scratch.borrow());
    module_ref.glwe_rotate_assign(-11, &mut expected, &mut expected_scratch.borrow());
    assert_eq!(actual, expected);
}

#[test]
fn public_core_dispatch_accepts_direct_impl_with_callable_reference() {
    let module = Module::<DelegatingFFT64Ref>::new(256);
    let reference = Module::<FFT64Ref>::new(256);
    let a = sample_glwe();
    let b = sample_glwe();
    let mut actual = module.glwe_alloc_from_infos(&a);
    let mut expected = reference.glwe_alloc_from_infos(&a);
    let before = ADD_IMPL_CALLS.get();
    module.glwe_add_into(&mut actual, &a, &b);
    reference.glwe_add_into(&mut expected, &a, &b);
    assert_eq!(ADD_IMPL_CALLS.get(), before + 1);
    assert_eq!(actual, expected);
    module.glwe_add_assign(&mut actual, &a);
    reference.glwe_add_assign(&mut expected, &a);
    assert_eq!(actual, expected);
}

#[test]
fn direct_reference_rotation_bypasses_backend_override() {
    let module = Module::<DelegatingFFT64Ref>::new(256);
    let reference = Module::<FFT64Ref>::new(256);
    let input = sample_glwe();
    let mut actual = module.glwe_alloc_from_infos(&input);
    let mut expected = reference.glwe_alloc_from_infos(&input);
    let before = ROTATE_OVERRIDE_CALLS.get();
    module.glwe_rotate_reference(19, &mut actual, &input);
    reference.glwe_rotate(19, &mut expected, &input);
    assert_eq!(ROTATE_OVERRIDE_CALLS.get(), before);
    assert_eq!(actual, expected);
}

#[test]
fn derived_ggsw_rotation_uses_selected_glwe_methods_and_scratch() {
    let module = Module::<DelegatingFFT64Ref>::new(256);
    let reference = Module::<FFT64Ref>::new(256);
    let layout = GGSWLayout {
        n: Degree(256),
        base2k: Base2K(17),
        dnum: Dnum(2),
        dsize: Dsize(1),
        k_aux: TorusPrecision(18),
        rank: Rank(2),
    };
    let mut input = reference.ggsw_alloc_from_infos(&layout);
    input.fill_uniform(17, &mut Source::new([43; 32]));
    let mut actual = module.ggsw_alloc_from_infos(&layout);
    let mut expected = reference.ggsw_alloc_from_infos(&layout);
    let rows = layout.dnum.as_usize() * (layout.rank.as_usize() + 1);

    let before = ROTATE_OVERRIDE_CALLS.get();
    module.ggsw_rotate(-13, &mut actual, &input);
    reference.ggsw_rotate(-13, &mut expected, &input);
    assert_eq!(ROTATE_OVERRIDE_CALLS.get(), before + rows);
    assert_eq!(actual, expected);

    let actual_bytes = module.ggsw_rotate_tmp_bytes();
    let expected_bytes = reference.ggsw_rotate_tmp_bytes();
    assert_eq!(actual_bytes, expected_bytes + ROTATE_EXTRA_SCRATCH);
    let before = ROTATE_ASSIGN_CALLS.get();
    let before_ggsw = GGSW_ASSIGN_CALLS.get();
    module.ggsw_rotate_assign(
        7,
        &mut actual,
        &mut poisoned_scratch::<DelegatingFFT64Ref>(actual_bytes).borrow(),
    );
    reference.ggsw_rotate_assign(7, &mut expected, &mut poisoned_scratch::<FFT64Ref>(expected_bytes).borrow());
    assert_eq!(ROTATE_ASSIGN_CALLS.get(), before + rows);
    assert_eq!(GGSW_ASSIGN_CALLS.get(), before_ggsw + 1);
    assert_eq!(actual, expected);
}

#[test]
fn derived_trace_uses_selected_assign_and_its_larger_scratch_query() {
    let module = Module::<DelegatingFFT64Ref>::new(256);
    let reference = Module::<FFT64Ref>::new(256);
    let mut input = sample_glwe();
    input.fill_uniform(17, &mut Source::new([61; 32]));
    let saved_input = input.clone();
    let key_layout = GLWEAutomorphismKeyLayout {
        n: input.n(),
        base2k: input.base2k(),
        dnum: Dnum(3),
        dsize: Dsize(1),
        k_aux: TorusPrecision(18),
        rank: input.rank(),
    };
    let rotations = reference.glwe_trace_galois_elements();
    let skip = rotations.len() - 1;
    let p = rotations[skip];
    let mut key = reference.glwe_automorphism_key_alloc_from_infos(&key_layout);
    key.fill_uniform(17, &mut Source::new([89; 32]));
    key.set_p(p);
    let saved_key = key.clone();
    let mut prepared_actual = module.glwe_automorphism_key_prepared_alloc_from_infos(&key);
    let mut prepared_expected = reference.glwe_automorphism_key_prepared_alloc_from_infos(&key);
    module.glwe_automorphism_key_prepare(
        &mut prepared_actual,
        &key,
        &mut poisoned_scratch::<DelegatingFFT64Ref>(module.glwe_automorphism_key_prepare_tmp_bytes(&key)).borrow(),
    );
    reference.glwe_automorphism_key_prepare(
        &mut prepared_expected,
        &key,
        &mut poisoned_scratch::<FFT64Ref>(reference.glwe_automorphism_key_prepare_tmp_bytes(&key)).borrow(),
    );
    let keys_actual = HashMap::from([(p, prepared_actual)]);
    let keys_expected = HashMap::from([(p, prepared_expected)]);
    let mut actual = module.glwe_alloc_from_infos(&input);
    let mut expected = reference.glwe_alloc_from_infos(&input);
    let actual_bytes = module.glwe_trace_tmp_bytes(&actual, &input, &key);
    let expected_bytes = reference.glwe_trace_tmp_bytes(&expected, &input, &key);
    assert_eq!(
        module.glwe_trace_assign_tmp_bytes(&input, &key),
        reference.glwe_trace_assign_tmp_bytes(&input, &key) + TRACE_EXTRA_SCRATCH,
    );
    assert_eq!(actual_bytes, expected_bytes + TRACE_EXTRA_SCRATCH);
    let before = TRACE_ASSIGN_CALLS.get();
    module.glwe_trace(
        &mut actual,
        skip,
        &input,
        &keys_actual,
        &mut poisoned_scratch::<DelegatingFFT64Ref>(actual_bytes).borrow(),
    );
    reference.glwe_trace(
        &mut expected,
        skip,
        &input,
        &keys_expected,
        &mut poisoned_scratch::<FFT64Ref>(expected_bytes).borrow(),
    );
    assert_eq!(TRACE_ASSIGN_CALLS.get(), before + 1);
    assert_eq!(actual, expected);
    assert_eq!(input, saved_input);
    assert_eq!(key, saved_key);
}

#[test]
fn core_parity_dispatches_comparison_backend_override_and_scratch() {
    let comparison = Module::<DelegatingFFT64Ref>::new(64);
    let tested = Module::<FFT64Ref>::new(64);
    let calls = DIGIT_PRODUCT_CALLS.get();
    let queries = DIGIT_PRODUCT_QUERY_CALLS.get();
    poulpy_core::test_suite::parity::test_gglwe_product_digits_strided_parity(
        &poulpy_hal::test_suite::TestParams {
            size: 64,
            n: 64,
            base2k: 12,
        },
        &poulpy_core::test_suite::parity::ParityShapes::default(),
        &comparison,
        &tested,
    );
    assert!(DIGIT_PRODUCT_CALLS.get() > calls, "comparison backend override was bypassed");
    assert_eq!(DIGIT_PRODUCT_QUERY_CALLS.get() - queries, DIGIT_PRODUCT_CALLS.get() - calls);
}
