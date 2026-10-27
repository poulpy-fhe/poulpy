//! Unit shifts inherit the selected constant-plaintext operation and its budget,
//! and the ciphertext and plaintext-vector assignments count the compositions entering them.
use super::OverrideBackend;
use poulpy_ckks::Scale;
use poulpy_ckks::{
    GLWEPlaintextMeta,
    api::{CKKSAddOps, CKKSSubOps},
    reference::{add::CKKSAddReference, sub::CKKSSubReference},
    test_suite::{
        CKKSTestParams,
        helpers::ckks_spec,
        parity::helpers::{fixture_ciphertext, snapshot, with_scratch},
    },
};
use poulpy_core::layouts::LWEInfos;
use poulpy_cpu_oracle::FFT64Oracle;
use poulpy_hal::layouts::{Backend, Module};
use std::cell::Cell;

thread_local! {
    pub(super) static ADD_CONST_CALLS: Cell<usize> = const { Cell::new(0) };
    static SUB_CONST_CALLS: Cell<usize> = const { Cell::new(0) };
    pub(super) static ADD_ASSIGN_CALLS: Cell<usize> = const { Cell::new(0) };
    pub(super) static SUB_ASSIGN_CALLS: Cell<usize> = const { Cell::new(0) };
    pub(super) static ADD_PT_VEC_CALLS: Cell<usize> = const { Cell::new(0) };
    pub(super) static SUB_PT_VEC_CALLS: Cell<usize> = const { Cell::new(0) };
}

/// Takes `EXTRA` bytes first, fills them with a pattern, runs `body` on the rest and
/// asserts the pattern intact.
pub(super) fn with_marked_region<const EXTRA: usize, R>(
    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    body: impl FnOnce(&mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>) -> R,
) -> R {
    let (mut marker, mut rest) = scratch.borrow().take_region(EXTRA);
    <OverrideBackend as Backend>::copy_host_to_view(&mut marker, &[0x5A; EXTRA]);
    let result = body(&mut rest);
    let mut after = [0u8; EXTRA];
    <OverrideBackend as Backend>::copy_view_to_host(&<OverrideBackend as Backend>::view_ref_mut(&marker), &mut after);
    assert!(after == [0x5A; EXTRA], "an override's extra region was overwritten");
    result
}

/// As [`with_marked_region`], for an extra of `bytes` chosen at run time.
pub(super) fn with_marked_bytes<R>(
    bytes: usize,
    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    body: impl FnOnce(&mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>) -> R,
) -> R {
    let (mut marker, mut rest) = scratch.borrow().take_region(bytes);
    <OverrideBackend as Backend>::copy_host_to_view(&mut marker, &vec![0x5A; bytes]);
    let result = body(&mut rest);
    let mut after = vec![0u8; bytes];
    <OverrideBackend as Backend>::copy_view_to_host(&<OverrideBackend as Backend>::view_ref_mut(&marker), &mut after);
    assert!(
        after.iter().all(|&byte| byte == 0x5A),
        "an override's extra region was overwritten"
    );
    result
}

/// Asserts `query` takes a hook's extra one for one once that extra outweighs every other term: `set` stores the
/// extra, and from the query without it, `STEP` more bytes of extra move the query by exactly `STEP`. Leaves the extra
/// at 0.
pub(super) fn assert_query_takes_extra(label: &str, set: impl Fn(usize), query: impl Fn() -> usize) {
    const STEP: usize = 4096;
    set(0);
    let without = query();
    set(without);
    let dominant = query();
    set(without + STEP);
    let stepped = query();
    set(0);
    assert_eq!(
        stepped,
        dominant + STEP,
        "{label}: the query does not take the hook's extra one for one"
    );
}

// The assignments count their calls and run behind a marked region their queries add; the `_into` forms forward.
macro_rules! impl_carry_with_plaintext_workspace {
    (
        $verb:ident,
        $Impl:ident,
        $Reference:ident,
        $calls:ident,
        $workspace:expr,
        $assign_calls:ident,
        $assign_workspace:expr,
        $pt_vec_calls:ident,
        $pt_vec_workspace:expr
    ) => {
        ::paste::paste! {
            unsafe impl ::poulpy_ckks::oep::$Impl for OverrideBackend {
                fn [<ckks_ $verb _tmp_bytes>](module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                    $assign_workspace + ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _tmp_bytes_reference>](module, res_size)
                }

                fn [<ckks_ $verb _into>]<Dst, A, B>(
                    module: &::poulpy_hal::layouts::Module<Self>,
                    dst: &mut Dst,
                    a: &A,
                    b: &B,
                    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
                ) -> ::poulpy_ckks::CKKSResult<()>
                where
                    Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                    A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds,
                    B: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds,
                {
                    ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _into_reference>](module, dst, a, b, scratch)
                }


                fn [<ckks_ $verb _assign>]<Dst, A>(
                    module: &::poulpy_hal::layouts::Module<Self>,
                    dst: &mut Dst,
                    a: &A,
                    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
                ) -> ::poulpy_ckks::CKKSResult<()>
                where
                    Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSInfos + ::poulpy_ckks::SetCKKSInfos,
                    A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSInfos,
                {
                    $assign_calls.set($assign_calls.get() + 1);
                    with_marked_region::<{ $assign_workspace }, _>(scratch, |rest| {
                        ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _assign_reference>](module, dst, a, rest)
                    })
                }



                fn [<ckks_ $verb _pt_vec_tmp_bytes>](module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                    $pt_vec_workspace + ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _pt_vec_tmp_bytes_reference>](module, res_size)
                }

                fn [<ckks_ $verb _pt_vec_into>]<Dst, A, P>(
                    module: &::poulpy_hal::layouts::Module<Self>,
                    dst: &mut Dst,
                    a: &A,
                    pt: &P,
                    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
                ) -> ::poulpy_ckks::CKKSResult<()>
                where
                    Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                    A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds,
                    P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
                {
                    ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _pt_vec_into_reference>](module, dst, a, pt, scratch)
                }


                fn [<ckks_ $verb _pt_vec_assign>]<Dst, P>(
                    module: &::poulpy_hal::layouts::Module<Self>,
                    dst: &mut Dst,
                    pt: &P,
                    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
                ) -> ::poulpy_ckks::CKKSResult<()>
                where
                    Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                    P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
                {
                    $pt_vec_calls.set($pt_vec_calls.get() + 1);
                    with_marked_region::<{ $pt_vec_workspace }, _>(scratch, |rest| {
                        ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _pt_vec_assign_reference>](module, dst, pt, rest)
                    })
                }


                fn [<ckks_ $verb _pt_const_tmp_bytes>](module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                    $workspace + ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _pt_const_tmp_bytes_reference>](module, res_size)
                }

                fn [<ckks_ $verb _pt_const_into>]<Dst, A, P>(
                    module: &::poulpy_hal::layouts::Module<Self>,
                    dst: &mut Dst,
                    dst_coeff: usize,
                    a: &A,
                    pt: &P,
                    pt_coeff: usize,
                    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
                ) -> ::poulpy_ckks::CKKSResult<()>
                where
                    Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                    A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds,
                    P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
                {
                    ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _pt_const_into_reference>](
                        module, dst, dst_coeff, a, pt, pt_coeff, scratch,
                    )
                }


                fn [<ckks_ $verb _pt_const_assign>]<Dst, P>(
                    module: &::poulpy_hal::layouts::Module<Self>,
                    dst: &mut Dst,
                    dst_coeff: usize,
                    pt: &P,
                    pt_coeff: usize,
                    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
                ) -> ::poulpy_ckks::CKKSResult<()>
                where
                    Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                    P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
                {
                    $calls.set($calls.get() + 1);
                    with_marked_region::<{ $workspace }, _>(scratch, |rest| {
                        ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _pt_const_assign_reference>](
                            module, dst, dst_coeff, pt, pt_coeff, rest,
                        )
                    })
                }

            }
        }
    };
}
const ADD_CONST_EXTRA: usize = 4096;
const SUB_CONST_EXTRA: usize = 8192;
pub(super) const ADD_ASSIGN_EXTRA: usize = 1024;
pub(super) const SUB_ASSIGN_EXTRA: usize = 1536;
pub(super) const ADD_PT_VEC_EXTRA: usize = 2048;
pub(super) const SUB_PT_VEC_EXTRA: usize = 3072;
impl_carry_with_plaintext_workspace!(
    add,
    CKKSAddImpl,
    CKKSAddReference,
    ADD_CONST_CALLS,
    ADD_CONST_EXTRA,
    ADD_ASSIGN_CALLS,
    ADD_ASSIGN_EXTRA,
    ADD_PT_VEC_CALLS,
    ADD_PT_VEC_EXTRA
);
impl_carry_with_plaintext_workspace!(
    sub,
    CKKSSubImpl,
    CKKSSubReference,
    SUB_CONST_CALLS,
    SUB_CONST_EXTRA,
    SUB_ASSIGN_CALLS,
    SUB_ASSIGN_EXTRA,
    SUB_PT_VEC_CALLS,
    SUB_PT_VEC_EXTRA
);

/// Each unit shift enters the selected constant-plaintext assignment once, its query is that assignment's, which adds
/// the probe's extra to the reference query, and its output equals the oracle's; the arithmetic parity, each operation
/// on its exact poisoned query, enters every counting assignment.
#[test]
fn unit_shifts_use_selected_plaintext_scratch() {
    let module_ref = Module::<FFT64Oracle>::new(64);
    let module_test = Module::<OverrideBackend>::new(64);
    for size in [1, 3, 8] {
        assert_eq!(
            module_test.ckks_add_pt_const_tmp_bytes(size),
            ADD_CONST_EXTRA + CKKSAddReference::ckks_add_pt_const_tmp_bytes_reference(&module_test, size)
        );
        assert_eq!(
            module_test.ckks_sub_pt_const_tmp_bytes(size),
            SUB_CONST_EXTRA + CKKSSubReference::ckks_sub_pt_const_tmp_bytes_reference(&module_test, size)
        );
        assert!(module_test.ckks_add_one_tmp_bytes(size) > module_test.ckks_add_tmp_bytes(size));
        assert!(module_test.ckks_sub_one_tmp_bytes(size) > module_test.ckks_sub_tmp_bytes(size));
        // The unit constant is carved from the same arena as the selected plaintext workspace.
        assert!(module_test.ckks_add_one_tmp_bytes(size) > module_test.ckks_add_pt_const_tmp_bytes(size));
        assert!(module_test.ckks_sub_one_tmp_bytes(size) > module_test.ckks_sub_pt_const_tmp_bytes(size));
    }
    let params = CKKSTestParams {
        n: 64,
        base2k: 16,
        k: 64,
        prec_meta: GLWEPlaintextMeta {
            scale: Scale::Log(16),
            ..Default::default()
        },
        prec_log_budget: 16,
        hw: 8,
        dsize: 1,
        rank: 1,
    };

    let layout = ckks_spec(params.n, params.base2k, 15, 2 * params.base2k + 6);
    let mut want = fixture_ciphertext(&module_ref, &layout, 41);
    let mut got = fixture_ciphertext(&module_test, &layout, 41);
    with_scratch::<FFT64Oracle, _>(module_ref.ckks_add_one_tmp_bytes(want.max_size()), |s| {
        module_ref.ckks_add_one_assign(&mut want, s)
    })
    .unwrap();
    ADD_CONST_CALLS.set(0);
    with_scratch::<OverrideBackend, _>(module_test.ckks_add_one_tmp_bytes(got.max_size()), |s| {
        module_test.ckks_add_one_assign(&mut got, s)
    })
    .unwrap();
    assert_eq!(ADD_CONST_CALLS.get(), 1);
    assert_eq!(
        snapshot::<OverrideBackend, _>(&got),
        snapshot::<FFT64Oracle, _>(&want),
        "ckks_add_one_assign"
    );
    with_scratch::<FFT64Oracle, _>(module_ref.ckks_sub_one_tmp_bytes(want.max_size()), |s| {
        module_ref.ckks_sub_one_assign(&mut want, s)
    })
    .unwrap();
    SUB_CONST_CALLS.set(0);
    with_scratch::<OverrideBackend, _>(module_test.ckks_sub_one_tmp_bytes(got.max_size()), |s| {
        module_test.ckks_sub_one_assign(&mut got, s)
    })
    .unwrap();
    assert_eq!(SUB_CONST_CALLS.get(), 1);
    assert_eq!(
        snapshot::<OverrideBackend, _>(&got),
        snapshot::<FFT64Oracle, _>(&want),
        "ckks_sub_one_assign"
    );

    let counters = [
        &ADD_CONST_CALLS,
        &SUB_CONST_CALLS,
        &ADD_ASSIGN_CALLS,
        &SUB_ASSIGN_CALLS,
        &ADD_PT_VEC_CALLS,
        &SUB_PT_VEC_CALLS,
    ];
    counters.iter().for_each(|calls| calls.set(0));
    poulpy_ckks::test_suite::parity::test_arithmetic_parity::<FFT64Oracle, OverrideBackend, f64>(
        params,
        &module_ref,
        &module_test,
    );
    assert!(
        counters.iter().all(|calls| calls.get() > 0),
        "the arithmetic parity skipped a counting assignment"
    );
}
