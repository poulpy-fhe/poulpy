//! Unit shifts inherit the selected assignment and its budget, and the assignment
//! counts the compositions entering each of its routes.
use super::OverrideBackend;
use poulpy_ckks::{
    CKKSCompositionError, GLWEPlaintextMeta,
    api::{CKKSAddOps, CKKSSubOps},
    reference::{add::CKKSAddReference, sub::CKKSSubReference},
    test_suite::{
        CKKSTestParams,
        helpers::{assert_ckks_error, ckks_spec},
        parity::helpers::{fixture_ciphertext, snapshot, with_scratch},
    },
};
use poulpy_ckks::{ReplicatedLayout, Scale};
use poulpy_core::layouts::LWEInfos;
use poulpy_cpu_oracle::FFT64Oracle;
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Module, Standard};
use std::cell::Cell;

thread_local! {
    pub(super) static ADD_ASSIGN_CALLS: Cell<usize> = const { Cell::new(0) };
    pub(super) static SUB_ASSIGN_CALLS: Cell<usize> = const { Cell::new(0) };
    pub(super) static ADD_PT_CALLS: Cell<usize> = const { Cell::new(0) };
    pub(super) static SUB_PT_CALLS: Cell<usize> = const { Cell::new(0) };
    pub(super) static CI_ADD_ASSIGN_CALLS: Cell<usize> = const { Cell::new(0) };
    static CI_ADD_PT_CALLS: Cell<usize> = const { Cell::new(0) };
}

/// A hook's result, refused when it is an error; a hook returning nothing refuses by panicking.
pub(super) trait Refusal {
    fn refused(&self) -> bool;
}

impl<T, E> Refusal for Result<T, E> {
    fn refused(&self) -> bool {
        self.is_err()
    }
}

impl Refusal for () {
    fn refused(&self) -> bool {
        false
    }
}

/// [`with_marked_bytes`] of `EXTRA` bytes.
pub(super) fn with_marked_region<const EXTRA: usize, R: Refusal>(
    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    body: impl FnOnce(&mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>) -> R,
) -> R {
    with_marked_bytes(EXTRA, scratch, body)
}

/// Takes `bytes` first, fills them with a pattern, runs `body` on the rest and
/// asserts the pattern intact; on an empty arena, runs `body` on it, for its refusals,
/// and asserts it refused, so a composition giving the hook no bytes fails.
pub(super) fn with_marked_bytes<R: Refusal>(
    bytes: usize,
    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    body: impl FnOnce(&mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>) -> R,
) -> R {
    if scratch.available() == 0 {
        let result = body(scratch);
        assert!(result.refused(), "an empty arena reached an override that accepted the call");
        return result;
    }
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

// The assignment counts its calls per route and runs behind a marked region its query adds; the `_into` form forwards.
macro_rules! impl_carry_with_plaintext_workspace {
    ($ring:ty, $verb:ident, $Impl:ident, $Reference:ident, $assign_calls:ident, $pt_calls:ident, $workspace:expr) => {
        ::paste::paste! {
            unsafe impl ::poulpy_ckks::oep::$Impl<$ring> for OverrideBackend {
                fn [<ckks_ $verb _tmp_bytes>](module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                    $workspace + $Reference::<Self, $ring>::[<ckks_ $verb _tmp_bytes_reference>](module, res_size)
                }

                fn [<ckks_ $verb _into>]<Dst, A, B>(
                    module: &::poulpy_hal::layouts::Module<Self>,
                    dst: &mut Dst,
                    a: &A,
                    b: &B,
                    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
                ) -> ::poulpy_ckks::CKKSResult<()>
                where
                    Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                    A: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + ::poulpy_ckks::CKKSCtBounds,
                    B: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + ::poulpy_ckks::CKKSCtBounds,
                {
                    $Reference::<Self, $ring>::[<ckks_ $verb _into_reference>](module, dst, a, b, scratch)
                }

                fn [<ckks_ $verb _assign>]<Dst, A>(
                    module: &::poulpy_hal::layouts::Module<Self>,
                    dst: &mut Dst,
                    a: &A,
                    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
                ) -> ::poulpy_ckks::CKKSResult<()>
                where
                    Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                    A: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + ::poulpy_ckks::CKKSCtBounds,
                {
                    let calls = if a.rank().as_usize() == 0 { &$pt_calls } else { &$assign_calls };
                    calls.set(calls.get() + 1);
                    with_marked_region::<{ $workspace }, _>(scratch, |rest| {
                        $Reference::<Self, $ring>::[<ckks_ $verb _assign_reference>](module, dst, a, rest)
                    })
                }
            }
        }
    };
}
pub(super) const ADD_ASSIGN_EXTRA: usize = 1024;
pub(super) const SUB_ASSIGN_EXTRA: usize = 1536;
impl_carry_with_plaintext_workspace!(
    Standard,
    add,
    CKKSAddImpl,
    CKKSAddReference,
    ADD_ASSIGN_CALLS,
    ADD_PT_CALLS,
    ADD_ASSIGN_EXTRA
);
impl_carry_with_plaintext_workspace!(
    Standard,
    sub,
    CKKSSubImpl,
    CKKSSubReference,
    SUB_ASSIGN_CALLS,
    SUB_PT_CALLS,
    SUB_ASSIGN_EXTRA
);
// The conjugate-invariant addition counts apart, behind an extra no standard-ring query takes.
pub(super) const CI_ADD_ASSIGN_EXTRA: usize = 1280;
impl_carry_with_plaintext_workspace!(
    ConjugateInvariant,
    add,
    CKKSAddImpl,
    CKKSAddReference,
    CI_ADD_ASSIGN_CALLS,
    CI_ADD_PT_CALLS,
    CI_ADD_ASSIGN_EXTRA
);

/// Each unit shift enters the plaintext route of the selected assignment once, its query is the unit plaintext beside
/// that assignment's query, which adds the probe's extra to the reference query, and its output equals the oracle's; a
/// refused one enters no route; the arithmetic parity, each operation on its exact poisoned query, enters every route of
/// every counting assignment.
#[test]
fn unit_shifts_use_selected_plaintext_scratch() {
    let module_ref = Module::<FFT64Oracle>::new(64);
    let module_test = Module::<OverrideBackend>::new(64);
    for size in [1, 3, 8] {
        assert_eq!(
            <Module<OverrideBackend> as CKKSAddOps<OverrideBackend, Standard>>::ckks_add_tmp_bytes(&module_test, size),
            ADD_ASSIGN_EXTRA + <_ as CKKSAddReference<_, Standard>>::ckks_add_tmp_bytes_reference(&module_test, size)
        );
        assert_eq!(
            <Module<OverrideBackend> as CKKSSubOps<OverrideBackend, Standard>>::ckks_sub_tmp_bytes(&module_test, size),
            SUB_ASSIGN_EXTRA + <_ as CKKSSubReference<_, Standard>>::ckks_sub_tmp_bytes_reference(&module_test, size)
        );
        // The unit plaintext, carved at the largest degree a masked one takes, beside the selected workspace.
        let unit = OverrideBackend::scratch_aligned(OverrideBackend::vec_znx_bytes_of(
            2 << ReplicatedLayout::MAX_LOG_R,
            1,
            size + 1,
        ));
        assert_eq!(
            <Module<OverrideBackend> as CKKSAddOps<OverrideBackend, Standard>>::ckks_add_one_tmp_bytes(&module_test, size),
            unit + <Module<OverrideBackend> as CKKSAddOps<OverrideBackend, Standard>>::ckks_add_tmp_bytes(&module_test, size)
        );
        assert_eq!(
            <Module<OverrideBackend> as CKKSSubOps<OverrideBackend, Standard>>::ckks_sub_one_tmp_bytes(&module_test, size),
            unit + <Module<OverrideBackend> as CKKSSubOps<OverrideBackend, Standard>>::ckks_sub_tmp_bytes(&module_test, size)
        );
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
    with_scratch::<FFT64Oracle, _>(
        <Module<FFT64Oracle> as CKKSAddOps<FFT64Oracle, Standard>>::ckks_add_one_tmp_bytes(&module_ref, want.max_size()),
        |s| module_ref.ckks_add_one_assign(&mut want, s),
    )
    .unwrap();
    ADD_PT_CALLS.set(0);
    with_scratch::<OverrideBackend, _>(
        <Module<OverrideBackend> as CKKSAddOps<OverrideBackend, Standard>>::ckks_add_one_tmp_bytes(&module_test, got.max_size()),
        |s| module_test.ckks_add_one_assign(&mut got, s),
    )
    .unwrap();
    assert_eq!(ADD_PT_CALLS.get(), 1);
    assert_eq!(
        snapshot::<OverrideBackend, _, Standard>(&got),
        snapshot::<FFT64Oracle, _, Standard>(&want),
        "ckks_add_one_assign"
    );
    with_scratch::<FFT64Oracle, _>(
        <Module<FFT64Oracle> as CKKSSubOps<FFT64Oracle, Standard>>::ckks_sub_one_tmp_bytes(&module_ref, want.max_size()),
        |s| module_ref.ckks_sub_one_assign(&mut want, s),
    )
    .unwrap();
    SUB_PT_CALLS.set(0);
    with_scratch::<OverrideBackend, _>(
        <Module<OverrideBackend> as CKKSSubOps<OverrideBackend, Standard>>::ckks_sub_one_tmp_bytes(&module_test, got.max_size()),
        |s| module_test.ckks_sub_one_assign(&mut got, s),
    )
    .unwrap();
    assert_eq!(SUB_PT_CALLS.get(), 1);
    assert_eq!(
        snapshot::<OverrideBackend, _, Standard>(&got),
        snapshot::<FFT64Oracle, _, Standard>(&want),
        "ckks_sub_one_assign"
    );

    let small = Module::<OverrideBackend>::new(32);
    ADD_PT_CALLS.set(0);
    SUB_PT_CALLS.set(0);
    for (op, add) in [("ckks_add_one_assign", true), ("ckks_sub_one_assign", false)] {
        let before = snapshot::<OverrideBackend, _, Standard>(&got);
        let bytes = if add {
            <Module<OverrideBackend> as CKKSAddOps<OverrideBackend, Standard>>::ckks_add_one_tmp_bytes(&small, got.max_size())
        } else {
            <Module<OverrideBackend> as CKKSSubOps<OverrideBackend, Standard>>::ckks_sub_one_tmp_bytes(&small, got.max_size())
        };
        let err = with_scratch::<OverrideBackend, _>(bytes, |s| {
            if add {
                small.ckks_add_one_assign(&mut got, s)
            } else {
                small.ckks_sub_one_assign(&mut got, s)
            }
        })
        .unwrap_err();
        let reason = "operand degree exceeds the module degree".into();
        assert_ckks_error(op, &err, CKKSCompositionError::InvalidOperand { op, reason });
        assert_eq!(
            snapshot::<OverrideBackend, _, Standard>(&got),
            before,
            "{op}: refused call changed dst"
        );
    }
    assert_eq!(
        (ADD_PT_CALLS.get(), SUB_PT_CALLS.get()),
        (0, 0),
        "a refused unit shift entered the selected assignment"
    );

    let counters = [&ADD_ASSIGN_CALLS, &SUB_ASSIGN_CALLS, &ADD_PT_CALLS, &SUB_PT_CALLS];
    counters.iter().for_each(|calls| calls.set(0));
    poulpy_ckks::test_suite::parity::test_arithmetic_parity::<FFT64Oracle, OverrideBackend, f64, Standard>(
        params,
        &module_ref,
        &module_test,
    );
    assert!(
        counters.iter().all(|calls| calls.get() > 0),
        "the arithmetic parity skipped a counting assignment"
    );
}
