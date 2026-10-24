//! Unit shifts inherit the selected constant-plaintext operation and its budget,
//! and the ciphertext and plaintext-vector assignments count the compositions entering them.
use super::OverrideBackend;
use crate::FFT64Portable;
use poulpy_ckks::Scale;
use poulpy_ckks::{
    GLWEPlaintextMeta,
    api::{CKKSAddOps, CKKSSubOps},
    test_suite::CKKSTestParams,
};
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
    assert_eq!(after, [0x5A; EXTRA], "an override's extra region was overwritten");
    result
}

// Forward the unchanged operations explicitly so the defaults compose a backend
// that gives only constant-plaintext assignment a larger scratch requirement.
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
                    a: &A,
                    dst_coeff: usize,
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
                        module, dst, a, dst_coeff, pt, pt_coeff, scratch,
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
                    let (mut region, mut remaining) = scratch.borrow().take_region($workspace);
                    <Self as Backend>::copy_host_to_view(&mut region, &[0x5A; $workspace]);
                    ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _pt_const_assign_reference>](
                        module, dst, dst_coeff, pt, pt_coeff, &mut remaining,
                    )
                }

            }
        }
    };
}
pub(super) const ADD_ASSIGN_EXTRA: usize = 1024;
pub(super) const SUB_ASSIGN_EXTRA: usize = 1536;
pub(super) const ADD_PT_VEC_EXTRA: usize = 2048;
pub(super) const SUB_PT_VEC_EXTRA: usize = 3072;
impl_carry_with_plaintext_workspace!(
    add,
    CKKSAddImpl,
    CKKSAddReference,
    ADD_CONST_CALLS,
    4096,
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
    8192,
    SUB_ASSIGN_CALLS,
    SUB_ASSIGN_EXTRA,
    SUB_PT_VEC_CALLS,
    SUB_PT_VEC_EXTRA
);

#[test]
fn unit_shifts_use_selected_plaintext_scratch() {
    let reference = Module::<FFT64Portable>::new(64);
    let module = Module::<OverrideBackend>::new(64);
    for size in [1, 3, 8] {
        assert!(module.ckks_add_one_tmp_bytes(size) > module.ckks_add_tmp_bytes(size));
        assert!(module.ckks_sub_one_tmp_bytes(size) > module.ckks_sub_tmp_bytes(size));
        assert_eq!(module.ckks_add_one_tmp_bytes(size), module.ckks_add_pt_const_tmp_bytes(size));
        assert_eq!(module.ckks_sub_one_tmp_bytes(size), module.ckks_sub_pt_const_tmp_bytes(size));
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
    ADD_CONST_CALLS.set(0);
    SUB_CONST_CALLS.set(0);
    // The parity runner gives each operation its exact reported, poisoned arena
    // and checks coefficient and metadata equality against the portable backend.
    poulpy_ckks::test_suite::parity::test_arithmetic_parity::<FFT64Portable, OverrideBackend, f64>(params, &reference, &module);
    assert!(ADD_CONST_CALLS.get() > 0);
    assert!(SUB_CONST_CALLS.get() > 0);
}
