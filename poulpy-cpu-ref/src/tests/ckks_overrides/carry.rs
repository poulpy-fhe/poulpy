//! Unit shifts inherit the selected constant-plaintext operation and its budget.
use super::OverrideBackend;
use crate::FFT64Ref;
use poulpy_ckks::{
    CKKSMeta,
    api::{CKKSAddOps, CKKSSubOps},
    test_suite::CKKSTestParams,
};
use poulpy_hal::layouts::{Backend, Module};
use std::cell::Cell;

thread_local! {
    static ADD_CONST_CALLS: Cell<usize> = const { Cell::new(0) };
    static SUB_CONST_CALLS: Cell<usize> = const { Cell::new(0) };
}

// Forward the unchanged operations explicitly so the defaults compose a backend
// that gives only constant-plaintext assignment a larger scratch requirement.
macro_rules! impl_carry_with_plaintext_workspace {
    ($verb:ident, $Impl:ident, $Reference:ident, $calls:ident, $workspace:expr) => {
        ::paste::paste! {
            unsafe impl ::poulpy_ckks::oep::$Impl for OverrideBackend {
                fn [<ckks_ $verb _tmp_bytes_impl>](module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                    ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _tmp_bytes_reference>](module, res_size)
                }

                fn [<ckks_ $verb _into_impl>]<Dst, A, B>(
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


                fn [<ckks_ $verb _assign_impl>]<Dst, A>(
                    module: &::poulpy_hal::layouts::Module<Self>,
                    dst: &mut Dst,
                    a: &A,
                    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
                ) -> ::poulpy_ckks::CKKSResult<()>
                where
                    Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSInfos + ::poulpy_ckks::SetCKKSInfos,
                    A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSInfos,
                {
                    ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _assign_reference>](module, dst, a, scratch)
                }



                fn [<ckks_ $verb _pt_vec_tmp_bytes_impl>](module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                    ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _pt_vec_tmp_bytes_reference>](module, res_size)
                }

                fn [<ckks_ $verb _pt_vec_into_impl>]<Dst, A, P>(
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


                fn [<ckks_ $verb _pt_vec_assign_impl>]<Dst, P>(
                    module: &::poulpy_hal::layouts::Module<Self>,
                    dst: &mut Dst,
                    pt: &P,
                    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
                ) -> ::poulpy_ckks::CKKSResult<()>
                where
                    Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                    P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
                {
                    ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _pt_vec_assign_reference>](module, dst, pt, scratch)
                }


                fn [<ckks_ $verb _pt_const_tmp_bytes_impl>](module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                    $workspace + ::poulpy_ckks::reference::$verb::$Reference::[<ckks_ $verb _pt_const_tmp_bytes_reference>](module, res_size)
                }

                fn [<ckks_ $verb _pt_const_into_impl>]<Dst, A, P>(
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


                fn [<ckks_ $verb _pt_const_assign_impl>]<Dst, P>(
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
impl_carry_with_plaintext_workspace!(add, CKKSAddImpl, CKKSAddReference, ADD_CONST_CALLS, 4096);
impl_carry_with_plaintext_workspace!(sub, CKKSSubImpl, CKKSSubReference, SUB_CONST_CALLS, 8192);

#[test]
fn unit_shifts_use_selected_plaintext_scratch() {
    let reference = Module::<FFT64Ref>::new(64);
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
        prec_meta: CKKSMeta {
            log_delta: 16,
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
    // and checks coefficient and metadata equality against the reference backend.
    poulpy_ckks::test_suite::parity::test_arithmetic_parity::<FFT64Ref, OverrideBackend, f64>(params, &reference, &module);
    assert!(ADD_CONST_CALLS.get() > 0);
    assert!(SUB_CONST_CALLS.get() > 0);
}
