//! The CKKS addition and subtraction families, registered as `poulpy-ckks`
//! registers them (every method forwarding to its reference body) except for
//! the add-one and sub-one forms.
//!
//! Unit plaintexts are cached by radix, slot layout and scale, so warmed
//! operations issue no host transfer and can be recorded for replay.

/// `impl_ckks_add_reference!` with a `ckks_add_one_assign` that issues no host transfer.
macro_rules! impl_add_cached_one {
    ($be:ty) => {
        $crate::ckks_add_sub::impl_add_cached_one!($be, ::poulpy_hal::layouts::Standard);
        $crate::ckks_add_sub::impl_add_cached_one!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl ::poulpy_ckks::oep::CKKSAddImpl<$ring> for $be {
            fn ckks_add_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                ::poulpy_ckks::reference::add::CKKSAddReference::<$be, $ring>::ckks_add_tmp_bytes_reference(module, res_size)
            }

            fn ckks_add_into<Dst, A, B>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                b: &B,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_ckks::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + ::poulpy_ckks::CKKSCtBounds,
                B: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + ::poulpy_ckks::CKKSCtBounds,
            {
                ::poulpy_ckks::reference::add::CKKSAddReference::<$be, $ring>::ckks_add_into_reference(module, dst, a, b, scratch)
            }

            fn ckks_add_assign<Dst, A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_ckks::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + ::poulpy_ckks::CKKSCtBounds,
            {
                ::poulpy_ckks::reference::add::CKKSAddReference::<$be, $ring>::ckks_add_assign_reference(module, dst, a, scratch)
            }

            // The constant `1` comes from the module's cache instead of a host upload per call.
            fn ckks_add_one_assign<Dst>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_ckks::SetCKKSInfos,
            {
                let bytes = ::poulpy_ckks::api::CKKSAddOps::<$be, $ring>::ckks_add_one_tmp_bytes(
                    module,
                    ::poulpy_core::layouts::LWEInfos::max_size(dst),
                );
                ::poulpy_ckks::oep::check_unit_shift::<_, _, $ring>("ckks_add_one_assign", module, &*dst, scratch, bytes)?;
                let one = $crate::ckks_impl::one_plaintext::<_, $ring>(
                    module,
                    ::poulpy_core::layouts::LWEInfos::base2k(dst),
                    ::poulpy_ckks::GLWEPlaintextInfos::slot_layout(dst),
                    ::poulpy_ckks::CKKSInfos::log_delta(dst),
                )?;
                ::poulpy_ckks::api::CKKSAddOps::<$be, $ring>::ckks_add_assign(module, dst, &*one, scratch)
            }
        }
    };
}
pub(crate) use impl_add_cached_one;

/// `impl_ckks_sub_reference!` with a `ckks_sub_one_assign` that issues no host
/// transfer: the upstream default uploads its constant plaintext at every
/// call, which keeps every operation built on it (the modular reduction of a
/// bootstrap, for one) from being recorded.
macro_rules! impl_sub_cached_one {
    ($be:ty) => {
        $crate::ckks_add_sub::impl_sub_cached_one!($be, ::poulpy_hal::layouts::Standard);
        $crate::ckks_add_sub::impl_sub_cached_one!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl ::poulpy_ckks::oep::CKKSSubImpl<$ring> for $be {
            fn ckks_sub_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                ::poulpy_ckks::reference::sub::CKKSSubReference::<$be, $ring>::ckks_sub_tmp_bytes_reference(module, res_size)
            }

            fn ckks_sub_into<Dst, A, B>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                b: &B,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_ckks::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + ::poulpy_ckks::CKKSCtBounds,
                B: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + ::poulpy_ckks::CKKSCtBounds,
            {
                ::poulpy_ckks::reference::sub::CKKSSubReference::<$be, $ring>::ckks_sub_into_reference(module, dst, a, b, scratch)
            }

            fn ckks_sub_assign<Dst, A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_ckks::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + ::poulpy_ckks::CKKSCtBounds,
            {
                ::poulpy_ckks::reference::sub::CKKSSubReference::<$be, $ring>::ckks_sub_assign_reference(module, dst, a, scratch)
            }

            // The constant `1` comes from the module's cache instead of a host upload per call.
            fn ckks_sub_one_assign<Dst>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_ckks::SetCKKSInfos,
            {
                let bytes = ::poulpy_ckks::api::CKKSSubOps::<$be, $ring>::ckks_sub_one_tmp_bytes(
                    module,
                    ::poulpy_core::layouts::LWEInfos::max_size(dst),
                );
                ::poulpy_ckks::oep::check_unit_shift::<_, _, $ring>("ckks_sub_one_assign", module, &*dst, scratch, bytes)?;
                let one = $crate::ckks_impl::one_plaintext::<_, $ring>(
                    module,
                    ::poulpy_core::layouts::LWEInfos::base2k(dst),
                    ::poulpy_ckks::GLWEPlaintextInfos::slot_layout(dst),
                    ::poulpy_ckks::CKKSInfos::log_delta(dst),
                )?;
                ::poulpy_ckks::api::CKKSSubOps::<$be, $ring>::ckks_sub_assign(module, dst, &*one, scratch)
            }
        }
    };
}
pub(crate) use impl_sub_cached_one;
