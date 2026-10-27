//! The CKKS addition and subtraction families, registered as `poulpy-ckks`
//! registers them (every method forwarding to its reference body) except for
//! the add-one and sub-one forms.
//!
//! The upstream defaults of those two upload their constant plaintext at
//! every call. On a device backend that is a host transfer inside every
//! operation built on them (the modular reduction of a bootstrap, for one),
//! which also keeps such an operation from being recorded as a graph. Here
//! the constant is built once per radix and kept by the module
//! (`ckks_impl::one_plaintext`).

/// `impl_ckks_add_reference!` with a `ckks_add_one_assign` that issues no host transfer.
macro_rules! impl_add_cached_one {
    ($be:ty) => {
        unsafe impl ::poulpy_ckks::oep::CKKSAddImpl for $be {
            fn ckks_add_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                ::poulpy_ckks::reference::add::CKKSAddReference::ckks_add_tmp_bytes_reference(module, res_size)
            }

            fn ckks_add_into<Dst, A, B>(
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
                ::poulpy_ckks::reference::add::CKKSAddReference::ckks_add_into_reference(module, dst, a, b, scratch)
            }

            fn ckks_add_assign<Dst, A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSInfos + ::poulpy_ckks::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSInfos,
            {
                ::poulpy_ckks::reference::add::CKKSAddReference::ckks_add_assign_reference(module, dst, a, scratch)
            }

            fn ckks_add_pt_vec_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                ::poulpy_ckks::reference::add::CKKSAddReference::ckks_add_pt_vec_tmp_bytes_reference(module, res_size)
            }

            fn ckks_add_pt_vec_into<Dst, A, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                pt: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_core::layouts::IntPolyInfos,
            {
                ::poulpy_ckks::reference::add::CKKSAddReference::ckks_add_pt_vec_into_reference(module, dst, a, pt, scratch)
            }

            fn ckks_add_pt_vec_assign<Dst, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                pt: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_core::layouts::IntPolyInfos,
            {
                ::poulpy_ckks::reference::add::CKKSAddReference::ckks_add_pt_vec_assign_reference(module, dst, pt, scratch)
            }

            fn ckks_add_pt_const_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                ::poulpy_ckks::reference::add::CKKSAddReference::ckks_add_pt_const_tmp_bytes_reference(module, res_size)
            }

            fn ckks_add_pt_const_into<Dst, A, P>(
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
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_core::layouts::IntPolyInfos,
            {
                ::poulpy_ckks::reference::add::CKKSAddReference::ckks_add_pt_const_into_reference(
                    module, dst, dst_coeff, a, pt, pt_coeff, scratch,
                )
            }

            fn ckks_add_pt_const_assign<Dst, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                dst_coeff: usize,
                pt: &P,
                pt_coeff: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_core::layouts::IntPolyInfos,
            {
                ::poulpy_ckks::reference::add::CKKSAddReference::ckks_add_pt_const_assign_reference(
                    module, dst, dst_coeff, pt, pt_coeff, scratch,
                )
            }

            // The constant `1` comes from the module's cache instead of a host upload per call.
            fn ckks_add_one_assign<Dst>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
            {
                let one = $crate::ckks_impl::one_plaintext(module, ::poulpy_core::layouts::LWEInfos::base2k(dst))?;
                ::poulpy_ckks::api::CKKSAddOps::ckks_add_pt_const_assign(module, dst, 0, &*one, 0, scratch)
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
        unsafe impl ::poulpy_ckks::oep::CKKSSubImpl for $be {
            fn ckks_sub_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                ::poulpy_ckks::reference::sub::CKKSSubReference::ckks_sub_tmp_bytes_reference(module, res_size)
            }

            fn ckks_sub_into<Dst, A, B>(
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
                ::poulpy_ckks::reference::sub::CKKSSubReference::ckks_sub_into_reference(module, dst, a, b, scratch)
            }

            fn ckks_sub_assign<Dst, A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSInfos + ::poulpy_ckks::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSInfos,
            {
                ::poulpy_ckks::reference::sub::CKKSSubReference::ckks_sub_assign_reference(module, dst, a, scratch)
            }

            fn ckks_sub_pt_vec_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                ::poulpy_ckks::reference::sub::CKKSSubReference::ckks_sub_pt_vec_tmp_bytes_reference(module, res_size)
            }

            fn ckks_sub_pt_vec_into<Dst, A, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                pt: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_core::layouts::IntPolyInfos,
            {
                ::poulpy_ckks::reference::sub::CKKSSubReference::ckks_sub_pt_vec_into_reference(module, dst, a, pt, scratch)
            }

            fn ckks_sub_pt_vec_assign<Dst, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                pt: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_core::layouts::IntPolyInfos,
            {
                ::poulpy_ckks::reference::sub::CKKSSubReference::ckks_sub_pt_vec_assign_reference(module, dst, pt, scratch)
            }

            fn ckks_sub_pt_const_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                ::poulpy_ckks::reference::sub::CKKSSubReference::ckks_sub_pt_const_tmp_bytes_reference(module, res_size)
            }

            fn ckks_sub_pt_const_into<Dst, A, P>(
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
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_core::layouts::IntPolyInfos,
            {
                ::poulpy_ckks::reference::sub::CKKSSubReference::ckks_sub_pt_const_into_reference(
                    module, dst, dst_coeff, a, pt, pt_coeff, scratch,
                )
            }

            fn ckks_sub_pt_const_assign<Dst, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                dst_coeff: usize,
                pt: &P,
                pt_coeff: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_core::layouts::IntPolyInfos,
            {
                ::poulpy_ckks::reference::sub::CKKSSubReference::ckks_sub_pt_const_assign_reference(
                    module, dst, dst_coeff, pt, pt_coeff, scratch,
                )
            }

            // The constant `1` comes from the module's cache instead of a host upload per call.
            fn ckks_sub_one_assign<Dst>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
            {
                let one = $crate::ckks_impl::one_plaintext(module, ::poulpy_core::layouts::LWEInfos::base2k(dst))?;
                ::poulpy_ckks::api::CKKSSubOps::ckks_sub_pt_const_assign(module, dst, 0, &*one, 0, scratch)
            }
        }
    };
}
pub(crate) use impl_sub_cached_one;
