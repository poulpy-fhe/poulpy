use crate::CKKSResult as Result;

use poulpy_core::layouts::GLWE;
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos,
    layouts::{CKKSCiphertext, UnnormalizedCKKSCiphertext, ciphertext::UnnormalizedCKKSCiphertextRefMut},
    oep::carry_verb::ckks_carry_verb_oep,
};

ckks_carry_verb_oep! {
    verb: sub,
    doc_verb: "subtraction",
    impl_trait: CKKSSubImpl,
}

/// Implements the sub contract with the callable reference algorithms.
#[macro_export]
macro_rules! impl_ckks_sub_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSSubImpl for $be {
            fn ckks_sub_tmp_bytes_impl(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                $crate::reference::sub::CKKSSubReference::ckks_sub_tmp_bytes_reference(module, res_size)
            }

            fn ckks_sub_into_impl<Dst, A, B>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                b: &B,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                B: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::sub::CKKSSubReference::ckks_sub_into_reference(module, dst, a, b, scratch)
            }

            fn ckks_sub_into_unnormalized_impl<Dst, A, B>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut $crate::layouts::UnnormalizedCKKSCiphertext<Dst, Self::ZnxWord>,
                a: &A,
                b: &B,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWE<Dst, Self::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                B: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::sub::ckks_sub_into_unnormalized_wrapped_reference(module, dst, a, b, scratch)
            }

            fn ckks_sub_assign_impl<Dst, A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSInfos + $crate::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos,
            {
                $crate::reference::sub::CKKSSubReference::ckks_sub_assign_reference(module, dst, a, scratch)
            }

            fn ckks_sub_assign_unnormalized_impl<Dst, A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut $crate::layouts::UnnormalizedCKKSCiphertext<Dst, Self::ZnxWord>,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWE<Dst, Self::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos,
            {
                $crate::reference::sub::ckks_sub_assign_unnormalized_wrapped_reference(module, dst, a, scratch)
            }

            fn ckks_sub_assign_unnormalized_ref_impl<Dst, A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut $crate::layouts::ciphertext::UnnormalizedCKKSCiphertextRefMut<'_, Dst, Self::ZnxWord>,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_hal::layouts::Data,
                $crate::layouts::CKKSCiphertext<Dst, Self::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos,
            {
                $crate::reference::sub::ckks_sub_assign_unnormalized_ref_wrapped_reference(module, dst, a, scratch)
            }

            fn ckks_sub_pt_vec_tmp_bytes_impl(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                $crate::reference::sub::CKKSSubReference::ckks_sub_pt_vec_tmp_bytes_reference(module, res_size)
            }

            fn ckks_sub_pt_vec_into_impl<Dst, A, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                pt: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
            {
                $crate::reference::sub::CKKSSubReference::ckks_sub_pt_vec_into_reference(module, dst, a, pt, scratch)
            }

            fn ckks_sub_pt_vec_into_unnormalized_impl<Dst, A, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut $crate::layouts::UnnormalizedCKKSCiphertext<Dst, Self::ZnxWord>,
                a: &A,
                pt: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWE<Dst, Self::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
            {
                $crate::reference::sub::ckks_sub_pt_vec_into_unnormalized_wrapped_reference(module, dst, a, pt, scratch)
            }

            fn ckks_sub_pt_vec_assign_impl<Dst, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                pt: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
            {
                $crate::reference::sub::CKKSSubReference::ckks_sub_pt_vec_assign_reference(module, dst, pt, scratch)
            }

            fn ckks_sub_pt_vec_assign_unnormalized_impl<Dst, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut $crate::layouts::UnnormalizedCKKSCiphertext<Dst, Self::ZnxWord>,
                pt: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWE<Dst, Self::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
            {
                $crate::reference::sub::ckks_sub_pt_vec_assign_unnormalized_wrapped_reference(module, dst, pt, scratch)
            }

            fn ckks_sub_pt_const_tmp_bytes_impl(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                $crate::reference::sub::CKKSSubReference::ckks_sub_pt_const_tmp_bytes_reference(module, res_size)
            }

            fn ckks_sub_pt_const_into_impl<Dst, A, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                dst_coeff: usize,
                pt: &P,
                pt_coeff: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
            {
                $crate::reference::sub::CKKSSubReference::ckks_sub_pt_const_into_reference(
                    module, dst, a, dst_coeff, pt, pt_coeff, scratch,
                )
            }

            fn ckks_sub_pt_const_into_unnormalized_impl<Dst, A, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut $crate::layouts::UnnormalizedCKKSCiphertext<Dst, Self::ZnxWord>,
                a: &A,
                dst_coeff: usize,
                pt: &P,
                pt_coeff: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWE<Dst, Self::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
            {
                $crate::reference::sub::ckks_sub_pt_const_into_unnormalized_wrapped_reference(
                    module, dst, a, dst_coeff, pt, pt_coeff, scratch,
                )
            }

            fn ckks_sub_pt_const_assign_impl<Dst, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                dst_coeff: usize,
                pt: &P,
                pt_coeff: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
            {
                $crate::reference::sub::CKKSSubReference::ckks_sub_pt_const_assign_reference(
                    module, dst, dst_coeff, pt, pt_coeff, scratch,
                )
            }

            fn ckks_sub_pt_const_assign_unnormalized_impl<Dst, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut $crate::layouts::UnnormalizedCKKSCiphertext<Dst, Self::ZnxWord>,
                dst_coeff: usize,
                pt: &P,
                pt_coeff: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWE<Dst, Self::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
            {
                $crate::reference::sub::ckks_sub_pt_const_assign_unnormalized_wrapped_reference(
                    module, dst, dst_coeff, pt, pt_coeff, scratch,
                )
            }
        }
    };
}
pub use crate::impl_ckks_sub_reference;
