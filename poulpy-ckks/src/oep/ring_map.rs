use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Module, ScratchArena, Standard};

use crate::{CKKSCtBounds, SetCKKSInfos, layouts::CKKSCiphertext};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSCIRingMapImpl: Backend<Ring = ConjugateInvariant> {
    fn ckks_ci_unfold_impl<D, Src>(
        module: &Module<Self>,
        dst: &mut CKKSCiphertext<D, Self::ZnxWord, Standard>,
        src: &Src,
    ) -> Result<()>
    where
        D: Data,
        GLWE<D, Self::ZnxWord>: GLWEToBackendMut<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;

    fn ckks_ci_fold_tmp_bytes_impl<R, A>(module: &Module<Self>, res_infos: &R, a_infos: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    fn ckks_ci_fold_impl<Dst, D>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &CKKSCiphertext<D, Self::ZnxWord, Standard>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        D: Data,
        GLWE<D, Self::ZnxWord>: GLWEToBackendRef<Self>;
}

/// Implements this contract with the callable reference algorithms.
#[macro_export]
macro_rules! impl_ckks_ci_ring_map_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSCIRingMapImpl for $be {
            fn ckks_ci_unfold_impl<D, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut $crate::layouts::CKKSCiphertext<
                    D,
                    <Self as ::poulpy_hal::layouts::Backend>::ZnxWord,
                    ::poulpy_hal::layouts::Standard,
                >,
                src: &Src,
            ) -> $crate::CKKSResult<()>
            where
                D: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWE<D, <Self as ::poulpy_hal::layouts::Backend>::ZnxWord>:
                    ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::ring_map::ckks_ci_unfold_reference(module, dst, src)
            }

            fn ckks_ci_fold_tmp_bytes_impl<R, A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res_infos: &R,
                a_infos: &A,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                $crate::reference::ring_map::ckks_ci_fold_tmp_bytes_reference(module, res_infos, a_infos)
            }

            fn ckks_ci_fold_impl<Dst, D>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &$crate::layouts::CKKSCiphertext<
                    D,
                    <Self as ::poulpy_hal::layouts::Backend>::ZnxWord,
                    ::poulpy_hal::layouts::Standard,
                >,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                D: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWE<D, <Self as ::poulpy_hal::layouts::Backend>::ZnxWord>:
                    ::poulpy_core::layouts::GLWEToBackendRef<Self>,
            {
                $crate::reference::ring_map::ckks_ci_fold_reference(module, dst, src, scratch)
            }
        }
    };
}

pub use crate::impl_ckks_ci_ring_map_reference;
