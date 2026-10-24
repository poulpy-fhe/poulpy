use crate::CKKSResult as Result;

use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey,
    prepared::GLWEAutomorphismKeyPreparedBackendRef,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena, Standard};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSConjugateImpl: Backend<Ring = Standard> {
    fn ckks_conjugate_rotate_into<Dst, Src, H>(
        module: &Module<Self>,
        k: i64,
        dst: &mut Dst,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        super::derived::conjugate::ckks_conjugate_rotate_into_derived(module, k, dst, src, keys, scratch)
    }

    fn ckks_conjugate_assign<Dst, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        super::derived::conjugate::ckks_conjugate_assign_derived(module, dst, keys, scratch)
    }

    fn ckks_conjugate_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<Self>, ct_infos: &C, key_infos: &K) -> usize;

    /// Hook without an API twin: the conjugation under the automorphism key its caller looked up.
    fn ckks_conjugate_with_key_into<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEInfos + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds;

    /// In-place [`Self::ckks_conjugate_with_key_into`].
    fn ckks_conjugate_with_key_assign<Dst>(
        module: &Module<Self>,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos;
}

/// Selects the reference implementation of `CKKSConjugateImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_conjugate_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSConjugateImpl for $be {
            fn ckks_conjugate_tmp_bytes<C: ::poulpy_core::layouts::GLWEInfos, K: ::poulpy_core::layouts::GGLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct_infos: &C,
                key_infos: &K,
            ) -> usize {
                $crate::reference::conjugate::CKKSConjugateReference::ckks_conjugate_tmp_bytes_reference(
                    module, ct_infos, key_infos,
                )
            }

            fn ckks_conjugate_with_key_into<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                key: &::poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEInfos
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_core::layouts::GLWEInfos + $crate::CKKSCtBounds,
            {
                $crate::reference::conjugate::CKKSConjugateReference::ckks_conjugate_with_key_into_reference(
                    module, dst, src, key, scratch,
                )
            }

            fn ckks_conjugate_with_key_assign<Dst>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                key: &::poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
            {
                $crate::reference::conjugate::CKKSConjugateReference::ckks_conjugate_with_key_assign_reference(
                    module, dst, key, scratch,
                )
            }
        }
    };
}
pub use crate::impl_ckks_conjugate_reference_full;
