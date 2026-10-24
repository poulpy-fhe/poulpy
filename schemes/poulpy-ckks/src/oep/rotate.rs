use crate::CKKSResult as Result;

use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey,
    prepared::GLWEAutomorphismKeyPreparedBackendRef,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    oep::derived::rotate::{ckks_rotate_assign_derived, ckks_rotate_into_derived, ckks_rotate_tmp_bytes_derived},
};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSRotateImpl: Backend + super::CKKSCopyImpl {
    /// Scratch of [`Self::ckks_rotate_with_key_into`] and [`Self::ckks_rotate_with_key_assign`].
    fn ckks_rotate_with_key_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<Self>, ct_infos: &C, key_infos: &K) -> usize;

    /// Hook without an API twin: the rotation under the automorphism key its caller looked up.
    fn ckks_rotate_with_key_into<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEInfos + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds;

    /// In-place [`Self::ckks_rotate_with_key_into`].
    fn ckks_rotate_with_key_assign<Dst>(
        module: &Module<Self>,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEInfos + CKKSCtBounds + SetCKKSInfos;

    /// Scratch of a rotation by a slot shift: the keyed rotation or the identity copy.
    fn ckks_rotate_tmp_bytes<C: CKKSCtBounds, K: GGLWEInfos>(module: &Module<Self>, ct_infos: &C, key_infos: &K) -> usize {
        ckks_rotate_tmp_bytes_derived(module, ct_infos, key_infos)
    }

    /// Rotation by `k` slots through the key of `ckks_galois_element(k)`; a
    /// shift by a multiple of the slot count is a copy and needs no key.
    fn ckks_rotate_into<Dst, Src, H>(
        module: &Module<Self>,
        k: i64,
        dst: &mut Dst,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        H: GetAutomorphismKey<Self>,
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        ckks_rotate_into_derived(module, k, dst, src, keys, scratch)
    }

    /// In-place [`Self::ckks_rotate_into`]; the identity leaves `dst` unchanged.
    fn ckks_rotate_assign<Dst, H>(
        module: &Module<Self>,
        k: i64,
        dst: &mut Dst,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        H: GetAutomorphismKey<Self>,
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
    {
        ckks_rotate_assign_derived(module, k, dst, keys, scratch)
    }
}

/// Selects the reference implementation of `CKKSRotateImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_rotate_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSRotateImpl for $be {
            fn ckks_rotate_with_key_tmp_bytes<C: ::poulpy_core::layouts::GLWEInfos, K: ::poulpy_core::layouts::GGLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct_infos: &C,
                key_infos: &K,
            ) -> usize {
                $crate::reference::rotate::CKKSRotateReference::ckks_rotate_with_key_tmp_bytes_reference(
                    module, ct_infos, key_infos,
                )
            }

            fn ckks_rotate_with_key_into<Dst, Src>(
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
                $crate::reference::rotate::CKKSRotateReference::ckks_rotate_with_key_into_reference(
                    module, dst, src, key, scratch,
                )
            }

            fn ckks_rotate_with_key_assign<Dst>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                key: &::poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEInfos
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos,
            {
                $crate::reference::rotate::CKKSRotateReference::ckks_rotate_with_key_assign_reference(module, dst, key, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_rotate_reference_full;
