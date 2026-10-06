use crate::CKKSResult as Result;

use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey,
    prepared::GLWEAutomorphismKeyPreparedBackendRef,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    oep::derived::rotate::{ckks_rotate_by_assign, ckks_rotate_by_into, ckks_rotate_by_tmp_bytes},
};

/// # Safety
/// Ciphertext outputs must reproduce the reference noise metadata and provenance
/// checks, including clearing invalidated estimates. Delegates only forward calls.
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSRotateImpl: Backend + super::CKKSCopyImpl {
    fn ckks_rotate_tmp_bytes_impl<C: GLWEInfos, K: GGLWEInfos>(module: &Module<Self>, ct_infos: &C, key_infos: &K) -> usize;

    fn ckks_rotate_into_impl<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEInfos + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds;

    fn ckks_rotate_assign_impl<Dst>(
        module: &Module<Self>,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEInfos + CKKSCtBounds + SetCKKSInfos;

    /// Scratch of a rotation by a slot shift: the keyed rotation or the identity copy.
    fn ckks_rotate_by_tmp_bytes_impl<C: CKKSCtBounds, K: GGLWEInfos>(
        module: &Module<Self>,
        ct_infos: &C,
        key_infos: &K,
    ) -> usize {
        ckks_rotate_by_tmp_bytes(module, ct_infos, key_infos)
    }

    /// Rotation by `k` slots through the key of `ckks_galois_element(k)`; a
    /// shift by a multiple of the slot count is a copy and needs no key.
    fn ckks_rotate_by_into_impl<Dst, Src, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        H: GetAutomorphismKey<Self>,
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        ckks_rotate_by_into(module, dst, src, k, keys, scratch)
    }

    /// In-place [`Self::ckks_rotate_by_into_impl`]; the identity leaves `dst` unchanged.
    fn ckks_rotate_by_assign_impl<Dst, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        H: GetAutomorphismKey<Self>,
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
    {
        ckks_rotate_by_assign(module, dst, k, keys, scratch)
    }
}

/// Implements this contract with the callable reference algorithms.
#[macro_export]
macro_rules! impl_ckks_rotate_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSRotateImpl for $be {
            fn ckks_rotate_tmp_bytes_impl<C: ::poulpy_core::layouts::GLWEInfos, K: ::poulpy_core::layouts::GGLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct_infos: &C,
                key_infos: &K,
            ) -> usize {
                $crate::reference::rotate::CKKSRotateReference::ckks_rotate_tmp_bytes_reference(module, ct_infos, key_infos)
            }

            fn ckks_rotate_into_impl<Dst, Src>(
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
                $crate::reference::rotate::CKKSRotateReference::ckks_rotate_into_reference(module, dst, src, key, scratch)
            }

            fn ckks_rotate_assign_impl<Dst>(
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
                $crate::reference::rotate::CKKSRotateReference::ckks_rotate_assign_reference(module, dst, key, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_rotate_reference;
