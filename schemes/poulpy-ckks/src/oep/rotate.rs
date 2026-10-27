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

/// Backend contract of [`CKKSRotateOps`](crate::api::CKKSRotateOps).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSRotateImpl: Backend + super::CKKSCopyImpl {
    /// Scratch bound of [`Self::ckks_rotate_with_key_into`] and [`Self::ckks_rotate_with_key_assign`]; may report less
    /// than the reference, never less than the body takes.
    fn ckks_rotate_with_key_tmp_bytes<C: CKKSCtBounds, K: GGLWEInfos>(
        module: &Module<Self>,
        ct_infos: &C,
        key_infos: &K,
    ) -> usize;

    /// Hook without an API twin. Reproduces
    /// [`ckks_rotate_with_key_into_reference`](crate::reference::rotate::CKKSRotateReference::ckks_rotate_with_key_into_reference)
    /// bit for bit, within [`Self::ckks_rotate_with_key_tmp_bytes`]; free in kernels and scratch use. The caller hands
    /// the automorphism key it looked up, and the hook stamps the output metadata as the reference does.
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

    /// Hook without an API twin. Reproduces
    /// [`ckks_rotate_with_key_assign_reference`](crate::reference::rotate::CKKSRotateReference::ckks_rotate_with_key_assign_reference)
    /// bit for bit, within [`Self::ckks_rotate_with_key_tmp_bytes`]; free in kernels and scratch use. The caller hands
    /// the automorphism key it looked up, and the hook stamps the output metadata as the reference does.
    fn ckks_rotate_with_key_assign<Dst>(
        module: &Module<Self>,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEInfos + CKKSCtBounds + SetCKKSInfos;

    /// Default body: `ckks_rotate_tmp_bytes_derived`, the larger of [`Self::ckks_rotate_with_key_tmp_bytes`] and the
    /// selected `ckks_copy_tmp_bytes` of `ct_infos` onto itself, the copy serving the identity shift. Bounds
    /// [`Self::ckks_rotate_into`] and [`Self::ckks_rotate_assign`].
    fn ckks_rotate_tmp_bytes<C: CKKSCtBounds, K: GGLWEInfos>(module: &Module<Self>, ct_infos: &C, key_infos: &K) -> usize {
        ckks_rotate_tmp_bytes_derived(module, ct_infos, key_infos)
    }

    /// Default body: `ckks_rotate_into_derived`. A shift whose Galois element `ckks_galois_element(k)` is `1` (a
    /// multiple of the slot count) is the selected `ckks_copy` then the noise record cleared, with no key lookup. Any
    /// other shift looks up the automorphism key of the element at `min(src.k(), dst.k())`, then
    /// [`Self::ckks_rotate_with_key_into`]. A missing key is `MissingAutomorphismKey`.
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

    /// Default body: `ckks_rotate_assign_derived`. A shift whose Galois element is `1` clears the noise record and does
    /// nothing else. Any other shift looks up the automorphism key of the element at `dst.k()`, then
    /// [`Self::ckks_rotate_with_key_assign`]. A missing key is `MissingAutomorphismKey`.
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
            fn ckks_rotate_with_key_tmp_bytes<C: $crate::CKKSCtBounds, K: ::poulpy_core::layouts::GGLWEInfos>(
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
