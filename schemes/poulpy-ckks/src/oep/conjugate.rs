use crate::CKKSResult as Result;

use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey,
    prepared::GLWEAutomorphismKeyPreparedBackendRef,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena, Standard};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Backend contract of [`CKKSConjugateOps`](crate::api::CKKSConjugateOps), on the standard ring.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSConjugateImpl: Backend<Ring = Standard> {
    /// Default body: `ckks_conjugate_into_derived`, the selected `ckks_conjugate_rotate_into` with `k = 0`, within
    /// [`Self::ckks_conjugate_tmp_bytes`].
    fn ckks_conjugate_into<Dst, Src, H>(
        module: &Module<Self>,
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
        super::derived::conjugate::ckks_conjugate_into_derived(module, dst, src, keys, scratch)
    }

    /// Default body: `ckks_conjugate_rotate_into_derived`, the lookup of the automorphism key of the conjugate-rotate
    /// Galois element of `k` at the switch width `min(src.k(), dst.k())`, then [`Self::ckks_conjugate_with_key_into`]
    /// with it. A missing key is `MissingAutomorphismKey`.
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

    /// Default body: `ckks_conjugate_assign_derived`, the lookup of the automorphism key of Galois element `-1` at
    /// `dst.k()`, then [`Self::ckks_conjugate_with_key_assign`] with it. A missing key is `MissingAutomorphismKey`.
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

    /// Scratch bound of [`Self::ckks_conjugate_with_key_into`] and [`Self::ckks_conjugate_with_key_assign`]; may report
    /// less than the reference, never less than the body takes.
    fn ckks_conjugate_with_key_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(
        module: &Module<Self>,
        ct_infos: &C,
        key_infos: &K,
    ) -> usize;

    /// Default body: `ckks_conjugate_tmp_bytes_derived`, [`Self::ckks_conjugate_with_key_tmp_bytes`]. Bounds
    /// [`Self::ckks_conjugate_into`], [`Self::ckks_conjugate_rotate_into`] and [`Self::ckks_conjugate_assign`].
    fn ckks_conjugate_tmp_bytes<C: CKKSCtBounds, K: GGLWEInfos>(module: &Module<Self>, ct_infos: &C, key_infos: &K) -> usize {
        super::derived::conjugate::ckks_conjugate_tmp_bytes_derived(module, ct_infos, key_infos)
    }

    /// Hook without an API twin. Reproduces
    /// [`ckks_conjugate_with_key_into_reference`](crate::reference::conjugate::CKKSConjugateReference::ckks_conjugate_with_key_into_reference)
    /// bit for bit, within [`Self::ckks_conjugate_with_key_tmp_bytes`]; free in kernels and scratch use. The caller hands the
    /// automorphism key it looked up, and the hook stamps the output metadata as the reference does.
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

    /// Hook without an API twin. Reproduces
    /// [`ckks_conjugate_with_key_assign_reference`](crate::reference::conjugate::CKKSConjugateReference::ckks_conjugate_with_key_assign_reference)
    /// bit for bit, within [`Self::ckks_conjugate_with_key_tmp_bytes`]; free in kernels and scratch use. The caller hands the
    /// automorphism key it looked up, and the hook stamps the output metadata as the reference does.
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
            fn ckks_conjugate_with_key_tmp_bytes<C: ::poulpy_core::layouts::GLWEInfos, K: ::poulpy_core::layouts::GGLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct_infos: &C,
                key_infos: &K,
            ) -> usize {
                $crate::reference::conjugate::CKKSConjugateReference::ckks_conjugate_with_key_tmp_bytes_reference(
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
