use crate::CKKSResult as Result;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWE, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEInfos, GLWEToBackendMut,
    GLWEToBackendRef,
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Module, ScratchArena, Standard};

use crate::{CKKSCtBounds, SetCKKSInfos, layouts::CKKSCiphertext};

/// Backend contract of [`CKKSCIRingMapOps`](crate::api::CKKSCIRingMapOps), on the standard ring.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSCIRingMapImpl: Backend<Ring = Standard> {
    /// Scratch bound of [`Self::ckks_ci_embed`]; may report less than the reference, never less than the body takes.
    fn ckks_ci_embed_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos;

    /// Reproduces [`ckks_ci_embed_reference`](crate::reference::ring_map::CKKSCIRingMapReference::ckks_ci_embed_reference)
    /// bit for bit, within [`Self::ckks_ci_embed_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_ci_embed<Dst, D, K>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &CKKSCiphertext<D, Self::ZnxWord, ConjugateInvariant>,
        key: &GLWECIEmbedKeyPrepared<K, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        D: Data,
        GLWE<D, Self::ZnxWord>: GLWEToBackendRef<Self>,
        K: Data,
        GLWECIEmbedKeyPrepared<K, Self>: GGLWEPreparedToBackendRef<Self>;

    /// Scratch bound of [`Self::ckks_ci_trace`]; may report less than the reference, never less than the body takes.
    fn ckks_ci_trace_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    /// Reproduces [`ckks_ci_trace_reference`](crate::reference::ring_map::CKKSCIRingMapReference::ckks_ci_trace_reference)
    /// bit for bit, within [`Self::ckks_ci_trace_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_ci_trace<D, Src, K>(
        module: &Module<Self>,
        dst: &mut CKKSCiphertext<D, Self::ZnxWord, ConjugateInvariant>,
        src: &Src,
        key: &GLWECITraceKeyPrepared<K, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        D: Data,
        GLWE<D, Self::ZnxWord>: GLWEToBackendMut<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        K: Data,
        GLWECITraceKeyPrepared<K, Self>: GGLWEPreparedToBackendRef<Self>;
}

/// Selects the reference implementation of `CKKSCIRingMapImpl` for the standard backend `$be`.
#[macro_export]
macro_rules! impl_ckks_ci_ring_map_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSCIRingMapImpl for $be {
            fn ckks_ci_embed_tmp_bytes<R, K>(module: &::poulpy_hal::layouts::Module<Self>, res_infos: &R, key_infos: &K) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::ring_map::CKKSCIRingMapReference::ckks_ci_embed_tmp_bytes_reference(
                    module, res_infos, key_infos,
                )
            }

            fn ckks_ci_embed<Dst, D, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &$crate::layouts::CKKSCiphertext<
                    D,
                    <Self as ::poulpy_hal::layouts::Backend>::ZnxWord,
                    ::poulpy_hal::layouts::ConjugateInvariant,
                >,
                key: &::poulpy_core::layouts::GLWECIEmbedKeyPrepared<K, Self>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                D: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWE<D, <Self as ::poulpy_hal::layouts::Backend>::ZnxWord>:
                    ::poulpy_core::layouts::GLWEToBackendRef<Self>,
                K: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWECIEmbedKeyPrepared<K, Self>: ::poulpy_core::layouts::GGLWEPreparedToBackendRef<Self>,
            {
                $crate::reference::ring_map::CKKSCIRingMapReference::ckks_ci_embed_reference(module, dst, src, key, scratch)
            }

            fn ckks_ci_trace_tmp_bytes<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res_infos: &R,
                a_infos: &A,
                key_infos: &K,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::ring_map::CKKSCIRingMapReference::ckks_ci_trace_tmp_bytes_reference(
                    module, res_infos, a_infos, key_infos,
                )
            }

            fn ckks_ci_trace<D, Src, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut $crate::layouts::CKKSCiphertext<
                    D,
                    <Self as ::poulpy_hal::layouts::Backend>::ZnxWord,
                    ::poulpy_hal::layouts::ConjugateInvariant,
                >,
                src: &Src,
                key: &::poulpy_core::layouts::GLWECITraceKeyPrepared<K, Self>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                D: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWE<D, <Self as ::poulpy_hal::layouts::Backend>::ZnxWord>:
                    ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                K: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWECITraceKeyPrepared<K, Self>: ::poulpy_core::layouts::GGLWEPreparedToBackendRef<Self>,
            {
                $crate::reference::ring_map::CKKSCIRingMapReference::ckks_ci_trace_reference(module, dst, src, key, scratch)
            }
        }
    };
}

pub use crate::impl_ckks_ci_ring_map_reference_full;
