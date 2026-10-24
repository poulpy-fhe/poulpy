//! Conjugate-invariant ring maps: a derived standard-only family over the core maps and key switch.
use crate::CKKSResult as Result;
use poulpy_core::{
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWE, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEInfos, GLWEToBackendMut,
        GLWEToBackendRef,
    },
    oep::{GLWECIConversionImpl, GLWEKeyswitchImpl},
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Module, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    layouts::CKKSCiphertext,
    oep::derived::ring_map::{
        ckks_ci_embed_derived, ckks_ci_embed_tmp_bytes_derived, ckks_ci_trace_derived, ckks_ci_trace_tmp_bytes_derived,
    },
};

/// Backend contract of [`CKKSCIRingMapOps`](crate::api::CKKSCIRingMapOps), on the standard ring.
///
/// Every method defaults to the core keyless map (`glwe_ci_embed`, `glwe_ci_trace`)
/// composed with the selected GLWE key switch; an override reproduces it bit for bit
/// within its own query.
///
/// # Safety
///
/// Implementations uphold `poulpy-hal/docs/backend_safety_contract.md` for every
/// operand and scratch region they touch.
pub unsafe trait CKKSRingMapImpl: Backend<Ring = Standard> + GLWECIConversionImpl + GLWEKeyswitchImpl {
    /// Scratch of [`Self::ckks_ci_embed`]: the selected key switch query at `res_infos`.
    fn ckks_ci_embed_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        ckks_ci_embed_tmp_bytes_derived(module, res_infos, key_infos)
    }

    /// Checks the layouts, stamps the metadata, `glwe_ci_embed`, then `glwe_keyswitch_assign` under `key`.
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
        GLWECIEmbedKeyPrepared<K, Self>: GGLWEPreparedToBackendRef<Self>,
    {
        ckks_ci_embed_derived(module, dst, src, key, scratch)
    }

    /// Scratch of [`Self::ckks_ci_trace`]: the switched copy of `a_infos` plus the larger of the key switch and
    /// `glwe_ci_trace` queries.
    fn ckks_ci_trace_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        ckks_ci_trace_tmp_bytes_derived(module, res_infos, a_infos, key_infos)
    }

    /// Checks the layouts, `glwe_keyswitch` of `src` under `key` into a scratch copy, then `glwe_ci_trace` into `dst`.
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
        GLWECITraceKeyPrepared<K, Self>: GGLWEPreparedToBackendRef<Self>,
    {
        ckks_ci_trace_derived(module, dst, src, key, scratch)
    }
}

/// Selects the derived defaults of `CKKSRingMapImpl` for the standard backend `$be`.
#[macro_export]
macro_rules! impl_ckks_ring_map_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSRingMapImpl for $be {}
    };
}

pub use crate::impl_ckks_ring_map_derived_full;
