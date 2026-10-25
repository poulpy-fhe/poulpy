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
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSRingMapImpl: Backend<Ring = Standard> + GLWECIConversionImpl + GLWEKeyswitchImpl {
    /// Default body: `ckks_ci_embed_tmp_bytes_derived`, the selected `glwe_keyswitch_tmp_bytes` with `res_infos` as
    /// both layouts. Scratch bound of [`Self::ckks_ci_embed`]; may report less than the reference, never less than the
    /// body takes.
    fn ckks_ci_embed_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        ckks_ci_embed_tmp_bytes_derived(module, res_infos, key_infos)
    }

    /// Default body: `ckks_ci_embed_derived`. A key whose degree is not the standard degree, a degree or rank pair that
    /// is not a conjugate-invariant `N` and standard `2N`, a destination too narrow for the output width, or a radix
    /// differing from the input's is a refusal. Then the metadata stamp (real slots, the input's width),
    /// `glwe_ci_embed`, and `glwe_keyswitch_assign` under `key`.
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

    /// Default body: `ckks_ci_trace_tmp_bytes_derived`, the bytes of a copy shaped like `a_infos` plus the larger of
    /// the selected `glwe_keyswitch_tmp_bytes` at `a_infos` and the selected `glwe_ci_trace_tmp_bytes` from `a_infos`
    /// to `res_infos`. Scratch bound of [`Self::ckks_ci_trace`]; may report less than the reference, never less than
    /// the body takes.
    fn ckks_ci_trace_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        ckks_ci_trace_tmp_bytes_derived(module, res_infos, a_infos, key_infos)
    }

    /// Default body: `ckks_ci_trace_derived`. A key whose degree is not the standard degree, a degree or rank pair that
    /// is not a conjugate-invariant `N` and standard `2N`, a `src` without the one bit of budget the trace costs, or a
    /// destination too narrow for the output width is a refusal. Then `glwe_keyswitch` of `src` under `key` into a
    /// scratch copy, the metadata stamp (real slots, one bit less width and budget), and `glwe_ci_trace` into `dst`.
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
