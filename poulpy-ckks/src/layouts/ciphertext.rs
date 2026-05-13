//! CKKS metadata attached to ciphertext storage.
//!
//! A CKKS ciphertext is represented as [`CKKSCiphertext<D>`], a thin wrapper
//! over `poulpy-core`'s `GLWE<D, CKKS>`.

use std::{
    fmt,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use anyhow::Result;
use poulpy_core::layouts::{Base2K, Degree, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GLWEViewMut, LWEInfos, Rank};
use poulpy_core::{GLWENormalize, ScratchArenaTakeCore};
use poulpy_hal::layouts::{Backend, Data, HostBackend, HostDataRef, Module, ScratchArena};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos, error::CKKSCompositionError, layouts::CKKSModuleAlloc};

mod sealed {
    pub trait Sealed {}
}

/// Marker for CKKS ciphertexts whose limb digits are carry-normalized.
pub struct Normalized;

/// Marker for CKKS ciphertexts whose limb digits may contain unpropagated carries.
pub struct Unnormalized;

impl sealed::Sealed for Normalized {}
impl sealed::Sealed for Unnormalized {}

/// Sealed CKKS ciphertext normalization state.
pub trait CKKSNormalizationState: sealed::Sealed {}

impl CKKSNormalizationState for Normalized {}
impl CKKSNormalizationState for Unnormalized {}

/// CKKS ciphertext storage plus semantic precision metadata.
///
/// `inner` contains the raw GLWE torus digits while `meta` describes the
/// semantic decimal scaling and remaining homomorphic capacity of the value.
pub struct CKKSCiphertext<D: Data, S: CKKSNormalizationState = Normalized> {
    /// Raw GLWE ciphertext storage.
    pub(crate) inner: GLWE<D>,
    /// Semantic CKKS metadata associated with `inner`.
    pub(crate) meta: CKKSMeta,
    _state: PhantomData<S>,
}

impl<D: Data, S: CKKSNormalizationState> CKKSCiphertext<D, S> {
    pub(crate) fn from_inner(inner: GLWE<D>, meta: CKKSMeta) -> Self {
        Self {
            inner,
            meta,
            _state: PhantomData,
        }
    }

    /// Rebuilds this backend-owned ciphertext as a host-owned [`CKKSCiphertext<Vec<u8>>`].
    pub fn to_host_owned<BE>(&self) -> CKKSCiphertext<Vec<u8>, S>
    where
        BE: Backend<OwnedBuf = D>,
    {
        CKKSCiphertext::<Vec<u8>, S>::from_inner(self.inner.to_host_owned::<BE>(), self.meta)
    }

    /// Formats this backend-owned ciphertext through the existing host [`fmt::Display`] implementation.
    pub fn display_host<BE>(&self) -> String
    where
        BE: Backend<OwnedBuf = D>,
    {
        self.to_host_owned::<BE>().to_string()
    }

    pub fn to_ref<BE: Backend>(&self) -> GLWE<BE::BufRef<'_>>
    where
        GLWE<D>: GLWEToBackendRef<BE>,
    {
        GLWEToBackendRef::to_backend_ref(&self.inner)
    }

    pub fn to_mut<BE: Backend>(&mut self) -> GLWE<BE::BufMut<'_>>
    where
        GLWE<D>: GLWEToBackendMut<BE>,
    {
        GLWEToBackendMut::to_backend_mut(&mut self.inner)
    }

    /// Replaces the semantic metadata after checking that the current storage
    /// can represent it.
    ///
    /// This is intended for callers that build ciphertext buffers manually.
    /// Normal CKKS operations update metadata themselves.
    pub fn set_meta_checked(&mut self, meta: CKKSMeta) -> Result<()> {
        anyhow::ensure!(
            meta.effective_k() <= self.max_k().as_usize(),
            CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
                max_k: self.max_k().as_usize(),
                log_delta: meta.log_delta(),
                base2k: self.base2k().as_usize(),
                requested_limbs: self.size(),
            }
        );
        self.meta = meta;
        Ok(())
    }
}

impl<D: Data, S: CKKSNormalizationState> Deref for CKKSCiphertext<D, S> {
    type Target = GLWE<D>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: Data, S: CKKSNormalizationState> DerefMut for CKKSCiphertext<D, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: Data, S: CKKSNormalizationState> LWEInfos for CKKSCiphertext<D, S> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}

impl<D: Data, S: CKKSNormalizationState> GLWEInfos for CKKSCiphertext<D, S> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data, S: CKKSNormalizationState> CKKSInfos for CKKSCiphertext<D, S> {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }

    fn log_delta(&self) -> usize {
        self.meta.log_delta()
    }

    fn log_budget(&self) -> usize {
        self.meta.log_budget()
    }
}

impl<D: Data, S: CKKSNormalizationState> SetCKKSInfos for CKKSCiphertext<D, S> {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.meta = meta;
    }
}

impl<D: HostDataRef, S: CKKSNormalizationState> fmt::Display for CKKSCiphertext<D, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl<BE: Backend, D: Data, S: CKKSNormalizationState> GLWEToBackendRef<BE> for CKKSCiphertext<D, S>
where
    GLWE<D>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>> {
        GLWEToBackendRef::to_backend_ref(&self.inner)
    }
}

impl<BE: Backend, D: Data, S: CKKSNormalizationState> GLWEToBackendMut<BE> for CKKSCiphertext<D, S>
where
    GLWE<D>: GLWEToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>> {
        GLWEToBackendMut::to_backend_mut(&mut self.inner)
    }
}

/// Scratch-backed mutable CKKS ciphertext view.
///
/// This is the CKKS analogue of core's [`GLWEViewMut`]: the limb storage is
/// borrowed from a [`ScratchArena`] in the backend-native buffer type, while the
/// CKKS semantic metadata is carried alongside the GLWE view.
pub struct CKKSCiphertextViewMut<'a, BE: Backend + 'a> {
    inner: GLWEViewMut<'a, BE>,
    meta: CKKSMeta,
}

impl<'a, BE: Backend + 'a> CKKSCiphertextViewMut<'a, BE> {
    pub(crate) fn from_inner(inner: GLWEViewMut<'a, BE>, meta: CKKSMeta) -> Self {
        Self { inner, meta }
    }

    #[allow(dead_code)]
    pub(crate) fn into_inner(self) -> GLWEViewMut<'a, BE> {
        self.inner
    }
}

impl<'a, BE: Backend + 'a> Deref for CKKSCiphertextViewMut<'a, BE> {
    type Target = GLWEViewMut<'a, BE>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, BE: Backend + 'a> DerefMut for CKKSCiphertextViewMut<'a, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a, BE: Backend + 'a> LWEInfos for CKKSCiphertextViewMut<'a, BE> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}

impl<'a, BE: Backend + 'a> GLWEInfos for CKKSCiphertextViewMut<'a, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<'a, BE: Backend + 'a> CKKSInfos for CKKSCiphertextViewMut<'a, BE> {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }

    fn log_delta(&self) -> usize {
        self.meta.log_delta()
    }

    fn log_budget(&self) -> usize {
        self.meta.log_budget()
    }
}

impl<'a, BE: Backend + 'a> SetCKKSInfos for CKKSCiphertextViewMut<'a, BE> {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.meta = meta;
    }
}

impl<'a, BE: Backend + 'a> GLWEToBackendRef<BE> for CKKSCiphertextViewMut<'a, BE> {
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>> {
        self.inner.to_backend_ref()
    }
}

impl<'a, BE: Backend + 'a> GLWEToBackendMut<BE> for CKKSCiphertextViewMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>> {
        self.inner.to_backend_mut()
    }
}

/// CKKS layout carving helpers for backend-native scratch arenas.
pub trait ScratchArenaTakeCKKS<'a, BE: Backend>: ScratchArenaTakeCore<'a, BE> + Sized {
    fn take_ckks_ciphertext_scratch<I>(self, infos: &I, meta: CKKSMeta) -> (CKKSCiphertextViewMut<'a, BE>, Self)
    where
        BE: 'a,
        I: GLWEInfos,
    {
        let (inner, scratch) = self.take_glwe_scratch(infos);
        (CKKSCiphertextViewMut::from_inner(inner, meta), scratch)
    }

    fn take_ckks_ciphertext_like_scratch<C>(self, ct: &C) -> (CKKSCiphertextViewMut<'a, BE>, Self)
    where
        BE: 'a,
        C: GLWEInfos + CKKSInfos,
    {
        self.take_ckks_ciphertext_scratch(ct, ct.meta())
    }

    fn take_unnormalized_ckks_ciphertext_scratch<I>(
        self,
        infos: &I,
        meta: CKKSMeta,
    ) -> (UnnormalizedCKKSCiphertext<BE::BufMut<'a>>, Self)
    where
        BE: 'a,
        I: GLWEInfos,
    {
        let (inner, scratch) = self.take_glwe_scratch(infos);
        (UnnormalizedCKKSCiphertext::from_inner(inner.into_inner(), meta), scratch)
    }

    fn take_unnormalized_ckks_ciphertext_like_scratch<C>(self, ct: &C) -> (UnnormalizedCKKSCiphertext<BE::BufMut<'a>>, Self)
    where
        BE: 'a,
        C: GLWEInfos + CKKSInfos,
    {
        self.take_unnormalized_ckks_ciphertext_scratch(ct, ct.meta())
    }
}

impl<'a, BE, T> ScratchArenaTakeCKKS<'a, BE> for T
where
    BE: Backend + 'a,
    T: ScratchArenaTakeCore<'a, BE>,
{
}

/// Maintenance operations for resizing ciphertext limb storage.
pub trait CKKSMaintainOps {
    /// Reallocates the owned backing buffer to exactly `size` limbs.
    ///
    /// Inputs:
    /// - `ct`: ciphertext whose owned limb buffer should be resized
    /// - `size`: requested number of limbs
    ///
    /// Output:
    /// - returns `Ok(())` after resizing `ct`
    ///
    /// Behavior:
    /// - preserves ciphertext metadata
    /// - rejects shrink operations that would make the buffer too small for the
    ///   current semantic precision
    ///
    /// Errors:
    /// - `LimbReallocationShrinksBelowMetadata` if the requested limb count
    ///   cannot represent the current metadata
    fn ckks_reallocate_limbs_checked(&self, ct: &mut CKKSCiphertext<Vec<u8>>, size: usize) -> Result<()>;

    /// Shrinks an owned ciphertext buffer to the minimum limb count that still
    /// preserves its current metadata.
    ///
    /// Inputs:
    /// - `ct`: ciphertext whose limb storage should be compacted
    ///
    /// Output:
    /// - returns `Ok(())` after compacting `ct`
    ///
    /// Errors:
    /// - propagates `ckks_reallocate_limbs_checked` if the computed compact
    ///   size would violate metadata constraints
    fn ckks_compact_limbs(&self, ct: &mut CKKSCiphertext<Vec<u8>>) -> Result<()>;

    /// Returns a newly allocated owned ciphertext holding a compacted copy of
    /// `ct`.
    ///
    /// Inputs:
    /// - `ct`: ciphertext to copy and compact
    ///
    /// Output:
    /// - a fresh owned ciphertext with the same metadata and the minimum limb
    ///   count needed to preserve it
    ///
    /// Errors:
    /// - `LimbReallocationShrinksBelowMetadata` if the compacted size would undercut the current metadata
    fn ckks_compact_limbs_copy<D>(&self, ct: &CKKSCiphertext<D>) -> Result<CKKSCiphertext<Vec<u8>>>
    where
        D: HostDataRef;
}

#[doc(hidden)]
pub trait CKKSMaintainOpsDefault<BE: Backend> {
    fn ckks_reallocate_limbs_checked_default(&self, ct: &mut CKKSCiphertext<Vec<u8>>, size: usize) -> Result<()> {
        let base2k = ct.base2k().as_usize();
        let required_limbs = ct.effective_k().div_ceil(base2k);
        anyhow::ensure!(
            size >= required_limbs,
            CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
                max_k: ct.max_k().as_usize(),
                log_delta: ct.log_delta(),
                base2k,
                requested_limbs: size,
            }
        );
        ct.data_mut().reallocate_limbs(size);
        Ok(())
    }

    fn ckks_compact_limbs_default(&self, ct: &mut CKKSCiphertext<Vec<u8>>) -> Result<()> {
        let size = ct.effective_k().div_ceil(ct.base2k().as_usize());
        self.ckks_reallocate_limbs_checked_default(ct, size)?;
        Ok(())
    }
}

#[macro_export]
macro_rules! impl_ckks_maintain_ops_defaults {
    ($be:ty) => {
        impl $crate::layouts::ciphertext::CKKSMaintainOpsDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_maintain_ops_defaults;

impl<BE: Backend> CKKSMaintainOps for Module<BE>
where
    BE: HostBackend<OwnedBuf = Vec<u8>>,
    Module<BE>: CKKSMaintainOpsDefault<BE> + CKKSModuleAlloc<BE>,
{
    fn ckks_reallocate_limbs_checked(&self, ct: &mut CKKSCiphertext<Vec<u8>>, size: usize) -> Result<()> {
        self.ckks_reallocate_limbs_checked_default(ct, size)
    }

    fn ckks_compact_limbs(&self, ct: &mut CKKSCiphertext<Vec<u8>>) -> Result<()> {
        self.ckks_compact_limbs_default(ct)
    }

    fn ckks_compact_limbs_copy<D>(&self, ct: &CKKSCiphertext<D>) -> Result<CKKSCiphertext<Vec<u8>>>
    where
        D: HostDataRef,
    {
        let size = ct.effective_k().div_ceil(ct.base2k().as_usize());
        let mut compact = self.ckks_ciphertext_alloc_from_infos(ct);
        compact.meta = ct.meta();
        self.ckks_reallocate_limbs_checked_default(&mut compact, size)?;
        let dst_len = compact.data().data.len();
        compact.data_mut().data.copy_from_slice(&ct.data().data.as_ref()[..dst_len]);
        Ok(compact)
    }
}

/// A CKKS ciphertext produced by an unnormalized linear operation.
///
/// Unnormalized ciphertexts have un-propagated carries: limb digits may hold
/// more than `base2k` bits. Any primitive that switches to the DFT domain —
/// keyswitching, convolution (`ckks_mul`), or automorphisms (`ckks_rotate`,
/// `ckks_conjugate`) — assumes each limb fits within `base2k` bits; passing
/// an unnormalized ciphertext to one will silently produce an incorrectly
/// decryptable result. CKKS-level APIs use the normalization state parameter to
/// route explicit unnormalized accumulation through the matching operations.
///
/// The only way to recover a [`CKKSCiphertext`] from an
/// `UnnormalizedCKKSCiphertext` is to call [`Self::normalize`], which applies
/// the missing `glwe_normalize_assign` step and consumes `self`.
pub type UnnormalizedCKKSCiphertext<D> = CKKSCiphertext<D, Unnormalized>;

impl<D: Data> CKKSCiphertext<D, Unnormalized> {
    /// Wraps `ct` in the unnormalized typestate.
    pub fn new(ct: CKKSCiphertext<D>) -> Self {
        Self::from_inner(ct.inner, ct.meta)
    }

    /// Returns a shared reference to the underlying ciphertext for read-only
    /// inspection (e.g. metadata checks or decryption in tests).
    pub fn as_inner(&self) -> &Self {
        self
    }

    /// Normalizes the ciphertext and returns the result as a [`CKKSCiphertext`].
    ///
    /// Propagates carries through the limb chain (only the top limb discards
    /// overflow), making each digit fit within `base2k` bits and the result
    /// safe to pass to any DFT-domain primitive (keyswitching, convolution,
    /// automorphisms).
    pub fn normalize<M, BE>(self, module: &M, scratch: &mut ScratchArena<'_, BE>) -> CKKSCiphertext<D>
    where
        BE: Backend,
        M: GLWENormalize<BE>,
        GLWE<D>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let mut normalized = CKKSCiphertext::<D>::from_inner(self.inner, self.meta);
        module.glwe_normalize_assign(&mut normalized, scratch);
        normalized
    }
}

pub struct UnnormalizedCKKSCiphertextRefMut<'a, D: Data> {
    pub(crate) inner: &'a mut CKKSCiphertext<D>,
}

impl<'a, D: Data> UnnormalizedCKKSCiphertextRefMut<'a, D> {
    pub(crate) fn new(inner: &'a mut CKKSCiphertext<D>) -> Self {
        Self { inner }
    }

    pub(crate) fn normalize<M, BE>(self, module: &M, scratch: &mut ScratchArena<'_, BE>)
    where
        BE: Backend,
        M: GLWENormalize<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
    {
        module.glwe_normalize_assign(self.inner, scratch);
    }
}

pub(crate) trait CKKSOffset: LWEInfos + CKKSInfos {
    fn offset_unary<A>(&self, a: &A) -> usize
    where
        A: LWEInfos + CKKSInfos,
    {
        crate::ckks_offset_unary(self, a)
    }

    fn offset_binary<A, B>(&self, a: &A, b: &B) -> usize
    where
        A: LWEInfos + CKKSInfos,
        B: LWEInfos + CKKSInfos,
    {
        crate::ckks_offset_binary(self, a, b)
    }
}

impl<T> CKKSOffset for T where T: LWEInfos + CKKSInfos {}
