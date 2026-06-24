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
use poulpy_core::layouts::{
    BSGSMeta, Base2K, Compact, Degree, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GLWEViewMut, LWEInfos, Rank, SetBSGSMeta, SetK, TorusPrecision,
};
use poulpy_core::{GLWENormalize, ScratchArenaTakeCore};
use poulpy_hal::layouts::{Backend, Data, HostDataRef, ScratchArena};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos, error::CKKSCompositionError};

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
        // The budget now lives in the wrapped GLWE's torus width `k`; this only
        // validates that the stored width fits the allocated storage and that the
        // requested scale fits within it.
        anyhow::ensure!(
            self.k().as_usize() <= self.max_k().as_usize() && meta.log_delta <= self.k().as_usize(),
            CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
                max_k: self.max_k().as_usize(),
                log_delta: meta.log_delta,
                base2k: self.base2k().as_usize(),
                requested_limbs: self.max_size(),
            }
        );
        self.meta = meta;
        Ok(())
    }
}

// Without this, `ct.clone()` silently resolves through `Deref` to
// `GLWE::clone` and drops the CKKS metadata.
impl<D: Data, S: CKKSNormalizationState> Clone for CKKSCiphertext<D, S>
where
    GLWE<D>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            meta: self.meta,
            _state: PhantomData,
        }
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

    fn max_size(&self) -> usize {
        self.inner.max_size()
    }

    // A CKKS ciphertext's processing limb count: the `precision_size()` limbs
    // that carry meaningful precision, plus one limb of keyswitch head-room
    // (capped at `max_size`). The `+1` is load-bearing; see the rationale and
    // the rejected localized alternative in `docs/issues/ckks_size_headroom.md`.
    fn size(&self) -> usize {
        (self.precision_size() + 1).max(1).min(self.max_size())
    }

    fn k(&self) -> TorusPrecision {
        self.inner.k()
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
        self.meta.log_delta
    }

    fn log_budget(&self) -> usize {
        self.inner.k().as_usize().saturating_sub(self.meta.log_delta)
    }
}

impl<D: Data, S: CKKSNormalizationState> SetCKKSInfos for CKKSCiphertext<D, S> {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.meta = meta;
    }

    fn set_k(&mut self, k: TorusPrecision) {
        SetK::set_k(&mut self.inner, k);
    }
}

impl<D: Data, S: CKKSNormalizationState> Compact for CKKSCiphertext<D, S> {
    fn compact(&mut self) {
        // Drop the active limbs below `k + log_n` (the `size()` working width):
        // the limbs beyond that hold only spent keyswitch noise, and leaving them
        // active lets carry-domain ops (normalize/copy) fold that noise back in.
        // Clamp to `[1, max_size()]`. `k()` / `log_budget` are unchanged.
        let limbs = (self.k().as_usize() + self.log_n())
            .div_ceil(self.base2k().as_usize())
            .max(1)
            .min(self.max_size());
        self.inner.data_mut().set_size(limbs);
    }
}

impl<D: Data, S: CKKSNormalizationState> BSGSMeta for CKKSCiphertext<D, S> {
    fn bsgs_log_budget(&self) -> usize {
        CKKSInfos::log_budget(self)
    }
    fn bsgs_log_delta(&self) -> usize {
        CKKSInfos::log_delta(self)
    }
}

impl<D: Data, S: CKKSNormalizationState> SetBSGSMeta for CKKSCiphertext<D, S> {
    fn set_bsgs_log_budget(&mut self, log_budget: usize) {
        SetCKKSInfos::set_log_budget(self, log_budget);
    }
    fn set_bsgs_log_delta(&mut self, log_delta: usize) {
        SetCKKSInfos::set_log_delta(self, log_delta);
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

    fn max_size(&self) -> usize {
        self.inner.max_size()
    }

    // One limb of keyswitch head-room above `precision_size()`; see
    // `CKKSCiphertext::size`.
    fn size(&self) -> usize {
        (self.precision_size() + 1).max(1).min(self.inner.max_size())
    }

    fn k(&self) -> TorusPrecision {
        self.inner.k()
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
        self.meta.log_delta
    }

    fn log_budget(&self) -> usize {
        self.inner.k().as_usize().saturating_sub(self.meta.log_delta)
    }
}

impl<'a, BE: Backend + 'a> SetCKKSInfos for CKKSCiphertextViewMut<'a, BE> {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.meta = meta;
    }

    fn set_k(&mut self, k: TorusPrecision) {
        SetK::set_k(&mut self.inner, k);
    }
}

impl<'a, BE: Backend + 'a> Compact for CKKSCiphertextViewMut<'a, BE> {
    fn compact(&mut self) {
        //let limbs = self.k().div_ceil(self.base2k().as_usize()).max(1).min(self.size());
        //self.inner.data_mut().set_size(limbs);
    }
}

impl<'a, BE: Backend + 'a> BSGSMeta for CKKSCiphertextViewMut<'a, BE> {
    fn bsgs_log_budget(&self) -> usize {
        CKKSInfos::log_budget(self)
    }
    fn bsgs_log_delta(&self) -> usize {
        CKKSInfos::log_delta(self)
    }
}

impl<'a, BE: Backend + 'a> SetBSGSMeta for CKKSCiphertextViewMut<'a, BE> {
    fn set_bsgs_log_budget(&mut self, log_budget: usize) {
        SetCKKSInfos::set_log_budget(self, log_budget);
    }
    fn set_bsgs_log_delta(&mut self, log_delta: usize) {
        SetCKKSInfos::set_log_delta(self, log_delta);
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
