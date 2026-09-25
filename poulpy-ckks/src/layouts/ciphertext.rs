//! CKKS metadata attached to ciphertext storage.
//!
//! A CKKS ciphertext is represented as [`CKKSCiphertext<D>`], a thin wrapper
//! over `poulpy-core`'s `GLWE<D, CKKS>`.

use poulpy_hal::AlignedBuf;
use std::{
    fmt,
    mem::align_of,
    ops::{Deref, DerefMut},
};

use anyhow::Result;
use poulpy_core::ScratchArenaTakeCore;
use poulpy_core::layouts::{
    BSGSMeta, Base2K, Degree, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GLWEViewMut, GLWEViewRef, LWEInfos, Rank,
    SetBSGSMeta, SetK, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Data, HostDataRef, ScratchArena, ZnxWord};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos, error::CKKSCompositionError};

use super::{CKKSEncodingBuffer, CKKSEncodingBufferViewMut, CKKSPlaintextViewMut};

/// CKKS ciphertext storage plus semantic precision metadata.
///
/// `inner` contains the raw GLWE torus digits while `meta` describes the
/// semantic decimal scaling and remaining homomorphic capacity of the value.
pub struct CKKSCiphertext<D: Data, W: ZnxWord> {
    /// Raw GLWE ciphertext storage.
    pub(crate) inner: GLWE<D, W>,
    /// Semantic CKKS metadata associated with `inner`.
    pub(crate) meta: CKKSMeta,
}

impl<D: Data, W: ZnxWord> CKKSCiphertext<D, W> {
    pub(crate) fn from_inner(inner: GLWE<D, W>, meta: CKKSMeta) -> Self {
        Self { inner, meta }
    }

    /// Rebuilds this backend-owned ciphertext as a host-owned [`CKKSCiphertext<AlignedBuf, i64>`].
    pub fn to_host_owned<BE>(&self) -> CKKSCiphertext<AlignedBuf, W>
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        CKKSCiphertext::<AlignedBuf, W>::from_inner(self.inner.to_host_owned::<BE>(), self.meta)
    }

    /// Formats this backend-owned ciphertext through the existing host [`fmt::Display`] implementation.
    pub fn display_host<BE>(&self) -> String
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        self.to_host_owned::<BE>().to_string()
    }

    pub fn to_ref<BE: Backend<ZnxWord = W>>(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord>
    where
        GLWE<D, W>: GLWEToBackendRef<BE>,
    {
        GLWEToBackendRef::to_backend_ref(&self.inner)
    }

    pub fn to_mut<BE: Backend<ZnxWord = W>>(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord>
    where
        GLWE<D, W>: GLWEToBackendMut<BE>,
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
        // validates that the claimed width fits the allocated storage and that the
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
impl<D: Data, W: ZnxWord> Clone for CKKSCiphertext<D, W>
where
    GLWE<D, W>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            meta: self.meta,
        }
    }
}

impl<D: Data, W: ZnxWord> Deref for CKKSCiphertext<D, W> {
    type Target = GLWE<D, W>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: Data, W: ZnxWord> DerefMut for CKKSCiphertext<D, W> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for CKKSCiphertext<D, W> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }

    fn max_size(&self) -> usize {
        self.inner.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.inner.k()
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for CKKSCiphertext<D, W> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data, W: ZnxWord> CKKSInfos for CKKSCiphertext<D, W> {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }
}

impl<D: Data, W: ZnxWord> SetCKKSInfos for CKKSCiphertext<D, W> {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.meta = meta;
    }

    fn set_k(&mut self, k: TorusPrecision) {
        SetK::set_k(&mut self.inner, k);
    }
}

impl<D: Data, W: ZnxWord> SetK for CKKSCiphertext<D, W> {
    fn set_k(&mut self, k: TorusPrecision) {
        SetK::set_k(&mut self.inner, k);
    }
}

impl<D: Data, W: ZnxWord> BSGSMeta for CKKSCiphertext<D, W> {
    fn bsgs_log_budget(&self) -> usize {
        CKKSInfos::log_budget(self)
    }
    fn bsgs_log_delta(&self) -> usize {
        CKKSInfos::log_delta(self)
    }
}

impl<D: Data, W: ZnxWord> SetBSGSMeta for CKKSCiphertext<D, W> {
    fn set_bsgs_log_budget(&mut self, log_budget: usize) {
        SetCKKSInfos::set_log_budget(self, log_budget);
    }
    fn set_bsgs_log_delta(&mut self, log_delta: usize) {
        SetCKKSInfos::set_log_delta(self, log_delta);
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for CKKSCiphertext<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl<BE: Backend, D: Data> GLWEToBackendRef<BE> for CKKSCiphertext<D, BE::ZnxWord>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        GLWEToBackendRef::to_backend_ref(&self.inner)
    }
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for CKKSCiphertext<D, BE::ZnxWord>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        GLWEToBackendMut::to_backend_mut(&mut self.inner)
    }

    fn set_canonical(&mut self, canonical: bool) {
        self.inner.set_canonical(canonical)
    }
}

/// Backend-owned CKKS ciphertext: the backend's buffer type and its coefficient word.
pub type CKKSCiphertextOwned<BE> = CKKSCiphertext<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

pub(crate) struct CKKSCiphertextViewRef<'a, BE: Backend + 'a> {
    inner: GLWEViewRef<'a, BE>,
    meta: CKKSMeta,
}

impl<'a, BE: Backend + 'a> Deref for CKKSCiphertextViewRef<'a, BE> {
    type Target = GLWEViewRef<'a, BE>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<BE: Backend> LWEInfos for CKKSCiphertextViewRef<'_, BE> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }

    fn max_size(&self) -> usize {
        self.inner.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.inner.k()
    }
}

impl<BE: Backend> GLWEInfos for CKKSCiphertextViewRef<'_, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<BE: Backend> CKKSInfos for CKKSCiphertextViewRef<'_, BE> {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }
}

impl<BE: Backend> GLWEToBackendRef<BE> for CKKSCiphertextViewRef<'_, BE> {
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        self.inner.to_backend_ref()
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

    pub(crate) fn to_backend_view_ref(&self) -> CKKSCiphertextViewRef<'_, BE> {
        CKKSCiphertextViewRef {
            inner: GLWEViewRef::from_inner(self.inner.to_backend_ref()),
            meta: self.meta,
        }
    }
}

impl<'a, BE: Backend + 'a> Deref for CKKSCiphertextViewMut<'a, BE> {
    type Target = GLWEViewMut<'a, BE>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<BE: Backend> DerefMut for CKKSCiphertextViewMut<'_, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

crate::impl_ckks_infos!(self_meta CKKSCiphertextViewMut);

impl<BE: Backend> GLWEToBackendRef<BE> for CKKSCiphertextViewMut<'_, BE> {
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        self.inner.to_backend_ref()
    }
}

impl<BE: Backend> GLWEToBackendMut<BE> for CKKSCiphertextViewMut<'_, BE> {
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        self.inner.to_backend_mut()
    }

    fn set_canonical(&mut self, canonical: bool) {
        GLWEToBackendMut::<BE>::set_canonical(&mut self.inner, canonical)
    }
}

/// CKKS layout carving helpers for backend-native scratch arenas.
pub trait ScratchArenaTakeCKKS<'a, BE: Backend>: ScratchArenaTakeCore<'a, BE> + Sized {
    /// Carves a backend-resident scalar workspace for CKKS encoding.
    fn take_ckks_encoding_buffer_scratch<F>(self, len: usize) -> (CKKSEncodingBufferViewMut<'a, BE, F>, Self)
    where
        BE: 'a;

    /// Carves a mutable CKKS plaintext view from backend-native scratch.
    fn take_ckks_plaintext_scratch<I>(self, infos: &I, meta: CKKSMeta) -> (CKKSPlaintextViewMut<'a, BE>, Self)
    where
        BE: 'a,
        I: GLWEInfos,
    {
        let (inner, scratch) = self.take_glwe_plaintext_scratch(infos);
        let inner = super::CKKSPlaintext::from_inner(inner.into_inner(), meta);
        (CKKSPlaintextViewMut::from_inner(inner), scratch)
    }

    /// Carves a mutable CKKS plaintext view with another value's layout and metadata.
    fn take_ckks_plaintext_like_scratch<P>(self, pt: &P) -> (CKKSPlaintextViewMut<'a, BE>, Self)
    where
        BE: 'a,
        P: GLWEInfos + CKKSInfos,
    {
        self.take_ckks_plaintext_scratch(pt, pt.meta())
    }

    fn take_ckks_ciphertext_scratch<I>(self, infos: &I, meta: CKKSMeta) -> (CKKSCiphertextViewMut<'a, BE>, Self)
    where
        BE: 'a,
        I: GLWEInfos,
    {
        let (inner, scratch) = self.take_glwe_scratch(infos);
        (CKKSCiphertextViewMut::from_inner(inner, meta), scratch)
    }

    /// Carves several same-layout CKKS ciphertexts from scratch space.
    fn take_ckks_ciphertext_slice_scratch<I>(
        self,
        size: usize,
        infos: &I,
        meta: CKKSMeta,
    ) -> (Vec<CKKSCiphertextViewMut<'a, BE>>, Self)
    where
        BE: 'a,
        I: GLWEInfos,
    {
        let (inner, scratch) = self.take_glwe_slice_scratch(size, infos);
        (
            inner
                .into_iter()
                .map(|ct| CKKSCiphertextViewMut::from_inner(ct, meta))
                .collect(),
            scratch,
        )
    }

    fn take_ckks_ciphertext_like_scratch<C>(self, ct: &C) -> (CKKSCiphertextViewMut<'a, BE>, Self)
    where
        BE: 'a,
        C: GLWEInfos + CKKSInfos,
    {
        self.take_ckks_ciphertext_scratch(ct, ct.meta())
    }
}

impl<'a, BE> ScratchArenaTakeCKKS<'a, BE> for ScratchArena<'a, BE>
where
    BE: Backend + 'a,
{
    fn take_ckks_encoding_buffer_scratch<F>(self, len: usize) -> (CKKSEncodingBufferViewMut<'a, BE, F>, Self) {
        assert!(
            BE::SCRATCH_ALIGN.is_multiple_of(align_of::<F>()),
            "backend scratch alignment {} is not a multiple of encoding scalar alignment {}",
            BE::SCRATCH_ALIGN,
            align_of::<F>()
        );
        let (data, scratch) = self.take_region(CKKSEncodingBuffer::<BE::BufMut<'a>, F>::bytes_of(len));
        (
            CKKSEncodingBufferViewMut::from_inner(CKKSEncodingBuffer::from_data(data, len)),
            scratch,
        )
    }
}
