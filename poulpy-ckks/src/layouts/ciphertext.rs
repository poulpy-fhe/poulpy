//! CKKS metadata attached to ciphertext storage.
//!
//! A CKKS ciphertext is represented as [`CKKSCiphertext<D>`], a thin wrapper
//! over `poulpy-core`'s `GLWE<D, CKKS>`.

use std::{
    fmt,
    marker::PhantomData,
    mem::align_of,
    ops::{Deref, DerefMut},
};

use anyhow::Result;
use poulpy_core::layouts::{
    BSGSMeta, Base2K, Compact, Degree, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GLWEViewMut, LWEInfos, Rank,
    SetBSGSMeta, SetK, SetSize, TorusPrecision,
};
use poulpy_core::{GLWENormalize, ScratchArenaTakeCore};
use poulpy_hal::layouts::{Backend, Data, HostDataRef, ScratchArena};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos, error::CKKSCompositionError};

use super::{CKKSEncodingBuffer, CKKSEncodingBufferViewMut, CKKSPlaintextViewMut};

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
}

impl<D: Data, S: CKKSNormalizationState> SetCKKSInfos for CKKSCiphertext<D, S> {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.meta = meta;
    }

    fn set_k(&mut self, k: TorusPrecision) {
        SetK::set_k(&mut self.inner, k);
    }
}

impl<D: Data, S: CKKSNormalizationState> SetK for CKKSCiphertext<D, S> {
    fn set_k(&mut self, k: TorusPrecision) {
        SetK::set_k(&mut self.inner, k);
    }
}

impl<D: Data, S: CKKSNormalizationState> SetSize for CKKSCiphertext<D, S> {
    fn set_size(&mut self, size: usize) {
        self.inner.data_mut().set_size(size);
    }
}

impl<D: Data, S: CKKSNormalizationState> Compact for CKKSCiphertext<D, S> {}

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

// Backend conversion is deliberately implemented for the `Normalized` state
// ONLY: every DFT-domain op (keyswitching, convolution, automorphisms) is
// generic over these traits, so keeping them off `Unnormalized` is what makes
// passing an unnormalized ciphertext to such an op a compile error (see the
// crate-level docs). The unnormalized write path goes through the crate-private
// [`UnnormalizedCKKSCiphertextWriteView`] instead.
impl<BE: Backend, D: Data> GLWEToBackendRef<BE> for CKKSCiphertext<D, Normalized>
where
    GLWE<D>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>> {
        GLWEToBackendRef::to_backend_ref(&self.inner)
    }
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for CKKSCiphertext<D, Normalized>
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

crate::impl_ckks_infos!(self_meta CKKSCiphertextViewMut);

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
///
/// Unlike [`CKKSCiphertext`], this type does **not** implement
/// `GLWEToBackendRef`/`GLWEToBackendMut`, so passing it to a DFT-domain op is
/// a compile error:
///
/// ```compile_fail
/// use poulpy_ckks::layouts::UnnormalizedCKKSCiphertext;
/// use poulpy_core::layouts::{GLWE, GLWEToBackendRef};
/// use poulpy_hal::layouts::{Backend, Data};
///
/// fn dft_domain_op<BE: Backend, T: GLWEToBackendRef<BE>>(_: &T) {}
///
/// fn reject<BE: Backend, D: Data>(ct: &UnnormalizedCKKSCiphertext<D>)
/// where
///     GLWE<D>: GLWEToBackendRef<BE>,
/// {
///     dft_domain_op::<BE, _>(ct); // ERROR: trait not implemented for Unnormalized
/// }
/// ```
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

    /// Crate-private backend write access for the unnormalized op defaults.
    ///
    /// `UnnormalizedCKKSCiphertext` deliberately does not implement
    /// `GLWEToBackendRef`/`GLWEToBackendMut` (that seal is what makes passing it
    /// to a DFT-domain op a compile error); the `_unnormalized` add/sub defaults
    /// obtain access through this view instead.
    pub(crate) fn write_view(&mut self) -> UnnormalizedCKKSCiphertextWriteView<'_, D> {
        UnnormalizedCKKSCiphertextWriteView { inner: self }
    }
}

/// Crate-private view granting backend access to an [`UnnormalizedCKKSCiphertext`].
///
/// Constructed only by [`CKKSCiphertext::<D, Unnormalized>::write_view`] inside
/// the `_unnormalized` op forwarders; never exposed publicly, so the seal on the
/// unnormalized type-state holds for all public op surfaces.
pub(crate) struct UnnormalizedCKKSCiphertextWriteView<'a, D: Data> {
    inner: &'a mut CKKSCiphertext<D, Unnormalized>,
}

impl<'a, D: Data> LWEInfos for UnnormalizedCKKSCiphertextWriteView<'a, D> {
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

impl<'a, D: Data> GLWEInfos for UnnormalizedCKKSCiphertextWriteView<'a, D> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<'a, D: Data> CKKSInfos for UnnormalizedCKKSCiphertextWriteView<'a, D> {
    fn meta(&self) -> CKKSMeta {
        self.inner.meta()
    }
}

impl<'a, D: Data> SetCKKSInfos for UnnormalizedCKKSCiphertextWriteView<'a, D> {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.inner.set_meta(meta);
    }

    fn set_k(&mut self, k: TorusPrecision) {
        SetCKKSInfos::set_k(self.inner, k);
    }
}

impl<'a, BE: Backend, D: Data> GLWEToBackendRef<BE> for UnnormalizedCKKSCiphertextWriteView<'a, D>
where
    GLWE<D>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>> {
        GLWEToBackendRef::to_backend_ref(&self.inner.inner)
    }
}

impl<'a, BE: Backend, D: Data> GLWEToBackendMut<BE> for UnnormalizedCKKSCiphertextWriteView<'a, D>
where
    GLWE<D>: GLWEToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>> {
        GLWEToBackendMut::to_backend_mut(&mut self.inner.inner)
    }
}

/// Crate-internal unnormalized reborrow of a [`CKKSCiphertext`], used by the
/// composite accumulators (dot products) to run repeated carry-verb steps on a
/// caller-owned normalized ciphertext without paying a normalization per step.
///
/// The type is `pub` only because it appears in OEP method signatures
/// ([`CKKSAddImpl::ckks_add_assign_unnormalized_ref_impl`](crate::oep::CKKSAddImpl::ckks_add_assign_unnormalized_ref_impl)
/// and the sub mirror); it is deliberately **unconstructible outside this
/// crate** (crate-private constructor and field). A public constructor — or an
/// api-layer counterpart for the OEP methods that consume it — would let
/// callers accumulate carries into a `Normalized`-typed ciphertext and release
/// it un-normalized, silently voiding the compile-time normalization guard the
/// crate documents. The internal users restore the invariant by calling
/// `normalize` (crate-private) before the borrow ends.
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
