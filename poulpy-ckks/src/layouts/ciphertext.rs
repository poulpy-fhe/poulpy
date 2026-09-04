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
    BSGSMeta, Base2K, Degree, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GLWEViewMut, GLWEViewRef, LWEInfos, Rank,
    SetBSGSMeta, SetK, TorusPrecision,
};
use poulpy_core::{GLWENormalize, ScratchArenaTakeCore};
use poulpy_hal::{
    api::VecZnxNormalizeAssignBackend,
    layouts::{ArithmeticState, Backend, Data, HostDataRef, ScratchArena, VecZnx, VecZnxToBackendMut, ZnxWord},
};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos, error::CKKSCompositionError};

use super::{CKKSEncodingBuffer, CKKSEncodingBufferViewMut, CKKSPlaintextViewMut};
use poulpy_hal::layouts::BorrowedCarryView;
use poulpy_hal::layouts::WeakenBackendRef;

pub use poulpy_hal::layouts::{CoeffNormalized, CoeffUnnormalized, CoefficientState};

/// Alias kept for the previous CKKS-local marker trait; the state now lives in
/// `poulpy_hal` and propagates from [`VecZnx`](poulpy_hal::layouts::VecZnx)
/// through [`GLWE`] up to this wrapper.
pub trait CKKSNormalizationState: CoefficientState {}
impl<S: CoefficientState> CKKSNormalizationState for S {}

/// CKKS ciphertext storage plus semantic precision metadata.
///
/// `inner` contains the raw GLWE torus digits while `meta` describes the
/// semantic decimal scaling and remaining homomorphic capacity of the value.
pub struct CKKSCiphertext<D: Data, W: ZnxWord, S: CoefficientState = CoeffNormalized> {
    /// Raw GLWE ciphertext storage; its state parameter is this wrapper's.
    pub(crate) inner: GLWE<D, W, S>,
    /// Semantic CKKS metadata associated with `inner`.
    pub(crate) meta: CKKSMeta,
    _state: PhantomData<S>,
}

impl<D: Data, W: ZnxWord, S: CoefficientState> CKKSCiphertext<D, W, S> {
    pub(crate) fn from_inner(inner: GLWE<D, W, S>, meta: CKKSMeta) -> Self {
        Self {
            inner,
            meta,
            _state: PhantomData,
        }
    }

    /// Rebuilds this backend-owned ciphertext as a host-owned [`CKKSCiphertext<Vec<u8>, i64>`].
    pub fn to_host_owned<BE>(&self) -> CKKSCiphertext<Vec<u8>, W, S>
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        CKKSCiphertext::<Vec<u8>, W, S>::from_inner(self.inner.to_host_owned::<BE>(), self.meta)
    }

    /// Formats this backend-owned ciphertext through the existing host [`fmt::Display`] implementation.
    pub fn display_host<BE>(&self) -> String
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        self.to_host_owned::<BE>().to_string()
    }

    pub fn to_ref<BE: Backend<ZnxWord = W>>(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord, S>
    where
        GLWE<D, W, S>: GLWEToBackendRef<BE, State = S>,
    {
        GLWEToBackendRef::to_backend_ref(&self.inner)
    }

    pub fn to_mut<BE: Backend<ZnxWord = W>>(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord, S>
    where
        GLWE<D, W, S>: GLWEToBackendMut<BE, State = S>,
    {
        GLWEToBackendMut::to_backend_mut(&mut self.inner)
    }

    /// Relabels this ciphertext as [`CoeffUnnormalized`] (free); see [`GLWE::into_unnormalized`].
    pub fn into_unnormalized(self) -> CKKSCiphertext<D, W, CoeffUnnormalized>
    where
        D: poulpy_hal::layouts::DataOwned,
    {
        CKKSCiphertext::from_inner(self.inner.into_unnormalized(), self.meta)
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
impl<D: Data, W: ZnxWord, S: CoefficientState> Clone for CKKSCiphertext<D, W, S>
where
    GLWE<D, W, S>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            meta: self.meta,
            _state: PhantomData,
        }
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> Deref for CKKSCiphertext<D, W, S> {
    type Target = GLWE<D, W, S>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> DerefMut for CKKSCiphertext<D, W, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> LWEInfos for CKKSCiphertext<D, W, S> {
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

impl<D: Data, W: ZnxWord, S: CoefficientState> GLWEInfos for CKKSCiphertext<D, W, S> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> CKKSInfos for CKKSCiphertext<D, W, S> {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> SetCKKSInfos for CKKSCiphertext<D, W, S> {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.meta = meta;
    }

    fn set_k(&mut self, k: TorusPrecision) {
        SetK::set_k(&mut self.inner, k);
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> SetK for CKKSCiphertext<D, W, S> {
    fn set_k(&mut self, k: TorusPrecision) {
        SetK::set_k(&mut self.inner, k);
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> BSGSMeta for CKKSCiphertext<D, W, S> {
    fn bsgs_log_budget(&self) -> usize {
        CKKSInfos::log_budget(self)
    }
    fn bsgs_log_delta(&self) -> usize {
        CKKSInfos::log_delta(self)
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> SetBSGSMeta for CKKSCiphertext<D, W, S> {
    fn set_bsgs_log_budget(&mut self, log_budget: usize) {
        SetCKKSInfos::set_log_budget(self, log_budget);
    }
    fn set_bsgs_log_delta(&mut self, log_delta: usize) {
        SetCKKSInfos::set_log_delta(self, log_delta);
    }
}

impl<D: HostDataRef, W: ZnxWord, S: CoefficientState> fmt::Display for CKKSCiphertext<D, W, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

// The backend view carries the wrapper's normalization state: DFT-domain ops in
// `poulpy-core` bound their inputs with `State = CoeffNormalized`, so passing an
// `CoeffUnnormalized` ciphertext to keyswitching, convolution or an automorphism is a
// compile error, while carry-producing ops demand `State = CoeffUnnormalized`.
impl<BE: Backend, D: Data, S: ArithmeticState> GLWEToBackendRef<BE> for CKKSCiphertext<D, BE::ZnxWord, S>
where
    GLWE<D, BE::ZnxWord, S>: GLWEToBackendRef<BE, State = S>,
{
    type State = S;
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord, S> {
        GLWEToBackendRef::to_backend_ref(&self.inner)
    }
}

impl<BE: Backend, D: Data, S: ArithmeticState> GLWEToBackendMut<BE> for CKKSCiphertext<D, BE::ZnxWord, S>
where
    GLWE<D, BE::ZnxWord, S>: GLWEToBackendMut<BE, State = S>,
{
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord, S> {
        GLWEToBackendMut::to_backend_mut(&mut self.inner)
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

impl<'a, BE: Backend + 'a> LWEInfos for CKKSCiphertextViewRef<'a, BE> {
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

impl<'a, BE: Backend + 'a> GLWEInfos for CKKSCiphertextViewRef<'a, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<'a, BE: Backend + 'a> CKKSInfos for CKKSCiphertextViewRef<'a, BE> {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }
}

impl<'a, BE: Backend + 'a> GLWEToBackendRef<BE> for CKKSCiphertextViewRef<'a, BE> {
    type State = CoeffNormalized;
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

impl<'a, BE: Backend + 'a> DerefMut for CKKSCiphertextViewMut<'a, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

crate::impl_ckks_infos!(self_meta CKKSCiphertextViewMut);

impl<'a, BE: Backend + 'a> GLWEToBackendRef<BE> for CKKSCiphertextViewMut<'a, BE> {
    type State = CoeffNormalized;
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        self.inner.to_backend_ref()
    }
}

impl<'a, BE: Backend + 'a> GLWEToBackendMut<BE> for CKKSCiphertextViewMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
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

    fn take_unnormalized_ckks_ciphertext_scratch<I>(
        self,
        infos: &I,
        meta: CKKSMeta,
    ) -> (UnnormalizedCKKSCiphertext<BE::BufMut<'a>, BE::ZnxWord>, Self)
    where
        BE: 'a,
        I: GLWEInfos,
    {
        let (inner, scratch) = self.take_glwe_scratch(infos);
        (
            UnnormalizedCKKSCiphertext::from_inner(inner.into_unnormalized().into_inner(), meta),
            scratch,
        )
    }

    fn take_unnormalized_ckks_ciphertext_like_scratch<C>(
        self,
        ct: &C,
    ) -> (UnnormalizedCKKSCiphertext<BE::BufMut<'a>, BE::ZnxWord>, Self)
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
/// CoeffUnnormalized ciphertexts have un-propagated carries: limb digits may hold
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
/// fn reject<BE: Backend, D: Data>(ct: &UnnormalizedCKKSCiphertext<D, i64>)
/// where
///     GLWE<D, i64>: GLWEToBackendRef<BE>,
/// {
///     dft_domain_op::<BE, _>(ct); // ERROR: trait not implemented for CoeffUnnormalized
/// }
/// ```
pub type UnnormalizedCKKSCiphertext<D, W> = CKKSCiphertext<D, W, CoeffUnnormalized>;

impl<D: Data + poulpy_hal::layouts::DataOwned, W: ZnxWord> CKKSCiphertext<D, W, CoeffUnnormalized> {
    /// Wraps `ct` in the unnormalized typestate (a free relabel).
    pub fn new(ct: CKKSCiphertext<D, W>) -> Self {
        ct.into_unnormalized()
    }

    /// Normalizes the ciphertext and returns the result as a [`CKKSCiphertext`].
    ///
    /// Propagates carries through the limb chain (only the top limb discards
    /// overflow), making each digit fit within `base2k` bits and the result
    /// safe to pass to any DFT-domain primitive (keyswitching, convolution,
    /// automorphisms).
    pub fn normalize<M, BE>(self, module: &M, scratch: &mut ScratchArena<'_, BE>) -> CKKSCiphertext<D, W>
    where
        D: poulpy_hal::layouts::DataOwned,
        BE: Backend<ZnxWord = W>,
        M: VecZnxNormalizeAssignBackend<BE> + ?Sized,
        VecZnx<D, W, CoeffUnnormalized>: VecZnxToBackendMut<BE, State = CoeffUnnormalized>,
    {
        let meta = self.meta;
        CKKSCiphertext::from_inner(self.inner.normalize(module, scratch), meta)
    }
}

/// Crate-private unnormalized write access to a caller-provided ciphertext.
///
/// The safe (normalizing) op defaults run their `_unnormalized` sibling on the
/// destination and then normalize it in place. `poulpy-core` types the
/// carry-producing verbs with `State = CoeffUnnormalized`, so this wrapper relabels a
/// temporary backend view of `T` while forwarding all CKKS metadata accessors to
/// it. It is sound only because every user normalizes `T` before the borrow is
/// relied upon again as `CoeffNormalized`; it is therefore never exposed publicly.
pub(crate) struct CKKSUnnormalizedWriteView<'a, T> {
    inner: &'a mut T,
}

impl<'a, T> CKKSUnnormalizedWriteView<'a, T> {
    pub(crate) fn new(inner: &'a mut T) -> Self {
        Self { inner }
    }
}

impl<'a, T: LWEInfos> LWEInfos for CKKSUnnormalizedWriteView<'a, T> {
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

impl<'a, T: GLWEInfos> GLWEInfos for CKKSUnnormalizedWriteView<'a, T> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<'a, T: CKKSInfos> CKKSInfos for CKKSUnnormalizedWriteView<'a, T> {
    fn meta(&self) -> CKKSMeta {
        self.inner.meta()
    }
}

impl<'a, T: SetCKKSInfos> SetCKKSInfos for CKKSUnnormalizedWriteView<'a, T> {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.inner.set_meta(meta);
    }

    fn set_k(&mut self, k: TorusPrecision) {
        SetCKKSInfos::set_k(self.inner, k);
    }
}

impl<'a, BE: Backend, T: GLWEToBackendRef<BE>> GLWEToBackendRef<BE> for CKKSUnnormalizedWriteView<'a, T> {
    type State = CoeffUnnormalized;
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord, CoeffUnnormalized> {
        self.inner.to_backend_ref().weaken_backend_ref()
    }
}

impl<'a, BE: Backend, T: GLWEToBackendMut<BE>> GLWEToBackendMut<BE> for CKKSUnnormalizedWriteView<'a, T> {
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord, CoeffUnnormalized> {
        self.inner.to_backend_mut().borrowed_carry_view()
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
/// callers accumulate carries into a `CoeffNormalized`-typed ciphertext and release
/// it un-normalized, silently voiding the compile-time normalization guard the
/// crate documents. The internal users restore the invariant by calling
/// `normalize` (crate-private) before the borrow ends.
pub struct UnnormalizedCKKSCiphertextRefMut<'a, D: Data, W: ZnxWord> {
    pub(crate) inner: &'a mut CKKSCiphertext<D, W>,
}

impl<'a, D: Data, W: ZnxWord> UnnormalizedCKKSCiphertextRefMut<'a, D, W> {
    pub(crate) fn new(inner: &'a mut CKKSCiphertext<D, W>) -> Self {
        Self { inner }
    }

    pub(crate) fn normalize<M, BE>(self, module: &M, scratch: &mut ScratchArena<'_, BE>)
    where
        BE: Backend<ZnxWord = W>,
        M: GLWENormalize<BE>,
        CKKSCiphertext<D, W>: GLWEToBackendMut<BE>,
    {
        module.glwe_normalize_assign(self.inner, scratch);
    }
}
