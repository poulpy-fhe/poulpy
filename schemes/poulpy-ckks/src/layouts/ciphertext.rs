//! CKKS metadata attached to ciphertext storage.
//!
//! A CKKS ciphertext is represented as [`CKKSCiphertext<D>`], a thin wrapper
//! over `poulpy-core`'s `GLWE<D, CKKS>`.

use poulpy_hal::AlignedBuf;
use std::{
    fmt,
    marker::PhantomData,
    mem::align_of,
    ops::{Deref, DerefMut},
};

use anyhow::Result;
use poulpy_core::layouts::{
    BSGSMeta, Base2K, Degree, GLWE, GLWEInfos, GLWEPlaintextInfos, GLWEPlaintextMeta, GLWEToBackendMut, GLWEToBackendRef,
    GLWEViewMut, GLWEViewRef, LWEInfos, Rank, SetBSGSMeta, SetGLWEPlaintextInfos, SetK, TorusPrecision,
};
use poulpy_core::{ScratchArenaTakeGLWE, ScratchArenaTakeGLWEPlaintext};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, Ring, ScratchArena, WriterTo, ZnxWord};

use crate::{CKKSInfos, SetCKKSInfos, ckks_log_delta, error::CKKSCompositionError};

use super::{CKKSEncodingBuffer, CKKSEncodingBufferViewMut, CKKSPlaintextViewMut};

/// CKKS ciphertext: a GLWE whose plaintext metadata holds the CKKS scale,
/// slot kind and sparsity.
/// `R` is the ring the value lives in; a module accepts only operands of its backend's ring.
pub struct CKKSCiphertext<D: Data, W: ZnxWord, R: Ring> {
    /// Raw GLWE ciphertext storage.
    pub(crate) inner: GLWE<D, W>,
    _ring: PhantomData<R>,
}

impl<D: Data, W: ZnxWord, R: Ring> CKKSCiphertext<D, W, R> {
    pub(crate) fn from_inner(inner: GLWE<D, W>, meta: GLWEPlaintextMeta) -> Self {
        let mut ct = Self {
            inner,
            _ring: PhantomData,
        };
        ct.set_meta(meta);
        ct
    }

    /// Rebuilds this backend-owned ciphertext as a host-owned [`CKKSCiphertext<AlignedBuf, W, R>`].
    pub fn to_host_owned<BE>(&self) -> CKKSCiphertext<AlignedBuf, W, R>
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        CKKSCiphertext {
            inner: self.inner.to_host_owned::<BE>(),
            _ring: PhantomData,
        }
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
    pub fn set_meta_checked(&mut self, meta: GLWEPlaintextMeta) -> Result<()> {
        ensure_meta_fits(self, &meta)?;
        self.set_meta(meta);
        Ok(())
    }
}

/// The budget lives in the wrapped GLWE's torus width `k`, so this only validates that the claimed width fits the
/// allocated storage and that the requested scale fits within it.
fn ensure_meta_fits(ct: &impl GLWEInfos, meta: &GLWEPlaintextMeta) -> Result<()> {
    anyhow::ensure!(
        ct.k().as_usize() <= ct.max_k().as_usize() && ckks_log_delta(meta) <= ct.k().as_usize(),
        CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
            max_k: ct.max_k().as_usize(),
            log_delta: ckks_log_delta(meta),
            base2k: ct.base2k().as_usize(),
            requested_limbs: ct.max_size(),
        }
    );
    Ok(())
}

// Binding `R` to `BE::Ring` in the backend conversions below rejects a
// ciphertext of another ring at compile time.

// Without this, `ct.clone()` silently resolves through `Deref` to `GLWE::clone`.
impl<D: Data, W: ZnxWord, R: Ring> Clone for CKKSCiphertext<D, W, R>
where
    GLWE<D, W>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _ring: PhantomData,
        }
    }
}

impl<D: Data, W: ZnxWord, R: Ring> Deref for CKKSCiphertext<D, W, R> {
    type Target = GLWE<D, W>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: Data, W: ZnxWord, R: Ring> DerefMut for CKKSCiphertext<D, W, R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: Data, W: ZnxWord, R: Ring> LWEInfos for CKKSCiphertext<D, W, R> {
    fn noise(&self) -> Option<poulpy_core::ComponentNoise> {
        self.inner.noise()
    }

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

impl<D: Data, W: ZnxWord, R: Ring> GLWEInfos for CKKSCiphertext<D, W, R> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data, W: ZnxWord, R: Ring> GLWEPlaintextInfos for CKKSCiphertext<D, W, R> {
    fn plaintext_meta(&self) -> Option<GLWEPlaintextMeta> {
        self.inner.plaintext_meta()
    }
}

impl<D: Data, W: ZnxWord, R: Ring> CKKSInfos for CKKSCiphertext<D, W, R> {}

impl<D: Data, W: ZnxWord, R: Ring> SetGLWEPlaintextInfos for CKKSCiphertext<D, W, R> {
    fn set_plaintext_meta(&mut self, meta: Option<GLWEPlaintextMeta>) {
        self.inner.set_plaintext_meta(meta)
    }
}

impl<D: Data, W: ZnxWord, R: Ring> SetK for CKKSCiphertext<D, W, R> {
    fn set_k(&mut self, k: TorusPrecision) {
        SetK::set_k(&mut self.inner, k);
    }
}

impl<D: Data, W: ZnxWord, R: Ring> BSGSMeta for CKKSCiphertext<D, W, R> {
    fn bsgs_log_budget(&self) -> usize {
        CKKSInfos::log_budget(self)
    }
    fn bsgs_log_delta(&self) -> usize {
        CKKSInfos::log_delta(self)
    }
}

impl<D: Data, W: ZnxWord, R: Ring> SetBSGSMeta for CKKSCiphertext<D, W, R> {
    fn set_bsgs_log_budget(&mut self, log_budget: usize) {
        SetCKKSInfos::set_log_budget(self, log_budget);
    }
    fn set_bsgs_log_delta(&mut self, log_delta: usize) {
        SetCKKSInfos::set_log_delta(self, log_delta);
    }
}

impl<D: HostDataRef, W: ZnxWord, R: Ring> fmt::Display for CKKSCiphertext<D, W, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl<D: HostDataRef, W: ZnxWord, R: Ring> WriterTo for CKKSCiphertext<D, W, R> {
    /// The format of the inner [`GLWE`], which rejects a flag-clear ciphertext with
    /// [`std::io::ErrorKind::InvalidInput`], writing nothing; the ring is the type's.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.inner.write_to(writer)
    }
}

impl<D: HostDataMut, W: ZnxWord, R: Ring> ReaderFrom for CKKSCiphertext<D, W, R> {
    /// Reads the inner [`GLWE`]'s format into a ciphertext of this ring.
    fn read_from<Rd: std::io::Read>(&mut self, reader: &mut Rd) -> std::io::Result<()> {
        self.inner.read_from(reader)
    }
}

impl<BE: Backend, D: Data> GLWEToBackendRef<BE> for CKKSCiphertext<D, BE::ZnxWord, BE::Ring>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        GLWEToBackendRef::to_backend_ref(&self.inner)
    }
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for CKKSCiphertext<D, BE::ZnxWord, BE::Ring>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        GLWEToBackendMut::to_backend_mut(&mut self.inner)
    }

    fn set_canonical(&mut self, canonical: bool) {
        self.inner.set_canonical(canonical)
    }

    fn set_noise(&mut self, metadata: Option<poulpy_core::ComponentNoise>) {
        GLWEToBackendMut::<BE>::set_noise(&mut self.inner, metadata);
    }
}

/// Backend-owned CKKS ciphertext: the backend's buffer type and its coefficient word.
pub type CKKSCiphertextOwned<BE> = CKKSCiphertext<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord, <BE as Backend>::Ring>;

/// Ciphertext of ring `R` in the buffers of backend `BE`.
pub type CKKSRingCiphertext<BE, R> = CKKSCiphertext<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord, R>;

/// Read-only view of a scratch-backed CKKS ciphertext, from [`CKKSCiphertextViewMut::to_backend_view_ref`].
pub struct CKKSCiphertextViewRef<'a, BE: Backend + 'a> {
    inner: GLWEViewRef<'a, BE>,
}

impl<'a, BE: Backend + 'a> Deref for CKKSCiphertextViewRef<'a, BE> {
    type Target = GLWEViewRef<'a, BE>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<BE: Backend> LWEInfos for CKKSCiphertextViewRef<'_, BE> {
    fn noise(&self) -> Option<poulpy_core::ComponentNoise> {
        self.inner.noise()
    }

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

impl<BE: Backend> GLWEPlaintextInfos for CKKSCiphertextViewRef<'_, BE> {
    fn plaintext_meta(&self) -> Option<GLWEPlaintextMeta> {
        self.inner.plaintext_meta()
    }
}

impl<BE: Backend> CKKSInfos for CKKSCiphertextViewRef<'_, BE> {}

impl<BE: Backend> GLWEToBackendRef<BE> for CKKSCiphertextViewRef<'_, BE> {
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        self.inner.to_backend_ref()
    }
}

/// Scratch-backed mutable CKKS ciphertext view.
///
/// This is the CKKS analogue of core's [`GLWEViewMut`]: the limb storage is
/// borrowed from a [`ScratchArena`] in the backend-native buffer type, and the
/// CKKS metadata lives in the GLWE view's plaintext metadata.
pub struct CKKSCiphertextViewMut<'a, BE: Backend + 'a> {
    inner: GLWEViewMut<'a, BE>,
}

impl<'a, BE: Backend + 'a> CKKSCiphertextViewMut<'a, BE> {
    pub(crate) fn from_inner(inner: GLWEViewMut<'a, BE>, meta: GLWEPlaintextMeta) -> Self {
        let mut ct = Self { inner };
        ct.set_meta(meta);
        ct
    }

    /// As [`CKKSCiphertext::set_meta_checked`].
    pub(crate) fn set_meta_checked(&mut self, meta: GLWEPlaintextMeta) -> Result<()> {
        ensure_meta_fits(self, &meta)?;
        self.set_meta(meta);
        Ok(())
    }

    /// Reborrows this view read-only, in the backend's shared buffer type.
    pub fn to_backend_view_ref(&self) -> CKKSCiphertextViewRef<'_, BE> {
        CKKSCiphertextViewRef {
            inner: GLWEViewRef::from_inner(self.inner.to_backend_ref()),
        }
    }
    /// The same view on backend `B`, which shares the buffers of `BE`.
    pub(crate) fn rebind<B: Backend + 'a>(self) -> CKKSCiphertextViewMut<'a, B>
    where
        BE: for<'b> Backend<BufMut<'b> = B::BufMut<'b>, ZnxWord = B::ZnxWord>,
    {
        let meta = self.meta();
        CKKSCiphertextViewMut::from_inner(GLWEViewMut::from_inner(self.inner.into_inner()), meta)
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

crate::impl_ckks_infos!(glwe_inner CKKSCiphertextViewMut);

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

    fn set_noise(&mut self, metadata: Option<poulpy_core::ComponentNoise>) {
        GLWEToBackendMut::<BE>::set_noise(&mut self.inner, metadata);
    }
}

/// CKKS layout carving helpers for backend-native scratch arenas.
pub trait ScratchArenaTakeCKKS<'a, BE: Backend>:
    ScratchArenaTakeGLWE<'a, BE> + ScratchArenaTakeGLWEPlaintext<'a, BE> + Sized
{
    /// Carves a backend-resident scalar workspace for CKKS encoding.
    fn take_ckks_encoding_buffer_scratch<F>(self, len: usize) -> (CKKSEncodingBufferViewMut<'a, BE, F>, Self)
    where
        BE: 'a;

    /// Carves a mutable CKKS plaintext view from backend-native scratch.
    fn take_ckks_plaintext_scratch<I>(self, infos: &I, meta: GLWEPlaintextMeta) -> (CKKSPlaintextViewMut<'a, BE>, Self)
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

    fn take_ckks_ciphertext_scratch<I>(self, infos: &I, meta: GLWEPlaintextMeta) -> (CKKSCiphertextViewMut<'a, BE>, Self)
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
        meta: GLWEPlaintextMeta,
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

#[cfg(test)]
mod tests {
    use poulpy_core::layouts::{Base2K, TorusPrecision};
    use poulpy_hal::layouts::{HostBytesBackend, Module, ReaderFrom, WriterTo, ZnxViewMut};

    use crate::{CKKSInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind, layouts::CKKSModuleAlloc};

    #[test]
    fn ciphertext_round_trip_keeps_the_inner_format() {
        let module = Module::<HostBytesBackend>::new(16);
        let mut ct = module.ckks_ciphertext_alloc(Base2K(12), TorusPrecision(36));
        let meta = GLWEPlaintextMeta {
            scale: Scale::Log(20),
            slots: SlotsKind::Complex,
            log_sparsity: 2,
        };
        ct.set_meta(meta);
        ct.data_mut()
            .raw_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(t, x)| *x = t as i64 % 11 - 5);
        let mut bytes = Vec::new();
        ct.write_to(&mut bytes).unwrap();
        let mut inner = Vec::new();
        ct.inner.write_to(&mut inner).unwrap();
        assert!(bytes == inner, "a ciphertext serializes as its GLWE");
        let mut back = module.ckks_ciphertext_alloc(Base2K(12), TorusPrecision(36));
        back.read_from(&mut bytes.as_slice()).unwrap();
        assert_eq!(back.meta(), meta);
        let mut again = Vec::new();
        back.write_to(&mut again).unwrap();
        assert!(again == bytes, "a read ciphertext serializes as the original");
    }
}
