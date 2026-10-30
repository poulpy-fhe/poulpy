//! Vectors of CKKS ciphertexts: entries spread over ciphertexts of `chunk` entries each, at strided slots repeated with
//! the period their label proves.

use anyhow::{Result, ensure};
use byteorder::{LittleEndian, WriteBytesExt};
use poulpy_ckks::{
    CKKSCtBounds, CKKSLayout, GLWEPlaintextInfos, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSModuleAlloc},
};
use poulpy_core::layouts::{GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, Ring, WriterTo, ZnxWord};

use super::ciphertext::{layout_of, slots_of};

/// Where a vector's entries sit: `len` entries, ciphertext `t` holding entries `t chunk .. (t + 1) chunk`, its entry `e`
/// at slots `offset + stride e + q P` for every `q`, `P` the period the ciphertexts' label proves
/// ([`vector_period`]). The last ciphertext holds the remaining `len - (count - 1) chunk` entries; the other slots are
/// free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BlasVectorShape {
    /// Number of entries.
    pub len: usize,
    /// Entries per ciphertext.
    pub chunk: usize,
    /// Slot of a ciphertext's first entry.
    pub offset: usize,
    /// Slots between a ciphertext's consecutive entries.
    pub stride: usize,
}

impl BlasVectorShape {
    /// `len` entries, `chunk` per ciphertext, at slots `offset + stride e`.
    pub fn new(len: usize, chunk: usize, offset: usize, stride: usize) -> Self {
        Self {
            len,
            chunk,
            offset,
            stride,
        }
    }

    /// Number of ciphertexts, `ceil(len / chunk)`; `0` for a shape of no entry per ciphertext.
    pub fn count(&self) -> usize {
        self.len.checked_div(self.chunk).map_or(0, |_| self.len.div_ceil(self.chunk))
    }

    /// The slots a period must hold, `offset + stride (chunk - 1) + 1`; `None` without an entry per ciphertext or on
    /// overflow.
    pub(crate) fn span(&self) -> Option<usize> {
        let last = self.chunk.checked_sub(1)?;
        self.stride.checked_mul(last)?.checked_add(self.offset)?.checked_add(1)
    }

    /// Whether a ciphertext of `slots` slots repeating every `period` slots holds the shape: at least one entry, one
    /// per ciphertext and a positive stride, entries within the period, and a period dividing the slots.
    pub(crate) fn fits(&self, slots: usize, period: usize) -> bool {
        self.len > 0
            && self.stride > 0
            && period > 0
            && slots.is_multiple_of(period)
            && self.span().is_some_and(|span| span <= period)
    }
}

/// A vector's shape and the layout every one of its ciphertexts shares.
pub trait BlasVectorInfos {
    /// The vector's shape and ciphertext layout.
    fn blas_vector_layout(&self) -> BlasVectorLayout;

    /// Where the entries sit.
    fn vector_shape(&self) -> BlasVectorShape {
        self.blas_vector_layout().shape
    }

    /// Layout of every ciphertext.
    fn ct_layout(&self) -> CKKSLayout {
        self.blas_vector_layout().ct
    }
}

impl<T: BlasVectorInfos + ?Sized> BlasVectorInfos for &T {
    fn blas_vector_layout(&self) -> BlasVectorLayout {
        (**self).blas_vector_layout()
    }
}

impl<T: BlasVectorInfos + ?Sized> BlasVectorInfos for &mut T {
    fn blas_vector_layout(&self) -> BlasVectorLayout {
        (**self).blas_vector_layout()
    }
}

/// A vector's shape and ciphertext layout, without ciphertexts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BlasVectorLayout {
    pub ct: CKKSLayout,
    pub shape: BlasVectorShape,
}

impl BlasVectorLayout {
    /// A vector of `shape` on ciphertexts of layout `ct`.
    pub fn new(ct: CKKSLayout, shape: BlasVectorShape) -> Self {
        Self { ct, shape }
    }
}

impl BlasVectorInfos for BlasVectorLayout {
    fn blas_vector_layout(&self) -> BlasVectorLayout {
        *self
    }
}

/// The period of the slots of `v`'s ciphertexts on ring `R` (`BE::Ring`), as their label proves: with `S` slots and
/// label `s`, `S >> s`, `0` past the word width.
pub fn vector_period<R: Ring>(v: &impl BlasVectorInfos) -> usize {
    let ct = v.ct_layout();
    slots_of::<R>(ct.n().as_usize())
        .checked_shr(u32::try_from(ct.log_sparsity()).unwrap_or(u32::MAX))
        .unwrap_or(0)
}

/// A vector held by CKKS ciphertexts, [`BlasVectorShape`] stating where its entries sit. Operations reject a vector
/// whose ciphertexts do not share one layout, whose `log_delta` exceeds its `k`, or whose entries do not fit the period
/// its label proves: with `S` slots, degree `n` and label `s`, `s <= log2 n` and `offset + stride (chunk - 1) < S >> s`.
pub struct BlasVector<D: Data, W: ZnxWord, R: Ring> {
    cts: Vec<CKKSCiphertext<D, W, R>>,
    shape: BlasVectorShape,
}

/// Backend-owned vector.
pub type BlasVectorOwned<BE> = BlasVector<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord, <BE as Backend>::Ring>;

impl<D: Data, W: ZnxWord, R: Ring> BlasVector<D, W, R> {
    /// Reads `cts` as a vector of `shape`, which must count `cts.len()` ciphertexts, at least one.
    pub fn from_cts(cts: Vec<CKKSCiphertext<D, W, R>>, shape: BlasVectorShape) -> Result<Self> {
        ensure!(
            shape.count() > 0 && shape.count() == cts.len(),
            "{} ciphertexts do not hold {} entries of {} per ciphertext",
            cts.len(),
            shape.len,
            shape.chunk
        );
        Ok(Self { cts, shape })
    }

    /// Ciphertext `t`.
    pub fn ct(&self, t: usize) -> &CKKSCiphertext<D, W, R> {
        assert!(t < self.cts.len(), "ciphertext {t} is outside a vector of {}", self.cts.len());
        &self.cts[t]
    }

    /// Ciphertext `t`, mutably.
    pub fn ct_mut(&mut self, t: usize) -> &mut CKKSCiphertext<D, W, R> {
        assert!(t < self.cts.len(), "ciphertext {t} is outside a vector of {}", self.cts.len());
        &mut self.cts[t]
    }

    /// Every ciphertext, in entry order.
    pub fn cts(&self) -> &[CKKSCiphertext<D, W, R>] {
        &self.cts
    }

    /// Every ciphertext, mutably.
    pub fn cts_mut(&mut self) -> &mut [CKKSCiphertext<D, W, R>] {
        &mut self.cts
    }

    /// Returns the ciphertexts, in entry order.
    pub fn into_cts(self) -> Vec<CKKSCiphertext<D, W, R>> {
        self.cts
    }

    /// The vector's header: its shape (u64 each), then the ciphertexts' `n` (u64), `base2k`, `k` (u32) and rank (u64).
    /// Ciphertexts that differ in `n`, `base2k`, `k` or rank, which it cannot describe, are
    /// [`std::io::ErrorKind::InvalidInput`].
    fn header(&self) -> std::io::Result<Vec<u8>> {
        let first = self.cts[0].glwe_layout();
        if self.cts.iter().any(|ct| ct.glwe_layout() != first) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "the ciphertexts of the vector do not share one layout",
            ));
        }
        let mut header = Vec::new();
        for field in [self.shape.len, self.shape.chunk, self.shape.offset, self.shape.stride] {
            header.write_u64::<LittleEndian>(field as u64)?;
        }
        header.write_u64::<LittleEndian>(first.n.as_usize() as u64)?;
        header.write_u32::<LittleEndian>(first.base2k.as_u32())?;
        header.write_u32::<LittleEndian>(first.k.as_u32())?;
        header.write_u64::<LittleEndian>(first.rank.as_usize() as u64)?;
        Ok(header)
    }
}

impl<D: Data, W: ZnxWord, R: Ring> Clone for BlasVector<D, W, R>
where
    CKKSCiphertext<D, W, R>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            cts: self.cts.clone(),
            shape: self.shape,
        }
    }
}

impl<D: Data, W: ZnxWord, R: Ring> BlasVectorInfos for BlasVector<D, W, R> {
    fn blas_vector_layout(&self) -> BlasVectorLayout {
        BlasVectorLayout::new(layout_of(&self.cts[0]), self.shape)
    }
}

impl<D: Data, W: ZnxWord, R: Ring> PartialEq for BlasVector<D, W, R>
where
    GLWE<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.cts.len() == other.cts.len() && self.cts.iter().zip(&other.cts).all(|(a, b)| **a == **b)
    }
}

impl<D: Data, W: ZnxWord, R: Ring> Eq for BlasVector<D, W, R> where GLWE<D, W>: Eq {}

/// The shape, the ciphertext count and the first ciphertext's layout, never a coefficient.
impl<D: Data, W: ZnxWord, R: Ring> std::fmt::Debug for BlasVector<D, W, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlasVector")
            .field("shape", &self.shape)
            .field("cts", &self.cts.len())
            .field("ct_layout", &layout_of(&self.cts[0]))
            .finish()
    }
}

impl<D: HostDataRef, W: ZnxWord, R: Ring> WriterTo for BlasVector<D, W, R> {
    /// Serialises the vector in little-endian binary format: its header, then every ciphertext, in entry order, as a
    /// [`CKKSCiphertext`] of its first `ceil(k / base2k)` limbs, so the stream follows the header's `k`, not the
    /// allocation. Ciphertexts that differ in `n`, `base2k`, `k` or rank, which the header cannot describe, and a
    /// flag-clear ciphertext, fail with [`std::io::ErrorKind::InvalidInput`], writing nothing: normalize it first.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        let header = self.header()?;
        if self.cts.iter().any(|ct| !GLWE::is_canonical(ct)) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "a ciphertext is not canonical: normalize it before serializing",
            ));
        }
        writer.write_all(&header)?;
        self.cts.iter().try_for_each(|ct| ct.write_live_to(writer))
    }
}

impl<D: HostDataMut, W: ZnxWord, R: Ring> ReaderFrom for BlasVector<D, W, R> {
    /// Deserialises into a vector allocated from the writer's layout, or one of more limbs per ciphertext, which then
    /// holds the written limbs: a receiver whose ciphertexts differ in layout fails with
    /// [`std::io::ErrorKind::InvalidInput`], another shape, or ciphertext `n`, `base2k`, `k` or rank with
    /// [`std::io::ErrorKind::InvalidData`], before any ciphertext is read.
    fn read_from<Rd: std::io::Read>(&mut self, reader: &mut Rd) -> std::io::Result<()> {
        let header = self.header()?;
        let mut read = vec![0u8; header.len()];
        reader.read_exact(&mut read)?;
        if read != header {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "the serialized layout differs from the receiver's",
            ));
        }
        self.cts.iter_mut().try_for_each(|ct| ct.read_from(reader))
    }
}

/// A vector the operations read: its ciphertexts, in entry order, shareable across the tasks of a parallel
/// implementation.
pub trait BlasVectorRef<BE: Backend>: BlasVectorInfos {
    /// One ciphertext.
    type Ct: GLWEToBackendRef<BE> + CKKSCtBounds + Sync;

    /// Every ciphertext, in entry order: exactly [`BlasVectorShape::count`] of them, at least one. Every operation
    /// rejects a vector that breaks this.
    fn cts(&self) -> &[Self::Ct];
}

/// A vector the operations write: each ciphertext can move into its own task.
pub trait BlasVectorMut<BE: Backend>: BlasVectorRef<BE, Ct: GLWEToBackendMut<BE> + SetCKKSInfos + Send> {
    /// Every ciphertext, in entry order, mutably.
    fn cts_mut(&mut self) -> &mut [Self::Ct];
}

impl<BE: Backend, D: Data + Sync> BlasVectorRef<BE> for BlasVector<D, BE::ZnxWord, BE::Ring>
where
    CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
{
    type Ct = CKKSCiphertext<D, BE::ZnxWord, BE::Ring>;

    fn cts(&self) -> &[Self::Ct] {
        &self.cts
    }
}

impl<BE: Backend, D: Data + Send + Sync> BlasVectorMut<BE> for BlasVector<D, BE::ZnxWord, BE::Ring>
where
    CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
{
    fn cts_mut(&mut self) -> &mut [Self::Ct] {
        &mut self.cts
    }
}

/// [`super::BlasModuleAlloc::blas_vector_alloc_from_infos`].
pub(crate) fn alloc<BE, M>(module: &M, ct: &CKKSLayout, shape: BlasVectorShape) -> BlasVectorOwned<BE>
where
    BE: Backend,
    M: CKKSModuleAlloc<BE>,
{
    let count = shape.count();
    assert!(count > 0, "a vector has at least one entry and one entry per ciphertext");
    let cts = (0..count).map(|_| module.ckks_ciphertext_alloc_from_infos(ct)).collect();
    BlasVector { cts, shape }
}

#[cfg(test)]
mod tests {
    use poulpy_ckks::SetGLWEPlaintextInfos;
    use poulpy_core::layouts::SetK;
    use poulpy_hal::layouts::{HostBytesBackend, Module, Standard, ZnxView, ZnxViewMut};

    use super::*;
    use crate::layouts::{BlasBytesOf, BlasModuleAlloc};

    #[test]
    fn shapes_count_their_ciphertexts_and_span() {
        let shape = BlasVectorShape::new(10, 4, 1, 3);
        assert_eq!((shape.count(), shape.span()), (3, Some(11)));
        assert_eq!(BlasVectorShape::new(8, 4, 0, 1).count(), 2);
        assert_eq!(BlasVectorShape::new(8, 0, 0, 1).count(), 0);
        assert_eq!(BlasVectorShape::new(8, 0, 0, 1).span(), None);
        assert_eq!(BlasVectorShape::new(8, 4, usize::MAX, 1).span(), None);
        assert!(shape.fits(128, 16));
        assert!(shape.fits(128, 128));
        assert!(!shape.fits(128, 8));
        assert!(!shape.fits(128, 24), "a period that does not divide the slots");
        assert!(!BlasVectorShape::new(0, 4, 0, 1).fits(128, 128));
        assert!(!BlasVectorShape::new(4, 4, 0, 0).fits(128, 128));
    }

    #[test]
    fn vectors_read_their_period_off_the_label() {
        let module = Module::<HostBytesBackend>::new(256);
        let shape = BlasVectorShape::new(6, 4, 0, 2);
        let mut v = module.blas_vector_alloc(shape, 19usize.into(), 60usize.into());
        assert_eq!((v.cts().len(), vector_period::<Standard>(&v)), (2, 128));
        v.cts_mut().iter_mut().for_each(|ct| ct.set_log_sparsity(3));
        assert_eq!(vector_period::<Standard>(&v), 16);
        v.cts_mut().iter_mut().for_each(|ct| ct.set_log_sparsity(64));
        assert_eq!(vector_period::<Standard>(&v), 0);
        assert!(BlasVector::from_cts(v.clone().into_cts(), BlasVectorShape::new(9, 4, 0, 2)).is_err());
        assert!(BlasVectorOwned::<HostBytesBackend>::from_cts(Vec::new(), BlasVectorShape::new(0, 4, 0, 2)).is_err());
    }

    #[test]
    fn vectors_round_trip_through_their_serialization() {
        let module = Module::<HostBytesBackend>::new(64);
        let shape = BlasVectorShape::new(5, 2, 1, 4);
        let mut v = module.blas_vector_alloc(shape, 13usize.into(), 39usize.into());
        for (t, ct) in v.cts_mut().iter_mut().enumerate() {
            ct.data_mut().raw_mut().iter_mut().for_each(|x| *x = t as i64 - 1);
            ct.set_log_sparsity(1);
        }
        let mut bytes = Vec::new();
        v.write_to(&mut bytes).unwrap();
        let mut back = module.blas_vector_alloc(shape, 13usize.into(), 39usize.into());
        back.cts_mut()[0].data_mut().raw_mut()[0] = 7;
        back.read_from(&mut bytes.as_slice()).unwrap();
        assert_eq!(back, v, "a read vector is the original, digits and metadata");
        let mut again = Vec::new();
        back.write_to(&mut again).unwrap();
        assert!(again == bytes, "a read vector serializes as the original");

        // Another shape, radix or precision is rejected before any ciphertext is read, the receiver left equal.
        for (shape, base2k, k) in [
            (BlasVectorShape::new(5, 2, 0, 4), 13usize, 39usize),
            (shape, 12, 39),
            (shape, 13, 38),
        ] {
            let mut other = module.blas_vector_alloc(shape, base2k.into(), k.into());
            other.cts_mut()[0].data_mut().raw_mut()[0] = 7;
            let before = other.clone();
            let err = other.read_from(&mut bytes.as_slice()).unwrap_err();
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{shape:?} {base2k} {k}");
            assert_eq!(other, before, "a rejected read changed the vector: {shape:?} {base2k} {k}");
        }
    }

    #[test]
    fn vectors_narrowed_below_their_allocation_read_into_their_layout() {
        let module = Module::<HostBytesBackend>::new(64);
        let shape = BlasVectorShape::new(4, 2, 0, 1);
        // Three limbs allocated, two live after `k` drops to 26, as an operation writing a narrower result leaves them.
        let narrowed = |fill: bool| {
            let mut v = module.blas_vector_alloc(shape, 13usize.into(), 39usize.into());
            for (t, ct) in v.cts_mut().iter_mut().enumerate() {
                if fill {
                    for (col, limb) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        ct.data_mut().at_mut(col, limb).fill(t as i64 + col as i64 - limb as i64);
                    }
                }
                ct.set_k(26usize.into());
                GLWE::set_canonical(ct, true);
            }
            v
        };
        let v = narrowed(true);
        let mut bytes = Vec::new();
        v.write_to(&mut bytes).unwrap();
        for (label, mut back) in [
            ("allocated from its layout", module.blas_vector_alloc_from_infos(&v)),
            ("allocated as the writer", narrowed(false)),
        ] {
            back.read_from(&mut bytes.as_slice()).unwrap();
            assert_eq!(back.blas_vector_layout(), v.blas_vector_layout(), "{label}");
            for (have, want) in back.cts().iter().zip(v.cts()) {
                assert_eq!(have.data().size(), 2, "{label}: the live limbs");
                for (col, limb) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    assert_eq!(have.data().at(col, limb), want.data().at(col, limb), "{label}");
                }
            }
            let mut again = Vec::new();
            back.write_to(&mut again).unwrap();
            assert!(again == bytes, "{label}: a read vector serializes as the original");
        }
    }

    #[test]
    fn vector_bytes_match_their_allocation() {
        let module = Module::<HostBytesBackend>::new(64);
        let v = module.blas_vector_alloc(BlasVectorShape::new(5, 2, 1, 4), 13usize.into(), 39usize.into());
        let held: usize = v
            .cts()
            .iter()
            .map(|ct| <HostBytesBackend as Backend>::len_bytes(ct.data().data()))
            .sum();
        assert_eq!(module.blas_vector_bytes_of_from_infos(&v), held);
        assert_eq!(module.blas_vector_bytes_of_from_infos(&v.blas_vector_layout()), held);
        let from_infos = module.blas_vector_alloc_from_infos(&v.blas_vector_layout());
        assert_eq!(from_infos.blas_vector_layout(), v.blas_vector_layout());
    }

    #[test]
    fn vector_write_rejects_a_flag_clear_ciphertext_writing_nothing() {
        let module = Module::<HostBytesBackend>::new(64);
        let mut v = module.blas_vector_alloc(BlasVectorShape::new(4, 2, 0, 1), 13usize.into(), 39usize.into());
        GLWE::set_canonical(v.ct_mut(1), false);
        let mut bytes = Vec::new();
        assert_eq!(v.write_to(&mut bytes).unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
        assert!(bytes.is_empty());
    }

    #[test]
    fn vectors_of_ciphertexts_of_different_layouts_neither_write_nor_read() {
        let module = Module::<HostBytesBackend>::new(64);
        let shape = BlasVectorShape::new(4, 2, 0, 1);
        let cts = (0..2usize)
            .map(|t| module.ckks_ciphertext_alloc(13usize.into(), (26 + 13 * t).into()))
            .collect();
        let mut mixed = BlasVector::from_cts(cts, shape).unwrap();
        let mut bytes = Vec::new();
        assert_eq!(
            mixed.write_to(&mut bytes).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert!(bytes.is_empty());

        // A stream whose header matches the first ciphertext is rejected before any ciphertext is read.
        let mut uniform = module.blas_vector_alloc(shape, 13usize.into(), 26usize.into());
        uniform.ct_mut(1).data_mut().raw_mut()[0] = 7;
        uniform.write_to(&mut bytes).unwrap();
        let before = mixed.clone();
        assert_eq!(
            mixed.read_from(&mut bytes.as_slice()).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert_eq!(mixed, before, "a rejected read changed the vector");
    }
}
