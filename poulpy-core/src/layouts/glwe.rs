use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, ReaderFrom, ToOwnedDeep, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, WriterTo,
};

use crate::layouts::{Base2K, Degree, LWEInfos, Rank, SetBase2k, SetK, TorusPrecision};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::layouts::ZnxWord;
use std::fmt;

/// Trait providing the parameter accessors for a GLWE (Generalised LWE) ciphertext.
///
/// A GLWE ciphertext is a polynomial-ring LWE ciphertext consisting of
/// a body polynomial and `rank` mask polynomials, all defined over `Z[X]/(X^n + 1)`.
/// Extends [`LWEInfos`] with the GLWE rank.
pub trait GLWEInfos
where
    Self: LWEInfos,
{
    /// Returns the GLWE rank (number of mask polynomials).
    fn rank(&self) -> Rank;
    /// Returns a plain-data [`GLWELayout`] snapshot of the current parameters.
    fn glwe_layout(&self) -> GLWELayout {
        GLWELayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.k(),
            rank: self.rank(),
        }
    }
}

impl<T: GLWEInfos + ?Sized> GLWEInfos for &T {
    fn rank(&self) -> Rank {
        (**self).rank()
    }
}

impl<T: GLWEInfos + ?Sized> GLWEInfos for &mut T {
    fn rank(&self) -> Rank {
        (**self).rank()
    }
}

/// Plain-data snapshot of the parameters that describe a [`GLWE`] ciphertext.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWELayout {
    /// Ring degree.
    pub n: Degree,
    /// Base-2-log of the limb width.
    pub base2k: Base2K,
    /// Torus precision.
    pub k: TorusPrecision,
    /// Number of mask polynomials.
    pub rank: Rank,
}

impl LWEInfos for GLWELayout {
    fn n(&self) -> Degree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn max_size(&self) -> usize {
        self.k.div_ceil(self.base2k) as usize
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl GLWEInfos for GLWELayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

/// A GLWE (Generalised LWE) ciphertext over the polynomial ring `Z[X]/(X^n + 1)`.
///
/// Wraps a [`VecZnx`] with `rank + 1` columns: the first column is the body
/// polynomial, and the remaining `rank` columns are the mask polynomials.
///
/// `D: Data` is the storage backend (e.g. `AlignedBuf`, `&[u8]`, `&mut [u8]`).
///
/// # Normalized form
///
/// A normalized GLWE is canonical at its `k`: its `ceil(k / base2k)` live limbs
/// hold digits in `[-2^(base2k - 1), 2^(base2k - 1)]`, the low
/// `(base2k - k % base2k) % base2k` bits of the bottom live limb are zero and
/// every limb past the live ones is zero. That is what normalizing at `k`
/// produces.
///
/// # Canonical flag
///
/// [`GLWE::is_canonical`] states that the data is in normalized form.
/// Normalizing operations set it, additions and subtractions clear it, digit
/// permutations keep the source's, lowering `k` clears it. Operations that read
/// the digits through a DFT normalize a flag-clear operand first. Serialization
/// rejects a flag-clear GLWE, so a loaded GLWE is flagged canonical; equality
/// ignores the flag.
/// Only a `GLWE` stores it: the GLWE views of GGLWE and GGSW rows, tensors and
/// plaintexts report it set and drop a clear, so their data must stay canonical;
/// normalize after writing a flag-clearing result into one.
#[derive(Clone)]
pub struct GLWE<D: Data, W: ZnxWord> {
    pub(crate) data: VecZnx<D, W>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) canonical: bool,
}

impl<D: Data, W: ZnxWord> PartialEq for GLWE<D, W>
where
    VecZnx<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.k == other.k && self.base2k == other.base2k
    }
}

impl<D: Data, W: ZnxWord> Eq for GLWE<D, W> where VecZnx<D, W>: Eq {}

pub type GLWEBackendRef<'a, BE> = GLWE<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord>;
pub type GLWEBackendMut<'a, BE> = GLWE<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord>;

impl<D: Data, W: ZnxWord> SetBase2k for GLWE<D, W> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data, W: ZnxWord> SetBase2k for &mut GLWE<D, W> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data, W: ZnxWord> SetK for GLWE<D, W> {
    fn set_k(&mut self, k: TorusPrecision) {
        self.canonical &= k >= self.k;
        self.k = k
    }
}

impl<D: Data, W: ZnxWord> SetK for &mut GLWE<D, W> {
    fn set_k(&mut self, k: TorusPrecision) {
        (**self).set_k(k)
    }
}

impl<D: Data, W: ZnxWord> GLWE<D, W> {
    /// Returns a shared reference to the underlying [`VecZnx`].
    pub fn data(&self) -> &VecZnx<D, W> {
        &self.data
    }

    pub fn is_canonical(&self) -> bool {
        self.canonical
    }

    /// For data written directly into the limbs, or to feed tolerated
    /// non-canonical digits to a DFT operation without its normalization.
    pub fn set_canonical(&mut self, canonical: bool) {
        self.canonical = canonical
    }
}

impl<D: Data, W: ZnxWord> GLWE<D, W> {
    /// Returns a mutable reference to the underlying [`VecZnx`].
    pub fn data_mut(&mut self) -> &mut VecZnx<D, W> {
        &mut self.data
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWE<D, W> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn max_size(&self) -> usize {
        self.data.size()
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GLWE<D, W> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: HostDataRef, W: ZnxWord> ToOwnedDeep for GLWE<D, W> {
    type Owned = GLWE<AlignedBuf, W>;
    fn to_owned_deep(&self) -> Self::Owned {
        GLWE {
            data: self.data.to_owned_deep(),
            base2k: self.base2k,
            k: self.k,
            canonical: self.canonical,
        }
    }
}

impl<D: Data, W: ZnxWord> GLWE<D, W> {
    /// Rebuilds this backend-owned ciphertext as a host-owned [`GLWE<AlignedBuf, W>`].
    pub fn to_host_owned<BE>(&self) -> GLWE<AlignedBuf, W>
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        GLWE {
            data: self.data.to_host_owned::<BE>(),
            base2k: self.base2k,
            k: self.k,
            canonical: self.canonical,
        }
    }

    /// Formats this backend-owned ciphertext through the existing host [`fmt::Display`] implementation.
    pub fn display_host<BE>(&self) -> String
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        self.to_host_owned::<BE>().to_string()
    }
}

impl<D: Data, W: ZnxWord> GLWE<D, W> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GLWE<To::OwnedBuf, To::ZnxWord>
    where
        To: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        let shape = self.data.shape();
        let data = self.data.into_data();
        GLWE {
            data: VecZnx::from_shape(data, shape),
            base2k: self.base2k,
            k: self.k,
            canonical: self.canonical,
        }
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWE<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GLWE<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GLWE: base2k={} k={}: {}", self.base2k().0, self.k().0, self.data)
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GLWE<AlignedBuf, W> {
    /// Allocates a new [`GLWE`] with the given parameters.
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    /// Allocates a new [`GLWE`] with the given parameters.
    ///
    /// * `n` -- ring degree.
    /// * `base2k` -- base-2-log of the limb width.
    /// * `k` -- torus precision.
    /// * `rank` -- number of mask polynomials.
    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        GLWE {
            data: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<AlignedBuf, W>::bytes_of(
                    n.into(),
                    (rank + 1).into(),
                    size,
                )),
                n.into(),
                (rank + 1).into(),
                size,
            ),
            base2k,
            k,
            canonical: true,
        }
    }

    /// Returns the byte count required for a [`GLWE`] with the given parameters.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    /// Returns the byte count required for a [`GLWE`] with the given parameters.
    ///
    /// * `n` -- ring degree.
    /// * `base2k` -- base-2-log of the limb width.
    /// * `k` -- torus precision.
    /// * `rank` -- number of mask polynomials.
    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        VecZnx::<AlignedBuf, W>::bytes_of(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWE<D, W> {
    /// Deserialises a [`GLWE`] in little-endian binary format.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.set_base2k(Base2K(reader.read_u32::<LittleEndian>()?));
        self.data.read_from(reader)?;
        self.canonical = true;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWE<D, W> {
    /// Serialises the [`GLWE`] in little-endian binary format.
    ///
    /// Fails with [`std::io::ErrorKind::InvalidInput`], writing nothing, when the
    /// canonical flag is clear: normalize first.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        if !self.canonical {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "GLWE is not canonical: normalize it before serializing",
            ));
        }
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        self.data.write_to(writer)
    }
}

pub trait GLWEToBackendRef<BE: Backend>: Sized {
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE>;

    fn is_canonical(&self) -> bool {
        self.to_backend_ref().is_canonical()
    }
}

impl<BE: Backend, D: Data> GLWEToBackendRef<BE> for GLWE<D, BE::ZnxWord>
where
    VecZnx<D, BE::ZnxWord>: VecZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
        GLWE {
            base2k: self.base2k,
            k: self.k,
            canonical: self.canonical,
            data: self.data.to_backend_ref(),
        }
    }
}

pub fn glwe_backend_ref_from_ref<'a, 'b, BE: Backend>(glwe: &'a GLWE<BE::BufRef<'b>, BE::ZnxWord>) -> GLWEBackendRef<'a, BE> {
    GLWE {
        base2k: glwe.base2k,
        k: glwe.k,
        canonical: glwe.canonical,
        data: poulpy_hal::layouts::vec_znx_backend_ref_from_ref::<BE>(&glwe.data),
    }
}

impl<BE: Backend> GLWEToBackendRef<BE> for &GLWE<BE::BufRef<'_>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
        glwe_backend_ref_from_ref::<BE>(self)
    }
}

pub fn glwe_backend_ref_from_mut<'a, 'b, BE: Backend>(glwe: &'a GLWE<BE::BufMut<'b>, BE::ZnxWord>) -> GLWEBackendRef<'a, BE> {
    GLWE {
        base2k: glwe.base2k,
        k: glwe.k,
        canonical: glwe.canonical,
        data: poulpy_hal::layouts::vec_znx_backend_ref_from_mut::<BE>(&glwe.data),
    }
}

pub trait GLWEToBackendMut<BE: Backend>: GLWEToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE>;

    /// Sets the owner's canonical flag; a flag set on the view returned by
    /// [`Self::to_backend_mut`] is lost. Types without a flag ignore it.
    fn set_canonical(&mut self, canonical: bool);
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for GLWE<D, BE::ZnxWord>
where
    VecZnx<D, BE::ZnxWord>: VecZnxToBackendRef<BE> + VecZnxToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE> {
        GLWE {
            base2k: self.base2k,
            k: self.k,
            canonical: self.canonical,
            data: self.data.to_backend_mut(),
        }
    }

    fn set_canonical(&mut self, canonical: bool) {
        self.canonical = canonical
    }
}

impl<BE: Backend> GLWEToBackendRef<BE> for &mut GLWE<BE::BufMut<'_>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
        glwe_backend_ref_from_mut::<BE>(self)
    }
}

impl<BE: Backend> GLWEToBackendMut<BE> for &mut GLWE<BE::BufMut<'_>, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE> {
        glwe_backend_mut_from_mut::<BE>(self)
    }

    fn set_canonical(&mut self, canonical: bool) {
        self.canonical = canonical
    }
}

pub fn glwe_backend_mut_from_mut<'a, 'b, BE: Backend>(glwe: &'a mut GLWE<BE::BufMut<'b>, BE::ZnxWord>) -> GLWEBackendMut<'a, BE> {
    GLWE {
        base2k: glwe.base2k,
        k: glwe.k,
        canonical: glwe.canonical,
        data: poulpy_hal::layouts::vec_znx_backend_mut_from_mut::<BE>(&mut glwe.data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_to_rejects_flag_clear_glwe() {
        let mut glwe = GLWE::<AlignedBuf, i64>::alloc(Degree(8), Base2K(12), TorusPrecision(33), Rank(1));
        glwe.set_canonical(false);
        let mut bytes = Vec::new();
        let err = glwe.write_to(&mut bytes).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(bytes.is_empty());
        glwe.set_canonical(true);
        glwe.write_to(&mut bytes).unwrap();
    }
}
