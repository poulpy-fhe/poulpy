use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, ReaderFrom, ToOwnedDeep, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, WriterTo,
};
use poulpy_hal::{AlignedBuf, alloc_aligned};

use crate::layouts::{
    Base2K, Degree, GLWEPlaintextInfos, GLWEPlaintextMeta, LWEInfos, Rank, SetBase2k, SetGLWEPlaintextInfos, SetK, TorusPrecision,
};
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
///
/// # Plaintext metadata
///
/// [`GLWEPlaintextMeta`] records what the plaintext is, for the scheme that
/// manages it; core operations never read or write it.
#[derive(Clone)]
pub struct GLWE<D: Data, W: ZnxWord> {
    pub(crate) noise: Option<crate::ComponentNoise>,
    pub(crate) data: VecZnx<D, W>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) canonical: bool,
    pub(crate) plaintext_meta: Option<GLWEPlaintextMeta>,
}

impl<D: Data, W: ZnxWord> PartialEq for GLWE<D, W>
where
    VecZnx<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.noise == other.noise
            && self.data == other.data
            && self.k == other.k
            && self.base2k == other.base2k
            && self.plaintext_meta == other.plaintext_meta
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
    /// Narrowing drops the canonical flag and the noise estimate.
    fn set_k(&mut self, k: TorusPrecision) {
        if k < self.k {
            self.canonical = false;
            self.noise = None;
        }
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
        self.noise = None;
        &mut self.data
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWE<D, W> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.noise.clone()
    }

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

impl<D: Data, W: ZnxWord> GLWEPlaintextInfos for GLWE<D, W> {
    fn plaintext_meta(&self) -> Option<GLWEPlaintextMeta> {
        self.plaintext_meta
    }
}

impl<D: Data, W: ZnxWord> SetGLWEPlaintextInfos for GLWE<D, W> {
    fn set_plaintext_meta(&mut self, meta: Option<GLWEPlaintextMeta>) {
        self.plaintext_meta = meta
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
            noise: self.noise.clone(),
            data: self.data.to_owned_deep(),
            base2k: self.base2k,
            k: self.k,
            canonical: self.canonical,
            plaintext_meta: self.plaintext_meta,
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
            noise: self.noise.clone(),
            data: self.data.to_host_owned::<BE>(),
            base2k: self.base2k,
            k: self.k,
            canonical: self.canonical,
            plaintext_meta: self.plaintext_meta,
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
            noise: self.noise.clone(),
            data: VecZnx::from_shape(data, shape),
            base2k: self.base2k,
            k: self.k,
            canonical: self.canonical,
            plaintext_meta: self.plaintext_meta,
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
            noise: None,
            data: VecZnx::from_data(
                alloc_aligned::<u8>(VecZnx::<AlignedBuf, W>::bytes_of(n.into(), (rank + 1).into(), size)),
                n.into(),
                (rank + 1).into(),
                size,
            ),
            base2k,
            k,
            canonical: true,
            plaintext_meta: None,
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
        self.noise = None;
        let components = self.data.cols();
        let noise = crate::ComponentNoise::read_optional(reader, components)?;
        let base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        let plaintext_meta = GLWEPlaintextMeta::read_from(reader, self.data.n())?;
        crate::layouts::read_vec_znx_with_shape(&mut self.data, reader, None, components)?;
        crate::layouts::validate_noise_components(noise.as_ref(), self.data.cols())?;
        self.set_base2k(base2k);
        self.noise = noise;
        self.canonical = true;
        self.plaintext_meta = plaintext_meta;
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
        crate::layouts::validate_noise_components(self.noise.as_ref(), self.data.cols())?;
        crate::ComponentNoise::write_optional(self.noise.as_ref(), writer)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        GLWEPlaintextMeta::write_to(&self.plaintext_meta, writer)?;
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
            noise: self.noise.clone(),
            base2k: self.base2k,
            k: self.k,
            canonical: self.canonical,
            plaintext_meta: self.plaintext_meta,
            data: self.data.to_backend_ref(),
        }
    }
}

pub fn glwe_backend_ref_from_ref<'a, 'b, BE: Backend>(glwe: &'a GLWE<BE::BufRef<'b>, BE::ZnxWord>) -> GLWEBackendRef<'a, BE> {
    GLWE {
        noise: crate::layouts::LWEInfos::noise(&glwe),
        base2k: glwe.base2k,
        k: glwe.k,
        canonical: glwe.canonical,
        plaintext_meta: glwe.plaintext_meta,
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
        noise: crate::layouts::LWEInfos::noise(&glwe),
        base2k: glwe.base2k,
        k: glwe.k,
        canonical: glwe.canonical,
        plaintext_meta: glwe.plaintext_meta,
        data: poulpy_hal::layouts::vec_znx_backend_ref_from_mut::<BE>(&glwe.data),
    }
}

pub trait GLWEToBackendMut<BE: Backend>: GLWEToBackendRef<BE> {
    /// Backend hook for recording or propagating component noise metadata.
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);

    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE>;

    /// Sets the owner's canonical flag; a flag set on the view returned by
    /// [`Self::to_backend_mut`] is lost. Types without a flag ignore it.
    fn set_canonical(&mut self, canonical: bool);
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for GLWE<D, BE::ZnxWord>
where
    VecZnx<D, BE::ZnxWord>: VecZnxToBackendRef<BE> + VecZnxToBackendMut<BE>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE> {
        self.noise = None;
        GLWE {
            noise: None,
            base2k: self.base2k,
            k: self.k,
            canonical: self.canonical,
            plaintext_meta: self.plaintext_meta,
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
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE> {
        glwe_backend_mut_from_mut::<BE>(self)
    }

    fn set_canonical(&mut self, canonical: bool) {
        self.canonical = canonical
    }
}

pub fn glwe_backend_mut_from_mut<'a, 'b, BE: Backend>(glwe: &'a mut GLWE<BE::BufMut<'b>, BE::ZnxWord>) -> GLWEBackendMut<'a, BE> {
    glwe.noise = None;
    GLWE {
        noise: None,
        base2k: glwe.base2k,
        k: glwe.k,
        canonical: glwe.canonical,
        plaintext_meta: glwe.plaintext_meta,
        data: poulpy_hal::layouts::vec_znx_backend_mut_from_mut::<BE>(&mut glwe.data),
    }
}

impl<D: Data, W: ZnxWord> GLWE<D, W> {
    pub(crate) fn record_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, self.data.cols());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutable_access_clears_metadata() {
        use poulpy_hal::layouts::HostBytesBackend;
        let mut glwe = GLWE::<AlignedBuf, i64>::alloc(Degree(8), Base2K(12), TorusPrecision(33), Rank(1));
        let fresh = Some(crate::ComponentNoise::from_secret(crate::Distribution::TernaryProb(0.5), 1));
        GLWEToBackendMut::<HostBytesBackend>::set_noise(&mut glwe, fresh.clone());
        assert_eq!(GLWEToBackendRef::<HostBytesBackend>::to_backend_ref(&glwe).noise(), fresh);
        assert_eq!(GLWEToBackendMut::<HostBytesBackend>::to_backend_mut(&mut glwe).noise(), None);
        assert_eq!(glwe.noise(), None);
        GLWEToBackendMut::<HostBytesBackend>::set_noise(&mut glwe, fresh.clone());
        glwe.data_mut();
        assert_eq!(glwe.noise(), None);
        GLWEToBackendMut::<HostBytesBackend>::set_noise(&mut glwe, fresh.clone());
        glwe.set_k(TorusPrecision(40));
        assert_eq!(glwe.noise(), fresh);
        glwe.set_k(TorusPrecision(30));
        assert_eq!(glwe.noise(), None);
    }

    #[test]
    fn failed_rank_change_read_clears_old_metadata() {
        use crate::{ComponentNoise, Distribution};
        let mut source = GLWE::<AlignedBuf, i64>::alloc(Degree(8), Base2K(12), TorusPrecision(33), Rank(1));
        source.noise = Some(ComponentNoise::from_secret_at(Distribution::TernaryProb(0.3), source.k(), 1));
        let mut receiver = GLWE::<AlignedBuf, i64>::alloc(Degree(8), Base2K(12), TorusPrecision(33), Rank(2));
        receiver.noise = Some(ComponentNoise::from_secret_at(
            Distribution::TernaryProb(0.3),
            receiver.k(),
            2,
        ));
        let mut bytes = Vec::new();
        source.write_to(&mut bytes).unwrap();
        assert!(receiver.read_from(&mut bytes.as_slice()).is_err());
        assert!(receiver.noise().is_none());
        receiver.write_to(&mut Vec::new()).unwrap();
    }

    #[test]
    fn serialization_rejects_noise_with_a_different_component_count() {
        let mut glwe = GLWE::<AlignedBuf, i64>::alloc(Degree(8), Base2K(12), TorusPrecision(33), Rank(1));
        let invalid = crate::ComponentNoise::from_secret_at(crate::Distribution::TernaryProb(0.5), glwe.k(), 2);

        let mut no_noise = Vec::new();
        crate::ComponentNoise::write_optional(None, &mut no_noise).unwrap();
        let mut valid = Vec::new();
        glwe.write_to(&mut valid).unwrap();
        let mut invalid_stream = Vec::new();
        crate::ComponentNoise::write_optional(Some(&invalid), &mut invalid_stream).unwrap();
        invalid_stream.extend_from_slice(&valid[no_noise.len()..]);
        assert_eq!(
            glwe.read_from(&mut invalid_stream.as_slice()).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert!(glwe.noise().is_none());

        glwe.noise = Some(invalid);
        let mut output = Vec::new();
        assert_eq!(
            glwe.write_to(&mut output).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert!(output.is_empty());
    }

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
