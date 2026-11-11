use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, Ring, Standard, WriterTo};

use crate::layouts::{
    Base2K, Degree, Dnum, GGLWE, GGLWEAtBackendMut, GGLWEAtBackendRef, GGLWEAtViewMut, GGLWEAtViewRef, GGLWEBackendMut,
    GGLWEBackendRef, GGLWEInfos, GGLWELayout, GGLWEToBackendMut, GGLWEToBackendRef, GLWE, GLWEInfos, GLWEViewMut, GLWEViewRef,
    LWEInfos, Rank, TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::layouts::ZnxWord;

use std::fmt;

/// Plain-data descriptor for a [`GLWEAutomorphismKey`] carrying only the
/// layout parameters (no backing buffer).
///
/// Implements [`LWEInfos`], [`GLWEInfos`] and [`GGLWEInfos`] so it can
/// be passed to any generic constructor that needs layout information.
/// For an automorphism key `rank_in == rank_out`. It has no row offset (see
/// [`GGLWEInfos::k_offset`]): no automorphism takes a key with one.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWEAutomorphismKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub dnum: Dnum,
    pub k_aux: TorusPrecision,
    pub rank: Rank,
    pub k_digit: TorusPrecision,
}

/// The automorphism key of a stored GGLWE layout (`rank_out` is its rank).
///
/// # Panics
///
/// On a coarsened view (`stride != 1`), which no key is stored as, and on
/// a row offset, which no automorphism key has.
impl From<GGLWELayout> for GLWEAutomorphismKeyLayout {
    fn from(layout: GGLWELayout) -> Self {
        assert_eq!(layout.stride, 1, "a coarsened view is not a stored layout");
        assert_eq!(layout.k_offset.0, 0, "an automorphism key has no row offset");
        Self {
            n: layout.n,
            base2k: layout.base2k,
            dnum: layout.dnum,
            k_aux: layout.k_aux,
            rank: layout.rank_out,
            k_digit: layout.k_digit,
        }
    }
}
/// GLWE automorphism (Galois) key.
///
/// Wraps a [`GGLWE`] together with the Galois element index `p` that
/// identifies which automorphism this key materialises.
///
/// `D: Data` is the backing storage type (e.g. `AlignedBuf`, `&[u8]`,
/// `&mut [u8]`).
#[derive(PartialEq, Eq, Clone)]
pub struct GLWEAutomorphismKey<D: Data, W: ZnxWord, Rg: Ring = Standard> {
    pub(crate) key: GGLWE<D, W, Rg>,
    pub(crate) p: i64,
}

/// Provides read access to the Galois element index `p`.
pub trait GetGaloisElement {
    /// Returns the Galois element index.
    fn p(&self) -> i64;
}

impl<T: GetGaloisElement + ?Sized> GetGaloisElement for &T {
    fn p(&self) -> i64 {
        (**self).p()
    }
}

impl<T: GetGaloisElement + ?Sized> GetGaloisElement for &mut T {
    fn p(&self) -> i64 {
        (**self).p()
    }
}

/// Provides write access to the Galois element index `p`.
pub trait SetGaloisElement {
    /// Sets the Galois element index.
    fn set_p(&mut self, p: i64);
}

impl<D: Data, W: ZnxWord, Rg: Ring> SetGaloisElement for GLWEAutomorphismKey<D, W, Rg> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GetGaloisElement for GLWEAutomorphismKey<D, W, Rg> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEAutomorphismKey<D, W, Rg> {
    /// Returns the Galois element index `p`.
    pub fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWEInfos for GLWEAutomorphismKey<D, W, Rg> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.key.noise()
    }

    fn n(&self) -> Degree {
        self.key.n()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn max_size(&self) -> usize {
        self.key.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.key.k()
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEInfos for GLWEAutomorphismKey<D, W, Rg> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GGLWEInfos for GLWEAutomorphismKey<D, W, Rg> {
    fn k_aux(&self) -> TorusPrecision {
        self.key.k_aux()
    }

    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.key.k_digit()
    }
    fn k_offset(&self) -> TorusPrecision {
        self.key.k_offset()
    }

    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }
}

impl LWEInfos for GLWEAutomorphismKeyLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn max_size(&self) -> usize {
        crate::layouts::key_size(self.base2k, self.dnum, self.k_digit, self.k_aux)
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::key_k(self.dnum, self.k_digit, self.k_aux)
    }
}

impl GLWEInfos for GLWEAutomorphismKeyLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl GGLWEInfos for GLWEAutomorphismKeyLayout {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }

    fn rank_in(&self) -> Rank {
        self.rank
    }

    fn k_digit(&self) -> TorusPrecision {
        self.k_digit
    }
    fn k_offset(&self) -> TorusPrecision {
        TorusPrecision(0)
    }

    fn rank_out(&self) -> Rank {
        self.rank
    }
}

impl<BE: Backend, Rg: Ring> GGLWEAtBackendRef<BE, Rg> for GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>, BE::ZnxWord, Rg> {
        <GGLWE<BE::OwnedBuf, BE::ZnxWord, Rg> as GGLWEAtBackendRef<BE, Rg>>::at_backend(&self.key, row, col)
    }
}

impl<BE: Backend, Rg: Ring> GGLWEAtBackendMut<BE, Rg> for GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>, BE::ZnxWord, Rg> {
        <GGLWE<BE::OwnedBuf, BE::ZnxWord, Rg> as GGLWEAtBackendMut<BE, Rg>>::at_backend_mut(&mut self.key, row, col)
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Debug for GLWEAutomorphismKey<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Display for GLWEAutomorphismKey<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(AutomorphismKey: p={}) {}", self.p, self.key)
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord, Rg: Ring> GLWEAutomorphismKey<AlignedBuf, W, Rg> {
    /// Allocates a new [`GLWEAutomorphismKey`] with the given parameters.
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.k_digit(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    /// Allocates a new [`GLWEAutomorphismKey`] with the given parameters.
    pub(crate) fn alloc(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> Self {
        GLWEAutomorphismKey {
            key: GGLWE::alloc(n, base2k, dnum, k_digit, k_aux, rank, rank),
            p: 0,
        }
    }

    /// Returns the byte count required for a [`GLWEAutomorphismKey`] with the given parameters.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for AutomorphismKey"
        );
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.k_digit(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    /// Returns the byte count required for a [`GLWEAutomorphismKey`] with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, k_digit: TorusPrecision, k_aux: TorusPrecision, rank: Rank) -> usize {
        GGLWE::<AlignedBuf, W, Rg>::bytes_of(n, base2k, dnum, k_digit, k_aux, rank, rank)
    }
}

impl_gglwe_to_backend_for_field!(GLWEAutomorphismKey<D, BE::ZnxWord, Rg>, key, GGLWE<D, BE::ZnxWord, Rg>);

impl_gglwe_at_view_for_field!(GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord, Rg>; key);

impl<D: Data, W: ZnxWord, Rg: Ring> SetGaloisElement for &mut GLWEAutomorphismKey<D, W, Rg> {
    fn set_p(&mut self, p: i64) {
        self.p = p;
    }
}

impl_glwe_host_at_for_field!(GLWEAutomorphismKey<D, W, Rg>; key);

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> ReaderFrom for GLWEAutomorphismKey<D, W, Rg> {
    /// Deserialises from little-endian binary format. A stream of another ring, or whose Galois element is even or not
    /// below the cyclotomic order of the ring at the stream's degree, is [`std::io::ErrorKind::InvalidData`], the
    /// Galois element and noise metadata left unset.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.key.noise = None;
        let p = reader.read_i64::<LittleEndian>()?;
        self.key.read_from(reader)?;
        if let Err(err) = crate::layouts::validate_galois_element::<Rg>(p, self.key.n(), std::io::ErrorKind::InvalidData) {
            self.key.noise = None;
            return Err(err);
        }
        self.p = p;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> WriterTo for GLWEAutomorphismKey<D, W, Rg> {
    /// Serialises in little-endian binary format: the Galois element `p` (`i64`), then the key. A `p` even or not below
    /// the ring's cyclotomic order is [`std::io::ErrorKind::InvalidInput`], writing nothing.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        if let Some(noise) = &self.key.noise {
            noise.validate_wire()?;
            noise.validate_components(self.key.data.cols_out())?;
        }
        crate::layouts::validate_galois_element::<Rg>(self.p, self.key.n(), std::io::ErrorKind::InvalidInput)?;
        writer.write_i64::<LittleEndian>(self.p)?;
        self.key.write_to(writer)
    }
}
