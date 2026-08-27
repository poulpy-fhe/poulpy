use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEAtBackendMut, GGLWEAtBackendRef, GGLWEAtViewMut, GGLWEAtViewRef, GGLWEBackendMut,
    GGLWEBackendRef, GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GLWE, GLWEInfos, GLWEViewMut, GLWEViewRef, LWEInfos, Rank,
    TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::layouts::ZnxWord;

use std::fmt;

/// Plain-data descriptor for a [`GLWEAutomorphismKey`] carrying only the
/// layout parameters (no backing buffer).
///
/// Implements [`LWEInfos`], [`GLWEInfos`] and [`GGLWEInfos`] so it can
/// be passed to any generic constructor that needs layout information.
/// For an automorphism key `rank_in == rank_out`.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWEAutomorphismKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub dnum: Dnum,
    pub k_aux: TorusPrecision,
    pub rank: Rank,
    pub dsize: Dsize,
}
/// GLWE automorphism (Galois) key.
///
/// Wraps a [`GGLWE`] together with the Galois element index `p` that
/// identifies which automorphism this key materialises.
///
/// `D: Data` is the backing storage type (e.g. `Vec<u8>`, `&[u8]`,
/// `&mut [u8]`).
#[derive(PartialEq, Eq, Clone)]
pub struct GLWEAutomorphismKey<D: Data, W: ZnxWord> {
    pub(crate) key: GGLWE<D, W>,
    pub(crate) p: i64,
}

/// Provides read access to the Galois element index `p`.
pub trait GetGaloisElement {
    /// Returns the Galois element index.
    fn p(&self) -> i64;
}

/// Provides write access to the Galois element index `p`.
pub trait SetGaloisElement {
    /// Sets the Galois element index.
    fn set_p(&mut self, p: i64);
}

impl<D: Data, W: ZnxWord> SetGaloisElement for GLWEAutomorphismKey<D, W> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<D: Data, W: ZnxWord> GetGaloisElement for GLWEAutomorphismKey<D, W> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, W: ZnxWord> GLWEAutomorphismKey<D, W> {
    /// Returns the Galois element index `p`.
    pub fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEAutomorphismKey<D, W> {
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

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEAutomorphismKey<D, W> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GLWEAutomorphismKey<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.key.k_aux()
    }

    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.key.dsize()
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
        crate::layouts::key_size(self.base2k, self.dnum, self.dsize, self.k_aux)
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::key_k(self.base2k, self.dnum, self.dsize, self.k_aux)
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

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn rank_out(&self) -> Rank {
        self.rank
    }
}

impl<BE: Backend> GGLWEAtBackendRef<BE> for GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        <GGLWE<BE::OwnedBuf, BE::ZnxWord> as GGLWEAtBackendRef<BE>>::at_backend(&self.key, row, col)
    }
}

impl<BE: Backend> GGLWEAtBackendMut<BE> for GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        <GGLWE<BE::OwnedBuf, BE::ZnxWord> as GGLWEAtBackendMut<BE>>::at_backend_mut(&mut self.key, row, col)
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWEAutomorphismKey<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut, W: ZnxWord> FillUniform for GLWEAutomorphismKey<D, W> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GLWEAutomorphismKey<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(AutomorphismKey: p={}) {}", self.p, self.key)
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GLWEAutomorphismKey<Vec<u8>, W> {
    /// Allocates a new [`GLWEAutomorphismKey`] with the given parameters.
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.dsize(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    /// Allocates a new [`GLWEAutomorphismKey`] with the given parameters.
    pub(crate) fn alloc(n: Degree, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> Self {
        GLWEAutomorphismKey {
            key: GGLWE::alloc(n, base2k, dnum, dsize, k_aux, rank, rank),
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
            infos.dsize(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    /// Returns the byte count required for a [`GLWEAutomorphismKey`] with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> usize {
        GGLWE::<Vec<u8>, W>::bytes_of(n, base2k, dnum, dsize, k_aux, rank, rank)
    }
}

impl_gglwe_to_backend_for_field!(GLWEAutomorphismKey<D, BE::ZnxWord>, key, GGLWE<D, BE::ZnxWord>);

impl_gglwe_at_view_for_field!(GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord>; key);

impl<D: Data, W: ZnxWord> SetGaloisElement for &mut GLWEAutomorphismKey<D, W> {
    fn set_p(&mut self, p: i64) {
        self.p = p;
    }
}

impl_glwe_host_at_for_field!(GLWEAutomorphismKey<D, W>; key);

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEAutomorphismKey<D, W> {
    /// Deserialises from little-endian binary format.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.p = reader.read_u64::<LittleEndian>()? as i64;
        self.key.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEAutomorphismKey<D, W> {
    /// Serialises in little-endian binary format.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.p as u64)?;
        self.key.write_to(writer)
    }
}
