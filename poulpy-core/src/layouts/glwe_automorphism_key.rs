use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, ReaderFrom, WriterTo},
    source::Source,
};

use crate::{
    DeclaredK,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEAtViewMut, GGLWEAtViewRef, GGLWEBackendMut, GGLWEBackendRef, GGLWEInfos,
        GGLWELayout, GGLWEToBackendMut, GGLWEToBackendRef, GLWE, GLWEInfos, GLWEViewMut, GLWEViewRef, LWEInfos, Rank,
        TorusPrecision,
    },
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

/// Provides lookup of automorphism keys by Galois element and access
/// to the shared layout information.
pub trait GLWEAutomorphismKeyHelper<K, BE: Backend> {
    /// Returns the automorphism key associated with the Galois element `k`, if present.
    fn get_automorphism_key(&self, k: i64) -> Option<&K>;
    /// Returns the [`GGLWELayout`] common to all stored automorphism keys.
    fn automorphism_key_infos(&self) -> GGLWELayout;
}

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
    pub k: TorusPrecision,
    pub rank: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

impl DeclaredK for GLWEAutomorphismKeyLayout {
    fn k(&self) -> TorusPrecision {
        self.k
    }
}

/// GLWE automorphism (Galois) key.
///
/// Wraps a [`GGLWE`] together with the Galois element index `p` that
/// identifies which automorphism this key materialises.
///
/// `D: Data` is the backing storage type (e.g. `Vec<u8>`, `&[u8]`,
/// `&mut [u8]`).
#[derive(PartialEq, Eq, Clone)]
pub struct GLWEAutomorphismKey<D: Data> {
    pub(crate) key: GGLWE<D>,
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

impl<D: HostDataMut> SetGaloisElement for GLWEAutomorphismKey<D> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<D: HostDataRef> GetGaloisElement for GLWEAutomorphismKey<D> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data> GLWEAutomorphismKey<D> {
    /// Returns the Galois element index `p`.
    pub fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data> LWEInfos for GLWEAutomorphismKey<D> {
    fn n(&self) -> Degree {
        self.key.n()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn size(&self) -> usize {
        self.key.size()
    }
}

impl<D: Data> GLWEInfos for GLWEAutomorphismKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWEAutomorphismKey<D> {
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

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
    }
}

impl GLWEInfos for GLWEAutomorphismKeyLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl GGLWEInfos for GLWEAutomorphismKeyLayout {
    fn rank_in(&self) -> Rank {
        self.rank
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn rank_out(&self) -> Rank {
        self.rank
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

impl<D: HostDataRef> fmt::Debug for GLWEAutomorphismKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut> FillUniform for GLWEAutomorphismKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef> fmt::Display for GLWEAutomorphismKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(AutomorphismKey: p={}) {}", self.p, self.key)
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl GLWEAutomorphismKey<Vec<u8>> {
    /// Allocates a new [`GLWEAutomorphismKey`] with the given parameters.
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Allocates a new [`GLWEAutomorphismKey`] with the given parameters.
    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        GLWEAutomorphismKey {
            key: GGLWE::alloc(n, base2k, k, rank, rank, dnum, dsize),
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
            infos.max_k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Returns the byte count required for a [`GLWEAutomorphismKey`] with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        GGLWE::bytes_of(n, base2k, k, rank, rank, dnum, dsize)
    }
}

impl_gglwe_to_backend_for_field!(GLWEAutomorphismKey<D>, key, GGLWE<D>);

impl_gglwe_at_view_for_field!(GLWEAutomorphismKey<BE::OwnedBuf>; key);

impl<D: Data> SetGaloisElement for &mut GLWEAutomorphismKey<D> {
    fn set_p(&mut self, p: i64) {
        self.p = p;
    }
}

impl_glwe_host_at_for_field!(GLWEAutomorphismKey<D>; key);

impl<D: HostDataMut> ReaderFrom for GLWEAutomorphismKey<D> {
    /// Deserialises from little-endian binary format.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.p = reader.read_u64::<LittleEndian>()? as i64;
        self.key.read_from(reader)
    }
}

impl<D: HostDataRef> WriterTo for GLWEAutomorphismKey<D> {
    /// Serialises in little-endian binary format.
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.p as u64)?;
        self.key.write_to(writer)
    }
}
