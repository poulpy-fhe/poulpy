use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEAtViewMut, GGLWEAtViewRef, GGLWEBackendMut, GGLWEBackendRef, GGLWECore, GGLWEInfos,
    GGLWELayout, GGLWEToBackendMut, GGLWEToBackendRef, GLWE, GLWEInfos, GLWEViewMut, GLWEViewRef, LWEInfos, Rank, TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::layouts::{MatZnx, MatZnxInfos, ZnxWord};

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
    pub dnum: Dnum,
    pub k_aux: TorusPrecision,
    pub rank: Rank,
    pub dsize: Dsize,
}
/// GLWE automorphism (Galois) key, generic over the HAL payload holding its
/// polynomials.
///
/// Wraps a [`GGLWECore`] together with the Galois element index `p` that
/// identifies which automorphism this key materialises.
///
/// `P` selects the computational domain: [`GLWEAutomorphismKey`] is the
/// coefficient-domain spelling (payload `MatZnx`) and
/// [`GLWEAutomorphismKeyPrepared`](crate::layouts::GLWEAutomorphismKeyPrepared)
/// the prepared one (payload `VmpPMat`); both are aliases of this struct. The
/// wrapper stays nominal in either domain.
#[derive(PartialEq, Clone)]
pub struct GLWEAutomorphismKeyCore<P> {
    pub(crate) key: GGLWECore<P>,
    pub(crate) p: i64,
}

/// Coefficient-domain GLWE automorphism key.
///
/// `D: Data` is the backing storage type (e.g. `Vec<u8>`, `&[u8]`,
/// `&mut [u8]`).
pub type GLWEAutomorphismKey<D, W> = GLWEAutomorphismKeyCore<MatZnx<D, W>>;

// `Eq` stays coefficient-domain only, mirroring `GGLWECore`.
impl<D: Data, W: ZnxWord> Eq for GLWEAutomorphismKeyCore<MatZnx<D, W>> {}

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

impl<P> SetGaloisElement for GLWEAutomorphismKeyCore<P> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<P> GetGaloisElement for GLWEAutomorphismKeyCore<P> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<P> GLWEAutomorphismKeyCore<P> {
    /// Returns the Galois element index `p`.
    pub fn p(&self) -> i64 {
        self.p
    }
}

/// Delegated wholesale to the wrapped gadget key; the wrapper adds only the
/// Galois element, which is not layout information.
impl<P: MatZnxInfos> LWEInfos for GLWEAutomorphismKeyCore<P> {
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

impl<P: MatZnxInfos> GLWEInfos for GLWEAutomorphismKeyCore<P> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<P: MatZnxInfos> GGLWEInfos for GLWEAutomorphismKeyCore<P> {
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
