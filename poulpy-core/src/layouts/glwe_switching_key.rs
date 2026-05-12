use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, ReaderFrom, WriterTo},
    source::Source,
};

use crate::{
    DeclaredK,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEAtViewMut, GGLWEAtViewRef, GGLWEBackendMut, GGLWEBackendRef, GGLWEInfos,
        GGLWEToBackendMut, GGLWEToBackendRef, GLWE, GLWEInfos, GLWEViewMut, GLWEViewRef, LWEInfos, Rank, TorusPrecision,
    },
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

/// Plain-data descriptor for a [`GLWESwitchingKey`] carrying only the
/// layout parameters (no backing buffer).
///
/// Implements [`LWEInfos`], [`GLWEInfos`] and [`GGLWEInfos`] so it can
/// be passed to any generic constructor that needs layout information.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWESwitchingKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank_in: Rank,
    pub rank_out: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

impl DeclaredK for GLWESwitchingKeyLayout {
    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl LWEInfos for GLWESwitchingKeyLayout {
    fn n(&self) -> Degree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
    }
}

impl GLWEInfos for GLWESwitchingKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for GLWESwitchingKeyLayout {
    fn rank_in(&self) -> Rank {
        self.rank_in
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

/// GLWE key-switching key.
///
/// Wraps a [`GGLWE`] and additionally stores the polynomial degrees of
/// the input and output secret keys (`input_degree` / `output_degree`).
///
/// `D: Data` is the backing storage type (e.g. `Vec<u8>`, `&[u8]`,
/// `&mut [u8]`).
#[derive(PartialEq, Eq, Clone)]
pub struct GLWESwitchingKey<D: Data> {
    pub(crate) key: GGLWE<D>,
    pub(crate) input_degree: Degree,  // Degree of sk_in
    pub(crate) output_degree: Degree, // Degree of sk_out
}

/// Provides read access to the input and output secret-key degrees
/// stored in a [`GLWESwitchingKey`].
pub trait GLWESwitchingKeyDegrees {
    /// Returns the polynomial degree of the input secret key.
    fn input_degree(&self) -> &Degree;
    /// Returns the polynomial degree of the output secret key.
    fn output_degree(&self) -> &Degree;
}

impl<D: HostDataRef> GLWESwitchingKeyDegrees for GLWESwitchingKey<D> {
    fn output_degree(&self) -> &Degree {
        &self.output_degree
    }

    fn input_degree(&self) -> &Degree {
        &self.input_degree
    }
}

/// Provides mutable access to the input and output secret-key degrees
/// stored in a [`GLWESwitchingKey`].
pub trait GLWESwitchingKeyDegreesMut {
    /// Returns a mutable reference to the input secret-key degree.
    fn input_degree(&mut self) -> &mut Degree;
    /// Returns a mutable reference to the output secret-key degree.
    fn output_degree(&mut self) -> &mut Degree;
}

impl<D: HostDataMut> GLWESwitchingKeyDegreesMut for GLWESwitchingKey<D> {
    fn output_degree(&mut self) -> &mut Degree {
        &mut self.output_degree
    }

    fn input_degree(&mut self) -> &mut Degree {
        &mut self.input_degree
    }
}

impl<D: Data> LWEInfos for GLWESwitchingKey<D> {
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

impl<D: Data> GLWEInfos for GLWESwitchingKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWESwitchingKey<D> {
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

impl<D: HostDataRef> fmt::Debug for GLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef> fmt::Display for GLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GLWESwitchingKey: sk_in_n={} sk_out_n={}) {}",
            self.input_degree,
            self.output_degree,
            self.key.data()
        )
    }
}

impl<D: HostDataMut> FillUniform for GLWESwitchingKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl GLWESwitchingKey<Vec<u8>> {
    /// Allocates a new [`GLWESwitchingKey`] with the given parameters.
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Allocates a new [`GLWESwitchingKey`] with the given parameters.
    pub(crate) fn alloc(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self {
        GLWESwitchingKey {
            key: GGLWE::alloc(n, base2k, k, rank_in, rank_out, dnum, dsize),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }

    /// Returns the byte count required for a [`GLWESwitchingKey`] with the given parameters.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Returns the byte count required for a [`GLWESwitchingKey`] with the given parameters.
    pub fn bytes_of(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        GGLWE::bytes_of(n, base2k, k, rank_in, rank_out, dnum, dsize)
    }
}

impl_gglwe_to_backend_for_field!(GLWESwitchingKey<D>, key, GGLWE<D>);

impl_gglwe_at_view_for_field!(GLWESwitchingKey<BE::OwnedBuf>; key);

impl_glwe_host_at_for_field!(GLWESwitchingKey<D>; key);

impl<D: HostDataMut> ReaderFrom for GLWESwitchingKey<D> {
    /// Deserialises from little-endian binary format.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.input_degree = Degree(reader.read_u32::<LittleEndian>()?);
        self.output_degree = Degree(reader.read_u32::<LittleEndian>()?);
        self.key.read_from(reader)
    }
}

impl<D: HostDataRef> WriterTo for GLWESwitchingKey<D> {
    /// Serialises in little-endian binary format.
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.input_degree.into())?;
        writer.write_u32::<LittleEndian>(self.output_degree.into())?;
        self.key.write_to(writer)
    }
}
