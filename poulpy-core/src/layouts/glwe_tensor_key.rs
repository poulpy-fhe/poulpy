use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEAtViewMut, GGLWEAtViewRef, GGLWEBackendMut, GGLWEBackendRef, GGLWEInfos,
    GGLWEToBackendMut, GGLWEToBackendRef, GLWEInfos, GLWEViewMut, GLWEViewRef, LWEInfos, Rank, TorusPrecision,
};

use poulpy_hal::layouts::ZnxWord;
use std::fmt;

/// Plain-data descriptor for a [`GLWETensorKey`] carrying only the
/// layout parameters (no backing buffer).
///
/// Implements [`LWEInfos`], [`GLWEInfos`] and [`GGLWEInfos`] so it can
/// be passed to any generic constructor that needs layout information.
/// The `rank_in` is derived from `rank` as `max(1, rank*(rank+1)/2)`.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWETensorKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub dnum: Dnum,
    pub k_aux: TorusPrecision,
    pub rank: Rank,
    pub dsize: Dsize,
}

/// GLWE tensor key used for relinearisation after a tensor product.
///
/// Wraps a [`GGLWE`] whose `rank_in` equals the number of unique
/// pairs `max(1, rank*(rank+1)/2)` produced by the tensor product.
///
/// `D: Data` is the backing storage type (e.g. `Vec<u8>`, `&[u8]`,
/// `&mut [u8]`).
#[derive(PartialEq, Eq, Clone)]
pub struct GLWETensorKey<D: Data, W: ZnxWord>(pub(crate) GGLWE<D, W>);

impl<D: Data, W: ZnxWord> LWEInfos for GLWETensorKey<D, W> {
    fn n(&self) -> Degree {
        self.0.n()
    }

    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn max_size(&self) -> usize {
        self.0.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.0.k()
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GLWETensorKey<D, W> {
    fn rank(&self) -> Rank {
        self.0.rank_out()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GLWETensorKey<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.0.k_aux()
    }

    fn rank_in(&self) -> Rank {
        let rank_out: usize = self.rank_out().as_usize();
        let pairs: usize = (((rank_out + 1) * rank_out) >> 1).max(1);
        pairs.into()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

impl LWEInfos for GLWETensorKeyLayout {
    fn n(&self) -> Degree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn max_size(&self) -> usize {
        crate::layouts::key_size(self.base2k, self.dnum, self.dsize, self.k_aux)
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::key_k(self.base2k, self.dnum, self.dsize, self.k_aux)
    }
}

impl GLWEInfos for GLWETensorKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for GLWETensorKeyLayout {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }

    fn rank_in(&self) -> Rank {
        let rank_out: usize = self.rank_out().as_usize();
        let pairs: usize = (((rank_out + 1) * rank_out) >> 1).max(1);
        pairs.into()
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn rank_out(&self) -> Rank {
        self.rank
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWETensorKey<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut, W: ZnxWord> FillUniform for GLWETensorKey<D, W> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source)
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GLWETensorKey<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GLWETensorKey)",)?;
        write!(f, "{}", self.0)?;
        Ok(())
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GLWETensorKey<Vec<u8>, W> {
    /// Allocates a new [`GLWETensorKey`] with the given parameters.
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

    /// Allocates a new [`GLWETensorKey`] with the given parameters.
    pub(crate) fn alloc(n: Degree, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> Self {
        let pairs: u32 = (((rank.0 + 1) * rank.0) >> 1).max(1);
        GLWETensorKey(GGLWE::alloc(n, base2k, dnum, dsize, k_aux, Rank(pairs), rank))
    }

    /// Returns the byte count required for a [`GLWETensorKey`] with the given parameters.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.dsize(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    /// Returns the byte count required for a [`GLWETensorKey`] with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> usize {
        let pairs: u32 = (((rank.0 + 1) * rank.0) >> 1).max(1);
        GGLWE::<Vec<u8>, W>::bytes_of(n, base2k, dnum, dsize, k_aux, Rank(pairs), rank)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWETensorKey<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)?;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWETensorKey<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.0.write_to(writer)?;
        Ok(())
    }
}

impl_gglwe_to_backend_for_field!(GLWETensorKey<D, BE::ZnxWord>, 0, GGLWE<D, BE::ZnxWord>);

impl_gglwe_at_view_for_field!(GLWETensorKey<BE::OwnedBuf, BE::ZnxWord>; 0);
