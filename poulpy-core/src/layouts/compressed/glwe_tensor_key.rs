use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWECompressed, GGLWECompressedSeedMut, GGLWEDecompress, GGLWEInfos, GGLWEToBackendMut,
    GLWEInfos, LWEInfos, Rank, TorusPrecision,
    compressed::{
        GGLWECompressedBackendMut, GGLWECompressedBackendRef, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef,
    },
};
use poulpy_hal::layouts::ZnxWord;
use std::fmt;

/// Seed-compressed GLWE tensor key layout.
///
/// A newtype wrapper around [`GGLWECompressed`] representing
/// the seed-compressed form of a GLWE tensor key.
#[derive(PartialEq, Eq, Clone)]
pub struct GLWETensorKeyCompressed<D: Data, W: ZnxWord>(pub(crate) GGLWECompressed<D, W>);

impl<D: HostDataMut, W: ZnxWord> GGLWECompressedSeedMut for GLWETensorKeyCompressed<D, W> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.0.seed
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWETensorKeyCompressed<D, W> {
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
impl<D: Data, W: ZnxWord> GLWEInfos for GLWETensorKeyCompressed<D, W> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GLWETensorKeyCompressed<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.0.k_aux()
    }

    fn rank_in(&self) -> Rank {
        self.rank_out()
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

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWETensorKeyCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut, W: ZnxWord> FillUniform for GLWETensorKeyCompressed<D, W> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GLWETensorKeyCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GLWETensorKeyCompressed)",)?;
        write!(f, "{}", self.0)?;
        Ok(())
    }
}

impl<D: Data, W: ZnxWord> GLWETensorKeyCompressed<D, W> {
    pub(crate) fn alloc_from_infos<B: Backend<OwnedBuf = D, ZnxWord = W>, A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc::<B>(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.dsize(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    pub(crate) fn alloc<B: Backend<OwnedBuf = D, ZnxWord = W>>(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> Self {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        GLWETensorKeyCompressed(GGLWECompressed::alloc::<B>(n, base2k, dnum, dsize, k_aux, Rank(pairs), rank))
    }

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

    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> usize {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        GGLWECompressed::<Vec<u8>, W>::bytes_of(n, base2k, dnum, dsize, k_aux, Rank(pairs))
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWETensorKeyCompressed<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)?;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWETensorKeyCompressed<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.0.write_to(writer)?;
        Ok(())
    }
}

pub trait GLWETensorKeyDecompress
where
    Self: GGLWEDecompress,
{
    fn decompress_tensor_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self::Backend> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<Self::Backend> + GGLWEInfos,
    {
        self.decompress_gglwe(res, other);
    }
}

impl<B: Backend> GLWETensorKeyDecompress for Module<B> where Self: GGLWEDecompress {}

// module-only API: decompression is provided by `GLWETensorKeyDecompress` on `Module`.

impl_gglwe_compressed_to_backend_for_field!(GLWETensorKeyCompressed<BE::OwnedBuf, BE::ZnxWord>, 0, GGLWECompressed<BE::OwnedBuf, BE::ZnxWord>);
