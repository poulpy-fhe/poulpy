use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, Ring, Standard, WriterTo};

use crate::layouts::{
    Base2K, Degree, Dnum, GGLWECompressed, GGLWECompressedSeedMut, GGLWEInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision,
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
pub struct GLWETensorKeyCompressed<D: Data, W: ZnxWord, Rg: Ring = Standard>(pub(crate) GGLWECompressed<D, W, Rg>);

impl<D: Data, W: ZnxWord, Rg: Ring> crate::layouts::GGLWECompressedSeed for GLWETensorKeyCompressed<D, W, Rg> {
    fn seed(&self) -> &Vec<[u8; 32]> {
        &self.0.seed
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GGLWECompressedSeedMut for GLWETensorKeyCompressed<D, W, Rg> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.0.seed
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWEInfos for GLWETensorKeyCompressed<D, W, Rg> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.0.noise()
    }

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
impl<D: Data, W: ZnxWord, Rg: Ring> GLWEInfos for GLWETensorKeyCompressed<D, W, Rg> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GGLWEInfos for GLWETensorKeyCompressed<D, W, Rg> {
    fn k_aux(&self) -> TorusPrecision {
        self.0.k_aux()
    }

    fn rank_in(&self) -> Rank {
        self.rank_out()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.0.k_digit()
    }

    fn k_offset(&self) -> TorusPrecision {
        self.0.k_offset()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Debug for GLWETensorKeyCompressed<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Display for GLWETensorKeyCompressed<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GLWETensorKeyCompressed)",)?;
        write!(f, "{}", self.0)?;
        Ok(())
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWETensorKeyCompressed<D, W, Rg> {
    pub(crate) fn alloc_from_infos<B: Backend<OwnedBuf = D, ZnxWord = W>, A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc::<B>(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.k_digit(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    pub(crate) fn alloc<B: Backend<OwnedBuf = D, ZnxWord = W>>(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> Self {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        GLWETensorKeyCompressed(GGLWECompressed::alloc::<B>(
            n,
            base2k,
            dnum,
            k_digit,
            k_aux,
            Rank(pairs),
            rank,
        ))
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.k_digit(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, k_digit: TorusPrecision, k_aux: TorusPrecision, rank: Rank) -> usize {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        GGLWECompressed::<AlignedBuf, W, Rg>::bytes_of(n, base2k, dnum, k_digit, k_aux, Rank(pairs))
    }
}

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> ReaderFrom for GLWETensorKeyCompressed<D, W, Rg> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)?;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> WriterTo for GLWETensorKeyCompressed<D, W, Rg> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.0.write_to(writer)?;
        Ok(())
    }
}

impl_gglwe_compressed_to_backend_for_field!(GLWETensorKeyCompressed<BE::OwnedBuf, BE::ZnxWord, Rg>, 0, GGLWECompressed<BE::OwnedBuf, BE::ZnxWord, Rg>);
