use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo};

use crate::layouts::{
    Base2K, Degree, Dnum, GGLWECompressed, GGLWECompressedSeedMut, GGLWEInfos, GLWEInfos, GetGaloisElement, LWEInfos, Rank,
    SetGaloisElement, TorusPrecision,
    compressed::{
        GGLWECompressedBackendMut, GGLWECompressedBackendRef, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef,
    },
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::layouts::ZnxWord;
use std::fmt;

/// Seed-compressed GLWE automorphism key layout.
///
/// Wraps a [`GGLWECompressed`] with a Galois element `p` for applying
/// automorphisms `X → X^p` on GLWE ciphertexts.
#[derive(PartialEq, Eq, Clone)]
pub struct GLWEAutomorphismKeyCompressed<D: Data, W: ZnxWord> {
    pub(crate) key: GGLWECompressed<D, W>,
    pub(crate) p: i64,
}

impl<D: Data, W: ZnxWord> GetGaloisElement for GLWEAutomorphismKeyCompressed<D, W> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEAutomorphismKeyCompressed<D, W> {
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

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEAutomorphismKeyCompressed<D, W> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GLWEAutomorphismKeyCompressed<D, W> {
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

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWEAutomorphismKeyCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GLWEAutomorphismKeyCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(AutomorphismKeyCompressed: p={}) {}", self.p, self.key)
    }
}

impl<D: Data, W: ZnxWord> GLWEAutomorphismKeyCompressed<D, W> {
    pub(crate) fn alloc_from_infos<B: Backend<OwnedBuf = D, ZnxWord = W>, A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        GLWEAutomorphismKeyCompressed {
            key: GGLWECompressed::alloc_from_infos::<B, _>(infos),
            p: 0,
        }
    }

    pub(crate) fn alloc<B: Backend<OwnedBuf = D, ZnxWord = W>>(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> Self {
        GLWEAutomorphismKeyCompressed {
            key: GGLWECompressed::alloc::<B>(n, base2k, dnum, k_digit, k_aux, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        GGLWECompressed::<AlignedBuf, W>::bytes_of_from_infos(infos)
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, k_digit: TorusPrecision, k_aux: TorusPrecision, rank: Rank) -> usize {
        GGLWECompressed::<AlignedBuf, W>::bytes_of(n, base2k, dnum, k_digit, k_aux, rank)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEAutomorphismKeyCompressed<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.key.noise = None;
        let p = reader.read_u64::<LittleEndian>()? as i64;
        self.key.read_from(reader)?;
        self.p = p;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEAutomorphismKeyCompressed<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        if let Some(noise) = &self.key.noise {
            noise.validate_wire()?;
            noise.validate_components(self.key.rank_out.as_usize() + 1)?;
        }
        writer.write_u64::<LittleEndian>(self.p as u64)?;
        self.key.write_to(writer)
    }
}

impl_gglwe_compressed_to_backend_for_field!(
    GLWEAutomorphismKeyCompressed<BE::OwnedBuf, BE::ZnxWord>,
    key,
    GGLWECompressed<BE::OwnedBuf, BE::ZnxWord>
);

impl<D: Data, W: ZnxWord> GGLWECompressedSeedMut for GLWEAutomorphismKeyCompressed<D, W> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.key.seed
    }
}

impl<D: Data, W: ZnxWord> crate::layouts::GGLWECompressedSeed for GLWEAutomorphismKeyCompressed<D, W> {
    fn seed(&self) -> &Vec<[u8; 32]> {
        &self.key.seed
    }
}

impl<D: Data, W: ZnxWord> SetGaloisElement for GLWEAutomorphismKeyCompressed<D, W> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}
