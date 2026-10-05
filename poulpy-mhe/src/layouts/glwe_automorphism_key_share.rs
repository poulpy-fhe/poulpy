use std::fmt;

use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, GetGaloisElement, LWEInfos, Rank, SetGaloisElement, TorusPrecision,
    compressed::{
        GGLWECompressedBackendMut, GGLWECompressedBackendRef, GGLWECompressedSeed, GGLWECompressedSeedMut,
        GGLWECompressedToBackendMut, GGLWECompressedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

use crate::layouts::GGLWEPatCompressed;

pub type GLWEAutomorphismKeyShareOwned<BE> = GLWEAutomorphismKeyShare<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// One party's share of a collective GLWE automorphism key: a
/// [`GGLWEPatCompressed`] with the Galois element `p`.
///
/// Serializes as core's `GLWEAutomorphismKeyCompressed`.
#[derive(PartialEq, Eq, Clone)]
pub struct GLWEAutomorphismKeyShare<D: Data, W: ZnxWord> {
    pub(crate) key: GGLWEPatCompressed<D, W>,
    pub(crate) p: i64,
}

impl<D: Data, W: ZnxWord> GetGaloisElement for GLWEAutomorphismKeyShare<D, W> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, W: ZnxWord> SetGaloisElement for GLWEAutomorphismKeyShare<D, W> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEAutomorphismKeyShare<D, W> {
    fn encryption_metadata(&self) -> Option<poulpy_core::EncryptionMetadata> {
        self.key.encryption_metadata()
    }

    fn n(&self) -> Degree {
        self.key.n()
    }

    fn k(&self) -> TorusPrecision {
        self.key.k()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn max_size(&self) -> usize {
        self.key.max_size()
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEAutomorphismKeyShare<D, W> {
    fn rank(&self) -> Rank {
        self.key.rank()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GLWEAutomorphismKeyShare<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.key.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }

    fn dsize(&self) -> Dsize {
        self.key.dsize()
    }

    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }
}

impl<D: Data, W: ZnxWord> GGLWECompressedSeed for GLWEAutomorphismKeyShare<D, W> {
    fn seed(&self) -> &Vec<[u8; 32]> {
        self.key.seed()
    }
}

impl<D: Data, W: ZnxWord> GGLWECompressedSeedMut for GLWEAutomorphismKeyShare<D, W> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        self.key.seed_mut()
    }
}

impl<BE: Backend, D: Data> GGLWECompressedToBackendRef<BE> for GLWEAutomorphismKeyShare<D, BE::ZnxWord>
where
    GGLWEPatCompressed<D, BE::ZnxWord>: GGLWECompressedToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGLWECompressedBackendRef<'_, BE> {
        self.key.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GGLWECompressedToBackendMut<BE> for GLWEAutomorphismKeyShare<D, BE::ZnxWord>
where
    GGLWEPatCompressed<D, BE::ZnxWord>: GGLWECompressedToBackendMut<BE>,
{
    fn set_encryption_metadata(&mut self, metadata: Option<poulpy_core::EncryptionMetadata>) {
        GGLWECompressedToBackendMut::<BE>::set_encryption_metadata(&mut self.key, metadata);
    }

    fn to_backend_mut(&mut self) -> GGLWECompressedBackendMut<'_, BE> {
        self.key.to_backend_mut()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWEAutomorphismKeyShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GLWEAutomorphismKeyShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(GLWEAutomorphismKeyShare: p={}) {}", self.p, self.key)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEAutomorphismKeyShare<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let mut p = [0u8; 8];
        reader.read_exact(&mut p)?;
        self.p = i64::from_le_bytes(p);
        self.key.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEAutomorphismKeyShare<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_all(&self.p.to_le_bytes())?;
        self.key.write_to(writer)
    }
}
