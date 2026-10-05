use std::fmt;

use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, LWEInfos, Rank,
    TorusPrecision,
    compressed::{
        GGLWECompressedBackendMut, GGLWECompressedBackendRef, GGLWECompressedSeed, GGLWECompressedSeedMut,
        GGLWECompressedToBackendMut, GGLWECompressedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

use crate::layouts::GGLWEPatCompressed;

pub type GLWESwitchingKeyShareOwned<BE> = GLWESwitchingKeyShare<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// One party's share of a collective GLWE switching key: a
/// [`GGLWEPatCompressed`] with the degrees of the input and output secrets.
///
/// Serializes as core's `GLWESwitchingKeyCompressed`.
#[derive(PartialEq, Eq, Clone)]
pub struct GLWESwitchingKeyShare<D: Data, W: ZnxWord> {
    pub(crate) key: GGLWEPatCompressed<D, W>,
    pub(crate) input_degree: Degree,
    pub(crate) output_degree: Degree,
}

impl<D: Data, W: ZnxWord> GLWESwitchingKeyDegrees for GLWESwitchingKeyShare<D, W> {
    fn output_degree(&self) -> &Degree {
        &self.output_degree
    }

    fn input_degree(&self) -> &Degree {
        &self.input_degree
    }
}

impl<D: Data, W: ZnxWord> GLWESwitchingKeyDegreesMut for GLWESwitchingKeyShare<D, W> {
    fn output_degree(&mut self) -> &mut Degree {
        &mut self.output_degree
    }

    fn input_degree(&mut self) -> &mut Degree {
        &mut self.input_degree
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWESwitchingKeyShare<D, W> {
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

impl<D: Data, W: ZnxWord> GLWEInfos for GLWESwitchingKeyShare<D, W> {
    fn rank(&self) -> Rank {
        self.key.rank()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GLWESwitchingKeyShare<D, W> {
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

impl<D: Data, W: ZnxWord> GGLWECompressedSeed for GLWESwitchingKeyShare<D, W> {
    fn seed(&self) -> &Vec<[u8; 32]> {
        self.key.seed()
    }
}

impl<D: Data, W: ZnxWord> GGLWECompressedSeedMut for GLWESwitchingKeyShare<D, W> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        self.key.seed_mut()
    }
}

impl<BE: Backend, D: Data> GGLWECompressedToBackendRef<BE> for GLWESwitchingKeyShare<D, BE::ZnxWord>
where
    GGLWEPatCompressed<D, BE::ZnxWord>: GGLWECompressedToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGLWECompressedBackendRef<'_, BE> {
        self.key.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GGLWECompressedToBackendMut<BE> for GLWESwitchingKeyShare<D, BE::ZnxWord>
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

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWESwitchingKeyShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GLWESwitchingKeyShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GLWESwitchingKeyShare: sk_in_n={} sk_out_n={}) {}",
            self.input_degree, self.output_degree, self.key
        )
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWESwitchingKeyShare<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let mut degree = [0u8; 4];
        reader.read_exact(&mut degree)?;
        self.input_degree = Degree(u32::from_le_bytes(degree));
        reader.read_exact(&mut degree)?;
        self.output_degree = Degree(u32::from_le_bytes(degree));
        self.key.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWESwitchingKeyShare<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_all(&self.input_degree.0.to_le_bytes())?;
        writer.write_all(&self.output_degree.0.to_le_bytes())?;
        self.key.write_to(writer)
    }
}
