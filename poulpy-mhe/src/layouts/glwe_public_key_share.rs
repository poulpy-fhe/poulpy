use std::fmt;

use poulpy_core::{
    Distribution, GetDistribution, GetDistributionMut,
    layouts::{
        Base2K, Degree, GLWEInfos, GLWEPublicKeyCompressed, GLWEPublicKeyCompressedSeed, GLWEPublicKeyCompressedSeedMut,
        GLWEPublicKeyCompressedToBackendMut, GLWEPublicKeyCompressedToBackendRef, LWEInfos, Rank, TorusPrecision,
        compressed::{GLWEPublicKeyCompressedBackendMut, GLWEPublicKeyCompressedBackendRef},
    },
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

pub type GLWEPublicKeyShareOwned<BE> = GLWEPublicKeyShare<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// One party's share of the collective public key: a core
/// [`GLWEPublicKeyCompressed`], one seeded body per key entry, tagged with the
/// distribution of the secret it was generated with.
///
/// Serializes as core's `GLWEPublicKeyCompressed`.
#[derive(Clone)]
pub struct GLWEPublicKeyShare<D: Data, W: ZnxWord> {
    pub(crate) key: GLWEPublicKeyCompressed<D, W>,
}

impl<D: Data, W: ZnxWord> PartialEq for GLWEPublicKeyShare<D, W>
where
    GLWEPublicKeyCompressed<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<D: Data, W: ZnxWord> Eq for GLWEPublicKeyShare<D, W> where GLWEPublicKeyCompressed<D, W>: Eq {}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEPublicKeyShare<D, W> {
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

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEPublicKeyShare<D, W> {
    fn rank(&self) -> Rank {
        self.key.rank()
    }
}

impl<D: Data, W: ZnxWord> GetDistribution for GLWEPublicKeyShare<D, W> {
    fn dist(&self) -> &Distribution {
        self.key.dist()
    }
}

impl<D: Data, W: ZnxWord> GetDistributionMut for GLWEPublicKeyShare<D, W> {
    fn dist_mut(&mut self) -> &mut Distribution {
        self.key.dist_mut()
    }
}

impl<D: Data, W: ZnxWord> GLWEPublicKeyCompressedSeed for GLWEPublicKeyShare<D, W> {
    fn seed(&self) -> &[[u8; 32]] {
        self.key.seed()
    }
}

impl<D: Data, W: ZnxWord> GLWEPublicKeyCompressedSeedMut for GLWEPublicKeyShare<D, W> {
    fn seed_mut(&mut self) -> &mut [[u8; 32]] {
        self.key.seed_mut()
    }
}

impl<BE: Backend, D: Data> GLWEPublicKeyCompressedToBackendRef<BE> for GLWEPublicKeyShare<D, BE::ZnxWord>
where
    GLWEPublicKeyCompressed<D, BE::ZnxWord>: GLWEPublicKeyCompressedToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWEPublicKeyCompressedBackendRef<'_, BE> {
        self.key.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GLWEPublicKeyCompressedToBackendMut<BE> for GLWEPublicKeyShare<D, BE::ZnxWord>
where
    GLWEPublicKeyCompressed<D, BE::ZnxWord>: GLWEPublicKeyCompressedToBackendMut<BE>,
{
    fn set_encryption_metadata(&mut self, metadata: Option<poulpy_core::EncryptionMetadata>) {
        GLWEPublicKeyCompressedToBackendMut::<BE>::set_encryption_metadata(&mut self.key, metadata);
    }

    fn to_backend_mut(&mut self) -> GLWEPublicKeyCompressedBackendMut<'_, BE> {
        self.key.to_backend_mut()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWEPublicKeyShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GLWEPublicKeyShare: {:?}", self.key)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEPublicKeyShare<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.key.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEPublicKeyShare<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.key.write_to(writer)
    }
}
