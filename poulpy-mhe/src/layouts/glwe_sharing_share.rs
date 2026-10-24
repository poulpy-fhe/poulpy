use std::{
    fmt,
    io::{self, Read, Write},
};

use poulpy_core::layouts::{
    Base2K, Degree, GLWE, GLWEBackendMut, GLWEBackendRef, GLWECompressedSeed, GLWECompressedSeedMut, GLWECompressedToBackendMut,
    GLWECompressedToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, TorusPrecision,
    compressed::{GLWECompressedBackendMut, GLWECompressedBackendRef},
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

use crate::layouts::{GLWEPatCompressed, glwe_share::glwe_share};

glwe_share!(
    /// One party's public share of an encryption-to-shares conversion: a rank-0
    /// GLWE, its partial decryption minus its mask.
    GLWEEncToShareShare,
    GLWEEncToShareShareOwned
);

pub type GLWEShareToEncShareOwned<BE> = GLWEShareToEncShare<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// One party's share of a shares-to-encryption conversion: a [`GLWEPatCompressed`],
/// the seeded encryption of its additive share.
///
/// Serializes as its [`GLWEPatCompressed`].
#[derive(Clone)]
pub struct GLWEShareToEncShare<D: Data, W: ZnxWord> {
    pub(crate) inner: GLWEPatCompressed<D, W>,
}

impl<D: Data, W: ZnxWord> PartialEq for GLWEShareToEncShare<D, W>
where
    GLWEPatCompressed<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<D: Data, W: ZnxWord> Eq for GLWEShareToEncShare<D, W> where GLWEPatCompressed<D, W>: Eq {}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEShareToEncShare<D, W> {
    fn noise(&self) -> Option<poulpy_core::ComponentNoise> {
        self.inner.noise()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }

    fn k(&self) -> TorusPrecision {
        self.inner.k()
    }

    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn max_size(&self) -> usize {
        self.inner.max_size()
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEShareToEncShare<D, W> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data, W: ZnxWord> GLWECompressedSeed for GLWEShareToEncShare<D, W> {
    fn seed(&self) -> &[u8; 32] {
        self.inner.seed()
    }
}

impl<D: Data, W: ZnxWord> GLWECompressedSeedMut for GLWEShareToEncShare<D, W> {
    fn seed_mut(&mut self) -> &mut [u8; 32] {
        self.inner.seed_mut()
    }
}

impl<BE: Backend, D: Data> GLWECompressedToBackendRef<BE> for GLWEShareToEncShare<D, BE::ZnxWord>
where
    GLWEPatCompressed<D, BE::ZnxWord>: GLWECompressedToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWECompressedBackendRef<'_, BE> {
        self.inner.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GLWECompressedToBackendMut<BE> for GLWEShareToEncShare<D, BE::ZnxWord>
where
    GLWEPatCompressed<D, BE::ZnxWord>: GLWECompressedToBackendMut<BE>,
{
    fn set_noise(&mut self, metadata: Option<poulpy_core::ComponentNoise>) {
        GLWECompressedToBackendMut::<BE>::set_noise(&mut self.inner, metadata);
    }

    fn to_backend_mut(&mut self) -> GLWECompressedBackendMut<'_, BE> {
        self.inner.to_backend_mut()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWEShareToEncShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GLWEShareToEncShare: {:?}", self.inner)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEShareToEncShare<D, W> {
    fn read_from<R: Read>(&mut self, reader: &mut R) -> io::Result<()> {
        self.inner.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEShareToEncShare<D, W> {
    fn write_to<Wr: Write>(&self, writer: &mut Wr) -> io::Result<()> {
        self.inner.write_to(writer)
    }
}
