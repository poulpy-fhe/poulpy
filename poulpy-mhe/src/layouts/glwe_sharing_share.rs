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

use crate::layouts::{MHEGLWEPatCompressed, glwe_share::glwe_share};

glwe_share!(
    /// One party's public share of an encryption-to-shares conversion: a rank-0
    /// GLWE, its partial decryption minus its mask.
    MHEGLWEEncToShareShare,
    MHEGLWEEncToShareShareOwned
);

pub type MHEGLWEShareToEncShareOwned<BE> = MHEGLWEShareToEncShare<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// One party's share of a shares-to-encryption conversion: a [`MHEGLWEPatCompressed`],
/// the seeded encryption of its additive share.
///
/// Serializes as its [`MHEGLWEPatCompressed`].
#[derive(Clone)]
pub struct MHEGLWEShareToEncShare<D: Data, W: ZnxWord> {
    pub(crate) inner: MHEGLWEPatCompressed<D, W>,
}

impl<D: Data, W: ZnxWord> PartialEq for MHEGLWEShareToEncShare<D, W>
where
    MHEGLWEPatCompressed<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<D: Data, W: ZnxWord> Eq for MHEGLWEShareToEncShare<D, W> where MHEGLWEPatCompressed<D, W>: Eq {}

impl<D: Data, W: ZnxWord> LWEInfos for MHEGLWEShareToEncShare<D, W> {
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

impl<D: Data, W: ZnxWord> GLWEInfos for MHEGLWEShareToEncShare<D, W> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data, W: ZnxWord> GLWECompressedSeed for MHEGLWEShareToEncShare<D, W> {
    fn seed(&self) -> &[u8; 32] {
        self.inner.seed()
    }
}

impl<D: Data, W: ZnxWord> GLWECompressedSeedMut for MHEGLWEShareToEncShare<D, W> {
    fn seed_mut(&mut self) -> &mut [u8; 32] {
        self.inner.seed_mut()
    }
}

impl<BE: Backend, D: Data> GLWECompressedToBackendRef<BE> for MHEGLWEShareToEncShare<D, BE::ZnxWord>
where
    MHEGLWEPatCompressed<D, BE::ZnxWord>: GLWECompressedToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWECompressedBackendRef<'_, BE> {
        self.inner.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GLWECompressedToBackendMut<BE> for MHEGLWEShareToEncShare<D, BE::ZnxWord>
where
    MHEGLWEPatCompressed<D, BE::ZnxWord>: GLWECompressedToBackendMut<BE>,
{
    fn set_noise(&mut self, metadata: Option<poulpy_core::ComponentNoise>) {
        GLWECompressedToBackendMut::<BE>::set_noise(&mut self.inner, metadata);
    }

    fn to_backend_mut(&mut self) -> GLWECompressedBackendMut<'_, BE> {
        self.inner.to_backend_mut()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for MHEGLWEShareToEncShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MHEGLWEShareToEncShare: {:?}", self.inner)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for MHEGLWEShareToEncShare<D, W> {
    fn read_from<R: Read>(&mut self, reader: &mut R) -> io::Result<()> {
        self.inner.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for MHEGLWEShareToEncShare<D, W> {
    fn write_to<Wr: Write>(&self, writer: &mut Wr) -> io::Result<()> {
        self.inner.write_to(writer)
    }
}

#[deprecated(note = "renamed MHEGLWEEncToShareShare")]
pub type GLWEEncToShareShare<D, W> = MHEGLWEEncToShareShare<D, W>;

#[deprecated(note = "renamed MHEGLWEEncToShareShareOwned")]
pub type GLWEEncToShareShareOwned<BE> = MHEGLWEEncToShareShareOwned<BE>;

#[deprecated(note = "renamed MHEGLWEShareToEncShare")]
pub type GLWEShareToEncShare<D, W> = MHEGLWEShareToEncShare<D, W>;

#[deprecated(note = "renamed MHEGLWEShareToEncShareOwned")]
pub type GLWEShareToEncShareOwned<BE> = MHEGLWEShareToEncShareOwned<BE>;
