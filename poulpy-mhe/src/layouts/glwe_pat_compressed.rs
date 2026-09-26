use std::fmt;

use poulpy_core::layouts::{
    Base2K, Degree, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    compressed::{
        GLWECompressed, GLWECompressedBackendMut, GLWECompressedBackendRef, GLWECompressedSeed, GLWECompressedSeedMut,
        GLWECompressedToBackendMut, GLWECompressedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, VecZnx, WriterTo, ZnxWord};

pub type GLWEPatCompressedOwned<BE> = GLWEPatCompressed<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// Seeded public aggregatable transcript of a GLWE: the body of a
/// [`GLWECompressed`] whose mask every party regenerates from the common seed.
///
/// Aggregation adds limbs without normalizing and clears the canonical flag;
/// serialization and finalization require it set.
#[derive(Clone)]
pub struct GLWEPatCompressed<D: Data, W: ZnxWord> {
    pub(crate) inner: GLWECompressed<D, W>,
    pub(crate) canonical: bool,
}

impl<D: Data, W: ZnxWord> GLWEPatCompressed<D, W> {
    pub fn data(&self) -> &VecZnx<D, W> {
        self.inner.data()
    }

    pub fn data_mut(&mut self) -> &mut VecZnx<D, W> {
        self.inner.data_mut()
    }

    pub fn is_canonical(&self) -> bool {
        self.canonical
    }

    /// For data written directly into the limbs.
    pub fn set_canonical(&mut self, canonical: bool) {
        self.canonical = canonical
    }
}

impl<D: Data, W: ZnxWord> PartialEq for GLWEPatCompressed<D, W>
where
    GLWECompressed<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<D: Data, W: ZnxWord> Eq for GLWEPatCompressed<D, W> where GLWECompressed<D, W>: Eq {}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEPatCompressed<D, W> {
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

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEPatCompressed<D, W> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data, W: ZnxWord> GLWECompressedSeed for GLWEPatCompressed<D, W> {
    fn seed(&self) -> &[u8; 32] {
        self.inner.seed()
    }
}

impl<D: Data, W: ZnxWord> GLWECompressedSeedMut for GLWEPatCompressed<D, W> {
    fn seed_mut(&mut self) -> &mut [u8; 32] {
        self.inner.seed_mut()
    }
}

impl<BE: Backend, D: Data> GLWECompressedToBackendRef<BE> for GLWEPatCompressed<D, BE::ZnxWord>
where
    GLWECompressed<D, BE::ZnxWord>: GLWECompressedToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWECompressedBackendRef<'_, BE> {
        self.inner.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GLWECompressedToBackendMut<BE> for GLWEPatCompressed<D, BE::ZnxWord>
where
    GLWECompressed<D, BE::ZnxWord>: GLWECompressedToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWECompressedBackendMut<'_, BE> {
        self.inner.to_backend_mut()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWEPatCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GLWEPatCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GLWEPatCompressed: canonical={} {}", self.canonical, self.inner)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEPatCompressed<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.inner.read_from(reader)?;
        self.canonical = true;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEPatCompressed<D, W> {
    /// Fails with [`std::io::ErrorKind::InvalidInput`], writing nothing, when the
    /// canonical flag is clear: normalize first.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        if !self.canonical {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "GLWEPatCompressed is not canonical: normalize it before serializing",
            ));
        }
        self.inner.write_to(writer)
    }
}
