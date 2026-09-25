use std::fmt;

use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEBackendMut, GGLWEBackendRef, GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef,
    GLWEInfos, LWEInfos, Rank, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

pub type GGLWEPatOwned<BE> = GGLWEPat<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// Unseeded public aggregatable transcript of a GGLWE: a full [`GGLWE`], for
/// transcripts whose masks are not uniform, such as public-key encryptions.
///
/// Aggregation adds limbs without normalizing and clears the canonical flag;
/// serialization and finalization require it set.
#[derive(Clone)]
pub struct GGLWEPat<D: Data, W: ZnxWord> {
    pub(crate) inner: GGLWE<D, W>,
    pub(crate) canonical: bool,
}

impl<D: Data, W: ZnxWord> GGLWEPat<D, W> {
    pub fn is_canonical(&self) -> bool {
        self.canonical
    }

    /// For data written directly into the limbs.
    pub fn set_canonical(&mut self, canonical: bool) {
        self.canonical = canonical
    }
}

impl<D: Data, W: ZnxWord> PartialEq for GGLWEPat<D, W>
where
    GGLWE<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<D: Data, W: ZnxWord> Eq for GGLWEPat<D, W> where GGLWE<D, W>: Eq {}

impl<D: Data, W: ZnxWord> LWEInfos for GGLWEPat<D, W> {
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

impl<D: Data, W: ZnxWord> GLWEInfos for GGLWEPat<D, W> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GGLWEPat<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.inner.dnum()
    }

    fn dsize(&self) -> Dsize {
        self.inner.dsize()
    }

    fn rank_in(&self) -> Rank {
        self.inner.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.inner.rank_out()
    }
}

impl<BE: Backend, D: Data> GGLWEToBackendRef<BE> for GGLWEPat<D, BE::ZnxWord>
where
    GGLWE<D, BE::ZnxWord>: GGLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        self.inner.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GGLWEToBackendMut<BE> for GGLWEPat<D, BE::ZnxWord>
where
    GGLWE<D, BE::ZnxWord>: GGLWEToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        self.inner.to_backend_mut()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GGLWEPat<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GGLWEPat<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GGLWEPat: canonical={} {}", self.canonical, self.inner)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GGLWEPat<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.inner.read_from(reader)?;
        self.canonical = true;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GGLWEPat<D, W> {
    /// Fails with [`std::io::ErrorKind::InvalidInput`], writing nothing, when the
    /// canonical flag is clear: normalize first.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        if !self.canonical {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "GGLWEPat is not canonical: normalize it before serializing",
            ));
        }
        self.inner.write_to(writer)
    }
}
