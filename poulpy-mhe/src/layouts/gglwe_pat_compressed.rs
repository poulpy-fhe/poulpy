use std::fmt;

use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    compressed::{
        GGLWECompressed, GGLWECompressedBackendMut, GGLWECompressedBackendRef, GGLWECompressedSeed, GGLWECompressedSeedMut,
        GGLWECompressedToBackendMut, GGLWECompressedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

pub type GGLWEPatCompressedOwned<BE> = GGLWEPatCompressed<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// Seeded public aggregatable transcript of a GGLWE: the bodies of a
/// [`GGLWECompressed`] whose masks every party regenerates from the common seed.
///
/// Aggregation adds limbs without normalizing and clears the canonical flag;
/// serialization and finalization require it set.
#[derive(Clone)]
pub struct GGLWEPatCompressed<D: Data, W: ZnxWord> {
    pub(crate) inner: GGLWECompressed<D, W>,
    pub(crate) canonical: bool,
}

impl<D: Data, W: ZnxWord> GGLWEPatCompressed<D, W> {
    pub fn is_canonical(&self) -> bool {
        self.canonical
    }

    /// For data written directly into the limbs.
    pub fn set_canonical(&mut self, canonical: bool) {
        self.canonical = canonical
    }
}

impl<D: Data, W: ZnxWord> PartialEq for GGLWEPatCompressed<D, W>
where
    GGLWECompressed<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<D: Data, W: ZnxWord> Eq for GGLWEPatCompressed<D, W> where GGLWECompressed<D, W>: Eq {}

impl<D: Data, W: ZnxWord> LWEInfos for GGLWEPatCompressed<D, W> {
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

impl<D: Data, W: ZnxWord> GLWEInfos for GGLWEPatCompressed<D, W> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GGLWEPatCompressed<D, W> {
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

impl<D: Data, W: ZnxWord> GGLWECompressedSeed for GGLWEPatCompressed<D, W> {
    fn seed(&self) -> &Vec<[u8; 32]> {
        self.inner.seed()
    }
}

impl<D: Data, W: ZnxWord> GGLWECompressedSeedMut for GGLWEPatCompressed<D, W> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        self.inner.seed_mut()
    }
}

impl<BE: Backend, D: Data> GGLWECompressedToBackendRef<BE> for GGLWEPatCompressed<D, BE::ZnxWord>
where
    GGLWECompressed<D, BE::ZnxWord>: GGLWECompressedToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGLWECompressedBackendRef<'_, BE> {
        self.inner.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GGLWECompressedToBackendMut<BE> for GGLWEPatCompressed<D, BE::ZnxWord>
where
    GGLWECompressed<D, BE::ZnxWord>: GGLWECompressedToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GGLWECompressedBackendMut<'_, BE> {
        self.inner.to_backend_mut()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GGLWEPatCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GGLWEPatCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GGLWEPatCompressed: canonical={} {}", self.canonical, self.inner)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GGLWEPatCompressed<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.inner.read_from(reader)?;
        self.canonical = true;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GGLWEPatCompressed<D, W> {
    /// Fails with [`std::io::ErrorKind::InvalidInput`], writing nothing, when the
    /// canonical flag is clear: normalize first.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        if !self.canonical {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "GGLWEPatCompressed is not canonical: normalize it before serializing",
            ));
        }
        self.inner.write_to(writer)
    }
}
