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

pub type GLWEAutomorphismKeyPatCompressedOwned<BE> =
    GLWEAutomorphismKeyPatCompressed<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// Seeded public aggregatable transcript of a GLWE automorphism key: a
/// [`GGLWEPatCompressed`] with the Galois element `p`.
///
/// Serializes as core's `GLWEAutomorphismKeyCompressed`.
#[derive(PartialEq, Eq, Clone)]
pub struct GLWEAutomorphismKeyPatCompressed<D: Data, W: ZnxWord> {
    pub(crate) key: GGLWEPatCompressed<D, W>,
    pub(crate) p: i64,
}

impl<D: Data, W: ZnxWord> GLWEAutomorphismKeyPatCompressed<D, W> {
    pub fn is_canonical(&self) -> bool {
        self.key.is_canonical()
    }

    /// For data written directly into the limbs.
    pub fn set_canonical(&mut self, canonical: bool) {
        self.key.set_canonical(canonical)
    }
}

impl<D: Data, W: ZnxWord> GetGaloisElement for GLWEAutomorphismKeyPatCompressed<D, W> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, W: ZnxWord> SetGaloisElement for GLWEAutomorphismKeyPatCompressed<D, W> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEAutomorphismKeyPatCompressed<D, W> {
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

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEAutomorphismKeyPatCompressed<D, W> {
    fn rank(&self) -> Rank {
        self.key.rank()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GLWEAutomorphismKeyPatCompressed<D, W> {
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

impl<D: Data, W: ZnxWord> GGLWECompressedSeed for GLWEAutomorphismKeyPatCompressed<D, W> {
    fn seed(&self) -> &Vec<[u8; 32]> {
        self.key.seed()
    }
}

impl<D: Data, W: ZnxWord> GGLWECompressedSeedMut for GLWEAutomorphismKeyPatCompressed<D, W> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        self.key.seed_mut()
    }
}

impl<BE: Backend, D: Data> GGLWECompressedToBackendRef<BE> for GLWEAutomorphismKeyPatCompressed<D, BE::ZnxWord>
where
    GGLWEPatCompressed<D, BE::ZnxWord>: GGLWECompressedToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGLWECompressedBackendRef<'_, BE> {
        self.key.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GGLWECompressedToBackendMut<BE> for GLWEAutomorphismKeyPatCompressed<D, BE::ZnxWord>
where
    GGLWEPatCompressed<D, BE::ZnxWord>: GGLWECompressedToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GGLWECompressedBackendMut<'_, BE> {
        self.key.to_backend_mut()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWEAutomorphismKeyPatCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GLWEAutomorphismKeyPatCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(GLWEAutomorphismKeyPatCompressed: p={}) {}", self.p, self.key)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEAutomorphismKeyPatCompressed<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let mut p = [0u8; 8];
        reader.read_exact(&mut p)?;
        self.p = i64::from_le_bytes(p);
        self.key.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEAutomorphismKeyPatCompressed<D, W> {
    /// Fails with [`std::io::ErrorKind::InvalidInput`], writing nothing, when the
    /// canonical flag is clear: normalize first.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        if !self.is_canonical() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "GLWEAutomorphismKeyPatCompressed is not canonical: normalize it before serializing",
            ));
        }
        writer.write_all(&self.p.to_le_bytes())?;
        self.key.write_to(writer)
    }
}
