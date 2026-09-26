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

pub type GLWESwitchingKeyPatCompressedOwned<BE> =
    GLWESwitchingKeyPatCompressed<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// Seeded public aggregatable transcript of a GLWE switching key: a
/// [`GGLWEPatCompressed`] with the degrees of the input and output secrets.
///
/// Serializes as core's `GLWESwitchingKeyCompressed`.
#[derive(PartialEq, Eq, Clone)]
pub struct GLWESwitchingKeyPatCompressed<D: Data, W: ZnxWord> {
    pub(crate) key: GGLWEPatCompressed<D, W>,
    pub(crate) input_degree: Degree,
    pub(crate) output_degree: Degree,
}

impl<D: Data, W: ZnxWord> GLWESwitchingKeyPatCompressed<D, W> {
    pub fn is_canonical(&self) -> bool {
        self.key.is_canonical()
    }

    /// For data written directly into the limbs.
    pub fn set_canonical(&mut self, canonical: bool) {
        self.key.set_canonical(canonical)
    }
}

impl<D: Data, W: ZnxWord> GLWESwitchingKeyDegrees for GLWESwitchingKeyPatCompressed<D, W> {
    fn output_degree(&self) -> &Degree {
        &self.output_degree
    }

    fn input_degree(&self) -> &Degree {
        &self.input_degree
    }
}

impl<D: Data, W: ZnxWord> GLWESwitchingKeyDegreesMut for GLWESwitchingKeyPatCompressed<D, W> {
    fn output_degree(&mut self) -> &mut Degree {
        &mut self.output_degree
    }

    fn input_degree(&mut self) -> &mut Degree {
        &mut self.input_degree
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWESwitchingKeyPatCompressed<D, W> {
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

impl<D: Data, W: ZnxWord> GLWEInfos for GLWESwitchingKeyPatCompressed<D, W> {
    fn rank(&self) -> Rank {
        self.key.rank()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GLWESwitchingKeyPatCompressed<D, W> {
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

impl<D: Data, W: ZnxWord> GGLWECompressedSeed for GLWESwitchingKeyPatCompressed<D, W> {
    fn seed(&self) -> &Vec<[u8; 32]> {
        self.key.seed()
    }
}

impl<D: Data, W: ZnxWord> GGLWECompressedSeedMut for GLWESwitchingKeyPatCompressed<D, W> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        self.key.seed_mut()
    }
}

impl<BE: Backend, D: Data> GGLWECompressedToBackendRef<BE> for GLWESwitchingKeyPatCompressed<D, BE::ZnxWord>
where
    GGLWEPatCompressed<D, BE::ZnxWord>: GGLWECompressedToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGLWECompressedBackendRef<'_, BE> {
        self.key.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GGLWECompressedToBackendMut<BE> for GLWESwitchingKeyPatCompressed<D, BE::ZnxWord>
where
    GGLWEPatCompressed<D, BE::ZnxWord>: GGLWECompressedToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GGLWECompressedBackendMut<'_, BE> {
        self.key.to_backend_mut()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWESwitchingKeyPatCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GLWESwitchingKeyPatCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GLWESwitchingKeyPatCompressed: sk_in_n={} sk_out_n={}) {}",
            self.input_degree, self.output_degree, self.key
        )
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWESwitchingKeyPatCompressed<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let mut degree = [0u8; 4];
        reader.read_exact(&mut degree)?;
        self.input_degree = Degree(u32::from_le_bytes(degree));
        reader.read_exact(&mut degree)?;
        self.output_degree = Degree(u32::from_le_bytes(degree));
        self.key.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWESwitchingKeyPatCompressed<D, W> {
    /// Fails with [`std::io::ErrorKind::InvalidInput`], writing nothing, when the
    /// canonical flag is clear: normalize first.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        if !self.is_canonical() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "GLWESwitchingKeyPatCompressed is not canonical: normalize it before serializing",
            ));
        }
        writer.write_all(&self.input_degree.0.to_le_bytes())?;
        writer.write_all(&self.output_degree.0.to_le_bytes())?;
        self.key.write_to(writer)
    }
}
