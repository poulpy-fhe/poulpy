use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWECompressed, GGLWECompressedSeedMut, GGLWEDecompress, GGLWEInfos, GGLWEToBackendMut,
    GLWEInfos, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, LWEInfos, Rank, TorusPrecision,
    compressed::{
        GGLWECompressedBackendMut, GGLWECompressedBackendRef, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef,
    },
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::layouts::ZnxWord;
use std::fmt;

/// Seed-compressed GLWE switching key layout.
///
/// Wraps a [`GGLWECompressed`] with additional input/output degree metadata
/// for key-switching between GLWE ciphertexts with different ring degrees.
#[derive(PartialEq, Eq, Clone)]
pub struct GLWESwitchingKeyCompressed<D: Data, W: ZnxWord> {
    pub(crate) key: GGLWECompressed<D, W>,
    pub(crate) input_degree: Degree,  // Degree of sk_in
    pub(crate) output_degree: Degree, // Degree of sk_out
}

impl<D: HostDataMut, W: ZnxWord> GGLWECompressedSeedMut for GLWESwitchingKeyCompressed<D, W> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.key.seed
    }
}

impl<D: HostDataRef, W: ZnxWord> GLWESwitchingKeyDegrees for GLWESwitchingKeyCompressed<D, W> {
    fn output_degree(&self) -> &Degree {
        &self.output_degree
    }

    fn input_degree(&self) -> &Degree {
        &self.input_degree
    }
}

impl<D: HostDataMut, W: ZnxWord> GLWESwitchingKeyDegreesMut for GLWESwitchingKeyCompressed<D, W> {
    fn output_degree(&mut self) -> &mut Degree {
        &mut self.output_degree
    }

    fn input_degree(&mut self) -> &mut Degree {
        &mut self.input_degree
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWESwitchingKeyCompressed<D, W> {
    fn n(&self) -> Degree {
        self.key.n()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn max_size(&self) -> usize {
        self.key.max_size()
    }
    fn k(&self) -> TorusPrecision {
        self.key.k()
    }
}
impl<D: Data, W: ZnxWord> GLWEInfos for GLWESwitchingKeyCompressed<D, W> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GLWESwitchingKeyCompressed<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.key.k_aux()
    }

    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.key.dsize()
    }

    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWESwitchingKeyCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut, W: ZnxWord> FillUniform for GLWESwitchingKeyCompressed<D, W> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GLWESwitchingKeyCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GLWESwitchingKeyCompressed: sk_in_n={} sk_out_n={}) {}",
            self.input_degree, self.output_degree, self.key.data
        )
    }
}

impl<D: Data, W: ZnxWord> GLWESwitchingKeyCompressed<D, W> {
    /// Allocates a new compressed GLWE switching key by copying parameters from an existing info provider.
    pub(crate) fn alloc_from_infos<B: Backend<OwnedBuf = D, ZnxWord = W>, A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc::<B>(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.dsize(),
            infos.k_aux(),
            infos.rank_in(),
            infos.rank_out(),
        )
    }

    /// Allocates a new compressed GLWE switching key with the given parameters.
    pub(crate) fn alloc<B: Backend<OwnedBuf = D, ZnxWord = W>>(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> Self {
        GLWESwitchingKeyCompressed {
            key: GGLWECompressed::alloc::<B>(n, base2k, dnum, dsize, k_aux, rank_in, rank_out),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }

    /// Returns the serialized byte size by copying parameters from an existing info provider.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        GGLWECompressed::<Vec<u8>, W>::bytes_of_from_infos(infos)
    }

    /// Returns the serialized byte size for a compressed GLWE switching key with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank_in: Rank) -> usize {
        GGLWECompressed::<Vec<u8>, W>::bytes_of(n, base2k, dnum, dsize, k_aux, rank_in)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWESwitchingKeyCompressed<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.input_degree = Degree(reader.read_u32::<LittleEndian>()?);
        self.output_degree = Degree(reader.read_u32::<LittleEndian>()?);
        self.key.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWESwitchingKeyCompressed<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.input_degree.into())?;
        writer.write_u32::<LittleEndian>(self.output_degree.into())?;
        self.key.write_to(writer)
    }
}

/// Trait for decompressing a [`GLWESwitchingKeyCompressed`] into a standard
/// [`GLWESwitchingKey`](crate::layouts::GLWESwitchingKey).
pub trait GLWESwitchingKeyDecompress
where
    Self: GGLWEDecompress,
{
    /// Decompresses `other` into `res`, copying degree metadata.
    fn decompress_glwe_switching_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self::Backend> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self::Backend> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        self.decompress_gglwe(res, other);

        *res.input_degree() = *other.input_degree();
        *res.output_degree() = *other.output_degree();
    }
}

impl<B: Backend> GLWESwitchingKeyDecompress for Module<B> where Self: GGLWEDecompress {}

// module-only API: decompression is provided by `GLWESwitchingKeyDecompress` on `Module`.

impl_gglwe_compressed_to_backend_for_field!(GLWESwitchingKeyCompressed<BE::OwnedBuf, BE::ZnxWord>, key, GGLWECompressed<BE::OwnedBuf, BE::ZnxWord>);
