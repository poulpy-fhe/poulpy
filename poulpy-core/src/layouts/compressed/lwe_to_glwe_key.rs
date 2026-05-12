use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut,
    LWEInfos, Rank, TorusPrecision,
    compressed::{
        GGLWECompressedBackendMut, GGLWECompressedBackendRef, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef,
        GLWESwitchingKeyCompressed, GLWESwitchingKeyDecompress,
    },
};
use std::fmt;

/// Seed-compressed LWE-to-GLWE conversion key layout.
///
/// A newtype wrapper around [`GLWESwitchingKeyCompressed`] for converting
/// LWE ciphertexts to GLWE ciphertexts.
#[derive(PartialEq, Eq, Clone)]
pub struct LWEToGLWEKeyCompressed<D: Data>(pub(crate) GLWESwitchingKeyCompressed<D>);

impl<D: Data> LWEInfos for LWEToGLWEKeyCompressed<D> {
    fn n(&self) -> Degree {
        self.0.n()
    }

    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}
impl<D: Data> GLWEInfos for LWEToGLWEKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for LWEToGLWEKeyCompressed<D> {
    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

impl<D: HostDataRef> fmt::Debug for LWEToGLWEKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut> FillUniform for LWEToGLWEKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef> fmt::Display for LWEToGLWEKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWEToGLWESwitchingKeyCompressed) {}", self.0)
    }
}

impl<D: HostDataMut> ReaderFrom for LWEToGLWEKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: HostDataRef> WriterTo for LWEToGLWEKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl LWEToGLWEKeyCompressed<Vec<u8>> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWEToGLWESwitchingKeyCompressed"
        );
        assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWEToGLWESwitchingKeyCompressed"
        );
        Self::alloc(infos.n(), infos.base2k(), infos.max_k(), infos.rank_out(), infos.dnum())
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> Self {
        LWEToGLWEKeyCompressed(GLWESwitchingKeyCompressed::alloc(
            n,
            base2k,
            k,
            Rank(1),
            rank_out,
            dnum,
            Dsize(1),
        ))
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWEToGLWESwitchingKeyCompressed"
        );
        assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWEToGLWESwitchingKeyCompressed"
        );
        GLWESwitchingKeyCompressed::bytes_of_from_infos(infos)
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize {
        GLWESwitchingKeyCompressed::bytes_of(n, base2k, k, Rank(1), dnum, Dsize(1))
    }
}

pub trait LWEToGLWEKeyDecompress
where
    Self: GLWESwitchingKeyDecompress,
{
    fn decompress_lwe_to_glwe_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self::Backend> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self::Backend> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        self.decompress_glwe_switching_key(res, other);
    }
}

impl<B: Backend> LWEToGLWEKeyDecompress for Module<B> where Self: GLWESwitchingKeyDecompress {}

// module-only API: decompression is provided by `LWEToGLWEKeyDecompress` on `Module`.

impl_gglwe_compressed_to_backend_for_field!(
    LWEToGLWEKeyCompressed<BE::OwnedBuf>,
    0,
    GLWESwitchingKeyCompressed<BE::OwnedBuf>
);
