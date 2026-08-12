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
use poulpy_hal::layouts::ZnxWord;
use std::fmt;

/// Seed-compressed LWE switching key layout.
///
/// A newtype wrapper around [`GLWESwitchingKeyCompressed`] for key-switching
/// between LWE ciphertexts encrypted under different keys.
#[derive(PartialEq, Eq, Clone)]
pub struct LWESwitchingKeyCompressed<D: Data, W: ZnxWord>(pub(crate) GLWESwitchingKeyCompressed<D, W>);

impl<D: Data, W: ZnxWord> LWEInfos for LWESwitchingKeyCompressed<D, W> {
    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn n(&self) -> Degree {
        self.0.n()
    }
    fn max_size(&self) -> usize {
        self.0.max_size()
    }
    fn k(&self) -> TorusPrecision {
        self.0.k()
    }
}
impl<D: Data, W: ZnxWord> GLWEInfos for LWESwitchingKeyCompressed<D, W> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for LWESwitchingKeyCompressed<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.0.k_aux()
    }

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

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for LWESwitchingKeyCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut, W: ZnxWord> FillUniform for LWESwitchingKeyCompressed<D, W> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for LWESwitchingKeyCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWESwitchingKeyCompressed) {}", self.0)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for LWESwitchingKeyCompressed<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for LWESwitchingKeyCompressed<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl<D: Data, W: ZnxWord> LWESwitchingKeyCompressed<D, W> {
    pub(crate) fn alloc_from_infos<B: Backend<OwnedBuf = D, ZnxWord = W>, A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWESwitchingKeyCompressed");
        assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKeyCompressed"
        );
        assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKeyCompressed"
        );
        Self::alloc::<B>(infos.n(), infos.base2k(), infos.dnum(), infos.k_aux())
    }

    pub(crate) fn alloc<B: Backend<OwnedBuf = D, ZnxWord = W>>(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        k_aux: TorusPrecision,
    ) -> Self {
        LWESwitchingKeyCompressed(GLWESwitchingKeyCompressed::alloc::<B>(
            n,
            base2k,
            dnum,
            Dsize(1),
            k_aux,
            Rank(1),
            Rank(1),
        ))
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWESwitchingKeyCompressed");
        assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKeyCompressed"
        );
        assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKeyCompressed"
        );
        GLWESwitchingKeyCompressed::<Vec<u8>, W>::bytes_of_from_infos(infos)
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, k_aux: TorusPrecision) -> usize {
        GLWESwitchingKeyCompressed::<Vec<u8>, W>::bytes_of(n, base2k, dnum, Dsize(1), k_aux, Rank(1))
    }
}

pub trait LWESwitchingKeyDecompress
where
    Self: GLWESwitchingKeyDecompress,
{
    fn decompress_lwe_switching_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self::Backend> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self::Backend> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        self.decompress_glwe_switching_key(res, other);
    }
}

impl<B: Backend> LWESwitchingKeyDecompress for Module<B> where Self: GLWESwitchingKeyDecompress {}

// module-only API: decompression is provided by `LWESwitchingKeyDecompress` on `Module`.

impl_gglwe_compressed_to_backend_for_field!(
    LWESwitchingKeyCompressed<BE::OwnedBuf, BE::ZnxWord>,
    0,
    GLWESwitchingKeyCompressed<BE::OwnedBuf, BE::ZnxWord>
);
