use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::ZnxWord;
use std::fmt;

use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, Ring, Standard, WriterTo};

use crate::layouts::{
    Base2K, Degree, Dnum, GGLWEInfos, GLWEInfos, GLWESwitchingKeyDegrees, LWEInfos, Rank, TorusPrecision,
    compressed::{
        GGLWECompressedBackendMut, GGLWECompressedBackendRef, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef,
        GLWESwitchingKeyCompressed,
    },
};

/// Seed-compressed GLWE-to-LWE switching key layout.
///
/// A newtype wrapper around [`GLWESwitchingKeyCompressed`] for converting
/// GLWE ciphertexts to LWE ciphertexts via key-switching.
#[derive(PartialEq, Eq, Clone)]
pub struct GLWEToLWEKeyCompressed<D: Data, W: ZnxWord, Rg: Ring = Standard>(pub(crate) GLWESwitchingKeyCompressed<D, W, Rg>);

#[deprecated(note = "renamed GLWEToLWEKeyCompressed")]
pub type GLWEToLWESwitchingKeyCompressed<D, W, Rg = Standard> = GLWEToLWEKeyCompressed<D, W, Rg>;

// The wrapper carries the same degree metadata as its switching-key storage.
impl<D: Data, W: ZnxWord, Rg: Ring> GLWESwitchingKeyDegrees for GLWEToLWEKeyCompressed<D, W, Rg> {
    fn input_degree(&self) -> &Degree {
        GLWESwitchingKeyDegrees::input_degree(&self.0)
    }

    fn output_degree(&self) -> &Degree {
        GLWESwitchingKeyDegrees::output_degree(&self.0)
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWEInfos for GLWEToLWEKeyCompressed<D, W, Rg> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.0.noise()
    }

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

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEInfos for GLWEToLWEKeyCompressed<D, W, Rg> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GGLWEInfos for GLWEToLWEKeyCompressed<D, W, Rg> {
    fn k_aux(&self) -> TorusPrecision {
        self.0.k_aux()
    }

    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.0.k_digit()
    }

    fn k_offset(&self) -> TorusPrecision {
        self.0.k_offset()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Debug for GLWEToLWEKeyCompressed<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Display for GLWEToLWEKeyCompressed<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(GLWEToLWEKeyCompressed) {}", self.0)
    }
}

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> ReaderFrom for GLWEToLWEKeyCompressed<D, W, Rg> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> WriterTo for GLWEToLWEKeyCompressed<D, W, Rg> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEToLWEKeyCompressed<D, W, Rg> {
    pub(crate) fn alloc_from_infos<B: Backend<OwnedBuf = D, ZnxWord = W>, A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is unsupported for GLWEToLWEKeyCompressed"
        );
        assert_eq!(
            infos.k_digit().0,
            infos.base2k().0,
            "GLWEToLWEKeyCompressed uses one-limb digits: k_digit must equal base2k"
        );
        Self::alloc::<B>(infos.n(), infos.base2k(), infos.dnum(), infos.k_aux(), infos.rank_in())
    }

    pub(crate) fn alloc<B: Backend<OwnedBuf = D, ZnxWord = W>>(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        k_aux: TorusPrecision,
        rank_in: Rank,
    ) -> Self {
        GLWEToLWEKeyCompressed(GLWESwitchingKeyCompressed::alloc::<B>(
            n,
            base2k,
            dnum,
            TorusPrecision::from(base2k),
            k_aux,
            rank_in,
            Rank(1),
        ))
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is unsupported for GLWEToLWEKeyCompressed"
        );
        assert_eq!(
            infos.k_digit().0,
            infos.base2k().0,
            "GLWEToLWEKeyCompressed uses one-limb digits: k_digit must equal base2k"
        );
        GLWESwitchingKeyCompressed::<AlignedBuf, W, Rg>::bytes_of_from_infos(infos)
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, k_aux: TorusPrecision, rank_in: Rank) -> usize {
        GLWESwitchingKeyCompressed::<AlignedBuf, W, Rg>::bytes_of(n, base2k, dnum, TorusPrecision::from(base2k), k_aux, rank_in)
    }
}

impl_gglwe_compressed_to_backend_for_field!(
    GLWEToLWEKeyCompressed<BE::OwnedBuf, BE::ZnxWord, Rg>,
    0,
    GLWESwitchingKeyCompressed<BE::OwnedBuf, BE::ZnxWord, Rg>
);
