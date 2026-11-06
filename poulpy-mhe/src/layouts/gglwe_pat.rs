use std::fmt;

use poulpy_core::{
    TransferInto,
    layouts::{
        Base2K, Degree, Dnum, GGLWE, GGLWEBackendMut, GGLWEBackendRef, GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef,
        GLWEInfos, LWEInfos, Rank, TorusPrecision,
    },
};
use poulpy_hal::layouts::{Backend, CopyFromHost, CopyToHost, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

pub type MHEGGLWEPatOwned<BE> = MHEGGLWEPat<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// Unseeded public aggregatable transcript of a GGLWE: a full [`GGLWE`], for
/// transcripts whose masks are not uniform, such as public-key encryptions.
///
/// Aggregation adds limbs without normalizing; finalization produces canonical
/// digits and leaves the PAT unchanged.
#[derive(Clone)]
pub struct MHEGGLWEPat<D: Data, W: ZnxWord> {
    pub(crate) inner: GGLWE<D, W>,
}

impl<D: Data, W: ZnxWord> PartialEq for MHEGGLWEPat<D, W>
where
    GGLWE<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<D: Data, W: ZnxWord> Eq for MHEGGLWEPat<D, W> where GGLWE<D, W>: Eq {}

impl<D: Data, W: ZnxWord> LWEInfos for MHEGGLWEPat<D, W> {
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

impl<D: Data, W: ZnxWord> GLWEInfos for MHEGGLWEPat<D, W> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for MHEGGLWEPat<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.inner.dnum()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.inner.k_digit()
    }

    fn k_offset(&self) -> TorusPrecision {
        self.inner.k_offset()
    }

    fn rank_in(&self) -> Rank {
        self.inner.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.inner.rank_out()
    }
}

impl<BE: Backend, D: Data> GGLWEToBackendRef<BE> for MHEGGLWEPat<D, BE::ZnxWord>
where
    GGLWE<D, BE::ZnxWord>: GGLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        self.inner.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GGLWEToBackendMut<BE> for MHEGGLWEPat<D, BE::ZnxWord>
where
    GGLWE<D, BE::ZnxWord>: GGLWEToBackendMut<BE>,
{
    fn set_noise(&mut self, metadata: Option<poulpy_core::ComponentNoise>) {
        GGLWEToBackendMut::<BE>::set_noise(&mut self.inner, metadata);
    }

    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        self.inner.to_backend_mut()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for MHEGGLWEPat<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for MHEGGLWEPat<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MHEGGLWEPat: {}", self.inner)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for MHEGGLWEPat<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.inner.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for MHEGGLWEPat<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.inner.write_to(writer)
    }
}

impl<D1, D2, W> TransferInto<MHEGGLWEPat<D2, W>> for MHEGGLWEPat<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut MHEGGLWEPat<D2, W>) {
        self.inner.transfer_into(&mut dst.inner);
    }
}

#[deprecated(note = "renamed MHEGGLWEPat")]
pub type GGLWEPat<D, W> = MHEGGLWEPat<D, W>;

#[deprecated(note = "renamed MHEGGLWEPatOwned")]
pub type GGLWEPatOwned<BE> = MHEGGLWEPatOwned<BE>;
