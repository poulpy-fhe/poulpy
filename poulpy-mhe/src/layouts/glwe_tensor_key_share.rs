use std::fmt;

use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEBackendMut, GGLWEBackendRef, GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GLWEInfos,
    LWEInfos, Rank, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

use crate::layouts::GGLWEPat;

pub type GLWETensorKeyShareOwned<BE> = GLWETensorKeyShare<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// One party's share of the collective tensor key: a [`GGLWEPat`] laid out as
/// core's `GLWETensorKey`, one entry per pair of secret components.
///
/// Serializes as its [`GGLWEPat`].
#[derive(Clone)]
pub struct GLWETensorKeyShare<D: Data, W: ZnxWord> {
    pub(crate) key: GGLWEPat<D, W>,
}

impl<D: Data, W: ZnxWord> PartialEq for GLWETensorKeyShare<D, W>
where
    GGLWEPat<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<D: Data, W: ZnxWord> Eq for GLWETensorKeyShare<D, W> where GGLWEPat<D, W>: Eq {}

impl<D: Data, W: ZnxWord> LWEInfos for GLWETensorKeyShare<D, W> {
    fn noise(&self) -> Option<poulpy_core::ComponentNoise> {
        self.key.noise()
    }

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

impl<D: Data, W: ZnxWord> GLWEInfos for GLWETensorKeyShare<D, W> {
    fn rank(&self) -> Rank {
        self.key.rank()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GLWETensorKeyShare<D, W> {
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

impl<BE: Backend, D: Data> GGLWEToBackendRef<BE> for GLWETensorKeyShare<D, BE::ZnxWord>
where
    GGLWEPat<D, BE::ZnxWord>: GGLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        self.key.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GGLWEToBackendMut<BE> for GLWETensorKeyShare<D, BE::ZnxWord>
where
    GGLWEPat<D, BE::ZnxWord>: GGLWEToBackendMut<BE>,
{
    fn set_noise(&mut self, metadata: Option<poulpy_core::ComponentNoise>) {
        GGLWEToBackendMut::<BE>::set_noise(&mut self.key, metadata);
    }

    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        self.key.to_backend_mut()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWETensorKeyShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GLWETensorKeyShare: {:?}", self.key)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWETensorKeyShare<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.key.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWETensorKeyShare<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.key.write_to(writer)
    }
}
