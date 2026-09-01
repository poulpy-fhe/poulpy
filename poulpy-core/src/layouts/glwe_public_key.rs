use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, Normalized, ReaderFrom, VecZnx, WriterTo, ZnxWord};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, TorusPrecision},
};

#[derive(PartialEq, Eq)]
pub struct GLWEPublicKey<D: Data, W: ZnxWord> {
    pub(crate) key: GLWE<D, W>,
    pub(crate) dist: Distribution,
}

impl<D: Data, W: ZnxWord> GetDistributionMut for GLWEPublicKey<D, W> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, W: ZnxWord> GetDistribution for GLWEPublicKey<D, W> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWEPublicKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEPublicKey<D, W> {
    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn n(&self) -> Degree {
        self.key.n()
    }

    fn max_size(&self) -> usize {
        self.key.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.key.k()
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEPublicKey<D, W> {
    fn rank(&self) -> Rank {
        self.key.rank()
    }
}

impl LWEInfos for GLWEPublicKeyLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn max_size(&self) -> usize {
        self.k.div_ceil(self.base2k) as usize
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl GLWEInfos for GLWEPublicKeyLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GLWEPublicKey<Vec<u8>, W> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        GLWEPublicKey {
            key: GLWE::alloc(n, base2k, k, rank),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        VecZnx::<Vec<u8>, W>::bytes_of(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEPublicKey<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.dist = Distribution::read_from(reader)?;
        self.key.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEPublicKey<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        match self.dist.write_to(writer) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        self.key.write_to(writer)
    }
}

impl<BE: Backend, D: Data> GLWEToBackendRef<BE> for GLWEPublicKey<D, BE::ZnxWord>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE, State = Normalized>,
{
    type State = Normalized;
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        self.key.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for GLWEPublicKey<D, BE::ZnxWord>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE, State = Normalized>,
{
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        self.key.to_backend_mut()
    }
}
