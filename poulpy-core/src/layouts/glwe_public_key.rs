use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, VecZnx, VecZnxInfos, WriterTo, ZnxWord};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWE, GLWECore, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, TorusPrecision},
};

/// A GLWE public key, generic over the HAL payload holding its polynomials.
///
/// `P` selects the computational domain: [`GLWEPublicKey`] is the
/// coefficient-domain spelling (payload `VecZnx`) and
/// [`GLWEPublicKeyPrepared`](crate::layouts::GLWEPublicKeyPrepared) the
/// prepared one (payload `VecZnxDft`); both are aliases of this struct.
///
/// It is a [`GLWECore`] plus the distribution the key was sampled from, which
/// public-key encryption needs and a ciphertext does not carry.
#[derive(PartialEq)]
pub struct GLWEPublicKeyCore<P> {
    pub(crate) key: GLWECore<P>,
    pub(crate) dist: Distribution,
}

/// Coefficient-domain GLWE public key.
pub type GLWEPublicKey<D, W> = GLWEPublicKeyCore<VecZnx<D, W>>;

// `Eq` stays coefficient-domain only, mirroring `GLWECore`.
impl<D: Data, W: ZnxWord> Eq for GLWEPublicKeyCore<VecZnx<D, W>> {}

impl<P> GetDistributionMut for GLWEPublicKeyCore<P> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<P> GetDistribution for GLWEPublicKeyCore<P> {
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

/// Delegated wholesale to the wrapped key: a public key advertises exactly the
/// shape of the ciphertext it is.
impl<P: VecZnxInfos> LWEInfos for GLWEPublicKeyCore<P> {
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

impl<P: VecZnxInfos> GLWEInfos for GLWEPublicKeyCore<P> {
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
    GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        self.key.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for GLWEPublicKey<D, BE::ZnxWord>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        self.key.to_backend_mut()
    }
}
