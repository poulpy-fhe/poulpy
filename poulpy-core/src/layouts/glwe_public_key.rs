use std::fmt;

use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, TorusPrecision},
};

/// GLWE public key of rank `r`: `r` encryptions of zero under the secret `s`,
/// `pk_l = (b_l, a_{l,1}, .., a_{l,r})` with `b_l = -Sum_j a_{l,j} s_j + e_l`,
/// one per ephemeral of a public-key encryption.
#[derive(PartialEq, Eq)]
pub struct GLWEPublicKey<D: Data, W: ZnxWord> {
    pub(crate) keys: Vec<GLWE<D, W>>,
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
        self.keys[0].base2k()
    }

    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn max_size(&self) -> usize {
        self.keys[0].max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.keys[0].k()
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEPublicKey<D, W> {
    fn rank(&self) -> Rank {
        self.keys[0].rank()
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
impl<W: ZnxWord> GLWEPublicKey<AlignedBuf, W> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        GLWEPublicKey {
            keys: (0..rank.as_usize()).map(|_| GLWE::alloc(n, base2k, k, rank)).collect(),
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
        rank.as_usize() * GLWE::<AlignedBuf, W>::bytes_of(n, base2k, k, rank)
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWEPublicKey<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GLWEPublicKey")
            .field("keys", &self.keys)
            .field("dist", &self.dist)
            .finish()
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEPublicKey<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.dist = Distribution::read_from(reader)?;
        for key in &mut self.keys {
            key.read_from(reader)?;
        }
        if self.keys.iter().any(|key| key.rank().as_usize() != self.keys.len()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid public key: entry count differs from its rank",
            ));
        }
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEPublicKey<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.dist.write_to(writer)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

pub type GLWEPublicKeyBackendRef<'a, BE> = GLWEPublicKey<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord>;
pub type GLWEPublicKeyBackendMut<'a, BE> = GLWEPublicKey<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord>;

pub trait GLWEPublicKeyToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GLWEPublicKeyBackendRef<'_, BE>;
}

impl<BE: Backend, D: Data> GLWEPublicKeyToBackendRef<BE> for GLWEPublicKey<D, BE::ZnxWord>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWEPublicKeyBackendRef<'_, BE> {
        GLWEPublicKey {
            keys: self.keys.iter().map(|key| key.to_backend_ref()).collect(),
            dist: self.dist,
        }
    }
}

pub trait GLWEPublicKeyToBackendMut<BE: Backend> {
    fn to_backend_mut(&mut self) -> GLWEPublicKeyBackendMut<'_, BE>;

    /// [`GLWEToBackendMut::set_canonical`] on every entry.
    fn set_canonical(&mut self, canonical: bool);
}

impl<BE: Backend, D: Data> GLWEPublicKeyToBackendMut<BE> for GLWEPublicKey<D, BE::ZnxWord>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWEPublicKeyBackendMut<'_, BE> {
        GLWEPublicKey {
            keys: self.keys.iter_mut().map(|key| key.to_backend_mut()).collect(),
            dist: self.dist,
        }
    }

    fn set_canonical(&mut self, canonical: bool) {
        for key in &mut self.keys {
            key.canonical = canonical
        }
    }
}
