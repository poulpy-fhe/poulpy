use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare},
    layouts::{Backend, Data, Module, SvpPPol, ZnxInfos},
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, GLWESecretToBackendRef, GetDegree, LWEInfos, Rank},
};

/// DFT-domain (prepared) variant of [`GLWESecret`].
///
/// Stores the GLWE secret key with polynomials in the frequency domain
/// for fast multiplication during encryption and decryption. Tied to a
/// specific backend via `B: Backend`.
pub struct GLWESecretPrepared<D: Data, B: Backend> {
    pub(crate) data: SvpPPol<D, B::DftWord>,
    pub(crate) dist: Distribution,
}

pub type GLWESecretPreparedBackendRef<'a, B> = GLWESecretPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWESecretPreparedBackendMut<'a, B> = GLWESecretPrepared<<B as Backend>::BufMut<'a>, B>;

impl<D: Data, BE: Backend> GetDistribution for GLWESecretPrepared<D, BE> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, BE: Backend> GetDistributionMut for GLWESecretPrepared<D, BE> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, B: Backend> LWEInfos for GLWESecretPrepared<D, B> {
    fn base2k(&self) -> Base2K {
        Base2K(0)
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn max_size(&self) -> usize {
        self.data.size()
    }
    fn k(&self) -> crate::layouts::TorusPrecision {
        unimplemented!("this method is not defined on secrets")
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWESecretPrepared<D, B> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32)
    }
}

pub trait GLWESecretPreparedFactory<B: Backend>
where
    Self: GetDegree + SvpPPolBytesOf + SvpPPolAlloc<B> + SvpPrepare<B>,
{
    fn glwe_secret_prepared_alloc(&self, rank: Rank) -> GLWESecretPrepared<B::OwnedBuf, B> {
        GLWESecretPrepared {
            data: self.svp_ppol_alloc(rank.into()),
            dist: Distribution::NONE,
        }
    }
    fn glwe_secret_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWESecretPrepared<B::OwnedBuf, B>
    where
        A: GLWEInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.glwe_secret_prepared_alloc(infos.rank())
    }

    fn glwe_secret_prepared_bytes_of(&self, rank: Rank) -> usize {
        self.bytes_of_svp_ppol(rank.into())
    }
    fn glwe_secret_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.glwe_secret_prepared_bytes_of(infos.rank())
    }

    fn glwe_secret_prepare<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretPreparedToBackendMut<B> + GetDistributionMut,
        O: GLWESecretToBackendRef<B> + GetDistribution,
    {
        {
            let mut res = res.to_backend_mut();
            let other = other.to_backend_ref();
            for i in 0..res.rank().into() {
                self.svp_prepare(&mut res.data, i, &other.data, i);
            }
        }

        *res.dist_mut() = *other.dist();
    }
}

impl<B: Backend> GLWESecretPreparedFactory<B> for Module<B> where
    Self: GetDegree + SvpPPolBytesOf + SvpPPolAlloc<B> + SvpPrepare<B>
{
}

// module-only API: allocation/size helpers are provided by `GLWESecretPreparedFactory` on `Module`.

impl<D: Data, B: Backend> GLWESecretPrepared<D, B> {
    pub fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    pub fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32)
    }
}

// module-only API: preparation is provided by `GLWESecretPreparedFactory` on `Module`.

pub trait GLWESecretPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWESecretPreparedBackendRef<'_, B>;
}

impl<B: Backend> GLWESecretPreparedToBackendRef<B> for GLWESecretPrepared<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> GLWESecretPreparedBackendRef<'_, B> {
        GLWESecretPrepared {
            dist: self.dist,
            data: self.data.to_backend_ref::<B>(),
        }
    }
}

pub trait GLWESecretPreparedToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> GLWESecretPreparedBackendMut<'_, B>;
}

impl<B: Backend> GLWESecretPreparedToBackendMut<B> for GLWESecretPrepared<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> GLWESecretPreparedBackendMut<'_, B> {
        GLWESecretPrepared {
            dist: self.dist,
            data: self.data.to_backend_mut::<B>(),
        }
    }
}
