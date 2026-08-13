use poulpy_hal::layouts::SvpPPolToBackendMut;
use poulpy_hal::layouts::SvpPPolToBackendRef;
use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPPolBytesOf, SvpPreparePPol},
    layouts::{Backend, Data, HostDataMut, HostDataRef, Module, SvpPPol, ZnxInfos},
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, GLWESecretPreparedFactory, GLWESecretTensorToBackendRef, GetDegree, LWEInfos, Rank},
};

/// DFT-domain (prepared) variant of [`GLWESecretTensor`].
///
/// Stores the GLWE secret tensor with polynomials in the frequency domain
/// for fast tensor operations. Tied to a specific backend via `B: Backend`.
pub struct GLWESecretTensorPrepared<D: Data, B: Backend> {
    pub(crate) data: SvpPPol<D, B::DftWord, B>,
    pub(crate) rank: Rank,
    pub(crate) dist: Distribution,
}

impl<D: HostDataRef, BE: Backend> GetDistribution for GLWESecretTensorPrepared<D, BE> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: HostDataMut, BE: Backend> GetDistributionMut for GLWESecretTensorPrepared<D, BE> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, B: Backend> LWEInfos for GLWESecretTensorPrepared<D, B> {
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
impl<D: Data, B: Backend> GLWEInfos for GLWESecretTensorPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

pub trait GLWESecretTensorPreparedFactory<B: Backend> {
    fn glwe_secret_tensor_prepared_alloc(&self, rank: Rank) -> GLWESecretTensorPrepared<B::OwnedBuf, B>;
    fn glwe_secret_tensor_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWESecretTensorPrepared<B::OwnedBuf, B>
    where
        A: GLWEInfos;

    fn glwe_secret_tensor_prepared_bytes_of(&self, rank: Rank) -> usize;
    fn glwe_secret_tensor_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Moves a [`GLWESecretTensor`](crate::layouts::GLWESecretTensor) to the
    /// DFT domain.
    ///
    /// Only a tensor secret is accepted: preparing a base
    /// [`GLWESecret`](crate::layouts::GLWESecret) is
    /// [`glwe_secret_prepare`](GLWESecretPreparedFactory::glwe_secret_prepare),
    /// and the two are not interchangeable (they do not even have the same
    /// number of columns).
    fn glwe_secret_tensor_prepared_prepare<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretTensorPreparedToBackendMut<B> + GetDistributionMut,
        O: GLWESecretTensorToBackendRef<B> + GetDistribution;
}

impl<B: Backend> GLWESecretTensorPreparedFactory<B> for Module<B>
where
    Self: GLWESecretPreparedFactory<B>,
{
    fn glwe_secret_tensor_prepared_alloc(&self, rank: Rank) -> GLWESecretTensorPrepared<B::OwnedBuf, B> {
        GLWESecretTensorPrepared {
            data: self.svp_ppol_alloc(crate::layouts::pairs(rank.into())),
            rank,
            dist: Distribution::NONE,
        }
    }
    fn glwe_secret_tensor_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWESecretTensorPrepared<B::OwnedBuf, B>
    where
        A: GLWEInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.glwe_secret_tensor_prepared_alloc(infos.rank())
    }

    fn glwe_secret_tensor_prepared_bytes_of(&self, rank: Rank) -> usize {
        self.bytes_of_svp_ppol(crate::layouts::pairs(rank.into()))
    }
    fn glwe_secret_tensor_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.glwe_secret_tensor_prepared_bytes_of(infos.rank())
    }

    fn glwe_secret_tensor_prepared_prepare<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretTensorPreparedToBackendMut<B> + GetDistributionMut,
        O: GLWESecretTensorToBackendRef<B> + GetDistribution,
    {
        {
            let mut res = res.to_backend_mut();
            let other = other.to_backend_ref();
            assert_eq!(
                res.rank, other.rank,
                "GLWESecretTensorPrepared rank must equal the source tensor's rank"
            );
            let cols: usize = other.data.cols();
            assert_eq!(res.data.cols(), cols);
            for i in 0..cols {
                self.svp_prepare_ppol(&mut res.data, i, &other.data, i);
            }
        }

        *res.dist_mut() = *other.dist();
    }
}

// module-only API: allocation/size helpers are provided by `GLWESecretTensorPreparedFactory` on `Module`.

impl<D: Data, B: Backend> GLWESecretTensorPrepared<D, B> {
    pub fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    /// Rank of the base secret this tensor was derived from, consistent with
    /// [`GLWEInfos::rank`]. The number of stored polynomials is `pairs(rank)`.
    pub fn rank(&self) -> Rank {
        self.rank
    }
}

// module-only API: preparation is provided by `GLWESecretTensorPreparedFactory` on `Module`.

pub type GLWESecretTensorPreparedBackendRef<'a, B> = GLWESecretTensorPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWESecretTensorPreparedBackendMut<'a, B> = GLWESecretTensorPrepared<<B as Backend>::BufMut<'a>, B>;

pub trait GLWESecretTensorPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWESecretTensorPreparedBackendRef<'_, B>;
}

impl<B: Backend> GLWESecretTensorPreparedToBackendRef<B> for GLWESecretTensorPrepared<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> GLWESecretTensorPreparedBackendRef<'_, B> {
        GLWESecretTensorPrepared {
            data: self.data.to_backend_ref(),
            rank: self.rank,
            dist: self.dist,
        }
    }
}

pub trait GLWESecretTensorPreparedToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> GLWESecretTensorPreparedBackendMut<'_, B>;
}

impl<B: Backend> GLWESecretTensorPreparedToBackendMut<B> for GLWESecretTensorPrepared<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> GLWESecretTensorPreparedBackendMut<'_, B> {
        GLWESecretTensorPrepared {
            data: self.data.to_backend_mut(),
            rank: self.rank,
            dist: self.dist,
        }
    }
}

// No `GLWESecretPreparedToBackendRef`/`Mut` for `GLWESecretTensorPrepared`:
// a prepared tensor secret holds `pairs(rank)` polynomials of secret products
// and is not a substitute for a prepared base secret. Consumers that need one
// must ask for `GLWESecretTensorPreparedToBackendRef` explicitly.
