use poulpy_hal::layouts::SvpPPolToBackendMut;
use poulpy_hal::layouts::SvpPPolToBackendRef;
use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPPolBytesOf},
    layouts::{Backend, Module, SvpPPol},
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{
        GLWEInfos, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretPreparedToBackendMut, GLWESecretPreparedToBackendRef,
        GLWESecretTensorCore, GLWESecretToBackendRef, GetDegree, Rank,
    },
};

/// DFT-domain (prepared) variant of [`GLWESecretTensor`](crate::layouts::GLWESecretTensor).
///
/// Stores the GLWE secret tensor with polynomials in the frequency domain
/// for fast tensor operations. Tied to a specific backend via `B: Backend`.
/// This is [`GLWESecretTensorCore`] over an `SvpPPol` payload: the same
/// semantic object as a coefficient-domain secret tensor, in the prepared
/// domain.
pub type GLWESecretTensorPrepared<D, B> = GLWESecretTensorCore<SvpPPol<D, <B as Backend>::DftWord, B>>;

pub trait GLWESecretTensorPreparedFactory<B: Backend> {
    fn glwe_secret_tensor_prepared_alloc(&self, rank: Rank) -> GLWESecretTensorPrepared<B::OwnedBuf, B>;
    fn glwe_secret_tensor_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWESecretTensorPrepared<B::OwnedBuf, B>
    where
        A: GLWEInfos;

    fn glwe_secret_tensor_prepared_bytes_of(&self, rank: Rank) -> usize;
    fn glwe_secret_tensor_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_secret_tensor_prepared_prepare<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretPreparedToBackendMut<B> + GetDistributionMut,
        O: GLWESecretToBackendRef<B> + GetDistribution;
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
        self.glwe_secret_prepared_bytes_of(infos.rank())
    }

    fn glwe_secret_tensor_prepared_prepare<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretPreparedToBackendMut<B> + GetDistributionMut,
        O: GLWESecretToBackendRef<B> + GetDistribution,
    {
        self.glwe_secret_prepare(res, other);
    }
}

// module-only API: allocation/size helpers are provided by `GLWESecretTensorPreparedFactory` on `Module`.
//
// `n()` and `rank()` come from `LWEInfos`/`GLWEInfos` on `GLWESecretTensorCore`.
// The inherent pair that used to live here reported `rank = data.cols()`, which
// is `pairs(rank)` for a tensor, and shadowed the correct trait method.

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

impl<B: Backend> GLWESecretPreparedToBackendRef<B> for GLWESecretTensorPrepared<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> crate::layouts::GLWESecretPreparedBackendRef<'_, B> {
        GLWESecretPrepared {
            data: self.data.to_backend_ref(),
            dist: self.dist,
        }
    }
}

impl<B: Backend> GLWESecretPreparedToBackendMut<B> for GLWESecretTensorPrepared<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> crate::layouts::GLWESecretPreparedBackendMut<'_, B> {
        GLWESecretPrepared {
            data: self.data.to_backend_mut(),
            dist: self.dist,
        }
    }
}
