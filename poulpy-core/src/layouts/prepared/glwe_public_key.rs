use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf},
    layouts::{Backend, Data, Module, VecZnxDft},
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{
        Base2K, GLWEInfos, GLWEPrepared, GLWEPreparedBackendMut, GLWEPreparedBackendRef, GLWEPreparedFactory,
        GLWEPreparedToBackendMut, GLWEPreparedToBackendRef, GLWEPublicKeyCore, GLWEToBackendRef, GetDegree, Rank, TorusPrecision,
    },
};

/// DFT-domain (prepared) variant of a GLWE public key.
///
/// Wraps a [`GLWEPrepared`] with distribution metadata for public-key
/// encryption. Tied to a specific backend via `B: Backend`.
///
/// This is [`GLWEPublicKeyCore`] over a `VecZnxDft` payload; `LWEInfos`,
/// `GLWEInfos` and the distribution accessors come from the payload-generic
/// impls there.
pub type GLWEPublicKeyPrepared<D, B> = GLWEPublicKeyCore<VecZnxDft<D, <B as Backend>::DftWord, B>>;

pub trait GLWEPublicKeyPreparedFactory<B: Backend>
where
    Self: GetDegree + GLWEPreparedFactory<B>,
{
    fn glwe_public_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
    ) -> GLWEPublicKeyPrepared<B::OwnedBuf, B> {
        GLWEPublicKeyPrepared {
            key: self.glwe_prepared_alloc(base2k, k, rank),
            dist: Distribution::NONE,
        }
    }

    fn glwe_public_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWEPublicKeyPrepared<B::OwnedBuf, B>
    where
        A: GLWEInfos,
    {
        self.glwe_public_key_prepared_alloc(infos.base2k(), infos.k(), infos.rank())
    }

    fn glwe_public_key_prepared_bytes_of(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        self.glwe_prepared_bytes_of(base2k, k, rank)
    }

    fn glwe_public_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        self.glwe_public_key_prepared_bytes_of(infos.base2k(), infos.k(), infos.rank())
    }

    fn glwe_public_key_prepare<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEPreparedToBackendMut<B> + GetDistributionMut,
        O: GLWEToBackendRef<B> + GetDistribution + GLWEInfos,
    {
        self.glwe_prepare(res, other);
        *res.dist_mut() = *other.dist();
    }
}

impl<B: Backend> GLWEPublicKeyPreparedFactory<B> for Module<B> where Self: VecZnxDftAlloc<B> + VecZnxDftBytesOf + VecZnxDftApply<B>
{}

// module-only API: allocation, sizing, and preparation are provided by
// `GLWEPublicKeyPreparedFactory` on `Module`.

pub type GLWEPublicKeyPreparedBackendRef<'a, B> = GLWEPublicKeyPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWEPublicKeyPreparedBackendMut<'a, B> = GLWEPublicKeyPrepared<<B as Backend>::BufMut<'a>, B>;

pub trait GLWEPublicKeyPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWEPublicKeyPreparedBackendRef<'_, B>;
}

impl<D: Data, B: Backend> GLWEPublicKeyPreparedToBackendRef<B> for GLWEPublicKeyPrepared<D, B>
where
    GLWEPrepared<D, B>: GLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GLWEPublicKeyPreparedBackendRef<'_, B> {
        GLWEPublicKeyPrepared {
            key: self.key.to_backend_ref(),
            dist: self.dist,
        }
    }
}

pub trait GLWEPublicKeyPreparedToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> GLWEPublicKeyPreparedBackendMut<'_, B>;
}

impl<D: Data, B: Backend> GLWEPublicKeyPreparedToBackendMut<B> for GLWEPublicKeyPrepared<D, B>
where
    GLWEPrepared<D, B>: GLWEPreparedToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> GLWEPublicKeyPreparedBackendMut<'_, B> {
        GLWEPublicKeyPrepared {
            key: self.key.to_backend_mut(),
            dist: self.dist,
        }
    }
}

impl<D: Data, B: Backend> GLWEPreparedToBackendMut<B> for GLWEPublicKeyPrepared<D, B>
where
    GLWEPrepared<D, B>: GLWEPreparedToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> GLWEPreparedBackendMut<'_, B> {
        self.key.to_backend_mut()
    }
}

impl<D: Data, B: Backend> GLWEPreparedToBackendRef<B> for GLWEPublicKeyPrepared<D, B>
where
    GLWEPrepared<D, B>: GLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GLWEPreparedBackendRef<'_, B> {
        self.key.to_backend_ref()
    }
}
