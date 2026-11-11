use poulpy_hal::layouts::ZnxWord;
use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, Ring, ScalarZnx, ScalarZnxToBackendRef, Standard, ZnxView, ZnxViewMut,
};
use poulpy_hal::{AlignedBuf, alloc_aligned};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, LWEInfos, Rank},
};

/// Number of distinct unordered secret-key products for a rank-`r` tensor key.
/// Depends only on the rank: neither the storage nor the coefficient word.
pub(crate) fn pairs(rank: usize) -> usize {
    (((rank + 1) * rank) >> 1).max(1)
}

/// Tensor of a [`GLWESecret`](crate::layouts::GLWESecret): the `(rank + 1) * rank / 2` distinct products
/// `s_i * s_j` of a base secret `(s_0, ..., s_{rank-1})`, e.g.
/// `(1, s_0, s_1)^(x)2 = (s_0^2, s_0*s_1, s_1^2)`.
///
/// Note that `dist` is the tag of the *base* secret, not of the stored
/// products: the coefficients held here are neither ternary nor binary, and
/// no [`Distribution`] variant describes them. The tag is kept because the
/// products' own statistics are a closed-form function of the base
/// distribution (see [`Distribution`] for the variance).
pub struct GLWESecretTensor<D: Data, W: ZnxWord, Rg: Ring = Standard> {
    pub(crate) data: ScalarZnx<D, W, Rg>,
    pub(crate) rank: Rank,
    /// Distribution of the base secret this tensor was derived from, *not*
    /// of the `s_i * s_j` coefficients stored in `data`.
    pub(crate) dist: Distribution,
}

impl<W: ZnxWord, Rg: Ring> GLWESecretTensor<AlignedBuf, W, Rg> {}

impl<D: Data, W: ZnxWord, Rg: Ring> GetDistribution for GLWESecretTensor<D, W, Rg> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GetDistributionMut for GLWESecretTensor<D, W, Rg> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWESecretTensor<D, W, Rg> {
    pub fn data(&self) -> &ScalarZnx<D, W, Rg> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut ScalarZnx<D, W, Rg> {
        &mut self.data
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWEInfos for GLWESecretTensor<D, W, Rg> {
    fn base2k(&self) -> Base2K {
        Base2K(0)
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn max_size(&self) -> usize {
        1
    }

    fn k(&self) -> super::TorusPrecision {
        unimplemented!("this method is not defined on secrets")
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> GLWESecretTensor<D, W, Rg> {
    pub fn at(&self, mut i: usize, mut j: usize) -> ScalarZnx<&[u8], W, Rg> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank().into();
        ScalarZnx::from_data(
            bytemuck::cast_slice(self.data.at(i * rank + j - (i * (i + 1) / 2), 0)),
            self.n().into(),
            1,
        )
    }
}

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> GLWESecretTensor<D, W, Rg> {
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> ScalarZnx<&mut [u8], W, Rg> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank().into();
        let n = self.n().into();
        ScalarZnx::from_data(
            bytemuck::cast_slice_mut(self.data.at_mut(i * rank + j - (i * (i + 1) / 2), 0)),
            n,
            1,
        )
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEInfos for GLWESecretTensor<D, W, Rg> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

pub type GLWESecretTensorBackendRef<'a, BE, Rg = Standard> =
    GLWESecretTensor<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord, Rg>;
pub type GLWESecretTensorBackendMut<'a, BE, Rg = Standard> =
    GLWESecretTensor<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord, Rg>;

/// Backend view of a tensor secret.
///
/// Deliberately distinct from [`GLWESecretToBackendRef`](crate::layouts::GLWESecretToBackendRef): a tensor secret is
/// not interchangeable with the base secret it was derived from, so it is
/// not accepted by APIs that ask for a [`GLWESecret`](crate::layouts::GLWESecret).
pub trait GLWESecretTensorToBackendRef<BE: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> GLWESecretTensorBackendRef<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> GLWESecretTensorToBackendRef<BE, Rg> for GLWESecretTensor<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> GLWESecretTensorBackendRef<'_, BE, Rg> {
        GLWESecretTensor {
            data: <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxToBackendRef<BE, Rg>>::to_backend_ref(&self.data),
            rank: self.rank,
            dist: self.dist,
        }
    }
}

/// Mutable backend view of a tensor secret. See [`GLWESecretTensorToBackendRef`].
pub trait GLWESecretTensorToBackendMut<BE: Backend, Rg: Ring = Standard>: GLWESecretTensorToBackendRef<BE, Rg> {
    fn to_backend_mut(&mut self) -> GLWESecretTensorBackendMut<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> GLWESecretTensorToBackendMut<BE, Rg> for GLWESecretTensor<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn to_backend_mut(&mut self) -> GLWESecretTensorBackendMut<'_, BE, Rg> {
        let rank = self.rank;
        GLWESecretTensor {
            data:
                <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as poulpy_hal::layouts::ScalarZnxToBackendMut<BE, Rg>>::to_backend_mut(
                    &mut self.data,
                ),
            rank,
            dist: self.dist,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord, Rg: Ring> GLWESecretTensor<AlignedBuf, W, Rg> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.rank())
    }

    pub(crate) fn alloc(n: Degree, rank: Rank) -> Self {
        GLWESecretTensor {
            data: ScalarZnx::from_data(
                alloc_aligned::<u8>(ScalarZnx::<AlignedBuf, W, Rg>::bytes_of(n.into(), pairs(rank.into()))),
                n.into(),
                pairs(rank.into()),
            ),
            rank,
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), pairs(infos.rank().into()).into())
    }

    pub fn bytes_of(n: Degree, rank: Rank) -> usize {
        ScalarZnx::<AlignedBuf, W, Rg>::bytes_of(n.into(), pairs(rank.into()))
    }
}
