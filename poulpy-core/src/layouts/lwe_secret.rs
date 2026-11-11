use poulpy_hal::layouts::ZnxWord;
use poulpy_hal::layouts::{
    Backend, Data, HostDataRef, Ring, ScalarZnx, ScalarZnxToBackendMut, ScalarZnxToBackendRef, Standard, ZnxView,
};
use poulpy_hal::{AlignedBuf, alloc_aligned};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, LWEInfos},
};

pub struct LWESecret<D: Data, W: ZnxWord, Rg: Ring = Standard> {
    pub(crate) data: ScalarZnx<D, W, Rg>,
    /// Distribution the base secret was sampled from. When this secret is
    /// obtained by flattening a GLWE secret, the tag is that of the source
    /// key's polynomial components and is not rescaled by the rank.
    /// See [`Distribution`].
    pub(crate) dist: Distribution,
}

pub type LWESecretBackendRef<'a, BE, Rg = Standard> = LWESecret<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord, Rg>;
pub type LWESecretBackendMut<'a, BE, Rg = Standard> = LWESecret<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord, Rg>;

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord, Rg: Ring> LWESecret<AlignedBuf, W, Rg> {
    pub(crate) fn alloc(n: Degree) -> Self {
        LWESecret {
            data: ScalarZnx::from_data(
                alloc_aligned::<u8>(ScalarZnx::<AlignedBuf, W, Rg>::bytes_of(n.into(), 1)),
                n.into(),
                1,
            ),
            dist: Distribution::NONE,
        }
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWESecret<D, W, Rg> {
    pub fn data(&self) -> &ScalarZnx<D, W, Rg> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut ScalarZnx<D, W, Rg> {
        &mut self.data
    }

    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> LWESecret<To::OwnedBuf, To::ZnxWord, Rg>
    where
        To: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        let n = self.data.n();
        let cols = self.data.cols();
        let data = self.data.data;
        LWESecret {
            data: ScalarZnx::from_data(data, n, cols),
            dist: self.dist,
        }
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GetDistribution for LWESecret<D, W, Rg> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GetDistributionMut for LWESecret<D, W, Rg> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> LWESecret<D, W, Rg> {
    /// Borrows the secret's coefficients in the layout's own word.
    pub fn raw(&self) -> &[W] {
        self.data.at(0, 0)
    }

    pub fn dist(&self) -> Distribution {
        self.dist
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWEInfos for LWESecret<D, W, Rg> {
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

pub trait LWESecretToBackendRef<BE: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> LWESecretToBackendRef<BE, Rg> for LWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE, Rg> {
        LWESecret {
            dist: self.dist,
            data: <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxToBackendRef<BE, Rg>>::to_backend_ref(&self.data),
        }
    }
}

impl<BE: Backend, Rg: Ring> LWESecretToBackendRef<BE, Rg> for &LWESecret<BE::BufRef<'_>, BE::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE, Rg> {
        LWESecret {
            dist: self.dist,
            data: ScalarZnx::from_data(BE::view_ref(&self.data.data), self.data.n(), self.data.cols()),
        }
    }
}

impl<BE: Backend, Rg: Ring> LWESecretToBackendRef<BE, Rg> for &mut LWESecret<BE::BufMut<'_>, BE::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE, Rg> {
        LWESecret {
            dist: self.dist,
            data: ScalarZnx::from_data(BE::view_ref_mut(&self.data.data), self.data.n(), self.data.cols()),
        }
    }
}

pub trait LWESecretToBackendMut<BE: Backend, Rg: Ring = Standard>: LWESecretToBackendRef<BE, Rg> {
    fn to_backend_mut(&mut self) -> LWESecretBackendMut<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> LWESecretToBackendMut<BE, Rg> for LWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn to_backend_mut(&mut self) -> LWESecretBackendMut<'_, BE, Rg> {
        LWESecret {
            dist: self.dist,
            data: <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxToBackendMut<BE, Rg>>::to_backend_mut(&mut self.data),
        }
    }
}

impl<BE: Backend, Rg: Ring> LWESecretToBackendMut<BE, Rg> for &mut LWESecret<BE::BufMut<'_>, BE::ZnxWord, Rg> {
    fn to_backend_mut(&mut self) -> LWESecretBackendMut<'_, BE, Rg> {
        let n = self.data.n();
        let cols = self.data.cols();
        LWESecret {
            dist: self.dist,
            data: ScalarZnx::from_data(BE::view_mut_ref(&mut self.data.data), n, cols),
        }
    }
}
