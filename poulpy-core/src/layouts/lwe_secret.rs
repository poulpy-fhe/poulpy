use poulpy_hal::layouts::ZnxWord;
use poulpy_hal::layouts::{Backend, Data, HostDataRef, ScalarZnx, ScalarZnxToBackendMut, ScalarZnxToBackendRef, ZnxView};
use poulpy_hal::{AlignedBuf, alloc_aligned};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, LWEInfos},
};

pub struct LWESecret<D: Data, W: ZnxWord> {
    pub(crate) data: ScalarZnx<D, W>,
    /// Distribution the base secret was sampled from. When this secret is
    /// obtained by flattening a GLWE secret, the tag is that of the source
    /// key's polynomial components and is not rescaled by the rank.
    /// See [`Distribution`].
    pub(crate) dist: Distribution,
}

pub type LWESecretBackendRef<'a, BE> = LWESecret<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord>;
pub type LWESecretBackendMut<'a, BE> = LWESecret<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord>;

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> LWESecret<AlignedBuf, W> {
    pub(crate) fn alloc(n: Degree) -> Self {
        LWESecret {
            data: ScalarZnx::from_data(
                alloc_aligned::<u8>(ScalarZnx::<AlignedBuf, W>::bytes_of(n.into(), 1)),
                n.into(),
                1,
            ),
            dist: Distribution::NONE,
        }
    }
}

impl<D: Data, W: ZnxWord> LWESecret<D, W> {
    pub fn data(&self) -> &ScalarZnx<D, W> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut ScalarZnx<D, W> {
        &mut self.data
    }

    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> LWESecret<To::OwnedBuf, To::ZnxWord>
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

impl<D: Data, W: ZnxWord> GetDistribution for LWESecret<D, W> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, W: ZnxWord> GetDistributionMut for LWESecret<D, W> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: HostDataRef, W: ZnxWord> LWESecret<D, W> {
    /// Borrows the secret's coefficients in the layout's own word.
    pub fn raw(&self) -> &[W] {
        self.data.at(0, 0)
    }

    pub fn dist(&self) -> Distribution {
        self.dist
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for LWESecret<D, W> {
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

pub trait LWESecretToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE>;
}

impl<BE: Backend> LWESecretToBackendRef<BE> for LWESecret<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE> {
        LWESecret {
            dist: self.dist,
            data: <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&self.data),
        }
    }
}

impl<BE: Backend> LWESecretToBackendRef<BE> for &LWESecret<BE::BufRef<'_>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE> {
        LWESecret {
            dist: self.dist,
            data: ScalarZnx::from_data(BE::view_ref(&self.data.data), self.data.n(), self.data.cols()),
        }
    }
}

impl<BE: Backend> LWESecretToBackendRef<BE> for &mut LWESecret<BE::BufMut<'_>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE> {
        LWESecret {
            dist: self.dist,
            data: ScalarZnx::from_data(BE::view_ref_mut(&self.data.data), self.data.n(), self.data.cols()),
        }
    }
}

pub trait LWESecretToBackendMut<BE: Backend>: LWESecretToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> LWESecretBackendMut<'_, BE>;
}

impl<BE: Backend> LWESecretToBackendMut<BE> for LWESecret<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> LWESecretBackendMut<'_, BE> {
        LWESecret {
            dist: self.dist,
            data: <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
        }
    }
}

impl<BE: Backend> LWESecretToBackendMut<BE> for &mut LWESecret<BE::BufMut<'_>, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> LWESecretBackendMut<'_, BE> {
        let n = self.data.n();
        let cols = self.data.cols();
        LWESecret {
            dist: self.dist,
            data: ScalarZnx::from_data(BE::view_mut_ref(&mut self.data.data), n, cols),
        }
    }
}
