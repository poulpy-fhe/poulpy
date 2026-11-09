use poulpy_hal::layouts::SvpPPolToBackendMut;
use poulpy_hal::layouts::SvpPPolToBackendRef;
use poulpy_hal::layouts::{Backend, Data, Ring, Standard, SvpPPol, ZnxInfos};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, LWEInfos, Rank},
};

/// DFT-domain (prepared) variant of [`GLWESecretTensor`](crate::layouts::GLWESecretTensor).
///
/// Stores the GLWE secret tensor with polynomials in the frequency domain
/// for fast tensor operations. Tied to a specific backend via `B: Backend`.
pub struct GLWESecretTensorPrepared<D: Data, B: Backend, Rg: Ring = Standard> {
    pub(crate) data: SvpPPol<D, B::DftWord, B, Rg>,
    pub(crate) rank: Rank,
    pub(crate) dist: Distribution,
}

impl<D: Data, BE: Backend, Rg: Ring> GetDistribution for GLWESecretTensorPrepared<D, BE, Rg> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, BE: Backend, Rg: Ring> GetDistributionMut for GLWESecretTensorPrepared<D, BE, Rg> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, B: Backend, Rg: Ring> LWEInfos for GLWESecretTensorPrepared<D, B, Rg> {
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
impl<D: Data, B: Backend, Rg: Ring> GLWEInfos for GLWESecretTensorPrepared<D, B, Rg> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: Data, B: Backend, Rg: Ring> GLWESecretTensorPrepared<D, B, Rg> {
    pub fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    /// Rank of the base secret this tensor was derived from, consistent with
    /// [`GLWEInfos::rank`]. The number of stored polynomials is `pairs(rank)`.
    pub fn rank(&self) -> Rank {
        self.rank
    }
}

pub type GLWESecretTensorPreparedBackendRef<'a, B, Rg = Standard> = GLWESecretTensorPrepared<<B as Backend>::BufRef<'a>, B, Rg>;
pub type GLWESecretTensorPreparedBackendMut<'a, B, Rg = Standard> = GLWESecretTensorPrepared<<B as Backend>::BufMut<'a>, B, Rg>;

pub trait GLWESecretTensorPreparedToBackendRef<B: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> GLWESecretTensorPreparedBackendRef<'_, B, Rg>;
}

impl<B: Backend, Rg: Ring> GLWESecretTensorPreparedToBackendRef<B, Rg> for GLWESecretTensorPrepared<B::OwnedBuf, B, Rg> {
    fn to_backend_ref(&self) -> GLWESecretTensorPreparedBackendRef<'_, B, Rg> {
        GLWESecretTensorPrepared {
            data: self.data.to_backend_ref(),
            rank: self.rank,
            dist: self.dist,
        }
    }
}

pub trait GLWESecretTensorPreparedToBackendMut<B: Backend, Rg: Ring = Standard> {
    fn to_backend_mut(&mut self) -> GLWESecretTensorPreparedBackendMut<'_, B, Rg>;
}

impl<B: Backend, Rg: Ring> GLWESecretTensorPreparedToBackendMut<B, Rg> for GLWESecretTensorPrepared<B::OwnedBuf, B, Rg> {
    fn to_backend_mut(&mut self) -> GLWESecretTensorPreparedBackendMut<'_, B, Rg> {
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
