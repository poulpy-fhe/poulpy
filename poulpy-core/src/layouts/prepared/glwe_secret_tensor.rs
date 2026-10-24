use poulpy_hal::layouts::SvpPPolToBackendMut;
use poulpy_hal::layouts::SvpPPolToBackendRef;
use poulpy_hal::layouts::{Backend, Data, SvpPPol, ZnxInfos};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, LWEInfos, Rank},
};

/// DFT-domain (prepared) variant of [`GLWESecretTensor`](crate::layouts::GLWESecretTensor).
///
/// Stores the GLWE secret tensor with polynomials in the frequency domain
/// for fast tensor operations. Tied to a specific backend via `B: Backend`.
pub struct GLWESecretTensorPrepared<D: Data, B: Backend> {
    pub(crate) data: SvpPPol<D, B::DftWord, B>,
    pub(crate) rank: Rank,
    pub(crate) dist: Distribution,
}

impl<D: Data, BE: Backend> GetDistribution for GLWESecretTensorPrepared<D, BE> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, BE: Backend> GetDistributionMut for GLWESecretTensorPrepared<D, BE> {
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
