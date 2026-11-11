use poulpy_hal::layouts::VecZnxDftToBackendMut;
use poulpy_hal::layouts::VecZnxDftToBackendRef;
use poulpy_hal::layouts::{Backend, Data, Ring, Standard, VecZnxDft};

use crate::layouts::{Base2K, Degree, GLWEInfos, LWEInfos, Rank, TorusPrecision};

/// DFT-domain (prepared) variant of [`GLWE`](crate::layouts::GLWE).
///
/// Stores polynomials in the frequency domain of the backend's DFT/NTT
/// transform, enabling O(N log N) polynomial multiplication.
/// Tied to a specific backend via `B: Backend`.
#[derive(PartialEq)]
pub struct GLWEPrepared<D: Data, B: Backend, Rg: Ring = Standard> {
    pub(crate) noise: Option<crate::ComponentNoise>,
    pub(crate) data: VecZnxDft<D, B::DftWord, B, Rg>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
}

pub type GLWEPreparedBackendRef<'a, B, Rg = Standard> = GLWEPrepared<<B as Backend>::BufRef<'a>, B, Rg>;
pub type GLWEPreparedBackendMut<'a, B, Rg = Standard> = GLWEPrepared<<B as Backend>::BufMut<'a>, B, Rg>;

impl<D: Data, B: Backend, Rg: Ring> LWEInfos for GLWEPrepared<D, B, Rg> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.noise.clone()
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn max_size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl<D: Data, B: Backend, Rg: Ring> GLWEInfos for GLWEPrepared<D, B, Rg> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

pub trait GLWEPreparedToBackendRef<B: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> GLWEPreparedBackendRef<'_, B, Rg>;
}

impl<B: Backend, Rg: Ring> GLWEPreparedToBackendRef<B, Rg> for GLWEPrepared<B::OwnedBuf, B, Rg> {
    fn to_backend_ref(&self) -> GLWEPreparedBackendRef<'_, B, Rg> {
        GLWEPrepared {
            noise: crate::layouts::LWEInfos::noise(&self),
            data: self.data.to_backend_ref(),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

pub trait GLWEPreparedToBackendMut<B: Backend, Rg: Ring = Standard> {
    /// Backend hook for recording or propagating component noise metadata.
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);

    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GLWEPreparedBackendMut<'_, B, Rg>;
}

impl<B: Backend, Rg: Ring> GLWEPreparedToBackendMut<B, Rg> for GLWEPrepared<B::OwnedBuf, B, Rg> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GLWEPreparedBackendMut<'_, B, Rg> {
        self.noise = None;
        GLWEPrepared {
            noise: None,
            data: self.data.to_backend_mut(),
            base2k: self.base2k,
            k: self.k,
        }
    }
}
