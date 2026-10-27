use poulpy_hal::layouts::VmpPMatToBackendMut;
use poulpy_hal::layouts::VmpPMatToBackendRef;
use poulpy_hal::layouts::{Backend, Data, VmpPMat};

use crate::layouts::{Base2K, Degree, Dnum, GGSWInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision};

/// DFT-domain (prepared) variant of [`GGSW`](crate::layouts::GGSW).
///
/// Stores the GGSW gadget matrix with polynomials in the frequency domain
/// of the backend's DFT/NTT transform, enabling O(N log N) polynomial
/// operations. Tied to a specific backend via `B: Backend`.
#[derive(PartialEq)]
pub struct GGSWPrepared<D: Data, B: Backend> {
    pub(crate) noise: Option<crate::ComponentNoise>,
    pub(crate) data: VmpPMat<D, B::DftWord, B>,
    pub(crate) k_aux: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) k_digit: TorusPrecision,
}

pub type GGSWPreparedBackendRef<'a, B> = GGSWPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GGSWPreparedBackendMut<'a, B> = GGSWPrepared<<B as Backend>::BufMut<'a>, B>;

impl<D: Data, B: Backend> LWEInfos for GGSWPrepared<D, B> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.noise.clone()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn max_size(&self) -> usize {
        crate::layouts::key_size(self.base2k, self.dnum(), self.k_digit, self.k_aux)
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::key_k(self.dnum(), self.k_digit, self.k_aux)
    }
}

impl<D: Data, B: Backend> GLWEInfos for GGSWPrepared<D, B> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }
}

impl<D: Data, B: Backend> GGSWInfos for GGSWPrepared<D, B> {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

    fn k_digit(&self) -> TorusPrecision {
        self.k_digit
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: Data, B: Backend> GGSWPrepared<D, B> {
    pub fn data(&self) -> &VmpPMat<D, B::DftWord, B> {
        &self.data
    }
}

pub trait GGSWPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GGSWPreparedBackendRef<'_, B>;
}

impl<B: Backend> GGSWPreparedToBackendRef<B> for GGSWPrepared<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> GGSWPreparedBackendRef<'_, B> {
        GGSWPrepared {
            noise: crate::layouts::LWEInfos::noise(&self),
            base2k: self.base2k,
            k_aux: self.k_aux,
            k_digit: self.k_digit,
            data: self.data.to_backend_ref(),
        }
    }
}

pub trait GGSWPreparedToBackendMut<B: Backend> {
    /// Backend hook for recording or propagating component noise metadata.
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);

    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GGSWPreparedBackendMut<'_, B>;
}

impl<B: Backend> GGSWPreparedToBackendMut<B> for GGSWPrepared<B::OwnedBuf, B> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GGSWPreparedBackendMut<'_, B> {
        self.noise = None;
        GGSWPrepared {
            noise: None,
            base2k: self.base2k,
            k_aux: self.k_aux,
            k_digit: self.k_digit,
            data: self.data.to_backend_mut(),
        }
    }
}
