use crate::{error::Result, layouts::GGLWELayout};
use poulpy_hal::layouts::VmpPMatToBackendMut;
use poulpy_hal::layouts::VmpPMatToBackendRef;
use poulpy_hal::layouts::vmp_pmat_backend_ref_from_ref;
use poulpy_hal::layouts::{Backend, Data, VmpPMat};

use crate::layouts::{Base2K, Degree, Dnum, GGLWEInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision};

/// DFT-domain (prepared) variant of [`GGLWE`](crate::layouts::GGLWE).
///
/// Stores the gadget GLWE matrix with polynomials in the frequency domain
/// of the backend's DFT/NTT transform, enabling O(N log N) polynomial
/// multiplication. The underlying data is held as a [`VmpPMat`], which
/// represents a prepared matrix suitable for vector-matrix products.
///
/// Tied to a specific backend via `B: Backend`.
#[derive(PartialEq)]
pub struct GGLWEPrepared<D: Data, B: Backend> {
    pub(crate) noise: Option<crate::ComponentNoise>,
    pub(crate) data: VmpPMat<D, B::DftWord, B>,
    pub(crate) k_aux: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) k_digit: TorusPrecision,
    /// Row offset in bits (see [`GGLWEInfos::k_offset`]).
    pub(crate) k_offset: TorusPrecision,
    /// Digits reachable at `k_digit`: the stored rows, or fewer for a view.
    pub(crate) dnum: Dnum,
    /// Rows to skip between digits: 1 as stored, `k_digit / stored k_digit` for a view.
    pub(crate) stride: usize,
}

pub type GGLWEPreparedBackendRef<'a, B> = GGLWEPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GGLWEPreparedBackendMut<'a, B> = GGLWEPrepared<<B as Backend>::BufMut<'a>, B>;

/// Provides LWE-level parameter accessors (degree, base2k, precision, size).
impl<D: Data, B: Backend> LWEInfos for GGLWEPrepared<D, B> {
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
        crate::layouts::offset_key_size(self.base2k, self.k_offset, self.dnum(), self.k_digit, self.k_aux)
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::offset_key_k(self.k_offset, self.dnum(), self.k_digit, self.k_aux)
    }
}

/// Provides the GLWE rank, derived from the output rank.
impl<D: Data, B: Backend> GLWEInfos for GGLWEPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

/// Provides GGLWE-specific parameter accessors (input/output rank, k_digit, dnum).
impl<D: Data, B: Backend> GGLWEInfos for GGLWEPrepared<D, B> {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

    fn rank_in(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }

    fn rank_out(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }

    fn k_digit(&self) -> TorusPrecision {
        self.k_digit
    }

    fn k_offset(&self) -> TorusPrecision {
        self.k_offset
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }

    fn stride(&self) -> usize {
        self.stride
    }
}

impl<D: Data, B: Backend> GGLWEPrepared<D, B> {
    /// Returns a shared reference to the underlying [`VmpPMat`].
    pub fn data(&self) -> &VmpPMat<D, B::DftWord, B> {
        &self.data
    }
}

impl<D: Data, B: Backend> GGLWEPrepared<D, B> {
    /// This key read through a coarser `k_digit`. No row is copied, only the
    /// scalars change; see [`GGLWELayout::at_k_digit`] for the rule.
    pub fn with_k_digit(&self, k_digit: TorusPrecision) -> Result<GGLWEPreparedBackendRef<'_, B>>
    where
        Self: GGLWEPreparedToBackendRef<B>,
    {
        let mut key: GGLWEPreparedBackendRef<'_, B> = self.to_backend_ref();
        let coarse: GGLWELayout = key.gglwe_layout().at_k_digit(k_digit)?;
        key.stride = coarse.stride;
        key.k_digit = coarse.k_digit;
        key.dnum = coarse.dnum;
        key.k_aux = coarse.k_aux;
        Ok(key)
    }
}

pub trait GGLWEPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B>;
}

impl<B: Backend> GGLWEPreparedToBackendRef<B> for GGLWEPrepared<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        GGLWEPrepared {
            noise: crate::layouts::LWEInfos::noise(&self),
            base2k: self.base2k,
            k_aux: self.k_aux,
            k_digit: self.k_digit,
            k_offset: self.k_offset,
            dnum: self.dnum,
            stride: self.stride,
            data: self.data.to_backend_ref(),
        }
    }
}

impl<B: Backend> GGLWEPreparedToBackendRef<B> for &GGLWEPrepared<B::BufRef<'_>, B> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        GGLWEPrepared {
            noise: crate::layouts::LWEInfos::noise(&self),
            base2k: self.base2k,
            k_aux: self.k_aux,
            k_digit: self.k_digit,
            k_offset: self.k_offset,
            dnum: self.dnum,
            stride: self.stride,
            data: vmp_pmat_backend_ref_from_ref::<B>(&self.data),
        }
    }
}

pub trait GGLWEPreparedToBackendMut<B: Backend> {
    /// Backend hook for recording or propagating component noise metadata.
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);

    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B>;
}

impl<B: Backend> GGLWEPreparedToBackendMut<B> for GGLWEPrepared<B::OwnedBuf, B> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B> {
        self.noise = None;
        GGLWEPrepared {
            noise: None,
            base2k: self.base2k,
            k_aux: self.k_aux,
            k_digit: self.k_digit,
            k_offset: self.k_offset,
            dnum: self.dnum,
            stride: self.stride,
            data: self.data.to_backend_mut(),
        }
    }
}
