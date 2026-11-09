use crate::error::Result;
use poulpy_hal::layouts::{Backend, Data, Ring, Standard};

use crate::layouts::prepared::{GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef};
use crate::layouts::{
    Base2K, Degree, Dnum, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendMut, GGLWEPreparedBackendRef, GLWEInfos, LWEInfos, Rank,
    TorusPrecision,
};

/// DFT-domain (prepared) variant of a GLWE tensor key.
///
/// A newtype wrapper around [`GGLWEPrepared`] for tensor operations.
/// Tied to a specific backend via `B: Backend`.
#[derive(PartialEq)]
pub struct GLWETensorKeyPrepared<D: Data, B: Backend, Rg: Ring = Standard>(pub(crate) GGLWEPrepared<D, B, Rg>);

impl<D: Data, B: Backend, Rg: Ring> LWEInfos for GLWETensorKeyPrepared<D, B, Rg> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.0.noise()
    }

    fn n(&self) -> Degree {
        self.0.n()
    }

    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn max_size(&self) -> usize {
        self.0.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.0.k()
    }
}

impl<D: Data, B: Backend, Rg: Ring> GLWEInfos for GLWETensorKeyPrepared<D, B, Rg> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend, Rg: Ring> GGLWEInfos for GLWETensorKeyPrepared<D, B, Rg> {
    fn k_aux(&self) -> TorusPrecision {
        self.0.k_aux()
    }

    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.0.k_digit()
    }

    fn k_offset(&self) -> TorusPrecision {
        self.0.k_offset()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }

    fn stride(&self) -> usize {
        self.0.stride()
    }
}

pub type GLWETensorKeyPreparedBackendRef<'a, B, Rg = Standard> = GLWETensorKeyPrepared<<B as Backend>::BufRef<'a>, B, Rg>;
pub type GLWETensorKeyPreparedBackendMut<'a, B, Rg = Standard> = GLWETensorKeyPrepared<<B as Backend>::BufMut<'a>, B, Rg>;

impl<D: Data, B: Backend, Rg: Ring> GLWETensorKeyPrepared<D, B, Rg> {
    /// This key read through a coarser `k_digit`. See [`GGLWEPrepared::with_k_digit`].
    pub fn with_k_digit(&self, k_digit: TorusPrecision) -> Result<GLWETensorKeyPreparedBackendRef<'_, B, Rg>>
    where
        GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendRef<B, Rg>,
    {
        Ok(GLWETensorKeyPrepared(self.0.with_k_digit(k_digit)?))
    }
}

pub trait GLWETensorKeyPreparedToBackendRef<B: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> GLWETensorKeyPreparedBackendRef<'_, B, Rg>;
}

impl<D: Data, B: Backend, Rg: Ring> GLWETensorKeyPreparedToBackendRef<B, Rg> for GLWETensorKeyPrepared<D, B, Rg>
where
    GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendRef<B, Rg>,
{
    fn to_backend_ref(&self) -> GLWETensorKeyPreparedBackendRef<'_, B, Rg> {
        GLWETensorKeyPrepared(self.0.to_backend_ref())
    }
}

pub trait GLWETensorKeyPreparedToBackendMut<B: Backend, Rg: Ring = Standard> {
    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GLWETensorKeyPreparedBackendMut<'_, B, Rg>;
}

impl<D: Data, B: Backend, Rg: Ring> GLWETensorKeyPreparedToBackendMut<B, Rg> for GLWETensorKeyPrepared<D, B, Rg>
where
    GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendMut<B, Rg>,
{
    fn to_backend_mut(&mut self) -> GLWETensorKeyPreparedBackendMut<'_, B, Rg> {
        GLWETensorKeyPrepared(self.0.to_backend_mut())
    }
}

impl<D: Data, B: Backend, Rg: Ring> GGLWEPreparedToBackendRef<B, Rg> for GLWETensorKeyPrepared<D, B, Rg>
where
    GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendRef<B, Rg>,
{
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B, Rg> {
        self.0.to_backend_ref()
    }
}

impl<'a, B: Backend + 'a, Rg: Ring> GGLWEPreparedToBackendRef<B, Rg> for &GLWETensorKeyPrepared<B::BufRef<'a>, B, Rg> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B, Rg> {
        GGLWEPrepared {
            noise: self.0.noise.clone(),
            base2k: self.0.base2k,
            k_aux: self.0.k_aux,
            k_digit: self.0.k_digit,
            k_offset: self.0.k_offset,
            dnum: self.0.dnum,
            stride: self.0.stride,
            data: poulpy_hal::layouts::vmp_pmat_backend_ref_from_ref::<B, Rg>(&self.0.data),
        }
    }
}

impl<D: Data, B: Backend, Rg: Ring> GGLWEPreparedToBackendMut<B, Rg> for GLWETensorKeyPrepared<D, B, Rg>
where
    GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendMut<B, Rg>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        <_ as GGLWEPreparedToBackendMut<B, Rg>>::set_noise(&mut self.0, metadata);
    }

    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B, Rg> {
        self.0.to_backend_mut()
    }
}
