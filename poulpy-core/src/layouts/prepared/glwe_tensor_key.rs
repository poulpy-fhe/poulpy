use crate::error::Result;
use poulpy_hal::layouts::{Backend, Data};

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
pub struct GLWETensorKeyPrepared<D: Data, B: Backend>(pub(crate) GGLWEPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for GLWETensorKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GLWEInfos for GLWETensorKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWETensorKeyPrepared<D, B> {
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

pub type GLWETensorKeyPreparedBackendRef<'a, B> = GLWETensorKeyPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWETensorKeyPreparedBackendMut<'a, B> = GLWETensorKeyPrepared<<B as Backend>::BufMut<'a>, B>;

impl<D: Data, B: Backend> GLWETensorKeyPrepared<D, B> {
    /// This key read through a coarser `k_digit`. See [`GGLWEPrepared::with_k_digit`].
    pub fn with_k_digit(&self, k_digit: TorusPrecision) -> Result<GLWETensorKeyPreparedBackendRef<'_, B>>
    where
        GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
    {
        Ok(GLWETensorKeyPrepared(self.0.with_k_digit(k_digit)?))
    }
}

pub trait GLWETensorKeyPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWETensorKeyPreparedBackendRef<'_, B>;
}

impl<D: Data, B: Backend> GLWETensorKeyPreparedToBackendRef<B> for GLWETensorKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GLWETensorKeyPreparedBackendRef<'_, B> {
        GLWETensorKeyPrepared(self.0.to_backend_ref())
    }
}

pub trait GLWETensorKeyPreparedToBackendMut<B: Backend> {
    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GLWETensorKeyPreparedBackendMut<'_, B>;
}

impl<D: Data, B: Backend> GLWETensorKeyPreparedToBackendMut<B> for GLWETensorKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> GLWETensorKeyPreparedBackendMut<'_, B> {
        GLWETensorKeyPrepared(self.0.to_backend_mut())
    }
}

impl<D: Data, B: Backend> GGLWEPreparedToBackendRef<B> for GLWETensorKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        self.0.to_backend_ref()
    }
}

impl<'a, B: Backend + 'a> GGLWEPreparedToBackendRef<B> for &GLWETensorKeyPrepared<B::BufRef<'a>, B> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        GGLWEPrepared {
            noise: self.0.noise.clone(),
            base2k: self.0.base2k,
            k_aux: self.0.k_aux,
            k_digit: self.0.k_digit,
            k_offset: self.0.k_offset,
            dnum: self.0.dnum,
            stride: self.0.stride,
            data: poulpy_hal::layouts::vmp_pmat_backend_ref_from_ref::<B>(&self.0.data),
        }
    }
}

impl<D: Data, B: Backend> GGLWEPreparedToBackendMut<B> for GLWETensorKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        <_ as GGLWEPreparedToBackendMut<B>>::set_noise(&mut self.0, metadata);
    }

    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B> {
        self.0.to_backend_mut()
    }
}
