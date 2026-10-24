use crate::error::Result;
use poulpy_hal::layouts::{Backend, Data};

use crate::layouts::prepared::{GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef};
use crate::layouts::{
    Base2K, Degree, Dnum, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendMut, GGLWEPreparedBackendRef, GLWEInfos,
    GetGaloisElement, LWEInfos, Rank, SetGaloisElement, TorusPrecision,
};
use poulpy_hal::layouts::vmp_pmat_backend_ref_from_ref;

#[derive(PartialEq)]
pub struct GLWEAutomorphismKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GGLWEPrepared<D, B>,
    pub(crate) p: i64,
}

impl<D: Data, B: Backend> LWEInfos for GLWEAutomorphismKeyPrepared<D, B> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.key.noise()
    }

    fn n(&self) -> Degree {
        self.key.n()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn max_size(&self) -> usize {
        self.key.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.key.k()
    }
}

impl<D: Data, B: Backend> GetGaloisElement for GLWEAutomorphismKeyPrepared<D, B> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, B: Backend> SetGaloisElement for GLWEAutomorphismKeyPrepared<D, B> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWEAutomorphismKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWEAutomorphismKeyPrepared<D, B> {
    fn k_aux(&self) -> TorusPrecision {
        self.key.k_aux()
    }

    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.key.k_digit()
    }

    fn k_offset(&self) -> TorusPrecision {
        self.key.k_offset()
    }

    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }

    fn stride(&self) -> usize {
        self.key.stride()
    }
}

pub type GLWEAutomorphismKeyPreparedBackendRef<'a, B> = GLWEAutomorphismKeyPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWEAutomorphismKeyPreparedBackendMut<'a, B> = GLWEAutomorphismKeyPrepared<<B as Backend>::BufMut<'a>, B>;

impl<D: Data, B: Backend> GLWEAutomorphismKeyPrepared<D, B> {
    /// This key read through a coarser `k_digit`. See [`GGLWEPrepared::with_k_digit`].
    pub fn with_k_digit(&self, k_digit: TorusPrecision) -> Result<GLWEAutomorphismKeyPreparedBackendRef<'_, B>>
    where
        GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
    {
        Ok(GLWEAutomorphismKeyPrepared {
            key: self.key.with_k_digit(k_digit)?,
            p: self.p,
        })
    }
}

pub trait GLWEAutomorphismKeyPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWEAutomorphismKeyPreparedBackendRef<'_, B>;
}

impl<D: Data, B: Backend> GLWEAutomorphismKeyPreparedToBackendRef<B> for GLWEAutomorphismKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GLWEAutomorphismKeyPreparedBackendRef<'_, B> {
        GLWEAutomorphismKeyPrepared {
            key: self.key.to_backend_ref(),
            p: self.p,
        }
    }
}

impl<B: Backend> GLWEAutomorphismKeyPreparedToBackendRef<B> for &GLWEAutomorphismKeyPrepared<B::BufRef<'_>, B> {
    fn to_backend_ref(&self) -> GLWEAutomorphismKeyPreparedBackendRef<'_, B> {
        GLWEAutomorphismKeyPrepared {
            key: GGLWEPreparedToBackendRef::to_backend_ref(self),
            p: self.p,
        }
    }
}

impl<D: Data, B: Backend> GGLWEPreparedToBackendRef<B> for GLWEAutomorphismKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        self.key.to_backend_ref()
    }
}

impl<B: Backend> GGLWEPreparedToBackendRef<B> for &GLWEAutomorphismKeyPrepared<B::BufRef<'_>, B> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        GGLWEPrepared {
            noise: self.key.noise.clone(),
            base2k: self.key.base2k,
            k_aux: self.key.k_aux,
            k_digit: self.key.k_digit,
            k_offset: self.key.k_offset,
            dnum: self.key.dnum,
            stride: self.key.stride,
            data: vmp_pmat_backend_ref_from_ref::<B>(&self.key.data),
        }
    }
}

impl<D: Data, B: Backend> GGLWEPreparedToBackendMut<B> for GLWEAutomorphismKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        <_ as GGLWEPreparedToBackendMut<B>>::set_noise(&mut self.key, metadata);
    }

    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B> {
        self.key.to_backend_mut()
    }
}

pub trait GLWEAutomorphismKeyPreparedToBackendMut<B: Backend> {
    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GLWEAutomorphismKeyPreparedBackendMut<'_, B>;
}

impl<D: Data, B: Backend> GLWEAutomorphismKeyPreparedToBackendMut<B> for GLWEAutomorphismKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> GLWEAutomorphismKeyPreparedBackendMut<'_, B> {
        GLWEAutomorphismKeyPrepared {
            key: self.key.to_backend_mut(),
            p: self.p,
        }
    }
}
