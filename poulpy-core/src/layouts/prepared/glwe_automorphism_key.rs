use crate::error::Result;
use poulpy_hal::layouts::{Backend, Data, Ring, Standard};

use crate::layouts::prepared::{GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef};
use crate::layouts::{
    Base2K, Degree, Dnum, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendMut, GGLWEPreparedBackendRef, GLWEInfos,
    GetGaloisElement, LWEInfos, Rank, SetGaloisElement, TorusPrecision,
};
use poulpy_hal::layouts::vmp_pmat_backend_ref_from_ref;

#[derive(PartialEq)]
pub struct GLWEAutomorphismKeyPrepared<D: Data, B: Backend, Rg: Ring = Standard> {
    pub(crate) key: GGLWEPrepared<D, B, Rg>,
    pub(crate) p: i64,
}

impl<D: Data, B: Backend, Rg: Ring> LWEInfos for GLWEAutomorphismKeyPrepared<D, B, Rg> {
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

impl<D: Data, B: Backend, Rg: Ring> GetGaloisElement for GLWEAutomorphismKeyPrepared<D, B, Rg> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, B: Backend, Rg: Ring> SetGaloisElement for GLWEAutomorphismKeyPrepared<D, B, Rg> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<D: Data, B: Backend, Rg: Ring> GLWEInfos for GLWEAutomorphismKeyPrepared<D, B, Rg> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend, Rg: Ring> GGLWEInfos for GLWEAutomorphismKeyPrepared<D, B, Rg> {
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

pub type GLWEAutomorphismKeyPreparedBackendRef<'a, B, Rg = Standard> =
    GLWEAutomorphismKeyPrepared<<B as Backend>::BufRef<'a>, B, Rg>;
pub type GLWEAutomorphismKeyPreparedBackendMut<'a, B, Rg = Standard> =
    GLWEAutomorphismKeyPrepared<<B as Backend>::BufMut<'a>, B, Rg>;

impl<D: Data, B: Backend, Rg: Ring> GLWEAutomorphismKeyPrepared<D, B, Rg> {
    /// This key read through a coarser `k_digit`. See [`GGLWEPrepared::with_k_digit`].
    pub fn with_k_digit(&self, k_digit: TorusPrecision) -> Result<GLWEAutomorphismKeyPreparedBackendRef<'_, B, Rg>>
    where
        GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendRef<B, Rg>,
    {
        Ok(GLWEAutomorphismKeyPrepared {
            key: self.key.with_k_digit(k_digit)?,
            p: self.p,
        })
    }
}

pub trait GLWEAutomorphismKeyPreparedToBackendRef<B: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> GLWEAutomorphismKeyPreparedBackendRef<'_, B, Rg>;
}

impl<D: Data, B: Backend, Rg: Ring> GLWEAutomorphismKeyPreparedToBackendRef<B, Rg> for GLWEAutomorphismKeyPrepared<D, B, Rg>
where
    GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendRef<B, Rg>,
{
    fn to_backend_ref(&self) -> GLWEAutomorphismKeyPreparedBackendRef<'_, B, Rg> {
        GLWEAutomorphismKeyPrepared {
            key: self.key.to_backend_ref(),
            p: self.p,
        }
    }
}

impl<B: Backend, Rg: Ring> GLWEAutomorphismKeyPreparedToBackendRef<B, Rg> for &GLWEAutomorphismKeyPrepared<B::BufRef<'_>, B, Rg> {
    fn to_backend_ref(&self) -> GLWEAutomorphismKeyPreparedBackendRef<'_, B, Rg> {
        GLWEAutomorphismKeyPrepared {
            key: GGLWEPreparedToBackendRef::to_backend_ref(self),
            p: self.p,
        }
    }
}

impl<D: Data, B: Backend, Rg: Ring> GGLWEPreparedToBackendRef<B, Rg> for GLWEAutomorphismKeyPrepared<D, B, Rg>
where
    GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendRef<B, Rg>,
{
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B, Rg> {
        self.key.to_backend_ref()
    }
}

impl<B: Backend, Rg: Ring> GGLWEPreparedToBackendRef<B, Rg> for &GLWEAutomorphismKeyPrepared<B::BufRef<'_>, B, Rg> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B, Rg> {
        GGLWEPrepared {
            noise: self.key.noise.clone(),
            base2k: self.key.base2k,
            k_aux: self.key.k_aux,
            k_digit: self.key.k_digit,
            k_offset: self.key.k_offset,
            dnum: self.key.dnum,
            stride: self.key.stride,
            data: vmp_pmat_backend_ref_from_ref::<B, Rg>(&self.key.data),
        }
    }
}

impl<D: Data, B: Backend, Rg: Ring> GGLWEPreparedToBackendMut<B, Rg> for GLWEAutomorphismKeyPrepared<D, B, Rg>
where
    GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendMut<B, Rg>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        <_ as GGLWEPreparedToBackendMut<B, Rg>>::set_noise(&mut self.key, metadata);
    }

    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B, Rg> {
        self.key.to_backend_mut()
    }
}

pub trait GLWEAutomorphismKeyPreparedToBackendMut<B: Backend, Rg: Ring = Standard> {
    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GLWEAutomorphismKeyPreparedBackendMut<'_, B, Rg>;
}

impl<D: Data, B: Backend, Rg: Ring> GLWEAutomorphismKeyPreparedToBackendMut<B, Rg> for GLWEAutomorphismKeyPrepared<D, B, Rg>
where
    GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendMut<B, Rg>,
{
    fn to_backend_mut(&mut self) -> GLWEAutomorphismKeyPreparedBackendMut<'_, B, Rg> {
        GLWEAutomorphismKeyPrepared {
            key: self.key.to_backend_mut(),
            p: self.p,
        }
    }
}
