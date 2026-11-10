use poulpy_hal::layouts::{Backend, Data};

use crate::layouts::{
    Base2K, Degree, Dnum, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendRef, GGLWEPreparedToBackendMut,
    GGLWEPreparedToBackendRef, GLWEInfos, GLWESwitchingKeyDegreesMut, LWEInfos, Rank, TorusPrecision,
    prepared::{GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedToBackendMut, GLWESwitchingKeyPreparedToBackendRef},
};

/// A special `GLWESwitchingKey` required for the conversion from `LWE` to `GLWE`.
#[derive(PartialEq)]
pub struct LWEToGLWEKeyPrepared<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for LWEToGLWEKeyPrepared<D, B> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.0.noise()
    }

    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn n(&self) -> Degree {
        self.0.n()
    }

    fn max_size(&self) -> usize {
        self.0.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.0.k()
    }
}

impl<D: Data, B: Backend> GLWEInfos for LWEToGLWEKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for LWEToGLWEKeyPrepared<D, B> {
    fn k_aux(&self) -> TorusPrecision {
        self.0.k_aux()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.0.k_digit()
    }

    fn k_offset(&self) -> TorusPrecision {
        self.0.k_offset()
    }

    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }

    fn stride(&self) -> usize {
        self.0.stride()
    }
}

impl<D: Data, B: Backend> GGLWEPreparedToBackendMut<B> for LWEToGLWEKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        <_ as GGLWEPreparedToBackendMut<B>>::set_noise(&mut self.0.key, metadata);
    }

    fn to_backend_mut(&mut self) -> crate::layouts::GGLWEPreparedBackendMut<'_, B> {
        self.0.key.to_backend_mut()
    }
}

impl<D: Data, B: Backend> GLWESwitchingKeyDegreesMut for LWEToGLWEKeyPrepared<D, B> {
    fn input_degree(&mut self) -> &mut Degree {
        &mut self.0.input_degree
    }

    fn output_degree(&mut self) -> &mut Degree {
        &mut self.0.output_degree
    }
}

pub type LWEToGLWEKeyPreparedBackendRef<'a, B> = LWEToGLWEKeyPrepared<<B as Backend>::BufRef<'a>, B>;
pub type LWEToGLWEKeyPreparedBackendMut<'a, B> = LWEToGLWEKeyPrepared<<B as Backend>::BufMut<'a>, B>;

pub trait LWEToGLWEKeyPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> LWEToGLWEKeyPreparedBackendRef<'_, B>;
}

impl<D: Data, B: Backend> LWEToGLWEKeyPreparedToBackendRef<B> for LWEToGLWEKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> LWEToGLWEKeyPreparedBackendRef<'_, B> {
        LWEToGLWEKeyPrepared(self.0.to_backend_ref())
    }
}

impl<D: Data, B: Backend> GGLWEPreparedToBackendRef<B> for LWEToGLWEKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        self.0.key.to_backend_ref()
    }
}

pub trait LWEToGLWEKeyPreparedToBackendMut<B: Backend> {
    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> LWEToGLWEKeyPreparedBackendMut<'_, B>;
}

impl<D: Data, B: Backend> LWEToGLWEKeyPreparedToBackendMut<B> for LWEToGLWEKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> LWEToGLWEKeyPreparedBackendMut<'_, B> {
        LWEToGLWEKeyPrepared(self.0.to_backend_mut())
    }
}
