use poulpy_hal::layouts::{Backend, Data};

use crate::layouts::{
    Base2K, Degree, Dnum, GGLWEInfos, GGLWEPreparedBackendMut, GGLWEPreparedBackendRef, GLWEInfos, GLWESwitchingKeyDegrees,
    GLWESwitchingKeyDegreesMut, LWEInfos, Rank, TorusPrecision,
    prepared::{GGLWEPrepared, GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef},
};

/// DFT-domain (prepared) variant of a GLWE switching key.
///
/// Wraps a [`GGLWEPrepared`] with input/output degree metadata for
/// key-switching between GLWE ciphertexts. Tied to a specific backend
/// via `B: Backend`.
#[derive(PartialEq)]
pub struct GLWESwitchingKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GGLWEPrepared<D, B>,
    pub(crate) input_degree: Degree,  // Degree of sk_in
    pub(crate) output_degree: Degree, // Degree of sk_out
}

impl<D: Data, BE: Backend> GLWESwitchingKeyDegrees for GLWESwitchingKeyPrepared<D, BE> {
    fn output_degree(&self) -> &Degree {
        &self.output_degree
    }

    fn input_degree(&self) -> &Degree {
        &self.input_degree
    }
}

impl<D: Data, BE: Backend> GLWESwitchingKeyDegreesMut for GLWESwitchingKeyPrepared<D, BE> {
    fn output_degree(&mut self) -> &mut Degree {
        &mut self.output_degree
    }

    fn input_degree(&mut self) -> &mut Degree {
        &mut self.input_degree
    }
}

impl<D: Data, B: Backend> LWEInfos for GLWESwitchingKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GLWEInfos for GLWESwitchingKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWESwitchingKeyPrepared<D, B> {
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

pub type GLWESwitchingKeyPreparedBackendRef<'a, B> = GLWESwitchingKeyPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWESwitchingKeyPreparedBackendMut<'a, B> = GLWESwitchingKeyPrepared<<B as Backend>::BufMut<'a>, B>;

pub trait GLWESwitchingKeyPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWESwitchingKeyPreparedBackendRef<'_, B>;
}

impl<D: Data, B: Backend> GLWESwitchingKeyPreparedToBackendRef<B> for GLWESwitchingKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GLWESwitchingKeyPreparedBackendRef<'_, B> {
        GLWESwitchingKeyPrepared {
            key: self.key.to_backend_ref(),
            input_degree: self.input_degree,
            output_degree: self.output_degree,
        }
    }
}

impl<D: Data, B: Backend> GGLWEPreparedToBackendRef<B> for GLWESwitchingKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        self.key.to_backend_ref()
    }
}

impl<D: Data, B: Backend> GGLWEPreparedToBackendMut<B> for GLWESwitchingKeyPrepared<D, B>
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

pub trait GLWESwitchingKeyPreparedToBackendMut<B: Backend> {
    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GLWESwitchingKeyPreparedBackendMut<'_, B>;
}

impl<D: Data, B: Backend> GLWESwitchingKeyPreparedToBackendMut<B> for GLWESwitchingKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> GLWESwitchingKeyPreparedBackendMut<'_, B> {
        GLWESwitchingKeyPrepared {
            key: self.key.to_backend_mut(),
            input_degree: self.input_degree,
            output_degree: self.output_degree,
        }
    }
}
