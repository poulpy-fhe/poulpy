use std::marker::PhantomData;

use poulpy_hal::layouts::{Backend, Data};

use crate::layouts::{
    Base2K, CIEmbed, CITrace, Degree, Dnum, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendMut, GGLWEPreparedBackendRef,
    GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef, GLWEInfos, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut,
    LWEInfos, Rank, TorusPrecision, prepared::GLWESwitchingKeyPrepared,
};

/// DFT-domain (prepared) [`GLWECIKey`](crate::layouts::GLWECIKey).
#[derive(PartialEq)]
pub struct GLWECIKeyPrepared<D: Data, B: Backend, M>(pub(crate) GLWESwitchingKeyPrepared<D, B>, pub(crate) PhantomData<M>);

/// Prepared [`GLWECIEmbedKey`](crate::layouts::GLWECIEmbedKey).
pub type GLWECIEmbedKeyPrepared<D, B> = GLWECIKeyPrepared<D, B, CIEmbed>;

/// Prepared [`GLWECITraceKey`](crate::layouts::GLWECITraceKey).
pub type GLWECITraceKeyPrepared<D, B> = GLWECIKeyPrepared<D, B, CITrace>;

impl<D: Data, B: Backend, M> LWEInfos for GLWECIKeyPrepared<D, B, M> {
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

impl<D: Data, B: Backend, M> GLWEInfos for GLWECIKeyPrepared<D, B, M> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend, M> GGLWEInfos for GLWECIKeyPrepared<D, B, M> {
    fn k_aux(&self) -> TorusPrecision {
        self.0.k_aux()
    }

    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.0.k_digit()
    }

    fn k_offset(&self) -> TorusPrecision {
        self.0.k_offset()
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

impl<D: Data, B: Backend, M> GGLWEPreparedToBackendRef<B> for GLWECIKeyPrepared<D, B, M>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        self.0.key.to_backend_ref()
    }
}

impl<D: Data, B: Backend, M> GGLWEPreparedToBackendMut<B> for GLWECIKeyPrepared<D, B, M>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        <_ as GGLWEPreparedToBackendMut<B>>::set_noise(&mut self.0.key, metadata);
    }

    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B> {
        self.0.key.to_backend_mut()
    }
}

impl<D: Data, B: Backend, M> GLWESwitchingKeyDegreesMut for GLWECIKeyPrepared<D, B, M> {
    fn input_degree(&mut self) -> &mut Degree {
        &mut self.0.input_degree
    }

    fn output_degree(&mut self) -> &mut Degree {
        &mut self.0.output_degree
    }
}

impl<D: Data, B: Backend, M> GLWESwitchingKeyDegrees for GLWECIKeyPrepared<D, B, M> {
    fn input_degree(&self) -> &Degree {
        &self.0.input_degree
    }

    fn output_degree(&self) -> &Degree {
        &self.0.output_degree
    }
}
