use std::marker::PhantomData;

use poulpy_hal::layouts::{Backend, Data, Ring, Standard};

use crate::layouts::{
    Base2K, CIEmbed, CITrace, Degree, Dnum, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendMut, GGLWEPreparedBackendRef,
    GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef, GLWEInfos, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut,
    LWEInfos, Rank, TorusPrecision, prepared::GLWESwitchingKeyPrepared,
};

/// DFT-domain (prepared) [`GLWECIKey`](crate::layouts::GLWECIKey).
#[derive(PartialEq)]
pub struct GLWECIKeyPrepared<D: Data, B: Backend, M, Rg: Ring = Standard>(
    pub(crate) GLWESwitchingKeyPrepared<D, B, Rg>,
    pub(crate) PhantomData<M>,
);

/// Prepared [`GLWECIEmbedKey`](crate::layouts::GLWECIEmbedKey).
pub type GLWECIEmbedKeyPrepared<D, B, Rg = Standard> = GLWECIKeyPrepared<D, B, CIEmbed, Rg>;

/// Prepared [`GLWECITraceKey`](crate::layouts::GLWECITraceKey).
pub type GLWECITraceKeyPrepared<D, B, Rg = Standard> = GLWECIKeyPrepared<D, B, CITrace, Rg>;

impl<D: Data, B: Backend, M, Rg: Ring> LWEInfos for GLWECIKeyPrepared<D, B, M, Rg> {
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

impl<D: Data, B: Backend, M, Rg: Ring> GLWEInfos for GLWECIKeyPrepared<D, B, M, Rg> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend, M, Rg: Ring> GGLWEInfos for GLWECIKeyPrepared<D, B, M, Rg> {
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

impl<D: Data, B: Backend, M, Rg: Ring> GGLWEPreparedToBackendRef<B, Rg> for GLWECIKeyPrepared<D, B, M, Rg>
where
    GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendRef<B, Rg>,
{
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B, Rg> {
        self.0.key.to_backend_ref()
    }
}

impl<D: Data, B: Backend, M, Rg: Ring> GGLWEPreparedToBackendMut<B, Rg> for GLWECIKeyPrepared<D, B, M, Rg>
where
    GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendMut<B, Rg>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        <_ as GGLWEPreparedToBackendMut<B, Rg>>::set_noise(&mut self.0.key, metadata);
    }

    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B, Rg> {
        self.0.key.to_backend_mut()
    }
}

impl<D: Data, B: Backend, M, Rg: Ring> GLWESwitchingKeyDegreesMut for GLWECIKeyPrepared<D, B, M, Rg> {
    fn input_degree(&mut self) -> &mut Degree {
        &mut self.0.input_degree
    }

    fn output_degree(&mut self) -> &mut Degree {
        &mut self.0.output_degree
    }
}

impl<D: Data, B: Backend, M, Rg: Ring> GLWESwitchingKeyDegrees for GLWECIKeyPrepared<D, B, M, Rg> {
    fn input_degree(&self) -> &Degree {
        &self.0.input_degree
    }

    fn output_degree(&self) -> &Degree {
        &self.0.output_degree
    }
}
