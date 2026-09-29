use std::marker::PhantomData;

use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::layouts::{
    Base2K, CIEmbed, CITrace, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendMut, GGLWEPreparedBackendRef,
    GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef, GGLWEToBackendRef, GLWECIKey, GLWEInfos, GLWESwitchingKey,
    GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, LWEInfos, Rank, TorusPrecision,
    prepared::{GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedFactory},
};

/// DFT-domain (prepared) [`GLWECIKey`].
#[derive(PartialEq)]
pub struct GLWECIKeyPrepared<D: Data, B: Backend, M>(pub(crate) GLWESwitchingKeyPrepared<D, B>, pub(crate) PhantomData<M>);

/// Prepared [`GLWECIEmbedKey`](crate::layouts::GLWECIEmbedKey).
pub type GLWECIEmbedKeyPrepared<D, B> = GLWECIKeyPrepared<D, B, CIEmbed>;

/// Prepared [`GLWECITraceKey`](crate::layouts::GLWECITraceKey).
pub type GLWECITraceKeyPrepared<D, B> = GLWECIKeyPrepared<D, B, CITrace>;

impl<D: Data, B: Backend, M> LWEInfos for GLWECIKeyPrepared<D, B, M> {
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

    fn dsize(&self) -> Dsize {
        self.0.dsize()
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

pub trait GLWECIKeyPreparedFactory<B: Backend>
where
    Self: GLWESwitchingKeyPreparedFactory<B>,
{
    fn glwe_ci_embed_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWECIEmbedKeyPrepared<B::OwnedBuf, B>
    where
        A: GGLWEInfos,
    {
        GLWECIKeyPrepared(self.glwe_switching_key_prepared_alloc_from_infos(infos), PhantomData)
    }

    fn glwe_ci_trace_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWECITraceKeyPrepared<B::OwnedBuf, B>
    where
        A: GGLWEInfos,
    {
        GLWECIKeyPrepared(self.glwe_switching_key_prepared_alloc_from_infos(infos), PhantomData)
    }

    fn glwe_ci_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.glwe_switching_key_prepared_bytes_of_from_infos(infos)
    }

    fn glwe_ci_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.glwe_switching_key_prepare_tmp_bytes(infos)
    }

    /// Prepares `other` into `res`; both carry the same direction.
    fn glwe_ci_key_prepare<DR, DO, M>(
        &self,
        res: &mut GLWECIKeyPrepared<DR, B, M>,
        other: &GLWECIKey<DO, B::ZnxWord, M>,
        scratch: &mut ScratchArena<'_, B>,
    ) where
        DR: Data,
        DO: Data,
        GLWESwitchingKeyPrepared<DR, B>: GGLWEPreparedToBackendMut<B>,
        GLWESwitchingKey<DO, B::ZnxWord>: GGLWEToBackendRef<B>,
    {
        self.glwe_switching_key_prepare(&mut res.0, &other.0, scratch);
    }
}

impl<B: Backend> GLWECIKeyPreparedFactory<B> for Module<B> where Self: GLWESwitchingKeyPreparedFactory<B> {}
