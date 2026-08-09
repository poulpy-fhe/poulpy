use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena, VmpPMat};

use crate::layouts::prepared::{GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef};
use crate::layouts::{
    Base2K, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendMut, GGLWEPreparedFactory, GGLWEToBackendRef,
    GLWETensorKeyCore, Rank, TorusPrecision,
};

/// DFT-domain (prepared) variant of a GLWE tensor key.
///
/// This is [`GLWETensorKeyCore`] over a `VmpPMat` payload; the `Infos` traits
/// come from the payload-generic impls there.
pub type GLWETensorKeyPrepared<D, B> = GLWETensorKeyCore<VmpPMat<D, <B as Backend>::DftWord, B>>;

pub trait GLWETensorKeyPreparedFactory<B: Backend>
where
    Self: GGLWEPreparedFactory<B>,
{
    fn alloc_tensor_key_prepared(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> GLWETensorKeyPrepared<B::OwnedBuf, B> {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        GLWETensorKeyCore(self.gglwe_prepared_alloc(base2k, dnum, dsize, k_aux, Rank(pairs), rank))
    }

    fn alloc_tensor_key_prepared_from_infos<A>(&self, infos: &A) -> GLWETensorKeyPrepared<B::OwnedBuf, B>
    where
        A: GGLWEInfos,
    {
        self.alloc_tensor_key_prepared(infos.base2k(), infos.dnum(), infos.dsize(), infos.k_aux(), infos.rank_out())
    }

    fn bytes_of_tensor_key_prepared(&self, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> usize {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        self.gglwe_prepared_bytes_of(base2k, dnum, dsize, k_aux, Rank(pairs), rank)
    }

    fn bytes_of_tensor_key_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.bytes_of_tensor_key_prepared(infos.base2k(), infos.dnum(), infos.dsize(), infos.k_aux(), infos.rank())
    }

    fn prepare_tensor_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.gglwe_prepare_tmp_bytes(infos);
        lvl_0
    }

    fn prepare_tensor_key<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, B>)
    where
        R: GGLWEPreparedToBackendMut<B>,
        O: GGLWEToBackendRef<B>,
    {
        let tmp_bytes = {
            let res_infos = res.to_backend_mut();
            self.prepare_tensor_key_tmp_bytes(&res_infos)
        };
        assert!(
            scratch.available() >= tmp_bytes,
            "scratch.available(): {} < GLWETensorKeyPreparedFactory::prepare_tensor_key_tmp_bytes: {}",
            scratch.available(),
            tmp_bytes
        );
        self.gglwe_prepare(res, other, scratch);
    }
}

impl<B: Backend> GLWETensorKeyPreparedFactory<B> for Module<B> where Module<B>: GGLWEPreparedFactory<B> {}

// module-only API: allocation/size helpers are provided by `GLWETensorKeyPreparedFactory` on `Module`.

// module-only API: preparation sizing is provided by `GLWETensorKeyPreparedFactory` on `Module`.

// module-only API: preparation is provided by `GLWETensorKeyPreparedFactory` on `Module`.

pub type GLWETensorKeyPreparedBackendRef<'a, B> = GLWETensorKeyPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWETensorKeyPreparedBackendMut<'a, B> = GLWETensorKeyPrepared<<B as Backend>::BufMut<'a>, B>;

pub trait GLWETensorKeyPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWETensorKeyPreparedBackendRef<'_, B>;
}

impl<D: Data, B: Backend> GLWETensorKeyPreparedToBackendRef<B> for GLWETensorKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GLWETensorKeyPreparedBackendRef<'_, B> {
        GLWETensorKeyCore(self.0.to_backend_ref())
    }
}

pub trait GLWETensorKeyPreparedToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> GLWETensorKeyPreparedBackendMut<'_, B>;
}

impl<D: Data, B: Backend> GLWETensorKeyPreparedToBackendMut<B> for GLWETensorKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> GLWETensorKeyPreparedBackendMut<'_, B> {
        GLWETensorKeyCore(self.0.to_backend_mut())
    }
}

impl<D: Data, B: Backend> GGLWEPreparedToBackendMut<B> for GLWETensorKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B> {
        self.0.to_backend_mut()
    }
}
