use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, Module, ScratchArena},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendRef, GGLWEPreparedToBackendMut,
    GGLWEPreparedToBackendRef, GGLWEToBackendRef, GLWEInfos, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, LWEInfos, Rank,
    TorusPrecision,
    prepared::{
        GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedFactory, GLWESwitchingKeyPreparedToBackendMut,
        GLWESwitchingKeyPreparedToBackendRef,
    },
};

/// A special `GLWESwitchingKey` required for the conversion from `LWE` to `GLWE`.
#[derive(PartialEq, Eq)]
pub struct LWEToGLWEKeyPrepared<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for LWEToGLWEKeyPrepared<D, B> {
    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn n(&self) -> Degree {
        self.0.n()
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<D: Data, B: Backend> GLWEInfos for LWEToGLWEKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for LWEToGLWEKeyPrepared<D, B> {
    fn dsize(&self) -> Dsize {
        self.0.dsize()
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
}

pub trait LWEToGLWEKeyPreparedFactory<B: Backend>
where
    Self: GLWESwitchingKeyPreparedFactory<B>,
{
    fn lwe_to_glwe_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_out: Rank,
        dnum: Dnum,
    ) -> LWEToGLWEKeyPrepared<B::OwnedBuf, B> {
        LWEToGLWEKeyPrepared(self.glwe_switching_key_prepared_alloc(base2k, k, Rank(1), rank_out, dnum, Dsize(1)))
    }
    fn lwe_to_glwe_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> LWEToGLWEKeyPrepared<B::OwnedBuf, B>
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWEToGLWEKey");
        debug_assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWEToGLWEKey");
        self.lwe_to_glwe_key_prepared_alloc(infos.base2k(), infos.max_k(), infos.rank_out(), infos.dnum())
    }

    fn lwe_to_glwe_key_prepared_bytes_of(&self, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> usize {
        self.bytes_of_glwe_key_prepared(base2k, k, Rank(1), rank_out, dnum, Dsize(1))
    }

    fn lwe_to_glwe_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWEToGLWEKey");
        debug_assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWEToGLWEKey");
        self.lwe_to_glwe_key_prepared_bytes_of(infos.base2k(), infos.max_k(), infos.rank_out(), infos.dnum())
    }

    fn lwe_to_glwe_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.glwe_switching_key_prepare_tmp_bytes(infos);
        lvl_0
    }

    fn lwe_to_glwe_key_prepare<'s, R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'s, B>)
    where
        R: GGLWEPreparedToBackendMut<B> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<B> + GLWESwitchingKeyDegrees,
        ScratchArena<'s, B>: ScratchAvailable,
        B: 's,
    {
        let tmp_bytes = {
            let res_infos = res.to_backend_mut();
            self.lwe_to_glwe_key_prepare_tmp_bytes(&res_infos)
        };
        assert!(
            scratch.available() >= tmp_bytes,
            "scratch.available(): {} < LWEToGLWEKeyPreparedFactory::lwe_to_glwe_key_prepare_tmp_bytes: {}",
            scratch.available(),
            tmp_bytes
        );
        self.glwe_switching_key_prepare(res, other, scratch);
    }
}

impl<B: Backend> LWEToGLWEKeyPreparedFactory<B> for Module<B> where Self: GLWESwitchingKeyPreparedFactory<B> {}

// module-only API: allocation, sizing, and preparation are provided by
// `LWEToGLWEKeyPreparedFactory` on `Module`.

impl<D: Data, B: Backend> GGLWEPreparedToBackendMut<B> for LWEToGLWEKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
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
