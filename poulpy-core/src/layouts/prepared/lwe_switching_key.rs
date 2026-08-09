use poulpy_hal::layouts::VmpPMat;
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::layouts::{
    Base2K, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendRef, GGLWEPreparedToBackendMut,
    GGLWEPreparedToBackendRef, GGLWEToBackendRef, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, LWESwitchingKeyCore, Rank,
    TorusPrecision,
    prepared::{
        GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedFactory, GLWESwitchingKeyPreparedToBackendMut,
        GLWESwitchingKeyPreparedToBackendRef,
    },
};

/// DFT-domain (prepared) variant of an LWE switching key.
///
/// A newtype wrapper around [`GLWESwitchingKeyPrepared`] for LWE key-switching.
/// Tied to a specific backend via `B: Backend`.
/// DFT-domain (prepared) variant of a LWE key-switching key.
///
/// This is [`LWESwitchingKeyCore`] over a `VmpPMat` payload; the `Infos` traits and the
/// degree accessors come from the payload-generic impls there.
pub type LWESwitchingKeyPrepared<D, B> = LWESwitchingKeyCore<VmpPMat<D, <B as Backend>::DftWord, B>>;

pub trait LWESwitchingKeyPreparedFactory<B: Backend>
where
    Self: GLWESwitchingKeyPreparedFactory<B>,
{
    fn lwe_switching_key_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_aux: TorusPrecision,
    ) -> LWESwitchingKeyPrepared<B::OwnedBuf, B> {
        LWESwitchingKeyCore(self.glwe_switching_key_prepared_alloc(base2k, dnum, Dsize(1), k_aux, Rank(1), Rank(1)))
    }

    fn lwe_switching_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> LWESwitchingKeyPrepared<B::OwnedBuf, B>
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWESwitchingKey");
        debug_assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWESwitchingKey");
        debug_assert_eq!(infos.rank_out().0, 1, "rank_out > 1 is not supported for LWESwitchingKey");
        self.lwe_switching_key_prepared_alloc(infos.base2k(), infos.dnum(), infos.k_aux())
    }

    fn lwe_switching_key_prepared_bytes_of(&self, base2k: Base2K, dnum: Dnum, k_aux: TorusPrecision) -> usize {
        self.bytes_of_glwe_key_prepared(base2k, dnum, Dsize(1), k_aux, Rank(1), Rank(1))
    }

    fn lwe_switching_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWESwitchingKey");
        debug_assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWESwitchingKey");
        debug_assert_eq!(infos.rank_out().0, 1, "rank_out > 1 is not supported for LWESwitchingKey");
        self.lwe_switching_key_prepared_bytes_of(infos.base2k(), infos.dnum(), infos.k_aux())
    }

    fn lwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.glwe_switching_key_prepare_tmp_bytes(infos);
        lvl_0
    }
    fn lwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, B>)
    where
        R: GGLWEPreparedToBackendMut<B> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<B> + GLWESwitchingKeyDegrees,
    {
        let tmp_bytes = {
            let res_infos = res.to_backend_mut();
            self.lwe_switching_key_prepare_tmp_bytes(&res_infos)
        };
        assert!(
            scratch.available() >= tmp_bytes,
            "scratch.available(): {} < LWESwitchingKeyPreparedFactory::lwe_switching_key_prepare_tmp_bytes: {}",
            scratch.available(),
            tmp_bytes
        );
        self.glwe_switching_key_prepare(res, other, scratch);
    }
}

impl<B: Backend> LWESwitchingKeyPreparedFactory<B> for Module<B> where Self: GLWESwitchingKeyPreparedFactory<B> {}

// module-only API: allocation, sizing, and preparation are provided by
// `LWESwitchingKeyPreparedFactory` on `Module`.

impl<D: Data, B: Backend> GGLWEPreparedToBackendMut<B> for LWESwitchingKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> crate::layouts::GGLWEPreparedBackendMut<'_, B> {
        self.0.key.to_backend_mut()
    }
}

pub type LWESwitchingKeyPreparedBackendRef<'a, B> = LWESwitchingKeyPrepared<<B as Backend>::BufRef<'a>, B>;
pub type LWESwitchingKeyPreparedBackendMut<'a, B> = LWESwitchingKeyPrepared<<B as Backend>::BufMut<'a>, B>;

pub trait LWESwitchingKeyPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> LWESwitchingKeyPreparedBackendRef<'_, B>;
}

impl<D: Data, B: Backend> LWESwitchingKeyPreparedToBackendRef<B> for LWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> LWESwitchingKeyPreparedBackendRef<'_, B> {
        LWESwitchingKeyCore(self.0.to_backend_ref())
    }
}

impl<D: Data, B: Backend> GGLWEPreparedToBackendRef<B> for LWESwitchingKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        self.0.key.to_backend_ref()
    }
}

pub trait LWESwitchingKeyPreparedToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> LWESwitchingKeyPreparedBackendMut<'_, B>;
}

impl<D: Data, B: Backend> LWESwitchingKeyPreparedToBackendMut<B> for LWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> LWESwitchingKeyPreparedBackendMut<'_, B> {
        LWESwitchingKeyCore(self.0.to_backend_mut())
    }
}
