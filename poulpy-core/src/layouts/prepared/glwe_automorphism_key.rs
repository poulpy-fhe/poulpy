use std::collections::HashMap;

use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena, VmpPMat, VmpPMatBackendRef};

use crate::layouts::prepared::{GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef, GGLWEPreparedVmpPMatRef};
use crate::layouts::{
    Base2K, Dnum, Dsize, GGLWEInfos, GGLWELayout, GGLWEPrepared, GGLWEPreparedBackendMut, GGLWEPreparedBackendRef,
    GGLWEPreparedFactory, GGLWEToBackendRef, GLWEAutomorphismKeyCore, GLWEAutomorphismKeyHelper, GetGaloisElement, Rank,
    SetGaloisElement, TorusPrecision,
};

impl<K, BE: Backend> GLWEAutomorphismKeyHelper<K, BE> for HashMap<i64, K>
where
    K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
{
    fn get_automorphism_key(&self, k: i64) -> Option<&K> {
        self.get(&k)
    }

    fn automorphism_key_infos(&self) -> GGLWELayout {
        let first_key = self.keys().min().copied().expect("automorphism key map is empty");
        self.get(&first_key).unwrap().gglwe_layout()
    }
}

/// DFT-domain (prepared) variant of a GLWE automorphism key.
///
/// This is [`GLWEAutomorphismKeyCore`] over a `VmpPMat` payload; the `Infos`
/// traits and the Galois-element accessors come from the payload-generic impls
/// there.
pub type GLWEAutomorphismKeyPrepared<D, B> = GLWEAutomorphismKeyCore<VmpPMat<D, <B as Backend>::DftWord, B>>;

pub trait GLWEAutomorphismKeyPreparedFactory<B: Backend>
where
    Self: GGLWEPreparedFactory<B>,
{
    fn glwe_automorphism_key_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> GLWEAutomorphismKeyPrepared<B::OwnedBuf, B> {
        GLWEAutomorphismKeyPrepared::<B::OwnedBuf, B> {
            key: self.gglwe_prepared_alloc(base2k, dnum, dsize, k_aux, rank, rank),
            p: 0,
        }
    }

    fn glwe_automorphism_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWEAutomorphismKeyPrepared<B::OwnedBuf, B>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for AutomorphismKeyPrepared"
        );
        self.glwe_automorphism_key_prepared_alloc(infos.base2k(), infos.dnum(), infos.dsize(), infos.k_aux(), infos.rank())
    }

    fn glwe_automorphism_key_prepared_bytes_of(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> usize {
        self.gglwe_prepared_bytes_of(base2k, dnum, dsize, k_aux, rank, rank)
    }

    fn glwe_automorphism_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for AutomorphismKeyPrepared"
        );
        self.glwe_automorphism_key_prepared_bytes_of(infos.base2k(), infos.dnum(), infos.dsize(), infos.k_aux(), infos.rank())
    }

    fn glwe_automorphism_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.gglwe_prepare_tmp_bytes(infos);
        lvl_0
    }

    fn glwe_automorphism_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, B>)
    where
        R: GGLWEPreparedToBackendMut<B> + SetGaloisElement,
        O: GGLWEToBackendRef<B> + GetGaloisElement,
    {
        let tmp_bytes = {
            let res_infos = res.to_backend_mut();
            self.glwe_automorphism_key_prepare_tmp_bytes(&res_infos)
        };
        assert!(
            scratch.available() >= tmp_bytes,
            "scratch.available(): {} < GLWEAutomorphismKeyPreparedFactory::glwe_automorphism_key_prepare_tmp_bytes: {}",
            scratch.available(),
            tmp_bytes
        );
        self.gglwe_prepare(res, other, scratch);
        res.set_p(other.p());
    }
}

impl<B: Backend> GLWEAutomorphismKeyPreparedFactory<B> for Module<B> where Module<B>: GGLWEPreparedFactory<B> {}

// module-only API: allocation, sizing, and preparation are provided by
// `GLWEAutomorphismKeyPreparedFactory` on `Module`.

pub type GLWEAutomorphismKeyPreparedBackendRef<'a, B> = GLWEAutomorphismKeyPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWEAutomorphismKeyPreparedBackendMut<'a, B> = GLWEAutomorphismKeyPrepared<<B as Backend>::BufMut<'a>, B>;

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

impl<D: Data, B: Backend> GGLWEPreparedToBackendRef<B> for GLWEAutomorphismKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        self.key.to_backend_ref()
    }
}

impl<D: Data, B: Backend> GGLWEPreparedVmpPMatRef<B> for GLWEAutomorphismKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedVmpPMatRef<B>,
{
    fn vmp_pmat_backend_ref(&self) -> VmpPMatBackendRef<'_, B> {
        self.key.vmp_pmat_backend_ref()
    }
}

impl<D: Data, B: Backend> GGLWEPreparedToBackendMut<B> for GLWEAutomorphismKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B> {
        self.key.to_backend_mut()
    }
}

pub trait GLWEAutomorphismKeyPreparedToBackendMut<B: Backend> {
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
