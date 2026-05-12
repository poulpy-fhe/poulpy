use std::collections::HashMap;

use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, Module, ScratchArena, vmp_pmat_backend_ref_from_ref},
};

use crate::layouts::prepared::{GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef};
use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWELayout, GGLWEPrepared, GGLWEPreparedBackendMut, GGLWEPreparedBackendRef,
    GGLWEPreparedFactory, GGLWEToBackendRef, GLWEAutomorphismKeyHelper, GLWEInfos, GetGaloisElement, LWEInfos, Rank,
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

#[derive(PartialEq, Eq)]
pub struct GLWEAutomorphismKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GGLWEPrepared<D, B>,
    pub(crate) p: i64,
}

impl<D: Data, B: Backend> LWEInfos for GLWEAutomorphismKeyPrepared<D, B> {
    fn n(&self) -> Degree {
        self.key.n()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn size(&self) -> usize {
        self.key.size()
    }
}

impl<D: Data, B: Backend> LWEInfos for &GLWEAutomorphismKeyPrepared<D, B> {
    fn n(&self) -> Degree {
        self.key.n()
    }
    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }
    fn size(&self) -> usize {
        self.key.size()
    }
}

impl<D: Data, B: Backend> LWEInfos for &mut GLWEAutomorphismKeyPrepared<D, B> {
    fn n(&self) -> Degree {
        self.key.n()
    }
    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }
    fn size(&self) -> usize {
        self.key.size()
    }
}

impl<D: Data, B: Backend> GetGaloisElement for GLWEAutomorphismKeyPrepared<D, B> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, B: Backend> GetGaloisElement for &GLWEAutomorphismKeyPrepared<D, B> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, B: Backend> GetGaloisElement for &mut GLWEAutomorphismKeyPrepared<D, B> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, B: Backend> SetGaloisElement for GLWEAutomorphismKeyPrepared<D, B> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWEAutomorphismKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GLWEInfos for &GLWEAutomorphismKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GLWEInfos for &mut GLWEAutomorphismKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWEAutomorphismKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.key.dsize()
    }

    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for &GLWEAutomorphismKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }
    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }
    fn dsize(&self) -> Dsize {
        self.key.dsize()
    }
    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for &mut GLWEAutomorphismKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }
    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }
    fn dsize(&self) -> Dsize {
        self.key.dsize()
    }
    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }
}

pub trait GLWEAutomorphismKeyPreparedFactory<B: Backend>
where
    Self: GGLWEPreparedFactory<B>,
{
    fn glwe_automorphism_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GLWEAutomorphismKeyPrepared<B::OwnedBuf, B> {
        GLWEAutomorphismKeyPrepared::<B::OwnedBuf, B> {
            key: self.gglwe_prepared_alloc(base2k, k, rank, rank, dnum, dsize),
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
        self.glwe_automorphism_key_prepared_alloc(infos.base2k(), infos.max_k(), infos.rank(), infos.dnum(), infos.dsize())
    }

    fn glwe_automorphism_key_prepared_bytes_of(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        self.gglwe_prepared_bytes_of(base2k, k, rank, rank, dnum, dsize)
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
        self.glwe_automorphism_key_prepared_bytes_of(infos.base2k(), infos.max_k(), infos.rank(), infos.dnum(), infos.dsize())
    }

    fn glwe_automorphism_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.gglwe_prepare_tmp_bytes(infos);
        lvl_0
    }

    fn glwe_automorphism_key_prepare<'s, R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'s, B>)
    where
        R: GGLWEPreparedToBackendMut<B> + SetGaloisElement,
        O: GGLWEToBackendRef<B> + GetGaloisElement,
        ScratchArena<'s, B>: ScratchAvailable,
        B: 's,
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

impl<'b, B: Backend + 'b> GLWEAutomorphismKeyPreparedToBackendRef<B> for &GLWEAutomorphismKeyPrepared<B::BufRef<'b>, B> {
    fn to_backend_ref(&self) -> GLWEAutomorphismKeyPreparedBackendRef<'_, B> {
        let key = &self.key;
        GLWEAutomorphismKeyPrepared {
            key: GGLWEPrepared {
                data: vmp_pmat_backend_ref_from_ref::<B>(&key.data),
                base2k: key.base2k,
                dsize: key.dsize,
            },
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

impl<'b, B: Backend + 'b> GGLWEPreparedToBackendRef<B> for &GLWEAutomorphismKeyPrepared<B::BufRef<'b>, B> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        GGLWEPrepared {
            data: vmp_pmat_backend_ref_from_ref::<B>(&self.key.data),
            base2k: self.key.base2k,
            dsize: self.key.dsize,
        }
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
