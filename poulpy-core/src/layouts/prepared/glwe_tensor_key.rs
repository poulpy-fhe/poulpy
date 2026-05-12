use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, Module, ScratchArena, vmp_pmat_backend_ref_from_ref},
};

use crate::layouts::prepared::{GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef};
use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendMut, GGLWEPreparedFactory, GGLWEToBackendRef,
    GLWEInfos, LWEInfos, Rank, TorusPrecision,
};

/// DFT-domain (prepared) variant of a GLWE tensor key.
///
/// A newtype wrapper around [`GGLWEPrepared`] for tensor operations.
/// Tied to a specific backend via `B: Backend`.
#[derive(PartialEq, Eq)]
pub struct GLWETensorKeyPrepared<D: Data, B: Backend>(pub(crate) GGLWEPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for GLWETensorKeyPrepared<D, B> {
    fn n(&self) -> Degree {
        self.0.n()
    }

    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<D: Data, B: Backend> LWEInfos for &GLWETensorKeyPrepared<D, B> {
    fn n(&self) -> Degree {
        (*self).n()
    }

    fn base2k(&self) -> Base2K {
        (*self).base2k()
    }

    fn size(&self) -> usize {
        (*self).size()
    }
}

impl<D: Data, B: Backend> LWEInfos for &mut GLWETensorKeyPrepared<D, B> {
    fn n(&self) -> Degree {
        (**self).n()
    }

    fn base2k(&self) -> Base2K {
        (**self).base2k()
    }

    fn size(&self) -> usize {
        (**self).size()
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWETensorKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GLWEInfos for &GLWETensorKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        (*self).rank()
    }
}

impl<D: Data, B: Backend> GLWEInfos for &mut GLWETensorKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        (**self).rank()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWETensorKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for &GLWETensorKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        (*self).rank_in()
    }

    fn rank_out(&self) -> Rank {
        (*self).rank_out()
    }

    fn dsize(&self) -> Dsize {
        (*self).dsize()
    }

    fn dnum(&self) -> Dnum {
        (*self).dnum()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for &mut GLWETensorKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        (**self).rank_in()
    }

    fn rank_out(&self) -> Rank {
        (**self).rank_out()
    }

    fn dsize(&self) -> Dsize {
        (**self).dsize()
    }

    fn dnum(&self) -> Dnum {
        (**self).dnum()
    }
}

pub trait GLWETensorKeyPreparedFactory<B: Backend>
where
    Self: GGLWEPreparedFactory<B>,
{
    fn alloc_tensor_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        dnum: Dnum,
        dsize: Dsize,
        rank: Rank,
    ) -> GLWETensorKeyPrepared<B::OwnedBuf, B> {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        GLWETensorKeyPrepared(self.gglwe_prepared_alloc(base2k, k, Rank(pairs), rank, dnum, dsize))
    }

    fn alloc_tensor_key_prepared_from_infos<A>(&self, infos: &A) -> GLWETensorKeyPrepared<B::OwnedBuf, B>
    where
        A: GGLWEInfos,
    {
        self.alloc_tensor_key_prepared(infos.base2k(), infos.max_k(), infos.dnum(), infos.dsize(), infos.rank_out())
    }

    fn bytes_of_tensor_key_prepared(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        self.gglwe_prepared_bytes_of(base2k, k, Rank(pairs), rank, dnum, dsize)
    }

    fn bytes_of_tensor_key_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.bytes_of_tensor_key_prepared(infos.base2k(), infos.max_k(), infos.rank(), infos.dnum(), infos.dsize())
    }

    fn prepare_tensor_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.gglwe_prepare_tmp_bytes(infos);
        lvl_0
    }

    fn prepare_tensor_key<'s, R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'s, B>)
    where
        R: GGLWEPreparedToBackendMut<B>,
        O: GGLWEToBackendRef<B>,
        ScratchArena<'s, B>: ScratchAvailable,
        B: 's,
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
        GLWETensorKeyPrepared(self.0.to_backend_ref())
    }
}

impl<'b, B: Backend + 'b> GLWETensorKeyPreparedToBackendRef<B> for &GLWETensorKeyPrepared<B::BufRef<'b>, B> {
    fn to_backend_ref(&self) -> GLWETensorKeyPreparedBackendRef<'_, B> {
        let inner = &self.0;
        GLWETensorKeyPrepared(GGLWEPrepared {
            data: vmp_pmat_backend_ref_from_ref::<B>(&inner.data),
            base2k: inner.base2k,
            dsize: inner.dsize,
        })
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
        GLWETensorKeyPrepared(self.0.to_backend_mut())
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
