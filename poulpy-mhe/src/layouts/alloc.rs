use poulpy_core::layouts::{
    Base2K, Dnum, Dsize, GGLWEInfos, GLWEInfos, ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Module};

use crate::layouts::{
    GGLWEPat, GGLWEPatCompressed, GGLWEPatCompressedOwned, GGLWEPatOwned, GLWEPatCompressed, GLWEPatCompressedOwned,
};

/// PAT allocation on a backend module.
///
/// Every method is default-bodied over the core allocation supertraits, so the
/// blanket impl for `Module<BE>` is empty. A fresh PAT is zero, hence canonical.
pub trait MHEModuleAlloc<BE: Backend>:
    ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
    + ModuleCoreCompressedAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
    fn glwe_pat_compressed_alloc_from_infos<A: GLWEInfos>(&self, infos: &A) -> GLWEPatCompressedOwned<BE> {
        GLWEPatCompressed {
            inner: self.glwe_compressed_alloc_from_infos(infos),
            canonical: true,
        }
    }

    fn glwe_pat_compressed_alloc(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> GLWEPatCompressedOwned<BE> {
        GLWEPatCompressed {
            inner: self.glwe_compressed_alloc(base2k, k, rank),
            canonical: true,
        }
    }

    fn gglwe_pat_compressed_alloc_from_infos<A: GGLWEInfos>(&self, infos: &A) -> GGLWEPatCompressedOwned<BE> {
        GGLWEPatCompressed {
            inner: self.gglwe_compressed_alloc_from_infos(infos),
            canonical: true,
        }
    }

    fn gglwe_pat_compressed_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> GGLWEPatCompressedOwned<BE> {
        GGLWEPatCompressed {
            inner: self.gglwe_compressed_alloc(base2k, dnum, dsize, k_aux, rank_in, rank_out),
            canonical: true,
        }
    }

    fn gglwe_pat_alloc_from_infos<A: GGLWEInfos>(&self, infos: &A) -> GGLWEPatOwned<BE> {
        GGLWEPat {
            inner: self.gglwe_alloc_from_infos(infos),
            canonical: true,
        }
    }

    fn gglwe_pat_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> GGLWEPatOwned<BE> {
        GGLWEPat {
            inner: self.gglwe_alloc(base2k, dnum, dsize, k_aux, rank_in, rank_out),
            canonical: true,
        }
    }
}

impl<BE: Backend> MHEModuleAlloc<BE> for Module<BE> where
    Self: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + ModuleCoreCompressedAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
}
