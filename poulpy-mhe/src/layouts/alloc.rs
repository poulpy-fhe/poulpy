use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Module};

use crate::layouts::{
    GGLWEPat, GGLWEPatCompressed, GGLWEPatCompressedOwned, GGLWEPatOwned, GLWEAutomorphismKeyPatCompressed,
    GLWEAutomorphismKeyPatCompressedOwned, GLWEPatCompressed, GLWEPatCompressedOwned, GLWESwitchingKeyPatCompressed,
    GLWESwitchingKeyPatCompressedOwned,
};

/// PAT allocation on a backend module.
///
/// Every method is default-bodied over the core allocation supertraits, so the
/// blanket impl for `Module<BE>` is empty. A fresh PAT is zero, hence canonical;
/// key metadata starts as in core (degrees `0`, Galois element `0`).
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

    fn glwe_switching_key_pat_compressed_alloc_from_infos<A: GGLWEInfos>(
        &self,
        infos: &A,
    ) -> GLWESwitchingKeyPatCompressedOwned<BE> {
        GLWESwitchingKeyPatCompressed {
            key: self.gglwe_pat_compressed_alloc_from_infos(infos),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }

    fn glwe_switching_key_pat_compressed_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> GLWESwitchingKeyPatCompressedOwned<BE> {
        GLWESwitchingKeyPatCompressed {
            key: self.gglwe_pat_compressed_alloc(base2k, dnum, dsize, k_aux, rank_in, rank_out),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }

    fn glwe_automorphism_key_pat_compressed_alloc_from_infos<A: GGLWEInfos>(
        &self,
        infos: &A,
    ) -> GLWEAutomorphismKeyPatCompressedOwned<BE> {
        self.glwe_automorphism_key_pat_compressed_alloc(infos.base2k(), infos.dnum(), infos.dsize(), infos.k_aux(), infos.rank())
    }

    fn glwe_automorphism_key_pat_compressed_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> GLWEAutomorphismKeyPatCompressedOwned<BE> {
        GLWEAutomorphismKeyPatCompressed {
            key: self.gglwe_pat_compressed_alloc(base2k, dnum, dsize, k_aux, rank, rank),
            p: 0,
        }
    }
}

impl<BE: Backend> MHEModuleAlloc<BE> for Module<BE> where
    Self: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + ModuleCoreCompressedAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
}
