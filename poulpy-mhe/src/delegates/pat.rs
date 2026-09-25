use poulpy_core::layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWEToBackendMut};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::{PatAggregate, PatFinalize, PatNormalize},
    layouts::{GGLWEPatCompressedOwned, GGLWEPatOwned, GLWEPatCompressedOwned},
    oep::{PatAggregateImpl, PatFinalizeImpl, PatNormalizeImpl},
};

impl<BE: Backend + PatAggregateImpl> PatAggregate<BE> for Module<BE> {
    fn glwe_pat_compressed_aggregate_assign(&self, res: &mut GLWEPatCompressedOwned<BE>, a: &GLWEPatCompressedOwned<BE>) {
        BE::glwe_pat_compressed_aggregate_assign(self, res, a)
    }

    fn gglwe_pat_compressed_aggregate_assign(&self, res: &mut GGLWEPatCompressedOwned<BE>, a: &GGLWEPatCompressedOwned<BE>) {
        BE::gglwe_pat_compressed_aggregate_assign(self, res, a)
    }

    fn gglwe_pat_aggregate_assign(&self, res: &mut GGLWEPatOwned<BE>, a: &GGLWEPatOwned<BE>) {
        BE::gglwe_pat_aggregate_assign(self, res, a)
    }
}

impl<BE: Backend + PatNormalizeImpl> PatNormalize<BE> for Module<BE> {
    fn pat_normalize_tmp_bytes(&self) -> usize {
        BE::pat_normalize_tmp_bytes(self)
    }

    fn glwe_pat_compressed_normalize_assign(&self, res: &mut GLWEPatCompressedOwned<BE>, scratch: &mut ScratchArena<'_, BE>) {
        BE::glwe_pat_compressed_normalize_assign(self, res, scratch)
    }

    fn gglwe_pat_compressed_normalize_assign(&self, res: &mut GGLWEPatCompressedOwned<BE>, scratch: &mut ScratchArena<'_, BE>) {
        BE::gglwe_pat_compressed_normalize_assign(self, res, scratch)
    }

    fn gglwe_pat_normalize_assign(&self, res: &mut GGLWEPatOwned<BE>, scratch: &mut ScratchArena<'_, BE>) {
        BE::gglwe_pat_normalize_assign(self, res, scratch)
    }
}

impl<BE: Backend + PatFinalizeImpl> PatFinalize<BE> for Module<BE> {
    fn pat_finalize_tmp_bytes(&self) -> usize {
        BE::pat_finalize_tmp_bytes(self)
    }

    fn glwe_pat_compressed_finalize<R>(&self, res: &mut R, pat: &GLWEPatCompressedOwned<BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        BE::glwe_pat_compressed_finalize(self, res, pat, scratch)
    }

    fn gglwe_pat_compressed_finalize<R>(&self, res: &mut R, pat: &GGLWEPatCompressedOwned<BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        BE::gglwe_pat_compressed_finalize(self, res, pat, scratch)
    }

    fn gglwe_pat_finalize<R>(&self, res: &mut R, pat: &GGLWEPatOwned<BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        BE::gglwe_pat_finalize(self, res, pat, scratch)
    }
}
