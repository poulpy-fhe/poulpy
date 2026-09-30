use poulpy_core::layouts::{
    GGLWECompressedSeed, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef, GGLWEInfos, GGLWEToBackendMut,
    GGLWEToBackendRef, GLWECompressedSeed, GLWECompressedToBackendMut, GLWECompressedToBackendRef, GLWEInfos, GLWEToBackendMut,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::{GGLWEPatCompressedOps, GGLWEPatOps, GLWEPatCompressedOps},
    oep::{GGLWEPatCompressedImpl, GGLWEPatImpl, GLWEPatCompressedImpl},
};

impl<BE: Backend + GLWEPatCompressedImpl> GLWEPatCompressedOps<BE> for Module<BE> {
    fn glwe_pat_compressed_aggregate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeed + GLWEInfos,
        A: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos,
    {
        BE::glwe_pat_compressed_aggregate_assign(self, res, a)
    }

    fn glwe_pat_compressed_finalize_tmp_bytes(&self) -> usize {
        BE::glwe_pat_compressed_finalize_tmp_bytes(self)
    }

    fn glwe_pat_compressed_finalize<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos,
    {
        BE::glwe_pat_compressed_finalize(self, res, pat, scratch)
    }
}

impl<BE: Backend + GGLWEPatCompressedImpl> GGLWEPatCompressedOps<BE> for Module<BE> {
    fn gglwe_pat_compressed_aggregate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeed + GGLWEInfos,
        A: GGLWECompressedToBackendRef<BE> + GGLWECompressedSeed + GGLWEInfos,
    {
        BE::gglwe_pat_compressed_aggregate_assign(self, res, a)
    }

    fn gglwe_pat_compressed_finalize_tmp_bytes(&self) -> usize {
        BE::gglwe_pat_compressed_finalize_tmp_bytes(self)
    }

    fn gglwe_pat_compressed_finalize<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        P: GGLWECompressedToBackendRef<BE> + GGLWEInfos,
    {
        BE::gglwe_pat_compressed_finalize(self, res, pat, scratch)
    }
}

impl<BE: Backend + GGLWEPatImpl> GGLWEPatOps<BE> for Module<BE> {
    fn gglwe_pat_aggregate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
    {
        BE::gglwe_pat_aggregate_assign(self, res, a)
    }

    fn gglwe_pat_finalize_tmp_bytes(&self) -> usize {
        BE::gglwe_pat_finalize_tmp_bytes(self)
    }

    fn gglwe_pat_finalize<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        P: GGLWEToBackendRef<BE> + GGLWEInfos,
    {
        BE::gglwe_pat_finalize(self, res, pat, scratch)
    }
}
