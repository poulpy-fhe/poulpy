use poulpy_core::layouts::{
    GGLWECompressedSeed, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef, GGLWEInfos, GGLWEToBackendMut,
    GGLWEToBackendRef, GLWECompressedSeed, GLWECompressedToBackendMut, GLWECompressedToBackendRef, GLWEInfos, GLWEToBackendMut,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::{MHEGGLWEPatCompressedOps, MHEGGLWEPatOps, MHEGLWEPatCompressedOps},
    oep::{MHEGGLWEPatCompressedImpl, MHEGGLWEPatImpl, MHEGLWEPatCompressedImpl},
};

impl<BE: Backend + MHEGLWEPatCompressedImpl> MHEGLWEPatCompressedOps<BE> for Module<BE> {
    fn mhe_glwe_pat_compressed_aggregate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeed + GLWEInfos,
        A: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos,
    {
        BE::mhe_glwe_pat_compressed_aggregate_assign(self, res, a)
    }

    fn mhe_glwe_pat_compressed_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_pat_compressed_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_pat_compressed_finalize<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos,
    {
        BE::mhe_glwe_pat_compressed_finalize(self, res, pat, scratch)
    }
}

impl<BE: Backend + MHEGGLWEPatCompressedImpl> MHEGGLWEPatCompressedOps<BE> for Module<BE> {
    fn mhe_gglwe_pat_compressed_aggregate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeed + GGLWEInfos,
        A: GGLWECompressedToBackendRef<BE> + GGLWECompressedSeed + GGLWEInfos,
    {
        BE::mhe_gglwe_pat_compressed_aggregate_assign(self, res, a)
    }

    fn mhe_gglwe_pat_compressed_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_gglwe_pat_compressed_finalize_tmp_bytes(self)
    }

    fn mhe_gglwe_pat_compressed_finalize<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        P: GGLWECompressedToBackendRef<BE> + GGLWEInfos,
    {
        BE::mhe_gglwe_pat_compressed_finalize(self, res, pat, scratch)
    }
}

impl<BE: Backend + MHEGGLWEPatImpl> MHEGGLWEPatOps<BE> for Module<BE> {
    fn mhe_gglwe_pat_aggregate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
    {
        BE::mhe_gglwe_pat_aggregate_assign(self, res, a)
    }

    fn mhe_gglwe_pat_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_gglwe_pat_finalize_tmp_bytes(self)
    }

    fn mhe_gglwe_pat_finalize<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        P: GGLWEToBackendRef<BE> + GGLWEInfos,
    {
        BE::mhe_gglwe_pat_finalize(self, res, pat, scratch)
    }
}
