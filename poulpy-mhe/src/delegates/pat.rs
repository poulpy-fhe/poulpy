use poulpy_core::layouts::{
    GGLWECompressedSeed, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef, GGLWEInfos, GGLWEToBackendMut,
    GGLWEToBackendRef, GLWECompressedSeed, GLWECompressedToBackendMut, GLWECompressedToBackendRef, GLWEInfos,
    GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, GLWEToBackendMut, GetGaloisElement, SetGaloisElement,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::{
        GGLWEPatCompressedOps, GGLWEPatOps, GLWEAutomorphismKeyPatCompressedOps, GLWEPatCompressedOps,
        GLWESwitchingKeyPatCompressedOps,
    },
    oep::{
        GGLWEPatCompressedImpl, GGLWEPatImpl, GLWEAutomorphismKeyPatCompressedImpl, GLWEPatCompressedImpl,
        GLWESwitchingKeyPatCompressedImpl,
    },
};

impl<BE: Backend + GLWEPatCompressedImpl> GLWEPatCompressedOps<BE> for Module<BE> {
    fn glwe_pat_compressed_aggregate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeed + GLWEInfos,
        A: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos,
    {
        BE::glwe_pat_compressed_aggregate_assign(self, res, a)
    }

    fn glwe_pat_compressed_normalize_tmp_bytes(&self) -> usize {
        BE::glwe_pat_compressed_normalize_tmp_bytes(self)
    }

    fn glwe_pat_compressed_normalize_assign<R>(&self, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWECompressedToBackendMut<BE> + GLWEInfos,
    {
        BE::glwe_pat_compressed_normalize_assign(self, res, scratch)
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

    fn gglwe_pat_compressed_normalize_tmp_bytes(&self) -> usize {
        BE::gglwe_pat_compressed_normalize_tmp_bytes(self)
    }

    fn gglwe_pat_compressed_normalize_assign<R>(&self, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWECompressedToBackendMut<BE> + GGLWEInfos,
    {
        BE::gglwe_pat_compressed_normalize_assign(self, res, scratch)
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

    fn gglwe_pat_normalize_tmp_bytes(&self) -> usize {
        BE::gglwe_pat_normalize_tmp_bytes(self)
    }

    fn gglwe_pat_normalize_assign<R>(&self, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        BE::gglwe_pat_normalize_assign(self, res, scratch)
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

impl<BE: Backend + GLWESwitchingKeyPatCompressedImpl> GLWESwitchingKeyPatCompressedOps<BE> for Module<BE> {
    fn glwe_switching_key_pat_compressed_aggregate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeed + GGLWEInfos + GLWESwitchingKeyDegrees,
        A: GGLWECompressedToBackendRef<BE> + GGLWECompressedSeed + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::glwe_switching_key_pat_compressed_aggregate_assign(self, res, a)
    }

    fn glwe_switching_key_pat_compressed_normalize_tmp_bytes(&self) -> usize {
        BE::glwe_switching_key_pat_compressed_normalize_tmp_bytes(self)
    }

    fn glwe_switching_key_pat_compressed_normalize_assign<R>(&self, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWECompressedToBackendMut<BE> + GGLWEInfos,
    {
        BE::glwe_switching_key_pat_compressed_normalize_assign(self, res, scratch)
    }

    fn glwe_switching_key_pat_compressed_finalize_tmp_bytes(&self) -> usize {
        BE::glwe_switching_key_pat_compressed_finalize_tmp_bytes(self)
    }

    fn glwe_switching_key_pat_compressed_finalize<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        P: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::glwe_switching_key_pat_compressed_finalize(self, res, pat, scratch)
    }
}

impl<BE: Backend + GLWEAutomorphismKeyPatCompressedImpl> GLWEAutomorphismKeyPatCompressedOps<BE> for Module<BE> {
    fn glwe_automorphism_key_pat_compressed_aggregate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeed + GGLWEInfos + GetGaloisElement,
        A: GGLWECompressedToBackendRef<BE> + GGLWECompressedSeed + GGLWEInfos + GetGaloisElement,
    {
        BE::glwe_automorphism_key_pat_compressed_aggregate_assign(self, res, a)
    }

    fn glwe_automorphism_key_pat_compressed_normalize_tmp_bytes(&self) -> usize {
        BE::glwe_automorphism_key_pat_compressed_normalize_tmp_bytes(self)
    }

    fn glwe_automorphism_key_pat_compressed_normalize_assign<R>(&self, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWECompressedToBackendMut<BE> + GGLWEInfos,
    {
        BE::glwe_automorphism_key_pat_compressed_normalize_assign(self, res, scratch)
    }

    fn glwe_automorphism_key_pat_compressed_finalize_tmp_bytes(&self) -> usize {
        BE::glwe_automorphism_key_pat_compressed_finalize_tmp_bytes(self)
    }

    fn glwe_automorphism_key_pat_compressed_finalize<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
        P: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    {
        BE::glwe_automorphism_key_pat_compressed_finalize(self, res, pat, scratch)
    }
}
