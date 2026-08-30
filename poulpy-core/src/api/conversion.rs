use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    LWEMatrixInfos, LWEMatrixToBackendMut, LWEToBackendMut, LWEToBackendRef,
    prepared::{GGLWEPreparedBackendRef, GGLWEToGGSWKeyPreparedBackendRef},
};

pub trait LWESampleExtract<BE: Backend> {
    fn lwe_sample_extract<R, A>(&self, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;
}

pub trait GLWEFromLWE<BE: Backend> {
    fn glwe_from_lwe_tmp_bytes<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe<R, A>(
        &self,
        res: &mut R,
        lwe: &A,
        ksk: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos;
}

pub trait LWEFromGLWE<BE: Backend> {
    fn lwe_from_glwe_tmp_bytes<R, A, K>(&self, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn lwe_from_glwe<R, A>(
        &self,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;
}

pub trait GGSWFromGGLWE<BE: Backend> {
    fn ggsw_from_gglwe_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_from_gglwe<R, A>(
        &self,
        res: &mut R,
        a: &A,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos;
}

pub trait GLWEExpandLWE<BE: Backend> {
    fn glwe_expand_lwe_tmp_bytes<R, A>(&self, lwe_infos: &R, a_infos: &A) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos;

    fn glwe_expand_lwe<R, A>(&self, res: &mut [R], a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;
}

pub trait GLWEExpandLWEMatrix<BE: Backend> {
    fn glwe_expand_lwe_matrix_tmp_bytes<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
    where
        R: LWEMatrixInfos,
        A: GLWEInfos;

    fn glwe_expand_lwe_matrix<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;
}

pub trait GGSWExpandRows<BE: Backend> {
    fn ggsw_expand_rows_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row<R>(&self, res: &mut R, tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos;
}
