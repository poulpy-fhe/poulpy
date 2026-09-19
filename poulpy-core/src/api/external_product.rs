use poulpy_hal::layouts::{Backend, ScratchArena, VecZnxDft};

use crate::layouts::{GGSWInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGSWPreparedBackendRef};

pub trait GLWEExternalProduct<BE: Backend> {
    fn glwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;

    fn glwe_external_product_assign<R>(
        &self,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;

    fn glwe_external_product<R, A>(
        &self,
        res: &mut R,
        lhs: &A,
        rhs: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;
}

pub trait GLWEExternalProductInternal<BE: Backend> {
    fn glwe_external_product_internal_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;

    fn glwe_external_product_dft<'r, A>(
        &self,
        res_dft: &mut VecZnxDft<<BE as Backend>::BufMut<'r>, BE::DftWord, BE>,
        a: &A,
        ggsw: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE>;
}
