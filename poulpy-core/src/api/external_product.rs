use poulpy_hal::layouts::{Backend, ScratchArena, VecZnxDft};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut,
        GGSWToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGSWPreparedToBackendRef,
    },
};

pub trait GLWEExternalProduct<BE: Backend> {
    fn glwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;

    fn glwe_external_product_assign<'s, R, D>(&self, res: &mut R, a: &D, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        D: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>;

    fn glwe_external_product<'s, R, A, D>(
        &self,
        res: &mut R,
        lhs: &A,
        rhs: &D,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        D: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>;
}

pub trait GLWEExternalProductInternal<BE: Backend> {
    fn glwe_external_product_internal_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;

    fn glwe_external_product_dft<'s, 'r, A, G>(
        &self,
        res_dft: &mut VecZnxDft<<BE as Backend>::BufMut<'r>, BE>,
        a: &A,
        ggsw: &G,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        A: GLWEToBackendRef<BE>,
        G: GGSWPreparedToBackendRef<BE>,
        BE: 's,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>;
}

pub trait GGLWEExternalProduct<BE: Backend> {
    fn gglwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos;

    fn gglwe_external_product<'s, R, A, B>(&self, res: &mut R, a: &A, b: &B, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>;

    fn gglwe_external_product_assign<'s, R, A>(&self, res: &mut R, a: &A, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>;
}

pub trait GGSWExternalProduct<BE: Backend> {
    fn ggsw_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos;

    fn ggsw_external_product<'s, R, A, B>(&self, res: &mut R, a: &A, b: &B, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>;

    fn ggsw_external_product_assign<'s, R, A>(&self, res: &mut R, a: &A, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>;
}
