use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::{GGLWEExternalProduct, GGSWExternalProduct, GLWEExternalProduct},
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut,
        GGSWToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGSWPreparedToBackendRef,
    },
    oep::{GGLWEExternalProductImpl, GGSWExternalProductImpl, GLWEExternalProductImpl},
};

macro_rules! impl_external_product_delegate {
    ($trait:ty, [$($bounds:tt)+], $($body:item)+) => {
        impl<BE> $trait for Module<BE>
        where
            $($bounds)+
        {
            $($body)+
        }
    };
}

impl_external_product_delegate!(
    GLWEExternalProduct<BE>,
    [BE: Backend + GLWEExternalProductImpl<BE>],
    fn glwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        BE::glwe_external_product_tmp_bytes(self, res_infos, a_infos, b_infos)
    }

    fn glwe_external_product_assign<'s, R, D>(
        &self,
        res: &mut R,
        rhs: &D,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        D: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::glwe_external_product_assign(self, res, rhs, key_size, scratch)
    }

    fn glwe_external_product<'s, R, A, D>(
        &self,
        res: &mut R,
        lhs: &A,
        rhs: &D,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        D: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::glwe_external_product(self, res, lhs, rhs, key_size, scratch)
    }
);

impl_external_product_delegate!(
    GGLWEExternalProduct<BE>,
    [BE: Backend + GGLWEExternalProductImpl<BE>, Module<BE>: GLWEExternalProduct<BE>],
    fn gglwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
    {
        BE::gglwe_external_product_tmp_bytes(self, res_infos, a_infos, b_infos)
    }

    fn gglwe_external_product<'s, R, A, B>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::gglwe_external_product(self, res, a, b, key_size, scratch)
    }

    fn gglwe_external_product_assign<'s, R, A>(
        &self,
        res: &mut R,
        a: &A,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::gglwe_external_product_assign(self, res, a, key_size, scratch)
    }
);

impl_external_product_delegate!(
    GGSWExternalProduct<BE>,
    [BE: Backend + GGSWExternalProductImpl<BE>, Module<BE>: GLWEExternalProduct<BE>],
    fn ggsw_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
    {
        BE::ggsw_external_product_tmp_bytes(self, res_infos, a_infos, b_infos)
    }

    fn ggsw_external_product<'s, R, A, B>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::ggsw_external_product(self, res, a, b, key_size, scratch)
    }

    fn ggsw_external_product_assign<'s, R, A>(
        &self,
        res: &mut R,
        a: &A,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::ggsw_external_product_assign(self, res, a, key_size, scratch)
    }
);
