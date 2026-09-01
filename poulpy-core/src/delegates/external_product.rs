use poulpy_hal::layouts::{Backend, Module, Normalized, ScratchArena};

use crate::{
    api::{GGLWEExternalProduct, GGSWExternalProduct, GLWEExternalProduct},
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut,
        GGSWToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGSWPreparedBackendRef,
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

    fn glwe_external_product_assign<R>(
        &self,
        res: &mut R,
        rhs: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GLWEToBackendMut<BE, State = Normalized> + GLWEInfos,
    {
        BE::glwe_external_product_assign(self, res, rhs, scratch)
    }

    fn glwe_external_product<R, A>(
        &self,
        res: &mut R,
        lhs: &A,
        rhs: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos,
    {
        BE::glwe_external_product(self, res, lhs, rhs, scratch)
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

    fn gglwe_external_product<R, A>(
        &self,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
    {
        BE::gglwe_external_product(self, res, a, b, scratch)
    }

    fn gglwe_external_product_assign<R>(
        &self,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        BE::gglwe_external_product_assign(self, res, a, scratch)
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

    fn ggsw_external_product<R, A>(
        &self,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
    {
        BE::ggsw_external_product(self, res, a, b, scratch)
    }

    fn ggsw_external_product_assign<R>(
        &self,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    {
        BE::ggsw_external_product_assign(self, res, a, scratch)
    }
);
