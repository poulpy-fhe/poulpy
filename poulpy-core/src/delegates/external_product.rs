use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

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
        impl<BE, Rg: Ring> $trait for Module<BE>
        where
            $($bounds)+
        {
            $($body)+
        }
    };
}

impl_external_product_delegate!(
    GLWEExternalProduct<BE, Rg>,
    [BE: Backend + GLWEExternalProductImpl<Rg>],
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
        rhs: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
    {
        BE::glwe_external_product_assign(self, res, rhs, scratch)
    }

    fn glwe_external_product<R, A>(
        &self,
        res: &mut R,
        lhs: &A,
        rhs: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_external_product(self, res, lhs, rhs, scratch)
    }
);

impl_external_product_delegate!(
    GGLWEExternalProduct<BE, Rg>,
    [BE: Backend + GGLWEExternalProductImpl<Rg>],
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
        b: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGLWEToBackendMut<BE, Rg> + GGLWEInfos,
        A: GGLWEToBackendRef<BE, Rg> + GGLWEInfos,
    {
        BE::gglwe_external_product(self, res, a, b, scratch)
    }

    fn gglwe_external_product_assign<R>(
        &self,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGLWEToBackendMut<BE, Rg> + GGLWEInfos,
    {
        BE::gglwe_external_product_assign(self, res, a, scratch)
    }
);

impl_external_product_delegate!(
    GGSWExternalProduct<BE, Rg>,
    [BE: Backend + GGSWExternalProductImpl<Rg>],
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
        b: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGSWToBackendMut<BE, Rg> + GGSWAtViewMut<BE, Rg> + GGSWInfos,
        A: GGSWToBackendRef<BE, Rg> + GGSWAtViewRef<BE, Rg> + GGSWInfos,
    {
        BE::ggsw_external_product(self, res, a, b, scratch)
    }

    fn ggsw_external_product_assign<R>(
        &self,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGSWToBackendMut<BE, Rg> + GGSWAtViewMut<BE, Rg> + GGSWInfos,
    {
        BE::ggsw_external_product_assign(self, res, a, scratch)
    }
);

impl<BE: Backend + GLWEExternalProductImpl<Rg>, Rg: Ring> crate::api::GLWEExternalProductDft<BE, Rg> for Module<BE> {
    fn glwe_external_product_dft_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        BE::glwe_external_product_dft_tmp_bytes(self, res_infos, a_infos, b_infos)
    }

    fn glwe_external_product_dft<'r, A>(
        &self,
        res_dft: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'r, BE, Rg>,
        a: &A,
        ggsw: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE, Rg>,
    {
        BE::glwe_external_product_dft(self, res_dft, a, ggsw, scratch)
    }
}
