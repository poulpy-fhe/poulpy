use crate::layouts::LWEInfos;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

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
    [BE: Backend + GLWEExternalProductImpl],
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
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        let metadata = super::matching_metadata(res.to_backend_ref().encryption_metadata(), rhs.encryption_metadata());
        BE::glwe_external_product_assign(self, res, rhs, scratch);
        res.set_encryption_metadata(metadata);
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
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        let metadata = super::matching_metadata(lhs.to_backend_ref().encryption_metadata(), rhs.encryption_metadata());
        BE::glwe_external_product(self, res, lhs, rhs, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_external_product_delegate!(
    GGLWEExternalProduct<BE>,
    [BE: Backend + GGLWEExternalProductImpl, Module<BE>: GLWEExternalProduct<BE>],
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
        let metadata = super::matching_metadata(a.to_backend_ref().encryption_metadata(), b.encryption_metadata());
        BE::gglwe_external_product(self, res, a, b, scratch);
        res.set_encryption_metadata(metadata);
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
        let metadata = super::matching_metadata(res.to_backend_ref().encryption_metadata(), a.encryption_metadata());
        BE::gglwe_external_product_assign(self, res, a, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_external_product_delegate!(
    GGSWExternalProduct<BE>,
    [BE: Backend + GGSWExternalProductImpl, Module<BE>: GLWEExternalProduct<BE>],
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
        let metadata = super::matching_metadata(a.to_backend_ref().encryption_metadata(), b.encryption_metadata());
        BE::ggsw_external_product(self, res, a, b, scratch);
        res.set_encryption_metadata(metadata);
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
        let metadata = super::matching_metadata(res.to_backend_ref().encryption_metadata(), a.encryption_metadata());
        BE::ggsw_external_product_assign(self, res, a, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl<BE: Backend + GLWEExternalProductImpl> crate::api::GLWEExternalProductInternal<BE> for Module<BE> {
    fn glwe_external_product_internal_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        BE::glwe_external_product_internal_tmp_bytes(self, res_infos, a_infos, b_infos)
    }

    fn glwe_external_product_dft<'r, A>(
        &self,
        res_dft: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'r, BE>,
        a: &A,
        ggsw: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_external_product_dft(self, res_dft, a, ggsw, scratch)
    }
}
