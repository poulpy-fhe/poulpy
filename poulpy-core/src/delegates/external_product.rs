use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::GLWEExternalProduct,
    layouts::{GGSWInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGSWPreparedBackendRef},
    oep::GLWEExternalProductImpl,
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
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_external_product(self, res, lhs, rhs, scratch)
    }
);
