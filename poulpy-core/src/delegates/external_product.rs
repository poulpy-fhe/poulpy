use poulpy_hal::layouts::{Backend, Module, Scratch};

use crate::{
    api::{GGLWEExternalProduct, GGSWExternalProduct, GLWEExternalProduct},
    external_product::{GGLWEExternalProductDefault, GGSWExternalProductDefault},
    layouts::{
        GGLWEInfos, GGLWEToMut, GGLWEToRef, GGSWInfos, GGSWPreparedToRef, GGSWToMut, GGSWToRef, GLWEInfos, GLWEToMut, GLWEToRef,
    },
    oep::CoreImpl,
};

impl<BE> GLWEExternalProduct<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        BE::glwe_external_product_tmp_bytes(self, res_infos, a_infos, b_infos)
    }

    fn glwe_external_product_assign<R, D>(&self, res: &mut R, rhs: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        D: GGSWPreparedToRef<BE> + GGSWInfos,
    {
        BE::glwe_external_product_assign(self, res, rhs, scratch)
    }

    fn glwe_external_product<R, A, D>(&self, res: &mut R, lhs: &A, rhs: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        D: GGSWPreparedToRef<BE> + GGSWInfos,
    {
        BE::glwe_external_product(self, res, lhs, rhs, scratch)
    }
}

impl<BE> GGLWEExternalProduct<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GGLWEExternalProductDefault<BE>,
{
    fn gglwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
    {
        BE::gglwe_external_product_tmp_bytes(self, res_infos, a_infos, b_infos)
    }

    fn gglwe_external_product<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + GGLWEInfos,
        A: GGLWEToRef + GGLWEInfos,
        B: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::gglwe_external_product(self, res, a, b, scratch)
    }

    fn gglwe_external_product_assign<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::gglwe_external_product_assign(self, res, a, scratch)
    }
}

impl<BE> GGSWExternalProduct<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GGSWExternalProductDefault<BE>,
{
    fn ggsw_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
    {
        BE::ggsw_external_product_tmp_bytes(self, res_infos, a_infos, b_infos)
    }

    fn ggsw_external_product<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        B: GGSWPreparedToRef<BE>,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::ggsw_external_product(self, res, a, b, scratch)
    }

    fn ggsw_external_product_assign<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::ggsw_external_product_assign(self, res, a, scratch)
    }
}
