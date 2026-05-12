use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWENegate, GLWERotate, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSImagOps, oep::CKKSImagImpl};

impl<BE: Backend + CKKSImagImpl<BE>> CKKSImagOps<BE> for Module<BE>
where
    Module<BE>: GLWECopy<BE> + GLWENegate<BE> + GLWERotate<BE> + GLWEShift<BE> + ModuleN,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_mul_i_tmp_bytes(&self) -> usize {
        BE::ckks_mul_i_tmp_bytes(self)
    }

    fn ckks_mul_i_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_mul_i_into(self, dst, src, scratch)
    }

    fn ckks_mul_i_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_mul_i_assign(self, dst, scratch)
    }

    fn ckks_div_i_tmp_bytes(&self) -> usize {
        BE::ckks_div_i_tmp_bytes(self)
    }

    fn ckks_div_i_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_div_i_into(self, dst, src, scratch)
    }

    fn ckks_div_i_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_div_i_assign(self, dst, scratch)
    }
}
