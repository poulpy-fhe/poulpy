use crate::CKKSResult as Result;
use poulpy_core::{
    GLWECopy, GLWENegate, GLWERotate, GLWEShift,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::CoeffNormalized;
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, CoeffFitsIn, Module, ScratchArena},
};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSImagOps, oep::CKKSImagImpl};

impl<BE: Backend + CKKSImagImpl<BE>> CKKSImagOps<BE> for Module<BE>
where
    Module<BE>: GLWECopy<BE> + GLWENegate<BE> + GLWERotate<BE> + GLWEShift<BE> + ModuleN,
{
    fn ckks_mul_i_tmp_bytes(&self) -> usize {
        BE::ckks_mul_i_tmp_bytes_impl(self)
    }

    fn ckks_mul_i_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = CoeffNormalized> + CKKSCtBounds,
        <Src as GLWEToBackendRef<BE>>::State: CoeffFitsIn<<Dst as GLWEToBackendRef<BE>>::State>,
    {
        BE::ckks_mul_i_into_impl(self, dst, src, scratch)
    }

    fn ckks_mul_i_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_mul_i_assign_impl(self, dst, scratch)
    }

    fn ckks_div_i_tmp_bytes(&self) -> usize {
        BE::ckks_div_i_tmp_bytes_impl(self)
    }

    fn ckks_div_i_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = CoeffNormalized> + CKKSCtBounds,
    {
        BE::ckks_div_i_into_impl(self, dst, src, scratch)
    }

    fn ckks_div_i_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_div_i_assign_impl(self, dst, scratch)
    }
}
