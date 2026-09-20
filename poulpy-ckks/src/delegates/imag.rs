use crate::CKKSResult as Result;
use crate::{api::CKKSModuleInfos, ckks_ensure};
use poulpy_core::{
    GLWECopy, GLWENegate, GLWERotate, GLWEShift,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSImagOps, oep::CKKSImagImpl};

impl<BE: Backend + CKKSImagImpl> CKKSImagOps<BE> for Module<BE>
where
    Module<BE>: GLWECopy<BE> + GLWENegate<BE> + GLWERotate<BE> + GLWEShift<BE> + ModuleN,
{
    fn ckks_mul_i_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_mul_i_tmp_bytes_impl(self, res_size)
    }

    fn ckks_mul_i_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_mul_i_into", dst)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_mul_i_into", src)?;
        ckks_ensure!(
            !self.ckks_is_conjugate_invariant(),
            "imaginary multiplication requires the standard CKKS ring"
        );
        BE::ckks_mul_i_into_impl(self, dst, src, scratch)
    }

    fn ckks_mul_i_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_mul_i_assign", dst)?;
        ckks_ensure!(
            !self.ckks_is_conjugate_invariant(),
            "imaginary multiplication requires the standard CKKS ring"
        );
        BE::ckks_mul_i_assign_impl(self, dst, scratch)
    }

    fn ckks_div_i_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_div_i_tmp_bytes_impl(self, res_size)
    }

    fn ckks_div_i_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_div_i_into", dst)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_div_i_into", src)?;
        ckks_ensure!(
            !self.ckks_is_conjugate_invariant(),
            "imaginary multiplication requires the standard CKKS ring"
        );
        BE::ckks_div_i_into_impl(self, dst, src, scratch)
    }

    fn ckks_div_i_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_div_i_assign", dst)?;
        ckks_ensure!(
            !self.ckks_is_conjugate_invariant(),
            "imaginary multiplication requires the standard CKKS ring"
        );
        BE::ckks_div_i_assign_impl(self, dst, scratch)
    }
}
