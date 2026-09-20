use crate::CKKSResult as Result;
use poulpy_core::{
    GLWECopy, GLWEShift,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSCopyOps, oep::CKKSCopyImpl};

impl<BE: Backend + CKKSCopyImpl> CKKSCopyOps<BE> for Module<BE>
where
    Module<BE>: GLWECopy<BE> + GLWEShift<BE>,
{
    fn ckks_copy_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_copy_tmp_bytes_impl(self, res_size)
    }

    fn ckks_copy<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_copy", dst)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_copy", src)?;
        BE::ckks_copy_impl(self, dst, src, scratch)
    }
}
