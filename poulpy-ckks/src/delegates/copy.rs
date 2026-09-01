use crate::CKKSResult as Result;
use poulpy_core::{
    GLWECopy, GLWEShift,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::Normalized;
use poulpy_hal::layouts::{Backend, FitsIn, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSCopyOps, oep::CKKSCopyImpl};

impl<BE: Backend + CKKSCopyImpl<BE>> CKKSCopyOps<BE> for Module<BE>
where
    Module<BE>: GLWECopy<BE> + GLWEShift<BE>,
{
    fn ckks_copy_tmp_bytes(&self) -> usize {
        BE::ckks_copy_tmp_bytes_impl(self)
    }

    fn ckks_copy<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        <Src as GLWEToBackendRef<BE>>::State: FitsIn<<Dst as GLWEToBackendRef<BE>>::State>,
    {
        BE::ckks_copy_impl(self, dst, src, scratch)
    }
}
