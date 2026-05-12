use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSCopyOps, oep::CKKSCopyImpl};

impl<BE: Backend + CKKSCopyImpl<BE>> CKKSCopyOps<BE> for Module<BE>
where
    Module<BE>: GLWECopy<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_copy_tmp_bytes(&self) -> usize {
        BE::ckks_copy_tmp_bytes(self)
    }

    fn ckks_copy<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_copy(self, dst, src, scratch)
    }
}
