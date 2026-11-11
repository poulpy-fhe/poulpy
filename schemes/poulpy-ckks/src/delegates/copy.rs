use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSCopyOps, oep::CKKSCopyImpl};

impl<BE: Backend + CKKSCopyImpl<Rg>, Rg: Ring> CKKSCopyOps<BE, Rg> for Module<BE> {
    fn ckks_copy_tmp_bytes<Dst: CKKSCtBounds, Src: CKKSCtBounds>(&self, dst: &Dst, src: &Src) -> usize {
        BE::ckks_copy_tmp_bytes(self, dst, src)
    }

    fn ckks_copy<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    {
        BE::ckks_copy(self, dst, src, scratch)
    }
}
