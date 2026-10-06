use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSCopyOps, oep::CKKSCopyImpl};

impl<BE: Backend + CKKSCopyImpl> CKKSCopyOps<BE> for Module<BE> {
    fn ckks_copy_tmp_bytes<Dst: CKKSCtBounds, Src: CKKSCtBounds>(&self, dst: &Dst, src: &Src) -> usize {
        BE::ckks_copy_tmp_bytes_impl(self, dst, src)
    }

    fn ckks_copy<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_copy_impl(self, dst, src, scratch)?;
        let rank = dst.to_backend_ref().rank().as_usize();
        dst.set_noise(src.noise().map(|noise| noise.with_rank(rank)));
        Ok(())
    }
}
