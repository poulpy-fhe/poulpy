use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSRoundOps, oep::CKKSRoundImpl};

impl<BE: Backend + CKKSRoundImpl> CKKSRoundOps<BE> for Module<BE> {
    fn ckks_round_tmp_bytes<Dst: CKKSCtBounds, Src: CKKSCtBounds>(&self, dst: &Dst, src: &Src) -> usize {
        BE::ckks_round_tmp_bytes_impl(self, dst, src)
    }

    fn ckks_round_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_round_into_impl(self, dst, src, scratch)
    }
}
