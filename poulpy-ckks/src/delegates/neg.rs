use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, oep::CKKSNegImpl};

use crate::api::CKKSNegOps;

impl<BE: Backend + CKKSNegImpl> CKKSNegOps<BE> for Module<BE> {
    fn ckks_neg_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_neg_tmp_bytes_impl(self, res_size)
    }

    fn ckks_neg_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_neg_into_impl(self, dst, src, scratch)
    }

    fn ckks_neg_assign<Dst>(&self, dst: &mut Dst) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_neg_assign_impl(self, dst)
    }
}
