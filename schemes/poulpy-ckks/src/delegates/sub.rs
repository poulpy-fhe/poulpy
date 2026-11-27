use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::api::CKKSSubOps;

use crate::{CKKSCtBounds, SetCKKSInfos, oep::CKKSSubImpl};

impl<BE: Backend + CKKSSubImpl<Rg>, Rg: Ring> CKKSSubOps<BE, Rg> for Module<BE> {
    fn ckks_sub_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_sub_tmp_bytes(self, res_size)
    }

    fn ckks_sub_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        B: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    {
        BE::ckks_sub_into(self, dst, a, b, scratch)
    }

    fn ckks_sub_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    {
        BE::ckks_sub_assign(self, dst, a, scratch)
    }

    fn ckks_sub_one_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_sub_one_tmp_bytes(self, res_size)
    }

    fn ckks_sub_one_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_sub_one_assign(self, dst, scratch)
    }
}
