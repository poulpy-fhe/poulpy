use anyhow::Result;
use poulpy_core::{
    GLWENegate, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSCtBounds, SetCKKSInfos, oep::CKKSNegImpl};

use crate::api::CKKSNegOps;

impl<BE: Backend + CKKSNegImpl<BE>> CKKSNegOps<BE> for Module<BE>
where
    Module<BE>: GLWENegate<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_neg_tmp_bytes(&self) -> usize {
        BE::ckks_neg_tmp_bytes(self)
    }

    fn ckks_neg_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_neg_into(self, dst, src, scratch)
    }

    fn ckks_neg_assign<Dst>(&self, dst: &mut Dst) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_neg_assign(self, dst)
    }
}
