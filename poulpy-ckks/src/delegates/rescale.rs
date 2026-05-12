use anyhow::Result;
use poulpy_core::{
    GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSRescaleOps, oep::CKKSRescaleImpl};

impl<BE: Backend + CKKSRescaleImpl<BE>> CKKSRescaleOps<BE> for Module<BE>
where
    Module<BE>: GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_rescale_tmp_bytes(&self) -> usize {
        BE::ckks_rescale_tmp_bytes(self)
    }

    fn ckks_rescale_assign<Dst>(&self, ct: &mut Dst, k: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_rescale_assign(self, ct, k, scratch)
    }

    fn ckks_rescale_into<Dst, Src>(&self, dst: &mut Dst, k: usize, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_rescale_into(self, dst, k, src, scratch)
    }

    fn ckks_align_pair<A, B>(&self, a: &mut A, b: &mut B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        A: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        B: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_align_pair(self, a, b, scratch)
    }

    fn ckks_align_tmp_bytes(&self) -> usize {
        BE::ckks_align_tmp_bytes(self)
    }
}
