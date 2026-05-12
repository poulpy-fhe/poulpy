use anyhow::Result;
use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_core::{
    GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEInfos, LWEInfos},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::GLWEToBackendRef;

use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub};

#[doc(hidden)]
pub trait CKKSRescaleOpsDefault<BE: Backend> {
    fn ckks_rescale_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_align_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_rescale_assign_default<Dst>(&self, ct: &mut Dst, k: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let log_budget = checked_log_budget_sub("rescale_assign", ct.log_budget(), k)?;
        self.glwe_lsh_assign(ct, k, scratch);
        ct.set_log_budget(log_budget);
        Ok(())
    }

    fn ckks_rescale_into_default<Dst, Src>(
        &self,
        dst: &mut Dst,
        k: usize,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let log_budget = checked_log_budget_sub("rescale", src.log_budget(), k)?;
        self.glwe_lsh(dst, src, k, scratch);
        dst.set_meta(src.meta());
        dst.set_log_budget(log_budget);
        Ok(())
    }

    fn ckks_align_pair_default<A, B>(&self, a: &mut A, b: &mut B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        A: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        B: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        if a.log_budget() < b.log_budget() {
            self.ckks_rescale_assign_default(b, b.log_budget() - a.log_budget(), scratch)
        } else {
            self.ckks_rescale_assign_default(a, a.log_budget() - b.log_budget(), scratch)
        }
    }
}
