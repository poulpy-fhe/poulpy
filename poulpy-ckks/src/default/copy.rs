use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, ScratchArena},
};

use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_unary};

pub trait CKKSCopyDefault<BE: Backend> {
    fn ckks_copy_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_copy_default<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let offset = ckks_offset_unary(dst, src);
        if offset == 0 {
            self.glwe_copy(dst, src);
            dst.set_meta(src.meta());
        } else {
            self.glwe_lsh(dst, src, offset, scratch);
            dst.set_meta(src.meta());
            dst.set_log_budget(checked_log_budget_sub("copy", src.log_budget(), offset)?);
        }
        Ok(())
    }
}
