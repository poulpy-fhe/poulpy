use anyhow::Result;
use poulpy_core::{
    GLWENegate, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, ScratchArena},
};

use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_unary};

pub trait CKKSNegDefault<BE: Backend> {
    fn ckks_neg_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_neg_into_default<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWENegate<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let offset = ckks_offset_unary(dst, src);
        if offset != 0 {
            self.glwe_lsh(dst, src, offset, scratch);
            dst.set_meta(src.meta());
            dst.set_log_budget(checked_log_budget_sub("neg", src.log_budget(), offset)?);
            self.glwe_negate_assign(dst);
        } else {
            self.glwe_negate(dst, src);
            dst.set_meta(src.meta());
        }
        Ok(())
    }

    fn ckks_neg_assign_default<Dst>(&self, dst: &mut Dst) -> Result<()>
    where
        Self: GLWENegate<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        self.glwe_negate_assign(dst);
        Ok(())
    }
}
