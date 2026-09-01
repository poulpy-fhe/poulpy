use crate::CKKSResult as Result;
use poulpy_core::{
    GLWENegate, GLWEShift,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::Normalized;
use poulpy_hal::layouts::{Backend, FitsIn, ScratchArena};

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
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos + CKKSInfos,
        <Src as GLWEToBackendRef<BE>>::State: FitsIn<<Dst as GLWEToBackendRef<BE>>::State>,
    {
        let offset = ckks_offset_unary(dst, src);
        if offset != 0 {
            // Validate before mutating: on error `dst` must remain untouched.
            let log_budget = checked_log_budget_sub("neg", src.log_budget(), offset)?;
            self.glwe_lsh(dst, src, offset, scratch);
            dst.set_meta(src.meta());
            dst.set_log_budget(log_budget);
            self.glwe_negate_assign(dst);
        } else {
            self.glwe_negate(dst, src);
            dst.set_meta(src.meta());
            // `set_meta` no longer carries the budget: propagate `src`'s width
            // explicitly so a wider `dst` does not keep a stale `k`.
            dst.set_log_budget(src.log_budget());
        }
        Ok(())
    }

    fn ckks_neg_assign_default<Dst>(&self, dst: &mut Dst) -> Result<()>
    where
        Self: GLWENegate<BE>,
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSInfos + SetCKKSInfos,
    {
        self.glwe_negate_assign(dst);
        Ok(())
    }
}
