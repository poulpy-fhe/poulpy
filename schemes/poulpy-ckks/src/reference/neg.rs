use crate::CKKSResult as Result;
use poulpy_core::{
    GLWENegate, GLWEShift,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Ring, ScratchArena, Standard},
};

use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_unary, ckks_unary_exact};

pub trait CKKSNegReference<BE: Backend, Rg: Ring = Standard>: ModuleN {
    fn ckks_neg_tmp_bytes_reference(&self, res_size: usize) -> usize
    where
        Self: GLWEShift<BE, Rg>,
    {
        self.glwe_shift_tmp_bytes(res_size)
    }

    fn ckks_neg_into_reference<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWENegate<BE, Rg> + GLWEShift<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, Rg> + GLWEInfos + CKKSInfos,
    {
        crate::ensure_operand_degrees("ckks_neg_into", self, [dst.n(), src.n()])?;
        let offset = ckks_offset_unary(dst, src);
        if !ckks_unary_exact(dst, src) {
            // Validate before mutating: on error `dst` must remain untouched.
            let log_budget = checked_log_budget_sub("ckks_neg_into", src.log_budget(), offset)?;
            // Stamp before the shift: it normalizes at `dst.k()`.
            dst.set_meta(src.meta());
            dst.set_log_budget(log_budget);
            self.glwe_lsh(dst, src, offset, scratch);
            self.glwe_negate_assign(dst);
        } else {
            dst.set_meta(src.meta());
            // `set_meta` no longer carries the budget: propagate `src`'s width
            // explicitly so a wider `dst` does not keep a stale `k`.
            dst.set_log_budget(src.log_budget());
            self.glwe_negate(dst, src);
        }
        Ok(())
    }

    fn ckks_neg_assign_reference<Dst>(&self, dst: &mut Dst) -> Result<()>
    where
        Self: GLWENegate<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos,
    {
        crate::ensure_operand_degrees("ckks_neg_assign", self, [dst.n()])?;
        self.glwe_negate_assign(dst);
        Ok(())
    }
}

impl<BE: Backend, Rg: Ring> CKKSNegReference<BE, Rg> for poulpy_hal::layouts::Module<BE> {}
