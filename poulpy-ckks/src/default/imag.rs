use crate::CKKSResult as Result;
use poulpy_core::{
    GLWECopy, GLWENegate, GLWERotate, GLWEShift,
    layouts::{GLWEInfos, GLWEToBackendMut},
};
use poulpy_hal::layouts::Normalized;
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, FitsIn, ScratchArena},
};

use crate::GLWEToBackendRef;
use crate::{CKKSInfos, SetCKKSInfos, SlotsKind, checked_log_budget_sub, ckks_offset_unary};

pub trait CKKSImagDefault<BE: Backend> {
    fn ckks_mul_i_tmp_bytes_default(&self) -> usize
    where
        Self: GLWERotate<BE> + GLWEShift<BE>,
    {
        self.glwe_rotate_tmp_bytes().max(self.glwe_shift_tmp_bytes())
    }

    fn ckks_div_i_tmp_bytes_default(&self) -> usize
    where
        Self: GLWERotate<BE> + GLWEShift<BE>,
    {
        self.glwe_rotate_tmp_bytes().max(self.glwe_shift_tmp_bytes())
    }

    fn ckks_mul_i_into_default<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWERotate<BE> + GLWEShift<BE> + ModuleN,
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos + CKKSInfos,
        <Src as GLWEToBackendRef<BE>>::State: FitsIn<<Dst as GLWEToBackendRef<BE>>::State>,
    {
        let offset = ckks_offset_unary(dst, src);
        // Validate before mutating: on error `dst` must remain untouched.
        let log_budget = checked_log_budget_sub("mul_i", src.log_budget(), offset)?;
        let k = (self.n() / 2) as i64;
        if offset == 0 {
            self.glwe_rotate(k, dst, src);
        } else {
            self.glwe_lsh(dst, src, offset, scratch);
            self.glwe_rotate_assign(k, dst, scratch);
        }
        dst.set_meta(src.meta());
        dst.set_log_budget(log_budget);
        // Multiplying by `i` maps the reals onto the imaginary axis.
        dst.set_slots(SlotsKind::Complex);
        Ok(())
    }

    fn ckks_mul_i_assign_default<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWERotate<BE> + ModuleN,
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSInfos + SetCKKSInfos,
    {
        self.glwe_rotate_assign((self.n() / 2) as i64, dst, scratch);
        dst.set_slots(SlotsKind::Complex);
        Ok(())
    }

    fn ckks_div_i_into_default<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWENegate<BE> + GLWERotate<BE> + GLWEShift<BE> + ModuleN,
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos + CKKSInfos,
    {
        self.ckks_mul_i_into_default(dst, src, scratch)?;
        self.glwe_negate_assign(dst);
        Ok(())
    }

    fn ckks_div_i_assign_default<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWENegate<BE> + GLWERotate<BE> + ModuleN,
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSInfos + SetCKKSInfos,
    {
        self.ckks_mul_i_assign_default(dst, scratch)?;
        self.glwe_negate_assign(dst);
        Ok(())
    }
}
