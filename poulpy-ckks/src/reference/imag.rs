use crate::CKKSResult as Result;
use poulpy_core::{
    GLWERotate, GLWEShift,
    layouts::{GLWEInfos, GLWEToBackendMut},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, ScratchArena},
};

use crate::GLWEToBackendRef;
use crate::{CKKSInfos, SetCKKSInfos, SlotsKind, checked_log_budget_sub, ckks_offset_unary};

/// Multiplication and division by `i` using core monomial rotations.
///
/// In the ring modulo `X^N + 1`, multiplication by `i` uses `X^(N/2)`
/// and division by `i` uses `X^(-N/2)`.
pub trait CKKSImagReference<BE: Backend> {
    fn ckks_mul_i_tmp_bytes_reference(&self, res_size: usize) -> usize
    where
        Self: GLWERotate<BE> + GLWEShift<BE>,
    {
        self.glwe_rotate_tmp_bytes().max(self.glwe_shift_tmp_bytes(res_size))
    }

    fn ckks_mul_i_into_reference<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWERotate<BE> + GLWEShift<BE> + ModuleN,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    {
        let offset = ckks_offset_unary(dst, src);
        // Validate before mutating: on error `dst` must remain untouched.
        let log_budget = checked_log_budget_sub("mul_i", src.log_budget(), offset)?;
        let k = (self.n() / 2) as i64;
        // Stamp before the write: the shift normalizes at `dst.k()`.
        dst.set_meta(src.meta());
        dst.set_log_budget(log_budget);
        // Multiplying by `i` maps the reals onto the imaginary axis.
        dst.set_slots(SlotsKind::Complex);
        if offset == 0 {
            self.glwe_rotate(k, dst, src);
        } else {
            self.glwe_lsh(dst, src, offset, scratch);
            self.glwe_rotate_assign(k, dst, scratch);
        }
        Ok(())
    }

    fn ckks_mul_i_assign_reference<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWERotate<BE> + ModuleN,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
    {
        self.glwe_rotate_assign((self.n() / 2) as i64, dst, scratch);
        dst.set_slots(SlotsKind::Complex);
        Ok(())
    }

    fn ckks_div_i_tmp_bytes_reference(&self, res_size: usize) -> usize
    where
        Self: GLWERotate<BE> + GLWEShift<BE>,
    {
        self.glwe_rotate_tmp_bytes().max(self.glwe_shift_tmp_bytes(res_size))
    }

    fn ckks_div_i_into_reference<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWERotate<BE> + GLWEShift<BE> + ModuleN,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    {
        let offset = ckks_offset_unary(dst, src);
        // Validate before mutating: on error `dst` must remain untouched.
        let log_budget = checked_log_budget_sub("div_i", src.log_budget(), offset)?;
        let k = -((self.n() / 2) as i64);
        // Stamp before the write: the shift normalizes at `dst.k()`.
        dst.set_meta(src.meta());
        dst.set_log_budget(log_budget);
        // Dividing by `i` maps the reals onto the imaginary axis.
        dst.set_slots(SlotsKind::Complex);
        if offset == 0 {
            self.glwe_rotate(k, dst, src);
        } else {
            self.glwe_lsh(dst, src, offset, scratch);
            self.glwe_rotate_assign(k, dst, scratch);
        }
        Ok(())
    }

    fn ckks_div_i_assign_reference<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWERotate<BE> + ModuleN,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
    {
        self.glwe_rotate_assign(-((self.n() / 2) as i64), dst, scratch);
        dst.set_slots(SlotsKind::Complex);
        Ok(())
    }
}

impl<BE: Backend> CKKSImagReference<BE> for poulpy_hal::layouts::Module<BE> {}
