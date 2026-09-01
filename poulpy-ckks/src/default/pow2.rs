use crate::CKKSResult as Result;
use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_core::{GLWECopy, GLWEShift, layouts::GLWEInfos};
use poulpy_hal::layouts::Normalized;
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::GLWEToBackendRef;

use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub};

pub trait CKKSPow2Default<BE: Backend> {
    fn ckks_mul_pow2_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_div_pow2_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_mul_pow2_into_default<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos + CKKSInfos,
    {
        crate::ckks_shift_stamp_unary(self, "mul_pow2", dst, src, bits, 0, scratch)?;
        Ok(())
    }

    fn ckks_mul_pow2_assign_default<Dst>(&self, dst: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSInfos + SetCKKSInfos,
    {
        self.glwe_lsh_assign(dst, bits, scratch);
        Ok(())
    }

    fn ckks_div_pow2_into_default<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWECopy<BE>,
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos + CKKSInfos,
    {
        crate::ckks_shift_stamp_unary(self, "div_pow2", dst, src, 0, bits, scratch)?;
        dst.set_log_delta(dst.log_delta() + bits);
        Ok(())
    }

    fn ckks_div_pow2_assign_default<Dst>(&self, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSInfos + SetCKKSInfos,
    {
        // Lossless relabel, mirroring `_into` with `offset = 0`: the `bits`
        // charged to the budget move under `log_delta`, leaving `k` unchanged
        // (`set_log_delta` preserves the budget by shifting `k` back up).
        dst.set_log_budget(checked_log_budget_sub("div_pow2_assign", dst.log_budget(), bits)?);
        dst.set_log_delta(dst.log_delta() + bits);
        Ok(())
    }
}
