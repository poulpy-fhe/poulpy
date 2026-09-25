use crate::CKKSResult as Result;
use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_core::{GLWEAdd, GLWECopy, GLWEShift, layouts::GLWEInfos};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::GLWEToBackendRef;

use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_unary};

pub trait CKKSPow2Reference<BE: Backend> {
    fn ckks_mul_pow2_tmp_bytes_reference(&self, res_size: usize) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes(res_size)
    }

    fn ckks_div_pow2_tmp_bytes_reference(&self, res_size: usize) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes(res_size)
    }

    fn ckks_mul_pow2_into_reference<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    {
        crate::ckks_shift_stamp_unary(self, "mul_pow2", dst, src, bits, 0, 0, scratch)?;
        Ok(())
    }

    fn ckks_mul_pow2_assign_reference<Dst>(&self, dst: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
    {
        self.glwe_lsh_assign(dst, bits, scratch);
        Ok(())
    }

    fn ckks_double_into_reference<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    {
        if ckks_offset_unary(dst, src) != 0 {
            return self.ckks_mul_pow2_into_reference(dst, src, 1, scratch);
        }
        self.glwe_add_into(dst, src, src);
        dst.set_meta(src.meta());
        dst.set_log_budget(src.log_budget());
        Ok(())
    }

    fn ckks_div_pow2_into_reference<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWECopy<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    {
        // The `bits` charged to the budget move under `log_delta` inside the
        // stamp, so the shift normalizes at the width the result reports.
        crate::ckks_shift_stamp_unary(self, "div_pow2", dst, src, 0, bits, bits, scratch)?;
        Ok(())
    }

    fn ckks_div_pow2_assign_reference<Dst>(&self, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
    {
        // Lossless relabel, mirroring `_into` with `offset = 0`: the `bits`
        // charged to the budget move under `log_delta`, leaving `k`, and so the
        // canonical flag, unchanged.
        checked_log_budget_sub("div_pow2_assign", dst.log_budget(), bits)?;
        let mut meta = dst.meta();
        meta.log_delta += bits;
        dst.set_meta(meta);
        Ok(())
    }
}

impl<BE: Backend> CKKSPow2Reference<BE> for poulpy_hal::layouts::Module<BE> {}
