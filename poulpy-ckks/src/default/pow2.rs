use anyhow::Result;
use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_core::{
    GLWECopy, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEInfos, LWEInfos},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::GLWEToBackendRef;

use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_unary};

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
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let offset = ckks_offset_unary(dst, src);
        self.glwe_lsh(dst, src, bits + offset, scratch);
        dst.set_meta(src.meta());
        dst.set_log_budget(checked_log_budget_sub("mul_pow2", dst.log_budget(), offset)?);
        Ok(())
    }

    fn ckks_mul_pow2_assign_default<Dst>(&self, dst: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
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
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let offset = ckks_offset_unary(dst, src);
        self.glwe_lsh(dst, src, offset, scratch);
        dst.set_meta(src.meta());
        dst.set_log_budget(checked_log_budget_sub("div_pow2", dst.log_budget(), bits + offset)?);
        dst.set_log_delta(dst.log_delta() + bits);
        Ok(())
    }

    fn ckks_div_pow2_assign_default<Dst>(&self, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        dst.set_log_budget(checked_log_budget_sub("div_pow2_assign", dst.log_budget(), bits)?);
        Ok(())
    }
}
