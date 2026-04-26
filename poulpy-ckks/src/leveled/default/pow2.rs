use anyhow::Result;
use poulpy_core::{GLWECopy, GLWEShift, ScratchTakeCore};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{
    CKKSInfos, checked_log_budget_sub,
    layouts::{CKKSCiphertext, ciphertext::CKKSOffset},
};

pub(crate) trait CKKSPow2Default<BE: Backend> {
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

    fn ckks_mul_pow2_into_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let offset = dst.offset_unary(src);
        self.glwe_lsh(dst, src, bits + offset, scratch);
        dst.meta = src.meta();
        dst.meta.log_budget = checked_log_budget_sub("mul_pow2", dst.log_budget(), offset)?;
        Ok(())
    }

    fn ckks_mul_pow2_assign_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.glwe_lsh_assign(dst, bits, scratch);
        Ok(())
    }

    fn ckks_div_pow2_into_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let offset = dst.offset_unary(src);
        self.glwe_lsh(dst, src, offset, scratch);
        dst.meta = src.meta();
        dst.meta.log_budget = checked_log_budget_sub("div_pow2", dst.log_budget(), bits + offset)?;
        Ok(())
    }

    fn ckks_div_pow2_assign_default(&self, dst: &mut CKKSCiphertext<impl DataMut>, bits: usize) -> Result<()> {
        dst.meta.log_budget = checked_log_budget_sub("div_pow2_assign", dst.log_budget(), bits)?;
        Ok(())
    }
}

impl<BE: Backend> CKKSPow2Default<BE> for Module<BE> {}
