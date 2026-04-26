use anyhow::Result;
use poulpy_core::{GLWEShift, ScratchTakeCore};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{CKKSInfos, checked_log_budget_sub, layouts::CKKSCiphertext};

#[doc(hidden)]
pub(crate) trait CKKSRescaleOpsDefault<BE: Backend> {
    fn ckks_rescale_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_align_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_rescale_assign_default(
        &self,
        ct: &mut CKKSCiphertext<impl DataMut>,
        k: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let log_budget = checked_log_budget_sub("rescale_assign", ct.log_budget(), k)?;
        self.glwe_lsh_assign(ct, k, scratch);
        ct.meta.log_budget = log_budget;
        Ok(())
    }

    fn ckks_rescale_into_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        k: usize,
        src: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let log_budget = checked_log_budget_sub("rescale", src.log_budget(), k)?;
        self.glwe_lsh(dst, src, k, scratch);
        dst.meta = src.meta();
        dst.meta.log_budget = log_budget;
        Ok(())
    }

    fn ckks_align_assign_default(
        &self,
        a: &mut CKKSCiphertext<impl DataMut>,
        b: &mut CKKSCiphertext<impl DataMut>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        if a.log_budget() < b.log_budget() {
            self.ckks_rescale_assign_default(b, b.log_budget() - a.log_budget(), scratch)
        } else {
            self.ckks_rescale_assign_default(a, a.log_budget() - b.log_budget(), scratch)
        }
    }
}

impl<BE: Backend> CKKSRescaleOpsDefault<BE> for Module<BE> {}
