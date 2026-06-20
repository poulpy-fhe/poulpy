use anyhow::Result;
use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_core::{
    GLWEShift,
    layouts::{GLWEInfos, LWEInfos},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::GLWEToBackendRef;

use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub, layouts::CKKSCiphertext, layouts::ciphertext::CKKSMaintainOpsDefault};

#[doc(hidden)]
pub trait CKKSRescaleOpsDefault<BE: Backend> {
    /// Increases `ct`'s `log_delta` by `bits`, keeping the encoded message and
    /// `log_budget`, so `effective_k` grows by `bits` (the added precision bits
    /// are zero LSBs). When the storage `max_k` cannot hold the new `effective_k`
    /// the owned buffer is reallocated wider; otherwise it is a pure metadata
    /// update.
    fn ckks_scale_up_default(&self, ct: &mut CKKSCiphertext<Vec<u8>>, bits: usize) -> Result<()>
    where
        Self: CKKSMaintainOpsDefault<BE>,
    {
        if bits == 0 {
            return Ok(());
        }
        let new_log_delta = ct.log_delta() + bits;
        let new_effective_k = new_log_delta + ct.log_budget();
        let required_limbs = new_effective_k.div_ceil(ct.base2k().as_usize());
        if ct.size() < required_limbs {
            self.ckks_reallocate_limbs_checked_default(ct, required_limbs)?;
        }
        ct.set_log_delta(new_log_delta);
        Ok(())
    }

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

    fn ckks_rescale_assign_default<Dst>(&self, ct: &mut Dst, k: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        let log_budget = checked_log_budget_sub("rescale_assign", ct.log_budget(), k)?;
        self.glwe_lsh_assign(ct, k, scratch);
        ct.set_log_budget(log_budget);
        Ok(())
    }

    fn ckks_rescale_into_default<Dst, Src>(
        &self,
        dst: &mut Dst,
        k: usize,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
    {
        let log_budget = checked_log_budget_sub("rescale", src.log_budget(), k)?;
        self.glwe_lsh(dst, src, k, scratch);
        dst.set_meta(src.meta());
        dst.set_log_budget(log_budget);
        Ok(())
    }

    fn ckks_align_pair_default<A, B>(&self, a: &mut A, b: &mut B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        A: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        B: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        if a.log_budget() < b.log_budget() {
            self.ckks_rescale_assign_default(b, b.log_budget() - a.log_budget(), scratch)
        } else {
            self.ckks_rescale_assign_default(a, a.log_budget() - b.log_budget(), scratch)
        }
    }
}
