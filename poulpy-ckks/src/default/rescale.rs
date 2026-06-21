use anyhow::Result;
use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_core::{
    GLWEShift,
    layouts::{GLWEInfos, LWEInfos},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::GLWEToBackendRef;

use crate::{
    CKKSInfos, SetCKKSInfos, checked_log_budget_sub, layouts::CKKSCiphertext, layouts::ciphertext::CKKSMaintainOpsDefault,
};

#[doc(hidden)]
pub trait CKKSRescaleOpsDefault<BE: Backend> {
    /// Sets `ct`'s encoding scale to `log_delta`, preserving the encoded message
    /// and `log_budget`.
    ///
    /// - **Increase** (`log_delta` larger): extends the precision window with zero
    ///   low-order bits, reallocating the owned buffer wider when the storage
    ///   `max_k` cannot hold the larger `effective_k` (otherwise a metadata-only
    ///   update).
    /// - **Decrease** (`log_delta` smaller): drops the low-order precision bits and
    ///   compacts the storage to the new (smaller) `effective_k`.
    ///
    /// A no-op when `ct` is already at `log_delta`.
    fn ckks_set_log_delta_default(&self, ct: &mut CKKSCiphertext<Vec<u8>>, log_delta: usize) -> Result<()>
    where
        Self: CKKSMaintainOpsDefault<BE>,
    {
        let current = ct.log_delta();
        if log_delta > current {
            let new_effective_k = log_delta + ct.log_budget();
            let required_limbs = new_effective_k.div_ceil(ct.base2k().as_usize());
            if ct.size() < required_limbs {
                self.ckks_reallocate_limbs_checked_default(ct, required_limbs)?;
            }
            ct.set_log_delta(log_delta);
        } else if log_delta < current {
            ct.set_log_delta(log_delta);
            self.ckks_compact_limbs_default(ct)?;
        }
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
