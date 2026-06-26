use anyhow::Result;
use poulpy_core::{GLWEShift, layouts::GLWEToBackendMut};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSScaleManage, checked_log_budget_sub, checked_log_delta_sub};

/// Scale management is a plain torus shift plus a metadata relabel: it never
/// touches the wrapped GLWE's torus width `k`, so `log_budget = k − log_delta`
/// follows the `log_delta` change for free. No backend specialization is needed,
/// so this is a single blanket impl over any `Module<BE>: GLWEShift<BE>` rather
/// than the api/oep/default/delegates dispatch the public ops use.
impl<BE: Backend> CKKSScaleManage<BE> for Module<BE>
where
    Module<BE>: GLWEShift<BE>,
{
    fn ckks_scale_down_assign<Dst>(&self, ct: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        let log_delta = checked_log_delta_sub("scale_down", ct.log_delta(), bits)?;
        self.glwe_rsh(bits, ct, scratch);
        let mut meta = ct.meta();
        meta.log_delta = log_delta;
        ct.set_meta(meta); // `k` unchanged ⇒ log_budget += bits
        Ok(())
    }

    fn ckks_scale_up_assign<Dst>(&self, ct: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        // Errors if there is not `bits` of headroom to absorb the left-shift.
        checked_log_budget_sub("scale_up", ct.log_budget(), bits)?;
        self.glwe_lsh_assign(ct, bits, scratch);
        let mut meta = ct.meta();
        meta.log_delta = ct.log_delta() + bits;
        ct.set_meta(meta); // `k` unchanged ⇒ log_budget -= bits
        Ok(())
    }
}
