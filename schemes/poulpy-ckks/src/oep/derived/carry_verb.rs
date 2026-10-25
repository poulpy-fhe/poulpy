//! Unit shifts compose the selected constant-plaintext add/sub operations.
use crate::api::{CKKSAddOps, CKKSSubOps};
use crate::{CKKSCtBounds, CKKSResult, SetCKKSInfos};
use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};
pub(crate) fn ckks_add_one_assign_derived<BE: Backend, Dst>(
    module: &Module<BE>,
    dst: &mut Dst,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    Module<BE>: CKKSAddOps<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let slot_layout = crate::GLWEPlaintextInfos::slot_layout(dst);
    let (one, mut scratch) =
        crate::reference::carry_verb::take_one_pt(dst.base2k(), slot_layout, dst.log_delta(), scratch.borrow())?;
    match slot_layout {
        crate::SlotLayout::Full => module.ckks_add_pt_const_assign(dst, 0, &one, 0, &mut scratch),
        crate::SlotLayout::Replicated(_) => module.ckks_add_pt_vec_assign(dst, &one, &mut scratch),
    }
}
pub(crate) fn ckks_sub_one_assign_derived<BE: Backend, Dst>(
    module: &Module<BE>,
    dst: &mut Dst,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    Module<BE>: CKKSSubOps<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let slot_layout = crate::GLWEPlaintextInfos::slot_layout(dst);
    let (one, mut scratch) =
        crate::reference::carry_verb::take_one_pt(dst.base2k(), slot_layout, dst.log_delta(), scratch.borrow())?;
    match slot_layout {
        crate::SlotLayout::Full => module.ckks_sub_pt_const_assign(dst, 0, &one, 0, &mut scratch),
        crate::SlotLayout::Replicated(_) => module.ckks_sub_pt_vec_assign(dst, &one, &mut scratch),
    }
}
