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
    let one = crate::reference::carry_verb::ckks_one_pt(module, dst.base2k())?;
    module.ckks_add_pt_const_assign(dst, 0, &one, 0, scratch)
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
    let one = crate::reference::carry_verb::ckks_one_pt(module, dst.base2k())?;
    module.ckks_sub_pt_const_assign(dst, 0, &one, 0, scratch)
}
