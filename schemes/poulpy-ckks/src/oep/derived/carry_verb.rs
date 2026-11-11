//! Unit shifts compose the selected constant-plaintext add/sub operations.
use crate::api::{CKKSAddOps, CKKSSubOps};
use crate::{CKKSCtBounds, CKKSResult, SetCKKSInfos};
use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};
pub(crate) fn ckks_add_one_assign_derived<BE: Backend, Dst, Rg: Ring>(
    module: &Module<BE>,
    dst: &mut Dst,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    Module<BE>: CKKSAddOps<BE, Rg>,
    Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
{
    let one = crate::layouts::slots::ckks_plaintext_one(module, dst.base2k())?;
    module.ckks_add_pt_const_assign(dst, 0, &one, 0, scratch)
}
pub(crate) fn ckks_sub_one_assign_derived<BE: Backend, Dst, Rg: Ring>(
    module: &Module<BE>,
    dst: &mut Dst,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    Module<BE>: CKKSSubOps<BE, Rg>,
    Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
{
    let one = crate::layouts::slots::ckks_plaintext_one(module, dst.base2k())?;
    module.ckks_sub_pt_const_assign(dst, 0, &one, 0, scratch)
}
