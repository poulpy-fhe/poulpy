//! Unit shifts compose the selected `ckks_add_assign` and `ckks_sub_assign`.
use crate::api::{CKKSAddOps, CKKSSubOps};
use crate::layouts::slots::take_one_pt;
use crate::layouts::validation::{checked_one, ensure_scratch};
use crate::{CKKSCtBounds, CKKSResult, SetCKKSInfos};
use poulpy_core::layouts::{GLWEToBackendMut, LWEInfos};
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
    const OP: &str = "ckks_add_one_assign";
    let one = checked_one::<BE, Dst, Rg>(OP, module, &*dst)?;
    ensure_scratch(OP, scratch, module.ckks_add_one_tmp_bytes(dst.max_size()))?;
    let (one, mut scratch) = take_one_pt(dst.base2k(), one, scratch.borrow());
    module.ckks_add_assign(dst, &one, &mut scratch)
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
    const OP: &str = "ckks_sub_one_assign";
    let one = checked_one::<BE, Dst, Rg>(OP, module, &*dst)?;
    ensure_scratch(OP, scratch, module.ckks_sub_one_tmp_bytes(dst.max_size()))?;
    let (one, mut scratch) = take_one_pt(dst.base2k(), one, scratch.borrow());
    module.ckks_sub_assign(dst, &one, &mut scratch)
}
