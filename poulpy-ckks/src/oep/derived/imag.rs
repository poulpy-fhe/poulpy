//! Division by i through the selected multiplication-by-i and negation contracts.
use crate::api::{CKKSImagOps, CKKSNegOps};
use crate::{CKKSCtBounds, CKKSResult, SetCKKSInfos};
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};
pub(crate) fn ckks_div_i_tmp_bytes_derived<BE: Backend>(module: &Module<BE>, res_size: usize) -> usize
where
    Module<BE>: CKKSImagOps<BE> + CKKSNegOps<BE>,
{
    module.ckks_mul_i_tmp_bytes(res_size)
}
pub(crate) fn ckks_div_i_into_derived<BE: Backend, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    Module<BE>: CKKSImagOps<BE> + CKKSNegOps<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSCtBounds,
{
    module.ckks_mul_i_into(dst, src, scratch)?;
    module.ckks_neg_assign(dst)
}
pub(crate) fn ckks_div_i_assign_derived<BE: Backend, Dst>(
    module: &Module<BE>,
    dst: &mut Dst,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    Module<BE>: CKKSImagOps<BE> + CKKSNegOps<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
{
    module.ckks_mul_i_assign(dst, scratch)?;
    module.ckks_neg_assign(dst)
}
