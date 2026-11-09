use crate::{
    CKKSInfos, CKKSResult,
    api::{CKKSLeftOperandPreparedFactory, CKKSRightOperandPreparedFactory},
    layouts::{CKKSLeftOperandPrepared, CKKSRightOperandPrepared},
    oep::CKKSMulImpl,
};
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendRef};
use poulpy_hal::layouts::{Module, PrepareHint, Ring, ScratchArena};

/// A left operand of `a`'s layout under `hint`, allocated then prepared into from `a`.
pub(crate) fn ckks_left_operand_prepare_alloc_derived<BE, A, Rg: Ring>(
    module: &Module<BE>,
    a: &A,
    hint: PrepareHint,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<CKKSLeftOperandPrepared<BE, Rg>>
where
    BE: CKKSMulImpl<Rg>,
    A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
{
    let mut res = module.ckks_left_operand_prepared_alloc_from_infos(a, hint);
    BE::ckks_left_operand_prepare(module, &mut res, a, scratch)?;
    Ok(res)
}

/// As [`ckks_left_operand_prepare_alloc_derived`], for a right operand.
pub(crate) fn ckks_right_operand_prepare_alloc_derived<BE, A, Rg: Ring>(
    module: &Module<BE>,
    a: &A,
    hint: PrepareHint,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<CKKSRightOperandPrepared<BE, Rg>>
where
    BE: CKKSMulImpl<Rg>,
    A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
{
    let mut res = module.ckks_right_operand_prepared_alloc_from_infos(a, hint);
    BE::ckks_right_operand_prepare(module, &mut res, a, scratch)?;
    Ok(res)
}
