use crate::{
    CKKSInfos, CKKSResult,
    api::{CKKSLeftOperandPreparedFactory, CKKSRightOperandPreparedFactory},
    layouts::{CKKSLeftOperandPrepared, CKKSRightOperandPrepared},
    oep::CKKSMulImpl,
};
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendRef};
use poulpy_hal::layouts::{Module, PrepareHint, ScratchArena};

/// A left operand of `a`'s layout under `hint`, allocated then prepared into from `a`.
pub(crate) fn ckks_left_operand_prepare_alloc_derived<BE, A>(
    module: &Module<BE>,
    a: &A,
    hint: PrepareHint,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<CKKSLeftOperandPrepared<BE>>
where
    BE: CKKSMulImpl,
    A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
{
    let mut res = module.ckks_left_operand_prepared_alloc_from_infos(a, hint);
    BE::ckks_left_operand_prepare(module, &mut res, a, scratch)?;
    Ok(res)
}

/// As [`ckks_left_operand_prepare_alloc_derived`], for a right operand.
pub(crate) fn ckks_right_operand_prepare_alloc_derived<BE, A>(
    module: &Module<BE>,
    a: &A,
    hint: PrepareHint,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<CKKSRightOperandPrepared<BE>>
where
    BE: CKKSMulImpl,
    A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
{
    let mut res = module.ckks_right_operand_prepared_alloc_from_infos(a, hint);
    BE::ckks_right_operand_prepare(module, &mut res, a, scratch)?;
    Ok(res)
}
