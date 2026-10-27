use crate::{
    CKKSInfos, CKKSResult,
    layouts::{CKKSModuleAlloc, CKKSPreparedLeft, CKKSPreparedRight},
    oep::CKKSMulImpl,
};
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendRef};
use poulpy_hal::layouts::{Module, PrepareHint, ScratchArena};

/// A left operand of `a`'s layout under `hint`, allocated then prepared into from `a`.
pub(crate) fn ckks_prepare_left_derived<BE, A>(
    module: &Module<BE>,
    a: &A,
    hint: PrepareHint,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<CKKSPreparedLeft<BE>>
where
    BE: CKKSMulImpl,
    A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
{
    let mut res = module.ckks_prepared_left_alloc_from_infos(a, hint);
    BE::ckks_prepare_left_into(module, &mut res, a, scratch)?;
    Ok(res)
}

/// As [`ckks_prepare_left_derived`], for a right operand.
pub(crate) fn ckks_prepare_right_derived<BE, A>(
    module: &Module<BE>,
    a: &A,
    hint: PrepareHint,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<CKKSPreparedRight<BE>>
where
    BE: CKKSMulImpl,
    A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
{
    let mut res = module.ckks_prepared_right_alloc_from_infos(a, hint);
    BE::ckks_prepare_right_into(module, &mut res, a, scratch)?;
    Ok(res)
}
