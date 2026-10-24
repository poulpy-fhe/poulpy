use crate::{
    CKKSInfos, CKKSResult,
    layouts::{CKKSPreparedLeft, CKKSPreparedRight},
    oep::CKKSMulImpl,
};
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendRef};
use poulpy_hal::layouts::{Module, PrepareHint, ScratchArena};

/// A left operand of `a`'s layout under `hint`, allocated then prepared into from `a`.
pub(crate) fn ckks_prepare_left<BE, A>(
    module: &Module<BE>,
    a: &A,
    hint: PrepareHint,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<CKKSPreparedLeft<BE>>
where
    BE: CKKSMulImpl,
    A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
{
    let mut res = BE::ckks_prepared_left_alloc_from_infos_impl(module, a, hint);
    BE::ckks_prepare_left_into_impl(module, &mut res, a, scratch)?;
    Ok(res)
}

/// As [`ckks_prepare_left`], a right operand.
pub(crate) fn ckks_prepare_right<BE, A>(
    module: &Module<BE>,
    a: &A,
    hint: PrepareHint,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<CKKSPreparedRight<BE>>
where
    BE: CKKSMulImpl,
    A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
{
    let mut res = BE::ckks_prepared_right_alloc_from_infos_impl(module, a, hint);
    BE::ckks_prepare_right_into_impl(module, &mut res, a, scratch)?;
    Ok(res)
}
