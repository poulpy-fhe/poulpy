//! Direct unary operation helpers and plan constants.

mod unary;

use anyhow::Result;
use poulpy_ckks::{
    CKKSLayout,
    layouts::{CKKSPlaintext, CKKSScalar, ckks_encode_host_constants},
};
use poulpy_core::layouts::Base2K;
use poulpy_hal::{
    AlignedBuf,
    layouts::{HostBytesBackend, Module, Ring},
};

pub(crate) use unary::{declare_unary_op, define_unary_plan, impl_unary_op};

/// The real constant `value` as a plaintext at the precision and slot layout of `coeff_meta`.
pub(crate) fn encode_constant<F, Rg: Ring>(
    module: &Module<HostBytesBackend>,
    base2k: Base2K,
    coeff_meta: CKKSLayout,
    value: F,
) -> Result<CKKSPlaintext<AlignedBuf, i64, Rg>>
where
    F: CKKSScalar,
{
    let mut constants = ckks_encode_host_constants::<F, Rg>(module, base2k, coeff_meta.into(), &[value])?;
    Ok(constants.pop().expect("one value, one constant"))
}
