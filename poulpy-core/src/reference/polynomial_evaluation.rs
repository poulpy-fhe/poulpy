//! Deprecated HAL bounds of the polynomial evaluation, whose arithmetic policy
//! is [`GLWEBSGSOps`](crate::api::GLWEBSGSOps).

use crate::{GLWEBytesOf, GLWENormalize};
use poulpy_hal::{
    api::{
        CnvOps, CnvPVecBytesOf, ModuleN, VecZnxAddAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopy, VecZnxDftBytesOf, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxNegate, VecZnxNormalizeAssign,
        VecZnxNormalizeTmpBytes, VecZnxSubAssign,
    },
    layouts::{Backend, Ring, Standard},
};

/// Former bundle of the HAL bounds behind the prepared-right tensor product.
///
/// The product now dispatches through
/// [`GLWETensoring::glwe_tensor_apply_prepared_right`](crate::GLWETensoring::glwe_tensor_apply_prepared_right),
/// so neither the BSGS engine nor its scheme implementations need these bounds.
#[deprecated(note = "bound the module by `GLWETensoring` and the HAL operations it calls directly")]
pub trait GiantStepTensorBounds<BE: Backend, Rg: Ring = Standard>:
    Sized
    + ModuleN
    + CnvPVecBytesOf
    + VecZnxDftBytesOf
    + VecZnxBigBytesOf
    + VecZnxIdftApply<BE, Rg>
    + VecZnxIdftApplyTmpBytes
    + VecZnxBigNormalize<BE, Rg>
    + CnvOps<BE, Rg>
    + VecZnxSubAssign<BE, Rg>
    + VecZnxAddAssign<BE, Rg>
    + VecZnxBigNormalizeTmpBytes
    + VecZnxCopy<BE, Rg>
    + VecZnxNegate<BE, Rg>
    + VecZnxNormalizeAssign<BE, Rg>
    + VecZnxNormalizeTmpBytes
    + GLWENormalize<BE, Rg>
    + GLWEBytesOf<BE>
{
}

#[allow(deprecated)]
impl<BE: Backend, M, Rg: Ring> GiantStepTensorBounds<BE, Rg> for M where
    M: Sized
        + ModuleN
        + CnvPVecBytesOf
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApply<BE, Rg>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE, Rg>
        + CnvOps<BE, Rg>
        + VecZnxSubAssign<BE, Rg>
        + VecZnxAddAssign<BE, Rg>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopy<BE, Rg>
        + VecZnxNegate<BE, Rg>
        + VecZnxNormalizeAssign<BE, Rg>
        + VecZnxNormalizeTmpBytes
        + GLWENormalize<BE, Rg>
        + GLWEBytesOf<BE>
{
}
