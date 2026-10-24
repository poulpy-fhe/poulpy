//! Deprecated HAL bounds of the polynomial evaluation, whose arithmetic policy
//! is [`BSGSOps`](crate::api::BSGSOps).

use crate::{GLWEBytesOf, GLWENormalize};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, VecZnxAddAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopy, VecZnxDftBytesOf, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxNegate, VecZnxNormalizeAssign,
        VecZnxNormalizeTmpBytes, VecZnxSubAssign,
    },
    layouts::Backend,
};

/// Former bundle of the HAL bounds behind the prepared-right tensor product.
///
/// The product now dispatches through
/// [`GLWETensoring::glwe_tensor_apply_prepared_right`](crate::GLWETensoring::glwe_tensor_apply_prepared_right),
/// so neither the BSGS engine nor its scheme implementations need these bounds.
#[deprecated(note = "bound the module by `GLWETensoring` and the HAL operations it calls directly")]
pub trait GiantStepTensorBounds<BE: Backend>:
    Sized
    + ModuleN
    + CnvPVecBytesOf
    + VecZnxDftBytesOf
    + VecZnxBigBytesOf
    + VecZnxIdftApply<BE>
    + VecZnxIdftApplyTmpBytes
    + VecZnxBigNormalize<BE>
    + Convolution<BE>
    + VecZnxSubAssign<BE>
    + VecZnxAddAssign<BE>
    + VecZnxBigNormalizeTmpBytes
    + VecZnxCopy<BE>
    + VecZnxNegate<BE>
    + VecZnxNormalizeAssign<BE>
    + VecZnxNormalizeTmpBytes
    + GLWENormalize<BE>
    + GLWEBytesOf<BE>
{
}

#[allow(deprecated)]
impl<BE: Backend, M> GiantStepTensorBounds<BE> for M where
    M: Sized
        + ModuleN
        + CnvPVecBytesOf
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + Convolution<BE>
        + VecZnxSubAssign<BE>
        + VecZnxAddAssign<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopy<BE>
        + VecZnxNegate<BE>
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes
        + GLWENormalize<BE>
        + GLWEBytesOf<BE>
{
}
