use poulpy_hal::layouts::Backend;

use super::{
    CKKSAddImpl, CKKSConjugateImpl, CKKSCopyImpl, CKKSEncryptionImpl, CKKSImagImpl, CKKSMulImpl, CKKSNegImpl,
    CKKSPlaintextZnxImpl, CKKSPow2Impl, CKKSRescaleImpl, CKKSRotateImpl, CKKSSubImpl,
};

/// Aggregate CKKS dispatch surface.
///
/// Concrete APIs can depend on narrower `CKKS*Impl` family traits. This
/// aggregate trait remains useful for composite operations that span multiple
/// CKKS families and for broad test/backend capability bundles.
///
/// # Safety
///
/// Implementations must satisfy the contracts of all bundled `CKKS*Impl`
/// traits, including the HAL-level invariants implied by their method
/// signatures.
pub unsafe trait CKKSImpl<BE: Backend>:
    Backend
    + CKKSPlaintextZnxImpl<BE>
    + CKKSCopyImpl<BE>
    + CKKSAddImpl<BE>
    + CKKSEncryptionImpl<BE>
    + CKKSSubImpl<BE>
    + CKKSNegImpl<BE>
    + CKKSPow2Impl<BE>
    + CKKSImagImpl<BE>
    + CKKSRescaleImpl<BE>
    + CKKSRotateImpl<BE>
    + CKKSConjugateImpl<BE>
    + CKKSMulImpl<BE>
{
}

unsafe impl<BE: Backend> CKKSImpl<BE> for BE where
    BE: CKKSPlaintextZnxImpl<BE>
        + CKKSCopyImpl<BE>
        + CKKSAddImpl<BE>
        + CKKSEncryptionImpl<BE>
        + CKKSSubImpl<BE>
        + CKKSNegImpl<BE>
        + CKKSPow2Impl<BE>
        + CKKSImagImpl<BE>
        + CKKSRescaleImpl<BE>
        + CKKSRotateImpl<BE>
        + CKKSConjugateImpl<BE>
        + CKKSMulImpl<BE>
{
}
