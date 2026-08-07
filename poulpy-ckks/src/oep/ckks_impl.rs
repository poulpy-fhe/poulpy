use poulpy_hal::layouts::Backend;

use super::{
    CKKSAddImpl, CKKSBootstrappingImpl, CKKSConjugateImpl, CKKSCopyImpl, CKKSEncryptionImpl, CKKSEvalModImpl, CKKSImagImpl,
    CKKSMulImpl, CKKSNegImpl, CKKSPlaintextZnxImpl, CKKSPolynomialEvaluationImpl, CKKSPow2Impl, CKKSRotateImpl, CKKSSubImpl,
    DFTImpl,
};

/// Aggregate CKKS dispatch surface.
///
/// Concrete APIs can depend on narrower `CKKS*Impl` family traits. This
/// aggregate trait remains useful for composite operations that span multiple
/// CKKS families and for broad test/backend capability bundles.
///
/// Bundles every scalar-independent family. The scalar-generic seams —
/// [`CKKSEncodingImpl<BE, F>`](super::CKKSEncodingImpl),
/// [`DFTMatrixImpl<BE, F>`](super::DFTMatrixImpl), and
/// [`CKKSPaCoCoeffEncodingImpl<BE>`](super::CKKSPaCoCoeffEncodingImpl) — carry
/// an encoding-scalar type parameter and therefore cannot be part of a
/// non-generic bundle; bound them separately where needed.
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
    + CKKSRotateImpl<BE>
    + CKKSConjugateImpl<BE>
    + CKKSMulImpl<BE>
    + CKKSPolynomialEvaluationImpl<BE>
    + DFTImpl<BE>
    + CKKSEvalModImpl<BE>
    + CKKSBootstrappingImpl<BE>
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
        + CKKSRotateImpl<BE>
        + CKKSConjugateImpl<BE>
        + CKKSMulImpl<BE>
        + CKKSPolynomialEvaluationImpl<BE>
        + DFTImpl<BE>
        + CKKSEvalModImpl<BE>
        + CKKSBootstrappingImpl<BE>
{
}
