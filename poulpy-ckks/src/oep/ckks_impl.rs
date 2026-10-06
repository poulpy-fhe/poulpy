use poulpy_hal::layouts::Backend;

use super::{
    CKKSAddImpl, CKKSComplexPolynomialEvaluationImpl, CKKSConjugateImpl, CKKSCopyImpl, CKKSEncryptionImpl, CKKSEvalModImpl,
    CKKSImagImpl, CKKSMulImpl, CKKSNegImpl, CKKSPlaintextZnxImpl, CKKSPolynomialEvaluationImpl, CKKSPow2Impl, CKKSRotateImpl,
    CKKSSubImpl, DFTImpl,
};

/// Aggregate CKKS dispatch surface.
///
/// Concrete APIs can depend on narrower `CKKS*Impl` family traits. This
/// aggregate trait remains useful for composite operations that span multiple
/// CKKS families and for broad test/backend capability bundles.
///
/// Bundles every scalar-independent family. The scalar-generic seams —
/// [`CKKSEncodingImpl<F>`](super::CKKSEncodingImpl),
/// [`DFTMatrixImpl<F>`](super::DFTMatrixImpl), and
/// [`CKKSPaCoCoeffEncodingImpl`](super::CKKSPaCoCoeffEncodingImpl) — carry
/// an encoding-scalar type parameter and therefore cannot be part of a
/// non-generic bundle; bound them separately where needed. It includes the
/// standard-only families, so conjugate-invariant backends do not implement it.
///
/// # Safety
/// Ciphertext outputs must reproduce the reference noise metadata and provenance
/// checks, including clearing invalidated estimates. Delegates only forward calls.
///
/// Implementations must satisfy the contracts of all bundled `CKKS*Impl`
/// traits, including the HAL-level invariants implied by their method
/// signatures.
pub unsafe trait CKKSImpl:
    Backend
    + CKKSPlaintextZnxImpl
    + CKKSCopyImpl
    + CKKSAddImpl
    + CKKSEncryptionImpl
    + CKKSSubImpl
    + CKKSNegImpl
    + CKKSPow2Impl
    + CKKSImagImpl
    + CKKSRotateImpl
    + CKKSConjugateImpl
    + CKKSMulImpl
    + CKKSPolynomialEvaluationImpl
    + CKKSComplexPolynomialEvaluationImpl
    + DFTImpl
    + CKKSEvalModImpl
{
}

unsafe impl<BE: Backend> CKKSImpl for BE where
    BE: CKKSPlaintextZnxImpl
        + CKKSCopyImpl
        + CKKSAddImpl
        + CKKSEncryptionImpl
        + CKKSSubImpl
        + CKKSNegImpl
        + CKKSPow2Impl
        + CKKSImagImpl
        + CKKSRotateImpl
        + CKKSConjugateImpl
        + CKKSMulImpl
        + CKKSPolynomialEvaluationImpl
        + CKKSComplexPolynomialEvaluationImpl
        + DFTImpl
        + CKKSEvalModImpl
{
}
