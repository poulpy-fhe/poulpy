use poulpy_hal::layouts::Backend;

use super::{
    CKKSAddImpl, CKKSAddManyImpl, CKKSAffineImpl, CKKSApproximationImpl, CKKSCIRingMapImpl, CKKSComplexPolynomialEvaluationImpl,
    CKKSConjugateImpl, CKKSCopyImpl, CKKSDFTImpl, CKKSDotProductImpl, CKKSEncryptionImpl, CKKSEvalModImpl, CKKSImagImpl,
    CKKSInnerSumImpl, CKKSLinearTransformationImpl, CKKSMulAddImpl, CKKSMulImpl, CKKSMulSubImpl, CKKSNegImpl,
    CKKSPlaintextVecImpl, CKKSPolynomialEvaluationImpl, CKKSPow2Impl, CKKSRotateImpl, CKKSSubImpl,
};

/// Aggregate CKKS dispatch surface: the scalar-independent evaluation families.
///
/// Concrete APIs depend on narrower `CKKS*Impl` families; this bundle serves code spanning several of them, such as
/// test and backend capability bounds. It includes standard-only families, so conjugate-invariant backends do not
/// implement it. The scalar-generic families ([`CKKSEncodingImpl<F>`](super::CKKSEncodingImpl),
/// [`CKKSDFTMatrixImpl<F>`](super::CKKSDFTMatrixImpl)) and the fold, bootstrapping, ModUp, PaCo and SHIP families
/// stay outside it and are bounded separately where needed.
///
/// # Safety
/// The trait has no method: it adds no obligation to those of the bundled `CKKS*Impl` traits.
pub unsafe trait CKKSImpl:
    Backend
    + CKKSPlaintextVecImpl
    + CKKSCopyImpl
    + CKKSAddImpl
    + CKKSEncryptionImpl
    + CKKSSubImpl
    + CKKSNegImpl
    + CKKSPow2Impl
    + CKKSImagImpl
    + CKKSRotateImpl
    + CKKSInnerSumImpl
    + CKKSConjugateImpl
    + CKKSMulImpl
    + CKKSDotProductImpl
    + CKKSLinearTransformationImpl
    + CKKSPolynomialEvaluationImpl
    + CKKSComplexPolynomialEvaluationImpl
    + CKKSDFTImpl
    + CKKSEvalModImpl
    + CKKSAddManyImpl
    + CKKSMulAddImpl
    + CKKSAffineImpl
    + CKKSMulSubImpl
    + CKKSApproximationImpl
    + CKKSCIRingMapImpl
{
}

unsafe impl<BE: Backend> CKKSImpl for BE where
    BE: CKKSPlaintextVecImpl
        + CKKSCopyImpl
        + CKKSAddImpl
        + CKKSEncryptionImpl
        + CKKSSubImpl
        + CKKSNegImpl
        + CKKSPow2Impl
        + CKKSImagImpl
        + CKKSRotateImpl
        + CKKSInnerSumImpl
        + CKKSConjugateImpl
        + CKKSMulImpl
        + CKKSDotProductImpl
        + CKKSLinearTransformationImpl
        + CKKSPolynomialEvaluationImpl
        + CKKSComplexPolynomialEvaluationImpl
        + CKKSDFTImpl
        + CKKSEvalModImpl
        + CKKSAddManyImpl
        + CKKSMulAddImpl
        + CKKSAffineImpl
        + CKKSMulSubImpl
        + CKKSApproximationImpl
        + CKKSCIRingMapImpl
{
}
