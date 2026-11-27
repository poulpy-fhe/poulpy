use crate::{CKKSCtBounds, CKKSResult, SetCKKSInfos, layouts::CKKSMaskedConstants, polynomial::EncodedFactoredPolynomial};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard};
/// Backend contract for [`CKKSFactoredPolynomialEvaluationOps`](crate::api::CKKSFactoredPolynomialEvaluationOps).
///
/// # Safety
///
/// The operation reproduces the reference's values, metadata and refusals. Its query must
/// cover its scratch use. The backend memory, aliasing and synchronization contract applies.
/// For a program compiled by BSGS, the implementation must fit the shared BSGS scratch bound.
/// [`FactoredPolynomial::uses_bsgs_workspace`](crate::polynomial::FactoredPolynomial::uses_bsgs_workspace) identifies those calls.
pub unsafe trait CKKSFactoredPolynomialEvaluationImpl<Rg: Ring = Standard>: Backend {
    /// Sizes the simultaneously live intermediate ciphertexts and the largest selected operation.
    fn ckks_eval_poly_factored_tmp_bytes<R, A, P, T>(
        module: &Module<Self>,
        res: &R,
        a: &[&A],
        poly: &EncodedFactoredPolynomial<'_, P, Rg>,
        tsk: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSCtBounds + CKKSMaskedConstants<Self, Rg>,
        T: GGLWEInfos;
    /// Evaluates the program on its ordered inputs. Coefficient plaintexts stay on the backend.
    fn ckks_eval_poly_factored<R, A, P, H>(
        module: &Module<Self>,
        res: &mut R,
        a: &[&A],
        poly: &EncodedFactoredPolynomial<'_, P, Rg>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        R: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<Self, Rg> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<Self, Rg>,
        H: GetTensorKey<Self, Rg>;
}

/// Selects the reference factored polynomial evaluator on one or both rings.
#[macro_export]
macro_rules! impl_ckks_factored_polynomial_evaluation_reference_full {
    ($be:ty) => {
        $crate::impl_ckks_factored_polynomial_evaluation_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_ckks_factored_polynomial_evaluation_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        const _: () = {
            use $crate::{CKKSCtBounds, CKKSResult, SetCKKSInfos, layouts::CKKSMaskedConstants, polynomial::EncodedFactoredPolynomial};
            use ::poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos};
            use ::poulpy_hal::layouts::{Module, ScratchArena};

            unsafe impl $crate::oep::CKKSFactoredPolynomialEvaluationImpl<$ring> for $be {
                fn ckks_eval_poly_factored_tmp_bytes<R, A, P, T>(
                    module: &Module<Self>,
                    res: &R,
                    a: &[&A],
                    poly: &EncodedFactoredPolynomial<'_, P, $ring>,
                    tsk: &T,
                ) -> usize
                where
                    R: CKKSCtBounds,
                    A: CKKSCtBounds,
                    P: CKKSCtBounds + CKKSMaskedConstants<Self, $ring>,
                    T: GGLWEInfos,
                {
                    $crate::reference::CKKSFactoredPolynomialEvaluationReference::<Self, $ring>::ckks_eval_poly_factored_tmp_bytes_reference(module, res, a, poly, tsk)
                }

                fn ckks_eval_poly_factored<R, A, P, H>(
                    module: &Module<Self>,
                    res: &mut R,
                    a: &[&A],
                    poly: &EncodedFactoredPolynomial<'_, P, $ring>,
                    tsk: &H,
                    scratch: &mut ScratchArena<'_, Self>,
                ) -> CKKSResult<()>
                where
                    R: GLWEToBackendMut<Self, $ring> + CKKSCtBounds + SetCKKSInfos,
                    A: GLWEToBackendRef<Self, $ring> + CKKSCtBounds,
                    P: GLWEToBackendRef<Self, $ring> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<Self, $ring>,
                    H: GetTensorKey<Self, $ring>,
                {
                    $crate::reference::CKKSFactoredPolynomialEvaluationReference::<Self, $ring>::ckks_eval_poly_factored_reference(module, res, a, poly, tsk, scratch)
                }
            }
        };
    };
}

pub use crate::impl_ckks_factored_polynomial_evaluation_reference_full;
