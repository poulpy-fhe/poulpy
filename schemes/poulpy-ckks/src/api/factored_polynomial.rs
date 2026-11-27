use crate::{CKKSCtBounds, CKKSResult, SetCKKSInfos, layouts::CKKSMaskedConstants, polynomial::EncodedFactoredPolynomial};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos};
use poulpy_hal::layouts::{Backend, Ring, ScratchArena, Standard};
/// Evaluation of a checked, ring-specific factored polynomial on one or more ciphertexts.
///
/// # Metadata
///
/// Each step inherits the scale, capacity, slots and sparsity of its selected CKKS operations.
/// The final copy respects the requested output width. Inputs have a common degree, rank, radix,
/// scale and slot layout. Input budgets may differ.
///
/// # Errors
///
/// Invalid operand shapes, short coefficient tables and incompatible masked plaintexts are
/// refused before writing the result. Products propagate the selected operation's capacity
/// and tensor-key errors.
pub trait CKKSFactoredPolynomialEvaluationOps<BE: Backend, Rg: Ring = Standard> {
    /// Sizes the simultaneously live intermediate ciphertexts and the largest selected operation.
    fn ckks_eval_poly_factored_tmp_bytes<R, A, P, T>(
        &self,
        res: &R,
        a: &[&A],
        poly: &EncodedFactoredPolynomial<'_, P, Rg>,
        tsk: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSCtBounds + CKKSMaskedConstants<BE, Rg>,
        T: GGLWEInfos;
    /// Evaluates the program on its ordered inputs. Coefficient plaintexts stay on the backend.
    fn ckks_eval_poly_factored<R, A, P, H>(
        &self,
        res: &mut R,
        a: &[&A],
        poly: &EncodedFactoredPolynomial<'_, P, Rg>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
        H: GetTensorKey<BE, Rg>;
}
