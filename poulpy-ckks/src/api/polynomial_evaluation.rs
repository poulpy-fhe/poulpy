use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, CoeffNormalized, ScratchArena};

use poulpy_core::layouts::{BSGSMeta, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};

use crate::{CKKSCtBounds, SetCKKSInfos, layouts::CKKSCiphertextOwned, polynomial::ComplexBSGSPolynomial};

pub use poulpy_core::layouts::{BSGSPolynomialInfos, BabyStep, Basis, Parity, PolynomialInputTransform, PowerBasisHelper};

pub trait CKKSPolynomialEvaluationOps<BE: Backend> {
    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, H>(
        &self,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<BE, State = CoeffNormalized> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>;

    /// Evaluates a complex-coefficient polynomial `Σ_k (a_k + i·b_k)·z^k`,
    /// where `poly.re`/`poly.im` are the matched real/imag BSGS decompositions
    /// (identical baby-step schedule).
    fn ckks_eval_poly_complex_const_coeffs_from_power_basis<R, C, A, G, H>(
        &self,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos,
        A: GLWEToBackendRef<BE, State = CoeffNormalized> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>;

    /// Builds the power basis internally then evaluates a real-coefficient
    /// polynomial.
    fn ckks_eval_poly_real_const_coeffs<R, S, B, H>(
        &self,
        dst: &mut R,
        src: &S,
        bsgs: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE, State = CoeffNormalized> + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE, State = CoeffNormalized>
            + GLWEToBackendRef<BE, State = CoeffNormalized>
            + CKKSCtBounds
            + SetCKKSInfos;

    /// Builds the power basis internally then evaluates a complex-coefficient
    /// polynomial.
    fn ckks_eval_poly_complex_const_coeffs<R, S, C, H>(
        &self,
        dst: &mut R,
        src: &S,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE, State = CoeffNormalized> + CKKSCtBounds,
        C: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE, State = CoeffNormalized>
            + GLWEToBackendRef<BE, State = CoeffNormalized>
            + CKKSCtBounds
            + SetCKKSInfos;
}
