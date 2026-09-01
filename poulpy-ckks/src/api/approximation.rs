//! Evaluation of prepared polynomial approximations.

use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, LWEInfos, SetBSGSMeta};
use poulpy_hal::layouts::Normalized;
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, CKKSInfos, CKKSResult as Result, SetCKKSInfos, layouts::PolynomialApproximation};

/// Homomorphic evaluation of reusable, interval-mapped polynomial plans.
pub trait CKKSApproximationOps<BE: Backend> {
    /// Scratch bytes for evaluation with these input and output layouts.
    fn ckks_approximation_tmp_bytes<R, I, T, P>(
        &self,
        res: &R,
        input: &I,
        tsk: &T,
        approximation: &PolynomialApproximation<P>,
    ) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// Applies the approximation's interval map, then evaluates its BSGS
    /// polynomial.
    fn ckks_eval_approximation<R, I, P, H>(
        &self,
        res: &mut R,
        input: &I,
        approximation: &PolynomialApproximation<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
        H: GetTensorKey<BE>;
}
