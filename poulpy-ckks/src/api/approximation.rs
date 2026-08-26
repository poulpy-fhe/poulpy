//! Evaluation of prepared polynomial approximations.

use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, LWEInfos, SetBSGSMeta,
    prepared::GLWETensorKeyPrepared,
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSResult as Result, SetCKKSInfos,
    layouts::{AdaptivePolynomialApproximation, PolynomialApproximation},
};

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
    fn ckks_eval_approximation<R, I, P>(
        &self,
        res: &mut R,
        input: &I,
        approximation: &PolynomialApproximation<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos;

    /// Scratch bytes for adaptive approximation evaluation.
    fn ckks_adaptive_approximation_tmp_bytes<R, I, T, P>(
        &self,
        res: &R,
        input: &I,
        tsk: &T,
        approximation: &AdaptivePolynomialApproximation<P>,
    ) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// Evaluates both adaptive branches after the shared input transforms.
    fn ckks_eval_adaptive_approximation<R, I, P>(
        &self,
        res: &mut R,
        input: &I,
        approximation: &AdaptivePolynomialApproximation<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos;
}
