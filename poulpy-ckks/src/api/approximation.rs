//! Evaluation of prepared polynomial approximations.

use poulpy_core::layouts::{
    BSGSMeta, Compact, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, SetBSGSMeta, prepared::GLWETensorKeyPrepared,
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, CKKSInfos, CKKSResult as Result, SetCKKSInfos, layouts::PolynomialApproximation};

/// Homomorphic evaluation of reusable, interval-mapped polynomial plans.
pub trait CKKSApproximationOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_eval_approximation`].
    fn ckks_approximation_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, approximation: &PolynomialApproximation<P>) -> usize
    where
        R: CKKSCtBounds,
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
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Compact,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta;
}
