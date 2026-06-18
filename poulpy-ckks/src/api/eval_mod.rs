use anyhow::Result;
use poulpy_hal::layouts::{Backend, ScratchArena};

use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};

use crate::{CKKSCtBounds, SetCKKSInfos, default::eval_mod::EvalModParameters};

/// Homomorphic modular reduction (`x mod 1`) via a trigonometric polynomial
/// approximation, the core non-linear step of CKKS bootstrapping.
pub trait CKKSEvalModOps<BE: Backend> {
    fn ckks_eval_mod_tmp_bytes<R, P, F, T>(&self, res: &R, params: &EvalModParameters<F, P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos;

    /// Evaluates the configured `x mod 1` approximation of `ct` into `res`.
    ///
    /// Consumes `params.depth() * log_delta` bits of `log_budget`; errors if
    /// `ct` has insufficient remaining capacity.
    fn ckks_eval_mod<R, C, P, F>(
        &self,
        res: &mut R,
        ct: &C,
        params: &EvalModParameters<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta;
}
