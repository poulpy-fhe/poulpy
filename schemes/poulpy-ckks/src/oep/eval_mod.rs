use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, layouts::eval_mod::EvalMod};

/// Backend override hook for [`CKKSEvalModOps`](crate::api::CKKSEvalModOps).
///
/// Evaluation and scratch sizing are independently selected backend methods.
/// [`impl_ckks_eval_mod_reference_full`] wires both to the callable reference circuit.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSEvalModImpl: Backend {
    /// Scratch bound of [`Self::ckks_eval_mod`]; may report less than the reference, never less than the body takes.
    /// Pass the `res`, `ct`, `params` and `tsk` the evaluation uses: EvalMod raises the input to the plan's
    /// `f_mod_log_delta` internally.
    fn ckks_eval_mod_tmp_bytes<R, C, P, F, T>(module: &Module<Self>, res: &R, ct: &C, params: &EvalMod<F, P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        C: CKKSCtBounds,
        P: CKKSCtBounds,
        T: poulpy_core::layouts::GGLWEInfos;

    /// Reproduces
    /// [`ckks_eval_mod_reference`](crate::reference::eval_mod::CKKSEvalModReference::ckks_eval_mod_reference) bit
    /// for bit, within [`Self::ckks_eval_mod_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_eval_mod<R, C, P, F, H>(
        module: &Module<Self>,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        H: GetTensorKey<Self>;
}

/// Selects the reference implementation of `CKKSEvalModImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_eval_mod_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSEvalModImpl for $be {
            fn ckks_eval_mod_tmp_bytes<R, C, P, F, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                ct: &C,
                params: &$crate::layouts::eval_mod::EvalMod<F, P>,
                tsk: &T,
            ) -> usize
            where
                R: $crate::CKKSCtBounds,
                C: $crate::CKKSCtBounds,
                P: $crate::CKKSCtBounds,
                T: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::eval_mod::CKKSEvalModReference::ckks_eval_mod_tmp_bytes_reference(module, res, ct, params, tsk)
            }
            fn ckks_eval_mod<R, C, P, F, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                ct: &C,
                params: &$crate::layouts::eval_mod::EvalMod<F, P>,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                R: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::SetBSGSMeta,
                C: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_core::layouts::IntPolyInfos
                    + $crate::CKKSCtBounds
                    + ::poulpy_core::layouts::BSGSMeta,
                H: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::eval_mod::CKKSEvalModReference::ckks_eval_mod_reference(module, res, ct, params, tsk, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_eval_mod_reference_full;
