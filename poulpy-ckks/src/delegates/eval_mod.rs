use crate::CKKSResult as Result;
use crate::layouts::EvalModBsgs;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSEvalModOps, layouts::eval_mod::EvalMod, oep::CKKSEvalModImpl};

impl<BE: Backend + CKKSEvalModImpl> CKKSEvalModOps<BE> for Module<BE> {
    fn ckks_eval_mod_tmp_bytes<R, C, P, F, T>(&self, res: &R, ct: &C, params: &EvalMod<F, P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        C: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        BE::ckks_eval_mod_tmp_bytes_impl(self, res, ct, params, tsk)
    }

    fn ckks_eval_mod<R, C, P, F, H>(
        &self,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        H: GetTensorKey<BE>,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_eval_mod", res)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_eval_mod", ct)?;
        let ring = crate::api::CKKSModuleInfos::ckks_ring(self);
        match &params.f_mod_bsgs {
            EvalModBsgs::Real(poly) => super::polynomial_evaluation::check_polynomial_ring::<BE, _>(ring, poly)?,
            EvalModBsgs::Complex(poly) => {
                crate::ckks_ensure!(
                    !crate::api::CKKSModuleInfos::ckks_is_conjugate_invariant(self),
                    "complex EvalMod requires the standard CKKS ring"
                );
                super::polynomial_evaluation::check_polynomial_ring::<BE, _>(ring, &poly.re)?;
                super::polynomial_evaluation::check_polynomial_ring::<BE, _>(ring, &poly.im)?;
            }
        }
        if let Some(poly) = &params.f_mod_inv_bsgs {
            super::polynomial_evaluation::check_polynomial_ring::<BE, _>(ring, poly)?;
        }
        for pt in params.range_extension_consts.iter().chain(params.f_mod_input_offset.iter()) {
            ring.check_coefficients("ckks_eval_mod", pt)?;
        }
        BE::ckks_eval_mod_impl::<R, C, P, F, H>(self, res, ct, params, tsk, scratch)
    }
}
