//! Composite evaluation of prepared polynomial approximations.

use poulpy_core::layouts::{
    BSGSMeta, Compact, GGLWEInfos, GLWE, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, SetBSGSMeta,
    prepared::{GLWETensorKeyPrepared, GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSResult as Result, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSAffineOps, CKKSAllOpsTmpBytes, CKKSApproximationOps, CKKSCopyOps, CKKSPolynomialEvaluationOps,
        CKKSPow2Ops,
    },
    ckks_ensure,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, PolynomialApproximation, ScratchArenaTakeCKKS},
};

impl<BE: Backend> CKKSApproximationOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSAffineOps<BE>
        + CKKSAllOpsTmpBytes<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSPow2Ops<BE>,
    CKKSCiphertext<BE::OwnedBuf>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Compact,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_approximation_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, approximation: &PolynomialApproximation<P>) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        let coeffs = approximation.poly.baby_step(0);
        let eval = self.ckks_all_ops_tmp_bytes(res, tsk, coeffs);
        match &approximation.affine {
            Some(affine) => {
                let ct = GLWE::<Vec<u8>>::bytes_of_from_infos(res);
                let map = if approximation.scale_pow2.is_some() {
                    eval
                } else {
                    eval.max(self.ckks_affine_pt_const_tmp_bytes(res, res, affine))
                };
                ct + map
            }
            None => eval,
        }
    }

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
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
    {
        let required = approximation.consumed_bits(input.log_delta());
        ckks_ensure!(
            input.log_budget() > required,
            "ckks_eval_approximation: log_budget {} <= {required} bits required at log_delta {}",
            input.log_budget(),
            input.log_delta()
        );
        if let Some(affine) = &approximation.affine {
            scratch.scope(|scratch_local| {
                let (mut normalized, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
                if let Some(exponent) = approximation.scale_pow2 {
                    self.ckks_copy(&mut normalized, input, &mut scratch_local)?;
                    if exponent < 0 {
                        self.ckks_div_pow2_assign(&mut normalized, exponent.unsigned_abs() as usize)?;
                    } else if exponent > 0 {
                        self.ckks_mul_pow2_assign(&mut normalized, exponent as usize, &mut scratch_local)?;
                    }
                    self.ckks_add_pt_const_assign(&mut normalized, 0, affine, 0, &mut scratch_local)?;
                } else {
                    self.ckks_affine_pt_const_into(&mut normalized, input, affine, 0, 1, &mut scratch_local)?;
                }
                self.ckks_eval_poly_real_const_coeffs(res, &normalized, &approximation.poly, tsk, &mut scratch_local)
            })
        } else {
            self.ckks_eval_poly_real_const_coeffs(res, input, &approximation.poly, tsk, scratch)
        }
    }
}
