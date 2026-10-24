//! Smooth extrema built on `sign`.

use anyhow::Result;
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GGLWEPreparedToBackendRef, GLWE, GLWETensorKeyPrepared, GLWEToBackendMut,
    GLWEToBackendRef, GetGaloisElement, LWEInfos, SetBSGSMeta,
    prepared::{GLWEAutomorphismKeyPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::layouts::{Backend, HostStaged, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSMulOps, CKKSNegOps,
        CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps,
    },
    layouts::{CKKSCiphertext, CKKSModuleAlloc, ScratchArenaTakeCKKS},
};

use super::{CKKSSignOps, SignComposite, ckks_sign_into};

/// Smooth extrema and clamp.
pub trait CKKSComparisonOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_fmax`] / [`Self::ckks_fmin`].
    fn ckks_comparison_tmp_bytes<R, T, A, P>(
        &self,
        res: &R,
        tsk: &T,
        conj_key: &A,
        coeff_prec: &P,
    ) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// `res ← fmax(op0, op1)`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_fmax<R, A, B, F, P, K>(
        &self,
        res: &mut R,
        op0: &A,
        op1: &B,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos;

    /// `res ← fmin(op0, op1)`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_fmin<R, A, B, F, P, K>(
        &self,
        res: &mut R,
        op0: &A,
        op1: &B,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos;

    /// `res ← fmax(x, bound[bound_coeff])`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_fmax_const<F, P, B, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        bound: &B,
        bound_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        B: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos;

    /// `res ← fmin(x, bound[bound_coeff])`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_fmin_const<F, P, B, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        bound: &B,
        bound_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        B: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos;

    /// `res ← min(max(x, bounds[lo_coeff]), bounds[hi_coeff])`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_clamp<F, P, B, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        bounds: &B,
        lo_coeff: usize,
        hi_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        B: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos;
}

impl<BE: Backend> CKKSComparisonOps<BE> for Module<BE>
where
    Module<BE>: CKKSSignOps<BE>
        + CKKSSubOps<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSMulOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSModuleAlloc<BE>
        + CKKSPolynomialEvaluationOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_comparison_tmp_bytes<R, T, A, P>(
        &self,
        res: &R,
        tsk: &T,
        conj_key: &A,
        coeff_prec: &P,
    ) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        let ct_bytes = GLWE::<Vec<u8>, BE::ZnxWord>::bytes_of_from_infos(res);
        3 * ct_bytes + self.ckks_sign_tmp_bytes(res, tsk, conj_key, coeff_prec)
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_fmax<R, A, B, F, P, K>(
        &self,
        res: &mut R,
        op0: &A,
        op1: &B,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
    {
        stepdiff_into(self, res, op0, op1, composite, tsk, conj_key, scratch)?;
        self.ckks_add_assign(res, op1, scratch)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_fmin<R, A, B, F, P, K>(
        &self,
        res: &mut R,
        op0: &A,
        op1: &B,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
    {
        stepdiff_into(self, res, op0, op1, composite, tsk, conj_key, scratch)?;
        self.ckks_neg_assign(res)?;
        self.ckks_add_assign(res, op0, scratch)?;
        Ok(())
    }

    fn ckks_fmax_const<F, P, B, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        bound: &B,
        bound_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        B: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
    {
        const_correction_into(
            self,
            res,
            x,
            bound,
            bound_coeff,
            true,
            composite,
            tsk,
            conj_key,
            scratch,
        )?;
        self.ckks_add_assign(res, x, scratch)?;
        Ok(())
    }

    fn ckks_fmin_const<F, P, B, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        bound: &B,
        bound_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        B: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
    {
        const_correction_into(
            self,
            res,
            x,
            bound,
            bound_coeff,
            false,
            composite,
            tsk,
            conj_key,
            scratch,
        )?;
        self.ckks_neg_assign(res)?;
        self.ckks_add_assign(res, x, scratch)?;
        Ok(())
    }

    fn ckks_clamp<F, P, B, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        bounds: &B,
        lo_coeff: usize,
        hi_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        B: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
    {
        self.ckks_fmax_const(res, x, bounds, lo_coeff, composite, tsk, conj_key, scratch)?;

        scratch.scope(|scratch_local| {
            let (mut correction, mut scratch_local) =
                scratch_local.take_ckks_ciphertext_like_scratch(&*res);
            const_correction_into(
                self,
                &mut correction,
                &*res,
                bounds,
                hi_coeff,
                false,
                composite,
                tsk,
                conj_key,
                &mut scratch_local,
            )?;
            self.ckks_sub_assign(res, &correction, &mut scratch_local)
        })?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn const_correction_into<BE, R, I, F, P, B, K>(
    module: &Module<BE>,
    res: &mut R,
    x: &I,
    bound: &B,
    bound_coeff: usize,
    reverse: bool,
    composite: &SignComposite<F, P>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    conj_key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + HostStaged,
    Module<BE>: CKKSSignOps<BE>
        + CKKSSubOps<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSMulOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSModuleAlloc<BE>
        + CKKSPolynomialEvaluationOps<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    I: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
    B: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
        + GGLWEPreparedToBackendRef<BE>
        + GetGaloisElement
        + GGLWEInfos,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    scratch.scope(|scratch_local| {
        let (mut diff, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
        let (mut sgn, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
        module.ckks_sub_pt_const_into(&mut diff, x, 0, bound, bound_coeff, &mut scratch_local)?;
        if reverse {
            module.ckks_neg_assign(&mut diff)?;
        }
        ckks_sign_into(
            module,
            &mut sgn,
            &diff,
            composite,
            tsk,
            conj_key,
            &mut scratch_local,
        )?;
        module.ckks_add_one_assign(&mut sgn, &mut scratch_local)?;
        module.ckks_mul_into(res, &sgn, &diff, tsk, &mut scratch_local)?;
        let (mut half, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*res);
        module.ckks_div_pow2_into(&mut half, &*res, 1, &mut scratch_local)?;
        module.ckks_copy(res, &half, &mut scratch_local)
    })?;
    Ok(())
}

/// Computes `step(a - b) * (a - b)`.
#[allow(clippy::too_many_arguments)]
fn stepdiff_into<BE, R, A, B, F, P, K>(
    module: &Module<BE>,
    res: &mut R,
    op0: &A,
    op1: &B,
    composite: &SignComposite<F, P>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    conj_key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + HostStaged,
    Module<BE>: CKKSSignOps<BE>
        + CKKSSubOps<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSMulOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSModuleAlloc<BE>
        + CKKSPolynomialEvaluationOps<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    B: GLWEToBackendRef<BE> + CKKSCtBounds,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
        + GGLWEPreparedToBackendRef<BE>
        + GetGaloisElement
        + GGLWEInfos,
{
    scratch.scope(|scratch_local| {
        let (mut diff, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(op0);
        let (mut sgn, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(op0);
        module.ckks_sub_into(&mut diff, op0, op1, &mut scratch_local)?;
        ckks_sign_into(
            module,
            &mut sgn,
            &diff,
            composite,
            tsk,
            conj_key,
            &mut scratch_local,
        )?;
        module.ckks_add_one_assign(&mut sgn, &mut scratch_local)?;
        module.ckks_mul_into(res, &sgn, &diff, tsk, &mut scratch_local)?;
        let (mut half, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*res);
        module.ckks_div_pow2_into(&mut half, &*res, 1, &mut scratch_local)?;
        module.ckks_copy(res, &half, &mut scratch_local)
    })?;
    Ok(())
}
