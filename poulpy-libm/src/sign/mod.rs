//! Composite-minimax sign and smooth predicates.

use anyhow::{Result, ensure};
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GGLWEPreparedToBackendRef, GLWE, GLWETensorKeyPrepared, GLWEToBackendMut,
    GLWEToBackendRef, GetGaloisElement, LWEInfos, SetBSGSMeta,
    prepared::{GLWEAutomorphismKeyPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::layouts::{Backend, HostStaged, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSCopyOps, CKKSPolynomialEvaluationOps,
        CKKSPow2Ops,
    },
    layouts::{CKKSCiphertext, ScratchArenaTakeCKKS},
};

mod compare;
mod composite;
mod predicate;

pub use compare::CKKSComparisonOps;
pub use composite::{
    COEFFS_SIGN_X2_CHEBYSHEV, COEFFS_SIGN_X4_CHEBYSHEV, DEFAULT_SIGN_COMPOSITE_CHEBYSHEV,
    SignComposite,
};
pub use predicate::CKKSPredicateOps;

/// Evaluates and real-cleans a sign composite.
pub(crate) fn ckks_sign_into<BE, R, I, F, P, K>(
    module: &Module<BE>,
    res: &mut R,
    input: &I,
    composite: &SignComposite<F, P>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    conj_key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>:
        CKKSPolynomialEvaluationOps<BE> + CKKSCopyOps<BE> + CKKSConjugateOps<BE> + CKKSAddOps<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    I: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
        + GGLWEPreparedToBackendRef<BE>
        + GetGaloisElement
        + GGLWEInfos,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    ensure!(
        !composite.is_empty(),
        "ckks_sign: composite polynomial is empty"
    );

    let log_delta = input.log_delta();
    let required = composite.consumed_bits(log_delta);
    ensure!(
        input.log_budget() > required,
        "ckks_sign: log_budget {} <= {required} bits required at log_delta {log_delta}",
        input.log_budget(),
    );

    scratch.scope(|scratch_local| {
        let (mut cur, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
        let (mut nxt, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
        let (mut conj, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
        module.ckks_copy(&mut cur, input, &mut scratch_local)?;

        for poly in &composite.polys_bsgs {
            module.ckks_eval_poly_real_const_coeffs(
                &mut nxt,
                &cur,
                poly,
                tsk,
                &mut scratch_local,
            )?;
            module.ckks_conjugate_into(&mut conj, &nxt, conj_key, &mut scratch_local)?;
            module.ckks_add_assign(&mut nxt, &conj, &mut scratch_local)?;
            std::mem::swap(&mut cur, &mut nxt);
        }

        module.ckks_copy(res, &cur, &mut scratch_local)
    })?;
    Ok(())
}

pub(crate) fn ckks_step_into<BE, R, I, F, P, K>(
    module: &Module<BE>,
    res: &mut R,
    input: &I,
    composite: &SignComposite<F, P>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    conj_key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + HostStaged,
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSPow2Ops<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    I: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
        + GGLWEPreparedToBackendRef<BE>
        + GetGaloisElement
        + GGLWEInfos,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    ckks_sign_into(module, res, input, composite, tsk, conj_key, scratch)?;
    module.ckks_add_one_assign(res, scratch)?;
    scratch.scope(|scratch_local| {
        let (mut half, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*res);
        module.ckks_div_pow2_into(&mut half, &*res, 1, &mut scratch_local)?;
        module.ckks_copy(res, &half, &mut scratch_local)
    })?;
    Ok(())
}

/// Homomorphic `sign` and `step`.
pub trait CKKSSignOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_sign`] / [`Self::ckks_step`].
    fn ckks_sign_tmp_bytes<R, T, A, P>(
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

    /// Evaluates `sign(input)` on `[−1, 1]` outside the composite gap.
    fn ckks_sign<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos;

    /// Evaluates `(sign(input) + 1) / 2`.
    fn ckks_step<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos;
}

impl<BE: Backend> CKKSSignOps<BE> for Module<BE>
where
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSAllOpsTmpBytes<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_sign_tmp_bytes<R, T, A, P>(
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
        3 * ct_bytes + self.ckks_all_ops_with_atk_tmp_bytes(res, tsk, conj_key, coeff_prec)
    }

    fn ckks_sign<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
    {
        // Each factor evaluates at half scale; `nxt += conj(nxt)` cleans the
        // imaginary part and restores full scale, keeping `log_delta` constant.
        ckks_sign_into(self, res, input, composite, tsk, conj_key, scratch)
    }

    fn ckks_step<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
    {
        ckks_step_into(self, res, input, composite, tsk, conj_key, scratch)
    }
}
