//! Predicates and selection built on `sign`.
//!
//! `gt`/`ge` and `lt`/`le` coincide under smooth CKKS comparison. Sign inputs
//! must lie in `[−1, 1]` outside the composite gap.

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

use super::{CKKSSignOps, SignComposite, ckks_sign_into, ckks_step_into};

/// Sign-based predicates and select.
pub trait CKKSPredicateOps<BE: Backend> {
    /// Scratch bytes for the sign-based predicates.
    fn ckks_predicate_tmp_bytes<R, T, A, P>(
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

    /// `res ← |x|` via `x·sign(x)`.
    fn ckks_fabs<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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

    /// `res ← sign(a − b)` (`+1`/`−1`/`0` for `a >`/`<`/`= b`).
    #[allow(clippy::too_many_arguments)]
    fn ckks_cmp<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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

    /// `res ← [a > b]` (`= step(a − b)`, `1`/`0`/`½`). `ge` is identical.
    #[allow(clippy::too_many_arguments)]
    fn ckks_gt<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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

    /// Alias of [`Self::ckks_gt`].
    #[allow(clippy::too_many_arguments)]
    fn ckks_ge<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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

    /// `res ← [a < b]` (`= step(b − a)`). `le` is identical.
    #[allow(clippy::too_many_arguments)]
    fn ckks_lt<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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

    /// Alias of [`Self::ckks_lt`].
    #[allow(clippy::too_many_arguments)]
    fn ckks_le<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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

    /// Smooth membership in `[bounds[lo_coeff], bounds[hi_coeff]]`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_indicator<F, P, B, K>(
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

    /// Smooth equality mask for `|a - b| <= epsilon[epsilon_coeff]`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_eq<F, P, E, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        epsilon: &E,
        epsilon_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        E: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos;

    /// `res ← max(a - b, 0)`; `a - b` must satisfy the sign plan's domain.
    #[allow(clippy::too_many_arguments)]
    fn ckks_fdim<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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

    /// `res ← |x| * sign(y)`; `x` and `y` must satisfy the sign plan's domain.
    #[allow(clippy::too_many_arguments)]
    fn ckks_copysign<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        y: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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

    /// `res ← mask·a + (1 − mask)·b` for a smooth binary mask.
    fn ckks_select(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        mask: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;
}

impl<BE: Backend> CKKSPredicateOps<BE> for Module<BE>
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
    fn ckks_predicate_tmp_bytes<R, T, A, P>(
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
        2 * ct_bytes + self.ckks_sign_tmp_bytes(res, tsk, conj_key, coeff_prec)
    }

    fn ckks_fabs<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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
        scratch.scope(|scratch_local| {
            let (mut sgn, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            ckks_sign_into(
                self,
                &mut sgn,
                x,
                composite,
                tsk,
                conj_key,
                &mut scratch_local,
            )?;
            self.ckks_mul_into(res, x, &sgn, tsk, &mut scratch_local)
        })?;
        Ok(())
    }

    fn ckks_cmp<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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
        scratch.scope(|scratch_local| {
            let (mut diff, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            self.ckks_sub_into(&mut diff, a, b, &mut scratch_local)?;
            ckks_sign_into(
                self,
                res,
                &diff,
                composite,
                tsk,
                conj_key,
                &mut scratch_local,
            )
        })
    }

    fn ckks_gt<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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
        step_of_diff(self, res, a, b, composite, tsk, conj_key, scratch)
    }

    fn ckks_ge<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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
        self.ckks_gt(res, a, b, composite, tsk, conj_key, scratch)
    }

    fn ckks_lt<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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
        // a < b  ⇔  b > a.
        step_of_diff(self, res, b, a, composite, tsk, conj_key, scratch)
    }

    fn ckks_le<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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
        self.ckks_lt(res, a, b, composite, tsk, conj_key, scratch)
    }

    fn ckks_indicator<F, P, B, K>(
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
        scratch.scope(|scratch_local| {
            let (mut diff, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            self.ckks_sub_pt_const_into(&mut diff, x, 0, bounds, lo_coeff, &mut scratch_local)?;
            ckks_step_into(
                self,
                res,
                &diff,
                composite,
                tsk,
                conj_key,
                &mut scratch_local,
            )
        })?;

        scratch.scope(|scratch_local| {
            let (mut diff, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            let (mut gate, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            self.ckks_sub_pt_const_into(&mut diff, x, 0, bounds, hi_coeff, &mut scratch_local)?;
            self.ckks_neg_assign(&mut diff)?;
            ckks_step_into(
                self,
                &mut gate,
                &diff,
                composite,
                tsk,
                conj_key,
                &mut scratch_local,
            )?;
            self.ckks_mul_assign(res, &gate, tsk, &mut scratch_local)
        })?;
        Ok(())
    }

    fn ckks_eq<F, P, E, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        epsilon: &E,
        epsilon_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        E: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
    {
        scratch.scope(|scratch_local| {
            let (mut diff, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            self.ckks_sub_into(&mut diff, a, b, &mut scratch_local)?;
            self.ckks_add_pt_const_assign(
                &mut diff,
                0,
                epsilon,
                epsilon_coeff,
                &mut scratch_local,
            )?;
            ckks_step_into(
                self,
                res,
                &diff,
                composite,
                tsk,
                conj_key,
                &mut scratch_local,
            )
        })?;

        scratch.scope(|scratch_local| {
            let (mut diff, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            let (mut gate, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            self.ckks_sub_into(&mut diff, b, a, &mut scratch_local)?;
            self.ckks_add_pt_const_assign(
                &mut diff,
                0,
                epsilon,
                epsilon_coeff,
                &mut scratch_local,
            )?;
            ckks_step_into(
                self,
                &mut gate,
                &diff,
                composite,
                tsk,
                conj_key,
                &mut scratch_local,
            )?;
            self.ckks_mul_assign(res, &gate, tsk, &mut scratch_local)
        })?;
        Ok(())
    }

    fn ckks_fdim<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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
        scratch.scope(|scratch_local| {
            let (mut diff, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            let (mut sign, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            self.ckks_sub_into(&mut diff, a, b, &mut scratch_local)?;
            ckks_sign_into(
                self,
                &mut sign,
                &diff,
                composite,
                tsk,
                conj_key,
                &mut scratch_local,
            )?;
            self.ckks_mul_into(res, &diff, &sign, tsk, &mut scratch_local)?;
            self.ckks_add_assign(res, &diff, &mut scratch_local)?;
            self.ckks_div_pow2_assign(res, 1)
        })?;
        Ok(())
    }

    fn ckks_copysign<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        y: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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
        scratch.scope(|scratch_local| {
            let (mut sign, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            ckks_sign_into(
                self,
                &mut sign,
                x,
                composite,
                tsk,
                conj_key,
                &mut scratch_local,
            )?;
            self.ckks_mul_into(res, x, &sign, tsk, &mut scratch_local)
        })?;
        scratch.scope(|scratch_local| {
            let (mut sign, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(y);
            ckks_sign_into(
                self,
                &mut sign,
                y,
                composite,
                tsk,
                conj_key,
                &mut scratch_local,
            )?;
            self.ckks_mul_assign(res, &sign, tsk, &mut scratch_local)
        })?;
        Ok(())
    }

    fn ckks_select(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        mask: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        scratch.scope(|scratch_local| {
            let (mut diff, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            self.ckks_sub_into(&mut diff, a, b, &mut scratch_local)?;
            self.ckks_mul_into(res, mask, &diff, tsk, &mut scratch_local)?;
            self.ckks_add_assign(res, b, &mut scratch_local)
        })?;
        Ok(())
    }
}

/// `res ← step(a − b) = (sign(a − b) + 1) / 2`.
#[allow(clippy::too_many_arguments)]
fn step_of_diff<BE, F, P, K>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
    a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
    b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
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
        + CKKSPow2Ops<BE>
        + CKKSPolynomialEvaluationOps<BE>,
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
        let (mut diff, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
        module.ckks_sub_into(&mut diff, a, b, &mut scratch_local)?;
        ckks_step_into(
            module,
            res,
            &diff,
            composite,
            tsk,
            conj_key,
            &mut scratch_local,
        )
    })
}
