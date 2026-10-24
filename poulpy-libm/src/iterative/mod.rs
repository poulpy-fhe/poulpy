//! Fixed-iteration reciprocal and square-root kernels.

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
        CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSMulOps, CKKSNegOps,
        CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps,
    },
    layouts::{CKKSCiphertext, CKKSModuleAlloc, ScratchArenaTakeCKKS},
};

use crate::range::IntervalNorm;
use crate::sign::{CKKSSignOps, SignComposite, ckks_sign_into};

fn ensure_goldschmidt_capacity<C: CKKSCtBounds>(
    op: &'static str,
    ct: &C,
    iters: usize,
) -> Result<()> {
    ensure!(iters > 0, "{op}: iters must be > 0");

    let log_delta = ct.log_delta();
    let consumed = (iters + 1) * log_delta;
    ensure!(
        ct.log_budget() > consumed,
        "{op}: log_budget {} <= {consumed} bits required for {iters} iterations at log_delta {log_delta}",
        ct.log_budget(),
    );
    Ok(())
}

fn ensure_rsqrt_capacity<C: CKKSCtBounds>(y: &C, r: usize) -> Result<()> {
    ensure!(r > 0, "ckks_rsqrt: r must be > 0");

    let log_delta = y.log_delta();
    let consumed = 2 * r * log_delta;
    ensure!(
        y.log_budget() > consumed,
        "ckks_rsqrt: log_budget {} <= {consumed} bits required for {r} iterations at log_delta {log_delta}",
        y.log_budget(),
    );
    Ok(())
}

/// Reciprocal and inverse square root.
pub trait CKKSInverseOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_goldschmidt_division`] / [`Self::ckks_rsqrt`].
    fn ckks_inverse_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos;

    /// Computes `1/ct` on `[0, 2]`. Consumes `(iters + 1)·log_delta` bits.
    fn ckks_goldschmidt_division<C>(
        &self,
        ct: &mut C,
        iters: usize,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Computes `1/sqrt(x)` with `r` Newton steps. Consumes `2r·log_delta` bits.
    fn ckks_rsqrt(
        &self,
        y: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        in_half: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        r: usize,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;
}

/// Reciprocal over signed intervals.
pub trait CKKSInverseDomainOps<BE: Backend> {
    /// Scratch bytes for inverse-domain operations.
    fn ckks_inverse_domain_tmp_bytes<R, T, A, P>(
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

    /// Compresses `ct` to `[−1, 1]` and writes its factor to `norm`.
    fn ckks_interval_normalization<P>(
        &self,
        norm: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        ct: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        params: &IntervalNorm<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos;

    /// `ct ← 1/ct` for `ct ∈ [Min, 2−Min]`. Alias for Goldschmidt division.
    fn ckks_inverse_positive_domain(
        &self,
        ct: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        iters: usize,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// `ct ← 1/ct` on `[−(2−Min), −Min]`.
    fn ckks_inverse_negative_domain(
        &self,
        ct: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        iters: usize,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// `ct ← 1/ct` on a signed domain outside the composite gap.
    #[allow(clippy::too_many_arguments)]
    fn ckks_inverse_full_domain<F, P, K>(
        &self,
        ct: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        iters: usize,
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
}

impl<BE: Backend + HostStaged> CKKSInverseOps<BE> for Module<BE>
where
    Module<BE>: CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_inverse_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
    {
        let ct_bytes = GLWE::<Vec<u8>, BE::ZnxWord>::bytes_of_from_infos(res);
        4 * ct_bytes
            + self
                .ckks_mul_tmp_bytes(res, res, res, tsk)
                .max(self.ckks_square_tmp_bytes(res, res, tsk))
                .max(self.ckks_add_tmp_bytes())
                .max(self.ckks_add_pt_const_tmp_bytes())
                .max(self.ckks_neg_tmp_bytes())
                .max(self.ckks_copy_tmp_bytes())
                .max(self.ckks_div_pow2_tmp_bytes())
    }

    fn ckks_goldschmidt_division<C>(
        &self,
        ct: &mut C,
        iters: usize,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        ensure_goldschmidt_capacity("ckks_goldschmidt_division", &*ct, iters)?;

        // a := 2 − m (in `ct`), b := 1 − m; a·b converges to 1/m.
        scratch.scope(|scratch_local| {
            let (mut b, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut tmp, mut scratch_local) =
                scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            self.ckks_neg_assign(ct)?;
            self.ckks_copy(&mut b, &*ct, &mut scratch_local)?;
            self.ckks_add_one_assign(&mut b, &mut scratch_local)?;
            self.ckks_add_one_assign(ct, &mut scratch_local)?;
            self.ckks_add_one_assign(ct, &mut scratch_local)?;

            for _ in 0..iters {
                self.ckks_square_assign(&mut b, tsk, &mut scratch_local)?;
                self.ckks_mul_into(&mut tmp, &*ct, &b, tsk, &mut scratch_local)?;
                self.ckks_add_assign(ct, &tmp, &mut scratch_local)?;
            }

            Ok(())
        })
    }

    fn ckks_rsqrt(
        &self,
        y: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        in_half: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        r: usize,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        ensure_rsqrt_capacity(&*y, r)?;

        // y ← 1.5·y − (x/2)·y·y², with 1.5·y = y + y/2.
        scratch.scope(|scratch_local| {
            let (mut ysqr, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*y);
            let (mut xy, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*y);
            let (mut t, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*y);
            let (mut half_y, mut scratch_local) =
                scratch_local.take_ckks_ciphertext_like_scratch(&*y);

            for _ in 0..r {
                self.ckks_square_into(&mut ysqr, &*y, tsk, &mut scratch_local)?;
                self.ckks_mul_into(&mut xy, in_half, &*y, tsk, &mut scratch_local)?;
                self.ckks_mul_into(&mut t, &ysqr, &xy, tsk, &mut scratch_local)?;
                self.ckks_neg_assign(&mut t)?;
                self.ckks_div_pow2_into(&mut half_y, &*y, 1, &mut scratch_local)?;
                self.ckks_add_assign(&mut t, &*y, &mut scratch_local)?;
                self.ckks_add_assign(&mut t, &half_y, &mut scratch_local)?;
                self.ckks_copy(y, &t, &mut scratch_local)?;
            }

            Ok(())
        })
    }
}

impl<BE: Backend + HostStaged> CKKSInverseDomainOps<BE> for Module<BE>
where
    Module<BE>: CKKSInverseOps<BE>
        + CKKSSignOps<BE>
        + CKKSMulOps<BE>
        + CKKSNegOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSModuleAlloc<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_inverse_domain_tmp_bytes<R, T, A, P>(
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
        let interval = 5 * ct_bytes
            + self
                .ckks_mul_tmp_bytes(res, res, res, tsk)
                .max(self.ckks_square_tmp_bytes(res, res, tsk))
                .max(self.ckks_mul_pt_const_tmp_bytes(res, res, coeff_prec))
                .max(self.ckks_copy_tmp_bytes())
                .max(self.ckks_neg_tmp_bytes())
                .max(self.ckks_add_tmp_bytes())
                .max(self.ckks_sub_tmp_bytes());
        self.ckks_inverse_tmp_bytes(res, tsk)
            .max(ct_bytes + self.ckks_sign_tmp_bytes(res, tsk, conj_key, coeff_prec))
            .max(ct_bytes + self.ckks_inverse_tmp_bytes(res, tsk))
            .max(interval)
    }

    fn ckks_interval_normalization<P>(
        &self,
        norm: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        ct: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        params: &IntervalNorm<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos,
    {
        let required = params.consumed_bits(ct.log_delta());
        ensure!(
            ct.log_budget() > required,
            "ckks_interval_normalization: log_budget {} <= {required} bits required",
            ct.log_budget(),
        );

        // Track each compression factor in norm.
        scratch.scope(|scratch_local| {
            let (mut z0, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut z1, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut z2, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut z0z1, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut z0z2, mut scratch_local) =
                scratch_local.take_ckks_ciphertext_like_scratch(&*ct);

            for i in 0..params.n {
                self.ckks_mul_pt_const_into(&mut z0, &*ct, &params.consts, i, &mut scratch_local)?;
                self.ckks_square_into(&mut z1, &*ct, tsk, &mut scratch_local)?;
                self.ckks_mul_into(&mut z0z1, &z0, &z1, tsk, &mut scratch_local)?;
                if i == 0 {
                    self.ckks_mul_into(&mut z0z2, &z0, &*ct, tsk, &mut scratch_local)?;
                    self.ckks_copy(norm, &z0z2, &mut scratch_local)?;
                    self.ckks_neg_assign(norm)?;
                    self.ckks_add_one_assign(norm, &mut scratch_local)?;
                } else {
                    self.ckks_mul_into(&mut z2, &*norm, &*ct, tsk, &mut scratch_local)?;
                    self.ckks_mul_into(&mut z0z2, &z0, &z2, tsk, &mut scratch_local)?;
                    self.ckks_sub_assign(norm, &z0z2, &mut scratch_local)?;
                }
                self.ckks_sub_assign(ct, &z0z1, &mut scratch_local)?;
            }

            Ok(())
        })
    }

    fn ckks_inverse_positive_domain(
        &self,
        ct: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        iters: usize,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        self.ckks_goldschmidt_division(ct, iters, tsk, scratch)
    }

    fn ckks_inverse_negative_domain(
        &self,
        ct: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        iters: usize,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        ensure_goldschmidt_capacity("ckks_inverse_negative_domain", &*ct, iters)?;
        self.ckks_neg_assign(ct)?;
        self.ckks_goldschmidt_division(ct, iters, tsk, scratch)?;
        self.ckks_neg_assign(ct)?;
        Ok(())
    }

    fn ckks_inverse_full_domain<F, P, K>(
        &self,
        ct: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        iters: usize,
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
        let log_delta = ct.log_delta();
        let consumed = composite.consumed_bits(log_delta) + (iters + 3) * log_delta;
        ensure!(iters > 0, "ckks_inverse_full_domain: iters must be > 0");
        ensure!(
            ct.log_budget() > consumed,
            "ckks_inverse_full_domain: log_budget {} <= {consumed} bits required for {iters} iterations at log_delta {log_delta}",
            ct.log_budget(),
        );

        // Invert the magnitude, then restore the sign.
        scratch.scope(|scratch_local| {
            let (mut sign, mut scratch_local) =
                scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            ckks_sign_into(
                self,
                &mut sign,
                &*ct,
                composite,
                tsk,
                conj_key,
                &mut scratch_local,
            )?;
            self.ckks_mul_assign(ct, &sign, tsk, &mut scratch_local)?;
            self.ckks_goldschmidt_division(ct, iters, tsk, &mut scratch_local)?;
            self.ckks_mul_assign(ct, &sign, tsk, &mut scratch_local)
        })?;
        Ok(())
    }
}

/// Division and square root over the iterative kernels.
pub trait CKKSDivSqrtOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_div`] / [`Self::ckks_sqrt`].
    fn ckks_div_sqrt_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos;

    /// `res ← a / b` for `b ∈ [Min, 2−Min]`.
    fn ckks_div(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        iters: usize,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// `res ← sqrt(x)` near `1`; `in_half` is `x/2` at the same scale.
    fn ckks_sqrt(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        in_half: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        r: usize,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;
}

impl<BE: Backend + HostStaged> CKKSDivSqrtOps<BE> for Module<BE>
where
    Module<BE>: CKKSInverseOps<BE> + CKKSMulOps<BE> + CKKSCopyOps<BE> + CKKSModuleAlloc<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_div_sqrt_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
    {
        self.ckks_inverse_tmp_bytes(res, tsk)
            .max(self.ckks_mul_tmp_bytes(res, res, res, tsk))
            .max(self.ckks_copy_tmp_bytes())
    }

    fn ckks_div(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        iters: usize,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        self.ckks_copy(res, b, scratch)?;
        self.ckks_goldschmidt_division(res, iters, tsk, scratch)?;
        self.ckks_mul_assign(res, a, tsk, scratch)?;
        Ok(())
    }

    fn ckks_sqrt(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        in_half: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        r: usize,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        self.ckks_copy(res, x, scratch)?;
        self.ckks_rsqrt(res, in_half, r, tsk, scratch)?;
        self.ckks_mul_assign(res, x, tsk, scratch)?;
        Ok(())
    }
}
