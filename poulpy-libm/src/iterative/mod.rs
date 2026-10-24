//! Fixed-iteration reciprocal and square-root kernels.
//!
//! The in-place operations run on scratch copies of their operands and write them back once, at the end, so a refusal
//! leaves them unchanged.

use anyhow::{Result, ensure};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, IntPolyInfos, LWEInfos,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena, Standard};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSCopyOps, CKKSMulOps, CKKSNegOps, CKKSPolynomialEvaluationOps,
        CKKSPow2Ops, CKKSSubOps,
    },
    layouts::ScratchArenaTakeCKKS,
};

use crate::{
    range::IntervalNorm,
    sign::{SignComposite, ckks_sign_into, ckks_sign_tmp_bytes},
};

fn ensure_goldschmidt_capacity<C: CKKSCtBounds>(op: &'static str, ct: &C, iters: usize) -> Result<()> {
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

fn ensure_rsqrt_capacity<C: CKKSCtBounds>(op: &'static str, y: &C, r: usize) -> Result<()> {
    ensure!(r > 0, "{op}: r must be > 0");

    let log_delta = y.log_delta();
    let consumed = 2 * r * log_delta;
    ensure!(
        y.log_budget() > consumed,
        "{op}: log_budget {} <= {consumed} bits required for {r} iterations at log_delta {log_delta}",
        y.log_budget(),
    );
    Ok(())
}

/// Reciprocal and inverse square root.
pub trait CKKSInverseOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_goldschmidt_division`].
    fn ckks_goldschmidt_division_tmp_bytes<C, T>(&self, ct: &C, tsk: &T) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos;

    /// Computes `ct <- 1/ct` on `[0, 2]`. Consumes `(iters + 1) * log_delta` bits.
    ///
    /// # Metadata
    ///
    /// `ct` keeps its scale and loses the consumed budget.
    ///
    /// # Errors
    ///
    /// An error for `iters == 0` or when `ct` holds no more budget than the iterations consume, and the errors of the
    /// arithmetic. The iterations run on a scratch copy, so `ct` is unchanged on `Err`.
    fn ckks_goldschmidt_division<C, H>(
        &self,
        ct: &mut C,
        iters: usize,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>;

    /// Scratch bytes for [`Self::ckks_rsqrt`].
    fn ckks_rsqrt_tmp_bytes<Y, X, T>(&self, y: &Y, in_half: &X, tsk: &T) -> usize
    where
        Y: CKKSCtBounds,
        X: CKKSCtBounds,
        T: GGLWEInfos;

    /// Refines `y <- 1/sqrt(x)` with `r` Newton steps, `in_half` holding `x/2`. Consumes `2r * log_delta` bits.
    ///
    /// # Metadata
    ///
    /// `y` keeps its scale and loses the consumed budget.
    ///
    /// # Errors
    ///
    /// An error for `r == 0` or when `y` holds no more budget than the steps consume, and the errors of the arithmetic.
    /// The steps run on a scratch copy, so `y` is unchanged on `Err`.
    fn ckks_rsqrt<Y, X, H>(&self, y: &mut Y, in_half: &X, r: usize, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Y: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>;
}

/// Reciprocal over signed intervals.
pub trait CKKSInverseDomainOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_interval_normalization`].
    fn ckks_interval_normalization_tmp_bytes<N, C, T, P>(&self, norm: &N, ct: &C, tsk: &T, params: &IntervalNorm<P>) -> usize
    where
        N: CKKSCtBounds,
        C: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// Compresses `ct` to `[-1, 1]` and writes its factor to `norm`.
    ///
    /// # Metadata
    ///
    /// Both outputs are computed at the layout of `ct`, which loses `params.consumed_bits(ct.log_delta())` bits, then
    /// copied to `ct` and `norm`.
    ///
    /// # Errors
    ///
    /// An error when `ct` holds no more budget than `params` consumes, and the errors of the arithmetic, with `ct` and
    /// `norm` unchanged. The two final `ckks_copy` calls write `ct`, then `norm`.
    fn ckks_interval_normalization<N, C, P, H>(
        &self,
        norm: &mut N,
        ct: &mut C,
        params: &IntervalNorm<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        N: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<BE>;

    /// Scratch bytes for [`Self::ckks_inverse_positive_domain`], [`Self::ckks_inverse_negative_domain`] and
    /// [`Self::ckks_inverse_full_domain`].
    fn ckks_inverse_domain_tmp_bytes<C, T, K, P>(&self, ct: &C, tsk: &T, conj_key: &K, coeff_prec: &P) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
        K: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// `ct <- 1/ct` for `ct` in `[min, 2 - min]`. Alias of Goldschmidt division.
    ///
    /// # Errors
    ///
    /// As [`CKKSInverseOps::ckks_goldschmidt_division`], with `ct` unchanged on `Err`.
    fn ckks_inverse_positive_domain<C, H>(
        &self,
        ct: &mut C,
        iters: usize,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>;

    /// `ct <- 1/ct` on `[-(2 - min), -min]`.
    ///
    /// # Errors
    ///
    /// As [`CKKSInverseOps::ckks_goldschmidt_division`], with `ct` unchanged on `Err`.
    fn ckks_inverse_negative_domain<C, H>(
        &self,
        ct: &mut C,
        iters: usize,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>;

    /// `ct <- 1/ct` on a signed domain outside the composite gap: the magnitude is inverted, then the sign restored.
    ///
    /// `conj_key` answers the conjugation, Galois element `-1`.
    ///
    /// # Metadata
    ///
    /// `ct` keeps its scale and loses `composite.consumed_bits(log_delta) + (iters + 3) * log_delta` bits.
    ///
    /// # Errors
    ///
    /// An error for `iters == 0` or when `ct` holds no more budget than the operation consumes, and the errors of the
    /// sign, the inversion and the products. Only the last product writes `ct`, so `ct` is unchanged on `Err`.
    fn ckks_inverse_full_domain<C, F, P, H, K>(
        &self,
        ct: &mut C,
        iters: usize,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;
}

impl<BE: Backend> CKKSInverseOps<BE> for Module<BE>
where
    Module<BE>: CKKSMulOps<BE> + CKKSAddOps<BE> + CKKSNegOps<BE> + CKKSPow2Ops<BE> + CKKSCopyOps<BE> + GLWEBytesOf<BE>,
{
    fn ckks_goldschmidt_division_tmp_bytes<C, T>(&self, ct: &C, tsk: &T) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
    {
        // `a`, `b` and the product, each with the layout of `ct`.
        3 * BE::scratch_aligned(self.glwe_bytes_of_from_infos(ct))
            + self
                .ckks_copy_tmp_bytes(ct, ct)
                .max(self.ckks_add_one_tmp_bytes(ct.size()))
                .max(self.ckks_square_tmp_bytes(ct, ct, tsk))
                .max(self.ckks_mul_tmp_bytes(ct, ct, ct, tsk))
                .max(self.ckks_add_tmp_bytes(ct.size()))
    }

    fn ckks_goldschmidt_division<C, H>(&self, ct: &mut C, iters: usize, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>,
    {
        ensure_goldschmidt_capacity("ckks_goldschmidt_division", &*ct, iters)?;

        // a := 2 - m and b := 1 - m, whose product converges to 1/m.
        scratch.scope(|scratch_local| {
            let (mut a, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut b, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            self.ckks_copy(&mut a, &*ct, &mut scratch_local)?;
            self.ckks_neg_assign(&mut a)?;
            self.ckks_copy(&mut b, &a, &mut scratch_local)?;
            self.ckks_add_one_assign(&mut b, &mut scratch_local)?;
            self.ckks_add_one_assign(&mut a, &mut scratch_local)?;
            self.ckks_add_one_assign(&mut a, &mut scratch_local)?;

            for _ in 0..iters {
                self.ckks_square_assign(&mut b, tsk, &mut scratch_local)?;
                self.ckks_mul_into(&mut tmp, &a, &b, tsk, &mut scratch_local)?;
                self.ckks_add_assign(&mut a, &tmp, &mut scratch_local)?;
            }

            self.ckks_copy(ct, &a, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_rsqrt_tmp_bytes<Y, X, T>(&self, y: &Y, in_half: &X, tsk: &T) -> usize
    where
        Y: CKKSCtBounds,
        X: CKKSCtBounds,
        T: GGLWEInfos,
    {
        // The iterate, its square, `x y / 2`, the update and `y / 2`, each with the layout of `y`.
        5 * BE::scratch_aligned(self.glwe_bytes_of_from_infos(y))
            + self
                .ckks_copy_tmp_bytes(y, y)
                .max(self.ckks_square_tmp_bytes(y, y, tsk))
                .max(self.ckks_mul_tmp_bytes(y, in_half, y, tsk))
                .max(self.ckks_mul_tmp_bytes(y, y, y, tsk))
                .max(self.ckks_div_pow2_tmp_bytes(y.size()))
                .max(self.ckks_add_tmp_bytes(y.size()))
    }

    fn ckks_rsqrt<Y, X, H>(&self, y: &mut Y, in_half: &X, r: usize, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Y: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        ensure_rsqrt_capacity("ckks_rsqrt", &*y, r)?;

        // y <- 1.5 y - (x/2) y y^2, with 1.5 y = y + y/2.
        scratch.scope(|scratch_local| {
            let (mut it, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*y);
            let (mut ysqr, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*y);
            let (mut xy, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*y);
            let (mut t, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*y);
            let (mut half_y, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*y);
            self.ckks_copy(&mut it, &*y, &mut scratch_local)?;

            for _ in 0..r {
                self.ckks_square_into(&mut ysqr, &it, tsk, &mut scratch_local)?;
                self.ckks_mul_into(&mut xy, in_half, &it, tsk, &mut scratch_local)?;
                self.ckks_mul_into(&mut t, &ysqr, &xy, tsk, &mut scratch_local)?;
                self.ckks_neg_assign(&mut t)?;
                self.ckks_div_pow2_into(&mut half_y, &it, 1, &mut scratch_local)?;
                self.ckks_add_assign(&mut t, &it, &mut scratch_local)?;
                self.ckks_add_assign(&mut t, &half_y, &mut scratch_local)?;
                self.ckks_copy(&mut it, &t, &mut scratch_local)?;
            }

            self.ckks_copy(y, &it, &mut scratch_local)?;
            Ok(())
        })
    }
}

impl<BE: Backend<Ring = Standard>> CKKSInverseDomainOps<BE> for Module<BE>
where
    Module<BE>: CKKSInverseOps<BE>
        + CKKSAllOpsTmpBytes<BE>
        + CKKSMulOps<BE>
        + CKKSNegOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSPow2Ops<BE>
        + GLWEBytesOf<BE>,
{
    fn ckks_interval_normalization_tmp_bytes<N, C, T, P>(&self, norm: &N, ct: &C, tsk: &T, params: &IntervalNorm<P>) -> usize
    where
        N: CKKSCtBounds,
        C: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        // The compressed copy, the factor and five terms, each with the layout of `ct`.
        7 * BE::scratch_aligned(self.glwe_bytes_of_from_infos(ct))
            + self
                .ckks_mul_pt_const_tmp_bytes(ct, ct, &params.consts)
                .max(self.ckks_square_tmp_bytes(ct, ct, tsk))
                .max(self.ckks_mul_tmp_bytes(ct, ct, ct, tsk))
                .max(self.ckks_copy_tmp_bytes(ct, ct))
                .max(self.ckks_add_one_tmp_bytes(ct.size()))
                .max(self.ckks_sub_tmp_bytes(ct.size()))
                .max(self.ckks_copy_tmp_bytes(norm, ct))
    }

    fn ckks_interval_normalization<N, C, P, H>(
        &self,
        norm: &mut N,
        ct: &mut C,
        params: &IntervalNorm<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        N: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<BE>,
    {
        let required = params.consumed_bits(ct.log_delta());
        ensure!(
            ct.log_budget() > required,
            "ckks_interval_normalization: log_budget {} <= {required} bits required",
            ct.log_budget(),
        );

        // Track each compression factor in `factor`.
        scratch.scope(|scratch_local| {
            let (mut x, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut factor, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut z0, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut z1, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut z2, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut z0z1, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut z0z2, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            self.ckks_copy(&mut x, &*ct, &mut scratch_local)?;

            for i in 0..params.n {
                self.ckks_mul_pt_const_into(&mut z0, &x, &params.consts, i, &mut scratch_local)?;
                self.ckks_square_into(&mut z1, &x, tsk, &mut scratch_local)?;
                self.ckks_mul_into(&mut z0z1, &z0, &z1, tsk, &mut scratch_local)?;
                if i == 0 {
                    self.ckks_mul_into(&mut z0z2, &z0, &x, tsk, &mut scratch_local)?;
                    self.ckks_copy(&mut factor, &z0z2, &mut scratch_local)?;
                    self.ckks_neg_assign(&mut factor)?;
                    self.ckks_add_one_assign(&mut factor, &mut scratch_local)?;
                } else {
                    self.ckks_mul_into(&mut z2, &factor, &x, tsk, &mut scratch_local)?;
                    self.ckks_mul_into(&mut z0z2, &z0, &z2, tsk, &mut scratch_local)?;
                    self.ckks_sub_assign(&mut factor, &z0z2, &mut scratch_local)?;
                }
                self.ckks_sub_assign(&mut x, &z0z1, &mut scratch_local)?;
            }

            self.ckks_copy(ct, &x, &mut scratch_local)?;
            self.ckks_copy(norm, &factor, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_inverse_domain_tmp_bytes<C, T, K, P>(&self, ct: &C, tsk: &T, conj_key: &K, coeff_prec: &P) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
        K: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        // The sign and the magnitude, or the negated copy, each with the layout of `ct`.
        2 * BE::scratch_aligned(self.glwe_bytes_of_from_infos(ct))
            + self
                .ckks_goldschmidt_division_tmp_bytes(ct, tsk)
                .max(ckks_sign_tmp_bytes(self, ct, ct, tsk, conj_key, coeff_prec))
                .max(self.ckks_mul_tmp_bytes(ct, ct, ct, tsk))
                .max(self.ckks_copy_tmp_bytes(ct, ct))
    }

    fn ckks_inverse_positive_domain<C, H>(
        &self,
        ct: &mut C,
        iters: usize,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>,
    {
        self.ckks_goldschmidt_division(ct, iters, tsk, scratch)
    }

    fn ckks_inverse_negative_domain<C, H>(
        &self,
        ct: &mut C,
        iters: usize,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>,
    {
        ensure_goldschmidt_capacity("ckks_inverse_negative_domain", &*ct, iters)?;
        scratch.scope(|scratch_local| {
            let (mut m, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            self.ckks_copy(&mut m, &*ct, &mut scratch_local)?;
            self.ckks_neg_assign(&mut m)?;
            self.ckks_goldschmidt_division(&mut m, iters, tsk, &mut scratch_local)?;
            self.ckks_neg_assign(&mut m)?;
            self.ckks_copy(ct, &m, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_inverse_full_domain<C, F, P, H, K>(
        &self,
        ct: &mut C,
        iters: usize,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
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
            let (mut sign, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            let (mut m, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*ct);
            ckks_sign_into(self, &mut sign, &*ct, composite, tsk, conj_key, false, &mut scratch_local)?;
            self.ckks_mul_into(&mut m, &*ct, &sign, tsk, &mut scratch_local)?;
            self.ckks_goldschmidt_division(&mut m, iters, tsk, &mut scratch_local)?;
            self.ckks_mul_into(ct, &m, &sign, tsk, &mut scratch_local)?;
            Ok(())
        })
    }
}

/// Division and square root over the iterative kernels.
pub trait CKKSDivSqrtOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_div`].
    fn ckks_div_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos;

    /// `res <- a / b` for `b` in `[min, 2 - min]`.
    ///
    /// # Metadata
    ///
    /// As `ckks_mul_into` of the reciprocal of `b`, computed at the layout of `b`, and `a`.
    ///
    /// # Errors
    ///
    /// The errors of [`CKKSInverseOps::ckks_goldschmidt_division`] on a copy of `b` and of the product, which is the
    /// only write of `res`, so `res` is untouched on `Err`.
    fn ckks_div<R, A, B, H>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        iters: usize,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>;

    /// Scratch bytes for [`Self::ckks_sqrt`].
    fn ckks_sqrt_tmp_bytes<R, X, Y, T>(&self, res: &R, x: &X, in_half: &Y, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        X: CKKSCtBounds,
        Y: CKKSCtBounds,
        T: GGLWEInfos;

    /// `res <- sqrt(x) = x / sqrt(x)` near `1`, `in_half` holding `x/2` at the same scale.
    ///
    /// # Metadata
    ///
    /// As `ckks_mul_into` of the inverse square root, refined at the layout of `x`, and `x`.
    ///
    /// # Errors
    ///
    /// The errors of [`CKKSInverseOps::ckks_rsqrt`] on a copy of `x` and of the product, which is the only write of
    /// `res`, so `res` is untouched on `Err`.
    fn ckks_sqrt<R, X, Y, H>(
        &self,
        res: &mut R,
        x: &X,
        in_half: &Y,
        r: usize,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        Y: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>;
}

impl<BE: Backend> CKKSDivSqrtOps<BE> for Module<BE>
where
    Module<BE>: CKKSInverseOps<BE> + CKKSMulOps<BE> + CKKSCopyOps<BE> + GLWEBytesOf<BE>,
{
    fn ckks_div_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        // The reciprocal has the layout of `b`.
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(b))
            + self
                .ckks_copy_tmp_bytes(b, b)
                .max(self.ckks_goldschmidt_division_tmp_bytes(b, tsk))
                .max(self.ckks_mul_tmp_bytes(res, b, a, tsk))
    }

    fn ckks_div<R, A, B, H>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        iters: usize,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut inv, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(b);
            self.ckks_copy(&mut inv, b, &mut scratch_local)?;
            self.ckks_goldschmidt_division(&mut inv, iters, tsk, &mut scratch_local)?;
            self.ckks_mul_into(res, &inv, a, tsk, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_sqrt_tmp_bytes<R, X, Y, T>(&self, res: &R, x: &X, in_half: &Y, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        X: CKKSCtBounds,
        Y: CKKSCtBounds,
        T: GGLWEInfos,
    {
        // The inverse square root has the layout of `x`.
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(x))
            + self
                .ckks_copy_tmp_bytes(x, x)
                .max(self.ckks_rsqrt_tmp_bytes(x, in_half, tsk))
                .max(self.ckks_mul_tmp_bytes(res, x, x, tsk))
    }

    fn ckks_sqrt<R, X, Y, H>(
        &self,
        res: &mut R,
        x: &X,
        in_half: &Y,
        r: usize,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        Y: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut inv, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            self.ckks_copy(&mut inv, x, &mut scratch_local)?;
            self.ckks_rsqrt(&mut inv, in_half, r, tsk, &mut scratch_local)?;
            self.ckks_mul_into(res, &inv, x, tsk, &mut scratch_local)?;
            Ok(())
        })
    }
}
