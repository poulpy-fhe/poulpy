//! Predicates and selection built on `sign`.
//!
//! `gt`/`ge` and `lt`/`le` coincide under smooth CKKS comparison. Sign inputs
//! must lie in `[-1, 1]` outside the composite gap.

use anyhow::Result;
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

use super::{SignComposite, ckks_sign_into, ckks_sign_tmp_bytes};

/// Sign-based predicates and select.
///
/// The predicates build their terms on scratch temporaries and write `res` once, with their last call. `conj_key`
/// answers the conjugation, Galois element `-1`.
pub trait CKKSPredicateOps<BE: Backend> {
    /// Scratch bytes for the sign-based predicates, whose two ciphertext operands are `a` and `b`: `x` and `y` for
    /// [`Self::ckks_copysign`], `x` for both in [`Self::ckks_fabs`] and [`Self::ckks_indicator`].
    fn ckks_predicate_tmp_bytes<R, A, B, T, K, P>(&self, res: &R, a: &A, b: &B, tsk: &T, conj_key: &K, coeff_prec: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
        K: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// `res <- |x|` via `x * sign(x)`.
    ///
    /// # Metadata
    ///
    /// As `ckks_mul_into` of `x` and its sign, computed at the layout of `x`.
    ///
    /// # Errors
    ///
    /// The errors of [`CKKSSignOps::ckks_sign`](crate::comparison::CKKSSignOps::ckks_sign) and of the product, which is
    /// the only write of `res`, so `res` is untouched on `Err`.
    fn ckks_fabs<R, X, F, P, H, K>(
        &self,
        res: &mut R,
        x: &X,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// `res <- sign(a - b)` (`+1`, `-1` or `0` for `a >`, `<` or `= b`).
    ///
    /// # Metadata
    ///
    /// As [`CKKSSignOps::ckks_sign`](crate::comparison::CKKSSignOps::ckks_sign) on the difference, computed at the layout
    /// of `a`.
    ///
    /// # Errors
    ///
    /// The errors of the difference and of `ckks_sign`, with `res` untouched on `Err`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_cmp<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// `res <- [a > b] = step(a - b)` (`1`, `0` or `1/2`). `ge` is identical.
    ///
    /// # Metadata
    ///
    /// As [`CKKSSignOps::ckks_step`](crate::comparison::CKKSSignOps::ckks_step) on the difference, computed at the layout
    /// of `a`.
    ///
    /// # Errors
    ///
    /// The errors of the difference and of `ckks_step`, with `res` untouched on `Err`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_gt<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// Alias of [`Self::ckks_gt`].
    #[allow(clippy::too_many_arguments)]
    fn ckks_ge<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// `res <- [a < b] = step(b - a)`. `le` is identical.
    ///
    /// # Metadata
    ///
    /// As [`CKKSSignOps::ckks_step`](crate::comparison::CKKSSignOps::ckks_step) on the difference, computed at the layout
    /// of `b`.
    ///
    /// # Errors
    ///
    /// The errors of the difference and of `ckks_step`, with `res` untouched on `Err`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_lt<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// Alias of [`Self::ckks_lt`].
    #[allow(clippy::too_many_arguments)]
    fn ckks_le<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// Smooth membership in `[bounds[lo_coeff], bounds[hi_coeff]]`: `step(x - lo) * step(hi - x)`.
    ///
    /// # Metadata
    ///
    /// As `ckks_mul_into` of the two steps, computed at the layout of `x`.
    ///
    /// # Errors
    ///
    /// The errors of the differences, of the two steps and of the product, which is the only write of `res`, so `res`
    /// is untouched on `Err`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_indicator<R, X, F, P, Q, H, K>(
        &self,
        res: &mut R,
        x: &X,
        bounds: &Q,
        lo_coeff: usize,
        hi_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        Q: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// Smooth equality mask for `|a - b| <= epsilon[epsilon_coeff]`: `step(a - b + eps) * step(b - a + eps)`.
    ///
    /// # Metadata
    ///
    /// As `ckks_mul_into` of the two steps, computed at the layout of `a`.
    ///
    /// # Errors
    ///
    /// As [`Self::ckks_indicator`], with `res` untouched on `Err`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_eq<R, A, B, F, P, E, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        epsilon: &E,
        epsilon_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        E: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// `res <- max(a - b, 0) = (d * sign(d) + d) / 2` for `d = a - b`, which must satisfy the sign plan's domain.
    ///
    /// # Metadata
    ///
    /// The terms run at the layout of `a`, the halving adding one bit to `log_delta`, then `ckks_copy` writes the result
    /// to `res`.
    ///
    /// # Errors
    ///
    /// The errors of the difference, of `ckks_sign` and of the arithmetic. Only the final `ckks_copy` writes `res`, so
    /// `res` is untouched on `Err`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_fdim<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// `res <- |x| * sign(y)`, for `x` and `y` in the sign plan's domain.
    ///
    /// # Metadata
    ///
    /// As `ckks_mul_into` of `|x|`, computed at the layout of `x`, and the sign of `y`, at the layout of `y`.
    ///
    /// # Errors
    ///
    /// The errors of the two signs and of the products. Only the last product writes `res`, so `res` is untouched on
    /// `Err`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_copysign<R, X, Y, F, P, H, K>(
        &self,
        res: &mut R,
        x: &X,
        y: &Y,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        Y: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// Scratch bytes for [`Self::ckks_select`].
    fn ckks_select_tmp_bytes<R, M, A, B, T>(&self, res: &R, mask: &M, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        M: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos;

    /// `res <- mask * a + (1 - mask) * b` for a smooth binary mask.
    ///
    /// # Metadata
    ///
    /// As `ckks_add_into` of `mask * (a - b)`, computed at the layout of `a`, and `b`.
    ///
    /// # Errors
    ///
    /// The errors of the difference, the product and the sum, which is the only write of `res`, so `res` is untouched
    /// on `Err`.
    fn ckks_select<R, M, A, B, H>(
        &self,
        res: &mut R,
        mask: &M,
        a: &A,
        b: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        M: GLWEToBackendRef<BE> + CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>;
}

impl<BE: Backend<Ring = Standard>> CKKSPredicateOps<BE> for Module<BE>
where
    Module<BE>: CKKSAllOpsTmpBytes<BE>
        + CKKSSubOps<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSMulOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + GLWEBytesOf<BE>,
{
    fn ckks_predicate_tmp_bytes<R, A, B, T, K, P>(&self, res: &R, a: &A, b: &B, tsk: &T, conj_key: &K, coeff_prec: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
        K: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        // At most three temporaries, each with the layout of `a` or `b`.
        let ct = BE::scratch_aligned(self.glwe_bytes_of_from_infos(a)).max(BE::scratch_aligned(self.glwe_bytes_of_from_infos(b)));
        let size = a.size().max(b.size());
        3 * ct
            + ckks_sign_tmp_bytes(self, a, a, tsk, conj_key, coeff_prec)
                .max(ckks_sign_tmp_bytes(self, b, b, tsk, conj_key, coeff_prec))
                .max(self.ckks_sub_tmp_bytes(size))
                .max(self.ckks_sub_pt_const_tmp_bytes(size))
                .max(self.ckks_add_pt_const_tmp_bytes(size))
                .max(self.ckks_add_tmp_bytes(size))
                .max(self.ckks_mul_tmp_bytes(a, a, a, tsk))
                .max(self.ckks_mul_tmp_bytes(a, a, b, tsk))
                .max(self.ckks_mul_tmp_bytes(res, a, b, tsk))
                .max(self.ckks_mul_tmp_bytes(res, a, a, tsk))
                .max(self.ckks_copy_tmp_bytes(res, a))
    }

    fn ckks_fabs<R, X, F, P, H, K>(
        &self,
        res: &mut R,
        x: &X,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut sgn, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            ckks_sign_into(self, &mut sgn, x, composite, tsk, conj_key, false, &mut scratch_local)?;
            self.ckks_mul_into(res, x, &sgn, tsk, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_cmp<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut diff, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            self.ckks_sub_into(&mut diff, a, b, &mut scratch_local)?;
            ckks_sign_into(self, res, &diff, composite, tsk, conj_key, false, &mut scratch_local)
        })
    }

    fn ckks_gt<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        step_of_diff(self, res, a, b, composite, tsk, conj_key, scratch)
    }

    fn ckks_ge<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        self.ckks_gt(res, a, b, composite, tsk, conj_key, scratch)
    }

    fn ckks_lt<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        // a < b  <=>  b > a.
        step_of_diff(self, res, b, a, composite, tsk, conj_key, scratch)
    }

    fn ckks_le<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        self.ckks_lt(res, a, b, composite, tsk, conj_key, scratch)
    }

    fn ckks_indicator<R, X, F, P, Q, H, K>(
        &self,
        res: &mut R,
        x: &X,
        bounds: &Q,
        lo_coeff: usize,
        hi_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        Q: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut lower, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            let (mut gate, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            let (mut diff, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            self.ckks_sub_pt_const_into(&mut diff, x, 0, bounds, lo_coeff, &mut scratch_local)?;
            ckks_sign_into(self, &mut lower, &diff, composite, tsk, conj_key, true, &mut scratch_local)?;
            self.ckks_sub_pt_const_into(&mut diff, x, 0, bounds, hi_coeff, &mut scratch_local)?;
            self.ckks_neg_assign(&mut diff)?;
            ckks_sign_into(self, &mut gate, &diff, composite, tsk, conj_key, true, &mut scratch_local)?;
            self.ckks_mul_into(res, &lower, &gate, tsk, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_eq<R, A, B, F, P, E, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        epsilon: &E,
        epsilon_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        E: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut lower, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            let (mut gate, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            let (mut diff, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            self.ckks_sub_into(&mut diff, a, b, &mut scratch_local)?;
            self.ckks_add_pt_const_assign(&mut diff, 0, epsilon, epsilon_coeff, &mut scratch_local)?;
            ckks_sign_into(self, &mut lower, &diff, composite, tsk, conj_key, true, &mut scratch_local)?;
            self.ckks_sub_into(&mut diff, b, a, &mut scratch_local)?;
            self.ckks_add_pt_const_assign(&mut diff, 0, epsilon, epsilon_coeff, &mut scratch_local)?;
            ckks_sign_into(self, &mut gate, &diff, composite, tsk, conj_key, true, &mut scratch_local)?;
            self.ckks_mul_into(res, &lower, &gate, tsk, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_fdim<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut diff, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            let (mut sign, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            let (mut acc, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            self.ckks_sub_into(&mut diff, a, b, &mut scratch_local)?;
            ckks_sign_into(self, &mut sign, &diff, composite, tsk, conj_key, false, &mut scratch_local)?;
            self.ckks_mul_into(&mut acc, &diff, &sign, tsk, &mut scratch_local)?;
            self.ckks_add_assign(&mut acc, &diff, &mut scratch_local)?;
            self.ckks_div_pow2_assign(1, &mut acc)?;
            self.ckks_copy(res, &acc, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_copysign<R, X, Y, F, P, H, K>(
        &self,
        res: &mut R,
        x: &X,
        y: &Y,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        Y: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut abs, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            let (mut sign_x, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            let (mut sign_y, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(y);
            ckks_sign_into(self, &mut sign_x, x, composite, tsk, conj_key, false, &mut scratch_local)?;
            self.ckks_mul_into(&mut abs, x, &sign_x, tsk, &mut scratch_local)?;
            ckks_sign_into(self, &mut sign_y, y, composite, tsk, conj_key, false, &mut scratch_local)?;
            self.ckks_mul_into(res, &abs, &sign_y, tsk, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_select_tmp_bytes<R, M, A, B, T>(&self, res: &R, mask: &M, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        M: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        // The difference and the product have the layout of `a`.
        let _ = b;
        2 * BE::scratch_aligned(self.glwe_bytes_of_from_infos(a))
            + self
                .ckks_sub_tmp_bytes(a.size())
                .max(self.ckks_mul_tmp_bytes(a, mask, a, tsk))
                .max(self.ckks_add_tmp_bytes(res.size()))
    }

    fn ckks_select<R, M, A, B, H>(
        &self,
        res: &mut R,
        mask: &M,
        a: &A,
        b: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        M: GLWEToBackendRef<BE> + CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut diff, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            let (mut prod, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
            self.ckks_sub_into(&mut diff, a, b, &mut scratch_local)?;
            self.ckks_mul_into(&mut prod, mask, &diff, tsk, &mut scratch_local)?;
            self.ckks_add_into(res, &prod, b, &mut scratch_local)?;
            Ok(())
        })
    }
}

/// `res <- step(a - b) = (sign(a - b) + 1) / 2`.
#[allow(clippy::too_many_arguments)]
fn step_of_diff<BE, R, A, B, F, P, H, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    b: &B,
    composite: &SignComposite<F, P>,
    tsk: &H,
    conj_key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend<Ring = Standard>,
    Module<BE>: CKKSSubOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSConjugateOps<BE> + CKKSPow2Ops<BE>,
    Module<BE>: CKKSPolynomialEvaluationOps<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    B: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
    H: GetTensorKey<BE>,
    K: GetAutomorphismKey<BE>,
{
    scratch.scope(|scratch_local| {
        let (mut diff, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(a);
        module.ckks_sub_into(&mut diff, a, b, &mut scratch_local)?;
        ckks_sign_into(module, res, &diff, composite, tsk, conj_key, true, &mut scratch_local)
    })
}
