//! Smooth extrema built on `sign`.

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
    layouts::ScratchArenaTakeCKKSCiphertext,
};

use super::{SignComposite, ckks_sign_into, ckks_sign_tmp_bytes};

/// Smooth extrema and clamp.
///
/// Every operation evaluates `step(d) * d` for a difference `d`, on scratch temporaries with the layout of its first
/// operand, and writes `res` once, with its last call. `keys` answers the conjugation, Galois element `-1`.
pub trait CKKSComparisonOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_fmax`] and [`Self::ckks_fmin`].
    fn ckks_comparison_tmp_bytes<R, A, B, P, T, K>(&self, res: &R, op0: &A, op1: &B, coeff_prec: &P, tsk: &T, keys: &K) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        P: CKKSInfos + LWEInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// Scratch bytes for [`Self::ckks_fmax_const`], [`Self::ckks_fmin_const`] and [`Self::ckks_clamp`].
    fn ckks_comparison_const_tmp_bytes<R, X, P, T, K>(&self, res: &R, x: &X, coeff_prec: &P, tsk: &T, keys: &K) -> usize
    where
        R: CKKSCtBounds,
        X: CKKSCtBounds,
        P: CKKSInfos + LWEInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// `res <- fmax(op0, op1) = op1 + step(op0 - op1) * (op0 - op1)`.
    ///
    /// # Metadata
    ///
    /// As `ckks_add_into` of the correction, computed at the layout of `op0`, and `op1`.
    ///
    /// # Errors
    ///
    /// The errors of [`CKKSSignOps::ckks_sign`](crate::comparison::CKKSSignOps::ckks_sign) on the difference and of the
    /// arithmetic around it. Only the last call writes `res`, so `res` is untouched on `Err`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_fmax<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        op0: &A,
        op1: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// `res <- fmin(op0, op1) = op0 - step(op0 - op1) * (op0 - op1)`.
    ///
    /// # Metadata
    ///
    /// As `ckks_add_into` of the negated correction, computed at the layout of `op0`, and `op0`.
    ///
    /// # Errors
    ///
    /// As [`Self::ckks_fmax`], with `res` untouched on `Err`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_fmin<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        op0: &A,
        op1: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// `res <- fmax(x, bound[bound_coeff])`.
    ///
    /// # Metadata
    ///
    /// As `ckks_add_into` of the correction, computed at the layout of `x`, and `x`.
    ///
    /// # Errors
    ///
    /// As [`Self::ckks_fmax`], with `res` untouched on `Err`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_fmax_const<R, X, F, P, B, H, K>(
        &self,
        res: &mut R,
        x: &X,
        bound: &B,
        bound_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        B: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// `res <- fmin(x, bound[bound_coeff])`.
    ///
    /// # Metadata
    ///
    /// As `ckks_add_into` of the negated correction, computed at the layout of `x`, and `x`.
    ///
    /// # Errors
    ///
    /// As [`Self::ckks_fmax`], with `res` untouched on `Err`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_fmin_const<R, X, F, P, B, H, K>(
        &self,
        res: &mut R,
        x: &X,
        bound: &B,
        bound_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        B: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// `res <- min(max(x, bounds[lo_coeff]), bounds[hi_coeff])`.
    ///
    /// # Metadata
    ///
    /// As `ckks_sub_into` of the upper correction from the lower clamp, both computed at the layout of `x`.
    ///
    /// # Errors
    ///
    /// As [`Self::ckks_fmax`], with `res` untouched on `Err`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_clamp<R, X, F, P, B, H, K>(
        &self,
        res: &mut R,
        x: &X,
        bounds: &B,
        lo_coeff: usize,
        hi_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        B: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;
}

impl<BE: Backend<Ring = Standard>> CKKSComparisonOps<BE> for Module<BE>
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
    fn ckks_comparison_tmp_bytes<R, A, B, P, T, K>(&self, res: &R, op0: &A, op1: &B, coeff_prec: &P, tsk: &T, keys: &K) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        P: CKKSInfos + LWEInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        // `op1` enters only the difference and the final sum, sized from their destinations.
        let _ = op1;
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(op0))
            + correction_tmp_bytes(self, op0, coeff_prec, tsk, keys).max(self.ckks_add_tmp_bytes(res.size()))
    }

    fn ckks_comparison_const_tmp_bytes<R, X, P, T, K>(&self, res: &R, x: &X, coeff_prec: &P, tsk: &T, keys: &K) -> usize
    where
        R: CKKSCtBounds,
        X: CKKSCtBounds,
        P: CKKSInfos + LWEInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        // `ckks_clamp` keeps the lower clamp and the upper correction.
        2 * BE::scratch_aligned(self.glwe_bytes_of_from_infos(x))
            + correction_tmp_bytes(self, x, coeff_prec, tsk, keys)
                .max(self.ckks_add_tmp_bytes(x.size()))
                .max(self.ckks_add_tmp_bytes(res.size()))
                .max(self.ckks_sub_tmp_bytes(res.size()))
    }

    fn ckks_fmax<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        op0: &A,
        op1: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
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
            let (mut corr, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(op0);
            correction_into(
                self,
                false,
                &mut corr,
                op0,
                Diff::<B, P>::Ct(op1),
                composite,
                tsk,
                keys,
                &mut scratch_local,
            )?;
            self.ckks_add_into(res, &corr, op1, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_fmin<R, A, B, F, P, H, K>(
        &self,
        res: &mut R,
        op0: &A,
        op1: &B,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
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
            let (mut corr, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(op0);
            correction_into(
                self,
                false,
                &mut corr,
                op0,
                Diff::<B, P>::Ct(op1),
                composite,
                tsk,
                keys,
                &mut scratch_local,
            )?;
            self.ckks_neg_assign(&mut corr)?;
            self.ckks_add_into(res, &corr, op0, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_fmax_const<R, X, F, P, B, H, K>(
        &self,
        res: &mut R,
        x: &X,
        bound: &B,
        bound_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        B: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut corr, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            let diff = Diff::<X, B>::Const(bound, bound_coeff);
            correction_into(self, true, &mut corr, x, diff, composite, tsk, keys, &mut scratch_local)?;
            self.ckks_add_into(res, &corr, x, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_fmin_const<R, X, F, P, B, H, K>(
        &self,
        res: &mut R,
        x: &X,
        bound: &B,
        bound_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        B: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut corr, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            let diff = Diff::<X, B>::Const(bound, bound_coeff);
            correction_into(self, false, &mut corr, x, diff, composite, tsk, keys, &mut scratch_local)?;
            self.ckks_neg_assign(&mut corr)?;
            self.ckks_add_into(res, &corr, x, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_clamp<R, X, F, P, B, H, K>(
        &self,
        res: &mut R,
        x: &X,
        bounds: &B,
        lo_coeff: usize,
        hi_coeff: usize,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        X: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        B: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut lower, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            let (mut corr, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            // lower = fmax(x, lo)
            let lo = Diff::<X, B>::Const(bounds, lo_coeff);
            correction_into(self, true, &mut corr, x, lo, composite, tsk, keys, &mut scratch_local)?;
            self.ckks_add_into(&mut lower, &corr, x, &mut scratch_local)?;
            // res = lower - step(lower - hi) * (lower - hi)
            let hi = Diff::<X, B>::Const(bounds, hi_coeff);
            correction_into(self, false, &mut corr, &lower, hi, composite, tsk, keys, &mut scratch_local)?;
            self.ckks_sub_into(res, &lower, &corr, &mut scratch_local)?;
            Ok(())
        })
    }
}

/// The subtrahend of a comparison difference: a ciphertext or a coefficient of a constant plaintext.
enum Diff<'a, C, B> {
    Ct(&'a C),
    Const(&'a B, usize),
}

/// Query of [`correction_into`] for a minuend with the layout of `x`.
fn correction_tmp_bytes<BE, X, P, T, K>(module: &Module<BE>, x: &X, coeff_prec: &P, tsk: &T, keys: &K) -> usize
where
    BE: Backend,
    Module<BE>: CKKSAllOpsTmpBytes<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSPow2Ops<BE> + CKKSSubOps<BE>,
    Module<BE>: GLWEBytesOf<BE>,
    X: CKKSCtBounds,
    P: CKKSInfos + LWEInfos,
    T: GGLWEInfos,
    K: GGLWEInfos,
{
    // The difference, the step and the halved product coexist.
    3 * BE::scratch_aligned(module.glwe_bytes_of_from_infos(x))
        + module
            .ckks_sub_tmp_bytes(x.size())
            .max(module.ckks_sub_pt_const_tmp_bytes(x.size()))
            .max(ckks_sign_tmp_bytes(module, x, x, coeff_prec, tsk, keys))
            .max(module.ckks_add_one_tmp_bytes(x.size()))
            .max(module.ckks_mul_tmp_bytes(x, x, x, tsk))
            .max(module.ckks_div_pow2_tmp_bytes(x.size()))
            .max(module.ckks_copy_tmp_bytes(x, x))
}

/// Writes `step(d) * d` into `corr`, for `d = x - subtrahend`, negated when `reverse` is set.
#[allow(clippy::too_many_arguments)]
fn correction_into<BE, C, X, S, B, F, P, H, K>(
    module: &Module<BE>,
    reverse: bool,
    corr: &mut C,
    x: &X,
    subtrahend: Diff<'_, S, B>,
    composite: &SignComposite<F, P>,
    tsk: &H,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend<Ring = Standard>,
    Module<BE>: CKKSSubOps<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSMulOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSPolynomialEvaluationOps<BE>,
    C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    X: GLWEToBackendRef<BE> + CKKSCtBounds,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
    B: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
    H: GetTensorKey<BE>,
    K: GetAutomorphismKey<BE>,
{
    scratch.scope(|scratch_local| {
        let (mut diff, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
        let (mut sgn, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
        match subtrahend {
            Diff::Ct(op1) => module.ckks_sub_into(&mut diff, x, op1, &mut scratch_local)?,
            Diff::Const(bound, coeff) => module.ckks_sub_pt_const_into(&mut diff, 0, x, bound, coeff, &mut scratch_local)?,
        }
        if reverse {
            module.ckks_neg_assign(&mut diff)?;
        }
        ckks_sign_into(module, false, &mut sgn, &diff, composite, tsk, keys, &mut scratch_local)?;
        module.ckks_add_one_assign(&mut sgn, &mut scratch_local)?;
        module.ckks_mul_into(corr, &sgn, &diff, tsk, &mut scratch_local)?;
        let (mut half, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*corr);
        module.ckks_div_pow2_into(1, &mut half, &*corr, &mut scratch_local)?;
        module.ckks_copy(corr, &half, &mut scratch_local)?;
        Ok(())
    })
}
