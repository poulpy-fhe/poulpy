//! Composite-minimax sign and smooth predicates.

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
    api::{CKKSAddOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSCopyOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops},
    layouts::ScratchArenaTakeCKKS,
};

mod compare;
mod composite;
mod predicate;

pub use compare::CKKSComparisonOps;
pub use composite::{COEFFS_SIGN_X2_CHEBYSHEV, COEFFS_SIGN_X4_CHEBYSHEV, DEFAULT_SIGN_COMPOSITE_CHEBYSHEV, SignComposite};
pub use predicate::CKKSPredicateOps;

/// Evaluates and real-cleans a sign composite on scratch copies of `input`, then `(s + 1) / 2` when `step` is set, and
/// copies the result to `res` with its last write.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ckks_sign_into<BE, R, I, F, P, H, K>(
    module: &Module<BE>,
    res: &mut R,
    input: &I,
    composite: &SignComposite<F, P>,
    tsk: &H,
    conj_key: &K,
    step: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend<Ring = Standard>,
    Module<BE>: CKKSPolynomialEvaluationOps<BE> + CKKSCopyOps<BE> + CKKSConjugateOps<BE> + CKKSAddOps<BE> + CKKSPow2Ops<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    I: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
    H: GetTensorKey<BE>,
    K: GetAutomorphismKey<BE>,
{
    let op = if step { "ckks_step" } else { "ckks_sign" };

    let log_delta = input.log_delta();
    // The step halves the sign, which costs one more bit.
    let required = composite.consumed_bits(log_delta) + usize::from(step);
    ensure!(
        input.log_budget() > required,
        "{op}: log_budget {} <= {required} bits required at log_delta {log_delta}",
        input.log_budget(),
    );

    scratch.scope(|scratch_local| {
        let (mut cur, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
        let (mut nxt, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
        let (mut conj, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
        module.ckks_copy(&mut cur, input, &mut scratch_local)?;

        // Each factor evaluates at half scale: `nxt += conj(nxt)` cleans the imaginary part and restores full scale,
        // keeping `log_delta` constant.
        for poly in &composite.polys_bsgs {
            module.ckks_eval_poly_real_const_coeffs(&mut nxt, &cur, poly, tsk, &mut scratch_local)?;
            module.ckks_conjugate_into(&mut conj, &nxt, conj_key, &mut scratch_local)?;
            module.ckks_add_assign(&mut nxt, &conj, &mut scratch_local)?;
            std::mem::swap(&mut cur, &mut nxt);
        }

        if step {
            module.ckks_add_one_assign(&mut cur, &mut scratch_local)?;
            module.ckks_div_pow2_into(1, &mut nxt, &cur, &mut scratch_local)?;
            std::mem::swap(&mut cur, &mut nxt);
        }

        module.ckks_copy(res, &cur, &mut scratch_local)
    })?;
    Ok(())
}

/// Query of [`ckks_sign_into`]: three temporaries with the layout of `input`.
pub(crate) fn ckks_sign_tmp_bytes<BE, R, I, T, A, P>(
    module: &Module<BE>,
    res: &R,
    input: &I,
    tsk: &T,
    conj_key: &A,
    coeff_prec: &P,
) -> usize
where
    BE: Backend,
    Module<BE>: CKKSAllOpsTmpBytes<BE> + CKKSCopyOps<BE> + GLWEBytesOf<BE>,
    R: CKKSCtBounds,
    I: CKKSCtBounds,
    T: GGLWEInfos,
    A: GGLWEInfos,
    P: CKKSInfos + LWEInfos,
{
    3 * BE::scratch_aligned(module.glwe_bytes_of_from_infos(input))
        + module
            .ckks_all_ops_with_atk_tmp_bytes(input, coeff_prec, tsk, conj_key)
            .max(module.ckks_copy_tmp_bytes(res, input))
}

/// Homomorphic `sign` and `step`.
pub trait CKKSSignOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_sign`] and [`Self::ckks_step`].
    fn ckks_sign_tmp_bytes<R, I, T, A, P>(&self, res: &R, input: &I, tsk: &T, conj_key: &A, coeff_prec: &P) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// Evaluates `sign(input)` on `[-1, 1]` outside the composite gap.
    ///
    /// `conj_key` answers the conjugation, Galois element `-1`.
    ///
    /// # Metadata
    ///
    /// The factors run at the layout of `input`, each keeping `log_delta` and consuming its share of
    /// `composite.consumed_bits(input.log_delta())`, then `ckks_copy` writes the result to `res`.
    ///
    /// # Errors
    ///
    /// An error when `input` holds no more budget than the composite consumes, and the errors of the factors and of the
    /// final `ckks_copy`. The factors run in scratch, so `res` is untouched on `Err`.
    fn ckks_sign<R, I, F, P, H, K>(
        &self,
        res: &mut R,
        input: &I,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// Evaluates `(sign(input) + 1) / 2`.
    ///
    /// # Metadata
    ///
    /// As [`Self::ckks_sign`], then the halving adds one bit to `log_delta` at the same width.
    ///
    /// # Errors
    ///
    /// As [`Self::ckks_sign`], the halving requiring one more bit of budget. `res` is untouched on `Err`.
    fn ckks_step<R, I, F, P, H, K>(
        &self,
        res: &mut R,
        input: &I,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;
}

impl<BE: Backend<Ring = Standard>> CKKSSignOps<BE> for Module<BE>
where
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSAllOpsTmpBytes<BE>
        + GLWEBytesOf<BE>,
{
    fn ckks_sign_tmp_bytes<R, I, T, A, P>(&self, res: &R, input: &I, tsk: &T, conj_key: &A, coeff_prec: &P) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        ckks_sign_tmp_bytes(self, res, input, tsk, conj_key, coeff_prec)
    }

    fn ckks_sign<R, I, F, P, H, K>(
        &self,
        res: &mut R,
        input: &I,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        ckks_sign_into(self, res, input, composite, tsk, conj_key, false, scratch)
    }

    fn ckks_step<R, I, F, P, H, K>(
        &self,
        res: &mut R,
        input: &I,
        composite: &SignComposite<F, P>,
        tsk: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        ckks_sign_into(self, res, input, composite, tsk, conj_key, true, scratch)
    }
}
