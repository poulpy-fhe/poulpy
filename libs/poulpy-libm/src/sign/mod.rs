//! Composite-minimax sign and smooth predicates.

use anyhow::{Result, ensure};
use poulpy_ckks::layouts::CKKSMaskedConstants;
use poulpy_core::{
    GLWEBytesOf,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, IntPolyInfos, LWEInfos,
    },
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Module, Ring, ScratchArena, Standard};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{CKKSAddOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSCopyOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops},
    layouts::ScratchArenaTakeCKKSCiphertext,
};

mod compare;
mod composite;
mod predicate;

pub use compare::CKKSComparisonOps;
pub use composite::{COEFFS_SIGN_X2_CHEBYSHEV, COEFFS_SIGN_X4_CHEBYSHEV, DEFAULT_SIGN_COMPOSITE_CHEBYSHEV, SignComposite};
pub use predicate::CKKSPredicateOps;

/// A ring the composite sign runs on.
///
/// The standard ring's slots are complex, so each factor is encoded at half scale and its output cleaned to
/// `x + conj(x)`, which drops the imaginary part and restores full scale. The conjugate-invariant ring's slots are
/// real: its factors are encoded at full scale and need no cleaning.
pub trait SignRing: Ring {
    /// Whether each factor is encoded at half scale and cleaned with a conjugation.
    const CONJUGATE_CLEANING: bool;
}

impl SignRing for Standard {
    const CONJUGATE_CLEANING: bool = true;
}

impl SignRing for ConjugateInvariant {
    const CONJUGATE_CLEANING: bool = false;
}

/// The cleaning step of [`SignRing`] on a backend.
pub trait SignRingOps<BE: Backend>: SignRing {
    /// Replaces `ct` by `ct + conj(ct)` through the temporary `tmp` when [`SignRing::CONJUGATE_CLEANING`] is set, and
    /// leaves it unchanged otherwise.
    fn ckks_sign_clean<C, T, K>(
        module: &Module<BE>,
        ct: &mut C,
        tmp: &mut T,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        C: GLWEToBackendMut<BE, Self> + CKKSCtBounds + SetCKKSInfos,
        T: GLWEToBackendMut<BE, Self> + CKKSCtBounds + SetCKKSInfos,
        K: GetAutomorphismKey<BE, Self>;
}

impl<BE: Backend> SignRingOps<BE> for Standard
where
    Module<BE>: CKKSConjugateOps<BE> + CKKSAddOps<BE>,
{
    fn ckks_sign_clean<C, T, K>(
        module: &Module<BE>,
        ct: &mut C,
        tmp: &mut T,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        C: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        T: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GetAutomorphismKey<BE>,
    {
        module.ckks_conjugate_into(tmp, ct, keys, scratch)?;
        module.ckks_add_assign(ct, tmp, scratch)?;
        Ok(())
    }
}

impl<BE: Backend> SignRingOps<BE> for ConjugateInvariant {
    fn ckks_sign_clean<C, T, K>(
        _module: &Module<BE>,
        _ct: &mut C,
        _tmp: &mut T,
        _keys: &K,
        _scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        C: GLWEToBackendMut<BE, Self> + CKKSCtBounds + SetCKKSInfos,
        T: GLWEToBackendMut<BE, Self> + CKKSCtBounds + SetCKKSInfos,
        K: GetAutomorphismKey<BE, Self>,
    {
        Ok(())
    }
}

/// Evaluates and cleans a sign composite on scratch copies of `input`, then `(s + 1) / 2` when `step` is set, and
/// copies the result to `res` with its last write.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ckks_sign_into<BE, Rg, R, I, F, P, H, K>(
    module: &Module<BE>,
    step: bool,
    res: &mut R,
    input: &I,
    composite: &SignComposite<F, P>,
    tsk: &H,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Rg: SignRingOps<BE>,
    Module<BE>: CKKSPolynomialEvaluationOps<BE, Rg> + CKKSCopyOps<BE, Rg> + CKKSAddOps<BE, Rg> + CKKSPow2Ops<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    I: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + BSGSMeta + CKKSMaskedConstants<BE, Rg>,
    H: GetTensorKey<BE, Rg>,
    K: GetAutomorphismKey<BE, Rg>,
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
        let (mut cur, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch::<_, Rg>(input);
        let (mut nxt, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch::<_, Rg>(input);
        module.ckks_copy(&mut cur, input, &mut scratch_local)?;

        // On the standard ring each factor evaluates at half scale: `nxt += conj(nxt)` cleans the imaginary part and
        // restores full scale, keeping `log_delta` constant. The conjugate-invariant ring evaluates at full scale.
        for poly in &composite.polys_bsgs {
            module.ckks_eval_poly_real_const_coeffs(&mut nxt, &cur, poly, tsk, &mut scratch_local)?;
            if Rg::CONJUGATE_CLEANING {
                let (mut conj, mut scratch_clean) = scratch_local.borrow().take_ckks_ciphertext_like_scratch::<_, Rg>(input);
                Rg::ckks_sign_clean(module, &mut nxt, &mut conj, keys, &mut scratch_clean)?;
            }
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

/// Query of [`ckks_sign_into`]: two temporaries with the layout of `input`, and a third for the cleaning when
/// [`SignRing::CONJUGATE_CLEANING`] is set.
pub(crate) fn ckks_sign_tmp_bytes<Rg, BE>(
    module: &Module<BE>,
    res: &impl CKKSCtBounds,
    input: &impl CKKSCtBounds,
    coeff_prec: &impl CKKSInfos,
    tsk: &impl GGLWEInfos,
    keys: &impl GGLWEInfos,
) -> usize
where
    Rg: SignRing,
    BE: Backend,
    Module<BE>: CKKSAllOpsTmpBytes<BE> + CKKSCopyOps<BE, Rg> + GLWEBytesOf<BE>,
{
    (2 + usize::from(Rg::CONJUGATE_CLEANING)) * BE::scratch_aligned(module.glwe_bytes_of_from_infos(input))
        + module
            .ckks_all_ops_with_atk_tmp_bytes(input, coeff_prec, tsk, keys)
            .max(<Module<BE> as CKKSCopyOps<BE, Rg>>::ckks_copy_tmp_bytes(module, res, input))
}

/// Homomorphic `sign` and `step` on the ring `Rg`.
pub trait CKKSSignOps<BE: Backend, Rg: Ring = Standard> {
    /// Scratch bytes for [`Self::ckks_sign`] and [`Self::ckks_step`].
    fn ckks_sign_tmp_bytes<R, I, P, T, A>(&self, res: &R, input: &I, coeff_prec: &P, tsk: &T, keys: &A) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        P: CKKSInfos + LWEInfos,
        T: GGLWEInfos,
        A: GGLWEInfos;

    /// Evaluates `sign(input)` on `[-1, 1]` outside the composite gap.
    ///
    /// On the standard ring `keys` answers the conjugation, Galois element `-1`. The conjugate-invariant ring looks up
    /// no key.
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
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + BSGSMeta + CKKSMaskedConstants<BE, Rg>,
        H: GetTensorKey<BE, Rg>,
        K: GetAutomorphismKey<BE, Rg>;

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
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + BSGSMeta + CKKSMaskedConstants<BE, Rg>,
        H: GetTensorKey<BE, Rg>,
        K: GetAutomorphismKey<BE, Rg>;
}

impl<BE: Backend, Rg: SignRingOps<BE>> CKKSSignOps<BE, Rg> for Module<BE>
where
    Module<BE>: CKKSPolynomialEvaluationOps<BE, Rg>
        + CKKSCopyOps<BE, Rg>
        + CKKSAddOps<BE, Rg>
        + CKKSPow2Ops<BE, Rg>
        + CKKSAllOpsTmpBytes<BE>
        + GLWEBytesOf<BE>,
{
    fn ckks_sign_tmp_bytes<R, I, P, T, A>(&self, res: &R, input: &I, coeff_prec: &P, tsk: &T, keys: &A) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        P: CKKSInfos + LWEInfos,
        T: GGLWEInfos,
        A: GGLWEInfos,
    {
        ckks_sign_tmp_bytes::<Rg, BE>(self, res, input, coeff_prec, tsk, keys)
    }

    fn ckks_sign<R, I, F, P, H, K>(
        &self,
        res: &mut R,
        input: &I,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + BSGSMeta + CKKSMaskedConstants<BE, Rg>,
        H: GetTensorKey<BE, Rg>,
        K: GetAutomorphismKey<BE, Rg>,
    {
        ckks_sign_into::<BE, Rg, _, _, _, _, _, _>(self, false, res, input, composite, tsk, keys, scratch)
    }

    fn ckks_step<R, I, F, P, H, K>(
        &self,
        res: &mut R,
        input: &I,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + BSGSMeta + CKKSMaskedConstants<BE, Rg>,
        H: GetTensorKey<BE, Rg>,
        K: GetAutomorphismKey<BE, Rg>,
    {
        ckks_sign_into::<BE, Rg, _, _, _, _, _, _>(self, true, res, input, composite, tsk, keys, scratch)
    }
}
