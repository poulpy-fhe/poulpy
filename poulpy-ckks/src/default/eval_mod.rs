//! Homomorphic evaluation of the `x mod 1` reduction, the core non-linear step
//! of CKKS bootstrapping.
//!
//! This module holds the *evaluation* — the backend-generic reference
//! [`CKKSEvalModOpsDefault`] and the `eval_mod` pipeline. The parameterization
//! it consumes (the periodic-function approximation polynomials and their
//! encoding) lives in [`crate::layouts::eval_mod`]; see there for the maths and
//! the [`EvalMod`] structure. The public entry point is
//! [`CKKSEvalModOps`](crate::api::CKKSEvalModOps).

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWECopy,
    layouts::{
        BSGSMeta, Base2K, Degree, GGLWEInfos, GLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
        LWEInfos, Rank, SetBSGSMeta, TorusPrecision, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::api::CnvPVecBytesOf;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};
use std::borrow::Borrow;

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps, PolynomialInputTransform},
    layouts::{
        CKKSCiphertextOwned, CKKSModuleAlloc, ScratchArenaTakeCKKS,
        eval_mod::{EvalMod, EvalModBsgs},
    },
    power_basis::{PowerBasis, PowerBasisGen},
};

/// Backend-generic reference implementation of [`CKKSEvalModOps`], and the
/// per-method override surface for the family.
///
/// Opt-in, not blanket: a backend inherits every method by invoking
/// [`impl_ckks_eval_mod_defaults`](crate::impl_ckks_eval_mod_defaults), which
/// emits the empty impl. To substitute one kernel — a fused paired EvalMod, say
/// — the backend writes the impl itself and overrides only that method,
/// inheriting the rest; the [`CKKSEvalModImpl`](crate::oep::CKKSEvalModImpl)
/// OEP's blanket impl is keyed on this marker and forwards each hook here. A
/// backend that owns the whole family instead skips the macro and implements
/// `CKKSEvalModImpl` directly, without an overlapping-impl error.
///
/// [`CKKSEvalModOps`]: crate::api::CKKSEvalModOps
pub trait CKKSEvalModOpsDefault<BE: Backend> {
    /// Reference `x mod 1` evaluation: see [`crate::layouts::eval_mod`] for the
    /// base-polynomial / range-extension / inverse pipeline and the `eval_mod`
    /// function for the implementation.
    fn ckks_eval_mod_default<R, C, P, F>(
        &self,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: Borrow<Module<BE>>,
        Module<BE>: CKKSPolynomialEvaluationOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + CKKSCopyOps<BE>
            + CKKSModuleAlloc<BE>
            + CKKSPow2Ops<BE>
            + GLWECopy<BE>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        eval_mod(self.borrow(), res, ct, params, tsk, scratch)
    }

    /// Reference scratch budget for [`Self::ckks_eval_mod_pair_default`]: the
    /// branches run one after the other, so the larger single budget covers
    /// both. Branch shapes may differ.
    #[allow(clippy::too_many_arguments)]
    fn ckks_eval_mod_pair_tmp_bytes_default<R0, R1, C0, C1, P, F, T>(
        &self,
        res_0: &R0,
        res_1: &R1,
        ct_0: &C0,
        ct_1: &C1,
        params: &EvalMod<F, P>,
        tsk: &T,
    ) -> usize
    where
        Self: Borrow<Module<BE>>,
        Module<BE>: CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSMulOps<BE> + CKKSCopyOps<BE> + CnvPVecBytesOf,
        R0: CKKSCtBounds,
        R1: CKKSCtBounds,
        C0: CKKSCtBounds,
        C1: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        let module = self.borrow();
        ckks_eval_mod_tmp_bytes_default(module, res_0, ct_0, params, tsk)
            .max(ckks_eval_mod_tmp_bytes_default(module, res_1, ct_1, params, tsk))
    }

    /// Reference paired `x mod 1`: sequential by design, since pairing is an
    /// opportunity for a backend and never a requirement. Routed through
    /// [`Self::ckks_eval_mod_default`], so a backend overriding only the single
    /// op gets its own kernel here too.
    #[allow(clippy::too_many_arguments)]
    fn ckks_eval_mod_pair_default<R0, R1, C0, C1, P, F>(
        &self,
        res_0: &mut R0,
        res_1: &mut R1,
        ct_0: &C0,
        ct_1: &C1,
        params: &EvalMod<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: Borrow<Module<BE>>,
        Module<BE>: CKKSPolynomialEvaluationOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + CKKSCopyOps<BE>
            + CKKSModuleAlloc<BE>
            + CKKSPow2Ops<BE>
            + GLWECopy<BE>,
        R0: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C0: GLWEToBackendRef<BE> + CKKSCtBounds,
        C1: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        self.ckks_eval_mod_default(res_0, ct_0, params, tsk, scratch)?;
        self.ckks_eval_mod_default(res_1, ct_1, params, tsk, scratch)
    }
}

/// The `x mod 1` pipeline (see [`crate::layouts::eval_mod`] for the maths).
///
/// 1. Verify `ct` still has `params.eval_depth() · log_delta` bits of `log_budget` —
///    the multiplicative levels the evaluation will consume — and copy it into a
///    working ciphertext.
/// 2. Evaluate the base `f` polynomial by BSGS into `res`.
/// 3. Apply the `f_mod_log_interval_reduction` range-extension steps. For the
///    trigonometric families these are `res ← 2·res² − dac` on the real path (the
///    `cos 2θ` identity with the encoded constant `dac`) and `res ← res²` on the
///    complex path.
/// 4. If configured, compose the inverse `f⁻¹` polynomial in place.
///
/// `res` receives the result; `tsk` is the relinearization (tensor) key for the
/// squarings, and `scratch` supplies the working memory sized by
/// [`CKKSEvalModOps::ckks_eval_mod_tmp_bytes`](crate::api::CKKSEvalModOps::ckks_eval_mod_tmp_bytes).
fn eval_mod<R, C, P, F, BE>(
    module: &Module<BE>,
    res: &mut R,
    ct: &C,
    params: &EvalMod<F, P>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSPow2Ops<BE>
        + GLWECopy<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    // EvalMod runs at its own plan scale `f_mod_log_delta`: reinterpret the
    // working ciphertext to it on entry, then return the result to the input
    // scale. `consumed_bits()` accounts for the arithmetic at the plan scale;
    // if the plan scale is higher than the input scale, returning to the input
    // scale also drops that extra precision from the externally visible budget.
    // Intermediates are allocated rank-1 (`ckks_ciphertext_alloc`); reject
    // higher-rank inputs instead of silently mis-shaping them.
    ckks_ensure!(
        ct.rank().as_usize() == 1,
        "ckks_eval_mod supports rank-1 ciphertexts only, got rank {}",
        ct.rank().as_usize()
    );
    let s_in = ct.log_delta();
    let s_eval = params.plan.f_mod_log_delta;
    let s_budget = ct.log_budget();

    let required = params.consumed_bits();
    ckks_ensure!(
        ct.log_budget() >= required,
        "ckks_eval_mod: input log_budget {got} < {required} bits required (consumed at scale {s_eval})",
        got = ct.log_budget(),
    );

    let work_layout = GLWELayout {
        n: ct.n(),
        base2k: ct.base2k(),
        k: (s_budget + s_eval).into(),
        rank: Rank(1),
    };
    let work_meta = CKKSMeta {
        log_delta: s_eval,
        log_sparsity: ct.log_sparsity(),
        slots: ct.slots(),
    };

    match &params.f_mod_bsgs {
        EvalModBsgs::Real(bsgs) => {
            if bsgs.input_transform() == PolynomialInputTransform::Identity {
                // The generic one-shot evaluator would copy this input again
                // before building its power basis. Hand ownership over directly.
                let x1 = eval_mod_input(module, ct, &work_layout, work_meta);
                let mut power_basis = PowerBasis::new(bsgs.basis(), x1);
                power_basis.populate(bsgs.degree(), bsgs.log_split(), bsgs.parity(), module, tsk, scratch)?;
                module.ckks_eval_poly_real_const_coeffs_from_power_basis(res, bsgs, &power_basis, tsk, scratch)?;
            } else {
                scratch.scope(|scratch_local| {
                    let (mut input, mut nested) = scratch_local.take_ckks_ciphertext_scratch(&work_layout, work_meta);
                    module.glwe_copy(&mut input, ct);
                    module.ckks_eval_poly_real_const_coeffs(res, &input, bsgs, tsk, &mut nested)
                })?;
            }

            if let Some(consts) = params.range_extension_consts.as_ref() {
                for i in 0..params.plan.f_mod_log_interval_reduction {
                    module.ckks_square_assign(res, tsk, scratch)?;
                    module.ckks_mul_pow2_assign(res, 1, scratch)?;
                    module.ckks_sub_pt_const_assign(res, 0, consts, i, scratch)?;
                }
            }

            if let Some(inv) = params.f_mod_inv_bsgs.as_ref() {
                // The inverse consumes the base result, so this is the only
                // stage that still needs a separate working copy.
                scratch.scope(|scratch_local| {
                    let (mut input, mut nested) = scratch_local.take_ckks_ciphertext_scratch(&work_layout, work_meta);
                    module.ckks_copy(&mut input, &*res, &mut nested)?;
                    module.ckks_eval_poly_real_const_coeffs(res, &input, inv, tsk, &mut nested)
                })?;
            }
        }
        EvalModBsgs::Complex(bsgs) => {
            if bsgs.re.input_transform() == PolynomialInputTransform::Identity
                && bsgs.im.input_transform() == PolynomialInputTransform::Identity
            {
                let x1 = eval_mod_input(module, ct, &work_layout, work_meta);
                let mut power_basis = PowerBasis::new(bsgs.re.basis(), x1);
                power_basis.populate(bsgs.re.degree(), bsgs.re.log_split(), bsgs.re.parity(), module, tsk, scratch)?;
                module.ckks_eval_poly_complex_const_coeffs_from_power_basis(res, bsgs, &power_basis, tsk, scratch)?;
            } else {
                scratch.scope(|scratch_local| {
                    let (mut input, mut nested) = scratch_local.take_ckks_ciphertext_scratch(&work_layout, work_meta);
                    module.glwe_copy(&mut input, ct);
                    module.ckks_eval_poly_complex_const_coeffs(res, &input, bsgs, tsk, &mut nested)
                })?;
            }
            for _ in 0..params.plan.f_mod_log_interval_reduction {
                module.ckks_square_assign(res, tsk, scratch)?;
            }
        }
    }

    // Restore the input scale on the result. This is a pure metadata relabel
    // (`set_log_delta`), not a rescale: entry raised the scale `s_in -> s_eval`
    // without spending budget (`set_log_budget(s_budget)` on an MSB-aligned copy,
    // which reinterprets the value at `2^-(s_eval - s_in)`); relabelling back to
    // `s_in` here undoes exactly that, so the scale round-trip is budget-neutral
    // and the only consumption is the EvalMod arithmetic, which `consumed_bits()`
    // accounts for in full.
    if s_eval != s_in {
        res.set_log_delta(s_in);
    }

    Ok(())
}

/// Allocates the single owned EvalMod input that becomes power-basis element 1.
/// Copying directly from `ct` avoids the scratch copy followed by the generic
/// polynomial evaluator's second owned copy.
fn eval_mod_input<BE, C>(module: &Module<BE>, ct: &C, layout: &GLWELayout, meta: CKKSMeta) -> CKKSCiphertextOwned<BE>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + GLWECopy<BE>,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let mut input = module.ckks_ciphertext_alloc(layout.base2k, layout.k);
    module.glwe_copy(&mut input, ct);
    input.set_meta(meta);
    input
}

#[derive(Clone, Copy)]
struct EvalModWorkCtInfos {
    n: Degree,
    base2k: Base2K,
    rank: Rank,
    max_size: usize,
    k: TorusPrecision,
    meta: CKKSMeta,
}

impl LWEInfos for EvalModWorkCtInfos {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn max_size(&self) -> usize {
        self.max_size
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl GLWEInfos for EvalModWorkCtInfos {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl CKKSInfos for EvalModWorkCtInfos {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }
}

/// Reference scratch budget for [`CKKSEvalModOps::ckks_eval_mod`].
///
/// Kept here rather than in the delegate so the paired operation's default can
/// reuse it without routing back through the public trait.
///
/// [`CKKSEvalModOps::ckks_eval_mod`]: crate::api::CKKSEvalModOps::ckks_eval_mod
pub fn ckks_eval_mod_tmp_bytes_default<BE, R, C, P, F, T>(
    module: &Module<BE>,
    res: &R,
    ct: &C,
    params: &EvalMod<F, P>,
    tsk: &T,
) -> usize
where
    BE: Backend,
    Module<BE>: CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSMulOps<BE> + CKKSCopyOps<BE> + CnvPVecBytesOf,
    R: CKKSCtBounds,
    C: CKKSCtBounds,
    P: CKKSCtBounds,
    T: GGLWEInfos,
{
    let work_k = ct.k().as_usize().max(ct.log_budget() + params.plan.f_mod_log_delta);
    let work = EvalModWorkCtInfos {
        n: ct.n(),
        base2k: ct.base2k(),
        rank: ct.rank(),
        max_size: work_k.div_ceil(ct.base2k().as_usize()).max(1),
        // Total torus width = budget carried into eval_mod + the plan scale.
        k: (ct.log_budget() + params.plan.f_mod_log_delta).into(),
        meta: CKKSMeta {
            log_sparsity: ct.log_sparsity(),
            log_delta: params.plan.f_mod_log_delta,
            slots: SlotsKind::Complex,
        },
    };

    let cols: usize = (work.rank() + 1).into();
    let compact_work = BE::bytes_of_vec_znx(work.n().into(), cols, work.max_size());
    // The giant step hoists the prepared `X^{gsp}` right operand, kept alive
    // across the baby-step pairs that share it.
    let hoisted_right = module.bytes_of_cnv_pvec_right(cols, work.max_size());
    let bsgs_giant = module
        .ckks_mul_tmp_bytes(&work, &work, &work, tsk)
        .max(module.ckks_add_tmp_bytes())
        + 3 * compact_work
        + hoisted_right;
    let square_scope = (module.ckks_square_tmp_bytes(&work, &work, tsk) + compact_work).max(
        // Scratch is a physical working-set budget: size the square-scope
        // copy off `res`'s allocated capacity, the upper bound on the limbs
        // any runtime re-expansion can expose.
        module.ckks_square_tmp_bytes(res, res, tsk)
            + BE::bytes_of_vec_znx(res.n().into(), (res.rank() + 1).into(), res.max_size()),
    );
    // Identity base polynomials transfer an owned, relabelled input directly
    // into the power basis. Scratch only needs a full working ciphertext for
    // a transformed base or for the optional inverse composition.
    let needs_work_copy = params.f_mod_inv_bsgs.is_some()
        || match &params.f_mod_bsgs {
            EvalModBsgs::Real(poly) => poly.input_transform() != PolynomialInputTransform::Identity,
            EvalModBsgs::Complex(poly) => {
                poly.re.input_transform() != PolynomialInputTransform::Identity
                    || poly.im.input_transform() != PolynomialInputTransform::Identity
            }
        };
    usize::from(needs_work_copy) * compact_work
        + module
            .ckks_copy_tmp_bytes()
            .max(module.ckks_add_pt_const_tmp_bytes())
            .max(module.ckks_sub_pt_const_tmp_bytes())
            .max(bsgs_giant)
            .max(square_scope)
}
