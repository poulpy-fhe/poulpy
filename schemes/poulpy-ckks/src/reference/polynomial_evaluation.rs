pub(crate) mod factored;

use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{
    BSGSMeta, GLWEInfos, GLWELayout, GLWEPlaintextInfos, GLWEPlaintextMeta, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
    SetBSGSMeta,
};
use poulpy_core::oep::{ConstTerm, GLWEMulConstImpl, GLWEMulPlainImpl, PlainTerm};
use poulpy_core::{GLWENormalize, glwe_left_operand_prepare};
use poulpy_hal::{
    api::{CnvOps, CnvPVecAlloc},
    layouts::{
        Backend, CnvPVecLOwned, CnvPVecLToBackendRef, Module, PrepareHint, Ring, ScratchArena, Standard,
        vec_znx_backend_ref_from_ref,
    },
};

use crate::{CKKSCompositionError, CKKSCtBounds, CKKSInfos, Scale, SlotLayout, api::CKKSRoundOps};
use crate::{
    SetCKKSInfos,
    api::{BSGSPolynomialInfos, CKKSAddOps, CKKSMulAddOps, CKKSMulOps, CKKSSubOps, PowerBasisHelper},
    layouts::{CKKSMaskedConstants, ScratchArenaTakeCKKSCiphertext},
    polynomial::ComplexBSGSPolynomial,
};

struct BabyPowers<BE: Backend, Rg: Ring> {
    /// The public method the evaluation serves, which its refusals carry.
    op: &'static str,
    /// The left operand of each power the shaped baby-step combinations take, by value number, prepared at its first use and
    /// reused by every baby step of the polynomial.
    powers: Vec<Option<CnvPVecLOwned<BE, Rg>>>,
}

impl<BE: Backend, Rg: Ring> BabyPowers<BE, Rg> {
    fn new(op: &'static str) -> Self {
        Self { op, powers: Vec::new() }
    }
}

/// Accumulate a scalar-coefficient block on the full layout in a single pass.
/// Returns false when the operands require the individual multiply/add path.
fn full_baby_linear_combination<BE, V, A, P, Rg: Ring>(
    op: &'static str,
    module: &Module<BE>,
    res: &mut V,
    terms: &[(&A, usize)],
    seed: &A,
    coeffs: &P,
    scratch: &mut ScratchArena<'_, BE>,
) -> bool
where
    BE: GLWEMulConstImpl<Rg>,
    V: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos,
{
    use crate::mul_pt_params_raw;
    let kb: usize = res.base2k().into();
    let delta = seed.log_delta();
    if kb == 0 {
        return false;
    }

    let mut budget = seed.log_budget();
    let mut term_params = Vec::with_capacity(terms.len());
    for (t, _) in terms {
        if usize::from(t.base2k()) != kb || t.log_delta() != delta {
            return false;
        }
        let Ok((b_i, d_i, off_i)) = mul_pt_params_raw(
            op,
            budget + delta,
            t.log_delta(),
            t.log_budget(),
            coeffs.log_delta(),
            coeffs.log_budget(),
            coeffs.encoded_k().as_usize(),
        ) else {
            return false;
        };
        if d_i != delta || off_i < kb {
            return false;
        }
        budget = budget.min(b_i);
        term_params.push((b_i, off_i));
    }
    if budget + delta > res.max_k().as_usize() {
        return false;
    }
    for (term_budget, offset) in &mut term_params {
        *offset += *term_budget - budget;
    }

    let mut sparsity = res.log_sparsity();
    let mut slots = seed.slots();
    for (t, _) in terms {
        sparsity = sparsity.min(t.log_sparsity());
        slots = slots.join(t.slots());
    }
    res.set_log_delta(delta);
    res.set_log_budget(budget);
    res.set_log_sparsity(sparsity);
    res.set_slots(slots);
    res.set_slot_layout(crate::SlotLayout::Full);
    let pt_bk = coeffs.to_backend_ref();
    let pt_size = pt_bk.data().size();
    let res_size = res.to_backend_ref().data().size();
    // Splits a term's total bit shift into the sub-limb right-shift of the
    // constant (`r`) and the whole-limb convolution offset (`rho`).
    let shift_split = |off: usize| {
        let r = (kb - off % kb) % kb;
        (r, (off + r) / kb - 1)
    };
    let mut acc_size = res_size;
    for ((t, _), (_, off)) in terms.iter().zip(term_params.iter()) {
        let (r, rho) = shift_split(*off);
        let g_size = pt_size + usize::from(r != 0);
        let needed = (t.to_backend_ref().data().size() + g_size - 1).saturating_sub(rho);
        acc_size = acc_size.max(needed);
    }

    res.set_canonical(true);
    let term_refs: Vec<_> = terms.iter().map(|(t, _)| t.to_backend_ref()).collect();
    let combination: Vec<ConstTerm<'_, BE, Rg>> = terms
        .iter()
        .zip(term_params.iter())
        .zip(term_refs.iter())
        .map(|(((_, coeff_idx), (_, off)), t_bk)| {
            let (shift, cnv_offset) = shift_split(*off);
            ConstTerm {
                a: vec_znx_backend_ref_from_ref::<BE, Rg>(t_bk.data()),
                b: vec_znx_backend_ref_from_ref::<BE, Rg>(pt_bk.data()),
                b_col: 0,
                b_coeff: *coeff_idx,
                shift,
                cnv_offset,
            }
        })
        .collect();
    let mut res_bk = res.to_backend_mut();
    BE::glwe_const_combination(module, res_bk.data_mut(), kb, budget + delta, acc_size, &combination, scratch);
    true
}

/// `res = Σ_t x_t·c_t·π_S` on a replicated layout: each product with its
/// masked constant, at the offset of the scalar combination, is added in the
/// transform domain at the shaping depth of a product under the common scale
/// ([`GLWEMulPlainImpl::glwe_plain_combination`]), and the sum is rounded once
/// with shaping. Each power is prepared as a left operand once for the
/// polynomial, in `ops`. Returns `false` without writing `res` when a term has
/// no masked constant, or does not share the radix and scales of the others.
fn shaped_baby_linear_combination<BE, V, A, P, Rg: Ring>(
    ops: &mut BabyPowers<BE, Rg>,
    module: &Module<BE>,
    res: &mut V,
    terms: &[(&A, usize)],
    coeffs: &P,
    power_index: impl Fn(usize) -> usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> anyhow::Result<bool>
where
    BE: GLWEMulPlainImpl<Rg>,
    Module<BE>: CnvOps<BE, Rg> + GLWENormalize<BE, Rg> + CKKSRoundOps<BE, Rg>,
    V: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    P: CKKSMaskedConstants<BE, Rg>,
{
    use crate::{layouts::validation::product_layout, mul_pt_params_raw, shaping::product_depth};

    let op = ops.op;
    let Some(&(seed, _)) = terms.last() else {
        return Ok(false);
    };
    let kb = res.base2k().as_usize();
    let delta = seed.log_delta();
    let operand = |i: usize| coeffs.masked_constant(i);
    let Some(first) = operand(terms[0].1) else {
        return Ok(false);
    };
    let pt_log_delta = first.log_delta();
    for &(t, i) in terms {
        let Some(pt) = operand(i) else {
            return Ok(false);
        };
        product_layout(op, &[seed.slot_layout(), t.slot_layout(), pt.slot_layout()])?;
        if t.base2k().as_usize() != kb || t.log_delta() != delta || pt.base2k().as_usize() != kb || pt.log_delta() != pt_log_delta
        {
            return Ok(false);
        }
    }
    let Some(lambda) = product_depth(seed.slot_layout(), pt_log_delta) else {
        return Ok(false);
    };

    // Each term at the rule of the scalar combination: its product's budget
    // and offset against the running smallest budget.
    let running = || {
        terms.iter().scan(seed.log_budget(), move |budget, &(t, i)| {
            let pt = operand(i)?;
            let (b, d, off) = mul_pt_params_raw(
                op,
                *budget + delta,
                t.log_delta(),
                t.log_budget(),
                pt.log_delta(),
                pt.log_budget(),
                pt.encoded_k().as_usize(),
            )
            .ok()?;
            *budget = (*budget).min(b);
            Some((t, i, pt, b, d, off))
        })
    };
    let mut budget = seed.log_budget();
    let mut count = 0;
    for (_, _, _, b, d, off) in running() {
        if d != delta || off < kb {
            return Ok(false);
        }
        budget = budget.min(b);
        count += 1;
    }
    if count != terms.len() {
        return Ok(false);
    }
    // The sub-limb part `r` of a term's offset shifts its constant, the rest
    // offsets its convolution by `rho` limbs.
    let split = |b: usize, off: usize| {
        let total = off + b - budget;
        let r = (kb - total % kb) % kb;
        (r, (total + r) / kb - 1)
    };

    let wide_k = budget + delta + lambda;
    let mut acc_size = wide_k.div_ceil(kb);
    let mut slots = seed.slots();
    let mut log_sparsity = res.log_sparsity();
    let (mut widest, mut pt_size) = (seed, 0);
    for (t, _, pt, b, _, off) in running() {
        let (r, rho) = split(b, off);
        let g_size = pt.to_backend_ref().data().size() + usize::from(r != 0);
        acc_size = acc_size.max((t.k().as_usize().div_ceil(kb) + g_size - 1).saturating_sub(rho));
        log_sparsity = log_sparsity.min(t.log_sparsity()).min(t.log_n() - pt.log_n());
        slots = slots.join(t.slots()).join(pt.slots());
        pt_size = pt_size.max(pt.to_backend_ref().data().size());
        if t.k() > widest.k() {
            widest = t;
        }
    }
    let bytes = crate::oep::derived::polynomial_evaluation::eval_shaped_baby_linear_combination_tmp_bytes::<BE, _, Rg>(
        module, widest, pt_size,
    );
    crate::layouts::validation::ensure_scratch(op, scratch, bytes)?;

    let powers = &mut ops.powers;
    for (position, &(t, _)) in terms.iter().enumerate() {
        let i = power_index(position);
        if powers.len() <= i {
            powers.resize_with(i + 1, || None);
        }
        if powers[i].is_none() {
            let mut prep =
                module.cnv_pvec_left_alloc(res.n().as_usize(), res.rank().as_usize() + 1, t.size(), PrepareHint::Reuse);
            glwe_left_operand_prepare(module, &mut prep, t, t.k().as_usize(), scratch);
            powers[i] = Some(prep);
        }
    }

    let layout = GLWELayout {
        n: res.n(),
        base2k: res.base2k(),
        k: wide_k.into(),
        rank: Rank(res.rank().as_u32()),
    };
    let meta = GLWEPlaintextMeta {
        scale: Scale::Log(delta + lambda),
        slots,
        log_sparsity,
        slot_layout: seed.slot_layout(),
    };
    let (mut wide, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&layout, meta);
    wide.set_log_budget(budget);
    {
        let pt_refs: Vec<_> = running().map(|(_, _, pt, ..)| pt.to_backend_ref()).collect();
        let combination: Vec<PlainTerm<'_, BE, Rg>> = running()
            .zip(pt_refs.iter())
            .enumerate()
            .map(|(position, ((_, _, _, b, _, off), pt_bk))| -> Result<_> {
                let (shift, cnv_offset) = split(b, off);
                Ok(PlainTerm {
                    a: powers[power_index(position)]
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("missing prepared baby-step power"))?
                        .to_backend_ref(),
                    b: vec_znx_backend_ref_from_ref::<BE, Rg>(pt_bk.data()),
                    shift,
                    cnv_offset,
                })
            })
            .collect::<Result<_>>()?;
        let mut wide_bk = wide.to_backend_mut();
        BE::glwe_plain_combination(module, wide_bk.data_mut(), kb, wide_k, acc_size, &combination, &mut scratch);
    }
    // Every column is normalized at the wide width.
    wide.set_canonical(true);
    res.set_k((budget + delta).into());
    module.ckks_round(res, &wide, &mut scratch)?;
    Ok(true)
}

/// The masked plaintext of constant `idx` when `slot_layout` is replicated,
/// whose constants are masked plaintexts.
fn masked<'p, BE: Backend, P: CKKSMaskedConstants<BE, Rg>, Rg: Ring>(
    op: &'static str,
    slot_layout: SlotLayout,
    coeffs: &'p P,
    idx: usize,
) -> Result<Option<&'p P::Masked>> {
    match slot_layout {
        SlotLayout::Full => Ok(None),
        SlotLayout::Replicated(_) => coeffs.masked_constant(idx).map(Some).ok_or_else(|| {
            CKKSCompositionError::InvalidOperand {
                op,
                reason: "a replicated layout takes its constants as masked plaintexts".into(),
            }
            .into()
        }),
    }
}

/// `res += c` for constant `c = coeffs[idx]`, as `c·π_S` on a replicated layout, refusing for the caller's `op`.
pub(crate) fn const_add_assign<BE, R, P, Rg: Ring>(
    op: &'static str,
    module: &Module<BE>,
    res: &mut R,
    coeffs: &P,
    idx: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSAddOps<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
{
    match masked(op, res.slot_layout(), coeffs, idx)? {
        Some(pt) => module.ckks_add_pt_vec_assign(res, pt, scratch),
        None => module.ckks_add_pt_const_assign(res, 0, coeffs, idx, scratch),
    }
}

/// `res -= c` for constant `c = coeffs[idx]`, as `c·π_S` on a replicated layout.
pub(crate) fn const_sub_assign<BE, R, P, Rg: Ring>(
    op: &'static str,
    module: &Module<BE>,
    res: &mut R,
    coeffs: &P,
    idx: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSSubOps<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
{
    match masked(op, res.slot_layout(), coeffs, idx)? {
        Some(pt) => module.ckks_sub_pt_vec_assign(res, pt, scratch),
        None => module.ckks_sub_pt_const_assign(res, 0, coeffs, idx, scratch),
    }
}

/// `res = a·c` for constant `c = coeffs[idx]`, as a shaped product by
/// `c·π_S` on a replicated layout.
pub(crate) fn const_mul_into<BE, R, A, P, Rg: Ring>(
    op: &'static str,
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    coeffs: &P,
    idx: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSMulOps<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
{
    match masked(op, a.slot_layout(), coeffs, idx)? {
        Some(pt) => module.ckks_mul_pt_vec_into(res, a, pt, scratch),
        None => module.ckks_mul_pt_const_into(res, a, coeffs, idx, scratch),
    }
}

/// `res += a·c` for constant `c = coeffs[idx]`, as a shaped product by
/// `c·π_S` on a replicated layout.
pub(crate) fn const_mul_add_into<BE, R, A, P, Rg: Ring>(
    op: &'static str,
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    coeffs: &P,
    idx: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSMulAddOps<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
{
    match masked(op, a.slot_layout(), coeffs, idx)? {
        Some(pt) => module.ckks_mul_add_pt_vec_into(res, a, pt, scratch),
        None => module.ckks_mul_add_pt_const_into(res, a, coeffs, idx, scratch),
    }
}

/// Fallback for real BSGS overrides through the selected factored evaluator.
pub trait CKKSPolynomialEvaluationReference<BE: Backend, Rg: Ring = Standard>: std::borrow::Borrow<Module<BE>> {
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_reference<R, B, A, G, H>(
        &self,
        res: &mut R,
        power_basis: &G,
        poly: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: crate::api::CKKSFactoredPolynomialEvaluationOps<BE, Rg>,
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<BE, Rg>,
        B::Coeffs: CKKSCtBounds + CKKSMaskedConstants<BE, Rg>,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<BE, A, Rg>,
        H: GetTensorKey<BE, Rg>,
    {
        crate::oep::derived::polynomial_evaluation::ckks_eval_poly_real_const_coeffs_from_power_basis_derived(
            self.borrow(),
            res,
            power_basis,
            poly,
            tsk,
            scratch,
        )
    }
}
impl<BE: Backend, Rg: Ring> CKKSPolynomialEvaluationReference<BE, Rg> for Module<BE> {}

/// Fallback for complex BSGS overrides through the selected factored evaluator.
pub trait CKKSComplexPolynomialEvaluationReference<BE: Backend>: std::borrow::Borrow<Module<BE>> {
    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_reference<R, C, A, G, H>(
        &self,
        res: &mut R,
        power_basis: &G,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: crate::api::CKKSFactoredPolynomialEvaluationOps<BE>,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        crate::oep::derived::polynomial_evaluation::ckks_eval_poly_complex_const_coeffs_from_power_basis_derived(
            self.borrow(),
            res,
            power_basis,
            poly,
            tsk,
            scratch,
        )
    }
}
impl<BE: Backend> CKKSComplexPolynomialEvaluationReference<BE> for Module<BE> {}
