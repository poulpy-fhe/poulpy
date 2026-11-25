use crate::{CKKSResult as Result, ckks_ensure_operand};
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{
    BSGSMeta, GLWEInfos, GLWELayout, GLWEPlaintextInfos, GLWEPlaintextLayout, GLWEPlaintextMeta, GLWEToBackendMut,
    GLWEToBackendRef, LWEInfos, Rank, SetBSGSMeta,
};
use poulpy_core::oep::GLWEMulPlainImpl;
use poulpy_core::{
    GLWEBSGSOps, GLWELeftOperandRef, GLWEMulPlain, GLWENormalize, GLWEPolynomialEvaluation, GLWEZero, glwe_left_operand_prepare,
};
use poulpy_hal::{
    api::{CnvOps, CnvPVecAlloc, ModuleN, VecZnxBigNormalize, VecZnxRsh, VecZnxRshTmpBytes},
    layouts::{Backend, CnvPVecLOwned, Data, Module, PrepareHint, Ring, ScratchArena, Standard, ZnxWord},
};
use std::cell::RefCell;

use crate::{
    CKKSCtBounds, Scale, SlotLayout, api::CKKSRoundOps, ensure_plaintext_degree_embeds,
    layouts::validation::ensure_masked_constants,
};
use crate::{
    SetCKKSInfos,
    api::{
        BSGSPolynomialInfos, BabyStep as BabyStepInfos, Basis, CKKSAddOps, CKKSCopyOps, CKKSImagOps, CKKSMulAddOps, CKKSMulOps,
        CKKSRightOperandPreparedFactory, PowerBasisHelper,
    },
    layouts::{CKKSCiphertext, CKKSCiphertextAlloc, CKKSRightOperandPrepared, ScratchArenaTakeCKKSCiphertext},
    polynomial::ComplexBSGSPolynomial,
};

struct EvaluatedBabyStep<D: Data, W: ZnxWord, Rg: Ring> {
    degree: usize,
    value: CKKSCiphertext<D, W, Rg>,
}

impl<BE, D, Rg: Ring> BabyStepInfos<BE, Rg> for EvaluatedBabyStep<D, BE::ZnxWord, Rg>
where
    BE: Backend,
    D: Data,
    CKKSCiphertext<D, BE::ZnxWord, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg>,
{
    type Value = CKKSCiphertext<D, BE::ZnxWord, Rg>;

    fn degree(&self) -> usize {
        self.degree
    }

    fn get(&self) -> &Self::Value {
        &self.value
    }

    fn get_mut(&mut self) -> &mut Self::Value {
        &mut self.value
    }
}

/// CKKS scale-aware operations for the scale-agnostic core BSGS engine, for one polynomial evaluation.
///
/// Every method is a thin dispatch to the CKKS API ([`CKKSMulOps`], [`CKKSAddOps`],
/// [`CKKSCopyOps`]); the non-dispatch bits are the accumulator seed, which
/// has no single existing API equivalent, and the fused baby-step linear
/// combinations.
struct CKKSBSGSOps<'a, BE: Backend + 'a, Rg: Ring> {
    /// The public method the evaluation serves, which its refusals carry.
    op: &'static str,
    /// The left operand of each power the shaped baby-step combinations take, by degree, prepared at its first use and
    /// reused by every baby step of the polynomial.
    powers: RefCell<Vec<Option<CnvPVecLOwned<BE, Rg>>>>,
    /// The left operands, plaintexts and offsets of the baby-step combinations, allocated once for the polynomial and
    /// refilled, the operand lists as lists of references of any type.
    left: RefCell<Vec<&'a ()>>,
    plain: RefCell<Vec<&'a ()>>,
    offsets: RefCell<Vec<usize>>,
}

impl<'a, BE: Backend + 'a, Rg: Ring> CKKSBSGSOps<'a, BE, Rg> {
    fn new(op: &'static str) -> Self {
        Self {
            op,
            powers: RefCell::new(Vec::new()),
            left: RefCell::new(Vec::new()),
            plain: RefCell::new(Vec::new()),
            offsets: RefCell::new(Vec::new()),
        }
    }

    /// `res = Σ_t l_t · c_t` over the `(l_t, c_t, offset_t)` of `terms`, through the lists kept for the polynomial.
    fn dot_product<'t, R, L, P>(
        &self,
        module: &Module<BE>,
        res: &mut R,
        terms: impl Iterator<Item = (&'t L, &'t P, usize)>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: GLWEMulPlainImpl<Rg>,
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        L: GLWELeftOperandRef<BE, Rg> + 't,
        P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + GLWEInfos + 't,
    {
        let (mut left, mut plain): (Vec<&L>, Vec<&P>) = (recycled(self.left.take()), recycled(self.plain.take()));
        let mut offsets = self.offsets.take();
        offsets.clear();
        for (l, p, offset) in terms {
            left.push(l);
            plain.push(p);
            offsets.push(offset);
        }
        module.glwe_mul_plain_sum(res, &left, &plain, &offsets, scratch);
        self.left.replace(recycled(left));
        self.plain.replace(recycled(plain));
        self.offsets.replace(offsets);
    }
}

/// `list` emptied, with its allocation, as a list of another element type: the term lists of a baby step borrow its
/// operands, and keep one allocation across baby steps.
fn recycled<T, U>(mut list: Vec<T>) -> Vec<U> {
    list.clear();
    list.into_iter().map(|_| unreachable!("the list is empty")).collect()
}

/// The terms of a baby step at the rule of their products by their constants, against the running smallest budget
/// from that of the last term: `(term, index, budget, scale, offset)`, ending at a product the rule refuses.
fn baby_terms<'t, A, P>(
    op: &'static str,
    terms: &'t [(&'t A, usize)],
    coeffs: &'t [P],
    delta: usize,
) -> impl Iterator<Item = (&'t A, usize, usize, usize, usize)> + 't
where
    A: CKKSCtBounds,
    P: CKKSCtBounds + IntPolyInfos,
{
    let budget = terms.last().map_or(0, |(t, _)| t.log_budget());
    terms.iter().scan(budget, move |budget, &(t, i)| {
        let pt = &coeffs[i];
        let (b, d, off) = crate::mul_pt_params_raw(
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
        Some((t, i, b, d, off))
    })
}

/// The budget a combination of `terms` lands at, the smallest of their products', or `None` when a product is
/// refused, lands at another scale than `delta`, or at an offset below a limb of `2^kb`.
fn baby_budget<A, P>(op: &'static str, terms: &[(&A, usize)], coeffs: &[P], delta: usize, kb: usize) -> Option<usize>
where
    A: CKKSCtBounds,
    P: CKKSCtBounds + IntPolyInfos,
{
    let mut budget = terms.last()?.0.log_budget();
    let mut count = 0;
    for (_, _, b, d, off) in baby_terms(op, terms, coeffs, delta) {
        if d != delta || off < kb {
            return None;
        }
        budget = budget.min(b);
        count += 1;
    }
    (count == terms.len()).then_some(budget)
}

impl<'a, BE: Backend + 'a, V, P, A, R, Rg: Ring> GLWEBSGSOps<BE, V, P, A, R, Rg> for CKKSBSGSOps<'a, BE, Rg>
where
    BE: GLWEMulPlainImpl<Rg>,
    Module<BE>: CKKSAddOps<BE, Rg>
        + CKKSCiphertextAlloc<BE>
        + CKKSMulOps<BE, Rg>
        + CKKSRightOperandPreparedFactory<BE, Rg>
        + CKKSMulAddOps<BE, Rg>
        + CKKSCopyOps<BE, Rg>
        + GLWEZero<BE, Rg>
        + CnvOps<BE, Rg>
        + GLWENormalize<BE, Rg>
        + CKKSRoundOps<BE, Rg>,
    V: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    P: CKKSCtBounds + IntPolyInfos + GLWEToBackendRef<BE, Rg>,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + BSGSMeta,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
{
    type Prepared = CKKSRightOperandPrepared<BE, Rg>;

    fn init_accumulator(
        &self,
        module: &Module<BE>,
        res: &mut V,
        seed: &A,
        _scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        res.set_log_delta(seed.log_delta());
        res.set_log_budget(seed.log_budget());
        res.set_slots(seed.slots());
        // The seed's layout, so a reused destination does not keep another.
        res.set_slot_layout(seed.slot_layout());
        module.glwe_zero(res);
        Ok(())
    }

    fn eval_baby_linear_combination(
        &self,
        module: &Module<BE>,
        res: &mut V,
        terms: &[(&A, usize)],
        coeffs: &[P],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<bool> {
        let Some(&(seed, _)) = terms.last() else {
            return Ok(false);
        };
        // `π_S = 1` on the full layout, which the label proves: the constants multiply as integers.
        if let SlotLayout::Replicated(_) = coeffs[0].slot_layout() {
            return shaped_baby_linear_combination(self, module, res, terms, coeffs, scratch);
        }
        let kb: usize = res.base2k().into();
        let delta = seed.log_delta();
        // One dot product takes coefficients of the terms' radix and of any degree dividing theirs, constants and
        // polynomials mixed; any other step takes the composed path, which refuses with a typed error.
        let fits = |t: &A, c: &P| {
            let (t_n, c_n) = (t.n().as_usize(), c.n().as_usize());
            usize::from(t.base2k()) == kb
                && t.log_delta() == delta
                && usize::from(c.base2k()) == kb
                && c_n.is_power_of_two()
                && t_n.is_multiple_of(c_n)
        };
        if kb == 0 || terms.iter().any(|&(t, i)| !fits(t, &coeffs[i])) {
            return Ok(false);
        }
        let Some(budget) = baby_budget(self.op, terms, coeffs, delta, kb) else {
            return Ok(false);
        };
        let running = || baby_terms(self.op, terms, coeffs, delta);
        // The sum's query at the term of largest `k` and the coefficients' largest degree and limbs.
        let widest = terms.iter().map(|&(t, _)| t).max_by_key(|t| t.k()).expect("checked above");
        let pt_n = terms
            .iter()
            .map(|&(_, i)| coeffs[i].n().as_usize())
            .max()
            .expect("checked above");
        let pt_size = terms
            .iter()
            .map(|&(_, i)| coeffs[i].encoded_k().as_usize().div_ceil(kb))
            .max()
            .expect("checked above");
        let pt = GLWEPlaintextLayout {
            n: pt_n.into(),
            base2k: res.base2k(),
            k: (pt_size * kb).into(),
        };
        crate::layouts::validation::ensure_scratch(self.op, scratch, module.glwe_mul_plain_sum_tmp_bytes(res, widest, &pt))?;

        let mut sparsity = res.log_sparsity();
        let mut slots = seed.slots();
        for &(t, i) in terms {
            sparsity = sparsity.min(t.log_sparsity()).min(t.log_n() - coeffs[i].log_n());
            slots = slots.join(t.slots()).join(coeffs[i].slots());
        }
        res.set_log_delta(delta);
        res.set_log_budget(budget);
        res.set_log_sparsity(sparsity);
        res.set_slots(slots);
        res.set_slot_layout(crate::SlotLayout::Full);
        let terms = running().map(|(t, i, b, _, off)| (t, &coeffs[i], off + b - budget));
        self.dot_product(module, res, terms, scratch);
        Ok(true)
    }

    fn add_pt_assign(&self, module: &Module<BE>, res: &mut V, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> anyhow::Result<()> {
        module.ckks_add_assign(res, pt, scratch).map_err(::anyhow::Error::from)
    }

    fn mul_pt(&self, module: &Module<BE>, res: &mut V, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> anyhow::Result<()> {
        module.ckks_mul_pt_into(res, a, pt, scratch).map_err(::anyhow::Error::from)
    }

    fn mul_add_pt(
        &self,
        module: &Module<BE>,
        res: &mut V,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        module
            .ckks_mul_add_pt_into(res, a, pt, scratch)
            .map_err(::anyhow::Error::from)
    }

    fn prepare_right(&self, module: &Module<BE>, a: &A, scratch: &mut ScratchArena<'_, BE>) -> anyhow::Result<Self::Prepared> {
        // `X^{gsp}` multiplies every pair of its giant-step level.
        module
            .ckks_right_operand_prepare_alloc(a, PrepareHint::Reuse, scratch)
            .map_err(::anyhow::Error::from)
    }

    fn mul_prepared_assign<H>(
        &self,
        module: &Module<BE>,
        dst: &mut V,
        prepared: &Self::Prepared,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()>
    where
        H: GetTensorKey<BE, Rg>,
    {
        module
            .ckks_mul_prepared_assign(dst, prepared, tsk, scratch)
            .map_err(::anyhow::Error::from)
    }

    fn add_assign(&self, module: &Module<BE>, dst: &mut V, a: &V, scratch: &mut ScratchArena<'_, BE>) -> anyhow::Result<()> {
        module.ckks_add_assign(dst, a, scratch).map_err(::anyhow::Error::from)
    }

    fn copy(&self, module: &Module<BE>, res: &mut R, src: &V, scratch: &mut ScratchArena<'_, BE>) -> anyhow::Result<()> {
        module.ckks_copy(res, src, scratch)?;
        Ok(())
    }
}

/// `res = Σ_t x_t·c_t·π_S` on a replicated layout: each product with its
/// masked constant, at the offset of the scalar combination, is added in the
/// transform domain at the shaping depth of a product under the common scale
/// ([`GLWEMulPlain::glwe_mul_plain_sum`]), and the sum is rounded once
/// with shaping. Each power is prepared as a left operand once for the
/// polynomial, in `ops`. Returns `false` without writing `res` when a term does
/// not share the radix and scales of the others.
fn shaped_baby_linear_combination<'a, BE, V, A, P, Rg: Ring>(
    ops: &CKKSBSGSOps<'a, BE, Rg>,
    module: &Module<BE>,
    res: &mut V,
    terms: &[(&A, usize)],
    coeffs: &[P],
    scratch: &mut ScratchArena<'_, BE>,
) -> anyhow::Result<bool>
where
    BE: GLWEMulPlainImpl<Rg> + 'a,
    Module<BE>: CnvOps<BE, Rg> + GLWENormalize<BE, Rg> + CKKSRoundOps<BE, Rg>,
    V: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos,
{
    use crate::{layouts::validation::product_layout, shaping::product_depth};

    let op = ops.op;
    let Some(&(seed, _)) = terms.last() else {
        return Ok(false);
    };
    let kb = res.base2k().as_usize();
    let delta = seed.log_delta();
    let operand = |i: usize| &coeffs[i];
    let pt_log_delta = operand(terms[0].1).log_delta();
    for &(t, i) in terms {
        let pt = operand(i);
        product_layout(op, &[seed.slot_layout(), t.slot_layout(), pt.slot_layout()])?;
        ensure_plaintext_degree_embeds(op, t.n().as_usize(), pt.n().as_usize())?;
        if t.base2k().as_usize() != kb || t.log_delta() != delta || pt.base2k().as_usize() != kb || pt.log_delta() != pt_log_delta
        {
            return Ok(false);
        }
    }
    let Some(lambda) = product_depth(seed.slot_layout(), pt_log_delta) else {
        return Ok(false);
    };

    let Some(budget) = baby_budget(op, terms, coeffs, delta, kb) else {
        return Ok(false);
    };
    let running = || baby_terms(op, terms, coeffs, delta);

    let wide_k = budget + delta + lambda;
    let mut slots = seed.slots();
    let mut log_sparsity = res.log_sparsity();
    let (mut widest, mut pt_size) = (seed, 0);
    for (t, i, ..) in running() {
        let pt = operand(i);
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

    let mut powers = ops.powers.borrow_mut();
    for &(t, i) in terms {
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
    let prepared = |i: usize| powers[i].as_ref().expect("prepared above");
    let terms = running().map(|(_, i, b, _, off)| (prepared(i), operand(i), off + b - budget));
    ops.dot_product(module, &mut wide, terms, &mut scratch);
    res.set_k((budget + delta).into());
    module.ckks_round(res, &wide, &mut scratch)?;
    Ok(true)
}

pub trait CKKSPolynomialEvaluationReference<BE: Backend, Rg: Ring = Standard>: ModuleN {
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_reference<R, B, A, G, H>(
        &self,
        res: &mut R,
        power_basis: &G,
        poly: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEZero<BE, Rg>
            + CKKSAddOps<BE, Rg>
            + CKKSMulOps<BE, Rg>
            + CKKSRightOperandPreparedFactory<BE, Rg>
            + CKKSMulAddOps<BE, Rg>
            + CKKSCiphertextAlloc<BE>
            + CKKSCopyOps<BE, Rg>
            + CnvOps<BE, Rg>
            + VecZnxBigNormalize<BE, Rg>
            + VecZnxRsh<BE, Rg>
            + VecZnxRshTmpBytes
            + GLWENormalize<BE, Rg>
            + CKKSRoundOps<BE, Rg>
            + Sized,
        R: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE, Rg>,
        B::Coeffs: CKKSCtBounds + GLWEToBackendRef<BE, Rg>,
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A, Rg>,
        H: GetTensorKey<BE, Rg>;
}

/// Reference of [`CKKSComplexPolynomialEvaluationImpl`](crate::oep::CKKSComplexPolynomialEvaluationImpl), which
/// [`impl_ckks_complex_polynomial_evaluation_reference_full!`](crate::impl_ckks_complex_polynomial_evaluation_reference_full)
/// forwards to: the complex-coefficient evaluation of [`CKKSPolynomialEvaluationReference`], on the standard ring, the
/// imaginary unit having no conjugate-invariant image.
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
        BE: GLWEMulPlainImpl,
        Module<BE>: GLWEZero<BE>
            + GLWEPolynomialEvaluation<BE>
            + CKKSAddOps<BE>
            + CKKSImagOps<BE>
            + CKKSMulOps<BE>
            + CKKSRightOperandPreparedFactory<BE>
            + CKKSMulAddOps<BE>
            + CKKSCiphertextAlloc<BE>
            + CKKSCopyOps<BE>
            + CnvOps<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxRsh<BE>
            + VecZnxRshTmpBytes
            + GLWENormalize<BE>
            + CKKSRoundOps<BE>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        C: GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos + GLWEToBackendRef<BE>,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        const OP: &str = "ckks_eval_poly_complex_const_coeffs_from_power_basis";
        let module: &Module<BE> = std::borrow::Borrow::borrow(self);
        let poly_re = &poly.re;
        let poly_im = &poly.im;
        ensure_complex_schedules::<BE, C>(OP, poly)?;
        ensure_power_basis(OP, BSGSPolynomialInfos::<BE>::basis(poly_re), power_basis.basis())?;
        let n_baby = BSGSPolynomialInfos::<BE>::baby_steps(poly_re);

        // Fold the trailing lone constant via the highest power when present.
        let last_re = BSGSPolynomialInfos::<BE>::baby_step(poly_re, n_baby - 1);
        let trailing_const_only = n_baby >= 2 && last_re.len() == 1;
        let fold_power = BSGSPolynomialInfos::<BE>::degree(poly_re);
        let can_fold = trailing_const_only && power_basis.has_power(fold_power);
        let n_to_process = if can_fold { n_baby - 1 } else { n_baby };

        // Per baby step: baby_i = eval(re_i) + i·eval(im_i). A single giant tree
        // over baby_i runs the relinearizations once.
        let parity = BSGSPolynomialInfos::<BE>::parity(poly_re);
        let x = power_basis.get(1)?;
        crate::ensure_operand_degrees(OP, module, [res.n(), x.n()])?;
        let constants = (0..n_baby).flat_map(|i| {
            [
                BSGSPolynomialInfos::<BE>::baby_step(poly_re, i),
                BSGSPolynomialInfos::<BE>::baby_step(poly_im, i),
            ]
        });
        ensure_masked_constants(OP, x.slot_layout(), x.n().as_usize(), constants)?;
        let precision = CKKSBSGSOps::new(OP);
        let mut baby_steps = Vec::with_capacity(n_to_process);
        for i in 0..n_to_process {
            let re_coeffs = BSGSPolynomialInfos::<BE>::baby_step(poly_re, i);
            let im_coeffs = BSGSPolynomialInfos::<BE>::baby_step(poly_im, i);
            let degree = re_coeffs.len() - 1;

            let mut value = module.ckks_ciphertext_alloc_from_infos(x);
            value.set_meta(x.meta());
            module.glwe_eval_baby_step::<_, _, C, A, G>(
                &precision,
                &mut value,
                parity,
                re_coeffs,
                power_basis,
                &mut scratch.borrow(),
            )?;

            let mut im_value = module.ckks_ciphertext_alloc_from_infos(x);
            im_value.set_meta(x.meta());
            module.glwe_eval_baby_step::<_, _, C, A, G>(
                &precision,
                &mut im_value,
                parity,
                im_coeffs,
                power_basis,
                &mut scratch.borrow(),
            )?;
            module.ckks_mul_i_assign(&mut im_value, &mut scratch.borrow())?;
            module.ckks_add_assign(&mut value, &im_value, &mut scratch.borrow())?;

            baby_steps.push(EvaluatedBabyStep { degree, value });
        }

        module.glwe_eval_giant_steps::<_, R, _, _, C, A, G, H>(
            &precision,
            res,
            &mut baby_steps,
            power_basis,
            tsk,
            &mut scratch.borrow(),
        )?;

        if can_fold {
            // res += a·x^fold + i·(b·x^fold), with a = last_re[0], b = last_im[0].
            let last_im = BSGSPolynomialInfos::<BE>::baby_step(poly_im, n_baby - 1);
            let xpow = power_basis.get(fold_power)?;
            module.ckks_mul_add_pt_into(res, xpow, &last_re[0], scratch)?;
            let mut im_fold = module.ckks_ciphertext_alloc_from_infos(res);
            module.ckks_mul_pt_into(&mut im_fold, xpow, &last_im[0], scratch)?;
            module.ckks_mul_i_assign(&mut im_fold, scratch)?;
            module.ckks_add_assign(res, &im_fold, scratch)?;
        }

        Ok(())
    }
}

impl<BE: Backend> CKKSComplexPolynomialEvaluationReference<BE> for Module<BE> {}

impl<BE: Backend, Rg: Ring> CKKSPolynomialEvaluationReference<BE, Rg> for Module<BE>
where
    BE: GLWEMulPlainImpl<Rg>,
    Module<BE>: GLWEPolynomialEvaluation<BE, Rg>,
{
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_reference<R, B, A, G, H>(
        &self,
        res: &mut R,
        power_basis: &G,
        poly: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEZero<BE, Rg>
            + GLWEPolynomialEvaluation<BE, Rg>
            + CKKSAddOps<BE, Rg>
            + CKKSMulOps<BE, Rg>
            + CKKSRightOperandPreparedFactory<BE, Rg>
            + CKKSMulAddOps<BE, Rg>
            + CKKSCiphertextAlloc<BE>
            + CKKSCopyOps<BE, Rg>
            + CnvOps<BE, Rg>
            + VecZnxBigNormalize<BE, Rg>
            + VecZnxRsh<BE, Rg>
            + VecZnxRshTmpBytes
            + GLWENormalize<BE, Rg>
            + CKKSRoundOps<BE, Rg>
            + Sized,
        R: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE, Rg>,
        B::Coeffs: CKKSCtBounds + GLWEToBackendRef<BE, Rg>,
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A, Rg>,
        H: GetTensorKey<BE, Rg>,
    {
        const OP: &str = "ckks_eval_poly_real_const_coeffs_from_power_basis";
        ensure_baby_steps(OP, poly)?;
        ensure_power_basis(OP, poly.basis(), power_basis.basis())?;

        let n_baby = poly.baby_steps();
        let last_coeffs = poly.baby_step(n_baby - 1);
        let trailing_const_only = n_baby >= 2 && last_coeffs.len() == 1;
        let fold_power = poly.degree();
        let can_fold = trailing_const_only && power_basis.has_power(fold_power);

        let n_to_process = if can_fold { n_baby - 1 } else { n_baby };
        // The evaluated baby steps are deliberately heap-allocated: their count
        // is polynomial-dependent (`√degree`-ish) and the giant-step fold
        // consumes them progressively, so a scratch carve would have to reserve
        // the full vector for the whole evaluation. Stage internals are
        // scratch-sized (see `ckks_eval_mod_tmp_bytes`).
        let mut baby_steps = Vec::with_capacity(n_to_process);
        let parity = poly.parity();
        let x = power_basis.get(1)?;
        crate::ensure_operand_degrees(OP, self, [res.n(), x.n()])?;
        ensure_masked_constants(OP, x.slot_layout(), x.n().as_usize(), (0..n_baby).map(|i| poly.baby_step(i)))?;
        let precision = CKKSBSGSOps::new(OP);
        for i in 0..n_to_process {
            let coeffs = poly.baby_step(i);
            let degree = coeffs.len() - 1;
            let mut value = self.ckks_ciphertext_alloc_from_infos(x);
            value.set_meta(x.meta());
            self.glwe_eval_baby_step(&precision, &mut value, parity, coeffs, power_basis, &mut scratch.borrow())?;
            baby_steps.push(EvaluatedBabyStep { degree, value });
        }

        self.glwe_eval_giant_steps::<_, R, _, _, B::Coeffs, A, G, H>(
            &precision,
            res,
            &mut baby_steps,
            power_basis,
            tsk,
            &mut scratch.borrow(),
        )?;

        if can_fold {
            let xpow = power_basis.get(fold_power)?;
            self.ckks_mul_add_pt_into(res, xpow, &last_coeffs[0], scratch)?;
        }

        Ok(())
    }
}

/// `InvalidOperand` for `op` unless `poly` has a baby step.
pub(crate) fn ensure_baby_steps<BE: Backend, B: BSGSPolynomialInfos<BE, Rg>, Rg: Ring>(op: &'static str, poly: &B) -> Result<()> {
    ckks_ensure_operand!(op, poly.baby_steps() > 0, "the polynomial has no baby step");
    Ok(())
}

fn ensure_power_basis(op: &'static str, poly: Basis, power_basis: Basis) -> Result<()> {
    ckks_ensure_operand!(
        op,
        poly == power_basis,
        "polynomial basis {poly:?} does not match power basis {power_basis:?}"
    );
    Ok(())
}

/// `InvalidOperand` for `op` unless the real and imaginary parts of `poly` share a nonempty baby-step schedule and their
/// degree, parity and basis, which fix the length of each step.
pub(crate) fn ensure_complex_schedules<BE, C>(op: &'static str, poly: &ComplexBSGSPolynomial<C>) -> Result<()>
where
    BE: Backend,
    C: GLWEInfos + BSGSMeta + IntPolyInfos,
{
    let (re, im) = (&poly.re, &poly.im);
    ensure_baby_steps::<BE, _, Standard>(op, re)?;
    let n_baby = BSGSPolynomialInfos::<BE>::baby_steps(re);
    ckks_ensure_operand!(
        op,
        BSGSPolynomialInfos::<BE>::baby_steps(im) == n_baby,
        "real/imag baby-step schedules differ"
    );
    ckks_ensure_operand!(
        op,
        BSGSPolynomialInfos::<BE>::degree(im) == BSGSPolynomialInfos::<BE>::degree(re),
        "real/imag degrees differ"
    );
    ckks_ensure_operand!(
        op,
        BSGSPolynomialInfos::<BE>::parity(im) == BSGSPolynomialInfos::<BE>::parity(re),
        "real/imag parities differ"
    );
    ckks_ensure_operand!(
        op,
        BSGSPolynomialInfos::<BE>::basis(im) == BSGSPolynomialInfos::<BE>::basis(re),
        "real/imag bases differ"
    );
    Ok(())
}
