use super::{
    BabyPowers, const_add_assign, const_mul_add_into, const_mul_into, full_baby_linear_combination,
    shaped_baby_linear_combination,
};
use crate::layouts::CKKSCiphertextAlloc;
use crate::{
    CKKSCtBounds, CKKSResult, SetCKKSInfos, SlotLayout,
    api::{
        CKKSAddOps, CKKSCopyOps, CKKSImagOps, CKKSMulAddOps, CKKSMulOps, CKKSPow2Ops, CKKSRightOperandPreparedFactory,
        CKKSRoundOps, CKKSSubOps,
    },
    ckks_ensure_operand,
    layouts::{
        CKKSCiphertextViewMut, CKKSCiphertextViewRef, CKKSMaskedConstants, CKKSRightOperandPrepared,
        ScratchArenaTakeCKKSCiphertext,
        validation::{ensure_masked_constants, ensure_scratch},
    },
    oep::derived::polynomial_evaluation::{
        eval_baby_linear_combination_tmp_bytes, eval_shaped_baby_linear_combination_tmp_bytes,
    },
    polynomial::{EncodedFactoredPolynomial, Factor, FactorUpdate},
};
use poulpy_core::{
    GLWENormalize, GLWEZero,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWEPlaintextInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos, LWEInfos,
        SetGLWEPlaintextInfos, SetK,
    },
    oep::{GLWEMulConstImpl, GLWEMulPlainImpl},
};
use poulpy_hal::{
    api::CnvOps,
    layouts::{Backend, ConjugateInvariant, Module, PrepareHint, Ring, ScratchArena, Standard},
};

const OP: &str = "ckks_eval_poly_factored";

/// Reference for [`CKKSFactoredPolynomialEvaluationOps`](crate::api::CKKSFactoredPolynomialEvaluationOps).
pub trait CKKSFactoredPolynomialEvaluationReference<BE: Backend, Rg: Ring = Standard> {
    fn ckks_eval_poly_factored_tmp_bytes_reference<R, A, P, T>(
        &self,
        res: &R,
        a: &[&A],
        poly: &EncodedFactoredPolynomial<'_, P, Rg>,
        tsk: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSCtBounds + CKKSMaskedConstants<BE, Rg>,
        T: GGLWEInfos;
    fn ckks_eval_poly_factored_reference<R, A, P, H>(
        &self,
        res: &mut R,
        a: &[&A],
        poly: &EncodedFactoredPolynomial<'_, P, Rg>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
        H: GetTensorKey<BE, Rg>;
}

// MulI is a standard-ring operation. The plan constructor excludes it on the other ring.
trait Imaginary<BE: Backend>: Ring {
    fn mul_i<R: GLWEToBackendMut<BE, Self> + CKKSCtBounds + SetCKKSInfos>(
        module: &Module<BE>,
        a: &mut R,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>;
    fn tmp_bytes(module: &Module<BE>, size: usize) -> usize;
}
impl<BE: Backend> Imaginary<BE> for Standard
where
    Module<BE>: CKKSImagOps<BE>,
{
    fn mul_i<R: GLWEToBackendMut<BE, Self> + CKKSCtBounds + SetCKKSInfos>(
        module: &Module<BE>,
        a: &mut R,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()> {
        module.ckks_mul_i_assign(a, scratch)
    }
    fn tmp_bytes(module: &Module<BE>, size: usize) -> usize {
        module.ckks_mul_i_tmp_bytes(size)
    }
}
impl<BE: Backend> Imaginary<BE> for ConjugateInvariant {
    fn mul_i<R: GLWEToBackendMut<BE, Self> + CKKSCtBounds + SetCKKSInfos>(
        _: &Module<BE>,
        _: &mut R,
        _: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()> {
        crate::ckks_bail!("imaginary update on the conjugate-invariant ring")
    }
    fn tmp_bytes(_: &Module<BE>, _: usize) -> usize {
        0
    }
}

fn buffer_bytes<BE: Backend>(ct: &impl CKKSCtBounds) -> usize {
    BE::scratch_aligned(BE::vec_znx_bytes_of(ct.n().into(), (ct.rank() + 1).into(), ct.size()))
}

impl<BE, Rg> CKKSFactoredPolynomialEvaluationReference<BE, Rg> for Module<BE>
where
    BE: Backend + GLWEMulConstImpl<Rg> + GLWEMulPlainImpl<Rg>,
    Rg: Imaginary<BE>,
    Self: CKKSAddOps<BE, Rg>
        + CKKSCopyOps<BE, Rg>
        + CKKSMulOps<BE, Rg>
        + CKKSMulAddOps<BE, Rg>
        + CKKSPow2Ops<BE, Rg>
        + CKKSSubOps<BE, Rg>
        + CKKSRightOperandPreparedFactory<BE, Rg>
        + GLWEZero<BE, Rg>
        + GLWENormalize<BE, Rg>
        + CnvOps<BE, Rg>
        + CKKSRoundOps<BE, Rg>,
{
    fn ckks_eval_poly_factored_tmp_bytes_reference<R, A, P, T>(
        &self,
        res: &R,
        inputs: &[&A],
        poly: &EncodedFactoredPolynomial<'_, P, Rg>,
        key: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSCtBounds + CKKSMaskedConstants<BE, Rg>,
        T: GGLWEInfos,
    {
        let Some(&widest) = inputs.iter().max_by_key(|v| v.k()) else {
            return 0;
        };
        let mut bound = crate::CKKSLayout {
            glwe_layout: widest.glwe_layout(),
            meta: widest.meta(),
        };
        if let Some(max) = inputs.iter().map(|v| v.max_k()).max() {
            bound.glwe_layout.k = max;
        }
        let width = bound.max_size();
        let mut nested = self
            .ckks_copy_tmp_bytes(res, &bound)
            .max(self.ckks_copy_tmp_bytes(&bound, &bound))
            .max(self.ckks_mul_tmp_bytes(&bound, &bound, &bound, key))
            .max(self.ckks_square_tmp_bytes(&bound, &bound, key))
            .max(self.ckks_right_operand_prepare_tmp_bytes(&bound))
            .max(self.ckks_add_tmp_bytes(width))
            .max(self.ckks_sub_tmp_bytes(width))
            .max(self.ckks_mul_pow2_tmp_bytes(width))
            .max(Rg::tmp_bytes(self, width))
            .max(self.ckks_add_pt_const_tmp_bytes(width))
            .max(self.ckks_add_pt_vec_tmp_bytes(width));
        for t in 0..poly.plan.table_sizes.len() {
            let c = poly.table(t);
            nested = nested
                .max(self.ckks_mul_add_pt_const_tmp_bytes(&bound, &bound, c))
                .max(self.ckks_mul_pt_const_tmp_bytes(&bound, &bound, c))
                .max(eval_baby_linear_combination_tmp_bytes::<BE, Rg>(self, width, c.max_size()));
            if matches!(widest.slot_layout(), SlotLayout::Replicated(_)) {
                for &j in &poly.plan.coefficients_read[t] {
                    if let Some(mask) = c.masked_constant(j) {
                        nested = nested
                            .max(self.ckks_mul_pt_vec_tmp_bytes(&bound, &bound, mask))
                            .max(self.ckks_mul_add_pt_vec_tmp_bytes(&bound, &bound, mask))
                            .max(eval_shaped_baby_linear_combination_tmp_bytes::<BE, _, Rg>(
                                self,
                                &bound,
                                mask.max_size(),
                            ));
                    }
                }
            }
        }
        if poly.plan.bsgs.is_some() {
            return nested;
        }
        let imaginary = poly.plan.steps.iter().any(|s| {
            s.updates
                .iter()
                .any(|u| matches!(u, FactorUpdate::MulAddImaginaryConstant(..)))
        });
        buffer_bytes::<BE>(widest) * (poly.plan.temporaries + usize::from(imaginary)) + nested
    }

    fn ckks_eval_poly_factored_reference<R, A, P, H>(
        &self,
        res: &mut R,
        inputs: &[&A],
        poly: &EncodedFactoredPolynomial<'_, P, Rg>,
        key: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
        H: GetTensorKey<BE, Rg>,
    {
        let plan = poly.plan;
        let op = plan.bsgs.map_or(OP, |b| b.op);
        validate_operands::<BE, Rg, _, _, _>(self, op, res, inputs, poly)?;
        let first = inputs[0];
        let widest = if plan.bsgs.is_some() {
            first
        } else {
            let widest = inputs.iter().copied().max_by_key(|a| a.k()).unwrap_or(first);
            let budget = widest.log_budget();
            let used = inputs.iter().map(|a| budget - a.log_budget()).collect();
            crate::error::checked_log_budget_sub(
                op,
                budget,
                plan.budget_cost(used, first.log_delta(), |t| poly.table(t).log_delta()),
            )?;
            widest
        };
        let layout = widest.glwe_layout();
        // BSGS's shared scratch query is independent of polynomial degree. Its intermediate
        // ciphertexts use the heap; standalone programs reserve their pool through the plan query.
        let mut heap = Vec::new();
        let (mut free, mut arena) = if plan.bsgs.is_some() {
            heap.extend((0..plan.temporaries).map(|_| self.ckks_ciphertext_alloc_from_infos::<_, Rg>(first)));
            (heap.iter_mut().map(CKKSCiphertextViewMut::of).collect(), scratch.borrow())
        } else {
            ensure_scratch(op, scratch, plan.temporaries * buffer_bytes::<BE>(widest))?;
            scratch
                .borrow()
                .take_ckks_ciphertext_slice_scratch::<_, Rg>(plan.temporaries, &layout, first.meta())
        };
        let mut values: Vec<Option<CKKSCiphertextViewMut<'_, BE, Rg>>> = (0..plan.steps.len()).map(|_| None).collect();
        let mut prepared: Vec<Option<CKKSRightOperandPrepared<BE, Rg>>> = (0..plan.steps.len()).map(|_| None).collect();
        let mut powers = BabyPowers::new(op);
        for (i, step) in plan.steps.iter().enumerate() {
            let finish = plan.bsgs.is_some() && i + 1 == plan.steps.len();
            let reuse = if finish { None } else { plan.reuse[i] };
            let mut out = reuse.and_then(|v| values[v - plan.inputs].take());
            let get = |v: usize| -> CKKSResult<CKKSCiphertextViewRef<'_, BE, Rg>> {
                if v < plan.inputs {
                    return Ok(CKKSCiphertextViewRef::of(inputs[v]));
                }
                values[v - plan.inputs]
                    .as_ref()
                    .map(|ct| ct.to_backend_view_ref())
                    .ok_or_else(|| anyhow::anyhow!("missing live program value {v}").into())
            };
            if finish {
                let Factor::Value(root) = step.factor else {
                    return Err(anyhow::anyhow!("missing output copy in the compiled program").into());
                };
                self.ckks_copy(res, &get(root)?, &mut arena)?;
                res.set_noise(None);
                return apply_updates(op, self, res, &step.updates, get, poly, &mut arena);
            }
            if let Factor::PrepareRight(v) = step.factor {
                prepared[i] = Some(self.ckks_right_operand_prepare_alloc(&get(v)?, PrepareHint::Reuse, &mut arena)?);
            } else {
                if reuse.is_none() {
                    out = free.pop();
                    if let Some(dst) = &mut out {
                        dst.set_k(layout.k);
                        dst.set_meta(first.meta());
                    }
                }
                let dst = out.as_mut().ok_or_else(|| anyhow::anyhow!("missing program accumulator"))?;
                match &step.factor {
                    Factor::Value(v) => {
                        if reuse.is_none() {
                            self.ckks_copy(dst, &get(*v)?, &mut arena)?;
                        }
                    }
                    Factor::Zero(v) => {
                        seed_zero(self, dst, &get(*v)?);
                    }
                    Factor::Square(v) => {
                        if reuse.is_some() {
                            self.ckks_square_assign(dst, key, &mut arena)?;
                        } else {
                            self.ckks_square_into(dst, &get(*v)?, key, &mut arena)?;
                        }
                    }
                    Factor::Product(a, b) => {
                        if reuse.is_some() {
                            self.ckks_mul_assign(dst, &get(*b)?, key, &mut arena)?;
                        } else {
                            self.ckks_mul_into(dst, &get(*a)?, &get(*b)?, key, &mut arena)?;
                        }
                    }
                    Factor::PreparedProduct(a, b) => {
                        if reuse.is_none() {
                            self.ckks_copy(dst, &get(*a)?, &mut arena)?;
                        }
                        let rhs = prepared[*b - plan.inputs]
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!("missing prepared operand"))?;
                        self.ckks_mul_prepared_assign(dst, rhs, key, &mut arena)?;
                    }
                    Factor::WeightedSum { table, terms, constant } => {
                        let operands = terms.iter().map(|&(v, _)| get(v)).collect::<CKKSResult<Vec<_>>>()?;
                        weighted_sum(
                            &mut powers,
                            self,
                            dst,
                            &operands,
                            terms,
                            poly.table(*table),
                            *constant,
                            &mut arena,
                        )?;
                    }
                    Factor::PrepareRight(_) => return Err(anyhow::anyhow!("prepared value in ciphertext expression").into()),
                }
                apply_updates(op, self, dst, &step.updates, get, poly, &mut arena)?;
                values[i] = out;
            }
            for &v in &plan.release[i] {
                let index = v - plan.inputs;
                if let Some(dead) = values[index].take() {
                    free.push(dead);
                }
                prepared[index] = None;
                if let Some(power) = powers.powers.get_mut(v) {
                    *power = None;
                }
            }
        }
        let result = values
            .last()
            .and_then(Option::as_ref)
            .ok_or_else(|| anyhow::anyhow!("missing program output"))?;
        self.ckks_copy(res, result, &mut arena)
    }
}

fn validate_operands<BE: Backend, Rg: Ring, R, A, P>(
    module: &Module<BE>,
    op: &'static str,
    res: &R,
    inputs: &[&A],
    poly: &EncodedFactoredPolynomial<'_, P, Rg>,
) -> CKKSResult<()>
where
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    P: CKKSCtBounds + CKKSMaskedConstants<BE, Rg>,
{
    let plan = poly.plan;
    ckks_ensure_operand!(op, inputs.len() == plan.inputs, "wrong number of program inputs");
    let first = inputs[0];
    crate::ensure_operand_degrees(op, module, [first.n(), res.n()])?;
    let standalone = plan.bsgs.is_none();
    if standalone {
        ckks_ensure_operand!(
            op,
            first.n() == res.n() && first.rank() == res.rank(),
            "output degree or rank differs from the inputs"
        );
    }
    for &a in inputs {
        ckks_ensure_operand!(
            op,
            a.n() == first.n() && a.rank() == first.rank() && a.slot_layout() == first.slot_layout(),
            "input degree, rank or slot layout differs"
        );
        if standalone {
            ckks_ensure_operand!(
                op,
                a.base2k() == first.base2k() && a.log_delta() == first.log_delta(),
                "input radix or scale differs"
            );
        }
    }
    for (t, &n) in plan.table_sizes.iter().enumerate() {
        let c = poly.table(t);
        ckks_ensure_operand!(op, c.n().as_usize() >= n, "coefficient table {t} is too short");
        ensure_masked_constants::<BE, Rg, P>(op, first.slot_layout(), [c])?;
        if let SlotLayout::Replicated(layout) = first.slot_layout() {
            for &j in &plan.coefficients_read[t] {
                ckks_ensure_operand!(
                    op,
                    c.masked_constant(j)
                        .is_some_and(|m| m.slot_layout() == SlotLayout::Replicated(layout) && m.n() <= first.n()),
                    "coefficient ({t}, {j}) lacks its masked plaintext in the input layout"
                );
            }
        }
    }
    Ok(())
}

/// Try the layout's fused baby step. The fallback seeds from the last power, then
/// accumulates the constant and the weighted terms in their original order.
#[allow(clippy::too_many_arguments)]
fn weighted_sum<BE, Rg: Ring, P>(
    powers: &mut BabyPowers<BE, Rg>,
    module: &Module<BE>,
    dst: &mut CKKSCiphertextViewMut<'_, BE, Rg>,
    operands: &[CKKSCiphertextViewRef<'_, BE, Rg>],
    terms: &[(usize, usize)],
    coefficients: &P,
    constant: Option<usize>,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: GLWEMulConstImpl<Rg> + GLWEMulPlainImpl<Rg>,
    Module<BE>: CKKSAddOps<BE, Rg>
        + CKKSMulAddOps<BE, Rg>
        + GLWEZero<BE, Rg>
        + CnvOps<BE, Rg>
        + GLWENormalize<BE, Rg>
        + CKKSRoundOps<BE, Rg>,
    P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
{
    let op = powers.op;
    let weighted: Vec<_> = operands.iter().zip(terms).map(|(a, &(_, c))| (a, c)).collect();
    let seed = operands.last().ok_or_else(|| anyhow::anyhow!("missing weighted-sum seed"))?;
    let fused = match seed.slot_layout() {
        SlotLayout::Full => full_baby_linear_combination(op, module, dst, &weighted, seed, coefficients, scratch),
        SlotLayout::Replicated(_) => {
            shaped_baby_linear_combination(powers, module, dst, &weighted, coefficients, |j| terms[j].0, scratch)?
        }
    };
    if !fused {
        seed_zero(module, dst, seed);
    }
    if let Some(c) = constant {
        const_add_assign(op, module, dst, coefficients, c, scratch)?;
    }
    if !fused {
        for &(ct, c) in &weighted {
            const_mul_add_into(op, module, dst, ct, coefficients, c, scratch)?;
        }
    }
    dst.set_noise(None);
    Ok(())
}

fn seed_zero<BE: Backend, Rg: Ring>(module: &Module<BE>, res: &mut CKKSCiphertextViewMut<'_, BE, Rg>, seed: &impl CKKSCtBounds)
where
    Module<BE>: GLWEZero<BE, Rg>,
{
    res.set_log_delta(seed.log_delta());
    res.set_log_budget(seed.log_budget());
    res.set_slots(seed.slots());
    res.set_slot_layout(seed.slot_layout());
    module.glwe_zero(res);
}

fn apply_updates<'a, BE, Rg, R, P>(
    op: &'static str,
    module: &Module<BE>,
    res: &mut R,
    updates: &[FactorUpdate],
    get: impl Fn(usize) -> CKKSResult<CKKSCiphertextViewRef<'a, BE, Rg>>,
    poly: &EncodedFactoredPolynomial<'_, P, Rg>,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: Backend + 'a,
    Rg: Imaginary<BE>,
    Module<BE>: CKKSAddOps<BE, Rg> + CKKSSubOps<BE, Rg> + CKKSMulOps<BE, Rg> + CKKSMulAddOps<BE, Rg> + CKKSPow2Ops<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
{
    for update in updates {
        match *update {
            FactorUpdate::Add(v) => module.ckks_add_assign(res, &get(v)?, scratch)?,
            FactorUpdate::Sub(v) => module.ckks_sub_assign(res, &get(v)?, scratch)?,
            FactorUpdate::MulAddConstant(v, (t, c)) => const_mul_add_into(op, module, res, &get(v)?, poly.table(t), c, scratch)?,
            FactorUpdate::AddConstant((t, c)) => const_add_assign(op, module, res, poly.table(t), c, scratch)?,
            FactorUpdate::MulPow2(bits) => module.ckks_mul_pow2_assign(bits as usize, res, scratch)?,
            FactorUpdate::MulI => Rg::mul_i(module, res, scratch)?,
            FactorUpdate::MulAddImaginaryConstant(v, (t, c)) => {
                // The imaginary fold is rounded at the width left by the real fold.
                let mut owned = poly.plan.bsgs.map(|_| module.ckks_ciphertext_alloc_from_infos::<_, Rg>(res));
                let (mut term, mut nested) = if let Some(ct) = &mut owned {
                    (CKKSCiphertextViewMut::of(ct), scratch.borrow())
                } else {
                    scratch.borrow().take_ckks_ciphertext_scratch(&res.glwe_layout(), res.meta())
                };
                const_mul_into(op, module, &mut term, &get(v)?, poly.table(t), c, &mut nested)?;
                Rg::mul_i(module, &mut term, &mut nested)?;
                module.ckks_add_assign(res, &term, &mut nested)?;
            }
        }
    }
    Ok(())
}
