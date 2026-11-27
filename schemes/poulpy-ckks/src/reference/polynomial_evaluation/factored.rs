use super::{CKKSBSGSOps, const_add_assign, const_mul_add_into, full_baby_linear_combination, shaped_baby_linear_combination};
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

// Only a standard-ring program can contain MulI. Keep its operation statically typed on that ring.
trait Imaginary<BE: Backend>: Ring {
    fn mul_i(
        module: &Module<BE>,
        a: &mut CKKSCiphertextViewMut<'_, BE, Self>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>;
    fn tmp_bytes(module: &Module<BE>, size: usize) -> usize;
}

impl<BE: Backend> Imaginary<BE> for Standard
where
    Module<BE>: CKKSImagOps<BE>,
{
    fn mul_i(
        module: &Module<BE>,
        a: &mut CKKSCiphertextViewMut<'_, BE, Self>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()> {
        module.ckks_mul_i_assign(a, scratch)
    }
    fn tmp_bytes(module: &Module<BE>, size: usize) -> usize {
        module.ckks_mul_i_tmp_bytes(size)
    }
}
impl<BE: Backend> Imaginary<BE> for ConjugateInvariant {
    fn mul_i(_: &Module<BE>, _: &mut CKKSCiphertextViewMut<'_, BE, Self>, _: &mut ScratchArena<'_, BE>) -> CKKSResult<()> {
        crate::ckks_bail!("imaginary update in a conjugate-invariant program")
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
                .max(eval_baby_linear_combination_tmp_bytes::<BE, Rg>(self, width, c.max_size()));
            if matches!(widest.slot_layout(), SlotLayout::Replicated(_)) {
                for j in 0..poly.plan.table_sizes[t] {
                    if let Some(mask) = c.masked_constant(j) {
                        nested = nested.max(self.ckks_mul_add_pt_vec_tmp_bytes(&bound, &bound, mask)).max(
                            eval_shaped_baby_linear_combination_tmp_bytes::<BE, _, Rg>(self, &bound, mask.max_size()),
                        );
                    }
                }
            }
        }
        buffer_bytes::<BE>(widest) * poly.plan.temporaries + nested
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
        ckks_ensure_operand!(OP, inputs.len() == plan.inputs, "wrong number of program inputs");
        let first = inputs[0];
        crate::ensure_operand_degrees(OP, self, [first.n(), res.n()])?;
        ckks_ensure_operand!(
            OP,
            first.n() == res.n() && first.rank() == res.rank(),
            "output degree or rank differs from the inputs"
        );
        for &a in inputs {
            ckks_ensure_operand!(
                OP,
                a.n() == first.n()
                    && a.rank() == first.rank()
                    && a.base2k() == first.base2k()
                    && a.log_delta() == first.log_delta()
                    && a.slot_layout() == first.slot_layout(),
                "input degree, rank, radix, scale or slot layout differs"
            );
        }
        for (t, &n) in plan.table_sizes.iter().enumerate() {
            let c = poly.table(t);
            ckks_ensure_operand!(OP, c.n().as_usize() >= n, "coefficient table {t} is too short");
            ensure_masked_constants::<BE, Rg, P>(OP, first.slot_layout(), [c])?;
            if let SlotLayout::Replicated(layout) = first.slot_layout() {
                for j in 0..n {
                    ckks_ensure_operand!(
                        OP,
                        c.masked_constant(j)
                            .is_some_and(|m| m.slot_layout() == SlotLayout::Replicated(layout) && m.n() <= first.n()),
                        "coefficient ({t}, {j}) lacks its masked plaintext in the input layout"
                    );
                }
            }
        }
        let widest = inputs.iter().copied().max_by_key(|a| a.k()).unwrap_or(first);
        let budget = widest.log_budget();
        let used = inputs.iter().map(|a| budget - a.log_budget()).collect();
        crate::error::checked_log_budget_sub(
            OP,
            budget,
            plan.budget_cost(used, first.log_delta(), |t| poly.table(t).log_delta()),
        )?;
        ensure_scratch(OP, scratch, plan.temporaries * buffer_bytes::<BE>(widest))?;
        let layout = widest.glwe_layout();
        let (mut free, mut arena) =
            scratch
                .borrow()
                .take_ckks_ciphertext_slice_scratch::<_, Rg>(plan.temporaries, &layout, first.meta());
        let mut values: Vec<Option<CKKSCiphertextViewMut<'_, BE, Rg>>> = (0..plan.steps.len()).map(|_| None).collect();
        let mut prepared: Vec<Option<CKKSRightOperandPrepared<BE, Rg>>> = (0..plan.steps.len()).map(|_| None).collect();
        let powers = CKKSBSGSOps::new(OP);
        for (i, step) in plan.steps.iter().enumerate() {
            let reuse = plan.reuse[i];
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
                        let coefficients = poly.table(*table);
                        let operands = terms.iter().map(|&(v, _)| get(v)).collect::<CKKSResult<Vec<_>>>()?;
                        let weighted: Vec<_> = operands.iter().zip(terms).map(|(a, &(_, c))| (a, c)).collect();
                        let Some(&(seed, _)) = weighted.last() else {
                            return Err(anyhow::anyhow!("missing weighted-sum seed").into());
                        };
                        let fused = match seed.slot_layout() {
                            SlotLayout::Full => {
                                full_baby_linear_combination(OP, self, dst, &weighted, seed, coefficients, &mut arena)
                            }
                            SlotLayout::Replicated(_) => shaped_baby_linear_combination(
                                &powers,
                                self,
                                dst,
                                &weighted,
                                coefficients,
                                |j| terms[j].0,
                                &mut arena,
                            )?,
                        };
                        if !fused {
                            seed_zero(self, dst, seed);
                        }
                        if let Some(c) = constant {
                            const_add_assign(OP, self, dst, coefficients, *c, &mut arena)?;
                        }
                        if !fused {
                            for &(ct, c) in &weighted {
                                const_mul_add_into(OP, self, dst, ct, coefficients, c, &mut arena)?;
                            }
                        }
                        dst.set_noise(None);
                    }
                    Factor::PrepareRight(_) => return Err(anyhow::anyhow!("prepared value in ciphertext expression").into()),
                }
                for update in &step.updates {
                    match *update {
                        FactorUpdate::Add(v) => self.ckks_add_assign(dst, &get(v)?, &mut arena)?,
                        FactorUpdate::Sub(v) => self.ckks_sub_assign(dst, &get(v)?, &mut arena)?,
                        FactorUpdate::MulAddConstant(v, (t, c)) => {
                            const_mul_add_into(OP, self, dst, &get(v)?, poly.table(t), c, &mut arena)?
                        }
                        FactorUpdate::AddConstant((t, c)) => const_add_assign(OP, self, dst, poly.table(t), c, &mut arena)?,
                        FactorUpdate::MulPow2(bits) => self.ckks_mul_pow2_assign(bits as usize, dst, &mut arena)?,
                        FactorUpdate::MulI => Rg::mul_i(self, dst, &mut arena)?,
                    }
                }
                values[i] = out;
            }
            for &v in &plan.release[i] {
                let index = v - plan.inputs;
                if let Some(dead) = values[index].take() {
                    free.push(dead);
                }
                prepared[index] = None;
                if let Some(power) = powers.powers.borrow_mut().get_mut(v) {
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
