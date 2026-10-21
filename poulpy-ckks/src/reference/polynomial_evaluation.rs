use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{
    BSGSMeta, GLWEInfos, GLWELayout, GLWEPlaintextInfos, GLWEPlaintextMeta, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
    SetBSGSMeta,
};
use poulpy_core::{BSGSOps, GLWENormalize, GLWEPolynomialEvaluation, GLWEZero, glwe_prepare_left, glwe_prepare_left_tmp_bytes};
use poulpy_hal::{
    api::{
        Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxRsh, VecZnxRshTmpBytes,
    },
    layouts::{
        Backend, CnvPVecLToBackendRef, CnvPVecRToBackendRef, Data, Module, PrepareHint, Ring, ScratchArena,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxWord, vec_znx_backend_ref_from_ref,
    },
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSLayout, Scale, SlotLayout, api::CKKSRoundOps, shaping::MAX_PASS_BITS,
};
use crate::{
    SetCKKSInfos,
    api::{
        BSGSPolynomialInfos, BabyStep as BabyStepInfos, CKKSAddOps, CKKSCopyOps, CKKSImagOps, CKKSMulAddOps, CKKSMulOps,
        CKKSSubOps, PowerBasisHelper,
    },
    layouts::{CKKSCiphertext, CKKSMaskedConstants, CKKSModuleAlloc, CKKSPreparedRight, ScratchArenaTakeCKKS},
    polynomial::ComplexBSGSPolynomial,
};

/// Scratch upper bound for [`BSGSOps::eval_baby_linear_combination`]: BIG
/// accumulator, shifted-constant buffer and per-term convolution/shift/normalize
/// scratch, for the widest term (`ct_size`) and constant (`pt_size`) in limbs.
pub(crate) fn eval_baby_linear_combination_tmp_bytes<BE, M>(module: &M, ct_size: usize, pt_size: usize) -> usize
where
    BE: Backend,
    M: ModuleN + Convolution<BE> + VecZnxRshTmpBytes + VecZnxBigNormalizeTmpBytes,
{
    let acc_size = ct_size + pt_size;
    BE::bytes_of_vec_znx(1, 1, pt_size + 1)
        + BE::bytes_of_vec_znx_big(module.n(), 1, acc_size)
        + module
            .cnv_by_const_apply_tmp_bytes(0, acc_size, ct_size, pt_size + 1)
            .max(module.cnv_by_const_apply_add_tmp_bytes(0, acc_size, ct_size, pt_size + 1))
            .max(module.vec_znx_rsh_tmp_bytes(acc_size))
            .max(module.vec_znx_big_normalize_tmp_bytes())
}

/// Scratch upper bound for the shaped baby-step linear combination of a
/// replicated layout, for terms of the layout of `ct` and masked constants of
/// `pt_size` limbs: each term prepared in turn and summed in the transform
/// domain up to `MAX_PASS_BITS` under the scale, then rounded with shaping.
pub(crate) fn eval_shaped_baby_linear_combination_tmp_bytes<BE, M, C>(module: &M, ct: &C, pt_size: usize) -> usize
where
    BE: Backend,
    M: ModuleN
        + Convolution<BE>
        + GLWENormalize<BE>
        + poulpy_core::GLWEBytesOf<BE>
        + VecZnxRshTmpBytes
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + CKKSRoundOps<BE>,
    C: CKKSCtBounds,
{
    let n = module.n();
    let cols = ct.rank().as_usize() + 1;
    let wide = CKKSLayout {
        glwe_layout: GLWELayout {
            k: (ct.k().as_usize() + MAX_PASS_BITS).into(),
            ..ct.glwe_layout()
        },
        meta: ct.meta(),
    };
    let wide_size = wide.size();
    let g_size = pt_size + 1;
    let acc_size = wide_size + g_size;
    let term = BE::bytes_of_cnv_pvec_left(n, cols, ct.size(), PrepareHint::OneShot)
        + BE::bytes_of_cnv_pvec_right(n, 1, g_size, PrepareHint::OneShot)
        + glwe_prepare_left_tmp_bytes(module, ct.size(), ct)
            .max(
                BE::bytes_of_vec_znx(n, 1, g_size)
                    + module
                        .vec_znx_rsh_tmp_bytes(g_size)
                        .max(module.cnv_prepare_right_tmp_bytes(g_size, g_size)),
            )
            .max(module.cnv_apply_dft_tmp_bytes(0, acc_size, ct.size(), g_size))
            .max(module.cnv_apply_dft_add_tmp_bytes(0, acc_size, ct.size(), g_size));
    let normalize = BE::bytes_of_vec_znx_big(n, 1, acc_size)
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    let combination = BE::bytes_of_vec_znx_dft(n, cols, acc_size) + term.max(normalize);
    BE::bytes_of_vec_znx(n, cols, wide_size) + combination.max(module.ckks_round_tmp_bytes(ct, &wide))
}

struct EvaluatedBabyStep<D: Data, W: ZnxWord, R: Ring> {
    degree: usize,
    value: CKKSCiphertext<D, W, R>,
}

impl<BE, D> BabyStepInfos<BE> for EvaluatedBabyStep<D, BE::ZnxWord, BE::Ring>
where
    BE: Backend,
    D: Data,
    CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    type Value = CKKSCiphertext<D, BE::ZnxWord, BE::Ring>;

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

/// CKKS scale-aware operations for the scale-agnostic core BSGS engine.
///
/// Every method is a thin dispatch to the CKKS API ([`CKKSMulOps`], [`CKKSAddOps`],
/// [`CKKSCopyOps`]); the non-dispatch bits are the accumulator seed, which
/// has no single existing API equivalent, and the fused baby-step linear
/// combinations.
struct CKKSBSGSOps;

impl<BE: Backend, V, P, A, R> BSGSOps<BE, V, P, A, R> for CKKSBSGSOps
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSMulOps<BE>
        + CKKSMulAddOps<BE>
        + CKKSCopyOps<BE>
        + GLWEZero<BE>
        + Convolution<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxRsh<BE>
        + VecZnxRshTmpBytes
        + VecZnxIdftApply<BE>
        + GLWENormalize<BE>
        + CKKSRoundOps<BE>,
    V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE>,
    A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
{
    type Prepared = CKKSPreparedRight<BE>;

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
        module.glwe_zero(res);
        Ok(())
    }

    fn eval_baby_linear_combination(
        &self,
        module: &Module<BE>,
        res: &mut V,
        terms: &[(&A, usize)],
        coeffs: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<bool> {
        use crate::reference::mul::mul_pt_params_raw;

        let Some(&(seed, _)) = terms.last() else {
            return Ok(false);
        };
        if let SlotLayout::Replicated(_) = seed.slot_layout() {
            return shaped_baby_linear_combination(module, res, terms, coeffs, scratch);
        }
        let kb: usize = res.base2k().into();
        let delta = seed.log_delta();
        if kb == 0 {
            return Ok(false);
        }

        let mut budget = seed.log_budget();
        let mut term_params = Vec::with_capacity(terms.len());
        for (t, _) in terms {
            if usize::from(t.base2k()) != kb || t.log_delta() != delta {
                return Ok(false);
            }
            let Ok((b_i, d_i, off_i)) = mul_pt_params_raw(
                budget + delta,
                t.log_delta(),
                t.log_budget(),
                coeffs.log_delta(),
                coeffs.log_budget(),
                coeffs.encoded_k().as_usize(),
            ) else {
                return Ok(false);
            };
            if d_i != delta || off_i < kb {
                return Ok(false);
            }
            budget = budget.min(b_i);
            term_params.push((b_i, off_i));
        }
        for (term_budget, offset) in &mut term_params {
            *offset += *term_budget - budget;
        }

        let cols: usize = res.rank().as_usize() + 1;
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
        scratch.scope(|scratch_local| -> anyhow::Result<()> {
            let (mut g, scratch_local) = scratch_local.take_vec_znx_scratch(1, 1, pt_size + 1);
            let (mut acc, mut scratch_local) = scratch_local.take_vec_znx_big_scratch(module.n(), 1, acc_size);
            for col in 0..cols {
                for (t_idx, ((t, coeff_idx), (_, off))) in terms.iter().zip(term_params.iter()).enumerate() {
                    let (r, rho) = shift_split(*off);
                    let t_bk = t.to_backend_ref();
                    let mut acc_bk = acc.to_backend_mut();
                    let (b_ref, b_coeff);
                    if r == 0 {
                        b_ref = vec_znx_backend_ref_from_ref::<BE>(pt_bk.data());
                        b_coeff = *coeff_idx;
                    } else {
                        {
                            let mut g_bk = g.to_backend_mut();
                            module.vec_znx_rsh(
                                kb,
                                r,
                                &mut g_bk,
                                0,
                                &vec_znx_backend_ref_from_ref::<BE>(pt_bk.data()).window_coeffs(*coeff_idx, 1),
                                0,
                                &mut scratch_local.borrow(),
                            );
                        }
                        b_ref = g.to_backend_ref();
                        b_coeff = 0;
                    }
                    if t_idx == 0 {
                        module.cnv_by_const_apply(
                            rho,
                            &mut acc_bk,
                            0,
                            t_bk.data(),
                            col,
                            &b_ref,
                            0,
                            b_coeff,
                            &mut scratch_local.borrow(),
                        );
                    } else {
                        module.cnv_by_const_apply_add(
                            rho,
                            &mut acc_bk,
                            0,
                            t_bk.data(),
                            col,
                            &b_ref,
                            0,
                            b_coeff,
                            &mut scratch_local.borrow(),
                        );
                    }
                }
                let acc_ref = acc.to_backend_ref();
                let mut res_bk = res.to_backend_mut();
                module.vec_znx_big_normalize(
                    res_bk.data_mut(),
                    kb,
                    budget + delta,
                    0,
                    col,
                    &acc_ref,
                    kb,
                    0,
                    &mut scratch_local.borrow(),
                );
            }
            Ok(())
        })?;
        Ok(true)
    }

    fn add_pt_const_assign(
        &self,
        module: &Module<BE>,
        res: &mut V,
        res_coeff: usize,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        if res_coeff == 0 {
            return const_add_assign(module, res, coeffs, idx, scratch).map_err(::anyhow::Error::from);
        }
        module
            .ckks_add_pt_const_assign(res, res_coeff, coeffs, idx, scratch)
            .map_err(::anyhow::Error::from)
    }

    fn mul_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut V,
        a: &A,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        const_mul_into(module, res, a, coeffs, idx, scratch).map_err(::anyhow::Error::from)
    }

    fn mul_add_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut V,
        a: &A,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        const_mul_add_into(module, res, a, coeffs, idx, scratch).map_err(::anyhow::Error::from)
    }

    fn prepare_right(&self, module: &Module<BE>, a: &A, scratch: &mut ScratchArena<'_, BE>) -> anyhow::Result<Self::Prepared> {
        // `X^{gsp}` multiplies every pair of its giant-step level.
        module
            .ckks_prepare_right(a, PrepareHint::Reuse, scratch)
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
        H: GetTensorKey<BE>,
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
/// transform domain at the shaping depth of a product under the common scale,
/// and the sum is rounded once with shaping. Returns `false` without writing
/// `res` when a term has no prepared power or masked constant, or does not
/// share the radix and scales of the others.
fn shaped_baby_linear_combination<BE, V, A, P>(
    module: &Module<BE>,
    res: &mut V,
    terms: &[(&A, usize)],
    coeffs: &P,
    scratch: &mut ScratchArena<'_, BE>,
) -> anyhow::Result<bool>
where
    BE: Backend,
    Module<BE>:
        Convolution<BE> + GLWENormalize<BE> + VecZnxRsh<BE> + VecZnxIdftApply<BE> + VecZnxBigNormalize<BE> + CKKSRoundOps<BE>,
    V: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: CKKSMaskedConstants<BE>,
{
    use crate::reference::mul::{mul_pt_params_raw, product_depth, product_layout};

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
        product_layout(
            "ckks_eval_baby_linear_combination",
            &[seed.slot_layout(), t.slot_layout(), pt.slot_layout()],
        )?;
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
                *budget + delta,
                t.log_delta(),
                t.log_budget(),
                pt.log_delta(),
                pt.log_budget(),
                pt.encoded_k().as_usize(),
            )
            .ok()?;
            *budget = (*budget).min(b);
            Some((t, pt, b, d, off))
        })
    };
    let mut budget = seed.log_budget();
    let mut count = 0;
    for (_, _, b, d, off) in running() {
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
    for (t, pt, b, _, off) in running() {
        let (r, rho) = split(b, off);
        let g_size = pt.to_backend_ref().data().size() + usize::from(r != 0);
        acc_size = acc_size.max((t.k().as_usize().div_ceil(kb) + g_size - 1).saturating_sub(rho));
        log_sparsity = log_sparsity.min(t.log_sparsity()).min(t.log_n() - pt.log_n());
        slots = slots.join(t.slots()).join(pt.slots());
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
    wide.set_canonical(true);
    let (n, cols) = (res.n().as_usize(), res.rank().as_usize() + 1);
    scratch.scope(|scratch_local| {
        let (mut acc, mut scratch_local) = scratch_local.take_vec_znx_dft_scratch(n, cols, acc_size);
        for (t_idx, (t, pt, b, _, off)) in running().enumerate() {
            let (r, rho) = split(b, off);
            let a_size = t.k().as_usize().div_ceil(kb);
            let pt_bk = pt.to_backend_ref();
            let (pt_n, pt_size) = (pt_bk.n().as_usize(), pt_bk.data().size());
            let g_size = pt_size + usize::from(r != 0);
            let pt_data = vec_znx_backend_ref_from_ref::<BE>(pt_bk.data());
            let (mut left, scratch_term) =
                scratch_local
                    .borrow()
                    .take_cnv_pvec_left_scratch(n, cols, a_size, PrepareHint::OneShot);
            let (mut right, mut scratch_term) = scratch_term.take_cnv_pvec_right_scratch(pt_n, 1, g_size, PrepareHint::OneShot);
            glwe_prepare_left(module, &mut left, t, t.k().as_usize(), &mut scratch_term);
            if r == 0 {
                module.cnv_prepare_right(&mut right, &pt_data, &mut scratch_term);
            } else {
                let (mut g, mut scratch_g) = scratch_term.borrow().take_vec_znx_scratch(pt_n, 1, g_size);
                module.vec_znx_rsh(kb, r, &mut g.to_backend_mut(), 0, &pt_data, 0, &mut scratch_g.borrow());
                module.cnv_prepare_right(&mut right, &g.to_backend_ref(), &mut scratch_g);
            }
            let (a, b) = (left.to_backend_ref(), right.to_backend_ref());
            for col in 0..cols {
                let mut acc_bk = acc.to_backend_mut();
                if t_idx == 0 {
                    module.cnv_apply_dft(rho, &mut acc_bk, col, &a, col, &b, 0, 1, &mut scratch_term);
                } else {
                    module.cnv_apply_dft_add(rho, &mut acc_bk, col, &a, col, &b, 0, 1, &mut scratch_term);
                }
            }
        }
        let (mut big, mut scratch_local) = scratch_local.take_vec_znx_big_scratch(n, 1, acc_size);
        for col in 0..cols {
            module.vec_znx_idft_apply(
                &mut big.to_backend_mut(),
                0,
                &acc.to_backend_ref(),
                col,
                &mut scratch_local.borrow(),
            );
            let mut wide_bk = wide.to_backend_mut();
            module.vec_znx_big_normalize(
                wide_bk.data_mut(),
                kb,
                wide_k,
                0,
                col,
                &big.to_backend_ref(),
                kb,
                0,
                &mut scratch_local.borrow(),
            );
        }
    });
    res.set_k((budget + delta).into());
    module.ckks_round_into(res, &wide, &mut scratch)?;
    Ok(true)
}

/// The masked plaintext of constant `idx` when `slot_layout` is replicated,
/// whose constants are masked plaintexts.
fn masked<'p, BE: Backend, P: CKKSMaskedConstants<BE>>(
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

/// `res += c` for constant `c = coeffs[idx]`, as `c·π_S` on a replicated layout.
pub(crate) fn const_add_assign<BE, R, P>(
    module: &Module<BE>,
    res: &mut R,
    coeffs: &P,
    idx: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSAddOps<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE>,
{
    match masked("ckks_add_const", res.slot_layout(), coeffs, idx)? {
        Some(pt) => module.ckks_add_pt_vec_assign(res, pt, scratch),
        None => module.ckks_add_pt_const_assign(res, 0, coeffs, idx, scratch),
    }
}

/// `res -= c` for constant `c = coeffs[idx]`, as `c·π_S` on a replicated layout.
pub(crate) fn const_sub_assign<BE, R, P>(
    module: &Module<BE>,
    res: &mut R,
    coeffs: &P,
    idx: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSSubOps<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE>,
{
    match masked("ckks_sub_const", res.slot_layout(), coeffs, idx)? {
        Some(pt) => module.ckks_sub_pt_vec_assign(res, pt, scratch),
        None => module.ckks_sub_pt_const_assign(res, 0, coeffs, idx, scratch),
    }
}

/// `res = a·c` for constant `c = coeffs[idx]`, as a shaped product by
/// `c·π_S` on a replicated layout.
pub(crate) fn const_mul_into<BE, R, A, P>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    coeffs: &P,
    idx: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSMulOps<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE>,
{
    match masked("ckks_mul_const", a.slot_layout(), coeffs, idx)? {
        Some(pt) => module.ckks_mul_pt_vec_into(res, a, pt, scratch),
        None => module.ckks_mul_pt_const_into(res, a, coeffs, idx, scratch),
    }
}

/// `res += a·c` for constant `c = coeffs[idx]`, as a shaped product by
/// `c·π_S` on a replicated layout.
pub(crate) fn const_mul_add_into<BE, R, A, P>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    coeffs: &P,
    idx: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSMulAddOps<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE>,
{
    match masked("ckks_mul_add_const", a.slot_layout(), coeffs, idx)? {
        Some(pt) => module.ckks_mul_add_pt_vec_into(res, a, pt, scratch),
        None => module.ckks_mul_add_pt_const_into(res, a, coeffs, idx, scratch),
    }
}

pub trait PolynomialEvaluationReference<BE: Backend> {
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_reference<R, B, A, G, H>(
        &self,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEZero<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSModuleAlloc<BE>
            + CKKSCopyOps<BE>
            + Convolution<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxRsh<BE>
            + VecZnxRshTmpBytes
            + VecZnxIdftApply<BE>
            + GLWENormalize<BE>
            + CKKSRoundOps<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds + CKKSMaskedConstants<BE>,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>;

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_reference<R, C, A, G, H>(
        &self,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEZero<BE>
            + CKKSAddOps<BE>
            + CKKSImagOps<BE>
            + CKKSMulOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSModuleAlloc<BE>
            + CKKSCopyOps<BE>
            + Convolution<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxRsh<BE>
            + VecZnxRshTmpBytes
            + VecZnxIdftApply<BE>
            + GLWENormalize<BE>
            + CKKSRoundOps<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE>,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>;
}

impl<BE: Backend> PolynomialEvaluationReference<BE> for Module<BE>
where
    Module<BE>: GLWEPolynomialEvaluation<BE>,
{
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_reference<R, B, A, G, H>(
        &self,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEZero<BE>
            + GLWEPolynomialEvaluation<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSModuleAlloc<BE>
            + CKKSCopyOps<BE>
            + Convolution<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxRsh<BE>
            + VecZnxRshTmpBytes
            + VecZnxIdftApply<BE>
            + GLWENormalize<BE>
            + CKKSRoundOps<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds + CKKSMaskedConstants<BE>,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        ckks_ensure!(
            poly.baby_steps() > 0,
            "ckks_eval_poly_real_const_coeffs_from_power_basis: polynomial must contain at least one baby step"
        );
        let poly_basis = poly.basis();
        let power_basis_basis = power_basis.basis();
        ckks_ensure!(
            poly_basis == power_basis_basis,
            "ckks_eval_poly_real_const_coeffs_from_power_basis: polynomial basis {poly_basis:?} does not match power basis {power_basis_basis:?}"
        );

        let n_baby = poly.baby_steps();
        let last_coeffs = poly.baby_step(n_baby - 1);
        let trailing_const_only = n_baby >= 2 && last_coeffs.n().as_usize() == 1;
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
        let precision = CKKSBSGSOps;
        for i in 0..n_to_process {
            let coeffs = poly.baby_step(i);
            let degree = coeffs.n().as_usize() - 1;
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
            const_mul_add_into(self, res, xpow, last_coeffs, 0, scratch)?;
        }

        Ok(())
    }

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_reference<R, C, A, G, H>(
        &self,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEZero<BE>
            + GLWEPolynomialEvaluation<BE>
            + CKKSAddOps<BE>
            + CKKSImagOps<BE>
            + CKKSMulOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSModuleAlloc<BE>
            + CKKSCopyOps<BE>
            + Convolution<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxRsh<BE>
            + VecZnxRshTmpBytes
            + VecZnxIdftApply<BE>
            + GLWENormalize<BE>
            + CKKSRoundOps<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE>,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        let poly_re = &poly.re;
        let poly_im = &poly.im;
        let n_baby = BSGSPolynomialInfos::<BE>::baby_steps(poly_re);
        ckks_ensure!(
            n_baby > 0,
            "ckks_eval_poly_complex_const_coeffs_from_power_basis: polynomial must contain at least one baby step"
        );
        ckks_ensure!(
            BSGSPolynomialInfos::<BE>::baby_steps(poly_im) == n_baby,
            "ckks_eval_poly_complex_const_coeffs_from_power_basis: real/imag baby-step schedules differ"
        );
        ckks_ensure!(
            BSGSPolynomialInfos::<BE>::degree(poly_im) == BSGSPolynomialInfos::<BE>::degree(poly_re),
            "ckks_eval_poly_complex_const_coeffs_from_power_basis: real/imag degrees differ"
        );
        ckks_ensure!(
            BSGSPolynomialInfos::<BE>::parity(poly_im) == BSGSPolynomialInfos::<BE>::parity(poly_re),
            "ckks_eval_poly_complex_const_coeffs_from_power_basis: real/imag parities differ"
        );
        ckks_ensure!(
            BSGSPolynomialInfos::<BE>::basis(poly_im) == BSGSPolynomialInfos::<BE>::basis(poly_re),
            "ckks_eval_poly_complex_const_coeffs_from_power_basis: real/imag bases differ"
        );
        let poly_basis = BSGSPolynomialInfos::<BE>::basis(poly_re);
        let power_basis_basis = power_basis.basis();
        ckks_ensure!(
            poly_basis == power_basis_basis,
            "ckks_eval_poly_complex_const_coeffs_from_power_basis: polynomial basis {poly_basis:?} does not match power basis {power_basis_basis:?}"
        );

        // Fold the trailing lone constant via the highest power when present.
        let last_re = BSGSPolynomialInfos::<BE>::baby_step(poly_re, n_baby - 1);
        let trailing_const_only = n_baby >= 2 && last_re.n().as_usize() == 1;
        let fold_power = BSGSPolynomialInfos::<BE>::degree(poly_re);
        let can_fold = trailing_const_only && power_basis.has_power(fold_power);
        let n_to_process = if can_fold { n_baby - 1 } else { n_baby };

        // Per baby step: baby_i = eval(re_i) + i·eval(im_i). A single giant tree
        // over baby_i runs the relinearizations once.
        let parity = BSGSPolynomialInfos::<BE>::parity(poly_re);
        let x = power_basis.get(1)?;
        let precision = CKKSBSGSOps;
        let mut baby_steps = Vec::with_capacity(n_to_process);
        for i in 0..n_to_process {
            let re_coeffs = BSGSPolynomialInfos::<BE>::baby_step(poly_re, i);
            let im_coeffs = BSGSPolynomialInfos::<BE>::baby_step(poly_im, i);
            ckks_ensure!(
                im_coeffs.n() == re_coeffs.n(),
                "ckks_eval_poly_complex_const_coeffs_from_power_basis: real/imag baby-step {i} lengths differ"
            );
            let degree = re_coeffs.n().as_usize() - 1;

            let mut value = self.ckks_ciphertext_alloc_from_infos(x);
            value.set_meta(x.meta());
            self.glwe_eval_baby_step::<_, _, C, A, G>(
                &precision,
                &mut value,
                parity,
                re_coeffs,
                power_basis,
                &mut scratch.borrow(),
            )?;

            let mut im_value = self.ckks_ciphertext_alloc_from_infos(x);
            im_value.set_meta(x.meta());
            self.glwe_eval_baby_step::<_, _, C, A, G>(
                &precision,
                &mut im_value,
                parity,
                im_coeffs,
                power_basis,
                &mut scratch.borrow(),
            )?;
            self.ckks_mul_i_assign(&mut im_value, &mut scratch.borrow())?;
            self.ckks_add_assign(&mut value, &im_value, &mut scratch.borrow())?;

            baby_steps.push(EvaluatedBabyStep { degree, value });
        }

        self.glwe_eval_giant_steps::<_, R, _, _, C, A, G, H>(
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
            ckks_ensure!(
                last_im.n() == last_re.n(),
                "ckks_eval_poly_complex_const_coeffs_from_power_basis: real/imag trailing baby-step lengths differ"
            );
            let xpow = power_basis.get(fold_power)?;
            const_mul_add_into(self, res, xpow, last_re, 0, scratch)?;
            let mut im_fold = self.ckks_ciphertext_alloc_from_infos(res);
            const_mul_into(self, &mut im_fold, xpow, last_im, 0, scratch)?;
            self.ckks_mul_i_assign(&mut im_fold, scratch)?;
            self.ckks_add_assign(res, &im_fold, scratch)?;
        }

        Ok(())
    }
}
