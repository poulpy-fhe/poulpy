//! Scheme-agnostic Baby-Step / Giant-Step polynomial-evaluation engine.
//!
//! Owns the BSGS schedule, parity loop and giant-step folding, composing core
//! GLWE primitives. Per-operation precision integers and the
//! plaintext-coefficient addition are supplied by the scheme through
//! [`BSGSPrecision`].

use anyhow::{Result, ensure};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxAddAssignBackend, VecZnxBigBytesOf,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxDftBytesOf, VecZnxIdftApplyTmpA,
        VecZnxNegateBackend, VecZnxSubAssignBackend,
    },
    layouts::{Backend, ScratchArena},
};

use crate::{
    GLWEAdd, GLWEMulConst, GLWENormalize, GLWEShift, GLWETensoring, GLWEZero, ScratchArenaTakeCore,
    default::operations::{glwe_prepare_right, glwe_tensor_apply_prepared_right},
    layouts::{
        BSGSMeta, BabyStep, GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Parity,
        PowerBasisHelper, SetBSGSMeta, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};

/// HAL bounds required to run the hoisted prepared-right tensor product.
pub trait GiantStepTensorBounds<BE: Backend>:
    Sized
    + ModuleN
    + CnvPVecBytesOf
    + VecZnxDftBytesOf
    + VecZnxBigBytesOf
    + VecZnxIdftApplyTmpA<BE>
    + VecZnxBigNormalize<BE>
    + Convolution<BE>
    + VecZnxSubAssignBackend<BE>
    + VecZnxAddAssignBackend<BE>
    + VecZnxBigNormalizeTmpBytes
    + VecZnxCopyBackend<BE>
    + VecZnxNegateBackend<BE>
{
}

impl<BE: Backend, M> GiantStepTensorBounds<BE> for M where
    M: Sized
        + ModuleN
        + CnvPVecBytesOf
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + Convolution<BE>
        + VecZnxSubAssignBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopyBackend<BE>
        + VecZnxNegateBackend<BE>
{
}

/// Scheme-supplied per-operation precision integers driving the BSGS engine.
pub trait BSGSPrecision<BE: Backend> {
    /// Returns `(log_budget, log_delta, cnv_offset)` for `res = a * b` (ct × ct).
    fn mul_ct_params<R, A, B>(&self, res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
    where
        R: GLWEInfos + BSGSMeta,
        A: GLWEInfos + BSGSMeta,
        B: GLWEInfos + BSGSMeta;

    /// Returns `(log_budget, log_delta, cnv_offset)` for `res = a * pt` (ct × pt).
    fn mul_pt_params<R, A, P>(&self, res: &R, a: &A, pt: &P) -> Result<(usize, usize, usize)>
    where
        R: GLWEInfos + BSGSMeta,
        A: GLWEInfos + BSGSMeta,
        P: GLWEInfos + BSGSMeta;
}

/// Scheme-supplied constant-coefficient addition into `R` from `P`.
pub trait BSGSConstAdd<BE: Backend, R, P> {
    /// Computes `res[res_coeff] += coeffs[idx]`, normalizing `res`.
    fn add_pt_const_assign(
        &self,
        res: &mut R,
        res_coeff: usize,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;
}

/// Evaluates a single baby step into `res`.
#[allow(clippy::too_many_arguments)]
pub fn eval_baby_step<M, PR, R, C, A, G, BE: Backend>(
    module: &M,
    precision: &PR,
    res: &mut R,
    parity: Parity,
    coeffs: &C,
    power_basis: &G,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: GLWEMulConst<BE> + GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE> + GLWEZero<BE>,
    PR: BSGSPrecision<BE> + BSGSConstAdd<BE, R, C>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta,
    C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
    G: PowerBasisHelper<BE, A>,
    for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
{
    let degree = coeffs.n().as_usize() - 1;
    let x = power_basis.get(1)?;
    res.set_bsgs_log_budget(x.bsgs_log_budget());
    res.set_bsgs_log_delta(x.bsgs_log_delta());
    module.glwe_zero(res);

    let mut has_value = false;
    let mut must_normalize = false;
    if parity != Parity::Odd {
        precision.add_pt_const_assign(res, 0, coeffs, 0, scratch)?;
        has_value = true;
        must_normalize = true;
    }

    let (first, step) = match parity {
        Parity::Even => (2, 2),
        Parity::Odd => (1, 2),
        Parity::Full => (1, 1),
    };

    for i in (first..=degree).step_by(step) {
        let xpow = power_basis.get(i)?;
        if has_value {
            mul_add_pt_const_unnormalized(module, precision, res, xpow, coeffs, i, scratch)?;
            must_normalize = true;
        } else {
            mul_pt_const(module, precision, res, xpow, coeffs, i, scratch)?;
            has_value = true;
        }
    }

    if must_normalize {
        module.glwe_normalize_assign(res, scratch);
    }

    Ok(())
}

/// Computes `res = a * coeffs[idx]`, setting `res` precision metadata.
fn mul_pt_const<M, R, A, P, BE: Backend>(
    module: &M,
    precision: &impl BSGSPrecision<BE>,
    res: &mut R,
    a: &A,
    coeffs: &P,
    idx: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: GLWEMulConst<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos + SetBSGSMeta,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
    P: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
{
    let (log_budget, log_delta, cnv_offset) = precision.mul_pt_params(res, a, coeffs)?;
    module.glwe_mul_const(cnv_offset, res, a, coeffs, idx, scratch);
    res.set_bsgs_log_budget(log_budget);
    res.set_bsgs_log_delta(log_delta);
    Ok(())
}

/// Computes `res += a * coeffs[idx]` without normalizing `res`.
fn mul_add_pt_const_unnormalized<M, R, A, P, BE: Backend>(
    module: &M,
    precision: &impl BSGSPrecision<BE>,
    res: &mut R,
    a: &A,
    coeffs: &P,
    idx: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: GLWEMulConst<BE> + GLWEAdd<BE> + GLWEShift<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
    P: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
    for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
{
    scratch.scope(|scratch_local| {
        let tmp_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: res.max_k(),
            rank: res.rank(),
        };
        let (tmp_inner, mut scratch_local) = scratch_local.take_glwe_scratch(&tmp_layout);
        let (tmp_log_budget, tmp_log_delta, cnv_offset) = precision.mul_pt_params(res, a, coeffs)?;
        let mut tmp = CompactCt(tmp_log_budget, tmp_log_delta, tmp_inner);
        module.glwe_mul_const(cnv_offset, &mut tmp, a, coeffs, idx, &mut scratch_local);
        add_assign_unnormalized(module, res, &tmp, &mut scratch_local);
        Ok(())
    })
}

/// Computes `res += a` with budget alignment, without normalizing `res`.
fn add_assign_unnormalized<M, R, A, BE: Backend>(module: &M, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
where
    M: GLWEAdd<BE> + GLWEShift<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos + SetBSGSMeta,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
{
    let res_log_budget = res.bsgs_log_budget();
    let a_log_budget = a.bsgs_log_budget();

    if res_log_budget < a_log_budget {
        module.glwe_lsh_add(res, a, a_log_budget - res_log_budget, scratch);
    } else if res_log_budget > a_log_budget {
        module.glwe_lsh_assign(res, res_log_budget - a_log_budget, scratch);
        module.glwe_add_assign(res, a);
    } else {
        module.glwe_add_assign(res, a);
    }

    res.set_bsgs_log_budget(res_log_budget.min(a_log_budget));
    res.set_bsgs_log_delta(res.bsgs_log_delta().min(a.bsgs_log_delta()));
}

/// Folds the evaluated baby steps into `res` using the giant-step schedule.
pub fn eval_giant_steps<M, R, B, A, G, T, BE: Backend>(
    module: &M,
    precision: &impl BSGSPrecision<BE>,
    res: &mut R,
    baby_steps: &mut [B],
    power_basis: &G,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: GiantStepTensorBounds<BE> + GLWEAdd<BE> + GLWEShift<BE> + GLWETensoring<BE> + GLWENormalize<BE> + crate::GLWECopy<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos + SetBSGSMeta,
    B: BabyStep<BE>,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
    G: PowerBasisHelper<BE, A>,
    T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
{
    ensure!(
        !baby_steps.is_empty(),
        "eval_giant_steps: polynomial must contain at least one baby step"
    );

    let mut active: Vec<(usize, usize)> = baby_steps
        .iter()
        .enumerate()
        .map(|(index, step)| (step.degree(), index))
        .collect();

    while active.len() > 1 {
        let mut next = Vec::with_capacity(active.len().div_ceil(2));
        let mut pairs: Vec<(usize, usize, usize)> = Vec::with_capacity(active.len() / 2);
        let mut i = 0;
        while i < active.len() {
            let is_last = i + 1 == active.len();
            if !is_last && active[i].0 == active[i + 1].0 {
                let gsp = giant_step_power(active[i].0);
                pairs.push((gsp, active[i].1, active[i + 1].1));
                next.push((2 * gsp - 1, active[i + 1].1));
                i += 2;
            } else if is_last && i > 0 {
                let degree = next.last().map(|(degree, _)| *degree).unwrap_or(active[i].0);
                next.push((degree, active[i].1));
                i += 1;
            } else {
                next.push(active[i]);
                i += 1;
            }
        }

        // Process pairs left-to-right, hoisting the prepared `X^{gsp}` across
        // consecutive pairs that share the same giant-step power.
        let mut p = 0;
        while p < pairs.len() {
            let gsp = pairs[p].0;
            let mut run_end = p + 1;
            while run_end < pairs.len() && pairs[run_end].0 == gsp {
                run_end += 1;
            }
            eval_monomial_run(
                module,
                precision,
                baby_steps,
                &pairs[p..run_end],
                power_basis.get(gsp)?,
                tsk,
                scratch,
            )?;
            p = run_end;
        }

        active = next;
    }

    let evaluated = baby_steps.last().expect("non-empty baby step vector");
    module.glwe_copy(res, evaluated.get());
    res.set_bsgs_log_budget(evaluated.get().bsgs_log_budget());
    res.set_bsgs_log_delta(evaluated.get().bsgs_log_delta());

    Ok(())
}

/// Evaluates a run of `b = b * xpow + a` pairs that share the same `xpow`.
///
/// The compacted `xpow` is prepared once into a scratch `CnvPVecR` and reused as
/// the right tensor operand across every pair in the run.
#[allow(clippy::too_many_arguments)]
fn eval_monomial_run<M, B, A, T, BE: Backend>(
    module: &M,
    precision: &impl BSGSPrecision<BE>,
    baby_steps: &mut [B],
    pairs: &[(usize, usize, usize)],
    xpow: &A,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: GiantStepTensorBounds<BE> + GLWEAdd<BE> + GLWEShift<BE> + GLWETensoring<BE> + GLWENormalize<BE> + crate::GLWECopy<BE>,
    B: BabyStep<BE>,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
    T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
{
    let xpow_log_budget = xpow.bsgs_log_budget();
    let xpow_log_delta = xpow.bsgs_log_delta();

    scratch.scope(|run_scratch| {
        // Hoist: compact `xpow` and prepare it into a reusable right operand.
        let (mut xpow_compact, run_scratch) = take_compact_scratch(run_scratch, xpow, xpow_log_budget, xpow_log_delta);
        module.glwe_copy(&mut xpow_compact, xpow);
        let cols = xpow_compact.rank().as_usize() + 1;
        let xpow_size = xpow_compact.size();
        let xpow_effective_k = xpow_compact.bsgs_effective_k();

        let (mut xpow_prep, mut run_scratch) = run_scratch.take_cnv_pvec_right_scratch(module, cols, xpow_size);
        run_scratch = run_scratch.apply_mut(|scratch_prep| {
            glwe_prepare_right(module, &mut xpow_prep, &xpow_compact, xpow_effective_k, scratch_prep);
        });

        for &(_, low_idx, high_idx) in pairs {
            ensure!(low_idx != high_idx, "eval_giant_steps: baby-step pair aliases itself");
            run_scratch.scope(|pair_scratch| {
                let (a, b) = if low_idx < high_idx {
                    let (low_steps, high_steps) = baby_steps.split_at_mut(high_idx);
                    (low_steps[low_idx].get(), high_steps[0].get_mut())
                } else {
                    let (high_steps, low_steps) = baby_steps.split_at_mut(low_idx);
                    (low_steps[0].get(), high_steps[high_idx].get_mut())
                };

                let (mut a_compact, pair_scratch) =
                    take_compact_scratch(pair_scratch, a, a.bsgs_log_budget(), a.bsgs_log_delta());
                let (mut b_compact, mut pair_scratch) =
                    take_compact_scratch(pair_scratch, b, b.bsgs_log_budget(), b.bsgs_log_delta());
                module.glwe_copy(&mut a_compact, a);
                module.glwe_copy(&mut b_compact, b);

                mul_assign_prepared(
                    module,
                    precision,
                    &mut b_compact,
                    &xpow_compact,
                    &xpow_prep,
                    xpow_size,
                    tsk,
                    &mut pair_scratch,
                )?;
                add_assign(module, &mut b_compact, &a_compact, &mut pair_scratch);

                module.glwe_copy(b, &b_compact);
                b.set_bsgs_log_budget(b_compact.0);
                b.set_bsgs_log_delta(b_compact.1);
                Result::<()>::Ok(())
            })?;
        }
        Ok(())
    })
}

/// A compact GLWE scratch buffer carrying its own precision metadata locally.
struct CompactCt<'a, BE: Backend>(usize, usize, crate::layouts::GLWEViewMut<'a, BE>);

impl<'a, BE: Backend> BSGSMeta for CompactCt<'a, BE> {
    fn bsgs_log_budget(&self) -> usize {
        self.0
    }
    fn bsgs_log_delta(&self) -> usize {
        self.1
    }
}

impl<'a, BE: Backend> SetBSGSMeta for CompactCt<'a, BE> {
    fn set_bsgs_log_budget(&mut self, log_budget: usize) {
        self.0 = log_budget;
    }
    fn set_bsgs_log_delta(&mut self, log_delta: usize) {
        self.1 = log_delta;
    }
}

impl<'a, BE: Backend> LWEInfos for CompactCt<'a, BE> {
    fn base2k(&self) -> crate::layouts::Base2K {
        self.2.base2k()
    }
    fn n(&self) -> crate::layouts::Degree {
        self.2.n()
    }
    fn size(&self) -> usize {
        self.2.size()
    }
}

impl<'a, BE: Backend> GLWEInfos for CompactCt<'a, BE> {
    fn rank(&self) -> crate::layouts::Rank {
        self.2.rank()
    }
}

impl<'a, BE: Backend> GLWEToBackendRef<BE> for CompactCt<'a, BE> {
    fn to_backend_ref(&self) -> crate::layouts::GLWE<BE::BufRef<'_>> {
        self.2.to_backend_ref()
    }
}

impl<'a, BE: Backend> GLWEToBackendMut<BE> for CompactCt<'a, BE> {
    fn to_backend_mut(&mut self) -> crate::layouts::GLWE<BE::BufMut<'_>> {
        self.2.to_backend_mut()
    }
}

fn take_compact_scratch<'a, S, C, BE>(scratch: S, ct: &C, log_budget: usize, log_delta: usize) -> (CompactCt<'a, BE>, S)
where
    S: ScratchArenaTakeCore<'a, BE>,
    C: GLWEInfos + BSGSMeta,
    BE: Backend + 'a,
{
    let layout = GLWELayout {
        n: ct.n(),
        base2k: ct.base2k(),
        k: ct.bsgs_effective_k().into(),
        rank: ct.rank(),
    };
    let (inner, scratch) = scratch.take_glwe_scratch(&layout);
    (CompactCt(log_budget, log_delta, inner), scratch)
}

/// Computes `dst *= a` (ct × ct) reusing the caller-prepared right operand `a_prep`.
///
/// `a_prep` is the prepared `CnvPVecR` of `a` and `a_size` its limb count, so the
/// tensor product feeds `mul_ct_params` the same operand metadata as `a`.
#[allow(clippy::too_many_arguments)]
fn mul_assign_prepared<M, V, A, AP, T, BE: Backend>(
    module: &M,
    precision: &impl BSGSPrecision<BE>,
    dst: &mut V,
    a: &A,
    a_prep: &AP,
    a_size: usize,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: GiantStepTensorBounds<BE> + GLWETensoring<BE>,
    V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
    AP: poulpy_hal::layouts::CnvPVecRToBackendRef<BE>,
    T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
{
    let (log_budget, log_delta, cnv_offset) = precision.mul_ct_params(dst, dst, a)?;

    let tensor_layout = GLWELayout {
        n: dst.n(),
        base2k: dst.base2k(),
        k: dst.max_k().max(a.max_k()),
        rank: dst.rank(),
    };
    let scratch_local = scratch.borrow();
    let (mut tmp, mut scratch_local) = scratch_local.take_glwe_tensor_scratch(&tensor_layout);
    let dst_effective_k = dst.bsgs_effective_k();
    glwe_tensor_apply_prepared_right(
        module,
        cnv_offset,
        &mut tmp,
        &*dst,
        dst_effective_k,
        a_prep,
        a_size,
        &mut scratch_local,
    );
    module.glwe_tensor_relinearize(dst, &tmp, tsk, tmp.size() + tsk.dsize().as_usize(), &mut scratch_local);

    dst.set_bsgs_log_budget(log_budget);
    dst.set_bsgs_log_delta(log_delta);
    Ok(())
}

/// Computes `dst += a` with budget alignment, normalizing `dst`.
fn add_assign<M, V, A, BE: Backend>(module: &M, dst: &mut V, a: &A, scratch: &mut ScratchArena<'_, BE>)
where
    M: GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE>,
    V: GLWEToBackendMut<BE> + GLWEInfos + SetBSGSMeta,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
{
    add_assign_unnormalized(module, dst, a, scratch);
    module.glwe_normalize_assign(dst, scratch);
}

fn giant_step_power(degree: usize) -> usize {
    (degree + 1).next_power_of_two()
}
