use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_core::{GLWENormalize, GLWEZero, ScratchArenaTakeCore};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, ScratchArena},
};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{
        BSGSPolynomialInfos, BabyStep as BabyStepInfos, CKKSAddOps, CKKSCopyOps, CKKSMulAddOps, CKKSMulOps, Parity,
        PowerBasisHelper,
    },
    layouts::{CKKSCiphertext, CKKSModuleAlloc, ScratchArenaTakeCKKS},
};
use anyhow::{Result, ensure};

struct EvaluatedBabyStep<D: poulpy_hal::layouts::Data> {
    degree: usize,
    value: CKKSCiphertext<D>,
}

impl<BE, D> BabyStepInfos<BE> for EvaluatedBabyStep<D>
where
    BE: Backend,
    D: poulpy_hal::layouts::Data,
    CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    type Value = CKKSCiphertext<D>;

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

pub trait PolynomialEvaluationDefault<BE: Backend> {
    fn ckks_eval_baby_step_default<R, C, A, G>(
        &self,
        res: &mut R,
        coeffs: &C,
        parity: Parity,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSAddOps<BE> + CKKSMulAddOps<BE> + CKKSMulOps<BE> + GLWENormalize<BE> + GLWEZero<BE> + Sized,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchAvailable + ScratchArenaTakeCore<'b, BE>,
    {
        eval_baby_step(self, res, parity, coeffs, power_basis, scratch)
    }

    fn ckks_eval_giant_steps_default<R, B, A, G, T>(
        &self,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSModuleAlloc<BE> + Sized,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        B: BabyStepInfos<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
    {
        eval_giant_steps(self, res, baby_steps, power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_real_const_coeffs_from_power_basis_default<R, B, A, G, T>(
        &self,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSAddOps<BE>
            + CKKSCopyOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSMulOps<BE>
            + GLWENormalize<BE>
            + GLWEZero<BE>
            + CKKSModuleAlloc<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        B: BSGSPolynomialInfos<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchAvailable + ScratchArenaTakeCore<'b, BE>,
    {
        ensure!(
            poly.baby_steps() > 0,
            "ckks_eval_poly_real_const_coeffs_from_power_basis: polynomial must contain at least one baby step"
        );

        let mut baby_steps = Vec::with_capacity(poly.baby_steps());
        let parity = poly.parity();
        let x = power_basis.get(1)?;
        for i in 0..poly.baby_steps() {
            let coeffs = poly.baby_step(i);
            let degree = coeffs.n().as_usize() - 1;
            let mut value = self.ckks_ciphertext_alloc_from_infos(x);
            value.set_meta(x.meta());
            self.ckks_eval_baby_step_default::<_, _, A, G>(&mut value, coeffs, parity, power_basis, &mut scratch.borrow())?;
            baby_steps.push(EvaluatedBabyStep { degree, value });
        }

        self.ckks_eval_giant_steps_default(res, &mut baby_steps, power_basis, tsk, &mut scratch.borrow())?;
        Ok(())
    }
}

fn eval_baby_step<M, R, C, A, G, BE: Backend>(
    module: &M,
    res: &mut R,
    parity: Parity,
    coeffs: &C,
    power_basis: &G,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: CKKSAddOps<BE> + CKKSMulAddOps<BE> + CKKSMulOps<BE> + GLWENormalize<BE> + GLWEZero<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    C: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    G: PowerBasisHelper<BE, A>,
    for<'b> ScratchArena<'b, BE>: ScratchAvailable + ScratchArenaTakeCore<'b, BE>,
{
    let degree = coeffs.n().as_usize() - 1;
    let x = power_basis.get(1)?;
    res.set_meta(x.meta());
    module.glwe_zero(res);

    // For odd polynomials the constant term is zero; skip it.
    let mut has_value = false;
    let mut must_normalize = false;
    if parity != Parity::Odd {
        module.ckks_add_pt_const_assign(res, 0, coeffs, 0, scratch)?;
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
            module.ckks_mul_add_pt_const_into_unnormalized(res, xpow, coeffs, i, scratch)?;
            must_normalize = true;
        } else {
            module.ckks_mul_pt_const_into(res, xpow, coeffs, i, scratch)?;
            has_value = true;
        }
    }

    if must_normalize {
        module.glwe_normalize_assign(res, scratch);
    }

    Ok(())
}

fn eval_giant_steps<M, R, B, A, G, T, BE: Backend>(
    module: &M,
    res: &mut R,
    baby_steps: &mut [B],
    power_basis: &G,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSModuleAlloc<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    B: BabyStepInfos<BE>,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    G: PowerBasisHelper<BE, A>,
    T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
{
    ensure!(
        !baby_steps.is_empty(),
        "ckks_eval_giant_steps: polynomial must contain at least one baby step"
    );

    let mut active: Vec<(usize, usize)> = baby_steps
        .iter()
        .enumerate()
        .map(|(index, step)| (step.degree(), index))
        .collect();

    // Each round pairs consecutive equal-degree steps: combined = high * X^gsp + low.
    // Unmatched tails inherit the degree of their predecessor so they pair in the next round.
    while active.len() > 1 {
        let mut next = Vec::with_capacity(active.len().div_ceil(2));
        let mut i = 0;
        while i < active.len() {
            let is_last = i + 1 == active.len();
            if !is_last && active[i].0 == active[i + 1].0 {
                let gsp = giant_step_power(active[i].0);
                eval_monomial_pair(
                    module,
                    baby_steps,
                    active[i].1,
                    active[i + 1].1,
                    power_basis.get(gsp)?,
                    tsk,
                    scratch,
                )?;
                next.push((2 * gsp - 1, active[i + 1].1));
                i += 2;
            } else if is_last && i > 0 {
                // Unmatched tail: assign the preceding element's degree so it can
                // pair correctly in the next round.
                let degree = next.last().map(|(degree, _)| *degree).unwrap_or(active[i].0);
                next.push((degree, active[i].1));
                i += 1;
            } else {
                next.push(active[i]);
                i += 1;
            }
        }
        active = next;
    }

    let evaluated = baby_steps.last().expect("non-empty baby step vector");
    module.ckks_copy(res, evaluated.get(), scratch)?;

    Ok(())
}

fn eval_monomial_pair<M, B, A, T, BE: Backend>(
    module: &M,
    baby_steps: &mut [B],
    low_idx: usize,
    high_idx: usize,
    xpow: &A,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSModuleAlloc<BE>,
    B: BabyStepInfos<BE>,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
{
    ensure!(low_idx != high_idx, "ckks_eval_giant_steps: baby-step pair aliases itself");
    if low_idx < high_idx {
        let (low_steps, high_steps) = baby_steps.split_at_mut(high_idx);
        eval_monomial(module, low_steps[low_idx].get(), high_steps[0].get_mut(), xpow, tsk, scratch)
    } else {
        let (high_steps, low_steps) = baby_steps.split_at_mut(low_idx);
        eval_monomial(module, low_steps[0].get(), high_steps[high_idx].get_mut(), xpow, tsk, scratch)
    }
}

fn eval_monomial<M, V, A, T, BE: Backend>(
    module: &M,
    a: &V,
    b: &mut V,
    xpow: &A,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSModuleAlloc<BE>,
    V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
{
    scratch.scope(|scratch_local| {
        let (mut a_compact, scratch_local) = scratch_local.take_compact_ckks_ciphertext_scratch(a);
        let (mut b_compact, scratch_local) = scratch_local.take_compact_ckks_ciphertext_scratch(b);
        let (mut xpow_compact, mut scratch_local) = scratch_local.take_compact_ckks_ciphertext_scratch(xpow);
        module.ckks_copy(&mut a_compact, a, &mut scratch_local)?;
        module.ckks_copy(&mut b_compact, b, &mut scratch_local)?;
        module.ckks_copy(&mut xpow_compact, xpow, &mut scratch_local)?;
        module.ckks_mul_assign(&mut b_compact, &xpow_compact, tsk, &mut scratch_local)?;
        module.ckks_add_assign(&mut b_compact, &a_compact, &mut scratch_local)?;
        module.ckks_copy(b, &b_compact, &mut scratch_local)
    })
}

fn giant_step_power(degree: usize) -> usize {
    (degree + 1).next_power_of_two()
}

impl<BE: Backend> PolynomialEvaluationDefault<BE> for poulpy_hal::layouts::Module<BE> {}
