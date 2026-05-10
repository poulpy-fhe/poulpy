use crate::api::CKKSMulAddOps;
use crate::layouts::UnnormalizedCKKSCiphertext;
use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_core::{GLWENormalize, ScratchArenaTakeCore};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{BSGSPolynomialInfos, CKKSAddOps, CKKSAddOpsUnnormalized, CKKSCopyOps, CKKSMulOps, PowerBasisHelper},
    layouts::{CKKSCiphertext, CKKSModuleAlloc, ScratchArenaTakeCKKS},
};
use anyhow::{Result, ensure};

struct BabyStep<D: poulpy_hal::layouts::Data> {
    degree: usize,
    value: CKKSCiphertext<D>,
}

pub trait PolynomialEvaluationDefault<BE: Backend> {
    fn ckks_eval_poly_const_coeffs_default<R, B, A, G, T>(
        &self,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSAddOps<BE>
            + CKKSAddOpsUnnormalized<BE>
            + CKKSMulAddOps<BE>
            + CKKSCopyOps<BE>
            + CKKSMulOps<BE>
            + GLWENormalize<BE>
            + CKKSModuleAlloc<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        B: BSGSPolynomialInfos<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
    {
        ensure!(
            poly.baby_steps() > 0,
            "ckks_eval_poly_const_coeffs: polynomial must contain at least one baby step"
        );

        let bs = poly.baby_steps();
        let mut baby_steps = Vec::with_capacity(bs);

        for i in 0..bs {
            let degree = poly.baby_degree(i);
            let value = eval_baby_step(self, degree, poly.baby_step(i), power_basis, &mut scratch.borrow())?;
            baby_steps.push(BabyStep { degree, value });
        }

        let evaluated = process_baby_steps(self, baby_steps, power_basis, tsk, &mut scratch.borrow())?;
        self.ckks_copy(res, &evaluated, scratch)
    }
}

fn compact_ct<M, S, BE: Backend>(module: &M, src: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<CKKSCiphertext<BE::OwnedBuf>>
where
    M: CKKSCopyOps<BE> + CKKSModuleAlloc<BE>,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE>,
{
    let mut compact = module.ckks_ciphertext_alloc(src.base2k(), src.effective_k().into());
    module.ckks_copy(&mut compact, src, scratch)?;
    Ok(compact)
}

fn eval_baby_step<M, C, A, G, BE: Backend>(
    module: &M,
    degree: usize,
    coeffs: &C,
    power_basis: &G,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<CKKSCiphertext<<BE as Backend>::OwnedBuf>>
where
    M: CKKSMulAddOps<BE>
        + CKKSAddOpsUnnormalized<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + GLWENormalize<BE>
        + CKKSModuleAlloc<BE>,
    C: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    G: PowerBasisHelper<BE, A>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
{
    let x = power_basis.get(1)?;
    let mut res = module.ckks_ciphertext_alloc_from_infos(x);
    res.set_meta(x.meta());

    let mut res_unormalized = UnnormalizedCKKSCiphertext::new(res);

    module.ckks_add_pt_const_assign_unnormalized(&mut res_unormalized, 0, coeffs, 0, scratch)?;

    for i in (1..degree + 1).rev() {
        let xpow = power_basis.get(i)?;
        module.ckks_mul_add_pt_const_into_unnormalized(&mut res_unormalized, xpow, coeffs, i, &mut scratch.borrow())?;
    }

    let res = res_unormalized.normalize(module, scratch);

    compact_ct(module, &res, scratch)
}

fn process_baby_steps<M, A, G, T, BE: Backend>(
    module: &M,
    mut baby_steps: Vec<BabyStep<BE::OwnedBuf>>,
    power_basis: &G,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<CKKSCiphertext<BE::OwnedBuf>>
where
    M: CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSModuleAlloc<BE>,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    G: PowerBasisHelper<BE, A>,
    T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    ensure!(
        !baby_steps.is_empty(),
        "ckks_eval_poly_const_coeffs: polynomial must contain at least one baby step"
    );

    // Each round pairs consecutive equal-degree steps: combined = high * X^gsp + low.
    // Unmatched tails inherit the degree of their predecessor so they pair in the next round.
    while baby_steps.len() > 1 {
        let mut i = 0;
        while i < baby_steps.len() {
            let is_last = i + 1 == baby_steps.len();
            if !is_last && baby_steps[i].degree == baby_steps[i + 1].degree {
                let gsp = giant_step_power(baby_steps[i].degree);
                // remove(i) shifts baby_steps[i+1] down to baby_steps[i].
                let even = baby_steps.remove(i);
                eval_monomial(
                    module,
                    &even.value,
                    &mut baby_steps[i].value,
                    power_basis.get(gsp)?,
                    tsk,
                    scratch,
                )?;
                let compacted = compact_ct(module, &baby_steps[i].value, scratch)?;
                baby_steps[i].value = compacted;
                baby_steps[i].degree = 2 * gsp - 1;
            } else if is_last && i > 0 {
                // Unmatched tail: assign the preceding element's degree so it can
                // pair correctly in the next round.
                baby_steps[i].degree = baby_steps[i - 1].degree;
            }
            i += 1;
        }
    }

    Ok(baby_steps.pop().expect("non-empty baby step vector").value)
}

fn eval_monomial<M, A, T, BE: Backend>(
    module: &M,
    a: &CKKSCiphertext<BE::OwnedBuf>,
    b: &mut CKKSCiphertext<BE::OwnedBuf>,
    xpow: &A,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSModuleAlloc<BE>,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    let xpow_layout = GLWELayout {
        n: xpow.n(),
        base2k: xpow.base2k(),
        k: xpow.effective_k().into(),
        rank: xpow.rank(),
    };

    scratch.scope(|scratch_local| {
        let (mut xpow_compact, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&xpow_layout, xpow.meta());
        module.ckks_copy(&mut xpow_compact, xpow, &mut scratch_local)?;
        module.ckks_mul_assign(b, &xpow_compact, tsk, &mut scratch_local)?;
        module.ckks_add_assign(b, a, &mut scratch_local)
    })
}

fn giant_step_power(degree: usize) -> usize {
    (degree + 1).next_power_of_two()
}

impl<BE: Backend> PolynomialEvaluationDefault<BE> for poulpy_hal::layouts::Module<BE> {}
