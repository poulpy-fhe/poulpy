//! Two-branch lockstep driver for [`CKKSEvalModOps::ckks_eval_mod_pair`].
//!
//! The two EvalMod DAGs of a bootstrap (real and imaginary halves) are
//! independent and, because they share `params`, structurally identical. This
//! driver advances both on one host thread by dependency frontier and hands
//! every tensor-product frontier to the CKKS batch operations, so a backend
//! sees `B = 2` (one ready operation per branch) or `B = 4` (two sibling
//! products per branch) instead of a stream of scalar multiplies.
//!
//! Independence is what makes this exact: interleaving cannot change either
//! branch's own operation sequence, so every result is byte-for-byte what two
//! sequential [`CKKSEvalModOps::ckks_eval_mod`] calls produce. The cost is a
//! doubled working set: both power bases and both baby-step vectors are live at
//! once, which
//! [`ckks_eval_mod_pair_lockstep_tmp_bytes_default`] accounts for.
//!
//! The frontier sequence is derived from the plan alone, so the scratch query
//! replays it without any ciphertext and prices every frontier through the
//! public batch queries; see [`lockstep_frontier_shapes`].
//!
//! [`CKKSEvalModOps::ckks_eval_mod_pair`]: crate::api::CKKSEvalModOps::ckks_eval_mod_pair
//! [`CKKSEvalModOps::ckks_eval_mod`]: crate::api::CKKSEvalModOps::ckks_eval_mod

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    GLWECopy, GLWEPolynomialEvaluation, GLWEZero,
    default::polynomial_evaluation::giant_step_schedule,
    layouts::{
        BSGSMeta, BSGSPolynomial, BSGSPolynomialInfos, Base2K, Degree, GGLWEInfos, GLWEInfos, GLWELayout, GLWETensorKeyPrepared,
        GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, LWEInfos, Parity, PowerBasis, PowerBasisHelper, Rank, SetBSGSMeta,
        TorusPrecision, prepared::GLWETensorKeyPreparedToBackendRef, split_degree,
    },
};
use poulpy_hal::api::CnvPVecBytesOf;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};
use std::collections::{HashMap, HashSet};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos,
    api::{
        Basis, CKKSAddOps, CKKSCopyOps, CKKSImagOps, CKKSMulAddOps, CKKSMulIntoItem, CKKSMulOps, CKKSPow2Ops,
        CKKSPreparedMulAssignItem, CKKSSquareAssignItem, CKKSSquareIntoItem, CKKSSubOps, PolynomialInputTransform,
    },
    default::{carry_verb::ckks_one_pt, polynomial_evaluation::CKKSBSGSOps},
    layouts::{
        CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPreparedRight, CKKSPreparedRightLayout,
        eval_mod::{EvalMod, EvalModBsgs},
    },
    polynomial::ComplexBSGSPolynomial,
    power_basis::mul_ct_k,
};

/// Ciphertext type every lockstep intermediate uses. Both branches share it,
/// which is what lets a frontier be one slice.
type Ct<BE> = CKKSCiphertextOwned<BE>;

/// Operation bundle the lockstep driver needs, gathered so the public helpers
/// state it once.
pub trait CKKSLockstepOps<BE: Backend>:
    CKKSAddOps<BE>
    + CKKSSubOps<BE>
    + CKKSMulOps<BE>
    + CKKSMulAddOps<BE>
    + CKKSCopyOps<BE>
    + CKKSModuleAlloc<BE>
    + CKKSPow2Ops<BE>
    + CKKSImagOps<BE>
    + GLWECopy<BE>
    + GLWEZero<BE>
    + GLWEPolynomialEvaluation<BE>
{
}

impl<BE: Backend, M> CKKSLockstepOps<BE> for M where
    M: CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSMulAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSPow2Ops<BE>
        + CKKSImagOps<BE>
        + GLWECopy<BE>
        + GLWEZero<BE>
        + GLWEPolynomialEvaluation<BE>
{
}

/// A baby step of one branch: its degree and its accumulator.
struct BabyStepPair<BE: Backend> {
    degree: usize,
    value: Ct<BE>,
}

/// One product of a power-basis generation plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PowerProduct {
    /// Power this product stores.
    n: usize,
    /// Split operands; `a == b` is a self-product and goes to the square batch.
    a: usize,
    b: usize,
    /// Chebyshev tail `2·T_a·T_b − T_c`; `None` subtracts one instead.
    c: Option<usize>,
}

/// The products a power-basis stage issues, in order.
///
/// Derived from the polynomial alone, so the executor and the scratch replay
/// walk the same frontier sequence and cannot drift.
#[derive(Clone, Debug)]
struct PowerPlan {
    basis: Basis,
    products: Vec<PowerProduct>,
}

fn plan_power(basis: Basis, n: usize, seen: &mut HashSet<usize>, out: &mut Vec<PowerProduct>) -> Result<()> {
    if seen.contains(&n) {
        return Ok(());
    }
    ckks_ensure!(
        n >= 2,
        "power_basis_plan: n={n} < 2; the degree-one power is provided at construction"
    );

    let (a, b) = split_degree(n);
    plan_power(basis, a, seen, out)?;
    plan_power(basis, b, seen, out)?;
    let c = match basis {
        Basis::Monomial => None,
        Basis::Chebyshev => {
            let c = a.abs_diff(b);
            if c != 0 {
                plan_power(basis, c, seen, out)?;
                Some(c)
            } else {
                None
            }
        }
    };
    seen.insert(n);
    out.push(PowerProduct { n, a, b, c });
    Ok(())
}

/// [`crate::power_basis::PowerBasisGen::populate`]'s generation order.
fn power_basis_plan(basis: Basis, degree: usize, log_split: usize, parity: Parity) -> Result<PowerPlan> {
    ckks_ensure!(degree >= 1, "power_basis_plan: degree must be >= 1");

    let log_degree = (usize::BITS - degree.leading_zeros()) as usize;
    let largest_pow2 = 1usize << (log_degree - 1);
    let base = 1usize << log_split;
    let mut seen: HashSet<usize> = HashSet::from([1usize]);
    let mut products: Vec<PowerProduct> = Vec::new();

    if largest_pow2 >= 2 {
        plan_power(basis, largest_pow2, &mut seen, &mut products)?;
    }
    let baby_limit = base.min(degree + 1);
    match parity {
        Parity::Even => {
            for i in (4..baby_limit).step_by(2) {
                plan_power(basis, i, &mut seen, &mut products)?;
            }
        }
        Parity::Odd => {
            for i in (3..baby_limit).step_by(2) {
                plan_power(basis, i, &mut seen, &mut products)?;
            }
        }
        Parity::Full => {
            for i in (3..baby_limit).rev() {
                plan_power(basis, i, &mut seen, &mut products)?;
            }
        }
    }
    Ok(PowerPlan { basis, products })
}

/// The single degree-two product a polynomial input transform performs.
fn transform_plan(transform: PolynomialInputTransform) -> Option<PowerPlan> {
    let basis = match transform {
        PolynomialInputTransform::Identity => return None,
        PolynomialInputTransform::Square | PolynomialInputTransform::SquareTimesInput => Basis::Monomial,
        PolynomialInputTransform::ChebyshevT2 | PolynomialInputTransform::ChebyshevT2TimesInput => Basis::Chebyshev,
    };
    Some(PowerPlan {
        basis,
        products: vec![PowerProduct {
            n: 2,
            a: 1,
            b: 1,
            c: None,
        }],
    })
}

/// Runs a [`PowerPlan`] on both branches, each product one two-item frontier.
fn execute_power_plan<BE>(
    plan: &PowerPlan,
    basis: &mut [PowerBasis<Ct<BE>>; 2],
    module: &Module<BE>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLockstepOps<BE>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    Ct<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    for product in &plan.products {
        let [basis_0, basis_1] = &mut *basis;
        let (mut r0, mut r1) = {
            let a0 = basis_0.get_stored(product.a).expect("plan generates operands first");
            let b0 = basis_0.get_stored(product.b).expect("plan generates operands first");
            let a1 = basis_1.get_stored(product.a).expect("plan generates operands first");
            let b1 = basis_1.get_stored(product.b).expect("plan generates operands first");
            (
                module.ckks_ciphertext_alloc(a0.base2k(), mul_ct_k(a0, b0)?.into()),
                module.ckks_ciphertext_alloc(a1.base2k(), mul_ct_k(a1, b1)?.into()),
            )
        };

        {
            let a0 = basis_0.get_stored(product.a).expect("stored above");
            let a1 = basis_1.get_stored(product.a).expect("stored above");
            if product.a == product.b {
                let mut items = [
                    CKKSSquareIntoItem { dst: &mut r0, a: a0 },
                    CKKSSquareIntoItem { dst: &mut r1, a: a1 },
                ];
                module.ckks_square_into_batch(&mut items, tsk, scratch)?;
            } else {
                let b0 = basis_0.get_stored(product.b).expect("stored above");
                let b1 = basis_1.get_stored(product.b).expect("stored above");
                let mut items = [
                    CKKSMulIntoItem {
                        dst: &mut r0,
                        a: a0,
                        b: b0,
                    },
                    CKKSMulIntoItem {
                        dst: &mut r1,
                        a: a1,
                        b: b1,
                    },
                ];
                module.ckks_mul_into_batch(&mut items, tsk, scratch)?;
            }
        }

        if plan.basis == Basis::Chebyshev {
            // `2·T_a·T_b − T_c`, no tensor product.
            for (r, source) in [(&mut r0, &*basis_0), (&mut r1, &*basis_1)] {
                module.ckks_mul_pow2_assign(r, 1, scratch)?;
                match product.c {
                    None => {
                        let one = ckks_one_pt(module, r.base2k())?;
                        module.ckks_sub_pt_const_assign(r, 0, &one, 0, scratch)?;
                    }
                    Some(c) => {
                        let c_val = source.get_stored(c).expect("plan generates T_c first");
                        module.ckks_sub_assign(r, c_val, scratch)?;
                    }
                }
            }
        }

        basis_0.set_power(product.n, r0);
        basis_1.set_power(product.n, r1);
    }
    Ok(())
}

/// [`polynomial_input`](crate::oep) for both branches: the working copy, then
/// the transform's square as one two-item frontier.
fn polynomial_input_pair<BE>(
    srcs: [&Ct<BE>; 2],
    transform: PolynomialInputTransform,
    module: &Module<BE>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<[Ct<BE>; 2]>
where
    BE: Backend,
    Module<BE>: CKKSLockstepOps<BE>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    Ct<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let mut inputs: [Ct<BE>; 2] = [
        module.ckks_ciphertext_alloc_from_infos(srcs[0]),
        module.ckks_ciphertext_alloc_from_infos(srcs[1]),
    ];
    for branch in 0..2 {
        module.ckks_copy(&mut inputs[branch], srcs[branch], scratch)?;
    }

    let [x0, x1] = inputs;
    let Some(plan) = transform_plan(transform) else {
        return Ok([x0, x1]);
    };
    let mut basis = [PowerBasis::new(plan.basis, x0), PowerBasis::new(plan.basis, x1)];
    execute_power_plan(&plan, &mut basis, module, tsk, scratch)?;
    let [b0, b1] = &mut basis;
    Ok([
        b0.take_power(2).expect("the transform plan stores the degree-two power"),
        b1.take_power(2).expect("the transform plan stores the degree-two power"),
    ])
}

/// Giant-step fold of both branches, level by level.
///
/// Each level's pairs are independent within a branch and across branches, so
/// the whole level is one frontier. Items are interleaved (branch 0, branch 1,
/// branch 0, ...) so a backend that consumes them two at a time gets the
/// corresponding operation from each branch, and four at a time gets two
/// sibling pairs from each.
fn eval_giant_steps_pair<BE>(
    steps: &mut [Vec<BabyStepPair<BE>>; 2],
    basis: &[PowerBasis<Ct<BE>>; 2],
    module: &Module<BE>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLockstepOps<BE>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    Ct<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    ckks_ensure!(
        !steps[0].is_empty() && !steps[1].is_empty(),
        "eval_giant_steps_pair: each branch must contain at least one baby step"
    );

    let levels: [Vec<Vec<poulpy_core::default::polynomial_evaluation::GiantStepPair>>; 2] = [
        giant_step_schedule(&steps[0].iter().map(|s| s.degree).collect::<Vec<_>>()),
        giant_step_schedule(&steps[1].iter().map(|s| s.degree).collect::<Vec<_>>()),
    ];
    ckks_ensure!(
        levels[0].len() == levels[1].len(),
        "eval_giant_steps_pair: branches have different giant-step depths ({} vs {})",
        levels[0].len(),
        levels[1].len()
    );

    for (pairs_0, pairs_1) in levels[0].iter().zip(levels[1].iter()) {
        let pairs = [pairs_0, pairs_1];
        for branch_pairs in pairs {
            for pair in branch_pairs.iter() {
                ckks_ensure!(pair.low != pair.high, "eval_giant_steps_pair: baby-step pair aliases itself");
            }
        }

        // One hoisted `X^{gsp}` per distinct power per branch.
        let mut prepared: [Vec<(usize, CKKSPreparedRight<BE>)>; 2] = [Vec::new(), Vec::new()];
        for branch in 0..2 {
            for pair in pairs[branch].iter() {
                if !prepared[branch].iter().any(|(g, _)| *g == pair.gsp) {
                    let source = basis[branch].get(pair.gsp)?;
                    prepared[branch].push((pair.gsp, module.ckks_prepare_right(source, scratch)?));
                }
            }
        }

        let (steps_0, steps_1) = steps.split_at_mut(1);
        let mut ordered: [Vec<(usize, &mut Ct<BE>)>; 2] = [
            frontier_destinations(&mut steps_0[0], pairs[0]),
            frontier_destinations(&mut steps_1[0], pairs[1]),
        ];

        let mut frontier: Vec<CKKSPreparedMulAssignItem<&mut Ct<BE>, &CKKSPreparedRight<BE>>> = Vec::new();
        let [ordered_0, ordered_1] = &mut ordered;
        let mut cursor_0 = ordered_0.drain(..);
        let mut cursor_1 = ordered_1.drain(..);
        loop {
            let mut progressed = false;
            for (branch, next) in [cursor_0.next(), cursor_1.next()].into_iter().enumerate() {
                if let Some((position, dst)) = next {
                    let gsp = pairs[branch][position].gsp;
                    let prep = &prepared[branch]
                        .iter()
                        .find(|(g, _)| *g == gsp)
                        .expect("every pair's giant-step power was prepared")
                        .1;
                    frontier.push(CKKSPreparedMulAssignItem { dst, prepared: prep });
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }
        drop(cursor_0);
        drop(cursor_1);

        module.ckks_mul_prepared_assign_batch(&mut frontier, tsk, scratch)?;
        drop(frontier);

        for branch in 0..2 {
            for pair in pairs[branch].iter() {
                let branch_steps = &mut steps[branch];
                let (a, b) = if pair.low < pair.high {
                    let (low, high) = branch_steps.split_at_mut(pair.high);
                    (&low[pair.low].value, &mut high[0].value)
                } else {
                    let (high, low) = branch_steps.split_at_mut(pair.low);
                    (&low[0].value, &mut high[pair.high].value)
                };
                module.ckks_add_assign(b, a, scratch)?;
            }
        }
    }

    Ok(())
}

/// Distinct `&mut` destinations of one branch's level, in pair order.
fn frontier_destinations<'a, BE: Backend>(
    steps: &'a mut [BabyStepPair<BE>],
    pairs: &[poulpy_core::default::polynomial_evaluation::GiantStepPair],
) -> Vec<(usize, &'a mut Ct<BE>)> {
    // Each baby step is the high operand of at most one pair, so the mutable
    // borrows are disjoint; `rank` restores pair order after the index-ordered
    // `iter_mut`.
    let mut rank: Vec<usize> = vec![usize::MAX; steps.len()];
    for (position, pair) in pairs.iter().enumerate() {
        rank[pair.high] = position;
    }
    let mut out: Vec<(usize, &mut Ct<BE>)> = steps
        .iter_mut()
        .enumerate()
        .filter(|(index, _)| rank[*index] != usize::MAX)
        .map(|(index, step)| (rank[index], &mut step.value))
        .collect();
    out.sort_by_key(|(position, _)| *position);
    out
}

/// Real BSGS polynomial over both branches: baby steps per branch (ct×pt only),
/// then one lockstep giant-step fold.
#[allow(clippy::too_many_arguments)]
fn eval_poly_real_pair<BE, B>(
    acc: &mut [Ct<BE>; 2],
    poly: &B,
    basis: &[PowerBasis<Ct<BE>>; 2],
    module: &Module<BE>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLockstepOps<BE>,
    B: BSGSPolynomialInfos<BE>,
    B::Coeffs: CKKSCtBounds,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    Ct<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let n_baby = poly.baby_steps();
    ckks_ensure!(
        n_baby > 0,
        "eval_poly_real_pair: polynomial must contain at least one baby step"
    );
    ckks_ensure!(
        poly.basis() == basis[0].basis(),
        "eval_poly_real_pair: polynomial basis {:?} does not match power basis {:?}",
        poly.basis(),
        basis[0].basis()
    );

    let last_coeffs = poly.baby_step(n_baby - 1);
    let trailing_const_only = n_baby >= 2 && last_coeffs.n().as_usize() == 1;
    let fold_power = poly.degree();
    let can_fold = trailing_const_only && basis[0].has_power(fold_power);
    let n_to_process = if can_fold { n_baby - 1 } else { n_baby };
    let parity = poly.parity();

    let mut steps: [Vec<BabyStepPair<BE>>; 2] = [Vec::with_capacity(n_to_process), Vec::with_capacity(n_to_process)];
    for branch in 0..2 {
        let x = basis[branch].get(1)?;
        for i in 0..n_to_process {
            let coeffs = poly.baby_step(i);
            let degree = coeffs.n().as_usize() - 1;
            let mut value = module.ckks_ciphertext_alloc_from_infos(x);
            value.set_meta(x.meta());
            module.glwe_eval_baby_step::<_, _, B::Coeffs, Ct<BE>, _>(
                &CKKSBSGSOps,
                &mut value,
                parity,
                coeffs,
                &basis[branch],
                &mut scratch.borrow(),
            )?;
            steps[branch].push(BabyStepPair { degree, value });
        }
    }

    eval_giant_steps_pair(&mut steps, basis, module, tsk, scratch)?;

    for branch in 0..2 {
        let evaluated = steps[branch].last().expect("non-empty baby step vector");
        module.ckks_copy(&mut acc[branch], &evaluated.value, &mut scratch.borrow())?;
    }

    if can_fold {
        for branch in 0..2 {
            let xpow = basis[branch].get(fold_power)?;
            module.ckks_mul_add_pt_const_into(&mut acc[branch], xpow, last_coeffs, 0, scratch)?;
        }
    }
    Ok(())
}

/// Complex BSGS polynomial over both branches. Same shape as
/// [`eval_poly_real_pair`]: only the giant-step fold is a frontier.
#[allow(clippy::too_many_arguments)]
fn eval_poly_complex_pair<BE, C>(
    acc: &mut [Ct<BE>; 2],
    poly: &ComplexBSGSPolynomial<C>,
    basis: &[PowerBasis<Ct<BE>>; 2],
    module: &Module<BE>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLockstepOps<BE>,
    C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    Ct<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let poly_re = &poly.re;
    let poly_im = &poly.im;
    let n_baby = BSGSPolynomialInfos::<BE>::baby_steps(poly_re);
    ckks_ensure!(
        n_baby > 0,
        "eval_poly_complex_pair: polynomial must contain at least one baby step"
    );
    ckks_ensure!(
        BSGSPolynomialInfos::<BE>::baby_steps(poly_im) == n_baby,
        "eval_poly_complex_pair: real/imag baby-step schedules differ"
    );
    ckks_ensure!(
        BSGSPolynomialInfos::<BE>::basis(poly_re) == basis[0].basis(),
        "eval_poly_complex_pair: polynomial basis does not match power basis"
    );

    let last_re = BSGSPolynomialInfos::<BE>::baby_step(poly_re, n_baby - 1);
    let last_im = BSGSPolynomialInfos::<BE>::baby_step(poly_im, n_baby - 1);
    let trailing_const_only = n_baby >= 2 && last_re.n().as_usize() == 1;
    let fold_power = BSGSPolynomialInfos::<BE>::degree(poly_re);
    let can_fold = trailing_const_only && basis[0].has_power(fold_power);
    let n_to_process = if can_fold { n_baby - 1 } else { n_baby };
    let parity = BSGSPolynomialInfos::<BE>::parity(poly_re);

    let mut steps: [Vec<BabyStepPair<BE>>; 2] = [Vec::with_capacity(n_to_process), Vec::with_capacity(n_to_process)];
    for branch in 0..2 {
        let x = basis[branch].get(1)?;
        for i in 0..n_to_process {
            let re_coeffs = BSGSPolynomialInfos::<BE>::baby_step(poly_re, i);
            let im_coeffs = BSGSPolynomialInfos::<BE>::baby_step(poly_im, i);
            ckks_ensure!(
                im_coeffs.n() == re_coeffs.n(),
                "eval_poly_complex_pair: real/imag baby-step {i} lengths differ"
            );
            let degree = re_coeffs.n().as_usize() - 1;

            let mut value = module.ckks_ciphertext_alloc_from_infos(x);
            value.set_meta(x.meta());
            module.glwe_eval_baby_step::<_, _, C, Ct<BE>, _>(
                &CKKSBSGSOps,
                &mut value,
                parity,
                re_coeffs,
                &basis[branch],
                &mut scratch.borrow(),
            )?;

            let mut im_value = module.ckks_ciphertext_alloc_from_infos(x);
            im_value.set_meta(x.meta());
            module.glwe_eval_baby_step::<_, _, C, Ct<BE>, _>(
                &CKKSBSGSOps,
                &mut im_value,
                parity,
                im_coeffs,
                &basis[branch],
                &mut scratch.borrow(),
            )?;
            module.ckks_mul_i_assign(&mut im_value, &mut scratch.borrow())?;
            module.ckks_add_assign(&mut value, &im_value, &mut scratch.borrow())?;

            steps[branch].push(BabyStepPair { degree, value });
        }
    }

    eval_giant_steps_pair(&mut steps, basis, module, tsk, scratch)?;

    for branch in 0..2 {
        let evaluated = steps[branch].last().expect("non-empty baby step vector");
        module.ckks_copy(&mut acc[branch], &evaluated.value, &mut scratch.borrow())?;
    }

    if can_fold {
        for branch in 0..2 {
            let xpow = basis[branch].get(fold_power)?;
            module.ckks_mul_add_pt_const_into(&mut acc[branch], xpow, last_re, 0, scratch)?;
            let mut im_fold = module.ckks_ciphertext_alloc_from_infos(&acc[branch]);
            module.ckks_mul_pt_const_into(&mut im_fold, xpow, last_im, 0, scratch)?;
            module.ckks_mul_i_assign(&mut im_fold, scratch)?;
            module.ckks_add_assign(&mut acc[branch], &im_fold, scratch)?;
        }
    }
    Ok(())
}

/// Borrowed view of a stage's polynomial: the base stage borrows from
/// [`EvalMod::f_mod_bsgs`], the inverse stage from `f_mod_inv_bsgs`.
enum StageBsgs<'a, P> {
    Real(&'a BSGSPolynomial<P>),
    Complex(&'a ComplexBSGSPolynomial<P>),
}

impl<'a, P> From<&'a EvalModBsgs<P>> for StageBsgs<'a, P> {
    fn from(bsgs: &'a EvalModBsgs<P>) -> Self {
        match bsgs {
            EvalModBsgs::Real(poly) => StageBsgs::Real(poly),
            EvalModBsgs::Complex(poly) => StageBsgs::Complex(poly),
        }
    }
}

/// The BSGS transform of `bsgs`, checked to agree across a complex pair.
fn stage_input_transform<BE: Backend, P>(bsgs: &StageBsgs<'_, P>) -> Result<PolynomialInputTransform>
where
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
{
    match bsgs {
        StageBsgs::Real(poly) => Ok(BSGSPolynomialInfos::<BE>::input_transform(*poly)),
        StageBsgs::Complex(poly) => {
            let transform = BSGSPolynomialInfos::<BE>::input_transform(&poly.re);
            ckks_ensure!(
                transform == BSGSPolynomialInfos::<BE>::input_transform(&poly.im),
                "eval_stage_pair: real and imaginary input transforms differ"
            );
            Ok(transform)
        }
    }
}

/// Power basis + baby steps + giant steps over both branches, from ready
/// power-basis roots.
fn eval_stage_from_roots_pair<BE, P>(
    acc: &mut [Ct<BE>; 2],
    roots: [Ct<BE>; 2],
    bsgs: &StageBsgs<'_, P>,
    module: &Module<BE>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLockstepOps<BE>,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    Ct<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let [x0, x1] = roots;
    let (poly_basis, degree, log_split, parity) = match bsgs {
        StageBsgs::Real(poly) => (
            BSGSPolynomialInfos::<BE>::basis(*poly),
            BSGSPolynomialInfos::<BE>::degree(*poly),
            BSGSPolynomialInfos::<BE>::log_split(*poly),
            BSGSPolynomialInfos::<BE>::parity(*poly),
        ),
        StageBsgs::Complex(poly) => (
            BSGSPolynomialInfos::<BE>::basis(&poly.re),
            BSGSPolynomialInfos::<BE>::degree(&poly.re),
            BSGSPolynomialInfos::<BE>::log_split(&poly.re),
            BSGSPolynomialInfos::<BE>::parity(&poly.re),
        ),
    };
    let plan = power_basis_plan(poly_basis, degree, log_split, parity)?;
    let mut basis = [PowerBasis::new(poly_basis, x0), PowerBasis::new(poly_basis, x1)];
    execute_power_plan(&plan, &mut basis, module, tsk, scratch)?;

    match bsgs {
        StageBsgs::Real(poly) => eval_poly_real_pair(acc, *poly, &basis, module, tsk, scratch),
        StageBsgs::Complex(poly) => eval_poly_complex_pair(acc, poly, &basis, module, tsk, scratch),
    }
}

/// One full BSGS stage over both branches: input transform, evaluation, and the
/// `TimesInput` tail.
fn eval_stage_pair<BE, P>(
    acc: &mut [Ct<BE>; 2],
    inputs: [Ct<BE>; 2],
    bsgs: &StageBsgs<'_, P>,
    module: &Module<BE>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLockstepOps<BE>,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    Ct<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let transform = stage_input_transform::<BE, P>(bsgs)?;
    let roots = polynomial_input_pair([&inputs[0], &inputs[1]], transform, module, tsk, scratch)?;
    eval_stage_from_roots_pair(acc, roots, bsgs, module, tsk, scratch)?;

    if matches!(
        transform,
        PolynomialInputTransform::SquareTimesInput | PolynomialInputTransform::ChebyshevT2TimesInput
    ) {
        // `acc *= input`, as one frontier. `ckks_mul_prepared_assign` against a
        // freshly prepared `input` is the same operation as `ckks_mul_assign`
        // against `input`: same parameters, same tensor width, same stamp (see
        // `test_mul_prepared_assign_matches_mul_assign`).
        let prepared = [
            module.ckks_prepare_right(&inputs[0], scratch)?,
            module.ckks_prepare_right(&inputs[1], scratch)?,
        ];
        let (first, second) = acc.split_at_mut(1);
        let mut items = [
            CKKSPreparedMulAssignItem {
                dst: &mut first[0],
                prepared: &prepared[0],
            },
            CKKSPreparedMulAssignItem {
                dst: &mut second[0],
                prepared: &prepared[1],
            },
        ];
        module.ckks_mul_prepared_assign_batch(&mut items, tsk, scratch)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scratch replay.
//
// The driver's frontier sequence and every item's layout are functions of
// `params` alone, so the query replays the whole metadata DAG on layout-only
// stand-ins and hands each frontier's *actual* item layouts to the matching
// public batch scratch query. A backend that picks a fused path for one layout
// and a fallback for another is therefore priced on the layouts it will really
// see.
//
// The mirror below reproduces the metadata rule of every step the driver takes.
// It is pinned end to end: the lockstep acceptance test records the layouts the
// backend actually receives and compares them against the ones priced here, so
// a rule that drifts fails the test rather than silently under-reporting.
// ---------------------------------------------------------------------------

/// Layout-only stand-in for a lockstep ciphertext: everything a batch scratch
/// query reads, and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CtLayout {
    n: Degree,
    base2k: Base2K,
    rank: Rank,
    /// Allocation width, fixed at construction.
    max_size: usize,
    /// Effective width `log_delta + log_budget`.
    k: TorusPrecision,
    meta: CKKSMeta,
}

impl LWEInfos for CtLayout {
    fn n(&self) -> Degree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn max_size(&self) -> usize {
        self.max_size
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl GLWEInfos for CtLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl CKKSInfos for CtLayout {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }
}

impl SetCKKSInfos for CtLayout {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.meta = meta;
    }

    fn set_k(&mut self, k: TorusPrecision) {
        self.k = k;
    }
}

impl CtLayout {
    /// `ckks_ciphertext_alloc(base2k, k)`: default metadata, buffer sized to `k`.
    fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        Self {
            n,
            base2k,
            rank,
            max_size: k.as_usize().div_ceil(base2k.as_usize()),
            k,
            meta: CKKSMeta::default(),
        }
    }

    /// `ckks_ciphertext_alloc_from_infos(infos)`: `glwe_alloc_from_infos` sizes
    /// the buffer from `infos.k()`, not from its capacity.
    fn alloc_from_infos<A: GLWEInfos + CKKSInfos>(infos: &A) -> Self {
        let mut out = Self::alloc(infos.n(), infos.base2k(), infos.k(), infos.rank());
        out.meta = infos.meta();
        out
    }

    /// `ckks_copy_default`: both branches stamp `src`'s metadata and charge the
    /// unary offset against its budget.
    fn copy_from(&mut self, src: &Self) {
        let offset = crate::ckks_offset_unary(self, src);
        self.meta = src.meta;
        self.set_log_budget(src.log_budget().saturating_sub(offset));
    }

    /// The metadata half of `ckks_{add,sub}_assign_unnormalized_default`.
    fn add_assign_from(&mut self, a: &Self) {
        self.set_log_budget(self.log_budget().min(a.log_budget()));
        self.set_log_delta(self.log_delta().min(a.log_delta()));
        self.set_log_sparsity(self.log_sparsity().min(a.log_sparsity()));
        self.set_slots(self.slots().join(a.slots()));
    }

    /// `CKKSBSGSOps::init_accumulator`.
    fn init_accumulator_from(&mut self, seed: &Self) {
        self.set_log_delta(seed.log_delta());
        self.set_log_budget(seed.log_budget());
        self.set_slots(seed.slots());
    }

    /// `stamp_meta`, the result metadata of every tensor multiply.
    fn stamp(&mut self, log_budget: usize, log_delta: usize, log_sparsity: Option<usize>, slots: Option<crate::SlotsKind>) {
        self.set_log_budget(log_budget);
        self.set_log_delta(log_delta);
        if let Some(log_sparsity) = log_sparsity {
            self.set_log_sparsity(log_sparsity);
        }
        if let Some(slots) = slots {
            self.set_slots(slots);
        }
    }

    /// `ckks_{add,sub}_pt_const_assign` at coefficient zero: the constant is
    /// real, so only the slot kind moves.
    fn join_real_slots(&mut self) {
        self.set_slots(self.slots().join(crate::SlotsKind::Real));
    }

    /// `ckks_mul_i_assign`.
    fn mul_i(&mut self) {
        self.set_slots(crate::SlotsKind::Complex);
    }

    /// The ordered `ckks_mul_add_pt_consts_into` chain's final state.
    fn mul_add_pt_consts<P>(&mut self, terms: &[(&Self, usize)], coeffs: &P) -> Result<()>
    where
        P: IntPolyInfos + CKKSCtBounds,
    {
        let plans = crate::default::mul::ckks_mul_add_pt_consts_plan(self, terms, coeffs)?;
        if let Some(last) = plans.last() {
            self.set_meta(last.dst_meta);
            self.set_log_budget(last.dst_log_budget);
        }
        Ok(())
    }
}

/// One frontier, with the exact layout of every item.
#[derive(Clone, Debug)]
enum Frontier {
    MulInto(Vec<(CtLayout, CtLayout, CtLayout)>),
    SquareInto(Vec<(CtLayout, CtLayout)>),
    SquareAssign(Vec<CtLayout>),
    PreparedAssign(Vec<(CtLayout, CKKSPreparedRightLayout)>),
}

impl Frontier {
    fn kind(&self) -> &'static str {
        match self {
            Frontier::MulInto(_) => "mul_into",
            Frontier::SquareInto(_) => "square_into",
            Frontier::SquareAssign(_) => "square_assign",
            Frontier::PreparedAssign(_) => "prepared_assign",
        }
    }

    /// `(destination k, destination capacity, left operand k, right operand
    /// limbs)` per item: the layout quantities the batch queries read.
    fn item_shapes(&self) -> Vec<(u32, usize, u32, usize)> {
        match self {
            Frontier::MulInto(items) => items
                .iter()
                .map(|(res, a, b)| (res.k.as_u32(), res.max_size, a.k.as_u32(), b.size()))
                .collect(),
            Frontier::SquareInto(items) => items
                .iter()
                .map(|(res, a)| (res.k.as_u32(), res.max_size, a.k.as_u32(), a.size()))
                .collect(),
            Frontier::SquareAssign(items) => items
                .iter()
                .map(|res| (res.k.as_u32(), res.max_size, res.k.as_u32(), res.size()))
                .collect(),
            Frontier::PreparedAssign(items) => items
                .iter()
                .map(|(res, prepared)| (res.k.as_u32(), res.max_size, res.k.as_u32(), prepared.size))
                .collect(),
        }
    }

    /// The advertised scratch of this frontier, queried through the public
    /// batch API with the item layouts the driver will really pass.
    fn tmp_bytes<BE, T>(&self, module: &Module<BE>, tsk: &T) -> usize
    where
        BE: Backend,
        Module<BE>: CKKSMulOps<BE>,
        T: GGLWEInfos,
    {
        match self {
            Frontier::MulInto(items) => {
                let batch: Vec<CKKSMulIntoItem<&CtLayout, &CtLayout, &CtLayout>> =
                    items.iter().map(|(res, a, b)| CKKSMulIntoItem { dst: res, a, b }).collect();
                module.ckks_mul_into_batch_tmp_bytes(&batch, tsk)
            }
            Frontier::SquareInto(items) => {
                let batch: Vec<CKKSSquareIntoItem<&CtLayout, &CtLayout>> =
                    items.iter().map(|(res, a)| CKKSSquareIntoItem { dst: res, a }).collect();
                module.ckks_square_into_batch_tmp_bytes(&batch, tsk)
            }
            Frontier::SquareAssign(items) => {
                let batch: Vec<CKKSSquareAssignItem<&CtLayout>> =
                    items.iter().map(|res| CKKSSquareAssignItem { dst: res }).collect();
                module.ckks_square_assign_batch_tmp_bytes(&batch, tsk)
            }
            Frontier::PreparedAssign(items) => {
                let batch: Vec<CKKSPreparedMulAssignItem<&CtLayout, &CKKSPreparedRightLayout>> = items
                    .iter()
                    .map(|(res, prepared)| CKKSPreparedMulAssignItem { dst: res, prepared })
                    .collect();
                module.ckks_mul_prepared_assign_batch_tmp_bytes(&batch, tsk)
            }
        }
    }
}

/// A stage's power-basis plan.
fn stage_plan<BE, P>(bsgs: &StageBsgs<'_, P>) -> Result<PowerPlan>
where
    BE: Backend,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
{
    let poly = match bsgs {
        StageBsgs::Real(poly) => *poly,
        StageBsgs::Complex(poly) => &poly.re,
    };
    power_basis_plan(
        BSGSPolynomialInfos::<BE>::basis(poly),
        BSGSPolynomialInfos::<BE>::degree(poly),
        BSGSPolynomialInfos::<BE>::log_split(poly),
        BSGSPolynomialInfos::<BE>::parity(poly),
    )
}

/// The mirror of the executing driver: it walks the same plan, applies the same
/// metadata rule at every step, and emits one [`Frontier`] per batch call.
struct Replay {
    frontiers: Vec<Frontier>,
    /// Peak per-branch count of simultaneously live ciphertexts.
    live_ciphertexts: usize,
    /// Peak count of distinct hoisted `X^{gsp}` operands in one level.
    hoisted_operands: usize,
}

type Powers = HashMap<usize, CtLayout>;

fn power(basis: &Powers, n: usize) -> Result<CtLayout> {
    basis
        .get(&n)
        .copied()
        .ok_or_else(|| crate::CKKSError::from(anyhow::anyhow!("lockstep scratch replay: power {n} is not in the basis")))
}

impl Replay {
    fn new() -> Self {
        Self {
            frontiers: Vec::new(),
            live_ciphertexts: 0,
            hoisted_operands: 0,
        }
    }

    /// `execute_power_plan` on layouts. The destinations are priced *before*
    /// their stamp, which is what the batch operation receives.
    fn power_plan(&mut self, plan: &PowerPlan, basis: &mut [Powers; 2]) -> Result<()> {
        for product in &plan.products {
            let mut before: Vec<CtLayout> = Vec::with_capacity(2);
            let mut after: Vec<CtLayout> = Vec::with_capacity(2);
            for entries in basis.iter() {
                let a = power(entries, product.a)?;
                let b = power(entries, product.b)?;
                let pre = CtLayout::alloc(a.n(), a.base2k(), (mul_ct_k(&a, &b)?).into(), a.rank());
                let (log_budget, log_delta, _) = crate::default::mul::get_mul_ct_params(&pre, &a, &b)?;
                let mut post = pre;
                post.stamp(
                    log_budget,
                    log_delta,
                    Some(a.log_sparsity().min(b.log_sparsity())),
                    Some(a.slots().join(b.slots())),
                );
                before.push(pre);
                after.push(post);
            }

            self.frontiers.push(if product.a == product.b {
                Frontier::SquareInto(
                    (0..2)
                        .map(|branch| Ok((before[branch], power(&basis[branch], product.a)?)))
                        .collect::<Result<Vec<_>>>()?,
                )
            } else {
                Frontier::MulInto(
                    (0..2)
                        .map(|branch| {
                            Ok((
                                before[branch],
                                power(&basis[branch], product.a)?,
                                power(&basis[branch], product.b)?,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            });

            if plan.basis == Basis::Chebyshev {
                // `ckks_mul_pow2_assign` leaves the metadata alone; the tail is
                // `− 1` or `− T_c`.
                for (branch, entries) in basis.iter().enumerate() {
                    match product.c {
                        None => after[branch].join_real_slots(),
                        Some(c) => {
                            let c_val = power(entries, c)?;
                            after[branch].add_assign_from(&c_val);
                        }
                    }
                }
            }

            for (branch, entries) in basis.iter_mut().enumerate() {
                entries.insert(product.n, after[branch]);
            }
        }
        Ok(())
    }

    /// `polynomial_input_pair` on layouts.
    fn polynomial_input(&mut self, srcs: [CtLayout; 2], transform: PolynomialInputTransform) -> Result<[CtLayout; 2]> {
        let mut inputs = [CtLayout::alloc_from_infos(&srcs[0]), CtLayout::alloc_from_infos(&srcs[1])];
        for branch in 0..2 {
            inputs[branch].copy_from(&srcs[branch]);
        }
        let Some(plan) = transform_plan(transform) else {
            return Ok(inputs);
        };
        let mut basis: [Powers; 2] = [Powers::from([(1, inputs[0])]), Powers::from([(1, inputs[1])])];
        self.power_plan(&plan, &mut basis)?;
        Ok([power(&basis[0], 2)?, power(&basis[1], 2)?])
    }

    /// `eval_baby_step` on layouts: seed from the highest scheduled power, then
    /// the ordered `ct×pt` chain, whose exact evolution comes from the shared
    /// planner.
    fn baby_step<P>(&self, basis: &Powers, parity: Parity, coeffs: &P) -> Result<CtLayout>
    where
        P: IntPolyInfos + CKKSCtBounds,
    {
        let degree = coeffs.n().as_usize() - 1;
        let (first, step) = match parity {
            Parity::Even => (2, 2),
            Parity::Odd => (1, 2),
            Parity::Full => (1, 1),
        };
        let mut value = CtLayout::alloc_from_infos(&power(basis, 1)?);
        let init_power = (first..=degree).step_by(step).last().unwrap_or(1);
        value.init_accumulator_from(&power(basis, init_power)?);
        if parity != Parity::Odd {
            value.join_real_slots();
        }
        let powers: Vec<CtLayout> = (first..=degree)
            .step_by(step)
            .map(|i| power(basis, i))
            .collect::<Result<_>>()?;
        let terms: Vec<(&CtLayout, usize)> = powers.iter().zip((first..=degree).step_by(step)).collect();
        value.mul_add_pt_consts(&terms, coeffs)?;
        Ok(value)
    }

    /// `eval_giant_steps_pair` on layouts, including the interleaving.
    fn giant_steps(&mut self, steps: &mut [Vec<(usize, CtLayout)>; 2], basis: &[Powers; 2]) -> Result<()> {
        let levels = [
            giant_step_schedule(&steps[0].iter().map(|(d, _)| *d).collect::<Vec<_>>()),
            giant_step_schedule(&steps[1].iter().map(|(d, _)| *d).collect::<Vec<_>>()),
        ];
        for (pairs_0, pairs_1) in levels[0].iter().zip(levels[1].iter()) {
            let pairs = [pairs_0, pairs_1];
            if pairs_0.is_empty() && pairs_1.is_empty() {
                continue;
            }
            for branch_pairs in pairs.iter() {
                let mut distinct: Vec<usize> = Vec::new();
                for pair in branch_pairs.iter() {
                    if !distinct.contains(&pair.gsp) {
                        distinct.push(pair.gsp);
                    }
                }
                self.hoisted_operands = self.hoisted_operands.max(distinct.len());
            }

            // Branch 0 pair 0, branch 1 pair 0, branch 0 pair 1, ...
            let mut items: Vec<(CtLayout, CKKSPreparedRightLayout)> = Vec::new();
            for position in 0..pairs_0.len().max(pairs_1.len()) {
                for branch in 0..2 {
                    if let Some(pair) = pairs[branch].get(position) {
                        let source = power(&basis[branch], pair.gsp)?;
                        items.push((steps[branch][pair.high].1, CKKSPreparedRightLayout::of(&source)));
                    }
                }
            }
            self.frontiers.push(Frontier::PreparedAssign(items));

            for branch in 0..2 {
                for pair in pairs[branch].iter() {
                    let source = power(&basis[branch], pair.gsp)?;
                    let dst = steps[branch][pair.high].1;
                    let (log_budget, log_delta, _) = crate::default::mul::mul_ct_params_raw(
                        dst.k().as_usize(),
                        dst.log_delta(),
                        dst.k().into(),
                        source.log_delta(),
                        source.k().into(),
                    )?;
                    steps[branch][pair.high].1.stamp(
                        log_budget,
                        log_delta,
                        Some(dst.log_sparsity().min(source.log_sparsity())),
                        Some(dst.slots().join(source.slots())),
                    );
                }
            }

            for branch in 0..2 {
                for pair in pairs[branch].iter() {
                    let low = steps[branch][pair.low].1;
                    steps[branch][pair.high].1.add_assign_from(&low);
                }
            }
        }
        Ok(())
    }

    /// `eval_poly_real_pair` / `eval_poly_complex_pair` on layouts.
    fn eval_poly<BE, P>(&mut self, acc: &mut [CtLayout; 2], bsgs: &StageBsgs<'_, P>, basis: &[Powers; 2]) -> Result<()>
    where
        BE: Backend,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
    {
        let poly = match bsgs {
            StageBsgs::Real(poly) => *poly,
            StageBsgs::Complex(poly) => &poly.re,
        };
        let n_baby = BSGSPolynomialInfos::<BE>::baby_steps(poly);
        ckks_ensure!(n_baby > 0, "lockstep scratch replay: polynomial has no baby step");
        let last = BSGSPolynomialInfos::<BE>::baby_step(poly, n_baby - 1);
        let fold_power = BSGSPolynomialInfos::<BE>::degree(poly);
        let can_fold = n_baby >= 2 && last.n().as_usize() == 1 && basis[0].contains_key(&fold_power);
        let n_to_process = if can_fold { n_baby - 1 } else { n_baby };
        let parity = BSGSPolynomialInfos::<BE>::parity(poly);

        let mut steps: [Vec<(usize, CtLayout)>; 2] = [Vec::with_capacity(n_to_process), Vec::with_capacity(n_to_process)];
        for branch in 0..2 {
            for i in 0..n_to_process {
                let coeffs = BSGSPolynomialInfos::<BE>::baby_step(poly, i);
                let degree = coeffs.n().as_usize() - 1;
                let mut value = self.baby_step(&basis[branch], parity, coeffs)?;
                if let StageBsgs::Complex(complex) = bsgs {
                    let im_coeffs = BSGSPolynomialInfos::<BE>::baby_step(&complex.im, i);
                    let mut im = self.baby_step(&basis[branch], parity, im_coeffs)?;
                    im.mul_i();
                    value.add_assign_from(&im);
                }
                steps[branch].push((degree, value));
            }
        }

        self.giant_steps(&mut steps, basis)?;

        for branch in 0..2 {
            let evaluated = steps[branch].last().expect("non-empty baby step vector").1;
            acc[branch].copy_from(&evaluated);
        }

        if can_fold {
            for branch in 0..2 {
                let xpow = power(&basis[branch], fold_power)?;
                acc[branch].mul_add_pt_consts(&[(&xpow, 0)], last)?;
                if let StageBsgs::Complex(complex) = bsgs {
                    let last_im = BSGSPolynomialInfos::<BE>::baby_step(&complex.im, n_baby - 1);
                    let mut im_fold = CtLayout::alloc_from_infos(&acc[branch]);
                    let (log_budget, log_delta, _) = crate::default::mul::get_mul_pt_params(&im_fold, &xpow, last_im)?;
                    im_fold.stamp(log_budget, log_delta, Some(xpow.log_sparsity()), Some(xpow.slots()));
                    im_fold.mul_i();
                    acc[branch].add_assign_from(&im_fold);
                }
            }
        }

        // Power basis, evaluated baby steps, the stage's working input and the
        // accumulator, all live at once.
        self.live_ciphertexts = self.live_ciphertexts.max(basis[0].len() + n_to_process + 2);
        Ok(())
    }

    /// `eval_stage_from_roots_pair` on layouts.
    fn stage_from_roots<BE, P>(&mut self, acc: &mut [CtLayout; 2], roots: [CtLayout; 2], bsgs: &StageBsgs<'_, P>) -> Result<()>
    where
        BE: Backend,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
    {
        let plan = stage_plan::<BE, P>(bsgs)?;
        let mut basis: [Powers; 2] = [Powers::from([(1, roots[0])]), Powers::from([(1, roots[1])])];
        self.power_plan(&plan, &mut basis)?;
        self.eval_poly::<BE, P>(acc, bsgs, &basis)
    }

    /// `eval_stage_pair` on layouts.
    fn stage<BE, P>(&mut self, acc: &mut [CtLayout; 2], inputs: [CtLayout; 2], bsgs: &StageBsgs<'_, P>) -> Result<()>
    where
        BE: Backend,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
    {
        let transform = stage_input_transform::<BE, P>(bsgs)?;
        let roots = self.polynomial_input(inputs, transform)?;
        self.stage_from_roots::<BE, P>(acc, roots, bsgs)?;

        if matches!(
            transform,
            PolynomialInputTransform::SquareTimesInput | PolynomialInputTransform::ChebyshevT2TimesInput
        ) {
            let prepared = [
                CKKSPreparedRightLayout::of(&inputs[0]),
                CKKSPreparedRightLayout::of(&inputs[1]),
            ];
            self.frontiers
                .push(Frontier::PreparedAssign(vec![(acc[0], prepared[0]), (acc[1], prepared[1])]));
            self.hoisted_operands = self.hoisted_operands.max(1);
            for branch in 0..2 {
                let dst = acc[branch];
                let src = inputs[branch];
                let (log_budget, log_delta, _) = crate::default::mul::mul_ct_params_raw(
                    dst.k().as_usize(),
                    dst.log_delta(),
                    dst.k().into(),
                    src.log_delta(),
                    src.k().into(),
                )?;
                acc[branch].stamp(
                    log_budget,
                    log_delta,
                    Some(dst.log_sparsity().min(src.log_sparsity())),
                    Some(dst.slots().join(src.slots())),
                );
            }
        }
        Ok(())
    }

    /// One range-extension level: both squares as one frontier.
    fn square_assign_frontier(&mut self, acc: &mut [CtLayout; 2]) -> Result<()> {
        self.frontiers.push(Frontier::SquareAssign(vec![acc[0], acc[1]]));
        for value in acc.iter_mut() {
            let (log_budget, log_delta, _) = crate::default::mul::get_mul_ct_params(value, value, value)?;
            value.stamp(log_budget, log_delta, None, None);
        }
        Ok(())
    }
}

/// Replays the whole lockstep pipeline on layout stand-ins.
fn replay_lockstep<BE, P, F>(
    acc: [CtLayout; 2],
    work_layout: [GLWELayout; 2],
    work_meta: [CKKSMeta; 2],
    params: &EvalMod<F, P>,
) -> Result<Replay>
where
    BE: Backend,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
{
    let mut replay = Replay::new();
    let mut acc = acc;

    // `eval_mod_input`: an owned copy at the working layout, relabelled to the
    // plan scale.
    let inputs: [CtLayout; 2] = std::array::from_fn(|branch| {
        let mut input = CtLayout::alloc(
            work_layout[branch].n,
            work_layout[branch].base2k,
            work_layout[branch].k,
            work_layout[branch].rank,
        );
        input.set_meta(work_meta[branch]);
        input
    });

    let base = StageBsgs::from(&params.f_mod_bsgs);
    let transform = stage_input_transform::<BE, P>(&base)?;
    let offset = params.f_mod_input_offset.as_ref();

    if transform == PolynomialInputTransform::Identity && offset.is_none() {
        replay.stage_from_roots::<BE, P>(&mut acc, inputs, &base)?;
    } else {
        let mut inputs = inputs;
        if offset.is_some() {
            for input in inputs.iter_mut() {
                input.join_real_slots();
            }
        }
        replay.stage::<BE, P>(&mut acc, inputs, &base)?;
    }

    match &params.f_mod_bsgs {
        EvalModBsgs::Real(_) => {
            if params.range_extension_consts.is_some() {
                for _ in 0..params.plan.f_mod_log_interval_reduction {
                    replay.square_assign_frontier(&mut acc)?;
                    for value in acc.iter_mut() {
                        // `ckks_mul_pow2_assign` leaves the metadata alone.
                        value.join_real_slots();
                    }
                }
            }
            if let Some(inv) = params.f_mod_inv_bsgs.as_ref() {
                let inputs: [CtLayout; 2] = std::array::from_fn(|branch| {
                    let mut input = CtLayout::alloc(
                        work_layout[branch].n,
                        work_layout[branch].base2k,
                        work_layout[branch].k,
                        work_layout[branch].rank,
                    );
                    input.set_meta(work_meta[branch]);
                    input.copy_from(&acc[branch]);
                    input
                });
                replay.stage::<BE, P>(&mut acc, inputs, &StageBsgs::Real(inv))?;
            }
        }
        EvalModBsgs::Complex(_) => {
            for _ in 0..params.plan.f_mod_log_interval_reduction {
                replay.square_assign_frontier(&mut acc)?;
            }
        }
    }

    Ok(replay)
}

/// The `(batch operation, per-item layout)` sequence
/// [`ckks_eval_mod_pair_lockstep_default`] issues for these operands.
///
/// Each item is reported as `(destination k, destination capacity, left operand
/// k, right operand limbs)`. Exposed so a backend test can assert that what it
/// observed is what the scratch query priced.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn lockstep_frontier_shapes<BE, R0, R1, C0, C1, P, F>(
    res_0: &R0,
    res_1: &R1,
    ct_0: &C0,
    ct_1: &C1,
    params: &EvalMod<F, P>,
) -> Result<Vec<(&'static str, Vec<(u32, usize, u32, usize)>)>>
where
    BE: Backend,
    R0: CKKSCtBounds,
    R1: CKKSCtBounds,
    C0: CKKSCtBounds,
    C1: CKKSCtBounds,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
{
    let (acc, work_layout, work_meta) = lockstep_entry_layouts(res_0, res_1, ct_0, ct_1, params);
    let replay = replay_lockstep::<BE, P, F>(acc, work_layout, work_meta, params)?;
    Ok(replay
        .frontiers
        .iter()
        .map(|frontier| (frontier.kind(), frontier.item_shapes()))
        .collect())
}

/// The accumulator and working-ciphertext layouts the driver starts from.
fn lockstep_entry_layouts<R0, R1, C0, C1, P, F>(
    res_0: &R0,
    res_1: &R1,
    ct_0: &C0,
    ct_1: &C1,
    params: &EvalMod<F, P>,
) -> ([CtLayout; 2], [GLWELayout; 2], [CKKSMeta; 2])
where
    R0: CKKSCtBounds,
    R1: CKKSCtBounds,
    C0: CKKSCtBounds,
    C1: CKKSCtBounds,
{
    let s_eval = params.plan.f_mod_log_delta;
    let work_layout = [
        GLWELayout {
            n: ct_0.n(),
            base2k: ct_0.base2k(),
            k: (ct_0.log_budget() + s_eval).into(),
            rank: Rank(1),
        },
        GLWELayout {
            n: ct_1.n(),
            base2k: ct_1.base2k(),
            k: (ct_1.log_budget() + s_eval).into(),
            rank: Rank(1),
        },
    ];
    let work_meta = [
        CKKSMeta {
            log_delta: s_eval,
            log_sparsity: ct_0.log_sparsity(),
            slots: ct_0.slots(),
        },
        CKKSMeta {
            log_delta: s_eval,
            log_sparsity: ct_1.log_sparsity(),
            slots: ct_1.slots(),
        },
    ];
    let acc = [CtLayout::alloc_from_infos(res_0), CtLayout::alloc_from_infos(res_1)];
    (acc, work_layout, work_meta)
}

/// Scratch for [`ckks_eval_mod_pair_lockstep_default`].
///
/// Three terms:
///
/// 1. the working set both branches hold at once (power bases, evaluated baby
///    steps, working inputs, accumulators and the hoisted `X^{gsp}` operands);
/// 2. the largest single-branch stage budget, covering every step the driver
///    runs outside a batch (copies, `ct×pt` baby steps, adds, the `pow2`/`sub`
///    tails);
/// 3. the largest of the frontiers, each queried through its public batch
///    `*_tmp_bytes` with the *exact* item layouts the driver will pass, so a
///    backend that selects a fused path for some layouts and a fallback for
///    others is priced on what it will really see.
///
/// The reference driver keeps term 1 on the heap and therefore runs inside
/// terms 2 and 3 alone; a backend that carves its working set from the arena is
/// covered by the full number.
#[allow(clippy::too_many_arguments)]
pub fn ckks_eval_mod_pair_lockstep_tmp_bytes_default<BE, R0, R1, C0, C1, P, F, T>(
    module: &Module<BE>,
    res_0: &R0,
    res_1: &R1,
    ct_0: &C0,
    ct_1: &C1,
    params: &EvalMod<F, P>,
    tsk: &T,
) -> usize
where
    BE: Backend,
    Module<BE>: CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSMulOps<BE> + CKKSCopyOps<BE> + CnvPVecBytesOf,
    R0: CKKSCtBounds,
    R1: CKKSCtBounds,
    C0: CKKSCtBounds,
    C1: CKKSCtBounds,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
    T: GGLWEInfos,
{
    let stages = super::eval_mod::ckks_eval_mod_tmp_bytes_default(module, res_0, ct_0, params, tsk).max(
        super::eval_mod::ckks_eval_mod_tmp_bytes_default(module, res_1, ct_1, params, tsk),
    );

    let (acc, work_layout, work_meta) = lockstep_entry_layouts(res_0, res_1, ct_0, ct_1, params);
    let Ok(replay) = replay_lockstep::<BE, P, F>(acc, work_layout, work_meta, params) else {
        // A malformed plan: the driver rejects it before allocating, so the
        // sequential budget is a sound answer.
        return stages;
    };

    let batches = replay
        .frontiers
        .iter()
        .map(|frontier| frontier.tmp_bytes(module, tsk))
        .max()
        .unwrap_or(0);

    // The working set is sized from the widest ciphertext either branch holds.
    let cols: usize = (ct_0.rank() + 1).into();
    let work_size = work_layout[0]
        .k
        .as_usize()
        .div_ceil(work_layout[0].base2k.as_usize())
        .max(work_layout[1].k.as_usize().div_ceil(work_layout[1].base2k.as_usize()))
        .max(res_0.max_size())
        .max(res_1.max_size())
        .max(1);
    let live_ct = BE::bytes_of_vec_znx(ct_0.n().as_usize(), cols, work_size);
    let hoisted = module.bytes_of_cnv_pvec_right(cols, work_size);
    let live = 2 * (replay.live_ciphertexts * live_ct + replay.hoisted_operands * hoisted);

    live + stages.max(batches)
}

/// Runs two EvalMod DAGs in lockstep on one thread and one module, advancing
/// them by dependency frontier and dispatching every tensor-product frontier
/// through the CKKS batch operations.
///
/// Semantically identical to two [`ckks_eval_mod`] calls: the branches are
/// independent, so interleaving leaves each one's own operation sequence
/// untouched and every result is byte-for-byte the sequential one. No worker
/// threads, broker, extra modules or streams are involved.
///
/// [`ckks_eval_mod`]: crate::api::CKKSEvalModOps::ckks_eval_mod
#[allow(clippy::too_many_arguments)]
pub fn ckks_eval_mod_pair_lockstep_default<BE, R0, R1, C0, C1, P, F>(
    module: &Module<BE>,
    res_0: &mut R0,
    res_1: &mut R1,
    ct_0: &C0,
    ct_1: &C1,
    params: &EvalMod<F, P>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLockstepOps<BE>,
    R0: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    C0: GLWEToBackendRef<BE> + CKKSCtBounds,
    C1: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    Ct<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let s_eval = params.plan.f_mod_log_delta;
    let required = params.consumed_bits();

    // Validate both branches before either is touched.
    let s_in = [ct_0.log_delta(), ct_1.log_delta()];
    let s_budget = [ct_0.log_budget(), ct_1.log_budget()];
    for (branch, (rank, budget)) in [(ct_0.rank().as_usize(), s_budget[0]), (ct_1.rank().as_usize(), s_budget[1])]
        .into_iter()
        .enumerate()
    {
        ckks_ensure!(
            rank == 1,
            "ckks_eval_mod_pair_lockstep supports rank-1 ciphertexts only, branch {branch} got rank {rank}"
        );
        ckks_ensure!(
            budget >= required,
            "ckks_eval_mod_pair_lockstep: branch {branch} log_budget {budget} < {required} bits required (consumed at scale {s_eval})"
        );
    }

    let work_layout = [
        GLWELayout {
            n: ct_0.n(),
            base2k: ct_0.base2k(),
            k: (s_budget[0] + s_eval).into(),
            rank: Rank(1),
        },
        GLWELayout {
            n: ct_1.n(),
            base2k: ct_1.base2k(),
            k: (s_budget[1] + s_eval).into(),
            rank: Rank(1),
        },
    ];
    let work_meta = [
        CKKSMeta {
            log_delta: s_eval,
            log_sparsity: ct_0.log_sparsity(),
            slots: ct_0.slots(),
        },
        CKKSMeta {
            log_delta: s_eval,
            log_sparsity: ct_1.log_sparsity(),
            slots: ct_1.slots(),
        },
    ];

    // The whole pipeline runs on owned accumulators of the destinations' own
    // layout, so both branches share one ciphertext type and every frontier is
    // one slice. The final copy is width-preserving.
    let mut acc: [Ct<BE>; 2] = [
        module.ckks_ciphertext_alloc_from_infos(&*res_0),
        module.ckks_ciphertext_alloc_from_infos(&*res_1),
    ];

    let offset = params.f_mod_input_offset.as_ref();
    let transform = stage_input_transform::<BE, P>(&StageBsgs::from(&params.f_mod_bsgs))?;

    if transform == PolynomialInputTransform::Identity && offset.is_none() {
        // Hand the relabelled inputs straight to the power bases, as the
        // single-branch pipeline does.
        let roots = [
            super::eval_mod::eval_mod_input(module, ct_0, &work_layout[0], work_meta[0]),
            super::eval_mod::eval_mod_input(module, ct_1, &work_layout[1], work_meta[1]),
        ];
        eval_stage_from_roots_pair(&mut acc, roots, &StageBsgs::from(&params.f_mod_bsgs), module, tsk, scratch)?;
    } else {
        let mut inputs = [
            super::eval_mod::eval_mod_input(module, ct_0, &work_layout[0], work_meta[0]),
            super::eval_mod::eval_mod_input(module, ct_1, &work_layout[1], work_meta[1]),
        ];
        if let Some(offset) = offset {
            for input in inputs.iter_mut() {
                module.ckks_add_pt_const_assign(input, 0, offset, 0, scratch)?;
            }
        }
        eval_stage_pair(&mut acc, inputs, &StageBsgs::from(&params.f_mod_bsgs), module, tsk, scratch)?;
    }

    match &params.f_mod_bsgs {
        EvalModBsgs::Real(_) => {
            if let Some(consts) = params.range_extension_consts.as_ref() {
                for i in 0..params.plan.f_mod_log_interval_reduction {
                    square_assign_frontier(&mut acc, module, tsk, scratch)?;
                    for value in acc.iter_mut() {
                        module.ckks_mul_pow2_assign(value, 1, scratch)?;
                        module.ckks_sub_pt_const_assign(value, 0, consts, i, scratch)?;
                    }
                }
            }

            if let Some(inv) = params.f_mod_inv_bsgs.as_ref() {
                let mut inputs: [Ct<BE>; 2] = [
                    module.ckks_ciphertext_alloc_from_glwe_infos(&work_layout[0]),
                    module.ckks_ciphertext_alloc_from_glwe_infos(&work_layout[1]),
                ];
                for branch in 0..2 {
                    inputs[branch].set_meta(work_meta[branch]);
                    module.ckks_copy(&mut inputs[branch], &acc[branch], scratch)?;
                }
                eval_stage_pair(&mut acc, inputs, &StageBsgs::Real(inv), module, tsk, scratch)?;
            }
        }
        EvalModBsgs::Complex(_) => {
            for _ in 0..params.plan.f_mod_log_interval_reduction {
                square_assign_frontier(&mut acc, module, tsk, scratch)?;
            }
        }
    }

    module.ckks_copy(res_0, &acc[0], scratch)?;
    module.ckks_copy(res_1, &acc[1], scratch)?;
    if s_eval != s_in[0] {
        res_0.set_log_delta(s_in[0]);
    }
    if s_eval != s_in[1] {
        res_1.set_log_delta(s_in[1]);
    }
    Ok(())
}

/// The two range-extension squares of one level, as one frontier.
fn square_assign_frontier<BE>(
    acc: &mut [Ct<BE>; 2],
    module: &Module<BE>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLockstepOps<BE>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    Ct<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let (first, second) = acc.split_at_mut(1);
    let mut items = [
        CKKSSquareAssignItem { dst: &mut first[0] },
        CKKSSquareAssignItem { dst: &mut second[0] },
    ];
    module.ckks_square_assign_batch(&mut items, tsk, scratch)
}
