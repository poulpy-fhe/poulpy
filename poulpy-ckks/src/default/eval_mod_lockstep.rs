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
//! sequential [`CKKSEvalModOps::ckks_eval_mod`] calls produce. The cost is
//! heap, not scratch: both power bases and both baby-step vectors are live at
//! once.
//!
//! [`CKKSEvalModOps::ckks_eval_mod_pair`]: crate::api::CKKSEvalModOps::ckks_eval_mod_pair
//! [`CKKSEvalModOps::ckks_eval_mod`]: crate::api::CKKSEvalModOps::ckks_eval_mod

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    GLWECopy, GLWEPolynomialEvaluation, GLWEZero,
    default::polynomial_evaluation::giant_step_schedule,
    layouts::{
        BSGSMeta, BSGSPolynomial, BSGSPolynomialInfos, GGLWEInfos, GLWEInfos, GLWELayout, GLWETensorKeyPrepared,
        GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, LWEInfos, Parity, PowerBasis, PowerBasisHelper, Rank, SetBSGSMeta,
        prepared::GLWETensorKeyPreparedToBackendRef, split_degree,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos,
    api::{
        Basis, CKKSAddOps, CKKSCopyOps, CKKSImagOps, CKKSMulAddOps, CKKSMulIntoItem, CKKSMulOps, CKKSPow2Ops,
        CKKSPreparedMulAssignItem, CKKSSquareAssignItem, CKKSSquareIntoItem, CKKSSubOps, PolynomialInputTransform,
    },
    default::{carry_verb::ckks_one_pt, polynomial_evaluation::CKKSBSGSOps},
    layouts::{
        CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPreparedRight,
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

/// Monomial `X^n` for both branches, batching the one tensor product.
fn gen_power_pair<BE>(
    n: usize,
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
    if basis[0].contains_power(n) {
        return Ok(());
    }
    ckks_ensure!(n >= 2, "gen_power_pair: n={n} < 2; X^1 must be provided at construction");

    let (a, b) = split_degree(n);
    gen_power_pair(a, basis, module, tsk, scratch)?;
    gen_power_pair(b, basis, module, tsk, scratch)?;

    let [basis_0, basis_1] = basis;
    let (mut r0, mut r1) = {
        let a0 = basis_0.get_stored(a).expect("gen_power_pair(a) just succeeded");
        let b0 = basis_0.get_stored(b).expect("gen_power_pair(b) just succeeded");
        let a1 = basis_1.get_stored(a).expect("gen_power_pair(a) just succeeded");
        let b1 = basis_1.get_stored(b).expect("gen_power_pair(b) just succeeded");
        (
            module.ckks_ciphertext_alloc(a0.base2k(), mul_ct_k(a0, b0)?.into()),
            module.ckks_ciphertext_alloc(a1.base2k(), mul_ct_k(a1, b1)?.into()),
        )
    };

    {
        let a0 = basis_0.get_stored(a).expect("stored above");
        let a1 = basis_1.get_stored(a).expect("stored above");
        if a == b {
            let mut items = [
                CKKSSquareIntoItem { dst: &mut r0, a: a0 },
                CKKSSquareIntoItem { dst: &mut r1, a: a1 },
            ];
            module.ckks_square_into_batch(&mut items, tsk, scratch)?;
        } else {
            let b0 = basis_0.get_stored(b).expect("stored above");
            let b1 = basis_1.get_stored(b).expect("stored above");
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

    basis_0.set_power(n, r0);
    basis_1.set_power(n, r1);
    Ok(())
}

/// Chebyshev `T_n` for both branches: the product is batched, the
/// `2·T_a·T_b − T_c` tail stays per branch (no tensor product).
fn gen_power_chebyshev_pair<BE>(
    n: usize,
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
    if basis[0].contains_power(n) {
        return Ok(());
    }
    ckks_ensure!(
        n >= 2,
        "gen_power_chebyshev_pair: n={n} < 2; T_1 must be provided at construction"
    );

    let (a, b) = split_degree(n);
    gen_power_chebyshev_pair(a, basis, module, tsk, scratch)?;
    gen_power_chebyshev_pair(b, basis, module, tsk, scratch)?;
    let c = a.abs_diff(b);
    if c != 0 {
        gen_power_chebyshev_pair(c, basis, module, tsk, scratch)?;
    }

    let [basis_0, basis_1] = basis;
    let (mut r0, mut r1) = {
        let a0 = basis_0.get_stored(a).expect("gen_power_chebyshev_pair(a) just succeeded");
        let b0 = basis_0.get_stored(b).expect("gen_power_chebyshev_pair(b) just succeeded");
        let a1 = basis_1.get_stored(a).expect("gen_power_chebyshev_pair(a) just succeeded");
        let b1 = basis_1.get_stored(b).expect("gen_power_chebyshev_pair(b) just succeeded");
        (
            module.ckks_ciphertext_alloc(a0.base2k(), mul_ct_k(a0, b0)?.into()),
            module.ckks_ciphertext_alloc(a1.base2k(), mul_ct_k(a1, b1)?.into()),
        )
    };

    {
        let a0 = basis_0.get_stored(a).expect("stored above");
        let a1 = basis_1.get_stored(a).expect("stored above");
        if a == b {
            let mut items = [
                CKKSSquareIntoItem { dst: &mut r0, a: a0 },
                CKKSSquareIntoItem { dst: &mut r1, a: a1 },
            ];
            module.ckks_square_into_batch(&mut items, tsk, scratch)?;
        } else {
            let b0 = basis_0.get_stored(b).expect("stored above");
            let b1 = basis_1.get_stored(b).expect("stored above");
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

    for (r, source) in [(&mut r0, &*basis_0), (&mut r1, &*basis_1)] {
        module.ckks_mul_pow2_assign(r, 1, scratch)?;
        if c == 0 {
            let one = ckks_one_pt(module, r.base2k())?;
            module.ckks_sub_pt_const_assign(r, 0, &one, 0, scratch)?;
        } else {
            let c_val = source.get_stored(c).expect("gen_power_chebyshev_pair(c) just succeeded");
            module.ckks_sub_assign(r, c_val, scratch)?;
        }
    }

    basis_0.set_power(n, r0);
    basis_1.set_power(n, r1);
    Ok(())
}

/// [`crate::power_basis::PowerBasisGen::populate`] for both branches, in the
/// same order, with each product issued as one two-item frontier.
fn populate_pair<BE>(
    degree: usize,
    log_split: usize,
    parity: Parity,
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
    ckks_ensure!(degree >= 1, "populate_pair: degree must be >= 1");

    let log_degree = (usize::BITS - degree.leading_zeros()) as usize;
    let largest_pow2 = 1usize << (log_degree - 1);
    let base = 1usize << log_split;
    let chebyshev = basis[0].basis() == Basis::Chebyshev;

    let generate = |n: usize, basis: &mut [PowerBasis<Ct<BE>>; 2], scratch: &mut ScratchArena<'_, BE>| -> Result<()> {
        if chebyshev {
            gen_power_chebyshev_pair(n, basis, module, tsk, scratch)
        } else {
            gen_power_pair(n, basis, module, tsk, scratch)
        }
    };

    if largest_pow2 >= 2 {
        generate(largest_pow2, basis, scratch)?;
    }

    let baby_limit = base.min(degree + 1);
    match parity {
        Parity::Even => {
            for i in (4..baby_limit).step_by(2) {
                generate(i, basis, scratch)?;
            }
        }
        Parity::Odd => {
            for i in (3..baby_limit).step_by(2) {
                generate(i, basis, scratch)?;
            }
        }
        Parity::Full => {
            for i in (3..baby_limit).rev() {
                generate(i, basis, scratch)?;
            }
        }
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
    match transform {
        PolynomialInputTransform::Identity => Ok([x0, x1]),
        PolynomialInputTransform::Square | PolynomialInputTransform::SquareTimesInput => {
            let mut basis = [PowerBasis::new(Basis::Monomial, x0), PowerBasis::new(Basis::Monomial, x1)];
            gen_power_pair(2, &mut basis, module, tsk, scratch)?;
            let [b0, b1] = &mut basis;
            Ok([
                b0.take_power(2).expect("generating x^2 must store the degree-two power"),
                b1.take_power(2).expect("generating x^2 must store the degree-two power"),
            ])
        }
        PolynomialInputTransform::ChebyshevT2 | PolynomialInputTransform::ChebyshevT2TimesInput => {
            let mut basis = [PowerBasis::new(Basis::Chebyshev, x0), PowerBasis::new(Basis::Chebyshev, x1)];
            gen_power_chebyshev_pair(2, &mut basis, module, tsk, scratch)?;
            let [b0, b1] = &mut basis;
            Ok([
                b0.take_power(2).expect("generating T2 must store the degree-two power"),
                b1.take_power(2).expect("generating T2 must store the degree-two power"),
            ])
        }
    }
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
    let mut basis = [PowerBasis::new(poly_basis, x0), PowerBasis::new(poly_basis, x1)];
    populate_pair(degree, log_split, parity, &mut basis, module, tsk, scratch)?;

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
        for (branch, input) in inputs.iter().enumerate() {
            module.ckks_mul_assign(&mut acc[branch], input, tsk, scratch)?;
        }
    }
    Ok(())
}

/// Scratch for [`ckks_eval_mod_pair_lockstep_default`].
///
/// The lockstep keeps its extra state (both power bases, both baby-step
/// vectors, both working inputs) on the heap, and every batch it issues carries
/// the sequential default's per-item bound, so the arena requirement is the
/// larger of the two single-branch budgets: the same value
/// `ckks_eval_mod_pair_tmp_bytes` returns.
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
    Module<BE>: CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSMulOps<BE> + CKKSCopyOps<BE> + poulpy_hal::api::CnvPVecBytesOf,
    R0: CKKSCtBounds,
    R1: CKKSCtBounds,
    C0: CKKSCtBounds,
    C1: CKKSCtBounds,
    P: CKKSCtBounds,
    T: GGLWEInfos,
{
    super::eval_mod::ckks_eval_mod_tmp_bytes_default(module, res_0, ct_0, params, tsk).max(
        super::eval_mod::ckks_eval_mod_tmp_bytes_default(module, res_1, ct_1, params, tsk),
    )
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
