//! Scheme-agnostic Baby-Step / Giant-Step (BSGS) polynomial-evaluation schedule
//! and host-side polynomial representation.
//!
//! Holds the integer BSGS planning (split-strategy selection, coefficient
//! decomposition into baby steps) and the cleartext [`Polynomial<F>`] /
//! decomposed [`BSGSPolynomial<C>`] types. Per-baby-step coefficient encoding is
//! supplied by the scheme layer through the closure passed to
//! [`Polynomial::decompose_bsgs_with`].

use std::collections::HashMap;
use std::fmt::Debug;

use anyhow::{Result, anyhow, ensure};
use poulpy_hal::layouts::Backend;

use crate::layouts::IntPolyInfos;
use rand_distr::num_traits::{Float, FloatConst, FromPrimitive};

use crate::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef};

// ── Basis / Parity ───────────────────────────────────────────────────────────

/// Polynomial evaluation basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    /// Standard monomial basis: {1, X, X², …}
    Monomial,
    /// Chebyshev first-kind basis: {T₀(X), T₁(X), T₂(X), …}
    Chebyshev,
}

/// Symmetry class of a polynomial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parity {
    /// No symmetry: all powers may be non-zero.
    Full,
    /// Even polynomial: only even-degree coefficients are non-zero.
    Even,
    /// Odd polynomial: only odd-degree coefficients are non-zero.
    Odd,
}

/// Input rewrite attached to a decomposed polynomial.
///
/// Chebyshev polynomials with a known parity can be folded through
/// `T₂(x) = 2x² - 1`, halving the encoded degree. Odd polynomials additionally
/// factor out one copy of the original input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PolynomialInputTransform {
    /// Evaluate the encoded polynomial directly on the input.
    #[default]
    Identity,
    /// Evaluate the encoded polynomial on `T₂(x)`.
    ChebyshevT2,
    /// Evaluate the encoded polynomial on `T₂(x)`, then multiply by `x`.
    ChebyshevT2TimesInput,
}

impl PolynomialInputTransform {
    fn extra_depth(self) -> usize {
        match self {
            Self::Identity => 0,
            Self::ChebyshevT2 => 1,
            Self::ChebyshevT2TimesInput => 2,
        }
    }
}

// ── BSGS split-strategy planning ─────────────────────────────────────────────

/// Chooses how [`Polynomial::decompose_bsgs_with`] picks `log_split`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitStrategy {
    /// Closed-form choice minimising multiplicative depth.
    MinDepth,
    /// Sweep `log_split` to minimise `(CT-CT, PT-CT)` lexicographically.
    MinMult,
}

/// Default planner picked by [`Polynomial::decompose_bsgs_with`].
pub const DEFAULT_SPLIT_STRATEGY: SplitStrategy = SplitStrategy::MinDepth;

/// Returns the BSGS log-split that minimises multiplication depth for a
/// polynomial of log-degree `log_degree`.
pub(crate) fn min_depth_split(log_degree: usize) -> usize {
    debug_assert!(log_degree >= 1, "min_depth_split requires log_degree ≥ 1");
    let s = log_degree >> 1;
    let a = (1 << s) + (1 << (log_degree - s)) + log_degree - s - 3;
    let b = (1 << (s + 1)) + (1 << (log_degree - s - 1)) + log_degree - s - 4;
    if a > b { s + 1 } else { s }
}

/// Returns the BSGS log-split that minimises total multiplication count
/// `(CT-CT, PT-CT)` lexicographically, sweeping `log_split ∈ [1, log_degree]`.
pub(crate) fn min_mult_split(degree: usize, parity: Parity, basis: Basis) -> usize {
    let log_degree = bit_len(degree);
    if log_degree <= 1 {
        return 1;
    }
    let mut best_log_split = 1usize;
    let mut best_score = (usize::MAX, usize::MAX);
    for log_split in 1..=log_degree {
        let score = estimate_op_counts(degree, log_split, parity, basis);
        if score < best_score {
            best_score = score;
            best_log_split = log_split;
        }
    }
    best_log_split
}

pub(crate) fn split_for_strategy(strategy: SplitStrategy, degree: usize, parity: Parity, basis: Basis) -> usize {
    match strategy {
        SplitStrategy::MinDepth => min_depth_split(bit_len(degree)),
        SplitStrategy::MinMult => min_mult_split(degree, parity, basis),
    }
}

/// `(ct_ct, pt_ct)` mul count for the given plan.
fn estimate_op_counts(degree: usize, log_split: usize, parity: Parity, basis: Basis) -> (usize, usize) {
    let mut baby_degrees: Vec<usize> = Vec::new();
    simulate_baby_step_decomposition(degree, log_split, &mut baby_degrees);
    if baby_degrees.is_empty() {
        return (usize::MAX, usize::MAX);
    }
    let pt_ct: usize = baby_degrees.iter().map(|&d| pt_ct_muls_for_baby_step(d, parity)).sum();
    let giant_steps = baby_degrees.len().saturating_sub(1);
    let power_basis = estimate_power_basis_muls(degree, log_split, parity, basis);
    let mut cc = giant_steps + power_basis;
    let mut pc = pt_ct;
    let trailing_const = baby_degrees.len() >= 2 && *baby_degrees.last().unwrap() == 0;
    if trailing_const && degree.is_power_of_two() && cc > 0 {
        cc -= 1;
        pc += 1;
    }
    (cc, pc)
}

/// Replays `decompose_bsgs_coeffs` structurally to collect baby-step degrees.
fn simulate_baby_step_decomposition(degree: usize, log_split: usize, out: &mut Vec<usize>) {
    let base = 1usize << log_split;
    if degree < base {
        out.push(degree);
        return;
    }
    let mut next_power = base;
    while next_power < (degree >> 1) + 1 {
        next_power <<= 1;
    }
    simulate_baby_step_decomposition(next_power - 1, log_split, out);
    simulate_baby_step_decomposition(degree - next_power, log_split, out);
}

fn pt_ct_muls_for_baby_step(degree: usize, parity: Parity) -> usize {
    let (first, step) = match parity {
        Parity::Even => (2usize, 2usize),
        Parity::Odd => (1, 2),
        Parity::Full => (1, 1),
    };
    if degree < first { 0 } else { (degree - first) / step + 1 }
}

fn estimate_power_basis_muls(degree: usize, log_split: usize, parity: Parity, basis: Basis) -> usize {
    if degree < 2 {
        return 0;
    }
    let log_degree = bit_len(degree);
    let largest_pow2 = 1usize << (log_degree - 1);
    let base = 1usize << log_split;

    let mut targets: Vec<usize> = Vec::new();
    if largest_pow2 >= 2 {
        targets.push(largest_pow2);
    }
    let baby_limit = base.min(degree + 1);
    match parity {
        Parity::Even => targets.extend((4..baby_limit).step_by(2)),
        Parity::Odd => targets.extend((3..baby_limit).step_by(2)),
        Parity::Full => targets.extend(3..baby_limit),
    }

    let mut needed = std::collections::HashSet::<usize>::new();
    let mut stack = targets;
    while let Some(n) = stack.pop() {
        if n <= 1 || !needed.insert(n) {
            continue;
        }
        let (a, b) = split_degree(n);
        stack.push(a);
        stack.push(b);
        if matches!(basis, Basis::Chebyshev) {
            let c = a.abs_diff(b);
            if c >= 2 {
                stack.push(c);
            }
        }
    }
    needed.len()
}

fn decompose_bsgs_coeffs<F>(
    basis: Basis,
    coeffs: &[F],
    log_split: usize,
    max_degree: usize,
    lead: bool,
    split_leading: bool,
    visit: &mut impl FnMut(&[F]) -> Result<()>,
) -> Result<()>
where
    F: Float,
{
    let degree = coeffs.len().saturating_sub(1);
    let base = 1usize << log_split;
    if degree < base {
        if split_leading && should_split_leading_baby_step(degree, log_split, max_degree, lead) {
            let log_degree = bit_len(degree);
            let smaller = min_depth_split(log_degree);
            if smaller < log_split {
                return decompose_bsgs_coeffs(basis, coeffs, smaller, max_degree, lead, split_leading, visit);
            }
        }
        return visit(coeffs);
    }

    let mut next_power = base;
    while next_power < (degree >> 1) + 1 {
        next_power <<= 1;
    }

    match basis {
        Basis::Monomial => {
            decompose_bsgs_coeffs(
                basis,
                &coeffs[..next_power],
                log_split,
                max_degree,
                false,
                split_leading,
                visit,
            )?;
            decompose_bsgs_coeffs(
                basis,
                &coeffs[next_power..],
                log_split,
                max_degree,
                lead,
                split_leading,
                visit,
            )
        }
        Basis::Chebyshev => {
            let (q, r) = factorize_coeffs_chebyshev(coeffs, next_power);
            decompose_bsgs_coeffs(basis, &r, log_split, max_degree, false, split_leading, visit)?;
            decompose_bsgs_coeffs(basis, &q, log_split, max_degree, lead, split_leading, visit)
        }
    }
}

fn should_split_leading_baby_step(degree: usize, log_split: usize, max_degree: usize, lead: bool) -> bool {
    if !lead || degree == 0 || log_split <= 1 {
        return false;
    }
    let next_strict_power_of_two = 1usize << bit_len(max_degree);
    let close_to_next_power = next_strict_power_of_two - (1usize << (log_split - 1));
    max_degree > close_to_next_power
}

fn bit_len(n: usize) -> usize {
    usize::BITS as usize - n.leading_zeros() as usize
}

fn factorize_coeffs_chebyshev<F>(coeffs: &[F], n: usize) -> (Vec<F>, Vec<F>)
where
    F: Float,
{
    let mut r = vec![F::zero(); n];
    r.copy_from_slice(&coeffs[..n]);

    let mut q = vec![F::zero(); coeffs.len() - n];
    q[0] = coeffs[n];

    let two = F::one() + F::one();
    for (i, j) in (n + 1..coeffs.len()).zip(1..) {
        q[i - n] = two * coeffs[i];
        r[n - j] = r[n - j] - coeffs[i];
    }

    (q, r)
}

/// Splits `n` into `(a, b)` with `n = a + b` and `|a – b|` minimised.
///
/// When `n` is a power of two `a = b = n/2`; otherwise uses the
/// Lee et al. (2020) strategy that maximises the number of odd-degree
/// Chebyshev terms.
pub fn split_degree(n: usize) -> (usize, usize) {
    assert!(n > 1);
    if n.is_power_of_two() {
        (n / 2, n / 2)
    } else {
        let k = (usize::BITS - (n - 1).leading_zeros()) as usize - 1;
        let a = (1usize << k) - 1;
        let b = n + 1 - (1usize << k);
        (a, b)
    }
}

/// Multiplicative depth (CT-CT multiplication levels) consumed by a BSGS
/// evaluation of a degree-`degree` polynomial under `strategy`.
///
/// The depth-optimal [`SplitStrategy::MinDepth`] split reaches `bit_len(degree)`
/// (`= ceil(log2(degree + 1))`). [`SplitStrategy::MinMult`] minimises the
/// multiplication count instead, which for the upper part of each
/// `[2^(b-1), 2^b)` band costs one extra level. The threshold within a band of
/// `b = bit_len(degree)` bits is `2^b − 2^((b-1)/2) + 1` (matching the giant-step
/// structure of `eval_giant_steps`).
/// `log_budget` bits a BSGS evaluation of a degree-`degree` polynomial consumes:
/// the multiplicative depth of the heaviest chain through its data-dependency
/// graph. Each `ct×ct` (power-basis build step and giant-step multiply) weighs
/// `input_log_delta`; the single baby-step `ct×pt` inner product weighs
/// `coeff_log_delta`.
///
/// Closed form. Any single path contains at most one `ct×pt` (build powers → one
/// baby-step inner product → giant `ct×ct`s), so the heaviest path is the larger
/// of the longest **pure** `ct×ct` chain and the longest chain ending in a
/// `ct×pt`:
///
/// ```text
/// consumed = max( P·Δ_in , R·Δ_in + Δ_coeff )
/// ```
///
/// `P`, the longest pure `ct×ct` chain, is `bit_len(degree)`, minus one when
/// `degree` is a power of two: then its leading term is a lone constant reached
/// only through a `ct×pt` (a `MinDepth` fold, or the deepest `MinMult` baby step),
/// so no pure chain reaches the top. For `MinMult` with `degree ≥ threshold` —
/// where [`bsgs_eval_depth`] already adds the extra level — `P = bit_len =
/// eval_depth − 1` drops out automatically, so this one expression covers both
/// strategies.
///
/// `R`, the `ct×ct` count of the longest `ct×pt`-terminated chain, is
/// `eval_depth − 1` (the highest term realises a `ct×pt` chain of the full
/// critical-path length [`bsgs_eval_depth`]) for every degree, with one
/// exception: `MinMult` degree 5, whose `[1,1,1]` baby-step layout strands the
/// third step as an additive carry, capping `R` at 1 (verified the sole exception
/// for all degrees up to 8191).
///
/// Parity- and basis-independent: those affect op *counts*, not critical-path
/// depth. Validated exhaustively against a faithful replay of the evaluator for
/// every degree `2..=511`, both strategies and bases (see the `consumed_bits_*`
/// tests).
pub fn bsgs_consumed_bits(
    degree: usize,
    strategy: SplitStrategy,
    _parity: Parity,
    _basis: Basis,
    input_log_delta: usize,
    coeff_log_delta: usize,
) -> usize {
    if degree == 0 {
        return 0;
    }
    let eval_depth = bsgs_eval_depth(degree, strategy);
    let pure_depth = bit_len(degree) - degree.is_power_of_two() as usize;
    let ctpt_depth = if matches!(strategy, SplitStrategy::MinMult) && degree == 5 {
        1
    } else {
        eval_depth - 1
    };
    (pure_depth * input_log_delta).max(ctpt_depth * input_log_delta + coeff_log_delta)
}

/// Faithful replay of the homomorphic BSGS schedule — flat baby-step list,
/// iterative giant-step pairing (`b = b·Xᵍˢᵖ + a`), trailing-constant fold —
/// computing the same heaviest-chain weight as [`bsgs_consumed_bits`]. Kept as the
/// structure-exact reference the closed form is tested against.
#[cfg(test)]
fn bsgs_consumed_bits_reference(
    degree: usize,
    strategy: SplitStrategy,
    parity: Parity,
    basis: Basis,
    input_log_delta: usize,
    coeff_log_delta: usize,
) -> usize {
    if degree == 0 {
        return 0;
    }
    let log_split = split_for_strategy(strategy, degree, parity, basis);
    // Accurate flat baby-step degree list, mirroring `decompose_bsgs_coeffs`
    // (including the `MinDepth` leading-split recursion).
    let split_leading = matches!(strategy, SplitStrategy::MinDepth);
    let mut baby_degrees: Vec<usize> = Vec::new();
    collect_baby_step_degrees(degree, log_split, degree, true, split_leading, &mut baby_degrees);

    // A trailing lone constant is folded into the top power as a `ct×pt` (see the
    // ckks evaluator); its fold power `degree` is a built giant power iff `degree`
    // is a power of two.
    let n_baby = baby_degrees.len();
    let trailing_const = n_baby >= 2 && baby_degrees[n_baby - 1] == 0;
    let can_fold = trailing_const && degree.is_power_of_two();
    let n_to_process = if can_fold { n_baby - 1 } else { n_baby };

    // Weight of one evaluated baby step: the build depth of its deepest used power
    // (`ct×ct`s) plus the single `ct×pt` inner product. A lone constant is free
    // (it is just an encoded plaintext at full budget).
    let baby_weight = |d: usize| -> usize {
        let highest = match parity {
            Parity::Full => d,
            Parity::Odd => d - (d + 1) % 2,
            Parity::Even => d - d % 2,
        };
        if highest == 0 {
            0
        } else {
            power_basis_depth(highest) * input_log_delta + coeff_log_delta
        }
    };

    let mut active: Vec<(usize, usize)> = baby_degrees[..n_to_process].iter().map(|&d| (d, baby_weight(d))).collect();

    // Replay the giant-step pairing of `eval_giant_steps`: adjacent equal-degree
    // steps combine as `b = b·Xᵍˢᵖ + a` (one `ct×ct`); the odd one out carries.
    while active.len() > 1 {
        let mut next: Vec<(usize, usize)> = Vec::with_capacity(active.len().div_ceil(2));
        let mut i = 0;
        while i < active.len() {
            let is_last = i + 1 == active.len();
            if !is_last && active[i].0 == active[i + 1].0 {
                let gsp = (active[i].0 + 1).next_power_of_two();
                let x_pow = power_basis_depth(gsp) * input_log_delta;
                // b = b·Xᵍˢᵖ (ct×ct) then + a.
                let combined = (x_pow.max(active[i + 1].1) + input_log_delta).max(active[i].1);
                next.push((2 * gsp - 1, combined));
                i += 2;
            } else if is_last && i > 0 {
                let degree_carry = next.last().map_or(active[i].0, |&(d, _)| d);
                next.push((degree_carry, active[i].1));
                i += 1;
            } else {
                next.push(active[i]);
                i += 1;
            }
        }
        active = next;
    }

    let mut consumed = active[0].1;
    if can_fold {
        // res += X^degree · last_const (build `ct×ct`s + one `ct×pt`).
        consumed = consumed.max(power_basis_depth(degree) * input_log_delta + coeff_log_delta);
    }
    consumed
}

/// Degree-only replay of [`decompose_bsgs_coeffs`] collecting the flat baby-step
/// degree list (basis-independent: both monomial and Chebyshev factorizations
/// split at the same `next_power`). Honors the `MinDepth` leading-split recursion.
#[cfg(test)]
fn collect_baby_step_degrees(
    degree: usize,
    log_split: usize,
    max_degree: usize,
    lead: bool,
    split_leading: bool,
    out: &mut Vec<usize>,
) {
    let base = 1usize << log_split;
    if degree < base {
        if split_leading && should_split_leading_baby_step(degree, log_split, max_degree, lead) {
            let smaller = min_depth_split(bit_len(degree));
            if smaller < log_split {
                collect_baby_step_degrees(degree, smaller, max_degree, lead, split_leading, out);
                return;
            }
        }
        out.push(degree);
        return;
    }
    let mut next_power = base;
    while next_power < (degree >> 1) + 1 {
        next_power <<= 1;
    }
    collect_baby_step_degrees(next_power - 1, log_split, max_degree, false, split_leading, out);
    collect_baby_step_degrees(degree - next_power, log_split, max_degree, lead, split_leading, out);
}

/// Multiplicative depth (chained `ct×ct`) to build the power-basis element of
/// index `i`, via the same balanced [`split_degree`] recursion the evaluator
/// uses. `X¹` (the input) has depth 0.
#[cfg(test)]
fn power_basis_depth(i: usize) -> usize {
    if i <= 1 {
        0
    } else {
        let (a, b) = split_degree(i);
        power_basis_depth(a).max(power_basis_depth(b)) + 1
    }
}

pub fn bsgs_eval_depth(degree: usize, strategy: SplitStrategy) -> usize {
    if degree == 0 {
        return 0;
    }
    let b = bit_len(degree);
    match strategy {
        SplitStrategy::MinDepth => b,
        SplitStrategy::MinMult => {
            let threshold = (1usize << b) - (1usize << ((b - 1) / 2)) + 1;
            if degree >= threshold { b + 1 } else { b }
        }
    }
}

// ── Polynomial ───────────────────────────────────────────────────────────────

/// Affine change of basis `(u, w)` such that `y = u·x + w` maps an evaluation
/// point `x` in the approximation interval `[a, b]` to the variable the
/// coefficients of `basis` are expressed in.
///
/// - [`Basis::Monomial`]: identity — `(1, 0)` (coefficients are in `x` directly).
/// - [`Basis::Chebyshev`]: normalization of `[a, b]` onto the canonical `[-1, 1]`
///   — `u = 2/(b−a)`, `w = −(a+b)/(b−a)`, i.e. `y = (2x − a − b)/(b − a)`.
///
/// The pair lets a caller apply the remap to a ciphertext (`ct ← u·ct + w`)
/// before a homomorphic evaluation that assumes coefficients in the normalized
/// variable. See [`Polynomial::change_of_basis`].
pub fn change_of_basis<F: Float>(basis: Basis, a: F, b: F) -> (F, F) {
    match basis {
        Basis::Monomial => (F::one(), F::zero()),
        Basis::Chebyshev => {
            let two = F::one() + F::one();
            let span = b - a;
            (two / span, -(a + b) / span)
        }
    }
}

/// A plaintext polynomial with real coefficients.
///
/// `coeffs[i]` is the coefficient of the degree-`i` term (monomial basis) or
/// of `Tᵢ(x)` (Chebyshev basis).
pub struct Polynomial<F> {
    pub basis: Basis,
    pub coeffs: Vec<F>,
    pub parity: Parity,
    /// Lower bound of the interval `[a, b]` the approximation is valid over.
    pub a: F,
    /// Upper bound of the interval `[a, b]` the approximation is valid over.
    ///
    /// The coefficients are expressed in the variable `y = u·x + w` of
    /// [`change_of_basis`](Self::change_of_basis): for `Chebyshev` this maps
    /// `[a, b]` onto `[-1, 1]`, for `Monomial` it is the identity (so `[a, b]` is
    /// pure metadata). Defaults to `[-1, 1]`; set via
    /// [`with_interval`](Self::with_interval) or
    /// [`chebyshev_interpolate`](Self::chebyshev_interpolate).
    pub b: F,
}

impl<F> Polynomial<F>
where
    F: Float + FloatConst + FromPrimitive + Debug,
{
    /// Constructs a polynomial and auto-detects even/odd symmetry.
    pub fn new(basis: Basis, coeffs: Vec<F>) -> Self {
        let parity = if coeffs.iter().enumerate().all(|(i, &c)| i.is_multiple_of(2) || c == F::zero()) {
            Parity::Even
        } else if coeffs
            .iter()
            .enumerate()
            .all(|(i, &c)| !i.is_multiple_of(2) || c == F::zero())
        {
            Parity::Odd
        } else {
            Parity::Full
        };
        Self::new_with_parity(basis, coeffs, parity)
    }

    pub fn new_with_parity(basis: Basis, coeffs: Vec<F>, parity: Parity) -> Self {
        let one = F::one();
        Self {
            basis,
            coeffs,
            parity,
            a: -one,
            b: one,
        }
    }

    /// Sets the approximation interval `[a, b]` (see the [`a`](Self::a)/[`b`](Self::b)
    /// fields). For `Chebyshev` this is the domain remapped onto `[-1, 1]`; for
    /// `Monomial` it is metadata only.
    pub fn with_interval(mut self, a: F, b: F) -> Self {
        self.a = a;
        self.b = b;
        self
    }

    /// The approximation interval `[a, b]`.
    pub fn interval(&self) -> (F, F) {
        (self.a, self.b)
    }

    /// Affine change of basis `(u, w)` mapping `x ∈ [a, b]` to the variable the
    /// coefficients are expressed in (`y = u·x + w`). See the free function
    /// [`change_of_basis`].
    pub fn change_of_basis(&self) -> (F, F) {
        change_of_basis(self.basis, self.a, self.b)
    }

    pub fn chebyshev_interpolate<Fun>(degree: usize, a: F, b: F, f: Fun) -> Result<Self>
    where
        Fun: Fn(F) -> F,
    {
        chebyshev_interpolate(degree, a, b, f)
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Evaluates the polynomial at `x`.
    ///
    /// Uses Horner's method (monomial) or Clenshaw's algorithm (Chebyshev).
    /// For Chebyshev, `x` should lie in `[−1, 1]`.
    pub fn evaluate(&self, x: F) -> F {
        evaluate_coeffs(self.basis, &self.coeffs, x)
    }

    /// Evaluates this polynomial at `x` in its original interval `[a, b]`,
    /// applying [`change_of_basis`](Self::change_of_basis) first: identity for
    /// `Monomial`, the `[a,b]→[-1,1]` remap for `Chebyshev`.
    pub fn evaluate_on_interval(&self, x: F) -> F {
        let (u, w) = self.change_of_basis();
        self.evaluate(u * x + w)
    }

    /// Folds an even or odd Chebyshev polynomial through `u = T₂(x)`.
    ///
    /// For even `P`, returns `Q` such that `P(x) = Q(T₂(x))`. For odd `P`,
    /// returns `Q` such that `P(x) = x·Q(T₂(x))`. The returned polynomial is
    /// expressed on the canonical `u ∈ [-1, 1]` interval; the accompanying
    /// transform records how to recover the source polynomial.
    pub fn fold_chebyshev_t2(&self) -> Result<(Self, PolynomialInputTransform)> {
        ensure!(self.basis == Basis::Chebyshev, "T₂ folding requires a Chebyshev polynomial");

        match self.parity {
            Parity::Even => {
                ensure!(self.degree() >= 2, "T₂ folding an even polynomial requires degree ≥ 2");
                ensure!(
                    self.coeffs
                        .iter()
                        .enumerate()
                        .all(|(i, &c)| i.is_multiple_of(2) || c == F::zero()),
                    "even T₂ folding requires zero odd coefficients"
                );
                let coeffs = self.coeffs.iter().step_by(2).copied().collect();
                Ok((
                    Self::new_with_parity(Basis::Chebyshev, coeffs, Parity::Full),
                    PolynomialInputTransform::ChebyshevT2,
                ))
            }
            Parity::Odd => {
                ensure!(self.degree() >= 3, "T₂ folding an odd polynomial requires degree ≥ 3");
                ensure!(
                    self.coeffs
                        .iter()
                        .enumerate()
                        .all(|(i, &c)| !i.is_multiple_of(2) || c == F::zero()),
                    "odd T₂ folding requires zero even coefficients"
                );

                // Rₖ(u) = T₂ₖ₊₁(x) / x in the Chebyshev basis of
                // u = T₂(x). Build Rₖ with the recurrence
                // R₀ = 1, R₁ = 2T₁ - T₀,
                // Rₖ₊₁ = 2T₁ Rₖ - Rₖ₋₁.
                let odd_coeffs: Vec<F> = self.coeffs.iter().skip(1).step_by(2).copied().collect();
                let mut q = vec![F::zero(); odd_coeffs.len()];
                let mut r_prev = Vec::new();
                let mut r = vec![F::one()];
                let two = F::one() + F::one();

                for (k, &coefficient) in odd_coeffs.iter().enumerate() {
                    for (i, &value) in r.iter().enumerate() {
                        q[i] = q[i] + coefficient * value;
                    }
                    if k + 1 == odd_coeffs.len() {
                        break;
                    }

                    let mut next = if k == 0 {
                        vec![-F::one(), two]
                    } else {
                        let mut next = vec![F::zero(); r.len() + 1];
                        for (i, &value) in r.iter().enumerate() {
                            if i == 0 {
                                next[1] = next[1] + two * value;
                            } else {
                                next[i - 1] = next[i - 1] + value;
                                next[i + 1] = next[i + 1] + value;
                            }
                        }
                        for (i, &value) in r_prev.iter().enumerate() {
                            next[i] = next[i] - value;
                        }
                        next
                    };
                    std::mem::swap(&mut r_prev, &mut r);
                    std::mem::swap(&mut r, &mut next);
                }

                Ok((
                    Self::new_with_parity(Basis::Chebyshev, q, Parity::Full),
                    PolynomialInputTransform::ChebyshevT2TimesInput,
                ))
            }
            Parity::Full => Err(anyhow!("T₂ folding requires even or odd parity")),
        }
    }

    /// Decomposes this polynomial into a [`BSGSPolynomial`], encoding each
    /// baby-step coefficient slice with the scheme-supplied `encode` closure.
    pub fn decompose_bsgs_with<C>(
        &self,
        split_strategy: SplitStrategy,
        mut encode: impl FnMut(&[F]) -> Result<C>,
    ) -> Result<BSGSPolynomial<C>> {
        ensure!(self.degree() >= 1, "polynomial must have degree ≥ 1");

        let degree = self.degree();
        let log_split = split_for_strategy(split_strategy, degree, self.parity, self.basis);
        let base = 1usize << log_split;
        let split_leading = matches!(split_strategy, SplitStrategy::MinDepth);

        let mut baby_steps = Vec::new();
        decompose_bsgs_coeffs(
            self.basis,
            &self.coeffs,
            log_split,
            degree,
            true,
            split_leading,
            &mut |baby_coeffs| {
                baby_steps.push(encode(baby_coeffs)?);
                Ok(())
            },
        )?;

        Ok(BSGSPolynomial {
            basis: self.basis,
            degree,
            base,
            baby_steps,
            parity: self.parity,
            split_strategy,
            input_transform: PolynomialInputTransform::Identity,
            a: self.a.to_f64().expect("interval lower bound must convert to f64"),
            b: self.b.to_f64().expect("interval upper bound must convert to f64"),
        })
    }

    /// Folds this even/odd Chebyshev polynomial through `T₂`, then decomposes
    /// and encodes the lower-degree polynomial for BSGS evaluation.
    ///
    /// This is explicit because the odd transform costs a final ciphertext
    /// multiplication and can increase depth for some degrees. The returned
    /// decomposition carries the transform and includes its cost in
    /// [`BSGSPolynomial::eval_depth`] and [`BSGSPolynomial::consumed_bits`].
    pub fn decompose_bsgs_t2_with<C>(
        &self,
        split_strategy: SplitStrategy,
        encode: impl FnMut(&[F]) -> Result<C>,
    ) -> Result<BSGSPolynomial<C>> {
        let (folded, input_transform) = self.fold_chebyshev_t2()?;
        let mut bsgs = folded.decompose_bsgs_with(split_strategy, encode)?;
        bsgs.input_transform = input_transform;
        // The public interval still describes the source input. Evaluation first
        // maps that interval to [-1, 1], then applies the attached T₂ transform.
        bsgs.a = self.a.to_f64().expect("interval lower bound must convert to f64");
        bsgs.b = self.b.to_f64().expect("interval upper bound must convert to f64");
        Ok(bsgs)
    }
}

/// Evaluates the polynomial with coefficients `coeffs` (in `basis`) at `x`.
///
/// Horner's method (monomial) or Clenshaw's algorithm (Chebyshev); for
/// Chebyshev, `x` should lie in `[−1, 1]`. Operates on a borrowed slice so
/// callers (e.g. complex polynomials) can evaluate their components without
/// allocating intermediate [`Polynomial`]s.
pub fn evaluate_coeffs<F>(basis: Basis, coeffs: &[F], x: F) -> F
where
    F: Float,
{
    match basis {
        Basis::Monomial => {
            let mut y = F::zero();
            for &c in coeffs.iter().rev() {
                y = y * x + c;
            }
            y
        }
        Basis::Chebyshev => {
            let n = coeffs.len();
            if n == 0 {
                return F::zero();
            }
            if n == 1 {
                return coeffs[0];
            }
            let two = F::one() + F::one();
            let mut b2 = F::zero();
            let mut b1 = F::zero();
            for i in (1..n).rev() {
                let tmp = two * x * b1 - b2 + coeffs[i];
                b2 = b1;
                b1 = tmp;
            }
            coeffs[0] + x * b1 - b2
        }
    }
}

/// Returns the Chebyshev interpolation polynomial of degree `degree` for `f`
/// on `[a, b]`.
///
/// The coefficients are expressed in the normalized Chebyshev variable
/// `u = (2x-a-b)/(b-a)`. Use [`Polynomial::evaluate_on_interval`] for host
/// evaluation on the original interval, or evaluate homomorphically on an
/// input ciphertext that has already been mapped to `u`.
fn chebyshev_interpolate<F, Fun>(degree: usize, a: F, b: F, f: Fun) -> Result<Polynomial<F>>
where
    F: Float + FloatConst + FromPrimitive + Debug,
    Fun: Fn(F) -> F,
{
    ensure!(a < b, "chebyshev_interpolate: expected a < b");

    let n = degree + 1;
    let two = F::one() + F::one();
    let half = F::from_f64(0.5).expect("0.5 must be representable");
    let center = (a + b) * half;
    let radius = (b - a) * half;
    let pi_over_n = F::PI() / F::from_usize(n).expect("n must fit in scalar");

    let mut coeffs = vec![F::zero(); n];
    for k in (1..=n).rev() {
        let theta = (F::from_usize(k).expect("k must fit in scalar") - half) * pi_over_n;
        let u = theta.cos();
        let val = f(center + radius * u);
        let mut t_prev = F::one();
        let mut t = u;
        for coeff in coeffs.iter_mut() {
            *coeff = *coeff + val * t_prev;
            let t_next = two * u * t - t_prev;
            t_prev = t;
            t = t_next;
        }
    }

    let inv_n = F::one() / F::from_usize(n).expect("n must fit in scalar");
    coeffs[0] = coeffs[0] * inv_n;
    let two_over_n = two * inv_n;
    for coeff in coeffs.iter_mut().skip(1) {
        *coeff = *coeff * two_over_n;
    }

    Ok(Polynomial::new(Basis::Chebyshev, coeffs).with_interval(a, b))
}

// ── BSGSPolynomial ────────────────────────────────────────────────────────────

/// A polynomial decomposed for Baby-Step-Giant-Step (BSGS) evaluation.
///
/// `baby_steps[0]` is the lowest-degree encoded baby polynomial containing the constant and
/// low-degree terms; `baby_steps[n−1]` is the highest-degree encoded baby polynomial.
///
/// Construct via [`Polynomial::decompose_bsgs_with`].
pub struct BSGSPolynomial<C> {
    basis: Basis,
    degree: usize,
    base: usize,
    baby_steps: Vec<C>,
    parity: Parity,
    split_strategy: SplitStrategy,
    input_transform: PolynomialInputTransform,
    /// Approximation interval `[a, b]`, carried from the source [`Polynomial`]
    /// (stored as `f64`, decoupled from the erased coefficient type `C`). See
    /// [`change_of_basis`](Self::change_of_basis).
    a: f64,
    b: f64,
}

impl<BE: Backend, C> BSGSPolynomialInfos<BE> for BSGSPolynomial<C>
where
    C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + IntPolyInfos,
{
    type Coeffs = C;

    fn degree(&self) -> usize {
        BSGSPolynomial::degree(self)
    }

    fn baby_steps(&self) -> usize {
        BSGSPolynomial::baby_steps(self).len()
    }

    fn baby_step(&self, i: usize) -> &Self::Coeffs {
        BSGSPolynomial::baby_step(self, i)
    }

    fn basis(&self) -> Basis {
        BSGSPolynomial::basis(self)
    }

    fn parity(&self) -> Parity {
        BSGSPolynomial::parity(self)
    }

    fn log_split(&self) -> usize {
        BSGSPolynomial::log_split(self)
    }

    fn split_strategy(&self) -> SplitStrategy {
        self.split_strategy
    }

    fn input_transform(&self) -> PolynomialInputTransform {
        self.input_transform
    }
}

impl<C> BSGSPolynomial<C> {
    /// Returns the polynomial basis used by this decomposition.
    pub fn basis(&self) -> Basis {
        self.basis
    }

    /// Returns the degree encoded by this BSGS decomposition.
    ///
    /// This is half the source degree when [`Self::input_transform`] is a T₂
    /// fold.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Returns the baby-step base used by this decomposition.
    pub fn base(&self) -> usize {
        self.base
    }

    /// Returns the baby-step split as `log2(base)`.
    pub fn log_split(&self) -> usize {
        self.base.trailing_zeros() as usize
    }

    /// Number consecutives multiplications needed to evaluate this polynomial.
    pub fn eval_depth(&self) -> usize {
        bsgs_eval_depth(self.degree(), self.split_strategy) + self.input_transform.extra_depth()
    }

    /// `log_budget` bits consumed evaluating this polynomial on a ciphertext, as
    /// the longest (heaviest) chain through the BSGS data-dependency graph.
    ///
    /// `input_log_delta` is the scale of the input ciphertext / its powers
    /// (consumed by every `ct×ct` along the chain — power basis and giant steps);
    /// `coeff_log_delta` is the scale of the encoded polynomial coefficients
    /// (consumed by the single `ct×pt` baby-step inner product on the chain).
    ///
    /// The chain has `eval_depth` multiplications; `eval_depth - 1` of them are
    /// `ct×ct` (weight `input_log_delta`). The weight of the deepest level
    /// depends on whether a full-depth all-`ct×ct` chain exists:
    ///
    /// - **`MinDepth`, non-degenerate**: the recursion keeps the baby-step
    ///   `ct×pt` shallow, so the deepest chain is all `ct×ct` (a giant squaring);
    ///   the top weight is `max(input, coeff)` (the `coeff` only wins if it
    ///   exceeds `input`).
    /// - **`MinMult`, or the degenerate `MinDepth` case** where the leading chunk
    ///   is a bare constant (a power-of-two degree with a trailing-constant
    ///   split): the deepest chain includes the baby-step `ct×pt`, so the top
    ///   weight is `coeff`.
    pub fn consumed_bits(&self, input_log_delta: usize, coeff_log_delta: usize) -> usize {
        bsgs_consumed_bits(
            self.degree,
            self.split_strategy,
            self.parity,
            self.basis,
            input_log_delta,
            coeff_log_delta,
        ) + self.input_transform.extra_depth() * input_log_delta
    }

    /// Returns all encoded baby-step coefficient polynomials.
    pub fn baby_steps(&self) -> &[C] {
        &self.baby_steps
    }

    /// Returns one encoded baby-step coefficient polynomial.
    ///
    /// Panics if `i >= self.baby_steps().len()`.
    pub fn baby_step(&self, i: usize) -> &C {
        &self.baby_steps[i]
    }

    /// Returns the polynomial parity carried by this decomposition.
    pub fn parity(&self) -> Parity {
        self.parity
    }

    /// Returns the input rewrite required by this decomposition.
    pub fn input_transform(&self) -> PolynomialInputTransform {
        self.input_transform
    }

    /// The approximation interval `[a, b]` carried from the source polynomial.
    pub fn interval(&self) -> (f64, f64) {
        (self.a, self.b)
    }

    /// Affine change of basis `(u, w)` mapping `x ∈ [a, b]` to the coefficient
    /// variable (`y = u·x + w`). See the free function [`change_of_basis`].
    pub fn change_of_basis(&self) -> (f64, f64) {
        change_of_basis(self.basis, self.a, self.b)
    }

    /// Rebuilds this BSGS polynomial by mapping borrowed baby-step coefficients.
    pub fn map_baby_steps_ref<D>(&self, mut f: impl FnMut(&C) -> D) -> BSGSPolynomial<D> {
        BSGSPolynomial {
            basis: self.basis,
            degree: self.degree,
            base: self.base,
            baby_steps: self.baby_steps.iter().map(&mut f).collect(),
            parity: self.parity,
            split_strategy: self.split_strategy,
            input_transform: self.input_transform,
            a: self.a,
            b: self.b,
        }
    }
}

// ── BSGS evaluation data traits ───────────────────────────────────────────────

/// Per-operation semantic precision carried by a value during BSGS evaluation.
///
/// `log_budget` is the remaining homomorphic headroom and `log_delta` the
/// encoded scaling precision; `k = log_budget + log_delta`.
pub trait BSGSMeta {
    fn bsgs_log_budget(&self) -> usize;
    fn bsgs_log_delta(&self) -> usize;
}

/// Mutable semantic precision access.
pub trait SetBSGSMeta: BSGSMeta {
    fn set_bsgs_log_budget(&mut self, log_budget: usize);
    fn set_bsgs_log_delta(&mut self, log_delta: usize);
}

/// Read access to a decomposed BSGS polynomial during evaluation.
pub trait BSGSPolynomialInfos<BE: Backend> {
    type Coeffs: GLWEToBackendRef<BE> + IntPolyInfos + GLWEInfos + BSGSMeta;
    fn degree(&self) -> usize;
    fn baby_steps(&self) -> usize;
    fn baby_step(&self, i: usize) -> &Self::Coeffs;
    fn basis(&self) -> Basis;
    fn parity(&self) -> Parity;
    fn log_split(&self) -> usize;
    fn split_strategy(&self) -> SplitStrategy;
    fn input_transform(&self) -> PolynomialInputTransform {
        PolynomialInputTransform::Identity
    }
}

/// A single evaluated baby step with its degree.
pub trait BabyStep<BE: Backend> {
    type Value: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta;
    fn degree(&self) -> usize;
    fn get(&self) -> &Self::Value;
    fn get_mut(&mut self) -> &mut Self::Value;
}

// ── PowerBasis ────────────────────────────────────────────────────────────────

/// Read access to the pre-computed powers feeding a BSGS evaluation.
pub trait PowerBasisHelper<BE: Backend, A> {
    fn basis(&self) -> Basis;
    fn has_power(&self, power: usize) -> bool;
    fn get(&self, power: usize) -> Result<&A>;
}

/// Stores pre-computed powers of a ciphertext for BSGS polynomial evaluation.
///
/// `values[n]` = X^n (monomial basis) or Tₙ(X) (Chebyshev basis).
/// `values[1]` must be provided at construction time.
pub struct PowerBasis<A> {
    pub(crate) basis: Basis,
    pub(crate) values: HashMap<usize, A>,
}

impl<A> PowerBasis<A> {
    /// Creates a power basis with `x` treated as X (or T₁(X) for Chebyshev).
    pub fn new(basis: Basis, x: A) -> Self {
        let mut values = HashMap::new();
        values.insert(1, x);
        Self { basis, values }
    }

    /// Returns the polynomial basis represented by the stored powers.
    pub fn basis(&self) -> Basis {
        self.basis
    }

    /// Returns a reference to the stored power at degree `n`, if computed.
    pub fn get_stored(&self, n: usize) -> Option<&A> {
        self.values.get(&n)
    }

    /// Returns whether the power at degree `n` is stored.
    pub fn contains_power(&self, n: usize) -> bool {
        self.values.contains_key(&n)
    }

    /// Stores `value` as the power at degree `n`, replacing any existing entry.
    pub fn set_power(&mut self, n: usize, value: A) {
        self.values.insert(n, value);
    }

    /// Removes and returns a stored power.
    pub fn take_power(&mut self, n: usize) -> Option<A> {
        self.values.remove(&n)
    }
}

impl<BE: Backend, A> PowerBasisHelper<BE, A> for PowerBasis<A>
where
    A: GLWEToBackendRef<BE>,
{
    fn basis(&self) -> Basis {
        self.basis
    }

    fn has_power(&self, power: usize) -> bool {
        self.values.contains_key(&power)
    }

    fn get(&self, power: usize) -> Result<&A> {
        self.values
            .get(&power)
            .ok_or_else(|| anyhow!("PowerBasis: X^{power} not computed; call gen_power or populate first"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_baby_step_degrees(basis: Basis, degree: usize, log_split: usize, split_leading: bool) -> Vec<usize> {
        let coeffs = vec![0.0f64; degree + 1];
        let mut degrees = Vec::new();
        decompose_bsgs_coeffs(basis, &coeffs, log_split, degree, true, split_leading, &mut |s| {
            degrees.push(s.len() - 1);
            Ok(())
        })
        .unwrap();
        degrees
    }

    /// The closed-form [`bsgs_consumed_bits`] must equal the structure-exact
    /// schedule replay [`bsgs_consumed_bits_reference`] for every degree, both
    /// strategies, both bases, and a spread of `(Δ_in, Δ_coeff)` orderings
    /// (including `Δ_in < Δ_coeff`, equal, and zeros).
    #[test]
    fn consumed_bits_closed_form_matches_reference_full_parity() {
        let deltas = [(6, 3), (3, 6), (1, 1), (10, 1), (1, 10), (5, 5), (7, 2), (0, 5), (5, 0)];
        for strategy in [SplitStrategy::MinDepth, SplitStrategy::MinMult] {
            for basis in [Basis::Monomial, Basis::Chebyshev] {
                for degree in 2..=511usize {
                    for &(din, dco) in &deltas {
                        let got = bsgs_consumed_bits(degree, strategy, Parity::Full, basis, din, dco);
                        let want = bsgs_consumed_bits_reference(degree, strategy, Parity::Full, basis, din, dco);
                        assert_eq!(got, want, "degree {degree} {strategy:?} {basis:?} din={din} dco={dco}");
                    }
                }
            }
        }
    }

    /// Same, restricted to odd-degree odd-parity polynomials (the eval_mod
    /// sine/arcsine families) to confirm the closed form is parity-independent.
    #[test]
    fn consumed_bits_closed_form_matches_reference_odd_parity() {
        let deltas = [(6, 3), (3, 6), (1, 1), (10, 1), (1, 10)];
        for strategy in [SplitStrategy::MinDepth, SplitStrategy::MinMult] {
            for basis in [Basis::Monomial, Basis::Chebyshev] {
                for degree in (3..=511usize).step_by(2) {
                    for &(din, dco) in &deltas {
                        let got = bsgs_consumed_bits(degree, strategy, Parity::Odd, basis, din, dco);
                        let want = bsgs_consumed_bits_reference(degree, strategy, Parity::Odd, basis, din, dco);
                        assert_eq!(got, want, "degree {degree} {strategy:?} {basis:?} din={din} dco={dco}");
                    }
                }
            }
        }
    }

    #[test]
    fn change_of_basis_maps_interval_endpoints() {
        // Monomial: identity regardless of interval.
        assert_eq!(change_of_basis(Basis::Monomial, -3.0_f64, 7.0), (1.0, 0.0));

        // Chebyshev: y = u·x + w must send a → −1 and b → +1.
        for &(a, b) in &[(-1.0_f64, 1.0), (0.0, 2.0), (-8.0, 8.0), (3.0, 11.0)] {
            let (u, w) = change_of_basis(Basis::Chebyshev, a, b);
            assert!((u * a + w + 1.0).abs() < 1e-12, "a→-1 failed for [{a},{b}]");
            assert!((u * b + w - 1.0).abs() < 1e-12, "b→+1 failed for [{a},{b}]");
        }

        // Canonical Chebyshev domain is the identity.
        assert_eq!(change_of_basis(Basis::Chebyshev, -1.0_f64, 1.0), (1.0, 0.0));

        // The interval propagates Polynomial → BSGSPolynomial, and both expose the
        // same change of basis.
        let poly = Polynomial::chebyshev_interpolate(8, 0.0_f64, 4.0, |x| x).unwrap();
        assert_eq!(poly.interval(), (0.0, 4.0));
        let (u, w) = poly.change_of_basis();
        assert_eq!((u, w), change_of_basis(Basis::Chebyshev, 0.0, 4.0));
        let bsgs = poly
            .decompose_bsgs_with(SplitStrategy::MinDepth, |c| Ok::<_, anyhow::Error>(c.to_vec()))
            .unwrap();
        assert_eq!(bsgs.interval(), (0.0, 4.0));
        assert_eq!(bsgs.change_of_basis(), (u, w));
    }

    #[test]
    fn even_chebyshev_t2_fold_matches_source() {
        let poly = Polynomial::new_with_parity(
            Basis::Chebyshev,
            vec![1.0_f64, 0.0, -0.5, 0.0, 0.25, 0.0, 0.125],
            Parity::Even,
        );
        let (folded, transform) = poly.fold_chebyshev_t2().unwrap();

        assert_eq!(transform, PolynomialInputTransform::ChebyshevT2);
        assert_eq!(folded.coeffs, vec![1.0, -0.5, 0.25, 0.125]);
        for i in 0..=64 {
            let x = -1.0 + 2.0 * i as f64 / 64.0;
            assert!((poly.evaluate(x) - folded.evaluate(2.0 * x * x - 1.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn odd_chebyshev_t2_fold_matches_source() {
        let poly = Polynomial::new_with_parity(
            Basis::Chebyshev,
            vec![0.0_f64, 1.0, 0.0, -0.5, 0.0, 0.25, 0.0, -0.125],
            Parity::Odd,
        );
        let (folded, transform) = poly.fold_chebyshev_t2().unwrap();

        assert_eq!(transform, PolynomialInputTransform::ChebyshevT2TimesInput);
        for i in 0..=64 {
            let x = -1.0 + 2.0 * i as f64 / 64.0;
            assert!((poly.evaluate(x) - x * folded.evaluate(2.0 * x * x - 1.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn t2_bsgs_preserves_source_interval_and_accounts_for_transform() {
        let even = Polynomial::new_with_parity(Basis::Chebyshev, vec![1.0_f64, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0], Parity::Even)
            .with_interval(-3.0, 5.0);
        let even_bsgs = even
            .decompose_bsgs_t2_with(SplitStrategy::MinDepth, |c| Ok::<_, anyhow::Error>(c.to_vec()))
            .unwrap();
        assert_eq!(even_bsgs.input_transform(), PolynomialInputTransform::ChebyshevT2);
        assert_eq!(even_bsgs.degree(), 3);
        assert_eq!(even_bsgs.parity(), Parity::Full);
        assert_eq!(even_bsgs.interval(), (-3.0, 5.0));
        assert_eq!(even_bsgs.eval_depth(), bsgs_eval_depth(3, SplitStrategy::MinDepth) + 1);

        let odd = Polynomial::new_with_parity(
            Basis::Chebyshev,
            vec![0.0_f64, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0],
            Parity::Odd,
        );
        let odd_bsgs = odd
            .decompose_bsgs_t2_with(SplitStrategy::MinDepth, |c| Ok::<_, anyhow::Error>(c.to_vec()))
            .unwrap();
        assert_eq!(odd_bsgs.input_transform(), PolynomialInputTransform::ChebyshevT2TimesInput);
        assert_eq!(odd_bsgs.degree(), 3);
        assert_eq!(odd_bsgs.eval_depth(), bsgs_eval_depth(3, SplitStrategy::MinDepth) + 2);
        assert_eq!(
            odd_bsgs.consumed_bits(5, 3),
            bsgs_consumed_bits(3, SplitStrategy::MinDepth, Parity::Full, Basis::Chebyshev, 5, 3) + 10
        );
    }

    #[test]
    fn power_basis_can_transfer_ownership_of_a_power() {
        let mut powers = PowerBasis::new(Basis::Monomial, 1_u32);
        powers.set_power(2, 4);
        assert_eq!(powers.take_power(2), Some(4));
        assert!(!powers.contains_power(2));
    }

    #[test]
    fn min_mult_chebyshev_degree31_uniform_baby_steps() {
        let log_split = min_mult_split(31, Parity::Full, Basis::Chebyshev);
        assert_eq!(log_split, 3);
        assert_eq!(
            collect_baby_step_degrees(Basis::Chebyshev, 31, log_split, false),
            vec![7, 7, 7, 7]
        );
    }

    #[test]
    fn min_depth_chebyshev_degree31_splits_leading_baby_step() {
        let log_split = min_depth_split(bit_len(31));
        assert_eq!(log_split, 3);
        assert_eq!(
            collect_baby_step_degrees(Basis::Chebyshev, 31, log_split, true),
            vec![7, 7, 7, 3, 1, 1]
        );
    }

    #[test]
    fn bsgs_eval_depth_matches_closed_form() {
        assert_eq!(bsgs_eval_depth(0, SplitStrategy::MinDepth), 0);
        assert_eq!(bsgs_eval_depth(0, SplitStrategy::MinMult), 0);
        // MinDepth reaches `k = bit_len(d) = ceil(log2(d + 1))`. MinMult costs one
        // extra level on the upper part of each `[2^(k-1), 2^k)` band, where
        // `2^k - d <= 2^((k-1)/2) - 1`.
        for d in 1..1024 {
            let k = bit_len(d);
            let min_mult = if (1usize << k) - d < (1usize << ((k - 1) / 2)) {
                k + 1
            } else {
                k
            };
            assert_eq!(bsgs_eval_depth(d, SplitStrategy::MinDepth), k, "MinDepth degree {d}");
            assert_eq!(bsgs_eval_depth(d, SplitStrategy::MinMult), min_mult, "MinMult degree {d}");
        }
    }

    #[test]
    fn power_basis_estimate_caps_baby_steps_at_degree() {
        assert_eq!(estimate_power_basis_muls(5, 3, Parity::Full, Basis::Monomial), 4);
        assert_eq!(estimate_power_basis_muls(5, 3, Parity::Even, Basis::Monomial), 2);
    }
}
