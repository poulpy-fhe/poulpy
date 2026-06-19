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
pub(crate) fn bsgs_eval_depth(degree: usize, strategy: SplitStrategy) -> usize {
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

/// A plaintext polynomial with real coefficients.
///
/// `coeffs[i]` is the coefficient of the degree-`i` term (monomial basis) or
/// of `Tᵢ(x)` (Chebyshev basis).
pub struct Polynomial<F> {
    pub basis: Basis,
    pub coeffs: Vec<F>,
    pub parity: Parity,
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
        Self { basis, coeffs, parity }
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

    /// Evaluates this polynomial on an input interval.
    ///
    /// Monomial polynomials are evaluated directly at `x`. Chebyshev
    /// polynomials first map `x` from `[a, b]` to the normalized Chebyshev
    /// variable `(2x-a-b)/(b-a)`.
    pub fn evaluate_on_interval(&self, x: F, a: F, b: F) -> F {
        assert!(a < b);
        match self.basis {
            Basis::Monomial => self.evaluate(x),
            Basis::Chebyshev => {
                let two = F::one() + F::one();
                self.evaluate((two * x - a - b) / (b - a))
            }
        }
    }

    /// Decomposes this polynomial into a [`BSGSPolynomial`], encoding each
    /// baby-step coefficient slice with the scheme-supplied `encode` closure.
    pub fn decompose_bsgs_with<C>(
        &self,
        strategy: SplitStrategy,
        mut encode: impl FnMut(&[F]) -> Result<C>,
    ) -> Result<BSGSPolynomial<C>> {
        ensure!(self.degree() >= 1, "polynomial must have degree ≥ 1");

        let degree = self.degree();
        let log_split = split_for_strategy(strategy, degree, self.parity, self.basis);
        let base = 1usize << log_split;
        let split_leading = matches!(strategy, SplitStrategy::MinDepth);

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
            eval_depth: bsgs_eval_depth(degree, strategy),
        })
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
    // Iterate nodes in ascending x order (k = n..1) to match Lattigo's convention.
    // u = cos(theta_k) is the normalized Chebyshev variable; x is only needed for f.
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

    Ok(Polynomial::new(Basis::Chebyshev, coeffs))
}

// ── BSGSPolynomial ────────────────────────────────────────────────────────────

/// A polynomial decomposed for Baby-Step-Giant-Step (BSGS) evaluation.
///
/// `baby_steps[0]` is the lowest-degree encoded baby polynomial containing the constant and
/// low-degree terms; `baby_steps[n−1]` is the highest-degree encoded baby polynomial.
///
/// Construct via [`Polynomial::decompose_bsgs_with`].
pub struct BSGSPolynomial<C> {
    pub(crate) basis: Basis,
    pub(crate) degree: usize,
    pub(crate) base: usize,
    pub(crate) baby_steps: Vec<C>,
    pub(crate) parity: Parity,
    /// Multiplicative depth this decomposition consumes, computed from the
    /// `SplitStrategy` at decomposition time (see [`bsgs_eval_depth`]).
    pub(crate) eval_depth: usize,
}

impl<BE: Backend, C> BSGSPolynomialInfos<BE> for BSGSPolynomial<C>
where
    C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
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
}

impl<C> BSGSPolynomial<C> {
    /// Returns the polynomial basis used by this decomposition.
    pub fn basis(&self) -> Basis {
        self.basis
    }

    /// Returns the original polynomial degree.
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

    /// Multiplicative depth (number of CT-CT multiplication levels) a BSGS
    /// evaluation of this polynomial consumes.
    ///
    /// Computed from the [`SplitStrategy`] at decomposition time, so it is exact
    /// for any strategy. In particular it is **not** simply `ceil(log2(degree))`:
    /// a `MinMult` split can cost one level more than the depth-optimal `MinDepth`
    /// split.
    pub fn eval_depth(&self) -> usize {
        self.eval_depth
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

    /// Rebuilds this BSGS polynomial by mapping borrowed baby-step coefficients.
    pub fn map_baby_steps_ref<D>(&self, mut f: impl FnMut(&C) -> D) -> BSGSPolynomial<D> {
        BSGSPolynomial {
            basis: self.basis,
            degree: self.degree,
            base: self.base,
            baby_steps: self.baby_steps.iter().map(&mut f).collect(),
            parity: self.parity,
            eval_depth: self.eval_depth,
        }
    }
}

// ── BSGS evaluation data traits ───────────────────────────────────────────────

/// Per-operation semantic precision carried by a value during BSGS evaluation.
///
/// `log_budget` is the remaining homomorphic headroom and `log_delta` the
/// encoded scaling precision; `effective_k = log_budget + log_delta`.
pub trait BSGSMeta {
    fn bsgs_log_budget(&self) -> usize;
    fn bsgs_log_delta(&self) -> usize;
    fn bsgs_effective_k(&self) -> usize {
        self.bsgs_log_budget() + self.bsgs_log_delta()
    }
}

/// Mutable semantic precision access.
pub trait SetBSGSMeta: BSGSMeta {
    fn set_bsgs_log_budget(&mut self, log_budget: usize);
    fn set_bsgs_log_delta(&mut self, log_delta: usize);
}

/// Read access to a decomposed BSGS polynomial during evaluation.
pub trait BSGSPolynomialInfos<BE: Backend> {
    type Coeffs: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta;
    fn degree(&self) -> usize;
    fn baby_steps(&self) -> usize;
    fn baby_step(&self, i: usize) -> &Self::Coeffs;
    fn basis(&self) -> Basis;
    fn parity(&self) -> Parity;
    fn log_split(&self) -> usize;
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
