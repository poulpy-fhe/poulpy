use std::fmt::Debug;

use anyhow::{Result, anyhow, ensure};
use poulpy_core::layouts::{Base2K, GLWEInfos, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module};
use rand_distr::num_traits::{Float, FloatConst, FromPrimitive};

use crate::{
    CKKSInfos, CKKSMeta,
    api::BSGSPolynomialInfos,
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec},
    power_basis::split_degree,
};

// Re-export so callers can use `polynomial::Basis`/`Parity` without reaching into `api`.
pub use crate::api::{Basis, Parity};

// ── BSGS helpers ─────────────────────────────────────────────────────────────

/// Chooses how `Polynomial::encode_bsgs` picks `log_split`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitStrategy {
    /// Closed-form choice minimising multiplicative depth.
    MinDepth,
    /// Sweep `log_split` to minimise `(CT-CT, PT-CT)` lexicographically.
    MinMult,
}

/// Default planner picked by [`Polynomial::encode_bsgs`].
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
    match parity {
        Parity::Even => targets.extend((4..base).step_by(2)),
        Parity::Odd => targets.extend((3..base).step_by(2)),
        Parity::Full => targets.extend(3..base),
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
        match self.basis {
            Basis::Monomial => {
                let mut y = F::zero();
                for &c in self.coeffs.iter().rev() {
                    y = y * x + c;
                }
                y
            }
            Basis::Chebyshev => {
                let n = self.coeffs.len();
                if n == 0 {
                    return F::zero();
                }
                if n == 1 {
                    return self.coeffs[0];
                }
                let two = F::one() + F::one();
                let mut b2 = F::zero();
                let mut b1 = F::zero();
                for i in (1..n).rev() {
                    let tmp = two * x * b1 - b2 + self.coeffs[i];
                    b2 = b1;
                    b1 = tmp;
                }
                self.coeffs[0] + x * b1 - b2
            }
        }
    }

    /// Evaluates this polynomial on an input interval.
    ///
    /// Monomial polynomials are evaluated directly at `x`. Chebyshev
    /// polynomials first map `x` from `[a, b]` to the normalized Chebyshev
    /// variable `(2x-a-b)/(b-a)`.
    pub fn evaluate_on_interval(&self, x: F, a: F, b: F) -> F {
        match self.basis {
            Basis::Monomial => self.evaluate(x),
            Basis::Chebyshev => {
                let two = F::one() + F::one();
                self.evaluate((two * x - a - b) / (b - a))
            }
        }
    }

    /// Decomposes this polynomial into a BSGS representation and encodes its
    /// coefficients as a `CKKSPlaintext`.
    ///
    /// The returned [`BSGSPolynomial`] implements [`BSGSPolynomialInfos`] and
    /// can be passed directly to `ckks_eval_poly_real_const_coeffs_from_power_basis`.
    ///
    /// `module` is used only for host-side encoding; it does not need to match
    /// the compute backend.
    ///
    /// Uses [`DEFAULT_SPLIT_STRATEGY`]; call [`Self::encode_bsgs_with`] to
    /// pick the strategy explicitly.
    pub fn encode_bsgs(
        &self,
        module: &Module<HostBytesBackend>,
        base2k: Base2K,
        coeff_meta: CKKSMeta,
    ) -> Result<BSGSPolynomial<CKKSPlaintext<Vec<u8>>>>
    where
        F: crate::layouts::CKKSScalar,
    {
        self.encode_bsgs_with(module, base2k, coeff_meta, DEFAULT_SPLIT_STRATEGY)
    }

    /// Same as [`Self::encode_bsgs`] with an explicit [`SplitStrategy`].
    pub fn encode_bsgs_with(
        &self,
        module: &Module<HostBytesBackend>,
        base2k: Base2K,
        coeff_meta: CKKSMeta,
        strategy: SplitStrategy,
    ) -> Result<BSGSPolynomial<CKKSPlaintext<Vec<u8>>>>
    where
        F: crate::layouts::CKKSScalar,
    {
        ensure!(self.degree() >= 1, "polynomial must have degree ≥ 1");

        let degree = self.degree();
        let log_split = split_for_strategy(strategy, degree, self.parity, self.basis);
        let base = 1usize << log_split;
        let split_leading = matches!(strategy, SplitStrategy::MinDepth);

        let mut baby_steps = Vec::new();
        let mut step_idx = 0usize;
        decompose_bsgs_coeffs(
            self.basis,
            &self.coeffs,
            log_split,
            degree,
            true,
            split_leading,
            &mut |baby_coeffs| {
                let mut pt = module.ckks_pt_coeffs_alloc(baby_coeffs.len(), base2k, coeff_meta);
                pt.encode_host_floats(baby_coeffs)
                    .map_err(|e| anyhow!("encode_bsgs: step {step_idx}: {e}"))?;
                baby_steps.push(pt);
                step_idx += 1;
                Ok(())
            },
        )?;

        Ok(BSGSPolynomial {
            basis: self.basis,
            degree,
            base,
            baby_steps,
            parity: self.parity,
        })
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

// ── BSGSPolynomial ────────────────────────────────────────────────────────────

/// A polynomial decomposed for Baby-Step-Giant-Step (BSGS) evaluation.
///
/// `baby_steps[0]` is the lowest-degree encoded baby polynomial containing the constant and
/// low-degree terms; `baby_steps[n−1]` is the highest-degree encoded baby polynomial.
///
/// Construct via [`Polynomial::encode_bsgs`].
pub struct BSGSPolynomial<C> {
    pub basis: Basis,
    pub degree: usize,
    pub base: usize,
    pub baby_steps: Vec<C>,
    pub parity: Parity,
}

impl<BE: Backend, C> BSGSPolynomialInfos<BE> for BSGSPolynomial<C>
where
    C: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
{
    type Coeffs = C;

    fn degree(&self) -> usize {
        self.degree
    }

    fn baby_steps(&self) -> usize {
        self.baby_steps.len()
    }

    fn baby_step(&self, i: usize) -> &Self::Coeffs {
        &self.baby_steps[i]
    }

    fn basis(&self) -> Basis {
        self.basis
    }

    fn parity(&self) -> Parity {
        self.parity
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
}
