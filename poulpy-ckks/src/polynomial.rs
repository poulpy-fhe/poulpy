use std::fmt::Debug;

use anyhow::{Result, anyhow, ensure};
use poulpy_core::layouts::{Base2K, GLWEInfos, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module};
use rand_distr::num_traits::{Float, FloatConst, FromPrimitive};

use crate::{
    CKKSInfos, CKKSMeta,
    api::BSGSPolynomialInfos,
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec},
};

// Re-export so callers can use `polynomial::Basis`/`Parity` without reaching into `api`.
pub use crate::api::{Basis, Parity};

// ── BSGS helpers ─────────────────────────────────────────────────────────────

/// Returns the BSGS log-split that minimises multiplication depth for a
/// polynomial of log-degree `log_degree`.
pub(crate) fn optimal_split(log_degree: usize) -> usize {
    debug_assert!(log_degree >= 1, "optimal_split requires log_degree ≥ 1");
    let s = log_degree >> 1;
    let a = (1 << s) + (1 << (log_degree - s)) + log_degree - s - 3;
    let b = (1 << (s + 1)) + (1 << (log_degree - s - 1)) + log_degree - s - 4;
    if a > b { s + 1 } else { s }
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
    pub fn encode_bsgs(
        &self,
        module: &Module<HostBytesBackend>,
        base2k: Base2K,
        coeff_meta: CKKSMeta,
    ) -> Result<BSGSPolynomial<CKKSPlaintext<Vec<u8>>>>
    where
        F: crate::layouts::CKKSScalar,
    {
        ensure!(self.degree() >= 1, "polynomial must have degree ≥ 1");

        let degree = self.degree();
        let log_degree = (usize::BITS - degree.leading_zeros()) as usize;
        let log_split = optimal_split(log_degree);
        let base = 1usize << log_split;

        let mut baby_steps = Vec::new();
        let mut step_idx = 0usize;
        decompose_bsgs_coeffs(self.basis, &self.coeffs, log_split, degree, true, &mut |baby_coeffs| {
            let mut pt = module.ckks_pt_coeffs_alloc(baby_coeffs.len(), base2k, coeff_meta);
            pt.encode_host_floats(baby_coeffs)
                .map_err(|e| anyhow!("encode_bsgs: step {step_idx}: {e}"))?;
            baby_steps.push(pt);
            step_idx += 1;
            Ok(())
        })?;

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

fn decompose_bsgs_coeffs<F>(
    basis: Basis,
    coeffs: &[F],
    log_split: usize,
    max_degree: usize,
    lead: bool,
    visit: &mut impl FnMut(&[F]) -> Result<()>,
) -> Result<()>
where
    F: Float,
{
    let degree = coeffs.len().saturating_sub(1);
    let base = 1usize << log_split;
    if degree < base {
        if should_split_leading_baby_step(degree, log_split, max_degree, lead) {
            let log_degree = bit_len(degree);
            let smaller_log_split = optimal_split(log_degree);
            if smaller_log_split < log_split {
                return decompose_bsgs_coeffs(basis, coeffs, smaller_log_split, max_degree, lead, visit);
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
            decompose_bsgs_coeffs(basis, &coeffs[..next_power], log_split, max_degree, false, visit)?;
            decompose_bsgs_coeffs(basis, &coeffs[next_power..], log_split, max_degree, lead, visit)
        }
        Basis::Chebyshev => {
            let (q, r) = factorize_coeffs_chebyshev(coeffs, next_power);
            decompose_bsgs_coeffs(basis, &r, log_split, max_degree, false, visit)?;
            decompose_bsgs_coeffs(basis, &q, log_split, max_degree, lead, visit)
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

    let mut q = vec![F::zero(); n];
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

    #[test]
    fn bsgs_decomposition_splits_leading_degree_close_to_next_power_of_two() {
        let degree = 31;
        let log_split = optimal_split(bit_len(degree));
        let coeffs = vec![0.0f64; degree + 1];

        let mut degrees = Vec::new();
        decompose_bsgs_coeffs(Basis::Chebyshev, &coeffs, log_split, degree, true, &mut |s| {
            degrees.push(s.len() - 1);
            Ok(())
        })
        .unwrap();

        assert_eq!(log_split, 3);
        assert_eq!(degrees, vec![7, 7, 7, 3, 1, 1]);
    }

    #[test]
    fn bsgs_decomposition_keeps_non_leading_baby_step_at_current_split() {
        let degree = 31;
        let log_split = optimal_split(bit_len(degree));
        let coeffs = vec![0.0f64; 8];

        let mut degrees = Vec::new();
        decompose_bsgs_coeffs(Basis::Chebyshev, &coeffs, log_split, degree, false, &mut |s| {
            degrees.push(s.len() - 1);
            Ok(())
        })
        .unwrap();

        assert_eq!(degrees, vec![7]);
    }
}
