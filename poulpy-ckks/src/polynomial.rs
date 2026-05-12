use std::{collections::HashMap, fmt::Debug};

use anyhow::{Result, anyhow, ensure};
use poulpy_core::layouts::{
    Base2K, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, Data, HostBytesBackend, Module, ScratchArena};
use rand_distr::num_traits::{Float, FloatConst, FromPrimitive};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos,
    api::{BSGSPolynomialInfos, CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSSubOps, PowerBasisHelper},
    checked_mul_ct_log_budget,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec},
};

// Re-export so callers can use `polynomial::Basis` without reaching into `api`.
pub use crate::api::Basis;

// ── BSGS helpers ─────────────────────────────────────────────────────────────

/// Splits `n` into `(a, b)` with `n = a + b` and `|a – b|` minimised.
///
/// When `n` is a power of two `a = b = n/2`; otherwise uses the
/// Lee et al. (2020) strategy that maximises the number of odd-degree
/// Chebyshev terms.
pub fn split_degree(n: usize) -> (usize, usize) {
    if n.is_power_of_two() {
        (n / 2, n / 2)
    } else {
        let k = (usize::BITS - (n - 1).leading_zeros()) as usize - 1;
        let a = (1usize << k) - 1;
        let b = n + 1 - (1usize << k);
        (a, b)
    }
}

/// Returns the BSGS log-split that minimises multiplication depth for a
/// polynomial of log-degree `log_degree`.
pub fn optimal_split(log_degree: usize) -> usize {
    let s = (log_degree >> 1) as i64;
    let d = log_degree as i64;
    let a = (1i64 << s) + (1i64 << (d - s)) + d - s - 3;
    let b = (1i64 << (s + 1)) + (1i64 << (d - s - 1)) + d - s - 4;
    if a > b { (s + 1) as usize } else { s as usize }
}

// ── Polynomial ───────────────────────────────────────────────────────────────

/// A plaintext polynomial with real coefficients.
///
/// `coeffs[i]` is the coefficient of the degree-`i` term (monomial basis) or
/// of `Tᵢ(x)` (Chebyshev basis).
pub struct Polynomial<F = f64> {
    pub basis: Basis,
    pub coeffs: Vec<F>,
    /// `true` when all odd-degree coefficients are zero.
    pub is_even: bool,
    /// `true` when all even-degree coefficients are zero.
    pub is_odd: bool,
}

impl<F> Polynomial<F>
where
    F: Float,
{
    /// Constructs a polynomial and auto-detects even/odd symmetry.
    pub fn new(basis: Basis, coeffs: Vec<F>) -> Self {
        let is_even = coeffs.iter().enumerate().all(|(i, &c)| i % 2 == 0 || c == F::zero());
        let is_odd = coeffs.iter().enumerate().all(|(i, &c)| i % 2 != 0 || c == F::zero());
        Self {
            basis,
            coeffs,
            is_even,
            is_odd,
        }
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
    /// can be passed directly to `ckks_eval_poly_const_coeffs`.
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

        let decomposed = decompose_bsgs_coeffs(self.basis, &self.coeffs, log_split, degree, true);
        let n_steps = decomposed.len();
        let mut baby_steps = Vec::with_capacity(n_steps);
        let mut baby_degrees = Vec::with_capacity(n_steps);

        for (step_idx, (baby_degree, baby_coeffs)) in decomposed.into_iter().enumerate() {
            // Use a full-ring plaintext so the encoded baby polynomials can be
            // uploaded and multiplied on any backend without limb-shape fixes.
            let mut pt = module.ckks_pt_vec_alloc(base2k, coeff_meta);
            let mut padded = vec![F::zero(); pt.n().as_usize()];
            padded[..baby_coeffs.len()].copy_from_slice(&baby_coeffs);

            pt.encode_host_floats(&padded)
                .map_err(|e| anyhow!("encode_bsgs: step {step_idx}: {e}"))?;

            baby_degrees.push(baby_degree);
            baby_steps.push(pt);
        }

        Ok(BSGSPolynomial {
            basis: self.basis,
            degree,
            base,
            baby_degrees,
            baby_steps,
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
pub fn chebyshev_interpolate<F, Fun>(degree: usize, a: F, b: F, f: Fun) -> Result<Polynomial<F>>
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

    let mut nodes = vec![F::zero(); n];
    let mut values = vec![F::zero(); n];
    for k in 1..=n {
        let theta = (F::from_usize(k).expect("k must fit in scalar") - half) * pi_over_n;
        let x = center + radius * theta.cos();
        // Match Lattigo's ascending node order.
        let idx = n - k;
        nodes[idx] = x;
        values[idx] = f(x);
    }

    let mut coeffs = vec![F::zero(); n];
    for i in 0..n {
        let u = (two * nodes[i] - a - b) / (b - a);
        let mut t_prev = F::one();
        let mut t = u;
        for coeff in coeffs.iter_mut() {
            *coeff = *coeff + values[i] * t_prev;
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

fn decompose_bsgs_coeffs<F>(basis: Basis, coeffs: &[F], log_split: usize, max_degree: usize, lead: bool) -> Vec<(usize, Vec<F>)>
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
                return decompose_bsgs_coeffs(basis, coeffs, smaller_log_split, max_degree, lead);
            }
        }
        return vec![(degree, coeffs.to_vec())];
    }

    let mut next_power = base;
    while next_power < (degree >> 1) + 1 {
        next_power <<= 1;
    }

    let (q, r) = factorize_coeffs(basis, coeffs, next_power);
    let mut steps = decompose_bsgs_coeffs(basis, &r, log_split, max_degree, false);
    steps.extend(decompose_bsgs_coeffs(basis, &q, log_split, max_degree, lead));
    steps
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

fn factorize_coeffs<F>(basis: Basis, coeffs: &[F], n: usize) -> (Vec<F>, Vec<F>)
where
    F: Float,
{
    let degree = coeffs.len().saturating_sub(1);
    let mut r = vec![F::zero(); n];
    r.copy_from_slice(&coeffs[..n]);

    let mut q = vec![F::zero(); n];
    q[0] = coeffs[n];

    match basis {
        Basis::Monomial => {
            for i in n + 1..=degree {
                q[i - n] = coeffs[i];
            }
        }
        Basis::Chebyshev => {
            let two = F::one() + F::one();
            for (i, j) in ((n + 1)..=degree).zip(1..) {
                q[i - n] = two * coeffs[i];
                r[n - j] = r[n - j] - coeffs[i];
            }
        }
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
    pub baby_degrees: Vec<usize>,
    pub baby_steps: Vec<C>,
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

    fn baby_degree(&self, i: usize) -> usize {
        self.baby_degrees[i]
    }

    fn baby_step(&self, i: usize) -> &Self::Coeffs {
        &self.baby_steps[i]
    }

    fn basis(&self) -> Basis {
        self.basis
    }
}

// ── PowerBasis ────────────────────────────────────────────────────────────────

/// Stores pre-computed powers of a ciphertext for BSGS polynomial evaluation.
///
/// `values[n]` = X^n (monomial basis) or Tₙ(X) (Chebyshev basis).
/// `values[1]` must be provided at construction time.
///
/// Implements [`PowerBasisHelper`] so it can be passed directly to
/// `ckks_eval_poly_const_coeffs`.
pub struct PowerBasis<A> {
    pub basis: Basis,
    values: HashMap<usize, A>,
}

impl<A> PowerBasis<A> {
    /// Creates a power basis with `x` treated as X (or T₁(X) for Chebyshev).
    pub fn new(basis: Basis, x: A) -> Self {
        let mut values = HashMap::new();
        values.insert(1, x);
        Self { basis, values }
    }

    /// Returns a reference to the stored power at degree `n`, if computed.
    pub fn get_stored(&self, n: usize) -> Option<&A> {
        self.values.get(&n)
    }

    /// Inserts a pre-computed power at degree `n`.
    pub fn insert(&mut self, n: usize, value: A) {
        self.values.insert(n, value);
    }
}

impl<BE: Backend, A> PowerBasisHelper<BE, A> for PowerBasis<A>
where
    A: GLWEToBackendRef<BE>,
{
    fn get(&self, power: usize) -> Result<&A> {
        self.values
            .get(&power)
            .ok_or_else(|| anyhow!("PowerBasis: X^{power} not computed; call gen_power or populate first"))
    }
}

impl<D: Data> PowerBasis<CKKSCiphertext<D>> {
    /// Recursively computes and stores X^`n` using `split_degree` to choose the
    /// multiplication tree: X^n = X^a · X^b where `split_degree(n) = (a, b)`.
    pub fn gen_power<BE>(
        &mut self,
        n: usize,
        module: &Module<BE>,
        tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<D, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: Backend<OwnedBuf = D>,
        Module<BE>: CKKSMulOps<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        poulpy_core::layouts::GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ensure!(
            self.basis == Basis::Monomial,
            "PowerBasis::gen_power only supports the monomial basis; use gen_power_chebyshev for Chebyshev"
        );

        if self.values.contains_key(&n) {
            return Ok(());
        }

        ensure!(n >= 2, "gen_power: n={n} < 2; X^1 must be provided at construction");

        let (a, b) = split_degree(n);
        self.gen_power(a, module, tsk, scratch)?;
        self.gen_power(b, module, tsk, scratch)?;

        // Hold immutable borrows only inside this block; insert afterwards.
        let result = {
            let a_val = self.values.get(&a).expect("gen_power(a) just succeeded");
            let b_val = self.values.get(&b).expect("gen_power(b) just succeeded");
            let k = mul_ct_effective_k(a_val, b_val)?;
            let mut r = module.ckks_ciphertext_alloc(a_val.base2k(), k.into());
            module.ckks_mul_into(&mut r, a_val, b_val, tsk, scratch)?;
            r
        };
        self.values.insert(n, result);
        Ok(())
    }

    /// Recursively computes and stores `T_n(X)` for the Chebyshev basis.
    ///
    /// Generates the plaintext `T_0 = 1` term on demand for
    /// `T_{a+b}(X) = 2*T_a(X)*T_b(X) - T_{|a-b|}(X)`.
    pub fn gen_power_chebyshev<BE>(
        &mut self,
        n: usize,
        module: &Module<BE>,
        tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<D, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: Backend<OwnedBuf = D>,
        Module<BE>: CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSSubOps<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        poulpy_core::layouts::GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ensure!(
            self.basis == Basis::Chebyshev,
            "gen_power_chebyshev requires a Chebyshev PowerBasis"
        );

        if self.values.contains_key(&n) {
            return Ok(());
        }

        ensure!(n >= 2, "gen_power_chebyshev: n={n} < 2; T_1 must be provided at construction");

        let (a, b) = split_degree(n);
        self.gen_power_chebyshev(a, module, tsk, scratch)?;
        self.gen_power_chebyshev(b, module, tsk, scratch)?;

        let c = a.abs_diff(b);
        if c != 0 {
            self.gen_power_chebyshev(c, module, tsk, scratch)?;
        }

        let result = {
            let a_val = self.values.get(&a).expect("gen_power_chebyshev(a) just succeeded");
            let b_val = self.values.get(&b).expect("gen_power_chebyshev(b) just succeeded");
            let k = mul_ct_effective_k(a_val, b_val)?;
            let mut product = module.ckks_ciphertext_alloc(a_val.base2k(), k.into());
            module.ckks_mul_into(&mut product, a_val, b_val, tsk, scratch)?;

            let mut doubled = module.ckks_ciphertext_alloc(product.base2k(), product.effective_k().into());
            module.ckks_add_into(&mut doubled, &product, &product, scratch)?;

            if c == 0 {
                module.ckks_sub_one_assign(&mut doubled, scratch)?;
            } else {
                let c_val = self.values.get(&c).expect("gen_power_chebyshev(c) just succeeded");
                module.ckks_sub_assign(&mut doubled, c_val, scratch)?;
            }

            compact_power_ct(module, &doubled, scratch)?
        };

        self.values.insert(n, result);
        Ok(())
    }

    /// Pre-computes all powers required to evaluate a polynomial of the given
    /// `degree` using BSGS with the Monomial basis.
    ///
    /// Populates powers of two up to `2^(⌈log₂ degree⌉−1)` and all
    /// intermediate powers from 2 up to `(2^log_split) − 1`.
    pub fn populate<BE>(
        &mut self,
        degree: usize,
        module: &Module<BE>,
        tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<D, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: Backend<OwnedBuf = D>,
        Module<BE>: CKKSMulOps<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        poulpy_core::layouts::GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ensure!(
            self.basis == Basis::Monomial,
            "PowerBasis::populate only supports the monomial basis; use populate_chebyshev for Chebyshev"
        );
        ensure!(degree >= 1, "populate: degree must be ≥ 1");

        let log_degree = (usize::BITS - degree.leading_zeros()) as usize;
        let log_split = optimal_split(log_degree);

        // Largest power of two needed (also computes all smaller powers of two
        // recursively via split_degree).
        let largest_pow2 = 1usize << (log_degree - 1);
        if largest_pow2 >= 2 {
            self.gen_power(largest_pow2, module, tsk, scratch)?;
        }

        // Intermediate powers from base−1 down to 3 (2 is computed transitively).
        let base = 1usize << log_split;
        for i in (3..base).rev() {
            self.gen_power(i, module, tsk, scratch)?;
        }

        Ok(())
    }

    /// Pre-computes all Chebyshev powers required to evaluate a polynomial of
    /// the given `degree` using the BSGS evaluator.
    pub fn populate_chebyshev<BE>(
        &mut self,
        degree: usize,
        module: &Module<BE>,
        tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<D, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: Backend<OwnedBuf = D>,
        Module<BE>: CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSSubOps<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        poulpy_core::layouts::GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ensure!(
            self.basis == Basis::Chebyshev,
            "populate_chebyshev requires a Chebyshev PowerBasis"
        );
        ensure!(degree >= 1, "populate_chebyshev: degree must be ≥ 1");

        let log_degree = (usize::BITS - degree.leading_zeros()) as usize;
        let log_split = optimal_split(log_degree);

        let largest_pow2 = 1usize << (log_degree - 1);
        if largest_pow2 >= 2 {
            self.gen_power_chebyshev(largest_pow2, module, tsk, scratch)?;
        }

        let base = 1usize << log_split;
        for i in (3..base).rev() {
            self.gen_power_chebyshev(i, module, tsk, scratch)?;
        }

        Ok(())
    }
}

fn compact_power_ct<M, S, BE: Backend>(
    module: &M,
    src: &S,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<CKKSCiphertext<BE::OwnedBuf>>
where
    M: CKKSCopyOps<BE> + CKKSModuleAlloc<BE>,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE>,
{
    let mut compact = module.ckks_ciphertext_alloc(src.base2k(), src.effective_k().into());
    module.ckks_copy(&mut compact, src, scratch)?;
    Ok(compact)
}

fn mul_ct_effective_k<A, B>(a: &A, b: &B) -> Result<usize>
where
    A: GLWEInfos + CKKSInfos,
    B: GLWEInfos + CKKSInfos,
{
    let log_budget = checked_mul_ct_log_budget("power_basis", a.log_budget(), b.log_budget(), a.log_delta(), b.log_delta())?;
    Ok(log_budget + a.log_delta().min(b.log_delta()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bsgs_decomposition_splits_leading_degree_close_to_next_power_of_two() {
        let degree = 31;
        let log_split = optimal_split(bit_len(degree));
        let coeffs = vec![0.0f64; degree + 1];

        let steps = decompose_bsgs_coeffs(Basis::Chebyshev, &coeffs, log_split, degree, true);
        let degrees: Vec<usize> = steps.iter().map(|(degree, _)| *degree).collect();

        assert_eq!(log_split, 3);
        assert_eq!(degrees, vec![7, 7, 7, 3, 1, 1]);
    }

    #[test]
    fn bsgs_decomposition_keeps_non_leading_baby_step_at_current_split() {
        let degree = 31;
        let log_split = optimal_split(bit_len(degree));
        let coeffs = vec![0.0f64; 8];

        let steps = decompose_bsgs_coeffs(Basis::Chebyshev, &coeffs, log_split, degree, false);
        let degrees: Vec<usize> = steps.iter().map(|(degree, _)| *degree).collect();

        assert_eq!(degrees, vec![7]);
    }
}
