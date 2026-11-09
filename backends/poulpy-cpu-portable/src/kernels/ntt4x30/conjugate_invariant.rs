//! Conjugate-invariant NTT4x30 arithmetic: the negacyclic NTT behind a basis change.
//!
//! With `alpha` a primitive `4n`-th root of unity, `i = alpha^n` and `f[k] = alpha^-k`, the transform of degree `n` changes
//! `a` into `out[j] = f[j] (a[j] - i a[n - j])` before the negacyclic NTT, and back with
//! `out[j] = (i f[n - j] a[j] - f[j] a[n - j]) / 2` after the inverse one, for `j` in `1..n`, keeping `out[0] = a[0]`.
//! Every degree holds its own table `f`, and each pair `(j, n - j)` is changed jointly: two products per output.

use std::{marker::PhantomData, sync::Arc};

use poulpy_hal::{
    AlignedVec, alloc_aligned,
    layouts::{ConjugateInvariant, Standard},
};

use crate::kernels::ntt4x30::{
    ntt::{NttTable, NttTableInv, intt_core, ntt_core},
    primes::PrimeSetCrt4,
    standard,
    vec_znx_dft::{NttAutomorphismPlan, NttPlan, NttPlanNew},
};

/// `a * b mod q`.
fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

/// `a^e mod q`.
fn pow_mod(mut a: u64, mut e: u64, q: u64) -> u64 {
    let mut r = 1;
    while e != 0 {
        if e & 1 != 0 {
            r = mul_mod(r, a, q);
        }
        a = mul_mod(a, a, q);
        e >>= 1;
    }
    r
}

/// The primitive `4n`-th root of unity `omega^(2^(max_log_n - 1) / n)` of a primitive `2^(max_log_n + 1)`-th root `omega`.
fn root(n: usize, q: u64, omega: u64, max_log_n: u32) -> u64 {
    assert!(
        n.is_power_of_two() && n <= 1 << (max_log_n - 1),
        "the degree must be a power of two within the root order"
    );
    pow_mod(omega, (1u64 << (max_log_n - 1)) / n as u64, q)
}

/// Shoup product `x * w mod q` in `[0, 2q)` for any `x`, `w < q < 2^63` and `w_quo = floor(w * 2^64 / q)`.
#[inline(always)]
fn shoup_mul(x: u64, w: u64, w_quo: u64, q: u64) -> u64 {
    let hi = ((x as u128 * w_quo as u128) >> 64) as u64;
    x.wrapping_mul(w).wrapping_sub(hi.wrapping_mul(q))
}

/// The basis change of degree `n` of the conjugate-invariant ring over the primes `P`: `f[k] = alpha^-k` for `k` in
/// `[0, n)`, `alpha` a primitive `4n`-th root of unity, and `i = alpha^n`.
pub struct BasisChangeTable<P: PrimeSetCrt4> {
    /// `f[k]` modulo prime `p` at `4 * k + p`, canonical.
    powers: AlignedVec<u64>,
    /// Shoup quotients `floor(f * 2^64 / q)` of `powers`.
    quotients: AlignedVec<u64>,
    /// `i` modulo each prime, canonical.
    i: [u64; 4],
    /// Shoup quotients of `i`.
    i_quotients: [u64; 4],
    _phantom: PhantomData<P>,
}

impl<P: PrimeSetCrt4> BasisChangeTable<P> {
    /// The table of degree `n`.
    pub(crate) fn new(n: usize) -> Self {
        let mut powers = alloc_aligned::<u64>(4 * n);
        let mut quotients = alloc_aligned::<u64>(4 * n);
        powers.truncate(4 * n);
        quotients.truncate(4 * n);
        let mut i = [0; 4];
        let mut i_quotients = [0; 4];
        for (p, &q) in P::Q.iter().enumerate() {
            let q = q as u64;
            let (m, r) = (u64::MAX / q, u64::MAX % q + 1);
            // floor(w 2^64 / q) = w floor(2^64 / q) + floor(w (2^64 mod q) / q), with no 128-bit division for w < q < 2^32.
            let quotient = |w: u64| w * m + w * r / q;
            let alpha = root(n, q, P::OMEGA[p] as u64, P::MAX_LOG_N);
            let step = pow_mod(alpha, q - 2, q);
            let step_quotient = quotient(step);
            let mut f = 1;
            for k in 0..n {
                powers[4 * k + p] = f;
                quotients[4 * k + p] = quotient(f);
                f = shoup_mul(f, step, step_quotient, q);
                f -= if f >= q { q } else { 0 };
            }
            i[p] = pow_mod(alpha, n as u64, q);
            i_quotients[p] = quotient(i[p]);
        }
        Self {
            powers,
            quotients,
            i,
            i_quotients,
            _phantom: PhantomData,
        }
    }

    /// The degree `n`.
    pub fn degree(&self) -> usize {
        self.powers.len() / 4
    }

    /// `f[k]` modulo prime `p` at `4 * k + p`, canonical.
    pub fn powers(&self) -> &[u64] {
        &self.powers
    }

    /// `i` modulo each prime, canonical.
    pub fn i(&self) -> &[u64; 4] {
        &self.i
    }

    /// Changes the q120b vector `data` of the degree of the table into the negacyclic basis, canonical.
    fn forward_portable(&self, data: &mut [u64]) {
        let n = self.degree();
        for (x, &q) in data[..4].iter_mut().zip(&P::Q) {
            *x %= q as u64;
        }
        for j in 1..=n / 2 {
            for (p, &q) in P::Q.iter().enumerate() {
                let q = q as u64;
                let (i, i_quo) = (self.i[p], self.i_quotients[p]);
                // `x - y mod q` below `2^64` for `y < 2q`.
                let sub = |x: u64, y: u64| if x < y { x + 2 * q - y } else { x - y };
                let change = |x: u64, y: u64, k: usize| {
                    let t = sub(x, shoup_mul(y, i, i_quo, q));
                    let r = shoup_mul(t, self.powers[k], self.quotients[k], q);
                    r - if r >= q { q } else { 0 }
                };
                let (direct, reflected) = (4 * j + p, 4 * (n - j) + p);
                let (a, b) = (data[direct], data[reflected]);
                data[direct] = change(a, b, direct);
                data[reflected] = change(b, a, reflected);
            }
        }
    }

    /// Changes the q120b vector `data` of the degree of the table back to invariant coefficients, canonical.
    fn inverse_portable(&self, data: &mut [u64]) {
        let n = self.degree();
        for (x, &q) in data[..4].iter_mut().zip(&P::Q) {
            *x %= q as u64;
        }
        for j in 1..=n / 2 {
            for (p, &q) in P::Q.iter().enumerate() {
                let q = q as u64;
                let (i, i_quo) = (self.i[p], self.i_quotients[p]);
                // `(i x - y) / 2`, canonical, for `x, y < 2q`.
                let change = |x: u64, y: u64| {
                    let mut r = shoup_mul(x, i, i_quo, q) + 2 * q - y;
                    r -= if r >= 2 * q { 2 * q } else { 0 };
                    r -= if r >= q { q } else { 0 };
                    (r + (r & 1) * q) >> 1
                };
                let (direct, reflected) = (4 * j + p, 4 * (n - j) + p);
                let (a, b) = (data[direct], data[reflected]);
                let u = shoup_mul(a, self.powers[reflected], self.quotients[reflected], q);
                let v = shoup_mul(b, self.powers[direct], self.quotients[direct], q);
                data[direct] = change(u, v);
                data[reflected] = change(v, u);
            }
        }
    }
}

impl<P: PrimeSetCrt4> NttTable<P, ConjugateInvariant> {
    pub fn new(n: usize) -> Self {
        NttTable::<P, Standard>::new(n).with_basis(Arc::new(BasisChangeTable::new(n)))
    }

    /// The basis change of this degree, applied before the NTT.
    pub fn basis_change(&self) -> &BasisChangeTable<P> {
        self.basis.as_deref().expect("conjugate-invariant tables hold a basis change")
    }
}

impl<P: PrimeSetCrt4> NttTableInv<P, ConjugateInvariant> {
    pub fn new(n: usize) -> Self {
        NttTableInv::<P, Standard>::new(n).with_basis(Arc::new(BasisChangeTable::new(n)))
    }

    /// The basis change of this degree, applied after the inverse NTT.
    pub fn basis_change(&self) -> &BasisChangeTable<P> {
        self.basis.as_deref().expect("conjugate-invariant tables hold a basis change")
    }
}

impl<P: PrimeSetCrt4> NttPlanNew for NttPlan<P, ConjugateInvariant> {
    fn new(n: usize) -> Self {
        Self::from_standard(&NttPlan::<P, Standard>::new(n))
    }
}

impl<P: PrimeSetCrt4> NttPlan<P, ConjugateInvariant> {
    /// The plan of the degree of `standard`, sharing its twiddles: only the basis change of the degree is computed, one
    /// table for both directions.
    pub fn from_standard(standard: &NttPlan<P, Standard>) -> Self {
        let basis = Arc::new(BasisChangeTable::new(standard.ntt.n));
        Self {
            ntt: standard.ntt.with_basis(Arc::clone(&basis)),
            intt: standard.intt.with_basis(basis),
        }
    }
}

/// Conjugate-invariant forward NTT: basis change into the negacyclic basis, then the
/// negacyclic NTT.
pub fn ntt_portable<P: PrimeSetCrt4>(table: &NttTable<P, ConjugateInvariant>, data: &mut [u64]) {
    let n = table.n;
    if n > 1 {
        table.basis_change().forward_portable(&mut data[..4 * n]);
    }
    ntt_core(table, data);
}

/// Conjugate-invariant inverse NTT: negacyclic inverse NTT, then basis change back to
/// invariant coefficients.
pub fn intt_portable<P: PrimeSetCrt4>(table: &NttTableInv<P, ConjugateInvariant>, data: &mut [u64]) {
    intt_core(table, data);
    let n = table.n;
    if n > 1 {
        table.basis_change().inverse_portable(&mut data[..4 * n]);
    }
}

/// Builds the [`NttAutomorphismPlan`]: the first half of the standard
/// degree-`2n` plan for the sign of `p` that is `1 mod 4`, which maps that half onto itself.
pub fn build_ntt4x30_automorphism_plan_portable(n: usize, p: i64) -> NttAutomorphismPlan<ConjugateInvariant> {
    let mut perm = standard::build_ntt4x30_automorphism_plan_portable(2 * n, if p & 3 == 1 { p } else { -p }).perm;
    perm.truncate(n);
    NttAutomorphismPlan {
        p,
        perm,
        ring: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kernels::ntt4x30::primes::{Primes29, Primes30, Primes31},
        test_suite::conjugate_invariant::test_conjugate_invariant_ntt_basis_change,
    };

    fn kernel<P: PrimeSetCrt4>(table: &BasisChangeTable<P>, inverse: bool, data: &mut [u64]) {
        if inverse {
            table.inverse_portable(data)
        } else {
            table.forward_portable(data)
        }
    }

    #[test]
    fn basis_change_matches_reference() {
        test_conjugate_invariant_ntt_basis_change::<Primes29>(kernel);
        test_conjugate_invariant_ntt_basis_change::<Primes30>(kernel);
        test_conjugate_invariant_ntt_basis_change::<Primes31>(kernel);
    }
}
