//! Conjugate-invariant NTT4x30 arithmetic: the negacyclic NTT behind a basis change.

use poulpy_hal::layouts::ConjugateInvariant;

use crate::kernels::ntt4x30::{
    ntt::{NttTable, NttTableInv, intt_core, ntt_core},
    primes::PrimeSetCrt4,
    standard,
    vec_znx_dft::{NttAutomorphismPlan, NttPlan, NttPlanNew},
};

/// Changes between invariant coefficients and a negacyclic NTT basis.
pub struct BasisChange {
    factors: Vec<[u64; 2]>,
    /// Shoup quotients `floor(factor * 2^64 / q)` of `factors`.
    quotients: Vec<[u64; 2]>,
    q: u64,
    n: usize,
}

impl BasisChange {
    /// Empty placeholder held by standard-ring tables.
    pub(super) const EMPTY: Self = Self {
        factors: Vec::new(),
        quotients: Vec::new(),
        q: 0,
        n: 0,
    };

    /// Direct and reflected coefficient multipliers, indexed by coefficient.
    pub fn factors(&self) -> &[[u64; 2]] {
        &self.factors
    }

    /// Uses a primitive `2^(max_log_n + 1)`-th root for the supplied prime.
    pub fn new(n: usize, q: u64, omega: u64, max_log_n: u32, inverse: bool) -> Self {
        assert!(n.is_power_of_two() && n <= (1usize << (max_log_n - 1)));
        let mul = |a: u64, b: u64| ((a as u128 * b as u128) % q as u128) as u64;
        let pow = |mut a: u64, mut e: u64| {
            let mut r = 1;
            while e != 0 {
                if e & 1 != 0 {
                    r = mul(r, a);
                }
                a = mul(a, a);
                e >>= 1;
            }
            r
        };
        let alpha = pow(omega, (1u64 << (max_log_n - 1)) / n as u64);
        let alpha_inv = pow(alpha, q - 2);
        // At x^n = i, the invariant basis becomes c[j] - i*c[n-j].
        let i = pow(alpha, n as u64);
        let half = q.div_ceil(2);
        let mut forward = 1;
        let mut backward = 1;
        let mut factors = vec![[1, 0]; n];
        for factors in factors.iter_mut().skip(1) {
            forward = mul(forward, alpha);
            backward = mul(backward, alpha_inv);
            *factors = if inverse {
                [mul(forward, half), q - mul(backward, half)]
            } else {
                [backward, q - mul(i, backward)]
            };
        }
        let shoup = |w: u64| (((w as u128) << 64) / q as u128) as u64;
        let quotients = factors.iter().map(|&[d, c]| [shoup(d), shoup(c)]).collect();
        Self {
            factors,
            quotients,
            q,
            n,
        }
    }

    /// Applies the basis change to one prime lane with the given coefficient stride.
    pub fn apply(&self, data: &mut [u64], stride: usize) {
        let q = self.q;
        // Shoup product `x * w mod q` in `[0, 2q)` for any `x < 2^64`.
        let mul = |x: u64, w: u64, w_quo: u64| {
            let hi = ((x as u128 * w_quo as u128) >> 64) as u64;
            x.wrapping_mul(w).wrapping_sub(hi.wrapping_mul(q))
        };
        let apply = |j: usize, a: u64, b: u64| {
            let ([d, c], [d_quo, c_quo]) = (self.factors[j], self.quotients[j]);
            let mut r = mul(a, d, d_quo) + mul(b, c, c_quo);
            if r >= 2 * q {
                r -= 2 * q;
            }
            if r >= q {
                r -= q;
            }
            r
        };
        data[0] %= q;
        for j in 1..=self.n / 2 {
            let other = self.n - j;
            let a = data[j * stride];
            let b = data[other * stride];
            data[j * stride] = apply(j, a, b);
            data[other * stride] = apply(other, b, a);
        }
    }
}

fn basis_changes<P: PrimeSetCrt4>(n: usize, inverse: bool) -> [BasisChange; 4] {
    std::array::from_fn(|k| BasisChange::new(n, P::Q[k] as u64, P::OMEGA[k] as u64, P::MAX_LOG_N, inverse))
}

impl<P: PrimeSetCrt4> NttTable<P, ConjugateInvariant> {
    pub fn new(n: usize) -> Self {
        Self::build(n, basis_changes::<P>(n, false))
    }

    /// Basis change into the negacyclic basis, applied before the NTT.
    pub fn basis_changes(&self) -> &[BasisChange; 4] {
        &self.basis
    }
}

impl<P: PrimeSetCrt4> NttTableInv<P, ConjugateInvariant> {
    pub fn new(n: usize) -> Self {
        Self::build(n, basis_changes::<P>(n, true))
    }

    /// Basis change back to invariant coefficients, applied after the inverse NTT.
    pub fn basis_changes(&self) -> &[BasisChange; 4] {
        &self.basis
    }
}

impl<P: PrimeSetCrt4> NttPlanNew for NttPlan<P, ConjugateInvariant> {
    fn new(n: usize) -> Self {
        Self {
            ntt: NttTable::<P, ConjugateInvariant>::new(n),
            intt: NttTableInv::<P, ConjugateInvariant>::new(n),
        }
    }
}

/// Conjugate-invariant forward NTT: basis change into the negacyclic basis, then the
/// negacyclic NTT.
pub fn ntt_portable<P: PrimeSetCrt4>(table: &NttTable<P, ConjugateInvariant>, data: &mut [u64]) {
    let n = table.n;
    if n > 1 {
        for k in 0..4 {
            table.basis_changes()[k].apply(&mut data[k..4 * n], 4);
        }
    }
    ntt_core(table, data);
}

/// Conjugate-invariant inverse NTT: negacyclic inverse NTT, then basis change back to
/// invariant coefficients.
pub fn intt_portable<P: PrimeSetCrt4>(table: &NttTableInv<P, ConjugateInvariant>, data: &mut [u64]) {
    intt_core(table, data);
    let n = table.n;
    if n > 1 {
        for k in 0..4 {
            table.basis_changes()[k].apply(&mut data[k..4 * n], 4);
        }
    }
}

/// Builds the [`NttAutomorphismPlan`]: the first half of the standard
/// degree-`2n` plan for the sign of `p` that is `1 mod 4`, which maps that half onto itself.
pub fn build_ntt4x30_automorphism_plan_portable(n: usize, p: i64) -> NttAutomorphismPlan {
    let mut plan = standard::build_ntt4x30_automorphism_plan_portable(2 * n, if p & 3 == 1 { p } else { -p });
    plan.perm.truncate(n);
    plan.p = p;
    plan
}
