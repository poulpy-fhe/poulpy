//! Conjugate invariant ring items of [`NTT3x42Ifma`].

use std::sync::Arc;

use core::arch::x86_64::{
    _mm512_add_epi64, _mm512_loadu_si512, _mm512_mask_add_epi64, _mm512_min_epu64, _mm512_permutexvar_epi64, _mm512_set_epi64,
    _mm512_set1_epi64, _mm512_srli_epi64, _mm512_storeu_si512, _mm512_sub_epi64, _mm512_test_epi64_mask,
};

use poulpy_cpu_portable::kernels::{
    ntt4x30::{conjugate_invariant::build_ntt4x30_automorphism_plan_portable, vec_znx_dft::NttAutomorphismPlan},
    znx::{ZnxAutomorphism, conjugate_invariant::znx_automorphism_portable},
};
use poulpy_hal::layouts::{ConjugateInvariant, Standard};

use super::{
    NTT3x42Ifma,
    kernels::{cond_sub_2q_si512, harvey_modmul_si512, intt_avx512, ntt_avx512},
    module::DegreeTables,
    primes::{PrimeSetNtt3x42Ifma, Primes42, modq_pow64},
    tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv, cond_sub_2q, harvey_modmul, harvey_quotient, max_ntt_degree},
    traits::Ntt3x42IfmaDFTExecute,
};

/// One prime of a [`BasisChangeTable`].
struct BasisChangePrime {
    q: u64,
    /// `f[k] = alpha^-k mod q` for `k < n`.
    factors: Vec<u64>,
    /// Harvey quotients of `factors`.
    quotients: Vec<u64>,
    /// `i = alpha^n`, a fourth root of unity, and its Harvey quotient.
    i: [u64; 2],
    /// `-i` and its Harvey quotient.
    neg_i: [u64; 2],
}

impl BasisChangePrime {
    fn new(n: usize, q: u64, omega: u64, max_log_n: u32) -> Self {
        let alpha = modq_pow64(omega, ((1usize << (max_log_n - 1)) / n) as i64, q);
        let step = modq_pow64(alpha, -1, q);
        let step_quot = harvey_quotient(step, q);
        // `w * 2^52 - (w * 2^52 mod q)` is a multiple of `q`, so multiplying it by `q^-1 mod 2^64` divides it exactly.
        let shift = (1u64 << 52) % q;
        let shift_quot = harvey_quotient(shift, q);
        let q_inv = (0..5).fold(q, |x, _| x.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(x))));
        let mut factors = Vec::with_capacity(n);
        let mut quotients = Vec::with_capacity(n);
        let mut w = 1;
        for _ in 0..n {
            factors.push(w);
            let rem = cond_sub_2q(harvey_modmul(w, shift, shift_quot, q), q);
            quotients.push((w << 52).wrapping_sub(rem).wrapping_mul(q_inv));
            w = cond_sub_2q(harvey_modmul(w, step, step_quot, q), q);
        }
        let i = modq_pow64(alpha, n as i64, q);
        Self {
            q,
            factors,
            quotients,
            i: [i, harvey_quotient(i, q)],
            neg_i: [q - i, harvey_quotient(q - i, q)],
        }
    }
}

/// The conjugate-invariant basis change of one degree `n`, read by both its forward and inverse transforms.
///
/// Per prime it holds `f[k] = alpha^-k mod q` for `k < n`, `alpha` a primitive `4n`-th root of unity, and `i = alpha^n`,
/// the same fourth root of unity at every degree: the forward change is `out[j] = f[j] (a[j] - i a[n - j])` and the
/// inverse `out[j] = (i f[n - j] a[j] - f[j] a[n - j]) / 2`, for `0 < j < n`, with `out[0] = a[0]`.
pub(super) struct BasisChangeTable([BasisChangePrime; 3]);

impl BasisChangeTable {
    fn new<P: PrimeSetNtt3x42Ifma>(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n <= max_ntt_degree::<P, ConjugateInvariant>(),
            "basis change degree must be a power of two within the conjugate-invariant ring"
        );
        Self(std::array::from_fn(|k| {
            BasisChangePrime::new(n, P::Q[k], P::OMEGA[k], P::MAX_LOG_N)
        }))
    }
}

impl<P: PrimeSetNtt3x42Ifma> Ntt3x42IfmaTable<P, ConjugateInvariant> {
    pub fn new(n: usize) -> Self {
        Ntt3x42IfmaTable::<P, Standard>::new(n).with_basis(Arc::new(BasisChangeTable::new::<P>(n)))
    }

    /// Basis change into the negacyclic basis, applied before the NTT.
    fn basis_change(&self) -> &BasisChangeTable {
        self.basis
            .as_deref()
            .expect("conjugate-invariant table without its basis change")
    }
}

impl<P: PrimeSetNtt3x42Ifma> Ntt3x42IfmaTableInv<P, ConjugateInvariant> {
    pub fn new(n: usize) -> Self {
        Ntt3x42IfmaTableInv::<P, Standard>::new(n).with_basis(Arc::new(BasisChangeTable::new::<P>(n)))
    }

    /// Basis change back to invariant coefficients, applied after the inverse NTT.
    fn basis_change(&self) -> &BasisChangeTable {
        self.basis
            .as_deref()
            .expect("conjugate-invariant table without its basis change")
    }
}

/// The conjugate-invariant tables of the degree of `standard`, sharing its twiddles and one basis change.
pub(super) fn tables(standard: &DegreeTables<Standard>) -> DegreeTables<ConjugateInvariant> {
    let basis = Arc::new(BasisChangeTable::new::<Primes42>(standard.ntt.n));
    DegreeTables {
        ntt: standard.ntt.with_basis(Arc::clone(&basis)),
        intt: standard.intt.with_basis(basis),
    }
}

impl ZnxAutomorphism<ConjugateInvariant> for NTT3x42Ifma {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_portable(p, res, a)
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_portable(p, res, a)
    }
}

impl Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, ConjugateInvariant>> for NTT3x42Ifma {
    #[inline(always)]
    fn ntt3x42_ifma_dft_execute(table: &Ntt3x42IfmaTable<Primes42, ConjugateInvariant>, data: &mut [u64]) {
        // SAFETY: NTT3x42Ifma::new() verifies AVX512-IFMA at construction time, and both kernels assert the lengths they read.
        unsafe {
            basis_change::<false>(table.basis_change(), data);
            ntt_avx512::<Primes42>(table, data, false)
        }
    }

    #[inline(always)]
    fn ntt3x42_ifma_dft_execute_lazy(table: &Ntt3x42IfmaTable<Primes42, ConjugateInvariant>, data: &mut [u64]) {
        // SAFETY: NTT3x42Ifma::new() verifies AVX512-IFMA at construction time, and both kernels assert the lengths they read.
        unsafe {
            basis_change::<false>(table.basis_change(), data);
            ntt_avx512::<Primes42>(table, data, true)
        }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan<ConjugateInvariant> {
        build_ntt4x30_automorphism_plan_portable(n, p)
    }
}

impl Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, ConjugateInvariant>> for NTT3x42Ifma {
    #[inline(always)]
    fn ntt3x42_ifma_dft_execute(table: &Ntt3x42IfmaTableInv<Primes42, ConjugateInvariant>, data: &mut [u64]) {
        // SAFETY: NTT3x42Ifma::new() verifies AVX512-IFMA at construction time, and both kernels assert the lengths they read.
        unsafe {
            intt_avx512::<Primes42>(table, data);
            basis_change::<true>(table.basis_change(), data)
        }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan<ConjugateInvariant> {
        build_ntt4x30_automorphism_plan_portable(n, p)
    }
}

/// Conjugate-invariant basis change of one prime plane of the degree of `prime`, with residues in `[0, 2q)` in and
/// canonical out.
///
/// Lane `l` of a vector step handles the pair `(j + l, n - j - l)`; degrees below 16 take the scalar loop, which handles
/// `(j, n - j)`. Reproduces the scalar definition `tests::reference`, which `basis_change_parity` compares with.
#[target_feature(enable = "avx512ifma")]
unsafe fn basis_change_plane<const INVERSE: bool>(prime: &BasisChangePrime, data: &mut [u64]) {
    let n = data.len();
    assert!(
        prime.factors.len() == n && prime.quotients.len() == n,
        "basis change: the table is not of the degree of the plane"
    );
    let q = prime.q;
    let [i, i_quot] = prime.i;
    let [neg_i, neg_i_quot] = prime.neg_i;
    data[0] = cond_sub_2q(data[0], q);
    let mut j = 1;
    // SAFETY: the table and `data` have length `n`, and `j + 7 <= n / 2` keeps every load and store below `n`.
    unsafe {
        let qv = _mm512_set1_epi64(q as i64);
        let q2 = _mm512_set1_epi64((2 * q) as i64);
        let (iv, iv_quot) = (_mm512_set1_epi64(i as i64), _mm512_set1_epi64(i_quot as i64));
        let (neg_iv, neg_iv_quot) = (_mm512_set1_epi64(neg_i as i64), _mm512_set1_epi64(neg_i_quot as i64));
        let one = _mm512_set1_epi64(1);
        let reverse = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        let (factors, quotients) = (prime.factors.as_ptr(), prime.quotients.as_ptr());
        let load = |p: *const u64| _mm512_loadu_si512(p.cast());
        let load_reversed = |p: *const u64| _mm512_permutexvar_epi64(reverse, _mm512_loadu_si512(p.cast()));
        let halve = |x| _mm512_srli_epi64::<1>(_mm512_mask_add_epi64(x, _mm512_test_epi64_mask(x, one), x, qv));
        // The last step reaches the middle coefficient from both sides, which compute the same value.
        while j + 7 <= n / 2 {
            let (f, f_quot) = (load(factors.add(j)), load(quotients.add(j)));
            let (g, g_quot) = (load_reversed(factors.add(n - j - 7)), load_reversed(quotients.add(n - j - 7)));
            let x = load(data.as_ptr().add(j));
            let y = load_reversed(data.as_ptr().add(n - j - 7));
            let (lo, hi) = if INVERSE {
                let u = harvey_modmul_si512(x, g, g_quot, qv);
                let v = harvey_modmul_si512(y, f, f_quot, qv);
                let combine = |u, v| {
                    let d = _mm512_sub_epi64(harvey_modmul_si512(u, iv, iv_quot, qv), v);
                    // A wrapped `d` is above `2^64 - 2q`, so the minimum picks `d + 2q`.
                    halve(cond_sub_2q_si512(_mm512_min_epu64(d, _mm512_add_epi64(d, q2)), qv))
                };
                (combine(u, v), combine(v, u))
            } else {
                let twist = |u, v| _mm512_add_epi64(u, harvey_modmul_si512(v, neg_iv, neg_iv_quot, qv));
                (
                    cond_sub_2q_si512(harvey_modmul_si512(twist(x, y), f, f_quot, qv), qv),
                    cond_sub_2q_si512(harvey_modmul_si512(twist(y, x), g, g_quot, qv), qv),
                )
            };
            _mm512_storeu_si512(data.as_mut_ptr().add(j).cast(), lo);
            _mm512_storeu_si512(data.as_mut_ptr().add(n - j - 7).cast(), _mm512_permutexvar_epi64(reverse, hi));
            j += 8;
        }
    }
    let mul = |x: u64, k: usize| harvey_modmul(x, prime.factors[k], prime.quotients[k], q);
    for j in j..=n / 2 {
        let (x, y) = (data[j], data[n - j]);
        let (lo, hi) = if INVERSE {
            let (u, v) = (mul(x, n - j), mul(y, j));
            let combine = |u: u64, v: u64| {
                let sum = cond_sub_2q(cond_sub_2q(harvey_modmul(u, i, i_quot, q) + 2 * q - v, 2 * q), q);
                (sum + (sum & 1) * q) >> 1
            };
            (combine(u, v), combine(v, u))
        } else {
            let twist = |u: u64, v: u64| u + harvey_modmul(v, neg_i, neg_i_quot, q);
            (cond_sub_2q(mul(twist(x, y), j), q), cond_sub_2q(mul(twist(y, x), n - j), q))
        };
        data[j] = lo;
        data[n - j] = hi;
    }
}

/// Conjugate-invariant basis change of the three prime planes of `data`.
#[target_feature(enable = "avx512ifma")]
unsafe fn basis_change<const INVERSE: bool>(basis: &BasisChangeTable, data: &mut [u64]) {
    let n = basis.0[0].factors.len();
    assert!(data.len() == 3 * n, "data must hold 3 planes of length n");
    for (prime, plane) in basis.0.iter().zip(data.chunks_exact_mut(n)) {
        // SAFETY: this function's `avx512ifma` feature covers the plane kernel, which asserts the lengths it reads.
        unsafe { basis_change_plane::<INVERSE>(prime, plane) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_hal::layouts::PrimeSet;

    /// The basis change from its definition: `out[j] = d[j] a[j] + c[j] a[n - j] mod q`, forward
    /// `[d, c] = [alpha^-j, -i alpha^-j]`, inverse `[alpha^j / 2, -alpha^-j / 2]`, `out[0] = a[0] mod q`.
    fn reference(a: &[u64], q: u64, omega: u64, inverse: bool) -> Vec<u64> {
        let n = a.len();
        let mul = |x: u64, y: u64| ((x as u128 * y as u128) % q as u128) as u64;
        let alpha = modq_pow64(omega, ((1usize << (Primes42::MAX_LOG_N - 1)) / n) as i64, q);
        let alpha_inv = modq_pow64(alpha, -1, q);
        let i = modq_pow64(alpha, n as i64, q);
        let half = q.div_ceil(2);
        let (mut forward, mut backward) = (1, 1);
        let mut out = vec![a[0] % q; n];
        for j in 1..n {
            forward = mul(forward, alpha);
            backward = mul(backward, alpha_inv);
            let [d, c] = if inverse {
                [mul(forward, half), q - mul(backward, half)]
            } else {
                [backward, q - mul(i, backward)]
            };
            out[j] = (mul(d, a[j]) + mul(c, a[n - j])) % q;
        }
        out
    }

    #[test]
    fn basis_change_parity() {
        for n in (1..=max_ntt_degree::<Primes42, ConjugateInvariant>().ilog2()).map(|log_n| 1usize << log_n) {
            let table = BasisChangeTable::new::<Primes42>(n);
            for (k, prime) in table.0.iter().enumerate() {
                let q = Primes42::Q[k];
                for (&w, &w_quot) in prime.factors.iter().zip(&prime.quotients) {
                    assert_eq!(w_quot, harvey_quotient(w, q), "n={n}, prime={k}");
                }
                for inverse in [false, true] {
                    let mut seed = 0x1234_5678_9abc_def0u64;
                    let mut actual: Vec<_> = (0..n)
                        .map(|j| {
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            match j % 6 {
                                0 => 0,
                                1 => 1,
                                2 => q - 1,
                                3 => q,
                                4 => 2 * q - 1,
                                _ => seed % (2 * q),
                            }
                        })
                        .collect();
                    actual[0] = if inverse { 2 * q - 1 } else { q - 1 };
                    let expected = reference(&actual, q, Primes42::OMEGA[k], inverse);
                    // SAFETY: this module only builds with `avx512ifma`, and the kernel asserts the table is of degree `n`.
                    unsafe {
                        if inverse {
                            basis_change_plane::<true>(prime, &mut actual)
                        } else {
                            basis_change_plane::<false>(prime, &mut actual)
                        }
                    };
                    assert_eq!(actual, expected, "n={n}, inverse={inverse}, prime={k}");
                }
            }
        }
    }

    #[test]
    fn basis_change_rejects_mismatched_degree() {
        let table = BasisChangeTable::new::<Primes42>(32);
        for n in [16, 64] {
            // SAFETY: this module only builds with `avx512ifma`, and the length check panics before any access.
            let mismatched = || unsafe { basis_change_plane::<false>(&table.0[0], &mut vec![0; n]) };
            let payload = std::panic::catch_unwind(mismatched).unwrap_err();
            assert_eq!(
                payload.downcast_ref::<&str>(),
                Some(&"basis change: the table is not of the degree of the plane")
            );
        }
    }
}
