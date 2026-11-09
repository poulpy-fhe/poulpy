//! Conjugate invariant ring items of [`NTT4x30Avx`].

use core::arch::x86_64::{
    __m256i, _mm_cvtsi64_si128, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_mul_epu32, _mm256_set1_epi64x,
    _mm256_setzero_si256, _mm256_srl_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi64,
};

use poulpy_cpu_portable::kernels::{
    ntt4x30::{
        NttDFTExecute,
        conjugate_invariant::{BasisChangeTable, build_ntt4x30_automorphism_plan_portable},
        ntt::{NttTable, NttTableInv},
        primes::{PrimeSetCrt4, Primes30},
        vec_znx_dft::NttAutomorphismPlan,
    },
    znx::{ZnxAutomorphism, conjugate_invariant::znx_automorphism_portable},
};
use poulpy_hal::layouts::ConjugateInvariant;

use super::{
    NTT4x30Avx,
    arithmetic_avx::cond_sub,
    ntt::{intt_avx2, ntt_avx2},
};

impl ZnxAutomorphism<ConjugateInvariant> for NTT4x30Avx {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_portable(p, res, a)
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_portable(p, res, a)
    }
}

impl NttDFTExecute<NttTable<Primes30, ConjugateInvariant>> for NTT4x30Avx {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        // SAFETY: NTT4x30Avx::new() verifies AVX2 availability at construction time.
        unsafe {
            basis_change::<Primes30, false>(table.basis_change(), data);
            ntt_avx2::<Primes30>(table, data)
        }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan<ConjugateInvariant> {
        build_ntt4x30_automorphism_plan_portable(n, p)
    }
}

impl NttDFTExecute<NttTableInv<Primes30, ConjugateInvariant>> for NTT4x30Avx {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        // SAFETY: NTT4x30Avx::new() verifies AVX2 availability at construction time.
        unsafe {
            intt_avx2::<Primes30>(table, data);
            basis_change::<Primes30, true>(table.basis_change(), data)
        }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan<ConjugateInvariant> {
        build_ntt4x30_automorphism_plan_portable(n, p)
    }
}

/// AVX2 conjugate-invariant basis change of the q120b vector `data` of the degree of `table`, one prime per lane: into
/// the negacyclic basis, or back with `INVERSE`, canonical.
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
unsafe fn basis_change<P: PrimeSetCrt4, const INVERSE: bool>(table: &BasisChangeTable<P>, data: &mut [u64]) {
    let n = table.degree();
    assert!(data.len() == 4 * n, "the vector must have the degree of the table");
    // SAFETY: AVX2 is the caller's contract; `data` and the table both hold `n` coefficients, and every index below is under `n`.
    unsafe {
        let primes = P::Q.map(u64::from);
        let mu = primes.map(|q| (1u64 << (P::LOG_Q + 31)) / q);
        let pow32 = primes.map(|q| (1u64 << 32) % q);
        let q = _mm256_loadu_si256(primes.as_ptr().cast());
        let mu = _mm256_loadu_si256(mu.as_ptr().cast());
        let pow32 = _mm256_loadu_si256(pow32.as_ptr().cast());
        let i = _mm256_loadu_si256(table.i().as_ptr().cast());
        let mask = _mm256_set1_epi64x(u32::MAX as i64);
        let one = _mm256_set1_epi64x(1);
        let high_shift = _mm_cvtsi64_si128((P::LOG_Q - 1) as i64);
        let low_shift = _mm_cvtsi64_si128((P::LOG_Q + 31) as i64);
        let reduce = |x| {
            let hi = _mm256_mul_epu32(_mm256_srli_epi64::<32>(x), mu);
            let lo = _mm256_mul_epu32(_mm256_and_si256(x, mask), mu);
            let quotient = _mm256_add_epi64(_mm256_srl_epi64(hi, high_shift), _mm256_srl_epi64(lo, low_shift));
            let r = _mm256_sub_epi64(x, _mm256_mul_epu32(quotient, q));
            cond_sub(cond_sub(r, q), q)
        };
        // Split before reduction so every u64 input, including negative lifts, is valid.
        let canonical = |x| {
            let hi = reduce(_mm256_srli_epi64::<32>(x));
            reduce(_mm256_add_epi64(_mm256_mul_epu32(hi, pow32), _mm256_and_si256(x, mask)))
        };
        let mul = |x, y| reduce(_mm256_mul_epu32(x, y));
        let sub = |x, y| cond_sub(_mm256_add_epi64(x, _mm256_sub_epi64(q, y)), q);
        let odd = |x| _mm256_sub_epi64(_mm256_setzero_si256(), _mm256_and_si256(x, one));
        let half = |x| _mm256_srli_epi64::<1>(_mm256_add_epi64(x, _mm256_and_si256(odd(x), q)));
        // Coefficient `j` of `a` against `n - j` of `b`, `fa = f[j]` and `fb = f[n - j]`.
        let change = |a, b, fa, fb| {
            if INVERSE {
                let (u, v) = (mul(a, fb), mul(b, fa));
                (half(sub(mul(u, i), v)), half(sub(mul(v, i), u)))
            } else {
                (mul(fa, sub(a, mul(i, b))), mul(fb, sub(b, mul(i, a))))
            }
        };
        let f = table.powers().as_ptr().cast::<__m256i>();
        let ptr = data.as_mut_ptr().cast::<__m256i>();
        _mm256_storeu_si256(ptr, canonical(_mm256_loadu_si256(ptr)));
        for j in 1..=n / 2 {
            let other = n - j;
            let a = canonical(_mm256_loadu_si256(ptr.add(j)));
            let b = canonical(_mm256_loadu_si256(ptr.add(other)));
            let (x, y) = change(a, b, _mm256_loadu_si256(f.add(j)), _mm256_loadu_si256(f.add(other)));
            _mm256_storeu_si256(ptr.add(j), x);
            _mm256_storeu_si256(ptr.add(other), y);
        }
    }
}

#[cfg(all(test, target_feature = "avx2"))]
mod tests {
    use poulpy_cpu_portable::{
        kernels::ntt4x30::{
            conjugate_invariant::BasisChangeTable,
            primes::{PrimeSetCrt4, Primes29, Primes30, Primes31},
        },
        test_suite::conjugate_invariant::test_conjugate_invariant_ntt_basis_change,
    };

    use super::basis_change;

    fn kernel<P: PrimeSetCrt4>(table: &BasisChangeTable<P>, inverse: bool, data: &mut [u64]) {
        // SAFETY: the test only builds with AVX2 enabled; the length check panics before any access.
        unsafe {
            if inverse {
                basis_change::<P, true>(table, data)
            } else {
                basis_change::<P, false>(table, data)
            }
        }
    }

    #[test]
    fn basis_change_parity() {
        test_conjugate_invariant_ntt_basis_change::<Primes29>(kernel);
        test_conjugate_invariant_ntt_basis_change::<Primes30>(kernel);
        test_conjugate_invariant_ntt_basis_change::<Primes31>(kernel);
    }
}
