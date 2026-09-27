//! Conjugate invariant ring items of [`NTT4x30Avx512`].

use core::arch::x86_64::{
    __m256i, _mm_cvtsi64_si128, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_mul_epu32, _mm256_set1_epi64x,
    _mm256_srl_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi64,
};

use poulpy_cpu_ref::reference::{
    ntt4x30::{
        NttDFTExecute,
        conjugate_invariant::{BasisChange, build_ntt4x30_automorphism_plan},
        ntt::{NttTable, NttTableInv},
        primes::{PrimeSetCrt4, Primes30},
        vec_znx_dft::NttAutomorphismPlan,
    },
    znx::{ZnxAutomorphism, conjugate_invariant::znx_automorphism_ref},
};
use poulpy_hal::layouts::ConjugateInvariant;

use super::{
    NTT4x30Avx512,
    ntt::{intt_avx512, ntt_avx512},
};

/// No failure model: a square's constant coefficient has a large positive mean on this ring.
impl poulpy_hal::layouts::MaxBase2k for NTT4x30Avx512<ConjugateInvariant> {
    fn max_base2k(_n: usize, _products: usize, _failure_bits: usize, _squaring: bool) -> Option<usize> {
        None
    }
}

impl ZnxAutomorphism for NTT4x30Avx512<ConjugateInvariant> {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_ref(p, res, a)
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_ref(p, res, a)
    }
}

impl NttDFTExecute<NttTable<Primes30, ConjugateInvariant>> for NTT4x30Avx512<ConjugateInvariant> {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        // SAFETY: NTT4x30Avx512::new() verifies AVX-512F availability at construction time.
        unsafe {
            basis_change::<Primes30>(table.basis_changes(), data);
            ntt_avx512::<Primes30>(table, data)
        }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan {
        build_ntt4x30_automorphism_plan(n, p)
    }
}

impl NttDFTExecute<NttTableInv<Primes30, ConjugateInvariant>> for NTT4x30Avx512<ConjugateInvariant> {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        // SAFETY: NTT4x30Avx512::new() verifies AVX-512F availability at construction time.
        unsafe {
            intt_avx512::<Primes30>(table, data);
            basis_change::<Primes30>(table.basis_changes(), data)
        }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan {
        build_ntt4x30_automorphism_plan(n, p)
    }
}

/// Conjugate-invariant basis change of a q120b vector, one prime per 256-bit lane.
#[target_feature(enable = "avx512f")]
unsafe fn basis_change<P: PrimeSetCrt4>(plans: &[BasisChange; 4], data: &mut [u64]) {
    use super::arithmetic_avx512::cond_sub;
    unsafe {
        let factors = plans.each_ref().map(|plan| plan.factors());
        let n = factors[0].len();
        assert_eq!(data.len(), 4 * n);
        assert!(factors.iter().all(|f| f.len() == n));
        let primes = P::Q.map(u64::from);
        let mu = primes.map(|q| (1u64 << (P::LOG_Q + 31)) / q);
        let pow32 = primes.map(|q| (1u64 << 32) % q);
        let q = _mm256_loadu_si256(primes.as_ptr().cast());
        let mu = _mm256_loadu_si256(mu.as_ptr().cast());
        let pow32 = _mm256_loadu_si256(pow32.as_ptr().cast());
        let mask = _mm256_set1_epi64x(u32::MAX as i64);
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
        let apply = |j: usize, a, b| {
            let direct: [u64; 4] = std::array::from_fn(|k| factors[k][j][0]);
            let reflected: [u64; 4] = std::array::from_fn(|k| factors[k][j][1]);
            let d = _mm256_loadu_si256(direct.as_ptr().cast());
            let c = _mm256_loadu_si256(reflected.as_ptr().cast());
            cond_sub(
                _mm256_add_epi64(reduce(_mm256_mul_epu32(a, d)), reduce(_mm256_mul_epu32(b, c))),
                q,
            )
        };
        let ptr = data.as_mut_ptr().cast::<__m256i>();
        _mm256_storeu_si256(ptr, canonical(_mm256_loadu_si256(ptr)));
        for j in 1..=n / 2 {
            let other = n - j;
            let a = canonical(_mm256_loadu_si256(ptr.add(j)));
            let b = canonical(_mm256_loadu_si256(ptr.add(other)));
            _mm256_storeu_si256(ptr.add(j), apply(j, a, b));
            _mm256_storeu_si256(ptr.add(other), apply(other, b, a));
        }
    }
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::basis_change;

    #[test]
    fn basis_change_parity() {
        use poulpy_cpu_ref::{
            reference::ntt4x30::primes::{Primes29, Primes30, Primes31},
            test_suite::conjugate_invariant::test_conjugate_invariant_ntt_basis_change,
        };
        test_conjugate_invariant_ntt_basis_change::<Primes29>(|plans, data| unsafe { basis_change::<Primes29>(plans, data) });
        test_conjugate_invariant_ntt_basis_change::<Primes30>(|plans, data| unsafe { basis_change::<Primes30>(plans, data) });
        test_conjugate_invariant_ntt_basis_change::<Primes31>(|plans, data| unsafe { basis_change::<Primes31>(plans, data) });
    }
}
