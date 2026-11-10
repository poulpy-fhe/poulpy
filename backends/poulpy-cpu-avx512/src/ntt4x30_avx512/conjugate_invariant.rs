//! Conjugate invariant ring items of [`NTT4x30Avx512`].

use core::arch::x86_64::{
    __m512i, _mm_cvtsi64_si128, _mm256_loadu_si256, _mm512_add_epi64, _mm512_and_si512, _mm512_broadcast_i64x4,
    _mm512_loadu_si512, _mm512_mask_add_epi64, _mm512_mask_blend_epi64, _mm512_mask_storeu_epi64, _mm512_maskz_loadu_epi64,
    _mm512_mul_epu32, _mm512_set1_epi64, _mm512_shuffle_i64x2, _mm512_srl_epi64, _mm512_srli_epi64, _mm512_storeu_si512,
    _mm512_sub_epi64, _mm512_test_epi64_mask,
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
    NTT4x30Avx512,
    arithmetic_avx512::cond_sub_512,
    ntt::{intt_avx512, ntt_avx512},
};

impl ZnxAutomorphism<ConjugateInvariant> for NTT4x30Avx512 {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_portable(p, res, a)
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_portable(p, res, a)
    }
}

impl NttDFTExecute<NttTable<Primes30, ConjugateInvariant>> for NTT4x30Avx512 {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        // SAFETY: NTT4x30Avx512::new() verifies AVX-512F availability at construction time.
        unsafe {
            basis_change::<Primes30, false>(table.basis_change(), data);
            ntt_avx512::<Primes30>(table, data)
        }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan<ConjugateInvariant> {
        build_ntt4x30_automorphism_plan_portable(n, p)
    }
}

impl NttDFTExecute<NttTableInv<Primes30, ConjugateInvariant>> for NTT4x30Avx512 {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        // SAFETY: NTT4x30Avx512::new() verifies AVX-512F availability at construction time.
        unsafe {
            intt_avx512::<Primes30>(table, data);
            basis_change::<Primes30, true>(table.basis_change(), data)
        }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan<ConjugateInvariant> {
        build_ntt4x30_automorphism_plan_portable(n, p)
    }
}

/// Conjugate-invariant basis change of the q120b vector `data` of the degree of `table`, two coefficients per vector and
/// one prime per lane: into the negacyclic basis, or back with `INVERSE`, canonical.
///
/// Reproduces the portable `BasisChangeTable::forward_portable` and `inverse_portable`, checked against the same
/// scalar reference by `test_conjugate_invariant_ntt_basis_change`.
///
/// # Safety
///
/// The CPU must support AVX-512F.
#[target_feature(enable = "avx512f")]
unsafe fn basis_change<P: PrimeSetCrt4, const INVERSE: bool>(table: &BasisChangeTable<P>, data: &mut [u64]) {
    let n = table.degree();
    assert!(data.len() == 4 * n, "the vector must have the degree of the table");
    // SAFETY: AVX-512F is the caller's contract; `data` and the table both hold `n` coefficients, and every access below
    // ends at coefficient `n - 1` at the latest.
    unsafe {
        let quad = |v: &[u64; 4]| _mm512_broadcast_i64x4(_mm256_loadu_si256(v.as_ptr().cast()));
        let primes = P::Q.map(u64::from);
        let q = quad(&primes);
        let mu = quad(&primes.map(|q| (1u64 << (P::LOG_Q + 31)) / q));
        let pow32 = quad(&primes.map(|q| (1u64 << 32) % q));
        let i = quad(table.i());
        let mask = _mm512_set1_epi64(u32::MAX as i64);
        let one = _mm512_set1_epi64(1);
        let high_shift = _mm_cvtsi64_si128((P::LOG_Q - 1) as i64);
        let low_shift = _mm_cvtsi64_si128((P::LOG_Q + 31) as i64);
        let reduce = |x| {
            let hi = _mm512_mul_epu32(_mm512_srli_epi64::<32>(x), mu);
            let lo = _mm512_mul_epu32(_mm512_and_si512(x, mask), mu);
            let quotient = _mm512_add_epi64(_mm512_srl_epi64(hi, high_shift), _mm512_srl_epi64(lo, low_shift));
            let r = _mm512_sub_epi64(x, _mm512_mul_epu32(quotient, q));
            cond_sub_512(cond_sub_512(r, q), q)
        };
        // Split before reduction so every u64 input, including negative lifts, is valid.
        let canonical = |x| {
            let hi = reduce(_mm512_srli_epi64::<32>(x));
            reduce(_mm512_add_epi64(_mm512_mul_epu32(hi, pow32), _mm512_and_si512(x, mask)))
        };
        let mul = |x, y| reduce(_mm512_mul_epu32(x, y));
        let sub = |x, y| cond_sub_512(_mm512_add_epi64(x, _mm512_sub_epi64(q, y)), q);
        let half = |x| _mm512_srli_epi64::<1>(_mm512_mask_add_epi64(x, _mm512_test_epi64_mask(x, one), x, q));
        // Coefficients `j` of `a` against `n - j` of `b`, `fa = f[j]` and `fb = f[n - j]`.
        let change = |a, b, fa, fb| {
            if INVERSE {
                let (u, v) = (mul(a, fb), mul(b, fa));
                (half(sub(mul(u, i), v)), half(sub(mul(v, i), u)))
            } else {
                (mul(fa, sub(a, mul(i, b))), mul(fb, sub(b, mul(i, a))))
            }
        };
        let f = table.powers().as_ptr();
        let swap = |x: __m512i| _mm512_shuffle_i64x2::<0b01_00_11_10>(x, x);
        let ptr = data.as_mut_ptr();
        if n == 2 {
            // Coefficient 1 is its own mirror; coefficient 0 is only reduced.
            let a = canonical(_mm512_loadu_si512(ptr.cast()));
            let fm = _mm512_loadu_si512(f.cast());
            let (x, _) = change(a, a, fm, fm);
            _mm512_storeu_si512(ptr.cast(), _mm512_mask_blend_epi64(0xF0, a, x));
        } else {
            let first = canonical(_mm512_maskz_loadu_epi64(0x0F, ptr.cast()));
            _mm512_mask_storeu_epi64(ptr.cast(), 0x0F, first);
            // Coefficients `j, j + 1` against `n - j, n - j - 1`: the last block holds the middle one twice, computed alike.
            for j in (1..n / 2).step_by(2) {
                let lo = ptr.add(4 * j);
                let hi = ptr.add(4 * (n - j - 1));
                let a = canonical(_mm512_loadu_si512(lo.cast()));
                let b = swap(canonical(_mm512_loadu_si512(hi.cast())));
                let fa = _mm512_loadu_si512(f.add(4 * j).cast());
                let fb = swap(_mm512_loadu_si512(f.add(4 * (n - j - 1)).cast()));
                let (x, y) = change(a, b, fa, fb);
                _mm512_storeu_si512(hi.cast(), swap(y));
                _mm512_storeu_si512(lo.cast(), x);
            }
        }
    }
}

#[cfg(all(test, target_feature = "avx512f"))]
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
        // SAFETY: the test only builds with AVX-512F enabled; the length check panics before any access.
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
