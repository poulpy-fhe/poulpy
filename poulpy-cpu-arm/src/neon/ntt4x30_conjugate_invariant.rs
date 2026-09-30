//! Conjugate invariant basis change — NEON kernel.

use core::arch::aarch64::{vdupq_n_s64, vshlq_u64};
use poulpy_cpu_portable::kernels::ntt4x30::{conjugate_invariant::BasisChange, primes::PrimeSetCrt4};

use super::{
    ntt4x30_ntt::broadcast_mask,
    q120::{Q120, add_q120, and_q120, load_const, load_q120, store_q120, sub_q120},
};

/// NEON conjugate-invariant basis change of a q120b vector, one prime per lane.
pub(crate) unsafe fn basis_change<P: PrimeSetCrt4>(plans: &[BasisChange; 4], data: &mut [u64]) {
    use super::q120::{cond_sub_q120, mul_epu32_q120, shr_q120};
    unsafe {
        let factors = plans.each_ref().map(|plan| plan.factors());
        let n = factors[0].len();
        assert_eq!(data.len(), 4 * n);
        assert!(factors.iter().all(|f| f.len() == n));
        let primes = P::Q.map(u64::from);
        let q = load_const(&primes);
        let mu = load_const(&primes.map(|q| (1u64 << (P::LOG_Q + 31)) / q));
        let pow32 = load_const(&primes.map(|q| (1u64 << 32) % q));
        let mask = broadcast_mask(u32::MAX as u64);
        let high_shift = vdupq_n_s64(-((P::LOG_Q - 1) as i64));
        let low_shift = vdupq_n_s64(-((P::LOG_Q + 31) as i64));
        let shift = |x: Q120, h| Q120 {
            lo: vshlq_u64(x.lo, h),
            hi: vshlq_u64(x.hi, h),
        };
        let reduce = |x| {
            let hi = mul_epu32_q120(shr_q120::<32>(x), mu);
            let lo = mul_epu32_q120(and_q120(x, mask), mu);
            let quotient = add_q120(shift(hi, high_shift), shift(lo, low_shift));
            let r = sub_q120(x, mul_epu32_q120(quotient, q));
            cond_sub_q120(cond_sub_q120(r, q), q)
        };
        // Split before reduction so every u64 input, including negative lifts, is valid.
        let canonical = |x| {
            let hi = reduce(shr_q120::<32>(x));
            reduce(add_q120(mul_epu32_q120(hi, pow32), and_q120(x, mask)))
        };
        let apply = |j: usize, a, b| {
            let direct = std::array::from_fn(|k| factors[k][j][0]);
            let reflected = std::array::from_fn(|k| factors[k][j][1]);
            let d = load_const(&direct);
            let c = load_const(&reflected);
            cond_sub_q120(add_q120(reduce(mul_epu32_q120(a, d)), reduce(mul_epu32_q120(b, c))), q)
        };
        let ptr = data.as_mut_ptr();
        store_q120(ptr, canonical(load_q120(ptr)));
        for j in 1..=n / 2 {
            let other = n - j;
            let a = canonical(load_q120(ptr.add(4 * j)));
            let b = canonical(load_q120(ptr.add(4 * other)));
            store_q120(ptr.add(4 * j), apply(j, a, b));
            store_q120(ptr.add(4 * other), apply(other, b, a));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::basis_change;

    #[test]
    fn basis_change_parity() {
        use poulpy_cpu_portable::{
            kernels::ntt4x30::primes::{Primes29, Primes30, Primes31},
            test_suite::conjugate_invariant::test_conjugate_invariant_ntt_basis_change,
        };
        test_conjugate_invariant_ntt_basis_change::<Primes29>(|plans, data| unsafe { basis_change::<Primes29>(plans, data) });
        test_conjugate_invariant_ntt_basis_change::<Primes30>(|plans, data| unsafe { basis_change::<Primes30>(plans, data) });
        test_conjugate_invariant_ntt_basis_change::<Primes31>(|plans, data| unsafe { basis_change::<Primes31>(plans, data) });
    }
}
