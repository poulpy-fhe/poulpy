//! Conjugate invariant ring items of [`NTT3x42Ifma`].

use core::arch::x86_64::{
    _mm512_add_epi64, _mm512_loadu_si512, _mm512_permutexvar_epi64, _mm512_set_epi64, _mm512_set1_epi64, _mm512_storeu_si512,
};

use poulpy_cpu_portable::reference::{
    ntt4x30::{
        NttHandleFactory,
        conjugate_invariant::{BasisChange, build_ntt4x30_automorphism_plan},
        vec_znx_dft::NttAutomorphismPlan,
    },
    znx::{ZnxAutomorphism, conjugate_invariant::znx_automorphism_ref},
};
use poulpy_hal::layouts::ConjugateInvariant;

use super::{
    NTT3x42Ifma,
    kernels::{cond_sub_2q_si512, harvey_modmul_si512, intt_avx512, ntt_avx512},
    module::NTT3x42IfmaHandle,
    primes::{PrimeSetNtt3x42Ifma, Primes42},
    tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv, cond_sub_2q, harvey_modmul, harvey_quotient},
    traits::Ntt3x42IfmaDFTExecute,
};

/// Conjugate-invariant basis-change factors of one prime plane and their Harvey quotients.
#[derive(Default)]
pub(super) struct BasisChangeTable {
    direct: Vec<u64>,
    reflected: Vec<u64>,
    direct_quot: Vec<u64>,
    reflected_quot: Vec<u64>,
}

impl BasisChangeTable {
    fn new(n: usize, q: u64, omega: u64, max_log_n: u32, inverse: bool) -> Self {
        let plan = BasisChange::new(n, q, omega, max_log_n, inverse);
        let direct: Vec<_> = plan.factors().iter().map(|f| f[0]).collect();
        let reflected: Vec<_> = plan.factors().iter().map(|f| f[1]).collect();
        let direct_quot = direct.iter().map(|&w| harvey_quotient(w, q)).collect();
        let reflected_quot = reflected.iter().map(|&w| harvey_quotient(w, q)).collect();
        Self {
            direct,
            reflected,
            direct_quot,
            reflected_quot,
        }
    }
}

fn basis_changes<P: PrimeSetNtt3x42Ifma>(n: usize, inverse: bool) -> [BasisChangeTable; 3] {
    std::array::from_fn(|k| BasisChangeTable::new(n, P::Q[k], P::OMEGA[k], P::MAX_LOG_N, inverse))
}

impl<P: PrimeSetNtt3x42Ifma> Ntt3x42IfmaTable<P, ConjugateInvariant> {
    pub fn new(n: usize) -> Self {
        Self::build(n, basis_changes::<P>(n, false))
    }

    /// Basis change into the negacyclic basis, applied before the NTT.
    fn basis_changes(&self) -> &[BasisChangeTable; 3] {
        &self.basis
    }
}

impl<P: PrimeSetNtt3x42Ifma> Ntt3x42IfmaTableInv<P, ConjugateInvariant> {
    pub fn new(n: usize) -> Self {
        Self::build(n, basis_changes::<P>(n, true))
    }

    /// Basis change back to invariant coefficients, applied after the inverse NTT.
    fn basis_changes(&self) -> &[BasisChangeTable; 3] {
        &self.basis
    }
}

unsafe impl NttHandleFactory for NTT3x42IfmaHandle<ConjugateInvariant> {
    fn create_ntt_handle(n: usize) -> Self {
        Self::with_tables(
            n,
            Ntt3x42IfmaTable::<Primes42, ConjugateInvariant>::new,
            Ntt3x42IfmaTableInv::<Primes42, ConjugateInvariant>::new,
        )
    }
}

/// No failure model: a square's constant coefficient has a large positive mean on this ring.
impl poulpy_hal::layouts::MaxBase2k for NTT3x42Ifma<ConjugateInvariant> {
    fn max_base2k(_n: usize, _products: usize, _failure_bits: usize, _squaring: bool) -> Option<usize> {
        None
    }
}

impl ZnxAutomorphism for NTT3x42Ifma<ConjugateInvariant> {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_ref(p, res, a)
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_ref(p, res, a)
    }
}

impl Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, ConjugateInvariant>> for NTT3x42Ifma<ConjugateInvariant> {
    #[inline(always)]
    fn ntt3x42_ifma_dft_execute(table: &Ntt3x42IfmaTable<Primes42, ConjugateInvariant>, data: &mut [u64]) {
        unsafe {
            basis_change::<Primes42>(table.basis_changes(), data);
            ntt_avx512::<Primes42>(table, data, false)
        }
    }

    #[inline(always)]
    fn ntt3x42_ifma_dft_execute_lazy(table: &Ntt3x42IfmaTable<Primes42, ConjugateInvariant>, data: &mut [u64]) {
        unsafe {
            basis_change::<Primes42>(table.basis_changes(), data);
            ntt_avx512::<Primes42>(table, data, true)
        }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan {
        build_ntt4x30_automorphism_plan(n, p)
    }
}

impl Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, ConjugateInvariant>> for NTT3x42Ifma<ConjugateInvariant> {
    #[inline(always)]
    fn ntt3x42_ifma_dft_execute(table: &Ntt3x42IfmaTableInv<Primes42, ConjugateInvariant>, data: &mut [u64]) {
        unsafe {
            intt_avx512::<Primes42>(table, data);
            basis_change::<Primes42>(table.basis_changes(), data)
        }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan {
        build_ntt4x30_automorphism_plan(n, p)
    }
}

/// Conjugate-invariant basis change of one prime plane with residues in `[0, 2q)`.
#[target_feature(enable = "avx512ifma")]
unsafe fn basis_change_plane(plan: &BasisChangeTable, data: &mut [u64], q: u64) {
    unsafe {
        let n = data.len();
        assert!(n > 0);
        assert_eq!(plan.direct.len(), n);
        assert_eq!(plan.reflected.len(), n);
        assert_eq!(plan.direct_quot.len(), n);
        assert_eq!(plan.reflected_quot.len(), n);
        let qv = _mm512_set1_epi64(q as i64);
        let q2 = _mm512_set1_epi64((2 * q) as i64);
        let reverse = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        let apply = |j, a, b| {
            let d = _mm512_loadu_si512(plan.direct.as_ptr().add(j).cast());
            let c = _mm512_loadu_si512(plan.reflected.as_ptr().add(j).cast());
            let dq = _mm512_loadu_si512(plan.direct_quot.as_ptr().add(j).cast());
            let cq = _mm512_loadu_si512(plan.reflected_quot.as_ptr().add(j).cast());
            let sum = _mm512_add_epi64(harvey_modmul_si512(a, d, dq, qv), harvey_modmul_si512(b, c, cq, qv));
            cond_sub_2q_si512(cond_sub_2q_si512(sum, q2), qv)
        };
        data[0] = cond_sub_2q(data[0], q);
        let mut j = 1;
        while j + 8 <= n / 2 {
            let other = n - j - 7;
            let a = _mm512_loadu_si512(data.as_ptr().add(j).cast());
            let b = _mm512_loadu_si512(data.as_ptr().add(other).cast());
            let lo = apply(j, a, _mm512_permutexvar_epi64(reverse, b));
            let hi = apply(other, b, _mm512_permutexvar_epi64(reverse, a));
            _mm512_storeu_si512(data.as_mut_ptr().add(j).cast(), lo);
            _mm512_storeu_si512(data.as_mut_ptr().add(other).cast(), hi);
            j += 8;
        }
        let apply = |j: usize, a, b| {
            let d = harvey_modmul(a, plan.direct[j], plan.direct_quot[j], q);
            let c = harvey_modmul(b, plan.reflected[j], plan.reflected_quot[j], q);
            cond_sub_2q(cond_sub_2q(d + c, 2 * q), q)
        };
        for j in j..=n / 2 {
            let other = n - j;
            let a = data[j];
            let b = data[other];
            data[j] = apply(j, a, b);
            data[other] = apply(other, b, a);
        }
    }
}

/// Conjugate-invariant basis change of the three prime planes of `data`.
#[target_feature(enable = "avx512ifma")]
unsafe fn basis_change<P: PrimeSetNtt3x42Ifma>(plans: &[BasisChangeTable; 3], data: &mut [u64]) {
    let n = data.len() / 3;
    for k in 0..3 {
        unsafe { basis_change_plane(&plans[k], &mut data[k * n..(k + 1) * n], P::Q[k]) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt3x42_ifma::primes::Primes42;
    use poulpy_cpu_portable::reference::ntt4x30::conjugate_invariant::BasisChange;
    use poulpy_hal::layouts::PrimeSet;

    #[test]
    fn basis_change_parity() {
        for n in [2, 4, 8, 16, 32, 64, 256, 1024, 8192, 32768, 65536] {
            let fwd = Ntt3x42IfmaTable::<Primes42, ConjugateInvariant>::new(n);
            let inv = Ntt3x42IfmaTableInv::<Primes42, ConjugateInvariant>::new(n);
            for (inverse, plans) in [(false, fwd.basis_changes()), (true, inv.basis_changes())] {
                for (k, plan) in plans.iter().enumerate() {
                    let q = Primes42::Q[k];
                    let reference = BasisChange::new(n, q, Primes42::OMEGA[k], Primes42::MAX_LOG_N, inverse);
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
                    let mut expected = actual.clone();
                    reference.apply(&mut expected, 1);
                    unsafe { basis_change_plane(plan, &mut actual, q) };
                    assert_eq!(actual, expected, "n={n}, inverse={inverse}, prime={k}");
                }
            }
        }
    }

    #[test]
    fn basis_change_rejects_short_factors() {
        for factor in 0..4 {
            let mut plan = BasisChangeTable::new(32, Primes42::Q[0], Primes42::OMEGA[0], Primes42::MAX_LOG_N, false);
            match factor {
                0 => &mut plan.direct,
                1 => &mut plan.reflected,
                2 => &mut plan.direct_quot,
                _ => &mut plan.reflected_quot,
            }
            .pop();
            assert!(
                std::panic::catch_unwind(|| unsafe {
                    basis_change_plane(&plan, &mut [0; 32], Primes42::Q[0]);
                })
                .is_err()
            );
        }
    }
}
