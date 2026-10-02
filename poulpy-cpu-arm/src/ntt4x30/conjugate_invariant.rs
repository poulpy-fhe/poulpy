//! Conjugate invariant ring items of [`NTT4x30Neon`].

use poulpy_cpu_ref::reference::{
    ntt4x30::{
        NttDFTExecute,
        conjugate_invariant::build_ntt4x30_automorphism_plan,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
        vec_znx_dft::NttAutomorphismPlan,
    },
    znx::{ZnxAutomorphism, conjugate_invariant::znx_automorphism_ref},
};
use poulpy_hal::layouts::ConjugateInvariant;

use super::NTT4x30Neon;
#[cfg(target_arch = "aarch64")]
use crate::neon::{
    ntt4x30_conjugate_invariant::basis_change,
    ntt4x30_ntt::{intt_neon, ntt_neon},
};
#[cfg(not(target_arch = "aarch64"))]
use poulpy_cpu_ref::reference::ntt4x30::conjugate_invariant::{intt_ref, ntt_ref};

/// No failure model: a square's constant coefficient has a large positive mean on this ring.
impl poulpy_hal::layouts::MaxBase2k for NTT4x30Neon<ConjugateInvariant> {
    fn max_base2k(_n: usize, _products: usize, _failure_bits: usize, _squaring: bool) -> Option<usize> {
        None
    }
}

impl ZnxAutomorphism for NTT4x30Neon<ConjugateInvariant> {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_ref(p, res, a)
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_ref(p, res, a)
    }
}

impl NttDFTExecute<NttTable<Primes30, ConjugateInvariant>> for NTT4x30Neon<ConjugateInvariant> {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            basis_change::<Primes30>(table.basis_changes(), data);
            ntt_neon::<Primes30>(table, data);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            ntt_ref::<Primes30>(table, data);
        }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan {
        build_ntt4x30_automorphism_plan(n, p)
    }
}

impl NttDFTExecute<NttTableInv<Primes30, ConjugateInvariant>> for NTT4x30Neon<ConjugateInvariant> {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            intt_neon::<Primes30>(table, data);
            basis_change::<Primes30>(table.basis_changes(), data);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            intt_ref::<Primes30>(table, data);
        }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan {
        build_ntt4x30_automorphism_plan(n, p)
    }
}
