//! Conjugate invariant ring items of [`NTT4x30Neon`].

use poulpy_cpu_portable::kernels::{
    ntt4x30::{
        NttDFTExecute,
        conjugate_invariant::build_ntt4x30_automorphism_plan_portable,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
        vec_znx_dft::NttAutomorphismPlan,
    },
    znx::{ZnxAutomorphism, conjugate_invariant::znx_automorphism_portable},
};
use poulpy_hal::layouts::ConjugateInvariant;

use super::NTT4x30Neon;
use poulpy_cpu_portable::kernels::ntt4x30::conjugate_invariant::{intt_portable, ntt_portable};

/// No failure model: a square's constant coefficient has a large positive mean on this ring.
impl poulpy_hal::layouts::MaxBase2k for NTT4x30Neon<ConjugateInvariant> {
    fn max_base2k(_n: usize, _products: usize, _failure_bits: usize, _squaring: bool) -> Option<usize> {
        None
    }
}

impl ZnxAutomorphism for NTT4x30Neon<ConjugateInvariant> {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_portable(p, res, a)
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_portable(p, res, a)
    }
}

impl NttDFTExecute<NttTable<Primes30, ConjugateInvariant>> for NTT4x30Neon<ConjugateInvariant> {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        ntt_portable::<Primes30>(table, data);
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan {
        build_ntt4x30_automorphism_plan_portable(n, p)
    }
}

impl NttDFTExecute<NttTableInv<Primes30, ConjugateInvariant>> for NTT4x30Neon<ConjugateInvariant> {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        intt_portable::<Primes30>(table, data);
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan {
        build_ntt4x30_automorphism_plan_portable(n, p)
    }
}
