//! Standard-ring items of [`NTT3x42Ifma`].

use poulpy_cpu_portable::kernels::{
    ntt4x30::{standard::build_ntt4x30_automorphism_plan_portable, vec_znx_dft::NttAutomorphismPlan},
    znx::{ZnxAutomorphism, ZnxAutomorphismRotate, standard::znx_automorphism_portable},
};
use poulpy_hal::layouts::Standard;

use super::{
    NTT3x42Ifma,
    kernels::{intt_avx512, ntt_avx512},
    module::DegreeTables,
    primes::Primes42,
    tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv},
    traits::Ntt3x42IfmaDFTExecute,
};
use crate::znx_avx512::{znx_automorphism_avx512, znx_automorphism_rotate_avx512};

/// The standard-ring tables of degree `n`.
pub(super) fn tables(n: usize) -> DegreeTables<Standard> {
    DegreeTables {
        ntt: Ntt3x42IfmaTable::<Primes42>::new(n),
        intt: Ntt3x42IfmaTableInv::<Primes42>::new(n),
    }
}

impl ZnxAutomorphism<Standard> for NTT3x42Ifma {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_automorphism_avx512(p, res, a) }
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_portable(p, res, a)
    }
}

impl ZnxAutomorphismRotate<Standard> for NTT3x42Ifma {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_automorphism_rotate_avx512(p, k, res, a) }
    }
}

impl Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42>> for NTT3x42Ifma {
    #[inline(always)]
    fn ntt3x42_ifma_dft_execute(table: &Ntt3x42IfmaTable<Primes42>, data: &mut [u64]) {
        // Non-lazy: fully reduce for the public DFT contract.
        unsafe { ntt_avx512::<Primes42>(table, data, false) }
    }

    #[inline(always)]
    fn ntt3x42_ifma_dft_execute_lazy(table: &Ntt3x42IfmaTable<Primes42>, data: &mut [u64]) {
        unsafe { ntt_avx512::<Primes42>(table, data, true) }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan<Standard> {
        build_ntt4x30_automorphism_plan_portable(n, p)
    }
}

impl Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42>> for NTT3x42Ifma {
    #[inline(always)]
    fn ntt3x42_ifma_dft_execute(table: &Ntt3x42IfmaTableInv<Primes42>, data: &mut [u64]) {
        unsafe { intt_avx512::<Primes42>(table, data) }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan<Standard> {
        build_ntt4x30_automorphism_plan_portable(n, p)
    }
}
