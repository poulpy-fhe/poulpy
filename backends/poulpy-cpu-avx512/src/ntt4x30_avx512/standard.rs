//! Standard-ring items of [`NTT4x30Avx512`].

use poulpy_cpu_portable::kernels::{
    ntt4x30::{
        NttDFTExecute,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
        standard::build_ntt4x30_automorphism_plan_portable,
        vec_znx_dft::NttAutomorphismPlan,
    },
    znx::{ZnxAutomorphism, ZnxAutomorphismRotate, standard::znx_automorphism_portable},
};
use poulpy_hal::layouts::Standard;

use super::{
    NTT4x30Avx512,
    ntt::{intt_avx512, ntt_avx512},
};
use crate::znx_avx512::{znx_automorphism_avx512, znx_automorphism_rotate_avx512};

impl ZnxAutomorphism<Standard> for NTT4x30Avx512 {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_automorphism_avx512(p, res, a) }
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_portable(p, res, a)
    }
}

impl ZnxAutomorphismRotate<Standard> for NTT4x30Avx512 {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_automorphism_rotate_avx512(p, k, res, a) }
    }
}

impl NttDFTExecute<NttTable<Primes30>> for NTT4x30Avx512 {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30>, data: &mut [u64]) {
        // SAFETY: NTT4x30Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { ntt_avx512::<Primes30>(table, data) }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan<Standard> {
        build_ntt4x30_automorphism_plan_portable(n, p)
    }
}

impl NttDFTExecute<NttTableInv<Primes30>> for NTT4x30Avx512 {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30>, data: &mut [u64]) {
        // SAFETY: NTT4x30Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { intt_avx512::<Primes30>(table, data) }
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan<Standard> {
        build_ntt4x30_automorphism_plan_portable(n, p)
    }
}
