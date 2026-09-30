//! Standard-ring items of [`NTT4x30Avx`].

use poulpy_cpu_portable::reference::{
    ntt4x30::{
        NttDFTExecute,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
    },
    znx::{ZnxAutomorphism, ZnxAutomorphismRotate, standard::znx_automorphism_ref},
};

use super::{
    NTT4x30Avx,
    ntt::{intt_avx2, ntt_avx2},
};
use crate::znx_avx::{znx_automorphism_avx, znx_automorphism_rotate_avx};

impl poulpy_hal::layouts::MaxBase2k for NTT4x30Avx {
    fn max_base2k(n: usize, products: usize, failure_bits: usize, squaring: bool) -> Option<usize> {
        Some(poulpy_hal::layouts::max_base2k_ntt::<Self>(
            <Primes30 as poulpy_hal::layouts::PrimeSet>::LOG_Q_PRODUCT,
            n,
            products,
            failure_bits,
            squaring,
        ))
    }
}

impl ZnxAutomorphism for NTT4x30Avx {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_automorphism_avx(p, res, a) }
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_ref(p, res, a)
    }
}

impl ZnxAutomorphismRotate for NTT4x30Avx {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_automorphism_rotate_avx(p, k, res, a) }
    }
}

impl NttDFTExecute<NttTable<Primes30>> for NTT4x30Avx {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30>, data: &mut [u64]) {
        // SAFETY: NTT4x30Avx::new() verifies AVX2 availability at construction time.
        unsafe { ntt_avx2::<Primes30>(table, data) }
    }
}

impl NttDFTExecute<NttTableInv<Primes30>> for NTT4x30Avx {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30>, data: &mut [u64]) {
        // SAFETY: NTT4x30Avx::new() verifies AVX2 availability at construction time.
        unsafe { intt_avx2::<Primes30>(table, data) }
    }
}
