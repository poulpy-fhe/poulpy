//! Standard-ring items of [`NTT3x42Ifma`].

use poulpy_cpu_portable::kernels::{
    ntt4x30::NttHandleFactory,
    znx::{ZnxAutomorphism, ZnxAutomorphismRotate, standard::znx_automorphism_portable},
};
use poulpy_hal::layouts::Standard;

use super::{
    NTT3x42Ifma,
    kernels::{intt_avx512, ntt_avx512},
    module::NTT3x42IfmaHandle,
    primes::{PrimeSetNtt3x42Ifma, Primes42},
    tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv},
    traits::Ntt3x42IfmaDFTExecute,
};
use crate::znx_avx512::{znx_automorphism_avx512, znx_automorphism_rotate_avx512};

impl<P: PrimeSetNtt3x42Ifma> Ntt3x42IfmaTable<P, Standard> {
    pub fn new(n: usize) -> Self {
        Self::build(n, Default::default())
    }
}

impl<P: PrimeSetNtt3x42Ifma> Ntt3x42IfmaTableInv<P, Standard> {
    pub fn new(n: usize) -> Self {
        Self::build(n, Default::default())
    }
}

unsafe impl NttHandleFactory for NTT3x42IfmaHandle {
    fn create_ntt_handle(n: usize) -> Self {
        Self::with_tables(n, Ntt3x42IfmaTable::<Primes42>::new, Ntt3x42IfmaTableInv::<Primes42>::new)
    }
}

impl poulpy_hal::layouts::MaxBase2k for NTT3x42Ifma {
    fn max_base2k(n: usize, products: usize, failure_bits: usize, squaring: bool) -> Option<usize> {
        Some(poulpy_hal::layouts::max_base2k_ntt::<Self>(
            <Primes42 as poulpy_hal::layouts::PrimeSet>::LOG_Q_PRODUCT,
            n,
            products,
            failure_bits,
            squaring,
        ))
    }
}

impl ZnxAutomorphism for NTT3x42Ifma {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_automorphism_avx512(p, res, a) }
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_portable(p, res, a)
    }
}

impl ZnxAutomorphismRotate for NTT3x42Ifma {
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
}

impl Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42>> for NTT3x42Ifma {
    #[inline(always)]
    fn ntt3x42_ifma_dft_execute(table: &Ntt3x42IfmaTableInv<Primes42>, data: &mut [u64]) {
        unsafe { intt_avx512::<Primes42>(table, data) }
    }
}
