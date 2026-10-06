//! Standard-ring items of [`NTT4x30Neon`].

use poulpy_cpu_portable::kernels::{
    ntt4x30::{
        NttDFTExecute,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
    },
    znx::{ZnxAutomorphism, ZnxAutomorphismRotate, standard::znx_automorphism_portable},
};

use super::NTT4x30Neon;
use crate::neon::znx::{znx_automorphism_neon as kn_automorphism, znx_automorphism_rotate_neon as kn_automorphism_rotate};
use poulpy_cpu_portable::kernels::ntt4x30::standard::{intt_portable, ntt_portable};

impl poulpy_hal::layouts::MaxBase2k for NTT4x30Neon {
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

impl ZnxAutomorphism for NTT4x30Neon {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        kn_automorphism(p, res, a);
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_portable(p, res, a)
    }
}

impl ZnxAutomorphismRotate for NTT4x30Neon {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        kn_automorphism_rotate(p, k, res, a);
    }
}

// The backend transforms packed limbs with its own kernels.
// These q120 transforms only serve the reference bodies that are generic over the q120 layout.
impl NttDFTExecute<NttTable<Primes30>> for NTT4x30Neon {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30>, data: &mut [u64]) {
        ntt_portable::<Primes30>(table, data);
    }
}

impl NttDFTExecute<NttTableInv<Primes30>> for NTT4x30Neon {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30>, data: &mut [u64]) {
        intt_portable::<Primes30>(table, data);
    }
}
