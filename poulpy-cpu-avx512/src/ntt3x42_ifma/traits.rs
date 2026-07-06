//! NTT-domain operation traits for the 3-prime IFMA backend.
//!
//! These traits are implemented by [`NTT3x42Ifma`](super::super::NTT3x42Ifma) using
//! AVX-512-IFMA SIMD intrinsics. Their scalar reference companions live in
//! [`super::reference`] and are used as test oracles.

use super::bbc_meta::Bbc126IfmaMeta;
use super::primes::Primes42;

/// Execute a forward or inverse NTT using a precomputed table.
pub trait Ntt3x42IfmaDFTExecute<Table> {
    fn ntt3x42_ifma_dft_execute(table: &Table, data: &mut [u64]);
}

/// Load a polynomial from i64 coefficients into 3-prime CRT format.
///
/// `res` has length `3 * a.len()` (one plane per CRT prime).
pub trait Ntt3x42IfmaFromZnx64 {
    fn ntt3x42_ifma_from_znx64(res: &mut [u64], a: &[i64]);

    fn ntt3x42_ifma_from_znx64_masked(res: &mut [u64], a: &[i64], mask: i64) {
        super::reference::arithmetic::b_ntt3x42_ifma_from_znx64_masked_ref(a.len(), res, a, mask)
    }
}

/// Recover `i128` coefficients from 3-prime CRT format via Garner's algorithm.
pub trait Ntt3x42IfmaToZnx128 {
    fn ntt3x42_ifma_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]);
}

/// Component-wise addition of two CRT vectors.
pub trait Ntt3x42IfmaAdd {
    fn ntt3x42_ifma_add(res: &mut [u64], a: &[u64], b: &[u64]);
}

/// In-place component-wise addition.
pub trait Ntt3x42IfmaAddAssign {
    fn ntt3x42_ifma_add_assign(res: &mut [u64], a: &[u64]);
}

/// Component-wise subtraction.
pub trait Ntt3x42IfmaSub {
    fn ntt3x42_ifma_sub(res: &mut [u64], a: &[u64], b: &[u64]);
}

/// In-place component-wise subtraction.
pub trait Ntt3x42IfmaSubAssign {
    fn ntt3x42_ifma_sub_assign(res: &mut [u64], a: &[u64]);
}

/// In-place swap-subtract: `res = a - res`.
pub trait Ntt3x42IfmaSubNegateAssign {
    fn ntt3x42_ifma_sub_negate_assign(res: &mut [u64], a: &[u64]);
}

/// Component-wise negation.
pub trait Ntt3x42IfmaNegate {
    fn ntt3x42_ifma_negate(res: &mut [u64], a: &[u64]);
}

/// In-place negation.
pub trait Ntt3x42IfmaNegateAssign {
    fn ntt3x42_ifma_negate_assign(res: &mut [u64]);
}

/// Zero a CRT vector.
pub trait Ntt3x42IfmaZero {
    fn ntt3x42_ifma_zero(res: &mut [u64]);
}

/// Copy a CRT vector.
pub trait Ntt3x42IfmaCopy {
    fn ntt3x42_ifma_copy(res: &mut [u64], a: &[u64]);
}

/// Pointwise product: b × c → b (overwrite).
///
/// `ntt_coeff` is in b format (as u32 view), `prepared` is in Harvey-prepared c format.
#[allow(dead_code)]
pub trait Ntt3x42IfmaMulBbc {
    fn ntt3x42_ifma_mul_bbc(meta: &Bbc126IfmaMeta<Primes42>, ell: usize, res: &mut [u64], ntt_coeff: &[u32], prepared: &[u32]);
}

/// Convert b → c (Harvey-prepared form).
pub trait Ntt3x42IfmaCFromB {
    fn ntt3x42_ifma_c_from_b(n: usize, res: &mut [u32], a: &[u64]);
}
