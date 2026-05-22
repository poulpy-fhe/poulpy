//! NTT-domain operation traits for the 3-prime IFMA backend.
//!
//! These traits are implemented by [`NTT126Ifma`](super::super::NTT126Ifma) using
//! AVX-512-IFMA SIMD intrinsics. Their scalar reference companions live in
//! [`super::reference`] and are used as test oracles.

use super::bbc_meta::Bbc126IfmaMeta;
use super::primes::Primes42;

/// Execute a forward or inverse NTT using a precomputed table.
pub trait Ntt126IfmaDFTExecute<Table> {
    fn ntt126_ifma_dft_execute(table: &Table, data: &mut [u64]);
}

/// Load a polynomial from i64 coefficients into 3-prime CRT format.
///
/// `res` has length `4 * a.len()` (3 active residues + 1 padding per coefficient).
pub trait Ntt126IfmaFromZnx64 {
    fn ntt126_ifma_from_znx64(res: &mut [u64], a: &[i64]);

    fn ntt126_ifma_from_znx64_masked(res: &mut [u64], a: &[i64], mask: i64) {
        super::reference::arithmetic::b_ntt126_ifma_from_znx64_masked_ref(a.len(), res, a, mask)
    }
}

/// Recover `i128` coefficients from 3-prime CRT format via Garner's algorithm.
pub trait Ntt126IfmaToZnx128 {
    fn ntt126_ifma_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]);
}

/// Component-wise addition of two CRT vectors.
pub trait Ntt126IfmaAdd {
    fn ntt126_ifma_add(res: &mut [u64], a: &[u64], b: &[u64]);
}

/// In-place component-wise addition.
pub trait Ntt126IfmaAddAssign {
    fn ntt126_ifma_add_assign(res: &mut [u64], a: &[u64]);
}

/// Component-wise subtraction.
pub trait Ntt126IfmaSub {
    fn ntt126_ifma_sub(res: &mut [u64], a: &[u64], b: &[u64]);
}

/// In-place component-wise subtraction.
pub trait Ntt126IfmaSubAssign {
    fn ntt126_ifma_sub_assign(res: &mut [u64], a: &[u64]);
}

/// In-place swap-subtract: `res = a - res`.
pub trait Ntt126IfmaSubNegateAssign {
    fn ntt126_ifma_sub_negate_assign(res: &mut [u64], a: &[u64]);
}

/// Component-wise negation.
pub trait Ntt126IfmaNegate {
    fn ntt126_ifma_negate(res: &mut [u64], a: &[u64]);
}

/// In-place negation.
pub trait Ntt126IfmaNegateAssign {
    fn ntt126_ifma_negate_assign(res: &mut [u64]);
}

/// Zero a CRT vector.
pub trait Ntt126IfmaZero {
    fn ntt126_ifma_zero(res: &mut [u64]);
}

/// Copy a CRT vector.
pub trait Ntt126IfmaCopy {
    fn ntt126_ifma_copy(res: &mut [u64], a: &[u64]);
}

/// Pointwise product: b × c → b (overwrite).
///
/// `ntt_coeff` is in b format (as u32 view), `prepared` is in Harvey-prepared c format.
pub trait Ntt126IfmaMulBbc {
    fn ntt126_ifma_mul_bbc(meta: &Bbc126IfmaMeta<Primes42>, ell: usize, res: &mut [u64], ntt_coeff: &[u32], prepared: &[u32]);
}

/// Convert b → c (Harvey-prepared form).
pub trait Ntt126IfmaCFromB {
    fn ntt126_ifma_c_from_b(n: usize, res: &mut [u32], a: &[u64]);
}
