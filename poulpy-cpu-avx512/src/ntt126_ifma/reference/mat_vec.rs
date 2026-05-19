//! Scalar reference matrix-vector product kernels for the 3-prime IFMA backend.
//!
//! Used by [`super::super::mat_vec_ifma`] tests as the per-coefficient
//! correctness oracle for the AVX-512-IFMA BBC kernels.  The shared
//! [`Bbc126IfmaMeta`](super::super::bbc_meta::Bbc126IfmaMeta) metadata
//! struct lives in the production module.

use super::super::bbc_meta::Bbc126IfmaMeta;
use super::super::primes::{PrimeSetNtt126Ifma, Primes42};

/// Reference BBC inner product: `res = Σᵢ x[i] · y[i]` for 3-prime IFMA.
///
/// Both `x` and `y` are in `[u32]` view but actually contain u64 data
/// (each pair of u32s forms one u64). The data layout is 4 u64 per
/// coefficient (3 active + 1 padding).
pub fn vec_mat1col_product_bbc_ntt126_ifma_ref(
    _meta: &Bbc126IfmaMeta<Primes42>,
    ell: usize,
    res: &mut [u64],
    x: &[u32],
    y: &[u32],
) {
    let q = <Primes42 as PrimeSetNtt126Ifma>::Q;
    // Reinterpret u32 slices as u64
    let x_u64: &[u64] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u64, x.len() / 2) };
    let y_u64: &[u64] = unsafe { std::slice::from_raw_parts(y.as_ptr() as *const u64, y.len() / 2) };

    let mut accum = [0u128; 3];
    for i in 0..ell {
        for k in 0..3 {
            let xv = x_u64[4 * i + k] % q[k];
            let yv = y_u64[4 * i + k] % q[k];
            accum[k] += xv as u128 * yv as u128;
        }
    }
    for k in 0..3 {
        res[k] = (accum[k] % q[k] as u128) as u64;
    }
    res[3] = 0;
}

/// Reference x2-block 1-column BBC product.
pub fn vec_mat1col_product_x2_bbc_ntt126_ifma_ref(
    _meta: &Bbc126IfmaMeta<Primes42>,
    ell: usize,
    res: &mut [u64],
    x: &[u32],
    y: &[u32],
) {
    let q = <Primes42 as PrimeSetNtt126Ifma>::Q;
    let x_u64: &[u64] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u64, x.len() / 2) };
    let y_u64: &[u64] = unsafe { std::slice::from_raw_parts(y.as_ptr() as *const u64, y.len() / 2) };

    let mut accum_a = [0u128; 3];
    let mut accum_b = [0u128; 3];

    for i in 0..ell {
        for k in 0..3 {
            // Pair A: x[2i], y[2i]
            let xa = x_u64[8 * i + k] % q[k];
            let ya = y_u64[8 * i + k] % q[k];
            accum_a[k] += xa as u128 * ya as u128;

            // Pair B: x[2i+1], y[2i+1]
            let xb = x_u64[8 * i + 4 + k] % q[k];
            let yb = y_u64[8 * i + 4 + k] % q[k];
            accum_b[k] += xb as u128 * yb as u128;
        }
    }

    for k in 0..3 {
        res[k] = (accum_a[k] % q[k] as u128) as u64;
        res[4 + k] = (accum_b[k] % q[k] as u128) as u64;
    }
    res[3] = 0;
    res[7] = 0;
}

/// Reference x2-block 2-column BBC product.
pub fn vec_mat2cols_product_x2_bbc_ntt126_ifma_ref(
    _meta: &Bbc126IfmaMeta<Primes42>,
    ell: usize,
    res: &mut [u64],
    x: &[u32],
    y: &[u32],
) {
    let q = <Primes42 as PrimeSetNtt126Ifma>::Q;
    let x_u64: &[u64] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u64, x.len() / 2) };
    let y_u64: &[u64] = unsafe { std::slice::from_raw_parts(y.as_ptr() as *const u64, y.len() / 2) };

    let mut acc = [[0u128; 3]; 4]; // [col0_a, col0_b, col1_a, col1_b]

    for i in 0..ell {
        for k in 0..3 {
            let xa = x_u64[8 * i + k] % q[k];
            let xb = x_u64[8 * i + 4 + k] % q[k];

            let yc0a = y_u64[16 * i + k] % q[k];
            let yc0b = y_u64[16 * i + 4 + k] % q[k];
            let yc1a = y_u64[16 * i + 8 + k] % q[k];
            let yc1b = y_u64[16 * i + 12 + k] % q[k];

            acc[0][k] += xa as u128 * yc0a as u128;
            acc[1][k] += xb as u128 * yc0b as u128;
            acc[2][k] += xa as u128 * yc1a as u128;
            acc[3][k] += xb as u128 * yc1b as u128;
        }
    }

    for j in 0..4 {
        for k in 0..3 {
            res[4 * j + k] = (acc[j][k] % q[k] as u128) as u64;
        }
        res[4 * j + 3] = 0;
    }
}

/// Extract one x2-block from a contiguous 3-prime CRT array.
pub fn extract_1blk_from_contiguous_ntt126_ifma_ref(n: usize, row_max: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    // Each x2-block = 2 consecutive coefficients = 8 u64
    let coeff_idx = blk * 2;
    for row in 0..row_max {
        let src_base = row * 4 * n + 4 * coeff_idx;
        let dst_base = row * 8;
        for j in 0..8 {
            dst[dst_base + j] = if src_base + j < src.len() { src[src_base + j] } else { 0 };
        }
    }
}
