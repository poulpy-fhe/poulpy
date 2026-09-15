//! Scalar dot products with wide products and explicit modular reduction.

use std::marker::PhantomData;

use super::primes::PrimeSetCrt4;
use bytemuck::{cast_slice, cast_slice_mut};

pub struct BaaMeta<P: PrimeSetCrt4>(PhantomData<P>);
impl<P: PrimeSetCrt4> BaaMeta<P> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}
impl<P: PrimeSetCrt4> Default for BaaMeta<P> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct BbbMeta<P: PrimeSetCrt4>(PhantomData<P>);
impl<P: PrimeSetCrt4> BbbMeta<P> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}
impl<P: PrimeSetCrt4> Default for BbbMeta<P> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct BbcMeta<P: PrimeSetCrt4>(PhantomData<P>);
impl<P: PrimeSetCrt4> BbcMeta<P> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}
impl<P: PrimeSetCrt4> Default for BbcMeta<P> {
    fn default() -> Self {
        Self::new()
    }
}

pub fn vec_mat1col_product_baa_ref<P: PrimeSetCrt4>(_: &BaaMeta<P>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    assert!(res.len() >= 4 && x.len() >= 4 * ell && y.len() >= 4 * ell);
    for k in 0..4 {
        let q = P::Q[k] as u128;
        let mut sum = 0u128;
        for i in 0..ell {
            sum = (sum + (x[4 * i + k] as u128 * y[4 * i + k] as u128) % q) % q;
        }
        res[k] = sum as u64;
    }
}

pub fn vec_mat1col_product_bbb_ref<P: PrimeSetCrt4>(_: &BbbMeta<P>, ell: usize, res: &mut [u64], x: &[u64], y: &[u64]) {
    assert!(res.len() >= 4 && x.len() >= 4 * ell && y.len() >= 4 * ell);
    for k in 0..4 {
        let q = P::Q[k] as u128;
        let mut sum = 0u128;
        for i in 0..ell {
            sum = (sum + (x[4 * i + k] as u128 * y[4 * i + k] as u128) % q) % q;
        }
        res[k] = sum as u64;
    }
}

pub(crate) fn accum_mul_q120_bc(s: &mut [u64; 8], x: &[u32; 8], y: &[u32; 8]) {
    for k in 0..4 {
        let sum = s[2 * k] as u128
            + ((s[2 * k + 1] as u128) << 64)
            + x[2 * k] as u128 * y[2 * k] as u128
            + x[2 * k + 1] as u128 * y[2 * k + 1] as u128;
        s[2 * k] = sum as u64;
        s[2 * k + 1] = (sum >> 64) as u64;
    }
}

pub(crate) fn accum_to_q120b<P: PrimeSetCrt4>(res: &mut [u64; 4], s: &[u64; 8], _: &BbcMeta<P>) {
    for k in 0..4 {
        let sum = s[2 * k] as u128 + ((s[2 * k + 1] as u128) << 64);
        res[k] = (sum % P::Q[k] as u128) as u64;
    }
}

pub fn vec_mat1col_product_bbc_ref<P: PrimeSetCrt4>(meta: &BbcMeta<P>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    assert!(res.len() >= 4);
    assert!(x.len() >= 8 * ell);
    assert!(y.len() >= 8 * ell);

    let xs: &[[u32; 8]] = cast_slice(&x[..8 * ell]);
    let ys: &[[u32; 8]] = cast_slice(&y[..8 * ell]);
    let mut s = [0u64; 8];
    for i in 0..ell {
        accum_mul_q120_bc(&mut s, &xs[i], &ys[i]);
    }
    let res4: &mut [u64; 4] = (&mut res[..4]).try_into().unwrap();
    accum_to_q120b::<P>(res4, &s, meta);
}

/// Computes two q120b dot products simultaneously (x2 variant).
///
/// `x` contains two interleaved q120b vectors (each of length `ell`),
/// and `y` contains two interleaved q120c vectors.
/// Both output q120b values are written into `res` (8 contiguous u64s).
pub fn vec_mat1col_product_x2_bbc_ref<P: PrimeSetCrt4>(meta: &BbcMeta<P>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    assert!(res.len() >= 8);
    assert!(x.len() >= 16 * ell);
    assert!(y.len() >= 16 * ell);

    let xs: &[[u32; 8]] = cast_slice(&x[..16 * ell]);
    let ys: &[[u32; 8]] = cast_slice(&y[..16 * ell]);
    let mut s = [[0u64; 8]; 2];
    for i in 0..ell {
        accum_mul_q120_bc(&mut s[0], &xs[2 * i], &ys[2 * i]);
        accum_mul_q120_bc(&mut s[1], &xs[2 * i + 1], &ys[2 * i + 1]);
    }
    let res4: &mut [[u64; 4]] = cast_slice_mut(&mut res[..8]);
    accum_to_q120b::<P>(&mut res4[0], &s[0], meta);
    accum_to_q120b::<P>(&mut res4[1], &s[1], meta);
}

/// Computes four q120b dot products (two output, two columns).
///
/// Equivalent to calling `vec_mat1col_product_x2_bbc_ref` twice with
/// two different column slices of `y`, accumulating into `res[0..8]`
/// and `res[8..16]` respectively.
pub fn vec_mat2cols_product_x2_bbc_ref<P: PrimeSetCrt4>(meta: &BbcMeta<P>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    assert!(res.len() >= 16);
    assert!(x.len() >= 16 * ell);
    assert!(y.len() >= 32 * ell);

    let xs: &[[u32; 8]] = cast_slice(&x[..16 * ell]);
    let ys: &[[u32; 8]] = cast_slice(&y[..32 * ell]);
    let mut s = [[0u64; 8]; 4];
    for i in 0..ell {
        accum_mul_q120_bc(&mut s[0], &xs[2 * i], &ys[4 * i]);
        accum_mul_q120_bc(&mut s[1], &xs[2 * i + 1], &ys[4 * i + 1]);
        accum_mul_q120_bc(&mut s[2], &xs[2 * i], &ys[4 * i + 2]);
        accum_mul_q120_bc(&mut s[3], &xs[2 * i + 1], &ys[4 * i + 3]);
    }
    let res4: &mut [[u64; 4]] = cast_slice_mut(&mut res[..16]);
    for (r, si) in res4.iter_mut().zip(&s) {
        accum_to_q120b::<P>(r, si, meta);
    }
}

/// Extracts one block of 4 q120b coefficients (= 8 u64 values) from
/// a q120b NTT vector of length `nn`, copying into `dst`.
///
/// A "block" here groups 2 consecutive NTT coefficients (indices
/// `2*blk` and `2*blk+1`), so `blk < nn/2`.
///
/// This is the Rust port of `q120x2_extract_1blk_from_q120b_ref`.
pub fn extract_1blk_from_q120b_ref(nn: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    assert!(blk < nn / 2);
    assert!(dst.len() >= 8);
    assert!(src.len() >= 4 * nn);

    dst[..8].copy_from_slice(&src[8 * blk..8 * blk + 8]);
}

/// Extracts one block from a contiguous array of `nrows` q120b NTT
/// vectors, each of length `nn`.
///
/// `dst` receives `nrows` consecutive blocks of 8 u64 each.
/// `src` is laid out as `[row_0 || row_1 || ... || row_{nrows-1}]`
/// where each row has `4*nn` u64 values.
///
/// Port of `q120x2_extract_1blk_from_contiguous_q120b_ref`.
pub fn extract_1blk_from_contiguous_q120b_ref(nn: usize, nrows: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    assert!(blk < nn / 2);
    assert!(dst.len() >= 8 * nrows);
    assert!(src.len() >= 4 * nn * nrows);

    for row in 0..nrows {
        let src_base = 4 * nn * row;
        let dst_base = 8 * row;
        dst[dst_base..dst_base + 8].copy_from_slice(&src[src_base + 8 * blk..src_base + 8 * blk + 8]);
    }
}

/// Saves one q120x2b block (8 u64 values) into the corresponding
/// position of a q120b NTT vector of length `nn`.
///
/// Port of `q120x2b_save_1blk_to_q120b_ref`.
pub fn save_1blk_to_q120b_ref(nn: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    assert!(blk < nn / 2);
    assert!(src.len() >= 8);
    assert!(dst.len() >= 4 * nn);

    dst[8 * blk..8 * blk + 8].copy_from_slice(&src[..8]);
}

#[cfg(test)]
mod tests {
    use super::super::primes::{PrimeSet, Primes30};
    use super::*;

    #[test]
    fn prepared_dot_matches_direct_residue_products() {
        let x = [u64::MAX, 1, 0, u64::MAX - 1, 7, 11, 13, 17];
        let y = [u64::MAX - 2, 19, 23, 29, 31, 37, 41, 43];
        let mut prepared = [0u32; 16];
        super::super::arithmetic::c_from_b_ref::<Primes30>(2, &mut prepared, &y);
        let mut actual = [0u64; 4];
        vec_mat1col_product_bbc_ref(&BbcMeta::<Primes30>::new(), 2, &mut actual, cast_slice(&x), &prepared);
        for k in 0..4 {
            let q = Primes30::Q[k] as u128;
            let expected = ((x[k] as u128 * y[k] as u128) % q + (x[k + 4] as u128 * y[k + 4] as u128) % q) % q;
            assert_eq!(actual[k], expected as u64);
        }
    }
}
