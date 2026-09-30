//! Trait implementations for [`NTT4x30Portable`] — primitive NTT-domain operations.
//!
//! Implements all `Ntt*` traits from [`crate::reference::ntt4x30`] for
//! [`NTT4x30Portable`], delegating to the `*_ref` scalar functions.
//!
//! This mirrors `poulpy_cpu_portable::fft64::reim` for the FFT64 backend.

use crate::reference::ntt4x30::{
    NttAdd, NttAddAssign, NttCFromB, NttCopy, NttDFTExecute, NttExtract1BlkContiguous, NttFromZnx64, NttMulBbb, NttMulBbc,
    NttMulBbc1ColX2, NttMulBbc2ColsX2, NttNegate, NttNegateAssign, NttPackLeft1BlkX2, NttPackRight1BlkX2,
    NttPairwisePackLeft1BlkX2, NttPairwisePackRight1BlkX2, NttSub, NttSubAssign, NttSubNegateAssign, NttToZnx128, NttZero,
    arithmetic::{add_bbb_ref, b_from_znx64_ref, b_to_znx128_ref, c_from_b_ref},
    conjugate_invariant,
    mat_vec::{
        BbbMeta, BbcMeta, extract_1blk_from_contiguous_q120b_ref, vec_mat1col_product_bbb_ref, vec_mat1col_product_bbc_ref,
        vec_mat1col_product_x2_bbc_ref, vec_mat2cols_product_x2_bbc_ref,
    },
    ntt::{NttTable, NttTableInv},
    primes::{PrimeSet, Primes30},
    standard,
    types::Q_SHIFTED,
};
use poulpy_hal::layouts::{ConjugateInvariant, Ring, Standard};

use super::NTT4x30Portable;

// ──────────────────────────────────────────────────────────────────────────────
// NTT execution
// ──────────────────────────────────────────────────────────────────────────────

impl NttDFTExecute<NttTable<Primes30, Standard>> for NTT4x30Portable<Standard> {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30, Standard>, data: &mut [u64]) {
        standard::ntt_ref::<Primes30>(table, data);
    }
}

impl NttDFTExecute<NttTableInv<Primes30, Standard>> for NTT4x30Portable<Standard> {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30, Standard>, data: &mut [u64]) {
        standard::intt_ref::<Primes30>(table, data);
    }
}

impl NttDFTExecute<NttTable<Primes30, ConjugateInvariant>> for NTT4x30Portable<ConjugateInvariant> {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        conjugate_invariant::ntt_ref::<Primes30>(table, data);
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> crate::reference::ntt4x30::vec_znx_dft::NttAutomorphismPlan {
        conjugate_invariant::build_ntt4x30_automorphism_plan(n, p)
    }
}

impl NttDFTExecute<NttTableInv<Primes30, ConjugateInvariant>> for NTT4x30Portable<ConjugateInvariant> {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        conjugate_invariant::intt_ref::<Primes30>(table, data);
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> crate::reference::ntt4x30::vec_znx_dft::NttAutomorphismPlan {
        conjugate_invariant::build_ntt4x30_automorphism_plan(n, p)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Domain conversion
// ──────────────────────────────────────────────────────────────────────────────

impl<R: Ring> NttFromZnx64 for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_from_znx64(res: &mut [u64], a: &[i64]) {
        b_from_znx64_ref::<Primes30>(a.len(), res, a);
    }
}

impl<R: Ring> NttToZnx128 for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]) {
        b_to_znx128_ref::<Primes30>(divisor_is_n, res, a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Addition / subtraction / negation / copy / zero
// ──────────────────────────────────────────────────────────────────────────────

impl<R: Ring> NttAdd for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        add_bbb_ref::<Primes30>(res.len() / 4, res, a, b);
    }
}

impl<R: Ring> NttAddAssign for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_add_assign(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = res[idx] % q_s + a[idx] % q_s;
            }
        }
    }
}

impl<R: Ring> NttSub for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = a[idx] % q_s + (q_s - b[idx] % q_s);
            }
        }
    }
}

impl<R: Ring> NttSubAssign for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_sub_assign(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = res[idx] % q_s + (q_s - a[idx] % q_s);
            }
        }
    }
}

impl<R: Ring> NttSubNegateAssign for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_sub_negate_assign(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = a[idx] % q_s + (q_s - res[idx] % q_s);
            }
        }
    }
}

/// **Output range:** For a zero input the result is `Q_SHIFTED\[k\]` (≡ 0 mod `Q\[k\]`), not `0`.
/// Output range is `(0, Q_SHIFTED\[k\]\]`. Use `val % Q\[k\] == 0`, not `val == 0`, to test for zero.
impl<R: Ring> NttNegate for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_negate(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = q_s - a[idx] % q_s;
            }
        }
    }
}

/// **Output range:** For a zero input the result is `Q_SHIFTED\[k\]` (≡ 0 mod `Q\[k\]`), not `0`.
/// Output range is `(0, Q_SHIFTED\[k\]\]`. Use `val % Q\[k\] == 0`, not `val == 0`, to test for zero.
impl<R: Ring> NttNegateAssign for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_negate_assign(res: &mut [u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = q_s - res[idx] % q_s;
            }
        }
    }
}

impl<R: Ring> NttZero for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl<R: Ring> NttCopy for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Multiply-accumulate
// ──────────────────────────────────────────────────────────────────────────────

impl<R: Ring> NttMulBbb for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_mul_bbb(meta: &BbbMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
        vec_mat1col_product_bbb_ref::<Primes30>(meta, ell, res, a, b);
    }
}

impl<R: Ring> NttMulBbc for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_mul_bbc(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], ntt_coeff: &[u32], prepared: &[u32]) {
        vec_mat1col_product_bbc_ref::<Primes30>(meta, ell, res, ntt_coeff, prepared);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// q120b → q120c conversion
// ──────────────────────────────────────────────────────────────────────────────

impl<R: Ring> NttCFromB for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_c_from_b(n: usize, res: &mut [u32], a: &[u64]) {
        c_from_b_ref::<Primes30>(n, res, a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// VMP x2-block kernels
// ──────────────────────────────────────────────────────────────────────────────

impl<R: Ring> NttMulBbc1ColX2 for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_mul_bbc_1col_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        vec_mat1col_product_x2_bbc_ref::<Primes30>(meta, ell, res, a, b);
    }
}

impl<R: Ring> NttMulBbc2ColsX2 for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_mul_bbc_2cols_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        vec_mat2cols_product_x2_bbc_ref::<Primes30>(meta, ell, res, a, b);
    }
}

impl<R: Ring> NttExtract1BlkContiguous for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_extract_1blk_contiguous(n: usize, row_max: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
        extract_1blk_from_contiguous_q120b_ref(n, row_max, blk, dst, src);
    }
}

impl<R: Ring> NttPackLeft1BlkX2 for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_pack_left_1blk_x2(dst: &mut [u32], a: &[u64], row_count: usize, row_stride: usize, blk: usize) {
        assert!(dst.len() >= 16 * row_count);
        assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);

        for row in 0..row_count {
            let row_base = row * row_stride + 8 * blk;
            let out_base = 16 * row;
            for coeff in 0..2 {
                for prime in 0..4 {
                    let idx = row_base + 4 * coeff + prime;
                    let q = Primes30::Q[prime] as u64;
                    let a_red = a[idx] % q;
                    dst[out_base + 8 * coeff + 2 * prime] = a_red as u32;
                    dst[out_base + 8 * coeff + 2 * prime + 1] = 0;
                }
            }
        }
    }
}

impl<R: Ring> NttPackRight1BlkX2 for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_pack_right_1blk_x2(dst: &mut [u32], a: &[u32], row_count: usize, row_stride: usize, blk: usize) {
        assert!(dst.len() >= 16 * row_count);
        assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 16 * blk + 16);

        for row in 0..row_count {
            let row_base = (row_count - 1 - row) * row_stride + 16 * blk;
            let out_base = 16 * row;
            dst[out_base..out_base + 16].copy_from_slice(&a[row_base..row_base + 16]);
        }
    }
}

impl<R: Ring> NttPairwisePackLeft1BlkX2 for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_pairwise_pack_left_1blk_x2(dst: &mut [u32], a: &[u64], b: &[u64], row_count: usize, row_stride: usize, blk: usize) {
        assert!(dst.len() >= 16 * row_count);
        assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
        assert!(b.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);

        for row in 0..row_count {
            let row_base = row * row_stride + 8 * blk;
            let out_base = 16 * row;
            for coeff in 0..2 {
                for prime in 0..4 {
                    let idx = row_base + 4 * coeff + prime;
                    let q = Primes30::Q[prime] as u64;
                    let mut sum = (a[idx] % q) + (b[idx] % q);
                    if sum >= q {
                        sum -= q;
                    }
                    dst[out_base + 8 * coeff + 2 * prime] = sum as u32;
                    dst[out_base + 8 * coeff + 2 * prime + 1] = 0;
                }
            }
        }
    }
}

impl<R: Ring> NttPairwisePackRight1BlkX2 for NTT4x30Portable<R> {
    #[inline(always)]
    fn ntt_pairwise_pack_right_1blk_x2(dst: &mut [u32], a: &[u32], b: &[u32], row_count: usize, row_stride: usize, blk: usize) {
        assert!(dst.len() >= 16 * row_count);
        assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 16 * blk + 16);
        assert!(b.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 16 * blk + 16);

        for row in 0..row_count {
            let row_base = (row_count - 1 - row) * row_stride + 16 * blk;
            let out_base = 16 * row;
            for idx in 0..16 {
                dst[out_base + idx] = a[row_base + idx] + b[row_base + idx];
            }
        }
    }
}
