//! Trait implementations for [`NTT120Neon`](super::NTT120Neon) — primitive NTT-domain operations.

// `PrimeSet` is needed in scope on x86 (the scalar fallback paths use
// `Primes30::Q[prime]`); on aarch64 the NEON kernels never reference it,
// hence the allow(unused_imports).
#[cfg(not(target_arch = "aarch64"))]
use poulpy_cpu_ref::reference::ntt120::arithmetic::{b_from_znx64_ref, b_to_znx128_ref, c_from_b_ref};
#[allow(unused_imports)]
use poulpy_cpu_ref::reference::ntt120::{
    NttAdd, NttAddAssign, NttCFromB, NttCopy, NttDFTExecute, NttExtract1BlkContiguous, NttFromZnx64, NttMulBbb, NttMulBbc,
    NttMulBbc1ColX2, NttMulBbc2ColsX2, NttNegate, NttNegateAssign, NttPackLeft1BlkX2, NttPackRight1BlkX2,
    NttPairwisePackLeft1BlkX2, NttPairwisePackRight1BlkX2, NttSub, NttSubAssign, NttSubNegateAssign, NttToZnx128, NttZero,
    mat_vec::{
        BbbMeta, BbcMeta, extract_1blk_from_contiguous_q120b_ref, vec_mat1col_product_bbb_ref, vec_mat1col_product_bbc_ref,
        vec_mat1col_product_x2_bbc_ref, vec_mat2cols_product_x2_bbc_ref,
    },
    ntt::{NttTable, NttTableInv, intt_ref, ntt_ref},
    primes::{PrimeSet, Primes30},
};

use super::NTT120Neon;

// On aarch64, the q120b lazy-modular kernels and the i64↔q120b↔i128↔q120c
// conversion kernels are NEON; on other targets we fall back to the scalar
// reference functions so the negative-gate path on x86 emits only the
// intended `compile_error!`.
#[cfg(target_arch = "aarch64")]
use crate::neon::{
    ntt120_arithmetic::{
        ntt_add_assign_neon, ntt_add_neon, ntt_negate_assign_neon, ntt_negate_neon, ntt_sub_assign_neon,
        ntt_sub_negate_assign_neon, ntt_sub_neon,
    },
    ntt120_convert::{
        b_from_znx64_masked_neon, b_from_znx64_neon, b_to_znx128_neon, c_from_b_neon, pack_left_1blk_x2_neon,
        pack_right_1blk_x2_neon, pairwise_pack_left_1blk_x2_neon, pairwise_pack_right_1blk_x2_neon,
    },
    ntt120_mat_vec::{
        vec_mat1col_product_bbb_neon, vec_mat1col_product_bbc_neon, vec_mat1col_product_x2_bbc_neon,
        vec_mat2cols_product_x2_bbc_neon,
    },
    ntt120_ntt::{intt_neon, ntt_neon},
};
#[cfg(not(target_arch = "aarch64"))]
use poulpy_cpu_ref::reference::ntt120::{arithmetic::add_bbb_ref, types::Q_SHIFTED};

impl NttDFTExecute<NttTable<Primes30>> for NTT120Neon {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30>, data: &mut [u64]) {
        #[cfg(target_arch = "aarch64")]
        {
            ntt_neon::<Primes30>(table, data);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            ntt_ref::<Primes30>(table, data);
        }
    }
}

impl NttDFTExecute<NttTableInv<Primes30>> for NTT120Neon {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30>, data: &mut [u64]) {
        #[cfg(target_arch = "aarch64")]
        {
            intt_neon::<Primes30>(table, data);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            intt_ref::<Primes30>(table, data);
        }
    }
}

impl NttFromZnx64 for NTT120Neon {
    #[inline(always)]
    fn ntt_from_znx64(res: &mut [u64], a: &[i64]) {
        #[cfg(target_arch = "aarch64")]
        {
            b_from_znx64_neon(a.len(), res, a);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            b_from_znx64_ref::<Primes30>(a.len(), res, a);
        }
    }

    #[inline(always)]
    fn ntt_from_znx64_masked(res: &mut [u64], a: &[i64], mask: i64) {
        #[cfg(target_arch = "aarch64")]
        {
            b_from_znx64_masked_neon(a.len(), res, a, mask);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // Fallback: mask scalar, then reuse the ref kernel.
            let masked: Vec<i64> = a.iter().map(|&v| v & mask).collect();
            b_from_znx64_ref::<Primes30>(a.len(), res, &masked);
        }
    }
}

impl NttToZnx128 for NTT120Neon {
    #[inline(always)]
    fn ntt_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]) {
        #[cfg(target_arch = "aarch64")]
        {
            b_to_znx128_neon(divisor_is_n, res, a);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            b_to_znx128_ref::<Primes30>(divisor_is_n, res, a);
        }
    }
}

impl NttAdd for NTT120Neon {
    #[inline(always)]
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        #[cfg(target_arch = "aarch64")]
        {
            ntt_add_neon(res.len() / 4, res, a, b);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            add_bbb_ref::<Primes30>(res.len() / 4, res, a, b);
        }
    }
}

impl NttAddAssign for NTT120Neon {
    #[inline(always)]
    fn ntt_add_assign(res: &mut [u64], a: &[u64]) {
        #[cfg(target_arch = "aarch64")]
        {
            ntt_add_assign_neon(res.len() / 4, res, a);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let n = res.len() / 4;
            for j in 0..n {
                for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                    let idx = 4 * j + k;
                    res[idx] = res[idx] % q_s + a[idx] % q_s;
                }
            }
        }
    }
}

impl NttSub for NTT120Neon {
    #[inline(always)]
    fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        #[cfg(target_arch = "aarch64")]
        {
            ntt_sub_neon(res.len() / 4, res, a, b);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let n = res.len() / 4;
            for j in 0..n {
                for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                    let idx = 4 * j + k;
                    res[idx] = a[idx] % q_s + (q_s - b[idx] % q_s);
                }
            }
        }
    }
}

impl NttSubAssign for NTT120Neon {
    #[inline(always)]
    fn ntt_sub_assign(res: &mut [u64], a: &[u64]) {
        #[cfg(target_arch = "aarch64")]
        {
            ntt_sub_assign_neon(res.len() / 4, res, a);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let n = res.len() / 4;
            for j in 0..n {
                for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                    let idx = 4 * j + k;
                    res[idx] = res[idx] % q_s + (q_s - a[idx] % q_s);
                }
            }
        }
    }
}

impl NttSubNegateAssign for NTT120Neon {
    #[inline(always)]
    fn ntt_sub_negate_assign(res: &mut [u64], a: &[u64]) {
        #[cfg(target_arch = "aarch64")]
        {
            ntt_sub_negate_assign_neon(res.len() / 4, res, a);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let n = res.len() / 4;
            for j in 0..n {
                for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                    let idx = 4 * j + k;
                    res[idx] = a[idx] % q_s + (q_s - res[idx] % q_s);
                }
            }
        }
    }
}

impl NttNegate for NTT120Neon {
    #[inline(always)]
    fn ntt_negate(res: &mut [u64], a: &[u64]) {
        #[cfg(target_arch = "aarch64")]
        {
            ntt_negate_neon(res.len() / 4, res, a);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let n = res.len() / 4;
            for j in 0..n {
                for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                    let idx = 4 * j + k;
                    res[idx] = q_s - a[idx] % q_s;
                }
            }
        }
    }
}

impl NttNegateAssign for NTT120Neon {
    #[inline(always)]
    fn ntt_negate_assign(res: &mut [u64]) {
        #[cfg(target_arch = "aarch64")]
        {
            ntt_negate_assign_neon(res.len() / 4, res);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let n = res.len() / 4;
            for j in 0..n {
                for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                    let idx = 4 * j + k;
                    res[idx] = q_s - res[idx] % q_s;
                }
            }
        }
    }
}

impl NttZero for NTT120Neon {
    #[inline(always)]
    fn ntt_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl NttCopy for NTT120Neon {
    #[inline(always)]
    fn ntt_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}

impl NttMulBbb for NTT120Neon {
    #[inline(always)]
    fn ntt_mul_bbb(meta: &BbbMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
        #[cfg(target_arch = "aarch64")]
        {
            vec_mat1col_product_bbb_neon(meta, ell, res, a, b);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            vec_mat1col_product_bbb_ref::<Primes30>(meta, ell, res, a, b);
        }
    }
}

impl NttMulBbc for NTT120Neon {
    #[inline(always)]
    fn ntt_mul_bbc(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], ntt_coeff: &[u32], prepared: &[u32]) {
        #[cfg(target_arch = "aarch64")]
        {
            vec_mat1col_product_bbc_neon(meta, ell, res, ntt_coeff, prepared);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            vec_mat1col_product_bbc_ref::<Primes30>(meta, ell, res, ntt_coeff, prepared);
        }
    }
}

impl NttCFromB for NTT120Neon {
    #[inline(always)]
    fn ntt_c_from_b(n: usize, res: &mut [u32], a: &[u64]) {
        #[cfg(target_arch = "aarch64")]
        {
            c_from_b_neon(n, res, a);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            c_from_b_ref::<Primes30>(n, res, a);
        }
    }
}

impl NttMulBbc1ColX2 for NTT120Neon {
    #[inline(always)]
    fn ntt_mul_bbc_1col_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        #[cfg(target_arch = "aarch64")]
        {
            vec_mat1col_product_x2_bbc_neon(meta, ell, res, a, b);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            vec_mat1col_product_x2_bbc_ref::<Primes30>(meta, ell, res, a, b);
        }
    }
}

impl NttMulBbc2ColsX2 for NTT120Neon {
    #[inline(always)]
    fn ntt_mul_bbc_2cols_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        #[cfg(target_arch = "aarch64")]
        {
            vec_mat2cols_product_x2_bbc_neon(meta, ell, res, a, b);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            vec_mat2cols_product_x2_bbc_ref::<Primes30>(meta, ell, res, a, b);
        }
    }
}

impl NttExtract1BlkContiguous for NTT120Neon {
    #[inline(always)]
    fn ntt_extract_1blk_contiguous(n: usize, row_max: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
        extract_1blk_from_contiguous_q120b_ref(n, row_max, blk, dst, src);
    }
}

impl NttPackLeft1BlkX2 for NTT120Neon {
    #[inline(always)]
    fn ntt_pack_left_1blk_x2(dst: &mut [u32], a: &[u64], row_count: usize, row_stride: usize, blk: usize) {
        #[cfg(target_arch = "aarch64")]
        {
            pack_left_1blk_x2_neon(dst, a, row_count, row_stride, blk);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
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
}

impl NttPackRight1BlkX2 for NTT120Neon {
    #[inline(always)]
    fn ntt_pack_right_1blk_x2(dst: &mut [u32], a: &[u32], row_count: usize, row_stride: usize, blk: usize) {
        #[cfg(target_arch = "aarch64")]
        {
            pack_right_1blk_x2_neon(dst, a, row_count, row_stride, blk);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for row in 0..row_count {
                let row_base = (row_count - 1 - row) * row_stride + 16 * blk;
                let out_base = 16 * row;
                dst[out_base..out_base + 16].copy_from_slice(&a[row_base..row_base + 16]);
            }
        }
    }
}

impl NttPairwisePackLeft1BlkX2 for NTT120Neon {
    #[inline(always)]
    fn ntt_pairwise_pack_left_1blk_x2(dst: &mut [u32], a: &[u64], b: &[u64], row_count: usize, row_stride: usize, blk: usize) {
        #[cfg(target_arch = "aarch64")]
        {
            pairwise_pack_left_1blk_x2_neon(dst, a, b, row_count, row_stride, blk);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
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
}

impl NttPairwisePackRight1BlkX2 for NTT120Neon {
    #[inline(always)]
    fn ntt_pairwise_pack_right_1blk_x2(dst: &mut [u32], a: &[u32], b: &[u32], row_count: usize, row_stride: usize, blk: usize) {
        #[cfg(target_arch = "aarch64")]
        {
            pairwise_pack_right_1blk_x2_neon(dst, a, b, row_count, row_stride, blk);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for row in 0..row_count {
                let row_base = (row_count - 1 - row) * row_stride + 16 * blk;
                let out_base = 16 * row;
                for idx in 0..16 {
                    dst[out_base + idx] = a[row_base + idx] + b[row_base + idx];
                }
            }
        }
    }
}
