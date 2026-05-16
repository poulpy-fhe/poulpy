//! AVX-512F fused iNTT + q120b → i128 Garner CRT compaction for `NTT120Avx512`.

use bytemuck::cast_slice_mut;
use core::arch::x86_64::{__m256i, __m512i, _mm512_extracti64x4_epi64, _mm512_loadu_si512, _mm512_mul_epu32};
use poulpy_cpu_ref::reference::ntt120::{ntt::NttTableInv, primes::Primes30, vec_znx_dft::NttModuleHandle};
use poulpy_hal::layouts::{Module, VecZnxBigBackendMut, VecZnxDftBackendMut, ZnxViewMut};

use super::{
    NTT120Avx512,
    arithmetic_avx512::{
        BARRETT_MU, CRT_VEC, POW16_CRT, POW32_CRT, Q_VEC, QM_HI, QM_LO, QM_MID, TOTAL_Q, TOTAL_Q_MULT, bcast_quad,
        crt_accumulate_avx512, hadd64_pub, reduce_b_and_apply_crt, reduce_b_and_apply_crt_512,
    },
    ntt::intt_avx512,
};

/// iNTT (in place on `src`) + Garner CRT-compact (writing i128 to `dst`).
///
/// # Safety
/// - `src_ptr` covers `4 * n * n_blocks` u64; `dst_ptr` covers `n * n_blocks` i128.
/// - If aliased, the dst window must lie in the first half of the src window.
/// - AVX-512F required at runtime.
#[target_feature(enable = "avx512f")]
unsafe fn intt_then_compact_avx512(
    n: usize,
    n_blocks: usize,
    src_ptr: *mut u64,
    dst_ptr: *mut i128,
    table: &NttTableInv<Primes30>,
) {
    use core::arch::x86_64::_mm256_loadu_si256;

    let half_q: u128 = TOTAL_Q.div_ceil(2);

    // 256-bit constants for the odd-coefficient tail.
    let q_avx = unsafe { _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i) };
    let mu_avx = unsafe { _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i) };
    let pow32_crt_avx = unsafe { _mm256_loadu_si256(POW32_CRT.as_ptr() as *const __m256i) };
    let pow16_crt_avx = unsafe { _mm256_loadu_si256(POW16_CRT.as_ptr() as *const __m256i) };
    let crt_avx = unsafe { _mm256_loadu_si256(CRT_VEC.as_ptr() as *const __m256i) };
    let qm_hi_avx = unsafe { _mm256_loadu_si256(QM_HI.as_ptr() as *const __m256i) };
    let qm_mid_avx = unsafe { _mm256_loadu_si256(QM_MID.as_ptr() as *const __m256i) };
    let qm_lo_avx = unsafe { _mm256_loadu_si256(QM_LO.as_ptr() as *const __m256i) };

    // 512-bit broadcast constants for the pair-packed inner loop.
    let q_512 = unsafe { bcast_quad(Q_VEC.as_ptr()) };
    let mu_512 = unsafe { bcast_quad(BARRETT_MU.as_ptr()) };
    let pow32_crt_512 = unsafe { bcast_quad(POW32_CRT.as_ptr()) };
    let pow16_crt_512 = unsafe { bcast_quad(POW16_CRT.as_ptr()) };
    let crt_512 = unsafe { bcast_quad(CRT_VEC.as_ptr()) };
    let qm_hi_512 = unsafe { bcast_quad(QM_HI.as_ptr()) };
    let qm_mid_512 = unsafe { bcast_quad(QM_MID.as_ptr()) };
    let qm_lo_512 = unsafe { bcast_quad(QM_LO.as_ptr()) };

    for k in 0..n_blocks {
        let src_off_u64 = 4 * n * k;
        let dst_off_i128 = n * k;

        {
            let blk: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(src_ptr.add(src_off_u64), 4 * n) };
            unsafe { intt_avx512::<Primes30>(table, blk) };
        }

        // Pair-packed compaction: 2 q120b coefficients per __m512i load → 2 i128 stores.
        unsafe {
            let pairs = n / 2;
            let mut c = 0usize;
            for _ in 0..pairs {
                let xv: __m512i = _mm512_loadu_si512(src_ptr.add(src_off_u64 + 4 * c) as *const __m512i);
                let t = reduce_b_and_apply_crt_512(xv, q_512, mu_512, pow32_crt_512, pow16_crt_512, crt_512);
                let p_hi = _mm512_mul_epu32(t, qm_hi_512);
                let p_mid = _mm512_mul_epu32(t, qm_mid_512);
                let p_lo = _mm512_mul_epu32(t, qm_lo_512);

                let pairs_iter = [
                    (
                        _mm512_extracti64x4_epi64::<0>(p_hi),
                        _mm512_extracti64x4_epi64::<0>(p_mid),
                        _mm512_extracti64x4_epi64::<0>(p_lo),
                    ),
                    (
                        _mm512_extracti64x4_epi64::<1>(p_hi),
                        _mm512_extracti64x4_epi64::<1>(p_mid),
                        _mm512_extracti64x4_epi64::<1>(p_lo),
                    ),
                ];
                for (j, (ph, pm, pl)) in pairs_iter.into_iter().enumerate() {
                    let s_hi = hadd64_pub(ph);
                    let s_mid = hadd64_pub(pm);
                    let s_lo = hadd64_pub(pl);
                    let mut v: u128 = ((s_hi as u128) << 64) + ((s_mid as u128) << 32) + (s_lo as u128);
                    let q_approx = (v >> 120) as usize;
                    v -= TOTAL_Q_MULT[q_approx];
                    if v >= TOTAL_Q {
                        v -= TOTAL_Q;
                    }
                    let val: i128 = if v >= half_q { v as i128 - TOTAL_Q as i128 } else { v as i128 };
                    dst_ptr.add(dst_off_i128 + c + j).write_unaligned(val);
                }
                c += 2;
            }

            // Tail (single 256-bit coefficient when n is odd).
            if n & 1 != 0 {
                let xv: __m256i = _mm256_loadu_si256(src_ptr.add(src_off_u64 + 4 * c) as *const __m256i);
                let t = reduce_b_and_apply_crt(xv, q_avx, mu_avx, pow32_crt_avx, pow16_crt_avx, crt_avx);
                let mut v = crt_accumulate_avx512(t, qm_hi_avx, qm_mid_avx, qm_lo_avx);
                let q_approx = (v >> 120) as usize;
                v -= TOTAL_Q_MULT[q_approx];
                if v >= TOTAL_Q {
                    v -= TOTAL_Q;
                }
                let val: i128 = if v >= half_q { v as i128 - TOTAL_Q as i128 } else { v as i128 };
                dst_ptr.add(dst_off_i128 + c).write_unaligned(val);
            }
        }
    }
}

/// `VecZnxIdftApplyTmpA` fast path: iNTT consumes `a` in place, Garner streams
/// into `res`. Limbs past `min(res.size(), a.size())` are zero-padded.
pub(crate) fn vec_znx_idft_apply_tmpa_avx512(
    module: &Module<NTT120Avx512>,
    res: &mut VecZnxBigBackendMut<'_, NTT120Avx512>,
    res_col: usize,
    a: &mut VecZnxDftBackendMut<'_, NTT120Avx512>,
    a_col: usize,
) {
    let table = module.get_intt_table();
    let n = a.n();
    let min_size = res.size().min(a.size());
    let a_cols = a.cols();
    let res_cols = res.cols();
    let res_size = res.size();

    let src_base: *mut u64 = cast_slice_mut::<_, u64>(a.raw_mut()).as_mut_ptr();
    let dst_base: *mut i128 = res.raw_mut().as_mut_ptr();

    for j in 0..min_size {
        let src_off_u64 = 4 * n * (j * a_cols + a_col);
        let dst_off_i128 = n * (j * res_cols + res_col);
        unsafe {
            intt_then_compact_avx512(n, 1, src_base.add(src_off_u64), dst_base.add(dst_off_i128), table);
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}
