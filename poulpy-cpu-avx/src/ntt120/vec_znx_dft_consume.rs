//! AVX2 override for the NTT120 `vec_znx_idft_apply_consume` hot path.
//!
//! The shared HAL defaults already cover the whole `vec_znx_dft` theme. This
//! module restores only the AVX2-accelerated in-place q120b -> i128 compaction
//! used by `NTT120Avx`.

use bytemuck::cast_slice_mut;
use core::arch::x86_64::__m256i;
use poulpy_cpu_ref::reference::ntt120::{ntt::NttTableInv, primes::Primes30, vec_znx_dft::NttModuleHandle};
use poulpy_hal::layouts::{Data, Module, VecZnxBig, VecZnxDft, VecZnxDftToBackendMut, ZnxViewMut};

use super::{
    NTT120Avx,
    arithmetic_avx::{
        BARRETT_MU, CRT_VEC, POW16_CRT, POW32_CRT, Q_VEC, QM_HI, QM_LO, QM_MID, TOTAL_Q, TOTAL_Q_MULT, crt_accumulate_avx2,
        reduce_b_and_apply_crt,
    },
    ntt::intt_avx2,
};

/// AVX2-accelerated in-place CRT compaction: q120b (32 bytes/coeff) -> i128 (16 bytes/coeff).
///
/// For each DFT block:
/// 1. apply the inverse NTT in place,
/// 2. reduce each 4-residue q120b coefficient against the CRT basis,
/// 3. accumulate the CRT combination directly into an `i128`.
///
/// # Safety
///
/// - `u64_ptr` must cover `4 * n * n_blocks` `u64` values.
/// - AVX2 support must be available at runtime.
/// - no aliased references to the same buffer may be live during the call.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
unsafe fn compact_all_blocks_avx2(n: usize, n_blocks: usize, u64_ptr: *mut u64, table: &NttTableInv<Primes30>) {
    use core::arch::x86_64::_mm256_loadu_si256;

    let half_q: u128 = TOTAL_Q.div_ceil(2);

    let q_avx = unsafe { _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i) };
    let mu_avx = unsafe { _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i) };
    let pow32_crt_avx = unsafe { _mm256_loadu_si256(POW32_CRT.as_ptr() as *const __m256i) };
    let pow16_crt_avx = unsafe { _mm256_loadu_si256(POW16_CRT.as_ptr() as *const __m256i) };
    let crt_avx = unsafe { _mm256_loadu_si256(CRT_VEC.as_ptr() as *const __m256i) };
    let qm_hi_avx = unsafe { _mm256_loadu_si256(QM_HI.as_ptr() as *const __m256i) };
    let qm_mid_avx = unsafe { _mm256_loadu_si256(QM_MID.as_ptr() as *const __m256i) };
    let qm_lo_avx = unsafe { _mm256_loadu_si256(QM_LO.as_ptr() as *const __m256i) };

    for k in 0..n_blocks {
        let src_start = 4 * n * k;
        let dst_start = 2 * n * k;

        {
            let blk: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(u64_ptr.add(src_start), 4 * n) };
            unsafe { intt_avx2::<Primes30>(table, blk) };
        }

        for c in 0..n {
            let xv: __m256i = unsafe { _mm256_loadu_si256(u64_ptr.add(src_start + 4 * c) as *const __m256i) };
            let t = unsafe { reduce_b_and_apply_crt(xv, q_avx, mu_avx, pow32_crt_avx, pow16_crt_avx, crt_avx) };
            let mut v = unsafe { crt_accumulate_avx2(t, qm_hi_avx, qm_mid_avx, qm_lo_avx) };

            let q_approx = (v >> 120) as usize;
            v -= TOTAL_Q_MULT[q_approx];
            if v >= TOTAL_Q {
                v -= TOTAL_Q;
            }

            let val: i128 = if v >= half_q { v as i128 - TOTAL_Q as i128 } else { v as i128 };
            unsafe { (u64_ptr.add(dst_start + 2 * c) as *mut i128).write_unaligned(val) };
        }
    }
}

#[allow(dead_code)]
pub(crate) fn vec_znx_idft_apply_consume<D: Data>(
    module: &Module<NTT120Avx>,
    mut a: VecZnxDft<D, NTT120Avx>,
) -> VecZnxBig<D, NTT120Avx>
where
    VecZnxDft<D, NTT120Avx>: VecZnxDftToBackendMut<NTT120Avx>,
{
    let table = module.get_intt_table();

    let (n, n_blocks, u64_ptr) = {
        let mut a_mut: VecZnxDft<&mut [u8], NTT120Avx> = a.to_backend_mut();
        let n = a_mut.n();
        let n_blocks = a_mut.cols() * a_mut.size();
        let ptr: *mut u64 = {
            let s = a_mut.raw_mut();
            cast_slice_mut::<_, u64>(s).as_mut_ptr()
        };
        (n, n_blocks, ptr)
    };

    unsafe { compact_all_blocks_avx2(n, n_blocks, u64_ptr, table) };
    a.into_big()
}
