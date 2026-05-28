//! NEON kernels for the FFT64 `I64Ops` family (i64 block move + convolution by-constant).

use core::arch::aarch64::{
    int64x2_t, vaddq_s64, vdupq_n_s32, vdupq_n_s64, vget_low_s32, vld1q_s64, vmovn_s64, vmull_s32, vst1q_s64,
};

/// Helper: low 32 bits of each i64 lane → int32x2_t (signed narrow).
#[inline(always)]
unsafe fn low32_s(v: int64x2_t) -> core::arch::aarch64::int32x2_t {
    unsafe { vmovn_s64(v) }
}

/// `dst[k0] = Σ_j a[(k-j) * 8 .. + 8] * b[j]` for one output coefficient `k`.
/// Caller must guarantee `|a[i]|, |b[j]| < 2^31` (so i32×i32→i64 is exact).
pub(crate) fn i64_convolution_by_const_1coeff_neon(k: usize, dst: &mut [i64; 8], a: &[i64], a_size: usize, b: &[i64]) {
    dst.fill(0);
    let b_size = b.len();
    if k >= a_size + b_size {
        return;
    }
    let j_min = k.saturating_sub(a_size - 1);
    let j_max = (k + 1).min(b_size);
    unsafe {
        let mut acc0 = vdupq_n_s64(0);
        let mut acc1 = vdupq_n_s64(0);
        let mut acc2 = vdupq_n_s64(0);
        let mut acc3 = vdupq_n_s64(0);
        let mut a_ptr = a.as_ptr().add(8 * (k - j_min));
        let mut b_ptr = b.as_ptr().add(j_min);
        for _ in 0..(j_max - j_min) {
            let br = vdupq_n_s32(*b_ptr as i32);
            let br_low_pair = vget_low_s32(br);
            let a_lo = vld1q_s64(a_ptr);
            let a_mid = vld1q_s64(a_ptr.add(2));
            let a_hi = vld1q_s64(a_ptr.add(4));
            let a_top = vld1q_s64(a_ptr.add(6));
            acc0 = vaddq_s64(acc0, vmull_s32(low32_s(a_lo), br_low_pair));
            acc1 = vaddq_s64(acc1, vmull_s32(low32_s(a_mid), br_low_pair));
            acc2 = vaddq_s64(acc2, vmull_s32(low32_s(a_hi), br_low_pair));
            acc3 = vaddq_s64(acc3, vmull_s32(low32_s(a_top), br_low_pair));
            a_ptr = a_ptr.sub(8);
            b_ptr = b_ptr.add(1);
        }
        let d_ptr = dst.as_mut_ptr();
        vst1q_s64(d_ptr, acc0);
        vst1q_s64(d_ptr.add(2), acc1);
        vst1q_s64(d_ptr.add(4), acc2);
        vst1q_s64(d_ptr.add(6), acc3);
    }
}

/// `dst[k0..k0+2] = Σ_j a[…] * b[j]` for two consecutive output coefficients.
pub(crate) fn i64_convolution_by_const_2coeffs_neon(k: usize, dst: &mut [i64; 16], a: &[i64], a_size: usize, b: &[i64]) {
    let b_size = b.len();
    debug_assert!(a.len() >= 8 * a_size);
    let k0 = k;
    let k1 = k + 1;
    let bound = a_size + b_size;
    if k0 >= bound {
        for v in dst.iter_mut() {
            *v = 0;
        }
        return;
    }
    unsafe {
        let mut a0 = vdupq_n_s64(0);
        let mut a1 = vdupq_n_s64(0);
        let mut a2 = vdupq_n_s64(0);
        let mut a3 = vdupq_n_s64(0);
        let mut b0 = vdupq_n_s64(0);
        let mut b1 = vdupq_n_s64(0);
        let mut b2 = vdupq_n_s64(0);
        let mut b3 = vdupq_n_s64(0);

        let j0_min = (k0 + 1).saturating_sub(a_size);
        let j0_max = (k0 + 1).min(b_size);

        let load_pairs =
            |p: *const i64| -> [int64x2_t; 4] { [vld1q_s64(p), vld1q_s64(p.add(2)), vld1q_s64(p.add(4)), vld1q_s64(p.add(6))] };

        if k1 >= bound {
            let mut a_k0_ptr = a.as_ptr().add(8 * (k0 - j0_min));
            let mut b_ptr = b.as_ptr().add(j0_min);
            for _ in 0..(j0_max - j0_min) {
                let br = vdupq_n_s32(*b_ptr as i32);
                let br_pair = vget_low_s32(br);
                let v = load_pairs(a_k0_ptr);
                a0 = vaddq_s64(a0, vmull_s32(low32_s(v[0]), br_pair));
                a1 = vaddq_s64(a1, vmull_s32(low32_s(v[1]), br_pair));
                a2 = vaddq_s64(a2, vmull_s32(low32_s(v[2]), br_pair));
                a3 = vaddq_s64(a3, vmull_s32(low32_s(v[3]), br_pair));
                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
        } else {
            let j1_min = (k1 + 1).saturating_sub(a_size);
            let j1_max = (k1 + 1).min(b_size);
            let mut a_k0_ptr = a.as_ptr().add(8 * (k0 - j0_min));
            let mut a_k1_ptr = a.as_ptr().add(8 * (k1 - j1_min));
            let mut b_ptr = b.as_ptr().add(j0_min);

            // Region 1: k0 only
            for _ in 0..(j1_min - j0_min) {
                let br = vdupq_n_s32(*b_ptr as i32);
                let br_pair = vget_low_s32(br);
                let v = load_pairs(a_k0_ptr);
                a0 = vaddq_s64(a0, vmull_s32(low32_s(v[0]), br_pair));
                a1 = vaddq_s64(a1, vmull_s32(low32_s(v[1]), br_pair));
                a2 = vaddq_s64(a2, vmull_s32(low32_s(v[2]), br_pair));
                a3 = vaddq_s64(a3, vmull_s32(low32_s(v[3]), br_pair));
                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
            // Region 2: both k0 and k1
            for _ in 0..(j0_max - j1_min) {
                let br = vdupq_n_s32(*b_ptr as i32);
                let br_pair = vget_low_s32(br);
                let v0 = load_pairs(a_k0_ptr);
                let v1 = load_pairs(a_k1_ptr);
                a0 = vaddq_s64(a0, vmull_s32(low32_s(v0[0]), br_pair));
                a1 = vaddq_s64(a1, vmull_s32(low32_s(v0[1]), br_pair));
                a2 = vaddq_s64(a2, vmull_s32(low32_s(v0[2]), br_pair));
                a3 = vaddq_s64(a3, vmull_s32(low32_s(v0[3]), br_pair));
                b0 = vaddq_s64(b0, vmull_s32(low32_s(v1[0]), br_pair));
                b1 = vaddq_s64(b1, vmull_s32(low32_s(v1[1]), br_pair));
                b2 = vaddq_s64(b2, vmull_s32(low32_s(v1[2]), br_pair));
                b3 = vaddq_s64(b3, vmull_s32(low32_s(v1[3]), br_pair));
                a_k0_ptr = a_k0_ptr.sub(8);
                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
            // Region 3: k1 only
            for _ in 0..(j1_max - j0_max) {
                let br = vdupq_n_s32(*b_ptr as i32);
                let br_pair = vget_low_s32(br);
                let v1 = load_pairs(a_k1_ptr);
                b0 = vaddq_s64(b0, vmull_s32(low32_s(v1[0]), br_pair));
                b1 = vaddq_s64(b1, vmull_s32(low32_s(v1[1]), br_pair));
                b2 = vaddq_s64(b2, vmull_s32(low32_s(v1[2]), br_pair));
                b3 = vaddq_s64(b3, vmull_s32(low32_s(v1[3]), br_pair));
                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
        }

        let d_ptr = dst.as_mut_ptr();
        vst1q_s64(d_ptr, a0);
        vst1q_s64(d_ptr.add(2), a1);
        vst1q_s64(d_ptr.add(4), a2);
        vst1q_s64(d_ptr.add(6), a3);
        vst1q_s64(d_ptr.add(8), b0);
        vst1q_s64(d_ptr.add(10), b1);
        vst1q_s64(d_ptr.add(12), b2);
        vst1q_s64(d_ptr.add(14), b3);
    }
}

/// Pure 8-i64 block copy from a contiguous reim-style i64 layout.
pub(crate) fn i64_extract_1blk_contiguous_neon(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
    unsafe {
        let mut src_ptr = src.as_ptr().add(offset + (blk << 3));
        let mut dst_ptr = dst.as_mut_ptr();
        let step = n;
        for _ in 0..rows {
            let v0 = vld1q_s64(src_ptr);
            let v1 = vld1q_s64(src_ptr.add(2));
            let v2 = vld1q_s64(src_ptr.add(4));
            let v3 = vld1q_s64(src_ptr.add(6));
            vst1q_s64(dst_ptr, v0);
            vst1q_s64(dst_ptr.add(2), v1);
            vst1q_s64(dst_ptr.add(4), v2);
            vst1q_s64(dst_ptr.add(6), v3);
            dst_ptr = dst_ptr.add(8);
            src_ptr = src_ptr.add(step);
        }
    }
}

/// Pure 8-i64 block copy back into a contiguous reim-style i64 layout.
pub(crate) fn i64_save_1blk_contiguous_neon(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
    unsafe {
        let mut src_ptr = src.as_ptr();
        let mut dst_ptr = dst.as_mut_ptr().add(offset + (blk << 3));
        let step = n;
        for _ in 0..rows {
            let v0 = vld1q_s64(src_ptr);
            let v1 = vld1q_s64(src_ptr.add(2));
            let v2 = vld1q_s64(src_ptr.add(4));
            let v3 = vld1q_s64(src_ptr.add(6));
            vst1q_s64(dst_ptr, v0);
            vst1q_s64(dst_ptr.add(2), v1);
            vst1q_s64(dst_ptr.add(4), v2);
            vst1q_s64(dst_ptr.add(6), v3);
            dst_ptr = dst_ptr.add(step);
            src_ptr = src_ptr.add(8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_cpu_ref::reference::fft64::convolution::{
        i64_convolution_by_const_1coeff_ref, i64_convolution_by_const_2coeffs_ref, i64_extract_1blk_contiguous_ref,
        i64_save_1blk_contiguous_ref,
    };
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0xa1b2_c3d4_e5f6_0708)
    }

    fn random_small_i64(r: &mut ChaCha8Rng, n: usize) -> Vec<i64> {
        (0..n).map(|_| (r.random::<i32>() as i64).wrapping_rem(1_000_000)).collect()
    }

    #[test]
    fn i64_convolution_by_const_1coeff_neon_matches_ref() {
        let mut r = rng();
        let a_size = 4usize;
        let b_size = 5usize;
        let a = random_small_i64(&mut r, 8 * a_size);
        let b = random_small_i64(&mut r, b_size);
        for k in 0..(a_size + b_size + 1) {
            let mut got = [0i64; 8];
            let mut want = [0i64; 8];
            i64_convolution_by_const_1coeff_neon(k, &mut got, &a, a_size, &b);
            i64_convolution_by_const_1coeff_ref(k, &mut want, &a, a_size, &b);
            assert_eq!(got, want, "k={k}");
        }
    }

    #[test]
    fn i64_convolution_by_const_2coeffs_neon_matches_ref() {
        let mut r = rng();
        let a_size = 4usize;
        let b_size = 5usize;
        let a = random_small_i64(&mut r, 8 * a_size);
        let b = random_small_i64(&mut r, b_size);
        for k in (0..(a_size + b_size + 2)).step_by(2) {
            let mut got = [0i64; 16];
            let mut want = [0i64; 16];
            i64_convolution_by_const_2coeffs_neon(k, &mut got, &a, a_size, &b);
            i64_convolution_by_const_2coeffs_ref(k, &mut want, &a, a_size, &b);
            assert_eq!(got, want, "k={k}");
        }
    }

    #[test]
    fn i64_extract_1blk_contiguous_neon_matches_ref() {
        let mut r = rng();
        let n = 32usize;
        let offset = 0usize;
        let rows = 4usize;
        let blk = 2usize;
        let src = random_small_i64(&mut r, rows * n);
        let mut got = vec![0i64; 8 * rows];
        let mut want = vec![0i64; 8 * rows];
        i64_extract_1blk_contiguous_neon(n, offset, rows, blk, &mut got, &src);
        i64_extract_1blk_contiguous_ref(n, offset, rows, blk, &mut want, &src);
        assert_eq!(got, want);
    }

    #[test]
    fn i64_save_1blk_contiguous_neon_matches_ref() {
        let mut r = rng();
        let n = 32usize;
        let offset = 0usize;
        let rows = 4usize;
        let blk = 1usize;
        let src = random_small_i64(&mut r, 8 * rows);
        let mut got = vec![0i64; rows * n];
        let mut want = vec![0i64; rows * n];
        i64_save_1blk_contiguous_neon(n, offset, rows, blk, &mut got, &src);
        i64_save_1blk_contiguous_ref(n, offset, rows, blk, &mut want, &src);
        assert_eq!(got, want);
    }
}
