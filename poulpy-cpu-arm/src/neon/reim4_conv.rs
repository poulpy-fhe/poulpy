//! NEON kernels for the FFT64 `Reim4Convolution` family.
//!
//! Mirrors the convolution kernels in
//! `poulpy-cpu-avx/src/fft64/reim4/arithmetic_avx.rs:295-647`. Each AVX
//! `__m256d` (4 × f64) is two NEON `float64x2_t` registers; one block iter
//! processes 4 f64 (8 doubles per complex coeff = 2 NEON pairs).
//!
//! Sign convention: AVX uses `_mm256_fnmadd_pd(a, b, c) = c - a*b` for the
//! `-ai*bi` term in complex multiplication. NEON's matching intrinsic is
//! `vfmsq_f64(c, a, b) = c - a*b` — same sign as `fnmadd`.

use core::arch::aarch64::{vaddq_f64, vdupq_n_f64, vfmaq_f64, vfmsq_f64, vld1q_f64, vst1q_f64};

/// `dst = Σ_j a[k-j] * b[j]` (complex × complex), one output coefficient.
/// Mirrors `reim4_convolution_1coeff_avx` at `arithmetic_avx.rs:295`.
pub(crate) fn reim4_convolution_1coeff_neon(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
    if k >= a_size + b_size {
        for v in dst.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let j_min = k.saturating_sub(a_size - 1);
    let j_max = (k + 1).min(b_size);
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let (mut acc_re_lo, mut acc_re_hi) = (zero, zero);
        let (mut acc_im_lo, mut acc_im_hi) = (zero, zero);
        let mut a_ptr = a.as_ptr().add(8 * (k - j_min));
        let mut b_ptr = b.as_ptr().add(8 * j_min);
        for _ in 0..(j_max - j_min) {
            let ar_lo = vld1q_f64(a_ptr);
            let ar_hi = vld1q_f64(a_ptr.add(2));
            let ai_lo = vld1q_f64(a_ptr.add(4));
            let ai_hi = vld1q_f64(a_ptr.add(6));
            let br_lo = vld1q_f64(b_ptr);
            let br_hi = vld1q_f64(b_ptr.add(2));
            let bi_lo = vld1q_f64(b_ptr.add(4));
            let bi_hi = vld1q_f64(b_ptr.add(6));
            // re += ar*br − ai*bi
            acc_re_lo = vfmaq_f64(acc_re_lo, ar_lo, br_lo);
            acc_re_hi = vfmaq_f64(acc_re_hi, ar_hi, br_hi);
            acc_re_lo = vfmsq_f64(acc_re_lo, ai_lo, bi_lo);
            acc_re_hi = vfmsq_f64(acc_re_hi, ai_hi, bi_hi);
            // im += ar*bi + ai*br
            acc_im_lo = vfmaq_f64(acc_im_lo, ar_lo, bi_lo);
            acc_im_hi = vfmaq_f64(acc_im_hi, ar_hi, bi_hi);
            acc_im_lo = vfmaq_f64(acc_im_lo, ai_lo, br_lo);
            acc_im_hi = vfmaq_f64(acc_im_hi, ai_hi, br_hi);
            a_ptr = a_ptr.sub(8);
            b_ptr = b_ptr.add(8);
        }
        let d_ptr = dst.as_mut_ptr();
        vst1q_f64(d_ptr, acc_re_lo);
        vst1q_f64(d_ptr.add(2), acc_re_hi);
        vst1q_f64(d_ptr.add(4), acc_im_lo);
        vst1q_f64(d_ptr.add(6), acc_im_hi);
    }
}

/// Two-coefficient complex × complex convolution.
/// Mirrors `reim4_convolution_2coeffs_avx` at `arithmetic_avx.rs:349`.
pub(crate) fn reim4_convolution_2coeffs_neon(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
    debug_assert!(a.len() >= 8 * a_size);
    debug_assert!(b.len() >= 8 * b_size);
    let k0 = k;
    let k1 = k + 1;
    let bound = a_size + b_size;
    if k0 >= bound {
        for v in dst.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let (mut re_k0_lo, mut re_k0_hi) = (zero, zero);
        let (mut im_k0_lo, mut im_k0_hi) = (zero, zero);
        let (mut re_k1_lo, mut re_k1_hi) = (zero, zero);
        let (mut im_k1_lo, mut im_k1_hi) = (zero, zero);

        let j0_min = (k0 + 1).saturating_sub(a_size);
        let j0_max = (k0 + 1).min(b_size);

        let mac_one =
            |re_lo: &mut _, re_hi: &mut _, im_lo: &mut _, im_hi: &mut _, a_ptr: *const f64, br_lo, br_hi, bi_lo, bi_hi| {
                let ar_lo = vld1q_f64(a_ptr);
                let ar_hi = vld1q_f64(a_ptr.add(2));
                let ai_lo = vld1q_f64(a_ptr.add(4));
                let ai_hi = vld1q_f64(a_ptr.add(6));
                *re_lo = vfmaq_f64(*re_lo, ar_lo, br_lo);
                *re_hi = vfmaq_f64(*re_hi, ar_hi, br_hi);
                *re_lo = vfmsq_f64(*re_lo, ai_lo, bi_lo);
                *re_hi = vfmsq_f64(*re_hi, ai_hi, bi_hi);
                *im_lo = vfmaq_f64(*im_lo, ar_lo, bi_lo);
                *im_hi = vfmaq_f64(*im_hi, ar_hi, bi_hi);
                *im_lo = vfmaq_f64(*im_lo, ai_lo, br_lo);
                *im_hi = vfmaq_f64(*im_hi, ai_hi, br_hi);
            };

        if k1 >= bound {
            let mut a_ptr = a.as_ptr().add(8 * (k0 - j0_min));
            let mut b_ptr = b.as_ptr().add(8 * j0_min);
            for _ in 0..(j0_max - j0_min) {
                let br_lo = vld1q_f64(b_ptr);
                let br_hi = vld1q_f64(b_ptr.add(2));
                let bi_lo = vld1q_f64(b_ptr.add(4));
                let bi_hi = vld1q_f64(b_ptr.add(6));
                mac_one(
                    &mut re_k0_lo,
                    &mut re_k0_hi,
                    &mut im_k0_lo,
                    &mut im_k0_hi,
                    a_ptr,
                    br_lo,
                    br_hi,
                    bi_lo,
                    bi_hi,
                );
                a_ptr = a_ptr.sub(8);
                b_ptr = b_ptr.add(8);
            }
        } else {
            let j1_min = (k1 + 1).saturating_sub(a_size);
            let j1_max = (k1 + 1).min(b_size);
            let mut a_k0_ptr = a.as_ptr().add(8 * (k0 - j0_min));
            let mut a_k1_ptr = a.as_ptr().add(8 * (k1 - j1_min));
            let mut b_ptr = b.as_ptr().add(8 * j0_min);

            // Region 1: k0 only
            for _ in 0..(j1_min - j0_min) {
                let br_lo = vld1q_f64(b_ptr);
                let br_hi = vld1q_f64(b_ptr.add(2));
                let bi_lo = vld1q_f64(b_ptr.add(4));
                let bi_hi = vld1q_f64(b_ptr.add(6));
                mac_one(
                    &mut re_k0_lo,
                    &mut re_k0_hi,
                    &mut im_k0_lo,
                    &mut im_k0_hi,
                    a_k0_ptr,
                    br_lo,
                    br_hi,
                    bi_lo,
                    bi_hi,
                );
                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(8);
            }
            // Region 2: both
            for _ in 0..(j0_max - j1_min) {
                let br_lo = vld1q_f64(b_ptr);
                let br_hi = vld1q_f64(b_ptr.add(2));
                let bi_lo = vld1q_f64(b_ptr.add(4));
                let bi_hi = vld1q_f64(b_ptr.add(6));
                mac_one(
                    &mut re_k0_lo,
                    &mut re_k0_hi,
                    &mut im_k0_lo,
                    &mut im_k0_hi,
                    a_k0_ptr,
                    br_lo,
                    br_hi,
                    bi_lo,
                    bi_hi,
                );
                mac_one(
                    &mut re_k1_lo,
                    &mut re_k1_hi,
                    &mut im_k1_lo,
                    &mut im_k1_hi,
                    a_k1_ptr,
                    br_lo,
                    br_hi,
                    bi_lo,
                    bi_hi,
                );
                a_k0_ptr = a_k0_ptr.sub(8);
                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(8);
            }
            // Region 3: k1 only
            for _ in 0..(j1_max - j0_max) {
                let br_lo = vld1q_f64(b_ptr);
                let br_hi = vld1q_f64(b_ptr.add(2));
                let bi_lo = vld1q_f64(b_ptr.add(4));
                let bi_hi = vld1q_f64(b_ptr.add(6));
                mac_one(
                    &mut re_k1_lo,
                    &mut re_k1_hi,
                    &mut im_k1_lo,
                    &mut im_k1_hi,
                    a_k1_ptr,
                    br_lo,
                    br_hi,
                    bi_lo,
                    bi_hi,
                );
                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(8);
            }
        }

        let d_ptr = dst.as_mut_ptr();
        vst1q_f64(d_ptr, re_k0_lo);
        vst1q_f64(d_ptr.add(2), re_k0_hi);
        vst1q_f64(d_ptr.add(4), im_k0_lo);
        vst1q_f64(d_ptr.add(6), im_k0_hi);
        vst1q_f64(d_ptr.add(8), re_k1_lo);
        vst1q_f64(d_ptr.add(10), re_k1_hi);
        vst1q_f64(d_ptr.add(12), im_k1_lo);
        vst1q_f64(d_ptr.add(14), im_k1_hi);
    }
}

/// Single-coefficient complex × real-scalar convolution.
/// Mirrors `reim4_convolution_by_real_const_1coeff_avx` at `arithmetic_avx.rs:481`.
pub(crate) fn reim4_convolution_by_real_const_1coeff_neon(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64]) {
    let b_size = b.len();
    if k >= a_size + b_size {
        for v in dst.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let j_min = k.saturating_sub(a_size - 1);
    let j_max = (k + 1).min(b_size);
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let (mut acc_re_lo, mut acc_re_hi) = (zero, zero);
        let (mut acc_im_lo, mut acc_im_hi) = (zero, zero);
        let mut a_ptr = a.as_ptr().add(8 * (k - j_min));
        let mut b_ptr = b.as_ptr().add(j_min);
        for _ in 0..(j_max - j_min) {
            let ar_lo = vld1q_f64(a_ptr);
            let ar_hi = vld1q_f64(a_ptr.add(2));
            let ai_lo = vld1q_f64(a_ptr.add(4));
            let ai_hi = vld1q_f64(a_ptr.add(6));
            let br = vdupq_n_f64(*b_ptr);
            acc_re_lo = vfmaq_f64(acc_re_lo, ar_lo, br);
            acc_re_hi = vfmaq_f64(acc_re_hi, ar_hi, br);
            acc_im_lo = vfmaq_f64(acc_im_lo, ai_lo, br);
            acc_im_hi = vfmaq_f64(acc_im_hi, ai_hi, br);
            a_ptr = a_ptr.sub(8);
            b_ptr = b_ptr.add(1);
        }
        let d_ptr = dst.as_mut_ptr();
        vst1q_f64(d_ptr, acc_re_lo);
        vst1q_f64(d_ptr.add(2), acc_re_hi);
        vst1q_f64(d_ptr.add(4), acc_im_lo);
        vst1q_f64(d_ptr.add(6), acc_im_hi);
    }
}

/// Two-coefficient complex × real-scalar convolution.
/// Mirrors `reim4_convolution_by_real_const_2coeffs_avx` at `arithmetic_avx.rs:532`.
pub(crate) fn reim4_convolution_by_real_const_2coeffs_neon(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64]) {
    let b_size = b.len();
    debug_assert!(a.len() >= 8 * a_size);
    let k0 = k;
    let k1 = k + 1;
    let bound = a_size + b_size;
    if k0 >= bound {
        for v in dst.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let (mut re_k0_lo, mut re_k0_hi) = (zero, zero);
        let (mut im_k0_lo, mut im_k0_hi) = (zero, zero);
        let (mut re_k1_lo, mut re_k1_hi) = (zero, zero);
        let (mut im_k1_lo, mut im_k1_hi) = (zero, zero);

        let j0_min = (k0 + 1).saturating_sub(a_size);
        let j0_max = (k0 + 1).min(b_size);

        let mac = |re_lo: &mut _, re_hi: &mut _, im_lo: &mut _, im_hi: &mut _, a_ptr: *const f64, br| {
            let ar_lo = vld1q_f64(a_ptr);
            let ar_hi = vld1q_f64(a_ptr.add(2));
            let ai_lo = vld1q_f64(a_ptr.add(4));
            let ai_hi = vld1q_f64(a_ptr.add(6));
            *re_lo = vfmaq_f64(*re_lo, ar_lo, br);
            *re_hi = vfmaq_f64(*re_hi, ar_hi, br);
            *im_lo = vfmaq_f64(*im_lo, ai_lo, br);
            *im_hi = vfmaq_f64(*im_hi, ai_hi, br);
        };

        if k1 >= bound {
            let mut a_ptr = a.as_ptr().add(8 * (k0 - j0_min));
            let mut b_ptr = b.as_ptr().add(j0_min);
            for _ in 0..(j0_max - j0_min) {
                let br = vdupq_n_f64(*b_ptr);
                mac(&mut re_k0_lo, &mut re_k0_hi, &mut im_k0_lo, &mut im_k0_hi, a_ptr, br);
                a_ptr = a_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
        } else {
            let j1_min = (k1 + 1).saturating_sub(a_size);
            let j1_max = (k1 + 1).min(b_size);
            let mut a_k0_ptr = a.as_ptr().add(8 * (k0 - j0_min));
            let mut a_k1_ptr = a.as_ptr().add(8 * (k1 - j1_min));
            let mut b_ptr = b.as_ptr().add(j0_min);

            for _ in 0..(j1_min - j0_min) {
                let br = vdupq_n_f64(*b_ptr);
                mac(&mut re_k0_lo, &mut re_k0_hi, &mut im_k0_lo, &mut im_k0_hi, a_k0_ptr, br);
                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
            for _ in 0..(j0_max - j1_min) {
                let br = vdupq_n_f64(*b_ptr);
                mac(&mut re_k0_lo, &mut re_k0_hi, &mut im_k0_lo, &mut im_k0_hi, a_k0_ptr, br);
                mac(&mut re_k1_lo, &mut re_k1_hi, &mut im_k1_lo, &mut im_k1_hi, a_k1_ptr, br);
                a_k0_ptr = a_k0_ptr.sub(8);
                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
            for _ in 0..(j1_max - j0_max) {
                let br = vdupq_n_f64(*b_ptr);
                mac(&mut re_k1_lo, &mut re_k1_hi, &mut im_k1_lo, &mut im_k1_hi, a_k1_ptr, br);
                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
        }

        let d_ptr = dst.as_mut_ptr();
        vst1q_f64(d_ptr, re_k0_lo);
        vst1q_f64(d_ptr.add(2), re_k0_hi);
        vst1q_f64(d_ptr.add(4), im_k0_lo);
        vst1q_f64(d_ptr.add(6), im_k0_hi);
        vst1q_f64(d_ptr.add(8), re_k1_lo);
        vst1q_f64(d_ptr.add(10), re_k1_hi);
        vst1q_f64(d_ptr.add(12), im_k1_lo);
        vst1q_f64(d_ptr.add(14), im_k1_hi);
    }
}

#[allow(dead_code)]
fn _keep() {
    let _ = vaddq_f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_cpu_ref::reference::fft64::reim4::{
        reim4_convolution_1coeff_ref, reim4_convolution_2coeffs_ref, reim4_convolution_by_real_const_1coeff_ref,
        reim4_convolution_by_real_const_2coeffs_ref,
    };
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0xdead_beef_dead_beef)
    }

    fn random(r: &mut ChaCha8Rng, n: usize) -> Vec<f64> {
        (0..n).map(|_| r.random::<f64>() * 100.0 - 50.0).collect()
    }

    fn close(got: &[f64], want: &[f64], tag: &str) {
        const REL: f64 = 1e-12;
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() / w.abs().max(1.0) < REL, "{tag}: idx={i} got={g} want={w}");
        }
    }

    #[test]
    fn reim4_convolution_1coeff_neon_close_to_ref() {
        let mut r = rng();
        let a_size = 4usize;
        let b_size = 4usize;
        let a = random(&mut r, 8 * a_size);
        let b = random(&mut r, 8 * b_size);
        for k in 0..(a_size + b_size + 1) {
            let mut got = [0f64; 8];
            let mut want = [0f64; 8];
            reim4_convolution_1coeff_neon(k, &mut got, &a, a_size, &b, b_size);
            reim4_convolution_1coeff_ref(k, &mut want, &a, a_size, &b, b_size);
            close(&got, &want, &format!("conv1coeff k={k}"));
        }
    }

    #[test]
    fn reim4_convolution_2coeffs_neon_close_to_ref() {
        let mut r = rng();
        let a_size = 4usize;
        let b_size = 4usize;
        let a = random(&mut r, 8 * a_size);
        let b = random(&mut r, 8 * b_size);
        for k in (0..(a_size + b_size + 1)).step_by(2) {
            let mut got = [0f64; 16];
            let mut want = [0f64; 16];
            reim4_convolution_2coeffs_neon(k, &mut got, &a, a_size, &b, b_size);
            reim4_convolution_2coeffs_ref(k, &mut want, &a, a_size, &b, b_size);
            close(&got, &want, &format!("conv2coeffs k={k}"));
        }
    }

    #[test]
    fn reim4_convolution_by_real_const_1coeff_neon_close_to_ref() {
        let mut r = rng();
        let a_size = 4usize;
        let b_size = 4usize;
        let a = random(&mut r, 8 * a_size);
        let b = random(&mut r, b_size);
        for k in 0..(a_size + b_size + 1) {
            let mut got = [0f64; 8];
            let mut want = [0f64; 8];
            reim4_convolution_by_real_const_1coeff_neon(k, &mut got, &a, a_size, &b);
            reim4_convolution_by_real_const_1coeff_ref(k, &mut want, &a, a_size, &b);
            close(&got, &want, &format!("conv_real_1coeff k={k}"));
        }
    }

    #[test]
    fn reim4_convolution_by_real_const_2coeffs_neon_close_to_ref() {
        let mut r = rng();
        let a_size = 4usize;
        let b_size = 4usize;
        let a = random(&mut r, 8 * a_size);
        let b = random(&mut r, b_size);
        for k in (0..(a_size + b_size + 1)).step_by(2) {
            let mut got = [0f64; 16];
            let mut want = [0f64; 16];
            reim4_convolution_by_real_const_2coeffs_neon(k, &mut got, &a, a_size, &b);
            reim4_convolution_by_real_const_2coeffs_ref(k, &mut want, &a, a_size, &b);
            close(&got, &want, &format!("conv_real_2coeffs k={k}"));
        }
    }
}
