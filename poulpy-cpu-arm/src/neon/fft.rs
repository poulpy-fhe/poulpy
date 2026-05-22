//! NEON `f64` FFT/IFFT butterfly kernels for [`FFT64Neon`].
//!
//! Direct port of `poulpy-cpu-avx/src/fft64/reim/{fft,ifft}_avx2_fma.rs`.
//! Each AVX `__m256d` (4×f64) becomes two `float64x2_t` registers (`_lo`/`_hi`),
//! and each AVX iteration (which advances the `f64` pointer by 4) becomes one
//! NEON iteration that processes both halves explicitly and advances the
//! pointer by 4 in `f64` units.
//!
//! NEON FMA semantics differ from AVX:
//! - `vfmaq_f64(c, a, b) = c + a*b`  (matches `_mm256_fmadd_pd(a, b, c)`)
//! - `vfmsq_f64(c, a, b) = c − a*b`  (matches `_mm256_fnmadd_pd(a, b, c)`)
//! - To compute `a*b − c` (AVX `_mm256_fmsub_pd`) we restructure as
//!   `vfmsq_f64(vmulq_f64(a, b), x, y)` to materialise `a*b − x*y` directly,
//!   avoiding the extra `vnegq_f64`.
//!
//! Sizes `m < 16` delegate to the scalar reference [`fft_ref`] /
//! [`ifft_ref`]. `m == 16` and the BFS leaves use the NEON-intrinsic
//! [`fft16_neon`] / [`ifft16_neon`], a 4-stage radix-2 port of
//! `fft16_ref` / `ifft16_ref`. An optional hand-written assembly
//! follow-up is gated on a real-AArch64 bench delta.

use core::arch::aarch64::{
    float64x2_t, vaddq_f64, vdupq_n_f64, vfmaq_f64, vfmsq_f64, vld1q_f64, vmulq_f64, vst1q_f64, vsubq_f64, vzip1q_f64, vzip2q_f64,
};

use poulpy_cpu_ref::reference::fft64::reim::{fft_ref, ifft_ref};

// ─── public dispatchers ────────────────────────────────────────────────────

/// Forward FFT in REIM split layout. Mirrors `fft_avx2_fma`.
pub(crate) fn fft_neon(m: usize, omg: &[f64], data: &mut [f64]) {
    if m < 16 {
        // m ∈ {1, 2, 4, 8} — scalar reference handles the small leaves.
        fft_ref(m, omg, data);
        return;
    }
    assert!(data.len() == 2 * m);
    let (re, im) = data.split_at_mut(m);
    if m == 16 {
        unsafe { fft16_neon(re, im, omg) };
    } else if m <= 2048 {
        unsafe { fft_bfs_16_neon(m, re, im, omg, 0) };
    } else {
        unsafe { fft_rec_16_neon(m, re, im, omg, 0) };
    }
}

/// Inverse FFT in REIM split layout. Mirrors `ifft_avx2_fma`.
pub(crate) fn ifft_neon(m: usize, omg: &[f64], data: &mut [f64]) {
    if m < 16 {
        ifft_ref(m, omg, data);
        return;
    }
    assert!(data.len() == 2 * m);
    let (re, im) = data.split_at_mut(m);
    if m == 16 {
        unsafe { ifft16_neon(re, im, omg) };
    } else if m <= 2048 {
        unsafe { ifft_bfs_16_neon(m, re, im, omg, 0) };
    } else {
        unsafe { ifft_rec_16_neon(m, re, im, omg, 0) };
    }
}

// ─── recursive layer ───────────────────────────────────────────────────────

unsafe fn fft_rec_16_neon(m: usize, re: &mut [f64], im: &mut [f64], omg: &[f64], mut pos: usize) -> usize {
    if m <= 2048 {
        return unsafe { fft_bfs_16_neon(m, re, im, omg, pos) };
    }
    let h = m >> 1;
    unsafe { twiddle_fft_neon(h, re, im, &omg[pos..pos + 2]) };
    pos += 2;
    pos = unsafe { fft_rec_16_neon(h, re, im, omg, pos) };
    pos = unsafe { fft_rec_16_neon(h, &mut re[h..], &mut im[h..], omg, pos) };
    pos
}

unsafe fn ifft_rec_16_neon(m: usize, re: &mut [f64], im: &mut [f64], omg: &[f64], mut pos: usize) -> usize {
    if m <= 2048 {
        return unsafe { ifft_bfs_16_neon(m, re, im, omg, pos) };
    }
    let h = m >> 1;
    pos = unsafe { ifft_rec_16_neon(h, re, im, omg, pos) };
    pos = unsafe { ifft_rec_16_neon(h, &mut re[h..], &mut im[h..], omg, pos) };
    unsafe { inv_twiddle_ifft_neon(h, re, im, &omg[pos..pos + 2]) };
    pos += 2;
    pos
}

// ─── BFS layer ─────────────────────────────────────────────────────────────

unsafe fn fft_bfs_16_neon(m: usize, re: &mut [f64], im: &mut [f64], omg: &[f64], mut pos: usize) -> usize {
    let log_m = (usize::BITS - (m - 1).leading_zeros()) as usize;
    let mut mm = m;

    if !log_m.is_multiple_of(2) {
        let h = mm >> 1;
        unsafe { twiddle_fft_neon(h, re, im, &omg[pos..pos + 2]) };
        pos += 2;
        mm = h;
    }

    while mm > 16 {
        let h = mm >> 2;
        for off in (0..m).step_by(mm) {
            unsafe { bitwiddle_fft_neon(h, &mut re[off..], &mut im[off..], &omg[pos..pos + 4]) };
            pos += 4;
        }
        mm = h;
    }

    for off in (0..m).step_by(16) {
        unsafe { fft16_neon(&mut re[off..off + 16], &mut im[off..off + 16], &omg[pos..pos + 16]) };
        pos += 16;
    }

    pos
}

unsafe fn ifft_bfs_16_neon(m: usize, re: &mut [f64], im: &mut [f64], omg: &[f64], mut pos: usize) -> usize {
    let log_m = (usize::BITS - (m - 1).leading_zeros()) as usize;

    for off in (0..m).step_by(16) {
        unsafe { ifft16_neon(&mut re[off..off + 16], &mut im[off..off + 16], &omg[pos..pos + 16]) };
        pos += 16;
    }

    let mut h = 16;
    let m_half = m >> 1;

    while h < m_half {
        let mm = h << 2;
        for off in (0..m).step_by(mm) {
            unsafe { inv_bitwiddle_ifft_neon(h, &mut re[off..], &mut im[off..], &omg[pos..pos + 4]) };
            pos += 4;
        }
        h = mm;
    }

    if !log_m.is_multiple_of(2) {
        unsafe { inv_twiddle_ifft_neon(h, re, im, &omg[pos..pos + 2]) };
        pos += 2;
    }

    pos
}

// ─── twiddle butterflies ───────────────────────────────────────────────────

/// Forward 2-way (Cooley–Tukey) butterfly. Mirrors `twiddle_fft_avx2_fma`.
#[inline]
unsafe fn twiddle_fft_neon(h: usize, re: &mut [f64], im: &mut [f64], omg: &[f64]) {
    debug_assert!(h.is_multiple_of(4));
    debug_assert!(omg.len() >= 2);
    unsafe {
        let omr: float64x2_t = vdupq_n_f64(omg[0]);
        let omi: float64x2_t = vdupq_n_f64(omg[1]);

        let mut r0 = re.as_mut_ptr();
        let mut r1 = re.as_mut_ptr().add(h);
        let mut i0 = im.as_mut_ptr();
        let mut i1 = im.as_mut_ptr().add(h);

        for _ in (0..h).step_by(4) {
            let mut ur0_lo = vld1q_f64(r0);
            let mut ur0_hi = vld1q_f64(r0.add(2));
            let mut ur1_lo = vld1q_f64(r1);
            let mut ur1_hi = vld1q_f64(r1.add(2));
            let mut ui0_lo = vld1q_f64(i0);
            let mut ui0_hi = vld1q_f64(i0.add(2));
            let mut ui1_lo = vld1q_f64(i1);
            let mut ui1_hi = vld1q_f64(i1.add(2));

            // tra = omr*ur1 - omi*ui1  (vfmsq(c, a, b) = c - a*b → seed c with omr*ur1)
            let mut tra_lo = vmulq_f64(omr, ur1_lo);
            let mut tra_hi = vmulq_f64(omr, ur1_hi);
            tra_lo = vfmsq_f64(tra_lo, omi, ui1_lo);
            tra_hi = vfmsq_f64(tra_hi, omi, ui1_hi);

            // tia = omr*ui1 + omi*ur1
            let mut tia_lo = vmulq_f64(omi, ur1_lo);
            let mut tia_hi = vmulq_f64(omi, ur1_hi);
            tia_lo = vfmaq_f64(tia_lo, omr, ui1_lo);
            tia_hi = vfmaq_f64(tia_hi, omr, ui1_hi);

            ur1_lo = vsubq_f64(ur0_lo, tra_lo);
            ur1_hi = vsubq_f64(ur0_hi, tra_hi);
            ui1_lo = vsubq_f64(ui0_lo, tia_lo);
            ui1_hi = vsubq_f64(ui0_hi, tia_hi);
            ur0_lo = vaddq_f64(ur0_lo, tra_lo);
            ur0_hi = vaddq_f64(ur0_hi, tra_hi);
            ui0_lo = vaddq_f64(ui0_lo, tia_lo);
            ui0_hi = vaddq_f64(ui0_hi, tia_hi);

            vst1q_f64(r0, ur0_lo);
            vst1q_f64(r0.add(2), ur0_hi);
            vst1q_f64(r1, ur1_lo);
            vst1q_f64(r1.add(2), ur1_hi);
            vst1q_f64(i0, ui0_lo);
            vst1q_f64(i0.add(2), ui0_hi);
            vst1q_f64(i1, ui1_lo);
            vst1q_f64(i1.add(2), ui1_hi);

            r0 = r0.add(4);
            r1 = r1.add(4);
            i0 = i0.add(4);
            i1 = i1.add(4);
        }
    }
}

/// Forward 4-way bitwiddle butterfly. Mirrors `bitwiddle_fft_avx2_fma`.
#[inline]
unsafe fn bitwiddle_fft_neon(h: usize, re: &mut [f64], im: &mut [f64], omg: &[f64]) {
    debug_assert!(h.is_multiple_of(4));
    debug_assert!(omg.len() >= 4);
    unsafe {
        let mut r0 = re.as_mut_ptr();
        let mut r1 = re.as_mut_ptr().add(h);
        let mut r2 = re.as_mut_ptr().add(2 * h);
        let mut r3 = re.as_mut_ptr().add(3 * h);
        let mut i0 = im.as_mut_ptr();
        let mut i1 = im.as_mut_ptr().add(h);
        let mut i2 = im.as_mut_ptr().add(2 * h);
        let mut i3 = im.as_mut_ptr().add(3 * h);

        let omar: float64x2_t = vdupq_n_f64(omg[0]);
        let omai: float64x2_t = vdupq_n_f64(omg[1]);
        let ombr: float64x2_t = vdupq_n_f64(omg[2]);
        let ombi: float64x2_t = vdupq_n_f64(omg[3]);

        for _ in (0..h).step_by(4) {
            let mut ur0_lo = vld1q_f64(r0);
            let mut ur0_hi = vld1q_f64(r0.add(2));
            let mut ur1_lo = vld1q_f64(r1);
            let mut ur1_hi = vld1q_f64(r1.add(2));
            let mut ur2_lo = vld1q_f64(r2);
            let mut ur2_hi = vld1q_f64(r2.add(2));
            let mut ur3_lo = vld1q_f64(r3);
            let mut ur3_hi = vld1q_f64(r3.add(2));
            let mut ui0_lo = vld1q_f64(i0);
            let mut ui0_hi = vld1q_f64(i0.add(2));
            let mut ui1_lo = vld1q_f64(i1);
            let mut ui1_hi = vld1q_f64(i1.add(2));
            let mut ui2_lo = vld1q_f64(i2);
            let mut ui2_hi = vld1q_f64(i2.add(2));
            let mut ui3_lo = vld1q_f64(i3);
            let mut ui3_hi = vld1q_f64(i3.add(2));

            // Stage 1: pair (r0,r2) and (r1,r3) with twiddle a.
            // tra = omar*ur2 - omai*ui2  (seed with omar*ur2; vfmsq subtracts omai*ui2)
            let mut tra_lo = vmulq_f64(omar, ur2_lo);
            let mut tra_hi = vmulq_f64(omar, ur2_hi);
            let mut trb_lo = vmulq_f64(omar, ur3_lo);
            let mut trb_hi = vmulq_f64(omar, ur3_hi);
            let mut tia_lo = vmulq_f64(omai, ur2_lo);
            let mut tia_hi = vmulq_f64(omai, ur2_hi);
            let mut tib_lo = vmulq_f64(omai, ur3_lo);
            let mut tib_hi = vmulq_f64(omai, ur3_hi);
            tra_lo = vfmsq_f64(tra_lo, omai, ui2_lo);
            tra_hi = vfmsq_f64(tra_hi, omai, ui2_hi);
            trb_lo = vfmsq_f64(trb_lo, omai, ui3_lo);
            trb_hi = vfmsq_f64(trb_hi, omai, ui3_hi);
            tia_lo = vfmaq_f64(tia_lo, omar, ui2_lo);
            tia_hi = vfmaq_f64(tia_hi, omar, ui2_hi);
            tib_lo = vfmaq_f64(tib_lo, omar, ui3_lo);
            tib_hi = vfmaq_f64(tib_hi, omar, ui3_hi);

            ur2_lo = vsubq_f64(ur0_lo, tra_lo);
            ur2_hi = vsubq_f64(ur0_hi, tra_hi);
            ur3_lo = vsubq_f64(ur1_lo, trb_lo);
            ur3_hi = vsubq_f64(ur1_hi, trb_hi);
            ui2_lo = vsubq_f64(ui0_lo, tia_lo);
            ui2_hi = vsubq_f64(ui0_hi, tia_hi);
            ui3_lo = vsubq_f64(ui1_lo, tib_lo);
            ui3_hi = vsubq_f64(ui1_hi, tib_hi);
            ur0_lo = vaddq_f64(ur0_lo, tra_lo);
            ur0_hi = vaddq_f64(ur0_hi, tra_hi);
            ur1_lo = vaddq_f64(ur1_lo, trb_lo);
            ur1_hi = vaddq_f64(ur1_hi, trb_hi);
            ui0_lo = vaddq_f64(ui0_lo, tia_lo);
            ui0_hi = vaddq_f64(ui0_hi, tia_hi);
            ui1_lo = vaddq_f64(ui1_lo, tib_lo);
            ui1_hi = vaddq_f64(ui1_hi, tib_hi);

            // Stage 2: cplx_twiddle on (r0,r1) and cplx_i_twiddle on (r2,r3) with twiddle b.
            // (r0, r1) line: tra = ombr*ur1 - ombi*ui1; tia = ombr*ui1 + ombi*ur1
            // (r2, r3) line: trb = ombi*ur3 + ombr*ui3; tib = ombi*ui3 - ombr*ur3
            tra_lo = vmulq_f64(ombr, ur1_lo);
            tra_hi = vmulq_f64(ombr, ur1_hi);
            trb_lo = vmulq_f64(ombr, ui3_lo);
            trb_hi = vmulq_f64(ombr, ui3_hi);
            tia_lo = vmulq_f64(ombi, ur1_lo);
            tia_hi = vmulq_f64(ombi, ur1_hi);
            tib_lo = vmulq_f64(ombi, ui3_lo);
            tib_hi = vmulq_f64(ombi, ui3_hi);
            tra_lo = vfmsq_f64(tra_lo, ombi, ui1_lo);
            tra_hi = vfmsq_f64(tra_hi, ombi, ui1_hi);
            trb_lo = vfmaq_f64(trb_lo, ombi, ur3_lo);
            trb_hi = vfmaq_f64(trb_hi, ombi, ur3_hi);
            tia_lo = vfmaq_f64(tia_lo, ombr, ui1_lo);
            tia_hi = vfmaq_f64(tia_hi, ombr, ui1_hi);
            tib_lo = vfmsq_f64(tib_lo, ombr, ur3_lo);
            tib_hi = vfmsq_f64(tib_hi, ombr, ur3_hi);

            ur1_lo = vsubq_f64(ur0_lo, tra_lo);
            ur1_hi = vsubq_f64(ur0_hi, tra_hi);
            ur3_lo = vaddq_f64(ur2_lo, trb_lo);
            ur3_hi = vaddq_f64(ur2_hi, trb_hi);
            ui1_lo = vsubq_f64(ui0_lo, tia_lo);
            ui1_hi = vsubq_f64(ui0_hi, tia_hi);
            ui3_lo = vaddq_f64(ui2_lo, tib_lo);
            ui3_hi = vaddq_f64(ui2_hi, tib_hi);
            ur0_lo = vaddq_f64(ur0_lo, tra_lo);
            ur0_hi = vaddq_f64(ur0_hi, tra_hi);
            ur2_lo = vsubq_f64(ur2_lo, trb_lo);
            ur2_hi = vsubq_f64(ur2_hi, trb_hi);
            ui0_lo = vaddq_f64(ui0_lo, tia_lo);
            ui0_hi = vaddq_f64(ui0_hi, tia_hi);
            ui2_lo = vsubq_f64(ui2_lo, tib_lo);
            ui2_hi = vsubq_f64(ui2_hi, tib_hi);

            vst1q_f64(r0, ur0_lo);
            vst1q_f64(r0.add(2), ur0_hi);
            vst1q_f64(r1, ur1_lo);
            vst1q_f64(r1.add(2), ur1_hi);
            vst1q_f64(r2, ur2_lo);
            vst1q_f64(r2.add(2), ur2_hi);
            vst1q_f64(r3, ur3_lo);
            vst1q_f64(r3.add(2), ur3_hi);
            vst1q_f64(i0, ui0_lo);
            vst1q_f64(i0.add(2), ui0_hi);
            vst1q_f64(i1, ui1_lo);
            vst1q_f64(i1.add(2), ui1_hi);
            vst1q_f64(i2, ui2_lo);
            vst1q_f64(i2.add(2), ui2_hi);
            vst1q_f64(i3, ui3_lo);
            vst1q_f64(i3.add(2), ui3_hi);

            r0 = r0.add(4);
            r1 = r1.add(4);
            r2 = r2.add(4);
            r3 = r3.add(4);
            i0 = i0.add(4);
            i1 = i1.add(4);
            i2 = i2.add(4);
            i3 = i3.add(4);
        }
    }
}

/// Inverse 2-way (Gentleman–Sande) butterfly. Mirrors `inv_twiddle_ifft_avx2_fma`.
#[inline]
unsafe fn inv_twiddle_ifft_neon(h: usize, re: &mut [f64], im: &mut [f64], omg: &[f64]) {
    debug_assert!(h.is_multiple_of(4));
    debug_assert!(omg.len() >= 2);
    unsafe {
        let omr: float64x2_t = vdupq_n_f64(omg[0]);
        let omi: float64x2_t = vdupq_n_f64(omg[1]);

        let mut r0 = re.as_mut_ptr();
        let mut r1 = re.as_mut_ptr().add(h);
        let mut i0 = im.as_mut_ptr();
        let mut i1 = im.as_mut_ptr().add(h);

        for _ in (0..h).step_by(4) {
            let mut ur0_lo = vld1q_f64(r0);
            let mut ur0_hi = vld1q_f64(r0.add(2));
            let mut ur1_lo = vld1q_f64(r1);
            let mut ur1_hi = vld1q_f64(r1.add(2));
            let mut ui0_lo = vld1q_f64(i0);
            let mut ui0_hi = vld1q_f64(i0.add(2));
            let mut ui1_lo = vld1q_f64(i1);
            let mut ui1_hi = vld1q_f64(i1.add(2));

            let tra_lo = vsubq_f64(ur0_lo, ur1_lo);
            let tra_hi = vsubq_f64(ur0_hi, ur1_hi);
            let tia_lo = vsubq_f64(ui0_lo, ui1_lo);
            let tia_hi = vsubq_f64(ui0_hi, ui1_hi);

            ur0_lo = vaddq_f64(ur0_lo, ur1_lo);
            ur0_hi = vaddq_f64(ur0_hi, ur1_hi);
            ui0_lo = vaddq_f64(ui0_lo, ui1_lo);
            ui0_hi = vaddq_f64(ui0_hi, ui1_hi);

            // ur1 = omr*tra - omi*tia
            ur1_lo = vmulq_f64(omr, tra_lo);
            ur1_hi = vmulq_f64(omr, tra_hi);
            ur1_lo = vfmsq_f64(ur1_lo, omi, tia_lo);
            ur1_hi = vfmsq_f64(ur1_hi, omi, tia_hi);

            // ui1 = omr*tia + omi*tra
            ui1_lo = vmulq_f64(omi, tra_lo);
            ui1_hi = vmulq_f64(omi, tra_hi);
            ui1_lo = vfmaq_f64(ui1_lo, omr, tia_lo);
            ui1_hi = vfmaq_f64(ui1_hi, omr, tia_hi);

            vst1q_f64(r0, ur0_lo);
            vst1q_f64(r0.add(2), ur0_hi);
            vst1q_f64(r1, ur1_lo);
            vst1q_f64(r1.add(2), ur1_hi);
            vst1q_f64(i0, ui0_lo);
            vst1q_f64(i0.add(2), ui0_hi);
            vst1q_f64(i1, ui1_lo);
            vst1q_f64(i1.add(2), ui1_hi);

            r0 = r0.add(4);
            r1 = r1.add(4);
            i0 = i0.add(4);
            i1 = i1.add(4);
        }
    }
}

/// Inverse 4-way bitwiddle butterfly. Mirrors `inv_bitwiddle_ifft_avx2_fma`.
#[inline]
unsafe fn inv_bitwiddle_ifft_neon(h: usize, re: &mut [f64], im: &mut [f64], omg: &[f64]) {
    debug_assert!(h.is_multiple_of(4));
    debug_assert!(omg.len() >= 4);
    unsafe {
        let mut r0 = re.as_mut_ptr();
        let mut r1 = re.as_mut_ptr().add(h);
        let mut r2 = re.as_mut_ptr().add(2 * h);
        let mut r3 = re.as_mut_ptr().add(3 * h);
        let mut i0 = im.as_mut_ptr();
        let mut i1 = im.as_mut_ptr().add(h);
        let mut i2 = im.as_mut_ptr().add(2 * h);
        let mut i3 = im.as_mut_ptr().add(3 * h);

        let omar: float64x2_t = vdupq_n_f64(omg[0]);
        let omai: float64x2_t = vdupq_n_f64(omg[1]);
        let ombr: float64x2_t = vdupq_n_f64(omg[2]);
        let ombi: float64x2_t = vdupq_n_f64(omg[3]);

        for _ in (0..h).step_by(4) {
            let mut ur0_lo = vld1q_f64(r0);
            let mut ur0_hi = vld1q_f64(r0.add(2));
            let mut ur1_lo = vld1q_f64(r1);
            let mut ur1_hi = vld1q_f64(r1.add(2));
            let mut ur2_lo = vld1q_f64(r2);
            let mut ur2_hi = vld1q_f64(r2.add(2));
            let mut ur3_lo = vld1q_f64(r3);
            let mut ur3_hi = vld1q_f64(r3.add(2));
            let mut ui0_lo = vld1q_f64(i0);
            let mut ui0_hi = vld1q_f64(i0.add(2));
            let mut ui1_lo = vld1q_f64(i1);
            let mut ui1_hi = vld1q_f64(i1.add(2));
            let mut ui2_lo = vld1q_f64(i2);
            let mut ui2_hi = vld1q_f64(i2.add(2));
            let mut ui3_lo = vld1q_f64(i3);
            let mut ui3_hi = vld1q_f64(i3.add(2));

            // Stage 1: inv_twiddle on (r0,r1) and inv_itwiddle on (r2,r3) with twiddle a.
            let mut tra_lo = vsubq_f64(ur0_lo, ur1_lo);
            let mut tra_hi = vsubq_f64(ur0_hi, ur1_hi);
            let mut trb_lo = vsubq_f64(ur2_lo, ur3_lo);
            let mut trb_hi = vsubq_f64(ur2_hi, ur3_hi);
            let mut tia_lo = vsubq_f64(ui0_lo, ui1_lo);
            let mut tia_hi = vsubq_f64(ui0_hi, ui1_hi);
            let mut tib_lo = vsubq_f64(ui2_lo, ui3_lo);
            let mut tib_hi = vsubq_f64(ui2_hi, ui3_hi);

            ur0_lo = vaddq_f64(ur0_lo, ur1_lo);
            ur0_hi = vaddq_f64(ur0_hi, ur1_hi);
            ur2_lo = vaddq_f64(ur2_lo, ur3_lo);
            ur2_hi = vaddq_f64(ur2_hi, ur3_hi);
            ui0_lo = vaddq_f64(ui0_lo, ui1_lo);
            ui0_hi = vaddq_f64(ui0_hi, ui1_hi);
            ui2_lo = vaddq_f64(ui2_lo, ui3_lo);
            ui2_hi = vaddq_f64(ui2_hi, ui3_hi);

            // ur1 = omar*tra - omai*tia    (inv_twiddle real)
            // ur3 = omai*trb + omar*tib    (inv_itwiddle real, +)
            // ui1 = omar*tia + omai*tra    (inv_twiddle imag)
            // ui3 = omai*tib - omar*trb    (inv_itwiddle imag, -)
            ur1_lo = vmulq_f64(omar, tra_lo);
            ur1_hi = vmulq_f64(omar, tra_hi);
            ur3_lo = vmulq_f64(omar, tib_lo);
            ur3_hi = vmulq_f64(omar, tib_hi);
            ui1_lo = vmulq_f64(omai, tra_lo);
            ui1_hi = vmulq_f64(omai, tra_hi);
            ui3_lo = vmulq_f64(omai, tib_lo);
            ui3_hi = vmulq_f64(omai, tib_hi);
            ur1_lo = vfmsq_f64(ur1_lo, omai, tia_lo);
            ur1_hi = vfmsq_f64(ur1_hi, omai, tia_hi);
            ur3_lo = vfmaq_f64(ur3_lo, omai, trb_lo);
            ur3_hi = vfmaq_f64(ur3_hi, omai, trb_hi);
            ui1_lo = vfmaq_f64(ui1_lo, omar, tia_lo);
            ui1_hi = vfmaq_f64(ui1_hi, omar, tia_hi);
            ui3_lo = vfmsq_f64(ui3_lo, omar, trb_lo);
            ui3_hi = vfmsq_f64(ui3_hi, omar, trb_hi);

            // Stage 2: inv_twiddle on (r0,r2) and (r1,r3) with twiddle b.
            tra_lo = vsubq_f64(ur0_lo, ur2_lo);
            tra_hi = vsubq_f64(ur0_hi, ur2_hi);
            trb_lo = vsubq_f64(ur1_lo, ur3_lo);
            trb_hi = vsubq_f64(ur1_hi, ur3_hi);
            tia_lo = vsubq_f64(ui0_lo, ui2_lo);
            tia_hi = vsubq_f64(ui0_hi, ui2_hi);
            tib_lo = vsubq_f64(ui1_lo, ui3_lo);
            tib_hi = vsubq_f64(ui1_hi, ui3_hi);

            ur0_lo = vaddq_f64(ur0_lo, ur2_lo);
            ur0_hi = vaddq_f64(ur0_hi, ur2_hi);
            ur1_lo = vaddq_f64(ur1_lo, ur3_lo);
            ur1_hi = vaddq_f64(ur1_hi, ur3_hi);
            ui0_lo = vaddq_f64(ui0_lo, ui2_lo);
            ui0_hi = vaddq_f64(ui0_hi, ui2_hi);
            ui1_lo = vaddq_f64(ui1_lo, ui3_lo);
            ui1_hi = vaddq_f64(ui1_hi, ui3_hi);

            // ur2 = ombr*tra - ombi*tia
            // ur3 = ombr*trb - ombi*tib
            // ui2 = ombr*tia + ombi*tra
            // ui3 = ombr*tib + ombi*trb
            ur2_lo = vmulq_f64(ombr, tra_lo);
            ur2_hi = vmulq_f64(ombr, tra_hi);
            ur3_lo = vmulq_f64(ombr, trb_lo);
            ur3_hi = vmulq_f64(ombr, trb_hi);
            ui2_lo = vmulq_f64(ombi, tra_lo);
            ui2_hi = vmulq_f64(ombi, tra_hi);
            ui3_lo = vmulq_f64(ombi, trb_lo);
            ui3_hi = vmulq_f64(ombi, trb_hi);
            ur2_lo = vfmsq_f64(ur2_lo, ombi, tia_lo);
            ur2_hi = vfmsq_f64(ur2_hi, ombi, tia_hi);
            ur3_lo = vfmsq_f64(ur3_lo, ombi, tib_lo);
            ur3_hi = vfmsq_f64(ur3_hi, ombi, tib_hi);
            ui2_lo = vfmaq_f64(ui2_lo, ombr, tia_lo);
            ui2_hi = vfmaq_f64(ui2_hi, ombr, tia_hi);
            ui3_lo = vfmaq_f64(ui3_lo, ombr, tib_lo);
            ui3_hi = vfmaq_f64(ui3_hi, ombr, tib_hi);

            vst1q_f64(r0, ur0_lo);
            vst1q_f64(r0.add(2), ur0_hi);
            vst1q_f64(r1, ur1_lo);
            vst1q_f64(r1.add(2), ur1_hi);
            vst1q_f64(r2, ur2_lo);
            vst1q_f64(r2.add(2), ur2_hi);
            vst1q_f64(r3, ur3_lo);
            vst1q_f64(r3.add(2), ur3_hi);
            vst1q_f64(i0, ui0_lo);
            vst1q_f64(i0.add(2), ui0_hi);
            vst1q_f64(i1, ui1_lo);
            vst1q_f64(i1.add(2), ui1_hi);
            vst1q_f64(i2, ui2_lo);
            vst1q_f64(i2.add(2), ui2_hi);
            vst1q_f64(i3, ui3_lo);
            vst1q_f64(i3.add(2), ui3_hi);

            r0 = r0.add(4);
            r1 = r1.add(4);
            r2 = r2.add(4);
            r3 = r3.add(4);
            i0 = i0.add(4);
            i1 = i1.add(4);
            i2 = i2.add(4);
            i3 = i3.add(4);
        }
    }
}

// ─── size-16 leaves (NEON-intrinsic radix-2 over 16 complex points) ────────
//
// Layout: each `float64x2_t` holds two consecutive doubles, so reg `rk` =
// (re[2k], re[2k+1]) for k ∈ [0..8) and likewise for im. Stages 1–3 of the
// forward FFT (and 2–4 of the inverse) butterfly whole regs; stage 4 (resp.
// stage 1 of the inverse) butterflies the two lanes within a single reg, so
// we pair consecutive regs and use `vzip1q`/`vzip2q` to deinterleave / re-
// interleave the per-lane operands.
//
// Algorithm shape lifted from `fft16_ref` / `ifft16_ref`
// (`poulpy-cpu-ref/src/reference/fft64/reim/{fft,ifft}_ref.rs`).

#[inline(always)]
unsafe fn cplx_twiddle_neon(
    ra: &mut float64x2_t,
    ia: &mut float64x2_t,
    rb: &mut float64x2_t,
    ib: &mut float64x2_t,
    omr: float64x2_t,
    omi: float64x2_t,
) {
    unsafe {
        // dr = rb*omr - ib*omi, di = rb*omi + ib*omr
        let dr = vfmsq_f64(vmulq_f64(*rb, omr), *ib, omi);
        let di = vfmaq_f64(vmulq_f64(*rb, omi), *ib, omr);
        let nra = vaddq_f64(*ra, dr);
        let nia = vaddq_f64(*ia, di);
        let nrb = vsubq_f64(*ra, dr);
        let nib = vsubq_f64(*ia, di);
        *ra = nra;
        *ia = nia;
        *rb = nrb;
        *ib = nib;
    }
}

#[inline(always)]
unsafe fn cplx_i_twiddle_neon(
    ra: &mut float64x2_t,
    ia: &mut float64x2_t,
    rb: &mut float64x2_t,
    ib: &mut float64x2_t,
    omr: float64x2_t,
    omi: float64x2_t,
) {
    unsafe {
        // dr = rb*omi + ib*omr, di = rb*omr - ib*omi
        let dr = vfmaq_f64(vmulq_f64(*rb, omi), *ib, omr);
        let di = vfmsq_f64(vmulq_f64(*rb, omr), *ib, omi);
        let nra = vsubq_f64(*ra, dr);
        let nia = vaddq_f64(*ia, di);
        let nrb = vaddq_f64(*ra, dr);
        let nib = vsubq_f64(*ia, di);
        *ra = nra;
        *ia = nia;
        *rb = nrb;
        *ib = nib;
    }
}

#[inline(always)]
unsafe fn inv_twiddle_neon(
    ra: &mut float64x2_t,
    ia: &mut float64x2_t,
    rb: &mut float64x2_t,
    ib: &mut float64x2_t,
    omr: float64x2_t,
    omi: float64x2_t,
) {
    unsafe {
        let r_diff = vsubq_f64(*ra, *rb);
        let i_diff = vsubq_f64(*ia, *ib);
        let nra = vaddq_f64(*ra, *rb);
        let nia = vaddq_f64(*ia, *ib);
        // rb' = r_diff*omr - i_diff*omi, ib' = r_diff*omi + i_diff*omr
        let nrb = vfmsq_f64(vmulq_f64(r_diff, omr), i_diff, omi);
        let nib = vfmaq_f64(vmulq_f64(r_diff, omi), i_diff, omr);
        *ra = nra;
        *ia = nia;
        *rb = nrb;
        *ib = nib;
    }
}

#[inline(always)]
unsafe fn inv_itwiddle_neon(
    ra: &mut float64x2_t,
    ia: &mut float64x2_t,
    rb: &mut float64x2_t,
    ib: &mut float64x2_t,
    omr: float64x2_t,
    omi: float64x2_t,
) {
    unsafe {
        let r_diff = vsubq_f64(*ra, *rb);
        let i_diff = vsubq_f64(*ia, *ib);
        let nra = vaddq_f64(*ra, *rb);
        let nia = vaddq_f64(*ia, *ib);
        // rb' = r_diff*omi + i_diff*omr, ib' = i_diff*omi - r_diff*omr
        let nrb = vfmaq_f64(vmulq_f64(r_diff, omi), i_diff, omr);
        let nib = vfmsq_f64(vmulq_f64(i_diff, omi), r_diff, omr);
        *ra = nra;
        *ia = nia;
        *rb = nrb;
        *ib = nib;
    }
}

#[inline]
unsafe fn fft16_neon(re: &mut [f64], im: &mut [f64], omg: &[f64]) {
    debug_assert!(re.len() >= 16 && im.len() >= 16 && omg.len() >= 16);
    unsafe {
        let r = re.as_mut_ptr();
        let i = im.as_mut_ptr();
        let o = omg.as_ptr();

        // Load: rk = (re[2k], re[2k+1]); ik = (im[2k], im[2k+1]).
        let mut r0 = vld1q_f64(r);
        let mut r1 = vld1q_f64(r.add(2));
        let mut r2 = vld1q_f64(r.add(4));
        let mut r3 = vld1q_f64(r.add(6));
        let mut r4 = vld1q_f64(r.add(8));
        let mut r5 = vld1q_f64(r.add(10));
        let mut r6 = vld1q_f64(r.add(12));
        let mut r7 = vld1q_f64(r.add(14));
        let mut i0 = vld1q_f64(i);
        let mut i1 = vld1q_f64(i.add(2));
        let mut i2 = vld1q_f64(i.add(4));
        let mut i3 = vld1q_f64(i.add(6));
        let mut i4 = vld1q_f64(i.add(8));
        let mut i5 = vld1q_f64(i.add(10));
        let mut i6 = vld1q_f64(i.add(12));
        let mut i7 = vld1q_f64(i.add(14));

        // Stage 1 (omg[0..2]): cplx_twiddle pairs (k, k+4) for k in [0..4).
        let omr = vdupq_n_f64(*o);
        let omi = vdupq_n_f64(*o.add(1));
        cplx_twiddle_neon(&mut r0, &mut i0, &mut r4, &mut i4, omr, omi);
        cplx_twiddle_neon(&mut r1, &mut i1, &mut r5, &mut i5, omr, omi);
        cplx_twiddle_neon(&mut r2, &mut i2, &mut r6, &mut i6, omr, omi);
        cplx_twiddle_neon(&mut r3, &mut i3, &mut r7, &mut i7, omr, omi);

        // Stage 2 (omg[2..4]): cplx_twiddle on top half, cplx_i_twiddle on bottom.
        let omr = vdupq_n_f64(*o.add(2));
        let omi = vdupq_n_f64(*o.add(3));
        cplx_twiddle_neon(&mut r0, &mut i0, &mut r2, &mut i2, omr, omi);
        cplx_twiddle_neon(&mut r1, &mut i1, &mut r3, &mut i3, omr, omi);
        cplx_i_twiddle_neon(&mut r4, &mut i4, &mut r6, &mut i6, omr, omi);
        cplx_i_twiddle_neon(&mut r5, &mut i5, &mut r7, &mut i7, omr, omi);

        // Stage 3 (omg[4..8]): two sub-groups, each cplx_twiddle + cplx_i_twiddle.
        let omr_a = vdupq_n_f64(*o.add(4));
        let omi_a = vdupq_n_f64(*o.add(5));
        let omr_b = vdupq_n_f64(*o.add(6));
        let omi_b = vdupq_n_f64(*o.add(7));
        cplx_twiddle_neon(&mut r0, &mut i0, &mut r1, &mut i1, omr_a, omi_a);
        cplx_twiddle_neon(&mut r4, &mut i4, &mut r5, &mut i5, omr_b, omi_b);
        cplx_i_twiddle_neon(&mut r2, &mut i2, &mut r3, &mut i3, omr_a, omi_a);
        cplx_i_twiddle_neon(&mut r6, &mut i6, &mut r7, &mut i7, omr_b, omi_b);

        // Stage 4 (omg[8..16]): within-register lane butterflies. Pair consecutive
        // regs so each NEON op runs two scalar butterflies (one per lane).
        let omr_lo = vld1q_f64(o.add(8)); // (omg[8],  omg[9])
        let omr_hi = vld1q_f64(o.add(10)); // (omg[10], omg[11])
        let omi_lo = vld1q_f64(o.add(12)); // (omg[12], omg[13])
        let omi_hi = vld1q_f64(o.add(14)); // (omg[14], omg[15])

        // Group covering r[0..3]: cplx_twiddle on r0,r2; cplx_i_twiddle on r1,r3.
        let mut xa_r = vzip1q_f64(r0, r2);
        let mut xb_r = vzip2q_f64(r0, r2);
        let mut xa_i = vzip1q_f64(i0, i2);
        let mut xb_i = vzip2q_f64(i0, i2);
        let mut ya_r = vzip1q_f64(r1, r3);
        let mut yb_r = vzip2q_f64(r1, r3);
        let mut ya_i = vzip1q_f64(i1, i3);
        let mut yb_i = vzip2q_f64(i1, i3);
        cplx_twiddle_neon(&mut xa_r, &mut xa_i, &mut xb_r, &mut xb_i, omr_lo, omi_lo);
        cplx_i_twiddle_neon(&mut ya_r, &mut ya_i, &mut yb_r, &mut yb_i, omr_lo, omi_lo);
        r0 = vzip1q_f64(xa_r, xb_r);
        r2 = vzip2q_f64(xa_r, xb_r);
        i0 = vzip1q_f64(xa_i, xb_i);
        i2 = vzip2q_f64(xa_i, xb_i);
        r1 = vzip1q_f64(ya_r, yb_r);
        r3 = vzip2q_f64(ya_r, yb_r);
        i1 = vzip1q_f64(ya_i, yb_i);
        i3 = vzip2q_f64(ya_i, yb_i);

        // Group covering r[4..7]: cplx_twiddle on r4,r6; cplx_i_twiddle on r5,r7.
        let mut xa_r = vzip1q_f64(r4, r6);
        let mut xb_r = vzip2q_f64(r4, r6);
        let mut xa_i = vzip1q_f64(i4, i6);
        let mut xb_i = vzip2q_f64(i4, i6);
        let mut ya_r = vzip1q_f64(r5, r7);
        let mut yb_r = vzip2q_f64(r5, r7);
        let mut ya_i = vzip1q_f64(i5, i7);
        let mut yb_i = vzip2q_f64(i5, i7);
        cplx_twiddle_neon(&mut xa_r, &mut xa_i, &mut xb_r, &mut xb_i, omr_hi, omi_hi);
        cplx_i_twiddle_neon(&mut ya_r, &mut ya_i, &mut yb_r, &mut yb_i, omr_hi, omi_hi);
        r4 = vzip1q_f64(xa_r, xb_r);
        r6 = vzip2q_f64(xa_r, xb_r);
        i4 = vzip1q_f64(xa_i, xb_i);
        i6 = vzip2q_f64(xa_i, xb_i);
        r5 = vzip1q_f64(ya_r, yb_r);
        r7 = vzip2q_f64(ya_r, yb_r);
        i5 = vzip1q_f64(ya_i, yb_i);
        i7 = vzip2q_f64(ya_i, yb_i);

        vst1q_f64(r, r0);
        vst1q_f64(r.add(2), r1);
        vst1q_f64(r.add(4), r2);
        vst1q_f64(r.add(6), r3);
        vst1q_f64(r.add(8), r4);
        vst1q_f64(r.add(10), r5);
        vst1q_f64(r.add(12), r6);
        vst1q_f64(r.add(14), r7);
        vst1q_f64(i, i0);
        vst1q_f64(i.add(2), i1);
        vst1q_f64(i.add(4), i2);
        vst1q_f64(i.add(6), i3);
        vst1q_f64(i.add(8), i4);
        vst1q_f64(i.add(10), i5);
        vst1q_f64(i.add(12), i6);
        vst1q_f64(i.add(14), i7);
    }
}

#[inline]
unsafe fn ifft16_neon(re: &mut [f64], im: &mut [f64], omg: &[f64]) {
    debug_assert!(re.len() >= 16 && im.len() >= 16 && omg.len() >= 16);
    unsafe {
        let r = re.as_mut_ptr();
        let i = im.as_mut_ptr();
        let o = omg.as_ptr();

        let mut r0 = vld1q_f64(r);
        let mut r1 = vld1q_f64(r.add(2));
        let mut r2 = vld1q_f64(r.add(4));
        let mut r3 = vld1q_f64(r.add(6));
        let mut r4 = vld1q_f64(r.add(8));
        let mut r5 = vld1q_f64(r.add(10));
        let mut r6 = vld1q_f64(r.add(12));
        let mut r7 = vld1q_f64(r.add(14));
        let mut i0 = vld1q_f64(i);
        let mut i1 = vld1q_f64(i.add(2));
        let mut i2 = vld1q_f64(i.add(4));
        let mut i3 = vld1q_f64(i.add(6));
        let mut i4 = vld1q_f64(i.add(8));
        let mut i5 = vld1q_f64(i.add(10));
        let mut i6 = vld1q_f64(i.add(12));
        let mut i7 = vld1q_f64(i.add(14));

        // Stage 1 (omg[0..8]): within-register lane butterflies. Twiddle k uses
        // (omg[k], omg[k+4]); inv_twiddle on r[2k], inv_itwiddle on r[2k+1].
        let omr_lo = vld1q_f64(o); // (omg[0], omg[1])
        let omr_hi = vld1q_f64(o.add(2)); // (omg[2], omg[3])
        let omi_lo = vld1q_f64(o.add(4)); // (omg[4], omg[5])
        let omi_hi = vld1q_f64(o.add(6)); // (omg[6], omg[7])

        // Group r[0..3]: inv_twiddle on r0,r2; inv_itwiddle on r1,r3 (twiddles lo).
        let mut xa_r = vzip1q_f64(r0, r2);
        let mut xb_r = vzip2q_f64(r0, r2);
        let mut xa_i = vzip1q_f64(i0, i2);
        let mut xb_i = vzip2q_f64(i0, i2);
        let mut ya_r = vzip1q_f64(r1, r3);
        let mut yb_r = vzip2q_f64(r1, r3);
        let mut ya_i = vzip1q_f64(i1, i3);
        let mut yb_i = vzip2q_f64(i1, i3);
        inv_twiddle_neon(&mut xa_r, &mut xa_i, &mut xb_r, &mut xb_i, omr_lo, omi_lo);
        inv_itwiddle_neon(&mut ya_r, &mut ya_i, &mut yb_r, &mut yb_i, omr_lo, omi_lo);
        r0 = vzip1q_f64(xa_r, xb_r);
        r2 = vzip2q_f64(xa_r, xb_r);
        i0 = vzip1q_f64(xa_i, xb_i);
        i2 = vzip2q_f64(xa_i, xb_i);
        r1 = vzip1q_f64(ya_r, yb_r);
        r3 = vzip2q_f64(ya_r, yb_r);
        i1 = vzip1q_f64(ya_i, yb_i);
        i3 = vzip2q_f64(ya_i, yb_i);

        // Group r[4..7]: inv_twiddle on r4,r6; inv_itwiddle on r5,r7 (twiddles hi).
        let mut xa_r = vzip1q_f64(r4, r6);
        let mut xb_r = vzip2q_f64(r4, r6);
        let mut xa_i = vzip1q_f64(i4, i6);
        let mut xb_i = vzip2q_f64(i4, i6);
        let mut ya_r = vzip1q_f64(r5, r7);
        let mut yb_r = vzip2q_f64(r5, r7);
        let mut ya_i = vzip1q_f64(i5, i7);
        let mut yb_i = vzip2q_f64(i5, i7);
        inv_twiddle_neon(&mut xa_r, &mut xa_i, &mut xb_r, &mut xb_i, omr_hi, omi_hi);
        inv_itwiddle_neon(&mut ya_r, &mut ya_i, &mut yb_r, &mut yb_i, omr_hi, omi_hi);
        r4 = vzip1q_f64(xa_r, xb_r);
        r6 = vzip2q_f64(xa_r, xb_r);
        i4 = vzip1q_f64(xa_i, xb_i);
        i6 = vzip2q_f64(xa_i, xb_i);
        r5 = vzip1q_f64(ya_r, yb_r);
        r7 = vzip2q_f64(ya_r, yb_r);
        i5 = vzip1q_f64(ya_i, yb_i);
        i7 = vzip2q_f64(ya_i, yb_i);

        // Stage 2 (omg[8..12]): pairs (j, j+2). inv_twiddle/inv_itwiddle interleaved.
        let omr_a = vdupq_n_f64(*o.add(8));
        let omi_a = vdupq_n_f64(*o.add(9));
        let omr_b = vdupq_n_f64(*o.add(10));
        let omi_b = vdupq_n_f64(*o.add(11));
        inv_twiddle_neon(&mut r0, &mut i0, &mut r1, &mut i1, omr_a, omi_a);
        inv_itwiddle_neon(&mut r2, &mut i2, &mut r3, &mut i3, omr_a, omi_a);
        inv_twiddle_neon(&mut r4, &mut i4, &mut r5, &mut i5, omr_b, omi_b);
        inv_itwiddle_neon(&mut r6, &mut i6, &mut r7, &mut i7, omr_b, omi_b);

        // Stage 3 (omg[12..14]): pairs (j, j+4). inv_twiddle on top, inv_itwiddle on bottom.
        let omr = vdupq_n_f64(*o.add(12));
        let omi = vdupq_n_f64(*o.add(13));
        inv_twiddle_neon(&mut r0, &mut i0, &mut r2, &mut i2, omr, omi);
        inv_twiddle_neon(&mut r1, &mut i1, &mut r3, &mut i3, omr, omi);
        inv_itwiddle_neon(&mut r4, &mut i4, &mut r6, &mut i6, omr, omi);
        inv_itwiddle_neon(&mut r5, &mut i5, &mut r7, &mut i7, omr, omi);

        // Stage 4 (omg[14..16]): pairs (k, k+4) for k in [0..4). All inv_twiddle.
        let omr = vdupq_n_f64(*o.add(14));
        let omi = vdupq_n_f64(*o.add(15));
        inv_twiddle_neon(&mut r0, &mut i0, &mut r4, &mut i4, omr, omi);
        inv_twiddle_neon(&mut r1, &mut i1, &mut r5, &mut i5, omr, omi);
        inv_twiddle_neon(&mut r2, &mut i2, &mut r6, &mut i6, omr, omi);
        inv_twiddle_neon(&mut r3, &mut i3, &mut r7, &mut i7, omr, omi);

        vst1q_f64(r, r0);
        vst1q_f64(r.add(2), r1);
        vst1q_f64(r.add(4), r2);
        vst1q_f64(r.add(6), r3);
        vst1q_f64(r.add(8), r4);
        vst1q_f64(r.add(10), r5);
        vst1q_f64(r.add(12), r6);
        vst1q_f64(r.add(14), r7);
        vst1q_f64(i, i0);
        vst1q_f64(i.add(2), i1);
        vst1q_f64(i.add(4), i2);
        vst1q_f64(i.add(6), i3);
        vst1q_f64(i.add(8), i4);
        vst1q_f64(i.add(10), i5);
        vst1q_f64(i.add(12), i6);
        vst1q_f64(i.add(14), i7);
    }
}

// ─── tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use poulpy_cpu_ref::reference::fft64::reim::{ReimFFTTable, ReimIFFTTable, fft_ref, ifft_ref};

    use super::{fft_neon, ifft_neon};

    /// FFT → IFFT round-trip residual matches the scalar reference's residual.
    #[test]
    fn fft_ifft_roundtrip_neon() {
        for log_m in 1..=8 {
            let m = 1usize << log_m;
            let fwd = ReimFFTTable::<f64>::new(m);
            let inv = ReimIFFTTable::<f64>::new(m);

            let data: Vec<f64> = (0..2 * m).map(|i| (i + 1) as f64 / m as f64).collect();

            let mut neon = data.clone();
            fft_neon(m, fwd.omg(), &mut neon);
            ifft_neon(m, inv.omg(), &mut neon);

            let mut reference = data.clone();
            fft_ref(m, fwd.omg(), &mut reference);
            ifft_ref(m, inv.omg(), &mut reference);

            let tol = 1e-10f64;
            for i in 0..2 * m {
                let diff = (neon[i] - reference[i]).abs();
                assert!(
                    diff <= tol,
                    "log_m={log_m} idx={i}: NEON={} ref={} diff={diff}",
                    neon[i],
                    reference[i]
                );
            }
        }
    }

    /// NEON FFT matches scalar `fft_ref` within ULP tolerance.
    #[test]
    fn fft_neon_vs_ref() {
        for log_m in 0..14 {
            let m = 1usize << log_m;
            let fwd = ReimFFTTable::<f64>::new(m);

            let mut values_neon: Vec<f64> = vec![0f64; m << 1];
            let scale: f64 = 1.0f64 / m as f64;
            values_neon
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = (i + 1) as f64 * scale);
            let mut values_ref = values_neon.clone();

            fft_neon(m, fwd.omg(), &mut values_neon);
            fft_ref(m, fwd.omg(), &mut values_ref);

            let max_diff: f64 = 1.0 / ((1u64 << (53 - log_m - 1)) as f64);
            for i in 0..m * 2 {
                let diff: f64 = (values_neon[i] - values_ref[i]).abs();
                assert!(
                    diff <= max_diff,
                    "log_m={log_m} idx={i} NEON={} ref={} diff={diff}",
                    values_neon[i],
                    values_ref[i]
                );
            }
        }
    }

    /// NEON IFFT matches scalar `ifft_ref` within ULP tolerance.
    #[test]
    fn ifft_neon_vs_ref() {
        for log_m in 0..14 {
            let m = 1usize << log_m;
            let inv = ReimIFFTTable::<f64>::new(m);

            let mut values_neon: Vec<f64> = vec![0f64; m << 1];
            let scale: f64 = 1.0f64 / m as f64;
            values_neon
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = (i + 1) as f64 * scale);
            let mut values_ref = values_neon.clone();

            ifft_neon(m, inv.omg(), &mut values_neon);
            ifft_ref(m, inv.omg(), &mut values_ref);

            let max_diff: f64 = 1.0 / ((1u64 << (53 - log_m - 1)) as f64);
            for i in 0..m * 2 {
                let diff: f64 = (values_neon[i] - values_ref[i]).abs();
                assert!(
                    diff <= max_diff,
                    "log_m={log_m} idx={i} NEON={} ref={} diff={diff}",
                    values_neon[i],
                    values_ref[i]
                );
            }
        }
    }

    /// Frequency-domain convolution via NEON matches the scalar reference.
    #[test]
    fn fft_convolution_neon() {
        let m = 32usize;
        let fwd = ReimFFTTable::<f64>::new(m);
        let inv = ReimIFFTTable::<f64>::new(m);

        let mut a_neon = vec![0f64; 2 * m];
        a_neon[0] = 1.0;
        a_neon[1] = 2.0;
        let mut b_neon = vec![0f64; 2 * m];
        b_neon[0] = 3.0;
        b_neon[1] = 4.0;

        let mut a_ref = a_neon.clone();
        let mut b_ref = b_neon.clone();

        fft_neon(m, fwd.omg(), &mut a_neon);
        fft_neon(m, fwd.omg(), &mut b_neon);
        fft_ref(m, fwd.omg(), &mut a_ref);
        fft_ref(m, fwd.omg(), &mut b_ref);

        let mut c_neon = vec![0f64; 2 * m];
        let mut c_ref = vec![0f64; 2 * m];
        for k in 0..m {
            c_neon[k] = a_neon[k] * b_neon[k] - a_neon[k + m] * b_neon[k + m];
            c_neon[k + m] = a_neon[k] * b_neon[k + m] + a_neon[k + m] * b_neon[k];
            c_ref[k] = a_ref[k] * b_ref[k] - a_ref[k + m] * b_ref[k + m];
            c_ref[k + m] = a_ref[k] * b_ref[k + m] + a_ref[k + m] * b_ref[k];
        }

        ifft_neon(m, inv.omg(), &mut c_neon);
        ifft_ref(m, inv.omg(), &mut c_ref);

        let tol = 1e-8f64;
        for i in 0..2 * m {
            let diff = (c_neon[i] - c_ref[i]).abs();
            assert!(diff <= tol, "idx={i}: NEON={} ref={} diff={diff}", c_neon[i], c_ref[i]);
        }
    }
}
