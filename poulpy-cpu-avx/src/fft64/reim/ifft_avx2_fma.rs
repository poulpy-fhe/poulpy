// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

use std::arch::x86_64::{
    __m128d, __m256d, _mm_load_pd, _mm256_add_pd, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_mul_pd,
    _mm256_permute2f128_pd, _mm256_set_m128d, _mm256_storeu_pd, _mm256_sub_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
};

use crate::fft64::reim::{as_arr, as_arr_mut};

#[target_feature(enable = "avx2,fma")]
pub(crate) fn ifft_avx2_fma(m: usize, omg: &[f64], data: &mut [f64]) {
    if m < 16 {
        use poulpy_cpu_portable::reference::fft64::reim::ifft_ref;
        ifft_ref(m, omg, data);
        return;
    }

    assert!(data.len() == 2 * m);
    let (re, im) = data.split_at_mut(m);

    if m == 16 {
        ifft16_avx2_fma(as_arr_mut::<16, f64>(re), as_arr_mut::<16, f64>(im), as_arr::<16, f64>(omg))
    } else if m <= 2048 {
        ifft_bfs_16_avx2_fma(m, re, im, omg, 0);
    } else {
        ifft_rec_16_avx2_fma(m, re, im, omg, 0);
    }
}

unsafe extern "sysv64" {
    unsafe fn ifft16_avx2_fma_asm(re: *mut f64, im: *mut f64, omg: *const f64);
}

#[target_feature(enable = "avx2,fma")]
fn ifft16_avx2_fma(re: &mut [f64; 16], im: &mut [f64; 16], omg: &[f64; 16]) {
    unsafe {
        ifft16_avx2_fma_asm(re.as_mut_ptr(), im.as_mut_ptr(), omg.as_ptr());
    }
}

#[target_feature(enable = "avx2,fma")]
fn ifft_rec_16_avx2_fma(m: usize, re: &mut [f64], im: &mut [f64], omg: &[f64], mut pos: usize) -> usize {
    if m <= 2048 {
        return ifft_bfs_16_avx2_fma(m, re, im, omg, pos);
    };
    let h: usize = m >> 1;
    pos = ifft_rec_16_avx2_fma(h, re, im, omg, pos);
    pos = ifft_rec_16_avx2_fma(h, &mut re[h..], &mut im[h..], omg, pos);
    inv_twiddle_ifft_avx2_fma(h, re, im, *as_arr::<2, f64>(&omg[pos..]));
    pos += 2;
    pos
}

#[target_feature(enable = "avx2,fma")]
fn ifft_bfs_16_avx2_fma(m: usize, re: &mut [f64], im: &mut [f64], omg: &[f64], mut pos: usize) -> usize {
    let log_m: usize = (usize::BITS - (m - 1).leading_zeros()) as usize;

    for off in (0..m).step_by(16) {
        ifft16_avx2_fma(
            as_arr_mut::<16, f64>(&mut re[off..]),
            as_arr_mut::<16, f64>(&mut im[off..]),
            as_arr::<16, f64>(&omg[pos..]),
        );
        pos += 16;
    }

    let mut h: usize = 16;
    let m_half: usize = m >> 1;

    while h < m_half {
        let mm: usize = h << 2;
        for off in (0..m).step_by(mm) {
            inv_bitwiddle_ifft_avx2_fma(h, &mut re[off..], &mut im[off..], as_arr::<4, f64>(&omg[pos..]));
            pos += 4;
        }
        h = mm;
    }

    if !log_m.is_multiple_of(2) {
        inv_twiddle_ifft_avx2_fma(h, re, im, *as_arr::<2, f64>(&omg[pos..]));
        pos += 2;
    }

    pos
}

#[target_feature(enable = "avx2,fma")]
fn inv_twiddle_ifft_avx2_fma(h: usize, re: &mut [f64], im: &mut [f64], omg: [f64; 2]) {
    unsafe {
        let omx: __m128d = _mm_load_pd(omg.as_ptr());
        let omra: __m256d = _mm256_set_m128d(omx, omx);
        let omi: __m256d = _mm256_unpackhi_pd(omra, omra);
        let omr: __m256d = _mm256_unpacklo_pd(omra, omra);
        let re_base: *mut f64 = re.as_mut_ptr();
        let im_base: *mut f64 = im.as_mut_ptr();
        let mut r0: *mut f64 = re_base;
        let mut r1: *mut f64 = re_base.add(h);
        let mut i0: *mut f64 = im_base;
        let mut i1: *mut f64 = im_base.add(h);
        for _ in (0..h).step_by(4) {
            let mut ur0: __m256d = _mm256_loadu_pd(r0);
            let mut ur1: __m256d = _mm256_loadu_pd(r1);
            let mut ui0: __m256d = _mm256_loadu_pd(i0);
            let mut ui1: __m256d = _mm256_loadu_pd(i1);
            let tra = _mm256_sub_pd(ur0, ur1);
            let tia = _mm256_sub_pd(ui0, ui1);
            ur0 = _mm256_add_pd(ur0, ur1);
            ui0 = _mm256_add_pd(ui0, ui1);
            ur1 = _mm256_mul_pd(omi, tia);
            ui1 = _mm256_mul_pd(omi, tra);
            ur1 = _mm256_fmsub_pd(omr, tra, ur1);
            ui1 = _mm256_fmadd_pd(omr, tia, ui1);
            _mm256_storeu_pd(r0, ur0);
            _mm256_storeu_pd(r1, ur1);
            _mm256_storeu_pd(i0, ui0);
            _mm256_storeu_pd(i1, ui1);

            r0 = r0.add(4);
            r1 = r1.add(4);
            i0 = i0.add(4);
            i1 = i1.add(4);
        }
    }
}

#[target_feature(enable = "avx2,fma")]
fn inv_bitwiddle_ifft_avx2_fma(h: usize, re: &mut [f64], im: &mut [f64], omg: &[f64; 4]) {
    unsafe {
        let re_base: *mut f64 = re.as_mut_ptr();
        let im_base: *mut f64 = im.as_mut_ptr();
        let mut r0: *mut f64 = re_base;
        let mut r1: *mut f64 = re_base.add(h);
        let mut r2: *mut f64 = re_base.add(2 * h);
        let mut r3: *mut f64 = re_base.add(3 * h);
        let mut i0: *mut f64 = im_base;
        let mut i1: *mut f64 = im_base.add(h);
        let mut i2: *mut f64 = im_base.add(2 * h);
        let mut i3: *mut f64 = im_base.add(3 * h);
        let om0: __m256d = _mm256_loadu_pd(omg.as_ptr());
        let omb: __m256d = _mm256_permute2f128_pd(om0, om0, 0x11);
        let oma: __m256d = _mm256_permute2f128_pd(om0, om0, 0x00);
        let omai: __m256d = _mm256_unpackhi_pd(oma, oma);
        let omar: __m256d = _mm256_unpacklo_pd(oma, oma);
        let ombi: __m256d = _mm256_unpackhi_pd(omb, omb);
        let ombr: __m256d = _mm256_unpacklo_pd(omb, omb);
        for _ in (0..h).step_by(4) {
            let mut ur0: __m256d = _mm256_loadu_pd(r0);
            let mut ur1: __m256d = _mm256_loadu_pd(r1);
            let mut ur2: __m256d = _mm256_loadu_pd(r2);
            let mut ur3: __m256d = _mm256_loadu_pd(r3);
            let mut ui0: __m256d = _mm256_loadu_pd(i0);
            let mut ui1: __m256d = _mm256_loadu_pd(i1);
            let mut ui2: __m256d = _mm256_loadu_pd(i2);
            let mut ui3: __m256d = _mm256_loadu_pd(i3);

            let mut tra: __m256d = _mm256_sub_pd(ur0, ur1);
            let mut trb: __m256d = _mm256_sub_pd(ur2, ur3);
            let mut tia: __m256d = _mm256_sub_pd(ui0, ui1);
            let mut tib: __m256d = _mm256_sub_pd(ui2, ui3);
            ur0 = _mm256_add_pd(ur0, ur1);
            ur2 = _mm256_add_pd(ur2, ur3);
            ui0 = _mm256_add_pd(ui0, ui1);
            ui2 = _mm256_add_pd(ui2, ui3);
            ur1 = _mm256_mul_pd(omai, tia);
            ur3 = _mm256_mul_pd(omar, tib);
            ui1 = _mm256_mul_pd(omai, tra);
            ui3 = _mm256_mul_pd(omar, trb);
            ur1 = _mm256_fmsub_pd(omar, tra, ur1);
            ur3 = _mm256_fmadd_pd(omai, trb, ur3);
            ui1 = _mm256_fmadd_pd(omar, tia, ui1);
            ui3 = _mm256_fmsub_pd(omai, tib, ui3);

            tra = _mm256_sub_pd(ur0, ur2);
            trb = _mm256_sub_pd(ur1, ur3);
            tia = _mm256_sub_pd(ui0, ui2);
            tib = _mm256_sub_pd(ui1, ui3);
            ur0 = _mm256_add_pd(ur0, ur2);
            ur1 = _mm256_add_pd(ur1, ur3);
            ui0 = _mm256_add_pd(ui0, ui2);
            ui1 = _mm256_add_pd(ui1, ui3);
            ur2 = _mm256_mul_pd(ombi, tia);
            ur3 = _mm256_mul_pd(ombi, tib);
            ui2 = _mm256_mul_pd(ombi, tra);
            ui3 = _mm256_mul_pd(ombi, trb);
            ur2 = _mm256_fmsub_pd(ombr, tra, ur2);
            ur3 = _mm256_fmsub_pd(ombr, trb, ur3);
            ui2 = _mm256_fmadd_pd(ombr, tia, ui2);
            ui3 = _mm256_fmadd_pd(ombr, tib, ui3);

            _mm256_storeu_pd(r0, ur0);
            _mm256_storeu_pd(r1, ur1);
            _mm256_storeu_pd(r2, ur2);
            _mm256_storeu_pd(r3, ur3);
            _mm256_storeu_pd(i0, ui0);
            _mm256_storeu_pd(i1, ui1);
            _mm256_storeu_pd(i2, ui2);
            _mm256_storeu_pd(i3, ui3);

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

#[test]
fn test_ifft_avx2_fma() {
    use super::*;

    #[target_feature(enable = "avx2,fma")]
    fn internal(log_m: usize) {
        use poulpy_cpu_portable::reference::fft64::reim::ReimIFFTRef;

        let m: usize = 1 << log_m;

        let table: ReimIFFTTable<f64> = ReimIFFTTable::<f64>::new(m);

        let mut values_0: Vec<f64> = vec![0f64; m << 1];
        let scale: f64 = 1.0f64 / m as f64;
        values_0.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as f64 * scale);

        let mut values_1: Vec<f64> = vec![0f64; m << 1];
        values_1.iter_mut().zip(values_0.iter()).for_each(|(y, x)| *y = *x);

        ReimIFFTAvx::reim_dft_execute(&table, &mut values_0);
        ReimIFFTRef::reim_dft_execute(&table, &mut values_1);

        let max_diff: f64 = 1.0 / ((1u64 << (53 - log_m - 1)) as f64);

        for i in 0..m * 2 {
            let diff: f64 = (values_0[i] - values_1[i]).abs();
            assert!(diff <= max_diff, "{} -> {}-{} = {}", i, values_0[i], values_1[i], diff)
        }
    }

    if std::is_x86_feature_detected!("avx2") {
        for log_m in 0..16 {
            unsafe { internal(log_m) }
        }
    } else {
        eprintln!("skipping: CPU lacks avx2");
    }
}
