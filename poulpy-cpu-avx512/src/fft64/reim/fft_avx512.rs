// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code adapted from the AVX2 / FMA C kernels of the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The 256-bit AVX2 originals were widened to 512-bit AVX-512 and translated
// to Rust intrinsics; algorithmic structure is preserved one-to-one with the
// spqlios sources to keep semantics identical.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

use std::arch::x86_64::{
    __m128d, __m256d, __m512d, __m512i, _mm_loadu_pd, _mm256_loadu_pd, _mm256_set_m128d, _mm256_storeu_pd, _mm512_add_pd,
    _mm512_castpd256_pd512, _mm512_castpd512_pd256, _mm512_extractf64x4_pd, _mm512_fmadd_pd, _mm512_fmsub_pd, _mm512_insertf64x4,
    _mm512_loadu_pd, _mm512_mul_pd, _mm512_permutex2var_pd, _mm512_set_epi64, _mm512_set1_pd, _mm512_shuffle_pd,
    _mm512_storeu_pd, _mm512_sub_pd, _mm512_unpackhi_pd, _mm512_unpacklo_pd,
};

use crate::fft64::reim::as_arr;

const FFT_RECURSION_CUTOFF: usize = 2048;

#[target_feature(enable = "avx512f")]
pub(crate) fn fft_avx512(m: usize, omg: &[f64], data: &mut [f64]) {
    // m <= 16 falls through to the reference implementation: it is too small
    // for the AVX-512 base case (`fft16x2_avx512` processes 2 blocks per call,
    // so it needs m >= 32). For m == 32 and above, the BFS dispatcher always
    // produces an even number of FFT16 blocks and uses `fft16x2_avx512`.
    if m <= 16 {
        use poulpy_cpu_ref::reference::fft64::reim::fft_ref;

        fft_ref(m, omg, data);
        return;
    }

    assert!(data.len() == 2 * m);
    let (re, im) = data.split_at_mut(m);

    if m <= FFT_RECURSION_CUTOFF {
        fft_bfs_16_avx512(m, re, im, omg, 0);
    } else {
        fft_rec_16_avx512(m, re, im, omg, 0);
    }
}

#[target_feature(enable = "avx512f")]
fn fft_rec_16_avx512(m: usize, re: &mut [f64], im: &mut [f64], omg: &[f64], mut pos: usize) -> usize {
    if m <= FFT_RECURSION_CUTOFF {
        return fft_bfs_16_avx512(m, re, im, omg, pos);
    };

    let h: usize = m >> 1;
    twiddle_fft_avx512(h, re, im, *as_arr::<2, f64>(&omg[pos..]));
    pos += 2;
    pos = fft_rec_16_avx512(h, re, im, omg, pos);
    pos = fft_rec_16_avx512(h, &mut re[h..], &mut im[h..], omg, pos);
    pos
}

#[target_feature(enable = "avx512f")]
fn fft_bfs_16_avx512(m: usize, re: &mut [f64], im: &mut [f64], omg: &[f64], mut pos: usize) -> usize {
    let log_m: usize = (usize::BITS - (m - 1).leading_zeros()) as usize;
    let mut mm: usize = m;

    if !log_m.is_multiple_of(2) {
        let h: usize = mm >> 1;
        twiddle_fft_avx512(h, re, im, *as_arr::<2, f64>(&omg[pos..]));
        pos += 2;
        mm = h
    }

    while mm > 16 {
        let h: usize = mm >> 2;
        for off in (0..m).step_by(mm) {
            bitwiddle_fft_avx512(h, &mut re[off..], &mut im[off..], as_arr::<4, f64>(&omg[pos..]));

            pos += 4;
        }
        mm = h
    }

    // After the higher-level stages, m is always a multiple of 32 (it's the
    // smallest BFS input that reaches this point: power-of-2 m >= 32 → m/16 is
    // a power of 2 >= 2, hence even). The fft16x2 base case processes 2 blocks
    // per call.
    let mut off = 0;
    while off + 32 <= m {
        unsafe {
            fft16x2_avx512(&mut re[off..off + 32], &mut im[off..off + 32], &omg[pos..pos + 32]);
        }
        pos += 32;
        off += 32;
    }

    pos
}

#[target_feature(enable = "avx512f")]
fn twiddle_fft_avx512(h: usize, re: &mut [f64], im: &mut [f64], omg: [f64; 2]) {
    unsafe {
        let omr: __m512d = _mm512_set1_pd(omg[0]);
        let omi: __m512d = _mm512_set1_pd(omg[1]);
        let mut r0: *mut f64 = re.as_mut_ptr();
        let mut r1: *mut f64 = re.as_mut_ptr().add(h);
        let mut i0: *mut f64 = im.as_mut_ptr();
        let mut i1: *mut f64 = im.as_mut_ptr().add(h);

        for _ in (0..h).step_by(8) {
            let mut ur0: __m512d = _mm512_loadu_pd(r0);
            let mut ur1: __m512d = _mm512_loadu_pd(r1);
            let mut ui0: __m512d = _mm512_loadu_pd(i0);
            let mut ui1: __m512d = _mm512_loadu_pd(i1);
            let mut tra: __m512d = _mm512_mul_pd(omi, ui1);
            let mut tia: __m512d = _mm512_mul_pd(omi, ur1);

            tra = _mm512_fmsub_pd(omr, ur1, tra);
            tia = _mm512_fmadd_pd(omr, ui1, tia);
            ur1 = _mm512_sub_pd(ur0, tra);
            ui1 = _mm512_sub_pd(ui0, tia);
            ur0 = _mm512_add_pd(ur0, tra);
            ui0 = _mm512_add_pd(ui0, tia);

            _mm512_storeu_pd(r0, ur0);
            _mm512_storeu_pd(r1, ur1);
            _mm512_storeu_pd(i0, ui0);
            _mm512_storeu_pd(i1, ui1);

            r0 = r0.add(8);
            r1 = r1.add(8);
            i0 = i0.add(8);
            i1 = i1.add(8);
        }
    }
}

#[target_feature(enable = "avx512f")]
fn bitwiddle_fft_avx512(h: usize, re: &mut [f64], im: &mut [f64], omg: &[f64; 4]) {
    unsafe {
        let mut r0: *mut f64 = re.as_mut_ptr();
        let mut r1: *mut f64 = re.as_mut_ptr().add(h);
        let mut r2: *mut f64 = re.as_mut_ptr().add(2 * h);
        let mut r3: *mut f64 = re.as_mut_ptr().add(3 * h);
        let mut i0: *mut f64 = im.as_mut_ptr();
        let mut i1: *mut f64 = im.as_mut_ptr().add(h);
        let mut i2: *mut f64 = im.as_mut_ptr().add(2 * h);
        let mut i3: *mut f64 = im.as_mut_ptr().add(3 * h);
        // omg[0..2] = oma (real, imag), omg[2..4] = omb (real, imag)
        let omar: __m512d = _mm512_set1_pd(omg[0]);
        let omai: __m512d = _mm512_set1_pd(omg[1]);
        let ombr: __m512d = _mm512_set1_pd(omg[2]);
        let ombi: __m512d = _mm512_set1_pd(omg[3]);
        for _ in (0..h).step_by(8) {
            let mut ur0: __m512d = _mm512_loadu_pd(r0);
            let mut ur1: __m512d = _mm512_loadu_pd(r1);
            let mut ur2: __m512d = _mm512_loadu_pd(r2);
            let mut ur3: __m512d = _mm512_loadu_pd(r3);
            let mut ui0: __m512d = _mm512_loadu_pd(i0);
            let mut ui1: __m512d = _mm512_loadu_pd(i1);
            let mut ui2: __m512d = _mm512_loadu_pd(i2);
            let mut ui3: __m512d = _mm512_loadu_pd(i3);

            let mut tra: __m512d = _mm512_mul_pd(omai, ui2);
            let mut trb: __m512d = _mm512_mul_pd(omai, ui3);
            let mut tia: __m512d = _mm512_mul_pd(omai, ur2);
            let mut tib: __m512d = _mm512_mul_pd(omai, ur3);
            tra = _mm512_fmsub_pd(omar, ur2, tra);
            trb = _mm512_fmsub_pd(omar, ur3, trb);
            tia = _mm512_fmadd_pd(omar, ui2, tia);
            tib = _mm512_fmadd_pd(omar, ui3, tib);
            ur2 = _mm512_sub_pd(ur0, tra);
            ur3 = _mm512_sub_pd(ur1, trb);
            ui2 = _mm512_sub_pd(ui0, tia);
            ui3 = _mm512_sub_pd(ui1, tib);
            ur0 = _mm512_add_pd(ur0, tra);
            ur1 = _mm512_add_pd(ur1, trb);
            ui0 = _mm512_add_pd(ui0, tia);
            ui1 = _mm512_add_pd(ui1, tib);

            tra = _mm512_mul_pd(ombi, ui1);
            trb = _mm512_mul_pd(ombr, ui3);
            tia = _mm512_mul_pd(ombi, ur1);
            tib = _mm512_mul_pd(ombr, ur3);
            tra = _mm512_fmsub_pd(ombr, ur1, tra);
            trb = _mm512_fmadd_pd(ombi, ur3, trb);
            tia = _mm512_fmadd_pd(ombr, ui1, tia);
            tib = _mm512_fmsub_pd(ombi, ui3, tib);
            ur1 = _mm512_sub_pd(ur0, tra);
            ur3 = _mm512_add_pd(ur2, trb);
            ui1 = _mm512_sub_pd(ui0, tia);
            ui3 = _mm512_add_pd(ui2, tib);
            ur0 = _mm512_add_pd(ur0, tra);
            ur2 = _mm512_sub_pd(ur2, trb);
            ui0 = _mm512_add_pd(ui0, tia);
            ui2 = _mm512_sub_pd(ui2, tib);

            _mm512_storeu_pd(r0, ur0);
            _mm512_storeu_pd(r1, ur1);
            _mm512_storeu_pd(r2, ur2);
            _mm512_storeu_pd(r3, ur3);
            _mm512_storeu_pd(i0, ui0);
            _mm512_storeu_pd(i1, ui1);
            _mm512_storeu_pd(i2, ui2);
            _mm512_storeu_pd(i3, ui3);

            r0 = r0.add(8);
            r1 = r1.add(8);
            r2 = r2.add(8);
            r3 = r3.add(8);
            i0 = i0.add(8);
            i1 = i1.add(8);
            i2 = i2.add(8);
            i3 = i3.add(8);
        }
    }
}

/// Process two consecutive FFT16 blocks in parallel using __m512d (AVX-512F).
///
/// `re`, `im` must each be at least 32 doubles long. `omg` must be at least 32 doubles long.
/// Block A is at offsets `[0..16]`, block B is at offsets `[16..32]`.
#[target_feature(enable = "avx512f")]
unsafe fn fft16x2_avx512(re: &mut [f64], im: &mut [f64], omg: &[f64]) {
    #[inline(always)]
    unsafe fn load_pair(p: *const f64, off: usize) -> __m512d {
        unsafe {
            let a: __m256d = _mm256_loadu_pd(p.add(off));
            let b: __m256d = _mm256_loadu_pd(p.add(off + 16));
            _mm512_insertf64x4::<1>(_mm512_castpd256_pd512(a), b)
        }
    }

    #[inline(always)]
    unsafe fn store_pair(p: *mut f64, off: usize, v: __m512d) {
        unsafe {
            let lo: __m256d = _mm512_castpd512_pd256(v);
            let hi: __m256d = _mm512_extractf64x4_pd::<1>(v);
            _mm256_storeu_pd(p.add(off), lo);
            _mm256_storeu_pd(p.add(off + 16), hi);
        }
    }

    #[inline(always)]
    unsafe fn load_narrow_twiddle(omg_ptr: *const f64, off_a: usize, off_b: usize) -> (__m512d, __m512d) {
        unsafe {
            let omx_a: __m128d = _mm_loadu_pd(omg_ptr.add(off_a));
            let omx_b: __m128d = _mm_loadu_pd(omg_ptr.add(off_b));
            let om_a256: __m256d = _mm256_set_m128d(omx_a, omx_a);
            let om_b256: __m256d = _mm256_set_m128d(omx_b, omx_b);
            let om_riri: __m512d = _mm512_insertf64x4::<1>(_mm512_castpd256_pd512(om_a256), om_b256);
            let omi: __m512d = _mm512_unpackhi_pd(om_riri, om_riri);
            let omr: __m512d = _mm512_unpacklo_pd(om_riri, om_riri);
            (omr, omi)
        }
    }

    #[inline(always)]
    unsafe fn load_wide_twiddle(omg_ptr: *const f64, off_a: usize, off_b: usize) -> (__m512d, __m512d) {
        unsafe {
            let om_a: __m256d = _mm256_loadu_pd(omg_ptr.add(off_a));
            let om_b: __m256d = _mm256_loadu_pd(omg_ptr.add(off_b));
            let om_full: __m512d = _mm512_insertf64x4::<1>(_mm512_castpd256_pd512(om_a), om_b);
            let omi: __m512d = _mm512_shuffle_pd::<0xFF>(om_full, om_full);
            let omr: __m512d = _mm512_shuffle_pd::<0x00>(om_full, om_full);
            (omr, omi)
        }
    }

    unsafe {
        let re_ptr: *mut f64 = re.as_mut_ptr();
        let im_ptr: *mut f64 = im.as_mut_ptr();
        let omg_ptr: *const f64 = omg.as_ptr();

        // Permute selectors for cross-half shuffles.
        let perm_high: __m512i = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
        let perm_low: __m512i = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);

        // Load two FFT16 blocks.
        let mut ra0: __m512d = load_pair(re_ptr, 0);
        let mut ra1: __m512d = load_pair(re_ptr, 4);
        let mut ra2: __m512d = load_pair(re_ptr, 8);
        let mut ra3: __m512d = load_pair(re_ptr, 12);
        let mut ia0: __m512d = load_pair(im_ptr, 0);
        let mut ia1: __m512d = load_pair(im_ptr, 4);
        let mut ia2: __m512d = load_pair(im_ptr, 8);
        let mut ia3: __m512d = load_pair(im_ptr, 12);

        // Stage 1.
        let (s1_omr, s1_omi) = load_narrow_twiddle(omg_ptr, 0, 16);
        let mut t8: __m512d = _mm512_mul_pd(ia2, s1_omi);
        let mut t9: __m512d = _mm512_mul_pd(ia3, s1_omi);
        let mut t10: __m512d = _mm512_mul_pd(ra2, s1_omi);
        let mut t11: __m512d = _mm512_mul_pd(ra3, s1_omi);
        t8 = _mm512_fmsub_pd(ra2, s1_omr, t8);
        t9 = _mm512_fmsub_pd(ra3, s1_omr, t9);
        t10 = _mm512_fmadd_pd(ia2, s1_omr, t10);
        t11 = _mm512_fmadd_pd(ia3, s1_omr, t11);
        ra2 = _mm512_sub_pd(ra0, t8);
        ra3 = _mm512_sub_pd(ra1, t9);
        ia2 = _mm512_sub_pd(ia0, t10);
        ia3 = _mm512_sub_pd(ia1, t11);
        ra0 = _mm512_add_pd(ra0, t8);
        ra1 = _mm512_add_pd(ra1, t9);
        ia0 = _mm512_add_pd(ia0, t10);
        ia1 = _mm512_add_pd(ia1, t11);

        // Stage 2.
        let (s2_omr, s2_omi) = load_narrow_twiddle(omg_ptr, 2, 18);
        t8 = _mm512_mul_pd(ia1, s2_omi);
        t9 = _mm512_mul_pd(ia3, s2_omr);
        t10 = _mm512_mul_pd(ra1, s2_omi);
        t11 = _mm512_mul_pd(ra3, s2_omr);
        t8 = _mm512_fmsub_pd(ra1, s2_omr, t8);
        t9 = _mm512_fmadd_pd(ra3, s2_omi, t9);
        t10 = _mm512_fmadd_pd(ia1, s2_omr, t10);
        t11 = _mm512_fmsub_pd(ia3, s2_omi, t11);
        ra1 = _mm512_sub_pd(ra0, t8);
        ra3 = _mm512_add_pd(ra2, t9);
        ia1 = _mm512_sub_pd(ia0, t10);
        ia3 = _mm512_add_pd(ia2, t11);
        ra0 = _mm512_add_pd(ra0, t8);
        ra2 = _mm512_sub_pd(ra2, t9);
        ia0 = _mm512_add_pd(ia0, t10);
        ia2 = _mm512_sub_pd(ia2, t11);

        // Stage 3.
        let (s3_omr, s3_omi) = load_wide_twiddle(omg_ptr, 4, 20);

        let mut t8: __m512d = _mm512_permutex2var_pd(ra0, perm_high, ra2);
        let mut t9: __m512d = _mm512_permutex2var_pd(ra1, perm_high, ra3);
        let mut t10: __m512d = _mm512_permutex2var_pd(ia0, perm_high, ia2);
        let mut t11: __m512d = _mm512_permutex2var_pd(ia1, perm_high, ia3);
        let new_ra0: __m512d = _mm512_permutex2var_pd(ra0, perm_low, ra2);
        let new_ra1: __m512d = _mm512_permutex2var_pd(ra1, perm_low, ra3);
        let new_ra2: __m512d = _mm512_permutex2var_pd(ia0, perm_low, ia2);
        let new_ra3: __m512d = _mm512_permutex2var_pd(ia1, perm_low, ia3);
        ra0 = new_ra0;
        ra1 = new_ra1;
        ra2 = new_ra2;
        ra3 = new_ra3;

        ia0 = _mm512_mul_pd(t10, s3_omi);
        ia1 = _mm512_mul_pd(t11, s3_omr);
        ia2 = _mm512_mul_pd(t8, s3_omi);
        ia3 = _mm512_mul_pd(t9, s3_omr);
        ia0 = _mm512_fmsub_pd(t8, s3_omr, ia0);
        ia1 = _mm512_fmadd_pd(t9, s3_omi, ia1);
        ia2 = _mm512_fmadd_pd(t10, s3_omr, ia2);
        ia3 = _mm512_fmsub_pd(t11, s3_omi, ia3);
        t8 = _mm512_sub_pd(ra0, ia0);
        t9 = _mm512_add_pd(ra1, ia1);
        t10 = _mm512_sub_pd(ra2, ia2);
        t11 = _mm512_add_pd(ra3, ia3);
        ra0 = _mm512_add_pd(ra0, ia0);
        ra1 = _mm512_sub_pd(ra1, ia1);
        ra2 = _mm512_add_pd(ra2, ia2);
        ra3 = _mm512_sub_pd(ra3, ia3);

        // Stage 4.
        let s4_omr: __m512d = {
            let a: __m256d = _mm256_loadu_pd(omg_ptr.add(8));
            let b: __m256d = _mm256_loadu_pd(omg_ptr.add(24));
            _mm512_insertf64x4::<1>(_mm512_castpd256_pd512(a), b)
        };
        let s4_omi: __m512d = {
            let a: __m256d = _mm256_loadu_pd(omg_ptr.add(12));
            let b: __m256d = _mm256_loadu_pd(omg_ptr.add(28));
            _mm512_insertf64x4::<1>(_mm512_castpd256_pd512(a), b)
        };

        let new_ia0: __m512d = _mm512_unpackhi_pd(ra0, ra1);
        let new_ia2: __m512d = _mm512_unpackhi_pd(ra2, ra3);
        let new_ia1: __m512d = _mm512_unpackhi_pd(t8, t9);
        let new_ia3: __m512d = _mm512_unpackhi_pd(t10, t11);
        let new_ra0: __m512d = _mm512_unpacklo_pd(ra0, ra1);
        let new_ra2: __m512d = _mm512_unpacklo_pd(ra2, ra3);
        let new_ra1: __m512d = _mm512_unpacklo_pd(t8, t9);
        let new_ra3: __m512d = _mm512_unpacklo_pd(t10, t11);
        ia0 = new_ia0;
        ia2 = new_ia2;
        ia1 = new_ia1;
        ia3 = new_ia3;
        ra0 = new_ra0;
        ra2 = new_ra2;
        ra1 = new_ra1;
        ra3 = new_ra3;

        t8 = _mm512_mul_pd(ia2, s4_omi);
        t9 = _mm512_mul_pd(ia3, s4_omr);
        t10 = _mm512_mul_pd(ia0, s4_omi);
        t11 = _mm512_mul_pd(ia1, s4_omr);
        t8 = _mm512_fmsub_pd(ia0, s4_omr, t8);
        t9 = _mm512_fmadd_pd(ia1, s4_omi, t9);
        t10 = _mm512_fmadd_pd(ia2, s4_omr, t10);
        t11 = _mm512_fmsub_pd(ia3, s4_omi, t11);
        ia0 = _mm512_sub_pd(ra0, t8);
        ia1 = _mm512_add_pd(ra1, t9);
        ia2 = _mm512_sub_pd(ra2, t10);
        ia3 = _mm512_add_pd(ra3, t11);
        ra0 = _mm512_add_pd(ra0, t8);
        ra1 = _mm512_sub_pd(ra1, t9);
        ra2 = _mm512_add_pd(ra2, t10);
        ra3 = _mm512_sub_pd(ra3, t11);

        // Final shuffle.
        t11 = _mm512_unpackhi_pd(ra3, ia3);
        t9 = _mm512_unpackhi_pd(ra1, ia1);
        t10 = _mm512_unpacklo_pd(ra3, ia3);
        t8 = _mm512_unpacklo_pd(ra1, ia1);
        let new_ra3: __m512d = _mm512_unpackhi_pd(ra2, ia2);
        let new_ra1: __m512d = _mm512_unpackhi_pd(ra0, ia0);
        let new_ra2: __m512d = _mm512_unpacklo_pd(ra2, ia2);
        let new_ra0: __m512d = _mm512_unpacklo_pd(ra0, ia0);
        ra3 = new_ra3;
        ra1 = new_ra1;
        ra2 = new_ra2;
        ra0 = new_ra0;

        let new_ia2: __m512d = _mm512_permutex2var_pd(ra2, perm_high, t10);
        let new_ia3: __m512d = _mm512_permutex2var_pd(ra3, perm_high, t11);
        let new_ia0: __m512d = _mm512_permutex2var_pd(ra2, perm_low, t10);
        let new_ia1: __m512d = _mm512_permutex2var_pd(ra3, perm_low, t11);
        let final_ra2: __m512d = _mm512_permutex2var_pd(ra0, perm_high, t8);
        let final_ra3: __m512d = _mm512_permutex2var_pd(ra1, perm_high, t9);
        let final_ra0: __m512d = _mm512_permutex2var_pd(ra0, perm_low, t8);
        let final_ra1: __m512d = _mm512_permutex2var_pd(ra1, perm_low, t9);
        ia2 = new_ia2;
        ia3 = new_ia3;
        ia0 = new_ia0;
        ia1 = new_ia1;
        ra2 = final_ra2;
        ra3 = final_ra3;
        ra0 = final_ra0;
        ra1 = final_ra1;

        // Store the two blocks back.
        store_pair(re_ptr, 0, ra0);
        store_pair(re_ptr, 4, ra1);
        store_pair(re_ptr, 8, ra2);
        store_pair(re_ptr, 12, ra3);
        store_pair(im_ptr, 0, ia0);
        store_pair(im_ptr, 4, ia1);
        store_pair(im_ptr, 8, ia2);
        store_pair(im_ptr, 12, ia3);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use poulpy_cpu_ref::reference::fft64::reim::{ReimFFTExecute, ReimFFTRef, ReimFFTTable, ReimIFFTRef, ReimIFFTTable};

    use crate::fft64::reim::{ReimFFTAvx512, ReimIFFTAvx512};

    /// AVX-512 FFT → IFFT round-trip matches reference FFT → IFFT (same residual error).
    #[test]
    fn fft_ifft_roundtrip_avx512() {
        let m = 64usize;
        let fwd = ReimFFTTable::<f64>::new(m);
        let inv = ReimIFFTTable::<f64>::new(m);

        let data: Vec<f64> = (0..2 * m).map(|i| (i + 1) as f64 / m as f64).collect();

        let mut ifma = data.clone();
        ReimFFTAvx512::reim_dft_execute(&fwd, &mut ifma);
        ReimIFFTAvx512::reim_dft_execute(&inv, &mut ifma);

        let mut reference = data.clone();
        ReimFFTRef::reim_dft_execute(&fwd, &mut reference);
        ReimIFFTRef::reim_dft_execute(&inv, &mut reference);

        let tol = 1e-10f64;
        for i in 0..2 * m {
            let diff = (ifma[i] - reference[i]).abs();
            assert!(
                diff <= tol,
                "idx={i}: AVX-512 round-trip={} ref round-trip={} diff={diff}",
                ifma[i],
                reference[i]
            );
        }
    }

    /// AVX-512 FFT-based convolution matches reference convolution.
    ///
    /// a = [1, 2, 0, …] (real), b = [3, 4, 0, …] (real).
    /// Multiplies in frequency domain (complex pointwise) and IFFTs.
    #[test]
    fn fft_convolution_avx512() {
        let m = 32usize;
        let fwd = ReimFFTTable::<f64>::new(m);
        let inv = ReimIFFTTable::<f64>::new(m);

        // Build reim input: real[0..m] then imag[0..m], purely real poly [1,2,0...]
        let mut a_avx512 = vec![0f64; 2 * m];
        a_avx512[0] = 1.0;
        a_avx512[1] = 2.0;
        let mut b_avx512 = vec![0f64; 2 * m];
        b_avx512[0] = 3.0;
        b_avx512[1] = 4.0;

        let a_ref = a_avx512.clone();
        let b_ref = b_avx512.clone();

        ReimFFTAvx512::reim_dft_execute(&fwd, &mut a_avx512);
        ReimFFTAvx512::reim_dft_execute(&fwd, &mut b_avx512);

        let mut a_ref2 = a_ref;
        let mut b_ref2 = b_ref;
        ReimFFTRef::reim_dft_execute(&fwd, &mut a_ref2);
        ReimFFTRef::reim_dft_execute(&fwd, &mut b_ref2);

        // Pointwise complex multiply: reim format = real[0..m], imag[m..2m]
        let mut c_avx512 = vec![0f64; 2 * m];
        let mut c_ref = vec![0f64; 2 * m];
        for k in 0..m {
            c_avx512[k] = a_avx512[k] * b_avx512[k] - a_avx512[k + m] * b_avx512[k + m];
            c_avx512[k + m] = a_avx512[k] * b_avx512[k + m] + a_avx512[k + m] * b_avx512[k];
            c_ref[k] = a_ref2[k] * b_ref2[k] - a_ref2[k + m] * b_ref2[k + m];
            c_ref[k + m] = a_ref2[k] * b_ref2[k + m] + a_ref2[k + m] * b_ref2[k];
        }

        ReimIFFTAvx512::reim_dft_execute(&inv, &mut c_avx512);
        ReimIFFTRef::reim_dft_execute(&inv, &mut c_ref);

        let tol = 1e-8f64;
        for i in 0..2 * m {
            let diff = (c_avx512[i] - c_ref[i]).abs();
            assert!(diff <= tol, "idx={i}: AVX-512={} ref={} diff={diff}", c_avx512[i], c_ref[i]);
        }
    }
}

#[test]
fn test_fft_avx512() {
    use super::*;

    #[target_feature(enable = "avx512f")]
    fn internal(log_m: usize) {
        use poulpy_cpu_ref::reference::fft64::reim::ReimFFTRef;

        let m = 1 << log_m;

        let table: ReimFFTTable<f64> = ReimFFTTable::<f64>::new(m);

        let mut values_0: Vec<f64> = vec![0f64; m << 1];
        let scale: f64 = 1.0f64 / m as f64;
        values_0.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as f64 * scale);

        let mut values_1: Vec<f64> = vec![0f64; m << 1];
        values_1.iter_mut().zip(values_0.iter()).for_each(|(y, x)| *y = *x);

        ReimFFTAvx512::reim_dft_execute(&table, &mut values_0);
        ReimFFTRef::reim_dft_execute(&table, &mut values_1);

        let max_diff: f64 = 1.0 / ((1u64 << (53 - log_m - 1)) as f64);

        for i in 0..m * 2 {
            let diff: f64 = (values_0[i] - values_1[i]).abs();
            assert!(diff <= max_diff, "{} -> {}-{} = {}", i, values_0[i], values_1[i], diff)
        }
    }

    for log_m in 0..16 {
        unsafe { internal(log_m) }
    }
}
