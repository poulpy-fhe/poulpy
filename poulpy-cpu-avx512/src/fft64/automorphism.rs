//! AVX-512F-accelerated DFT-domain automorphism for the FFT64 layout.
//!
//! Each iteration of the hot loop gathers 8 complex slots: an 8-wide u32
//! index load, widen to 8 × i64 lanes, two `_mm512_i64gather_pd` (real
//! and imaginary halves), an optional `_mm512_xor_pd` for the global
//! conjugate flag, then two `_mm512_storeu_pd`. Tail slots fall through
//! to AVX2 4-wide (≥4) or scalar (<4).

use core::arch::x86_64::{
    __m128i, __m256d, __m256i, __m512d, __m512i, _mm_loadu_si128, _mm256_cvtepu32_epi64, _mm256_i64gather_pd, _mm256_loadu_si256,
    _mm256_set1_pd, _mm256_storeu_pd, _mm256_xor_pd, _mm512_cvtepu32_epi64, _mm512_i64gather_pd, _mm512_set1_pd,
    _mm512_storeu_pd, _mm512_xor_pd,
};

use poulpy_cpu_ref::reference::fft64::vec_znx_dft::{Fft64AutomorphismPlan, vec_znx_dft_automorphism as fft64_automorphism_ref};
use poulpy_hal::layouts::{Backend, HostDataMut, HostDataRef, VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView, ZnxViewMut};

/// AVX-512F entry point for [`Fft64AutomorphismPlan`].
///
/// Falls back to the scalar reference when `m < 8` (below the SIMD
/// width). The tail (`m & 7`) is handled by an AVX2 4-wide pass plus
/// scalar mop-up.
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F (verified at module
/// construction for `FFT64Avx512`).
pub fn fft64_vec_znx_dft_automorphism_avx512<BE>(
    plan: &Fft64AutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = f64> + poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(plan.perm.len(), res.n() >> 1);
    }

    let m: usize = res.n() >> 1;
    if m < 8 {
        // 8-wide gather has no slots to fill; defer to the scalar path.
        fft64_automorphism_ref(plan, res, res_col, a, a_col);
        return;
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let min_size: usize = res_size.min(a_size);
    let perm: &[u32] = &plan.perm;

    for limb in 0..min_size {
        let (res_re, res_im) = res.at_mut(res_col, limb).split_at_mut(m);
        let a_limb = a.at(a_col, limb);
        let (a_re, a_im) = a_limb.split_at(m);

        unsafe {
            if plan.conj {
                automorphism_conj_inner(m, perm, a_re, a_im, res_re, res_im);
            } else {
                automorphism_no_conj_inner(m, perm, a_re, a_im, res_re, res_im);
            }
        }
    }

    for limb in min_size..res_size {
        let slice = res.at_mut(res_col, limb);
        for v in slice.iter_mut() {
            *v = 0.0;
        }
    }
}

/// AVX-512F hot loop, no global conjugation.
///
/// Tail strategy: 8-wide AVX-512 main loop, 4-wide AVX2 fallback for the
/// `m & 4` slot (if present), scalar mop-up for `m & 3`.
#[target_feature(enable = "avx512f", enable = "avx2", enable = "fma")]
unsafe fn automorphism_no_conj_inner(m: usize, perm: &[u32], a_re: &[f64], a_im: &[f64], res_re: &mut [f64], res_im: &mut [f64]) {
    unsafe {
        let main = m & !7; // largest multiple of 8 ≤ m
        let mut i: usize = 0;
        while i < main {
            // 8 × u32 → 8 × i64 lane indices.
            let idx32: __m256i = _mm256_loadu_si256(perm.as_ptr().add(i) as *const __m256i);
            let idx64: __m512i = _mm512_cvtepu32_epi64(idx32);

            let re_v: __m512d = _mm512_i64gather_pd::<8>(idx64, a_re.as_ptr());
            let im_v: __m512d = _mm512_i64gather_pd::<8>(idx64, a_im.as_ptr());

            _mm512_storeu_pd(res_re.as_mut_ptr().add(i), re_v);
            _mm512_storeu_pd(res_im.as_mut_ptr().add(i), im_v);

            i += 8;
        }
        // 4-wide AVX2 pass for the next group of 4 if present.
        if i + 4 <= m {
            let idx32: __m128i = _mm_loadu_si128(perm.as_ptr().add(i) as *const __m128i);
            let idx64: __m256i = _mm256_cvtepu32_epi64(idx32);
            let re_v: __m256d = _mm256_i64gather_pd::<8>(a_re.as_ptr(), idx64);
            let im_v: __m256d = _mm256_i64gather_pd::<8>(a_im.as_ptr(), idx64);
            _mm256_storeu_pd(res_re.as_mut_ptr().add(i), re_v);
            _mm256_storeu_pd(res_im.as_mut_ptr().add(i), im_v);
            i += 4;
        }
        // Scalar mop-up for `m & 3`.
        while i < m {
            let s = *perm.get_unchecked(i) as usize;
            *res_re.get_unchecked_mut(i) = *a_re.get_unchecked(s);
            *res_im.get_unchecked_mut(i) = *a_im.get_unchecked(s);
            i += 1;
        }
    }
}

/// AVX-512F hot loop with global imaginary-half negation.
#[target_feature(enable = "avx512f", enable = "avx2", enable = "fma")]
unsafe fn automorphism_conj_inner(m: usize, perm: &[u32], a_re: &[f64], a_im: &[f64], res_re: &mut [f64], res_im: &mut [f64]) {
    unsafe {
        let sign_bit_512: __m512d = _mm512_set1_pd(-0.0);
        let sign_bit_256: __m256d = _mm256_set1_pd(-0.0);
        let main = m & !7;
        let mut i: usize = 0;
        while i < main {
            let idx32: __m256i = _mm256_loadu_si256(perm.as_ptr().add(i) as *const __m256i);
            let idx64: __m512i = _mm512_cvtepu32_epi64(idx32);

            let re_v: __m512d = _mm512_i64gather_pd::<8>(idx64, a_re.as_ptr());
            let im_v: __m512d = _mm512_i64gather_pd::<8>(idx64, a_im.as_ptr());
            let im_neg: __m512d = _mm512_xor_pd(im_v, sign_bit_512);

            _mm512_storeu_pd(res_re.as_mut_ptr().add(i), re_v);
            _mm512_storeu_pd(res_im.as_mut_ptr().add(i), im_neg);

            i += 8;
        }
        if i + 4 <= m {
            let idx32: __m128i = _mm_loadu_si128(perm.as_ptr().add(i) as *const __m128i);
            let idx64: __m256i = _mm256_cvtepu32_epi64(idx32);
            let re_v: __m256d = _mm256_i64gather_pd::<8>(a_re.as_ptr(), idx64);
            let im_v: __m256d = _mm256_i64gather_pd::<8>(a_im.as_ptr(), idx64);
            let im_neg: __m256d = _mm256_xor_pd(im_v, sign_bit_256);
            _mm256_storeu_pd(res_re.as_mut_ptr().add(i), re_v);
            _mm256_storeu_pd(res_im.as_mut_ptr().add(i), im_neg);
            i += 4;
        }
        while i < m {
            let s = *perm.get_unchecked(i) as usize;
            *res_re.get_unchecked_mut(i) = *a_re.get_unchecked(s);
            *res_im.get_unchecked_mut(i) = -*a_im.get_unchecked(s);
            i += 1;
        }
    }
}
