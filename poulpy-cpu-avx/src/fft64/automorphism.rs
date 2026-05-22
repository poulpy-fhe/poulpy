//! AVX2-accelerated DFT-domain automorphism for the FFT64 layout.
//!
//! Each output complex slot reads from a permuted source slot, so the hot
//! loop is two parallel gathers (real and imaginary halves), with an
//! optional global sign flip on the imaginary half when the plan's
//! `conj` flag is set. The `conj` branch is hoisted outside the inner
//! loop so each arm is a straight-line gather/gather/store sequence.

use core::arch::x86_64::{
    __m128i, __m256d, __m256i, _mm_loadu_si128, _mm256_cvtepu32_epi64, _mm256_i64gather_pd, _mm256_set1_pd, _mm256_storeu_pd,
    _mm256_xor_pd,
};

use poulpy_cpu_ref::reference::fft64::vec_znx_dft::{Fft64AutomorphismPlan, vec_znx_dft_automorphism as fft64_automorphism_ref};
use poulpy_hal::layouts::{Backend, HostDataMut, HostDataRef, VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView, ZnxViewMut};

/// AVX2 entry point for [`Fft64AutomorphismPlan`].
///
/// Falls back to the scalar reference when `m < 4` (below the SIMD width)
/// or when the residual after the SIMD block is non-empty (handled by an
/// inner scalar tail).
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 + FMA (verified by the
/// module-handle constructor for `FFT64Avx`).
pub fn fft64_vec_znx_dft_automorphism_avx<BE>(
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

    // Below the SIMD width the gather setup costs more than it saves;
    // defer to the scalar reference. Also covers degenerate `m == 0`.
    if m < 4 {
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

    // Zero trailing limbs not reached by the source.
    for limb in min_size..res_size {
        let slice = res.at_mut(res_col, limb);
        for v in slice.iter_mut() {
            *v = 0.0;
        }
    }
}

/// AVX2 hot loop, no global conjugation. Two parallel gathers (re, im)
/// indexed by 4 u32 entries of `perm`, two unaligned stores.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn automorphism_no_conj_inner(m: usize, perm: &[u32], a_re: &[f64], a_im: &[f64], res_re: &mut [f64], res_im: &mut [f64]) {
    unsafe {
        let blocks = m & !3; // largest multiple of 4 ≤ m
        let mut i: usize = 0;
        while i < blocks {
            // Load 4 × u32 indices and widen to 4 × i64 lanes.
            let idx32: __m128i = _mm_loadu_si128(perm.as_ptr().add(i) as *const __m128i);
            let idx64: __m256i = _mm256_cvtepu32_epi64(idx32);

            let re_v: __m256d = _mm256_i64gather_pd::<8>(a_re.as_ptr(), idx64);
            let im_v: __m256d = _mm256_i64gather_pd::<8>(a_im.as_ptr(), idx64);

            _mm256_storeu_pd(res_re.as_mut_ptr().add(i), re_v);
            _mm256_storeu_pd(res_im.as_mut_ptr().add(i), im_v);

            i += 4;
        }
        // Scalar tail for the last `m & 3` slots.
        while i < m {
            let s = *perm.get_unchecked(i) as usize;
            *res_re.get_unchecked_mut(i) = *a_re.get_unchecked(s);
            *res_im.get_unchecked_mut(i) = *a_im.get_unchecked(s);
            i += 1;
        }
    }
}

/// AVX2 hot loop with global imaginary-half negation. Same gather
/// pattern, plus one `_mm256_xor_pd` against the sign-bit broadcast.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn automorphism_conj_inner(m: usize, perm: &[u32], a_re: &[f64], a_im: &[f64], res_re: &mut [f64], res_im: &mut [f64]) {
    unsafe {
        let sign_bit: __m256d = _mm256_set1_pd(-0.0);
        let blocks = m & !3;
        let mut i: usize = 0;
        while i < blocks {
            let idx32: __m128i = _mm_loadu_si128(perm.as_ptr().add(i) as *const __m128i);
            let idx64: __m256i = _mm256_cvtepu32_epi64(idx32);

            let re_v: __m256d = _mm256_i64gather_pd::<8>(a_re.as_ptr(), idx64);
            let im_v: __m256d = _mm256_i64gather_pd::<8>(a_im.as_ptr(), idx64);
            let im_neg: __m256d = _mm256_xor_pd(im_v, sign_bit);

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
