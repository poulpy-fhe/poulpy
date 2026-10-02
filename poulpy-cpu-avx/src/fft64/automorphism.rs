//! AVX2-accelerated DFT-domain automorphism for the FFT64 layout.
//!
//! Each output complex slot reads from a permuted source slot. Although this
//! is an AVX backend, the FFT64 layout makes the source slots bit-reversed, so
//! the AVX2 gather instructions are slower than a tight scalar indexed-copy
//! loop on the collapse workload. The kernel below therefore keeps the AVX
//! backend-specific entry point, but uses unchecked scalar loads/stores.

use poulpy_cpu_ref::reference::fft64::vec_znx_dft::Fft64AutomorphismPlan;

/// One limb of [`Fft64AutomorphismPlan`]: `res = tau_p(a)`.
#[inline(always)]
pub(crate) fn reim_automorphism_avx(plan: &Fft64AutomorphismPlan, res: &mut [f64], a: &[f64]) {
    let m: usize = res.len() >> 1;
    assert_eq!(a.len(), res.len());
    assert_eq!(plan.perm.len(), m);
    let (res_re, res_im) = res.split_at_mut(m);
    let (a_re, a_im) = a.split_at(m);
    if plan.conj {
        automorphism_conj_inner(m, &plan.perm, a_re, a_im, res_re, res_im);
    } else {
        automorphism_no_conj_inner(m, &plan.perm, a_re, a_im, res_re, res_im);
    }
}

/// Hot loop, no global conjugation. The source permutation is already
/// precomputed in `perm`, so the fastest path for FFT64 is scalar indexed
/// loads with contiguous stores.
#[inline(always)]
fn automorphism_no_conj_inner(m: usize, perm: &[u32], a_re: &[f64], a_im: &[f64], res_re: &mut [f64], res_im: &mut [f64]) {
    unsafe {
        let mut i: usize = 0;
        while i < m {
            let s = *perm.get_unchecked(i) as usize;
            *res_re.get_unchecked_mut(i) = *a_re.get_unchecked(s);
            *res_im.get_unchecked_mut(i) = *a_im.get_unchecked(s);
            i += 1;
        }
    }
}

/// Hot loop with global imaginary-half negation.
#[inline(always)]
fn automorphism_conj_inner(m: usize, perm: &[u32], a_re: &[f64], a_im: &[f64], res_re: &mut [f64], res_im: &mut [f64]) {
    unsafe {
        let mut i: usize = 0;
        while i < m {
            let s = *perm.get_unchecked(i) as usize;
            *res_re.get_unchecked_mut(i) = *a_re.get_unchecked(s);
            *res_im.get_unchecked_mut(i) = -*a_im.get_unchecked(s);
            i += 1;
        }
    }
}
