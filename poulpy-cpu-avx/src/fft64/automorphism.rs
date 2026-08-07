//! AVX2-accelerated DFT-domain automorphism for the FFT64 layout.
//!
//! Each output complex slot reads from a permuted source slot. Although this
//! is an AVX backend, the FFT64 layout makes the source slots bit-reversed, so
//! the AVX2 gather instructions are slower than a tight scalar indexed-copy
//! loop on the collapse workload. The kernel below therefore keeps the AVX
//! backend-specific entry point, but uses unchecked scalar loads/stores.

use poulpy_cpu_ref::reference::fft64::vec_znx_dft::{Fft64AutomorphismPlan, vec_znx_dft_automorphism as fft64_automorphism_ref};
use poulpy_hal::layouts::{Backend, HostDataMut, HostDataRef, VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView, ZnxViewMut};

/// AVX2 entry point for [`Fft64AutomorphismPlan`].
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 + FMA (verified by the
/// module-handle constructor for `FFT64Avx`).
#[inline(always)]
pub fn fft64_vec_znx_dft_automorphism_avx<BE>(
    plan: &Fft64AutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64> + poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(plan.perm.len(), res.n() >> 1);
    }

    let m: usize = res.n() >> 1;
    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let min_size: usize = res_size.min(a_size);
    let perm: &[u32] = &plan.perm;

    if m == 0 {
        fft64_automorphism_ref::<BE>(plan, res, res_col, a, a_col);
        return;
    }

    for limb in 0..min_size {
        let (res_re, res_im) = res.at_mut(res_col, limb).split_at_mut(m);
        let a_limb = a.at(a_col, limb);
        let (a_re, a_im) = a_limb.split_at(m);

        if plan.conj {
            automorphism_conj_inner(m, perm, a_re, a_im, res_re, res_im);
        } else {
            automorphism_no_conj_inner(m, perm, a_re, a_im, res_re, res_im);
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
