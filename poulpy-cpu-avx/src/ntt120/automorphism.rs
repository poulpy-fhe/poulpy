//! AVX2-accelerated DFT-domain automorphism for the NTT120 layout.
//!
//! Each output slot is a contiguous 32-byte block (4 × u64 CRT residues),
//! so a single `_mm256_loadu_si256` / `_mm256_storeu_si256` pair copies
//! one slot per iteration. There is no sign / conjugation logic — the
//! full-spectrum NTT layout makes the action a pure permutation.

use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};

use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_cpu_ref::reference::ntt120::{
    types::Q120bScalar,
    vec_znx_dft::{NttAutomorphismPlan, ntt120_vec_znx_dft_automorphism as ntt120_automorphism_ref},
    NttZero,
};
use poulpy_hal::layouts::{Backend, HostDataMut, HostDataRef, VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView, ZnxViewMut};

/// AVX2 entry point for [`NttAutomorphismPlan`].
///
/// For the lazy q120b layout (4 × u64 per slot), one slot fits exactly
/// in an `__m256i`, so the hot loop is one indexed load and one
/// sequential store per iteration. Falls back to the scalar reference if
/// `n == 0`.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 (verified at module
/// construction for `NTT120Avx`).
pub fn ntt120_vec_znx_dft_automorphism_avx<BE>(
    plan: &NttAutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttZero,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(plan.perm.len(), res.n());
    }

    let n: usize = res.n();
    if n == 0 {
        ntt120_automorphism_ref::<BE>(plan, res, res_col, a, a_col);
        return;
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let min_size: usize = res_size.min(a_size);
    let perm: &[u32] = &plan.perm;

    for limb in 0..min_size {
        let a_slice: &[u64] = cast_slice(a.at(a_col, limb));
        let res_slice: &mut [u64] = cast_slice_mut(res.at_mut(res_col, limb));
        unsafe {
            automorphism_inner(n, perm, a_slice, res_slice);
        }
    }

    for limb in min_size..res_size {
        BE::ntt_zero(cast_slice_mut(res.at_mut(res_col, limb)));
    }
}

/// AVX2 hot loop: one 256-bit load from `a[4 * perm[i]..]`, one 256-bit
/// store to `res[4 * i..]`. No arithmetic.
#[target_feature(enable = "avx2")]
unsafe fn automorphism_inner(n: usize, perm: &[u32], a: &[u64], res: &mut [u64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let res_ptr = res.as_mut_ptr() as *mut __m256i;

        for i in 0..n {
            let s = *perm.get_unchecked(i) as usize;
            // Each "slot" is one __m256i (4 × u64 = 32 bytes).
            let v = _mm256_loadu_si256(a_ptr.add(s));
            _mm256_storeu_si256(res_ptr.add(i), v);
        }
    }
}
