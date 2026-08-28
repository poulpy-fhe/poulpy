use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m256i, _mm256_castsi256_si128, _mm256_cvtepu32_epi64, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_mul_epu32,
    _mm256_storeu_si256,
};
use poulpy_cpu_ref::reference::ntt4x30::{NttDFTExecute, NttFromZnx64, vec_znx_dft::NttModuleHandle};
use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftApply},
    layouts::{
        DataView, DataViewMut, Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendRef,
        VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftReborrowBackendRef, VecZnxDftToBackendMut, ZnxView,
    },
};

use super::{
    NTT4x30Avx,
    arithmetic_avx::{BARRETT_MU, Q_VEC, barrett_reduce},
    vec_znx_dft::{pack_limb_q120, pack_two_q120, packed_limb, packed_limb_mut},
};

#[target_feature(enable = "avx2")]
unsafe fn mul_packed_limb(n: usize, dst: &mut [u32], src: &[u32], factor: &[u32]) {
    unsafe {
        let q = _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i);
        let mu = _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i);
        for pair in 0..n / 2 {
            let off = 8 * pair;
            let x = _mm256_loadu_si256(src.as_ptr().add(off) as *const __m256i);
            let y = _mm256_loadu_si256(factor.as_ptr().add(off) as *const __m256i);
            let x0 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(x));
            let x1 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(x));
            let y0 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(y));
            let y1 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(y));
            let p0 = barrett_reduce(_mm256_mul_epu32(x0, y0), q, mu);
            let p1 = barrett_reduce(_mm256_mul_epu32(x1, y1), q, mu);
            _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, pack_two_q120(p0, p1));
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_packed_limb_assign(n: usize, dst: &mut [u32], factor: &[u32]) {
    unsafe {
        let q = _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i);
        let mu = _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i);
        for pair in 0..n / 2 {
            let off = 8 * pair;
            let x = _mm256_loadu_si256(dst.as_ptr().add(off) as *const __m256i);
            let y = _mm256_loadu_si256(factor.as_ptr().add(off) as *const __m256i);
            let p0 = barrett_reduce(
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm256_castsi256_si128(x)),
                    _mm256_cvtepu32_epi64(_mm256_castsi256_si128(y)),
                ),
                q,
                mu,
            );
            let p1 = barrett_reduce(
                _mm256_mul_epu32(
                    _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(x)),
                    _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(y)),
                ),
                q,
                mu,
            );
            _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, pack_two_q120(p0, p1));
        }
    }
}

pub(crate) fn svp_prepare(
    module: &Module<NTT4x30Avx>,
    res: &mut SvpPPolBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
) {
    let n = res.n();
    let mut tmp = vec![0u64; 4 * n];
    NTT4x30Avx::ntt_from_znx64(&mut tmp, a.at(a_col, 0));
    NTT4x30Avx::ntt_dft_execute(module.get_ntt_table(), &mut tmp);
    let data: &mut [u32] = cast_slice_mut(res.data_mut());
    unsafe { pack_limb_q120(n, &mut data[4 * n * res_col..][..4 * n], &tmp) };
}

pub(crate) fn svp_ppol_copy_backend(
    res: &mut SvpPPolBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
) {
    let n = res.n();
    let dst: &mut [u32] = cast_slice_mut(res.data_mut());
    let src: &[u32] = cast_slice(a.data());
    dst[4 * n * res_col..][..4 * n].copy_from_slice(&src[4 * n * a_col..][..4 * n]);
}

pub(crate) fn svp_apply_dft(
    module: &Module<NTT4x30Avx>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT4x30Avx>,
    b_col: usize,
) {
    let mut b_dft_owned = module.vec_znx_dft_alloc(1, b.size());
    let mut b_dft = b_dft_owned.to_backend_mut();
    module.vec_znx_dft_apply(1, 0, &mut b_dft, 0, b, b_col);
    svp_apply_dft_to_dft(module, res, res_col, a, a_col, &b_dft.reborrow_backend_ref(), 0);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn svp_apply_dft_to_dft(
    _module: &Module<NTT4x30Avx>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    b_col: usize,
) {
    let n = res.n();
    let (res_cols, b_cols) = (res.cols(), b.cols());
    let min_size = res.size().min(b.size());
    let factor_data: &[u32] = cast_slice(a.data());
    let factor = &factor_data[4 * n * a_col..][..4 * n];
    let b_data: &[u32] = cast_slice(b.data());
    let res_size = res.size();
    let res_data: &mut [u32] = cast_slice_mut(res.data_mut());
    for limb in 0..res_size {
        let dst = packed_limb_mut(res_data, n, res_cols, res_col, limb);
        if limb < min_size {
            unsafe { mul_packed_limb(n, dst, packed_limb(b_data, n, b_cols, b_col, limb), factor) };
        } else {
            dst.fill(0);
        }
    }
}

pub(crate) fn svp_apply_dft_to_dft_assign(
    _module: &Module<NTT4x30Avx>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
) {
    let n = res.n();
    let cols = res.cols();
    let factor_data: &[u32] = cast_slice(a.data());
    let factor = &factor_data[4 * n * a_col..][..4 * n];
    let size = res.size();
    let data: &mut [u32] = cast_slice_mut(res.data_mut());
    for limb in 0..size {
        let dst = packed_limb_mut(data, n, cols, res_col, limb);
        unsafe { mul_packed_limb_assign(n, dst, factor) };
    }
}
