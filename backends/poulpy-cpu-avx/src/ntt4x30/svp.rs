use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m256i, _mm256_castsi256_si128, _mm256_cvtepu32_epi64, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_mul_epu32,
    _mm256_storeu_si256,
};
use poulpy_cpu_portable::kernels::ntt4x30::{NttDFTExecute, NttFromZnx64, vec_znx_dft::NttModuleHandle};
use poulpy_cpu_portable::kernels::ntt4x30::{
    ntt::{NttTable, NttTableInv},
    primes::Primes30,
};
use poulpy_hal::layouts::Ring;
use poulpy_hal::layouts::{
    DataView, DataViewMut, Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendRef,
    VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView, check_degree,
};

use super::{
    NTT4x30Avx,
    arithmetic_avx::{BARRETT_MU, Q_VEC, barrett_reduce},
    vec_znx_dft::{check_operand, dft_limb_wide, pack_limb_q120, pack_two_q120, packed_limb, packed_limb_mut},
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

pub(crate) fn svp_prepare<R: Ring>(
    module: &Module<NTT4x30Avx<R>>,
    res: &mut SvpPPolBackendMut<'_, NTT4x30Avx<R>>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'_, NTT4x30Avx<R>>,
    a_col: usize,
) where
    NTT4x30Avx<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    let n = res.n();
    check_degree::<NTT4x30Avx<R>>(module.n(), n);
    assert!(a.n() == n, "svp_prepare: a.n() != res.n()");
    let data: &mut [u32] = cast_slice_mut(res.data_mut());
    // The q120 form takes twice the destination limb, and `svp_prepare` has no scratch.
    poulpy_cpu_portable::kernels::with_thread_buffer(4 * n, |tmp| {
        NTT4x30Avx::<R>::ntt_from_znx64(tmp, a.at(a_col, 0));
        NTT4x30Avx::<R>::ntt_dft_execute(module.get_ntt_table_for(n), tmp);
        unsafe { pack_limb_q120(n, &mut data[4 * n * res_col..][..4 * n], tmp) };
    });
}

pub(crate) fn svp_ppol_copy<R: Ring>(
    res: &mut SvpPPolBackendMut<'_, NTT4x30Avx<R>>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Avx<R>>,
    a_col: usize,
) {
    assert_eq!(res.n(), a.n(), "svp_ppol_copy: res.n() {} != a.n() {}", res.n(), a.n());
    assert_eq!(
        res.hint(),
        a.hint(),
        "svp_ppol_copy: res and a must carry the same PrepareHint ({:?} != {:?})",
        res.hint(),
        a.hint()
    );
    let n = res.n();
    let dst: &mut [u32] = cast_slice_mut(res.data_mut());
    let src: &[u32] = cast_slice(a.data());
    dst[4 * n * res_col..][..4 * n].copy_from_slice(&src[4 * n * a_col..][..4 * n]);
}

/// Scratch of [`svp_apply_dft`]: the q120 form of one limb, twice the destination limb.
pub(crate) fn svp_apply_dft_tmp_bytes(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

/// `res = a * dft(b)`: each limb of `b` is transformed into its destination through `tmp`, then multiplied there.
#[allow(clippy::too_many_arguments)]
pub(crate) fn svp_apply_dft<R: Ring>(
    module: &Module<NTT4x30Avx<R>>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx<R>>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Avx<R>>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT4x30Avx<R>>,
    b_col: usize,
    tmp: &mut [u64],
) where
    NTT4x30Avx<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    poulpy_hal::layouts::assert_dense(b, "svp_apply_dft");
    let n = res.n();
    check_operand(n, res, res_col);
    check_degree::<NTT4x30Avx<R>>(module.n(), n);
    check_operand(n, b, b_col);
    assert_eq!(a.n(), n, "svp_apply_dft: a.n() != res.n()");
    assert!(tmp.len() >= 4 * n, "svp_apply_dft: scratch below its query");
    let tmp = &mut tmp[..4 * n];
    let (res_cols, res_size) = (res.cols(), res.size());
    let min_size = res_size.min(b.size());
    let factor_data: &[u32] = cast_slice(a.data());
    let factor = &factor_data[4 * n * a_col..][..4 * n];
    let res_data: &mut [u32] = cast_slice_mut(res.data_mut());
    for limb in 0..res_size {
        let dst = packed_limb_mut(res_data, n, res_cols, res_col, limb);
        if limb < min_size {
            dft_limb_wide(module, n, tmp, b.at(b_col, limb));
            unsafe {
                pack_limb_q120(n, dst, tmp);
                mul_packed_limb_assign(n, dst, factor);
            }
        } else {
            dst.fill(0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn svp_apply_dft_to_dft<R: Ring>(
    _module: &Module<NTT4x30Avx<R>>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx<R>>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Avx<R>>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT4x30Avx<R>>,
    b_col: usize,
) {
    let n = res.n();
    assert_eq!(a.n(), n, "svp_apply_dft_to_dft: a.n():{} != res.n():{n}", a.n());
    assert_eq!(b.n(), n, "svp_apply_dft_to_dft: b.n():{} != res.n():{n}", b.n());
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

pub(crate) fn svp_apply_dft_to_dft_assign<R: Ring>(
    _module: &Module<NTT4x30Avx<R>>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx<R>>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Avx<R>>,
    a_col: usize,
) {
    let n = res.n();
    assert_eq!(a.n(), n, "svp_apply_dft_to_dft_assign: a.n():{} != res.n():{n}", a.n());
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
