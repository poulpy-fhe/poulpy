use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_castsi128_si256, _mm256_castsi256_si128, _mm256_cmpgt_epi32,
    _mm256_cvtepu32_epi64, _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_permutevar8x32_epi32, _mm256_set1_epi32,
    _mm256_setr_epi32, _mm256_storeu_si256, _mm256_sub_epi32,
};
use poulpy_cpu_ref::reference::ntt4x30::{
    NttDFTExecute, NttFromZnx64, NttToZnx128,
    primes::{PrimeSet, Primes30},
    vec_znx_dft::{NttAutomorphismPlan, NttModuleHandle},
};
use poulpy_hal::execution::TaskExecutor;
use poulpy_hal::layouts::{
    DataView, DataViewMut, Module, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView,
    ZnxViewMut,
};

use super::{
    NTT4x30Avx,
    arithmetic_avx::{BARRETT_MU, POW32, Q_VEC, reduce_b_to_canonical},
};

const Q32X8: [u32; 8] = [
    Primes30::Q[0],
    Primes30::Q[1],
    Primes30::Q[2],
    Primes30::Q[3],
    Primes30::Q[0],
    Primes30::Q[1],
    Primes30::Q[2],
    Primes30::Q[3],
];

#[inline(always)]
pub(crate) fn packed_limb(data: &[u32], n: usize, cols: usize, col: usize, limb: usize) -> &[u32] {
    let start = 4 * n * (limb * cols + col);
    &data[start..start + 4 * n]
}

#[inline(always)]
pub(crate) fn packed_limb_mut(data: &mut [u32], n: usize, cols: usize, col: usize, limb: usize) -> &mut [u32] {
    let start = 4 * n * (limb * cols + col);
    &mut data[start..start + 4 * n]
}

#[inline(always)]
pub(crate) unsafe fn pack_two_q120(a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        let a = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(a, idx));
        let b = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(b, idx));
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(a), b)
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pack_limb_q120(n: usize, dst: &mut [u32], src: &[u64]) {
    debug_assert!(dst.len() >= 4 * n);
    debug_assert!(src.len() >= 4 * n);
    unsafe {
        let q = _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i);
        let mu = _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i);
        let pow32 = _mm256_loadu_si256(POW32.as_ptr() as *const __m256i);
        for pair in 0..n / 2 {
            let a = _mm256_loadu_si256(src.as_ptr().add(8 * pair) as *const __m256i);
            let b = _mm256_loadu_si256(src.as_ptr().add(8 * pair + 4) as *const __m256i);
            let a = reduce_b_to_canonical(a, q, mu, pow32);
            let b = reduce_b_to_canonical(b, q, mu, pow32);
            _mm256_storeu_si256(dst.as_mut_ptr().add(8 * pair) as *mut __m256i, pack_two_q120(a, b));
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn canonicalize_limb_q120(n: usize, src: &mut [u64]) {
    debug_assert!(src.len() >= 4 * n);
    unsafe {
        let q = _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i);
        let mu = _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i);
        let pow32 = _mm256_loadu_si256(POW32.as_ptr() as *const __m256i);
        for coeff in 0..n {
            let ptr = src.as_mut_ptr().add(4 * coeff) as *mut __m256i;
            _mm256_storeu_si256(ptr, reduce_b_to_canonical(_mm256_loadu_si256(ptr), q, mu, pow32));
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn unpack_limb_q120(n: usize, dst: &mut [u64], src: &[u32]) {
    debug_assert!(dst.len() >= 4 * n);
    debug_assert!(src.len() >= 4 * n);
    unsafe {
        for pair in 0..n / 2 {
            let x = _mm256_loadu_si256(src.as_ptr().add(8 * pair) as *const __m256i);
            let a = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(x));
            let b = _mm256_cvtepu32_epi64(core::arch::x86_64::_mm256_extracti128_si256::<1>(x));
            _mm256_storeu_si256(dst.as_mut_ptr().add(8 * pair) as *mut __m256i, a);
            _mm256_storeu_si256(dst.as_mut_ptr().add(8 * pair + 4) as *mut __m256i, b);
        }
    }
}

pub(crate) fn dft_limb(module: &Module<NTT4x30Avx>, dst: &mut [u32], src: Option<&[i64]>, tmp: &mut [u64]) {
    if let Some(src) = src {
        NTT4x30Avx::ntt_from_znx64(tmp, src);
        NTT4x30Avx::ntt_dft_execute(module.get_ntt_table(), tmp);
        unsafe { pack_limb_q120(module.n(), dst, tmp) };
    } else {
        dst.fill(0);
    }
}

pub(crate) fn idft_limb(module: &Module<NTT4x30Avx>, dst: &mut [i128], src: &[u32], tmp: &mut [u64]) {
    let n = module.n();
    unsafe { unpack_limb_q120(n, tmp, src) };
    NTT4x30Avx::ntt_dft_execute(module.get_intt_table(), tmp);
    NTT4x30Avx::ntt_to_znx128(dst, n, tmp);
}

#[inline(always)]
unsafe fn reduce_u32(x: __m256i, q: __m256i, q_minus_one: __m256i) -> __m256i {
    unsafe { _mm256_sub_epi32(x, _mm256_and_si256(_mm256_cmpgt_epi32(x, q_minus_one), q)) }
}

#[target_feature(enable = "avx2")]
unsafe fn packed_add(n: usize, dst: &mut [u32], a: &[u32], b: &[u32]) {
    unsafe {
        let q = _mm256_loadu_si256(Q32X8.as_ptr() as *const __m256i);
        let one = _mm256_set1_epi32(1);
        let qm1 = _mm256_sub_epi32(q, one);
        for pair in 0..n / 2 {
            let off = 8 * pair;
            let av = _mm256_loadu_si256(a.as_ptr().add(off) as *const __m256i);
            let bv = _mm256_loadu_si256(b.as_ptr().add(off) as *const __m256i);
            _mm256_storeu_si256(
                dst.as_mut_ptr().add(off) as *mut __m256i,
                reduce_u32(_mm256_add_epi32(av, bv), q, qm1),
            );
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn packed_add_assign(n: usize, dst: &mut [u32], a: &[u32]) {
    unsafe {
        let q = _mm256_loadu_si256(Q32X8.as_ptr() as *const __m256i);
        let qm1 = _mm256_sub_epi32(q, _mm256_set1_epi32(1));
        for pair in 0..n / 2 {
            let off = 8 * pair;
            let dv = _mm256_loadu_si256(dst.as_ptr().add(off) as *const __m256i);
            let av = _mm256_loadu_si256(a.as_ptr().add(off) as *const __m256i);
            _mm256_storeu_si256(
                dst.as_mut_ptr().add(off) as *mut __m256i,
                reduce_u32(_mm256_add_epi32(dv, av), q, qm1),
            );
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn packed_sub(n: usize, dst: &mut [u32], a: &[u32], b: &[u32]) {
    unsafe {
        let q = _mm256_loadu_si256(Q32X8.as_ptr() as *const __m256i);
        let qm1 = _mm256_sub_epi32(q, _mm256_set1_epi32(1));
        for pair in 0..n / 2 {
            let off = 8 * pair;
            let av = _mm256_loadu_si256(a.as_ptr().add(off) as *const __m256i);
            let bv = _mm256_loadu_si256(b.as_ptr().add(off) as *const __m256i);
            let x = _mm256_sub_epi32(_mm256_add_epi32(av, q), bv);
            _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, reduce_u32(x, q, qm1));
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn packed_sub_assign(n: usize, dst: &mut [u32], a: &[u32]) {
    unsafe {
        let q = _mm256_loadu_si256(Q32X8.as_ptr() as *const __m256i);
        let qm1 = _mm256_sub_epi32(q, _mm256_set1_epi32(1));
        for pair in 0..n / 2 {
            let off = 8 * pair;
            let dv = _mm256_loadu_si256(dst.as_ptr().add(off) as *const __m256i);
            let av = _mm256_loadu_si256(a.as_ptr().add(off) as *const __m256i);
            let x = _mm256_sub_epi32(_mm256_add_epi32(dv, q), av);
            _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, reduce_u32(x, q, qm1));
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn packed_sub_negate_assign(n: usize, dst: &mut [u32], a: &[u32]) {
    unsafe {
        let q = _mm256_loadu_si256(Q32X8.as_ptr() as *const __m256i);
        let qm1 = _mm256_sub_epi32(q, _mm256_set1_epi32(1));
        for pair in 0..n / 2 {
            let off = 8 * pair;
            let dv = _mm256_loadu_si256(dst.as_ptr().add(off) as *const __m256i);
            let av = _mm256_loadu_si256(a.as_ptr().add(off) as *const __m256i);
            let x = _mm256_sub_epi32(_mm256_add_epi32(av, q), dv);
            _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, reduce_u32(x, q, qm1));
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn packed_negate_assign(n: usize, dst: &mut [u32]) {
    unsafe {
        let q = _mm256_loadu_si256(Q32X8.as_ptr() as *const __m256i);
        let qm1 = _mm256_sub_epi32(q, _mm256_set1_epi32(1));
        for pair in 0..n / 2 {
            let off = 8 * pair;
            let dv = _mm256_loadu_si256(dst.as_ptr().add(off) as *const __m256i);
            _mm256_storeu_si256(
                dst.as_mut_ptr().add(off) as *mut __m256i,
                reduce_u32(_mm256_sub_epi32(q, dv), q, qm1),
            );
        }
    }
}

pub(crate) fn vec_znx_dft_apply(
    module: &Module<NTT4x30Avx>,
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
) {
    let n = res.n();
    let cols = res.cols();
    let res_size = res.size();
    let a_size = a.size();
    let mut tmp = vec![0u64; 4 * n];
    let res_data: &mut [u32] = cast_slice_mut(res.data_mut());
    for limb in 0..res_size {
        let dst = packed_limb_mut(res_data, n, cols, res_col, limb);
        let src_limb = offset + limb * step;
        dft_limb(module, dst, (src_limb < a_size).then(|| a.at(a_col, src_limb)), &mut tmp);
    }
}

pub(crate) fn vec_znx_idft_apply_tmp_bytes(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

pub(crate) fn vec_znx_idft_apply(
    module: &Module<NTT4x30Avx>,
    res: &mut VecZnxBigBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let min_size = res.size().min(a.size());
    let a_cols = a.cols();
    let a_data: &[u32] = cast_slice(a.data());
    for limb in 0..min_size {
        idft_limb(
            module,
            res.at_mut(res_col, limb),
            packed_limb(a_data, n, a_cols, a_col, limb),
            tmp,
        );
    }
    for limb in min_size..res.size() {
        res.at_mut(res_col, limb).fill(0);
    }
}

pub(crate) fn vec_znx_idft_apply_tmpa(
    module: &Module<NTT4x30Avx>,
    res: &mut VecZnxBigBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    a_col: usize,
) {
    let mut tmp = vec![0u64; 4 * a.n()];
    let a_ref = poulpy_hal::layouts::vec_znx_dft_backend_ref_from_mut(a);
    vec_znx_idft_apply(module, res, res_col, &a_ref, a_col, &mut tmp);
}

pub(crate) fn idft_compact_in_place(
    module: &Module<NTT4x30Avx>,
    a: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    a_col: usize,
    tmp: &mut [u64],
) {
    let n = a.n();
    let cols = a.cols();
    let size = a.size();
    let data: &mut [u32] = cast_slice_mut(a.data_mut());
    for limb in 0..size {
        let slot = packed_limb_mut(data, n, cols, a_col, limb);
        unsafe { unpack_limb_q120(n, tmp, slot) };
        NTT4x30Avx::ntt_dft_execute(module.get_intt_table(), tmp);
        let dst = unsafe { std::slice::from_raw_parts_mut(slot.as_mut_ptr() as *mut i128, n) };
        NTT4x30Avx::ntt_to_znx128(dst, n, tmp);
    }
}

pub(crate) fn vec_znx_dft_add_into(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    b_col: usize,
) {
    let n = res.n();
    let (rc, ac, bc) = (res.cols(), a.cols(), b.cols());
    let (rs, asz, bsz) = (res.size(), a.size(), b.size());
    let (sum_size, copy_size, copy_b) = if asz <= bsz {
        (asz.min(rs), bsz.min(rs), true)
    } else {
        (bsz.min(rs), asz.min(rs), false)
    };
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    let bp: &[u32] = cast_slice(b.data());
    for limb in 0..rs {
        let dst = packed_limb_mut(rp, n, rc, res_col, limb);
        if limb < sum_size {
            unsafe {
                packed_add(
                    n,
                    dst,
                    packed_limb(ap, n, ac, a_col, limb),
                    packed_limb(bp, n, bc, b_col, limb),
                )
            };
        } else if limb < copy_size {
            let src = if copy_b {
                packed_limb(bp, n, bc, b_col, limb)
            } else {
                packed_limb(ap, n, ac, a_col, limb)
            };
            dst.copy_from_slice(src);
        } else {
            dst.fill(0);
        }
    }
}

pub(crate) fn vec_znx_dft_add_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let size = res.size().min(a.size());
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for limb in 0..size {
        unsafe {
            packed_add_assign(
                n,
                packed_limb_mut(rp, n, rc, res_col, limb),
                packed_limb(ap, n, ac, a_col, limb),
            )
        };
    }
}

pub(crate) fn vec_znx_dft_add_scaled_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
    scale: i64,
) {
    let (res_shift, a_shift, size) = if scale > 0 {
        let shift = (scale as usize).min(a.size());
        (0, shift, a.size().min(res.size()).saturating_sub(shift))
    } else if scale < 0 {
        let shift = (scale.unsigned_abs() as usize).min(res.size());
        (shift, 0, a.size().min(res.size().saturating_sub(shift)))
    } else {
        (0, 0, a.size().min(res.size()))
    };
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for limb in 0..size {
        unsafe {
            packed_add_assign(
                n,
                packed_limb_mut(rp, n, rc, res_col, limb + res_shift),
                packed_limb(ap, n, ac, a_col, limb + a_shift),
            )
        };
    }
}

pub(crate) fn vec_znx_dft_sub(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    b_col: usize,
) {
    let n = res.n();
    let (rc, ac, bc) = (res.cols(), a.cols(), b.cols());
    let (rs, asz, bsz) = (res.size(), a.size(), b.size());
    let (sub_size, copy_size, negate_b) = if asz <= bsz {
        (asz.min(rs), bsz.min(rs), true)
    } else {
        (bsz.min(rs), asz.min(rs), false)
    };
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    let bp: &[u32] = cast_slice(b.data());
    for limb in 0..rs {
        let dst = packed_limb_mut(rp, n, rc, res_col, limb);
        if limb < sub_size {
            unsafe {
                packed_sub(
                    n,
                    dst,
                    packed_limb(ap, n, ac, a_col, limb),
                    packed_limb(bp, n, bc, b_col, limb),
                )
            };
        } else if limb < copy_size {
            if negate_b {
                dst.copy_from_slice(packed_limb(bp, n, bc, b_col, limb));
                unsafe { packed_negate_assign(n, dst) };
            } else {
                dst.copy_from_slice(packed_limb(ap, n, ac, a_col, limb));
            }
        } else {
            dst.fill(0);
        }
    }
}

pub(crate) fn vec_znx_dft_sub_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let size = res.size().min(a.size());
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for limb in 0..size {
        unsafe {
            packed_sub_assign(
                n,
                packed_limb_mut(rp, n, rc, res_col, limb),
                packed_limb(ap, n, ac, a_col, limb),
            )
        };
    }
}

pub(crate) fn vec_znx_dft_sub_negate_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let rs = res.size();
    let size = rs.min(a.size());
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for limb in 0..rs {
        let dst = packed_limb_mut(rp, n, rc, res_col, limb);
        if limb < size {
            unsafe { packed_sub_negate_assign(n, dst, packed_limb(ap, n, ac, a_col, limb)) };
        } else {
            unsafe { packed_negate_assign(n, dst) };
        }
    }
}

pub(crate) fn vec_znx_dft_copy(
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let size = res.size();
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for limb in 0..size {
        let dst = packed_limb_mut(rp, n, rc, res_col, limb);
        let src_limb = offset + limb * step;
        if src_limb < a.size() {
            dst.copy_from_slice(packed_limb(ap, n, ac, a_col, src_limb));
        } else {
            dst.fill(0);
        }
    }
}

pub(crate) fn vec_znx_dft_zero(res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>, res_col: usize) {
    let n = res.n();
    let cols = res.cols();
    let size = res.size();
    let data: &mut [u32] = cast_slice_mut(res.data_mut());
    for limb in 0..size {
        packed_limb_mut(data, n, cols, res_col, limb).fill(0);
    }
}

pub(crate) fn vec_znx_dft_automorphism(
    plan: &NttAutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let size = res.size().min(a.size());
    let rs = res.size();
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for limb in 0..rs {
        let dst = packed_limb_mut(rp, n, rc, res_col, limb);
        if limb < size {
            let src = packed_limb(ap, n, ac, a_col, limb);
            for (i, &p) in plan.perm.iter().enumerate() {
                dst[4 * i..4 * i + 4].copy_from_slice(&src[4 * p as usize..4 * p as usize + 4]);
            }
        } else {
            dst.fill(0);
        }
    }
}

pub(crate) fn vec_znx_dft_automorphism_add<E: TaskExecutor>(
    plan: &NttAutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let size = res.size().min(a.size());
    let res_ptr = cast_slice_mut::<_, u32>(res.data_mut()).as_mut_ptr() as usize;
    let ap: &[u32] = cast_slice(a.data());
    E::for_each(size, |limb| {
        let start = 4 * n * (limb * rc + res_col);
        let dst = unsafe { std::slice::from_raw_parts_mut((res_ptr as *mut u32).add(start), 4 * n) };
        let src = packed_limb(ap, n, ac, a_col, limb);
        for (i, &p) in plan.perm.iter().enumerate() {
            let p = p as usize;
            for prime in 0..4 {
                let sum = dst[4 * i + prime] as u64 + src[4 * p + prime] as u64;
                let q = Primes30::Q[prime] as u64;
                dst[4 * i + prime] = if sum >= q { (sum - q) as u32 } else { sum as u32 };
            }
        }
    });
}
