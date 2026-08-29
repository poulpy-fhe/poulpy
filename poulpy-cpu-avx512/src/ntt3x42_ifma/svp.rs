//! Scalar-vector product (SVP) operations for [`NTT3x42Ifma`].

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{__m512i, _mm512_loadu_si512, _mm512_set1_epi64, _mm512_storeu_si512};
use poulpy_hal::layouts::PrimeSet;

use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftApply},
    layouts::{
        DataView, DataViewMut, Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendRef,
        VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftReborrowBackendRef, VecZnxDftToBackendMut, ZnxView,
    },
};

use crate::NTT3x42Ifma;
use crate::ntt3x42_ifma::{
    execution::{SendPtr, for_index_exec},
    kernels::{cond_sub_2q_si512, harvey_modmul_si512, ntt_avx512},
    module::handle,
    primes::Primes42,
    tables::harvey_quotient,
    traits::{Ntt3x42IfmaCFromB, Ntt3x42IfmaFromZnx64},
    vec_znx_dft::{MASK20, MASK22, MASK42},
    vmp::{pack_y, unpack_y},
};

#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn mul_packed_limb(n: usize, dst: *mut u64, src: *const u64, prepared: &[u64]) {
    unsafe {
        let m42 = _mm512_set1_epi64(MASK42 as i64);
        let m20 = _mm512_set1_epi64(MASK20 as i64);
        let m22 = _mm512_set1_epi64(MASK22 as i64);
        let q = Primes42::Q.map(|q| _mm512_set1_epi64(q as i64));

        for g in 0..n / 8 {
            let off = 16 * g;
            let w0 = _mm512_loadu_si512(src.add(off) as *const __m512i);
            let w1 = _mm512_loadu_si512(src.add(off + 8) as *const __m512i);
            let mut y = unpack_y(w0, w1, m42, m20);
            for p in 0..3 {
                let factor = _mm512_loadu_si512(prepared.as_ptr().add(p * n + 8 * g) as *const __m512i);
                let quotient = _mm512_loadu_si512(prepared.as_ptr().add((3 + p) * n + 8 * g) as *const __m512i);
                y[p] = cond_sub_2q_si512(harvey_modmul_si512(y[p], factor, quotient, q[p]), q[p]);
            }
            let [r0, r1] = pack_y(y, m22);
            _mm512_storeu_si512(dst.add(off) as *mut __m512i, r0);
            _mm512_storeu_si512(dst.add(off + 8) as *mut __m512i, r1);
        }
    }
}

/// Encode a scalar polynomial into IFMA prepared format.
pub(crate) fn svp_prepare(
    module: &Module<NTT3x42Ifma>,
    res: &mut SvpPPolBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    let n = res.n();

    let mut tmp = vec![0u64; 3 * n];
    NTT3x42Ifma::ntt3x42_ifma_from_znx64(&mut tmp, a.at(a_col, 0));
    // Lazy [0, 4q): consumed only by c_from_b (re-reduces).
    unsafe { ntt_avx512::<Primes42>(&handle(module).table_ntt, &mut tmp, true) };

    let res_u64: &mut [u64] = cast_slice_mut(res.data_mut());
    let prepared = &mut res_u64[6 * n * res_col..][..6 * n];
    let res_u32: &mut [u32] = cast_slice_mut(&mut prepared[..3 * n]);
    NTT3x42Ifma::ntt3x42_ifma_c_from_b(n, res_u32, &tmp);
    for p in 0..3 {
        let q = Primes42::Q[p];
        for i in 0..n {
            prepared[(3 + p) * n + i] = harvey_quotient(prepared[p * n + i], q);
        }
    }
}

pub(crate) fn svp_ppol_copy_backend(
    res: &mut SvpPPolBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    let n = res.n();
    let res_u64: &mut [u64] = cast_slice_mut(res.data_mut());
    let a_u64: &[u64] = cast_slice(a.data());
    res_u64[6 * n * res_col..][..6 * n].copy_from_slice(&a_u64[6 * n * a_col..][..6 * n]);
}

/// Lift `a` (`VecZnx`) to DFT-domain via the forward NTT, then apply the
/// prepared SVP factor: `res = svp ⊙ NTT(a)`.
pub(crate) fn svp_apply_dft<E: poulpy_hal::execution::TaskExecutor>(
    module: &Module<NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
) {
    let b_size = b.size();
    let mut b_dft_owned = module.vec_znx_dft_alloc(1, b_size);
    let mut b_dft = b_dft_owned.to_backend_mut();
    <Module<NTT3x42Ifma> as VecZnxDftApply<NTT3x42Ifma>>::vec_znx_dft_apply(module, 1, 0, &mut b_dft, 0, b, b_col);
    let b_dft_ref = b_dft.reborrow_backend_ref();
    svp_apply_dft_to_dft::<E>(module, res, res_col, a, a_col, &b_dft_ref, 0);
}

/// Pointwise DFT-domain multiply: `res = a ⊙ b` (`b` and `res` packed).
pub(crate) fn svp_apply_dft_to_dft<E: poulpy_hal::execution::TaskExecutor>(
    _module: &Module<NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
) {
    let n = res.n();
    let res_size = res.size();
    let res_cols = res.cols();
    let b_cols = b.cols();
    let b_size = b.size();
    let min_size = res_size.min(b_size);

    let a_u64: &[u64] = cast_slice(a.data());
    let prepared = &a_u64[6 * n * a_col..][..6 * n];
    let b_u64: &[u64] = cast_slice(b.data());
    let res_data: &mut [u64] = cast_slice_mut(res.data_mut());
    let res_ptr = SendPtr(res_data.as_mut_ptr());

    for_index_exec::<E>(res_size, 2 * n * res_size, |j| {
        let start = 2 * n * (j * res_cols + res_col);
        let res_u64 = unsafe { std::slice::from_raw_parts_mut(res_ptr.get().add(start), 2 * n) };
        if j < min_size {
            let b_limb: &[u64] = &b_u64[2 * n * (j * b_cols + b_col)..][..2 * n];
            unsafe { mul_packed_limb(n, res_u64.as_mut_ptr(), b_limb.as_ptr(), prepared) };
        } else {
            res_u64.fill(0);
        }
    });
}

/// Pointwise DFT-domain multiply in place: `res = a ⊙ res`.
pub(crate) fn svp_apply_dft_to_dft_assign<E: poulpy_hal::execution::TaskExecutor>(
    _module: &Module<NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    let n = res.n();
    let res_size = res.size();
    let res_cols = res.cols();

    let a_u64: &[u64] = cast_slice(a.data());
    let prepared = &a_u64[6 * n * a_col..][..6 * n];
    let res_data: &mut [u64] = cast_slice_mut(res.data_mut());
    let res_ptr = SendPtr(res_data.as_mut_ptr());

    for_index_exec::<E>(res_size, 2 * n * res_size, |j| {
        let start = 2 * n * (j * res_cols + res_col);
        let res_u64 = unsafe { std::slice::from_raw_parts_mut(res_ptr.get().add(start), 2 * n) };
        let ptr = res_u64.as_mut_ptr();
        unsafe { mul_packed_limb(n, ptr, ptr, prepared) };
    });
}
