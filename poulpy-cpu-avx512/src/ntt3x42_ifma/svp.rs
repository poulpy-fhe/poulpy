//! Scalar-vector product (SVP) operations for [`NTT3x42Ifma`].

use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_hal::layouts::PrimeSet;

use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftApply},
    layouts::{
        DataView, DataViewMut, Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendRef,
        VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftReborrowBackendRef, VecZnxDftToBackendMut, ZnxView, ZnxViewMut,
    },
};

use crate::NTT3x42Ifma;
use crate::ntt3x42_ifma::{
    kernels::ntt_avx512,
    module::handle,
    primes::Primes42,
    serial::{SendPtr, for_index},
    tables::{harvey_modmul, harvey_quotient},
    traits::{Ntt3x42IfmaCFromB, Ntt3x42IfmaFromZnx64},
    vec_znx_dft::{pack_scalar_3x42, unpack_scalar_3x42},
};

/// `(a * b) mod q`, canonical output in `[0, q)`.
#[inline(always)]
fn mul_mod_canonical(a: u64, b: u64, prime: usize) -> u64 {
    let q = Primes42::Q[prime];
    let b = if b >= q { b - q } else { b };
    let r = harvey_modmul(a, b, harvey_quotient(b, q), q);
    if r >= q { r - q } else { r }
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

    let res_u32: &mut [u32] = cast_slice_mut(res.at_mut(res_col, 0));
    NTT3x42Ifma::ntt3x42_ifma_c_from_b(n, res_u32, &tmp);
}

/// Lift `a` (`VecZnx`) to DFT-domain via the forward NTT, then apply the
/// prepared SVP factor: `res = svp ⊙ NTT(a)`.
pub(crate) fn svp_apply_dft(
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
    svp_apply_dft_to_dft(module, res, res_col, a, a_col, &b_dft_ref, 0);
}

/// Pointwise DFT-domain multiply: `res = a ⊙ b` (`b` and `res` packed).
pub(crate) fn svp_apply_dft_to_dft(
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

    let a_u64: &[u64] = cast_slice(a.at(a_col, 0));
    let b_u64: &[u64] = cast_slice(b.data());
    let base_ptr = SendPtr(cast_slice_mut::<_, u64>(res.data_mut()).as_mut_ptr());

    for_index(res_size, 2 * n * res_size, |j| {
        let res_u64: &mut [u64] =
            unsafe { std::slice::from_raw_parts_mut(base_ptr.get().add(2 * n * (j * res_cols + res_col)), 2 * n) };
        if j < min_size {
            let b_limb: &[u64] = &b_u64[2 * n * (j * b_cols + b_col)..][..2 * n];
            for i in 0..n {
                let off = 16 * (i >> 3) + (i & 7);
                let (p0, p1, p2) = unpack_scalar_3x42(b_limb[off], b_limb[off + 8]);
                let r0 = mul_mod_canonical(p0, a_u64[i], 0);
                let r1 = mul_mod_canonical(p1, a_u64[n + i], 1);
                let r2 = mul_mod_canonical(p2, a_u64[2 * n + i], 2);
                let (w0, w1) = pack_scalar_3x42(r0, r1, r2);
                res_u64[off] = w0;
                res_u64[off + 8] = w1;
            }
        } else {
            res_u64.fill(0);
        }
    });
}

/// Pointwise DFT-domain multiply in place: `res = a ⊙ res`.
pub(crate) fn svp_apply_dft_to_dft_assign(
    _module: &Module<NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    let n = res.n();
    let res_size = res.size();
    let res_cols = res.cols();

    let a_u64: &[u64] = cast_slice(a.at(a_col, 0));
    let base_ptr = SendPtr(cast_slice_mut::<_, u64>(res.data_mut()).as_mut_ptr());

    for_index(res_size, 2 * n * res_size, |j| {
        let res_u64: &mut [u64] =
            unsafe { std::slice::from_raw_parts_mut(base_ptr.get().add(2 * n * (j * res_cols + res_col)), 2 * n) };
        for i in 0..n {
            let off = 16 * (i >> 3) + (i & 7);
            let (p0, p1, p2) = unpack_scalar_3x42(res_u64[off], res_u64[off + 8]);
            let r0 = mul_mod_canonical(p0, a_u64[i], 0);
            let r1 = mul_mod_canonical(p1, a_u64[n + i], 1);
            let r2 = mul_mod_canonical(p2, a_u64[2 * n + i], 2);
            let (w0, w1) = pack_scalar_3x42(r0, r1, r2);
            res_u64[off] = w0;
            res_u64[off + 8] = w1;
        }
    });
}
