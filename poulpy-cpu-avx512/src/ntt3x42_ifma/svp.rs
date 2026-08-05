//! Scalar-vector product (SVP) operations for [`NTT3x42Ifma`].

use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_cpu_ref::reference::ntt4x30::primes::PrimeSet;

use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftApply},
    layouts::{
        Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftToBackendMut, ZnxView, ZnxViewMut,
    },
};

use crate::NTT3x42Ifma;
use crate::ntt3x42_ifma::{
    kernels::ntt_avx512,
    module::handle,
    primes::Primes42,
    tables::{harvey_modmul, harvey_quotient},
    traits::{Ntt3x42IfmaCFromB, Ntt3x42IfmaFromZnx64, Ntt3x42IfmaZero},
};

#[inline(always)]
fn mul_mod_lazy(a: u64, b: u64, prime: usize) -> u64 {
    let q = Primes42::Q[prime];
    let b = if b >= q { b - q } else { b };
    harvey_modmul(a, b, harvey_quotient(b, q), q)
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
    let b_dft_ref = poulpy_hal::layouts::VecZnxDftReborrowBackendRef::<NTT3x42Ifma>::reborrow_backend_ref(&b_dft);
    svp_apply_dft_to_dft(module, res, res_col, a, a_col, &b_dft_ref, 0);
}

/// Pointwise DFT-domain multiply: `res = a ⊙ b`.
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
    let b_size = b.size();
    let min_size = res_size.min(b_size);

    let a_u64: &[u64] = cast_slice(a.at(a_col, 0));

    for j in 0..min_size {
        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, j));
        let b_u64: &[u64] = cast_slice(b.at(b_col, j));
        for prime in 0..3 {
            let base = prime * n;
            for i in 0..n {
                res_u64[base + i] = mul_mod_lazy(b_u64[base + i], a_u64[base + i], prime);
            }
        }
    }

    for j in min_size..res_size {
        NTT3x42Ifma::ntt3x42_ifma_zero(cast_slice_mut(res.at_mut(res_col, j)));
    }
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

    let a_u64: &[u64] = cast_slice(a.at(a_col, 0));

    for j in 0..res_size {
        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, j));
        for prime in 0..3 {
            let base = prime * n;
            for i in 0..n {
                res_u64[base + i] = mul_mod_lazy(res_u64[base + i], a_u64[base + i], prime);
            }
        }
    }
}
