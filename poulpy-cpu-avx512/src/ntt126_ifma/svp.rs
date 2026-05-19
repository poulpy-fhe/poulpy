//! Scalar-vector product (SVP) operations for [`NTT126Ifma`].

use bytemuck::{cast_slice, cast_slice_mut};

use poulpy_cpu_ref::reference::ntt120::types::Q120bScalar;
use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftApply},
    layouts::{
        Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftReborrowBackendRef, VecZnxDftToBackendMut, ZnxView, ZnxViewMut,
    },
};

use crate::NTT126Ifma;
use crate::ntt126_ifma::{
    module::handle,
    traits::{Ntt126IfmaCFromB, Ntt126IfmaDFTExecute, Ntt126IfmaFromZnx64, Ntt126IfmaMulBbc, Ntt126IfmaZero},
};

/// Encode a scalar polynomial into IFMA prepared format.
pub(crate) fn svp_prepare(
    module: &Module<NTT126Ifma>,
    res: &mut SvpPPolBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'_, NTT126Ifma>,
    a_col: usize,
) {
    let n = res.n();

    let mut tmp = vec![0u64; 4 * n];
    NTT126Ifma::ntt126_ifma_from_znx64(&mut tmp, a.at(a_col, 0));
    NTT126Ifma::ntt126_ifma_dft_execute(&handle(module).table_ntt, &mut tmp);

    let res_u32: &mut [u32] = cast_slice_mut(res.at_mut(res_col, 0));
    NTT126Ifma::ntt126_ifma_c_from_b(n, res_u32, &tmp);
}

/// Lift `a` (`VecZnx`) to DFT-domain via the forward NTT, then apply the
/// prepared SVP factor: `res = svp ⊙ NTT(a)`.
pub(crate) fn svp_apply_dft(
    module: &Module<NTT126Ifma>,
    res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT126Ifma>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT126Ifma>,
    b_col: usize,
) {
    let b_size = b.size();
    let mut b_dft_owned = module.vec_znx_dft_alloc(1, b_size);
    let mut b_dft = b_dft_owned.to_backend_mut();
    <Module<NTT126Ifma> as VecZnxDftApply<NTT126Ifma>>::vec_znx_dft_apply(module, 1, 0, &mut b_dft, 0, b, b_col);
    let b_dft_ref = b_dft.reborrow_backend_ref();
    svp_apply_dft_to_dft(module, res, res_col, a, a_col, &b_dft_ref, 0);
}

/// Pointwise DFT-domain multiply: `res = a ⊙ b`.
pub(crate) fn svp_apply_dft_to_dft(
    module: &Module<NTT126Ifma>,
    res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT126Ifma>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT126Ifma>,
    b_col: usize,
) {
    let meta = &handle(module).meta_bbc;
    let n = res.n();
    let res_size = res.size();
    let b_size = b.size();
    let min_size = res_size.min(b_size);

    let a_u32: &[u32] = cast_slice(a.at(a_col, 0));

    for j in 0..min_size {
        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, j));
        let b_u32: &[u32] = cast_slice(b.at(b_col, j));
        for n_i in 0..n {
            NTT126Ifma::ntt126_ifma_mul_bbc(
                meta,
                1,
                &mut res_u64[4 * n_i..4 * n_i + 4],
                &b_u32[8 * n_i..8 * n_i + 8],
                &a_u32[8 * n_i..8 * n_i + 8],
            );
        }
    }

    for j in min_size..res_size {
        NTT126Ifma::ntt126_ifma_zero(cast_slice_mut(res.at_mut(res_col, j)));
    }
}

/// Pointwise DFT-domain multiply in place: `res = a ⊙ res`.
pub(crate) fn svp_apply_dft_to_dft_assign(
    module: &Module<NTT126Ifma>,
    res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT126Ifma>,
    a_col: usize,
) {
    let meta = &handle(module).meta_bbc;
    let n = res.n();
    let res_size = res.size();

    let a_u32: &[u32] = cast_slice(a.at(a_col, 0));

    for j in 0..res_size {
        let res_slice: &mut [Q120bScalar] = res.at_mut(res_col, j);
        let mut product = [0u64; 4];
        for n_i in 0..n {
            let x_elem: Q120bScalar = res_slice[n_i];
            let x_u32: &[u32] = cast_slice(std::slice::from_ref(&x_elem));
            NTT126Ifma::ntt126_ifma_mul_bbc(meta, 1, &mut product, x_u32, &a_u32[8 * n_i..8 * n_i + 8]);
            res_slice[n_i] = Q120bScalar(product);
        }
    }
}
