use poulpy_hal::{
    api::ScratchArenaTakeBasic,
    layouts::{
        Backend, MatZnxBackendRef, Module, Ring, ScratchArena, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VecZnxDftToBackendMut, VmpPMatBackendMut, VmpPMatBackendRef, assert_dense, check_degree,
    },
    oep::{HalVecZnxDftImpl, HalVmpImpl},
};

use super::operand::{all_blocks, all_dft_blocks, column_ref, dft_column_mut, dft_column_ref, mat_blocks, pmat};
use crate::{
    NTT4x30Gpu,
    module::handle,
    ntt4x30::{DftTerm, combine, digit_product, extract_rows, forward},
    runtime::GpuRuntime,
};

/// The interleaved-digit product of `a` by `pmat` into `res`, in one launch:
/// digit `di` gathers the limbs of `a` congruent to `dsize - 1 - di` modulo
/// `dsize`, digit 0 writes every limb of `res` from the matrix limbs
/// `0..res.size()`, and digit `di >= 1` accumulates the matrix limbs from `di`
/// on into the first
/// `min(res.size(), pmat.size() - max(dsize - di - product_limbs, 0))` limbs.
// The body of the core hook `GGLWEProductDigitsStridedImpl`.
#[cfg_attr(not(feature = "enable-core"), allow(dead_code))]
pub(crate) fn gglwe_product_digits_strided<RT: GpuRuntime, R: Ring>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Gpu<RT, R>>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Gpu<RT, R>>,
    dsize: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Gpu<RT, R>>,
) {
    product::<RT, R>(res, a, pmat, dsize, product_limbs, 0, false);
}

fn product<RT: GpuRuntime, R: Ring>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Gpu<RT, R>>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Gpu<RT, R>>,
    b: &VmpPMatBackendRef<'_, NTT4x30Gpu<RT, R>>,
    dsize: usize,
    product_limbs: usize,
    limb_offset: usize,
    accumulate: bool,
) {
    assert!(
        a.n() == res.n() && b.n() == res.n(),
        "vmp_apply_dft_to_dft: operand degrees differ"
    );
    assert!(a.cols() == b.cols_in(), "vmp_apply_dft_to_dft: a.cols() != pmat.cols_in()");
    assert!(
        res.cols() == b.cols_out(),
        "vmp_apply_dft_to_dft: res.cols() != pmat.cols_out()"
    );
    let Some(res) = dft_column_mut(res, 0) else { return };
    let Some(b) = pmat(b) else { return };
    let Some(a) = dft_column_ref(a, 0) else {
        if !accumulate {
            let all = all_dft_blocks(&res, res.size);
            combine::<RT>(&all, all.size, DftTerm::zero(&all), DftTerm::zero(&all));
        }
        return;
    };
    digit_product::<RT>(&res, &a, &b, dsize, product_limbs, limb_offset, accumulate);
}

unsafe impl<RT: GpuRuntime, R: Ring> HalVmpImpl for NTT4x30Gpu<RT, R> {
    fn vmp_prepare_tmp_bytes(_module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
        0
    }

    fn vmp_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        check_degree::<Self>(module.n(), res.n());
        assert!(
            a.n() == res.n()
                && a.rows() == res.rows()
                && a.cols_in() == res.cols_in()
                && a.cols_out() == res.cols_out()
                && a.size() == res.size(),
            "vmp_prepare: the matrices differ in degree or dimensions"
        );
        let Some(res) = pmat(res) else { return };
        let a = mat_blocks(a).expect("vmp_prepare: a holds no storage");
        // Both matrices order their polynomials alike: one batched transform.
        let blocks = res.blocks();
        forward::<RT>(handle(module).tables(), &blocks, &a, blocks.size);
    }

    fn vmp_apply_dft_tmp_bytes(
        module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        Self::bytes_of_vec_znx_dft(module.n(), b_cols_in, a_size.min(b_rows))
    }

    fn vmp_apply_dft<D>(
        module: &Module<Self>,
        res: &mut D,
        a: &VecZnxBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        D: VecZnxDftToBackendMut<Self>,
    {
        assert_dense(a, "vmp_apply_dft");
        check_degree::<Self>(module.n(), a.n());
        let (cols_in, limbs) = (b.cols_in(), a.size().min(b.rows()));
        let (mut a_dft, _) = scratch.borrow().take_vec_znx_dft_scratch(a.n(), cols_in, limbs);
        // Trailing columns of `a` align with the highest input columns of the matrix.
        let shared = a.cols().min(cols_in);
        let (first, skipped) = (cols_in - shared, a.cols() - shared);
        if let Some(a_dft0) = dft_column_mut(&a_dft, 0) {
            if first == 0 && skipped == 0 {
                // Same columns on both sides: every block in one batched transform.
                let source = column_ref(a, 0).expect("vmp_apply_dft: a holds no storage");
                let all = all_dft_blocks(&a_dft0, limbs);
                forward::<RT>(handle(module).tables(), &all, &all_blocks(&source, cols_in, limbs), all.size);
            } else {
                for col in 0..first {
                    Self::vec_znx_dft_zero(module, &mut a_dft, col);
                }
                for col in 0..shared {
                    Self::vec_znx_dft_apply(module, 1, 0, &mut a_dft, first + col, a, skipped + col);
                }
            }
        }
        let mut res = res.to_backend_mut();
        product::<RT, R>(
            &mut res,
            &poulpy_hal::layouts::vec_znx_dft_backend_ref_from_mut(&a_dft),
            b,
            1,
            1,
            0,
            false,
        );
    }

    fn vmp_apply_dft_to_dft_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        _a_size: usize,
        _b_rows: usize,
        _b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    fn vmp_apply_dft_to_dft(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        product::<RT, R>(res, a, b, 1, 1, limb_offset, false);
    }

    fn vmp_apply_dft_to_dft_add_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        _a_size: usize,
        _b_rows: usize,
        _b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    fn vmp_apply_dft_to_dft_add(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        product::<RT, R>(res, a, b, 1, 1, limb_offset, true);
    }

    fn vmp_extract_selected_rows(
        _module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &VmpPMatBackendRef<'_, Self>,
        first_row: usize,
        row_step: usize,
    ) {
        assert!(a.hint() == res.hint(), "vmp_extract_selected_rows: a.hint() != res.hint()");
        let Some(res) = pmat(res) else { return };
        let a = pmat(a).expect("vmp_extract_selected_rows: a holds no storage");
        extract_rows::<RT>(&res, &a, first_row, row_step);
    }

    fn vmp_zero(_module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        let Some(res) = pmat(res) else { return };
        let blocks = res.blocks();
        combine::<RT>(&blocks, blocks.size, DftTerm::zero(&blocks), DftTerm::zero(&blocks));
    }
}
