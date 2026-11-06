use poulpy_hal::{
    layouts::{
        Module, Ring, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut,
        VecZnxBigBackendRef,
    },
    oep::HalVecZnxBigImpl,
};

use super::operand::{
    automorphism_map, big_column_mut, big_column_ref, column_mut, column_ref, empty_like, permute_assign, scalar_column,
};
use crate::{
    NTT4x30Gpu,
    runtime::GpuRuntime,
    znx::{Term, col_weighted_sum, inner_sum, lincomb, normalize, permute, scalar_product},
};

unsafe impl<RT: GpuRuntime, R: Ring> HalVecZnxBigImpl for NTT4x30Gpu<RT, R> {
    fn vec_znx_big_from_small(
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = big_column_mut(res, res_col) else { return };
        let a = column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        lincomb::<RT>(&res, res.size, Term::plus(&a), Term::zero(&res));
    }

    fn vec_znx_big_add(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let Some(res) = big_column_mut(res, res_col) else { return };
        let a = big_column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        let b = big_column_ref(b, b_col).unwrap_or_else(|| empty_like(&res));
        lincomb::<RT>(&res, res.size, Term::plus(&a), Term::plus(&b));
    }

    fn vec_znx_big_add_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = big_column_mut(res, res_col) else { return };
        let Some(a) = big_column_ref(a, a_col) else { return };
        lincomb::<RT>(&res, res.size.min(a.size), Term::plus(&res), Term::plus(&a));
    }

    fn vec_znx_big_add_small_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = big_column_mut(res, res_col) else { return };
        let Some(a) = column_ref(a, a_col) else { return };
        lincomb::<RT>(&res, res.size.min(a.size), Term::plus(&res), Term::plus(&a));
    }

    fn vec_znx_big_sub(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let Some(res) = big_column_mut(res, res_col) else { return };
        let a = big_column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        let b = big_column_ref(b, b_col).unwrap_or_else(|| empty_like(&res));
        lincomb::<RT>(&res, res.size, Term::plus(&a), Term::minus(&b));
    }

    fn vec_znx_big_sub_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = big_column_mut(res, res_col) else { return };
        let Some(a) = big_column_ref(a, a_col) else { return };
        lincomb::<RT>(&res, res.size.min(a.size), Term::plus(&res), Term::minus(&a));
    }

    fn vec_znx_big_sub_negate_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = big_column_mut(res, res_col) else { return };
        let a = big_column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        lincomb::<RT>(&res, res.size, Term::plus(&a), Term::minus(&res));
    }

    fn vec_znx_big_sub_small_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = big_column_mut(res, res_col) else { return };
        let Some(a) = column_ref(a, a_col) else { return };
        lincomb::<RT>(&res, res.size.min(a.size), Term::plus(&res), Term::minus(&a));
    }

    fn vec_znx_big_sub_small_negate_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = big_column_mut(res, res_col) else { return };
        let a = column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        lincomb::<RT>(&res, res.size, Term::plus(&a), Term::minus(&res));
    }

    fn vec_znx_big_inner_sum(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        res_coeff: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        poulpy_hal::layouts::assert_dense(res, "vec_znx_big_inner_sum");
        poulpy_hal::layouts::assert_dense(a, "vec_znx_big_inner_sum");
        let Some(res) = big_column_mut(res, res_col) else { return };
        let a = big_column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        inner_sum::<RT>(&res, res_coeff, &a);
    }

    fn vec_znx_big_col_weighted_sum(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        weights: &ScalarZnxBackendRef<'_, Self>,
        weights_col: usize,
        cols: usize,
        coeffs: usize,
    ) {
        poulpy_hal::layouts::assert_dense(res, "vec_znx_big_col_weighted_sum");
        poulpy_hal::layouts::assert_dense(a, "vec_znx_big_col_weighted_sum");
        assert!(
            cols <= a.cols().min(weights.n()),
            "vec_znx_big_col_weighted_sum: cols out of range"
        );
        assert!(
            weights_col < weights.cols(),
            "vec_znx_big_col_weighted_sum: weights_col out of range"
        );
        let Some(res) = big_column_mut(res, res_col) else { return };
        let a_col_stride = a.shape().n_full() as u32;
        // The kernel reads columns `0..cols`: the last one is checked against the storage as column 0 is.
        if cols > 0 {
            let _ = column_ref(a, cols - 1);
        }
        let a0 = column_ref(a, 0).unwrap_or_else(|| empty_like(&res));
        let weights = scalar_column(weights, weights_col).unwrap_or_else(|| empty_like(&res));
        col_weighted_sum::<RT>(&res, &a0, a_col_stride, &weights.binding, weights.base, cols, coeffs);
    }

    fn vec_znx_big_scalar_product(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &ScalarZnxBackendRef<'_, Self>,
        b_col: usize,
    ) {
        poulpy_hal::layouts::assert_dense(res, "vec_znx_big_scalar_product");
        poulpy_hal::layouts::assert_dense(a, "vec_znx_big_scalar_product");
        assert!(b.n() == a.n(), "operand degree differs from the result degree");
        assert!(b_col < b.cols(), "vec_znx_big_scalar_product: b_col out of range");
        let Some(res) = big_column_mut(res, res_col) else { return };
        let Some(a) = column_ref(a, a_col) else { return };
        let Some(b) = scalar_column(b, b_col) else { return };
        scalar_product::<RT>(&res, &a, &b.binding, b.base);
    }

    fn vec_znx_big_negate(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = big_column_mut(res, res_col) else { return };
        let a = big_column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        assert!(a.n == res.n, "operand degree differs from the result degree");
        lincomb::<RT>(&res, res.size, Term::minus(&a), Term::zero(&a));
    }

    fn vec_znx_big_negate_assign(_module: &Module<Self>, a: &mut VecZnxBigBackendMut<'_, Self>, a_col: usize) {
        let Some(a) = big_column_mut(a, a_col) else { return };
        lincomb::<RT>(&a, a.size, Term::minus(&a), Term::zero(&a));
    }

    fn vec_znx_big_normalize_tmp_bytes(_module: &Module<Self>) -> usize {
        0
    }

    fn vec_znx_big_normalize(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_base2k: usize,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        let Some(res) = column_mut(res, res_col) else { return };
        let a = big_column_ref(a, a_col).unwrap_or_else(|| {
            let mut empty = empty_like(&res);
            empty.words = 2;
            empty
        });
        normalize::<RT>(&res, res_base2k, res_k, res_offset, &a, a_base2k);
    }

    fn vec_znx_big_automorphism(
        _module: &Module<Self>,
        k: i64,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        poulpy_hal::layouts::assert_dense(res, "vec_znx_big_automorphism");
        poulpy_hal::layouts::assert_dense(a, "vec_znx_big_automorphism");
        assert!(a.n() == res.n(), "operand degree differs from the result degree");
        let Some(res) = big_column_mut(res, res_col) else { return };
        let a = big_column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        permute::<RT>(&res, &a, automorphism_map::<R>(k));
    }

    fn vec_znx_big_automorphism_assign_tmp_bytes(_module: &Module<Self>) -> usize {
        0
    }

    fn vec_znx_big_automorphism_assign(
        module: &Module<Self>,
        k: i64,
        a: &mut VecZnxBigBackendMut<'_, Self>,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        poulpy_hal::layouts::assert_dense(a, "vec_znx_big_automorphism_assign");
        let Some(a) = big_column_mut(a, a_col) else { return };
        permute_assign(module, &a, automorphism_map::<R>(k));
    }
}
