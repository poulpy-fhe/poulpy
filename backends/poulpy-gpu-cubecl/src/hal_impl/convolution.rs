use poulpy_hal::{
    layouts::{
        CnvDftAccTerm, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, Module, Ring,
        ScratchArena, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, assert_dense, check_degree, check_galois,
    },
    oep::HalConvolutionImpl,
};

use super::operand::{
    all_blocks, all_dft_blocks, big_column_mut, cnv_left_column, cnv_right_column, column_ref, dft_column_mut, empty_dft_like,
};
use crate::{
    NTT4x30Gpu,
    module::handle,
    ntt4x30::{
        AutomorphismPlan, DftColumn, DftTerm, combine, convolution, convolution_pairwise, convolution_sum, convolution_tensor,
        forward,
    },
    runtime::GpuRuntime,
    znx::{Term, convolution_by_const, lincomb},
};

/// The three product columns of a rank-1 tensor in one launch: columns 0 and 1
/// of `diag` receive `a0 * b0` and `a1 * b1`, column 0 of `cross` receives
/// `(a0 + a1) * (b0 + b1) - diag0 - diag1`.
// The body of the core hook `GLWETensoringImpl::glwe_mul_columns`.
#[cfg_attr(not(feature = "enable-core"), allow(dead_code))]
pub(crate) fn glwe_mul_columns<RT: GpuRuntime, R: Ring>(
    cnv_offset: usize,
    diag: &mut VecZnxDftBackendMut<'_, NTT4x30Gpu<RT, R>>,
    cross: &mut VecZnxDftBackendMut<'_, NTT4x30Gpu<RT, R>>,
    a: &CnvPVecLBackendRef<'_, NTT4x30Gpu<RT, R>>,
    b: &CnvPVecRBackendRef<'_, NTT4x30Gpu<RT, R>>,
) {
    assert!(
        diag.cols() >= 2 && a.cols() >= 2 && b.cols() >= 2,
        "glwe_mul_columns: rank-1 operands have two columns"
    );
    assert!(
        a.n() == diag.n() && cross.n() == diag.n(),
        "glwe_mul_columns: operand degrees differ"
    );
    let column = |column: Option<DftColumn>| column.expect("glwe_mul_columns: an operand holds no storage");
    convolution_tensor::<RT>(
        &column(dft_column_mut(diag, 0)),
        &column(dft_column_mut(diag, 1)),
        &column(dft_column_mut(cross, 0)),
        &column(cnv_left_column(a, 0)),
        &column(cnv_left_column(a, 1)),
        &column(cnv_right_column(b, 0)),
        &column(cnv_right_column(b, 1)),
        cnv_offset,
    );
}

/// Transforms every column of `a` into the prepared vector whose column 0 is
/// `res0`, truncating or zero-extending to its limbs.
fn prepare<RT: GpuRuntime, R: Ring>(
    module: &Module<NTT4x30Gpu<RT, R>>,
    res0: &DftColumn,
    cols: usize,
    a: &VecZnxBackendRef<'_, NTT4x30Gpu<RT, R>>,
) {
    assert_dense(a, "cnv_prepare");
    check_degree::<NTT4x30Gpu<RT, R>>(module.n(), res0.n);
    assert_eq!(a.n(), res0.n, "cnv_prepare: a.n() != res.n()");
    assert_eq!(a.cols(), cols, "cnv_prepare: res.cols() != a.cols()");
    let limbs = res0.size.min(a.size());
    let all = all_dft_blocks(res0, res0.size);
    if let Some(a0) = column_ref(a, 0).filter(|_| limbs != 0) {
        // Both vectors order their blocks alike: one batched transform.
        forward::<RT>(handle(module).tables(), &all, &all_blocks(&a0, cols, limbs), limbs * cols);
    }
    let tail = all.limbs(limbs * cols, 1, all.size - limbs * cols);
    combine::<RT>(&tail, tail.size, DftTerm::zero(&tail), DftTerm::zero(&tail));
}

/// The read of a right operand through `sigma_{b_galois}`, at its own degree.
fn right_plan<RT: GpuRuntime, R: Ring>(
    module: &Module<NTT4x30Gpu<RT, R>>,
    b: &CnvPVecRBackendRef<'_, NTT4x30Gpu<RT, R>>,
    b_galois: i64,
) -> AutomorphismPlan {
    check_galois(b_galois);
    AutomorphismPlan::new(b.n(), b_galois, handle(module).tables().invariant())
}

unsafe impl<RT: GpuRuntime, R: Ring> HalConvolutionImpl for NTT4x30Gpu<RT, R> {
    fn cnv_prepare_left_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    fn cnv_prepare_left(
        module: &Module<Self>,
        res: &mut CnvPVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_eq!(a.cols(), res.cols(), "cnv_prepare_left: res.cols() != a.cols()");
        let Some(res0) = cnv_left_column(res, 0) else { return };
        prepare(module, &res0, res.cols(), a);
    }

    fn cnv_prepare_right_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    fn cnv_prepare_right(
        module: &Module<Self>,
        res: &mut CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_eq!(a.cols(), res.cols(), "cnv_prepare_right: res.cols() != a.cols()");
        let Some(res0) = cnv_right_column(res, 0) else { return };
        prepare(module, &res0, res.cols(), a);
    }

    fn cnv_prepare_self_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    fn cnv_prepare_self(
        module: &Module<Self>,
        left: &mut CnvPVecLBackendMut<'_, Self>,
        right: &mut CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_eq!(right.size(), left.size(), "cnv_prepare_self: right.size() != left.size()");
        assert_eq!(right.cols(), left.cols(), "cnv_prepare_self: right.cols() != left.cols()");
        assert_eq!(right.n(), left.n(), "cnv_prepare_self: right.n() != left.n()");
        assert_eq!(a.cols(), left.cols(), "cnv_prepare_self: left.cols() != a.cols()");
        let Some(left0) = cnv_left_column(left, 0) else { return };
        prepare(module, &left0, left.cols(), a);
        // Both prepared forms are the transform: the right one is a copy.
        let Some(right0) = cnv_right_column(right, 0) else { return };
        let (from, to) = (all_dft_blocks(&left0, left0.size), all_dft_blocks(&right0, right0.size));
        combine::<RT>(&to, to.size, DftTerm::plus(&from), DftTerm::zero(&from));
    }

    fn cnv_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        _res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    fn cnv_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        b_galois: i64,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_eq!(a.n(), res.n(), "cnv_apply_dft: a.n() != res.n()");
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let a = cnv_left_column(a, a_col).unwrap_or_else(|| empty_dft_like(&res));
        let plan = right_plan(module, b, b_galois);
        let b = cnv_right_column(b, b_col).unwrap_or_else(|| empty_dft_like(&res));
        convolution::<RT>(&res, &a, &b, &plan, cnv_offset, false);
    }

    fn cnv_apply_dft_add_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        _res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    fn cnv_apply_dft_add(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        b_galois: i64,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_eq!(a.n(), res.n(), "cnv_apply_dft_add: a.n() != res.n()");
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let plan = right_plan(module, b, b_galois);
        let (Some(a), Some(b)) = (cnv_left_column(a, a_col), cnv_right_column(b, b_col)) else {
            return;
        };
        convolution::<RT>(&res, &a, &b, &plan, cnv_offset, true);
    }

    fn cnv_apply_dft_sum_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        _res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    fn cnv_apply_dft_sum(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        terms: &[CnvDftAccTerm<'_, Self>],
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        // Every term is checked before the first launch writes `res`.
        terms.iter().for_each(|term| check_galois(term.b_galois));
        let Some(res) = dft_column_mut(res, res_col) else { return };
        // A term with an operand without storage is zero.
        let terms: Vec<(DftColumn, DftColumn, AutomorphismPlan)> = terms
            .iter()
            .filter_map(|term| {
                assert_eq!(term.a.n(), res.n, "cnv_apply_dft_sum: a.n() != res.n()");
                let plan = right_plan(module, &term.b, term.b_galois);
                Some((
                    cnv_left_column(&term.a, term.a_col)?,
                    cnv_right_column(&term.b, term.b_col)?,
                    plan,
                ))
            })
            .collect();
        if terms.is_empty() {
            combine::<RT>(&res, res.size, DftTerm::zero(&res), DftTerm::zero(&res));
            return;
        }
        convolution_sum::<RT>(&res, &terms, cnv_offset, false);
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        _res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    fn cnv_pairwise_apply_dft(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        i: usize,
        j: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_eq!(a.n(), res.n(), "cnv_pairwise_apply_dft: a.n() != res.n()");
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let left = |col: usize| cnv_left_column(a, col).unwrap_or_else(|| empty_dft_like(&res));
        let right = |col: usize| cnv_right_column(b, col).unwrap_or_else(|| empty_dft_like(&res));
        if i == j {
            convolution::<RT>(
                &res,
                &left(i),
                &right(i),
                &AutomorphismPlan::new(b.n(), 1, false),
                cnv_offset,
                false,
            );
        } else {
            convolution_pairwise::<RT>(&res, &left(i), &left(j), &right(i), &right(j), cnv_offset);
        }
    }

    fn cnv_by_const_apply_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        _res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    fn cnv_by_const_apply(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        by_const::<RT, R>(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, false);
    }

    fn cnv_by_const_apply_add_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        _res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    fn cnv_by_const_apply_add(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        by_const::<RT, R>(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, true);
    }
}

#[allow(clippy::too_many_arguments)]
fn by_const<RT: GpuRuntime, R: Ring>(
    cnv_offset: usize,
    res: &mut VecZnxBigBackendMut<'_, NTT4x30Gpu<RT, R>>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, NTT4x30Gpu<RT, R>>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT4x30Gpu<RT, R>>,
    b_col: usize,
    b_coeff: usize,
    accumulate: bool,
) {
    assert_dense(res, "cnv_by_const_apply");
    assert_dense(a, "cnv_by_const_apply");
    assert_dense(b, "cnv_by_const_apply");
    assert_eq!(a.n(), res.n(), "cnv_by_const_apply: a.n() != res.n()");
    let Some(res) = big_column_mut(res, res_col) else { return };
    let (Some(a), Some(b)) = (column_ref(a, a_col), column_ref(b, b_col)) else {
        if !accumulate {
            lincomb::<RT>(&res, res.size, Term::zero(&res), Term::zero(&res));
        }
        return;
    };
    convolution_by_const::<RT>(&res, &a, &b, b_coeff, cnv_offset, accumulate);
}
