use poulpy_hal::{
    layouts::{
        Module, Ring, ScalarZnxBackendRef, ScratchArena, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendRef,
        VecZnxDftBackendMut, VecZnxDftBackendRef, check_degree,
    },
    oep::{HalSvpImpl, HalVecZnxDftImpl},
};

use super::operand::{dft_column_mut, dft_column_ref, empty_dft_like, scalar_column, svp_column};
use crate::{
    NTT4x30Gpu,
    module::handle,
    ntt4x30::{DftTerm, combine, forward, scale},
    runtime::GpuRuntime,
};

unsafe impl<RT: GpuRuntime, R: Ring> HalSvpImpl for NTT4x30Gpu<RT, R> {
    fn svp_prepare(
        module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        check_degree::<Self>(module.n(), res.n());
        assert!(a.n() == res.n(), "svp_prepare: operand degrees differ");
        let Some(res) = svp_column(res, res_col) else { return };
        let a = scalar_column(a, a_col).expect("svp_prepare: a holds no storage");
        forward::<RT>(handle(module).tables(), &res, &a, 1);
    }

    fn svp_ppol_copy(
        _module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert!(a.n() == res.n(), "svp_ppol_copy: operand degrees differ");
        assert!(a.hint() == res.hint(), "svp_ppol_copy: operand hints differ");
        let Some(res) = svp_column(res, res_col) else { return };
        let a = svp_column(a, a_col).expect("svp_ppol_copy: a holds no storage");
        combine::<RT>(&res, 1, DftTerm::plus(&a), DftTerm::zero(&a));
    }

    fn svp_apply_dft_tmp_bytes(_module: &Module<Self>, _b_size: usize) -> usize {
        0
    }

    fn svp_apply_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        // `a` is checked before the transform of `b` lands in `res`, which is then scaled in place.
        assert!(a.n() == res.n(), "svp_apply_dft: operand degrees differ");
        svp_column(a, a_col);
        Self::vec_znx_dft_apply(module, 1, 0, res, res_col, b, b_col);
        Self::svp_apply_dft_to_dft_assign(module, res, res_col, a, a_col);
    }

    fn svp_apply_dft_to_dft(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        assert!(
            a.n() == res.n() && b.n() == res.n(),
            "svp_apply_dft_to_dft: operand degrees differ"
        );
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let a = svp_column(a, a_col).expect("svp_apply_dft_to_dft: a holds no storage");
        let b = dft_column_ref(b, b_col).unwrap_or_else(|| empty_dft_like(&res));
        scale::<RT>(&res, &a, &b);
    }

    fn svp_apply_dft_to_dft_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert!(a.n() == res.n(), "svp_apply_dft_to_dft_assign: operand degrees differ");
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let a = svp_column(a, a_col).expect("svp_apply_dft_to_dft_assign: a holds no storage");
        scale::<RT>(&res, &a, &res);
    }
}
