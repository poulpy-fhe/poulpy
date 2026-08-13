//! Scalar-vector product kernels for the FFT64 backend.
//!
//! Every kernel takes its prepared operand as the concrete layout type: the
//! `ppol` set consumes [`SvpPPol`](crate::layouts::SvpPPol), the `tpol` set
//! [`SvpTPol`](crate::layouts::SvpTPol). Both tiers are the same reim DFT on
//! this backend, so each pair forwards to one private inner routine; a backend
//! that later builds them differently splits the pair.

use crate::{
    layouts::{
        Backend, HostDataMut, HostDataRef, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, SvpTPolBackendMut,
        SvpTPolBackendRef, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView, ZnxViewMut,
    },
    reference::fft64::reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
};

fn prepare_inner<BE>(table: &ReimFFTTable<f64>, res: &mut [f64], a: &[i64])
where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
{
    BE::reim_from_znx(res, a);
    BE::reim_dft_execute(table, res);
}

fn small_to_dft_inner<'r, 'b, BE>(
    table: &ReimFFTTable<f64>,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    pol: &[f64],
    b: &VecZnxBackendRef<'b, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'b>: HostDataRef,
{
    let res_size: usize = res.size();
    let min_size: usize = res_size.min(b.size());

    for j in 0..min_size {
        let out: &mut [f64] = res.at_mut(res_col, j);
        BE::reim_from_znx(out, b.at(b_col, j));
        BE::reim_dft_execute(table, out);
        BE::reim_mul_assign(out, pol);
    }

    for j in min_size..res_size {
        BE::reim_zero(res.at_mut(res_col, j));
    }
}

fn dft_to_dft_inner<'r, 'b, BE>(
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    pol: &[f64],
    b: &VecZnxDftBackendRef<'b, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'b>: HostDataRef,
{
    let res_size: usize = res.size();
    let min_size: usize = res_size.min(b.size());

    for j in 0..min_size {
        BE::reim_mul(res.at_mut(res_col, j), pol, b.at(b_col, j));
    }

    for j in min_size..res_size {
        BE::reim_zero(res.at_mut(res_col, j));
    }
}

fn dft_to_dft_assign_inner<'r, BE>(res: &mut VecZnxDftBackendMut<'r, BE>, res_col: usize, pol: &[f64])
where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    BE::BufMut<'r>: HostDataMut,
{
    for j in 0..res.size() {
        BE::reim_mul_assign(res.at_mut(res_col, j), pol);
    }
}

/// Transforms a scalar polynomial into an [`SvpPPol`](crate::layouts::SvpPPol).
pub fn svp_prepare_ppol<'r, 'a, BE>(
    table: &ReimFFTTable<f64>,
    res: &mut SvpPPolBackendMut<'r, BE>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    prepare_inner::<BE>(table, res.at_mut(res_col, 0), a.at(a_col, 0));
}

/// Transforms a scalar polynomial into an [`SvpTPol`](crate::layouts::SvpTPol).
pub fn svp_prepare_tpol<'r, 'a, BE>(
    table: &ReimFFTTable<f64>,
    res: &mut SvpTPolBackendMut<'r, BE>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    prepare_inner::<BE>(table, res.at_mut(res_col, 0), a.at(a_col, 0));
}

/// `res = a * DFT(b)`.
pub fn svp_apply_ppol_small_to_dft<'r, 'a, 'b, BE>(
    table: &ReimFFTTable<f64>,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpPPolBackendRef<'a, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'b, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
    BE::BufMut<'r>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    small_to_dft_inner::<BE>(table, res, res_col, a.at(a_col, 0), b, b_col);
}

/// `res = a * DFT(b)`.
pub fn svp_apply_tpol_small_to_dft<'r, 'a, 'b, BE>(
    table: &ReimFFTTable<f64>,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpTPolBackendRef<'a, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'b, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
    BE::BufMut<'r>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    small_to_dft_inner::<BE>(table, res, res_col, a.at(a_col, 0), b, b_col);
}

/// `res = a * b`.
pub fn svp_apply_ppol_dft_to_dft<'r, 'a, 'b, BE>(
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpPPolBackendRef<'a, BE>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'b, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    BE::BufMut<'r>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    dft_to_dft_inner::<BE>(res, res_col, a.at(a_col, 0), b, b_col);
}

/// `res = a * b`.
pub fn svp_apply_tpol_dft_to_dft<'r, 'a, 'b, BE>(
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpTPolBackendRef<'a, BE>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'b, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    BE::BufMut<'r>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    dft_to_dft_inner::<BE>(res, res_col, a.at(a_col, 0), b, b_col);
}

/// `res = a * res`.
pub fn svp_apply_ppol_dft_to_dft_assign<'r, 'a, BE>(
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpPPolBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    dft_to_dft_assign_inner::<BE>(res, res_col, a.at(a_col, 0));
}

/// `res = a * res`.
pub fn svp_apply_tpol_dft_to_dft_assign<'r, 'a, BE>(
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpTPolBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    dft_to_dft_assign_inner::<BE>(res, res_col, a.at(a_col, 0));
}
