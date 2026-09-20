use crate::{
    layouts::{
        Backend, HostDataMut, HostDataRef, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, ZnxView, ZnxViewMut,
    },
    reference::fft64::{
        module::FFT64Plan,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
    },
};

pub fn svp_prepare<'r, 'a, BE>(
    plan: &FFT64Plan<f64>,
    res: &mut SvpPPolBackendMut<'r, BE>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64>
        + ReimArith
        + ReimFFTExecute<ReimFFTTable<f64>, f64>
        + ReimFFTExecute<ReimIFFTTable<f64>, f64>,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    BE::reim_from_znx(res.at_mut(res_col, 0), a.at(a_col, 0));
    plan.forward::<BE>(res.at_mut(res_col, 0));
}

pub fn svp_apply_dft_to_dft<'r, 'a, BE>(
    plan: &FFT64Plan<f64>,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpPPolBackendRef<'a, BE>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'a, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let res_size: usize = res.size();
    let b_size: usize = b.size();
    let min_size: usize = res_size.min(b_size);

    let ppol: &[f64] = a.at(a_col, 0);
    for j in 0..min_size {
        if plan.is_conjugate_invariant() {
            BE::reim_real_mul(res.at_mut(res_col, j), ppol, b.at(b_col, j));
        } else {
            BE::reim_mul(res.at_mut(res_col, j), ppol, b.at(b_col, j));
        }
    }

    for j in min_size..res_size {
        BE::reim_zero(res.at_mut(res_col, j));
    }
}

pub fn svp_apply_dft_to_dft_assign<'r, 'a, BE>(
    plan: &FFT64Plan<f64>,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpPPolBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let ppol: &[f64] = a.at(a_col, 0);
    for j in 0..res.size() {
        if plan.is_conjugate_invariant() {
            BE::reim_real_mul_assign(res.at_mut(res_col, j), ppol);
        } else {
            BE::reim_mul_assign(res.at_mut(res_col, j), ppol);
        }
    }
}
