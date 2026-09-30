use crate::reference::fft64::ring_arith::Fft64RingArith;
use crate::{
    layouts::{
        Backend, HostDataMut, HostDataRef, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, ZnxView, ZnxViewMut,
    },
    reference::fft64::{module::FFT64Plan, reim::ReimArith},
};

pub fn svp_prepare<'r, 'a, BE>(
    plan: &FFT64Plan<f64, BE::Ring>,
    res: &mut SvpPPolBackendMut<'r, BE>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Fft64RingArith,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    BE::reim_from_znx(res.at_mut(res_col, 0), a.at(a_col, 0));
    BE::fft64_forward(plan, res.at_mut(res_col, 0));
}

pub fn svp_apply_dft_to_dft<'r, 'a, BE>(
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpPPolBackendRef<'a, BE>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'a, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Fft64RingArith,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let res_size: usize = res.size();
    let b_size: usize = b.size();
    let min_size: usize = res_size.min(b_size);

    let ppol: &[f64] = a.at(a_col, 0);
    for j in 0..min_size {
        BE::fft64_mul(res.at_mut(res_col, j), ppol, b.at(b_col, j));
    }

    for j in min_size..res_size {
        BE::reim_zero(res.at_mut(res_col, j));
    }
}

pub fn svp_apply_dft_to_dft_assign<'r, 'a, BE>(
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpPPolBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Fft64RingArith,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let ppol: &[f64] = a.at(a_col, 0);
    for j in 0..res.size() {
        BE::fft64_mul_assign(res.at_mut(res_col, j), ppol);
    }
}
