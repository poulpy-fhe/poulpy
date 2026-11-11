use crate::kernels::fft64::ring_arith::Fft64RingArith;
use crate::{
    kernels::fft64::{module::FFT64Plan, reim::ReimArith},
    layouts::{
        Backend, HostDataMut, HostDataRef, Ring, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, ZnxView, ZnxViewMut,
    },
};

pub fn svp_ppol_prepare_portable<'r, 'a, BE, Rg: Ring>(
    plan: &FFT64Plan<f64, Rg>,
    res: &mut SvpPPolBackendMut<'r, BE, Rg>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'a, BE, Rg>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Fft64RingArith<Rg>,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    BE::reim_from_znx(res.at_mut(res_col, 0), a.at(a_col, 0));
    BE::fft64_forward(plan, res.at_mut(res_col, 0));
}

pub fn svp_apply_dft_to_dft_portable<'r, 'a, BE, Rg: Ring>(
    res: &mut VecZnxDftBackendMut<'r, BE, Rg>,
    res_col: usize,
    a: &SvpPPolBackendRef<'a, BE, Rg>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'a, BE, Rg>,
    b_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Fft64RingArith<Rg>,
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

pub fn svp_apply_dft_to_dft_assign_portable<'r, 'a, BE, Rg: Ring>(
    res: &mut VecZnxDftBackendMut<'r, BE, Rg>,
    res_col: usize,
    a: &SvpPPolBackendRef<'a, BE, Rg>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Fft64RingArith<Rg>,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let ppol: &[f64] = a.at(a_col, 0);
    for j in 0..res.size() {
        BE::fft64_mul_assign(res.at_mut(res_col, j), ppol);
    }
}
