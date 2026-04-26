use crate::{
    layouts::{
        Backend, ScalarZnx, ScalarZnxToRef, SvpPPol, SvpPPolToMut, SvpPPolToRef, VecZnx, VecZnxDft, VecZnxDftToMut,
        VecZnxDftToRef, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut,
    },
    reference::fft64::reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
};

pub fn svp_prepare<R, A, BE>(table: &ReimFFTTable<f64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
    R: SvpPPolToMut<BE>,
    A: ScalarZnxToRef,
{
    let mut res: SvpPPol<&mut [u8], BE> = res.to_mut();
    let a: ScalarZnx<&[u8]> = a.to_ref();
    BE::reim_from_znx(res.at_mut(res_col, 0), a.at(a_col, 0));
    BE::reim_dft_execute(table, res.at_mut(res_col, 0));
}

pub fn svp_apply_dft<R, A, B, BE>(
    table: &ReimFFTTable<f64>,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
    R: VecZnxDftToMut<BE>,
    A: SvpPPolToRef<BE>,
    B: VecZnxToRef,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: SvpPPol<&[u8], BE> = a.to_ref();
    let b: VecZnx<&[u8]> = b.to_ref();

    let res_size: usize = res.size();
    let b_size: usize = b.size();
    let min_size: usize = res_size.min(b_size);

    let ppol: &[f64] = a.at(a_col, 0);
    for j in 0..min_size {
        let out: &mut [f64] = res.at_mut(res_col, j);
        BE::reim_from_znx(out, b.at(b_col, j));
        BE::reim_dft_execute(table, out);
        BE::reim_mul_assign(out, ppol);
    }

    for j in min_size..res_size {
        BE::reim_zero(res.at_mut(res_col, j));
    }
}

pub fn svp_apply_dft_to_dft<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    R: VecZnxDftToMut<BE>,
    A: SvpPPolToRef<BE>,
    B: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: SvpPPol<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    let res_size: usize = res.size();
    let b_size: usize = b.size();
    let min_size: usize = res_size.min(b_size);

    let ppol: &[f64] = a.at(a_col, 0);
    for j in 0..min_size {
        BE::reim_mul(res.at_mut(res_col, j), ppol, b.at(b_col, j));
    }

    for j in min_size..res_size {
        BE::reim_zero(res.at_mut(res_col, j));
    }
}

pub fn svp_apply_dft_to_dft_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    R: VecZnxDftToMut<BE>,
    A: SvpPPolToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: SvpPPol<&[u8], BE> = a.to_ref();

    let ppol: &[f64] = a.at(a_col, 0);
    for j in 0..res.size() {
        BE::reim_mul_assign(res.at_mut(res_col, j), ppol);
    }
}
