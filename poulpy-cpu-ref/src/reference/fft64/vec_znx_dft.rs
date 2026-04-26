use bytemuck::cast_slice_mut;

use crate::{
    layouts::{
        Backend, Data, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, ZnxInfos,
        ZnxView, ZnxViewMut,
    },
    reference::{
        fft64::reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
        znx::ZnxZero,
    },
};

pub fn vec_znx_dft_add_into<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    B: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(b.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    if a_size <= b_size {
        let sum_size: usize = a_size.min(res_size);
        let cpy_size: usize = b_size.min(res_size);

        for j in 0..sum_size {
            BE::reim_add(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            BE::reim_copy(res.at_mut(res_col, j), b.at(b_col, j));
        }

        for j in cpy_size..res_size {
            BE::reim_zero(res.at_mut(res_col, j));
        }
    } else {
        let sum_size: usize = b_size.min(res_size);
        let cpy_size: usize = a_size.min(res_size);

        for j in 0..sum_size {
            BE::reim_add(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            BE::reim_copy(res.at_mut(res_col, j), a.at(a_col, j));
        }

        for j in cpy_size..res_size {
            BE::reim_zero(res.at_mut(res_col, j));
        }
    }
}

pub fn vec_znx_dft_add_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let sum_size: usize = a_size.min(res_size);

    for j in 0..sum_size {
        BE::reim_add_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

/// res = res + a * 2^{a_scale * base2k}.
pub fn vec_znx_dft_add_scaled_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, a_scale: i64)
where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    if a_scale > 0 {
        let shift: usize = (a_scale as usize).min(a_size);
        let sum_size: usize = a_size.min(res_size).saturating_sub(shift);
        for j in 0..sum_size {
            BE::reim_add_assign(res.at_mut(res_col, j), a.at(a_col, j + shift));
        }
    } else if a_scale < 0 {
        let shift: usize = (a_scale.unsigned_abs() as usize).min(res_size);
        let sum_size: usize = a_size.min(res_size.saturating_sub(shift));
        for j in 0..sum_size {
            BE::reim_add_assign(res.at_mut(res_col, j + shift), a.at(a_col, j));
        }
    } else {
        let sum_size: usize = a_size.min(res_size);
        for j in 0..sum_size {
            BE::reim_add_assign(res.at_mut(res_col, j), a.at(a_col, j));
        }
    }
}

pub fn vec_znx_dft_copy<R, A, BE>(step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let steps: usize = a.size().div_ceil(step);
    let min_steps: usize = res.size().min(steps);

    (0..min_steps).for_each(|j| {
        let limb: usize = offset + j * step;
        if limb < a.size() {
            BE::reim_copy(res.at_mut(res_col, j), a.at(a_col, limb));
        } else {
            BE::reim_zero(res.at_mut(res_col, j));
        }
    });
    (min_steps..res.size()).for_each(|j| {
        BE::reim_zero(res.at_mut(res_col, j));
    })
}

pub fn vec_znx_dft_apply<R, A, BE>(
    table: &ReimFFTTable<f64>,
    step: usize,
    offset: usize,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert!(step > 0);
        assert_eq!(table.m() << 1, res.n());
        assert_eq!(a.n(), res.n());
    }

    let a_size: usize = a.size();
    let res_size: usize = res.size();

    let steps: usize = a_size.div_ceil(step);
    let min_steps: usize = res_size.min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a_size {
            BE::reim_from_znx(res.at_mut(res_col, j), a.at(a_col, limb));
            BE::reim_dft_execute(table, res.at_mut(res_col, j));
        }
    }

    (min_steps..res.size()).for_each(|j| {
        BE::reim_zero(res.at_mut(res_col, j));
    });
}

pub fn vec_znx_idft_apply<R, A, BE>(table: &ReimIFFTTable<f64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64> + ZnxZero,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(table.m() << 1, res.n());
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let min_size: usize = res_size.min(a.size());

    let divisor: f64 = table.m() as f64;

    for j in 0..min_size {
        let res_slice_f64: &mut [f64] = cast_slice_mut(res.at_mut(res_col, j));
        BE::reim_copy(res_slice_f64, a.at(a_col, j));
        BE::reim_dft_execute(table, res_slice_f64);
        BE::reim_to_znx_assign(res_slice_f64, divisor);
    }

    for j in min_size..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_idft_apply_tmpa<R, A, BE>(table: &ReimIFFTTable<f64>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64> + ZnxZero,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToMut<BE>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let mut a: VecZnxDft<&mut [u8], BE> = a.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(table.m() << 1, res.n());
        assert_eq!(a.n(), res.n());
    }

    let res_size = res.size();
    let min_size: usize = res_size.min(a.size());

    let divisor: f64 = table.m() as f64;

    for j in 0..min_size {
        BE::reim_dft_execute(table, a.at_mut(a_col, j));
        BE::reim_to_znx(res.at_mut(res_col, j), divisor, a.at(a_col, j));
    }

    for j in min_size..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_idft_apply_consume<D: Data, BE>(table: &ReimIFFTTable<f64>, mut res: VecZnxDft<D, BE>) -> VecZnxBig<D, BE>
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64>,
    VecZnxDft<D, BE>: VecZnxDftToMut<BE>,
{
    {
        let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(table.m() << 1, res.n());
        }

        let divisor: f64 = table.m() as f64;

        for i in 0..res.cols() {
            for j in 0..res.size() {
                BE::reim_dft_execute(table, res.at_mut(i, j));
                BE::reim_to_znx_assign(res.at_mut(i, j), divisor);
            }
        }
    }

    res.into_big()
}

pub fn vec_znx_dft_sub<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    B: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(b.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    if a_size <= b_size {
        let sum_size: usize = a_size.min(res_size);
        let cpy_size: usize = b_size.min(res_size);

        for j in 0..sum_size {
            BE::reim_sub(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            BE::reim_negate(res.at_mut(res_col, j), b.at(b_col, j));
        }

        for j in cpy_size..res_size {
            BE::reim_zero(res.at_mut(res_col, j));
        }
    } else {
        let sum_size: usize = b_size.min(res_size);
        let cpy_size: usize = a_size.min(res_size);

        for j in 0..sum_size {
            BE::reim_sub(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            BE::reim_copy(res.at_mut(res_col, j), a.at(a_col, j));
        }

        for j in cpy_size..res_size {
            BE::reim_zero(res.at_mut(res_col, j));
        }
    }
}

pub fn vec_znx_dft_sub_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let sum_size: usize = a_size.min(res_size);

    for j in 0..sum_size {
        BE::reim_sub_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

pub fn vec_znx_dft_sub_negate_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let sum_size: usize = a_size.min(res_size);

    for j in 0..sum_size {
        BE::reim_sub_negate_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in sum_size..res_size {
        BE::reim_negate_assign(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_dft_zero<R, BE>(res: &mut R, res_col: usize)
where
    R: VecZnxDftToMut<BE>,
    BE: Backend<ScalarPrep = f64> + ReimArith,
{
    let res: &mut VecZnxDft<&mut [u8], BE> = &mut res.to_mut();
    for j in 0..res.size() {
        BE::reim_zero(res.at_mut(res_col, j))
    }
}
