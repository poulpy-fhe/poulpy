use bytemuck::cast_slice_mut;

use crate::{
    layouts::{
        Backend, HostDataMut, HostDataRef, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef,
        ZnxView, ZnxViewMut,
    },
    reference::{
        fft64::reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
        znx::ZnxZero,
    },
};

pub fn vec_znx_dft_add_into<BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, BE>,
    b_col: usize,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
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

pub fn vec_znx_dft_add_assign<BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
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
pub fn vec_znx_dft_add_scaled_assign<BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
    a_scale: i64,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
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

pub fn vec_znx_dft_copy<BE>(
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
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

pub fn vec_znx_dft_apply<BE>(
    table: &ReimFFTTable<f64>,
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
{
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

pub fn vec_znx_idft_apply<BE>(
    table: &ReimIFFTTable<f64>,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64> + ZnxZero,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
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

pub fn vec_znx_idft_apply_tmpa<BE>(
    table: &ReimIFFTTable<f64>,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &mut VecZnxDftBackendMut<'_, BE>,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64> + ZnxZero,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
{
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

// Kept as dormant internal code for the removed consume path.
// It is intentionally retained because the in-place DFT -> big conversion
// may still be useful as a future optimization, even though the current
// public API now applies IDFT into a separately allocated VecZnxBig.
#[allow(dead_code)]
pub fn vec_znx_idft_apply_consume<'a, BE>(
    table: &ReimIFFTTable<f64>,
    mut res: VecZnxDftBackendMut<'a, BE>,
) -> VecZnxBigBackendMut<'a, BE>
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64>,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
{
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

    res.into_big()
}

pub fn vec_znx_dft_sub<BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, BE>,
    b_col: usize,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
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

pub fn vec_znx_dft_sub_assign<BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
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

pub fn vec_znx_dft_sub_negate_assign<BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
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

pub fn vec_znx_dft_zero<BE>(res: &mut VecZnxDftBackendMut<'_, BE>, res_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
{
    for j in 0..res.size() {
        BE::reim_zero(res.at_mut(res_col, j))
    }
}
