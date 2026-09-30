use crate::kernels::fft64::ring_arith::Fft64RingArith;
use bytemuck::cast_slice_mut;

use crate::{
    kernels::{
        SendPtr,
        fft64::{module::FFT64Plan, reim::ReimArith},
        znx::ZnxZero,
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef,
        ZnxView, ZnxViewMut,
    },
};

pub fn vec_znx_dft_add<BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
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
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
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

pub fn vec_znx_dft_copy<BE>(
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    {
        assert_eq!(res.n(), a.n());
        assert!(step >= 1, "vec_znx_dft_copy: step must be >= 1");
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
    plan: &FFT64Plan<f64, BE::Ring>,
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Fft64RingArith + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
{
    poulpy_hal::layouts::assert_dense(a, "vec_znx_dft_apply");
    {
        assert!(step >= 1, "vec_znx_dft_apply: step must be >= 1");
        assert_eq!(plan.fft().m() << 1, res.n());
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
            BE::fft64_forward(plan, res.at_mut(res_col, j));
        } else {
            BE::reim_zero(res.at_mut(res_col, j));
        }
    }

    (min_steps..res.size()).for_each(|j| {
        BE::reim_zero(res.at_mut(res_col, j));
    });
}

pub fn vec_znx_idft_apply<BE>(
    plan: &FFT64Plan<f64, BE::Ring>,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, BigWord = i64, ZnxWord = i64> + ReimArith + Fft64RingArith + ZnxZero,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_idft_apply");
    {
        assert_eq!(plan.fft().m() << 1, res.n());
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let min_size: usize = res_size.min(a.size());

    let divisor = BE::fft64_divisor(plan);

    for j in 0..min_size {
        let res_slice_f64: &mut [f64] = cast_slice_mut(res.at_mut(res_col, j));
        BE::reim_copy(res_slice_f64, a.at(a_col, j));
        BE::fft64_inverse(plan, res_slice_f64);
        BE::reim_to_znx_assign(res_slice_f64, divisor);
    }

    for j in min_size..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_idft_apply_tmpa<BE>(
    plan: &FFT64Plan<f64, BE::Ring>,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &mut VecZnxDftBackendMut<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, BigWord = i64, ZnxWord = i64> + ReimArith + Fft64RingArith + ZnxZero,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_idft_apply_tmpa");
    {
        assert_eq!(plan.fft().m() << 1, res.n());
        assert_eq!(a.n(), res.n());
    }

    let res_size = res.size();
    let min_size: usize = res_size.min(a.size());

    let divisor = BE::fft64_divisor(plan);

    for j in 0..min_size {
        BE::fft64_inverse(plan, a.at_mut(a_col, j));
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
    plan: &FFT64Plan<f64, BE::Ring>,
    mut res: VecZnxDftBackendMut<'a, BE>,
) -> VecZnxBigBackendMut<'a, BE>
where
    BE: Backend<DftWord = f64, BigWord = i64, ZnxWord = i64> + ReimArith + Fft64RingArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
{
    {
        assert_eq!(plan.fft().m() << 1, res.n());
    }

    let divisor = BE::fft64_divisor(plan);

    for i in 0..res.cols() {
        for j in 0..res.size() {
            BE::fft64_inverse(plan, res.at_mut(i, j));
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
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
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
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
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
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
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
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
{
    for j in 0..res.size() {
        BE::reim_zero(res.at_mut(res_col, j))
    }
}

/// Precomputed permutation for `VecZnxDft` automorphism `tau_p : X -> X^p`
/// in the FFT64 half-spectrum layout.
///
/// `perm[i]` is the source complex slot that supplies output slot `i`. If
/// `conj` is set, the imaginary half is globally negated on apply (driven
/// by `p mod 4`). The conjugate-invariant layout is the real half of the
/// standard degree-`2n` layout: `perm` indexes its real slots and `conj` is
/// unused.
#[derive(Clone, Debug)]
pub struct Fft64AutomorphismPlan {
    pub p: i64,
    pub perm: Vec<u32>,
    pub conj: bool,
}

/// Applies a precomputed DFT-domain automorphism plan to `a`, writing the
/// result into `res` (out-of-place).
///
/// This is a pure data movement op, applied limb by limb through the
/// backend ring's [`Fft64RingArith::fft64_automorphism`].
pub fn vec_znx_dft_automorphism<BE>(
    plan: &Fft64AutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Fft64RingArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    assert_eq!(a.n(), res.n());
    let res_size: usize = res.size();
    let min_size: usize = res_size.min(a.size());
    for limb in 0..min_size {
        BE::fft64_automorphism(plan, res.at_mut(res_col, limb), a.at(a_col, limb));
    }
    for limb in min_size..res_size {
        BE::reim_zero(res.at_mut(res_col, limb));
    }
}

pub fn vec_znx_dft_automorphism_add<BE, E: poulpy_hal::execution::TaskExecutor>(
    plan: &Fft64AutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + Fft64RingArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    assert_eq!(a.n(), res.n());
    let n = res.n();
    let cols = res.cols();
    let size = res.size().min(a.size());
    let res_ptr = SendPtr::new(res.raw_mut().as_mut_ptr());
    let apply = |limb: usize| {
        let start = n * (limb * cols + res_col);
        let res_limb = unsafe { std::slice::from_raw_parts_mut(res_ptr.get().add(start), n) };
        BE::fft64_automorphism_add(plan, res_limb, a.at(a_col, limb));
    };
    if E::IS_PARALLEL {
        E::for_each(size, apply);
    } else {
        for limb in 0..size {
            apply(limb);
        }
    }
}
