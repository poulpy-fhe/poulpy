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
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
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
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
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
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
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
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
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
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
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
    BE: Backend<DftWord = f64, BigWord = i64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64> + ZnxZero,
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
    BE: Backend<DftWord = f64, BigWord = i64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64> + ZnxZero,
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
    BE: Backend<DftWord = f64, BigWord = i64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64>,
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
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
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
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
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
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
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
/// by `p mod 4`).
#[derive(Clone, Debug)]
pub struct Fft64AutomorphismPlan {
    pub p: i64,
    pub perm: Vec<u32>,
    pub conj: bool,
}

/// Builds the [`Fft64AutomorphismPlan`] for ring dimension `n` and odd `p`.
///
/// Closed-form derivation: the DIF FFT places output slot `i` at the
/// evaluation point `omega^{2 * ir(i) + 1}` mod `2n`, where `ir(i)` is the
/// bit-reversal of `i` over `log2(n)` bits. For odd `p`:
///
/// - `p ≡ 1 (mod 4)` keeps the stored half-spectrum closed under
///   `e -> p*e mod 2n`: pure permutation.
/// - `p ≡ 3 (mod 4)` maps into the conjugate half. Substituting `-p`
///   (now `≡ 1 mod 4`) brings the action back at the cost of a single
///   global imag negation, signalled by `conj`.
pub fn build_fft64_automorphism_plan(n: usize, p: i64) -> Fft64AutomorphismPlan {
    assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
    assert!(p & 1 == 1, "p must be odd for an R/(X^N+1) automorphism, got {p}");

    let m = n >> 1;
    let mask = (2 * n - 1) as i64;
    let conj = (p & 3) != 1;
    // p_eff: positive representative in [0, 2n) of either p or -p, chosen
    // so that p_eff ≡ 1 (mod 4) and the stored half-spectrum is closed
    // under multiplication by p_eff.
    let p_eff = if conj { (-p) & mask } else { p & mask };

    let log_n = n.trailing_zeros();
    let ir = |i: u32| -> u32 { i.reverse_bits() >> (32 - log_n) };

    let mut perm: Vec<u32> = vec![0u32; m];
    for (i, mi) in perm.iter_mut().enumerate().take(m) {
        let e: i64 = 2 * ir(i as u32) as i64 + 1;
        let e_src: i64 = (p_eff * e) & mask;
        let src: u32 = ((e_src - 1) >> 1) as u32;
        *mi = ir(src);
    }
    Fft64AutomorphismPlan { p, perm, conj }
}

/// Applies a precomputed DFT-domain automorphism plan to `a`, writing the
/// result into `res` (out-of-place).
///
/// This is a pure data movement op: one source-side gather and one
/// destination-side contiguous store per complex slot, plus an optional
/// global imaginary-half sign flip selected outside the inner loop on
/// `plan.conj`.
pub fn vec_znx_dft_automorphism<BE>(
    plan: &Fft64AutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(plan.perm.len(), res.n() >> 1);
    }

    let m: usize = res.n() >> 1;
    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let min_size: usize = res_size.min(a_size);
    let perm: &[u32] = &plan.perm;

    for limb in 0..min_size {
        let (res_re, res_im) = res.at_mut(res_col, limb).split_at_mut(m);
        let a_limb = a.at(a_col, limb);
        let (a_re, a_im) = a_limb.split_at(m);

        if plan.conj {
            for i in 0..m {
                let s = perm[i] as usize;
                res_re[i] = a_re[s];
                res_im[i] = -a_im[s];
            }
        } else {
            for i in 0..m {
                let s = perm[i] as usize;
                res_re[i] = a_re[s];
                res_im[i] = a_im[s];
            }
        }
    }

    for limb in min_size..res_size {
        BE::reim_zero(res.at_mut(res_col, limb));
    }
}
