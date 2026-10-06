//! FFT64 DFT-domain arithmetic that differs between the standard and the conjugate-invariant ring.

use crate::{
    kernels::fft64::{module::FFT64Plan, vec_znx_dft::Fft64AutomorphismPlan},
    layouts::Backend,
};

/// DFT-domain FFT64 arithmetic, implemented once per backend ring with
/// [`fft64_ring_arith_standard!`](crate::fft64_ring_arith_standard) or
/// [`fft64_ring_arith_ci!`](crate::fft64_ring_arith_ci).
///
/// Standard slots are complex evaluations; conjugate-invariant slots are real.
#[allow(clippy::too_many_arguments)]
pub trait Fft64RingArith {
    fn fft64_forward(plan: &FFT64Plan<f64, <Self as Backend>::Ring>, data: &mut [f64])
    where
        Self: Backend;
    fn fft64_inverse(plan: &FFT64Plan<f64, <Self as Backend>::Ring>, data: &mut [f64])
    where
        Self: Backend;
    fn fft64_divisor(plan: &FFT64Plan<f64, <Self as Backend>::Ring>) -> f64
    where
        Self: Backend;
    fn fft64_mul(res: &mut [f64], a: &[f64], b: &[f64]);
    fn fft64_mul_assign(res: &mut [f64], a: &[f64]);
    fn fft64_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]);
    fn fft64_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]);
    fn fft64_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]);
    fn fft64_convolution_apply(
        m: usize,
        min_size: usize,
        offset: usize,
        dst: &mut [f64],
        dst_stride: usize,
        a: &[f64],
        a_size: usize,
        b: &[f64],
        b_size: usize,
        b_log_gap: usize,
        tmp: &mut [f64],
    );
    fn fft64_convolution_apply_accumulate(
        m: usize,
        min_size: usize,
        offset: usize,
        dst: &mut [f64],
        dst_stride: usize,
        a: &[f64],
        a_size: usize,
        b: &[f64],
        b_size: usize,
        b_log_gap: usize,
        tmp: &mut [f64],
    );
    fn fft64_convolution_pairwise_apply(
        m: usize,
        min_size: usize,
        offset: usize,
        dst: &mut [f64],
        dst_stride: usize,
        a0: &[f64],
        a1: &[f64],
        a_size: usize,
        b0: &[f64],
        b1: &[f64],
        b_size: usize,
        b_log_gap: usize,
        tmp: &mut [f64],
    );
    fn fft64_automorphism_plan(n: usize, p: i64) -> Fft64AutomorphismPlan;
    /// One limb: `res = tau_p(a)`.
    fn fft64_automorphism(plan: &Fft64AutomorphismPlan, res: &mut [f64], a: &[f64]);
    /// One limb: `res += tau_p(a)`.
    fn fft64_automorphism_add(plan: &Fft64AutomorphismPlan, res: &mut [f64], a: &[f64]);
}
