//! Standard-ring FFT64 arithmetic: negacyclic FFT and complex slots.

use std::{fmt::Debug, marker::PhantomData};

use bytemuck::Zeroable;
use poulpy_hal::layouts::Standard;
use rand_distr::num_traits::{Float, FloatConst};

use crate::kernels::fft64::{
    module::{FFT64Plan, FFT64PlanNew, plan_half_degree},
    reim::{ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
    vec_znx_dft::Fft64AutomorphismPlan,
};

impl<F> FFT64PlanNew for FFT64Plan<F, Standard>
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    fn new(n: usize) -> Self {
        let m = plan_half_degree(n);
        Self {
            fft: ReimFFTTable::new(m),
            ifft: ReimIFFTTable::new(m),
            dct: Default::default(),
            ring: PhantomData,
        }
    }
}

impl<F> FFT64Plan<F, Standard>
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    /// Normalisation of [`Self::inverse`].
    pub fn divisor(&self) -> F {
        F::from(self.fft.m()).unwrap()
    }

    /// Negacyclic forward transform.
    pub fn forward<BE: ReimFFTExecute<ReimFFTTable<F>, F>>(&self, data: &mut [F]) {
        assert_eq!(data.len(), self.fft.m() << 1);
        BE::reim_dft_execute(&self.fft, data);
    }

    /// Negacyclic inverse transform, unnormalised.
    pub fn inverse<BE: ReimFFTExecute<ReimIFFTTable<F>, F>>(&self, data: &mut [F]) {
        assert_eq!(data.len(), self.fft.m() << 1);
        BE::reim_dft_execute(&self.ifft, data);
    }
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

/// One limb of the DFT automorphism: `res = tau_p(a)`.
pub fn fft64_automorphism_portable(plan: &Fft64AutomorphismPlan, res: &mut [f64], a: &[f64]) {
    let m = res.len() >> 1;
    assert_eq!(plan.perm.len(), m);
    let (res_re, res_im) = res.split_at_mut(m);
    let (a_re, a_im) = a.split_at(m);
    if plan.conj {
        for (i, &s) in plan.perm.iter().enumerate() {
            res_re[i] = a_re[s as usize];
            res_im[i] = -a_im[s as usize];
        }
    } else {
        for (i, &s) in plan.perm.iter().enumerate() {
            res_re[i] = a_re[s as usize];
            res_im[i] = a_im[s as usize];
        }
    }
}

/// One limb of the DFT automorphism: `res += tau_p(a)`.
pub fn fft64_automorphism_add_portable(plan: &Fft64AutomorphismPlan, res: &mut [f64], a: &[f64]) {
    let m = res.len() >> 1;
    assert_eq!(plan.perm.len(), m);
    let (res_re, res_im) = res.split_at_mut(m);
    let (a_re, a_im) = a.split_at(m);
    for (i, &s) in plan.perm.iter().enumerate() {
        res_re[i] += a_re[s as usize];
        res_im[i] += if plan.conj { -a_im[s as usize] } else { a_im[s as usize] };
    }
}

/// [`Fft64RingArith`](crate::kernels::fft64::ring_arith::Fft64RingArith) for a standard-ring backend: negacyclic FFT and complex slots.
#[macro_export]
macro_rules! fft64_ring_arith_standard {
    () => {
        fn fft64_forward(
            plan: &$crate::kernels::fft64::module::FFT64Plan<f64, ::poulpy_hal::layouts::Standard>,
            data: &mut [f64],
        ) {
            plan.forward::<Self>(data)
        }
        fn fft64_inverse(
            plan: &$crate::kernels::fft64::module::FFT64Plan<f64, ::poulpy_hal::layouts::Standard>,
            data: &mut [f64],
        ) {
            plan.inverse::<Self>(data)
        }
        fn fft64_divisor(plan: &$crate::kernels::fft64::module::FFT64Plan<f64, ::poulpy_hal::layouts::Standard>) -> f64 {
            plan.divisor()
        }
        fn fft64_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
            <Self as $crate::kernels::fft64::reim::ReimArith>::reim_mul(res, a, b)
        }
        fn fft64_mul_assign(res: &mut [f64], a: &[f64]) {
            <Self as $crate::kernels::fft64::reim::ReimArith>::reim_mul_assign(res, a)
        }
        fn fft64_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
            <Self as $crate::kernels::fft64::reim4::Reim4BlkMatVec>::reim4_mat1col_prod(nrows, dst, u, v)
        }
        fn fft64_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
            <Self as $crate::kernels::fft64::reim4::Reim4BlkMatVec>::reim4_mat2cols_prod(nrows, dst, u, v)
        }
        fn fft64_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
            <Self as $crate::kernels::fft64::reim4::Reim4BlkMatVec>::reim4_mat2cols_2ndcol_prod(nrows, dst, u, v)
        }
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
        ) {
            <Self as $crate::kernels::fft64::reim4::Reim4Convolution>::reim4_convolution_apply(
                m, min_size, offset, dst, dst_stride, a, a_size, b, b_size, b_log_gap, tmp,
            )
        }
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
        ) {
            <Self as $crate::kernels::fft64::reim4::Reim4Convolution>::reim4_convolution_apply_accumulate(
                m, min_size, offset, dst, dst_stride, a, a_size, b, b_size, b_log_gap, tmp,
            )
        }
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
        ) {
            <Self as $crate::kernels::fft64::reim4::Reim4Convolution>::reim4_convolution_pairwise_apply(
                m, min_size, offset, dst, dst_stride, a0, a1, a_size, b0, b1, b_size, b_log_gap, tmp,
            )
        }
        fn fft64_automorphism_plan(n: usize, p: i64) -> $crate::kernels::fft64::vec_znx_dft::Fft64AutomorphismPlan {
            $crate::kernels::fft64::standard::build_fft64_automorphism_plan(n, p)
        }
        fn fft64_automorphism(plan: &$crate::kernels::fft64::vec_znx_dft::Fft64AutomorphismPlan, res: &mut [f64], a: &[f64]) {
            <Self as $crate::kernels::fft64::reim::ReimArith>::reim_automorphism(plan, res, a)
        }
        fn fft64_automorphism_add(plan: &$crate::kernels::fft64::vec_znx_dft::Fft64AutomorphismPlan, res: &mut [f64], a: &[f64]) {
            <Self as $crate::kernels::fft64::reim::ReimArith>::reim_automorphism_add(plan, res, a)
        }
    };
}
