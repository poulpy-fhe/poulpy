//! Conjugate-invariant FFT64 arithmetic: DCT transforms and real slots.

use std::{fmt::Debug, marker::PhantomData};

use bytemuck::Zeroable;
use poulpy_hal::layouts::ConjugateInvariant;
use rand_distr::num_traits::{Float, FloatConst};

use crate::kernels::fft64::{
    module::{FFT64Plan, FFT64PlanNew, plan_half_degree},
    reim::{ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
    standard,
    vec_znx_dft::Fft64AutomorphismPlan,
};

/// DCT tables of the conjugate-invariant transform.
pub(super) struct DctPlan<F> {
    pack_swaps: Vec<(usize, usize)>,
    paired_swaps: Vec<(usize, usize)>,
    cos: Vec<F>,
    sin: Vec<F>,
    rotation_cos: Vec<F>,
    rotation_sin: Vec<F>,
    bit_reverse: Vec<usize>,
    sqrt_two: F,
}

impl<F: Float> Default for DctPlan<F> {
    fn default() -> Self {
        Self {
            pack_swaps: Vec::new(),
            paired_swaps: Vec::new(),
            cos: Vec::new(),
            sin: Vec::new(),
            rotation_cos: Vec::new(),
            rotation_sin: Vec::new(),
            bit_reverse: Vec::new(),
            sqrt_two: F::zero(),
        }
    }
}

impl<F> DctPlan<F>
where
    F: Float + FloatConst,
{
    fn new(n: usize) -> Self {
        let m = n >> 1;
        let pack = |source: usize| {
            let y = if source.is_multiple_of(2) {
                source >> 1
            } else {
                n - 1 - (source >> 1)
            };
            if y.is_multiple_of(2) { y >> 1 } else { m + (y >> 1) }
        };
        // Slot `s` holds the evaluation at the 4n-th root with exponent `4 * bitrev(s) + 1`,
        // the real half of the standard degree-2n layout; natural slot `j` has exponent `2j + 1`.
        let log_n = n.trailing_zeros();
        let natural = |s: usize| {
            let e = 4 * (s.reverse_bits() >> (usize::BITS - log_n)) + 1;
            if e < 2 * n { (e - 1) >> 1 } else { (4 * n - e - 1) >> 1 }
        };
        let log_m = m.trailing_zeros();
        let bit_reverse = |value: usize| {
            if m == 1 {
                0
            } else {
                value.reverse_bits() >> (usize::BITS - log_m)
            }
        };
        let paired = |source: usize| {
            if source < m {
                bit_reverse(source)
            } else {
                m + bit_reverse(n - source)
            }
        };
        let bit_reverse = (0..m).map(bit_reverse).collect();
        let angle = F::PI() / F::from(2 * n).unwrap();
        let mut cos = Vec::with_capacity(m);
        let mut sin = Vec::with_capacity(m);
        let mut rotation_cos = Vec::with_capacity(m);
        let mut rotation_sin = Vec::with_capacity(m);
        let rotation_angle = F::PI() / F::from(m).unwrap();
        for k in 0..m {
            let theta = angle * F::from(k).unwrap();
            cos.push(theta.cos());
            sin.push(theta.sin());
            let rotation = rotation_angle * F::from(k).unwrap();
            rotation_cos.push(rotation.cos());
            rotation_sin.push(rotation.sin());
        }
        Self {
            pack_swaps: permutation_swaps(n, |s| pack(natural(s))),
            paired_swaps: permutation_swaps(n, paired),
            cos,
            sin,
            rotation_cos,
            rotation_sin,
            bit_reverse,
            sqrt_two: F::from(2).unwrap().sqrt(),
        }
    }
}

impl<F> FFT64PlanNew for FFT64Plan<F, ConjugateInvariant>
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    fn new(n: usize) -> Self {
        let m = plan_half_degree(n);
        Self {
            fft: ReimFFTTable::new_cyclic(m),
            ifft: ReimIFFTTable::new_cyclic(m),
            dct: DctPlan::new(n),
            ring: PhantomData,
        }
    }
}

impl<F> FFT64Plan<F, ConjugateInvariant>
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    /// Normalisation of [`Self::inverse`].
    pub fn divisor(&self) -> F {
        F::from(4 * self.fft.m()).unwrap()
    }

    /// Conjugate-invariant forward transform (DCT-III through a cyclic FFT).
    pub fn forward<BE: ReimFFTExecute<ReimIFFTTable<F>, F>>(&self, data: &mut [F]) {
        assert_eq!(data.len(), self.fft.m() << 1);
        apply_swaps(data, &self.dct.paired_swaps);
        dct3_preprocess(data, &self.dct);
        BE::reim_dft_execute(&self.ifft, data);
        apply_swaps_inverse(data, &self.dct.pack_swaps);
        let four = F::from(4).unwrap();
        data.iter_mut().for_each(|value| *value = *value * four);
    }

    /// Conjugate-invariant inverse transform (DCT-II through a cyclic FFT), unnormalised.
    pub fn inverse<BE: ReimFFTExecute<ReimFFTTable<F>, F>>(&self, data: &mut [F]) {
        assert_eq!(data.len(), self.fft.m() << 1);
        apply_swaps(data, &self.dct.pack_swaps);
        BE::reim_dft_execute(&self.fft, data);
        dct2_postprocess(data, &self.dct);
        apply_swaps_inverse(data, &self.dct.paired_swaps);
    }
}

fn permutation_swaps(n: usize, destination: impl Fn(usize) -> usize) -> Vec<(usize, usize)> {
    let mut seen = vec![false; n];
    let mut swaps = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut current = start;
        seen[current] = true;
        loop {
            let next = destination(current);
            if next == start {
                break;
            }
            swaps.push((start, next));
            current = next;
            assert!(!seen[current], "FFT permutation is not bijective");
            seen[current] = true;
        }
    }
    swaps
}

fn apply_swaps<F>(data: &mut [F], swaps: &[(usize, usize)]) {
    for &(a, b) in swaps {
        data.swap(a, b);
    }
}

fn apply_swaps_inverse<F>(data: &mut [F], swaps: &[(usize, usize)]) {
    for &(a, b) in swaps.iter().rev() {
        data.swap(a, b);
    }
}

fn dct2_value<F>(zr: F, zi: F, zmr: F, zmi: F, c: F, s: F, wc: F, ws: F) -> (F, F)
where
    F: Float,
{
    let half = F::from(0.5).unwrap();
    let er = (zr + zmr) * half;
    let ei = (zi - zmi) * half;
    let or = (zi + zmi) * half;
    let oi = (zmr - zr) * half;
    let yr = er + wc * or - ws * oi;
    let yi = ei + ws * or + wc * oi;
    (
        F::from(2).unwrap() * (yr * c - yi * s),
        F::from(2).unwrap() * (yr * s + yi * c),
    )
}

fn dct2_postprocess<F>(data: &mut [F], plan: &DctPlan<F>)
where
    F: Float,
{
    let m = data.len() >> 1;
    let a = data[0];
    let b = data[m];
    data[0] = F::from(2).unwrap() * (a + b);
    data[m] = plan.sqrt_two * (a - b);

    for k in 1..m.div_ceil(2) {
        let mk = m - k;
        let pk = plan.bit_reverse[k];
        let pmk = plan.bit_reverse[mk];
        let (akr, aki) = (data[pk], data[m + pk]);
        let (amr, ami) = (data[pmk], data[m + pmk]);
        let (xk, xnk) = dct2_value(
            akr,
            aki,
            amr,
            ami,
            plan.cos[k],
            plan.sin[k],
            plan.rotation_cos[k],
            plan.rotation_sin[k],
        );
        let (xmk, xnmk) = dct2_value(
            amr,
            ami,
            akr,
            aki,
            plan.cos[mk],
            plan.sin[mk],
            plan.rotation_cos[mk],
            plan.rotation_sin[mk],
        );
        data[pk] = xk;
        data[m + pk] = xnk;
        data[pmk] = xmk;
        data[m + pmk] = xnmk;
    }
    if m > 1 {
        let k = m >> 1;
        let p = plan.bit_reverse[k];
        let (zr, zi) = (data[p], data[m + p]);
        let (xk, xnk) = dct2_value(zr, zi, zr, zi, plan.cos[k], plan.sin[k], F::zero(), F::one());
        data[p] = xk;
        data[m + p] = xnk;
    }
}

#[allow(clippy::too_many_arguments)]
fn dct3_z<F>(ck: F, cnk: F, cmk: F, cmnk: F, c: F, s: F, cm: F, sm: F, wc: F, ws: F) -> (F, F)
where
    F: Float,
{
    let half = F::from(0.5).unwrap();
    let ykr = (ck * c + cnk * s) * half;
    let yki = (cnk * c - ck * s) * half;
    let ymr = (cmk * cm + cmnk * sm) * half;
    let ymi = (cmnk * cm - cmk * sm) * half;
    let er = (ykr + ymr) * half;
    let ei = (yki - ymi) * half;
    let dr = (ykr - ymr) * half;
    let di = (yki + ymi) * half;
    let or = dr * wc - di * ws;
    let oi = dr * ws + di * wc;
    (er - oi, ei + or)
}

fn dct3_preprocess<F>(data: &mut [F], plan: &DctPlan<F>)
where
    F: Float,
{
    let m = data.len() >> 1;
    let y0 = data[0] * F::from(0.5).unwrap();
    let ym = data[m] / plan.sqrt_two;
    data[0] = (y0 + ym) * F::from(0.5).unwrap();
    data[m] = (y0 - ym) * F::from(0.5).unwrap();

    for k in 1..m.div_ceil(2) {
        let mk = m - k;
        let pk = plan.bit_reverse[k];
        let pmk = plan.bit_reverse[mk];
        let (ck, cnk) = (data[pk], data[m + pk]);
        let (cmk, cmnk) = (data[pmk], data[m + pmk]);
        let (zkr, zki) = dct3_z(
            ck,
            cnk,
            cmk,
            cmnk,
            plan.cos[k],
            plan.sin[k],
            plan.cos[mk],
            plan.sin[mk],
            plan.rotation_cos[k],
            -plan.rotation_sin[k],
        );
        let (zmr, zmi) = dct3_z(
            cmk,
            cmnk,
            ck,
            cnk,
            plan.cos[mk],
            plan.sin[mk],
            plan.cos[k],
            plan.sin[k],
            plan.rotation_cos[mk],
            -plan.rotation_sin[mk],
        );
        data[pk] = zkr;
        data[m + pk] = zki;
        data[pmk] = zmr;
        data[m + pmk] = zmi;
    }
    if m > 1 {
        let k = m >> 1;
        let p = plan.bit_reverse[k];
        let (ck, cnk) = (data[p], data[m + p]);
        let (zr, zi) = dct3_z(
            ck,
            cnk,
            ck,
            cnk,
            plan.cos[k],
            plan.sin[k],
            plan.cos[k],
            plan.sin[k],
            F::zero(),
            -F::one(),
        );
        data[p] = zr;
        data[m + p] = zi;
    }
}

/// Builds the [`Fft64AutomorphismPlan`]: the standard degree-`2n` plan,
/// whose real half is the conjugate-invariant layout; `conj` only touches the absent
/// imaginary half.
pub fn build_fft64_automorphism_plan_portable(n: usize, p: i64) -> Fft64AutomorphismPlan {
    let plan = standard::build_fft64_automorphism_plan_portable(2 * n, p);
    Fft64AutomorphismPlan {
        p,
        perm: plan.perm,
        conj: false,
    }
}

/// One limb of the DFT automorphism: `res = tau_p(a)`.
pub fn fft64_automorphism_portable(plan: &Fft64AutomorphismPlan, res: &mut [f64], a: &[f64]) {
    assert_eq!(plan.perm.len(), res.len());
    for (value, &source) in res.iter_mut().zip(&plan.perm) {
        *value = a[source as usize];
    }
}

/// One limb of the DFT automorphism: `res += tau_p(a)`.
pub fn fft64_automorphism_add_portable(plan: &Fft64AutomorphismPlan, res: &mut [f64], a: &[f64]) {
    assert_eq!(plan.perm.len(), res.len());
    for (value, &source) in res.iter_mut().zip(&plan.perm) {
        *value += a[source as usize];
    }
}

/// [`Fft64RingArith`](crate::kernels::fft64::ring_arith::Fft64RingArith) for a conjugate-invariant backend: DCT transforms and real slots.
#[macro_export]
macro_rules! fft64_ring_arith_ci {
    () => {
        fn fft64_forward(
            plan: &$crate::kernels::fft64::module::FFT64Plan<f64, ::poulpy_hal::layouts::ConjugateInvariant>,
            data: &mut [f64],
        ) {
            plan.forward::<Self>(data)
        }
        fn fft64_inverse(
            plan: &$crate::kernels::fft64::module::FFT64Plan<f64, ::poulpy_hal::layouts::ConjugateInvariant>,
            data: &mut [f64],
        ) {
            plan.inverse::<Self>(data)
        }
        fn fft64_divisor(
            plan: &$crate::kernels::fft64::module::FFT64Plan<f64, ::poulpy_hal::layouts::ConjugateInvariant>,
        ) -> f64 {
            plan.divisor()
        }
        fn fft64_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
            <Self as $crate::kernels::fft64::reim::ReimArith>::reim_real_mul(res, a, b)
        }
        fn fft64_mul_assign(res: &mut [f64], a: &[f64]) {
            <Self as $crate::kernels::fft64::reim::ReimArith>::reim_real_mul_assign(res, a)
        }
        fn fft64_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
            <Self as $crate::kernels::fft64::reim4::Reim4BlkMatVec>::reim4_real_mat1col_prod(nrows, dst, u, v)
        }
        fn fft64_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
            <Self as $crate::kernels::fft64::reim4::Reim4BlkMatVec>::reim4_real_mat2cols_prod(nrows, dst, u, v)
        }
        fn fft64_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
            <Self as $crate::kernels::fft64::reim4::Reim4BlkMatVec>::reim4_real_mat2cols_2ndcol_prod(nrows, dst, u, v)
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
            <Self as $crate::kernels::fft64::reim4::Reim4Convolution>::reim4_real_convolution_apply(
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
            <Self as $crate::kernels::fft64::reim4::Reim4Convolution>::reim4_real_convolution_apply_accumulate(
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
            <Self as $crate::kernels::fft64::reim4::Reim4Convolution>::reim4_real_convolution_pairwise_apply(
                m, min_size, offset, dst, dst_stride, a0, a1, a_size, b0, b1, b_size, b_log_gap, tmp,
            )
        }
        fn fft64_automorphism_plan(n: usize, p: i64) -> $crate::kernels::fft64::vec_znx_dft::Fft64AutomorphismPlan {
            $crate::kernels::fft64::conjugate_invariant::build_fft64_automorphism_plan_portable(n, p)
        }
        fn fft64_automorphism(plan: &$crate::kernels::fft64::vec_znx_dft::Fft64AutomorphismPlan, res: &mut [f64], a: &[f64]) {
            $crate::kernels::fft64::conjugate_invariant::fft64_automorphism_portable(plan, res, a)
        }
        fn fft64_automorphism_add(plan: &$crate::kernels::fft64::vec_znx_dft::Fft64AutomorphismPlan, res: &mut [f64], a: &[f64]) {
            $crate::kernels::fft64::conjugate_invariant::fft64_automorphism_add_portable(plan, res, a)
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::FFT64Portable;
    use crate::kernels::fft64::module::{FFT64Plan, FFT64PlanNew};
    use poulpy_hal::layouts::ConjugateInvariant;

    #[test]
    fn conjugate_invariant_fft_matches_direct_dct() {
        for n in [2usize, 4, 8, 16, 32] {
            let plan = FFT64Plan::<f64, ConjugateInvariant>::new(n);
            let coeffs = (0..n).map(|i| (i as f64 + 1.0) / 17.0).collect::<Vec<_>>();
            let want = (0..n)
                .map(|j| {
                    coeffs[0]
                        + (1..n)
                            .map(|k| 2.0 * coeffs[k] * (std::f64::consts::PI * k as f64 * (j as f64 + 0.5) / n as f64).cos())
                            .sum::<f64>()
                })
                .collect::<Vec<_>>();
            let mut got = coeffs.clone();
            plan.forward::<FFT64Portable>(&mut got);
            // Slot `s` evaluates at exponent `4 * bitrev(s) + 1`, the natural slot of `+-` that exponent.
            let natural = |s: usize| {
                let e = 4 * (s.reverse_bits() >> (usize::BITS - n.trailing_zeros())) + 1;
                if e < 2 * n { (e - 1) >> 1 } else { (4 * n - e - 1) >> 1 }
            };
            for (s, got) in got.iter().enumerate() {
                let want = want[natural(s)];
                assert!((got - want).abs() < 1e-10, "n={n}: {got} != {want}");
            }
            plan.inverse::<FFT64Portable>(&mut got);
            for (got, want) in got.iter().zip(&coeffs) {
                assert!((got / plan.divisor() - want).abs() < 1e-10, "n={n}: {got} != {want}");
            }
        }
    }

    #[test]
    fn conjugate_invariant_fft_multiplication_matches_ambient_ring() {
        for n in [8usize, 16, 32] {
            let plan = FFT64Plan::<f64, ConjugateInvariant>::new(n);
            let a = (0..n).map(|i| (i as f64 - 3.0) / 11.0).collect::<Vec<_>>();
            let b = (0..n).map(|i| (5.0 - i as f64) / 13.0).collect::<Vec<_>>();
            let unfold = |value: &[f64]| {
                let mut out = vec![0.0; 2 * n];
                out[..n].copy_from_slice(value);
                for k in 1..n {
                    out[2 * n - k] = -value[k];
                }
                out
            };
            let (ua, ub) = (unfold(&a), unfold(&b));
            let mut want = vec![0.0; 2 * n];
            for (i, &a) in ua.iter().enumerate() {
                for (j, &b) in ub.iter().enumerate() {
                    let degree = i + j;
                    if degree < 2 * n {
                        want[degree] += a * b;
                    } else {
                        want[degree - 2 * n] -= a * b;
                    }
                }
            }

            let (mut fa, mut fb) = (a.clone(), b.clone());
            plan.forward::<FFT64Portable>(&mut fa);
            plan.forward::<FFT64Portable>(&mut fb);
            for (a, b) in fa.iter_mut().zip(&fb) {
                *a *= *b;
            }
            plan.inverse::<FFT64Portable>(&mut fa);
            for (got, want) in fa.iter().zip(&want[..n]) {
                assert!((got / plan.divisor() - want).abs() < 1e-9, "n={n}: {got} != {want}");
            }
        }
    }
}
