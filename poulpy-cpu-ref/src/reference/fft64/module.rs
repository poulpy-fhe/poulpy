use crate::ring::{CpuRing, RingData, Standard};
use std::fmt::Debug;

use bytemuck::Zeroable;
use rand_distr::num_traits::{Float, FloatConst};

use crate::{
    layouts::{Backend, Module},
    reference::fft64::reim::{ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
};

struct ConjugateInvariantPlan<F> {
    pack_swaps: Vec<(usize, usize)>,
    paired_swaps: Vec<(usize, usize)>,
    cos: Vec<F>,
    sin: Vec<F>,
    rotation_cos: Vec<F>,
    rotation_sin: Vec<F>,
    bit_reverse: Vec<usize>,
    sqrt_two: F,
}

/// Forward and inverse evaluation transforms for one ring degree.
pub struct FFT64Plan<F, R: CpuRing = Standard>
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    fft: ReimFFTTable<F>,
    ifft: ReimIFFTTable<F>,
    ci: R::Data<ConjugateInvariantPlan<F>>,
}

impl<F, R: CpuRing> FFT64Plan<F, R>
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    /// Creates the transform plan selected by the backend ring type.
    pub fn new(n: usize) -> Self {
        assert!(
            n >= 2 && n.is_power_of_two(),
            "ring degree must be a power of two >= 2, got {n}"
        );
        let m = n >> 1;
        Self {
            fft: if R::IS_CI {
                ReimFFTTable::new_cyclic(m)
            } else {
                ReimFFTTable::new(m)
            },
            ifft: if R::IS_CI {
                ReimIFFTTable::new_cyclic(m)
            } else {
                ReimIFFTTable::new(m)
            },
            ci: R::Data::new(|| ConjugateInvariantPlan::new(n)),
        }
    }

    pub fn is_conjugate_invariant(&self) -> bool {
        R::IS_CI
    }

    pub fn fft(&self) -> &ReimFFTTable<F> {
        &self.fft
    }

    pub fn ifft(&self) -> &ReimIFFTTable<F> {
        &self.ifft
    }

    pub fn divisor(&self) -> F {
        F::from(if R::IS_CI { 4 * self.fft.m() } else { self.fft.m() }).unwrap()
    }

    pub fn forward<BE>(&self, data: &mut [F])
    where
        BE: ReimFFTExecute<ReimFFTTable<F>, F> + ReimFFTExecute<ReimIFFTTable<F>, F>,
    {
        assert_eq!(data.len(), self.fft.m() << 1);
        if let Some(ci) = self.ci.get() {
            apply_swaps(data, &ci.paired_swaps);
            ci_dct3_preprocess(data, ci);
            BE::reim_dft_execute(&self.ifft, data);
            apply_swaps_inverse(data, &ci.pack_swaps);
            let four = F::from(4).unwrap();
            data.iter_mut().for_each(|value| *value = *value * four);
        } else {
            BE::reim_dft_execute(&self.fft, data);
        }
    }

    pub fn inverse<BE>(&self, data: &mut [F])
    where
        BE: ReimFFTExecute<ReimFFTTable<F>, F> + ReimFFTExecute<ReimIFFTTable<F>, F>,
    {
        assert_eq!(data.len(), self.fft.m() << 1);
        if let Some(ci) = self.ci.get() {
            apply_swaps(data, &ci.pack_swaps);
            BE::reim_dft_execute(&self.fft, data);
            ci_dct2_postprocess(data, ci);
            apply_swaps_inverse(data, &ci.paired_swaps);
        } else {
            BE::reim_dft_execute(&self.ifft, data);
        }
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

fn ci_dct2_value<F>(zr: F, zi: F, zmr: F, zmi: F, c: F, s: F, wc: F, ws: F) -> (F, F)
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

fn ci_dct2_postprocess<F>(data: &mut [F], plan: &ConjugateInvariantPlan<F>)
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
        let (xk, xnk) = ci_dct2_value(
            akr,
            aki,
            amr,
            ami,
            plan.cos[k],
            plan.sin[k],
            plan.rotation_cos[k],
            plan.rotation_sin[k],
        );
        let (xmk, xnmk) = ci_dct2_value(
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
        let (xk, xnk) = ci_dct2_value(zr, zi, zr, zi, plan.cos[k], plan.sin[k], F::zero(), F::one());
        data[p] = xk;
        data[m + p] = xnk;
    }
}

#[allow(clippy::too_many_arguments)]
fn ci_dct3_z<F>(ck: F, cnk: F, cmk: F, cmnk: F, c: F, s: F, cm: F, sm: F, wc: F, ws: F) -> (F, F)
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

fn ci_dct3_preprocess<F>(data: &mut [F], plan: &ConjugateInvariantPlan<F>)
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
        let (zkr, zki) = ci_dct3_z(
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
        let (zmr, zmi) = ci_dct3_z(
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
        let (zr, zi) = ci_dct3_z(
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

/// Complete geometric family of FFT plans up to a maximum ring degree.
pub struct FFT64PlanSet<F, R: CpuRing = Standard>
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    plans: Vec<FFT64Plan<F, R>>,
    max_n: usize,
}

impl<F, R: CpuRing> FFT64PlanSet<F, R>
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    pub fn new(max_n: usize) -> Self {
        assert!(
            max_n >= 2 && max_n.is_power_of_two(),
            "maximum ring degree must be a power of two >= 2, got {max_n}"
        );
        let plans = (1..=max_n.ilog2() as usize)
            .map(|log_n| FFT64Plan::new(1usize << log_n))
            .collect();
        Self { plans, max_n }
    }

    pub fn max_n(&self) -> usize {
        self.max_n
    }

    pub fn for_ring(&self, n: usize) -> &FFT64Plan<F, R> {
        assert!(
            n >= 2 && n.is_power_of_two() && n <= self.max_n,
            "unsupported ring degree {n}; maximum is {}",
            self.max_n
        );
        &self.plans[n.ilog2() as usize - 1]
    }

    pub fn for_slots(&self, slots: usize) -> &FFT64Plan<F, R> {
        self.for_ring(slots.checked_mul(2).expect("slot count overflow"))
    }
}

/// Access to the precomputed FFT/iFFT tables stored inside a `Module<B>` handle.
///
/// Backend crates implement [`FFTHandleProvider`] for their concrete handle type.
/// `poulpy-hal` then provides this blanket trait on `Module<B>`, which lets family
/// defaults share the same FFT64 handle contract across scalar and accelerated backends.
pub trait FFTModuleHandle<F>: poulpy_hal::api::ModuleN
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    type Ring: CpuRing;
    fn get_fft_plan(&self, n: usize) -> &FFT64Plan<F, Self::Ring>;

    fn get_fft_table_for(&self, n: usize) -> &ReimFFTTable<F> {
        self.get_fft_plan(n).fft()
    }

    fn get_ifft_table_for(&self, n: usize) -> &ReimIFFTTable<F> {
        self.get_fft_plan(n).ifft()
    }
}

/// Implemented by FFT64 backend handle types that own precomputed FFT tables.
///
/// # Safety
///
/// Implementors must return references that stay valid for the lifetime of `&self`.
/// The handle must be fully initialized before `Module::new()` returns.
pub unsafe trait FFTHandleProvider<F>
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    type Ring: CpuRing;
    fn get_fft_plan(&self, n: usize) -> &FFT64Plan<F, Self::Ring>;
}

/// Construct FFT64 backend handles for [`Module::new`](crate::api::ModuleNew::new).
///
/// # Safety
///
/// Implementors must return a fully initialized handle for the requested `n`.
/// The handle is boxed and stored inside the `Module`, so it must be safe to
/// drop via [`crate::layouts::Backend::destroy`].
pub unsafe trait FFT64HandleFactory: Sized {
    /// Builds a fully initialized handle for ring dimension `n`.
    fn create_fft64_handle(n: usize) -> Self;

    /// Optional runtime capability check (default: no-op).
    fn assert_fft64_runtime_support() {}
}

impl<BE: Backend<ZnxWord = i64>> FFTModuleHandle<BE::DftWord> for Module<BE>
where
    BE::DftWord: Float + FloatConst + Debug + Send + Sync,
    BE::Handle: FFTHandleProvider<BE::DftWord>,
{
    type Ring = <BE::Handle as FFTHandleProvider<BE::DftWord>>::Ring;
    fn get_fft_plan(&self, n: usize) -> &FFT64Plan<BE::DftWord, Self::Ring> {
        unsafe { (&*self.ptr()).get_fft_plan(n) }
    }
}

impl<F> ConjugateInvariantPlan<F>
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
            pack_swaps: permutation_swaps(n, pack),
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

#[cfg(test)]
mod tests {
    use super::FFT64Plan;
    use crate::FFT64Ref;
    use crate::ring::ConjugateInvariant;

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
            plan.forward::<FFT64Ref>(&mut got);
            for (got, want) in got.iter().zip(&want) {
                assert!((got - want).abs() < 1e-10, "n={n}: {got} != {want}");
            }
            plan.inverse::<FFT64Ref>(&mut got);
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
            plan.forward::<FFT64Ref>(&mut fa);
            plan.forward::<FFT64Ref>(&mut fb);
            for (a, b) in fa.iter_mut().zip(&fb) {
                *a *= *b;
            }
            plan.inverse::<FFT64Ref>(&mut fa);
            for (got, want) in fa.iter().zip(&want[..n]) {
                assert!((got / plan.divisor() - want).abs() < 1e-9, "n={n}: {got} != {want}");
            }
        }
    }
}
