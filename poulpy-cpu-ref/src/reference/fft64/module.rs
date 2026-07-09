use std::fmt::Debug;

use rand_distr::num_traits::{Float, FloatConst};

use crate::{
    layouts::{Backend, Module},
    reference::fft64::reim::{ReimFFTTable, ReimIFFTTable},
};

/// Forward and inverse negacyclic FFT tables for one ring degree.
pub struct FFT64Plan<F>
where
    F: Float + FloatConst + Debug,
{
    fft: ReimFFTTable<F>,
    ifft: ReimIFFTTable<F>,
}

impl<F> FFT64Plan<F>
where
    F: Float + FloatConst + Debug,
{
    /// Creates the plan for `Z[X]/(X^n + 1)`.
    pub fn new(n: usize) -> Self {
        assert!(
            n >= 2 && n.is_power_of_two(),
            "ring degree must be a power of two >= 2, got {n}"
        );
        Self {
            fft: ReimFFTTable::new(n >> 1),
            ifft: ReimIFFTTable::new(n >> 1),
        }
    }

    pub fn fft(&self) -> &ReimFFTTable<F> {
        &self.fft
    }

    pub fn ifft(&self) -> &ReimIFFTTable<F> {
        &self.ifft
    }
}

/// Complete geometric family of FFT plans up to a maximum ring degree.
pub struct FFT64PlanSet<F>
where
    F: Float + FloatConst + Debug,
{
    plans: Vec<FFT64Plan<F>>,
    max_n: usize,
}

impl<F> FFT64PlanSet<F>
where
    F: Float + FloatConst + Debug,
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

    pub fn for_ring(&self, n: usize) -> &FFT64Plan<F> {
        assert!(
            n >= 2 && n.is_power_of_two() && n <= self.max_n,
            "unsupported ring degree {n}; maximum is {}",
            self.max_n
        );
        &self.plans[n.ilog2() as usize - 1]
    }

    pub fn for_slots(&self, slots: usize) -> &FFT64Plan<F> {
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
    F: Float + FloatConst + Debug,
{
    fn get_fft_plan(&self, n: usize) -> &FFT64Plan<F>;

    fn get_fft_table_for(&self, n: usize) -> &ReimFFTTable<F> {
        self.get_fft_plan(n).fft()
    }

    fn get_ifft_table_for(&self, n: usize) -> &ReimIFFTTable<F> {
        self.get_fft_plan(n).ifft()
    }

    fn get_fft_table(&self) -> &ReimFFTTable<F> {
        self.get_fft_table_for(self.n())
    }

    fn get_ifft_table(&self) -> &ReimIFFTTable<F> {
        self.get_ifft_table_for(self.n())
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
    F: Float + FloatConst + Debug,
{
    fn get_fft_plan(&self, n: usize) -> &FFT64Plan<F>;
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

impl<BE: Backend> FFTModuleHandle<BE::ScalarPrep> for Module<BE>
where
    BE::ScalarPrep: Float + FloatConst + Debug,
    BE::Handle: FFTHandleProvider<BE::ScalarPrep>,
{
    fn get_fft_plan(&self, n: usize) -> &FFT64Plan<BE::ScalarPrep> {
        unsafe { (&*self.ptr()).get_fft_plan(n) }
    }
}
