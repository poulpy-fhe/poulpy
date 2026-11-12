use poulpy_hal::{
    api::{ReimFFT, ReimIFFT, ReimM},
    layouts::{Ring, Standard},
};
use std::{fmt::Debug, marker::PhantomData};

use bytemuck::Zeroable;
use rand_distr::num_traits::{Float, FloatConst};

use crate::{
    kernels::fft64::{
        conjugate_invariant::DctPlan,
        reim::{ReimFFTTable, ReimIFFTTable},
    },
    layouts::{Backend, Module},
};

/// Forward and inverse evaluation transforms for one ring degree.
pub struct FFT64Plan<F, R: Ring = Standard>
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    pub(super) fft: ReimFFTTable<F>,
    pub(super) ifft: ReimIFFTTable<F>,
    /// DCT tables, empty on the standard ring.
    pub(super) dct: DctPlan<F>,
    pub(super) ring: PhantomData<R>,
}

impl<F, R: Ring> FFT64Plan<F, R>
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    pub fn fft(&self) -> &ReimFFTTable<F> {
        &self.fft
    }

    pub fn ifft(&self) -> &ReimIFFTTable<F> {
        &self.ifft
    }
}

/// Ring-specific construction of an [`FFT64Plan`].
pub trait FFT64PlanNew: Sized {
    /// Builds the plan for ring degree `n`.
    fn new(n: usize) -> Self;
}

pub(super) fn plan_half_degree(n: usize) -> usize {
    assert!(
        n >= 2 && n.is_power_of_two(),
        "ring degree must be a power of two >= 2, got {n}"
    );
    n >> 1
}

/// Access to the precomputed FFT/iFFT tables stored inside a `Module<B>` handle.
///
/// Backend crates implement [`FFTHandleProvider`] for their concrete handle type.
/// `poulpy-hal` then provides this blanket trait on `Module<B>`, which lets family
/// defaults share the same FFT64 handle contract across scalar and accelerated backends.
pub trait FFTModuleHandle<F, Rg: Ring = Standard>: poulpy_hal::api::ModuleN
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    fn get_fft_plan(&self, n: usize) -> &FFT64Plan<F, Rg>;

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
/// Implementors must return references that stay valid for the lifetime of `&self`, to plans of both rings
/// fully built before `Module::new()` returns.
pub unsafe trait FFTHandleProvider<F, Rg: Ring = Standard>
where
    F: Float + FloatConst + Debug + Zeroable + Send + Sync,
{
    fn get_fft_plan(&self, n: usize) -> &FFT64Plan<F, Rg>;
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

impl<BE: Backend<ZnxWord = i64>, Rg: Ring> FFTModuleHandle<BE::DftWord, Rg> for Module<BE>
where
    BE::DftWord: Float + FloatConst + Debug + Send + Sync,
    BE::Handle: FFTHandleProvider<BE::DftWord, Rg>,
{
    fn get_fft_plan(&self, n: usize) -> &FFT64Plan<BE::DftWord, Rg> {
        unsafe { (&*self.ptr()).get_fft_plan(n) }
    }
}

impl<F: Float + FloatConst + Debug + Zeroable + Send + Sync> ReimM for FFT64Plan<F> {
    fn m(&self) -> usize {
        self.fft().m()
    }
}

impl<F: Float + FloatConst + Debug + Zeroable + Send + Sync> ReimFFT<F> for FFT64Plan<F> {
    fn reim_fft(&self, data: &mut [F]) {
        self.fft().execute(data);
    }
}

impl<F: Float + FloatConst + Debug + Zeroable + Send + Sync> ReimIFFT<F> for FFT64Plan<F> {
    fn reim_ifft(&self, data: &mut [F]) {
        self.ifft().execute(data);
    }
}
