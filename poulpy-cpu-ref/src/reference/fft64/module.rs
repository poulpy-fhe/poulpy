use std::fmt::Debug;

use rand_distr::num_traits::{Float, FloatConst};

use crate::{
    layouts::{Backend, Module},
    reference::fft64::reim::{ReimFFTTable, ReimIFFTTable},
};

/// Access to the precomputed FFT/iFFT tables stored inside a `Module<B>` handle.
///
/// Backend crates implement [`FFTHandleProvider`] for their concrete handle type.
/// `poulpy-hal` then provides this blanket trait on `Module<B>`, which lets family
/// defaults share the same FFT64 handle contract across scalar and accelerated backends.
pub trait FFTModuleHandle<F>
where
    F: Float + FloatConst + Debug,
{
    fn get_fft_table(&self) -> &ReimFFTTable<F>;
    fn get_ifft_table(&self) -> &ReimIFFTTable<F>;
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
    fn get_fft_table(&self) -> &ReimFFTTable<F>;
    fn get_ifft_table(&self) -> &ReimIFFTTable<F>;
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
    fn get_fft_table(&self) -> &ReimFFTTable<BE::ScalarPrep> {
        unsafe { (&*self.ptr()).get_fft_table() }
    }

    fn get_ifft_table(&self) -> &ReimIFFTTable<BE::ScalarPrep> {
        unsafe { (&*self.ptr()).get_ifft_table() }
    }
}
