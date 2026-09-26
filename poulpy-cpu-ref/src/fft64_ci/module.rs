//! Backend handle and module initialization for [`FFT64CIRef`](super::FFT64CIRef).

use super::FFT64CIRef;

use std::ptr::NonNull;

use poulpy_hal::layouts::Backend;

use crate::{
    reference::fft64::module::{FFT64HandleFactory, FFT64Plan, FFT64PlanSet, FFTHandleProvider},
    ring::ConjugateInvariant,
};

/// Opaque handle for the [`FFT64CIRef`](super::FFT64CIRef) backend, holding
/// the conjugate invariant FFT plans.
#[repr(C)]
pub struct FFT64CIRefHandle {
    ring_plans: FFT64PlanSet<f64, ConjugateInvariant>,
    table_cache: crate::table_cache::ModuleTableCache,
}

impl poulpy_hal::execution::ScratchWorkers for FFT64CIRef {}

impl poulpy_hal::layouts::MaxBase2k for FFT64CIRef {
    fn max_base2k(n: usize, products: usize, failure_bits: usize, squaring: bool) -> Option<usize> {
        Some(poulpy_hal::layouts::max_base2k_fft64::<Self>(
            n,
            products,
            failure_bits,
            squaring,
        ))
    }
}

impl Backend for FFT64CIRef {
    crate::forward_backend_storage!(crate::FFT64Ref);

    type Handle = FFT64CIRefHandle;
    type Ring = ConjugateInvariant;

    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }
}

/// # Safety
///
/// The returned handle must be fully initialized for `n`.
unsafe impl FFT64HandleFactory for FFT64CIRefHandle {
    fn create_fft64_handle(n: usize) -> Self {
        FFT64CIRefHandle {
            table_cache: Default::default(),
            ring_plans: FFT64PlanSet::new(n),
        }
    }
}

unsafe impl FFTHandleProvider<f64> for FFT64CIRefHandle {
    type Ring = ConjugateInvariant;
    fn get_fft_plan(&self, n: usize) -> &FFT64Plan<f64, ConjugateInvariant> {
        self.ring_plans.for_ring(n)
    }
}

unsafe impl crate::table_cache::ModuleTableCacheProvider for FFT64CIRefHandle {
    fn module_plan_cache(&self) -> &crate::table_cache::ModuleTableCache {
        &self.table_cache
    }
}
