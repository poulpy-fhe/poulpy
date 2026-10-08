//! Module handle and construction.

use std::{marker::PhantomData, ptr::NonNull};

use poulpy_hal::{
    layouts::{Module, Ring},
    oep::HalModuleImpl,
};

use crate::{
    NTT4x30Gpu,
    runtime::{GpuBuf, GpuRuntime},
};

/// State a module keeps on the device.
pub struct GpuHandle<RT: GpuRuntime> {
    n: usize,
    _runtime: PhantomData<fn() -> RT>,
}

impl<RT: GpuRuntime> GpuHandle<RT> {
    /// Largest ring degree the module serves.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Runs `f` on an uninitialized device temporary of `len` bytes, the one of
    /// the in-place permutations. Each call takes its own from the allocator,
    /// which recycles memory per stream: threads sharing the module run on
    /// different streams and must not share a temporary.
    pub(crate) fn with_temp<T>(&self, len: usize, f: impl FnOnce(&mut GpuBuf<RT>) -> T) -> T {
        f(&mut GpuBuf::uninit(len))
    }
}

/// The device state of `module`.
pub(crate) fn handle<RT: GpuRuntime, R: Ring>(module: &Module<NTT4x30Gpu<RT, R>>) -> &GpuHandle<RT> {
    // SAFETY: the pointer was created by `HalModuleImpl::new` and lives as long as the module.
    unsafe { &*module.ptr() }
}

unsafe impl<RT: GpuRuntime, R: Ring> HalModuleImpl for NTT4x30Gpu<RT, R> {
    fn new(n: u64) -> Module<Self> {
        // The client is created here so that a missing device fails at construction.
        RT::client();
        let handle = Box::new(GpuHandle::<RT> {
            n: n as usize,
            _runtime: PhantomData,
        });
        unsafe { Module::from_nonnull(NonNull::new_unchecked(Box::into_raw(handle)), n) }
    }

    fn synchronize(_module: &Module<Self>) {
        pollster::block_on(RT::client().sync()).unwrap_or_else(|error| panic!("device synchronization failed: {error:?}"));
    }
}
