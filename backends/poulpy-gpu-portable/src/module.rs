//! Module handle and construction.

use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Module, ModulePlanCache, Ring},
    oep::HalModuleImpl,
};

use crate::{
    NTT4x30Gpu,
    ntt4x30::Tables,
    runtime::{GpuBuf, GpuRuntime, access},
};

/// State a module keeps on the device.
pub struct GpuHandle<RT: GpuRuntime> {
    n: usize,
    /// Transform tables of every degree the module serves.
    tables: Tables<RT>,
    /// Plans the layers above memoize per module, such as the encoding tables.
    #[cfg_attr(not(feature = "enable-ckks"), allow(dead_code))]
    plans: ModulePlanCache,
}

impl<RT: GpuRuntime> GpuHandle<RT> {
    /// Largest ring degree the module serves.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The module-owned plan cache.
    #[cfg_attr(not(feature = "enable-ckks"), allow(dead_code))]
    pub(crate) fn plan_cache(&self) -> &ModulePlanCache {
        &self.plans
    }

    pub(crate) fn tables(&self) -> &Tables<RT> {
        &self.tables
    }

    /// Runs `f` on an uninitialized device temporary of `len` bytes, the one of
    /// the in-place permutations and of the inverse transform that keeps its
    /// source. Each call takes its own from the allocator, which recycles
    /// memory per stream: threads sharing the module run on different streams
    /// and must not share a temporary.
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
            tables: Tables::new(n as usize, R::CYCLOTOMIC_ORDER_FACTOR == 4),
            plans: ModulePlanCache::default(),
        });
        // SAFETY: the pointer comes from a live `Box`, which the module owns from here on.
        unsafe { Module::from_nonnull(NonNull::new_unchecked(Box::into_raw(handle)), n) }
    }

    fn synchronize(_module: &Module<Self>) {
        crate::runtime::host_event(crate::runtime::HostEvent::Sync, 0);
        let _access = access::shared();
        pollster::block_on(RT::client().sync()).unwrap_or_else(|error| panic!("device synchronization failed: {error:?}"));
    }
}
