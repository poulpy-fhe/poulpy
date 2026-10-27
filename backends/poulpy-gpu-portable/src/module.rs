//! Module handle and construction.

use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Module, ModulePlanCache, Ring},
    oep::HalModuleImpl,
};

use crate::{
    NTT4x30Gpu,
    ntt4x30::Tables,
    runtime::{GpuBuf, GpuRuntime},
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
    /// the in-place permutations, of the inverse transform that keeps its
    /// source and of the wide-radix normalization of an inverse transform, whose
    /// queries cover the kernel paths. Each call takes its own from the
    /// allocator, which recycles freed memory, so that threads sharing the
    /// module never share a temporary. The temporary is zeroed before it is
    /// released: it may hold secret-derived words, out of reach of the
    /// caller's `ScratchArena::wipe`. On an Apple M5 Pro (Metal), the zeroing
    /// costs `vec_znx_idft_apply` of the standard bench 4% at degree `2^14`
    /// and 10 to 14% at `2^15`.
    pub(crate) fn with_temp<T>(&self, len: usize, f: impl FnOnce(&mut GpuBuf<RT>) -> T) -> T {
        let mut temp = GpuBuf::uninit(len);
        let result = f(&mut temp);
        temp.wipe();
        result
    }
}

/// The device state of `module`.
pub(crate) fn handle<RT: GpuRuntime, R: Ring>(module: &Module<NTT4x30Gpu<RT, R>>) -> &GpuHandle<RT> {
    // SAFETY: the pointer was created by `HalModuleImpl::new` and lives as long as the module.
    unsafe { &*module.ptr() }
}

unsafe impl<RT: GpuRuntime, R: Ring> HalModuleImpl for NTT4x30Gpu<RT, R> {
    fn new(n: u64) -> Module<Self> {
        // The degree is checked before any table is built or uploaded.
        const { assert!(1 << crate::ntt4x30::tables::MIN_LOG_N == <Self as poulpy_hal::layouts::Backend>::MIN_DEGREE) };
        assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
        assert!(
            n as usize >= <Self as poulpy_hal::layouts::Backend>::MIN_DEGREE,
            "n is below the backend's minimum degree"
        );
        assert!(
            n <= poulpy_hal::layouts::MAX_RING_DEGREE,
            "ring degree {n} exceeds the supported maximum {}",
            poulpy_hal::layouts::MAX_RING_DEGREE
        );
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
        crate::runtime::synchronize::<RT>();
    }
}
