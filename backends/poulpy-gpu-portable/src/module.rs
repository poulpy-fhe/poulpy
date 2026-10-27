//! Module handle and construction.

use std::{
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

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
    /// See [`Self::convolution_capacity_step`].
    capacity_step: AtomicUsize,
}

impl<RT: GpuRuntime> GpuHandle<RT> {
    /// Largest ring degree the module serves.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Limbs by which the convolution kernels grow the operand copies each
    /// thread holds. A kernel is compiled per capacity: a larger step compiles
    /// fewer kernels, and spends registers on limbs a product does not read.
    /// The best step depends on the device's registers.
    pub fn convolution_capacity_step(&self) -> usize {
        self.capacity_step.load(Ordering::Relaxed)
    }

    /// Sets [`Self::convolution_capacity_step`], which changes the launches of
    /// the convolutions: a closure recorded with [`crate::Replay`] before the
    /// change no longer repeats its recording after it.
    ///
    /// # Panics
    ///
    /// Panics unless `limbs` is in `1..=MAX_CAPACITY_STEP`: a thread holds
    /// up to a step more words per operand than a product reads.
    pub fn set_convolution_capacity_step(&self, limbs: usize) {
        assert!(
            (1..=MAX_CAPACITY_STEP).contains(&limbs),
            "the capacity step is between one limb and MAX_CAPACITY_STEP"
        );
        self.capacity_step.store(limbs, Ordering::Relaxed);
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
    /// module never share a temporary.
    ///
    /// The temporary may hold secret-derived words, out of reach of the
    /// caller's `ScratchArena::wipe`: `f` leaves it zeroed, the kernel reading
    /// it last clearing each word it reads, at no launch of its own. When `f`
    /// panics, the temporary is zeroed by a launch before it is released.
    pub(crate) fn with_temp<T>(&self, len: usize, f: impl FnOnce(&mut GpuBuf<RT>) -> T) -> T {
        /// Zeroes the temporary when `f` unwinds.
        struct Unwinding<RT: GpuRuntime>(GpuBuf<RT>);
        impl<RT: GpuRuntime> Drop for Unwinding<RT> {
            fn drop(&mut self) {
                if std::thread::panicking() {
                    self.0.wipe();
                }
            }
        }
        let mut temp = Unwinding(GpuBuf::uninit(len));
        f(&mut temp.0)
    }
}

/// Largest [`GpuHandle::set_convolution_capacity_step`]: the operand copies
/// of a convolution thread live in its registers and local memory.
pub const MAX_CAPACITY_STEP: usize = 64;

/// The device state of `module`.
pub fn handle<RT: GpuRuntime, R: Ring>(module: &Module<NTT4x30Gpu<RT, R>>) -> &GpuHandle<RT> {
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
            capacity_step: AtomicUsize::new(crate::ntt4x30::CAPACITY_STEP),
        });
        // SAFETY: the pointer comes from a live `Box`, which the module owns from here on.
        unsafe { Module::from_nonnull(NonNull::new_unchecked(Box::into_raw(handle)), n) }
    }

    fn synchronize(_module: &Module<Self>) {
        crate::runtime::host_event(crate::runtime::HostEvent::Sync, 0);
        crate::runtime::synchronize::<RT>();
    }
}
