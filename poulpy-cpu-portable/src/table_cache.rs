//! Compatibility re-exports for the module plan cache, which now lives in
//! `poulpy-hal` so CPU and future device backends share the same neutral
//! ownership contract.

pub use poulpy_hal::layouts::{ModulePlanCache as ModuleTableCache, ModulePlanCacheProvider as ModuleTableCacheProvider};

use poulpy_hal::layouts::{Backend, Module};

/// Access to the module-owned [`ModuleTableCache`] through `Module<BE>`.
pub trait ModuleTableCacheAccess {
    fn module_table_cache(&self) -> &ModuleTableCache;
}

impl<BE: Backend<ZnxWord = i64>> ModuleTableCacheAccess for Module<BE>
where
    BE::Handle: ModuleTableCacheProvider,
{
    fn module_table_cache(&self) -> &ModuleTableCache {
        unsafe { (*self.ptr()).module_plan_cache() }
    }
}
