//! Module-owned transform and scheme tables.

pub use poulpy_hal::layouts::{ModulePlanCache as ModuleTableCache, ModulePlanCacheProvider as ModuleTableCacheProvider};

#[cfg(feature = "enable-ckks")]
use poulpy_hal::layouts::{Backend, Module};

/// Access to the module-owned [`ModuleTableCache`] through `Module<BE>`.
#[cfg(feature = "enable-ckks")]
pub trait ModuleTableCacheAccess {
    fn module_table_cache(&self) -> &ModuleTableCache;
}

#[cfg(feature = "enable-ckks")]
impl<BE: Backend<ZnxWord = i64>> ModuleTableCacheAccess for Module<BE>
where
    BE::Handle: ModuleTableCacheProvider,
{
    fn module_table_cache(&self) -> &ModuleTableCache {
        unsafe { (*self.ptr()).module_plan_cache() }
    }
}
