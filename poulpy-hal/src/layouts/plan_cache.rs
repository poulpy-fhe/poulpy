//! Backend-handle-owned cache for optional, typed execution plans.
//!
//! Ring backends have a closed set of native plans and normally store those
//! directly in their handle. Higher layers may need additional plans whose
//! concrete type is selected by a generic API parameter (for example CKKS
//! encoding precision). `ModulePlanCache` provides the small amount of type
//! erasure needed for those open-ended extensions without teaching the HAL
//! about any particular scheme or scalar type.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Mutex, OnceLock, RwLock},
};

use anyhow::Result;

/// A heterogeneous cache of immutable plan families.
///
/// One entry should normally contain the complete geometric family for a
/// concrete logical plan key, rather than one entry per ring dimension. The
/// key and stored value are separate type parameters: callers can give a
/// stable identity to a service even if its concrete plan representation
/// changes. Access is closure-based so callers borrow the cached value while
/// the cache itself is borrowed; no reference counting is required. The
/// erased entries are boxed only to keep their addresses stable when the map
/// grows, allowing the map lock to be released before initialization or use.
#[derive(Default)]
pub struct ModulePlanCache {
    plans: RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>,
}

struct PlanEntry<T> {
    value: OnceLock<T>,
    initialize: Mutex<()>,
}

impl<T> Default for PlanEntry<T> {
    fn default() -> Self {
        Self {
            value: OnceLock::new(),
            initialize: Mutex::new(()),
        }
    }
}

impl ModulePlanCache {
    /// Runs `use_plans` with the value stored under `K`, constructing it once
    /// with `init`.
    ///
    /// `init` may access other keys in this cache, but must not recursively
    /// request its own still-uninitialized key.
    pub fn with_or_create<K, T, R>(&self, init: impl FnOnce() -> Result<T>, use_plans: impl FnOnce(&T) -> R) -> Result<R>
    where
        K: 'static,
        T: Any + Send + Sync,
    {
        let key = TypeId::of::<K>();
        let mut entry = {
            let entries = self.plans.read().expect("module plan cache is poisoned");
            entries.get(&key).map(|entry| {
                entry
                    .downcast_ref::<PlanEntry<T>>()
                    .expect("module plan cache key has a different value type") as *const PlanEntry<T>
            })
        };

        if entry.is_none() {
            let mut entries = self.plans.write().expect("module plan cache is poisoned");
            let stored = entries.entry(key).or_insert_with(|| Box::new(PlanEntry::<T>::default()));
            entry = Some(
                stored
                    .downcast_ref::<PlanEntry<T>>()
                    .expect("module plan cache key has a different value type") as *const PlanEntry<T>,
            );
        }

        // SAFETY: entries are boxed and never removed or replaced, so their
        // pointees keep a stable address even when the map grows. Borrowing
        // `self` also guarantees that the map outlives this callback.
        let entry = unsafe { &*entry.expect("module plan cache entry disappeared") };
        if entry.value.get().is_none() {
            // The per-key lock prevents duplicate construction without
            // holding the map lock, so independent plan families may be
            // initialized concurrently or compose other cached families.
            let _initialize = entry.initialize.lock().expect("module plan cache initializer is poisoned");
            if entry.value.get().is_none() {
                entry
                    .value
                    .set(init()?)
                    .unwrap_or_else(|_| unreachable!("plan entry initialized while holding its initializer lock"));
            }
        }
        let plans = entry.value.get().expect("module plan cache entry was not initialized");
        Ok(use_plans(plans))
    }
}

/// Implemented by backend handles that own an extension-plan cache.
///
/// Keeping the cache in the handle makes its location and destruction order a
/// backend concern, which is important for plans that own device resources.
///
/// # Safety
///
/// The returned reference must remain valid for the lifetime of `&self`.
pub unsafe trait ModulePlanCacheProvider {
    fn module_plan_cache(&self) -> &ModulePlanCache;
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::ModulePlanCache;

    struct KeyA;
    struct KeyB;

    #[test]
    fn logical_keys_can_store_the_same_value_type_independently() {
        let cache = ModulePlanCache::default();
        assert_eq!(cache.with_or_create::<KeyA, _, _>(|| Ok(3usize), |value| *value).unwrap(), 3);
        assert_eq!(cache.with_or_create::<KeyB, _, _>(|| Ok(7usize), |value| *value).unwrap(), 7);
    }

    #[test]
    fn concurrent_first_use_initializes_once() {
        let cache = Arc::new(ModulePlanCache::default());
        let constructions = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let cache = Arc::clone(&cache);
                let constructions = Arc::clone(&constructions);
                scope.spawn(move || {
                    let value = cache
                        .with_or_create::<KeyA, _, _>(
                            || {
                                constructions.fetch_add(1, Ordering::Relaxed);
                                Ok(11usize)
                            },
                            |value| *value,
                        )
                        .unwrap();
                    assert_eq!(value, 11);
                });
            }
        });
        assert_eq!(constructions.load(Ordering::Relaxed), 1);
    }
}
