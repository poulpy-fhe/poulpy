//! Backend handle and module initialisation for [`NTT4x30CIRef`](super::NTT4x30CIRef).

use super::NTT4x30CIRef;

use std::ptr::NonNull;

use poulpy_hal::layouts::Backend;

use crate::{
    reference::ntt4x30::{
        mat_vec::{BbbMeta, BbcMeta},
        primes::Primes30,
        vec_znx_dft::{NttHandleFactory, NttHandleProvider, NttPlan, NttPlanSet},
    },
    ring::ConjugateInvariant,
};

/// Opaque handle for the [`NTT4x30CIRef`](super::NTT4x30CIRef) backend,
/// holding the conjugate invariant NTT plans and the multiply-accumulate
/// metadata.
#[repr(C)]
pub struct NTT4x30CIRefHandle {
    ring_plans: NttPlanSet<Primes30, ConjugateInvariant>,
    meta_bbc: BbcMeta<Primes30>,
    meta_bbb: BbbMeta<Primes30>,
    table_cache: crate::table_cache::ModuleTableCache,
}

impl poulpy_hal::execution::ScratchWorkers for NTT4x30CIRef {}

impl poulpy_hal::layouts::MaxBase2k for NTT4x30CIRef {
    fn max_base2k(n: usize, products: usize, failure_bits: usize, squaring: bool) -> Option<usize> {
        Some(poulpy_hal::layouts::max_base2k_ntt::<Self>(
            <Primes30 as poulpy_hal::layouts::PrimeSet>::LOG_Q_PRODUCT,
            n,
            products,
            failure_bits,
            squaring,
        ))
    }
}

impl Backend for NTT4x30CIRef {
    crate::forward_backend_storage!(crate::NTT4x30Ref);

    type Handle = NTT4x30CIRefHandle;
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
unsafe impl NttHandleFactory for NTT4x30CIRefHandle {
    fn create_ntt_handle(n: usize) -> Self {
        NTT4x30CIRefHandle {
            table_cache: Default::default(),
            ring_plans: NttPlanSet::new(n),
            meta_bbc: BbcMeta::new(),
            meta_bbb: BbbMeta::new(),
        }
    }
}

/// # Safety
///
/// The returned references are valid for the lifetime of `&self`.
/// All fields are fully initialised by the [`NttHandleFactory`] impl above.
unsafe impl NttHandleProvider for NTT4x30CIRefHandle {
    type Ring = ConjugateInvariant;
    fn get_ntt_plan(&self, n: usize) -> &NttPlan<Primes30, ConjugateInvariant> {
        self.ring_plans.for_ring(n)
    }

    fn get_bbc_meta(&self) -> &BbcMeta<Primes30> {
        &self.meta_bbc
    }

    fn get_bbb_meta(&self) -> &BbbMeta<Primes30> {
        &self.meta_bbb
    }
}

unsafe impl crate::table_cache::ModuleTableCacheProvider for NTT4x30CIRefHandle {
    fn module_plan_cache(&self) -> &crate::table_cache::ModuleTableCache {
        &self.table_cache
    }
}
