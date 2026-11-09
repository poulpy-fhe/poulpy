//! One value per ring, for backend handles serving both rings.

use std::{any::TypeId, marker::PhantomData, sync::OnceLock};

use poulpy_hal::layouts::{ConjugateInvariant, Ring, Standard};

use crate::kernels::{
    fft64::module::FFT64PlanSet,
    ntt4x30::{primes::PrimeSetCrt4, vec_znx_dft::NttPlanSet},
};

/// A type indexed by the ring: the plans or tables a handle holds for each ring.
pub trait RingFamily {
    type Of<R: Ring>: Send + Sync;
}

/// A standard-ring value built eagerly and a conjugate-invariant value built on
/// first use, so that a module only serving the standard ring never builds the
/// second.
pub struct PerRing<F: RingFamily> {
    standard: F::Of<Standard>,
    conjugate_invariant: OnceLock<F::Of<ConjugateInvariant>>,
    build: fn(usize) -> F::Of<ConjugateInvariant>,
    max_n: usize,
}

impl<F: RingFamily> PerRing<F> {
    /// `build(max_n)` makes the conjugate-invariant value when first asked for.
    pub fn new(standard: F::Of<Standard>, max_n: usize, build: fn(usize) -> F::Of<ConjugateInvariant>) -> Self {
        Self {
            standard,
            conjugate_invariant: OnceLock::new(),
            build,
            max_n,
        }
    }

    /// The value of ring `R`.
    pub fn on_ring<R: Ring>(&self) -> &F::Of<R> {
        if TypeId::of::<R>() == TypeId::of::<Standard>() {
            // SAFETY: `R` is `Standard`, so `F::Of<R>` and `F::Of<Standard>` are one type.
            unsafe { &*(&self.standard as *const F::Of<Standard> as *const F::Of<R>) }
        } else {
            let ci = self.conjugate_invariant.get_or_init(|| (self.build)(self.max_n));
            // SAFETY: `Ring` is sealed with two implementors, so `R` is `ConjugateInvariant`.
            unsafe { &*(ci as *const F::Of<ConjugateInvariant> as *const F::Of<R>) }
        }
    }
}

/// The FFT64 plans of each ring.
pub struct FFT64Plans;

impl RingFamily for FFT64Plans {
    type Of<R: Ring> = FFT64PlanSet<f64, R>;
}

/// The NTT plans of each ring over the primes `P`.
pub struct NttPlans<P>(PhantomData<P>);

impl<P: PrimeSetCrt4> RingFamily for NttPlans<P>
where
    NttPlanSet<P, Standard>: Send + Sync,
{
    type Of<R: Ring> = NttPlanSet<P, R>;
}
