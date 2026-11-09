//! One table per degree and ring, for backend handles serving both rings.

use std::{any::TypeId, marker::PhantomData};

use poulpy_hal::layouts::{ConjugateInvariant, Ring, Standard};

use crate::kernels::{
    fft64::module::FFT64Plan,
    ntt4x30::{primes::PrimeSetCrt4, vec_znx_dft::NttPlan},
};

/// A type indexed by the ring: the plan or table of one degree a handle holds for each ring.
pub trait RingFamily {
    type Of<R: Ring>: Send + Sync;
}

/// The tables of both rings of every power-of-two degree from a minimum to the module degree, all built on construction.
pub struct PerRing<F: RingFamily> {
    log_min_n: usize,
    /// Indexed by `log2(n) - log_min_n`, as is `conjugate_invariant`.
    standard: Vec<F::Of<Standard>>,
    conjugate_invariant: Vec<F::Of<ConjugateInvariant>>,
}

impl<F: RingFamily> PerRing<F> {
    /// Builds `standard(n)`, then `derive(&standard(n), n)`, for every power of two `n` from `min_n` to `max_n`.
    pub fn new(
        min_n: usize,
        max_n: usize,
        standard: impl Fn(usize) -> F::Of<Standard>,
        derive: impl Fn(&F::Of<Standard>, usize) -> F::Of<ConjugateInvariant>,
    ) -> Self {
        assert!(
            min_n.is_power_of_two() && max_n.is_power_of_two() && min_n <= max_n,
            "table degrees must be powers of two, the minimum at most the maximum"
        );
        let log_min_n = min_n.trailing_zeros() as usize;
        let standard: Vec<F::Of<Standard>> = (log_min_n..=max_n.trailing_zeros() as usize)
            .map(|log_n| standard(1 << log_n))
            .collect();
        let conjugate_invariant = standard
            .iter()
            .enumerate()
            .map(|(i, table)| derive(table, min_n << i))
            .collect();
        Self {
            log_min_n,
            standard,
            conjugate_invariant,
        }
    }

    /// The table of degree `n` on ring `R`.
    ///
    /// Panics unless `n` is a power of two between the minimum and the maximum degree.
    pub fn on_ring<R: Ring>(&self, n: usize) -> &F::Of<R> {
        let i = (n.trailing_zeros() as usize).wrapping_sub(self.log_min_n);
        assert!(n.is_power_of_two() && i < self.standard.len(), "no table of this degree");
        if TypeId::of::<R>() == TypeId::of::<Standard>() {
            // SAFETY: `R` is `Standard`, so `F::Of<R>` and `F::Of<Standard>` are one type.
            unsafe { &*(&self.standard[i] as *const F::Of<Standard> as *const F::Of<R>) }
        } else {
            // SAFETY: `Ring` is sealed with two implementors, so `R` is `ConjugateInvariant`.
            unsafe { &*(&self.conjugate_invariant[i] as *const F::Of<ConjugateInvariant> as *const F::Of<R>) }
        }
    }
}

/// The FFT64 plan of one degree on each ring.
pub struct FFT64Plans;

impl RingFamily for FFT64Plans {
    type Of<R: Ring> = FFT64Plan<f64, R>;
}

/// The NTT plan of one degree on each ring over the primes `P`.
pub struct NttPlans<P>(PhantomData<P>);

impl<P: PrimeSetCrt4> RingFamily for NttPlans<P> {
    type Of<R: Ring> = NttPlan<P, R>;
}
