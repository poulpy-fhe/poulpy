//! Compile-time ring selection shared by the CPU backend families.

use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

/// Sealed ring choice for the CPU transform implementations.
pub trait CpuRing: sealed::Sealed + Copy + Eq + Send + Sync + 'static {
    /// Whether coefficients use the conjugate-invariant basis.
    const IS_CI: bool;
    /// CI-specific state, absent from standard-ring handles.
    type Data<T: Send + Sync>: RingData<T> + Send + Sync;
}

/// The standard negacyclic ring.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Standard;

/// The conjugate-invariant subring of a degree-doubled negacyclic ring.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConjugateInvariant;

impl sealed::Sealed for Standard {}
impl sealed::Sealed for ConjugateInvariant {}

impl CpuRing for Standard {
    const IS_CI: bool = false;
    type Data<T: Send + Sync> = NoRingData<T>;
}

impl CpuRing for ConjugateInvariant {
    const IS_CI: bool = true;
    type Data<T: Send + Sync> = T;
}

/// Statically present or absent CI transform state.
pub trait RingData<T>: Sized {
    /// Builds state only when the ring requires it.
    fn new(build: impl FnOnce() -> T) -> Self;
    /// Borrows the state, if present for this ring.
    fn get(&self) -> Option<&T>;
}

/// Zero-sized storage for standard rings without CI transform state.
pub struct NoRingData<T>(PhantomData<fn() -> T>);

impl<T> RingData<T> for NoRingData<T> {
    #[inline]
    fn new(_: impl FnOnce() -> T) -> Self {
        Self(PhantomData)
    }

    #[inline]
    fn get(&self) -> Option<&T> {
        None
    }
}

impl<T> RingData<T> for T {
    #[inline]
    fn new(build: impl FnOnce() -> T) -> Self {
        build()
    }

    #[inline]
    fn get(&self) -> Option<&T> {
        Some(self)
    }
}
