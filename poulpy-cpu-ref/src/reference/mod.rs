//! Pure-Rust reference implementations of all polynomial operations.
//!
//! Contains scalar polynomial arithmetic (`znx`), vector-level operations
//! (`vec_znx`), and an FFT64 implementation (`fft64`). Used as a
//! correctness oracle for backend testing via the
//! [`poulpy_hal::test_suite`] module.

pub mod fft64;
pub mod ntt4x30;
pub mod vec_znx;
pub mod vmp_select;

#[derive(Clone, Copy)]
pub(crate) struct SendPtr<T>(*mut T);

impl<T> SendPtr<T> {
    pub(crate) fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    pub(crate) fn get(self) -> *mut T {
        self.0
    }
}

// Dereferencing remains unsafe; users must enforce the pointee's aliasing rules.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// Re-exported from [`poulpy_hal::reference::znx`], where the portable scalar
/// kernels now live so that every crate can reach them without depending on a
/// backend.
pub use poulpy_hal::reference::znx;

/// Kernel-side mutable word access to a typed destination view: a statement-scoped
/// state-erased view over the same storage, via the sealed capability
/// `poulpy_hal::oep::vec_znx_kernel_words_mut`. Each call marks a kernel write whose
/// enclosing operation declares the destination's postcondition (spec invariants 7
/// and 12); the erased view exists only for the statement that writes.
pub fn kernel_words_mut<
    D: poulpy_hal::layouts::HostDataMut,
    W: poulpy_hal::layouts::ZnxWord,
    S: poulpy_hal::layouts::CoefficientState,
>(
    v: &mut poulpy_hal::layouts::VecZnx<D, W, S>,
) -> poulpy_hal::layouts::VecZnx<&mut [u8], W, poulpy_hal::layouts::CoeffUnnormalized> {
    unsafe { poulpy_hal::oep::vec_znx_kernel_words_mut(v) }
}
