//! Pure-Rust reference implementations of all polynomial operations.
//!
//! Contains scalar polynomial arithmetic (`znx`), vector-level operations
//! (`vec_znx`), and an FFT64 implementation (`fft64`). Used as a
//! correctness oracle for backend testing via the
//! [`poulpy_hal::test_suite`] module.

pub mod fft64;
pub mod ntt4x30;
pub mod vec_znx;

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
