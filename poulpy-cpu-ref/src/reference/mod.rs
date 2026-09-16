//! Pure-Rust reference implementations of all polynomial operations.
//!
//! Contains scalar polynomial arithmetic (`znx`), vector-level operations
//! (`vec_znx`), and an FFT64 implementation (`fft64`). Used as a
//! correctness oracle for backend testing via the
//! [`poulpy_hal::test_suite`] module.

pub mod fft64;
pub mod normalization;
pub mod ntt4x30;
pub mod vec_znx;
pub mod vmp_select;

/// `log2(N/n)` for a degree-`n` operand under a module of degree `N`; `0` when
/// the degrees match.
///
/// A sparse operand needs a power-of-two degree that divides `N` and is at
/// least `floor`, the backend's `MIN_SPARSE_DEGREE`. Asserted once, at kernel
/// entry.
pub fn sparse_log_gap(module_n: usize, n: usize, floor: usize) -> usize {
    if n == module_n {
        return 0;
    }
    assert!(
        n.is_power_of_two() && n >= floor && n < module_n && module_n.is_multiple_of(n),
        "operand of degree {n} does not embed into the module degree {module_n} (floor {floor})"
    );
    (module_n / n).trailing_zeros() as usize
}

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

/// Portable HAL primitives and CPU-specific normalization kernels.
pub mod znx;
