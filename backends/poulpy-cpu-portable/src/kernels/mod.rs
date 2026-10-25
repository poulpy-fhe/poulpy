//! Portable production implementations of polynomial operations.
//!
//! Contains scalar polynomial arithmetic (`znx`), vector-level operations
//! (`vec_znx`), and FFT64 and NTT4x30 kernels shared by the portable and SIMD
//! backends. Independent correctness oracles live in `poulpy-cpu-oracle`.
//!
//! Every exported free kernel function uses the `_portable` suffix, including
//! generic compositions, table builders, and scratch-size queries. Backend
//! trait methods retain their contract names, and inherent methods on plans
//! and tables retain their type-scoped names.

pub mod fft64;
pub mod galois;
pub mod ntt4x30;
pub mod vec_znx;
pub mod vec_znx_big_normalize;
pub mod vmp_select;
pub mod znx_normalize;

pub use galois::{GaloisSlots, RightRead, right_read_fft64, right_read_ntt};

/// `log2(N / n)` for an operand of degree `n` under a module of degree `N`:
/// zero for a dense operand. An embedded operand needs a power-of-two degree that
/// divides `N` and is at least 8, the CPU block floor (spec 4.5, #266). Asserted
/// once, at kernel entry; the kernels' loops then index on it without checks.
pub fn embed_log_gap_portable(module_n: usize, n: usize) -> usize {
    if n == module_n {
        return 0;
    }
    assert!(
        n.is_power_of_two() && n >= 8 && n < module_n && module_n.is_multiple_of(n),
        "operand of degree {n} does not embed into the module degree {module_n}"
    );
    (module_n / n).trailing_zeros() as usize
}

/// A pointer shared by tasks that write distinct ranges behind it.
///
/// Shared by the CPU backends and their Rayon variants.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct SendPtr<T>(pub *mut T);

impl<T> SendPtr<T> {
    #[inline(always)]
    pub fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    #[inline(always)]
    pub fn get(self) -> *mut T {
        self.0
    }
}

// Dereferencing remains unsafe; users must enforce the pointee's aliasing rules.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

std::thread_local! {
    static THREAD_BUFFER: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Runs `f` on `len` words of a buffer this thread keeps across calls, for kernels whose HAL entry point takes no scratch.
///
/// The buffer grows to the largest `len` asked on the thread and is never shrunk, and its words hold whatever an earlier call left.
/// A call nested in `f` on the same thread gets a buffer of its own.
#[doc(hidden)]
pub fn with_thread_buffer<T>(len: usize, f: impl FnOnce(&mut [u64]) -> T) -> T {
    THREAD_BUFFER.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buf) => {
            if buf.len() < len {
                buf.resize(len, 0);
            }
            f(&mut buf[..len])
        }
        Err(_) => f(&mut vec![0u64; len]),
    })
}

/// Portable HAL primitives.
pub mod znx;

pub mod noise;
