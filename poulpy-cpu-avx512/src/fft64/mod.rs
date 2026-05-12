mod convolution;
mod module;
mod reim;
mod reim4;

/// AVX-512F-accelerated CPU backend for Poulpy HAL.
///
/// `FFT64Avx512` is a zero-sized marker type that selects the AVX-512F-optimized CPU backend
/// when used as the type parameter `B` in [`Module<B>`](poulpy_hal::layouts::Module)
/// and related HAL types. It implements all open extension point (OEP) traits from
/// `poulpy_hal::oep` using hand-tuned AVX-512F SIMD intrinsics and assembly kernels.
///
/// # Backend characteristics
///
/// - **ScalarPrep**: `f64` — DFT-domain coefficients are 64-bit IEEE 754 floats.
/// - **ScalarBig**: `i64` — large-coefficient ring elements use 64-bit signed integers.
/// - **FFT tables**: precomputed twiddle factors stored in the module handle
///   ([`FFTModuleHandle`]), shared across all operations on the same module.
///
/// # CPU feature requirements
///
/// **Runtime check**: [`Module::new()`](poulpy_hal::api::ModuleNew::new) verifies that
/// the CPU supports AVX-512F. If the feature is missing, the constructor panics.
///
/// **Compile-time requirement**: Code must be compiled with `-C target-feature=+avx512f`.
/// Failure to do so results in a compile error.
///
/// # Thread safety
///
/// `FFT64Avx512` is `Send + Sync` (derived from being a zero-sized, field-less struct).
/// The `Module<FFT64Avx512>` that holds the FFT tables is also `Send + Sync`, so modules can
/// be shared across threads. Individual operations require exclusive (`&mut`) access to their
/// output buffers and scratch space, preventing data races at the API level.
///
/// # Performance notes
///
/// - Optimized for x86-64 CPUs with AVX-512F support.
/// - Hand-written assembly FFT kernels are reused for the 16-point base case.
/// - Memory layout is vectorized (8 × `i64` per AVX-512 register) where the integer-domain helpers use SIMD lanes.
/// - Scalar fallback remains for non-multiple-of-8 tails in the integer-domain helpers.
///
/// # Example usage pattern
///
/// This type is typically not used directly but via HAL generic code:
///
/// ```text
/// use poulpy_hal::layouts::Module;
/// use poulpy_cpu_avx512::FFT64Avx512;
///
/// let module: Module<FFT64Avx512> = Module::new(1024);
/// // Use module for FHE operations...
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FFT64Avx512;

#[cfg(test)]
pub mod tests;

#[allow(unused_imports)]
pub use poulpy_cpu_ref::reference::fft64::module::FFTModuleHandle;
pub use reim::{FFT64Avx512ReimTable, ReimFFTAvx512, ReimIFFTAvx512};
