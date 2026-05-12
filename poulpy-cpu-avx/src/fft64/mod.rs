mod convolution;
mod module;
mod reim;
mod reim4;

/// AVX2/FMA-accelerated CPU backend for Poulpy HAL.
///
/// `FFT64Avx` is a zero-sized marker type that selects the AVX2-optimized CPU backend
/// when used as the type parameter `B` in [`Module<B>`](poulpy_hal::layouts::Module)
/// and related HAL types. It implements all open extension point (OEP) traits from
/// `poulpy_hal::oep` using hand-tuned AVX2/FMA SIMD intrinsics and assembly kernels.
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
/// the CPU supports AVX2, AVX, and FMA. If any feature is missing, the constructor panics.
///
/// **Compile-time requirement**: Code must be compiled with `-C target-feature=+avx2,+fma`.
/// Failure to do so results in a compile error.
///
/// # Thread safety
///
/// `FFT64Avx` is `Send + Sync` (derived from being a zero-sized, field-less struct).
/// The `Module<FFT64Avx>` that holds the FFT tables is also `Send + Sync`, so modules can
/// be shared across threads. Individual operations require exclusive (`&mut`) access to their
/// output buffers and scratch space, preventing data races at the API level.
///
/// # Performance notes
///
/// - Optimized for x86-64 CPUs with AVX2 (2013+) and FMA (Haswell and later).
/// - Hand-written assembly FFT kernels outperform compiler-generated intrinsics on
///   supported micro-architectures; the exact gain is host-dependent.
/// - Memory layout is vectorized (4 × `i64` per AVX2 register) with 64-byte alignment.
/// - Scalar fallback for non-multiple-of-4 lengths (negligible overhead for typical FHE parameters).
///
/// # Example usage pattern
///
/// This type is typically not used directly but via HAL generic code:
///
/// ```text
/// use poulpy_hal::layouts::Module;
/// use poulpy_cpu_avx::FFT64Avx;
///
/// let module: Module<FFT64Avx> = Module::new(1024);
/// // Use module for FHE operations...
/// ```
pub struct FFT64Avx {}

#[cfg(test)]
pub mod tests;

#[allow(unused_imports)]
pub use poulpy_cpu_ref::reference::fft64::module::FFTModuleHandle;
pub use reim::{FFT64AvxReimTable, ReimFFTAvx, ReimIFFTAvx};
