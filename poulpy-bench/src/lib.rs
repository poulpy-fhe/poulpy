//! Shared helpers for `poulpy-bench` benchmark binaries.
//!
//! # Public dispatch macros
//!
//! Three macros are exported for bench files to use:
//!
//! ```text
//! for_each_fft_backend!(path [, leading_arg]* ; criterion_ref)
//! for_each_ntt_backend!(path [, leading_arg]* ; criterion_ref)
//! for_each_backend!(path [, leading_arg]* ; criterion_ref)
//! ```
//!
//! Each expands to one call per matching backend:
//!
//! ```text
//! path::<BackendType>(leading_args..., criterion_ref, "backend-label");
//! ```
//!
//! Use:
//! - `for_each_fft_backend!` for FFT64-specific operations (`fft` primitive benches)
//! - `for_each_ntt_backend!` for NTT-family-specific operations (`ntt` primitive benches)
//! - `for_each_backend!` for operations that work with any backend (generic GLWE ops, vec_znx, etc.)
//!
//! # Adding a new backend
//!
//! 1. Add the crate to `[dependencies]` in `poulpy-bench/Cargo.toml` (optionally behind a feature).
//! 2. Add one `#[cfg(...)] { use $fn as __f; __f::<NewType>(...); }` line to the appropriate
//!    private family macro below (`for_each_fft_backend_family!` or `for_each_ntt_backend_family!`).
//! 3. No bench files need to change.

pub mod bench_suite;
pub mod params;

/// Return the shared Criterion configuration used by all bench binaries.
///
/// Uses 100 samples with a 5-second measurement budget per benchmark.
/// Fast benchmarks complete in ~5 s; for slow benchmarks whose single
/// iteration exceeds the per-sample budget Criterion automatically extends
/// the run to collect at least a few samples (it will never cut a sample
/// short), so scheme-level benchmarks (blind rotate, CBS) may take longer.
pub fn criterion_config() -> criterion::Criterion {
    criterion::Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(5))
}

/// Return the Criterion configuration used by CKKS benchmarks.
///
/// CKKS operations are slow enough that the default 100-sample policy can make
/// a benchmark run impractically long. This keeps a 10-second measurement
/// budget with 20 samples, giving a small but useful floor without trying to
/// force 100 samples for each expensive operation.
pub fn ckks_criterion_config() -> criterion::Criterion {
    criterion::Criterion::default()
        .sample_size(20)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(10))
}

/// Private: expands to every FFT64 backend in tier order (ref → avx → gpu).
#[doc(hidden)]
#[macro_export]
macro_rules! for_each_fft_backend_family {
    ($fn:path $(, $arg:expr)* ; $c:expr) => {{
        {
            use $fn as __f;
            __f::<poulpy_cpu_ref::FFT64Ref>($($arg,)* $c, "fft64-ref");
        }
        #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
        {
            use $fn as __f;
            __f::<poulpy_cpu_avx::FFT64Avx>($($arg,)* $c, "fft64-avx");
        }
        #[cfg(all(feature = "enable-avx512f", target_arch = "x86_64"))]
        {
            use $fn as __f;
            __f::<poulpy_cpu_avx512::FFT64Avx512>($($arg,)* $c, "fft64-avx512");
        }
        // #[cfg(feature = "enable-gpu")]
        // { use $fn as __f; __f::<poulpy_gpu::FFT64GPU>($($arg,)* $c, "fft64-gpu"); }
    }};
}

/// Private: expands to every NTT-family backend in tier order
/// (ntt120-ref → ntt120-avx → ntt120-avx512 → ntt-ifma → gpu).
#[doc(hidden)]
#[macro_export]
macro_rules! for_each_ntt_backend_family {
    ($fn:path $(, $arg:expr)* ; $c:expr) => {{
        {
            use $fn as __f;
            __f::<poulpy_cpu_ref::NTT120Ref>($($arg,)* $c, "ntt120-ref");
        }
        #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
        {
            use $fn as __f;
            __f::<poulpy_cpu_avx::NTT120Avx>($($arg,)* $c, "ntt120-avx");
        }
        #[cfg(all(feature = "enable-avx512f", target_arch = "x86_64"))]
        {
            use $fn as __f;
            __f::<poulpy_cpu_avx512::NTT120Avx512>($($arg,)* $c, "ntt120-avx512");
        }
        #[cfg(all(feature = "enable-ifma", target_arch = "x86_64"))]
        {
            use $fn as __f;
            __f::<poulpy_cpu_avx512::NTT126Ifma>($($arg,)* $c, "ntt-ifma");
        }
        // #[cfg(feature = "enable-gpu")]
        // { use $fn as __f; __f::<poulpy_gpu::NTT120GPU>($($arg,)* $c, "ntt120-gpu"); }
    }};
}

/// Run a bench function against every FFT64 backend.
///
/// Use for operations that are specific to the FFT64 transform domain
/// (DFT, convolution, VMP/SVP with DFT).
#[macro_export]
macro_rules! for_each_fft_backend {
    ($fn:path $(, $arg:expr)* ; $c:expr) => {{
        poulpy_bench::for_each_fft_backend_family!($fn $(, $arg)* ; $c);
    }};
}

/// Run a bench function against every NTT120 backend.
///
/// Use for operations that are specific to the NTT120 transform domain.
#[macro_export]
macro_rules! for_each_ntt_backend {
    ($fn:path $(, $arg:expr)* ; $c:expr) => {{
        poulpy_bench::for_each_ntt_backend_family!($fn $(, $arg)* ; $c);
    }};
}

/// Run a bench function against every available backend (FFT64 and NTT-family).
///
/// Use for operations that work with any backend: generic GLWE operations,
/// `vec_znx` / `vec_znx_big` arithmetic, encryption, decryption, key-switching, etc.
#[macro_export]
macro_rules! for_each_backend {
    ($fn:path $(, $arg:expr)* ; $c:expr) => {{
        poulpy_bench::for_each_fft_backend_family!($fn $(, $arg)* ; $c);
        poulpy_bench::for_each_ntt_backend_family!($fn $(, $arg)* ; $c);
    }};
}
