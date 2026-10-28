//! The BLAS layer at large blocks (`poulpy_bench::blas::scale`): the radix table of every transform family on both
//! rings, then the checks on the NTT backends, once each, outside Criterion. Arguments that do not start with `--` filter
//! the tables and checks by label, `max_base2k` for the tables alone, `!word` to leave out the labels holding `word`:
//!
//! ```text
//! CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C target-feature=+avx2,+fma,+avx512f,+avx512ifma,+avx512vl,+bmi2,+adx" \
//!     cargo bench --target x86_64-unknown-linux-gnu -p poulpy-cpu-avx512 --features enable-blas,enable-ifma \
//!     --bench blas_scale -- max_base2k
//! ```

use poulpy_bench::{blas::scale, heap::Counting};
use poulpy_cpu_avx512::{FFT64Avx512, FFT64CIAvx512, NTT4x30Avx512, NTT4x30CIAvx512};

#[global_allocator]
static HEAP: Counting = Counting;

fn main() {
    let filter: Vec<String> = std::env::args().skip(1).filter(|arg| !arg.starts_with("--")).collect();
    scale::max_base2k_table::<FFT64Avx512>(&filter);
    scale::max_base2k_table::<FFT64CIAvx512>(&filter);
    scale::max_base2k_table::<NTT4x30Avx512>(&filter);
    scale::max_base2k_table::<NTT4x30CIAvx512>(&filter);
    #[cfg(feature = "enable-ifma")]
    {
        scale::max_base2k_table::<poulpy_cpu_avx512::NTT3x42Ifma>(&filter);
        scale::max_base2k_table::<poulpy_cpu_avx512::NTT3x42CIIfma>(&filter);
    }
    let failed = scale::validate::<NTT4x30Avx512>(&filter);
    #[cfg(feature = "enable-ifma")]
    let failed = [failed, scale::validate::<poulpy_cpu_avx512::NTT3x42Ifma>(&filter)].concat();
    assert!(failed.is_empty(), "{} checks failed: {failed:#?}", failed.len());
}
