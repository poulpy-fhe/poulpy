#![allow(clippy::too_many_arguments)]

//! Independent scalar backends for correctness testing, not production execution.
//!
//! [`FFT64Oracle`] and [`NTT4x30Oracle`] own their arithmetic and numerical
//! tables. They do not depend on `poulpy-cpu-ref` or the SIMD CPU backends.
//! Core integration uses the generic HAL/Core compositions.

#[cfg(feature = "enable-core")]
#[doc(hidden)]
mod core_impl;
mod embed;
mod fft64;
mod hal_defaults;
mod hal_impl;
mod normalize;
mod ntt4x30;
mod products;
#[cfg(feature = "enable-core")]
mod sampling;
#[cfg(feature = "enable-core")]
mod scalar_znx_fill;

mod kernels;
mod table_cache;

#[cfg(test)]
mod tests;

pub(crate) mod layouts {
    pub use poulpy_hal::layouts::*;
}

pub(crate) mod source {
    pub use poulpy_hal::source::*;
}

#[cfg(feature = "enable-core")]
pub(crate) use scalar_znx_fill::ScalarZnxFill;

pub use fft64::{FFT64Oracle, FFT64ReimTable};
pub use ntt4x30::{NTT4x30Oracle, NTT4x30OracleHandle};
