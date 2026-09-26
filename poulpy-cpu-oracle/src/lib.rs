#![allow(clippy::too_many_arguments)]

//! Independent scalar backends for correctness testing, not production execution.
//!
//! [`FFT64Oracle`] and [`NTT4x30Oracle`] own their arithmetic and numerical
//! tables. They do not depend on portable or SIMD CPU implementations.
//! Core and scheme integration uses the generic HAL/Core/CKKS compositions.

#[cfg(feature = "enable-ckks")]
mod ckks_encoding;
#[cfg(feature = "enable-ckks")]
mod ckks_impl;
#[cfg(feature = "enable-ckks")]
mod ckks_paco;
#[cfg(feature = "enable-ckks")]
mod ckks_ship;
#[cfg(feature = "enable-core")]
#[doc(hidden)]
mod core_impl;
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

mod reference;
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

#[cfg(feature = "enable-ckks")]
pub mod ckks_encoding_fft;
#[cfg(feature = "enable-ckks")]
pub mod ckks_roots;
