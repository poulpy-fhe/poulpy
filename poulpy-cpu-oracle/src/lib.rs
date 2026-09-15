#![allow(clippy::too_many_arguments)]

//! Independent scalar backends for correctness testing, not production execution.
//!
//! [`FFT64Oracle`] and [`NTT4x30Oracle`] own their arithmetic and numerical
//! tables. They do not depend on portable or SIMD CPU implementations.
//! Core and scheme integration uses the generic HAL/Core/CKKS compositions.

#[cfg(feature = "enable-ckks")]
pub mod ckks_encoding;
#[cfg(feature = "enable-ckks")]
mod ckks_impl;
#[cfg(feature = "enable-ckks")]
pub mod ckks_paco;
#[cfg(feature = "enable-ckks")]
pub mod ckks_ship;
#[cfg(feature = "enable-core")]
#[doc(hidden)]
pub mod core_impl;
pub mod fft64;
pub mod hal_defaults;
mod hal_impl;
pub mod ntt4x30;
mod sampling;
mod scalar_znx_fill;

pub mod reference;
pub mod table_cache;
pub mod test_suite;

#[cfg(test)]
mod tests;

pub use poulpy_hal::cast_mut;

pub mod api {
    pub use poulpy_hal::api::*;
}

pub mod layouts {
    pub use poulpy_hal::layouts::*;
}

pub mod source {
    pub use poulpy_hal::source::*;
}

pub use scalar_znx_fill::ScalarZnxFill;

pub use fft64::{FFT64Oracle, FFT64ReimTable};
pub use ntt4x30::{NTT4x30Oracle, NTT4x30OracleHandle};
