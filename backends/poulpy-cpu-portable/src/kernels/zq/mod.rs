//! Negacyclic number-theoretic transform over `Z/q` for a modulus of any size.
//!
//! The same structure as the FFT64 transform: a forward table [`ZqNTTTable`] and an
//! inverse table [`ZqINTTTable`], an execute trait [`ZqDFTExecute`] that runs a table,
//! the portable executors [`ZqNTTPortable`] and [`ZqINTTPortable`] over the kernels
//! [`zq_ntt_portable`] and [`zq_intt_portable`], and the plan [`ZqPlan`] that pairs both
//! tables with an executor each and implements the HAL traits
//! [`ZqNew`](poulpy_hal::api::ZqNew), [`ZqNTT`](poulpy_hal::api::ZqNTT) and
//! [`ZqINTT`](poulpy_hal::api::ZqINTT). A SIMD crate can add its own executors and plug
//! them into [`ZqPlan`].
//!
//! For a power-of-two `n` and an odd `q`, `psi^n = -1 mod q` makes `psi^2` a principal
//! `n`-th root of unity, so the transform is invertible; `q` need not be prime.

mod kernels;
mod plan;
mod table_intt;
mod table_ntt;

pub use kernels::*;
pub use plan::*;
pub use table_intt::*;
pub use table_ntt::*;

/// Runs the transform a precomputed table describes on `data`, in place.
pub trait ZqDFTExecute<Table, W> {
    fn zq_dft_execute(table: &Table, data: &mut [W]);
}
