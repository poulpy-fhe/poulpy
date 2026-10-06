//! CKKS-specific multiparty protocols, layered as the crate: [`api`] the
//! operations, [`layouts`] their shares, [`oep`] the backend extension points
//! and [`reference`](mod@crate::ckks::reference) the default implementation.
pub mod api;
mod delegates;
pub mod layouts;
pub mod oep;
pub mod reference;
pub mod test_suite;

pub use api::*;
pub use layouts::*;
